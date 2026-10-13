/* Dedicated gfx950 grouped-MoE prefill kernels for block-fp8 [128,128] experts with bf16
 * activations (W8A16): ops 85 (MoeGroupGluPf, i3=1), 86 (MoeGroupDownPf, i3=1) and the k=8 f32
 * combine (op 87), each a standalone object segment (PLOW_GLM_MOE_PF_LEAN).
 *
 * BIT-IDENTICAL to the interpreter arms (op_moe.h d_moe_group_pf_t / d_moe_combine_pf) by
 * construction: every output element sees the same v_mfma_f32_32x32x16_bf16 sequence (same
 * fragment map, K ascending in 16-steps), the same per-128-K promotion `accf += acc * bs`, the same
 * epilogue expression. Only the tiling and the schedule change.
 *
 * Geometry: a work item is up to 128 sorted rows of ONE expert (two of the align's 64-row tiles)
 * x one N tile (GLU 128 fused gate|up columns, DOWN 256 columns), BK = 64. 8 waves as 2 (M) x 4 (N),
 * each wave 64 rows x two 32-column fragments. A persistent workgroup walks its items as one
 * flattened k-step stream with two register stages over an LDS ping-pong, so the next item's
 * first loads are in flight under the current item's last steps and epilogue. B is staged as raw
 * fp8 and decoded (exactly) to bf16 at the LDS commit.
 *
 * The A operand is staged through `mfp_load_a` / `mfp_commit_a` only, so an fp8-activation
 * variant (1x128 per-token scales) swaps those two and the promotion scale. */
#define PLOW_WG_WAVES 8
#define PLOW_MOE_PREFILL 1
#include "amd_common.h"
#include "op_moe.h"

extern "C" __device__ unsigned plow_moe_fp8_pf_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_fp8_pf_bm128_bk64_1 = 1;

/* XCD-aware item order: the gridDim/8 workgroups of one XCD (dispatch is round-robin over the
 * 8 XCDs) take consecutive items, so the n-tiles of one row block and the blocks of one expert
 * run together on one L2. Ordering only; values are unchanged. */
#ifndef MFP_XCD
#define MFP_XCD 1
#endif
/* MFP_SWAP: issue each MFMA as weight x activation (D^T), so a lane holds ONE row and four
 * consecutive columns per accumulator group: 16-byte part stores and 8-byte fu stores instead of
 * 4/2-byte ones. Same 16 products per k-step in the same k order. */
#ifndef MFP_SWAP
#define MFP_SWAP 1
#endif
/* MFP_DEPI (DOWN, with MFP_SWAP): stage each 32x32 f32 quadrant of a wave through a private LDS
 * slab so every part row leaves as one 128-byte run (8 lanes x 16 B) instead of 32-byte pieces.
 * Same values, same addresses. */
#ifndef MFP_DEPI
#define MFP_DEPI 0
#endif
/* Ceiling instrument only (wrong output): 1 = no MFMA, 2 = no global loads, 3 = no epilogue. */
#ifndef MFP_ABL
#define MFP_ABL 0
#endif

namespace mfp {
constexpr unsigned BM = 128, BK = 64, NTHR = 512;
constexpr unsigned A_HALVES = BM * BK;   /* 16 KB */
constexpr unsigned B_ROWS = 256;
constexpr unsigned B_HALVES = B_ROWS * BK / 2; /* 16 KB of raw fp8 */
constexpr unsigned STAGE = A_HALVES + B_HALVES;
constexpr unsigned MAXE = 512;
} // namespace mfp

/* LDS element offset of 8-half octet `oct` in row `row` of a BK=64 bf16 tile (16-byte XOR swizzle,
 * the same cell map op_moe.h uses). */
__device__ __forceinline__ unsigned mfp_cell(unsigned row, unsigned oct) {
    return row * mfp::BK + ((oct ^ (row & 7u)) << 3);
}

/* Byte offset of 16-byte chunk `c` (k 16c..16c+15) in row `row` of the raw-fp8 BK=64 B tile. The
 * XOR spreads the 32 rows of one b64 fragment read over 16 bank pairs (2-way, the minimum for
 * 8-byte reads of one k-octet). */
__device__ __forceinline__ unsigned mfp_bcell(unsigned row, unsigned c) {
    return row * 64u + ((c ^ ((row >> 2) & 3u)) << 4);
}

/* 8 raw e4m3 bytes -> 8 bf16, the exact decode fp8_to_bf16v8 uses (byte order preserved). */
__device__ __forceinline__ bf16x8 mfp_fp8x8_bf16(unsigned w0, unsigned w1) {
    typedef bf16_t bf16_2 __attribute__((ext_vector_type(2)));
    union { bf16x8 v; unsigned u[4]; } o;
    o.u[0] = __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w0, 1.0f, false));
    o.u[1] = __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w0, 1.0f, true));
    o.u[2] = __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w1, 1.0f, false));
    o.u[3] = __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w1, 1.0f, true));
    return o.v;
}

/* Per-item metadata the producer resolves and hands to the consumer. */
struct MfpItem {
    unsigned e, row0, valid, n0;
    const unsigned char* w0;
    const unsigned char* w1;
    const float* s0;
    const float* s1;
};

__device__ __forceinline__ unsigned mfp_remap(unsigned s, unsigned total) {
#if MFP_XCD
    const unsigned g = gridDim.x, j = s / g, b = s - j * g;
    if ((g & 7u) == 0u && (j + 1u) * g <= total) return j * g + (b & 7u) * (g >> 3) + (b >> 3);
#endif
    (void)total;
    return s;
}

template <bool GLU>
__device__ __forceinline__ MfpItem mfp_decode(unsigned it, unsigned tn, const int* bp,
                                              const int* meta, unsigned n_exp,
                                              const unsigned long long* wtab,
                                              const unsigned long long* stab) {
    const unsigned blk = it / tn, nt = it % tn;
    unsigned lo = 0, hi = n_exp;
    while (lo + 1u < hi) {
        const unsigned mid = (lo + hi) >> 1;
        if ((unsigned)bp[mid] <= blk) lo = mid;
        else hi = mid;
    }
    MfpItem m;
    m.e = __builtin_amdgcn_readfirstlane(lo);
    const int* tilep = meta + 2u * n_exp;
    const unsigned t0 = (unsigned)tilep[m.e], t1 = (unsigned)tilep[m.e + 1u];
    m.row0 = (unsigned)meta[m.e] + (blk - (unsigned)bp[m.e]) * mfp::BM;
    const unsigned rowend = (unsigned)meta[m.e] + (t1 - t0) * 64u;
    m.valid = rowend > m.row0 ? min(rowend - m.row0, mfp::BM) : 0u;
    m.n0 = nt * (GLU ? 128u : 256u);
    const unsigned long long wb0 = wtab[(size_t)m.e * 3 + (GLU ? 0 : 2)];
    m.w0 = (const unsigned char*)(size_t)wb0;
    m.w1 = GLU ? (const unsigned char*)(size_t)wtab[(size_t)m.e * 3 + 1] : nullptr;
    m.s0 = (const float*)(size_t)stab[(size_t)m.e * 3 + (GLU ? 0 : 2)];
    m.s1 = GLU ? (const float*)(size_t)stab[(size_t)m.e * 3 + 1] : nullptr;
    if (wb0 == 0ull) m.valid = 0u; /* EP "not my expert": no loads, no stores */
    /* Everything here is workgroup-uniform; pin it to SGPRs (the LDS prefix reads are VGPR). */
    m.row0 = __builtin_amdgcn_readfirstlane(m.row0);
    m.valid = __builtin_amdgcn_readfirstlane(m.valid);
    m.n0 = __builtin_amdgcn_readfirstlane(m.n0);
    return m;
}

/* Grouped expert GEMM. GLU: Cout = fu_g bf16 [rows][N], A = x gathered by row_token, B = gate|up.
 * DOWN: Cout = part f32 [T*k][N] scattered by row_partidx * row_gate, A = fu_g rows. */
template <bool GLU>
__device__ __forceinline__ void mfp_body(void* __restrict__ Cout, const bf16* __restrict__ A,
                                         const unsigned long long* __restrict__ wtab,
                                         const unsigned long long* __restrict__ stab,
                                         const int* __restrict__ meta,
                                         const unsigned* __restrict__ row_token,
                                         const unsigned* __restrict__ row_partidx,
                                         const float* __restrict__ row_gate, unsigned N,
                                         unsigned K, unsigned n_exp, unsigned act, float beta,
                                         float lbeta) {
    using namespace mfp;
    __shared__ __align__(16) bf16 lds[2 * STAGE];
    __shared__ int bp[MAXE + 1];
    /* meta / weight table / scale table, staged once: item decode is then LDS-only and never
     * drains the k-step loads in flight behind a dependent global read. */
    __shared__ int s_meta[3 * MAXE + 1];
    __shared__ unsigned long long s_wtab[3 * MAXE], s_stab[3 * MAXE];
    __shared__ unsigned s_pidx[2][BM];
    __shared__ float s_gate[2][BM];
#if MFP_DEPI
    __shared__ __align__(16) float s_epi[GLU ? 1 : 8 * 32 * 36];
#endif

    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned wm = wave >> 2, wn = wave & 3u;
    const int* tilep = meta + 2u * n_exp;

    for (unsigned i = tid; i < 3u * n_exp + 1u; i += NTHR) s_meta[i] = meta[i];
    for (unsigned i = tid; i < 3u * n_exp; i += NTHR) {
        s_wtab[i] = wtab[i];
        s_stab[i] = stab[i];
    }
    /* 128-row block prefix over experts: bp[e] = sum_{e'<e} ceil(tiles_e'/2). */
    if (tid < n_exp) bp[tid + 1u] = (tilep[tid + 1u] - tilep[tid] + 1) >> 1;
    if (tid == 0) bp[0] = 0;
    __syncthreads();
    if (wave == 0) {
        const unsigned per = (n_exp + 63u) >> 6;
        const unsigned b0 = 1u + lane * per;
        int s = 0;
        for (unsigned i = 0; i < per; i++)
            if (b0 + i <= n_exp) s += bp[b0 + i];
        int incl = s;
#pragma unroll
        for (int d = 1; d < 64; d <<= 1) {
            const int v = __shfl_up(incl, d, 64);
            if ((int)lane >= d) incl += v;
        }
        int run = incl - s;
        for (unsigned i = 0; i < per; i++)
            if (b0 + i <= n_exp) {
                run += bp[b0 + i];
                bp[b0 + i] = run;
            }
    }
    __syncthreads();

    const unsigned tn = GLU ? N / 128u : N / 256u;
    const unsigned total = (unsigned)bp[n_exp] * tn;
    const unsigned NT = K / BK, KB = K / 128u;
    if (blockIdx.x >= total) return;

    /* ---- producer state ---- */
    /* The producer walks items p_it = blockIdx + j*grid. `pre` is the item after the one being
     * loaded, decoded (LDS only) and with its source-row loads already issued, so an item switch
     * consumes registers that landed a whole item ago. */
    unsigned p_it = blockIdx.x, p_kt = 0, p_seq = 0;
    auto decode = [&](unsigned it) {
        return mfp_decode<GLU>(mfp_remap(it, total), tn, bp, s_meta, n_exp, s_wtab, s_stab);
    };
    MfpItem pre = decode(p_it);
    unsigned pre_src[2] = {0u, 0u}; /* GLU: row_token of this thread's two A rows */
    unsigned pre_pidx = 0u;         /* DOWN: row tid's partidx / gate */
    float pre_gate = 0.0f;
    /* Last step's prefetch, copied at the head of the next load (it has landed by then); an item
     * switch reads these, never the loads it just issued. */
    unsigned snap_src[2] = {0u, 0u}, snap_pidx = 0u;
    float snap_gate = 0.0f;
    auto prefetch = [&]() {
        if constexpr (GLU) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const unsigned r = (tid + (unsigned)i * NTHR) >> 3;
                pre_src[i] = as_glob(row_token)[pre.row0 + (r < pre.valid ? r : 0u)];
            }
        } else {
            const unsigned ix = pre.row0 + (tid < pre.valid ? tid : 0u);
            pre_pidx = as_glob(row_partidx)[ix];
            pre_gate = as_glob(row_gate)[ix];
        }
    };
    prefetch();
    /* Every fetch is unconditional (a guarded load makes the compiler drain vmcnt inside the
     * branch): absent A rows read row 0 of A and are zeroed at the commit by `amask`. */
    const bf16* aptr[2];
    const unsigned char* bptr[2];
    unsigned amask = 0;
    unsigned m_pidx = PLOW_EXPERT_UNUSED; /* DOWN: row tid's metadata for the item being loaded */
    float m_gate = 0.0f;
    MfpItem pi = pre;
    /* Make `pre` the item being loaded and start prefetching the one after it. */
    auto advance = [&]() {
        pi = pre;
        const MfpItem& m = pi;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const unsigned ch = tid + (unsigned)i * NTHR;
            const unsigned r = ch >> 3, c = ch & 7u;
            const bool live = r < m.valid && (!GLU || snap_src[i] != PLOW_EXPERT_UNUSED);
            const unsigned src = GLU ? snap_src[i] : m.row0 + r;
            aptr[i] = A + (size_t)(live ? src : 0u) * K + c * 8u;
            amask = (amask & ~(1u << i)) | ((unsigned)live << i);
            const unsigned rb = ch >> 2, cb = ch & 3u;
            bptr[i] = (const unsigned char*)A + cb * 16u;
            if (m.valid) {
                if constexpr (GLU) {
                    const unsigned char* w = rb < 128u ? m.w0 : m.w1;
                    bptr[i] = w + (size_t)(m.n0 + (rb & 127u)) * K + cb * 16u;
                } else {
                    bptr[i] = m.w0 + (size_t)(m.n0 + rb) * K + cb * 16u;
                }
            }
        }
        if constexpr (!GLU) {
            m_pidx = tid < m.valid ? snap_pidx : PLOW_EXPERT_UNUSED;
            m_gate = snap_gate;
        }
        if (p_it + gridDim.x < total) pre = decode(p_it + gridDim.x);
    };
    auto snapshot = [&]() {
        snap_src[0] = pre_src[0];
        snap_src[1] = pre_src[1];
        snap_pidx = pre_pidx;
        snap_gate = pre_gate;
    };
    snapshot(); /* one-time wait for the first item's rows */
    advance();
    MfpItem nxt = pi;

    bf16v8 ra0[2], ra1[2];
    fp8v16 rb0[2], rb1[2];
    bool mp0 = false, mp1 = false; /* stage carries an item's first step: commit row metadata */
    bool v0 = false, v1 = false;   /* stage holds a loaded step */
    unsigned am0 = 0, am1 = 0;     /* stage's live-A-row mask */
    unsigned ms0 = 0, ms1 = 0;
    unsigned mpx0 = 0, mpx1 = 0;
    float mg0 = 0.f, mg1 = 0.f;

/* One producer k-step: snapshot + re-issue the next item's row prefetch, then the stage loads.
 * Unconditional (after the last item it re-reads valid addresses into a stage nobody commits),
 * so the waitcnt pass sees the same in-order count on every path. */
#define MFP_LOAD(RA, RB, MP, MS, MPX, MG, V, AM)                                                 \
    {                                                                                          \
        snapshot();                                                                            \
        prefetch();                                                                            \
        const unsigned k0 = p_kt * BK;                                                         \
        _Pragma("unroll") for (int i = 0; i < 2; i++) {                                         \
            if (MFP_ABL != 2) {                                                                 \
                RA[i] = ld_glob8(as_glob(aptr[i] + k0));                                       \
                RB[i] = ld_glob_fp8v16(bptr[i] + k0);                                          \
            }                                                                                  \
        }                                                                                      \
        MP = (p_kt == 0u);                                                                     \
        AM = amask;                                                                            \
        V = p_it < total;                                                                      \
        MS = p_seq;                                                                            \
        MPX = m_pidx;                                                                          \
        MG = m_gate;                                                                           \
        if (p_it < total && ++p_kt == NT) {                                                    \
            p_kt = 0;                                                                          \
            p_it += gridDim.x;                                                                 \
            p_seq++;                                                                           \
            if (p_it < total) {                                                                \
                advance();                                                                     \
                nxt = pi;                                                                      \
            }                                                                                  \
        }                                                                                      \
    }

/* Stage-to-LDS commit, split in pieces so they interleave with the MFMAs of the step in flight:
 * piece 0/1 = A chunk 0/1 (bf16), piece 2/3 = B chunk 0/1 (raw fp8) + DOWN's row metadata. */
#define MFP_COMMIT_PIECE(PC, BUF, RA, RB, MP, MS, MPX, MG, AM)                                       \
    {                                                                                          \
        bf16* const la_ = lds + (BUF) * STAGE;                                                 \
        unsigned char* const lb_ = (unsigned char*)(la_ + A_HALVES);                           \
        const unsigned ch = tid + (unsigned)((PC) & 1) * NTHR;                                 \
        if ((PC) < 2) {                                                                        \
            const unsigned r = ch >> 3, c = ch & 7u;                                           \
            *(bf16v8*)&la_[mfp_cell(r, c)] =                                                   \
                ((AM) >> ((PC) & 1)) & 1u ? RA[(PC) & 1] : bf16v8_zero();                      \
        } else {                                                                               \
            *(fp8v16*)&lb_[mfp_bcell(ch >> 2, ch & 3u)] = RB[(PC) & 1];                        \
        }                                                                                      \
        if constexpr (!GLU) {                                                                  \
            if ((PC) == 3 && (MP) && tid < BM) {                                               \
                s_pidx[(MS) & 1u][tid] = (MPX);                                                \
                s_gate[(MS) & 1u][tid] = (MG);                                                 \
            }                                                                                  \
        }                                                                                      \
    }
#define MFP_COMMIT(BUF, RA, RB, MP, MS, MPX, MG, AM)                                               \
    MFP_COMMIT_PIECE(0, BUF, RA, RB, MP, MS, MPX, MG, AM)                                          \
    MFP_COMMIT_PIECE(1, BUF, RA, RB, MP, MS, MPX, MG, AM)                                          \
    MFP_COMMIT_PIECE(2, BUF, RA, RB, MP, MS, MPX, MG, AM)                                          \
    MFP_COMMIT_PIECE(3, BUF, RA, RB, MP, MS, MPX, MG, AM)

    /* ---- consumer state ---- */
    unsigned c_it = blockIdx.x, c_kt = 0, c_seq = 0;
    MfpItem ci = pi;
    f32x16 acc[2][2], accf[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            acc[i][j] = (f32x16)(0.0f);
            accf[i][j] = (f32x16)(0.0f);
        }
    float bs[2] = {0.f, 0.f};

#if MFP_SWAP
#define MFP_MFMA(a, b, c) plow_mfma_bf16_32x32(b, a, c)
#else
#define MFP_MFMA(a, b, c) plow_mfma_bf16_32x32(a, b, c)
#endif
/* One k-step's MFMAs out of LDS buffer BUF, with the commit of the next step's register stage
 * into buffer CBUF interleaved (CV: that stage holds a loaded step). */
#define MFP_COMPUTE(BUF, CBUF, CV, RA, RB, MP, MS, MPX, MG, AM)                                      \
    {                                                                                          \
        const bf16* const la = lds + (BUF) * STAGE;                                            \
        const unsigned char* const lb = (const unsigned char*)(la + A_HALVES);                 \
        const bool live0 = MFP_ABL != 1 && wm * 64u < ci.valid;                                \
        const bool live1 = MFP_ABL != 1 && wm * 64u + 32u < ci.valid;                          \
        _Pragma("unroll") for (int q = 0; q < (int)(BK / MFMA_K); q++) {                        \
            const unsigned oct = (unsigned)q * 2u + (lane >> 5);                               \
            bf16x8 af[2], bf[2];                                                               \
            _Pragma("unroll") for (int i = 0; i < 2; i++) {                                     \
                const unsigned arow = wm * 64u + (unsigned)i * 32u + (lane & 31u);             \
                __builtin_memcpy(&af[i], &la[mfp_cell(arow, oct)], 16);                        \
            }                                                                                  \
            _Pragma("unroll") for (int j = 0; j < 2; j++) {                                     \
                const unsigned brow = (GLU ? (unsigned)j * 128u + wn * 32u                      \
                                           : wn * 64u + (unsigned)j * 32u) +                    \
                                      (lane & 31u);                                            \
                unsigned w2[2];                                                                \
                __builtin_memcpy(w2, &lb[mfp_bcell(brow, (unsigned)q) + (lane >> 5) * 8u], 8); \
                bf[j] = mfp_fp8x8_bf16(w2[0], w2[1]);                                          \
            }                                                                                  \
            __builtin_amdgcn_s_setprio(1);                                                     \
            if (live0) {                                                                       \
                acc[0][0] = MFP_MFMA(af[0], bf[0], acc[0][0]);                                 \
                acc[0][1] = MFP_MFMA(af[0], bf[1], acc[0][1]);                                 \
            }                                                                                  \
            if (live1) {                                                                       \
                acc[1][0] = MFP_MFMA(af[1], bf[0], acc[1][0]);                                 \
                acc[1][1] = MFP_MFMA(af[1], bf[1], acc[1][1]);                                 \
            }                                                                                  \
            __builtin_amdgcn_s_setprio(0);                                                     \
            if (CV) MFP_COMMIT_PIECE(q, CBUF, RA, RB, MP, MS, MPX, MG, AM)                          \
        }                                                                                      \
    }

    /* Scale-block rows: GLU's 128-wide tile is one row for gate and up; DOWN's wave covers 64. */
    auto load_scales = [&](unsigned kb) {
        if (!ci.valid) return;
        if constexpr (GLU) {
            const unsigned nr = ci.n0 >> 7;
            bs[0] = as_glob(ci.s0)[(size_t)nr * KB + kb];
            bs[1] = as_glob(ci.s1)[(size_t)nr * KB + kb];
        } else {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned nr = (ci.n0 + wn * 64u + (unsigned)j * 32u) >> 7;
                bs[j] = as_glob(ci.s0)[(size_t)nr * KB + kb];
            }
        }
    };

    auto promote = [&]() {
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                accf[i][j] += acc[i][j] * bs[j];
                acc[i][j] = (f32x16)(0.0f);
            }
    };

    auto epilogue = [&]() {
        if (ci.valid && MFP_ABL != 3) {
#if MFP_SWAP
            /* lane: row wm*64 + i*32 + lane%32; columns base + 8g + 4*(lane/32) + 0..3 for el = 4g..4g+3 */
            const unsigned rl = lane & 31u, h = lane >> 5;
            if constexpr (GLU) {
                bf16* const fu = (bf16*)Cout;
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const unsigned rr = wm * 64u + (unsigned)i * 32u + rl;
                    if (rr >= ci.valid) continue;
#pragma unroll
                    for (int g = 0; g < 4; g++) {
                        const unsigned nn = ci.n0 + wn * 32u + 8u * (unsigned)g + 4u * h;
                        typedef unsigned u2v __attribute__((ext_vector_type(2)));
                        union { bf16 b[4]; u2v u; } o;
#pragma unroll
                        for (int e = 0; e < 4; e++)
                            o.b[e] = f2bf(moe_glu(accf[i][0][4 * g + e], accf[i][1][4 * g + e], act,
                                                  beta, lbeta));
                        *(PLOW_GLOB u2v*)&fu[(size_t)(ci.row0 + rr) * N + nn] = o.u;
                    }
                }
            } else {
                float* const part = (float*)Cout;
                const unsigned par = c_seq & 1u;
#if MFP_DEPI
                typedef float f4e __attribute__((ext_vector_type(4)));
                float* const slab = s_epi + wave * (32u * 36u);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const unsigned rr = wm * 64u + (unsigned)i * 32u + rl;
                    const float gv = rr < ci.valid ? s_gate[par][rr] : 0.0f;
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        __builtin_amdgcn_wave_barrier();
#pragma unroll
                        for (int g = 0; g < 4; g++) {
                            f4e v;
#pragma unroll
                            for (int e = 0; e < 4; e++) v[e] = gv * accf[i][j][4 * g + e];
                            *(f4e*)&slab[rl * 36u + 8u * (unsigned)g + 4u * h] = v;
                        }
                        __builtin_amdgcn_wave_barrier();
                        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
                        for (int ps = 0; ps < 4; ps++) {
                            const unsigned r2 = (lane >> 3) + 8u * (unsigned)ps;
                            const unsigned rr2 = wm * 64u + (unsigned)i * 32u + r2;
                            const f4e v = *(const f4e*)&slab[r2 * 36u + (lane & 7u) * 4u];
                            if (rr2 >= ci.valid) continue;
                            const unsigned pidx = s_pidx[par][rr2];
                            if (pidx == PLOW_EXPERT_UNUSED) continue;
                            const unsigned nn = ci.n0 + wn * 64u + (unsigned)j * 32u + (lane & 7u) * 4u;
                            __builtin_nontemporal_store(
                                v, (PLOW_GLOB f4e*)&part[(size_t)pidx * N + nn]);
                        }
                    }
                }
#else
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const unsigned rr = wm * 64u + (unsigned)i * 32u + rl;
                    if (rr >= ci.valid) continue;
                    const unsigned pidx = s_pidx[par][rr];
                    if (pidx == PLOW_EXPERT_UNUSED) continue;
                    const float gv = s_gate[par][rr];
                    typedef float f4 __attribute__((ext_vector_type(4)));
#pragma unroll
                    for (int j = 0; j < 2; j++)
#pragma unroll
                        for (int g = 0; g < 4; g++) {
                            const unsigned nn =
                                ci.n0 + wn * 64u + (unsigned)j * 32u + 8u * (unsigned)g + 4u * h;
                            f4 v;
#pragma unroll
                            for (int e = 0; e < 4; e++) v[e] = gv * accf[i][j][4 * g + e];
                            __builtin_nontemporal_store(
                                v, (PLOW_GLOB f4*)&part[(size_t)pidx * N + nn]);
                        }
                }
#endif
            }
#else
            if constexpr (GLU) {
                bf16* const fu = (bf16*)Cout;
                const unsigned nn = ci.n0 + wn * 32u + (lane & 31u);
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr = wm * 64u + (unsigned)i * 32u + mfma_acc_m(lane, el);
                        if (rr < ci.valid)
                            st_act1(&fu[(size_t)(ci.row0 + rr) * N + nn],
                                    f2bf(moe_glu(accf[i][0][el], accf[i][1][el], act, beta,
                                                 lbeta)));
                    }
            } else {
                float* const part = (float*)Cout;
                const unsigned par = c_seq & 1u;
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const unsigned nn = ci.n0 + wn * 64u + (unsigned)j * 32u + (lane & 31u);
#pragma unroll
                        for (int el = 0; el < 16; el++) {
                            const unsigned rr =
                                wm * 64u + (unsigned)i * 32u + mfma_acc_m(lane, el);
                            if (rr >= ci.valid) continue;
                            const unsigned pidx = s_pidx[par][rr];
                            if (pidx == PLOW_EXPERT_UNUSED) continue;
                            const float gv = s_gate[par][rr];
                            __builtin_nontemporal_store(
                                gv * accf[i][j][el],
                                (PLOW_GLOB float*)&part[(size_t)pidx * N + nn]);
                        }
                    }
            }
#endif
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) accf[i][j] = (f32x16)(0.0f);
    };

    /* Prologue: steps 0 and 1 into the two register stages, step 0 into LDS buffer 0. */
    MFP_LOAD(ra0, rb0, mp0, ms0, mpx0, mg0, v0, am0)
    MFP_LOAD(ra1, rb1, mp1, ms1, mpx1, mg1, v1, am1)
    MFP_COMMIT(0u, ra0, rb0, mp0, ms0, mpx0, mg0, am0)
    __syncthreads();

    /* NT is even (K % 128 == 0), and every item starts on an even flattened step, so buffer 0 /
     * stage 0 always holds an even k-tile and buffer 1 / stage 1 an odd one. */
    for (;;) {
        /* EVEN step: LDS 0 = (c_it, c_kt); stage 1 = next step; stage 0 is free. */
        MFP_LOAD(ra0, rb0, mp0, ms0, mpx0, mg0, v0, am0)
        load_scales(c_kt >> 1);
        MFP_COMPUTE(0u, 1u, true, ra1, rb1, mp1, ms1, mpx1, mg1, am1)
        __syncthreads();
        c_kt++;
        /* ODD step: LDS 1 = (c_it, c_kt); stage 0 = next step; stage 1 is free. */
        MFP_LOAD(ra1, rb1, mp1, ms1, mpx1, mg1, v1, am1)
        MFP_COMPUTE(1u, 0u, v0, ra0, rb0, mp0, ms0, mpx0, mg0, am0)
        promote();
        if (c_kt == NT - 1u) {
            epilogue();
            c_it += gridDim.x;
            c_seq++;
            c_kt = 0;
            if (c_it >= total) break;
            ci = nxt;
        } else {
            c_kt++;
        }
        __syncthreads();
    }
#undef MFP_LOAD
#undef MFP_COMMIT
#undef MFP_COMMIT_PIECE
#undef MFP_COMPUTE
}

extern "C" __global__ __launch_bounds__(512, 1)
void plow_moe_fp8_glu_pf_gfx950(bf16* __restrict__ out, const bf16* __restrict__ x,
                                const unsigned long long* __restrict__ wtab,
                                const unsigned long long* __restrict__ stab,
                                const int* __restrict__ meta,
                                const unsigned* __restrict__ row_token, unsigned inter,
                                unsigned hidden, unsigned experts, unsigned act, float beta,
                                float lbeta) {
    if (experts > mfp::MAXE || inter % 128u != 0u || hidden % 128u != 0u) return;
    mfp_body<true>(out, x, wtab, stab, meta, row_token, nullptr, nullptr, inter, hidden,
                   experts, act, beta, lbeta);
}

extern "C" __global__ __launch_bounds__(512, 1)
void plow_moe_fp8_down_pf_gfx950(float* __restrict__ part, const bf16* __restrict__ fu,
                                 const unsigned long long* __restrict__ wtab,
                                 const unsigned long long* __restrict__ stab,
                                 const int* __restrict__ meta,
                                 const unsigned* __restrict__ row_partidx,
                                 const float* __restrict__ row_gate, unsigned hidden,
                                 unsigned inter, unsigned experts, unsigned reserved) {
    (void)reserved;
    if (experts > mfp::MAXE || hidden % 256u != 0u || inter % 128u != 0u || inter < 256u) return;
    mfp_body<false>(part, fu, wtab, stab, meta, nullptr, row_partidx, row_gate, hidden, inter,
                    experts, 0u, 0.0f, 0.0f);
}
