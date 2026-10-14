/* Dedicated gfx950 grouped-MoE prefill kernels for block-fp8 [128,128] experts: ops 85
 * (MoeGroupGluPf) and 86 (MoeGroupDownPf) with i3 = 1, each a standalone object segment
 * (PLOW_MOE_FP8_PF_LEAN). Two numerics, one structure:
 *
 *   W8A16 (i7 = 0): bf16 activations, v_mfma_f32_32x32x16_bf16 on exactly decoded fp8 weights,
 *     promoted per 128 K as `accf += acc * ws` -- BIT-IDENTICAL to op_moe.h d_moe_group_pf_t
 *     (same fragment map, K ascending in 16-steps, same promotion and epilogue expressions).
 *   W8A8 (i7 = 2): 1x128 fp8 activations (QuantFp8Block128 rows + [H/128][pitch] scales for GLU,
 *     the GLU's own fq rows + [row][I/128] scales for DOWN), v_mfma_scale 32x32x64 fp8, promoted
 *     per 128 K as `accf = fma(t, as * ws, accf)`, GLU output re-quantized per 1x128 --
 *     BIT-IDENTICAL to moe_pf_w8a8.h d_moe_pf_w8a8 (same MFMA chain, promotion, quant).
 *
 * Geometry: a work item is up to 128 sorted rows of ONE expert (two of the align's 64-row tiles)
 * x one N tile (GLU 128 fused gate|up columns = one 1x128 output group, DOWN 256 columns), BK = 64.
 * 8 waves as 2 (M) x 4 (N), each wave 64 rows x two 32-column fragments. A persistent workgroup
 * walks its items as one flattened k-step stream: two register stages over an LDS ping-pong, the
 * next stage's LDS commit interleaved with the current step's MFMAs, and the next item's metadata
 * decoded from LDS copies of meta / weight / scale tables and its source rows prefetched one item
 * ahead, so item switches never drain the loads in flight. B is staged as raw fp8. */
#define PLOW_WG_WAVES 8
#define PLOW_MOE_PREFILL 1
#include "amd_common.h"
#include "op_moe.h"

extern "C" __device__ unsigned plow_moe_fp8_pf_abi_2 = 1;
extern "C" __device__ unsigned plow_moe_fp8_pf_w8a8_1 = 1;

/* XCD-aware item order: the gridDim/8 workgroups of one XCD (dispatch is round-robin over the
 * 8 XCDs) take consecutive items, so the n-tiles of one row block and the blocks of one expert
 * run together on one L2. Ordering only; values are unchanged. */
#ifndef MFP_XCD
#define MFP_XCD 1
#endif
/* MFP_DEPI (DOWN): stage each wave's 32x32 f32 quadrant through a private LDS slab so every part
 * row leaves as one 128-byte run (8 lanes x 16 B) instead of 4-byte lane stores. Same values. */
#ifndef MFP_DEPI
#define MFP_DEPI 0
#endif
/* Ceiling instrument only (wrong output): 1 = no MFMA, 2 = no global loads. */
#ifndef MFP_ABL
#define MFP_ABL 0
#endif

namespace mfp {
constexpr unsigned BM = 128, BK = 64, NTHR = 512;
constexpr unsigned A_BYTES = BM * BK * 2;  /* bf16 A k-tile (W8A8 uses the first half) */
constexpr unsigned B_BYTES = 256 * BK;     /* raw fp8 B k-tile */
constexpr unsigned STAGE = A_BYTES + B_BYTES;
constexpr unsigned MAXE = 512;
} // namespace mfp

/* Byte offset of 16-byte chunk `c` in row `row` of a 128-byte-row (bf16 BK=64) tile: the XOR
 * swizzle op_moe.h uses (MPF_XORSWZ), conflict-free for b128 fragment reads. */
__device__ __forceinline__ unsigned mfp_cell(unsigned row, unsigned c) {
    return row * 128u + ((c ^ (row & 7u)) << 4);
}

/* Byte offset of 16-byte chunk `c` in row `row` of a 64-byte-row (fp8 BK=64) tile. */
__device__ __forceinline__ unsigned mfp_bcell(unsigned row, unsigned c) {
    return row * 64u + ((c ^ ((row >> 2) & 3u)) << 4);
}

/* 8 raw e4m3 bytes -> 8 bf16, the exact decode fp8_to_bf16v8 uses (byte order preserved). */
__device__ __forceinline__ bf16x8 mfp_fp8x8_bf16(unsigned w0, unsigned w1) {
    union { bf16x8 v; unsigned u[4]; } o;
    o.u[0] = __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w0, 1.0f, false));
    o.u[1] = __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w0, 1.0f, true));
    o.u[2] = __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w1, 1.0f, false));
    o.u[3] = __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w1, 1.0f, true));
    return o.v;
}

/* 32 fp8 of row `row`, k 32h..32h+31, from a 64-byte-row tile: the K64 MFMA operand fragment. */
__device__ __forceinline__ fp8v32 mfp_frag32(const unsigned char* t, unsigned row, unsigned h) {
    const fp8v16 lo = *(const fp8v16*)(t + mfp_bcell(row, 2u * h));
    const fp8v16 hi = *(const fp8v16*)(t + mfp_bcell(row, 2u * h + 1u));
    fp8v32 v;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        v[j] = (int)lo[j];
        v[4 + j] = (int)hi[j];
    }
    return v;
}

/* Per-item metadata the producer resolves and hands to the consumer. */
struct MfpItem {
    unsigned row0, valid, n0;
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
    const unsigned e = __builtin_amdgcn_readfirstlane(lo);
    MfpItem m;
    const int* tilep = meta + 2u * n_exp;
    const unsigned t0 = (unsigned)tilep[e], t1 = (unsigned)tilep[e + 1u];
    m.row0 = (unsigned)meta[e] + (blk - (unsigned)bp[e]) * mfp::BM;
    const unsigned rowend = (unsigned)meta[e] + (t1 - t0) * 64u;
    m.valid = rowend > m.row0 ? min(rowend - m.row0, mfp::BM) : 0u;
    m.n0 = nt * (GLU ? 128u : 256u);
    const unsigned long long wb0 = wtab[(size_t)e * 3 + (GLU ? 0 : 2)];
    m.w0 = (const unsigned char*)(size_t)wb0;
    m.w1 = GLU ? (const unsigned char*)(size_t)wtab[(size_t)e * 3 + 1] : m.w0;
    m.s0 = (const float*)(size_t)stab[(size_t)e * 3 + (GLU ? 0 : 2)];
    m.s1 = GLU ? (const float*)(size_t)stab[(size_t)e * 3 + 1] : m.s0;
    if (wb0 == 0ull) m.valid = 0u; /* EP "not my expert": no MFMA, no stores */
    /* Workgroup-uniform; pin to SGPRs (the LDS prefix reads are VGPR). */
    m.row0 = __builtin_amdgcn_readfirstlane(m.row0);
    m.valid = __builtin_amdgcn_readfirstlane(m.valid);
    m.n0 = __builtin_amdgcn_readfirstlane(m.n0);
    return m;
}

/* Grouped expert GEMM.
 *   GLU : A = x rows gathered by row_token (bf16, or fp8 + as[kb*pitch + token]),
 *         Cout = fu_g (bf16 [rows][N], or fp8 + fs_out[row][N/128]).
 *   DOWN: A = fu_g rows (bf16, or fp8 + as[row*KB + kb]),
 *         Cout = part f32 [T*k][N] scattered by row_partidx, times row_gate. */
template <bool GLU, bool W8>
__device__ __forceinline__ void mfp_body(void* __restrict__ Cout, const void* __restrict__ Av,
                                         const unsigned long long* __restrict__ wtab,
                                         const unsigned long long* __restrict__ stab,
                                         const int* __restrict__ meta,
                                         const unsigned* __restrict__ row_token,
                                         const unsigned* __restrict__ row_partidx,
                                         const float* __restrict__ row_gate,
                                         const float* __restrict__ a_scale,
                                         float* __restrict__ fs_out, unsigned a_pitch, unsigned N,
                                         unsigned K, unsigned n_exp, unsigned act, float beta,
                                         float lbeta) {
    using namespace mfp;
    const unsigned char* const A = (const unsigned char*)Av;
    /* Bytes per A row. */
    const size_t arow = W8 ? (size_t)K : (size_t)K * 2u;
    __shared__ __align__(16) unsigned char lds[2 * STAGE];
    __shared__ int bp[MAXE + 1];
    /* meta / weight table / scale table, staged once: item decode is then LDS-only and never
     * drains the k-step loads in flight behind a dependent global read. */
    __shared__ int s_meta[3 * MAXE + 1];
    __shared__ unsigned long long s_wtab[3 * MAXE], s_stab[3 * MAXE];
    __shared__ unsigned s_pidx[2][BM];
    __shared__ float s_gate[2][BM];
    __shared__ float s_as[W8 ? 2 : 1][BM];            /* W8A8: row act scales, by kb parity */
    __shared__ float s_red[W8 && GLU ? 4 : 1][BM];    /* W8A8 GLU: per-wave row amax */
    __shared__ float s_fs[W8 && GLU ? BM : 1];        /* W8A8 GLU: output row scales */
#if MFP_DEPI
    __shared__ __align__(16) float s_epi[GLU ? 1 : 8 * 32 * 36];
#endif

    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned wm = wave >> 2, wn = wave & 3u, l32 = lane & 31u, hl = lane >> 5;
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

    /* A staging map. bf16: two 16-byte chunks per thread (rows tid>>3 and 64 + tid>>3, chunk
     * tid&7 of 8). fp8: one chunk (row tid>>2, chunk tid&3 of 4). B: two fp8 chunks (rows tid>>2
     * and 128 + tid>>2, chunk tid&3). */
    constexpr int NA = W8 ? 1 : 2;

    /* ---- producer state ---- */
    /* The producer walks items p_it = blockIdx + j*grid. `pre` is the item after the one being
     * loaded, decoded (LDS only), with its source rows prefetched every step; an item switch reads
     * the previous step's copy (`snap_*`), which has landed. */
    unsigned p_it = blockIdx.x, p_kt = 0, p_seq = 0;
    auto decode = [&](unsigned it) {
        return mfp_decode<GLU>(mfp_remap(it, total), tn, bp, s_meta, n_exp, s_wtab, s_stab);
    };
    MfpItem pre = decode(p_it);
    /* GLU: tokens of this thread's A rows (W8A8: one row, whose scale the row's chunk-0 thread
     * also stages). */
    unsigned pre_src[2] = {0u, 0u}, snap_src[2] = {0u, 0u};
    unsigned pre_pidx = 0u, snap_pidx = 0u; /* DOWN: row tid's partidx / gate */
    float pre_gate = 0.0f, snap_gate = 0.0f;
    auto arow_of = [&](int i) -> unsigned {
        return W8 ? (tid >> 2) : ((tid + (unsigned)i * NTHR) >> 3);
    };
    auto prefetch = [&]() {
        if constexpr (GLU) {
#pragma unroll
            for (int i = 0; i < NA; i++) {
                const unsigned r = arow_of(i);
                pre_src[i] = as_glob(row_token)[pre.row0 + (r < pre.valid ? r : 0u)];
            }
        } else {
            const unsigned ix = pre.row0 + (tid < pre.valid ? tid : 0u);
            pre_pidx = as_glob(row_partidx)[ix];
            pre_gate = as_glob(row_gate)[ix];
        }
    };
    auto snapshot = [&]() {
#pragma unroll
        for (int i = 0; i < 2; i++) snap_src[i] = pre_src[i];
        snap_pidx = pre_pidx;
        snap_gate = pre_gate;
    };
    prefetch();
    /* Every fetch is unconditional (a guarded load makes the compiler drain vmcnt inside the
     * branch): absent A rows read row 0 of A and are zeroed at the commit by `amask`; the W8A8
     * row scale of an absent row is zeroed the same way (`smask`). */
    const unsigned char* aptr[2] = {A, A};
    const unsigned char* bptr[2];
    const float* sptr = a_scale; /* W8A8: row tid's scale at kb = 0 */
    unsigned amask = 0, smask = 0;
    unsigned m_pidx = PLOW_EXPERT_UNUSED; /* DOWN: row tid's metadata for the item being loaded */
    float m_gate = 0.0f;
    MfpItem pi = pre;
    /* Make `pre` the item being loaded and point `pre` at the one after it. */
    auto advance = [&]() {
        pi = pre;
        const MfpItem& m = pi;
#pragma unroll
        for (int i = 0; i < NA; i++) {
            const unsigned r = arow_of(i);
            const unsigned c = W8 ? (tid & 3u) : (tid & 7u);
            const bool live = r < m.valid && (!GLU || snap_src[i] != PLOW_EXPERT_UNUSED);
            const unsigned src = GLU ? snap_src[i] : m.row0 + r;
            aptr[i] = A + (size_t)(live ? src : 0u) * arow + c * 16u;
            amask = (amask & ~(1u << i)) | ((unsigned)live << i);
        }
        if constexpr (W8) {
            const bool live = amask & 1u;
            const unsigned src = GLU ? snap_src[0] : m.row0 + (tid >> 2);
            sptr = a_scale + (live ? (GLU ? (size_t)src : (size_t)src * KB) : 0u);
            smask = live;
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const unsigned ch = tid + (unsigned)i * NTHR;
            const unsigned rb = ch >> 2, cb = ch & 3u;
            bptr[i] = A + cb * 16u;
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
    snapshot(); /* one-time wait for the first item's rows */
    advance();
    MfpItem nxt = pi;

    /* Register stage s (s = step parity): A chunks, B chunks, W8A8 row scale, and the flags the
     * commit needs. */
    fp8v16 ra0[2], ra1[2], rb0[2], rb1[2];
    float sc0 = 0.f, sc1 = 0.f;
    bool mp0 = false, mp1 = false; /* stage carries an item's first step: commit row metadata */
    bool v0 = false, v1 = false;   /* stage holds a loaded step */
    unsigned am0 = 0, am1 = 0;     /* live A chunks (bit i), bit 2 = live W8A8 scale row */
    unsigned ms0 = 0, ms1 = 0, kp0 = 0, kp1 = 0;
    unsigned mpx0 = 0, mpx1 = 0;
    float mg0 = 0.f, mg1 = 0.f;

/* One producer k-step: snapshot + re-issue the next item's row prefetch, then the stage loads.
 * Unconditional (after the last item it re-reads valid addresses into a stage nobody commits),
 * so the waitcnt pass sees the same in-order count on every path. */
#define MFP_LOAD(RA, RB, SC, MP, MS, MPX, MG, V, AM, KP)                                          \
    {                                                                                          \
        snapshot();                                                                            \
        prefetch();                                                                            \
        const unsigned k0 = p_kt * BK;                                                         \
        if (MFP_ABL != 2) {                                                                    \
            _Pragma("unroll") for (int i = 0; i < NA; i++)                                      \
                RA[i] = ld_glob_fp8v16(aptr[i] + k0 * (W8 ? 1u : 2u));                         \
            _Pragma("unroll") for (int i = 0; i < 2; i++)                                       \
                RB[i] = ld_glob_fp8v16(bptr[i] + k0);                                          \
            if constexpr (W8)                                                                  \
                SC = as_glob(sptr)[(size_t)(p_kt >> 1) * (GLU ? a_pitch : 1u)];                \
        }                                                                                      \
        MP = (p_kt == 0u);                                                                     \
        KP = (p_kt >> 1) & 1u;                                                                 \
        AM = amask | (smask << 2);                                                             \
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

/* Stage-to-LDS commit, in pieces that interleave with the MFMAs of the step in flight:
 * 0/1 = A chunk 0/1 (W8A8: chunk 0 + the row scale), 2/3 = B chunk 0/1 (+ DOWN row metadata). */
#define MFP_COMMIT_PIECE(PC, BUF, RA, RB, SC, MP, MS, MPX, MG, AM, KP)                             \
    {                                                                                          \
        unsigned char* const la_ = lds + (BUF) * STAGE;                                        \
        unsigned char* const lb_ = la_ + A_BYTES;                                              \
        if ((PC) == 0 || ((PC) == 1 && !W8)) {                                                 \
            const unsigned r = arow_of(PC);                                                    \
            const fp8v16 v = ((AM) >> (PC)) & 1u ? RA[PC] : (fp8v16)(0u);                      \
            if constexpr (W8)                                                                  \
                *(fp8v16*)&la_[mfp_bcell(r, tid & 3u)] = v;                                    \
            else                                                                               \
                *(fp8v16*)&la_[mfp_cell(r, tid & 7u)] = v;                                     \
        }                                                                                      \
        if constexpr (W8) {                                                                    \
            if ((PC) == 1 && (tid & 3u) == 0u)                                                 \
                s_as[KP][tid >> 2] = ((AM) >> 2) & 1u ? (SC) : 0.0f;                           \
        }                                                                                      \
        if ((PC) >= 2) {                                                                       \
            const unsigned ch = tid + (unsigned)((PC) & 1) * NTHR;                             \
            *(fp8v16*)&lb_[mfp_bcell(ch >> 2, ch & 3u)] = RB[(PC) & 1];                        \
        }                                                                                      \
        if constexpr (!GLU) {                                                                  \
            if ((PC) == 3 && (MP) && tid < BM) {                                               \
                s_pidx[(MS) & 1u][tid] = (MPX);                                                \
                s_gate[(MS) & 1u][tid] = (MG);                                                 \
            }                                                                                  \
        }                                                                                      \
    }
#define MFP_COMMIT(BUF, ...)                                                                   \
    MFP_COMMIT_PIECE(0, BUF, __VA_ARGS__)                                                      \
    MFP_COMMIT_PIECE(1, BUF, __VA_ARGS__)                                                      \
    MFP_COMMIT_PIECE(2, BUF, __VA_ARGS__)                                                      \
    MFP_COMMIT_PIECE(3, BUF, __VA_ARGS__)

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

/* One k-step's MFMAs out of LDS buffer BUF, with the commit of the next step's register stage
 * into buffer CBUF interleaved (CV: that stage holds a loaded step). */
#define MFP_COMPUTE(BUF, CBUF, CV, ...)                                                        \
    {                                                                                          \
        const unsigned char* const la = lds + (BUF) * STAGE;                                   \
        const unsigned char* const lb = la + A_BYTES;                                          \
        const bool live0 = MFP_ABL != 1 && wm * 64u < ci.valid;                                \
        /* Each 128-K block's first MFMA starts from zero, so acc is dead between promotions. */ \
        const f32x16 zacc = (f32x16)(0.0f);                                                    \
        const bool live1 = MFP_ABL != 1 && wm * 64u + 32u < ci.valid;                          \
        if constexpr (W8) {                                                                    \
            fp8v32 af[2], bf[2];                                                               \
            _Pragma("unroll") for (int i = 0; i < 2; i++)                                       \
                af[i] = mfp_frag32(la, wm * 64u + (unsigned)i * 32u + l32, hl);                \
            _Pragma("unroll") for (int j = 0; j < 2; j++) {                                     \
                const unsigned brow = (GLU ? (unsigned)j * 128u + wn * 32u                      \
                                           : wn * 64u + (unsigned)j * 32u) + l32;               \
                bf[j] = mfp_frag32(lb, brow, hl);                                              \
            }                                                                                  \
            __builtin_amdgcn_s_setprio(1);                                                     \
            if (live0) {                                                                       \
                acc[0][0] = plow_mfma_fp8_32x32(af[0], bf[0], (BUF) ? acc[0][0] : zacc);                      \
                acc[0][1] = plow_mfma_fp8_32x32(af[0], bf[1], (BUF) ? acc[0][1] : zacc);                      \
            }                                                                                  \
            if (live1) {                                                                       \
                acc[1][0] = plow_mfma_fp8_32x32(af[1], bf[0], (BUF) ? acc[1][0] : zacc);                      \
                acc[1][1] = plow_mfma_fp8_32x32(af[1], bf[1], (BUF) ? acc[1][1] : zacc);                      \
            }                                                                                  \
            __builtin_amdgcn_s_setprio(0);                                                     \
            if (CV) MFP_COMMIT(CBUF, __VA_ARGS__)                                              \
        } else {                                                                               \
            _Pragma("unroll") for (int q = 0; q < (int)(BK / MFMA_K); q++) {                    \
                const unsigned oct = (unsigned)q * 2u + hl;                                    \
                bf16x8 af[2], bf[2];                                                           \
                _Pragma("unroll") for (int i = 0; i < 2; i++) {                                 \
                    const unsigned r = wm * 64u + (unsigned)i * 32u + l32;                     \
                    __builtin_memcpy(&af[i], &la[mfp_cell(r, oct)], 16);                       \
                }                                                                              \
                _Pragma("unroll") for (int j = 0; j < 2; j++) {                                 \
                    const unsigned brow = (GLU ? (unsigned)j * 128u + wn * 32u                  \
                                               : wn * 64u + (unsigned)j * 32u) + l32;           \
                    unsigned w2[2];                                                            \
                    __builtin_memcpy(w2, &lb[mfp_bcell(brow, (unsigned)q) + hl * 8u], 8);      \
                    bf[j] = mfp_fp8x8_bf16(w2[0], w2[1]);                                      \
                }                                                                              \
                __builtin_amdgcn_s_setprio(1);                                                 \
                if (live0) {                                                                   \
                    acc[0][0] = plow_mfma_bf16_32x32(af[0], bf[0], ((BUF) || q) ? acc[0][0] : zacc);                 \
                    acc[0][1] = plow_mfma_bf16_32x32(af[0], bf[1], ((BUF) || q) ? acc[0][1] : zacc);                 \
                }                                                                              \
                if (live1) {                                                                   \
                    acc[1][0] = plow_mfma_bf16_32x32(af[1], bf[0], ((BUF) || q) ? acc[1][0] : zacc);                 \
                    acc[1][1] = plow_mfma_bf16_32x32(af[1], bf[1], ((BUF) || q) ? acc[1][1] : zacc);                 \
                }                                                                              \
                __builtin_amdgcn_s_setprio(0);                                                 \
                if (CV) MFP_COMMIT_PIECE(q, CBUF, __VA_ARGS__)                                 \
            }                                                                                  \
        }                                                                                      \
    }

    /* Weight-scale rows: GLU's 128-wide tile is one row for gate and up; DOWN's wave covers 64. */
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

    auto promote = [&](unsigned kb) {
        if constexpr (W8) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                float as[16];
#pragma unroll
                for (int el = 0; el < 16; el++)
                    as[el] = s_as[kb & 1u][wm * 64u + (unsigned)i * 32u + mfma_acc_m(lane, el)];
#pragma unroll
                for (int j = 0; j < 2; j++) {
#pragma unroll
                    for (int el = 0; el < 16; el++)
                        accf[i][j][el] = fmaf(acc[i][j][el], as[el] * bs[j], accf[i][j][el]);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    accf[i][j] += acc[i][j] * bs[j];
                }
        }
    };

    auto epilogue = [&]() {
        if (ci.valid) {
            if constexpr (GLU && W8) {
                /* SwiGLU in f32, then the 1x128 output quant: the tile's 128 columns are the four
                 * wn waves' 32. Lane-reduce within a wave, then LDS across the four. */
                /* The SwiGLU value overwrites the gate accumulator (register budget). */
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        accf[i][0][el] = moe_glu(accf[i][0][el], accf[i][1][el], act, beta, lbeta);
                        float m = fabsf(accf[i][0][el]);
                        m = fmaxf(m, __shfl_xor(m, 1));
                        m = fmaxf(m, __shfl_xor(m, 2));
                        m = fmaxf(m, __shfl_xor(m, 4));
                        m = fmaxf(m, __shfl_xor(m, 8));
                        m = fmaxf(m, __shfl_xor(m, 16));
                        if (l32 == 0u)
                            s_red[wn][wm * 64u + 32u * (unsigned)i + mfma_acc_m(lane, el)] = m;
                    }
                __syncthreads();
                if (tid < BM) {
                    float amax = 1e-10f; /* AITER floors amax, not the scale */
#pragma unroll
                    for (int w = 0; w < 4; w++) amax = fmaxf(amax, s_red[w][tid]);
                    const float s = amax * (1.0f / 448.0f);
                    s_fs[tid] = s;
                    if (tid < ci.valid)
                        st_act<float>(as_glob(fs_out) + (size_t)(ci.row0 + tid) * (N >> 7) +
                                          (ci.n0 >> 7),
                                      s);
                }
                __syncthreads();
                unsigned char* const fq = (unsigned char*)Cout;
                const unsigned nn = ci.n0 + wn * 32u + l32;
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr = wm * 64u + (unsigned)i * 32u + mfma_acc_m(lane, el);
                        const float inv = 1.0f / s_fs[rr];
                        const float xv = accf[i][0][el] * inv;
                        const unsigned q = __builtin_amdgcn_cvt_pk_fp8_f32(xv, 0.0f, 0u, false);
                        if (rr < ci.valid)
                            st_act1_u8(as_glob(fq) + (size_t)(ci.row0 + rr) * N + nn,
                                       (unsigned char)q);
                    }
            } else if constexpr (GLU) {
                bf16* const fu = (bf16*)Cout;
                const unsigned nn = ci.n0 + wn * 32u + l32;
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
#if MFP_DEPI
                typedef float f4e __attribute__((ext_vector_type(4)));
                float* const slab = s_epi + wave * (32u * 36u);
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        __builtin_amdgcn_wave_barrier();
#pragma unroll
                        for (int el = 0; el < 16; el++) {
                            const unsigned r = mfma_acc_m(lane, el);
                            const unsigned rr = wm * 64u + (unsigned)i * 32u + r;
                            const float gv = rr < ci.valid ? s_gate[par][rr] : 0.0f;
                            slab[r * 36u + l32] = gv * accf[i][j][el];
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
                            const unsigned nn =
                                ci.n0 + wn * 64u + (unsigned)j * 32u + (lane & 7u) * 4u;
                            __builtin_nontemporal_store(
                                v, (PLOW_GLOB f4e*)&part[(size_t)pidx * N + nn]);
                        }
                    }
#else
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const unsigned nn = ci.n0 + wn * 64u + (unsigned)j * 32u + l32;
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
#endif
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) accf[i][j] = (f32x16)(0.0f);
    };

    /* Prologue: steps 0 and 1 into the two register stages, step 0 into LDS buffer 0. */
    MFP_LOAD(ra0, rb0, sc0, mp0, ms0, mpx0, mg0, v0, am0, kp0)
    MFP_LOAD(ra1, rb1, sc1, mp1, ms1, mpx1, mg1, v1, am1, kp1)
    MFP_COMMIT(0u, ra0, rb0, sc0, mp0, ms0, mpx0, mg0, am0, kp0)
    __syncthreads();

    /* NT is even (K % 128 == 0), and every item starts on an even flattened step, so buffer 0 /
     * stage 0 always holds an even k-tile and buffer 1 / stage 1 an odd one. */
    for (;;) {
        /* EVEN step: LDS 0 = (c_it, c_kt); stage 1 = next step; stage 0 is free. */
        MFP_LOAD(ra0, rb0, sc0, mp0, ms0, mpx0, mg0, v0, am0, kp0)
        load_scales(c_kt >> 1);
        MFP_COMPUTE(0u, 1u, true, ra1, rb1, sc1, mp1, ms1, mpx1, mg1, am1, kp1)
        __syncthreads();
        c_kt++;
        /* ODD step: LDS 1 = (c_it, c_kt); stage 0 = next step; stage 1 is free. */
        MFP_LOAD(ra1, rb1, sc1, mp1, ms1, mpx1, mg1, v1, am1, kp1)
        MFP_COMPUTE(1u, 0u, v0, ra0, rb0, sc0, mp0, ms0, mpx0, mg0, am0, kp0)
        promote(c_kt >> 1);
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

/* ------------------------------------------------------------------------------------------------
 * MFP_DMA: the same items, fragment maps, MFMA chains, promotion and epilogues, but every k-step
 * operand (A rows, B rows, the W8A8 row scales, the step's weight scales) is staged with
 * global_load_lds straight into a D-deep LDS ring, and so is each item's row metadata (one item
 * ahead). No register stages: the producer runs D-1 k-steps ahead and the consumer waits for its
 * step with a fixed vmcnt -- every wave issues exactly OPS ops per step, so the steps after the
 * one being consumed always account for (D-2)*OPS outstanding ops. Extra ops (metadata) and the
 * DOWN epilogue's stores only make that wait stricter, never weaker. Barriers are raw s_barrier
 * after lgkmcnt(0): __syncthreads' release fence would drain the LDS DMA (vmcnt(0)). */
#ifndef MFP_DMA
#define MFP_DMA 1
#endif
#ifndef MFP_D8
#define MFP_D8 4
#endif
#ifndef MFP_D16
#define MFP_D16 3
#endif

__device__ __forceinline__ void mfp_bar() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
}

template <int N>
__device__ __forceinline__ void mfp_vmwait() {
    asm volatile("s_waitcnt vmcnt(%0)" ::"i"(N) : "memory");
}

/* LDS DMA hidden from the compiler (inline asm): its LDS-DMA alias tracking otherwise makes the
 * fragment reads wait for the k-step just issued, i.e. no lookahead at all. The waits are the
 * explicit mfp_vmwait ones. Sound because no compiler-tracked global load is in flight across
 * the k-loop (its only vector-memory ops are these and the epilogue stores).
 * Lane l writes LDS [dst + W l, +W), dst wave-uniform. */
template <int W>
__device__ __forceinline__ void mfp_dma(const void* src, void* dst) {
#ifdef __HIP_DEVICE_COMPILE__
    const unsigned m0 = __builtin_amdgcn_readfirstlane((unsigned)(size_t)dst);
    if constexpr (W == 16)
        asm volatile("s_mov_b32 m0, %1\n\tglobal_load_lds_dwordx4 %0, off" ::"v"(src), "s"(m0)
                     : "memory", "m0");
    else
        asm volatile("s_mov_b32 m0, %1\n\tglobal_load_lds_dword %0, off" ::"v"(src), "s"(m0)
                     : "memory", "m0");
#else
    (void)src;
    (void)dst;
#endif
}

template <bool GLU, bool W8, int D>
__device__ __forceinline__ void mfp_body_dma(void* __restrict__ Cout, const void* __restrict__ Av,
                                             const unsigned long long* __restrict__ wtab,
                                             const unsigned long long* __restrict__ stab,
                                             const int* __restrict__ meta,
                                             const unsigned* __restrict__ row_token,
                                             const unsigned* __restrict__ row_partidx,
                                             const float* __restrict__ row_gate,
                                             const float* __restrict__ a_scale,
                                             float* __restrict__ fs_out, unsigned a_pitch,
                                             unsigned N, unsigned K, unsigned n_exp, unsigned act,
                                             float beta, float lbeta) {
    using namespace mfp;
    static_assert(D >= 3, "the producer needs a free stage beyond the D-2 in flight");
    static_assert(GLU || D <= 4, "an item's metadata lands >= NT steps ahead; DOWN has NT = I/64 >= 4");
    const unsigned char* const A = (const unsigned char*)Av;
    const size_t arow = W8 ? (size_t)K : (size_t)K * 2u;
    constexpr unsigned DA_BYTES = W8 ? BM * BK : BM * BK * 2u; /* A k-tile */
    constexpr unsigned AS_OFF = DA_BYTES + B_BYTES;             /* [BM] f32 row scales (W8A8) */
    constexpr unsigned WS_OFF = AS_OFF + BM * 4u;               /* [64] f32: lanes 0/1 used */
    constexpr unsigned DSTAGE = WS_OFF + 256u;
    constexpr int NA = W8 ? 1 : 2;
    constexpr int OPS = NA + 2 + 1;
    __shared__ __align__(16) unsigned char lds[D * DSTAGE];
    __shared__ __align__(16) unsigned char s_junk[256];
    __shared__ int bp[MAXE + 1];
    __shared__ int s_meta[3 * MAXE + 1];
    __shared__ unsigned long long s_wtab[3 * MAXE], s_stab[3 * MAXE];
    /* Per-item row metadata ring (by item sequence & 3): GLU tokens, DOWN partidx + gate. */
    __shared__ __align__(16) unsigned s_mt[4][BM];
    __shared__ __align__(16) float s_mg[GLU ? 1 : 4][BM];
    __shared__ float s_red[W8 && GLU ? 4 : 1][BM];
    __shared__ float s_fs[W8 && GLU ? BM : 1];

    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned wm = wave >> 2, wn = wave & 3u, l32 = lane & 31u, hl = lane >> 5;
    const int* tilep = meta + 2u * n_exp;

    for (unsigned i = tid; i < 3u * n_exp + 1u; i += NTHR) s_meta[i] = meta[i];
    for (unsigned i = tid; i < 3u * n_exp; i += NTHR) {
        s_wtab[i] = wtab[i];
        s_stab[i] = stab[i];
    }
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
    auto decode = [&](unsigned it) {
        return mfp_decode<GLU>(mfp_remap(it, total), tn, bp, s_meta, n_exp, s_wtab, s_stab);
    };

    /* Item metadata DMA into ring slot `slot`: waves 0/1 the 128 tokens (GLU) or partidx (DOWN),
     * waves 2/3 the DOWN gates. Rows past `valid` re-read the item's first row. */
    auto meta_dma = [&](const MfpItem& m, unsigned slot) {
        if (wave < 2u) {
            const unsigned r = wave * 64u + lane;
            const unsigned* src = (GLU ? row_token : row_partidx) + m.row0 + (r < m.valid ? r : 0u);
            mfp_dma<4>(src, &s_mt[slot][wave * 64u]);
        } else if (!GLU && wave < 4u) {
            const unsigned r = (wave - 2u) * 64u + lane;
            mfp_dma<4>(row_gate + m.row0 + (r < m.valid ? r : 0u), &s_mg[slot][(wave - 2u) * 64u]);
        }
    };

    /* ---- producer ---- */
    unsigned p_it = blockIdx.x, p_kt = 0, p_seq = 0, p_stage = 0;
    MfpItem pm = decode(p_it), pre = pm;
    const unsigned char* aptr[NA];
    const unsigned char* bptr[2];
    const unsigned char* xptr = A; /* this wave's dword op: row / weight scales or junk */
    unsigned xstride = 0;          /* bytes per kb */
    unsigned xoff = 0;             /* stage offset of its destination when xstage, else junk */
    bool xstage = false;
    /* Point the per-lane sources at item `pm`, whose row metadata is in slot p_seq & 3. */
    auto setup = [&]() {
        const MfpItem& m = pm;
        const unsigned slot = p_seq & 3u;
#pragma unroll
        for (int i = 0; i < NA; i++) {
            const unsigned r = W8 ? wave * 16u + (lane >> 2) : (2u * wave + (unsigned)i) * 8u + (lane >> 3);
            const unsigned c = W8 ? (lane & 3u) ^ ((r >> 2) & 3u) : (lane & 7u) ^ (r & 7u);
            unsigned src = m.row0 + r;
            bool live = r < m.valid;
            if constexpr (GLU) {
                src = s_mt[slot][r];
                live = live && src != PLOW_EXPERT_UNUSED;
            }
            aptr[i] = A + (size_t)(live ? src : 0u) * arow + c * 16u;
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const unsigned r = (2u * wave + (unsigned)i) * 16u + (lane >> 2);
            const unsigned c = (lane & 3u) ^ ((r >> 2) & 3u);
            bptr[i] = A + c * 16u;
            if (m.valid) {
                if constexpr (GLU) {
                    const unsigned char* w = r < 128u ? m.w0 : m.w1;
                    bptr[i] = w + (size_t)(m.n0 + (r & 127u)) * K + c * 16u;
                } else {
                    bptr[i] = m.w0 + (size_t)(m.n0 + r) * K + c * 16u;
                }
            }
        }
        xptr = A;
        xstride = 0;
        xstage = false;
        if (W8 && wave < 2u && m.valid) {
            const unsigned r = wave * 64u + lane;
            if constexpr (GLU) {
                const unsigned src = s_mt[slot][r];
                const bool live = r < m.valid && src != PLOW_EXPERT_UNUSED;
                xptr = (const unsigned char*)(a_scale + (live ? src : 0u));
                xstride = a_pitch * 4u;
            } else {
                xptr = (const unsigned char*)(a_scale + (size_t)(m.row0 + (r < m.valid ? r : 0u)) * KB);
                xstride = 4u;
            }
            xoff = AS_OFF + wave * 256u;
            xstage = true;
        } else if (wave == 2u && m.valid) {
            const unsigned j = lane & 1u;
            const float* s = GLU ? (j ? m.s1 : m.s0) + (size_t)(m.n0 >> 7) * KB
                                 : m.s0 + (size_t)((m.n0 >> 7) + j) * KB;
            xptr = (const unsigned char*)s;
            xstride = 4u;
            xoff = WS_OFF;
            xstage = true;
        }
    };
    /* One k-step: OPS ops per wave, unconditionally (past the last item it re-reads the last
     * valid addresses into the free stage). An item's first step also stages the next item's
     * row metadata. */
    auto issue = [&]() {
        unsigned char* const st = lds + p_stage * DSTAGE;
        const unsigned k0 = p_kt * BK;
#pragma unroll
        for (int i = 0; i < NA; i++)
            mfp_dma<16>(aptr[i] + k0 * (W8 ? 1u : 2u), st + (W8 ? wave : 2u * wave + (unsigned)i) * 1024u);
#pragma unroll
        for (int i = 0; i < 2; i++) mfp_dma<16>(bptr[i] + k0, st + DA_BYTES + (2u * wave + (unsigned)i) * 1024u);
        mfp_dma<4>(xptr + (size_t)(p_kt >> 1) * xstride, xstage ? st + xoff : s_junk);
        if (++p_stage == (unsigned)D) p_stage = 0;
        if (p_it < total && ++p_kt == NT) {
            p_kt = 0;
            p_it += gridDim.x;
            p_seq++;
            if (p_it < total) {
                pm = pre;
                setup();
                if (p_it + gridDim.x < total) {
                    pre = decode(p_it + gridDim.x);
                    meta_dma(pre, (p_seq + 1u) & 3u);
                }
            }
        }
    };

    /* Prologue: item 0's metadata (waited), item 1's in flight, then D-1 steps. */
    meta_dma(pm, 0u);
    mfp_vmwait<0>();
    mfp_bar();
    setup();
    if (p_it + gridDim.x < total) {
        pre = decode(p_it + gridDim.x);
        meta_dma(pre, 1u);
    }
#pragma unroll
    for (int s = 0; s < D - 1; s++) issue();

    /* ---- consumer ---- */
    unsigned c_it = blockIdx.x, c_kt = 0, c_seq = 0, c_stage = 0;
    MfpItem ci = pm;
    ci = decode(c_it);
    f32x16 acc[2][2], accf[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            acc[i][j] = (f32x16)(0.0f);
            accf[i][j] = (f32x16)(0.0f);
        }

    auto compute = [&](const unsigned char* la, bool first) {
        const unsigned char* const lb = la + DA_BYTES;
        const bool live0 = wm * 64u < ci.valid, live1 = wm * 64u + 32u < ci.valid;
        const f32x16 zacc = (f32x16)(0.0f);
        if constexpr (W8) {
            fp8v32 af[2], bf[2];
#pragma unroll
            for (int i = 0; i < 2; i++) af[i] = mfp_frag32(la, wm * 64u + (unsigned)i * 32u + l32, hl);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned brow = (GLU ? (unsigned)j * 128u + wn * 32u : wn * 64u + (unsigned)j * 32u) + l32;
                bf[j] = mfp_frag32(lb, brow, hl);
            }
            __builtin_amdgcn_s_setprio(1);
            if (live0) {
                acc[0][0] = plow_mfma_fp8_32x32(af[0], bf[0], first ? zacc : acc[0][0]);
                acc[0][1] = plow_mfma_fp8_32x32(af[0], bf[1], first ? zacc : acc[0][1]);
            }
            if (live1) {
                acc[1][0] = plow_mfma_fp8_32x32(af[1], bf[0], first ? zacc : acc[1][0]);
                acc[1][1] = plow_mfma_fp8_32x32(af[1], bf[1], first ? zacc : acc[1][1]);
            }
            __builtin_amdgcn_s_setprio(0);
        } else {
#pragma unroll
            for (int q = 0; q < (int)(BK / MFMA_K); q++) {
                const unsigned oct = (unsigned)q * 2u + hl;
                bf16x8 af[2], bf[2];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const unsigned r = wm * 64u + (unsigned)i * 32u + l32;
                    __builtin_memcpy(&af[i], &la[mfp_cell(r, oct)], 16);
                }
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const unsigned brow = (GLU ? (unsigned)j * 128u + wn * 32u : wn * 64u + (unsigned)j * 32u) + l32;
                    unsigned w2[2];
                    __builtin_memcpy(w2, &lb[mfp_bcell(brow, (unsigned)q) + hl * 8u], 8);
                    bf[j] = mfp_fp8x8_bf16(w2[0], w2[1]);
                }
                const bool z = first && q == 0;
                __builtin_amdgcn_s_setprio(1);
                if (live0) {
                    acc[0][0] = plow_mfma_bf16_32x32(af[0], bf[0], z ? zacc : acc[0][0]);
                    acc[0][1] = plow_mfma_bf16_32x32(af[0], bf[1], z ? zacc : acc[0][1]);
                }
                if (live1) {
                    acc[1][0] = plow_mfma_bf16_32x32(af[1], bf[0], z ? zacc : acc[1][0]);
                    acc[1][1] = plow_mfma_bf16_32x32(af[1], bf[1], z ? zacc : acc[1][1]);
                }
                __builtin_amdgcn_s_setprio(0);
            }
        }
    };

    auto promote = [&](const unsigned char* st) {
        const float* const ws = (const float*)(st + WS_OFF);
        float bs[2];
        if constexpr (GLU) {
            bs[0] = ws[0];
            bs[1] = ws[1];
        } else {
            bs[0] = bs[1] = ws[wn >> 1];
        }
        if constexpr (W8) {
            const float* const sa = (const float*)(st + AS_OFF);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                float as[16];
#pragma unroll
                for (int el = 0; el < 16; el++) as[el] = sa[wm * 64u + (unsigned)i * 32u + mfma_acc_m(lane, el)];
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int el = 0; el < 16; el++)
                        accf[i][j][el] = fmaf(acc[i][j][el], as[el] * bs[j], accf[i][j][el]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) accf[i][j] += acc[i][j] * bs[j];
        }
    };

    auto epilogue = [&]() {
        const unsigned slot = c_seq & 3u;
        if (ci.valid) {
            if constexpr (GLU) {
                /* Pad rows (token UNUSED) are zero A rows on the interpreter: accf = +0. */
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr = wm * 64u + (unsigned)i * 32u + mfma_acc_m(lane, el);
                        if (s_mt[slot][rr] == PLOW_EXPERT_UNUSED) {
                            accf[i][0][el] = 0.0f;
                            accf[i][1][el] = 0.0f;
                        }
                    }
            }
            if constexpr (GLU && W8) {
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        accf[i][0][el] = moe_glu(accf[i][0][el], accf[i][1][el], act, beta, lbeta);
                        float m = fabsf(accf[i][0][el]);
                        m = fmaxf(m, __shfl_xor(m, 1));
                        m = fmaxf(m, __shfl_xor(m, 2));
                        m = fmaxf(m, __shfl_xor(m, 4));
                        m = fmaxf(m, __shfl_xor(m, 8));
                        m = fmaxf(m, __shfl_xor(m, 16));
                        if (l32 == 0u) s_red[wn][wm * 64u + 32u * (unsigned)i + mfma_acc_m(lane, el)] = m;
                    }
                mfp_bar();
                if (tid < BM) {
                    float amax = 1e-10f;
#pragma unroll
                    for (int w = 0; w < 4; w++) amax = fmaxf(amax, s_red[w][tid]);
                    const float s = amax * (1.0f / 448.0f);
                    s_fs[tid] = s;
                    if (tid < ci.valid)
                        st_act<float>(as_glob(fs_out) + (size_t)(ci.row0 + tid) * (N >> 7) + (ci.n0 >> 7), s);
                }
                mfp_bar();
                unsigned char* const fq = (unsigned char*)Cout;
                const unsigned nn = ci.n0 + wn * 32u + l32;
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr = wm * 64u + (unsigned)i * 32u + mfma_acc_m(lane, el);
                        const float inv = 1.0f / s_fs[rr];
                        const float xv = accf[i][0][el] * inv;
                        const unsigned q = __builtin_amdgcn_cvt_pk_fp8_f32(xv, 0.0f, 0u, false);
                        if (rr < ci.valid) st_act1_u8(as_glob(fq) + (size_t)(ci.row0 + rr) * N + nn, (unsigned char)q);
                    }
            } else if constexpr (GLU) {
                bf16* const fu = (bf16*)Cout;
                const unsigned nn = ci.n0 + wn * 32u + l32;
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int el = 0; el < 16; el++) {
                        const unsigned rr = wm * 64u + (unsigned)i * 32u + mfma_acc_m(lane, el);
                        if (rr < ci.valid)
                            st_act1(&fu[(size_t)(ci.row0 + rr) * N + nn],
                                    f2bf(moe_glu(accf[i][0][el], accf[i][1][el], act, beta, lbeta)));
                    }
            } else {
                float* const part = (float*)Cout;
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const unsigned nn = ci.n0 + wn * 64u + (unsigned)j * 32u + l32;
#pragma unroll
                        for (int el = 0; el < 16; el++) {
                            const unsigned rr = wm * 64u + (unsigned)i * 32u + mfma_acc_m(lane, el);
                            if (rr >= ci.valid) continue;
                            const unsigned pidx = s_mt[slot][rr];
                            if (pidx == PLOW_EXPERT_UNUSED) continue;
                            __builtin_nontemporal_store(s_mg[slot][rr] * accf[i][j][el],
                                                        (PLOW_GLOB float*)&part[(size_t)pidx * N + nn]);
                        }
                    }
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) accf[i][j] = (f32x16)(0.0f);
    };

    for (;;) {
        mfp_vmwait<(D - 2) * OPS>();
        mfp_bar();
        const unsigned char* const st = lds + c_stage * DSTAGE;
        issue();
        compute(st, (c_kt & 1u) == 0u);
        if (c_kt & 1u) promote(st);
        if (++c_stage == (unsigned)D) c_stage = 0;
        if (c_kt == NT - 1u) {
            epilogue();
            c_it += gridDim.x;
            c_seq++;
            c_kt = 0;
            if (c_it >= total) break;
            ci = decode(c_it);
        } else {
            c_kt++;
        }
    }
    mfp_vmwait<0>();
}

/* Measured (T=8192, bench best-of-10, ms, register-staged vs DMA): W8A8 GLU 0.529 vs 0.503,
 * W8A8 down 0.493 vs 0.518, W8A16 GLU 0.656 vs 0.670, W8A16 down 0.547 vs 0.580. So only the
 * W8A8 GLU takes the DMA body by default (MFP_DMA=2 forces it everywhere, 0 nowhere). */
#if MFP_DMA == 2
#define MFP_BODY(G, W) mfp_body_dma<G, W, (W) ? MFP_D8 : MFP_D16>
#elif MFP_DMA == 1
#define MFP_BODY(G, W) mfp_body_sel<G, W>
template <bool G, bool W, typename... Args>
__device__ __forceinline__ void mfp_body_sel(Args... a) {
    if constexpr (G && W)
        mfp_body_dma<G, W, MFP_D8>(a...);
    else
        mfp_body<G, W>(a...);
}
#else
#define MFP_BODY(G, W) mfp_body<G, W>
#endif

#define MFP_GLU_ARGS                                                                           \
    void *__restrict__ out, const void *__restrict__ x,                                        \
        const unsigned long long *__restrict__ wtab, const unsigned long long *__restrict__ stab, \
        const int *__restrict__ meta, const unsigned *__restrict__ row_token,                    \
        const float *__restrict__ x_scale, float *__restrict__ out_scale, unsigned inter,        \
        unsigned hidden, unsigned experts, unsigned act, float beta, float lbeta,                \
        unsigned x_pitch, unsigned reserved
#define MFP_DOWN_ARGS                                                                          \
    float *__restrict__ part, const void *__restrict__ fu,                                     \
        const unsigned long long *__restrict__ wtab, const unsigned long long *__restrict__ stab, \
        const int *__restrict__ meta, const unsigned *__restrict__ row_partidx,                  \
        const float *__restrict__ row_gate, const float *__restrict__ fu_scale, unsigned hidden, \
        unsigned inter, unsigned experts, unsigned reserved

extern "C" __global__ __launch_bounds__(512, 1) void plow_moe_fp8_glu_pf_gfx950(MFP_GLU_ARGS) {
    (void)reserved;
    if (experts > mfp::MAXE || inter % 128u != 0u || hidden % 128u != 0u) return;
    MFP_BODY(true, false)(out, x, wtab, stab, meta, row_token, nullptr, nullptr, nullptr,
                          nullptr, 0u, inter, hidden, experts, act, beta, lbeta);
}

extern "C" __global__ __launch_bounds__(512, 1) void plow_moe_w8a8_glu_pf_gfx950(MFP_GLU_ARGS) {
    (void)reserved;
    if (experts > mfp::MAXE || inter % 128u != 0u || hidden % 128u != 0u) return;
    MFP_BODY(true, true)(out, x, wtab, stab, meta, row_token, nullptr, nullptr, x_scale,
                         out_scale, x_pitch, inter, hidden, experts, act, beta, lbeta);
}

extern "C" __global__ __launch_bounds__(512, 1) void plow_moe_fp8_down_pf_gfx950(MFP_DOWN_ARGS) {
    (void)reserved;
    (void)fu_scale;
    if (experts > mfp::MAXE || hidden % 256u != 0u || inter % 128u != 0u || inter < 256u) return;
    MFP_BODY(false, false)(part, fu, wtab, stab, meta, nullptr, row_partidx, row_gate, nullptr,
                           nullptr, 0u, hidden, inter, experts, 0u, 0.0f, 0.0f);
}

extern "C" __global__ __launch_bounds__(512, 1) void plow_moe_w8a8_down_pf_gfx950(MFP_DOWN_ARGS) {
    (void)reserved;
    if (experts > mfp::MAXE || hidden % 256u != 0u || inter % 128u != 0u || inter < 256u) return;
    MFP_BODY(false, true)(part, fu, wtab, stab, meta, nullptr, row_partidx, row_gate, fu_scale,
                          nullptr, 0u, hidden, inter, experts, 0u, 0.0f, 0.0f);
}
