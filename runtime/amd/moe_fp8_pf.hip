/* Dedicated gfx950 grouped-MoE prefill kernels for block-fp8 [128,128] experts with bf16
 * activations (W8A16): ops 85 (MoeGroupGluPf, i3=1), 86 (MoeGroupDownPf, i3=1) and the k=8 f32
 * combine (op 87), each a standalone object segment (PLOW_GLM_MOE_PF_LEAN).
 *
 * BIT-IDENTICAL to the interpreter arms (op_moe.h d_moe_group_pf_t / d_moe_combine_pf) by
 * construction: every output element sees the same v_mfma_f32_32x32x16_bf16 sequence (same
 * fragment map, K ascending in 16-steps), the same per-128-K promotion `accf += acc * bs`, the same
 * epilogue expression and, for the combine, the same f32 operand order. Only the tiling changes.
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
extern "C" __device__ unsigned plow_moe_fp8_pf_combine_k8_1 = 1;

namespace mfp {
constexpr unsigned BM = 128, BK = 64, NTHR = 512;
constexpr unsigned A_HALVES = BM * BK;   /* 16 KB */
constexpr unsigned B_ROWS = 256;
constexpr unsigned B_HALVES = B_ROWS * BK; /* 32 KB as bf16 */
constexpr unsigned STAGE = A_HALVES + B_HALVES;
constexpr unsigned MAXE = 512;
} // namespace mfp

/* LDS element offset of 8-half octet `oct` in row `row` of a BK=64 bf16 tile (16-byte XOR swizzle,
 * the same cell map op_moe.h uses). */
__device__ __forceinline__ unsigned mfp_cell(unsigned row, unsigned oct) {
    return row * mfp::BK + ((oct ^ (row & 7u)) << 3);
}

/* Per-item metadata the producer resolves and hands to the consumer. */
struct MfpItem {
    unsigned e, row0, valid, n0;
    const unsigned char* w0;
    const unsigned char* w1;
    const float* s0;
    const float* s1;
};

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
    __shared__ unsigned s_pidx[2][BM];
    __shared__ float s_gate[2][BM];

    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned wm = wave >> 2, wn = wave & 3u;
    const int* tilep = meta + 2u * n_exp;

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
    unsigned p_it = blockIdx.x, p_kt = 0, p_seq = 0;
    MfpItem pi = mfp_decode<GLU>(p_it, tn, bp, meta, n_exp, wtab, stab);
    const bf16* aptr[2];
    const unsigned char* bptr[2];
    unsigned m_pidx = PLOW_EXPERT_UNUSED; /* DOWN: row tid's metadata for the item being loaded */
    float m_gate = 0.0f;
    unsigned m_seq = 0;
    bool m_pending = false;
    MfpItem nxt = pi;

    auto resolve = [&](const MfpItem& m) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const unsigned ch = tid + (unsigned)i * NTHR;
            const unsigned r = ch >> 3, c = ch & 7u;
            aptr[i] = nullptr;
            if (r < m.valid) {
                if constexpr (GLU) {
                    const unsigned tok = row_token[m.row0 + r];
                    if (tok != PLOW_EXPERT_UNUSED) aptr[i] = A + (size_t)tok * K + c * 8u;
                } else {
                    aptr[i] = A + (size_t)(m.row0 + r) * K + c * 8u;
                }
            }
            const unsigned rb = ch >> 2, cb = ch & 3u;
            bptr[i] = nullptr;
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
            m_pidx = PLOW_EXPERT_UNUSED;
            m_gate = 0.0f;
            if (tid < BM && tid < m.valid) {
                m_pidx = row_partidx[m.row0 + tid];
                m_gate = row_gate[m.row0 + tid];
            }
        }
    };
    resolve(pi);

    bf16v8 ra0[2], ra1[2];
    fp8v16 rb0[2], rb1[2];
    bool mp0 = false, mp1 = false; /* stage carries an item's first step: commit row metadata */
    unsigned ms0 = 0, ms1 = 0;
    unsigned mpx0 = 0, mpx1 = 0;
    float mg0 = 0.f, mg1 = 0.f;

#define MFP_LOAD(RA, RB, MP, MS, MPX, MG)                                                        \
    {                                                                                          \
        const unsigned k0 = p_kt * BK;                                                         \
        _Pragma("unroll") for (int i = 0; i < 2; i++) {                                         \
            RA[i] = aptr[i] ? ld_glob8(as_glob(aptr[i] + k0)) : bf16v8_zero();                 \
            RB[i] = bptr[i] ? ld_glob_fp8v16(bptr[i] + k0) : (fp8v16)(0u);                      \
        }                                                                                      \
        MP = (p_kt == 0u);                                                                     \
        MS = p_seq;                                                                            \
        MPX = m_pidx;                                                                          \
        MG = m_gate;                                                                           \
        if (++p_kt == NT) {                                                                    \
            p_kt = 0;                                                                          \
            p_it += gridDim.x;                                                                 \
            p_seq++;                                                                           \
            if (p_it < total) {                                                                \
                pi = mfp_decode<GLU>(p_it, tn, bp, meta, n_exp, wtab, stab);                   \
                nxt = pi;                                                                      \
                resolve(pi);                                                                   \
            }                                                                                  \
        }                                                                                      \
    }

#define MFP_COMMIT(BUF, RA, RB, MP, MS, MPX, MG)                                                 \
    {                                                                                          \
        bf16* const la = lds + (BUF) * STAGE;                                                  \
        bf16* const lb = la + A_HALVES;                                                        \
        _Pragma("unroll") for (int i = 0; i < 2; i++) {                                         \
            const unsigned ch = tid + (unsigned)i * NTHR;                                       \
            const unsigned r = ch >> 3, c = ch & 7u;                                            \
            *(bf16v8*)&la[mfp_cell(r, c)] = RA[i];                                              \
            const unsigned rb = ch >> 2, cb = ch & 3u;                                          \
            bf16v8 lo, hi;                                                                      \
            fp8_to_bf16v8(RB[i], lo, hi);                                                       \
            *(bf16v8*)&lb[mfp_cell(rb, 2u * cb)] = lo;                                          \
            *(bf16v8*)&lb[mfp_cell(rb, 2u * cb + 1u)] = hi;                                     \
        }                                                                                      \
        if constexpr (!GLU) {                                                                  \
            if ((MP) && tid < BM) {                                                            \
                s_pidx[(MS) & 1u][tid] = (MPX);                                                \
                s_gate[(MS) & 1u][tid] = (MG);                                                 \
            }                                                                                  \
        }                                                                                      \
    }

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

    auto compute = [&](unsigned buf) {
        const bf16* const la = lds + buf * STAGE;
        const bf16* const lb = la + A_HALVES;
        const bool live0 = wm * 64u < ci.valid;
        const bool live1 = wm * 64u + 32u < ci.valid;
#pragma unroll
        for (unsigned q = 0; q < BK / MFMA_K; q++) {
            const unsigned oct = q * 2u + (lane >> 5);
            bf16x8 af[2], bf[2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const unsigned arow = wm * 64u + (unsigned)i * 32u + (lane & 31u);
                __builtin_memcpy(&af[i], &la[mfp_cell(arow, oct)], 16);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned brow = (GLU ? (unsigned)j * 128u + wn * 32u
                                           : wn * 64u + (unsigned)j * 32u) +
                                      (lane & 31u);
                __builtin_memcpy(&bf[j], &lb[mfp_cell(brow, oct)], 16);
            }
            __builtin_amdgcn_s_setprio(1);
            if (live0) {
                acc[0][0] = plow_mfma_bf16_32x32(af[0], bf[0], acc[0][0]);
                acc[0][1] = plow_mfma_bf16_32x32(af[0], bf[1], acc[0][1]);
            }
            if (live1) {
                acc[1][0] = plow_mfma_bf16_32x32(af[1], bf[0], acc[1][0]);
                acc[1][1] = plow_mfma_bf16_32x32(af[1], bf[1], acc[1][1]);
            }
            __builtin_amdgcn_s_setprio(0);
        }
    };

    /* Scale-block rows: GLU's 128-wide tile is one row for gate and up; DOWN's wave covers 64. */
    auto load_scales = [&](unsigned kb) {
        if (!ci.valid) return;
        if constexpr (GLU) {
            const unsigned nr = ci.n0 >> 7;
            bs[0] = ci.s0[(size_t)nr * KB + kb];
            bs[1] = ci.s1[(size_t)nr * KB + kb];
        } else {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned nr = (ci.n0 + wn * 64u + (unsigned)j * 32u) >> 7;
                bs[j] = ci.s0[(size_t)nr * KB + kb];
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
        if (ci.valid) {
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
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) accf[i][j] = (f32x16)(0.0f);
    };

    /* Prologue: steps 0 and 1 into the two register stages, step 0 into LDS buffer 0. */
    MFP_LOAD(ra0, rb0, mp0, ms0, mpx0, mg0)
    if (p_it < total) MFP_LOAD(ra1, rb1, mp1, ms1, mpx1, mg1)
    MFP_COMMIT(0u, ra0, rb0, mp0, ms0, mpx0, mg0)
    __syncthreads();

    /* NT is even (K % 128 == 0), and every item starts on an even flattened step, so buffer 0 /
     * stage 0 always holds an even k-tile and buffer 1 / stage 1 an odd one. */
    for (;;) {
        /* EVEN step: LDS 0 = (c_it, c_kt); stage 1 = next step; stage 0 is free. */
        if (p_it < total) MFP_LOAD(ra0, rb0, mp0, ms0, mpx0, mg0)
        load_scales(c_kt >> 1);
        compute(0u);
        MFP_COMMIT(1u, ra1, rb1, mp1, ms1, mpx1, mg1)
        __syncthreads();
        c_kt++;
        /* ODD step: LDS 1 = (c_it, c_kt); stage 0 = next step; stage 1 is free. */
        if (p_it < total) MFP_LOAD(ra1, rb1, mp1, ms1, mpx1, mg1)
        compute(1u);
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
        MFP_COMMIT(0u, ra0, rb0, mp0, ms0, mpx0, mg0)
        __syncthreads();
    }
#undef MFP_LOAD
#undef MFP_COMMIT
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

/* k=8 f32 combine: out[t][h] = bf16(0 + shared[t][h] + part[t*8+0][h] + ... + part[t*8+7][h]),
 * the interpreter's operand order (d_moe_combine_pf, null residual), all eight slots in flight. */
extern "C" __global__ __launch_bounds__(256, 2)
void plow_moe_fp8_combine_k8_gfx950(bf16* __restrict__ out, const bf16* __restrict__ shared,
                                    const float* __restrict__ part, unsigned hidden,
                                    unsigned topk, unsigned tokens, unsigned reserved) {
    (void)reserved;
    if (topk != 8u || (hidden & 7u) != 0u) return;
    const size_t vt = (size_t)tokens * hidden / 8u;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    typedef float f4 __attribute__((ext_vector_type(4)));
    const f4* pf = (const f4*)part;
    for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < vt; v += stride) {
        const size_t e = v * 8u;
        const unsigned tok = (unsigned)(e / hidden), h = (unsigned)(e - (size_t)tok * hidden);
        const bf16v8 vs = shared ? ld_glob8(as_glob(shared) + e) : bf16v8_zero();
        const size_t q0 = ((size_t)tok * 8u * hidden + h) / 4u, qs = hidden / 4u;
        f4 p[8][2];
#pragma unroll
        for (int s = 0; s < 8; s++) {
            p[s][0] = __builtin_nontemporal_load(&pf[q0 + s * qs]);
            p[s][1] = __builtin_nontemporal_load(&pf[q0 + s * qs + 1]);
        }
        float acc[8];
#pragma unroll
        for (int c = 0; c < 8; c++) {
            acc[c] = 0.0f;
            if (shared) acc[c] += bf2f(vs[c]);
        }
#pragma unroll
        for (int s = 0; s < 8; s++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                acc[c] += p[s][0][c];
                acc[4 + c] += p[s][1][c];
            }
        }
        bf16v8 o;
#pragma unroll
        for (int c = 0; c < 8; c++) o[c] = f2bf(acc[c]);
        *(PLOW_GLOB bf16v8*)(as_glob(out) + e) = o;
    }
}
