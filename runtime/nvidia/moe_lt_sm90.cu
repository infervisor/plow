/* moe_lt_sm90.cu — device glue of the cuBLASLt GROUPED-GEMM MoE routes (PLOW_MOE_PF_LT prefill,
 * PLOW_MOE_DEC_LT decode).
 *
 * The route replaces MoeGroupGluGemmaPf + MoeGroupDownGemmaPf with two cuBLASLt grouped matmuls
 * (CUBLASLT_BATCH_MODE_GROUPED: per-expert row counts and matrix pointers live ON THE DEVICE, so
 * there is no host sync and the chain stays graph-capturable). MoeAlignGemmaPf still runs in the
 * interpreter; these kernels read ITS tables:
 *   meta (i32): [0,E) rowoff (padded segment start row), [E,2E) cnt, [2E,3E] tile prefix
 *   row_token / row_partidx (u32, PLOW_EXPERT_UNUSED on pad rows), row_gate (f32)
 * so the expert-contiguous layout, the gate-scaled f32 part[token*k+slot] contract and the
 * combine op are untouched. Every kernel launches on a fixed grid and walks only the live rows
 * (sum cnt): align pads each expert to a 64-row tile, so at a decode rung the padded extent is
 * ~40x the live rows, and the matmuls cover cnt[e] rows per expert, so pad rows are never read.
 *
 * Launch: grid = nblk blocks x 256 threads, smem 0. Block `b` owns live rows b, b+nblk, ...; its
 * threads stride the flattened (owned row, 8-wide chunk) space.
 */
#include "dev_isa.h"
#include "sm120_common.cuh"
#include "op_norm.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
#error "MoE cuBLASLt glue object requires sm_90a"
#endif

extern "C" {
/* ABI 2 adds plow_moe_lt_norm; a decode route needs it, the prefill route accepts ABI 1.
 * ABI 3 adds plow_moe_lt_norm_gather (the decode route's setup + norm + gather).
 * ABI 4 adds plow_moe_lt_combine_nrn (a decode segment that carries the layer tail).
 * ABI 5 adds plow_moe_lt_gather_inv + plow_moe_lt_combine_pf (a prefill segment that carries
 * MoeCombineNormGemmaPf).
 * ABI 6 adds the W8A8 prefill chain: plow_moe_lt_gather8_inv, plow_moe_lt_glu8,
 * plow_moe_lt_combine_pf8. */
__device__ unsigned plow_moe_lt_abi = 6;
}

/* pre[0..n_exp] = exclusive prefix of cnt; returns the live row count. n_exp <= 256 (align's
 * PLOW_MOE_MAXE) = one count per thread. */
__device__ unsigned moe_lt_live_prefix(unsigned* pre, const int* meta, unsigned n_exp) {
    __shared__ unsigned wsum[PLOW_NV_THREADS / 32u];
    const unsigned tid = threadIdx.x, lane = tid & 31u;
    unsigned v = tid < n_exp ? (unsigned)meta[n_exp + tid] : 0u;
#pragma unroll
    for (unsigned o = 1u; o < 32u; o <<= 1) {
        const unsigned t = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += t;
    }
    if (lane == 31u) wsum[tid >> 5] = v;
    __syncthreads();
    for (unsigned w = 0; w < (tid >> 5); w++) v += wsum[w];
    pre[tid + 1u] = v;
    if (tid == 0u) pre[0] = 0u;
    __syncthreads();
    return pre[n_exp];
}

/* Padded row of live row i: rowoff[e] + i - pre[e] for the e with pre[e] <= i < pre[e + 1]. */
__device__ __forceinline__ unsigned moe_lt_live_row(const unsigned* pre, const int* meta,
                                                    unsigned n_exp, unsigned i) {
    unsigned lo = 0u, hi = n_exp;
    while (hi - lo > 1u) {
        const unsigned mid = (lo + hi) >> 1;
        if (pre[mid] <= i) lo = mid;
        else hi = mid;
    }
    return (unsigned)meta[lo] + (i - pre[lo]);
}

/* Flattened (owned row, chunk) cursor of one thread: j = tid, tid + 256, ... over rows
 * slice, slice + nblk, ... each `ch` chunks wide. No division in the loop. */
struct MoeLtCursor {
    unsigned row, c, ch, drow, dc, nblk;
    __device__ MoeLtCursor(unsigned slice, unsigned nblk_, unsigned ch_)
        : row(slice + (threadIdx.x / ch_) * nblk_), c(threadIdx.x % ch_), ch(ch_),
          drow((PLOW_NV_THREADS / ch_) * nblk_), dc(PLOW_NV_THREADS % ch_), nblk(nblk_) {}
    __device__ void next() {
        row += drow;
        c += dc;
        if (c >= ch) {
            c -= ch;
            row += nblk;
        }
    }
};

/* out = bf16(row * rsqrt(mean(row^2) + eps) * gamma), the norm that MoeExpertGluNormGemma fuses
 * (plow_moe_stage_xn), by the whole block. H % 8 == 0. The reduction partition differs from the
 * interpreter's, so inv may differ in the last ulp. */
__device__ void moe_lt_norm_row(__nv_bfloat16* out, const __nv_bfloat16* row,
                                const __nv_bfloat16* gamma, unsigned H, float eps) {
    __shared__ float red[PLOW_NV_THREADS / 32u];
    __shared__ float inv_s;
    const unsigned nvec = H / 8u;
    float part = 0.0f;
    for (unsigned c = threadIdx.x; c < nvec; c += blockDim.x) {
        const bf16v8 v = ld_glob8(row + c * 8u);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float f = __bfloat162float(v.x[j]);
            part += f * f;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
    if ((threadIdx.x & 31u) == 0u) red[threadIdx.x >> 5] = part;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned w = 0; w < blockDim.x / 32u; w++) s += red[w];
        inv_s = rsqrtf(s / (float)H + eps);
    }
    __syncthreads();
    const float inv = inv_s;
    for (unsigned c = threadIdx.x; c < nvec; c += blockDim.x) {
        const bf16v8 v = ld_glob8(row + c * 8u), g = ld_glob8(gamma + c * 8u);
        bf16v8 o;
#pragma unroll
        for (int j = 0; j < 8; j++)
            o.x[j] = __float2bfloat16(__bfloat162float(v.x[j]) * inv * __bfloat162float(g.x[j]));
        st_glob8(out + c * 8u, o);
    }
}

/* Decode route, ABI 2 objects' chain: xn2[r] = norm(x[r]). One block per row (grid = rows). */
extern "C" __global__ void plow_moe_lt_norm(__nv_bfloat16* xn2, const __nv_bfloat16* x,
                                            const __nv_bfloat16* gamma, unsigned H, float eps) {
    moe_lt_norm_row(xn2 + (size_t)blockIdx.x * H, x + (size_t)blockIdx.x * H, gamma, H, eps);
}

/* Per-expert group shapes and matrix pointers for both grouped matmuls.
 * rows[e] = cnt[e]; ptrs = [xs | gu | fu | dn] x E, each base + rowoff[e] * width * 2. */
__device__ void moe_lt_setup_tables(int* rows, unsigned long long* ptrs, const int* meta,
                                    unsigned long long xs, unsigned long long gu,
                                    unsigned long long fu, unsigned long long dn, unsigned n_exp,
                                    unsigned H, unsigned I) {
    for (unsigned e = threadIdx.x; e < n_exp; e += blockDim.x) {
        const unsigned long long off = (unsigned long long)(unsigned)meta[e];
        rows[e] = meta[n_exp + e];
        ptrs[e] = xs + off * H * 2ull;
        ptrs[n_exp + e] = gu + off * I * 4ull;
        ptrs[2u * n_exp + e] = fu + off * I * 2ull;
        ptrs[3u * n_exp + e] = dn + off * H * 2ull;
    }
}

extern "C" __global__ void plow_moe_lt_setup(int* rows, unsigned long long* ptrs, const int* meta,
                                             unsigned long long xs, unsigned long long gu,
                                             unsigned long long fu, unsigned long long dn,
                                             unsigned n_exp, unsigned H, unsigned I) {
    if (blockIdx.x == 0) moe_lt_setup_tables(rows, ptrs, meta, xs, gu, fu, dn, n_exp, H, I);
}

/* Decode route, ABI 3: setup + norm + gather in one launch. Block b normalizes live rows b,
 * b+nblk, ... straight from x[row_token[r]] with the norm's own partition, so xs is
 * bit-identical to the three-kernel chain and xn2 is never written. */
extern "C" __global__ void plow_moe_lt_norm_gather(
    __nv_bfloat16* xs, const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const unsigned* row_token, int* rows, unsigned long long* ptrs, const int* meta,
    unsigned long long gu, unsigned long long fu, unsigned long long dn, unsigned n_exp,
    unsigned H, unsigned I, float eps, unsigned nblk) {
    if (blockIdx.x == 0)
        moe_lt_setup_tables(rows, ptrs, meta, (unsigned long long)xs, gu, fu, dn, n_exp, H, I);
    __shared__ unsigned pre[PLOW_NV_THREADS + 1u];
    const unsigned live = moe_lt_live_prefix(pre, meta, n_exp);
    for (unsigned i = blockIdx.x; i < live; i += nblk) {
        const unsigned r = moe_lt_live_row(pre, meta, n_exp, i);
        moe_lt_norm_row(xs + (size_t)r * H, x + (size_t)row_token[r] * H, gamma, H, eps);
    }
}

/* xs[r] = xn2[row_token[r]] for every live gathered row. H % 8 == 0. */
extern "C" __global__ void plow_moe_lt_gather(__nv_bfloat16* xs, const __nv_bfloat16* xn2,
                                              const unsigned* row_token, const int* meta,
                                              unsigned n_exp, unsigned H, unsigned nblk) {
    __shared__ unsigned pre[PLOW_NV_THREADS + 1u];
    const unsigned live = moe_lt_live_prefix(pre, meta, n_exp);
    for (MoeLtCursor k(blockIdx.x, nblk, H / 8u); k.row < live; k.next()) {
        const unsigned r = moe_lt_live_row(pre, meta, n_exp, k.row);
        st_glob8(xs + (size_t)r * H + k.c * 8u, ld_glob8(xn2 + (size_t)row_token[r] * H + k.c * 8u));
    }
}

/* fu[r] = act(gu[r][0..I)) * gu[r][I..2I) for every live gathered row. I % 8 == 0. */
extern "C" __global__ void plow_moe_lt_glu(__nv_bfloat16* fu, const __nv_bfloat16* gu,
                                           const int* meta, unsigned n_exp, unsigned I,
                                           unsigned act, unsigned nblk) {
    __shared__ unsigned pre[PLOW_NV_THREADS + 1u];
    const unsigned live = moe_lt_live_prefix(pre, meta, n_exp);
    for (MoeLtCursor k(blockIdx.x, nblk, I / 8u); k.row < live; k.next()) {
        const size_t r = moe_lt_live_row(pre, meta, n_exp, k.row);
        const __nv_bfloat16* g = gu + r * 2u * I + k.c * 8u;
        const bf16v8 vg = ld_glob8(g), vu = ld_glob8(g + I);
        bf16v8 vo;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float x = __bfloat162float(vg.x[j]);
            const float a = (act == PLOW_ACT_SILU_) ? act_silu(x) : act_gelu_tanh(x);
            vo.x[j] = __float2bfloat16(a * __bfloat162float(vu.x[j]));
        }
        st_glob8(fu + r * I + k.c * 8u, vo);
    }
}

/* Decode route, ABI 4: the layer tail the interpreter would run next, MoeCombineNormGemma then
 * NormResidualNorm, straight from the down matmul's rows, so part[] is never written. One block
 * per token row. Bit-exact to scatter + both ops: each slot product is rounded (__fmul_rn) as
 * scatter stores it, the combine is op70's k = 8 body (slot order, float4 partition, warp sums
 * folded by thread 0), and the NRN is the interpreter's d_norm_residual_norm.
 * k == 8, H % 8 == 0 and H <= 12 * 256 are checked on the host. */
extern "C" __global__ void plow_moe_lt_combine_nrn(
    __nv_bfloat16* hn, __nv_bfloat16* x, __nv_bfloat16* comb, const __nv_bfloat16* dn,
    const unsigned* row_partidx, const float* row_gate, const int* meta,
    const __nv_bfloat16* h1, const __nv_bfloat16* g_pf2, const __nv_bfloat16* g_po,
    const __nv_bfloat16* gn, unsigned n_exp, unsigned H, float eps_comb, float eps, float scale) {
    constexpr unsigned K = 8u;
    __shared__ unsigned pre[PLOW_NV_THREADS + 1u];
    __shared__ unsigned slot_row[K];
    __shared__ float slot_gate[K];
    __shared__ __align__(16) float acc[12u * PLOW_NV_THREADS];
    __shared__ float red[PLOW_NV_THREADS / 32u];
    const unsigned t = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned live = moe_lt_live_prefix(pre, meta, n_exp);
    for (unsigned i = tid; i < live; i += PLOW_NV_THREADS) {
        const unsigned r = moe_lt_live_row(pre, meta, n_exp, i);
        const unsigned p = row_partidx[r];
        if (p / K == t) {
            slot_row[p % K] = r;
            slot_gate[p % K] = row_gate[r];
        }
    }
    __syncthreads();

    const size_t base = (size_t)t * H;
    float ss = 0.0f;
#pragma unroll
    for (unsigned it = 0; it < 3u; it++) {
        const unsigned h = tid * 4u + it * PLOW_NV_THREADS * 4u;
        if (h >= H) continue;
        float4 v[K];
#pragma unroll
        for (unsigned s = 0; s < K; s++) {
            const uint2 raw = *(const uint2*)(dn + (size_t)slot_row[s] * H + h);
            const __nv_bfloat162 lo = *(const __nv_bfloat162*)&raw.x, hi = *(const __nv_bfloat162*)&raw.y;
            const float g = slot_gate[s];
            v[s] = make_float4(__fmul_rn(__low2float(lo), g), __fmul_rn(__high2float(lo), g),
                               __fmul_rn(__low2float(hi), g), __fmul_rn(__high2float(hi), g));
        }
        float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (unsigned s = 0; s < K; s++) {
            acc4.x += v[s].x; acc4.y += v[s].y; acc4.z += v[s].z; acc4.w += v[s].w;
        }
        *(float4*)(acc + h) = acc4;
        ss += acc4.x * acc4.x + acc4.y * acc4.y + acc4.z * acc4.z + acc4.w * acc4.w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o, 32);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (unsigned i = 0; i < PLOW_NV_THREADS / 32u; i++) s += red[i];
        red[0] = rsqrtf(s / (float)H + eps_comb);
    }
    __syncthreads();
    const float inv = red[0];
    for (unsigned c = tid; c < (H >> 3); c += PLOW_NV_THREADS) {
        const bf16v8 g = ld_glob8(g_pf2 + c * 8u), r = ld_glob8(h1 + base + c * 8u);
        bf16v8 ov;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float v = acc[c * 8u + j] * inv * __bfloat162float(g.x[j]);
            ov.x[j] = __float2bfloat16(v + __bfloat162float(r.x[j]));
        }
        st_glob8(comb + base + c * 8u, ov);
    }
    __syncthreads();
    d_norm_residual_norm(hn + base, x + base, x + base, comb + base, g_po, gn, 1u, H, eps, scale,
                         0u, 1u, red);
}

/* part[row_partidx[r]] = f32(dn[r]) * row_gate[r] for every live gathered row. H % 8 == 0. */
extern "C" __global__ void plow_moe_lt_scatter(float* part, const __nv_bfloat16* dn,
                                               const unsigned* row_partidx, const float* row_gate,
                                               const int* meta, unsigned n_exp, unsigned H,
                                               unsigned nblk) {
    __shared__ unsigned pre[PLOW_NV_THREADS + 1u];
    const unsigned live = moe_lt_live_prefix(pre, meta, n_exp);
    for (MoeLtCursor k(blockIdx.x, nblk, H / 8u); k.row < live; k.next()) {
        const unsigned r = moe_lt_live_row(pre, meta, n_exp, k.row);
        const unsigned pidx = row_partidx[r];
        const float gate = row_gate[r];
        const bf16v8 v = ld_glob8(dn + (size_t)r * H + k.c * 8u);
        float* o = part + (size_t)pidx * H + k.c * 8u;
        *(float4*)o = make_float4(__bfloat162float(v.x[0]) * gate, __bfloat162float(v.x[1]) * gate,
                                  __bfloat162float(v.x[2]) * gate, __bfloat162float(v.x[3]) * gate);
        *(float4*)(o + 4) =
            make_float4(__bfloat162float(v.x[4]) * gate, __bfloat162float(v.x[5]) * gate,
                        __bfloat162float(v.x[6]) * gate, __bfloat162float(v.x[7]) * gate);
    }
}

/* Prefill route, ABI 5: gather as plow_moe_lt_gather, plus inv[row_partidx[r]] = r for every
 * live row, the (token, slot) -> gathered row map plow_moe_lt_combine_pf reads. */
extern "C" __global__ void plow_moe_lt_gather_inv(__nv_bfloat16* xs, unsigned* inv,
                                                  const __nv_bfloat16* xn2,
                                                  const unsigned* row_token,
                                                  const unsigned* row_partidx, const int* meta,
                                                  unsigned n_exp, unsigned H, unsigned nblk) {
    __shared__ unsigned pre[PLOW_NV_THREADS + 1u];
    const unsigned live = moe_lt_live_prefix(pre, meta, n_exp);
    for (MoeLtCursor k(blockIdx.x, nblk, H / 8u); k.row < live; k.next()) {
        const unsigned r = moe_lt_live_row(pre, meta, n_exp, k.row);
        if (k.c == 0u) inv[row_partidx[r]] = r;
        st_glob8(xs + (size_t)r * H + k.c * 8u, ld_glob8(xn2 + (size_t)row_token[r] * H + k.c * 8u));
    }
}

/* Prefill route, ABI 5: MoeCombineNormGemmaPf (op77) straight from the down matmul's rows, so
 * part[] is never written. One block per token (grid-stride). Bit-exact to scatter + op77's
 * k = 8 float4 body at 256 threads (PLOW_NV_GEMV_RB, PLOW_MOE_COMBINE_PF_V4 = 2): each slot
 * product is rounded (__fmul_rn) as scatter stores it, slots summed in order, the same float4
 * partition and warp-sum fold. k == 8, H % 8 == 0 and H <= 12 * 256 are checked on the host.
 * Rows are clamped to the capacity: a token the align never placed reads a stale row, as op77
 * reads a stale part row. */
extern "C" __global__ void plow_moe_lt_combine_pf(
    __nv_bfloat16* out, const __nv_bfloat16* dn, const unsigned* inv, const float* row_gate,
    const __nv_bfloat16* h1, const __nv_bfloat16* gamma, unsigned H, unsigned T, unsigned cap,
    float eps) {
    constexpr unsigned K = 8u;
    __shared__ float red[PLOW_NV_THREADS / 32u];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    for (unsigned t = blockIdx.x; t < T; t += gridDim.x) {
        unsigned row[K];
        float gate[K];
#pragma unroll
        for (unsigned s = 0; s < K; s++) {
            row[s] = min(inv[(size_t)t * K + s], cap - 1u);
            gate[s] = row_gate[row[s]];
        }
        float4 acc[3];
        float ss = 0.0f;
#pragma unroll
        for (unsigned it = 0; it < 3u; it++) {
            const unsigned h = tid * 4u + it * PLOW_NV_THREADS * 4u;
            if (h >= H) continue;
            float4 v[K];
#pragma unroll
            for (unsigned s = 0; s < K; s++) {
                const uint2 raw = *(const uint2*)(dn + (size_t)row[s] * H + h);
                const __nv_bfloat162 lo = *(const __nv_bfloat162*)&raw.x;
                const __nv_bfloat162 hi = *(const __nv_bfloat162*)&raw.y;
                v[s] = make_float4(__fmul_rn(__low2float(lo), gate[s]), __fmul_rn(__high2float(lo), gate[s]),
                                   __fmul_rn(__low2float(hi), gate[s]), __fmul_rn(__high2float(hi), gate[s]));
            }
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (unsigned s = 0; s < K; s++) {
                a.x += v[s].x; a.y += v[s].y; a.z += v[s].z; a.w += v[s].w;
            }
            acc[it] = a;
            ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o, 32);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            for (unsigned i = 0; i < PLOW_NV_THREADS / 32u; i++) s += red[i];
            red[0] = rsqrtf(s / (float)H + eps);
        }
        __syncthreads();
        const float inv_rms = red[0];
        const size_t base = (size_t)t * H;
#pragma unroll
        for (unsigned it = 0; it < 3u; it++) {
            const unsigned h = tid * 4u + it * PLOW_NV_THREADS * 4u;
            if (h >= H) continue;
            alignas(8) __nv_bfloat16 g[4], r[4], ov[4];
            *(uint2*)g = *(const uint2*)(gamma + h);
            *(uint2*)r = *(const uint2*)(h1 + base + h);
            ov[0] = __float2bfloat16(acc[it].x * inv_rms * __bfloat162float(g[0]) + __bfloat162float(r[0]));
            ov[1] = __float2bfloat16(acc[it].y * inv_rms * __bfloat162float(g[1]) + __bfloat162float(r[1]));
            ov[2] = __float2bfloat16(acc[it].z * inv_rms * __bfloat162float(g[2]) + __bfloat162float(r[2]));
            ov[3] = __float2bfloat16(acc[it].w * inv_rms * __bfloat162float(g[3]) + __bfloat162float(r[3]));
            *(uint2*)(out + base + h) = *(const uint2*)ov;
        }
        __syncthreads(); /* red reused next token */
    }
}

/* ---- W8A8 prefill route (ABI 6) ------------------------------------------------------------
 * The grouped matmuls run e4m3 x e4m3 with unit scales and write bf16: gu and dn hold the
 * UNSCALED accumulators, and the glue applies the scales the W8A8 bodies fold into their
 * epilogues: activation row scale x weight channel scale (gate|up), then the per-row dynamic
 * re-quantization of fu (QuantFp8), then fu row scale x down channel scale x gate.
 * est[e * 2 + 0] = f32 [2I] gate|up channel scales, est[e * 2 + 1] = f32 [H] down scales. */

/* Expert of padded row r: the last e with rowoff[e] <= r (an empty expert's range is empty). */
__device__ __forceinline__ unsigned moe_lt_row_expert(const int* meta, unsigned n_exp, unsigned r) {
    unsigned lo = 0u, hi = n_exp;
    while (hi - lo > 1u) {
        const unsigned mid = (lo + hi) >> 1;
        if ((unsigned)meta[mid] <= r) lo = mid;
        else hi = mid;
    }
    return lo;
}

/* plow_moe_lt_setup with e4m3 xs and fu rows: ptrs = [xs8 | gu | fu8 | dn] x E. */
extern "C" __global__ void plow_moe_lt_setup8(int* rows, unsigned long long* ptrs, const int* meta,
                                              unsigned long long xs, unsigned long long gu,
                                              unsigned long long fu, unsigned long long dn,
                                              unsigned n_exp, unsigned H, unsigned I) {
    for (unsigned e = threadIdx.x; e < n_exp; e += blockDim.x) {
        const unsigned long long off = (unsigned long long)(unsigned)meta[e];
        rows[e] = meta[n_exp + e];
        ptrs[e] = xs + off * H;
        ptrs[n_exp + e] = gu + off * I * 4ull;
        ptrs[2u * n_exp + e] = fu + off * I;
        ptrs[3u * n_exp + e] = dn + off * H * 2ull;
    }
}

/* xs8[r] = xq[row_token[r]] (e4m3 rows, H bytes) and inv[row_partidx[r]] = r. H % 16 == 0. */
extern "C" __global__ void plow_moe_lt_gather8_inv(uint8_t* xs8, unsigned* inv, const uint8_t* xq,
                                                   const unsigned* row_token,
                                                   const unsigned* row_partidx, const int* meta,
                                                   unsigned n_exp, unsigned H, unsigned nblk) {
    __shared__ unsigned pre[PLOW_NV_THREADS + 1u];
    const unsigned live = moe_lt_live_prefix(pre, meta, n_exp);
    for (MoeLtCursor k(blockIdx.x, nblk, H / 16u); k.row < live; k.next()) {
        const unsigned r = moe_lt_live_row(pre, meta, n_exp, k.row);
        if (k.c == 0u) inv[row_partidx[r]] = r;
        *(uint4*)(xs8 + (size_t)r * H + k.c * 16u) =
            *(const uint4*)(xq + (size_t)row_token[r] * H + k.c * 16u);
    }
}

/* Warp per live row: f = bf16(act(g * as * sg) * (u * as * su)) from the unscaled gate|up row,
 * then QuantFp8 of the row: fs = max(amax / 448, 1e-12), fu8 = e4m3(f / fs). I % 8 == 0,
 * I <= 96 * 8. */
extern "C" __global__ void plow_moe_lt_glu8(uint8_t* fu8, float* fs, const __nv_bfloat16* gu,
                                            const float* ascale, const unsigned long long* est,
                                            const unsigned* row_token, const int* meta,
                                            unsigned n_exp, unsigned I, unsigned act,
                                            unsigned nblk) {
    constexpr unsigned C = 3u; /* chunks of 8 per lane */
    __shared__ unsigned pre[PLOW_NV_THREADS + 1u];
    const unsigned live = moe_lt_live_prefix(pre, meta, n_exp);
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned nch = I / 8u;
    for (unsigned i = blockIdx.x * (PLOW_NV_THREADS / 32u) + warp; i < live;
         i += nblk * (PLOW_NV_THREADS / 32u)) {
        unsigned lo = 0u, hi = n_exp;
        while (hi - lo > 1u) {
            const unsigned mid = (lo + hi) >> 1;
            if (pre[mid] <= i) lo = mid;
            else hi = mid;
        }
        const size_t r = (unsigned)meta[lo] + (i - pre[lo]);
        const float as = ascale[row_token[r]];
        const float* sc = (const float*)(size_t)est[(size_t)lo * 2u];
        const __nv_bfloat16* g = gu + r * 2u * I;
        float f[C][8];
        float amax = 0.0f;
#pragma unroll
        for (unsigned c = 0; c < C; c++) {
            const unsigned ch = lane + c * 32u;
            if (ch >= nch) continue;
            const bf16v8 vg = ld_glob8(g + ch * 8u), vu = ld_glob8(g + I + ch * 8u);
            const float4 sg0 = *(const float4*)(sc + ch * 8u), sg1 = *(const float4*)(sc + ch * 8u + 4u);
            const float4 su0 = *(const float4*)(sc + I + ch * 8u), su1 = *(const float4*)(sc + I + ch * 8u + 4u);
            const float sg[8] = {sg0.x, sg0.y, sg0.z, sg0.w, sg1.x, sg1.y, sg1.z, sg1.w};
            const float su[8] = {su0.x, su0.y, su0.z, su0.w, su1.x, su1.y, su1.z, su1.w};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float gv = __bfloat162float(vg.x[j]) * as * sg[j];
                const float uv = __bfloat162float(vu.x[j]) * as * su[j];
                const float a = (act == PLOW_ACT_SILU_) ? act_silu(gv) : act_gelu_tanh(gv);
                f[c][j] = __bfloat162float(__float2bfloat16(a * uv));
                amax = fmaxf(amax, fabsf(f[c][j]));
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float s = fmaxf(amax * (1.0f / 448.0f), 1e-12f);
        const float inv = 1.0f / s;
        if (lane == 0) fs[r] = s;
#pragma unroll
        for (unsigned c = 0; c < C; c++) {
            const unsigned ch = lane + c * 32u;
            if (ch >= nch) continue;
            uint2 q8;
            unsigned short* q2 = (unsigned short*)&q8;
#pragma unroll
            for (int j = 0; j < 4; j++) q2[j] = pack_fp8_e4m3(f[c][2 * j] * inv, f[c][2 * j + 1] * inv);
            *(uint2*)(fu8 + r * I + ch * 8u) = q8;
        }
    }
}

/* plow_moe_lt_combine_pf on the unscaled down rows: each slot's product is (gate * fs[row]) *
 * dsc[h] * dn, the W8A8 down epilogue's part value, then the same sum, norm and residual. */
extern "C" __global__ void plow_moe_lt_combine_pf8(
    __nv_bfloat16* out, const __nv_bfloat16* dn, const unsigned* inv, const float* row_gate,
    const float* fs, const unsigned long long* est, const int* meta, const __nv_bfloat16* h1,
    const __nv_bfloat16* gamma, unsigned n_exp, unsigned H, unsigned T, unsigned cap, float eps) {
    constexpr unsigned K = 8u;
    __shared__ float red[PLOW_NV_THREADS / 32u];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    for (unsigned t = blockIdx.x; t < T; t += gridDim.x) {
        unsigned row[K];
        float rs[K];
        const float* dsc[K];
#pragma unroll
        for (unsigned s = 0; s < K; s++) {
            row[s] = min(inv[(size_t)t * K + s], cap - 1u);
            rs[s] = row_gate[row[s]] * fs[row[s]];
            dsc[s] = (const float*)(size_t)est[(size_t)moe_lt_row_expert(meta, n_exp, row[s]) * 2u + 1u];
        }
        float4 acc[3];
        float ss = 0.0f;
#pragma unroll
        for (unsigned it = 0; it < 3u; it++) {
            const unsigned h = tid * 4u + it * PLOW_NV_THREADS * 4u;
            if (h >= H) continue;
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (unsigned s = 0; s < K; s++) {
                const uint2 raw = *(const uint2*)(dn + (size_t)row[s] * H + h);
                const __nv_bfloat162 lo = *(const __nv_bfloat162*)&raw.x;
                const __nv_bfloat162 hi = *(const __nv_bfloat162*)&raw.y;
                const float4 d = *(const float4*)(dsc[s] + h);
                a.x += rs[s] * d.x * __low2float(lo);
                a.y += rs[s] * d.y * __high2float(lo);
                a.z += rs[s] * d.z * __low2float(hi);
                a.w += rs[s] * d.w * __high2float(hi);
            }
            acc[it] = a;
            ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o, 32);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            for (unsigned i = 0; i < PLOW_NV_THREADS / 32u; i++) s += red[i];
            red[0] = rsqrtf(s / (float)H + eps);
        }
        __syncthreads();
        const float inv_rms = red[0];
        const size_t base = (size_t)t * H;
#pragma unroll
        for (unsigned it = 0; it < 3u; it++) {
            const unsigned h = tid * 4u + it * PLOW_NV_THREADS * 4u;
            if (h >= H) continue;
            alignas(8) __nv_bfloat16 g[4], r[4], ov[4];
            *(uint2*)g = *(const uint2*)(gamma + h);
            *(uint2*)r = *(const uint2*)(h1 + base + h);
            ov[0] = __float2bfloat16(acc[it].x * inv_rms * __bfloat162float(g[0]) + __bfloat162float(r[0]));
            ov[1] = __float2bfloat16(acc[it].y * inv_rms * __bfloat162float(g[1]) + __bfloat162float(r[1]));
            ov[2] = __float2bfloat16(acc[it].z * inv_rms * __bfloat162float(g[2]) + __bfloat162float(r[2]));
            ov[3] = __float2bfloat16(acc[it].w * inv_rms * __bfloat162float(g[3]) + __bfloat162float(r[3]));
            *(uint2*)(out + base + h) = *(const uint2*)ov;
        }
        __syncthreads();
    }
}
