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
 * combine op are untouched. Every kernel launches on a fixed grid; the live row extent
 * (rowoff[E-1] + cnt[E-1]) is read from meta on the device.
 *
 * Launch: grid = nblk blocks x 256 threads, smem 0. Block `b` owns rows b, b+nblk, ...; its
 * threads stride the flattened (owned row, 8-wide chunk) space.
 */
#include "dev_isa.h"
#include "sm120_common.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
#error "MoE cuBLASLt glue object requires sm_90a"
#endif

extern "C" {
/* ABI 2 adds plow_moe_lt_norm; a decode route needs it, the prefill route accepts ABI 1. */
__device__ unsigned plow_moe_lt_abi = 2;
}

__device__ __forceinline__ unsigned moe_lt_extent(const int* meta, unsigned n_exp) {
    return (unsigned)(meta[n_exp - 1u] + meta[2u * n_exp - 1u]);
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

/* Decode route only: xn2[r] = bf16(x[r] * rsqrt(mean(x[r]^2) + eps) * gamma), the norm that
 * MoeExpertGluNormGemma fuses (plow_moe_stage_xn). One block per row (grid = rows), H % 8 == 0.
 * The reduction partition differs from the interpreter's, so inv may differ in the last ulp. */
extern "C" __global__ void plow_moe_lt_norm(__nv_bfloat16* xn2, const __nv_bfloat16* x,
                                            const __nv_bfloat16* gamma, unsigned H, float eps) {
    __shared__ float red[PLOW_NV_THREADS / 32u];
    __shared__ float inv_s;
    const __nv_bfloat16* row = x + (size_t)blockIdx.x * H;
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
    __nv_bfloat16* out = xn2 + (size_t)blockIdx.x * H;
    for (unsigned c = threadIdx.x; c < nvec; c += blockDim.x) {
        const bf16v8 v = ld_glob8(row + c * 8u), g = ld_glob8(gamma + c * 8u);
        bf16v8 o;
#pragma unroll
        for (int j = 0; j < 8; j++)
            o.x[j] = __float2bfloat16(__bfloat162float(v.x[j]) * inv * __bfloat162float(g.x[j]));
        st_glob8(out + c * 8u, o);
    }
}

/* Per-expert group shapes and matrix pointers for both grouped matmuls.
 * rows[e] = cnt[e]; ptrs = [xs | gu | fu | dn] x E, each base + rowoff[e] * width * 2. */
extern "C" __global__ void plow_moe_lt_setup(int* rows, unsigned long long* ptrs, const int* meta,
                                             unsigned long long xs, unsigned long long gu,
                                             unsigned long long fu, unsigned long long dn,
                                             unsigned n_exp, unsigned H, unsigned I) {
    if (blockIdx.x != 0) return;
    for (unsigned e = threadIdx.x; e < n_exp; e += blockDim.x) {
        const unsigned long long off = (unsigned long long)(unsigned)meta[e];
        rows[e] = meta[n_exp + e];
        ptrs[e] = xs + off * H * 2ull;
        ptrs[n_exp + e] = gu + off * I * 4ull;
        ptrs[2u * n_exp + e] = fu + off * I * 2ull;
        ptrs[3u * n_exp + e] = dn + off * H * 2ull;
    }
}

/* xs[r] = xn2[row_token[r]] for every live gathered row. H % 8 == 0. */
extern "C" __global__ void plow_moe_lt_gather(__nv_bfloat16* xs, const __nv_bfloat16* xn2,
                                              const unsigned* row_token, const int* meta,
                                              unsigned n_exp, unsigned H, unsigned nblk) {
    const unsigned extent = moe_lt_extent(meta, n_exp);
    for (MoeLtCursor k(blockIdx.x, nblk, H / 8u); k.row < extent; k.next()) {
        const unsigned tok = row_token[k.row];
        if (tok == PLOW_EXPERT_UNUSED) continue;
        st_glob8(xs + (size_t)k.row * H + k.c * 8u, ld_glob8(xn2 + (size_t)tok * H + k.c * 8u));
    }
}

/* fu[r] = act(gu[r][0..I)) * gu[r][I..2I). I % 8 == 0. Pad rows inside the extent are computed
 * too: nothing reads them (the grouped down matmul covers cnt[e] rows per expert). */
extern "C" __global__ void plow_moe_lt_glu(__nv_bfloat16* fu, const __nv_bfloat16* gu,
                                           const int* meta, unsigned n_exp, unsigned I,
                                           unsigned act, unsigned nblk) {
    const unsigned extent = moe_lt_extent(meta, n_exp);
    for (MoeLtCursor k(blockIdx.x, nblk, I / 8u); k.row < extent; k.next()) {
        const __nv_bfloat16* g = gu + (size_t)k.row * 2u * I + k.c * 8u;
        const bf16v8 vg = ld_glob8(g), vu = ld_glob8(g + I);
        bf16v8 vo;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float x = __bfloat162float(vg.x[j]);
            const float a = (act == PLOW_ACT_SILU_) ? act_silu(x) : act_gelu_tanh(x);
            vo.x[j] = __float2bfloat16(a * __bfloat162float(vu.x[j]));
        }
        st_glob8(fu + (size_t)k.row * I + k.c * 8u, vo);
    }
}

/* part[row_partidx[r]] = f32(dn[r]) * row_gate[r] for every live gathered row. H % 8 == 0. */
extern "C" __global__ void plow_moe_lt_scatter(float* part, const __nv_bfloat16* dn,
                                               const unsigned* row_partidx, const float* row_gate,
                                               const int* meta, unsigned n_exp, unsigned H,
                                               unsigned nblk) {
    const unsigned extent = moe_lt_extent(meta, n_exp);
    for (MoeLtCursor k(blockIdx.x, nblk, H / 8u); k.row < extent; k.next()) {
        const unsigned pidx = row_partidx[k.row];
        if (pidx == PLOW_EXPERT_UNUSED) continue;
        const float gate = row_gate[k.row];
        const bf16v8 v = ld_glob8(dn + (size_t)k.row * H + k.c * 8u);
        float* o = part + (size_t)pidx * H + k.c * 8u;
        *(float4*)o = make_float4(__bfloat162float(v.x[0]) * gate, __bfloat162float(v.x[1]) * gate,
                                  __bfloat162float(v.x[2]) * gate, __bfloat162float(v.x[3]) * gate);
        *(float4*)(o + 4) =
            make_float4(__bfloat162float(v.x[4]) * gate, __bfloat162float(v.x[5]) * gate,
                        __bfloat162float(v.x[6]) * gate, __bfloat162float(v.x[7]) * gate);
    }
}
