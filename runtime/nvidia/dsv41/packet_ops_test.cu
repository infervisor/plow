// Test-only wrappers: packet op bodies (the code the interpreter arms call) as plain kernels, each
// block playing one interpreter slice, so scripts/dsv41_nv/test_packet_ops.py can check them
// against the reference before they run inside a packet. Not part of any shipped object.
#include <cuda_bf16.h>
#include "../../common/op_act_quant_mx.h"

extern "C" __global__ void t_act_quant_mx(uint16_t* out, const uint16_t* x, unsigned rows, unsigned k) {
    d_act_quant_mx(out, x, rows, k, blockIdx.x, gridDim.x);
}
extern "C" __global__ void t_act_quant_mx8(uint8_t* out, uint8_t* scale, const uint16_t* x, unsigned rows, unsigned k) {
    d_act_quant_mx(reinterpret_cast<uint16_t*>(out), x, rows, k, blockIdx.x, gridDim.x, scale);
}

#include "../op_gemm_f32.cuh"
#include "../op_gemv_fp8mx.cuh"
extern "C" __global__ void __launch_bounds__(256, 1) t_gemv_fp8mx(__nv_bfloat16* c, const uint8_t* x, const uint8_t* xs, const uint8_t* w,
                                                               const uint8_t* ws, unsigned t, unsigned n, unsigned k, unsigned groups,
                                                               unsigned fp8, unsigned reps, unsigned copies, unsigned arena_f) {
    extern __shared__ float arena[];
    /* rep r reads weight copy r mod copies (laid out back to back), so timing sees HBM, not L2 */
    const size_t wb = (size_t)groups * n * k, sb = (size_t)groups * (n / 32) * (k / 32);
    for (unsigned r = 0, cp = 0; r < reps; r++, cp = cp + 1u == copies ? 0u : cp + 1u)
        d_gemv_fp8mx(c, x, xs, w + cp * wb, ws + cp * sb, t, n, k, groups, fp8 != 0, blockIdx.x, gridDim.x, arena, arena_f);
}
extern "C" __global__ void __launch_bounds__(256) t_gemm_f32(float* c, const __nv_bfloat16* a, const __nv_bfloat16* w, unsigned m, unsigned n,
                                                             unsigned k, unsigned char* scratch, unsigned splits) {
    extern __shared__ float arena[];
    d_gemm_f32(c, a, w, m, n, k, blockIdx.x, gridDim.x, arena, 4u * 192u * 72u / 2u, scratch, splits);
}
extern "C" __global__ void __launch_bounds__(256) t_gemv_f32(float* c, const __nv_bfloat16* x, const float* w, unsigned m, unsigned n,
                                                            unsigned k, unsigned char* scratch, unsigned splits) {
    extern __shared__ float arena[];
    d_gemv_f32(c, x, w, m, n, k, blockIdx.x, gridDim.x, arena, plow_f32::SK_ARENA_FLOATS, scratch, splits);
}
extern "C" __global__ void __launch_bounds__(256) t_stream_read(const uint4* p, size_t n16, unsigned* sink, unsigned reps) {
    unsigned acc = 0;
    for (unsigned r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x * 4) {
            uint4 v[4];
#pragma unroll
            for (int u = 0; u < 4; u++) v[u] = i + (size_t)u * gridDim.x * blockDim.x < n16 ? __ldcs(p + i + (size_t)u * gridDim.x * blockDim.x) : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
    if (acc == 0x12345678u) *sink = acc;
}
#include "../op_sparse_attn_decode.cuh"
/* rep r reads ring/cmp copy r % copies and writes partial copy r % copies, so timing sees HBM,
 * not L2 */
extern "C" __global__ void __launch_bounds__(256, 1) t_sparse_attn_decode(__nv_bfloat16* o, const __nv_bfloat16* q, const __nv_bfloat16* ring,
                                                                         const __nv_bfloat16* cmp, const int* idx, const int* pos,
                                                                         const float* sink, float* scratch, unsigned B, unsigned H, unsigned W,
                                                                         unsigned cmp_stride, unsigned topk, unsigned nsplit, float scale,
                                                                         unsigned reps, unsigned copies) {
    extern __shared__ unsigned char smem_sad[];
    const size_t rb = (size_t)B * W * 512, cb = (size_t)B * cmp_stride * 512;
    const size_t sb = plow_sad::scratch_bytes(B, H / 16, nsplit) / 4;
    for (unsigned r = 0; r < reps; r++)
        d_sparse_attn_decode(o, q, ring + (r % copies) * rb, cmp ? cmp + (r % copies) * cb : nullptr, idx, pos, sink,
                             scratch + (r % copies) * sb, B, H, W, cmp_stride, topk, nsplit, scale, blockIdx.x, gridDim.x, smem_sad);
}
extern "C" __global__ void __launch_bounds__(256, 1) t_sparse_attn_merge(__nv_bfloat16* o, const float* scratch, const float* sink, unsigned B,
                                                                        unsigned H, unsigned nsplit, unsigned reps, unsigned copies) {
    const size_t sb = plow_sad::scratch_bytes(B, H / 16, nsplit) / 4;
    for (unsigned r = 0; r < reps; r++)
        d_sparse_attn_merge(o, scratch + (r % copies) * sb, sink, B, H, nsplit, blockIdx.x, gridDim.x);
}
#include "../op_index_decode.cuh"
/* rep r reads key copy r % copies, so timing sees HBM, not L2 */
extern "C" __global__ void __launch_bounds__(256, 1) t_index_score_decode(float* score, const __nv_bfloat16* q, const __nv_bfloat16* w, float wscale,
                                                                         const __nv_bfloat16* k, const int* pos, unsigned B, unsigned HI,
                                                                         unsigned cap, unsigned ratio, unsigned reps, unsigned copies) {
    const size_t kb = (size_t)B * cap * 128;
    for (unsigned r = 0; r < reps; r++)
        d_index_score_decode(score, q, w, wscale, k + (r % copies) * kb, pos, B, HI, cap, ratio, blockIdx.x, gridDim.x);
}
extern "C" __global__ void __launch_bounds__(256, 1) t_index_select_decode(int* idx, const float* score, const int* pos, unsigned B,
                                                                          unsigned cap, unsigned ratio, unsigned reps, unsigned smem_words) {
    extern __shared__ unsigned smem_idd[];
    for (unsigned r = 0; r < reps; r++)
        d_index_select_decode(idx, score, pos, B, cap, ratio, blockIdx.x, gridDim.x, smem_idd, smem_words);
}
#include "../op_compress.cuh"
extern "C" __global__ void __launch_bounds__(256) t_compress_decode_step(__nv_bfloat16* latent, float* st_kv, float* st_sc, const float* kv,
                                                                        const float* score, const __nv_bfloat16* gamma, const int* pos,
                                                                        unsigned B, unsigned ratio, unsigned d, float eps, unsigned seed) {
    __shared__ float arena[512 + 32];
    d_compress_decode_step(latent, st_kv, st_sc, kv, score, gamma, pos, B, ratio, d, eps, blockIdx.x, gridDim.x, arena, seed);
}
extern "C" __global__ void __launch_bounds__(256) t_compress_rope_quant(__nv_bfloat16* out, const __nv_bfloat16* src, const float* cosb,
                                                                       const float* sinb, const int* pos, unsigned n_rows, unsigned d,
                                                                       unsigned rd, unsigned qblk, unsigned ratio, unsigned row_base,
                                                                       unsigned qmode, unsigned n_head, unsigned batched, unsigned slot_stride,
                                                                       unsigned ring_mask, const int* kvlen) {
    d_compress_rope_quant(out, src, cosb, sinb, n_rows, d, rd, qblk, ratio, row_base, qmode, blockIdx.x, gridDim.x, pos, n_head, batched,
                          slot_stride, ring_mask, kvlen);
}
extern "C" __global__ void __launch_bounds__(256) t_rope_inverse_o(__nv_bfloat16* o, const float* cosb, const float* sinb, const int* pos,
                                                                  unsigned n_tok, unsigned n_head, unsigned D, unsigned rd, unsigned per_row) {
    d_rope_inverse_o(o, cosb, sinb, n_tok, n_head, D, rd, 0, blockIdx.x, gridDim.x, pos, per_row);
}
#include "../op_moe_decode_v41.cuh"
/* rep r reads weight-table copy r % copies (tables mapping the experts to different physical ones), so
 * timing sees HBM, not L2; the K-split scratch and counters rotate with it, so back-to-back reps never
 * share an item counter */
extern "C" __global__ void __launch_bounds__(256, 1) t_moe_glu_decode(__nv_bfloat16* fu, const __nv_bfloat16* x, const unsigned long long* wtab,
                                                                     const unsigned long long* stab, const int* meta, const unsigned* row_token,
                                                                     const float* row_gate, unsigned I, unsigned H, unsigned E, unsigned act,
                                                                     float lim, float* scratch, unsigned scratch_floats, unsigned* ctr,
                                                                     unsigned ctr_cap, unsigned reps, unsigned copies) {
    extern __shared__ __align__(16) unsigned char dyn[];
    for (unsigned r = 0; r < reps; r++)
        d_moe_glu_decode_v41(fu, x, wtab + (size_t)(r % copies) * E * 3, stab + (size_t)(r % copies) * E * 3, meta, row_token, row_gate, I, H,
                             E, act, lim, scratch + (size_t)(r % copies) * scratch_floats, ctr + (size_t)(r % copies) * ctr_cap,
                             blockIdx.x, gridDim.x, dyn);
}
extern "C" __global__ void __launch_bounds__(256, 1) t_moe_down_decode(float* part, const __nv_bfloat16* fu, const unsigned long long* wtab,
                                                                      const unsigned long long* stab, const int* meta, const unsigned* row_partidx,
                                                                      const float* row_gate, unsigned H, unsigned I, unsigned E, unsigned reps,
                                                                      unsigned copies) {
    extern __shared__ __align__(16) unsigned char dyn[];
    for (unsigned r = 0; r < reps; r++)
        d_moe_down_decode_v41(part, fu, wtab + (size_t)(r % copies) * E * 3, stab + (size_t)(r % copies) * E * 3, meta, row_partidx, row_gate,
                              H, I, E, blockIdx.x, gridDim.x, dyn);
}
