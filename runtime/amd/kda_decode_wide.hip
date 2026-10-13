/* Standalone batched-decode KDA chain (op 125, descriptor version 3, PLOW_KDA_DECODE_WIDE).
 *
 * One workgroup per (row, head) runs the interpreter's KdaConv3 -> KdaStateStepG -> KdaGatedNorm
 * chain for that pair through the SAME device bodies (kda_conv_range's independent-row arm,
 * d_kda_state_step_t's wide arm, d_kda_gated_norm's per-row body), so every element sees the
 * identical arithmetic and the outputs are bit-identical to the three interpreter packets. The
 * intermediates (mix q/k/v, o) still round-trip through their bf16 tensors; only the workgroup
 * that wrote them reads them. Unlike the interpreter (one 256-VGPR workgroup per CU, B*H/256
 * items serial per workgroup) every item is resident at once. The intermediates never cross a
 * workgroup, so __syncthreads' workgroup-scope fences order them (no agent-scope L2 writeback). */
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda.h"

extern "C" __device__ unsigned plow_kda_decode_wide_3 = 3;

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void plow_kda_decode_wide_v3(
    bf16* y, const bf16* q_raw, const bf16* k_raw, const bf16* v_raw,
    const float* wq, const float* wk, const float* wv, float* csq, float* csk,
    float* csv, const bf16* forget_raw, const bf16* beta_raw,
    const bf16* output_gate_raw, const float* a_log, const float* dt_bias, float* state,
    const float* norm_w, const unsigned* parked, unsigned rows, unsigned heads, unsigned dim,
    unsigned bv, unsigned conv_w, unsigned flags, unsigned gate_mode, float lower_bound,
    float scale, float norm_eps, bf16* mq, bf16* mk, bf16* mv, bf16* o) {
    if (blockDim.x != PLOW_THREADS || rows < 2u || heads == 0u || dim != 128u || bv == 0u ||
        conv_w == 0u || conv_w > 8u || !(flags & PLOW_KDA_F_SEQ_ROWS) ||
        (flags & ~(PLOW_KDA_F_QK_L2NORM | PLOW_KDA_F_SEQ_ROWS)) || gridDim.x != rows * heads)
        __builtin_trap();
    __shared__ float lds[5u * 128u + 2u * PLOW_WAVES];
    const unsigned item = blockIdx.x;
    const unsigned row = item / heads;
    const unsigned h = item - row * heads;
    const unsigned proj = heads * dim;
    const size_t rp = (size_t)row * proj;
    const size_t rcs = rp * conv_w;
    const unsigned* pk = parked ? parked + row : nullptr;
    const unsigned c0 = h * dim, c1 = c0 + dim;
    /* op 109, independent-sequence arm: this row's window, this head's channels. */
    kda_conv_range(mq + rp, q_raw + rp, wq, csq + rcs, 1u, proj, conv_w, 1u, c0, c1,
                   (size_t)proj * conv_w, pk);
    kda_conv_range(mk + rp, k_raw + rp, wk, csk + rcs, 1u, proj, conv_w, 1u, c0, c1,
                   (size_t)proj * conv_w, pk);
    kda_conv_range(mv + rp, v_raw + rp, wv, csv + rcs, 1u, proj, conv_w, 1u, c0, c1,
                   (size_t)proj * conv_w, pk);
    __syncthreads();
    /* op 110, wide arm: grid == T*H items, so this workgroup's only item is `item`. */
    d_kda_state_step_g(o, mq, mk, mv, forget_raw, beta_raw, a_log, dt_bias, gate_mode,
                       lower_bound, state, rows, heads, dim, bv, flags, scale, item,
                       rows * heads, lds, (size_t)heads * dim * dim, parked);
    __syncthreads();
    /* op 103 for row `item` (= row * H + h) on wave 0: d_kda_gated_norm's body. */
    if ((threadIdx.x >> 6) == 0u) {
        const unsigned lane = threadIdx.x & (PLOW_WAVE - 1);
        const size_t base = (size_t)item * dim;
        float ss = 0.0f;
        for (unsigned d = lane; d < dim; d += PLOW_WAVE) {
            const float x = bf2f(o[base + d]);
            ss += x * x;
        }
        const float inv = rsqrtf(wave_sum(ss) / (float)dim + norm_eps);
        for (unsigned d = lane; d < dim; d += PLOW_WAVE)
            y[base + d] = f2bf(bf2f(o[base + d]) * inv * norm_w[d] *
                               kda_sigmoid(bf2f(output_gate_raw[base + d])));
    }
}
