/* Standalone entry points for the Mamba-2 interpreter ops (op_mamba.cuh), one kernel per op with
 * the interpreter's (slice = blockIdx.x, nblk = gridDim.x) decomposition and 256 threads.
 * Built to a cubin and driven by runtime/tests/mamba2_oracle.py against a torch fp32 reference:
 *   nvcc -arch=sm_121a -O3 -cubin -I runtime/common -I runtime/nvidia \
 *        -o mamba2_ops.cubin runtime/tests/mamba2_ops_kernels.cu */
#include <cuda_bf16.h>
#include "../nvidia/op_mamba.cuh"

typedef __nv_bfloat16 bf16;

extern "C" __global__ void __launch_bounds__(256) mamba2_conv_k(bf16* out, const bf16* in,
    const bf16* w, const bf16* bias, bf16* hist, const int* active, unsigned C, unsigned W,
    unsigned rows, unsigned T, unsigned prefill, unsigned source) {
    d_mamba2_conv(out, in, w, bias, hist, active, C, W, rows, T, prefill, source, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_step_k(bf16* y, const bf16* xbc,
    const bf16* proj, const bf16* A_log, const bf16* D, const bf16* dt_bias, __half* state,
    const int* active, unsigned H, unsigned P, unsigned N, unsigned G, unsigned B,
    unsigned dt_source, float lo, float hi, unsigned seed_word, unsigned layer) {
    d_mamba2_step(y, xbc, proj, A_log, D, dt_bias, state, active, H, P, N, G, B, dt_source, lo, hi,
                  MAMBA_STATE_BYTES, seed_word, layer, blockIdx.x, gridDim.x);
}

/* The step's stochastic rounding alone: out[i] = SR(x[i]) with the Philox word of element i as
 * the step draws it (key from (seed, layer), counter = i rounded down to 4, word i % 4). */
extern "C" __global__ void __launch_bounds__(256) mamba2_sr_k(unsigned short* out, unsigned* bits,
    const float* x, unsigned n, unsigned seed, unsigned layer) {
    const unsigned long long key = mamba_layer_seed(seed, layer);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uint4 r = mamba_philox4<5>(key, i & ~3u);
        const unsigned w = (i & 3u) == 0 ? r.x : (i & 3u) == 1 ? r.y : (i & 3u) == 2 ? r.z : r.w;
        bits[i] = w;
        out[i] = (unsigned short)mamba_sr_f16(x[i], w);
    }
}

extern "C" __global__ void __launch_bounds__(256) mamba2_norm_k(bf16* out, const bf16* y,
    const bf16* z, const bf16* gamma, const int* active, unsigned groups, unsigned gsize,
    unsigned rows, float eps, unsigned z_source) {
    d_mamba2_gated_norm(out, y, z, gamma, active, groups, gsize, rows, eps, z_source, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_state_k(float* cstate,
    const bf16* xbc, const bf16* proj, const bf16* A_log, const bf16* dt_bias, float* dtv,
    float* cum, unsigned H, unsigned P, unsigned R, unsigned T, unsigned N, unsigned G, float lo,
    float hi, unsigned dt_source) {
    extern __shared__ float smem[];
    d_mamba2_chunk_state(cstate, xbc, proj, A_log, dt_bias, dtv, cum, H, P, R, T, N, G, lo, hi,
                         dt_source, smem, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_pass_k(float* cstate,
    const float* cum, __half* state, unsigned H, unsigned P, unsigned R, unsigned T, unsigned N) {
    d_mamba2_chunk_pass(cstate, cum, state, H, P, R, T, N, MAMBA_STATE_BYTES, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_out_k(bf16* y, const bf16* xbc,
    const float* enter, const float* cum, const float* dtv, const bf16* D, unsigned H, unsigned P,
    unsigned R, unsigned T, unsigned N, unsigned G) {
    extern __shared__ float smem[];
    d_mamba2_chunk_out(y, xbc, enter, cum, dtv, D, H, P, R, T, N, G, smem, blockIdx.x, gridDim.x);
}

/* Tensor-core prefill bodies (op_mamba_pf.cuh): the prefill object's arms. */
#include "../nvidia/op_mamba_pf.cuh"

extern "C" __global__ void __launch_bounds__(256) mamba2_conv_tc_k(bf16* out, const bf16* in,
    const bf16* w, const bf16* bias, bf16* hist, const int* active, unsigned C, unsigned W,
    unsigned rows, unsigned T, unsigned prefill, unsigned source) {
    if (!prefill) {
        d_mamba2_conv(out, in, w, bias, hist, active, C, W, rows, T, prefill, source, blockIdx.x, gridDim.x);
        return;
    }
    d_mamba2_conv_pf(out, in, w, bias, hist, C, W, rows, T, source, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_state_tc_k(float* cstate,
    const bf16* xbc, const bf16* proj, const bf16* A_log, const bf16* dt_bias, float* dtv,
    float* cum, unsigned H, unsigned P, unsigned R, unsigned T, unsigned N, unsigned G, float lo,
    float hi, unsigned dt_source) {
    extern __shared__ float smem[];
    d_mamba2_chunk_state_tc(cstate, xbc, proj, A_log, dt_bias, dtv, cum, H, P, R, T, N, G, lo, hi,
                            dt_source, smem, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_pass_tc_k(float* cstate,
    const float* cum, __half* state, unsigned H, unsigned P, unsigned R, unsigned T, unsigned N) {
    d_mamba2_chunk_pass_pf(cstate, cum, state, H, P, R, T, N, MAMBA_STATE_BYTES, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_out_tc_k(bf16* y, const bf16* xbc,
    const float* enter, const float* cum, const float* dtv, const bf16* D, unsigned H, unsigned P,
    unsigned R, unsigned T, unsigned N, unsigned G) {
    extern __shared__ float smem[];
    d_mamba2_chunk_out_tc(y, xbc, enter, cum, dtv, D, H, P, R, T, N, G, smem, blockIdx.x, gridDim.x);
}

/* Packed prefill: the same bodies over a request table ([n, {row0, len, slot, end}]); history and
 * state are the unshifted slot-major buffers. */
extern "C" __global__ void __launch_bounds__(256) mamba2_conv_pk(bf16* out, const bf16* in,
    const bf16* w, const bf16* bias, bf16* hist, const int* req, unsigned C, unsigned W,
    unsigned rows, unsigned T, unsigned source) {
    d_mamba2_conv_pf<true>(out, in, w, bias, hist, C, W, rows, T, source, blockIdx.x, gridDim.x, req);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_state_pk(float* cstate,
    const bf16* xbc, const bf16* proj, const bf16* A_log, const bf16* dt_bias, float* dtv,
    float* cum, const int* req, unsigned H, unsigned P, unsigned R, unsigned T, unsigned N,
    unsigned G, float lo, float hi, unsigned dt_source) {
    extern __shared__ float smem[];
    d_mamba2_chunk_state_tc<true>(cstate, xbc, proj, A_log, dt_bias, dtv, cum, H, P, R, T, N, G, lo, hi,
                            dt_source, smem, blockIdx.x, gridDim.x, req);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_pass_pk(float* cstate,
    const float* cum, __half* state, const int* req, unsigned H, unsigned P, unsigned R,
    unsigned T, unsigned N) {
    d_mamba2_chunk_pass_pf<true>(cstate, cum, state, H, P, R, T, N, MAMBA_STATE_BYTES, blockIdx.x,
                           gridDim.x, req);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_out_pk(bf16* y, const bf16* xbc,
    const float* enter, const float* cum, const float* dtv, const bf16* D, const int* req,
    unsigned H, unsigned P, unsigned R, unsigned T, unsigned N, unsigned G) {
    extern __shared__ float smem[];
    d_mamba2_chunk_out_tc<true>(y, xbc, enter, cum, dtv, D, H, P, R, T, N, G, smem, blockIdx.x,
                          gridDim.x, req);
}

/* CHUNK_STATE + CHUNK_PASS in one walk (the interpreter's PLOW_NV_MAMBA_SCAN_FUSE pair). */
extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_scan_tc_k(float* cstate,
    const bf16* xbc, const bf16* proj, const bf16* A_log, const bf16* dt_bias, float* dtv,
    float* cum, __half* state, unsigned H, unsigned P, unsigned R, unsigned T, unsigned N, unsigned G,
    float lo, float hi, unsigned dt_source) {
    extern __shared__ float smem[];
    d_mamba2_chunk_scan_state_tc(cstate, xbc, proj, A_log, dt_bias, dtv, cum, state, H, P, R, T, N, G, lo, hi,
                                 dt_source, smem, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_scan_pk(float* cstate,
    const bf16* xbc, const bf16* proj, const bf16* A_log, const bf16* dt_bias, float* dtv,
    float* cum, __half* state, const int* req, unsigned H, unsigned P, unsigned R, unsigned T, unsigned N,
    unsigned G, float lo, float hi, unsigned dt_source) {
    extern __shared__ float smem[];
    d_mamba2_chunk_scan_state_tc<true>(cstate, xbc, proj, A_log, dt_bias, dtv, cum, state, H, P, R, T, N, G, lo,
                                       hi, dt_source, smem, blockIdx.x, gridDim.x, req);
}

/* The fused walk storing bf16 entering states and the CHUNK_OUT that reads them (the interpreter's
 * pair when CHUNK_OUT follows the pass). */
extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_scan_bf_tc_k(float* cstate,
    const bf16* xbc, const bf16* proj, const bf16* A_log, const bf16* dt_bias, float* dtv,
    float* cum, __half* state, unsigned H, unsigned P, unsigned R, unsigned T, unsigned N, unsigned G,
    float lo, float hi, unsigned dt_source) {
    extern __shared__ float smem[];
    d_mamba2_chunk_scan_state_tc(cstate, xbc, proj, A_log, dt_bias, dtv, cum, state, H, P, R, T, N, G, lo, hi,
                                 dt_source, smem, blockIdx.x, gridDim.x, nullptr, true);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_scan_bf_pk(float* cstate,
    const bf16* xbc, const bf16* proj, const bf16* A_log, const bf16* dt_bias, float* dtv,
    float* cum, __half* state, const int* req, unsigned H, unsigned P, unsigned R, unsigned T, unsigned N,
    unsigned G, float lo, float hi, unsigned dt_source) {
    extern __shared__ float smem[];
    d_mamba2_chunk_scan_state_tc<true>(cstate, xbc, proj, A_log, dt_bias, dtv, cum, state, H, P, R, T, N, G, lo,
                                       hi, dt_source, smem, blockIdx.x, gridDim.x, req, true);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_out_bf_tc_k(bf16* y, const bf16* xbc,
    const float* enter, const float* cum, const float* dtv, const bf16* D, unsigned H, unsigned P,
    unsigned R, unsigned T, unsigned N, unsigned G) {
    extern __shared__ float smem[];
    d_mamba2_chunk_out_tc(y, xbc, enter, cum, dtv, D, H, P, R, T, N, G, smem, blockIdx.x, gridDim.x, nullptr, true);
}

extern "C" __global__ void __launch_bounds__(256) mamba2_chunk_out_bf_pk(bf16* y, const bf16* xbc,
    const float* enter, const float* cum, const float* dtv, const bf16* D, const int* req,
    unsigned H, unsigned P, unsigned R, unsigned T, unsigned N, unsigned G) {
    extern __shared__ float smem[];
    d_mamba2_chunk_out_tc<true>(y, xbc, enter, cum, dtv, D, H, P, R, T, N, G, smem, blockIdx.x, gridDim.x, req,
                                true);
}
