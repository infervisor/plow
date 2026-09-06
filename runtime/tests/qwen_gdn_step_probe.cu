// Standalone recurrence reuse experiment; production op137 stays unchanged.
#include <cuda_runtime.h>
#include "../nvidia/op_qwen_gdn.cuh"
#ifndef GDN_VROWS
#define GDN_VROWS 4
#endif
static_assert(GDN_VROWS == 1 || GDN_VROWS == 4 || GDN_VROWS == 8);
static __device__ void d_qwen_gdn_step_tiled(__nv_bfloat16* out, const __nv_bfloat16* qkv,
    const __nv_bfloat16* a, const __nv_bfloat16* b, const void* a_log,
    const __nv_bfloat16* dt_bias, float* state, const int* active,
    unsigned hk, unsigned hv, unsigned kdim, unsigned vdim, unsigned batch,
    float scale, float eps, unsigned alog_f32, unsigned slice, unsigned nblk) {
    if (!hk || !hv || hv % hk || kdim != 128 || !vdim) {
        if (threadIdx.x == 0) __trap(); return;
    }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned warps = blockDim.x >> 5;
    const unsigned packed = 2 * hk * kdim + hv * vdim;
    const unsigned tiles = (vdim + GDN_VROWS - 1) / GDN_VROWS;
    for (unsigned tile = slice * warps + warp; tile < batch * hv * tiles; tile += nblk * warps) {
        const unsigned slot = tile / (hv * tiles);
        if (active && active[slot] <= 0) continue;
        const unsigned head = tile / tiles % hv, first_v = (tile % tiles) * GDN_VROWS;
        const unsigned qhead = head / (hv / hk);
        const __nv_bfloat16* x = qkv + (size_t)slot * packed;
        float q[4], k[4], h[4], qq = 0.f, kk = 0.f;
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            unsigned d = lane + 32 * j;
            q[j] = __bfloat162float(x[qhead * kdim + d]);
            k[j] = __bfloat162float(x[hk * kdim + qhead * kdim + d]);
            qq += q[j] * q[j]; kk += k[j] * k[j];
        }
        const float qs = scale / sqrtf(qwen_warp_sum(qq) + eps);
        const float ks = 1.f / sqrtf(qwen_warp_sum(kk) + eps);
        const unsigned gate = slot * hv + head;
        const float dt = __bfloat162float(a[gate]) + __bfloat162float(dt_bias[head]);
        const float softplus = dt > 20.f ? dt : log1pf(expf(dt));
        const float al = alog_f32 ? ((const float*)a_log)[head] : __bfloat162float(((const __nv_bfloat16*)a_log)[head]);
        const float decay = expf(-expf(al) * softplus);
        // vLLM's packed decode rounds beta to the projection dtype before recurrence.
        const float beta = __bfloat162float(__float2bfloat16(1.f / (1.f + expf(-__bfloat162float(b[gate])))));
#pragma unroll
        for (unsigned j = 0; j < 4; j++) { q[j] *= qs; k[j] *= ks; }
#pragma unroll 1
        for (unsigned vcol = first_v; vcol < vdim && vcol < first_v + GDN_VROWS; vcol++) {
            const unsigned row = (slot * hv + head) * vdim + vcol;
            float projection = 0.f;
#pragma unroll
            for (unsigned j = 0; j < 4; j++) {
                h[j] = state[(size_t)row * kdim + lane + 32 * j] * decay;
                projection += h[j] * k[j];
            }
            const float value = __bfloat162float(x[2 * hk * kdim + head * vdim + vcol]);
            const float delta = (value - qwen_warp_sum(projection)) * beta;
            float result = 0.f;
#pragma unroll
            for (unsigned j = 0; j < 4; j++) {
                h[j] = fmaf(delta, k[j], h[j]);
                state[(size_t)row * kdim + lane + 32 * j] = h[j];
                result += h[j] * q[j];
            }
            result = qwen_warp_sum(result);
            if (lane == 0) out[row] = __float2bfloat16(result);
        }
    }
}

struct GdnArgs { void* t[8]; int i[8]; float f[2]; };
__global__ __launch_bounds__(256, 1) void gdn_step_probe(GdnArgs a) {
    d_qwen_gdn_step_tiled((__nv_bfloat16*)a.t[0], (const __nv_bfloat16*)a.t[1],
        (const __nv_bfloat16*)a.t[2], (const __nv_bfloat16*)a.t[3], a.t[4],
        (const __nv_bfloat16*)a.t[5], (float*)a.t[6], (const int*)a.t[7],
        a.i[0], a.i[1], a.i[2], a.i[3], a.i[4], a.f[0], a.f[1], a.i[5],
        blockIdx.x, gridDim.x);
}
extern "C" int qwen_test(unsigned op, void** tensors, const int* integers, const float* floats, void* stream) {
    if (op != 137) return (int)cudaErrorInvalidValue;
    GdnArgs a={};
    for (int i=0;i<8;i++) { a.t[i]=tensors[i]; a.i[i]=integers[i]; }
    for (int i=0;i<2;i++) a.f[i]=floats[i];
    gdn_step_probe<<<132,256,0,(cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
}
