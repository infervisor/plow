/* ROCm/HIP golden attention (BACKEND_ROCM, variant 0x00) — naive single-thread
 * reference for MI300, must match the CPU oracle. Performant MFMA flash lives in
 * mfma.hip. Mirrors runtime/nvidia/flash.cu. */
#include <hip/hip_runtime.h>
#include "kernel.h"
#include "packet.h"

__global__ void hip_flash_naive_f32(float* o, const float* q, const float* k,
                                    const float* v, unsigned sq, unsigned skv,
                                    unsigned hd, unsigned heads, int causal) {
    float scale = rsqrtf((float)hd);
    for (unsigned h = 0; h < heads; ++h) {
        const float* qh = q + (size_t)h * sq * hd;
        const float* kh = k + (size_t)h * skv * hd;
        const float* vh = v + (size_t)h * skv * hd;
        float* oh = o + (size_t)h * sq * hd;
        for (unsigned i = 0; i < sq; ++i) {
            unsigned last = causal ? (i < skv ? i : skv - 1) : skv - 1;
            float mx = -1e30f, sum = 0.0f;
            for (unsigned j = 0; j <= last; ++j) {
                float s = 0.0f;
                for (unsigned d = 0; d < hd; ++d) s += qh[i * hd + d] * kh[j * hd + d];
                s *= scale; if (s > mx) mx = s;
            }
            for (unsigned d = 0; d < hd; ++d) oh[i * hd + d] = 0.0f;
            for (unsigned j = 0; j <= last; ++j) {
                float s = 0.0f;
                for (unsigned d = 0; d < hd; ++d) s += qh[i * hd + d] * kh[j * hd + d];
                float e = expf(s * scale - mx); sum += e;
                for (unsigned d = 0; d < hd; ++d) oh[i * hd + d] += e * vh[j * hd + d];
            }
            for (unsigned d = 0; d < hd; ++d) oh[i * hd + d] /= sum;
        }
    }
}

extern "C" void hip_flash_golden(const void* body, kctx* ctx) {
    const PlowFlashBody* f = (const PlowFlashBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const float* Q = (const float*)ctx->slots[bd->in0];
    const float* K = (const float*)ctx->slots[bd->in1];
    const float* V = (const float*)ctx->slots[bd->in2];
    float* O = (float*)ctx->slots[f->out];
    hipLaunchKernelGGL(hip_flash_naive_f32, dim3(1), dim3(1), 0, (hipStream_t)ctx->stream,
                       O, Q, K, V, f->seq_q, f->seq_kv, f->head_dim, f->heads, bd->detail != 0);
}
