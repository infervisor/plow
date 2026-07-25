/* CPU golden scaled-dot-product attention — reference softmax, per head. */
#include "cpu_kernels.h"
#include <math.h>
#include <stdlib.h>

/* o[heads,sq,hd] = softmax(q·k^T/sqrt(hd) + causal_mask) · v. Layout is
 * [head][seq][dim], row-major. */
void plow_flash_ref(float* o, const float* q, const float* k, const float* v,
                    uint32_t sq, uint32_t skv, uint32_t hd, uint32_t heads, int causal) {
    float scale = 1.0f / sqrtf((float)hd);
    float* row = (float*)malloc((size_t)skv * sizeof(float));
    for (uint32_t h = 0; h < heads; h++) {
        const float* qh = q + (size_t)h * sq * hd;
        const float* kh = k + (size_t)h * skv * hd;
        const float* vh = v + (size_t)h * skv * hd;
        float* oh = o + (size_t)h * sq * hd;
        for (uint32_t i = 0; i < sq; i++) {
            uint32_t last = causal ? (i < skv ? i : skv - 1) : skv - 1;
            float mx = -INFINITY;
            for (uint32_t j = 0; j <= last; j++) {
                float s = 0.0f;
                for (uint32_t d = 0; d < hd; d++)
                    s += qh[i * hd + d] * kh[j * hd + d];
                s *= scale;
                row[j] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (uint32_t j = 0; j <= last; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
            for (uint32_t d = 0; d < hd; d++) {
                float acc = 0.0f;
                for (uint32_t j = 0; j <= last; j++)
                    acc += (row[j] / sum) * vh[j * hd + d];
                oh[i * hd + d] = acc;
            }
        }
    }
    free(row);
}

void cpu_flash(const void* body, kctx* ctx) {
    const PlowFlashBody* f = (const PlowFlashBody*)body;
    const PlowBinding* b = ctx->bind;
    if (!b) return;
    const float* Q = (const float*)ctx->slots[b->in0];
    const float* K = (const float*)ctx->slots[b->in1];
    const float* V = (const float*)ctx->slots[b->in2];
    float* O = (float*)ctx->slots[f->out];
    int causal = b->detail != 0;
    plow_flash_ref(O, Q, K, V, f->seq_q, f->seq_kv, f->head_dim, f->heads, causal);
}
