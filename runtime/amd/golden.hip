/* ROCm/HIP golden kernels (BACKEND_ROCM, variant 0x00) — naive single-thread
 * references for MI300 (CDNA3), must match the CPU oracle. Build-gated on hipcc.
 * Performant MFMA variants live in mfma.hip. */
#include <hip/hip_runtime.h>
#include "kernel.h"
#include "packet.h"

__global__ void hip_gemm_naive_f32(float* c, const float* a, const float* b,
                                   const float* bias, unsigned m, unsigned n, unsigned k) {
    for (unsigned i = 0; i < m; ++i)
        for (unsigned j = 0; j < n; ++j) {
            float acc = bias ? bias[j] : 0.0f;
            for (unsigned p = 0; p < k; ++p) acc += a[i * k + p] * b[j * k + p];
            c[i * n + j] = acc;
        }
}

extern "C" void hip_gemm_golden(const void* body, kctx* ctx) {
    const PlowGemmBody* g = (const PlowGemmBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const float* A = (const float*)ctx->slots[bd->in0];
    const float* B = (const float*)ctx->slots[bd->in1];
    const float* bias = (bd->in2 != PLOW_SLOT_NONE) ? (const float*)ctx->slots[bd->in2] : nullptr;
    float* C = (float*)ctx->slots[g->out];
    hipLaunchKernelGGL(hip_gemm_naive_f32, dim3(1), dim3(1), 0,
                       (hipStream_t)ctx->stream, C, A, B, bias, g->m, g->n, g->k);
}

extern "C" void hip_dma_golden(const void* body, kctx* ctx) {
    const PlowDmaBody* d = (const PlowDmaBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    hipStream_t s = (hipStream_t)ctx->stream;
    if (bd->detail == 0)
        hipMemcpyAsync(ctx->slots[d->slot], ctx->tensors[bd->in0], d->bytes, hipMemcpyDefault, s);
    else
        hipMemcpyAsync(ctx->tensors[bd->in0], ctx->slots[d->slot], d->bytes, hipMemcpyDefault, s);
}

extern "C" void hip_rdma_golden(const void* body, kctx* ctx) {
    const PlowRdmaBody* r = (const PlowRdmaBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    hipMemcpyAsync(ctx->slots[bd->in2], ctx->slots[bd->in0], r->bytes,
                   hipMemcpyDeviceToDevice, (hipStream_t)ctx->stream);
}

/* LAYOUT (v4 descriptor): kind 0 contiguous copy; kind 1 strided gather; kind 2
 * binary concat scatter (golden; mirrors cpu_layout). */
__device__ void hip_scatter_piece(uint8_t* dst, const uint8_t* src, uint32_t r, uint32_t es,
                                  const uint32_t* pshape, const uint32_t* out_stride,
                                  size_t out_base) {
    size_t in_str[PLOW_LAYOUT_MAX_RANK];
    size_t acc = 1;
    for (int d = (int)r - 1; d >= 0; --d) { in_str[d] = acc; acc *= pshape[d]; }
    size_t total = (r > 0) ? 1 : 0;
    for (uint32_t d = 0; d < r; ++d) total *= pshape[d];
    uint32_t idx[PLOW_LAYOUT_MAX_RANK];
    for (uint32_t d = 0; d < r; ++d) idx[d] = 0;
    for (size_t n = 0; n < total; ++n) {
        size_t in_off = 0, out_off = out_base;
        for (uint32_t d = 0; d < r; ++d) {
            in_off += (size_t)idx[d] * in_str[d];
            out_off += (size_t)idx[d] * out_stride[d];
        }
        for (uint32_t k = 0; k < es; ++k) dst[out_off * es + k] = src[in_off * es + k];
        for (int d = (int)r - 1; d >= 0; --d) { if (++idx[d] < pshape[d]) break; idx[d] = 0; }
    }
}

__global__ void hip_layout_strided(uint8_t* dst, const uint8_t* src, const uint8_t* src1,
                                   PlowLayoutBody L) {
    const uint32_t r = L.rank, es = L.elem_size;
    if (L.kind == 2) {
        const uint32_t axis = L.in_base, split = L.out_base;
        uint32_t p0[PLOW_LAYOUT_MAX_RANK], p1[PLOW_LAYOUT_MAX_RANK];
        for (uint32_t d = 0; d < r; ++d) { p0[d] = L.shape[d]; p1[d] = L.shape[d]; }
        p0[axis] = split;
        p1[axis] = L.shape[axis] - split;
        hip_scatter_piece(dst, src, r, es, p0, L.out_stride, 0);
        hip_scatter_piece(dst, src1, r, es, p1, L.out_stride, (size_t)split * L.out_stride[axis]);
        return;
    }
    size_t total = (r > 0) ? 1 : 0;
    for (uint32_t d = 0; d < r; ++d) total *= L.shape[d];
    uint32_t idx[PLOW_LAYOUT_MAX_RANK];
    for (uint32_t d = 0; d < r; ++d) idx[d] = 0;
    for (size_t n = 0; n < total; ++n) {
        size_t in_off = L.in_base, out_off = L.out_base;
        for (uint32_t d = 0; d < r; ++d) {
            in_off += (size_t)idx[d] * L.in_stride[d];
            out_off += (size_t)idx[d] * L.out_stride[d];
        }
        for (uint32_t k = 0; k < es; ++k) dst[out_off * es + k] = src[in_off * es + k];
        for (int d = (int)r - 1; d >= 0; --d) {
            if (++idx[d] < L.shape[d]) break;
            idx[d] = 0;
        }
    }
}

extern "C" void hip_layout_golden(const void* body, kctx* ctx) {
    const PlowLayoutBody* L = (const PlowLayoutBody*)body;
    const PlowBinding* bd = ctx->bind;
    if (!bd) return;
    const uint8_t* src = (const uint8_t*)ctx->slots[bd->in0];
    const uint8_t* src1 = (bd->in1 != PLOW_SLOT_NONE) ? (const uint8_t*)ctx->slots[bd->in1] : nullptr;
    uint8_t* dst = (uint8_t*)ctx->slots[L->out];
    if (L->kind == 0) {
        size_t total = (L->rank > 0) ? 1 : 0;
        for (uint32_t d = 0; d < L->rank; ++d) total *= L->shape[d];
        hipMemcpyAsync(dst, src, total * L->elem_size, hipMemcpyDeviceToDevice,
                       (hipStream_t)ctx->stream);
        return;
    }
    hipLaunchKernelGGL(hip_layout_strided, dim3(1), dim3(1), 0,
                       (hipStream_t)ctx->stream, dst, src, src1, *L);
}

/* CONTROL family (nop / host-coord): no data effect. Mirrors cpu_host. */
extern "C" void hip_host(const void* body, kctx* ctx) {
    (void)body;
    (void)ctx;
}
