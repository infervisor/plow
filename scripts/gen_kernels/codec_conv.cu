/* Standalone probe of the speech interpreter's Conv1dF32 / UnaryF32 arms behind a C ABI for the
 * codec gen_search specs (ctypes): the production kernel code (op_speech_f32.cuh) launched as the
 * interpreter runs it (nblk blocks x 256 threads, the launch arena), one op per launch, with the
 * flags word and weight layout a packet carries.
 *   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -shared -Xcompiler -fPIC -DPLOW_NV_HOPPER=1 -I runtime/common \
 *        -I runtime/nvidia scripts/gen_kernels/codec_conv.cu -o codec_conv.so
 */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "op_speech_f32.cuh"

__global__ void __launch_bounds__(256, 1) k_speech(PlowDevInst in, void* const* T) {
    extern __shared__ float arena[];
#if defined(PLOW_NV_HOPPER)
    if (sp_wg_conv1d(&in, T, blockIdx.x, gridDim.x)) return;
#endif
    d_speech_f32(&in, T, blockIdx.x, gridDim.x, arena);
}

struct Op { PlowDevInst in; void** table; };

static uint16_t bf16_rn(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

extern "C" {

/* The weight tensor a packet carries: tap-major f32 [n][k] (k = kernel * cin), then with `split`
 * (flags bit 18) the pre-split wgmma image at the next 1 KiB (devgen wgmma_weight_split). Returns
 * a device buffer. */
void* cc_weights(const float* w, unsigned n, unsigned k, int split) {
    std::vector<uint8_t> h((size_t)n * k * 4);
    memcpy(h.data(), w, h.size());
    if (split) {
        const size_t base = (h.size() + 1023) & ~(size_t)1023, np = (n + 127) & ~127u, kt = (k + 63) / 64;
        h.resize(base + kt * 2 * np * 128, 0);
        for (size_t t = 0; t < kt; t++)
            for (size_t r = 0; r < n; r++)
                for (size_t j = 0; j < 64 && t * 64 + j < k; j++) {
                    const float v = w[r * k + t * 64 + j];
                    const uint16_t hi = bf16_rn(v);
                    const uint32_t hb = (uint32_t)hi << 16;
                    float hf;
                    memcpy(&hf, &hb, 4);
                    const uint16_t lo = bf16_rn(v - hf);
                    const size_t off = base + t * 2 * np * 128 + r * 128 + (((j / 8) ^ (r & 7)) * 16) + (j % 8) * 2;
                    memcpy(&h[off], &hi, 2);
                    memcpy(&h[off + np * 128], &lo, 2);
                }
    }
    void* d = nullptr;
    if (cudaMalloc(&d, h.size()) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice);
    return d;
}

static Op* make(const PlowDevInst& in, void* const* ptrs, int n) {
    Op* op = new Op{in, nullptr};
    if (cudaMalloc(&op->table, n * sizeof(void*)) != cudaSuccess) return nullptr;
    cudaMemcpy(op->table, ptrs, n * sizeof(void*), cudaMemcpyHostToDevice);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(k_speech, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(SP_LAUNCH_ARENA_FLOATS * 4));
        attr = true;
    }
    return op;
}

/* Conv1dF32: tensors out, x, w, bias, alpha, residual, scratch (0 = none); `flags` is the packet's
 * flags word (fj[2]); t7 = scratch when flags bit 20 is set. */
void* cc_conv(void* out, void* x, void* w, void* bias, void* alpha, void* residual, void* scratch, unsigned batch,
              unsigned in_rows, unsigned cin, unsigned cout, unsigned kernel, unsigned dil, unsigned before,
              unsigned flags) {
    void* ptrs[7] = {out, x, w, bias, alpha, residual, scratch};
    PlowDevInst in{};
    in.op = PLOW_DOP_CONV1D_F32;
    for (int i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2;
    if (bias) in.t[3] = 3;
    if (alpha) in.t[4] = 4;
    if (residual) in.t[5] = 5;
    if (scratch) in.t[7] = 6;
    in.i[0] = batch; in.i[1] = in_rows; in.i[2] = cin; in.i[3] = cout; in.i[4] = kernel; in.i[5] = 1;
    in.i[6] = dil; in.i[7] = 1;
    in.fj[1].u = before;
    in.fj[2].u = flags;
    return make(in, ptrs, 7);
}

/* UnaryF32 in place or out of place: kind 19 = SnakeBeta with param [a | r]. */
void* cc_unary(void* out, void* x, void* param, unsigned rows, unsigned width, unsigned kind) {
    void* ptrs[3] = {out, x, param};
    PlowDevInst in{};
    in.op = PLOW_DOP_UNARY_F32;
    for (int i = 0; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    in.t[0] = 0; in.t[1] = 1;
    if (param) in.t[2] = 2;
    in.i[0] = rows; in.i[1] = width; in.i[2] = kind;
    return make(in, ptrs, 3);
}

int cc_run(void* handle, int nblk, void* stream) {
    const Op* op = (const Op*)handle;
    k_speech<<<nblk, 256, SP_LAUNCH_ARENA_FLOATS * 4, (cudaStream_t)stream>>>(op->in, op->table);
    return (int)cudaGetLastError();
}

}
