/* AttentionF32's stream arm, with the packet's row layout and persistent block count. */
#include <cuda_runtime.h>
#define PLOW_NV_SPEECH 1
#include "op_speech_f32.cuh"

__global__ void __launch_bounds__(256, 1) codec_attention(PlowDevInst in, void* const* tensors) {
    extern __shared__ float arena[];
    d_attention_f32(&in, tensors, blockIdx.x, gridDim.x, arena);
}

struct CodecAttention {
    PlowDevInst inst;
    void** tensors;
};

extern "C" void* ca_make(void* out, void* qkv, void* positions, void* lengths,
                         unsigned batch, unsigned queries, unsigned history,
                         unsigned heads, unsigned width, unsigned window) {
    auto* op = new CodecAttention{};
    for (auto& t : op->inst.t) t = PLOW_TENSOR_NONE;
    void* tensors[] = {out, qkv, positions, lengths};
    op->inst.t[0] = 0;
    op->inst.t[1] = op->inst.t[2] = op->inst.t[3] = 1;
    if (lengths) op->inst.t[4] = 3;
    op->inst.t[7] = 2;
    const unsigned fields[] = {batch, queries, history + queries, heads, width,
                               3 * heads * width, 9u | window << 16, history};
    for (unsigned i = 0; i < 8; ++i) op->inst.i[i] = fields[i];
    op->inst.fj[0].f = 1.f / sqrtf((float)width);
    op->inst.fj[1].u = heads * width;
    op->inst.fj[2].u = 2 * heads * width;
    if (cudaMalloc(&op->tensors, sizeof(tensors)) != cudaSuccess) {
        delete op;
        return nullptr;
    }
    if (cudaMemcpy(op->tensors, tensors, sizeof(tensors), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaFuncSetAttribute(codec_attention, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SP_LAUNCH_ARENA_FLOATS * 4) != cudaSuccess) {
        cudaFree(op->tensors);
        delete op;
        return nullptr;
    }
    return op;
}

extern "C" int ca_run(void* handle, unsigned blocks, void* stream) {
    auto* op = static_cast<CodecAttention*>(handle);
    codec_attention<<<blocks, 256, SP_LAUNCH_ARENA_FLOATS * 4,
                       static_cast<cudaStream_t>(stream)>>>(op->inst, op->tensors);
    return (int)cudaGetLastError();
}

extern "C" void ca_free(void* handle) {
    auto* op = static_cast<CodecAttention*>(handle);
    cudaFree(op->tensors);
    delete op;
}
