#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>

static hipModule_t module;
static hipFunction_t attention;

extern "C" int native_load(const char *path) {
    hipError_t status = hipModuleLoad(&module, path);
    if (status != hipSuccess)
        return status;
    return hipModuleGetFunction(
        &attention, module, "_ZN5aiter36mla_a16w16_qh8_qseqlen1_gqaratio8_v3E");
}

extern "C" int native_attention(void *q, void *kv, void *idx, void *qp,
                                void *kp, void *last, void *splits,
                                void *partial, void *lse, unsigned rows,
                                float scale, void *stream) {
    // The gfx942 QH8 code object has 20 padded 16-byte argument slots.
    uint64_t args[40] = {};
    args[0] = (uint64_t)partial;
    args[2] = (uint64_t)lse;
    args[4] = (uint64_t)q;
    args[6] = (uint64_t)kv;
    args[8] = (uint64_t)kp;
    args[10] = (uint64_t)idx;
    args[12] = (uint64_t)last;
    std::memcpy(&args[14], &scale, sizeof(scale));
    args[16] = 8;
    args[18] = 2;
    args[20] = 8 * 576 * 2;
    args[22] = 576 * 2;
    args[26] = (uint64_t)qp;
    args[28] = (uint64_t)splits;
    size_t size = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    return hipModuleLaunchKernel(attention, 1, rows, 2, 256, 1, 1, 0,
                                 (hipStream_t)stream, nullptr, config);
}

__global__ void reduce_two(float *out, float *ml, const float *part,
                           const float *lse, unsigned rows) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < rows * 8u * 128u; i += blockDim.x * gridDim.x) {
        const unsigned hrow = i / 128u, col = i % 128u;
        const unsigned row = hrow / 8u, head = hrow % 8u;
        const unsigned l = row * 16u + head;
        const float a = lse[l], b = lse[l + 8u], m = fmaxf(a, b);
        const float wa = expf(a - m), wb = expf(b - m);
        const float4 pa = ((const float4 *)part)[l * 128u + col];
        const float4 pb = ((const float4 *)part)[(l + 8u) * 128u + col];
        ((float4 *)out)[i] = make_float4((pa.x * wa + pb.x * wb) / (wa + wb),
                                         (pa.y * wa + pb.y * wb) / (wa + wb),
                                         (pa.z * wa + pb.z * wb) / (wa + wb),
                                         (pa.w * wa + pb.w * wb) / (wa + wb));
        if (col == 0) {
            ml[hrow * 2u] = 0;
            ml[hrow * 2u + 1u] = 1;
        }
    }
}

extern "C" int native_reduce(void *out, void *ml, void *partial, void *lse,
                             unsigned rows, void *stream) {
    reduce_two<<<304, 256, 0, (hipStream_t)stream>>>((float *)out, (float *)ml,
                                                     (const float *)partial,
                                                     (const float *)lse, rows);
    return hipGetLastError();
}
