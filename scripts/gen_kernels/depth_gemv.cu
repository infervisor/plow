/* Standalone probe of the Maya 2 depth projection as the interpreter runs it: GemvF32Out's flag
 * modes (d_gemv_f32out_mode -> gemv_f16w_mma) on nblk blocks x 256 threads with the speech arena,
 * behind a C ABI for gen_search (spec_maya2_depth_gemv.py, ctypes).
 *   nvcc -gencode arch=compute_90a,code=sm_90a -DPLOW_NV_HOPPER=1 -O3 -shared -Xcompiler -fPIC -I runtime/common -I runtime/nvidia \
 *        scripts/gen_kernels/depth_gemv.cu -o depth_gemv.so
 * flags = the emitted i[7] (maya2_depth.rs): 1 x f32, 2 W f16, 4 bias, 8 residual, 16 GELU,
 * 256 LayerNorm prologue, 1024 mma arm. */
#include <cuda_runtime.h>
#include <cstring>
#define PLOW_NV_SPEECH 1
#include "op_speech_f32.cuh"
#include "op_elementwise.cuh"

struct DgArgs {
    float* c;
    const float* x;
    const unsigned short* w;
    const float *bias, *res, *gamma, *beta;
    unsigned flags, m, n, k;
};

__global__ void __launch_bounds__(256, 1) k_depth_gemv(DgArgs a) {
    d_gemv_f32out_mode(a.c, a.x, a.w, a.bias, a.res, a.gamma, a.beta, 1e-5f, a.flags, 0u, a.m, a.n, a.k,
                       blockIdx.x, gridDim.x, sp_smem, nullptr);
}

extern "C" {
void* dg_make(float* c, const float* x, const unsigned short* w, const float* bias, const float* res,
              const float* gamma, const float* beta, unsigned flags, unsigned m, unsigned n, unsigned k) {
    DgArgs* a = new DgArgs{c, x, w, bias, res, gamma, beta, flags, m, n, k};
    return a;
}
int dg_run(void* hp, int nblk, void* stream) {
    static bool init = false;
    const size_t arena = SP_ARENA_FLOATS * sizeof(float);
    if (!init) {
        init = true;
        cudaFuncSetAttribute(k_depth_gemv, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena);
    }
    k_depth_gemv<<<nblk, 256, arena, (cudaStream_t)stream>>>(*(DgArgs*)hp);
    return (int)cudaGetLastError();
}
}
