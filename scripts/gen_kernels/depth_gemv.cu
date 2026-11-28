/* Standalone probe of the Maya 2 depth projection as the interpreter runs it: GemvF32Out's flag
 * modes (d_gemv_f32out_mode -> gemv_f16w_mma) on nblk blocks x 256 threads with the speech arena,
 * behind a C ABI for gen_search (spec_maya2_depth_gemv.py, ctypes).
 *   nvcc -gencode arch=compute_90a,code=sm_90a -DPLOW_NV_HOPPER=1 -O3 -shared -Xcompiler -fPIC -I runtime/common -I runtime/nvidia \
 *        scripts/gen_kernels/depth_gemv.cu -o depth_gemv.so
 * flags = the emitted i[7] (maya2_depth.rs): 1 x f32, 2 W f16, 4 bias, 8 residual, 16 GELU,
 * 256 LayerNorm prologue, 1024 mma arm. */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#define PLOW_NV_SPEECH 1
#include "op_speech_f32.cuh"
#include "op_elementwise.cuh"

struct DgArgs {
    float* c;
    const float* x;
    const unsigned short* w;
    const float *bias, *res, *gamma, *beta;
    unsigned flags, m, n, k, bgroup;
};

__global__ void __launch_bounds__(256, 1) k_depth_gemv(DgArgs a) {
    d_gemv_f32out_mode(a.c, a.x, a.w, a.bias, a.res, a.gamma, a.beta, 1e-5f, a.flags, a.bgroup, a.m, a.n, a.k,
                       blockIdx.x, gridDim.x, sp_smem, nullptr);
}

#if defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE
template <unsigned BM, bool HALF_OUT, unsigned BK = 64>
__global__ void __launch_bounds__(256, 1) k_depth_gemv_wide(DgArgs a) {
    gemv_f16w_wide<BM, HALF_OUT, BK>(a.c, (const unsigned short*)a.x, a.w,
        a.flags & 4u ? a.bias : nullptr, a.flags & 512u ? a.bgroup : 0u,
        a.flags & 8u ? a.res : nullptr, a.flags >> 4 & 15u,
        a.m, a.n, a.k, blockIdx.x, gridDim.x, sp_smem);
}

template <unsigned BM, bool HALF_OUT, unsigned BK = 64>
static int run_wide(DgArgs a, int nblk, void* stream) {
    static bool init = false;
    const size_t arena = SP_ARENA_FLOATS * sizeof(float);
    if (!init) {
        auto error = cudaFuncSetAttribute(k_depth_gemv_wide<BM, HALF_OUT, BK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena);
        if (error != cudaSuccess) return (int)error;
        init = true;
    }
    k_depth_gemv_wide<BM, HALF_OUT, BK><<<nblk, 256, arena, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
}

template <unsigned BM, unsigned BK, bool HALF_OUT>
__global__ void __launch_bounds__(256, 1) k_depth_gemv_half(DgArgs a) {
    gemv_f16w_half<BM, HALF_OUT, BK>(a.c, (const unsigned short*)a.x, a.w,
        a.flags & 4u ? a.bias : nullptr, a.flags & 512u ? a.bgroup : 0u,
        a.flags & 8u ? a.res : nullptr, a.flags >> 4 & 15u,
        a.m, a.n, a.k, blockIdx.x, gridDim.x, sp_smem);
}

template <unsigned BM, unsigned BK, bool HALF_OUT>
static int run_half(DgArgs a, int nblk, void* stream) {
    static bool init = false;
    const size_t arena = SP_ARENA_FLOATS * sizeof(float);
    if (!init) {
        auto error = cudaFuncSetAttribute(k_depth_gemv_half<BM, BK, HALF_OUT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena);
        if (error != cudaSuccess) return (int)error;
        init = true;
    }
    k_depth_gemv_half<BM, BK, HALF_OUT><<<nblk, 256, arena, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
}

template <unsigned BM, unsigned BK>
__global__ void __launch_bounds__(256, 1) k_depth_gemv_output_view(
    DgArgs a, unsigned rows, unsigned stride, unsigned offset) {
    gemv_f16w_half<BM, false, BK, true>(a.c, (const unsigned short*)a.x, a.w,
        nullptr, 0, nullptr, 0, a.m, a.n, a.k, blockIdx.x, gridDim.x, sp_smem,
        rows, stride, offset);
}

template <unsigned BM, unsigned BK = 64>
static int run_output_view(DgArgs a, int nblk, void* stream,
                            unsigned rows, unsigned stride, unsigned offset) {
    static bool init = false;
    const size_t arena = SP_ARENA_FLOATS * sizeof(float);
    if (!init) {
        auto error = cudaFuncSetAttribute(k_depth_gemv_output_view<BM, BK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena);
        if (error != cudaSuccess) return (int)error;
        init = true;
    }
    k_depth_gemv_output_view<BM, BK><<<nblk, 256, arena, (cudaStream_t)stream>>>(a, rows, stride, offset);
    return (int)cudaGetLastError();
}

__global__ void k_depth_copy_output_view(const float* src, float* dst, unsigned m, unsigned n,
                                         unsigned rows, unsigned stride, unsigned offset) {
    PlowDevInst in = {};
    in.op = PLOW_DOP_COPY_COLS_F32;
    for (unsigned t = 0; t < 8; t++) in.t[t] = PLOW_TENSOR_NONE;
    in.t[0] = 0; in.t[1] = 1;
    in.i[0] = m / rows; in.i[1] = rows; in.i[2] = n;
    in.i[3] = n; in.i[5] = n; in.i[6] = offset;
    in.fj[1].u = rows * n; in.fj[2].u = stride;
    void* tensors[] = {dst, (void*)src};
    d_copy_cols_f32(&in, tensors, blockIdx.x, gridDim.x);
}

static bool valid_output_view(unsigned m, unsigned n, unsigned rows,
                                unsigned out_rows, unsigned offset, size_t capacity) {
    return rows && m % rows == 0 && offset <= out_rows && rows <= out_rows - offset &&
           (size_t)out_rows * n <= 0xffffffffu &&
           (size_t)(m / rows) * out_rows * n <= SIZE_MAX / sizeof(float) &&
           (size_t)(m / rows) * out_rows * n <= capacity;
}

static bool buffers_overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    const auto aa = (uintptr_t)a, bb = (uintptr_t)b;
    return aa <= bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}
#endif

extern "C" {
void* dg_make(float* c, const float* x, const unsigned short* w, const float* bias, const float* res,
              const float* gamma, const float* beta, unsigned flags, unsigned m, unsigned n, unsigned k) {
    DgArgs* a = new DgArgs{c, x, w, bias, res, gamma, beta, flags, m, n, k, 0};
    return a;
}
void* dg_make_grouped(float* c, const float* x, const unsigned short* w, const float* bias, const float* res,
                     const float* gamma, const float* beta, unsigned flags, unsigned m, unsigned n,
                     unsigned k, unsigned bgroup) {
    return new DgArgs{c, x, w, bias, res, gamma, beta, flags, m, n, k, bgroup};
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
#if defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE
int dg_run_variant(void* hp, int nblk, void* stream, unsigned variant) {
    if (!hp) return (int)cudaErrorInvalidValue;
    if (variant == 0) return dg_run(hp, nblk, stream);
    DgArgs a = *(DgArgs*)hp;
    if (nblk <= 0 || a.m < 16u || !a.n || !a.k || a.n % (variant <= 2 || variant == 7 ? 128u : 64u) || a.k % 64u ||
        !a.c || !a.x || !a.w || ((((size_t)a.x) | ((size_t)a.w)) & 15u) ||
        (a.flags & (3u | 1024u | 8192u | 256u | 4096u)) != (3u | 1024u | 8192u) ||
        ((a.flags & 512u) && !a.bgroup) || ((a.flags & 16384u) && a.c == a.res)) return (int)cudaErrorInvalidValue;
    if (variant == 1) return a.flags & 16384u ? run_wide<64, true>(a, nblk, stream) : run_wide<64, false>(a, nblk, stream);
    if (variant == 2) return a.flags & 16384u ? run_wide<128, true>(a, nblk, stream) : run_wide<128, false>(a, nblk, stream);
    if (variant == 3) return a.flags & 16384u ? run_half<32, 64, true>(a, nblk, stream) : run_half<32, 64, false>(a, nblk, stream);
    if (variant == 4) return a.flags & 16384u ? run_half<64, 64, true>(a, nblk, stream) : run_half<64, 64, false>(a, nblk, stream);
    if (variant == 5) return a.flags & 16384u ? run_half<128, 64, true>(a, nblk, stream) : run_half<128, 64, false>(a, nblk, stream);
    if (variant == 6 && a.k % 128u == 0) return a.flags & 16384u ? run_half<64, 128, true>(a, nblk, stream) : run_half<64, 128, false>(a, nblk, stream);
    if (variant == 7 && a.k % 128u == 0) return a.flags & 16384u ? run_wide<64, true, 128>(a, nblk, stream) : run_wide<64, false, 128>(a, nblk, stream);
    if (variant == 8 && a.k % 128u == 0) return a.flags & 16384u ? run_half<128, 128, true>(a, nblk, stream) : run_half<128, 128, false>(a, nblk, stream);
    if (variant == 9) return a.flags & 16384u ? run_half<16, 64, true>(a, nblk, stream) : run_half<16, 64, false>(a, nblk, stream);
    return (int)cudaErrorInvalidValue;
}

int dg_run_output_view(void* hp, int nblk, void* stream, unsigned rows,
                       unsigned out_rows, unsigned offset, size_t capacity) {
    if (!hp || nblk <= 0) return (int)cudaErrorInvalidValue;
    const auto a = *(DgArgs*)hp;
    if (a.m < 16 || a.m > 0xffffff00u || !a.n || !a.k || (size_t)a.m * a.n > 0xffffffffu ||
        a.n % 64u || a.k % 64u || !a.c || !a.x || !a.w ||
        ((((size_t)a.x) | ((size_t)a.w) | ((size_t)a.c)) & 15u) ||
        a.flags != (3u | 1024u | 8192u) || a.c == a.x || a.c == (const void*)a.w ||
        !valid_output_view(a.m, a.n, rows, out_rows, offset, capacity)) return (int)cudaErrorInvalidValue;
    const size_t output_bytes = (size_t)(a.m / rows) * out_rows * a.n * sizeof(float);
    if (buffers_overlap(a.c, output_bytes, a.x, (size_t)a.m * a.k * 2u) ||
        buffers_overlap(a.c, output_bytes, a.w, (size_t)a.n * a.k * 2u)) return (int)cudaErrorInvalidValue;
    const unsigned stride = out_rows * a.n, col0 = offset * a.n;
    if (a.m >= 256u && a.k % 128u == 0u) return run_output_view<128, 128>(a, nblk, stream, rows, stride, col0);
    if (a.m >= 256u) return run_output_view<128>(a, nblk, stream, rows, stride, col0);
    if (a.n <= 1024u) return run_output_view<16>(a, nblk, stream, rows, stride, col0);
    if ((size_t)((a.m + 63u) / 64u) * (a.n / 64u) >= 96u)
        return run_output_view<64>(a, nblk, stream, rows, stride, col0);
    return run_output_view<32>(a, nblk, stream, rows, stride, col0);
}

int dg_copy_output_view(const float* src, float* dst, unsigned m, unsigned n, int nblk,
                        void* stream, unsigned rows, unsigned out_rows, unsigned offset, size_t capacity) {
    if (!src || !dst || src == dst || ((((uintptr_t)src) | ((uintptr_t)dst)) & 3u) ||
        !m || !n || nblk <= 0 || (size_t)m * n > 0xffffffffu ||
        !valid_output_view(m, n, rows, out_rows, offset, capacity)) return (int)cudaErrorInvalidValue;
    if (buffers_overlap(src, (size_t)m * n * sizeof(float), dst,
                        (size_t)(m / rows) * out_rows * n * sizeof(float))) return (int)cudaErrorInvalidValue;
    k_depth_copy_output_view<<<nblk, 256, 0, (cudaStream_t)stream>>>(src, dst, m, n, rows,
                                                                 out_rows * n, offset * n);
    return (int)cudaGetLastError();
}
#endif
}
