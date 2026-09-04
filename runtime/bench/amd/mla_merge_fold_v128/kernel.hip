#include <hip/hip_runtime.h>

#include "amd_common.h"

#define PLOW_MLA_FOLD_TB 8
#include "op_attention.h"

typedef union __align__(16) {
    float f[PLOW_MLA_FOLD_TB_LDS];
} FoldSmem;

#define FOLD_ARGS                                                                                 \
    bf16 *out, const float *opart, const float *mlpart, const bf16 *wuv, unsigned n_batch,        \
        unsigned n_head, unsigned v, unsigned nsplit

#define FOLD_BASE(NAME)                                                                           \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(FOLD_ARGS) {                  \
        __shared__ FoldSmem sm;                                                                   \
        d_mla_merge_fold<512, 128>(out, opart, mlpart, wuv, n_batch, n_head, v, nsplit,           \
                                   blockIdx.x, gridDim.x, sm.f);                                  \
    }

#define FOLD_TB(NAME, TB)                                                                         \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void NAME(FOLD_ARGS) {                  \
        __shared__ FoldSmem sm;                                                                   \
        d_mla_merge_fold_tb<512, 128, TB>(out, opart, mlpart, wuv, n_batch, n_head, v, nsplit,    \
                                          blockIdx.x, gridDim.x, sm.f);                           \
    }

FOLD_BASE(k_fold_v128_tb1)
FOLD_TB(k_fold_v128_tb2, 2)
FOLD_TB(k_fold_v128_tb4, 4)
FOLD_TB(k_fold_v128_tb8, 8)

extern "C" __global__ void k_cache_flush(const unsigned* in, unsigned* out, size_t n) {
    unsigned v = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        v ^= in[i];
    if (v) atomicXor(out, v);
}
