/* Exact Gemma-4 12B batch-1 BF16 gate/up+GeGLU decode rung.
 *
 * The generic body uses UN=6. K=3840 is exactly eight 512-element chunks, so the generic loop
 * executes two groups and drains four dead slots. The native arm fixes N/K and uses UN=8, which
 * also lets the compiler remove the runtime loop/tail arithmetic. Both call the production
 * gemv_glu_rows body verbatim and preserve its accumulation order.
 */
#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_gemm.h"

static constexpr unsigned G12_N = 15360;
static constexpr unsigned G12_K = 3840;

template <int UN, bool Exact>
__device__ __forceinline__ void run_g12_glu(bf16* C, const bf16* x, const bf16* Wg,
                                            const bf16* Wu, unsigned nrep, unsigned N,
                                            unsigned K, unsigned act) {
    __shared__ bf16 lds[GM_LDS_HALVES];
    const unsigned k = Exact ? G12_K : K;
    const unsigned n = Exact ? G12_N : N;
    for (unsigned i = threadIdx.x; i < k; i += PLOW_THREADS) lds[i] = x[i];
    __syncthreads();
    const size_t slab = (size_t)n * k;
    for (unsigned r = 0; r < nrep; r++)
        gemv_glu_rows<1, UN>(C, Wg + (size_t)r * slab, Wu + (size_t)r * slab, 1, n, k, act,
                             blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void gemma12_gemv_glu_generic_un6(bf16* C, const bf16* x, const bf16* Wg, const bf16* Wu,
                                  unsigned nrep, unsigned N, unsigned K, unsigned act) {
    run_g12_glu<6, false>(C, x, Wg, Wu, nrep, N, K, act);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void gemma12_gemv_glu_runtime_un8(bf16* C, const bf16* x, const bf16* Wg, const bf16* Wu,
                                  unsigned nrep, unsigned N, unsigned K, unsigned act) {
    run_g12_glu<8, false>(C, x, Wg, Wu, nrep, N, K, act);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void gemma12_gemv_glu_native_un8(bf16* C, const bf16* x, const bf16* Wg, const bf16* Wu,
                                 unsigned nrep, unsigned N, unsigned K, unsigned act) {
    run_g12_glu<8, true>(C, x, Wg, Wu, nrep, N, K, act);
}
