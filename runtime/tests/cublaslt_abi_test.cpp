#include <cublasLt.h>
#include <cstddef>
static_assert(sizeof(cublasLtMatmulAlgo_t) == 64);
static_assert(alignof(cublasLtMatmulAlgo_t) == 8);
static_assert(sizeof(cublasLtMatmulHeuristicResult_t) == 96);
static_assert(offsetof(cublasLtMatmulHeuristicResult_t, workspaceSize) == 64);
static_assert(offsetof(cublasLtMatmulHeuristicResult_t, state) == 72);
static_assert(offsetof(cublasLtMatmulHeuristicResult_t, wavesCount) == 76);
static_assert(offsetof(cublasLtMatmulHeuristicResult_t, reserved) == 80);
static_assert(sizeof(cublasStatus_t) == 4);
static_assert(CUDA_R_16BF == 14 && CUDA_R_32F == 0 && CUBLAS_COMPUTE_32F == 68);
static_assert(CUBLASLT_MATMUL_DESC_TRANSA == 3 && CUBLAS_OP_T == 1 && CUBLAS_OP_N == 0);
static_assert(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES == 1);
int main() {}
