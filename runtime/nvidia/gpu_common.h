/* gpu_common.h — shared helpers for the CUDA/HIP kernel adapters.
 *
 * The adapters live behind the same `kernel_fn` entry as the CPU kernels. On
 * GPU, kctx.slots/tensors hold *device* pointers and kctx.stream is a
 * cudaStream_t / hipStream_t. Naive (variant 0x00) kernels are single-thread
 * references that must match the CPU oracle; performant variants (0x01+) are
 * extracted from fast.cu / ThunderKittens / DeepGEMM / LiquidGEMM / HipKittens.
 */
#ifndef PLOW_GPU_COMMON_H
#define PLOW_GPU_COMMON_H

#include "kernel.h"
#include "packet.h"

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define GPU_STREAM hipStream_t
#else
  #include <cuda_runtime.h>
  #define GPU_STREAM cudaStream_t
#endif

#endif /* PLOW_GPU_COMMON_H */
