#include "op_attention.h"

/* The outlined entry preserves the interpreter's d_flash_prefill call boundary. Flatten is the
 * only intentional difference in the native entry; both instantiate the production body. */
#define FLASH_ENTRY(D)                                                                         \
  extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)                        \
  void gemma_flash_prefill_##D##_outlined(                                                     \
      float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, unsigned n_q,  \
      unsigned n_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window,    \
      float scale, unsigned nsplit) {                                                          \
    __shared__ bf16 lds[FA_LDS_HALVES(D)];                                                     \
    d_flash_prefill<D>(Opart, mlpart, nullptr, Q, K, V, n_q, n_kv, n_head, n_kv_head, q_pos0,  \
                       window, scale, n_kv, PLOW_KV_MASK_NONE, nsplit, blockIdx.x, gridDim.x,   \
                       lds);                                                                   \
  }                                                                                            \
  extern "C" __global__ __attribute__((flatten))                                              \
  __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void gemma_flash_prefill_##D##_native(       \
      float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, unsigned n_q,  \
      unsigned n_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window,    \
      float scale, unsigned nsplit) {                                                          \
    __shared__ bf16 lds[FA_LDS_HALVES(D)];                                                     \
    d_flash_prefill<D>(Opart, mlpart, nullptr, Q, K, V, n_q, n_kv, n_head, n_kv_head, q_pos0, \
                       window, scale, n_kv, PLOW_KV_MASK_NONE, nsplit, blockIdx.x, gridDim.x,   \
                       lds);                                                                   \
  }

FLASH_ENTRY(256)
FLASH_ENTRY(512)

#define FLASH_FUSED_ENTRY(D)                                                               \
  extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)                   \
  void gemma_flash_prefill_##D##_fused_outlined(                                           \
      bf16* O, const bf16* Q, const bf16* K, const bf16* V, unsigned n_q, unsigned n_kv,  \
      unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window, float scale, \
      unsigned kv_stride, unsigned kv_mask) {                                              \
    __shared__ bf16 lds[FA_LDS_HALVES(D)];                                                 \
    d_flash_prefill<D>(nullptr, nullptr, O, Q, K, V, n_q, n_kv, n_head, n_kv_head, q_pos0, \
                       window, scale, kv_stride, kv_mask, 1u, blockIdx.x, gridDim.x, lds); \
  }                                                                                        \
  extern "C" __global__ __attribute__((flatten))                                         \
  __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)                                         \
  void gemma_flash_prefill_##D##_fused_native(                                             \
      bf16* O, const bf16* Q, const bf16* K, const bf16* V, unsigned n_q, unsigned n_kv,  \
      unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window, float scale, \
      unsigned kv_stride, unsigned kv_mask) {                                              \
    __shared__ bf16 lds[FA_LDS_HALVES(D)];                                                 \
    d_flash_prefill<D>(nullptr, nullptr, O, Q, K, V, n_q, n_kv, n_head, n_kv_head, q_pos0, \
                       window, scale, kv_stride, kv_mask, 1u, blockIdx.x, gridDim.x, lds); \
  }

FLASH_FUSED_ENTRY(256)
FLASH_FUSED_ENTRY(512)

#define FLASH_MERGE_ENTRY(D)                                                                \
  extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)                     \
  void gemma_flash_merge_##D(bf16* O, const float* Opart, const float* mlpart,               \
                             unsigned n_batch, unsigned n_head, unsigned nsplit) {           \
    d_flash_merge<D>(O, Opart, mlpart, n_batch, n_head, nsplit, blockIdx.x, gridDim.x);      \
  }

FLASH_MERGE_ENTRY(256)
FLASH_MERGE_ENTRY(512)
