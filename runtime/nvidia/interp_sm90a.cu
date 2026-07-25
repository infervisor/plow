/* interp_sm90a.cu — Hopper (H100/H200) persistent packet interpreter.
 *
 * The packet ABI, counter protocol, and correctness-sensitive op bodies are
 * shared with the validated warp32 NVIDIA implementation. Hopper gets its own
 * translation unit and public symbols so a native sm_90a cubin can never be
 * mistaken for the sm_120a image by the driver or asset loader.
 *
 * Decode deliberately retains the vectorized FFMA/FP8 GEMV path: at M=1 it is
 * weight-bandwidth bound, and WGMMA would add staging/reduction overhead while
 * reading the same bytes. Prefill uses Hopper's cp.async.bulk+mbarrier path for
 * the hd512 attention arm. Tiled GEMM already uses bf16/fp8 tensor cores and is
 * kept in the separate `_pf` object so its register footprint cannot reduce
 * decode occupancy. A WGMMA prefill replacement must remain a separate segment
 * object until it passes numeric tests on real Hopper hardware.
 */
#define PLOW_NV_HOPPER 1

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
#error "interp_sm90a.cu must be compiled for sm_90a"
#endif

/* Hopper TMA is compile-time selectable for compiler-only A/B analysis.
 * fp8-KV objects stay on cp.async staging: the raw-e4m3 tile path and the
 * fp8mma arm are cp.async-only (op_attention.cuh stages bf16 under TMA). */
#ifndef PLOW_NV_FA_TMA
#ifdef PLOW_FP8_KV
#define PLOW_NV_FA_TMA 0
#else
#define PLOW_NV_FA_TMA 1
#endif
#endif

/* Architecture-specific public ABI. These aliases are expanded by the
 * two-level PLOW_SYM paste in interp_sm120.cu. */
#define interp_sm120 interp_sm90a
#define plow_sm120_grid plow_sm90a_grid
#define plow_sm120_smem plow_sm90a_smem
#define plow_sm120_sched plow_sm90a_sched
#define plow_sm120_skeleton plow_sm90a_skeleton
#define plow_sm120_launch plow_sm90a_launch

#include "interp_sm120.cu"
