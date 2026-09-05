#include <cstdint>
#include "dev_isa.h"
#define PLOW_NV_PREFILL 0
// Keep fixed-width attention helpers at 256 threads; only GEMV bodies use 16 warps.
#include "op_attention.cuh"
#undef PLOW_NV_THREADS
#undef PLOW_NV_WARPS
#define PLOW_NV_THREADS 512
#define PLOW_NV_WARPS 16u
#include "op_gemm.cuh"
#include "op_moe.cuh"
#define PLOW_NV_HOPPER 1
#define PLOW_NV_GEMV512_ROLE 1
#define PLOW_NV_EMBED_SMEM 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_FORCE_MINBLK 1
#define interp_sm120 interp_sm90a
#define plow_sm120_grid plow_sm90a_grid
#define plow_sm120_smem plow_sm90a_smem
#define plow_sm120_sched plow_sm90a_sched
#define plow_sm120_skeleton plow_sm90a_skeleton
#define plow_sm120_launch plow_sm90a_launch

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
#error "GEMV512 role requires sm_90a"
#endif

#include "interp_sm120.cu"
