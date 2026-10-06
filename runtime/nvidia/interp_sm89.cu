/* interp_sm89.cu — Ada (L4 / RTX 4090) persistent packet interpreter.
 *
 * The non-Hopper warp32 interpreter (interp_sm120.cu) built for sm_89: mma.sync
 * tensor cores and cp.async only, no wgmma/TMA/clusters. Its own TU and public
 * symbols keep an sm_89 cubin from being mistaken for the sm_120a image.
 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 890
#error "interp_sm89.cu must be compiled for sm_89"
#endif

/* Architecture-specific public ABI. These aliases are expanded by the
 * two-level PLOW_SYM paste in interp_sm120.cu. */
#define interp_sm120 interp_sm89
#define plow_sm120_grid plow_sm89_grid
#define plow_sm120_smem plow_sm89_smem
#define plow_sm120_sched plow_sm89_sched
#define plow_sm120_skeleton plow_sm89_skeleton
#define plow_sm120_launch plow_sm89_launch
#define plow_sm120_light plow_sm89_light
#define plow_sm120_light_norm_quant plow_sm89_light_norm_quant
#define plow_sm120_light_attn plow_sm89_light_attn
#define plow_sm120_light_tail plow_sm89_light_tail
#define plow_sm120_light_capmax plow_sm89_light_capmax
#define plow_sm120_light_attn_s plow_sm89_light_attn_s
#define plow_sm120_light_fp8_flash256 plow_sm89_light_fp8_flash256
#define plow_sm120_light_head plow_sm89_light_head
#define plow_sm120_light_flash plow_sm89_light_flash
#define plow_sm120_light_pf plow_sm89_light_pf
#define plow_sm120_rider_flash256 plow_sm89_rider_flash256
#define plow_sm120_rider_flash512 plow_sm89_rider_flash512
#define plow_sm120_rider_merge plow_sm89_rider_merge

#include "interp_sm120.cu"

extern "C" __device__ unsigned plow_norm_weight_offset = PLOW_NV_GEMMA3;
