/* op_attention_gfx950.h — the CDNA4 (MI350X / MI355X, gfx950) arm of the FlashAttention family.
 *
 * WHAT LIVES HERE: every attention knob whose DEFAULT differs between the two CDNA generations,
 * the code paths that exist on only one arch, and the REFUSALS for knobs that have no meaning
 * here. The bodies are shared and live in op_attention_common.h, which this file includes once
 * its defaults are set.
 *
 * Include op_attention.h, not this file: it selects the arm by PLOW_CDNA4 (amd_arch.h). The HOST
 * pass of a hipcc executable build has PLOW_CDNA4 == 0 and parses the gfx942 arm; no device code
 * runs there, so that is harmless, and it is why the gfx950 guard below is device-pass only.
 *
 * TO RUN AN EXPERIMENT ON THIS ARCH ONLY: add the knob's `#ifndef` default here (it beats the
 * fallback in op_attention_common.h because this file is included first), gate the body on it
 * in op_attention_common.h, add its PLOW_GEOM_MARK to geom_contract.h so `asm_audit.py
 * --contract` can prove a `-D` reached the object, and add the knob to the OTHER arch file's
 * `#error` list if it has no meaning there.
 *
 * THE POLICY THESE DEFAULTS FOLLOW: gfx950 is a shipped, validated target with no unit in the
 * gfx942 lab, so every last-bit-moving change measured on gfx942 (FA_FAST_RCP, FA_MERGE_UNROLL4,
 * FA_MLA_PF2_DEFER) defaults ON here and OFF there until someone with the hardware re-runs its
 * goldens. The LDS-driven ones (WPM, KSPLIT, SMX) are the only eight-wave layout that fits 64 KiB.
 */
#ifndef PLOW_OP_ATTENTION_GFX950_H
#define PLOW_OP_ATTENTION_GFX950_H
#include "amd_common.h"
#if !PLOW_CDNA4 && defined(__HIP_DEVICE_COMPILE__)
#error "op_attention_gfx950.h is the CDNA4 arm but this is not a gfx950 device compile; include op_attention.h, which selects"
#endif
#define PLOW_OP_ATTENTION_ARCH 950

/* ---- gfx950-ONLY knobs and helpers. */
/* gfx950-only PV transpose loads. The existing scalar path reconstructs one bf16x8
 * fragment with eight LDS reads and four packs; ds_read_b64_tr_b16 produces four
 * transposed bf16 values directly, so two instructions form the same fragment. */
#ifndef PLOW_MLA_PF_TR16
#define PLOW_MLA_PF_TR16 0
#endif
#if PLOW_MLA_PF_TR16
extern "C" __device__ unsigned plow_mla_pf_tr16_arm_1 = 1;
typedef bf16_t mla_pf_bf16x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ mla_pf_bf16x4 mla_pf_ds_read_tr16(const bf16* p) {
    auto* lp = (mla_pf_bf16x4 __attribute__((address_space(3)))*)(void*)p;
    return __builtin_amdgcn_ds_read_tr16_b64_v4bf16(lp);
}
#endif

/* ---- arch-DEFAULTED knobs. `#ifndef`-guarded, so -D wins; PLOW_GEOM_MARK'd in geom_contract.h. */
/* Bit-identical to the qualified gfx950 objects: the IEEE divide, the serial merge and the
 * running-max V2 softmax. Each is measured and argued in op_attention_gfx942.h / _common.h; flip
 * here once the goldens have been re-run on this silicon. */
#ifndef FA_FAST_RCP
#define FA_FAST_RCP 0
#endif
#ifndef FA_MERGE_UNROLL4
#define FA_MERGE_UNROLL4 0
#endif
#ifndef FA_MLA_PF2_DEFER
#define FA_MLA_PF2_DEFER 0
#endif
/* The historical row-per-lane K-phase map. The 8 that gfx942 picks is a gfx942 memory-system
 * measurement; CDNA4 has a different L1, LDS budget and wave-per-SIMD budget, so the same probe
 * has to be re-run here before the map changes under a shipped, tuned object. */
#ifndef FA_DEC_KL
#define FA_DEC_KL 1
#endif
#ifndef FA_MLA_KDU
#define FA_MLA_KDU 64 /* >= any trip count: FULL unroll, the historical CDNA4 lowering */
#endif
/* Four waves per query M-tile and the whole latent staged in one pass: 160 KiB affords the
 * Qsm[64][DK+DR+PAD] + full Ksm + P arena that 64 KiB cannot hold (op_attention_gfx942.h). */
#ifndef PLOW_MLA_PF_WPM
#define PLOW_MLA_PF_WPM 4
#endif
#ifndef PLOW_MLA_PF_KSPLIT
#define PLOW_MLA_PF_KSPLIT 1
#endif
/* The redundant per-wave softmax (4x at WPM=4). The split form is unmeasured here. */
#ifndef PLOW_MLA_PF_SMX
#define PLOW_MLA_PF_SMX 0
#endif

#include "op_attention_common.h"

#endif /* PLOW_OP_ATTENTION_GFX950_H */
