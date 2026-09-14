/* op_attention_gfx942.h — the CDNA3 (MI300X, gfx942) arm of the FlashAttention family.
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
#ifndef PLOW_OP_ATTENTION_GFX942_H
#define PLOW_OP_ATTENTION_GFX942_H
#include "amd_common.h"
#if PLOW_CDNA4
#error "op_attention_gfx942.h is the CDNA3 arm but this is a gfx950 compile; include op_attention.h, which selects"
#endif
#define PLOW_OP_ATTENTION_ARCH 942

/* ---- knobs with NO MEANING on CDNA3. */
/* ds_read_b64_tr_b16 is a gfx950 instruction; the scalar 8-read PV transpose is the only path here. */
#ifndef PLOW_MLA_PF_TR16
#define PLOW_MLA_PF_TR16 0
#endif
#if PLOW_MLA_PF_TR16
#error "PLOW_MLA_PF_TR16 requires gfx950"
#endif

/* ---- arch-DEFAULTED knobs. `#ifndef`-guarded, so -D wins; PLOW_GEOM_MARK'd in geom_contract.h. */
/* v_rcp_f32 for the softmax reciprocal (op_attention_common.h, SOFTMAX RECIPROCAL): 1 ULP into a
 * bf16 store, 135/99/48 IEEE divides gone from the decode/prefill/flash objects. */
#ifndef FA_FAST_RCP
#define FA_FAST_RCP 1
#endif
/* 4-banked merge accumulators (op_attention_common.h, [MERGE-UNROLL4]); reassociates the f32 sums. */
#ifndef FA_MERGE_UNROLL4
#define FA_MERGE_UNROLL4 1
#endif
/* Lanes per K row in the flash-decode score phase: 8, from the gfx942 memory-system probe
 * (decode_bw_probe.hip, 304 CU, 16 GB; [K-PHASE-KL8] in d_flash_decode). */
#ifndef FA_DEC_KL
#define FA_DEC_KL 8
#endif
/* Partial unroll of the MLA latent dot; the full unroll put the standalone decode at 256 VGPR with
 * 1836 B/lane of scratch. */
#ifndef FA_MLA_KDU
#define FA_MLA_KDU 4
#endif
/* Waves per query M-tile in the tiled MLA prefill: every wave shares ONE M-tile. This is an LDS
 * decision -- see the measured table above FA_MLA_PF_WPM in op_attention_common.h: WPM=PLOW_WAVES
 * with KSPLIT=2 is the ONLY eight-wave layout that fits 64 KiB, and it keeps occupancy 2. */
#ifndef PLOW_MLA_PF_WPM
#define PLOW_MLA_PF_WPM PLOW_WAVES
#endif
/* One on CDNA4, which has the 160 KiB to stage the whole latent. Two on CDNA3 -- see the
 * measured table above `PLOW_MLA_PF_WPM`: it is the only split that fits 64 KiB at either
 * wave count, and the re-staging it costs is L2-resident against a ~1000 FLOP/byte body. */
#ifndef PLOW_MLA_PF_KSPLIT
#define PLOW_MLA_PF_KSPLIT 2
#endif
/* Split softmax across the M-tile's column groups (op_attention_common.h, PLOW_MLA_PF_SMX): the
 * redundancy it removes is 8x at WPM=8. Bit-identical by construction. */
#ifndef PLOW_MLA_PF_SMX
#define PLOW_MLA_PF_SMX 1
#endif
/* The V2 MLA prefill's deferred-frame online softmax (op_attention_common.h, FA_MLA_PF2_DEFER).
 * CDNA3 ONLY BY DEFAULT, the same policy FA_FAST_RCP and FA_MERGE_UNROLL4 state above: this
 * is measured and content-gated on gfx942 (docs/amd/glm53-longctx-and-throughput.md) and
 * there is no gfx950 in this machine to re-run its goldens on, so that target keeps the
 * running-max form until someone with the hardware measures it. Force either way with the
 * FA_MLA_PF2_DEFER build axis in scripts/build_gfx942.sh. */
#ifndef FA_MLA_PF2_DEFER
#define FA_MLA_PF2_DEFER 1
#endif

#include "op_attention_common.h"

#endif /* PLOW_OP_ATTENTION_GFX942_H */
