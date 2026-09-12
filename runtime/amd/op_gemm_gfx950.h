/* op_gemm_gfx950.h — the CDNA4 (MI350X / MI355X, gfx950) arm of the GEMM/GEMV family.
 *
 * WHAT LIVES HERE: every GEMM/GEMV knob whose DEFAULT differs between the two CDNA generations,
 * every arch POLICY macro (a fixed per-arch fact the shared bodies key on), the code paths that
 * exist on only one arch, and the REFUSALS for knobs that have no meaning here. The bodies are
 * shared and live in op_gemm_common.h, which this file includes once its defaults are set.
 *
 * Include op_gemm.h, not this file: it selects the arm by PLOW_CDNA4 (amd_arch.h). Note that the
 * HOST pass of a hipcc executable build has PLOW_CDNA4 == 0 and therefore parses the gfx942 arm;
 * no device code runs there, so that is harmless, and it is why the gfx950 guard below is
 * device-pass only.
 *
 * TO RUN AN EXPERIMENT ON THIS ARCH ONLY: add the knob's `#ifndef` default here (it beats the
 * fallback in op_gemm_common.h because this file is included first), gate the body on it in
 * op_gemm_common.h, add its PLOW_GEOM_MARK to geom_contract.h so `asm_audit.py --contract` can
 * prove a `-D` reached the object, and add the knob to the OTHER arch file's `#error` list if it
 * has no meaning there. The other arch's objects then cannot move: they never see the default
 * and refuse the flag.
 */
#ifndef PLOW_OP_GEMM_GFX950_H
#define PLOW_OP_GEMM_GFX950_H
#include "amd_common.h"
#if !PLOW_CDNA4 && defined(__HIP_DEVICE_COMPILE__)
#error "op_gemm_gfx950.h is the CDNA4 arm but this is not a gfx950 device compile; include op_gemm.h, which selects"
#endif
#define PLOW_OP_GEMM_ARCH 950

/* ---- knobs with NO MEANING on CDNA4. Refused rather than silently redefined (the recipe compiles -w). */
#if defined(GV_UNROLL_M4) || defined(GV_UNROLL_M8)
#error "GV_UNROLL_M4 / GV_UNROLL_M8 are gfx942 knobs: CDNA4 has the packed bf16 dot, its M buckets never reached the AGPRs, and it keeps the flat GV_UNROLL (op_gemm_gfx942.h has the measured table)"
#endif
#if defined(PLOW_GV_UN_BIG) || defined(PLOW_GV_UN_LEGACY)
#error "PLOW_GV_UN_BIG / PLOW_GV_UN_LEGACY probe the K-divisor fp8 unroll rule, which gfx950 does not run (PLOW_GV_UN_FP8_KDIV 0): dead code here"
#endif

/* ---- arch POLICY. Fixed facts about this silicon, not tunables: bare defines with no `#ifndef`,
 * so a -D for one would be a redefinition the header wins (and the recipe compiles -w). Refused. */
#if defined(PLOW_GM_DIRECT_STAGE) || defined(PLOW_GM_FP8_PACK2) || defined(PLOW_GV_UN_FP8_KDIV) || \
    defined(GM8_FIX8)
#error "PLOW_GM_DIRECT_STAGE / PLOW_GM_FP8_PACK2 / PLOW_GV_UN_FP8_KDIV / GM8_FIX8 are arch policy set by op_gemm_gfx950.h, not -D knobs"
#endif
/* 16-byte global_load_lds is real here (amd_arch.h), so the plain bf16 d_gemm_t rungs stage A/B
 * straight into LDS and GM_DBUF=2 gives the DMA an idle buffer to stream into a cluster ahead. */
#define PLOW_GM_DIRECT_STAGE 1
/* The hardware packed OCP e4m3 encoder. */
#define PLOW_GM_FP8_PACK2(a, b) __builtin_amdgcn_cvt_pk_fp8_f32((a), (b), 0u, false)
/* gfx950: constant, so the switches in gv_un_fp8 fold and the object keeps its single pre-fix
 * UN instantiation per op. The K-divisor rule was measured on gfx942 only, and the GF=8
 * lesson in op_gemm_common.h records object growth alone regressing gfx950 decode 32% — CDNA3
 * only until someone with a gfx950 measures the rule there. */
#define PLOW_GV_UN_FP8_KDIV 0
/* The matrix core reads OCP e4m3 directly: nothing to mask. */
#define GM8_FIX8(p) ((void)0)
/* CDNA4 keeps the flat GV_UNROLL at every M bucket (see the refusal above). */
#define GV_UNROLL_M4 GV_UNROLL
#define GV_UNROLL_M8 GV_UNROLL

/* ---- arch-DEFAULTED knobs. `#ifndef`-guarded, so -D wins; PLOW_GEOM_MARK'd in geom_contract.h. */
/* THE DEFAULT RUNG: 256x256x64, double-buffered = 144 KiB of the 160 KiB LDS (op_gemm_common.h, TILE).
 * 256 stays the default because it does not regress Gemma and wins the low-load / single-request
 * latency case for every model; a Qwen prefill object is built with -DGM_BM=192 (the sweep in
 * op_gemm_common.h). Keep in sync with GFX950_TILES in crates/plowc/src/bin/gemma4.rs. */
#ifndef GM_BM
#define GM_BM 256
#endif
/* 160 KiB holds two stage buffers at the shipped tile; the ping-pong needs them. */
#ifndef GM_DBUF
#define GM_DBUF 2
#endif
/* The SMALL rung's BK: the rung gfx950 was tuned at. CDNA3's BK=128 re-cut (op_gemm_gfx942.h) runs
 * this ladder single-buffered; doubled here 64x128 at BK=128 is 104,448 B -- a different regime. */
#ifndef GM_SM_BK
#define GM_SM_BK 64
#endif
/* Both measured on gfx942 only (op_gemm_gfx942.h); gfx950 stays byte-identical to its qualified
 * objects until someone with the hardware re-runs its goldens. */
#ifndef GV_USCALAR
#define GV_USCALAR 0
#endif
#ifndef GV_RS_WIDE
#define GV_RS_WIDE 0
#endif

#include "op_gemm_common.h"

/* ---- gfx950-ONLY bodies (need the shared templates above). */
/* Internal gfx950 specialization of GEMM_WIDE. A 128x384 tile consumes 144 KiB of LDS, so it
 * cannot exist in the gfx942 object. It has no opcode of its own: the u8 opcode space is full,
 * and exec_gemm_wide selects it only when this tile maps exactly one workgroup to every CU. */
__device__ void d_gemm_c8(bf16* C, const bf16* A, const bf16* B, unsigned M, unsigned N,
                          unsigned K, unsigned slice, unsigned nblk, bf16* lds) {
    d_gemm_t<GM_C8_BM, GM_C8_BN, GM_C8_BK, GM_WM, GM_WN, false>(C, A, B, nullptr, nullptr, M, N, K,
                                                                slice, nblk, lds);
}

#endif /* PLOW_OP_GEMM_GFX950_H */
