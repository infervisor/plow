/* op_gemm_gfx942.h — the CDNA3 (MI300X, gfx942) arm of the GEMM/GEMV family.
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
#ifndef PLOW_OP_GEMM_GFX942_H
#define PLOW_OP_GEMM_GFX942_H
#include "amd_common.h"
#if PLOW_CDNA4
#error "op_gemm_gfx942.h is the CDNA3 arm but this is a gfx950 compile; include op_gemm.h, which selects"
#endif
#define PLOW_OP_GEMM_ARCH 942

/* ---- arch POLICY. Fixed facts about this silicon, not tunables: bare defines with no `#ifndef`,
 * so a -D for one would be a redefinition the header wins (and the recipe compiles -w). Refused. */
#if defined(PLOW_GM_DIRECT_STAGE) || defined(PLOW_GM_FP8_PACK2) || defined(PLOW_GV_UN_FP8_KDIV) || \
    defined(GM8_FIX8)
#error "PLOW_GM_DIRECT_STAGE / PLOW_GM_FP8_PACK2 / PLOW_GV_UN_FP8_KDIV / GM8_FIX8 are arch policy set by op_gemm_gfx942.h, not -D knobs"
#endif
/* No 16-byte global_load_lds on CDNA3 (amd_arch.h), so d_gemm_t's DMA-direct stage is never
 * selected here: the register-staging path, with GM_PGR2 as its two-deep prefetch, is the ladder. */
#define PLOW_GM_DIRECT_STAGE 0
/* Encoding x/2 as FNUZ produces the same finite byte as encoding x as OCP because the formats'
 * exponent biases differ by one. Exhaustively compared on gfx942 over every finite FP32 bit
 * pattern in d_quant_fp8's [-448,448] domain: zero mismatches. Balanced M1024 timing wins every
 * 3840/4096/5376/8192 width by 1.21x--1.33x, so gfx942 defaults it on. */
#ifndef GM_NATIVE_OCP_QUANT
#define GM_NATIVE_OCP_QUANT 1
#endif
#if GM_NATIVE_OCP_QUANT
#define PLOW_GM_FP8_PACK2(a, b)                                                             \
    (plow_fp8_mask_neg0(__builtin_amdgcn_cvt_pk_fp8_f32((a) * 0.5f, (b) * 0.5f, 0u, false)) \
     & 0xffffu)
#else
#define PLOW_GM_FP8_PACK2(a, b) \
    ((unsigned)plow_f32_to_fp8_ocp(a) | ((unsigned)plow_f32_to_fp8_ocp(b) << 8))
#endif
/* gv_un_fp8 picks the fp8 GEMV unroll by the K-divisor rule (op_gemm_common.h). Measured HERE:
 * K=3840/4096 -18..-27% standalone; gfx950 keeps the constant until someone measures the rule there. */
#define PLOW_GV_UN_FP8_KDIV 1
/* CDNA3 reads e4m3FNUZ, where OCP's 0x80 (-0) is NaN. Production operands are already
 * canonical: d_quant_fp8 never emits 0x80, and HsaUploadRing scrubs every F8_E4M3 checkpoint
 * payload once while uploading it. Repeating the SWAR scrub for every operand K-tile only burns
 * VALU and registers in the hot GEMM loop. Keep it in C5: deleting it there crosses a compiler
 * allocation cliff (+2 scratch operations), while Gemma's M1024 projection rung uses E2. */
#define GM8_FIX8(p)                                                                          \
    do {                                                                                      \
        if constexpr (BM == 192 && BN == 256) {                                               \
            unsigned w_[2];                                                                   \
            __builtin_memcpy(w_, (p), 8);                                                     \
            w_[0] = plow_fp8_mask_neg0(w_[0]);                                                \
            w_[1] = plow_fp8_mask_neg0(w_[1]);                                                \
            __builtin_memcpy((p), w_, 8);                                                     \
        }                                                                                     \
    } while (0)

/* ---- arch-DEFAULTED knobs. `#ifndef`-guarded, so -D wins; PLOW_GEOM_MARK'd in geom_contract.h. */
/* The K64 FP8 MFMA does not need the priority boost used by the BF16 ping-pong. Across the
 * Gemma-4 12B/31B M1024 ladder, priority-off wins all 2K/4K/8K production metrics and leaves
 * BF16 instruction selection unchanged. */
#ifndef GM_PRIO8
#define GM_PRIO8 0
#endif
/* Overridable at compile time so plowc can bucket the tile per shape without an ISA
 * change, and so a Qwen prefill object can be built with -DGM_BM=192 (see the sweep above).
 * Keep in sync with GFX950_TILES in crates/plowc/src/bin/gemma4.rs. */
/* The DEFAULT rung -- what `d_gemm` (and therefore GemmGlu) instantiates. Its default is
 * arch-conditional for the same reason the ladder below is: 256x256 at BK=64 is a 147,456 B
 * double-buffered stage, 2.25x over CDNA3's 64 KiB, so gfx942 shipped it single-buffered and
 * lost the ping-pong on the op that costs the most (GEMM_GLU is 24% of a Gemma-4 prefill
 * chunk's body). 128x256 at BK=32 fits DOUBLE-buffered in 61,440 B. BN stays 256 because the
 * fused-GLU epilogue's SN==2 assert pins it there at 8 waves.
 *
 * These were previously forced from the outside (`-DGM_BM=... -DGM_BN=...` in
 * scripts/build_gfx942.sh). Defaulting them here means a CDNA3 build that forgets the flags
 * still gets a tile that FITS, rather than one that fails the LDS limit at link time. Still
 * overridable, which is what plowc's per-shape bucketing and every A/B sweep rely on. */
#ifndef GM_BM
#define GM_BM 192
#endif
/* DEFAULTED PER ARCH, not passed in by the build. gfx950's 160 KiB holds two buffers at the
 * shipped 256x256/BK=64 tile; CDNA3's 64 KiB does not hold two at ANY tile the ladder uses
 * (192x256/BK=64 double-buffered is 129,024 B), so it is single-stage. This used to be
 * `-DGM_DBUF=1` in scripts/build_gfx942.sh, which meant a CDNA3 build that forgot the flag
 * silently asked for a 129 KiB arena on a 64 KiB part. See the ladder note above GM_SM_BM for
 * the measurement that says buying the second buffer back by halving BK does not pay. */
#ifndef GM_DBUF
#define GM_DBUF 1
#endif
/* The SMALL rung's BK. 128 is the CDNA3 re-cut measured in op_gemm_common.h above GM_SM_BM: 1.10x to
 * 1.56x on every Gemma-4 31B prefill shape, arithmetic-preserving, single-buffered. */
#ifndef GM_SM_BK
#define GM_SM_BK 128
#endif
/* Scalar (SGPR) GEMV row descriptors: -0.55% ms/token, ranges disjoint, token-identical on the
 * Gemma-4 12B asset (op_gemm_common.h, above PLOW_GV_RSRC). CDNA3 only because it changes codegen
 * on a shipped, validated gfx950 target that has not been re-run. */
#ifndef GV_USCALAR
#define GV_USCALAR 1
#endif
/* The MM==4 R-split GEMV arm on staged shapes (op_gemm_common.h, above gemv_rows_r): -0.96 ms of a
 * 27.29 ms plain-GEMV total at T=4, bit-exact. Measured here only. */
#ifndef GV_RS_WIDE
#define GV_RS_WIDE 1
#endif
/* THE UNROLL IS A REGISTER BUDGET, AND IT IS SHARED WITH THE M ROWS.
 *
 * GV_UNROLL is how many independent weight loads a wave puts in flight, and the comment above
 * picks 11 from `nchunk` for the M=1 decode. But each of the MM activation rows also holds live
 * vectors, so at MM>1 the same unroll asks for MM times the registers -- and on CDNA3 it asks
 * for more still, because there is no v_dot2c_f32_bf16 and the software dot has to materialise
 * f32 operands (amd_arch.h). Past the budget the allocator starts using AGPRs as spill relief,
 * which is exactly what happened: the M=8 and M=16 buckets landed at 256 VGPR + 256 AGPR and
 * every accumulate paid a v_accvgpr round trip.
 *
 * MEASURED on MI300X, 4-wave object, Gemma-4 31B shapes (ms, lower is better):
 *
 * SWEPT AT THE SHIPPING CONFIG -- 8 waves, single-buffered stage (ms, min of runs):
 *
 *            gate/up 21504x5376            down 5376x21504
 *   UN     M=1    M=4    M=8   M=16      M=1    M=4    M=8   M=16
 *   11    0.059  0.305  0.767  2.420    0.044  0.145  0.337  0.932
 *    6    0.068  0.098  0.206  1.804    0.044  0.047  0.092  0.706
 *    4    0.070  0.101  0.172  1.566    0.047  0.051  0.079  0.585
 *    3    0.070  0.102  0.160  0.265    0.046  0.053  0.075  0.119
 *    2    0.074  0.104  0.165  0.275    0.051  0.057  0.070  0.114
 *
 * The optimum FALLS with MM: 11 at M=1, 6 at M=4, 3 at M>=8. Taking it is 4.8x at M=8 and 9.1x
 * at M=16 over the flat 11 on gate/up. The earlier 4-wave sweep put M=16 at 2; at 8 waves the
 * allocator caps itself at 256 instead of reaching for AGPRs, and 3 wins -- which is why this
 * table was re-measured rather than carried over.
 *
 * CDNA4 keeps the flat 11 -- it has the packed dot, its buckets never reached the AGPRs, and
 * these numbers were not measured there. */
#ifndef GV_UNROLL_M4
#define GV_UNROLL_M4 6
#endif
#ifndef GV_UNROLL_M8
#define GV_UNROLL_M8 3
#endif

#include "op_gemm_common.h"

#endif /* PLOW_OP_GEMM_GFX942_H */
