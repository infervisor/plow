#pragma once
/* geom_contract.h — the compiled VALUE of every geometry/schedule knob, in the object.
 *
 * THE BUG THIS EXISTS FOR. `build_gfx942.sh` documents GM_SM_BK and GM_MD_* as reachable
 * through its `GM_AX` raw-`-D` escape hatch. They were not: a bare `#define` in a header the
 * command line has already defined is a REDEFINITION, the header wins (last definition), and
 * the recipe compiles with `-w`, so the warning is invisible. Every A/B ever run through
 * GM_AX against those five macros measured an UNCHANGED OBJECT -- including the published
 * "GM_SM_BK=128 is a flat null" row, which is worth -15.2% TTFT at 128 tokens now that the
 * macros are `#ifndef`-guarded. The `#ifndef` guards fix the five that were found by hand.
 * This file is the general defence: the object states what actually compiled, and
 * scripts/asm_audit.py --contract asserts it against the `-D` set the build passed.
 *
 * NAME UNEXPANDED, VALUE FROM THE ELF. The marker is `plow_geom_<MACRO>` initialised to the
 * macro -- not `plow_geom_<macro>_<value>` in the style of `plow_gemv_mm_cap_4`. That idiom
 * needs the value to be a single pasteable token, and a third of these knobs are expressions
 * (`FA_FAST_RCP` is `(!PLOW_CDNA4)`, `GM_CLBAR` is `(GM_DBUF == 2)`). The audit reads the
 * initialiser out of `.data` instead, which is as static as the name is. The loader never
 * needs these -- only the audit does -- so the reason `plow_gemv_mm_cap_N` encodes its value
 * in the NAME (plowrt reads `.symtab` before the object is on a device, and a value would
 * cost a device round-trip on the load path) does not apply here.
 *
 * DERIVED, never restated: `PLOW_GEOM_MARK(GM_SM_BK)` names the symbol from the token and
 * initialises it from the same token in one preprocessor pass, so the advertised value cannot
 * disagree with the compiled one.
 *
 * THE LIST IS CHECKED, NOT TRUSTED. asm_audit.py --contract scans amd_arch.h, op_moe.h and the
 * op_attention / op_gemm families (the selector, the _common body and both _gfx942 / _gfx950 arch
 * files) for `#ifndef (GM_|FA_|GV_|MPF_)*` and fails when one of them has no
 * PLOW_GEOM_MARK line here. A knob added without a marker is therefore a build failure and
 * not a silent hole -- which is the only way this guard does not rot.
 *
 * `#ifdef`-guarded one by one because several of these are defined inside an `#if` and are
 * genuinely absent from some configurations; an absent macro must not be a compile error, and
 * the audit only ever asserts markers for macros the build actually passed.
 *
 * COST: one 4-byte `.data` word per knob, ~350 B, no code. Included from interp.hip after
 * every op_*.h so each macro has already resolved to the value that compiled. */

/* ONE level, deliberately: macro-expansion of an argument is decided PER OCCURRENCE, so `m`
 * next to ## stays the macro's NAME while the `(m)` in the initialiser expands to its value.
 * A second level (PLOW_GEOM_MARK -> PLOW_GEOM_MARK_) expands the argument on the way in and
 * pastes `plow_geom_` onto the VALUE -- "pasting formed 'plow_geom_(', an invalid
 * preprocessing token" for every expression-valued knob. */
#define PLOW_GEOM_MARK(m) extern "C" __device__ unsigned plow_geom_##m = (unsigned)(m);

/* The GM_/FA_/GV_/MPF_ knobs `#ifndef`-guarded in amd_arch.h, op_moe.h and the op_attention* /
 * op_gemm* files (arch-defaulted knobs are guarded in op_*_gfx942.h and op_*_gfx950.h, shared
 * ones in op_*_common.h). asm_audit.py --contract re-derives this list from those headers and
 * fails on a knob that is missing here. */
#ifdef FA_ABL
PLOW_GEOM_MARK(FA_ABL)
#endif
#ifdef FA_BKV_D128
PLOW_GEOM_MARK(FA_BKV_D128)
#endif
#ifdef FA_DBUF
PLOW_GEOM_MARK(FA_DBUF)
#endif
#ifdef FA_DC
PLOW_GEOM_MARK(FA_DC)
#endif
#ifdef FA_DEC_ILV
PLOW_GEOM_MARK(FA_DEC_ILV)
#endif
#ifdef FA_DEC_KL
PLOW_GEOM_MARK(FA_DEC_KL)
#endif
#ifdef FA_LDS_DMA
PLOW_GEOM_MARK(FA_LDS_DMA)
#endif
#ifdef FA_DEC_KUNROLL
PLOW_GEOM_MARK(FA_DEC_KUNROLL)
#endif
#ifdef FA_DEC_LIVE
PLOW_GEOM_MARK(FA_DEC_LIVE)
#endif
#ifdef FA_DEC_VPIPE
PLOW_GEOM_MARK(FA_DEC_VPIPE)
#endif
#ifdef FA_DEC_V_UNROLL
PLOW_GEOM_MARK(FA_DEC_V_UNROLL)
#endif
#ifdef FA_DEC_WPEU
PLOW_GEOM_MARK(FA_DEC_WPEU)
#endif
#ifdef FA_FASTMASK
PLOW_GEOM_MARK(FA_FASTMASK)
#endif
#ifdef FA_FAST_RCP
PLOW_GEOM_MARK(FA_FAST_RCP)
#endif
#ifdef FA_HEAD_MAJOR
PLOW_GEOM_MARK(FA_HEAD_MAJOR)
#endif
#ifdef FA_LAZY_RESCALE
PLOW_GEOM_MARK(FA_LAZY_RESCALE)
#endif
#ifdef FA_MERGE_UNROLL4
PLOW_GEOM_MARK(FA_MERGE_UNROLL4)
#endif
#ifdef FA_MLA_KDU
PLOW_GEOM_MARK(FA_MLA_KDU)
#endif
#ifdef FA_MLA_KUNROLL
PLOW_GEOM_MARK(FA_MLA_KUNROLL)
#endif
#ifdef FA_MLA_PF2_BKV
PLOW_GEOM_MARK(FA_MLA_PF2_BKV)
#endif
#ifdef FA_MLA_PF2_DEFER
PLOW_GEOM_MARK(FA_MLA_PF2_DEFER)
#endif
#ifdef FA_MLA_PF2_FASTBF
PLOW_GEOM_MARK(FA_MLA_PF2_FASTBF)
#endif
#ifdef FA_MLA_PF2_FRAME
PLOW_GEOM_MARK(FA_MLA_PF2_FRAME)
#endif
#ifdef FA_USE_EXP2
PLOW_GEOM_MARK(FA_USE_EXP2)
#endif
#ifdef GM_BK
PLOW_GEOM_MARK(GM_BK)
#endif
#ifdef GM_BLK_BK
PLOW_GEOM_MARK(GM_BLK_BK)
#endif
#ifdef GM_BLK_BM
PLOW_GEOM_MARK(GM_BLK_BM)
#endif
#ifdef GM_BLK_BN
PLOW_GEOM_MARK(GM_BLK_BN)
#endif
#ifdef GM_BM
PLOW_GEOM_MARK(GM_BM)
#endif
#ifdef GM_BN
PLOW_GEOM_MARK(GM_BN)
#endif
#ifdef GM_CLBAR
PLOW_GEOM_MARK(GM_CLBAR)
#endif
#ifdef GM_CLFENCE
PLOW_GEOM_MARK(GM_CLFENCE)
#endif
#ifdef GM_DBUF
PLOW_GEOM_MARK(GM_DBUF)
#endif
#ifdef GM_HACK_NOFETCH
PLOW_GEOM_MARK(GM_HACK_NOFETCH)
#endif
#ifdef GM_MD_BK
PLOW_GEOM_MARK(GM_MD_BK)
#endif
#ifdef GM_MD_BM
PLOW_GEOM_MARK(GM_MD_BM)
#endif
#ifdef GM_MD_BN
PLOW_GEOM_MARK(GM_MD_BN)
#endif
#ifdef GM_PGR2
PLOW_GEOM_MARK(GM_PGR2)
#endif
#ifdef GM_PLR
PLOW_GEOM_MARK(GM_PLR)
#endif
#ifdef GM_PLR8
PLOW_GEOM_MARK(GM_PLR8)
#endif
#ifdef GM_PRIO
PLOW_GEOM_MARK(GM_PRIO)
#endif
#ifdef GM_SLICE
PLOW_GEOM_MARK(GM_SLICE)
#endif
#ifdef GM_SM_BK
PLOW_GEOM_MARK(GM_SM_BK)
#endif
#ifdef GM_SM_BM
PLOW_GEOM_MARK(GM_SM_BM)
#endif
#ifdef GM_SM_BN
PLOW_GEOM_MARK(GM_SM_BN)
#endif
#ifdef GM_SWZ
PLOW_GEOM_MARK(GM_SWZ)
#endif
#ifdef GM_WGM
PLOW_GEOM_MARK(GM_WGM)
#endif
#ifdef GV_BATCH
PLOW_GEOM_MARK(GV_BATCH)
#endif
#ifdef GV_BLOCKED
PLOW_GEOM_MARK(GV_BLOCKED)
#endif
#ifdef GV_DMA
PLOW_GEOM_MARK(GV_DMA)
#endif
#ifdef GV_HACK_CHEAPDOT
PLOW_GEOM_MARK(GV_HACK_CHEAPDOT)
#endif
#ifdef GV_HACK_NOSUM
PLOW_GEOM_MARK(GV_HACK_NOSUM)
#endif
#ifdef GV_KS
PLOW_GEOM_MARK(GV_KS)
#endif
#ifdef GV_KS_MAXRPW
PLOW_GEOM_MARK(GV_KS_MAXRPW)
#endif
#ifdef GV_KS_UN
PLOW_GEOM_MARK(GV_KS_UN)
#endif
#ifdef GV_MFMA
PLOW_GEOM_MARK(GV_MFMA)
#endif
#ifdef GV_MFMA4
PLOW_GEOM_MARK(GV_MFMA4)
#endif
#if defined(GV_MFMA4_MAXK) && GV_MFMA4
PLOW_GEOM_MARK(GV_MFMA4_MAXK)
#endif
#ifdef GV_MFMA4_UN_M2
PLOW_GEOM_MARK(GV_MFMA4_UN_M2)
#endif
#ifdef GV_MFMA4_UN_M4
PLOW_GEOM_MARK(GV_MFMA4_UN_M4)
#endif
#ifdef GV_MFMA4_UN_M8
PLOW_GEOM_MARK(GV_MFMA4_UN_M8)
#endif
#ifdef GV_MFMA4_YT_M2
PLOW_GEOM_MARK(GV_MFMA4_YT_M2)
#endif
#ifdef GV_MFMA4_YT_M4
PLOW_GEOM_MARK(GV_MFMA4_YT_M4)
#endif
#ifdef GV_MFMA4_YT_M8
PLOW_GEOM_MARK(GV_MFMA4_YT_M8)
#endif
#ifdef GV_NT
PLOW_GEOM_MARK(GV_NT)
#endif
#ifdef GV_RING
PLOW_GEOM_MARK(GV_RING)
#endif
#ifdef GV_RS_MAXNCH
PLOW_GEOM_MARK(GV_RS_MAXNCH)
#endif
#ifdef GV_RS_R
PLOW_GEOM_MARK(GV_RS_R)
#endif
#ifdef GV_RS_UN
PLOW_GEOM_MARK(GV_RS_UN)
#endif
#ifdef GV_RS_WIDE
PLOW_GEOM_MARK(GV_RS_WIDE)
#endif
#ifdef GV_RS_WIDE_R
PLOW_GEOM_MARK(GV_RS_WIDE_R)
#endif
#ifdef GV_RS_WIDE_UN
PLOW_GEOM_MARK(GV_RS_WIDE_UN)
#endif
#ifdef GV_UNROLL
PLOW_GEOM_MARK(GV_UNROLL)
#endif
#ifdef GV_UNROLL_FP8
PLOW_GEOM_MARK(GV_UNROLL_FP8)
#endif
#ifdef GV_UNROLL_GLU
PLOW_GEOM_MARK(GV_UNROLL_GLU)
#endif
#ifdef GV_UNROLL_GLU_FP8
PLOW_GEOM_MARK(GV_UNROLL_GLU_FP8)
#endif
#ifdef GV_UNROLL_GLU_MXFP4
PLOW_GEOM_MARK(GV_UNROLL_GLU_MXFP4)
#endif
#ifdef GV_UNROLL_M4
PLOW_GEOM_MARK(GV_UNROLL_M4)
#endif
#ifdef GV_UNROLL_M8
PLOW_GEOM_MARK(GV_UNROLL_M8)
#endif
#ifdef GV_UN_K8192
PLOW_GEOM_MARK(GV_UN_K8192)
#endif
#ifdef GV_URSRC
PLOW_GEOM_MARK(GV_URSRC)
#endif
#ifdef GV_USCALAR
PLOW_GEOM_MARK(GV_USCALAR)
#endif
#ifdef MPF_BK
PLOW_GEOM_MARK(MPF_BK)
#endif
#ifdef MPF_BM
PLOW_GEOM_MARK(MPF_BM)
#endif
#ifdef MPF_DBUF
PLOW_GEOM_MARK(MPF_DBUF)
#endif

/* The numeric PLOW_ knobs build_gfx942.sh passes as raw `-D`. PLOW_GEMV_MM is marked here as
 * well as by `plow_gemv_mm_cap_N` -- that symbol is the LOADER's (plowrt refuses a packet
 * wider than the compiled bucket) and this one is the audit's, and the two agreeing is
 * itself worth something. */
#ifdef PLOW_WG_WAVES
PLOW_GEOM_MARK(PLOW_WG_WAVES)
#endif
#ifdef PLOW_WPE
PLOW_GEOM_MARK(PLOW_WPE)
#endif
#ifdef PLOW_GEMV_MM
PLOW_GEOM_MARK(PLOW_GEMV_MM)
#endif
