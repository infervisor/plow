#!/usr/bin/env bash
# Build the gfx942 (MI300X / CDNA3) persistent-interpreter code objects.
#
# The gfx950 twin of this is scripts/build_gfx950.sh; this one exists separately
# rather than as a `$ARCH` knob on that script because CDNA3 diverges on two axes
# that are not a substitution:
#
#   LDS      64 KiB/workgroup, not 160. The gfx950 GEMM stage arena
#            (GM_BM=256 GM_BN=256 GM_BK=64 double-buffered, 147,456 B) is 2.25x
#            over budget here. The CDNA3 tile and GM_DBUF=1 are now DEFAULTS in
#            op_gemm.h keyed on PLOW_CDNA4, not flags this script passes, so a
#            build that forgets them still gets an arena that fits.
#   SYMBOL   the loader builds the kernel name from the LIVE HSA AGENT NAME
#            (exec/amd.rs symbol_name), so an object built here must export
#            plow_interp*_gfx942. -DPLOW_ARCH_SUFFIX is what does that; without
#            it interp.hip defaults to gfx950 and the object loads and then
#            fails symbol resolution.
#
# Output dir defaults to build-amd/hsaco/gfx942 -- ARCH-QUALIFIED, because the
# .elf stems carry no arch and a shared directory lets a gfx950 object be handed
# to an MI300X. Point plowrt at it with PLOW_HSACO=<dir>.
#
# Measured on ROCm 7.14 / clang-23, all rows within the cliff:
#   interp_decode          VGPR 253  AGPR   0  LDS 64,520  spill  0   occ 2
#   interp_prefill         VGPR 256  AGPR   0  LDS 64,520  spill  6   occ 2
#   interp_flash           VGPR 512  AGPR 256  LDS 58,368  spill 98   occ 1
#   interp_prefill_mla_moe VGPR 256  AGPR   0  LDS 64,520  spill  6   occ 2
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-$REPO/build-amd/hsaco/gfx942}}"
ARCH=gfx942
[ -n "${IN_NIX_SHELL:-}" ] || { echo "FAIL: run this script through nix develop" >&2; exit 2; }
: "${PLOW_HIPCC:?nix develop did not set PLOW_HIPCC}"
: "${PLOW_BUNDLER:?nix develop did not set PLOW_BUNDLER}"
: "${PLOW_READELF:?nix develop did not set PLOW_READELF}"
[ "${PLOW_TOOLCHAIN_LABEL:-}" = "rocm-7.14.0-nix" ] || {
  echo "FAIL: expected ROCm 7.14.0 from the flake, got ${PLOW_TOOLCHAIN_LABEL:-unset}" >&2; exit 2; }
HIPCC="$PLOW_HIPCC"
BUN="$PLOW_BUNDLER"
READELF="$PLOW_READELF"
require_nix_tool() {
  local name="$1" path="$2" target
  target="$(readlink -f "$path")"
  case "$target" in
    /nix/store/*) ;;
    *) echo "FAIL: $name must resolve into /nix/store, got ${target:-missing}" >&2; exit 2 ;;
  esac
  [ -x "$target" ] || { echo "FAIL: $name not executable at $target" >&2; exit 2; }
}
require_nix_tool hipcc "$HIPCC"
require_nix_tool clang-offload-bundler "$BUN"
require_nix_tool llvm-readelf "$READELF"
HIP_VERSION="$("$HIPCC" --version)"
case "$HIP_VERSION" in
  *"HIP version: 7.14."*) ;;
  *) echo "FAIL: expected HIP 7.14 from the flake" >&2; exit 2 ;;
esac
INC="-I$R/amd -I$R/common"
JOBS="${JOBS:-8}"
mkdir -p "$OUT"; cd "$OUT"

# THE CDNA3 TILE IS NOW A DEFAULT IN op_gemm.h, NOT A FLAG HERE.
#
# This used to force -DGM_DBUF=1 -DGM_BM=192 -DGM_BN=256; op_gemm.h now defaults
# exactly that on !PLOW_CDNA4 (GM_DBUF=1, 192x256, the single-buffered stage that
# fits 64 KiB). The BK=32 double-buffer re-cut was TRIED and REJECTED (+11.7% --
# see op_gemm.h's GM_DBUF note), so the ping-pong stays OFF on CDNA3. Forcing the
# tile from outside would silently override the header's defaults; the flags stay
# only as an A/B escape hatch.
CDNA3_TILE="-DPLOW_WG_WAVES=8${GM_BM:+ -DGM_BM=$GM_BM}${GM_BN:+ -DGM_BN=$GM_BN}${GM_BK:+ -DGM_BK=$GM_BK}${GM_DBUF:+ -DGM_DBUF=$GM_DBUF}"
# The 4-wave flash object: GM_WN=2 there, so the wave-grid assert is satisfied at
# BN=128 and the smaller tile leaves room for the 58,368 B flash arena.
CDNA3_TILE_4W="-DGM_BM=64 -DGM_BN=128${GM_DBUF:+ -DGM_DBUF=$GM_DBUF}"

# GEMMA-4 MoE (26B-A4B), ops 61-77 + 81/82. Folded into the STANDARD rows here
# rather than built as the separate interp_{prefill,decode}_gmoe.elf pair that
# scripts/build_gfx950.sh emits, because plowrt has no `_gmoe` object name:
# exec/amd.rs `object_name()` composes stem + variant + prefill-arm + sched and
# there is no Gemma-MoE arm in `PrefillArm`, so it opens plain
# `interp_prefill_gq.elf` and then REFUSES on the missing marker symbol
# (`check_moe_gemma_arms`). The separately-named objects are unreachable by that
# path -- which is why a 26B-A4B serve dies at load with
#   "this packet dispatches MoeRouterGemmaPf (op 73), but interp_prefill_gq.elf
#    was compiled without PLOW_MOE_GEMMA_PF".
# Folding them in is only affordable because both halves are free at the cliff
# (measured, and re-checked by the table this script prints).
AX_GMOE="-DPLOW_MOE_GEMMA=1 -DPLOW_MOE_GEMMA_PF=1"
AX_PREFILL="-DPLOW_BUCKET_DECODE=0 $CDNA3_TILE $AX_GMOE"
# PLOW_GEMV_MM is next_pow2(PLOW_DECODE_BATCH) CLAMPED TO 16, not the batch itself. The GEMV
# ladder instantiates MM in {1,2,4,8,16} and one instantiation with a runtime M serves every
# M <= MM, so the bucket is a CEILING. Passing the raw batch through was a bug in this script:
# PLOW_DECODE_BATCH=32 handed -DPLOW_GEMV_MM=32 to hipcc and every decode row failed to build.
GVMM=1
while [ "$GVMM" -lt "${PLOW_DECODE_BATCH:-1}" ] && [ "$GVMM" -lt 16 ]; do GVMM=$((GVMM * 2)); done
AX_DECODE="-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=$GVMM $CDNA3_TILE $AX_GMOE"
# OPT-IN (PLOW_GEMV_WALK=1): the §6g-WALK row-block outer loop — the object serves any M in
# ceil(M/MM) passes of the compiled bucket, and the LDS staging bound becomes min(MM,M)*K.
# REQUIRED for a PLOW_DECODE_BATCH>16 ladder (PLOW_GEMV_MAXM caps the bucket at 16; without
# the walk, rows 16.. would never be written — plowrt refuses the pair on the missing
# `plow_gemv_walk_1` marker rather than serving stale rows).
if [ "${PLOW_GEMV_WALK:-0}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_GEMV_WALK=1"
fi
# PLOW_OCC4=1 -- THE OCCUPANCY-4 DECODE PROFILE. MEASURED -10.4% on bf16 and -19.1% on fp8
# (Gemma-4-12B, ctx 4096, three interleaved pairs, token-identical):
#
#   bf16 20.748 -> 18.584 ms/token      fp8 19.198 -> 15.527      bf16 base -> fp8+occ4: -25.2%
#
# The decode object is normally pinned to 2 waves/SIMD by TWO megakernel unions, and BOTH have to
# move or neither does: LDS 64,520 B (one workgroup per CU) and VGPR 253. The four pieces:
#   GM 128x256x32   the smallest arena the GLU SN==2 assert allows, 30,720 B
#   fa[] gated      flash-PREFILL tiles, provably dead here (interp_decode_gq.elf carries ZERO
#                   d_flash_prefill symbols) -- done in interp.hip, bucket-conditional
#   PLOW_NO_MLA_DEC MLA latent decode, 42,064 B, which sets the union once fa[] is gone
#   PLOW_WPE=5      asks for 5 waves/EU so the allocator stops at 104 VGPR instead of spending
#                   the 253 that __launch_bounds__(512,2) merely PERMITS
# -> LDS 30,736 B and VGPR 104, i.e. 4 waves/SIMD on both axes.
#
# NOT THE DEFAULT, and the reason is PLOW_NO_MLA_DEC: plowrt has no MLA arm-check for decode the
# way it has `check_moe_gemma_arms`, so a GLM/DeepSeek/Kimi packet handed this object would
# silently skip its MLA ops rather than refuse. Shipping it on needs that check first -- the
# marker-symbol pattern in exec/amd.rs is the template.
if [ "${PLOW_OCC4:-0}" = 1 ]; then
  # -DPLOW_MOE_GEMMA only, NOT _PF: the grouped-MoE PREFILL tile is (64+256)*64 halves = 40,960 B
  # and its static_assert wants that much `raw`, which the 30,720 B arena cannot give. It is a
  # prefill arm and the decode bucket never dispatches op 73, so dropping it costs nothing here.
  AX_DECODE="-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=$GVMM -DPLOW_WG_WAVES=8 -DPLOW_MOE_GEMMA=1 \
             -DGM_BM=128 -DGM_BN=256 -DGM_BK=32 -DPLOW_NO_MLA_DEC=1 -DPLOW_WPE=5"
fi
# $AX_GMOE on the FLASH row too. The flash object runs only class-4 flash
# segments and has no use for op 73 -- but plowrt's `check_moe_gemma_arms` is a
# blanket check over EVERY object it loads, so a flash object without the marker
# symbol is REJECTED, and the rejection is an `info!` degrade ("no flash object
# -- flash segments run on the 8-wave interpreter"), not an error. See the note
# on AX_GMOE above for why that degrade is not benign.
AX_FLASH="-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 $CDNA3_TILE_4W $AX_GMOE"
# V2 MLA prefill arm (d_flash_mla_prefill_v2): the full-column-wave layout that needs this
# object's 512-register budget. Marker `plow_mla_pf_v2_arm_1`; the host routes FlashMlaPrefill
# segments here only under PLOW_MLA_PF_V2=1, so carrying the arm costs Gemma nothing.
AX_FLASH="$AX_FLASH -DPLOW_MLA_PF_V2_ARM=1"
# OPT-IN (PLOW_FA_LAZY=1): wave-voted skip of the online-softmax corr/rescale when the
# running max did not move (bit-identical; see op_attention.h FA_LAZY_RESCALE). FLASH
# OBJECT ONLY — the 8-wave prefill interpreter sits on the 256-reg cliff and even a
# no-op perturbation of the flash branch's allocator can drop it to 1 wave/SIMD; the
# 4-wave flash object has the 512-reg budget. Default OFF.
if [ "${PLOW_FA_LAZY:-0}" = 1 ]; then
  AX_FLASH="$AX_FLASH -DFA_LAZY_RESCALE=1"
fi
AX_FP8="-DPLOW_FP8=1"
AX_FP8KV="-DPLOW_FP8_KV=1"
AX_MLA="-DPLOW_MLA_PREFILL=1"
AX_MOE="-DPLOW_MOE_PREFILL=1"

# BATCHED DECODE (PLOW_DECODE_BATCH > 1): the GLM decode program emits its MoE/dense FFN with
# the grouped PREFILL family at T = rows (the decode MoE ops carry no token dimension), so the
# DECODE object must compile those case arms too — ops 83-87 are gated on PLOW_MOE_PREFILL,
# which historically only the interp_prefill_*_mla_moe rows carried. Without this the load
# refuses on the missing arm ("this packet dispatches MoeGroupGluPf, but interp_decode_gq.elf
# was built without it"). Gated on the batch so a B=1 build stays byte-identical; watch the
# cliff table below for the register cost the extra arms put on the decode megakernel.
if [ "${PLOW_DECODE_BATCH:-1}" -gt 1 ]; then
  AX_DECODE="$AX_DECODE $AX_MOE"
  # THE GROUPED TILE IS NO LONGER THE OCC4 BLOCKER (2026-08-10 bisect). BM=64/BK=32 is FIXED:
  # op_moe.h's MPF_SUBQ masked A-staging arm serves the sub-quantum tile (waves 0-3 stage the
  # full 8-half vector each; the scale-block-edge fp8 promotion and kk-derived preshuffle
  # address ride with it), and a BK=32 batched object at occ2 is BYTE-IDENTICAL in served
  # output to the proven BK=64 object (48-token solo trajectories, one serve each). What
  # HANGS the batched first dispatch is the OCC4 REGISTER RATION itself: with the identical
  # 30,720 B arena, GM 128x256x32, NO_MLA_DEC and BK=32, the serve passes at PLOW_WPE=3
  # (168 VGPR) and hangs at WPE=4 (128) and WPE=5 (104) — tile, LDS, NO_MLA_DEC and
  # PLOW_GATE_HIER each exonerated one at a time, and the B=1 OCC4 object serves fine on the
  # same binary. The BM=128 (SM=2) recut stays CLOSED for an unrelated reason: at SM=2 the
  # acc+accf promotion accumulators alone are 128 VGPRs — the whole WPE=5 budget.
  if [ "${PLOW_OCC4:-0}" = 1 ]; then
    echo "FAIL: PLOW_OCC4=1 with PLOW_DECODE_BATCH>1 — the WPE=5/4 register ration hangs the"
    echo "      batched program's first decode dispatch (2026-08-10 bisect; the BK32 grouped"
    echo "      tile is fixed and exonerated — occ2+MPF_BK=32 and the WPE=3 recut both serve)."
    echo "      Build without PLOW_OCC4, or take PLOW_DEC_SQUEEZE=1 (the validated WPE=3 recut)."
    exit 1
  fi
  if [ "${PLOW_DEC_SQUEEZE:-0}" = 1 ]; then
    # THE VALIDATED REGISTER-SQUEEZE RECUT (opt-in, 2026-08-10): the OCC4 profile's pieces at
    # the deepest ration that still serves — GM 128x256x32 (30,720 B arena), NO_MLA_DEC (fmla
    # aimed at raw, GF=4 fits, asserted in interp.hip), MPF_BK=32/DBUF=1 (the MPF_SUBQ tile),
    # PLOW_WPE=3 -> 168 VGPR, spill 20-26. Gate record, one serve each, same session: smoke
    # coherent; solo trajectory BYTE-IDENTICAL to the shipped BK=64 occ2 object over 48
    # tokens; needle-content PASS @3000; rung-1 TPOT 40.218 -> 35.291 ms p50 (-12.3%, in=8192
    # out=128, TTFT unmoved 2393->2391 — the correct negative control). WPE=4 (128 VGPR) and
    # WPE=5 (104) HANG the first batched dispatch — that cliff is the open OCC4 task.
    AX_DECODE="$AX_DECODE -DGM_BM=128 -DGM_BN=256 -DGM_BK=32 -DPLOW_NO_MLA_DEC=1 -DPLOW_WPE=3 \
               -DPLOW_MOE_GEMMA_PF=0 -DMPF_BK=32 -DMPF_DBUF=1"
  else
    # MPF_BK=32 A/B escape hatch (like GM_BM/GM_BK above): the arena-fitting tile at occ2,
    # validated byte-identical to BK=64 in served output. It drops the Gemma _PF arm exactly
    # as the OCC4 profile does — the Gemma grouped twin has no sub-quantum arm and its
    # static_assert refuses the pairing.
    AX_DECODE="$AX_DECODE${MPF_BK:+ -DMPF_BK=$MPF_BK -DPLOW_MOE_GEMMA_PF=0} -DMPF_DBUF=1"
  fi
fi
AX_GQ="-DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=${PLOW_GQ_BATCH:-1}"
# WEIGHT encoding: MXFP4 e2m1 + E8M0 (w4a16). This used to be gfx950-only because CDNA3 has no
# fp4 datatype and amd_arch.h poisoned the decode to NaN; it now decodes in software, exactly (the
# fp16-subnormal identity, verified against the ladder on this silicon). It gates the fp4
# PROJECTION ops -- GemvMxfp4 (91), GemvGluMxfp4 (92), GemmMxfp4 (93) and the four extra GEMM
# rungs, GemmGluMxfp4 (113), GemvQkvMxfp4 (114) -- and nothing else: the mxfp4 EXPERT walks that
# K3 decode actually runs (ops 45/46 with i6 = PLOW_MOE_ENC_MXFP4) are not behind it.
AX_MXFP4="-DPLOW_MXFP4=1"
# KIMI-K3. `GV_UNROLL=14` is on the K3 rows only and is measured, not derived: K3's dominant
# decode GEMV is K=7168, whose nchunk is exactly 14 (runtime/CMakeLists.txt records the sweep).
AX_K3="-DPLOW_K3=1 -DGV_UNROLL=14"
AX_MLA_K3="$AX_MLA -DPLOW_K3=1"
# THE A4W4 ROWS BUILD HERE TOO, as the SIMULATED arm. True A4W4 (fp4 on both operands through
# v_mfma_scale_f32_32x32x64_f8f6f4) has no CDNA3 analogue, but the ops do not ask for an
# instruction: without PLOW_HAS_MX_MMA, `d_moe_group_pf_a4w4` compiles as the CDNA3 body --
# fp4 dequantized to bf16 in staging (EXACT: <= 3 significant bits, power-of-two scale) and fed
# to the ordinary bf16 MFMA, same packet contract, same GLU bridge writing fu as MXFP4 + E8M0.
# Verified on this silicon by runtime/tests/moe_prefill_a4w4_cdna3_test.hip: the bridge output
# is quantized-value-IDENTICAL to an f64 host reference and DOWN agrees to 2.5e-8 rms. The arm
# costs the prefill object NOTHING (256 VGPR / occ 2 / 8 spill, byte-for-byte the same resource
# report as the object without it).
AX_A4W4="-DPLOW_MOE_PF_A4W4=1"
AX_K3_A4W4=""
# Value-identical CDNA3 DOWN metadata hoist: 6.155 -> 5.490 ms at the emitted TP8
# 4096-token/896-expert shape. Keep it on the K3 A4W4 rows so unrelated model objects do not move.
if [ "${PLOW_K3_A4W4_EPI:-1}" != 0 ]; then
  AX_K3_A4W4="$AX_K3_A4W4 -DPLOW_MOE_PF_EPI_SIB=1"
fi

# PER-XCD QUEUES + TWO-LEVEL GATE MAINTENANCE -- ON BY DEFAULT. MEASURED -16.0%, TOKEN-IDENTICAL.
#
# DEFAULT ON (opt out with PLOW_L2HIER=0). This is the shipped gfx942 decode path, not a tuning
# knob: L2-domain windowing drains EIGHT per-XCD queues concurrently instead of one global one,
# and the two-level gate is what that windowing buys. Measured on this box, together they are the
# largest win on gfx942 decode by a wide margin -- placement -1.5%, hierarchy -16.0% on top.
#
# SAFE WITH AN UNPLACED BLOB, verified rather than assumed: `hier_base` is only non-zero on an
# L2-placed blob and `nper` is zero otherwise, which interp.hip reads as "no hierarchy" and
# compiles the ordinary path. Objects built this way run an UNPLACED blob at 15.60/15.71 ms vs
# 15.67 for objects built without it -- i.e. identical. So turning this on cannot break existing
# assets; it only ACTIVATES when the blob is placed.
#
# TO GET THE WIN the blob must be placed too -- compile assets with PLOW_L2_PLACE=1 and run with
# PLOW_L2_PLACE_DISPATCH=1 (see the note below). An unplaced blob silently gets the old path.
#
# The single largest lever found on gfx942 decode, and it is not a kernel change. Every workgroup
# in a packet issues `buffer_wbl2` + `buffer_inv` at the gate; those are PER-L2, so each XCD does
# the same writeback and the same invalidate once per participating workgroup and they SERIALISE.
# At b=304 that is ~30 us per packet before a single weight byte moves -- measured directly by
# emptying the GEMV body (PLOW_GV_ABL=3), which removes every load, convert and dot and still
# leaves ~9 ms of a 10.54 ms GEMV union standing.
#
# Priced with the in-tree ceiling knobs, Gemma-4-12B fp8+occ4, ctx 4096, 48 steps:
#
#   shipped                                    15.32 ms/token
#   PLOW_GATE_NOINV     drop buffer_inv        14.76   -3.7%
#   PLOW_GATE_RELAXSIG  drop buffer_wbl2       12.75  -16.8%
#   NOINV + RELAXSIG    (bound on the total)   12.31  -19.6%
#   PLOW_GATE_HIER_CEIL unsound per-XCD leader 12.91  -15.8%
#   PLOW_GATE_HIER      THE SOUND FORM         12.66  -16.0%   <- this
#
# The sound form MATCHES its own unsound ceiling, so the two XCD-local rendezvous it adds cost
# nothing measurable. Token-identical to the same blob without it (last id 236761 both).
#
# REQUIRES AN L2-PLACED BLOB -- compile the model with PLOW_L2_PLACE=1 and run plowrt with
# PLOW_L2_PLACE_DISPATCH=1. Without placement `nper` is a run-time property, the emitter leaves it
# zero, and interp.hip reads that as "no hierarchy" and compiles the ordinary path. That is why
# this is opt-in rather than a default: the objects and the blob have to be built as a PAIR, and
# plowrt REFUSES a placed blob handed to an unplaced object (it does not silently mis-dispatch).
# Placement itself is worth a further -1.5% (15.31 -> 15.08 unplaced -> placed, no hierarchy).
# DECODE ROWS ONLY, and this is MEASURED, not cautious. Applying it to every row -- which the
# "objects that cannot support the hierarchy compile WITHOUT it" note in interp.hip appears to
# license -- builds all 28 objects cleanly and then HANGS: `amd-bench --ctx 4096` sat at 100% GPU
# for 680 s (vs ~150 s for a good run) and had to be killed. That note is about COMPILING, not
# about running: a flash-class object that silently drops the hierarchy is fine on its own, but
# the decode program's segments are then gated by two different protocols and the run deadlocks.
# The measurement above was taken with the flag on the DECODE object alone, so that is where it
# goes. Do not widen this without re-running the hang test.
if [ "${PLOW_L2HIER:-1}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_L2_PLACE_DISPATCH=1 -DPLOW_GATE_HIER=1"
fi

# OPT-IN (PLOW_GLM_GF8=1): compile the GF=8 MLA flash-decode arm so PLOW_GLM_GF=4-vs-8
# can be A/B'd ON THE SAME OBJECT (op_attention.h PLOW_GLM_GF8_ARM: comparing across
# objects confounds with the +32% I$-growth effect). NEVER ship an arm-present object's
# numbers as a default-config result — the arm's presence alone is the confound.
#
# NOT COMPOSABLE WITH PLOW_OCC4, and hipcc now says so instead of corrupting LDS: OCC4 passes
# -DPLOW_NO_MLA_DEC=1, which aims the MLA decode arena at the 30,720 B GEMM tile, and the GF=8
# layout is 42,048 B — 11,328 B past it. The static_assert in interp.hip next to the MPF one
# refuses the pair at compile time. A GF=8 A/B under OCC4 needs a deliberately widened arena.
if [ "${PLOW_GLM_GF8:-0}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_GLM_GF8_ARM=1"
fi

# OPT-IN (PLOW_GLM_FUSE_QNORM=1): THE Q-NORM FOLD ARM on op 22 `GemvQkv` (op_gemm.h
# PLOW_GLM_FUSE_QNORM). GLM-5.2 decode runs GemvQkv-A -> RmsNorm(q_a_layernorm) -> GemvQkv-G
# and the middle packet is ONE workgroup: the traced window between the two GEMVs is 12.2 us
# for a 4.6 us body, the largest packet-boundary window left on the decode chain
# (perf-data/plow-gfx942/glm52-decode-packet-folds.md section 7 prices it at ~-0.9 ms). This
# arm normalizes the staged copy of x in place -- d_gemv_t's `norm == 2` mechanism, bit-exact
# to the deleted packet -- and the blob must be emitted with PLOW_GLM_FUSE_QNORM=1 too. Both
# halves are needed: plowrt refuses a folded blob on an unarmed object via
# `plow_glm_fuse_qnorm_arm`, because an unarmed object would silently run the GEMV over an
# UNNORMED q_a row. DECODE ROWS ONLY (op 22 is decode-only; prefill picks a Gemm). Default
# OFF, so the shipped objects are byte-unchanged.
# ARM DEFAULT ON for gfx942 (opt out with PLOW_GLM_FUSE_QNORM=0), 2026-08-09: the arm is a
# runtime branch discriminated on the packet's t[7], so an armed object serves an unfolded
# blob byte-identically — defaulting it on just makes folded blobs loadable without a matched
# hand-built object. The EMIT side stays opt-in.
if [ "${PLOW_GLM_FUSE_QNORM:-1}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_GLM_FUSE_QNORM=1"
fi

# OPT-IN (PLOW_L2HIER_PF=1): the same pair on the PREFILL objects, for blobs whose
# prefill program is PLACED (GLM: uni-segment prefill, one object per program run, so
# the mixed-protocol deadlock above cannot arise -- that hang needed one placed
# program's segments split across objects with and without the define). On an
# UNPLACED blob the define is inert (GATE_HIER's runtime precondition is
# l2_domains != 0). Still: re-run the hang test (amd-bench --prompt at 2-4k completes
# in normal wall time) before trusting any build that widens this.
if [ "${PLOW_L2HIER_PF:-0}" = 1 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_L2_PLACE_DISPATCH=1 -DPLOW_GATE_HIER=1"
fi

# OPT-IN (PLOW_MLA_PF_QK1=1): MLA prefill computes QK^T + softmax on ONE wave per M-tile
# and shares P + the per-row corrections through LDS instead of every wave recomputing
# them (8x redundant on CDNA3, where the arena forces WPM = PLOW_WAVES). Byte-identical
# output by construction — same values, same multiplies (op_attention.h PLOW_MLA_PF_QK1).
# OPT-IN (PLOW_DSA_PF=1): the GATHERED arm of the V2 MLA prefill (DSA sparse prefill,
# runtime ops 117-119 + t7 on op 51). FLASH OBJECT ONLY, and opt-in because the megakernel's
# register allocation is the worst case over every inlined arm: instantiating the gathered
# body costs the flash object spill 98 -> 287 even for blobs that never emit a union table.
# A sparse blob loaded against an object built WITHOUT this reads no t7 and runs dense.
if [ "${PLOW_DSA_PF:-0}" = 1 ]; then
  AX_FLASH="$AX_FLASH -DPLOW_DSA_PF_ARM=1"
fi

if [ "${PLOW_MLA_PF_QK1:-0}" = 1 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MLA_PF_QK1=1"
fi

# PLOW_MLA_PF_SMX=0 opts OUT of the split-softmax MLA prefill (default ON for CDNA3 —
# op_attention.h PLOW_MLA_PF_SMX; bit-identical, kill switch for A/B only).
if [ "${PLOW_MLA_PF_SMX:-}" = 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MLA_PF_SMX=0"
fi

# CEILING INSTRUMENT ONLY (PLOW_MLA_PF_ABL=1..4): MLA-prefill ablation probes — one cost
# term deleted each (op_attention.h PLOW_MLA_PF_ABL). WRONG OUTPUT by construction, never a
# serve asset. Do not combine with PLOW_MLA_PF_QK1 (the ABL=2 arm changes what the QK1
# cgrp-0 guard binds to).
if [ "${PLOW_MLA_PF_ABL:-0}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MLA_PF_ABL=${PLOW_MLA_PF_ABL}"
fi

# CEILING INSTRUMENT ONLY (PLOW_XR_NOWAIT=1): prefill objects with BOTH two-shot rendezvous
# waits deleted (op_collective.h). The output is WRONG by construction — this prices what the
# collective's synchronization costs, never ships, and must not touch a serve asset.
if [ "${PLOW_XR_NOWAIT:-0}" = 1 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_XR_NOWAIT=1"
fi

# OPT-IN (PLOW_XR_MLP=1): PEER-BATCHED REDUCE in the cross-GPU collectives (op_collective.h).
# The reduce bodies walked the N peers one serialised round trip at a time (ISA: a pointer
# re-load + s_waitcnt vmcnt(0), then the remote load + another vmcnt(0), PER PEER PER ELEMENT);
# this hoists the eight peer bases and issues all eight remote loads before consuming any.
# BIT-IDENTICAL (same r=0..N-1 f32 sum, same element->thread map, same 2 B load width) and TP8
# only. DECODE and PREFILL objects both -- the one-shot XREDUCE is decode's MoE-seam collective
# and the two-shot's reduce-scatter is half of prefill's fabric bytes. Default OFF: unset, the
# objects are byte-identical to a build from before this axis existed.
# MEASURED NULL, slightly negative (+1.1%/+1.6% TTFT @4k/8k against a 0.7-1.3% control spread;
# 181.7 vs 289.4 GB/s in the reduce-scatter microbench). This fabric is limited by request
# concurrency ACROSS THREADS, not per-thread depth. Kept as the record; see op_collective.h and
# perf-data/plow-gfx942/glm52-collective-tuning-mi300x.md. DO NOT turn it on expecting a win.
if [ "${PLOW_XR_MLP:-0}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_XR_MLP=1"
  AX_PREFILL="$AX_PREFILL -DPLOW_XR_MLP=1"
fi

# OPT-IN (PLOW_XR_AGG=1): DEVICE-LOCAL AGGREGATION of the two-shot collective's `gate_ag`
# signal (op_collective.h PLOW_XR_AGG). As built, all nblk workgroups each issue nranks
# SYSTEM-scope returning RMWs on one 128 B line per peer -- 2432 remote atomics per rank per
# collective at nblk=304/tp=8, measured at 51.8 us against 8.2 us for a 1-signaller gate.
# This aggregates them on word 1 of the same counter line (PLOW_CTR_STRIDE is 32 words and
# only word 0 is used) and lets the closing workgroup issue nranks signals carrying nblk each
# -- so word 0 still lands on exactly nranks*nblk and plowrt's host audit is unchanged.
# BIT-IDENTICAL (no value is touched) and objects-only: no blob, no emitter, no arm marker;
# an object built without it is correct-just-slower. PREFILL ROWS ONLY -- `XReduceTwoShot`
# is emitted 156x per prefill program and ZERO times in the decode program, so this is TTFT
# work and exactly 0.0% of TPOT.
# HISTORY: default-on 2026-08-09, reverted same day (an XR_AGG-only build FAILED the
# 3000-token needle gate, '741' for '7413'), RE-ADOPTED 2026-08-10 after the ordering fix.
# The failing cut released with a FENCE and arrived with a relaxed AGENT-scope RMW — that
# orders nobody's stores for a remote observer, and agent-scope RMWs run cached in the
# arriving XCD's L2 on the very line the peers' signals update memory-side. The fixed form
# (op_collective.h): the arrival RMW itself is the release at SYSTEM scope, and the closing
# workgroup takes the SYSTEM acquire before speaking. Gate record (2026-08-10, r4b4 ladder
# asset): fixed arm PASSES needle 3000/8000 x2 each; a freshly rebuilt PRE-fix control ALSO
# passed 2/2 the same day, so the original failure was INTERMITTENT — the fix stands on the
# memory-model repair, the gate is its non-refutation. Opt out with PLOW_XR_AGG=0.
if [ "${PLOW_XR_AGG:-1}" = 1 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_XR_AGG=1"
fi

# OPT-IN (PLOW_MOE_PF_SCHED=1): sched_group_barrier pipeline shaping in the grouped MoE
# prefill k-loop (op_moe.h). Instruction ORDER only — bit-identical output; the A/B judges
# whether the aiter-style load/MFMA interleave beats LLVM's default schedule.
if [ "${PLOW_MOE_PF_SCHED:-0}" = 1 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MOE_PF_SCHED=1"
fi

# PLOW_MOE_PF_PIPE=0 forces the shipped single-stage grouped-prefill k-loop (the aiter-shape
# two-tile register pipeline is the CDNA3 DEFAULT in op_moe.h). Only "0" is meaningful here —
# it is the A/B control against the default-on pipeline.
if [ "${PLOW_MOE_PF_PIPE:-}" = 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MOE_PF_PIPE=0"
fi

# OPT-IN (PLOW_MOE_PF_GH=1|2): GATHER HIDING in the grouped MoE prefill A-gather (op_moe.h
# PLOW_MOE_PF_GH). 1 hoists the K-INVARIANT `row_token[rowbase+r]` index out of the k-loop --
# every k-tile currently re-loads the SAME dword and stalls on `s_waitcnt vmcnt(0)` before its
# A row can issue; 2 additionally software-pipelines that one load a full OUTPUT TILE ahead.
# Value-identical (same indices, same bytes, same LDS cells) -- this is an A/B on latency
# exposure, not on arithmetic. Default OFF, so the shipped object is byte-unchanged.
if [ "${PLOW_MOE_PF_GH:-0}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MOE_PF_GH=${PLOW_MOE_PF_GH}"
fi

# OPT-IN (PLOW_MOE_PF_EPI=1|2): DOWN-EPILOGUE ROW-METADATA HOIST (op_moe.h PLOW_MOE_PF_EPI).
# The shipped DOWN epilogue issues 128 flat_load_dword at max-outstanding 1, each followed by
# a full `s_waitcnt vmcnt(0) lgkmcnt(0)`, to re-read the k- AND n-INVARIANT
# row_partidx/row_gate pair once per OUTPUT ELEMENT. 1 loads the 64-row block one row per
# lane at the tile head (latency covered by the k-loop) and bpermutes it back in the epilogue;
# 2 issues the same two loads just before the epilogue instead, so nothing is live across the
# k-loop. Same addresses, same dwords, same arithmetic -- BYTE-IDENTICAL output, an A/B on
# round-trip serialization only. Default OFF, so the shipped object is byte-unchanged.
# DEFAULT ON for gfx942 (opt out with PLOW_MOE_PF_EPI=0), 2026-08-09. The output is
# BYTE-IDENTICAL -- same addresses, same dwords, same arithmetic -- so this is an A/B on
# round-trip serialization and nothing else, and it is the difference between the MoE grouped
# pair sitting 2.76x and 2.25x off aiter at the measured M=2048 shape
# (glm52-current-cost-decomposition.md sec 1.6). It was already passed explicitly by the
# canonical GLM object recipe; carrying it as a default removes the chance of building the
# "shipped" objects without it. `=0` restores the pre-arm object exactly.
if [ "${PLOW_MOE_PF_EPI:-1}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MOE_PF_EPI=${PLOW_MOE_PF_EPI:-1}"
fi

# OPT-IN (PLOW_MOE_PF_EPI_SIB=1): THE SAME HOIST AT THE TWO SIBLING SITES (op_moe.h
# PLOW_MOE_PF_EPI_SIB) -- `d_moe_group_pf_a4w4` (native CDNA4 and simulated CDNA3) and
# `d_moe_group_gemma_pf_t` (the Gemma-4 MoE twin, ops 75/76 and 81/82; its w8a8 arm carries a
# THIRD k/n-invariant per-row load, `ascale`, which this takes with the other two). Same
# addresses, same dwords, same arithmetic -- BYTE-IDENTICAL output. A SEPARATE flag from
# PLOW_MOE_PF_EPI so the GLM canonical recipe is unperturbed by a change to kernels GLM never
# dispatches. Rides AX_PREFILL and AX_DECODE alike, because the Gemma MoE prefill bodies are
# folded into every gfx942 row by AX_GMOE. Default OFF globally; K3 A4W4 rows enable it above.
if [ "${PLOW_MOE_PF_EPI_SIB:-0}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MOE_PF_EPI_SIB=1"
  AX_DECODE="$AX_DECODE -DPLOW_MOE_PF_EPI_SIB=1"
  AX_FLASH="$AX_FLASH -DPLOW_MOE_PF_EPI_SIB=1"
fi

# Simulated-A4W4 CDNA3 staging experiments. Defaults preserve BK64 and its MFMA priority bracket.
AX_K3_A4W4_TUNE=""
if [ -n "${PLOW_MOE_PF_A4W4_C3_BK:-}" ]; then
  case "$PLOW_MOE_PF_A4W4_C3_BK" in 32|64) ;; *) echo "FAIL: PLOW_MOE_PF_A4W4_C3_BK must be 32 or 64" >&2; exit 2;; esac
  AX_K3_A4W4_TUNE="$AX_K3_A4W4_TUNE -DPLOW_MOE_PF_A4W4_C3_BK=$PLOW_MOE_PF_A4W4_C3_BK"
fi
if [ "${PLOW_MOE_PF_A4W4_PRIO:-1}" = 0 ]; then
  AX_K3_A4W4_TUNE="$AX_K3_A4W4_TUNE -DPLOW_MOE_PF_A4W4_PRIO=0"
fi

# FALSIFICATION ARM (PLOW_F2BF_SELECT=1): the REFUTED branchless f2bf. Default 0 = the shipped branched form.
# MEASURED AND REFUTED: the branchless form is -5.0% static instructions on the prefill
# megakernel and +4.5/+5.3/+7.0% SERVED TTFT at 4k/8k/16k (4 interleaved arms, 2 rounds). It is
# also not output-identical in situ despite being value-identical over all 2^32 float bit
# patterns -- GSM8K 0.960 vs 0.970, reproducible per arm -- because a function this widely
# inlined perturbs surrounding codegen and fp contraction. Kept so the falsification is
# reproducible rather than a claim. Rides all three object groups: every one stores bf16.
if [ "${PLOW_F2BF_SELECT:-0}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_F2BF_SELECT=1"
  AX_DECODE="$AX_DECODE -DPLOW_F2BF_SELECT=1"
  AX_FLASH="$AX_FLASH -DPLOW_F2BF_SELECT=1"
fi

# OPT-IN (PLOW_MOE_PF_ATOMIC=1): FUSE the grouped MoE prefill's ops 86 -> 87 (op_moe.h
# PLOW_MOE_PF_ATOMIC). The DOWN epilogue stops scattering part[T*k, H] and atomically adds into
# a [T, H] f32 accumulator that op 83 zeroes and op 87 reads with k=1 -- removing 1.611 GB
# written + 1.611 GB read per layer per rank at T=8192 and collapsing op 87 from k=8 streams at
# 24 KB stride to one contiguous stream. This is aiter's decomposition (its shipped
# fmoe_..._g1u1 gfx942 object carries 96 global_atomic_pk_add_bf16 and no scatter at all).
# The BLOB must be emitted with PLOW_MOE_PF_ATOMIC=1 too; plow_moe_pf_atomic_arm refuses the
# mismatch. NUMERICS-CHANGING (atomic-arrival-order f32 sum, and run-to-run nondeterministic),
# so it is opt-in on both sides and the default object is byte-identical without it.
if [ "${PLOW_MOE_PF_ATOMIC:-0}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MOE_PF_ATOMIC=${PLOW_MOE_PF_ATOMIC}"
fi

# OPT-IN (PLOW_MOE_PF_DET=1): the DETERMINISTIC form of the same 86 -> 87 fusion (op_moe.h
# PLOW_MOE_PF_DET). Op 86 accumulates rint(gate*value * 2^32) into a [T,H] f64 accumulator with a
# device-scope f64 atomic: every partial sum is an integer below 2^53, so every add is EXACT and
# the k-way total does not depend on which workgroup arrives first. Op 87 reads one contiguous
# stream and scales by 2^-32. Twice the accumulator bytes of PLOW_MOE_PF_ATOMIC, and run-to-run
# BIT-REPRODUCIBLE, which that arm is not. Mutually exclusive with it (the header #errors).
# The BLOB must be emitted with PLOW_MOE_PF_DET=1 too; plow_moe_pf_det_arm refuses the mismatch.
#
# DEFAULT ON for gfx942 (opt out with PLOW_MOE_PF_DET=0), 2026-08-09. The numerics blocker that
# kept this opt-in is CLEARED BY MEASUREMENT, not waived: full-set paired GSM8K, per-question,
# one server load per arm, GPU-locked and HSA/coherence gated --
#     control 1268/1319 = 0.9613     det 1268/1319 = 0.9613     paired difference +0.00 pp
#     discordant b = 10, c = 10      McNemar exact two-sided p = 1.0000
#     minimum detectable difference at this discordance ~0.66 pp
# with TTFT -1.79/-2.88/-1.89/-1.66%% at 1k/4k/8k/16k against control spreads of 0.1-1.1%%, and
# DRAM -1.711 GB per MoE layer per rank. The arm was NEVER going to pass the character-identity
# gate it was held to -- sec 2 of glm52-moe-deterministic-writer.md proves no scheme can be
# bit-identical here -- and that gate cannot tell "degraded the model" from "reworded a correct
# answer". It changes 1.8%% of served answers (ten correctness flips each way, netting zero),
# which is a product property, not a defect. An OLD blob on a new object is unaffected (the arm
# is only reached when the packet arms i[5]); a new blob on a PRE-ARM object is a LOUD refusal.
if [ "${PLOW_MOE_PF_DET:-1}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MOE_PF_DET=${PLOW_MOE_PF_DET:-1}"
fi

# CEILING INSTRUMENT ONLY (PLOW_MLA_PF2_ABL=1..4): the V2 MLA prefill's ablation probes —
# one cost term deleted each (op_attention.h d_flash_mla_prefill_v2): 1 = no K-slab stage,
# 2 = no QK MFMA, 3 = no softmax math, 4 = no PV. WRONG OUTPUT by construction, never a
# serve asset. FLASH object only (the V2 body lives there).
if [ "${PLOW_MLA_PF2_ABL:-0}" != 0 ]; then
  AX_FLASH="$AX_FLASH -DPLOW_MLA_PF2_ABL=${PLOW_MLA_PF2_ABL}"
fi

# PLOW_MLA_PF2_DBUF=0 kill switch: the V2 kernel's register-prefetch K-slab pipeline
# (default ON in op_attention.h — bit-identical, see the kernel note). The switch exists
# for A/B only.
if [ "${PLOW_MLA_PF2_DBUF:-1}" = 0 ]; then
  AX_FLASH="$AX_FLASH -DPLOW_MLA_PF2_DBUF=0"
fi

# OPT-IN (PLOW_MLA_FOLD_TB=<G>): TOKEN-BLOCKED MlaMergeFold (op_attention.h
# d_mla_merge_fold_tb, dispatched by interp.hip's exec_mla_merge_fold).
#
# The shipped fold gives one workgroup one (token, head) row and streams the whole 256 KiB
# W_uv[head] panel to produce one 256-wide output row, so at GLM-5.2 TP8 T=8192 the packet
# re-reads 16.8 GB of W_uv out of L2 per layer to do 8.6 GMAC -- 8.6% of a prefill layer's CU
# budget, 120 ms of TTFT at 8k. This arm gives a workgroup G consecutive token-rows of ONE head,
# so a W_uv element in a register is consumed by G accumulators and the stream divides by G.
# Nothing else moves: same lane->column map, same l-slice split, same unroll, same fold tree, and
# the per-token accumulation ORDER is untouched -- so the output is BIT-IDENTICAL and this is an
# OBJECT-level knob like PLOW_MLA_PF_SV: no blob, no emit, no host plumbing, no manifest
# `requires`. PREFILL objects only (the prefill packet is the only one whose n_batch is the token
# count; decode's n_batch=1 fails the arm's own guard). Default OFF, so a build without it is
# byte-identical to one from before this block. Measured standalone at TP8 T=8192 ns=2
# (runtime/bench/amd/glm52_kbench_fold_pf, perf-data/plow-gfx942/glm52-mla-merge-fold.md):
# 1626 us/packet -> 616 at G=8, 692 at G=4, 940 at G=2.
# HISTORY: default-8 2026-08-09, reverted same day, RE-ADOPTED 2026-08-10. The 08-09 revert
# note said "alone or with XR_AGG", but the bisect record (686a3bf) is explicit that this arm
# was tested only IN COMBINATION with the then-broken XR_AGG and "was never content-gated at
# length alone" — it was condemned by association. Solo gate 2026-08-10 (r4b4 ladder asset):
# PASSES needle 3000/8000 x2 each; combined with the FIXED XR_AGG it is gated again before
# every recipe publish. TTFT −3.8/−5.5/−6.2% @4k/8k/16k. Opt out with PLOW_MLA_FOLD_TB=0.
if [ "${PLOW_MLA_FOLD_TB:-8}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MLA_FOLD_TB=${PLOW_MLA_FOLD_TB:-8}"
fi

# OPT-IN (PLOW_MLA_PF_SV=1): the V2 kernel's V-STAGE arm — kv-block LDS swizzle that makes
# the PV transpose read bank-conflict-free, plus double-buffered QK/PV LDS fragments (see
# op_attention.h PLOW_MLA_PF_SV). FLASH OBJECT ONLY (the V2 body lives there);
# BIT-IDENTICAL (LDS addresses and load issue order only), so it is an OBJECT-level knob
# like PLOW_MLA_PF2_DBUF — no blob/emit/host plumbing, no manifest `requires`. Default OFF:
# with it unset every row is byte-identical to a build without this block.
# DEFAULT ON for gfx942 (opt out with PLOW_MLA_PF_SV=0), 2026-08-09. BIT-IDENTICAL by
# construction (LDS addresses and load issue order only), and the adoption gate is already on
# record: objects-only A/B, 3 interleaved rounds, TTFT -1.2%% @4k / -2.5%% @8k / -2.5%% @16k
# against a control whose own round-to-round spread is 0.38-0.40%%, with EVERY sv round below
# EVERY control round at 8k and 16k, and 4/4 character-identical answers including two long
# free-form generations (glm52-flash-streamed-v.md, ADOPTION GATE). The win scaling with KV-tile
# count is what an LDS-side fix predicts and an MFMA-issue-bound loop would not produce.
# Object-level knob: no blob, no emit, no manifest `requires`, so an object built WITHOUT it
# stays fully correct and merely slower -- degrade, not corrupt.
if [ "${PLOW_MLA_PF_SV:-1}" = 1 ]; then
  AX_FLASH="$AX_FLASH -DPLOW_MLA_PF_SV=1"
fi

# FA_DEC_ILV -- interleave the flash-decode K-phase row->wave map. DECODE ROWS ONLY (d_flash_decode
# lives in the decode object; the flash object runs PREFILL). The blocked default gives wave w rows
# [w*64,(w+1)*64) of a 512-row tile, so a split shorter than 64 rows leaves SEVEN OF EIGHT WAVES
# idle -- and Gemma-4's 1024-token sliding window at nsplit=38 is 27 rows. MEASURED here, 3-4 reps
# each, L2-placed blob + PLOW_GATE_HIER, `amd-bench --steps 48`:
#
#   ctx 4096   12.646 -> 12.426  -1.7%
#   ctx 8192   12.690 -> 12.466  -1.6%
#   ctx 16000  12.793 -> 12.592  -1.6%
#
# Uniform across context (it also makes a pass contiguous in kv, which helps even when the tile is
# full), and the serve coherence gate answers "Paris" with it on. Not made the header default
# because gfx950 cannot be measured on this box.
AX_DECODE="$AX_DECODE -DFA_DEC_ILV=1"

# PLOW_INST_PF -- hoist the 64-byte PlowDevInst fetch above the gate poll. DECODE ROWS ONLY, so
# gfx950 objects stay byte-identical (the probe defaults to 0 in interp.hip).
#
# The gate metadata lives on the STREAM ENTRY (e.wait_ofs/len), so nothing in the poll depends on
# the instruction -- but every `in->` read sits after the poll in PROGRAM ORDER, so its
# scalar-cache miss lands after the wait instead of inside it. `insts` is address_space(4), i.e.
# s_load through the scalar cache, which LLVM already hoists aggressively, so this was expected to
# be a no-op. It is not, quite.
#
# MEASURED THREE TIMES, and reported with its weakness: the arms OVERLAP at n=8, so this is a
# small effect, not a clean win.
#   ctx4096 steps48  3 reps   12.095 -> 12.040   -0.45%   (clean separation at n=3)
#   ctx4096 steps128 5 reps   11.808 -> 11.738   -0.59%   (excluding one pf outlier at 12.091)
#   ctx4096 steps128 8 reps   11.823 -> 11.776   -0.40%   (warm-up declared IN ADVANCE; overlap)
#   ctx8192 steps48  3 reps   12.159 -> 12.113   -0.38%
# Every run negative, magnitude 0.38-0.59%. Read-only (two forced scalar loads), no correctness
# surface. Opt out with PLOW_INST_PF=0.
if [ "${PLOW_INST_PF:-1}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_INST_PF=1"
fi

# MEASUREMENT INSTRUMENT ONLY (PLOW_TRACE_PHASE=1): two extra `s_memrealtime` per (workgroup,
# packet), inside `if (prog.trace)`, that split the traced packet into claim+gate / acquire /
# body / publish and pack the two extra deltas into the trace record's unused `pc` field (see
# interp.hip). An UNTRACED run is unaffected; a TRACED run carries the same instrument in every
# arm. DECODE ROWS ONLY. Never ship it on -- it is how the packet-protocol decomposition was
# taken, not a tuning axis.
if [ "${PLOW_TRACE_PHASE:-0}" != 0 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_TRACE_PHASE=${PLOW_TRACE_PHASE}"
fi

# OPT-IN (PLOW_MOE_DEC_X2=1): the block-fp8 DECODE experts run gate|up as ONE loop with both
# weight streams in flight and the activation fragments read once (op_moe.h
# `wave_dot_fp8_blk_x2`). Bit-identical; the fp4 twin of the same pairing measured 1.44x.
# DECODE ROWS ONLY — the grouped PREFILL bodies are a different kernel entirely.
if [ "${PLOW_MOE_DEC_X2:-0}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_MOE_DEC_X2=1${PLOW_MOE_DEC_X2_UN:+ -DPLOW_MOE_DEC_X2_UN=$PLOW_MOE_DEC_X2_UN}"
fi

# OPT-IN (PLOW_MOE_DEC_LG=1): the block-fp8 DECODE expert DOWN takes the narrow-K lane-group map —
# RG row-groups of 64/RG lanes, UNR consecutive row-batches issued before any is consumed, so
# RG*UNR rows are in flight (op_moe.h `moe_down_lg_fp8_blk`). GLM-5.2 TP8 routes DOWN at
# K = I_moe = 256, where the shipped wave-per-row body leaves 48 of 64 lanes dead and keeps ONE
# load outstanding. Bit-identical (modulo the sign of a zero, which MoeCombine cannot see);
# anything wider than LPG*16 falls through to the shipped walk.
#
# DEFAULT ON (opt out with PLOW_MOE_DEC_LG=0). MEASURED -7.6% TPOT, CHARACTER-IDENTICAL.
#
# It shipped OFF because the campaign that built it measured it NULL -- and that measurement was
# taken on `interp_decode_fp8_gq.elf`, which a GLM-5.2 packet never loads (`Variant::detect`
# matches `GemvFp8`, not the block-scaled `GemvFp8Blk` family, so the blob detects as Bf16 and
# decode runs on `interp_decode_gq.elf`). Rebuilt into the object the run does open, three
# interleaved rounds of `scripts/bench_speed.sh`, port 8195:
#
#   ctx 1024   TPOT 28.957 -> 26.760 ms/token   -7.6%   control spread 0.14%
#   ctx 4096        31.290 -> 29.013            -7.3%   control spread 0.16%
#
# i.e. 50x the control's own round-to-round spread, no distribution overlap, TTFT unmoved (this
# is a decode-only axis). Serve gate PASSES on the three canonical prompts AND on a ~14.7k-token
# long-context prompt, character-identical to the control on all four; all 8 ranks
# token-identical on every step of 15 amd-bench runs. Bit-identical by construction -- see
# op_moe.h [FP8-DECODE-DOWN-LG] and glm52-decode-gemv-aiter.md section 3 for the off-device
# reduction and row-coverage proofs. Guarded to K <= LPG*16 and a 16-multiple K, so a wider
# contraction (K3 at TP8 routes DOWN with I_moe=384) falls through to the shipped walk, and
# Gemma's MoE decode is a different function entirely (`d_moe_expert_down_gemma_fp8`).
# Full record: perf-data/plow-gfx942/glm52-packet-protocol-xcd.md.
#
# Its twin PLOW_MOE_DEC_X2 stays OPT-IN: it adds 0.9% on top, which is at the edge of what this
# box can resolve, and its own census row moves the wrong way (GLU busy 4386 -> 4571 CU-us/layer).
if [ "${PLOW_MOE_DEC_LG:-1}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_MOE_DEC_LG=1${PLOW_MOE_DEC_LG_RG:+ -DPLOW_MOE_DEC_LG_RG=$PLOW_MOE_DEC_LG_RG}${PLOW_MOE_DEC_LG_UNR:+ -DPLOW_MOE_DEC_LG_UNR=$PLOW_MOE_DEC_LG_UNR}"
fi

# DEFAULT ON for gfx942 (opt out with PLOW_GEMV_LG=0): the bf16 DECODE GEMV takes the narrow-K
# lane-group map -- RG row-groups
# of 64/RG lanes, UNR consecutive row-batches issued before any is consumed, so RG*UNR rows are in
# flight (op_gemm.h `gemv_rows_lg`, [BF16-GEMV-NARROWK-LG]). The bf16 twin of PLOW_MOE_DEC_LG, one
# kernel over: GLM-5.2 TP8 runs the SHARED-EXPERT DOWN at N=6144 K=256, and `gemv_rows` hands lane L
# the 8 halves at k=8*L, so at K=256 half the wave is out of range and `nchunk = ceil(K/512) = 1`
# leaves the shipped R-split issuing 14 buffer loads of which 12 fetch nothing. Bit-identical by
# construction (same lane->k map per row, same xor-butterfly; only the leading +0.0 butterfly step
# and which wave owns which row change). Guarded to M==1, an 8-multiple K and K <= (64/RG)*8, so
# o_proj (K=2048) and the router gate (K=6144) fall through to the shipped body -- at K >= 512 every
# lane is already live and neither has this defect.
#
# FLIPPED TO DEFAULT-ON FOR gfx942 (2026-08-09). It was landed default-OFF and then simply never
# passed by any recipe, so the shipped configuration left a measured win on the floor for weeks.
#
# ORIGINAL MEASUREMENT (branch `gemv-narrowk`, commits 52d6dd5 / 2f6af04): -1.57/-1.31/-1.36% TPOT
# at ctx 1k/4k/8k against a 0.11-0.21% control spread, ranges disjoint over 3 interleaved rounds;
# traced shared-down packet busy 1506 -> 386 CU-us/layer (-74%) and span 35.0 -> 9.7 us (-72%),
# every other op row flat; 5/5 serve answers CHARACTER-IDENTICAL including a 14.1k-token prompt;
# 108 VGPR / 0 spill unchanged.
#
# INDEPENDENTLY REPRODUCED before flipping (2026-08-09, one session, one client, same box):
#   control  hsaco_r2  TPOT 26.503 ms  (reps 26.494/26.503/26.505, spread 0.04%)
#   arm      hsaco_t0  TPOT 26.077 ms  (reps 26.070/26.077/26.152, spread 0.31%)
#   => -1.61%, i.e. 5x the round-to-round spread, and inside the original -1.3..-1.6% band.
# TTFT unchanged (+0.1/+0.4/+1.1% at 1k/4k/8k, all inside spread) -- the correct negative control
# for a DECODE-only flag. Coherence gate PASS.
#
# It is smaller than PLOW_MOE_DEC_LG's -7.5% on the same defect because this packet already ran
# concurrently with the routed-expert slices, so only the unhidden part of its span reaches the
# token. Full record: perf-data/plow-gfx942/glm52-gemv-narrowk.md.
#
# The header default in op_gemm.h stays 0 -- SAFE value in the header, POLICY in this script, the
# same split PLOW_L2HIER and PLOW_MOE_DEC_LG already use. Including the header never changes
# behaviour; building via this script does.
if [ "${PLOW_GEMV_LG:-1}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_GEMV_LG=1${PLOW_GEMV_LG_RG:+ -DPLOW_GEMV_LG_RG=$PLOW_GEMV_LG_RG}${PLOW_GEMV_LG_UNR:+ -DPLOW_GEMV_LG_UNR=$PLOW_GEMV_LG_UNR}"
fi

# CEILING INSTRUMENT ONLY (PLOW_MOE_DEC_ABL=1|2): the block-fp8 decode expert DOWN with its body
# deleted — 1 keeps the walk and the store and drops every load + the dot, 2 retires the op. WRONG
# OUTPUT by construction; this prices what the packet costs when the kernel costs nothing, and must
# never touch a serve asset.
if [ "${PLOW_MOE_DEC_ABL:-0}" != 0 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_MOE_DEC_ABL=${PLOW_MOE_DEC_ABL}"
fi

# THE TABLE: <stem>|<axes>. Names must match exec/amd.rs `object_name()`
# EXACTLY -- it composes stem + variant infix + arm infix + sched suffix and
# opens the result by literal filename.
ROWS=(
  "interp_prefill|$AX_PREFILL"
  "interp_decode|$AX_DECODE"
  "interp_flash|$AX_FLASH"
  "interp_prefill_fp8|$AX_PREFILL $AX_FP8"
  "interp_decode_fp8|$AX_DECODE $AX_FP8"
  "interp_prefill_fp8kv|$AX_PREFILL $AX_FP8 $AX_FP8KV"
  "interp_decode_fp8kv|$AX_DECODE $AX_FP8 $AX_FP8KV"
  "interp_flash_fp8kv|$AX_FLASH $AX_FP8KV"
  "interp_prefill_mla|$AX_PREFILL $AX_MLA"
  "interp_prefill_mla_moe|$AX_PREFILL $AX_MLA $AX_MOE"
  "interp_prefill_fp8_mla|$AX_PREFILL $AX_MLA $AX_FP8"
  "interp_prefill_fp8_mla_moe|$AX_PREFILL $AX_MLA $AX_MOE $AX_FP8"
  "interp_prefill_fp8kv_mla|$AX_PREFILL $AX_MLA $AX_FP8 $AX_FP8KV"
  "interp_prefill_fp8kv_mla_moe|$AX_PREFILL $AX_MLA $AX_MOE $AX_FP8 $AX_FP8KV"
  # KIMI-K3. `interp_decode_k3` is the row a K3 decode packet actually loads (exec/amd.rs folds
  # K3Moe and K3MoeA4w4 onto PrefillArm::K3 for the decode phase), and it carries the mxfp4
  # EXPERT walks by default. `$AX_MXFP4` rides along so an all-fp4 packet finds its fp4
  # PROJECTION ops in the same object rather than falling through the silent dispatch `default:`.
  "interp_decode_k3|$AX_DECODE $AX_K3 $AX_MXFP4"
  "interp_decode_fp8kv_k3|$AX_DECODE $AX_K3 $AX_MXFP4 $AX_FP8KV"
  # ATTENTION-ONLY, exactly as on gfx950: without $AX_MOE the grouped expert packets fall through
  # `default:` and write nothing. A whole-layer K3 prompt needs the `_moe` rows below.
  "interp_prefill_k3|$AX_PREFILL $AX_MLA_K3 $AX_MXFP4"
  "interp_prefill_k3_moe|$AX_PREFILL $AX_MLA_K3 $AX_MOE $AX_MXFP4"
  # THE ROW A K3 PREFILL PACKET ACTUALLY LOADS (exec/amd.rs: grouped ops with i[3] == MXFP4
  # resolve the K3MoeA4w4 arm). On this arch it contains the simulated body -- see AX_A4W4.
  "interp_prefill_k3_moe_a4w4|$AX_PREFILL $AX_MLA_K3 $AX_MOE $AX_A4W4 $AX_K3_A4W4 $AX_K3_A4W4_TUNE $AX_MXFP4"
  "interp_prefill_fp8kv_k3_moe_a4w4|$AX_PREFILL $AX_MLA_K3 $AX_MOE $AX_A4W4 $AX_K3_A4W4 $AX_K3_A4W4_TUNE $AX_MXFP4 $AX_FP8KV"
)

# PLOW_ROWS_ONLY=<substring>: build only the rows whose stem matches — for iterating on
# ONE object family (e.g. interp_flash) without paying the full 28-object build. The
# resulting dir is PARTIAL; copy it over a full set before serving from it.
if [ -n "${PLOW_ROWS_ONLY:-}" ]; then
  FILTERED=()
  for row in "${ROWS[@]}"; do
    case "${row%%|*}" in *"${PLOW_ROWS_ONLY}"*) FILTERED+=("$row");; esac
  done
  # A mistyped filter matching NOTHING must refuse, not print "ready (0 objects)" — that state
  # has already invalidated performance work once (see LESSONS).
  [ "${#FILTERED[@]}" -gt 0 ] || {
    echo "FAIL: PLOW_ROWS_ONLY='${PLOW_ROWS_ONLY}' matches no object row; stems are:"
    for row in "${ROWS[@]}"; do echo "  ${row%%|*}"; done
    exit 1
  }
  ROWS=("${FILTERED[@]}")
  echo ">>> PLOW_ROWS_ONLY=${PLOW_ROWS_ONLY}: building ${#ROWS[@]} row(s)"
fi

# Delete FIRST. A build that dies must leave nothing behind to run: a stale .elf
# that a test prints CORRECT against is the failure every guard here exists for.
for row in "${ROWS[@]}"; do rm -f "${row%%|*}.elf" "${row%%|*}.co"; done

one() {  # <stem> <axes...>
  local stem="$1"; shift
  if ! "$HIPCC" --offload-arch="$ARCH" -O3 -w -DPLOW_ARCH_SUFFIX="$ARCH" \
        $* --genco "$R/amd/interp.hip" -o "$stem.co" $INC > "$stem.log" 2>&1; then
    echo "FAIL  $stem"; tail -20 "$stem.log"; return 1
  fi
  "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
      --input="$stem.co" --output="$stem.elf"
  rm -f "$stem.co" "$stem.log"
  echo "ok    $stem"
}
export -f one; export HIPCC ARCH R INC BUN

# Both scheduler twins: which one a packet needs is decided by the packet
# (gq_seg_ofs), not by this build, and plowrt opens the twin by literal name.
printf '%s\n' "${ROWS[@]}" | while IFS='|' read -r stem axes; do
  echo "$stem|$axes"
  echo "${stem}_gq|$axes $AX_GQ"
done | xargs -P "$JOBS" -I{} bash -c 'IFS="|" read -r s a <<< "{}"; one "$s" $a'

# test_kernels.elf -- the golden __device__ wrappers, which call the SAME op_*.h bodies the
# interpreter runs, so they must be rebuilt WITH it or a test passes against a stale kernel.
#
# ADDED 2026-08-09, and its absence was the root of a four-link failure. `plowc tune gemm --obj
# <dir>` needs a freshly built test_kernels.elf to time; build_gfx950.sh has always produced one
# and this script never did. So the gfx942 tuning cell could not be REFRESHED by any command in
# the repo -- it was seeded once by hand, went stale on the first runtime/amd/ edit after that,
# and stayed stale, while `tuned_tile_selection` (gfx950-only until today) stayed green. Net
# effect: every gfx942 compile selected GEMM tiles from the analytical model and reported tier
# `portable`, which is what it reports when nothing was ever measured.
#
# Skipped under PLOW_ROWS_ONLY, which is for iterating on one interpreter family and does not
# want the extra minute.
if [ -z "${PLOW_ROWS_ONLY:-}" ]; then
  if "$HIPCC" --offload-arch="$ARCH" -O3 -w --genco "$R/amd/test_kernels.hip" \
        -o tk.co $INC > test_kernels.log 2>&1; then
    "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
        --input=tk.co --output=test_kernels.elf
    rm -f tk.co test_kernels.log
    echo "ok    test_kernels"
  else
    echo "FAIL  test_kernels"; tail -20 test_kernels.log; exit 1
  fi
fi

# THE CLIFF CHECK. Over budget is HSA_STATUS_ERROR_INVALID_ISA at launch, which
# surfaces as a dead run rather than a build failure -- so it is checked here.
echo ""
printf '%-34s %6s %6s %9s %7s\n' object vgpr agpr lds spill
fail=0
for row in "${ROWS[@]}"; do
  for stem in "${row%%|*}" "${row%%|*}_gq"; do
    [ -f "$stem.elf" ] || { echo "MISSING $stem.elf"; fail=1; continue; }
    n=$("$READELF" --notes "$stem.elf" 2>/dev/null)
    v=$(sed -n 's/.*\.vgpr_count: *//p' <<<"$n" | head -1)
    a=$(sed -n 's/.*\.agpr_count: *//p' <<<"$n" | head -1)
    l=$(sed -n 's/.*\.group_segment_fixed_size: *//p' <<<"$n" | head -1)
    s=$(sed -n 's/.*\.vgpr_spill_count: *//p' <<<"$n" | head -1)
    printf '%-34s %6s %6s %9s %7s\n' "$stem" "$v" "$a" "$l" "$s"
    # 65536 B is the CDNA3 workgroup LDS ceiling; the 4-wave flash rows get the
    # 512-register budget, every 8-wave row must hold 256 total.
    #
    # `.vgpr_count` is ALREADY the unified arch+acc total on gfx90a and later --
    # `.agpr_count` is the accumulator SUBSET of it, not an addition. Verified
    # against -Rpass-analysis on interp_flash: that pass reports VGPRs 256 +
    # AGPRs 256 at Occupancy 1, and the note reads vgpr_count 512 / agpr_count
    # 256. Summing the two note fields says 768, which is over a register file
    # that only has 512 -- so adding them fails every 4-wave row for no reason.
    [ "$l" -le 65536 ] || { echo "  OVER LDS: $l > 65536"; fail=1; }
    case "$stem" in
      interp_flash*) [ "$v" -le 512 ] || { echo "  OVER REG: $v > 512"; fail=1; } ;;
      *)             [ "$v" -le 256 ] || { echo "  OVER REG: $v > 256"; fail=1; } ;;
    esac
  done
done
# INSTRUCTION-SELECTION gate, the CDNA3 twin of the one in build_gfx950.sh. The cliff table above
# catches a kernel that will not launch; it does not catch one that launches, is numerically
# correct, and quietly runs on the wrong matrix instruction. The gfx950 expectations cannot be
# reused -- CDNA3's bf16 MFMA is v_mfma_f32_32x32x8_bf16 (half the K) and its only fp8 MFMA is
# the one that file FORBIDS -- so the contract has its own file, and asm_audit.py refuses the
# cross-arch pairing by reading each object's ELF header. Skipped when the file is absent.
EXPECT="$REPO/scripts/asm_expect_gfx942.json"
if [ -f "$EXPECT" ] && command -v python3 >/dev/null; then
  echo ""
  echo "   --- instruction-selection audit ---"
  # Captured rather than piped: `cmd | tail` reports tail's status, so piping would swallow the
  # audit's exit code and the gate would print FAIL lines and then say the build is ready.
  audit=$(python3 "$REPO/scripts/asm_audit.py" --expect "$EXPECT" ./*.elf) || fail=1
  echo "$audit" | tail -20
fi

echo ""
[ "$fail" = 0 ] && echo ">>> $OUT ready ($(ls "$OUT"/*.elf | wc -l) objects)" || {
  echo "!!! one or more rows are over the cliff or missing"; exit 1; }
