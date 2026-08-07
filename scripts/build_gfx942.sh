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
HIPCC="${PLOW_HIPCC:-/opt/rocm/bin/hipcc}"
BUN="${PLOW_BUNDLER:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
        /opt/rocm-*/lib/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)}"
# Same discovery for readelf: the cliff check below dies with empty fields (every row reported
# OVER) when the hardcoded /opt/rocm path is absent, which fails a perfectly good build.
READELF="${PLOW_READELF:-$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/llvm-readelf \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/llvm-readelf \
        /opt/rocm-*/lib/llvm/bin/llvm-readelf 2>/dev/null | head -1)}"
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
AX_FP8="-DPLOW_FP8=1"
AX_FP8KV="-DPLOW_FP8_KV=1"
AX_MLA="-DPLOW_MLA_PREFILL=1"
AX_MOE="-DPLOW_MOE_PREFILL=1"
AX_GQ="-DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=${PLOW_GQ_BATCH:-1}"

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
)

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
echo ""
[ "$fail" = 0 ] && echo ">>> $OUT ready ($(ls "$OUT"/*.elf | wc -l) objects)" || {
  echo "!!! one or more rows are over the cliff or missing"; exit 1; }
