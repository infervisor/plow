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
# PLOW_HSACO_CONFIG=<assets dir | plow_config.h> -- BUILD FOR ONE PACKET. plowc writes
# `plow_config.h` beside `model.pkt`; this is the gfx942 twin of the cmake path's
# PLOW_HSACO_CONFIG (runtime/CMakeLists.txt). Two things happen, and the second is why it
# exists at all:
#   1. every row compiles with -DPLOW_CONFIG="plow_config.h", so the object carries the
#      packet's PLOW_PACKET_HASH as plow_packet_hash_{lo,hi} and plowrt REFUSES it against
#      any other packet (an unstamped object is accepted with a warning -- that is the
#      shipped state, not the goal);
#   2. the object set is DERIVED from the packet instead of hand-set env: the decode batch
#      (PLOW_DECODE_BATCH, and PLOW_GEMV_WALK=1 above 16 rows), the low-rung tiers from the
#      packet's decode ladder (PLOW_DECODE_TIERS), the packed operator-family rows
#      (PLOW_PACKED_PREFILL_CONSUMERS), the token-batch `_tb` twins for a packet with body
#      programs (PLOW_TOKEN_BATCH_TP_OBJECTS), and the opt-in arms the packet's `requires`
#      names (PLOW_DSA_PF, PLOW_MLA_PF_NOPE, PLOW_KDA_CHUNK, PLOW_MOE_PF_ATOMIC/DET, ...).
# An explicit env var still wins -- the header is #ifndef-guarded and this script only fills
# what nothing else set -- EXCEPT where it would build an object the loader refuses by name
# (a GM_BM/GM_BN/GM_DBUF that disagrees with the packet, a narrower decode batch): those
# fail here instead of at load. Without PLOW_HSACO_CONFIG every row is BYTE-IDENTICAL to a
# build from before this knob existed. `build_defines.json` records the -DPLOW_CONFIG on
# every row, so `asm_audit.py --contract` sees a config build as a different axis set.
#
# PLOW_HSACO_EXTENSION=<extension dir> -- build only the rows ONE packet extension declares,
# stamped with the extension's own packet hash. See the block below for the two rules it applies.
#
# THE RESOURCE TABLE IS NO LONGER A COMMENT HERE. It used to be four hand-maintained lines
# claiming interp_prefill "spill 6", and it was stale by two orders of magnitude: the note says
# 126 and the ISA says 1799 scratch ops across the object, 1588 of them in outlined bodies that
# `.vgpr_spill_count` cannot see at all. A table nobody diffs goes stale; the table now lives in
# scripts/obj_baseline_gfx942.json, is DERIVED from each object's own ELF, is asserted on every
# build by `asm_audit.py --contract` at the bottom of this script, and is re-blessed explicitly
# (--bless) so a change to it lands in a commit.
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
# F5 DEBUG BUILD (`PLOW_NORM_RANGE_CHECK=1`, optionally `PLOW_NORM_SS_MAX=1e12f`): arm op_norm.h's
# sum-of-squares assertion. It goes on the INCLUDE line, not into an AX_* set, because it has to
# reach EVERY row -- GM_AX lands only on the GEMM-tile rows, and the point of a range campaign is
# that no norm anywhere escapes it. NOT A SHIPPING CONFIGURATION: the objects trap on violation,
# they are not bit-identical to the default build (measured +61 instructions in `d_rmsnorm`, ~9%
# on a decode norm), and the object contract at the bottom of this script will report the drift.
# See op_norm.h for the contract and the measured margins that keep it off by default.
if [ -n "${PLOW_NORM_RANGE_CHECK:-}" ]; then
  INC="$INC -DPLOW_NORM_RANGE_CHECK=$PLOW_NORM_RANGE_CHECK"
  [ -n "${PLOW_NORM_SS_MAX:-}" ] && INC="$INC -DPLOW_NORM_SS_MAX=$PLOW_NORM_SS_MAX"
  echo "   !! NORM RANGE CHECK ARMED (debug build, not shippable): ${INC#*-I$R/common }"
fi

# PLOW_HSACO_EXTENSION=<extension dir> -- EXTENSION MODE (docs/arch/19, phase 3). An extension
# adds programs to a frozen packet; its objects must be built for the EXTENSION, not the parent.
# This mode is the two facts that follow from that, and nothing else:
#
#   1. PLOW_HSACO_CONFIG points at the extension's own plow_config.h, so every row stamps
#      plow_packet_hash_{lo,hi} with the EXTENSION's PLOW_PACKET_HASH. An object stamped with
#      the parent's hash is refused by name at load (plow_asset::extension::check_object_stamp):
#      it was compiled against the parent's arm set and has none of this bucket's arms, and on
#      AMD a missing arm does not trap, it writes nothing.
#   2. PLOW_ROWS_ONLY is derived from requires.json's object stems, so only the rows the
#      extension declares are built. That is the whole economy of the mechanism -- one more rung
#      costs one object, not twenty-eight.
#
# The resulting directory is PARTIAL by construction. Copy it over the parent's object set, or
# point PLOW_HSACO at a directory holding both.
if [ -n "${PLOW_HSACO_EXTENSION:-}" ]; then
  EXTD="$PLOW_HSACO_EXTENSION"
  [ -d "$EXTD" ] || { echo "FAIL: PLOW_HSACO_EXTENSION=$EXTD is not a directory" >&2; exit 2; }
  REQ="$EXTD/requires.json"
  ECFG="$EXTD/plow_config.h"
  [ -f "$REQ" ]  || { echo "FAIL: $REQ: an extension must declare its objects" >&2; exit 2; }
  [ -f "$ECFG" ] || { echo "FAIL: $ECFG: no plow_config.h in the extension" >&2; exit 2; }
  [ -f "$EXTD/extension.pkt" ] || { echo "FAIL: $EXTD holds no extension.pkt" >&2; exit 2; }

  # requires.json is machine-written (serde_json pretty), so one `"key": value` per line is the
  # grammar -- the same assumption `cfg_get` makes about plow_config.h.
  jstr() { sed -n "s/.*\"$1\": *\"\([^\"]*\)\".*/\1/p" "$REQ"; }
  ext_arch="$(jstr arch | head -1)"
  [ "$ext_arch" = "$ARCH" ] || {
    echo "FAIL: $REQ declares arch '$ext_arch'; this script builds $ARCH" >&2; exit 2; }

  # The hash the extension's objects will be stamped with must be the one requires.json pins, or
  # this build produces objects the loader it is building them for will refuse.
  ext_pin="$(jstr pairing_hash | head -1 | tr 'A-F' 'a-f')"
  ext_pin="${ext_pin#0x}"
  ext_own="$(sed -n 's/^#define PLOW_PACKET_HASH 0x\([0-9a-fA-F]*\)ull$/\1/p' "$ECFG" | head -1 | tr 'A-F' 'a-f')"
  [ -n "$ext_own" ] || { echo "FAIL: $ECFG carries no PLOW_PACKET_HASH" >&2; exit 2; }
  [ -n "$ext_pin" ] || { echo "FAIL: $REQ carries no pairing_hash" >&2; exit 2; }
  # Compare as numbers, so 0x0000dead and 0xdead are the same pin.
  if [ "$((16#$ext_pin))" != "$((16#$ext_own))" ]; then
    echo "FAIL: $REQ pins packet 0x$ext_pin but $ECFG is 0x$ext_own;" >&2
    echo "      plowrt refuses objects stamped for a different artifact." >&2
    exit 1
  fi

  ext_stems="$(jstr stem | sed 's/^/=/' | paste -sd, -)"
  [ -n "$ext_stems" ] || { echo "FAIL: $REQ lists no object stems" >&2; exit 2; }
  if [ -n "${PLOW_ROWS_ONLY:-}" ] && [ "$PLOW_ROWS_ONLY" != "$ext_stems" ]; then
    echo "FAIL: PLOW_ROWS_ONLY='$PLOW_ROWS_ONLY' is set, but the extension declares '$ext_stems';" >&2
    echo "      an extension builds exactly the rows it declares. Unset PLOW_ROWS_ONLY." >&2
    exit 1
  fi
  PLOW_ROWS_ONLY="$ext_stems"
  if [ -n "${PLOW_HSACO_CONFIG:-}" ] && [ "$PLOW_HSACO_CONFIG" != "$EXTD" ] \
     && [ "$PLOW_HSACO_CONFIG" != "$ECFG" ]; then
    echo "FAIL: PLOW_HSACO_CONFIG=$PLOW_HSACO_CONFIG disagrees with the extension at $EXTD;" >&2
    echo "      an extension's objects are stamped with the EXTENSION's packet hash." >&2
    exit 1
  fi
  PLOW_HSACO_CONFIG="$ECFG"
  echo ">>> EXTENSION MODE: $EXTD"
  echo "    stamping packet 0x$ext_own, building rows: $(jstr stem | paste -sd' ' -)"
fi

# PLOW_HSACO_CONFIG (see the header): resolve the packet's plow_config.h, stamp every row
# with it, and derive the object set from what it says. Resolved BEFORE the axes below, which
# read the knobs this block fills. `cfg_get` reads one `#define NAME value` line; the header
# is machine-written (devgen::manifest::config_header), so the grammar is exactly that.
CFG=""; AX_CONFIG=""; AX_CONFIG_JSON=""
# PLOW_HSACO_EXTRA_DEFINES="-DX=1 ...": raw -D appended to EVERY row, for an opt-in kernel-arm
# A/B whose header default is the shipped body (e.g. -DPLOW_COMBINE_VEC=1 -DPLOW_RN_ROWS=2).
# Recorded in build_defines.json beside AX_CONFIG so the contract audit sees the axis.
AX_EXTRA="${PLOW_HSACO_EXTRA_DEFINES:-}"
# It may NOT carry the axes the loader pairs against the packet (tile geometry, wave count,
# decode batch): those have dedicated variables that this script cross-checks against the
# packet's `requires`, and a -D smuggled in here would bypass that check and be refused at load
# — or, worse, silently redefine a header default (the GM_AX class of defect).
case " $AX_EXTRA " in
  *" -DGM_BM"*|*" -DGM_BN"*|*" -DGM_BK"*|*" -DGM_DBUF"*|*" -DPLOW_WG_WAVES"*|*" -DPLOW_DECODE_BATCH"*|*" -DPLOW_GEMV_MM"*)
    echo "FAIL: PLOW_HSACO_EXTRA_DEFINES must not set tile/wave/decode-batch axes; use their own variables" >&2
    exit 2 ;;
esac
if [ -n "${PLOW_HSACO_CONFIG:-}" ]; then
  CFG="$PLOW_HSACO_CONFIG"
  [ -d "$CFG" ] && CFG="$CFG/plow_config.h"
  [ -f "$CFG" ] || { echo "FAIL: PLOW_HSACO_CONFIG=$PLOW_HSACO_CONFIG: no plow_config.h there" >&2; exit 2; }
  CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
  cfg_get() { sed -n "s/^#define $1 //p" "$CFG" | head -1; }
  cfg_str() { cfg_get "$1" | sed 's/^"//; s/"$//'; }
  CFG_HASH="$(cfg_get PLOW_PACKET_HASH)"
  case "$CFG_HASH" in
    0x*ull) ;;
    *) echo "FAIL: $CFG carries no PLOW_PACKET_HASH -- not a plowc plow_config.h" >&2; exit 2 ;;
  esac
  CFG_ARCH="$(cfg_str PLOW_PACKET_OBJECT_ARCH)"
  if [ -n "$CFG_ARCH" ] && [ "$CFG_ARCH" != "$ARCH" ]; then
    echo "FAIL: $CFG describes a $CFG_ARCH packet; this script builds $ARCH" >&2; exit 2
  fi
  INC="$INC -I$(dirname "$CFG")"
  AX_CONFIG="-DPLOW_CONFIG=\"$(basename "$CFG")\""
  AX_CONFIG_JSON=" -DPLOW_CONFIG=\\\"$(basename "$CFG")\\\""
  # Decode batch and ladder. The packet's batch is a FLOOR for the object's bucket (the bucket
  # is a ceiling, so a wider object serves it); a narrower explicit batch is the object the
  # loader refuses (`check_gemv_capacity`), so refuse it here by name.
  cfg_batch="$(cfg_get PLOW_PACKET_DECODE_BATCH)"
  [ -n "$cfg_batch" ] || cfg_batch="$(cfg_get GV_MM_MAX)"
  if [ -n "$cfg_batch" ]; then
    if [ -n "${PLOW_DECODE_BATCH:-}" ] && [ "$PLOW_DECODE_BATCH" -lt "$cfg_batch" ]; then
      echo "FAIL: packet decodes at batch $cfg_batch but PLOW_DECODE_BATCH=$PLOW_DECODE_BATCH is set;" >&2
      echo "      an object narrower than the packet's widest GEMV is refused at load." >&2; exit 1
    fi
    PLOW_DECODE_BATCH="${PLOW_DECODE_BATCH:-$cfg_batch}"
    [ "$PLOW_DECODE_BATCH" -gt 16 ] && : "${PLOW_GEMV_WALK:=1}"
  fi
  # The ladder's rungs below the batch become the tiers; an unset PLOW_DECODE_TIERS would
  # otherwise fall to the script's own 1/2/4/8 default below.
  cfg_ladder="$(cfg_str PLOW_PACKET_DECODE_LADDER)"
  if [ -n "$cfg_ladder" ] && [ -z "${PLOW_DECODE_TIER:-}" ] && [ -z "${PLOW_DECODE_TIERS+x}" ]; then
    tiers=""
    for w in ${cfg_ladder//,/ }; do
      [ "$w" -lt "${PLOW_DECODE_BATCH:-1}" ] && tiers="$tiers${tiers:+,}$w"
    done
    PLOW_DECODE_TIERS="$tiers"
  fi
  # The packed operator-family objects, opened by literal name under PLOW_PACKED_PREFILL_ROUTE,
  # and their `_tb` twins for a packet that carries token-batch BODY programs (exec/amd.rs
  # opens `interp_packed_mla_{norm,flash}_tb*` for those and nothing else).
  # Bodies need the packed family rows too. An explicit opt-out the packet contradicts is
  # refused: the set would load-fail on the first body (or sibling) that opens a missing object.
  cfg_packed=0; cfg_tb=0
  grep -q '^#define PLOW_PACKET_HAS_PACKED_PREFILL_TOPOLOGY 1$' "$CFG" && cfg_packed=1
  grep -q '^#define PLOW_PACKET_HAS_TOKEN_BATCH_BODIES 1$' "$CFG" && { cfg_packed=1; cfg_tb=1; }
  for pair in "PLOW_PACKED_PREFILL_CONSUMERS:$cfg_packed" "PLOW_TOKEN_BATCH_TP_OBJECTS:$cfg_tb"; do
    key="${pair%%:*}"; want="${pair#*:}"; cur="${!key:-}"
    [ "$want" = 1 ] || continue
    if [ -n "$cur" ] && [ "$cur" != 1 ]; then
      echo "FAIL: packet carries programs that open the $key objects but $key=$cur is set;" >&2
      echo "      plowrt refuses to load a packet whose family objects are missing." >&2
      exit 1
    fi
    printf -v "$key" 1
  done
  # `backends.<arch>.requires`, verbatim. Three kinds of entry: row-selecting axes (every
  # variant row is built anyway -- the small-rung MLA and split objects included -- and plowrt
  # picks by filename and refuses by marker), the tile geometry (op_gemm.h's CDNA3 defaults ARE
  # the requirement -- an A/B override that disagrees is refused here rather than at load), and
  # the opt-in arms this script has a knob for. Anything else is either an unconditional arm, a
  # runtime branch, or #ifndef-defaulted from the header itself; it is listed in the summary
  # line so nothing is silently dropped.
  cfg_unmapped=""
  for tok in $(cfg_str PLOW_PACKET_OBJECT_REQUIRES); do
    key="${tok%%=*}"; val="${tok#*=}"; [ "$tok" = "$key" ] && val=1
    case "$key" in
      GM_BM|GM_BN|GM_DBUF)
        cur="${!key:-}"
        if [ -n "$cur" ] && [ "$cur" != "$val" ]; then
          echo "FAIL: packet requires $key=$val but $key=$cur is set in the environment;" >&2
          echo "      plowrt refuses a prefill object whose plow_geom_$key disagrees with the packet." >&2
          exit 1
        fi ;;
      PLOW_DSA_PF_ARM)       [ "$val" = 1 ] && : "${PLOW_DSA_PF:=1}" ;;
      PLOW_MLA_PF2_NOPE_ARM) [ "$val" = 1 ] && : "${PLOW_MLA_PF_NOPE:=1}" ;;
      PLOW_KDA_CHUNK)        [ "$val" = 1 ] && : "${PLOW_KDA_CHUNK:=1}" ;;
      PLOW_KDA_CONV_STEP_DB) [ "$val" = 1 ] && : "${PLOW_K3_KDA_CONV_STEP_DB:=1}" ;;
      PLOW_MOE_PF_ATOMIC)    [ "$val" = 1 ] && : "${PLOW_MOE_PF_ATOMIC:=1}" ;;
      PLOW_MOE_PF_DET)       [ "$val" = 1 ] && : "${PLOW_MOE_PF_DET:=1}" ;;
      PLOW_GLM_FUSE_QNORM)   [ "$val" = 1 ] && : "${PLOW_GLM_FUSE_QNORM:=1}" ;;
      PLOW_DSA_SELECT_SPLIT) [ "$val" = 1 ] && : "${PLOW_DSA_SELECT_SPLIT:=1}" ;;
      PLOW_WG_WAVES|PLOW_BUCKET_DECODE|PLOW_FP8|PLOW_FP8_KV|PLOW_MXFP4|PLOW_W8A8|PLOW_MLA_PREFILL|PLOW_MOE_PREFILL|PLOW_MOE_PF_A4W4|PLOW_K3|PLOW_MLA_PREFILL_FP8_SPLIT) ;;
      *) cfg_unmapped="$cfg_unmapped $tok" ;;
    esac
  done
  echo "   packet config: $CFG"
  echo "   packet config: hash=${CFG_HASH%ull} batch=${PLOW_DECODE_BATCH:-1} walk=${PLOW_GEMV_WALK:-0} tiers=${PLOW_DECODE_TIERS:-none} packed=${PLOW_PACKED_PREFILL_CONSUMERS:-0} tb=${PLOW_TOKEN_BATCH_TP_OBJECTS:-0} dsa_pf=${PLOW_DSA_PF:-0} moe_pf_atomic=${PLOW_MOE_PF_ATOMIC:-0} moe_pf_det=${PLOW_MOE_PF_DET:-1} kda_chunk=${PLOW_KDA_CHUNK:-0}"
  [ -z "$cfg_unmapped" ] || echo "   packet config: requires with no build knob here (marker-checked at load):$cfg_unmapped"
fi
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
# GM_AX is the same escape hatch one level down: raw -D for the per-rung geometry and schedule
# knobs op_gemm.h `#ifndef`-guards (GM_SM_BK, GM_MD_*, GM_PGR2, GM_PLR, GM_PRIO). They have no
# dedicated variable because there are a dozen of them and each is an A/B, not a policy.
CDNA3_TILE="-DPLOW_WG_WAVES=8${GM_BM:+ -DGM_BM=$GM_BM}${GM_BN:+ -DGM_BN=$GM_BN}${GM_BK:+ -DGM_BK=$GM_BK}${GM_DBUF:+ -DGM_DBUF=$GM_DBUF}${GM_AX:+ $GM_AX}"
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
AX_PREFILL="-DPLOW_PACKED_PREFILL_DENSE_CONSUMERS=1 -DPLOW_BUCKET_DECODE=0 $CDNA3_TILE $AX_GMOE"
# PLOW_GEMV_MM is next_pow2(PLOW_DECODE_BATCH) CLAMPED TO 16, not the batch itself. The GEMV
# ladder instantiates MM in {1,2,4,8,16} and one instantiation with a runtime M serves every
# M <= MM, so the bucket is a CEILING. Passing the raw batch through was a bug in this script:
# Passing PLOW_DECODE_BATCH directly handed unsupported MM values to hipcc and every decode row
# failed to build. Walking objects keep MM capped at 16 while serving batches through 128.
RAW_BATCH="${PLOW_DECODE_BATCH:-1}"
case "$RAW_BATCH" in
  ''|*[!0-9]*) echo "PLOW_DECODE_BATCH must be an integer in 1..128" >&2; exit 1 ;;
esac
if [ "$RAW_BATCH" -lt 1 ] || [ "$RAW_BATCH" -gt 128 ]; then
  echo "PLOW_DECODE_BATCH must be in 1..128, got $RAW_BATCH" >&2
  exit 1
fi

# PLOW_DECODE_TIER=<n>: THIS DECODE OBJECT IS A LOW-RUNG TIER and will only ever be handed
# packets of at most n rows. plowrt's `PLOW_HSACO_LOWRUNG=<dir>:n` co-loads such an object and
# runs its pairing checks with n, not with the blob's widest batch, so the GEMV bucket must
# follow the TIER width here too. Everything else about the row is unchanged, which is the point:
# the tier and the wide object differ in PLOW_GEMV_MM and in nothing else.
#
# WHY A TIER IS WORTH BUILDING. `PLOW_GEMV_MM` is a compiled CEILING and one instantiation serves
# every M <= MM by predicating each activation row with `live = (m < M)` -- and then computing its
# dot product anyway. On gfx942, which has no `v_dot2c_f32_bf16`, that discarded work is 24 VALU
# operations per 16 bytes of weight per dead row, and the unroll is cut with it (GV_UNROLL_M4 = 6
# against GV_UNROLL = 11). Measured on the Gemma-4 31B decode shapes
# (runtime/bench/amd/gemma31_gemv_decode_bench.*, per-token totals over the T=4 instance counts):
#
#   compiled bucket / runtime rows      MM=1 M=1   MM=4 M=1   MM=4 M=2   MM=4 M=4
#   plain-GEMV projection total, ms       15.43      26.04      26.77      27.49
#
# i.e. a batch-1 packet on the MM=4 object pays 1.69x for arithmetic it discards. A blob whose
# decode ladder is 1/2/4 therefore wants three decode objects, not one.
TIER="${PLOW_DECODE_TIER:-0}"
case "$TIER" in
  ''|*[!0-9]*) echo "PLOW_DECODE_TIER must be an integer" >&2; exit 1 ;;
esac
if [ "$TIER" -gt 0 ]; then
  if [ "$TIER" -gt "$RAW_BATCH" ]; then
    echo "REFUSING: PLOW_DECODE_TIER=$TIER exceeds PLOW_DECODE_BATCH=$RAW_BATCH." >&2
    echo "  A tier serves a SUBSET of the ladder; widen the batch or narrow the tier." >&2
    exit 1
  fi
  WIDTH="$TIER"
else
  WIDTH="$RAW_BATCH"
fi
P2=1
while [ "$P2" -lt "$WIDTH" ]; do P2=$((P2 * 2)); done
[ "$P2" -gt 16 ] && P2=16
GVMM="$P2"

# The row bucket and the packet batch are separate axes when the walk is enabled. A B16 packet
# on MM8 is sound only with PLOW_GEMV_WALK=1: the kernel makes two weight passes and advertises
# both markers, which plowrt validates before launch. Without the walk, rows MM..B-1 stay stale.
WALK="${PLOW_GEMV_WALK:-0}"
case "$WALK" in
  0) AX_GEMV_WALK="" ;;
  1) AX_GEMV_WALK="-DPLOW_GEMV_WALK=1" ;;
  *) echo "PLOW_GEMV_WALK must be 0 or 1" >&2; exit 1 ;;
esac
if [ "$WIDTH" -gt 16 ] && [ "$WALK" != 1 ]; then
  echo "REFUSING: decode width $WIDTH requires PLOW_GEMV_WALK=1 above 16 rows." >&2
  exit 1
fi
if [ -n "${PLOW_GEMV_MM:-}" ]; then
  case "$PLOW_GEMV_MM" in
    ''|*[!0-9]*) echo "PLOW_GEMV_MM must be a number" >&2; exit 1 ;;
    1|2|4|8|16) ;;
    *) echo "PLOW_GEMV_MM must be one of 1,2,4,8,16" >&2; exit 1 ;;
  esac
  if [ "$PLOW_GEMV_MM" -lt "$WIDTH" ] && [ "$WALK" != 1 ]; then
    echo "REFUSING: PLOW_GEMV_MM=$PLOW_GEMV_MM < width $WIDTH with PLOW_GEMV_WALK unset." >&2
    echo "  Without the walk, gemv_rows<MM> writes rows 0..MM-1 and leaves the rest STALE." >&2
    exit 1
  fi
  GVMM="$PLOW_GEMV_MM"
fi
AX_DECODE="-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=$GVMM $AX_GEMV_WALK $CDNA3_TILE $AX_GMOE"
echo "   decode GEMV batch bucket: PLOW_GEMV_MM=$GVMM walk=$WALK (PLOW_DECODE_BATCH=${PLOW_DECODE_BATCH:-1} tier=${PLOW_DECODE_TIER:-0})"
# OPT-IN (PLOW_GEMV_WALK=1): the §6g-WALK row-block outer loop — the object serves any M in
# ceil(M/MM) passes of the compiled bucket, and the LDS staging bound becomes min(MM,M)*K.
# REQUIRED for a PLOW_DECODE_BATCH>16 ladder (PLOW_GEMV_MAXM caps the bucket at 16; without
# the walk, rows 16.. would never be written — plowrt refuses the pair on the missing
# `plow_gemv_walk_1` marker rather than serving stale rows).
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
  AX_DECODE="-DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=$GVMM $AX_GEMV_WALK -DPLOW_WG_WAVES=8 -DPLOW_MOE_GEMMA=1 \
             -DGM_BM=128 -DGM_BN=256 -DGM_BK=32 -DPLOW_NO_MLA_DEC=1 -DPLOW_WPE=5"
fi
# $AX_GMOE on the FLASH row too. The flash object runs only class-4 flash
# segments and has no use for op 73 -- but plowrt's `check_moe_gemma_arms` is a
# blanket check over EVERY object it loads, so a flash object without the marker
# symbol is REJECTED, and the rejection is an `info!` degrade ("no flash object
# -- flash segments run on the 8-wave interpreter"), not an error. See the note
# on AX_GMOE above for why that degrade is not benign.
AX_FLASH="-DPLOW_PACKED_PREFILL_DENSE_CONSUMERS=1 -DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 $CDNA3_TILE_4W $AX_GMOE"
# V2 MLA prefill arm (d_flash_mla_prefill_v2): the full-column-wave layout that needs this
# object's 512-register budget. Marker `plow_mla_pf_v2_arm_1`; the host routes FlashMlaPrefill
# segments here only under PLOW_MLA_PF_V2=1, so carrying the arm costs Gemma nothing.
AX_FLASH="$AX_FLASH -DPLOW_MLA_PF_V2_ARM=1"
# The V2 MLA-prefill KV slab depth. 32 is shipped; 48 is the deepest that fits the 64 KiB
# LDS budget (62,304 B against 83,040 B at 64) and pays the per-(query,tile) softmax
# bookkeeping a third fewer times. FLASH OBJECT ONLY — the V2 body lives there.
if [ -n "${FA_MLA_PF2_BKV:-}" ]; then
  AX_FLASH="$AX_FLASH -DFA_MLA_PF2_BKV=$FA_MLA_PF2_BKV"
fi
# The V2 MLA prefill's DEFERRED-FRAME online softmax (op_attention.h FA_MLA_PF2_DEFER),
# DEFAULT ON for CDNA3. The accumulator lives in an integer exponent frame instead of
# tracking the running max, so the 128-mul 512-wide rescale and both quarter-wave reduces
# leave the per-KV-tile path: the rescale rides on frame re-takes (a handful over the ~1000
# tiles of a 32k row) and the l reduce happens once in the epilogue. FLASH OBJECT ONLY — the
# V2 body lives there, behind PLOW_MLA_PF_V2_ARM, so no other object's register cliff moves.
#
# NOT BIT-IDENTICAL, which is what separates it from PLOW_MLA_PF_SV and PLOW_MLA_FOLD_TB
# above: the exp arguments move by the frame offset, so greedy completions of a long prompt
# eventually diverge on a near-tie and a character-identity gate CANNOT pass by construction.
# It is gated instead on (a) mla_test's decode-oracle error, which stays in the shipped
# body's class on all 16 shapes including the fp8 arm, and (b) long-context needle retrieval
# and coherence at 8k/32k. Measured TTFT at TP4 (2 interleaved rounds, control spread 0.3%):
# -2.0% @4k, -3.2% @8k, -4.8% @16k, -7.1% @32k; the fitted T^2 (attention) coefficient falls
# 12.5% while the linear coefficient moves 0.3%. See docs/amd/glm53-longctx-and-throughput.md
# section 7. FA_MLA_PF2_DEFER=0 restores the shipped running-max form for A/B.
if [ -n "${FA_MLA_PF2_DEFER:-}" ]; then
  AX_FLASH="$AX_FLASH -DFA_MLA_PF2_DEFER=$FA_MLA_PF2_DEFER"
fi
# Log2 headroom the re-taken frame keeps above the row max (default 8). Larger = rarer
# re-takes, at the cost of dynamic range below the max.
if [ -n "${FA_MLA_PF2_FRAME:-}" ]; then
  AX_FLASH="$AX_FLASH -DFA_MLA_PF2_FRAME=$FA_MLA_PF2_FRAME"
fi
# The frame arm's NaN-free f32->bf16 for the P strip (default on; only reachable under
# FA_MLA_PF2_DEFER, where p is provably in [0,1] and f2bf's guard is dead code). Worth
# another -1.0% at 32k on top of the frame, and value-identical over that domain.
if [ -n "${FA_MLA_PF2_FASTBF:-}" ]; then
  AX_FLASH="$AX_FLASH -DFA_MLA_PF2_FASTBF=$FA_MLA_PF2_FASTBF"
fi
# Small K3 buckets remain L2-placed while machine-filling V2 buckets use wave segments.
# The dispatch arm is inert when `l2_domains == 0`, so one object safely serves both forms.
AX_FLASH="$AX_FLASH -DPLOW_L2_PLACE_DISPATCH=1"
# OPT-IN (PLOW_FA_LAZY=1): wave-voted skip of the online-softmax corr/rescale when the
# running max did not move (bit-identical; see op_attention.h FA_LAZY_RESCALE). FLASH
# OBJECT ONLY — the 8-wave prefill interpreter sits on the 256-reg cliff and even a
# no-op perturbation of the flash branch's allocator can drop it to 1 wave/SIMD; the
# 4-wave flash object has the 512-reg budget. Default OFF.
if [ "${PLOW_FA_LAZY:-0}" = 1 ]; then
  AX_FLASH="$AX_FLASH -DFA_LAZY_RESCALE=1"
fi
# OPT-IN (PLOW_FA_LDS_DMA=1): stage the flash K/V tile global->LDS with `global_load_lds_dword`
# instead of through VGPRs (op_attention.h FA_LDS_DMA / amd_common.h cp_async4). FLASH OBJECT
# ONLY and bf16 only -- the fp8-KV arm dequantizes DURING staging and a DMA cannot do arithmetic
# on the way.
#
# WHY IT EXISTS: AITER's shipped gfx942 MLA prefill issues 132 `buffer_load_dword ... lds` and
# never lands K/V in a register; plow's flash object issued ZERO and staged through a register
# file already fully committed at 512 VGPR / one wave per SIMD. The tiling was already the same
# (BM 128, BN 32, 4 waves) -- the staging was the difference, and the flash segment is ~75% of
# long-context prefill wall time. docs/amd/tp-bringup-mi300x.md 7f.
#
# NOT bit-identical in scheduling and NOT yet measured, so it is OFF by default: cp_async4 moves
# 4 B/lane (the only width CDNA3 implements), which lays lanes down at a lane*4 stride, so the
# staging loop is per-WAVE rather than per-thread and the LDS destination must be wave-uniform.
if [ "${PLOW_FA_LDS_DMA:-0}" = 1 ]; then
  AX_FLASH="$AX_FLASH -DFA_LDS_DMA=1"
fi
# DPP/swizzle half-wave reductions (PLOW_FA_RED_DPP, default ON) and the interior-tile mask
# skip (PLOW_FA_FASTMASK, default ON). Both are bit-identical and both are FLASH-OBJECT ONLY
# for the same register-cliff reason as PLOW_FA_LAZY above: the 8-wave prefill/decode rows hold
# 256 registers and are not re-qualified here. Measured on d_flash_prefill<256>, one KV tile:
# 1651 -> 1584 instructions, ds_bpermute 160 -> 64 ds_swizzle, s_waitcnt 163 -> 91, and
# d_flash_prefill<512> scratch spill 47 -> 10 B. Gemma-4 31B cold-prefill TTFT -0.6/-2.0/-3.5%
# at 128/1024/4096 tokens with identical greedy output; docs/amd/gemma4-31b-mi300x.md.
[ "${PLOW_FA_RED_DPP:-1}" = 0 ] || AX_FLASH="$AX_FLASH -DPLOW_WAVE_RED_DPP=1"
[ "${PLOW_FA_FASTMASK:-1}" = 0 ] || AX_FLASH="$AX_FLASH -DFA_FASTMASK=1"
# OPT-IN (PLOW_FA_HEAD_MAJOR=1): head-slowest prefill work order, for the KV-window/L2 study.
# Measured and a small loss (op_attention.h FA_HEAD_MAJOR); kept as the arm that prices it.
if [ "${PLOW_FA_HEAD_MAJOR:-0}" = 1 ]; then
  AX_FLASH="$AX_FLASH -DFA_HEAD_MAJOR=1"
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
case "${PLOW_MXFP4_DEC_NT:-1}" in
  0) AX_MXFP4="$AX_MXFP4 -DPLOW_MXFP4_DEC_NT=0" ;;
  1) ;;
  *) echo "FAIL: PLOW_MXFP4_DEC_NT must be 0 or 1" >&2; exit 2 ;;
esac
# KIMI-K3. `GV_UNROLL=14` is on the K3 rows only and is measured, not derived: K3's dominant
# decode GEMV is K=7168, whose nchunk is exactly 14 (runtime/CMakeLists.txt records the sweep).
AX_K3="-DPLOW_K3=1 -DGV_UNROLL=14"
AX_MLA_K3="$AX_MLA -DPLOW_K3=1"
case "${PLOW_KDA_CHUNK:-0}" in
  0) AX_KDA_CHUNK="" ;;
  1) AX_KDA_CHUNK="-DPLOW_KDA_CHUNK=1" ;;
  *) echo "FAIL: PLOW_KDA_CHUNK must be 0 or 1" >&2; exit 2 ;;
esac
AX_MLA_K3="$AX_MLA_K3 $AX_KDA_CHUNK"
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
case "${PLOW_MOE_PF_A4W4_WEIGHT_NT:-0}" in
  0) ;;
  1) AX_A4W4="$AX_A4W4 -DPLOW_MOE_PF_A4W4_WEIGHT_NT=1" ;;
  *) echo "FAIL: PLOW_MOE_PF_A4W4_WEIGHT_NT must be 0 or 1" >&2; exit 2 ;;
esac
# K3's full prefill program is L2-placed on gfx942, unlike the segmented prefill programs this
# script otherwise builds. Arm the queue interpretation on exactly the K3 A4W4 rows; hierarchy
# remains a separate, unmeasured PLOW_L2HIER_PF experiment.
AX_K3_A4W4="-DPLOW_L2_PLACE_DISPATCH=1"
case "${PLOW_KDA_PF_STATE_RESIDENT:-0}" in
  0) AX_K3_PF_STATE="" ;;
  1) AX_K3_PF_STATE="-DPLOW_KDA_PF_STATE_RESIDENT=1" ;;
  *) echo "FAIL: PLOW_KDA_PF_STATE_RESIDENT must be 0 or 1" >&2; exit 2 ;;
esac
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
AX_DECODE_GQ=""
if [ "${PLOW_L2HIER:-1}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_L2_PLACE_DISPATCH=1"
  # A/B ARM (PLOW_GATE_HIER=0): placement WITHOUT the two-level rendezvous. The table above
  # prices the two together and the pair only activates on an L2-PLACED blob, so until that
  # blob existed on this model there was no way to ask which half pays. Placement's half is
  # locality (a consumer reads its producer out of its own XCD's L2, and it lands in the op
  # BODIES); the hierarchy's half is the per-workgroup wbl2/inv at the gate. Objects with the
  # define off are byte-identical to a build from before it existed.
  [ "${PLOW_GATE_HIER:-1}" = 0 ] || AX_DECODE_GQ="-DPLOW_GATE_HIER=1"
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

# OPT-IN (PLOW_DSA_SELECT_SPLIT=1): the split-row batched decode selection (op 59 i[4]=2,
# op_attention_common.h d_index_select_split), emitted by PLOW_GLM_SELECT_SPLIT. A packet that
# carries it requires the arm (set here from its `requires`), and plowrt refuses it on an object
# without `plow_dsa_select_split_arm`. The selection runs inside the decode MLA segment, so the
# FLASH rows get it too. Default OFF: the shipped objects are byte-unchanged.
case "${PLOW_DSA_SELECT_SPLIT:-0}" in
  0) ;;
  1) AX_DECODE="$AX_DECODE -DPLOW_DSA_SELECT_SPLIT=1"; AX_FLASH="$AX_FLASH -DPLOW_DSA_SELECT_SPLIT=1" ;;
  *) echo "FAIL: PLOW_DSA_SELECT_SPLIT must be 0 or 1" >&2; exit 2 ;;
esac

# OPT-IN (PLOW_L2HIER_PF=1): L2-DOMAIN DISPATCH ON THE PREFILL ROWS, for blobs emitted with
# PLOW_L2_PLACE_PREFILL=1 (which is NOT the AMD default -- see crates/devgen/src/lib.rs).
# Without this the prefill objects lack the axis and plowrt refuses a prefill-placed blob. The
# FLASH rows already carry it unconditionally (AX_FLASH above), which is why Gemma's split
# prefill program only needs this one knob.
#
# PLACEMENT ONLY, and that is a hard limit rather than caution. This block used to add
# `-DPLOW_GATE_HIER=1` as well, and an object built that way DOES NOT COMPILE: the guard at the
# top of interp.hip is
#     #if PLOW_GATE_HIER && (!PLOW_BUCKET_DECODE || !PLOW_GLOBAL_QUEUE || !PLOW_L2_PLACE_DISPATCH)
#     #error "PLOW_GATE_HIER requires a decode global-queue object with L2-domain dispatch"
# and AX_PREFILL carries -DPLOW_BUCKET_DECODE=0. Verified by compiling the row by hand: one
# error, no object. plowrt's `check_gate_hier_object` refuses the same pairing a second time at
# load. So the two-level gate is DECODE-ONLY by construction, and the hierarchy half of
# "PLOW_L2HIER_PF" was never buildable; what remains here is the placement half.
#
# MEASURED AND NOT DEFAULTED. With these objects a prefill-placed Gemma-4-31B blob (dense, whose
# prefill program spans the prefill AND flash objects -- the shape recorded above as hanging
# amd-bench) runs to completion in normal wall time and moves TTFT -1.4/-2.1/-6.2% at 128/2048/8192
# tokens solo, with one reproducible +3.2% at 2048/conc-4 and decode untouched. Left opt-in for
# blast radius, not for the number: it makes every prefill object in a tree incompatible with a
# default emit. docs/amd/gemma4-31b-mi300x.md has the table.
if [ "${PLOW_L2HIER_PF:-0}" = 1 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_L2_PLACE_DISPATCH=1"
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

# OPT-IN (PLOW_MLA_PF_NOPE=1): the DR=0 arm of the V2 MLA prefill -- DeepSeek-V4-Flash's
# geometry, where the rope strip lives inside the cached 512-wide row so QK and PV both run
# over the full 512 and there is no Krope cache. FLASH OBJECT ONLY, and opt-in for the same
# reason PLOW_DSA_PF is: the megakernel inlines every arm and its register allocation is the
# worst case over all of them, so two more full-column-wave bodies must not be forced on the
# GLM / V3 blobs that never emit a NoPE packet. Without it the NoPE bit still TRAPS, so a
# blob that needs the arm and an object that lacks it is a hard stop, not a wrong answer.
if [ "${PLOW_MLA_PF_NOPE:-0}" = 1 ]; then
  AX_FLASH="$AX_FLASH -DPLOW_MLA_PF_NOPE_ARM=1"
fi

# OPT-IN (PLOW_DSA_IDX64=1): the 64-index-head arm of the DSA prefill indexer score (op 117).
# GLM-5.3 has 32 index heads and DeepSeek-V4 has 64; the 32x32 MFMA A-tile's M axis IS the head
# axis, so 64 is a second head group and a second live query fragment set. PREFILL OBJECT ONLY,
# opt-in so the GLM blobs that never emit a 64-head packet do not pay the registers. Without it
# a 64-head packet TRAPS rather than being scored against half its heads.
if [ "${PLOW_DSA_IDX64:-0}" = 1 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_DSA_IDX64_ARM=1"
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

# Diagnostic-only XREDUCE2 / XREDUCE phase timeline in PlowTraceRec. Never a serve asset.
if [ "${PLOW_XR_TRACE_PHASES:-0}" = 1 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_XR_TRACE_PHASES=1"
  AX_DECODE="$AX_DECODE -DPLOW_XR_TRACE_PHASES=1"
fi

# DEFAULT (PLOW_XR_SCHED=aiter; rollback PLOW_XR_SCHED=twoshot, `off` is an alias): the 16-byte
# prefill collective schedule (op_collective.h PLOW_XR_SCHED_AITER) on the first
# PLOW_XR_SCHED_NWG (24) workgroups of each two-shot / op 25 / op 26 packet, the two-shot's
# reduce-scatter on the first PLOW_XR_SCHED_NWG_RS (8); the other workgroups only arrive. Objects
# only, prefill rows only (decode emits only the one-shot XReduce); the packet is unchanged.
# Bit-identical: same r = 0..7 f32 sum and bf16 round per element (strict-order oracle,
# tp_allreduce_prefill_bench TP_RANDOM=1). MEASURED, 8x MI300X: 8192x6144 two-shot 1002 -> 728 us
# isolated; 4-layer TP8 1.06 -> 0.92 ms per collective; served A/B, credited as a pair with
# PLOW_AMD_NUMA_HOST_POOLS: full 8192 chunks 823.4 / 793.9 / 824.4 ms (ctrl / treat / ctrl2,
# -29.9 ms per chunk against a 1.0 ms control gap), +3.66 % out tok/s (control spread 1.14 %),
# retrieval 18/18 + 21/21 in every arm.
# `twoshot` compiles none of it: the prefill rows are then the shipped 2-byte two-shot.
case "${PLOW_XR_SCHED:-aiter}" in
  twoshot|off) ;;
  aiter) AX_PREFILL="$AX_PREFILL -DPLOW_XR_SCHED_AITER=1 -DPLOW_XR_SCHED_NWG=${PLOW_XR_SCHED_NWG:-24} -DPLOW_XR_SCHED_NWG_RS=${PLOW_XR_SCHED_NWG_RS:-8} -DPLOW_XR_SCHED_AG_U=${PLOW_XR_SCHED_AG_U:-1}"
         # Seam caps (ops 25 / 26). The seam reduce-scatter runs on 8 workgroups by default, like the
         # two-shot's (micro5: 8192x6144 394.5 -> ~347 us, -7 ms per 8192 chunk; bit-identical);
         # rollback PLOW_XR_SCHED_NWG_SRS=24, the all-gather cap it used before. The seam all-gather
         # keeps PLOW_XR_SCHED_NWG unless PLOW_XR_SCHED_NWG_SAG is set (no cap below 24 pays).
         AX_PREFILL="$AX_PREFILL -DPLOW_XR_SCHED_NWG_SRS=${PLOW_XR_SCHED_NWG_SRS:-8}"
         [ -n "${PLOW_XR_SCHED_NWG_SAG:-}" ] && AX_PREFILL="$AX_PREFILL -DPLOW_XR_SCHED_NWG_SAG=$PLOW_XR_SCHED_NWG_SAG"
         ;;
  *) echo "FAIL: PLOW_XR_SCHED must be aiter or twoshot" >&2; exit 2 ;;
esac

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
# BIT-IDENTICAL (no value is touched) and objects-only: no blob or emitter change. Generic
# decode uses the one-shot, while batched K3 decode emits 186 two-shot collectives at B32 and
# enables the same mechanism through PLOW_K3_DECODE_XR_AGG below.
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

# CEILING INSTRUMENT ONLY (PLOW_GEMM_ABL=1): the DENSE GEMM with its k-loop capped at one tile
# (op_gemm.h PLOW_GEMM_ABL) -- the twin of the MoE ablation below, for the projections. WRONG
# OUTPUT, never a serve asset.
if [ "${PLOW_GEMM_ABL:-0}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_GEMM_ABL=${PLOW_GEMM_ABL}"
fi

# CEILING INSTRUMENT ONLY (PLOW_MOE_PF_ABL=1): the grouped MoE prefill GEMM with its k-loop
# capped at one tile (op_moe.h PLOW_MOE_PF_ABL). WRONG OUTPUT by construction, never a serve
# asset -- the same contract as PLOW_MLA_PF_ABL and PLOW_XR_NOWAIT above. It prices the k-loop
# against everything else the `interpreter` segment contains (router, gather/scatter, norms,
# collectives), which is the measurement that decides whether a faster GEMM can carry this
# target or whether the term is elsewhere.
if [ "${PLOW_MOE_PF_ABL:-0}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MOE_PF_ABL=${PLOW_MOE_PF_ABL}"
fi

# MPF_BM A/B escape hatch for the PREFILL objects (the decode row has carried its MPF_BK twin
# since the OCC4 recut). The grouped MoE prefill GEMM is the term that binds once attention is
# sparse -- 11.8 ms per layer per rank, 87.7 TF/s, 3.4% of this part's fp8 peak -- and TP8 is
# what makes its overhead PER-TILE rather than per-byte: `down`'s K is moe_intermediate/TP =
# 2048/8 = 256, so its k-loop runs FOUR iterations while gemm1's runs ninety-six, and it emits
# 24 n-tiles per m-tile against gemm1's 4. Raising BM halves the tile count and so halves the
# fixed cost that k-loop is too short to amortize. At 128 the arena still fits: the single
# buffer is (128+256)*64*2 = 49,152 B against `plow_smem`'s 64,512, and MPF_DBUF stays 1
# exactly as it already is at BM=64 (2*40,960 is already over budget). docs 7h.
#
# PLOW_MOE_PF_EPI must come OFF with it: that hoist puts one m-tile row per LANE of a single
# wave and reads it back with `ds_bpermute_b32`, so op_moe.h `#error`s unless
# MPF_BM == PLOW_WAVE. A BM=128 arm therefore has to be compared against a BM=64 arm built with
# EPI=0 too, or the measurement is the hoist and not the tile.
if [ -n "${MPF_BM:-}" ]; then
  AX_PREFILL="$AX_PREFILL -DMPF_BM=$MPF_BM"
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
# Batched K3 decode lowers routed experts to the grouped prefill opcodes (83-87). MXFP4 packets
# therefore need the A4W4 body in the decode object too; without it the deliberate refusal path
# writes NaNs. Keep B=1 byte-identical because it uses the per-expert decode opcodes instead.
AX_K3_DECODE_A4W4=""
case "${PLOW_K3_DECODE_TILE_BINSEARCH:-1}" in
  0) AX_K3_TILE_SEARCH="" ;;
  1) AX_K3_TILE_SEARCH="-DPLOW_MOE_TILE_BINSEARCH=1" ;;
  *) echo "FAIL: PLOW_K3_DECODE_TILE_BINSEARCH must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_K3_DECODE_ALIGN_PAR_PREFIX:-1}" in
  0) AX_K3_ALIGN_PREFIX="" ;;
  1) AX_K3_ALIGN_PREFIX="-DPLOW_MOE_ALIGN_PAR_PREFIX=1" ;;
  *) echo "FAIL: PLOW_K3_DECODE_ALIGN_PAR_PREFIX must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_K3_DECODE_ROUTER_LOCAL:-1}" in
  0) AX_K3_ROUTER_LOCAL="" ;;
  1) AX_K3_ROUTER_LOCAL="-DPLOW_MOE_ROUTER_SELECT_LOCAL=1" ;;
  *) echo "FAIL: PLOW_K3_DECODE_ROUTER_LOCAL must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_K3_DECODE_XR_AGG:-1}" in
  0) AX_K3_DECODE_XR_AGG="" ;;
  1) AX_K3_DECODE_XR_AGG="-DPLOW_XR_AGG=1" ;;
  *) echo "FAIL: PLOW_K3_DECODE_XR_AGG must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_K3_DECODE_GROUPED:-0}" in
  0|1) ;;
  *) echo "FAIL: PLOW_K3_DECODE_GROUPED must be 0 or 1" >&2; exit 2 ;;
esac
if [ "${PLOW_DECODE_BATCH:-1}" -gt 1 ] || [ "${PLOW_K3_DECODE_GROUPED:-0}" = 1 ]; then
  AX_K3_DECODE_A4W4="$AX_A4W4 $AX_K3_A4W4 $AX_K3_A4W4_TUNE $AX_K3_TILE_SEARCH $AX_K3_ALIGN_PREFIX $AX_K3_ROUTER_LOCAL"
fi
# Compile-only falsification axis: the grouped MXFP4 expert body is gated by
# PLOW_MOE_PF_A4W4, independently of the standalone MXFP4 projection ops. Keep those projection
# ops by default; `=0` removes them only from the two K3 decode rows.
AX_K3_DECODE_MXFP4="$AX_MXFP4"
if [ "${PLOW_K3_DECODE_MXFP4_PROJ:-1}" = 0 ]; then
  AX_K3_DECODE_MXFP4=""
fi
case "${PLOW_K3_MOE_GROUP_FORCEINLINE:-0}" in
  0) AX_K3_MOE_GROUP_INLINE="" ;;
  1) AX_K3_MOE_GROUP_INLINE="-DPLOW_MOE_GROUP_FORCEINLINE=1" ;;
  *) echo "FAIL: PLOW_K3_MOE_GROUP_FORCEINLINE must be 0 or 1" >&2; exit 2 ;;
esac
case "${PLOW_K3_KDA_CONV_STEP_DB:-0}" in
  0) AX_K3_KDA_CONV_STEP_DB="" ;;
  1) AX_K3_KDA_CONV_STEP_DB="-DPLOW_KDA_CONV_STEP_DB=1" ;;
  *) echo "FAIL: PLOW_K3_KDA_CONV_STEP_DB must be 0 or 1" >&2; exit 2 ;;
esac

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
  # THE DECODE ROW NEEDS IT TOO ONCE A DECODE BATCH IS COMPILED IN, for exactly the reason
  # $AX_MOE is added to $AX_DECODE above: at rows>1 the GLM decode program emits its MoE seam
  # with the grouped PREFILL family (ops 83-87), so op 86/87's deterministic arm is reached from
  # the DECODE object. Without this a blob emitted PLOW_MOE_PF_DET=1 is refused at load —
  # "requires PLOW_MOE_PF_DET=1 but the DECODE object was built WITHOUT it" — which is what a
  # GLM-5.3 TP4 serve did here on its first attempt.
  if [ "${PLOW_DECODE_BATCH:-1}" -gt 1 ]; then
    AX_DECODE="$AX_DECODE -DPLOW_MOE_PF_DET=${PLOW_MOE_PF_DET:-1}"
  fi
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
# THE FLASH OBJECT NEEDS IT TOO, and that is where GLM-5.3 actually runs the op. The note above
# says "PREFILL objects only" meaning "not decode" — the guard it cites is `n_batch == token
# count`, which is a property of the PACKET, not of the object file. On the GLM-5.3 sparse TP8
# recipe the 8192-row chunk's MlaMergeFold is dispatched from `interp_flash_fp8kv*` (the segment
# after the native AITER attention: fold, o_proj, the two-shot, residual, norm, router), so with
# the axis on AX_PREFILL alone the arm the measurement above bought was never reached by the
# shipped recipe: traced at 594 us/packet x 78 = 45.6 ms/chunk on the scalar arm.
# The arm's own guards still decide per packet (decode's n_batch=1 fails `n_work >= nblk`), so
# adding it here cannot change a decode dispatch — the flash object serves prefill buckets.
if [ "${PLOW_MLA_FOLD_TB:-8}" != 0 ]; then
  AX_PREFILL="$AX_PREFILL -DPLOW_MLA_FOLD_TB=${PLOW_MLA_FOLD_TB:-8}"
  # OPT-IN (PLOW_MLA_FOLD_TB_FLASH=1), default off: the same arm in the FLASH object. Default-off
  # and not default-on with AX_PREFILL because the header is explicit that "bit-identical" is the
  # INTENT and the gfx950 V=128 oracle rejected TB>1 (245-1,463 outputs differ, packed-FMA
  # schedule) — every new shape needs its own character-identical gate. This is a new shape only
  # in the sense of a different object; the map (V=256, TB=8, gfx942) is the one already gated on
  # AX_PREFILL. Flip the default only after the 18-case retrieval screen on this object.
  #
  # >>> MEASURED ON THE GLM-5.3 TP8 SPARSE CHUNK, AND IT IS A NULL. <<< (2026-09-11, packet trace
  # of the last chunk of a 73,728-token prompt, 8 GPUs, rank 0: MlaMergeFold body 46.59 ms/chunk
  # without the arm and 46.59 ms with it; chunk 963.65 -> 964.21 ms, inside run-to-run spread.)
  # The reachability half of the finding is real — segment 11 of the 8192 program is
  # `object_class: flash`, so the arm on AX_PREFILL alone never ran for this recipe — but the
  # PRIZE is not: at nsplit=1 (the sparse arm) the merge phase is a pass-through and the fold is
  # VALU-bound, 17.2 GFLOP in 597 us = 29 TFLOP/s against a ~163 TFLOP/s f32 VALU peak, while the
  # W_uv panel (8 heads x 256 KiB = 2 MiB) is L2-resident. TB=8 divides an L2 stream that was not
  # the constraint. The 1626 -> 616 us standalone number was ns=2 on a cold cache; in situ the
  # scalar arm already runs at 597 us. What is left is the arithmetic itself: the fold is a
  # batched GEMM and wants MFMA, not a better stream.
  [ "${PLOW_MLA_FOLD_TB_FLASH:-0}" = 0 ] ||
    AX_FLASH="$AX_FLASH -DPLOW_MLA_FOLD_TB=${PLOW_MLA_FOLD_TB:-8}"
fi

# DEFAULT ON for the gfx942 PREFILL and FLASH objects (2026-09-11): the three glue memory arms —
# PLOW_COMBINE_VEC (8-wide k==1 MoE combine, op_moe.h), PLOW_RN_ROWS (RMSNorm issues R rows of
# loads before reducing any, op_norm.h) and PLOW_RESID_U (residual keeps U iterations of loads in
# flight, op_elementwise.h). Rollback per axis: PLOW_COMBINE_VEC=0, PLOW_RN_ROWS=0, PLOW_RESID_U=0
# (1 is also the shipped single-row / non-unrolled body for the last two).
#   BIT-IDENTICAL, measured: the three device bodies built with each object's exact -D set (8-wave
#   prefill; 4-wave flash with PLOW_WAVE_RED_DPP), with and without the arms, 304 blocks, identical
#   inputs — 606,699,522 output bytes per build equal in both geometries across 12 cases, ragged
#   ones included (RMSNorm 1000x6144, residual 6,144,001 elements, combine T=1000, the k=8 path).
#   FASTER, measured on GLM-5.3 TP8 (packet trace of a steady 8192-row sparse chunk, 8 GPUs):
#   MoeCombinePf 31.7 -> 6.7 ms/chunk (0.74 -> 3.5 TB/s), RmsNorm 22.2 -> 20.4, Residual 12.2 ->
#   11.5, chunk 963.7 -> 937.9 ms (-2.7%); served 70k/C20 clean interleaved A/B 50.17 -> 51.15 out
#   tok/s, candidate ahead in both rounds, retrieval 18/18 both arms.
# Decode, mixed, token-batch, packed and small/split MLA objects keep the shipped bodies: their
# end-to-end effect was not measured (the decode megakernel sits at its register limit).
AX_GLUE=""
[ "${PLOW_COMBINE_VEC:-1}" = 0 ] || AX_GLUE="$AX_GLUE -DPLOW_COMBINE_VEC=${PLOW_COMBINE_VEC:-1}"
case "${PLOW_RN_ROWS:-2}" in 0|1) ;; *) AX_GLUE="$AX_GLUE -DPLOW_RN_ROWS=${PLOW_RN_ROWS:-2}" ;; esac
case "${PLOW_RESID_U:-4}" in 0|1) ;; *) AX_GLUE="$AX_GLUE -DPLOW_RESID_U=${PLOW_RESID_U:-4}" ;; esac
AX_PREFILL="$AX_PREFILL$AX_GLUE"
AX_FLASH="$AX_FLASH$AX_GLUE"

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

# DEFAULT OFF, AND IT MUST STAY OFF UNTIL A MODEL OWNER SAYS OTHERWISE (PLOW_GEMV_MFMA4=1):
# the BATCHED bf16 decode GEMV runs on the matrix core -- `gemv_rows_mfma4` (op_gemm.h), the
# 4x4x4 bf16 diagonal arrangement vLLM 0.28's `wvSplitK_hf_sml_` uses, with the shipped memory
# pattern unchanged (one wave owns one output row, 64 lanes read 1024 contiguous bytes of it
# through one buffer descriptor).
#
# THIS FLAG REDEFINES PLOW'S DECODE REFERENCE ARITHMETIC. `gemv_rows` reduces a column as a
# per-lane chain of `dot8(w, x, 0.0f)` -- four nested fma pairs -- then an xor-butterfly
# `wave_sum`. Every MFMA on gfx942 reduces >= 4 products in hardware order inside ONE
# instruction, so no MFMA arrangement reproduces that nesting; bit identity is impossible by
# construction and every decode golden in the tree moves. That is why this is a build flag and
# not a measured default: it is a model-owner decision, not a performance one.
#
# WHAT IT BUYS, and it is only at BATCH > 1 (gemma31_mfma_decode_bench, Gemma-4 31B BF16 TP1
# shapes, per-token ms over the T=4 instance counts): MM=2 19.007 -> 15.039, MM=4 27.832 ->
# 16.388 (1.70x), MM=8 61.599 -> 41.844. MM=1 is 1.05x and the arm REFUSES it (see the
# `MM >= 2` guard in d_gemv_t), so a concurrency-1 tier keeps the shipped bit-identical body
# and turning this on cannot perturb a batch-1 sequence.
if [ "${PLOW_GEMV_MFMA4:-0}" = 1 ]; then
  AX_DECODE="$AX_DECODE -DGV_MFMA4=1${PLOW_GEMV_MFMA4_UN_M4:+ -DGV_MFMA4_UN_M4=$PLOW_GEMV_MFMA4_UN_M4}${PLOW_GEMV_MFMA4_YT_M4:+ -DGV_MFMA4_YT_M4=$PLOW_GEMV_MFMA4_YT_M4}${PLOW_GEMV_MFMA4_UN_M2:+ -DGV_MFMA4_UN_M2=$PLOW_GEMV_MFMA4_UN_M2}${PLOW_GEMV_MFMA4_YT_M2:+ -DGV_MFMA4_YT_M2=$PLOW_GEMV_MFMA4_YT_M2}"
  AX_DECODE="$AX_DECODE${PLOW_GEMV_MFMA4_MAXK:+ -DGV_MFMA4_MAXK=$PLOW_GEMV_MFMA4_MAXK}"
fi

# CEILING INSTRUMENT ONLY (PLOW_MOE_DEC_ABL=1|2): the block-fp8 decode expert DOWN with its body
# deleted — 1 keeps the walk and the store and drops every load + the dot, 2 retires the op. WRONG
# OUTPUT by construction; this prices what the packet costs when the kernel costs nothing, and must
# never touch a serve asset.
if [ "${PLOW_MOE_DEC_ABL:-0}" != 0 ]; then
  AX_DECODE="$AX_DECODE -DPLOW_MOE_DEC_ABL=${PLOW_MOE_DEC_ABL}"
fi

# PACKED-PREFILL OPERATOR-FAMILY OBJECTS (PLOW_PACKED_PREFILL_CONSUMERS=1, default off).
#
# These are the three LEAN objects `PLOW_PACKED_PREFILL_ROUTE=1` opens by literal name
# (exec/amd.rs `load_packed_family`). gfx942 never built them: the gfx950 side gets them from
# runtime/CMakeLists.txt behind PLOW_HSACO_PACKED_PREFILL_CONSUMERS (also default OFF), and this
# script has no cmake path at all. The consequence was silent -- `load_packed_family` returns
# `Ok(None)` on a missing file -- so PLOW_PACKED_PREFILL_ROUTE=1 on MI300X loaded nothing and
# then refused every MLA co-pack at `check_packed_prefill_program`.
#
# LEAN IS THE POINT. Each object carries ONE operator family and nothing else:
#   interp_packed_mla_norm   class 5 -- RmsNorm / HeadNormRope(+Fp8), the 8-wave bucket.
#   interp_packed_mla_flash  class 6 -- FlashMlaPrefill(+Fp8), the 4-wave / 512-reg bucket.
#   interp_packed_kda        class 7 -- KDA serial (and chunk under PLOW_KDA_CHUNK=1).
# Compiling any packed call arm into the full interpreter raises its spill count (the cmake
# comment records 116 VGPR spills on the K3 row), which is why they are separate objects rather
# than defines on the rows above.
#
# NO $AX_GMOE and no $AX_MOE: a family object only ever runs its own class, so the Gemma-MoE and
# expert arms it cannot dispatch would only grow the register union. `check_moe_gemma_arms` is
# not a blanket check for these -- exec/amd.rs validates them through `load_packed_family`,
# which asks for the family markers instead.
AX_PACKED_MLA_NORM="-DPLOW_BUCKET_DECODE=0 $CDNA3_TILE $AX_MLA \
  -DPLOW_BUCKET_PACKED_MLA_NORM=1 -DPLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS=1"
AX_PACKED_MLA_FLASH="-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 \
  -DFA_DC=256 -DFA_DBUF=1 $CDNA3_TILE_4W -DPLOW_MLA_PF_V2_ARM=1 \
  -DPLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS=1"
AX_PACKED_KDA="-DPLOW_BUCKET_DECODE=0 $CDNA3_TILE -DPLOW_K3=1 $AX_KDA_CHUNK \
  -DPLOW_BUCKET_PACKED_KDA=1 -DPLOW_PACKED_PREFILL_KDA_CONSUMERS=1"
AX_MLA_SMALL="-DPLOW_BUCKET_DECODE=0 -DPLOW_WG_WAVES=8 $CDNA3_TILE $AX_MLA \
  -DPLOW_BUCKET_MLA_PREFILL_SMALL=1 -DPLOW_L2_PLACE_DISPATCH=1"

# THE TABLE: <stem>|<axes>. Names must match exec/amd.rs `object_name()`
# EXACTLY -- it composes stem + variant infix + arm infix + sched suffix and
# opens the result by literal filename.
ROWS=(
  "interp_prefill|$AX_PREFILL"
  "interp_decode|$AX_DECODE"
  "interp_flash|$AX_FLASH"
  "interp_mla_small|$AX_MLA_SMALL"
  "interp_mla_small_fp8kv|$AX_MLA_SMALL $AX_FP8KV"
  "interp_mla_split_fp8kv|-DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 $CDNA3_TILE_4W $AX_FP8KV -DPLOW_BUCKET_MLA_PREFILL_SPLIT=1 -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_L2_PLACE_DISPATCH=1"
  "interp_mixed|-DPLOW_MIXED_STEP=1 -DPLOW_BUCKET_DECODE=0 -DPLOW_WG_WAVES=4 -DPLOW_GEMV_MM=4 -DGM_BM=64 -DGM_BN=128 -DFA_DC=256 -DFA_DBUF=1"
  # UNIFIED TOKEN BATCH, dense GQA (plans/unified-token-batch.md §8 Phase 2). The mixed object's
  # exact shape plus PLOW_TOKEN_BATCH=1, which REPLACES its phase-band projections with one
  # matmul over the combined M and its per-span attention loop with a flat query-tile schedule.
  # A separate row, not a define on the one above: the two routes are different dispatch and
  # `interp_mixed` stays byte-for-byte the qualified object it is today.
  # GM_BM=256, NOT the mixed row's 64, and it is the single biggest thing measured about this
  # route. `mixed_program::synthesize` rewrites every GEMM opcode -- `GemmWide` and `GemmC5`
  # included -- onto plain `Gemm`, so this object's ONE compiled tile runs every dense
  # projection of a prefill chunk. At 64x128 that is the slowest rung in op_gemm.h's own
  # inventory: 332-458 TF/s against 192x256's 1033-1236 on exactly the Gemma-31B shapes
  # (see the tile-inventory table above GM_WD_BM). BN stays 128 because the fused-GLU
  # epilogue's `SN == 2` pins it there at four waves; BM is free, and 256x128 is 2x the
  # arithmetic intensity of 64x128 for NOTHING: 64/192/256 all measure 446 vgpr / 190 agpr /
  # 64,544 B LDS / 0 spill on this row, because the arena is already sized at `GM_C5_*` (see
  # PLOW_GM_ARENA under PLOW_MIXED_STEP) and the wave grid is 2x2 either way. The object was
  # paying for a 192x256 arena and running 64x128 inside it.
  # Serving delta on Gemma-4 31B at concurrency 8, token batch against the ordinary route:
  # 7168 input -18.4% -> -1.8%, 4096 -17.8% -> -4.5%, 2048 -12.4% -> -2.4% output tokens/s;
  # at concurrency 1 with `--amd-token-batch-solo` (no packing at all, so the prefill chunk
  # alone) 7168 goes -24.8% -> -6.5% and its TTFT +64.8% -> +13.0%.
  # `TB_GM_BM`/`TB_GM_BN` stay overridable so the A/B that found this is one env away.
  "interp_tokbatch|-DPLOW_TOKEN_BATCH=1 -DPLOW_MIXED_STEP=1 -DPLOW_BUCKET_DECODE=0 -DPLOW_WG_WAVES=4 -DPLOW_GEMV_MM=4 -DGM_BM=${TB_GM_BM:-256} -DGM_BN=${TB_GM_BN:-128} -DFA_DC=256 -DFA_DBUF=1"
  "interp_prefill_fp8|$AX_PREFILL $AX_FP8"
  "interp_decode_fp8|$AX_DECODE $AX_FP8"
  "interp_prefill_fp8kv|$AX_PREFILL $AX_FP8 $AX_FP8KV"
  "interp_decode_fp8kv|$AX_DECODE $AX_FP8 $AX_FP8KV"
  "interp_flash_fp8kv|$AX_FLASH $AX_FP8KV -DPLOW_K3=1"
  "interp_prefill_mla|$AX_PREFILL $AX_MLA"
  "interp_prefill_mla_moe|$AX_PREFILL $AX_MLA $AX_MOE"
  "interp_prefill_fp8_mla|$AX_PREFILL $AX_MLA $AX_FP8"
  "interp_prefill_fp8_mla_moe|$AX_PREFILL $AX_MLA $AX_MOE $AX_FP8"
  "interp_prefill_fp8kv_mla|$AX_PREFILL $AX_MLA $AX_FP8 $AX_FP8KV"
  "interp_prefill_fp8kv_mla_moe|$AX_PREFILL $AX_MLA $AX_MOE $AX_FP8 $AX_FP8KV"
  # KIMI-K3. `interp_decode_k3` is the row a K3 decode packet actually loads (exec/amd.rs folds
  # K3Moe and K3MoeA4w4 onto PrefillArm::K3 for the decode phase), and it carries the mxfp4
  # EXPERT walks by default. `$AX_K3_DECODE_MXFP4` rides along so an all-fp4 packet finds its fp4
  # PROJECTION ops in the same object rather than falling through the silent dispatch `default:`.
  "interp_decode_k3|$AX_DECODE $AX_K3 $AX_K3_DECODE_A4W4 $AX_K3_DECODE_MXFP4 $AX_K3_DECODE_XR_AGG $AX_K3_MOE_GROUP_INLINE $AX_K3_KDA_CONV_STEP_DB"
  "interp_decode_fp8kv_k3|$AX_DECODE $AX_K3 $AX_K3_DECODE_A4W4 $AX_K3_DECODE_MXFP4 $AX_FP8KV $AX_K3_DECODE_XR_AGG $AX_K3_MOE_GROUP_INLINE $AX_K3_KDA_CONV_STEP_DB"
  # ATTENTION-ONLY, exactly as on gfx950: without $AX_MOE the grouped expert packets fall through
  # `default:` and write nothing. A whole-layer K3 prompt needs the `_moe` rows below.
  "interp_prefill_k3|$AX_PREFILL $AX_MLA_K3 $AX_MXFP4"
  "interp_prefill_fp8kv_k3|$AX_PREFILL $AX_MLA_K3 $AX_MXFP4 $AX_FP8KV"
  "interp_prefill_k3_moe|$AX_PREFILL $AX_MLA_K3 $AX_MOE $AX_MXFP4"
  # THE ROW A K3 PREFILL PACKET ACTUALLY LOADS (exec/amd.rs: grouped ops with i[3] == MXFP4
  # resolve the K3MoeA4w4 arm). On this arch it contains the simulated body -- see AX_A4W4.
  "interp_prefill_k3_moe_a4w4|$AX_PREFILL $AX_MLA_K3 $AX_MOE $AX_A4W4 $AX_K3_A4W4 $AX_K3_A4W4_TUNE $AX_K3_PF_STATE $AX_MXFP4"
  "interp_prefill_fp8kv_k3_moe_a4w4|$AX_PREFILL $AX_MLA_K3 $AX_MOE $AX_A4W4 $AX_K3_A4W4 $AX_K3_A4W4_TUNE $AX_K3_PF_STATE $AX_MXFP4 $AX_FP8KV"
)
# The `_fp8kv` twins serve FP8-KV packets (GLM-5.3 TP8 ships `--glm-fp8-kv=true`); exec/amd.rs
# opens `interp_packed_mla_{norm,flash}_fp8kv` for those and refuses the bf16 object by name.
if [ "${PLOW_PACKED_PREFILL_CONSUMERS:-0}" = 1 ]; then
  ROWS+=(
    "interp_packed_mla_norm|$AX_PACKED_MLA_NORM"
    "interp_packed_mla_flash|$AX_PACKED_MLA_FLASH"
    "interp_packed_mla_norm_fp8kv|$AX_PACKED_MLA_NORM $AX_FP8KV"
    "interp_packed_mla_flash_fp8kv|$AX_PACKED_MLA_FLASH $AX_FP8KV"
    "interp_packed_kda|$AX_PACKED_KDA"
  )
fi
# TOKEN-BATCH BODY FAMILY OBJECTS (PLOW_TOKEN_BATCH_TP_OBJECTS=1, default off): the packed MLA
# norm/flash objects with the slot-band resolver compiled in (`PLOW_PACKED_PREFILL_BAND=1`,
# marker `plow_packed_prefill_band_1`). Separate rows rather than a define on the rows above so
# the shipped packed objects stay byte-identical; `exec/amd.rs` opens them by the `_tb` stem for
# token-batch body programs only. The `_fp8kv` twins serve FP8-KV packets.
if [ "${PLOW_TOKEN_BATCH_TP_OBJECTS:-0}" = 1 ]; then
  ROWS+=(
    "interp_packed_mla_norm_tb|$AX_PACKED_MLA_NORM -DPLOW_PACKED_PREFILL_BAND=1"
    "interp_packed_mla_flash_tb|$AX_PACKED_MLA_FLASH -DPLOW_PACKED_PREFILL_BAND=1"
    "interp_packed_mla_norm_tb_fp8kv|$AX_PACKED_MLA_NORM $AX_FP8KV -DPLOW_PACKED_PREFILL_BAND=1"
    "interp_packed_mla_flash_tb_fp8kv|$AX_PACKED_MLA_FLASH $AX_FP8KV -DPLOW_PACKED_PREFILL_BAND=1"
  )
fi

# PLOW_ROWS_ONLY=<substring> (or =<exact-stem>): build only matching rows — for iterating on
# ONE object family (e.g. interp_flash) without paying the full 28-object build. The
# resulting dir is PARTIAL; copy it over a full set before serving from it.
if [ -n "${PLOW_ROWS_ONLY:-}" ]; then
  FILTERED=()
  for row in "${ROWS[@]}"; do
    # COMMA-SEPARATED: a list of filters, each `=exact-stem` or a substring, matching if ANY
    # does. Extension mode (PLOW_HSACO_EXTENSION) builds a set of named rows, and one substring
    # cannot name a set.
    for filt in ${PLOW_ROWS_ONLY//,/ }; do
      if [[ "$filt" == =* ]]; then
        if [ "${row%%|*}" = "${filt#=}" ]; then FILTERED+=("$row"); break; fi
      else
        case "${row%%|*}" in *"${filt}"*) FILTERED+=("$row"); break;; esac
      fi
    done
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
        $* $AX_CONFIG $AX_EXTRA --genco "$R/amd/interp.hip" -o "$stem.co" $INC > "$stem.log" 2>&1; then
    echo "FAIL  $stem"; tail -20 "$stem.log"; return 1
  fi
  "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
      --input="$stem.co" --output="$stem.elf"
  rm -f "$stem.co" "$stem.log"
  echo "ok    $stem"
}
export -f one; export HIPCC ARCH R INC BUN AX_CONFIG AX_EXTRA

# test_kernels.elf is STARTED HERE, alongside the row batch, and waited on after it.
# It shares no input with the rows and nothing between here and the wait consumes it, so
# the only thing its old position bought was serialisation: measured 29.7 s of a 79.4 s
# build at JOBS=48, against a 30.1 s critical path for the whole parallel batch. See the
# body below the wait for what it is and why it must be rebuilt with the interpreter.
if [ -z "${PLOW_ROWS_ONLY:-}" ]; then
  (
    if "$HIPCC" --offload-arch="$ARCH" -O3 -w --genco "$R/amd/test_kernels.hip" \
          -o tk.co $INC > test_kernels.log 2>&1; then
      "$BUN" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
          --input=tk.co --output=test_kernels.elf
      rm -f tk.co test_kernels.log
    else
      exit 1
    fi
  ) &
  TK_PID=$!
fi

# Both scheduler twins: which one a packet needs is decided by the packet
# (gq_seg_ofs), not by this build, and plowrt opens the twin by literal name.
# PLOW_DEFINES_ONLY=1 resolves every axis and writes build_defines.json without
# compiling anything. `scripts/check_recipe.py` uses it to verify that a recipe's
# `[objects].env` produces the `-D` set it claims — seconds instead of the ~25
# minutes a full build takes. The defines block below is the SAME code either
# way, so what this mode reports is what a real build would compile.
if [ "${PLOW_DEFINES_ONLY:-0}" = 0 ]; then
printf '%s\n' "${ROWS[@]}" | while IFS='|' read -r stem axes; do
  echo "$stem|$axes"
  case "$stem" in
    interp_decode*) echo "${stem}_gq|$axes $AX_GQ $AX_DECODE_GQ" ;;
    *) echo "${stem}_gq|$axes $AX_GQ" ;;
  esac
done | xargs -P "$JOBS" -I{} bash -c 'IFS="|" read -r s a <<< "{}"; one "$s" $a'
fi

# THE `-D` SET EACH OBJECT WAS ACTUALLY COMPILED WITH, written next to the objects.
#
# `asm_audit.py --contract` reads this and asserts, for every geometry macro in it, that the
# object carries a `plow_geom_<MACRO>` marker holding that value. That is the check for the
# defect this file's GM_AX hatch had: five per-rung GEMM knobs were bare `#define`s in
# op_gemm.h, a bare #define after a command-line -D is a redefinition the header wins, and -w
# hides the warning -- so every A/B ever run through GM_AX measured an unchanged object.
#
# Composed from the same $ROWS and the same $AX_GQ branch the compile loop uses, one line up,
# rather than from a second table that would drift from it. `$(echo $a)` re-splits on
# whitespace to fold the backslash-continued axis strings onto one JSON line.
{
  printf '{\n'
  sep=""
  for row in "${ROWS[@]}"; do
    stem="${row%%|*}"; axes="${row#*|}"
    case "$stem" in
      interp_decode*) gq_axes="$axes $AX_GQ $AX_DECODE_GQ" ;;
      *) gq_axes="$axes $AX_GQ" ;;
    esac
    for pair in "$stem|$axes" "${stem}_gq|$gq_axes"; do
      printf '%s "%s": "-DPLOW_ARCH_SUFFIX=%s%s %s%s"' "$sep" "${pair%%|*}" "$ARCH" "$AX_CONFIG_JSON" "$(echo ${pair#*|})" "${AX_EXTRA:+ $AX_EXTRA}"
      sep=$',\n'
    done
  done
  # test_kernels.elf takes no axes at all; it is listed so the contract audits it as an
  # object rather than skipping it for want of a defines entry.
  printf '%s "test_kernels": "-DPLOW_ARCH_SUFFIX=%s"' "$sep" "$ARCH"
  printf '\n}\n'
} > build_defines.json

if [ "${PLOW_DEFINES_ONLY:-0}" != 0 ]; then
  echo ">>> $OUT/build_defines.json written (PLOW_DEFINES_ONLY: nothing compiled)"
  exit 0
fi

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
# Started above, alongside the row batch. `wait` on a specific PID returns that job's
# exit status, so a test_kernels failure still fails the build here rather than earlier.
if [ -n "${TK_PID:-}" ]; then
  if wait "$TK_PID"; then
    echo "ok    test_kernels"
  else
    echo "FAIL  test_kernels"; [ -f test_kernels.log ] && tail -20 test_kernels.log; exit 1
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
    symbols=$("$READELF" -sW "$stem.elf" 2>/dev/null)
    grep -qE "OBJECT .* plow_packed_prefill_abi_1$" <<<"$symbols" || {
      echo "  MISSING PACKED-PREFILL ABI: expected plow_packed_prefill_abi_1"
      fail=1
    }
    # A config build's whole point: every object names the packet it was built for.
    if [ -n "$CFG" ]; then
      for m in plow_packet_hash_lo plow_packet_hash_hi; do
        grep -qE "OBJECT .* $m\$" <<<"$symbols" || {
          echo "  MISSING PACKET STAMP: expected $m (PLOW_HSACO_CONFIG=$CFG)"
          fail=1
        }
      done
    fi
    case "$stem" in
      # The family objects EXIST to carry these markers, and exec/amd.rs refuses one that does
      # not advertise them. Assert them positively here instead: a family object that silently
      # lost its marker is loaded as `Ok(None)` and the route degrades to nothing.
      interp_packed_mla_norm*)
        grep -qE "OBJECT .* plow_packed_prefill_mla_norm_segments_1$" <<<"$symbols" || {
          echo "  MISSING PACKED MLA-NORM SEGMENTS: expected plow_packed_prefill_mla_norm_segments_1"
          fail=1
        }
        if [[ "$stem" == *_tb* ]]; then
          grep -qE "OBJECT .* plow_packed_prefill_band_1$" <<<"$symbols" || {
            echo "  MISSING SLOT-BAND RESOLVER: expected plow_packed_prefill_band_1"
            fail=1
          }
        fi ;;
      interp_packed_mla_flash*)
        grep -qE "OBJECT .* plow_packed_prefill_mla_flash_segments_1$" <<<"$symbols" || {
          echo "  MISSING PACKED MLA-FLASH SEGMENTS: expected plow_packed_prefill_mla_flash_segments_1"
          fail=1
        }
        if [[ "$stem" == *_tb* ]]; then
          grep -qE "OBJECT .* plow_packed_prefill_band_1$" <<<"$symbols" || {
            echo "  MISSING SLOT-BAND RESOLVER: expected plow_packed_prefill_band_1"
            fail=1
          }
        fi ;;
      interp_packed_kda*)
        for m in plow_kda_family_segments_1 plow_packed_prefill_kda_serial_segments_1 \
                 plow_packed_prefill_kda_consumers_1; do
          grep -qE "OBJECT .* $m\$" <<<"$symbols" || {
            echo "  MISSING PACKED KDA MARKER: expected $m"
            fail=1
          }
        done ;;
      *)
        if grep -qE "OBJECT .* plow_packed_prefill_(mla|kda)_consumers_1$" <<<"$symbols"; then
          echo "  UNEXPECTED PACKED-PREFILL CONSUMERS: default objects must remain resource-clean"
          fail=1
        fi ;;
    esac
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
      interp_flash*|interp_mla_split*|interp_mixed*|interp_tokbatch*|interp_packed_mla_flash*) [ "$v" -le 512 ] || { echo "  OVER REG: $v > 512"; fail=1; } ;;
      *)             [ "$v" -le 256 ] || { echo "  OVER REG: $v > 256"; fail=1; } ;;
    esac
    case "$stem" in
      interp_mixed*)
        grep -qE "OBJECT .* plow_mixed_step_bf16_1$" <<<"$symbols" || {
          echo "  MISSING MIXED BF16 CONSUMERS: expected plow_mixed_step_bf16_1"
          fail=1
        }
        grep -qE "OBJECT .* plow_mixed_block$" <<<"$symbols" || {
          echo "  MISSING MIXED BLOCK CONTRACT: expected plow_mixed_block"
          fail=1
        }
        grep -qE "OBJECT .* plow_mixed_gemm_glu_1$" <<<"$symbols" || {
          echo "  MISSING MIXED GLU: expected plow_mixed_gemm_glu_1"
          fail=1
        }
        grep -qE "OBJECT .* plow_mixed_prefill_split_1$" <<<"$symbols" || {
          echo "  MISSING MIXED PREFILL MERGE: expected plow_mixed_prefill_split_1"
          fail=1
        }
        grep -qE "OBJECT .* plow_mixed_dynamic_rows_1$" <<<"$symbols" || {
          echo "  MISSING MIXED DYNAMIC ROWS: expected plow_mixed_dynamic_rows_1"
          fail=1
        }
        grep -qE "OBJECT .* plow_mixed_glu_lds_halves$" <<<"$symbols" || {
          echo "  MISSING MIXED GLU LDS CAPACITY: expected plow_mixed_glu_lds_halves"
          fail=1
        }
        grep -qE "OBJECT .* plow_gemv_mm_cap_4$" <<<"$symbols" || {
          echo "  MISSING MIXED SMALL-M CAPACITY: expected plow_gemv_mm_cap_4"
          fail=1
        }
        ;;
      interp_tokbatch*)
        # The four capability markers exec/amd.rs reads out of .symtab before the object reaches
        # a device. AMD's dispatch `default:` writes nothing and does not trap, so an object
        # missing an arm a packet needs is a silent wrong answer; a name costs nothing here and
        # a device round trip everywhere else.
        for m in plow_token_batch_1 plow_token_batch_dense_gqa_1 \
                 plow_token_batch_combined_m_1 plow_token_batch_span_attn_1; do
          grep -qE "OBJECT .* $m\$" <<<"$symbols" || {
            echo "  MISSING TOKEN-BATCH MARKER: expected $m"
            fail=1
          }
        done
        # The route is built ON the mixed object, so it must still answer for that contract.
        grep -qE "OBJECT .* plow_mixed_step_bf16_1$" <<<"$symbols" || {
          echo "  MISSING MIXED BF16 CONSUMERS on the token-batch row"
          fail=1
        }
        ;;
      interp_prefill_k3*|interp_prefill_fp8kv_k3*)
        if [ -n "$AX_KDA_CHUNK" ]; then
          grep -qE "OBJECT .* plow_kda_chunk_bt64_arm_1$" <<<"$symbols" || {
            echo "  MISSING CHUNK-KDA: expected plow_kda_chunk_bt64_arm_1"
            fail=1
          }
        elif grep -qE "OBJECT .* plow_kda_chunk_bt64_arm_1$" <<<"$symbols"; then
          echo "  UNEXPECTED CHUNK-KDA: PLOW_KDA_CHUNK=0"
          fail=1
        fi
        ;;
      interp_flash*)
        grep -qE "OBJECT .* plow_mla_pf_v2_arm_1$" <<<"$symbols" || {
          echo "  MISSING MLA V2: expected plow_mla_pf_v2_arm_1"
          fail=1
        }
        grep -qE "OBJECT .* plow_l2_place_dispatch_1$" <<<"$symbols" || {
          echo "  MISSING L2 DISPATCH: expected plow_l2_place_dispatch_1"
          fail=1
        }
        if [[ "$stem" == interp_flash_fp8kv* ]]; then
          grep -qE "OBJECT .* plow_mla_pf_v2_fp8_arm_1$" <<<"$symbols" || {
            echo "  MISSING FP8 MLA V2: expected plow_mla_pf_v2_fp8_arm_1"
            fail=1
          }
        fi
        ;;
      interp_decode*)
        grep -qE "OBJECT .* plow_gemv_mm_cap_${GVMM}$" <<<"$symbols" || {
          echo "  WRONG GEMV CAP: expected plow_gemv_mm_cap_${GVMM}"
          fail=1
        }
        if [ "$WALK" = 1 ]; then
          grep -qE "OBJECT .* plow_gemv_walk_1$" <<<"$symbols" || {
            echo "  MISSING GEMV WALK: expected plow_gemv_walk_1"
            fail=1
          }
        elif grep -qE "OBJECT .* plow_gemv_walk_1$" <<<"$symbols"; then
          echo "  UNEXPECTED GEMV WALK: PLOW_GEMV_WALK=0"
          fail=1
        fi
        case "$stem" in
          interp_decode_k3*|interp_decode_fp8kv_k3*)
            if [ -n "$AX_K3_DECODE_XR_AGG" ]; then
              grep -qE "OBJECT .* plow_xr_agg_1$" <<<"$symbols" || {
                echo "  MISSING XR AGG: expected plow_xr_agg_1"
                fail=1
              }
            elif grep -qE "OBJECT .* plow_xr_agg_1$" <<<"$symbols"; then
              echo "  UNEXPECTED XR AGG: PLOW_K3_DECODE_XR_AGG=0"
              fail=1
            fi
            if [ -n "$AX_K3_KDA_CONV_STEP_DB" ]; then
              grep -qE "OBJECT .* plow_kda_conv_step_db_arm$" <<<"$symbols" || {
                echo "  MISSING KDA CONV-STEP DB: expected plow_kda_conv_step_db_arm"
                fail=1
              }
            elif grep -qE "OBJECT .* plow_kda_conv_step_db_arm$" <<<"$symbols"; then
              echo "  UNEXPECTED KDA CONV-STEP DB: PLOW_K3_KDA_CONV_STEP_DB=0"
              fail=1
            fi
            ;;
        esac
        ;;
    esac
  done
done
# INSTRUCTION-SELECTION gate, the CDNA3 twin of the one in build_gfx950.sh. The cliff table above
# catches a kernel that will not launch; it does not catch one that launches, is numerically
# correct, and quietly runs on the wrong matrix instruction. The gfx950 expectations cannot be
# reused -- CDNA3's bf16 MFMA is v_mfma_f32_32x32x8_bf16 (half the K) and its only fp8 MFMA is
# the one that file FORBIDS -- so the contract has its own file, and asm_audit.py refuses the
# cross-arch pairing by reading each object's ELF header. Skipped when the file is absent.
#
# THE STATIC PERFORMANCE CONTRACT rides in the SAME invocation, sharing the one disassembly
# pass -- it is the object-side twin of crates/devgen/src/dispatch_audit.rs and it costs about
# as much as the audit above, so running it separately would double a step that is already 20%
# of this build. Six checks: the geometry `-D` markers (a `-D` that did not take), spill counted
# from the ISA rather than from `.vgpr_spill_count`, LDS-crossbar and 2-byte-LDS density
# (reported, not failed), head-dim arm coverage, and the resource budget that used to be the
# stale comment at the top of this file. PLOW_AUDIT_STRICT=1 promotes the two reports to
# refusals, as it does on the emit side. NOT skipped under PLOW_ROWS_ONLY: the rules that need
# no baseline still apply to a partial build.
EXPECT="$REPO/scripts/asm_expect_gfx942.json"
BASELINE="$REPO/scripts/obj_baseline_gfx942.json"
if [ -f "$EXPECT" ] && command -v python3 >/dev/null; then
  echo ""
  echo "   --- instruction-selection audit + static performance contract ---"
  # Captured rather than piped: `cmd | tail` reports tail's status, so piping would swallow the
  # audit's exit code and the gate would print FAIL lines and then say the build is ready.
  # --quiet drops the per-kernel instruction dump; the contract table below replaces it.
  audit=$(python3 "$REPO/scripts/asm_audit.py" --quiet --expect "$EXPECT" \
      ${BASELINE:+--contract "$BASELINE"} --defines build_defines.json ./*.elf) || fail=1
  echo "$audit"
fi

# B>1 K3 decode dispatches grouped ops 85/86 with MXFP4 encoding. Check the
# capability marker before the body: the generic grouped arm also contains
# bf16 MFMA, so disassembly alone cannot prove that the enc=2 arm exists.
if { [ "${PLOW_DECODE_BATCH:-1}" -gt 1 ] || [ "${PLOW_K3_DECODE_GROUPED:-0}" = 1 ]; } && command -v python3 >/dev/null; then
  K3_BATCH_EXPECT="$REPO/scripts/asm_expect_gfx942_k3_batched.json"
  k3_batch_objects=()
  for object in \
      interp_decode_k3.elf interp_decode_k3_gq.elf \
      interp_decode_fp8kv_k3.elf interp_decode_fp8kv_k3_gq.elf; do
    [ -f "$object" ] || continue
    k3_batch_objects+=("$object")
    symbols=$("$READELF" -sW "$object" 2>/dev/null)
    grep -qE "OBJECT .* plow_moe_pf_a4w4_arm$" <<<"$symbols" || {
      echo "FAIL  $object: missing plow_moe_pf_a4w4_arm for PLOW_DECODE_BATCH=${PLOW_DECODE_BATCH}"
      fail=1
    }
  done
  if [ "${PLOW_DECODE_BATCH:-1}" -gt 1 ] && [ "${#k3_batch_objects[@]}" -gt 0 ] && [ -f "$K3_BATCH_EXPECT" ]; then
    echo ""
    echo "   --- K3 batched A4W4 instruction-selection audit ---"
    batch_audit=$(python3 "$REPO/scripts/asm_audit.py" --expect "$K3_BATCH_EXPECT" \
        "${k3_batch_objects[@]}") || fail=1
    echo "$batch_audit" | tail -20
  fi
fi

echo ""
[ "$fail" = 0 ] || { echo "!!! one or more rows are over the cliff or missing"; exit 1; }

# Compile narrow widths separately to avoid paying the widest GEMV's dead-row arithmetic.
# An explicit empty PLOW_DECODE_TIERS keeps a single-object build for comparisons.
if [ "${PLOW_DECODE_TIERS+x}" != x ] && [ "$TIER" = 0 ]; then
  PLOW_DECODE_TIERS=""
  case "${PLOW_ROWS_ONLY:-}" in
    ''|*interp_decode*)
      for w in 1 2 4 8; do
        if [ "$w" -lt "$GVMM" ]; then PLOW_DECODE_TIERS+="${PLOW_DECODE_TIERS:+,}$w"; fi
      done ;;
  esac
fi
if [ -n "${PLOW_DECODE_TIERS:-}" ] && [ "$TIER" = 0 ]; then
  tier_rows=interp_decode
  case "${PLOW_ROWS_ONLY:-}" in *interp_decode*) tier_rows="$PLOW_ROWS_ONLY" ;; esac
  for w in ${PLOW_DECODE_TIERS//,/ }; do
    case "$w" in ''|*[!0-9]*) echo "PLOW_DECODE_TIERS must be a comma list of widths" >&2; exit 1;; esac
    if [ "$w" -ge "$RAW_BATCH" ]; then continue; fi  # the wide object already serves this rung
    echo ""
    echo ">>> low-rung decode tier: width $w -> $OUT/lowrung$w"
    ( unset PLOW_DECODE_TIERS PLOW_GEMV_MM
      [ -z "$CFG" ] || export PLOW_HSACO_CONFIG="$CFG"
      PLOW_DECODE_TIER="$w" PLOW_ROWS_ONLY="$tier_rows" \
        "$REPO/scripts/build_gfx942.sh" "$OUT/lowrung$w" ) || exit 1
  done
fi

echo ">>> $OUT ready ($(ls "$OUT"/*.elf | wc -l) objects)"
