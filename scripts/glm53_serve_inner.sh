#!/usr/bin/env bash
# The INSIDE-nix half of `glm53_mi300x.sh serve`. Split out so no quoting has to survive
# gpulease -> nix develop -> bash -c.
#
# WHY INSIDE NIX AT ALL: libhsa-runtime64.so.1 lives in the flake's ROCm 7.14 tree, which is
# also the toolchain build_gfx942.sh compiled the objects with. Outside it the dlopen fails and
# plowrt falls back to the CPU REFERENCE interpreter — which serves coherent answers at
# fictional speed. That happened on the first attempt here.
set -uo pipefail
ASSETS="${1:?assets}"; PORT="${2:?port}"; OBJ="${3:?objdir}"; BIN="${4:?bindir}"
export LD_LIBRARY_PATH="${ROCM_PATH:?nix develop did not set ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PLOW_HSACO="$OBJ"
export PLOW_MLA_PF_V2=1
export PLOW_L2_PLACE_DISPATCH=1
# BATCH-WIDTH-MATCHED DECODE TIERS, picked up automatically. `scripts/build_gfx942.sh` with
# PLOW_DECODE_TIERS=1,2 writes `$OBJ/lowrung1` and `$OBJ/lowrung2`: the same decode rows compiled
# for those widths. `PLOW_GEMV_MM` is a compiled CEILING, so without them a rung-1 packet runs the
# wide object's body and computes -- then discards -- one dot product per dead activation row.
# Measured on the Gemma-4 31B BF16 blob, three repeats of the six-cell packed-serve bench, all 45
# completions character-identical: concurrency-1 throughput +38.5 / +33.5 / +22.4% at 128 / 1024 /
# 4096 input tokens, TPOT -29.1 / -28.2 / -28.1%, concurrency 4 within noise
# (docs/amd/gemma4-31b-mi300x.md, "Batch-width-matched decode objects").
# An explicit PLOW_HSACO_LOWRUNG still wins: this only fills it in when it is unset.
if [ -z "${PLOW_HSACO_LOWRUNG:-}" ]; then
  tiers=""
  for w in 1 2 4 8; do
    [ -f "$OBJ/lowrung$w/interp_decode_gq.elf" ] || continue
    tiers="${tiers:+$tiers,}$OBJ/lowrung$w:$w"
  done
  [ -n "$tiers" ] && export PLOW_HSACO_LOWRUNG="$tiers"
fi
echo "serve tiers: PLOW_HSACO=$PLOW_HSACO PLOW_HSACO_LOWRUNG=${PLOW_HSACO_LOWRUNG:-<none>}"
exec "$BIN/plowrt" serve --assets "$ASSETS" --port "$PORT"
