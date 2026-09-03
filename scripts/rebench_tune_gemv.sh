#!/usr/bin/env bash
# Run a prebuilt GEMV harness over the supported subset of a live TUNEDUMP census.
# Compilation belongs to gemv_campaign_lease.sh and must happen before its one GPU lease.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="${1:?object dir}"
RAW="${2:?raw output jsonl}"
CENSUS="${3:?TUNEDUMP census log}"
FILTER="${PLOW_GEMV_CENSUS_FILTER:-}"
SWEEP="$OBJ/gemv_row_sweep"

[[ -x "$SWEEP" ]] || { echo "FAIL: prebuilt sweep missing: $SWEEP" >&2; exit 1; }
[[ -f "$OBJ/test_kernels.elf" ]] || { echo "FAIL: current test_kernels.elf missing: $OBJ" >&2; exit 1; }
[[ ! -e "$RAW" ]] || { echo "FAIL: refusing to overwrite raw samples: $RAW" >&2; exit 1; }

cd "$OBJ"
while IFS=$'\t' read -r n k label arms; do
  echo "=== $label N=$n K=$k"
  PLOW_GEMV_JSONL="$RAW" PLOW_GEMV_ARMS="$arms" ./gemv_row_sweep "$n" "$k" "$label"
done < <(python3 "$ROOT/scripts/gemv_campaign_census.py" plan \
  --census "$CENSUS" --filter "$FILTER" --obj-mm "${PLOW_GEMV_OBJ_MM:-16}")

[[ -s "$RAW" ]] || { echo "FAIL: sweep produced no samples" >&2; exit 1; }
