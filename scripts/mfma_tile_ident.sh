#!/usr/bin/env bash
# Is the MFMA arm's arithmetic INDEPENDENT of its (UN, YT) tile? It has to be: neither reassociates
# a row -- they move only which wave takes which column and how many loads are in flight -- so two
# tiles must agree BIT FOR BIT, not merely closely. This dumps one arm's logits for three prompts
# so scripts/mfma_numerics_report.py can compare them against another arm's existing dumps.
#
#   GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 gemma31-mfma-tile-ident \
#     nix develop /app/plow --command scripts/mfma_tile_ident.sh <objdir> <outdir/ARM>
set -uo pipefail
OBJ="${1:?objdir}"
OUT="${2:?outdir}"
ASSETS="${PLOW_ASSETS:-/app/plow/build-gemma31/assets-ctx131072-chunk8192}"
BIN="${PLOW_BIN_DIR:-/app/plow/target-glm53/release}"
PROMPTS="${PROMPTS_DIR:-/tmp/mfma_prompts}"
export LD_LIBRARY_PATH="${ROCM_PATH:?nix develop did not set ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
for tag in n128_p0 n1024_p0 n4096_p0; do
  n="${tag#n}"; n="${n%_*}"; p="${tag##*_p}"
  "$BIN/plowrt" amd-bench --blob "$ASSETS/model.pkt" --hsaco "$OBJ" \
    --checkpoint "$ASSETS/checkpoint" --steps 64 --dump-logits "$OUT/$tag" \
    --prompt "$(cat "$PROMPTS/p${p}_${n}.ids")" > "$OUT/$tag.log" 2>&1 || {
      echo "FAIL $tag"; tail -5 "$OUT/$tag.log"; exit 1; }
done
echo "=== dumps complete: $OUT"
