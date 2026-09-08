#!/usr/bin/env bash
# Kernel-level batch-4 TPOT for one or more object sets, through `plowrt amd-bench --batched`.
# No server, no scheduler: this is the megakernel's own per-dispatch time, which is what the
# tile probe needs (the served A/B costs half an hour per arm and answers the same question
# about the decode step more slowly).
#
#   GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 gemma31-mfma-tile \
#     nix develop /app/plow --command scripts/mfma_tile_bench.sh <objdir>...
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS="${PLOW_ASSETS:-/app/plow/build-gemma31/assets-ctx131072-chunk8192}"
BIN="${PLOW_BIN_DIR:-/app/plow/target-glm53/release}"
PROMPTS="${PROMPTS_DIR:-/tmp/mfma_prompts}"
REPS="${REPS:-2}"
export LD_LIBRARY_PATH="${ROCM_PATH:?nix develop did not set ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

a="$(cat "$PROMPTS/p0_1024.ids")"; b="$(cat "$PROMPTS/p1_128.ids")"
c="$(cat "$PROMPTS/p2_4096.ids")"; d="$(cat "$PROMPTS/p1_1024.ids")"

# Palindromic over the object list so drift between the first and last arm cancels.
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) = 1 ]; then set -- "$@"; fi
  for o in "$@"; do
    out=$("$BIN/plowrt" amd-bench --blob "$ASSETS/model.pkt" --hsaco "$o" \
      --checkpoint "$ASSETS/checkpoint" --steps 64 --batched --prompt "$a;$b;$c;$d" 2>&1)
    echo "r$r $(basename "$o")  $(echo "$out" | grep -E 'tpot|agree' | tr -d '\n')"
  done
done
