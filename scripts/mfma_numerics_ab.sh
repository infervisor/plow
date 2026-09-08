#!/usr/bin/env bash
# Full-model numerics for the MFMA batched-decode candidate: logits rel-L2, token agreement and
# first divergence, control vs candidate, at 128/1024/4096 input tokens x 3 prompts x 64 steps.
# Runs INSIDE one gpulease and INSIDE `nix develop`.
#
#   GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 gemma31-mfma-numerics \
#     nix develop /app/plow --command scripts/mfma_numerics_ab.sh <ctl-obj> <cand-obj> <outdir>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CTL="${1:?control objdir}"
CAND="${2:?candidate objdir}"
OUT="${3:?output dir}"
ASSETS="${PLOW_ASSETS:-/app/plow/build-gemma31/assets-ctx131072-chunk8192}"
BIN="${PLOW_BIN_DIR:-/app/plow/target-glm53/release}"
PROMPTS="${PROMPTS_DIR:-/tmp/mfma_prompts}"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="${ROCM_PATH:?nix develop did not set ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

run() { # $1 arm  $2 objdir  $3 tag  $4.. extra args
  local arm="$1" obj="$2" tag="$3"; shift 3
  local d="$OUT/$arm/$tag"
  mkdir -p "$d"
  "$BIN/plowrt" amd-bench --blob "$ASSETS/model.pkt" --hsaco "$obj" \
    --checkpoint "$ASSETS/checkpoint" --steps 64 --dump-logits "$d" "$@" \
    > "$OUT/$arm/$tag.log" 2>&1
  local rc=$?
  [ $rc = 0 ] || { echo "FAIL $arm/$tag rc=$rc"; tail -5 "$OUT/$arm/$tag.log"; }
  return $rc
}

for n in 128 1024 4096; do
  for p in 0 1 2; do
    ids="$(cat "$PROMPTS/p${p}_${n}.ids")"
    for arm in ctl cand; do
      case "$arm" in ctl) o="$CTL";; cand) o="$CAND";; esac
      run "$arm" "$o" "n${n}_p${p}" --prompt "$ids" || exit 1
    done
  done
done

# BATCH-WIDTH INDEPENDENCE at the model level: the same prompt in slot 0 of a ragged batch of 4
# must give the same logit stream as that prompt decoded alone.
for arm in ctl cand; do
  case "$arm" in ctl) o="$CTL";; cand) o="$CAND";; esac
  a="$(cat "$PROMPTS/p0_1024.ids")"; b="$(cat "$PROMPTS/p1_128.ids")"
  c="$(cat "$PROMPTS/p2_4096.ids")"; d="$(cat "$PROMPTS/p1_1024.ids")"
  run "$arm" "$o" "ragged4" --batched --prompt "$a;$b;$c;$d" || exit 1
  run "$arm" "$o" "solo1024p0" --prompt "$a" || exit 1
done
echo "=== dumps complete"
