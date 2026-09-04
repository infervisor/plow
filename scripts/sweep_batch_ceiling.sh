#!/usr/bin/env bash
# SCOPE-1: does device-level decode throughput scale with compiled batch width on gfx950?
# Production bench prefills every row; diagnostics time the exact batched decode call. The
# final N+4 full-width dispatches preserve four warmups followed by N measured calls.
set -euo pipefail

WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT="${PLOW_BATCH_CEILING_CKPT:-/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475}"
PLOWRT="${PLOWRT:-$WT/target/release/plowrt}"
NIX="${PLOW_NIX_BIN:-nix}"
LEASE="${PLOW_GPULEASE_BIN:-$WT/perf-data/tools/gpulease}"
ASSET_ROOT="${PLOW_BATCH_CEILING_ASSET_ROOT:-$WT/build-amd}"
OBJECT_ROOT="${PLOW_BATCH_CEILING_OBJECT_ROOT:-$WT/build-amd}"
OUT="${PLOW_BATCH_CEILING_OUT:-/tmp/plow-batch-ceiling}"
CTXS="${CTXS:-1024 4096}"
STEPS="${STEPS:-64}"
WARMUP_STEPS="${WARMUP_STEPS:-4}"

run_one() { # <batch> <ctx>
  local batch=$1 ctx=$2 assets objects arm rows report log
  assets="$ASSET_ROOT/g31b-db$batch"
  objects="$OBJECT_ROOT/hsaco-b$batch"
  arm="$OUT/b${batch}-c${ctx}"
  rows="$arm/prompt_rows.txt"
  report="$arm/bench.json"
  log="$arm/bench.log"
  mkdir -p "$arm"
  python3 - "$batch" "$ctx" "$rows" <<'PY'
import sys
batch, ctx, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
if batch < 1 or ctx < 1:
    raise SystemExit("batch and context must be positive")
row = ",".join(["1"] * ctx)
with open(path, "w") as output:
    output.write((row + "\n") * batch)
PY
  echo "======== B=$batch ctx=$ctx hsaco=$objects"
  "$NIX" develop "$WT" --command "$PLOWRT" \
    --rt-checkpoint "$CKPT" --rt-hsaco "$objects" --multistep 1 \
    bench --assets "$assets" --prompt-rows "$rows" \
    --concurrency "$batch" --requests "$batch" --warmup-requests 0 \
    --output-len "$((WARMUP_STEPS + STEPS + 1))" \
    --max-hold-ms 8 --slo-ms 60000 --token-audit --engine-diagnostics \
    >"$report" 2>"$log" || { cat "$log" >&2; return 1; }
  python3 "$WT/scripts/batch_ceiling_validate.py" \
    --report "$report" --assets "$assets" --objects "$objects" \
    --checkpoint "$CKPT" --prompts "$rows" --batch "$batch" --ctx "$ctx" \
    --warmup "$WARMUP_STEPS" --measured "$STEPS"
}

if [[ -z "${PLOW_BATCH_CEILING_LEASED:-}" ]]; then
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  exec env PLOW_BATCH_CEILING_LEASED=1 "$LEASE" -n 1 batch-ceiling "$0" "$@"
fi

mkdir -p "$OUT"
for batch in ${BS:-1 2 4 8}; do
  for ctx in $CTXS; do
    run_one "$batch" "$ctx"
  done
done
