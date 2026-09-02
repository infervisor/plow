#!/usr/bin/env bash
# One benchmark pass over one serving endpoint with the vLLM bench client
# (see docs/bringup/07-perf-campaign.md: same client for EVERY arm, sequential + exclusive).
#   bringup_bench.sh <tag> <base_url> <model> <tokenizer_dir> [round]
# Env: IN_LENS (default "128 1024 4096"), NPROMPT (default 6), OUTLEN (default 8),
#      BRINGUP_OUT (default /tmp/bringup-$USER).
# Appends one line per cell to $BRINGUP_OUT/cells.tsv:
#   tag round inlen ttft_mean ttft_median ttft_p99 tpot_median
set -euo pipefail
TAG="$1"; URL="$2"; MODEL="$3"; TOK="$4"; ROUND="${5:-1}"
IN_LENS="${IN_LENS:-128 1024 4096}"; NPROMPT="${NPROMPT:-6}"; OUTLEN="${OUTLEN:-8}"
OUTDIR="${BRINGUP_OUT:-/tmp/bringup-$USER}"
mkdir -p "$OUTDIR"

for IL in $IN_LENS; do
  LOG="$OUTDIR/$TAG-r$ROUND-in$IL.log"
  vllm bench serve --backend openai-chat --base-url "$URL" --endpoint /v1/chat/completions \
    --model "$MODEL" --tokenizer "$TOK" \
    --dataset-name random --random-input-len "$IL" --random-output-len "$OUTLEN" \
    --num-prompts "$NPROMPT" --max-concurrency 1 --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el >"$LOG" 2>&1
  ROW=$(python3 - "$TAG" "$ROUND" "$IL" "$LOG" <<'EOF'
import re, sys
tag, rnd, il, log = sys.argv[1:5]
txt = open(log).read()
def grab(name):
    m = re.search(rf"(?:Mean|Average) {name} \(ms\):\s+([\d.]+)", txt)
    med = re.search(rf"Median {name} \(ms\):\s+([\d.]+)", txt)
    p99 = re.search(rf"P99 {name} \(ms\):\s+([\d.]+)", txt)
    if not (m and med and p99):
        raise SystemExit(f"missing {name} metrics in {log}")
    return m.group(1), med.group(1), p99.group(1)
t = grab("TTFT"); p = grab("TPOT")
print("\t".join([tag, rnd, il, t[0], t[1], t[2], p[1]]))
EOF
  )
  printf '%s\n' "$ROW" >> "$OUTDIR/cells.tsv"
done
