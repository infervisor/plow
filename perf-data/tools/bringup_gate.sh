#!/usr/bin/env bash
# Token-identity correctness gate (see docs/bringup/07-perf-campaign.md): serve a bundle, run fixed
# greedy prompts, dump outputs for diffing against the reference arm.
#   bringup_gate.sh <assets-dir> <tag> <port> [plowrt-binary]
# Wrap in gpulease. Compare arms with: diff $BRINGUP_OUT/gate-out/<a>.txt <b>.txt
set -u
ASSETS="$1"; TAG="$2"; PORT="$3"
PLOWRT="${4:-$(dirname "$0")/../../target/release/plowrt}"
OUTDIR="${BRINGUP_OUT:-/tmp/bringup-$USER}/gate-out"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/serve-$TAG.log"

"$PLOWRT" serve --assets "$ASSETS" --port "$PORT" >"$LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT

for i in $(seq 1 600); do
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -30 "$LOG"; exit 1; }
  sleep 1
done
MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "server up: $MODEL"

PROMPTS=(
  "The capital of France is"
  "Water boils at a temperature of"
  "def fibonacci(n):"
  "The three primary colors are"
)
: > "$OUTDIR/$TAG.txt"
for p in "${PROMPTS[@]}"; do
  R=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":32,\"temperature\":0}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr(d['choices'][0]['message']['content']))" 2>&1)
  echo "PROMPT: $p" >> "$OUTDIR/$TAG.txt"
  echo "OUT:    $R"  >> "$OUTDIR/$TAG.txt"
done
cat "$OUTDIR/$TAG.txt"
