#!/usr/bin/env bash
# Token-identity correctness gate (see docs/bringup/07-perf-campaign.md): serve a bundle, run fixed
# greedy prompts, dump outputs for diffing against the reference arm.
#   bringup_gate.sh <assets-dir> <tag> <port> [plowrt-binary]
# Wrap in gpulease. Compare arms with: diff $BRINGUP_OUT/gate-out/<a>.txt <b>.txt
set -euo pipefail
ASSETS="$1"; TAG="$2"; PORT="$3"
PLOWRT="${4:-$(dirname "$0")/../../target/release/plowrt}"
OUTDIR="${BRINGUP_OUT:-/tmp/bringup-$USER}/gate-out"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/serve-$TAG.log"

python3 - "$ASSETS/build.json" <<'EOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"missing build manifest: {p}")
d = json.loads(p.read_text())
if d.get("schema") != 1:
    raise SystemExit(f"unsupported build manifest schema in {p}")
lean = d.get("lean", {})
if lean.get("verified") is not True or lean.get("oracle") is not True:
    raise SystemExit(f"unverified build manifest in {p}: lean={lean}")
if not d.get("pairing", {}).get("hash"):
    raise SystemExit(f"build manifest has no packet/object pairing hash: {p}")
EOF

"$PLOWRT" serve --assets "$ASSETS" --port "$PORT" >"$LOG" 2>&1 &
SPID=$!
cleanup() {
  kill "$SPID" 2>/dev/null || true
  wait "$SPID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for i in $(seq 1 "${BRINGUP_READY_ATTEMPTS:-600}"); do
  if curl --fail --silent --show-error --max-time 2 \
      "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -30 "$LOG"; exit 1; }
  sleep "${BRINGUP_READY_SLEEP:-1}"
done
[ "$READY" -eq 1 ] || { echo "SERVER READINESS TIMEOUT" >&2; tail -30 "$LOG" >&2; exit 1; }
MODEL=$(curl --fail-with-body --silent --show-error "http://127.0.0.1:$PORT/v1/models" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); model=d['data'][0]['id']; assert isinstance(model,str) and model; print(model)")
grep -q "backend ready.*GPU accelerated" "$LOG" || {
  echo "SERVER IS NOT GPU ACCELERATED" >&2
  tail -30 "$LOG" >&2
  exit 1
}
echo "server up: $MODEL"

PROMPTS=(
  "The capital of France is"
  "Water boils at a temperature of"
  "def fibonacci(n):"
  "The three primary colors are"
)
: > "$OUTDIR/$TAG.txt"
for p in "${PROMPTS[@]}"; do
  R=$(curl --fail-with-body --silent --show-error \
    "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":32,\"temperature\":0}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); out=d['choices'][0]['message']['content']; assert isinstance(out,str) and out; print(repr(out))")
  echo "PROMPT: $p" >> "$OUTDIR/$TAG.txt"
  echo "OUT:    $R"  >> "$OUTDIR/$TAG.txt"
done
cat "$OUTDIR/$TAG.txt"
