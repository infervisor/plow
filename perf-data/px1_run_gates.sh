#!/usr/bin/env bash
# px1_run_gates.sh — PX-1 stage-1 correctness gates against `plowrt serve`.
#
# Boots the server twice (PLOW_PF_BATCH=1, then unset = legacy control) on the
# PX-1 assets dir and runs perf-data/px1_gates.py:
#   Gate A: burst outputs byte-identical to solo outputs (batching on)
#   Gate B: victim outputs identical solo/concurrent/both orders (batching on)
#   Cross-check (informative): legacy solo vs batched solo
#
# Usage: ASSETS=/root/gpu-assets-px1 OUT=/tmp/px1-gates perf-data/px1_run_gates.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${ASSETS:?set ASSETS}"
OUT="${OUT:-/tmp/px1-gates}"
PORT="${PORT:-8091}"
BIN="$ROOT/target/release/plowrt"
mkdir -p "$OUT"

SRVPID=""
stop_srv() {
  if [ -n "$SRVPID" ]; then
    kill -TERM "$SRVPID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 2; done
    kill -KILL "$SRVPID" 2>/dev/null || true
    SRVPID=""
  fi
}
trap stop_srv EXIT

start_srv() { # $1 = logfile
  NO_COLOR=1 RUST_LOG=info,plowrt=debug "$BIN" serve \
    --assets "$ASSETS" --port "$PORT" > "$1" 2>&1 &
  SRVPID=$!
  for _ in $(seq 1 900); do
    grep -q "serving OpenAI API over TCP" "$1" && return 0
    kill -0 "$SRVPID" 2>/dev/null || { echo "server died"; tail -30 "$1"; exit 1; }
    sleep 1
  done
  echo "server never came up"; exit 1
}

echo "=== batched server (PLOW_PF_BATCH=1) ==="
PLOW_PF_BATCH=1 start_srv "$OUT/server-on.log"
grep -q "batched prefill enabled" "$OUT/server-on.log" \
  || { echo "FATAL: PX-1 mode did not engage"; grep -i "PLOW_PF_BATCH\|prefill" "$OUT/server-on.log"; exit 1; }
echo "--- solo (batching on) ---"
PORT=$PORT python3 "$ROOT/perf-data/px1_gates.py" solo "$OUT/on-solo.json"
echo "--- burst (batching on) ---"
PORT=$PORT python3 "$ROOT/perf-data/px1_gates.py" burst "$OUT/on-burst.json"
BATCHED_LAUNCHES=$(grep -c "batched prefill launch" "$OUT/server-on.log" || true)
MULTI=$(grep "batched prefill launch" "$OUT/server-on.log" | grep -cv "requests=1" || true)
echo "batched prefill launches: $BATCHED_LAUNCHES (multi-request: $MULTI)"
stop_srv

echo "=== legacy server (PLOW_PF_BATCH unset) ==="
start_srv "$OUT/server-off.log"
echo "--- solo (legacy) ---"
PORT=$PORT python3 "$ROOT/perf-data/px1_gates.py" solo "$OUT/off-solo.json"
stop_srv

echo
echo "=== GATE A + B: burst vs solo (batching on) — must PASS ==="
python3 "$ROOT/perf-data/px1_gates.py" cmp "$OUT/on-solo.json" "$OUT/on-burst.json" \
  && echo "GATE A/B: PASS" || { echo "GATE A/B: FAIL"; exit 1; }
[ "$MULTI" -gt 0 ] || { echo "GATE VOID: no multi-request packed launch observed"; exit 1; }

echo
echo "=== cross-check (informative): legacy solo vs batched solo ==="
python3 "$ROOT/perf-data/px1_gates.py" cmp "$OUT/off-solo.json" "$OUT/on-solo.json" \
  || echo "(informative only — first-token path differs: prefill lm_head vs decode step)"

echo
echo "=== Gate B sensitivity control ==="
python3 - "$OUT/on-solo.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
v, c = d["victim"], d["concat_control"]
print("victim (isolated):", repr(v[:60]))
print("concat control   :", repr(c[:60]))
print("SENSITIVITY:", "OK (concat differs from isolated victim)" if v != c else "WEAK (concat == isolated)")
EOF
