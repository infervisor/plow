#!/usr/bin/env bash
# Post-flip headline ladder: restate published GLM-5.2 TTFT at the FOUR lengths
# the shipped harness prompts actually use, on the three configurations the flip
# creates. Reuses `chunk_policy_battery.py --mode ttft` (interleaved reps, exact
# token counts) rather than growing a second timing harness.
#
#   ctl8    PLOW_RAGGED_CHUNK=0, 8192 ladder   -- the pre-flip control
#   rag8    shipped default,     8192 ladder   -- the flip, existing blob
#   rag16   shipped default,     16384 ladder  -- the flip, re-emitted blob
#
# The caller holds the box lock. SIGTERM only, never SIGKILL.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${PORT:-8195}"
OUT="${OUT:-$WT/perf-data/plow-gfx942/facts-gate-raw}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
ARMS="${ARMS:-ctl8 rag8 rag16}"
# The lengths plow's own published harness prompts land on (docs/arch/13 §7).
LENS="${LENS:-1023,4101,8196,16386}"
REPS="${REPS:-3}"
FACTS_ARMS="${FACTS_ARMS:-rag16}"   # arms that also run the facts gate
FACTS_LENS="${FACTS_LENS:-1025,3073,4097,6145,8193,10369,12345}"
export PATH="/root/.nix-profile/bin:/nix/var/nix/profiles/default/bin:$PATH"
export ROCM_LIB="${ROCM_LIB:-/opt/rocm-7.2.4/lib:/opt/amdgpu/lib/x86_64-linux-gnu}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.2.4}" HIP_PATH="${HIP_PATH:-/opt/rocm-7.2.4}"
mkdir -p "$OUT"

SPGID=""
stop_server() { [ -n "$SPGID" ] && kill -TERM "-$SPGID" 2>/dev/null; sleep 8; SPGID=""; }
trap 'stop_server; exit 130' INT TERM
trap 'stop_server' EXIT

for arm in $ARMS; do
  case "$arm" in
    ctl8)  ENVV=(PLOW_MLA_PF_V2=1 PLOW_RAGGED_CHUNK=0); AS=/workspace/assets/gfx942/glm52-tp8-final2 ;;
    rag8)  ENVV=(PLOW_MLA_PF_V2=1);                     AS=/workspace/assets/gfx942/glm52-tp8-final2 ;;
    rag16) ENVV=(PLOW_MLA_PF_V2=1);                     AS=/workspace/assets/gfx942/cp-rung16 ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  echo "=== arm $arm : ${ENVV[*]} assets=$AS"
  LOG="/tmp/flip_${arm}_$PORT.log"
  setsid env "${ENVV[@]}" nix develop "$WT" --command bash -c \
      'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_LIB"; exec "$0" serve --assets "$1" --port "$2"' \
      "$BIN" "$AS" "$PORT" > "$LOG" 2>&1 &
  sleep 3
  SPGID="$(ps -o pgid= -p "$(pgrep -f "serve --assets $AS --port $PORT" | head -1)" 2>/dev/null | tr -d ' ')"
  for _ in $(seq 1 900); do
    curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
  grep -q "backend ready — GPU accelerated" "$LOG" || { echo ">>> NOT ON THE GPU ($arm)"; exit 1; }
  MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  echo "  model: $MODEL"
  curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":32,\"temperature\":0}" \
    | grep -qi paris || { echo ">>> coherence gate FAIL ($arm)"; exit 1; }
  echo ">>> coherence gate: PASS ($arm)"

  python3 "$WT/perf-data/probes/chunk_policy_battery.py" --port "$PORT" --arm "$arm" \
    --mode ttft --lens "$LENS" --reps "$REPS" --out "$OUT/ttft_${arm}.json" || exit 1
  case " $FACTS_ARMS " in *" $arm "*)
    python3 "$WT/perf-data/probes/facts_gate.py" run --port "$PORT" --arm "$arm" \
      --lens "$FACTS_LENS" --max-tokens 288 --out "$OUT/facts_${arm}.json" || exit 1 ;;
  esac
  # Attribute every cell to a plan that was LOGGED, not one that was assumed.
  grep -iE "prefill chunk policy" "$LOG" | head -2
  stop_server
done
echo "FLIP LADDER DONE -> $OUT"
