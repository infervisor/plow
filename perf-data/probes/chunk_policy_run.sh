#!/usr/bin/env bash
# Drive the chunk-policy battery across ARMS, one server at a time.
#
#   ARMS   space-separated arm names from the case below (default: all three)
#   MODES  space-separated modes for chunk_policy_battery.py (default: "ttft ident")
#   LENS_TTFT / LENS_IDENT   exact prompt token counts per mode
#
# One server per arm because PLOW_LAUNCH_ROWS / PLOW_RAGGED_CHUNK are read from
# RuntimeConfig, a OnceLock, at process start. SIGTERM only, never SIGKILL.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS="${ASSETS:-/workspace/assets/gfx942/glm52-tp8-final2}"
PORT="${PORT:-8195}"
OUT="${OUT:-$WT/perf-data/plow-gfx942/chunk-policy-raw}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
ARMS="${ARMS:-ctrl reprice ragged}"
MODES="${MODES:-ttft ident}"
LENS_TTFT="${LENS_TTFT:-128,512,1024,1025,2048,3073,4096,4097,6145,8192,8193,10369,12345,71808}"
LENS_FACTS="${LENS_FACTS:-1024,1025,4097,8193}"
LENS_IDENT="${LENS_IDENT:-1024,1025,3073,4096,4097,6145,8193}"
REPS="${REPS:-3}"
MAXTOK="${MAXTOK:-256}"
export ROCM_LIB="${ROCM_LIB:-/opt/rocm-7.2.4/lib:/opt/amdgpu/lib/x86_64-linux-gnu}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.2.4}" HIP_PATH="${HIP_PATH:-/opt/rocm-7.2.4}"
mkdir -p "$OUT"

SPGID=""
cleanup() { [ -n "$SPGID" ] && kill -TERM "-$SPGID" 2>/dev/null; sleep 5; SPGID=""; }
# A trap that does not exit leaves the harness running with a dead server.
trap 'cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

for arm in $ARMS; do
  case "$arm" in
    ctrl)    ENVV=(PLOW_MLA_PF_V2=1) ;;
    reprice) ENVV=(PLOW_MLA_PF_V2=1 PLOW_LAUNCH_ROWS="${LR:-1780}") ;;
    ragged)  ENVV=(PLOW_MLA_PF_V2=1 PLOW_RAGGED_CHUNK=1) ;;
    both)    ENVV=(PLOW_MLA_PF_V2=1 PLOW_RAGGED_CHUNK=1 PLOW_LAUNCH_ROWS="${LR:-1780}") ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  echo "=== arm $arm : ${ENVV[*]}"
  # ROCm is NOT on the nix shell's library path: without this the HSA probe
  # fails, plowrt silently falls back to the CPU reference interpreter, and the
  # only thing that catches it is the coherence gate below.
  setsid env "${ENVV[@]}" nix develop "$WT" --command bash -c \
      'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_LIB"; exec "$0" serve --assets "$1" --port "$2"' \
      "$BIN" "$ASSETS" "$PORT" > "/tmp/cp_serve_${arm}_$PORT.log" 2>&1 &
  SPID=$!
  SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"
  for _ in $(seq 1 1800); do
    curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SPID" 2>/dev/null || { echo "FAIL: server died"; tail -30 "/tmp/cp_serve_${arm}_$PORT.log"; exit 1; }
    sleep 1
  done
  MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  echo "  model: $MODEL"
  # A fast wrong server is not a result.
  curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":32,\"temperature\":0}" \
    | grep -qi paris || { echo ">>> coherence gate FAIL ($arm)"; exit 1; }
  echo ">>> coherence gate: PASS ($arm)"
  # Record what the planner actually did, from the server's own log, so the
  # measured cell is attributable to a PLAN and not to an assumed one.
  grep -iE "prefill chunk policy|PLOW_RAGGED_CHUNK" "/tmp/cp_serve_${arm}_$PORT.log" | head -3 \
    || echo "  WARNING: no chunk-policy line yet (it is emitted at the FIRST prefill)"

  for mode in $MODES; do
    case "$mode" in
      ttft)  L="$LENS_TTFT";  EXTRA=(--reps "$REPS") ;;
      ident) L="$LENS_IDENT"; EXTRA=(--max-tokens "$MAXTOK") ;;
      facts) L="$LENS_FACTS"; EXTRA=(--max-tokens "$MAXTOK") ;;
    esac
    python3 "$WT/perf-data/probes/chunk_policy_battery.py" --port "$PORT" --arm "$arm" \
      --mode "$mode" --lens "$L" "${EXTRA[@]}" --out "$OUT/${mode}_${arm}.json" || exit 1
  done
  cleanup
done
echo "ALL ARMS DONE -> $OUT"
