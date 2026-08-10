#!/usr/bin/env bash
# Drive the FACTS GATE across arms, one server at a time, under the box lock.
#
#   ARMS    space-separated arm names from the case below (default "ctrl ragged")
#   LENS    exact OFF-RUNG prompt token counts (default: the seven where the
#           ctrl and ragged plans differ). On-rung lengths are byte-identical
#           across arms and the verdict REFUSES them.
#   INJECT  fault to inject into every prompt (e.g. drop-tail:400). This is the
#           gate's failure proof, not a measurement mode.
#
# One server per arm: PLOW_RAGGED_CHUNK / PLOW_LAUNCH_ROWS are read from
# RuntimeConfig, a OnceLock, at process start. SIGTERM only, never SIGKILL — a
# kill -9 leaves the persistent megakernel resident on the card.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS="${ASSETS:-/workspace/assets/gfx942/glm52-tp8-final2}"
PORT="${PORT:-8195}"
OUT="${OUT:-$WT/perf-data/plow-gfx942/facts-gate-raw}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
ARMS="${ARMS:-ctrl ragged}"
LENS="${LENS:-1025,3073,4097,6145,8193,10369,12345}"
ITEMS="${ITEMS:-}"
MAXTOK="${MAXTOK:-288}"
INJECT="${INJECT:-}"
SUFFIX="${SUFFIX:-}"
LOCK="${LOCK:-/tmp/plow_gpu.lock}"
# `nix` is not on PATH in a non-login shell, and `env ... nix develop` then dies
# with "No such file or directory" AFTER the lock is taken.
export PATH="/root/.nix-profile/bin:/nix/var/nix/profiles/default/bin:$PATH"
# The flake puts /opt/rocm/lib on LD_LIBRARY_PATH and THAT DIRECTORY DOES NOT
# EXIST on this box. Without a real ROCm lib dir the HSA probe fails and plowrt
# serves from the CPU reference interpreter, correctly and meaninglessly.
export ROCM_LIB="${ROCM_LIB:-/opt/rocm-7.2.4/lib:/opt/amdgpu/lib/x86_64-linux-gnu}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.2.4}" HIP_PATH="${HIP_PATH:-/opt/rocm-7.2.4}"
mkdir -p "$OUT"

SPGID=""; HAVE_LOCK=""
stop_server() { [ -n "$SPGID" ] && kill -TERM "-$SPGID" 2>/dev/null; sleep 5; SPGID=""; }
cleanup() {
  stop_server
  # Only ever remove a lock this process actually took. An unconditional rmdir
  # in a trap deletes a SIBLING agent's lock — that has happened here. The lock
  # is held for the WHOLE battery, not per arm: releasing it between arms lets a
  # sibling take the card halfway through an A/B.
  [ -n "$HAVE_LOCK" ] && rmdir "$LOCK" 2>/dev/null; HAVE_LOCK=""
}
# A trap that does not exit leaves the harness running with a dead server.
trap 'cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

echo "waiting for the box lock $LOCK ..."
until mkdir "$LOCK" 2>/dev/null; do sleep 20; done
HAVE_LOCK=1   # set ONLY after mkdir succeeded
# Holding the lock is not sufficient: a sibling may run a RENAMED binary, which
# `pgrep -x plowrt` misses. Match on the comm PREFIX.
while pgrep '^plowrt' >/dev/null 2>&1; do echo "  foreign plowrt still up..."; sleep 10; done
ss -lptn "sport = :$PORT" 2>/dev/null | grep -q LISTEN && { echo "FAIL: port $PORT busy"; exit 1; }
echo "lock held; GPU use:"; rocm-smi --showuse 2>/dev/null | grep -E "GPU\[[0-9]\]" | head -8

for arm in $ARMS; do
  case "$arm" in
    # EXPLICIT on both arms. Once `ragged_chunk` defaults ON, an arm that merely
    # omits the flag is no longer the control -- it is the candidate under
    # another name, and the A/B silently becomes an A/A.
    ctrl)     ENVV=(PLOW_MLA_PF_V2=1 PLOW_RAGGED_CHUNK=0) ;;
    ragged)   ENVV=(PLOW_MLA_PF_V2=1 PLOW_RAGGED_CHUNK=1) ;;
    reprice)  ENVV=(PLOW_MLA_PF_V2=1 PLOW_RAGGED_CHUNK=0 PLOW_LAUNCH_ROWS="${LR:-1780}") ;;
    default)  ENVV=(PLOW_MLA_PF_V2=1) ;;   # whatever the shipped default IS
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  echo "=== arm $arm : ${ENVV[*]}"
  LOG="/tmp/fg_serve_${arm}${SUFFIX}_$PORT.log"
  # ROCm is NOT on the nix shell's library path. Without this the HSA probe
  # fails, plowrt falls back to the CPU reference interpreter, and it SERVES
  # PERFECTLY -- a whole battery of meaningless answers. Asserted below.
  setsid env "${ENVV[@]}" nix develop "$WT" --command bash -c \
      'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_LIB"; exec "$0" serve --assets "$1" --port "$2"' \
      "$BIN" "$ASSETS" "$PORT" > "$LOG" 2>&1 &
  SPID=$!
  SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"
  for _ in $(seq 1 1800); do
    curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SPID" 2>/dev/null || { echo "FAIL: server died"; tail -30 "$LOG"; exit 1; }
    sleep 1
  done
  MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  echo "  model: $MODEL"
  # A CPU-interpreter server answers every prompt correctly and measures nothing.
  grep -q "backend ready — GPU accelerated" "$LOG" || { echo ">>> NOT ON THE GPU ($arm)"; grep -iE "backend|hsa=" "$LOG" | head; exit 1; }
  grep -q "HSA backend selected" "$LOG" || { echo ">>> not the HSA backend ($arm)"; exit 1; }
  curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":32,\"temperature\":0}" \
    | grep -qi paris || { echo ">>> coherence gate FAIL ($arm)"; exit 1; }
  echo ">>> coherence gate: PASS ($arm)"
  # Attribute the cell to a PLAN that was logged, not to one that was assumed.
  grep -iE "prefill chunk policy" "$LOG" | head -2 \
    || echo "  (chunk-policy line is emitted at the FIRST prefill)"

  python3 "$WT/perf-data/probes/facts_gate.py" run --port "$PORT" --arm "$arm" \
    --lens "$LENS" --items "$ITEMS" --max-tokens "$MAXTOK" \
    ${INJECT:+--inject "$INJECT"} --out "$OUT/facts_${arm}${SUFFIX}.json" || exit 1
  grep -iE "prefill chunk policy" "$LOG" | head -2
  stop_server
done
echo "ALL ARMS DONE -> $OUT"
