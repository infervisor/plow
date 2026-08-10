#!/usr/bin/env bash
# Serve one GLM asset to readiness + coherence, then STOP. The load-fault smoke test.
#
# Exists because ad-hoc versions of this got the liveness check wrong twice in one session:
# polling the `nix develop` wrapper pid says DIED while the real plowrt is still loading
# (75-250 s), and a `kill -9` on the tree leaves the megakernel resident and wedges the box
# for every later test. This script polls the PORT, detects the fault out of the LOG, and
# tears down with TERM -> wait -> KILL on the process GROUP.
#
#   scripts/glm52_serve_smoke.sh <assets> <port> [hsaco-override]
# Exit: 0 ready+coherent, 2 GPU fault, 3 died, 4 timeout.
set -uo pipefail
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ASSETS="${1:?assets}"; PORT="${2:?port}"; HSACO="${3:-}"
LOG="${SMOKE_LOG:-/tmp/glm52_smoke_$PORT.log}"
NIX="${PLOW_NIX:-/nix/var/nix/profiles/default/bin/nix}"
ROCM_LIB="${PLOW_ROCM_LIB:-/opt/rocm-7.2.4/lib}"
SERVE_ENV="${SERVE_ENV:-PLOW_MLA_PF_V2=1}"
TIMEOUT="${SMOKE_TIMEOUT:-420}"


# THE GPU LEASE. Every timing-bearing runner on this box takes /tmp/plow_gpu.lock; a serve
# without it can silently corrupt a lease-holder's arm (and vice versa). Same protocol as
# run_plow.sh: mkdir-as-mutex, owner file for diagnosis, released by the teardown trap.
LOCK="${PLOW_GPU_LOCK:-/tmp/plow_gpu.lock}"
HAVE_LOCK=0
for i in $(seq 1 "${LOCK_TRIES:-360}"); do
  mkdir "$LOCK" 2>/dev/null && { HAVE_LOCK=1; break; }
  sleep 5
done
[ "$HAVE_LOCK" = 1 ] || { echo "FAIL: could not take the GPU lock ($LOCK)"; exit 1; }
echo "$$ $(basename "$0")" > "$LOCK/owner" 2>/dev/null

if pgrep '^plowrt' >/dev/null 2>&1; then
  echo "FAIL: a plowrt is already running:"; pgrep -a '^plowrt'; exit 1
fi

HS=""
[ -n "$HSACO" ] && HS="export PLOW_HSACO='$HSACO';"
setsid "$NIX" develop "$WT" -c bash -c \
  "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:$ROCM_LIB\"; export $SERVE_ENV; $HS \
   exec '$WT/target/release/plowrt' serve --assets '$ASSETS' --port '$PORT'" \
  > "$LOG" 2>&1 &
WRAP=$!
SPGID="$(ps -o pgid= "$WRAP" 2>/dev/null | tr -d ' ')"

teardown() {
  [ -n "${SPGID:-}" ] && kill -TERM "-$SPGID" 2>/dev/null
  # plowrt's HSA teardown after a fault can take a while; give it real time before KILL,
  # then wait for /dev/kfd to actually be released so the NEXT test starts clean.
  for i in $(seq 1 30); do pgrep '^plowrt' >/dev/null 2>&1 || break; sleep 2; done
  pgrep '^plowrt' >/dev/null 2>&1 && { kill -KILL "-$SPGID" 2>/dev/null; sleep 5; }
  for i in $(seq 1 60); do
    held=0
    for p in $(pgrep '^plowrt' 2>/dev/null); do held=1; done
    [ "$held" = 0 ] && break; sleep 2
  done
  # Release the lease LAST — after the serve is actually gone, or a new holder races a
  # still-resident megakernel.
  [ "${HAVE_LOCK:-0}" = 1 ] && rm -rf "$LOCK"
}
trap 'teardown' EXIT

rc=4
for i in $(seq 1 "$TIMEOUT"); do
  if grep -q "Memory access fault" "$LOG" 2>/dev/null; then rc=2; break; fi
  # The real serve process, not the wrapper: comm-exact match.
  if ! pgrep '^plowrt' >/dev/null 2>&1 && [ "$i" -gt 15 ]; then
    grep -q "Memory access fault" "$LOG" && rc=2 || rc=3; break
  fi
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then rc=0; break; fi
  sleep 2
done

if [ "$rc" = 0 ]; then
  MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
          | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  GATE=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":64,\"temperature\":0}")
  if echo "$GATE" | grep -qi paris; then echo "SMOKE PASS: ready + coherent"
  else
    grep -q "Memory access fault" "$LOG" && { echo "SMOKE FAULT (during gate)"; exit 2; }
    echo "SMOKE COHERENCE FAIL:"; echo "$GATE" | head -c 300; exit 5
  fi
else
  case "$rc" in
    2) echo "SMOKE FAULT (GPU memory access fault)";;
    3) echo "SMOKE DIED (no fault line — see $LOG)"; tail -5 "$LOG";;
    4) echo "SMOKE TIMEOUT after ${TIMEOUT} polls";;
  esac
fi
exit "$rc"
