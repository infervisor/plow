#!/usr/bin/env bash
# BATCH-DETERMINISM probe: row X's tokens at rung>=2 must equal X served solo.
#
# Accuracy cannot localize a batched-decode defect on a truncated model (its tokens are
# gibberish either way); DETERMINISM can, and it is the property the ladder promises anyway —
# "a sequence keeps its slot while the program under it changes rung to rung". Serve the
# asset, run prompt X solo, then X concurrently with Y, and diff X's completions.
#
#   scripts/glm52_batch_determinism.sh <assets> <port> [max_tokens]
# Exit: 0 identical, 6 diverged, else serve failure codes.
set -uo pipefail
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ASSETS="${1:?assets}"; PORT="${2:?port}"; MAXTOK="${3:-48}"
NIX="${PLOW_NIX:-/nix/var/nix/profiles/default/bin/nix}"
ROCM_LIB="${PLOW_ROCM_LIB:-/opt/rocm-7.2.4/lib}"
LOG="/tmp/glm52_det_$PORT.log"
TMP="${TMPDIR:-/tmp}/glm52_det_$PORT"; mkdir -p "$TMP"


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

pgrep '^plowrt' >/dev/null 2>&1 && { echo "FAIL: plowrt already running"; exit 1; }
setsid "$NIX" develop "$WT" -c bash -c \
  "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:$ROCM_LIB\"; export PLOW_MLA_PF_V2=1; \
   exec '$WT/target/release/plowrt' serve --assets '$ASSETS' --port '$PORT'" > "$LOG" 2>&1 &
WRAP=$!; SPGID="$(ps -o pgid= "$WRAP" 2>/dev/null | tr -d ' ')"
teardown() {
  [ -n "${SPGID:-}" ] && kill -TERM "-$SPGID" 2>/dev/null
  for i in $(seq 1 30); do pgrep '^plowrt' >/dev/null 2>&1 || break; sleep 2; done
  pgrep '^plowrt' >/dev/null 2>&1 && { kill -KILL "-$SPGID" 2>/dev/null; sleep 5; }
  # Release the lease LAST — after the serve is actually gone, or a new holder races a
  # still-resident megakernel.
  [ "${HAVE_LOCK:-0}" = 1 ] && rm -rf "$LOCK"
}
trap 'teardown' EXIT

for i in $(seq 1 200); do
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  grep -q "Memory access fault" "$LOG" && { echo "FAULT during load"; exit 2; }
  sleep 2
done
MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
        | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
[ -n "$MODEL" ] || { echo "no model id"; exit 3; }

ask() { # <outfile> <prompt>
  curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":$MAXTOK,\"temperature\":0}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" > "$1" 2>/dev/null
}

PX="The quick brown fox jumps over the lazy dog. Continue this story."
PY="List the planets of the solar system in order from the sun."

ask "$TMP/x_solo.txt" "$PX"
ask "$TMP/x_pair.txt" "$PX" & ask "$TMP/y_pair.txt" "$PY" & wait

echo "--- X solo:  $(head -c 90 "$TMP/x_solo.txt" | tr '\n' ' ')"
echo "--- X pair:  $(head -c 90 "$TMP/x_pair.txt" | tr '\n' ' ')"
echo "--- Y pair:  $(head -c 90 "$TMP/y_pair.txt" | tr '\n' ' ')"
if cmp -s "$TMP/x_solo.txt" "$TMP/x_pair.txt"; then
  echo "DETERMINISM PASS: X identical solo vs paired"
else
  echo "DETERMINISM FAIL: X diverges under batching"
  exit 6
fi
