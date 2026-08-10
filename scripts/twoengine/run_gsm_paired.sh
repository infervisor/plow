#!/usr/bin/env bash
# Serve ONE asset dir under the same four gates as run_plow.sh, then run the FULL-SET
# per-question GSM8K (gsm_paired.py) against it.
#
# This is the DECISIVE accuracy gate, used when an arm changes numerics and the question is
# "did the answers get worse". run_plow.sh's n=100 aggregate cannot answer that: measured
# 2026-08-09, 0.970 vs 0.950 is 0.72 sigma unpaired and McNemar p ~= 0.50. Use this instead,
# then `mcnemar.py control.json arm.json`.
#
# $1 assets  $2 port  $3 label
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="${PLOW_REPO:-$(cd "$HERE/../.." && pwd)}"
ASSETS="${1:?assets}"; PORT="${2:?port}"; LABEL="${3:?label}"
OUT="${OUT:-${TMPDIR:-/tmp}/twoengine}"; mkdir -p "$OUT"
export N="${N:-1319}" SHOTS="${SHOTS:-8}" MAXTOK="${MAXTOK:-320}" CONC="${CONC:-1}"
export GSM8K_DIR="${GSM8K_DIR:-$HOME/.cache/gsm8k}"
LOG="$OUT/serve_$LABEL.log"
LOCK=/tmp/plow_gpu.lock

# GATE 0 -- the binary itself. `cargo build`/`cargo test` WITHOUT `--features hsa` silently
# replaces target/release/plowrt with a CPU-only binary, and that binary SERVES PERFECTLY:
# correct answers, coherence gate green, every timing fiction. Measured 2026-08-09: four
# interleaved A/B arms were destroyed this way by an unrelated `cargo test --workspace` running
# in the same session. Gate 2 below catches it after a 75 s model load; this catches it in a
# millisecond, before the GPU lock is even taken.
PLOWRT_BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
[ -x "$PLOWRT_BIN" ] || { echo "FAIL: no plowrt at $PLOWRT_BIN"; exit 1; }
grep -aq "libhsa-runtime64" "$PLOWRT_BIN" || {
  echo "FAIL: $PLOWRT_BIN was built WITHOUT --features hsa (no libhsa-runtime64 reference)."
  echo "      It would serve correct answers at fictional speed. Rebuild:"
  echo "      nix develop . -c cargo build --release -p plowrt --features hsa"
  exit 1; }

HAVE_LOCK=0

release() {
  [ -n "${SPGID:-}" ] && kill -TERM "-$SPGID" 2>/dev/null
  sleep 3
  [ -n "${SPGID:-}" ] && kill -KILL "-$SPGID" 2>/dev/null
  # rm -rf, not rmdir: we drop an `owner` file in the lock dir and rmdir refuses a non-empty
  # directory SILENTLY under 2>/dev/null, leaking the lock.
  [ "$HAVE_LOCK" = 1 ] && rm -rf "$LOCK"
  return 0
}
trap 'release; exit 143' INT TERM
trap 'release' EXIT

for i in $(seq 1 720); do
  mkdir "$LOCK" 2>/dev/null && { HAVE_LOCK=1; break; }
  sleep 5
done
[ "$HAVE_LOCK" = 1 ] || { echo "FAIL: could not take GPU lock"; exit 1; }
echo "$$ $LABEL" > "$LOCK/owner" 2>/dev/null

if pgrep '^plowrt' >/dev/null 2>&1; then
  echo "FAIL: a plowrt is already running:"; pgrep -a '^plowrt'; exit 1
fi

echo "=== [$LABEL] serving $ASSETS on :$PORT (n=$N shots=$SHOTS) ==="
cd "$WT"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
NIX="${PLOW_NIX:-/nix/var/nix/profiles/default/bin/nix}"
[ -x "$NIX" ] || { echo "FAIL: nix not found at '$NIX' (set PLOW_NIX)"; exit 1; }
# LD_LIBRARY_PATH must be set INSIDE the nix shell: the flake does not carry /opt/rocm-*/lib,
# so dlopen of libhsa-runtime64 fails, plowrt falls back to the CPU reference interpreter, and
# it SERVES PERFECTLY -- correct answers, fictional timings. Gate 2 below is the only catch.
ROCM_LIB="${PLOW_ROCM_LIB:-/opt/rocm-7.2.4/lib}"
[ -e "$ROCM_LIB/libhsa-runtime64.so.1" ] || {
  echo "FAIL: no libhsa-runtime64.so.1 under '$ROCM_LIB' (set PLOW_ROCM_LIB)"; exit 1; }
SERVE_ENV="${SERVE_ENV:-PLOW_MLA_PF_V2=1}"
setsid "$NIX" develop "$WT" -c bash -c \
  "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:$ROCM_LIB\"; export $SERVE_ENV; \
   exec ./target/release/plowrt serve --assets '$ASSETS' --port '$PORT'" \
  > "$LOG" 2>&1 &
SPID=$!
SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"

for i in $(seq 1 1800); do
  kill -0 "$SPID" 2>/dev/null || { echo "FAIL: server died during load"; tail -30 "$LOG"; exit 1; }
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "FAIL: never became ready"; tail -30 "$LOG"; exit 1; }

if grep -q "CPU reference backend active" "$LOG"; then
  echo "FAIL: plowrt selected the CPU REFERENCE BACKEND -- every number below would be fiction."
  exit 1
fi
grep -qE "HSA backend selected|hsa=true" "$LOG" && echo "  HSA backend: OK" \
  || echo "  >>> WARN: no HSA banner (check $LOG)"

MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
        | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
[ -n "$MODEL" ] || { echo "FAIL: no model id"; exit 1; }
echo "  model: $MODEL"

GATE=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":200,\"temperature\":0}")
echo "$GATE" | grep -qi paris || {
  echo ">>> COHERENCE GATE FAIL -- refusing to score a wrong server."; echo "$GATE" | head -c 600; exit 1; }
echo "  coherence gate: PASS"

MODEL="$MODEL" PORT="$PORT" LABEL="$LABEL" OUT="$OUT" python3 "$HERE/gsm_paired.py"
echo "=== [$LABEL] done ==="
