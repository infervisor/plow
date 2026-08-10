#!/usr/bin/env bash
# Task 2: does a 16384 prefill rung BUILD, RUN, and pay for its memory?
#
# Two blobs, ONE binary (PLOWRT_BIN must be built with MAX_CHUNK >= 16384, else
# `plan_chunks` filters the 16384 bucket out and the rung arm is a silent null):
#   cp-ctl16   the shipped ladder, verified byte-identical to glm52-tp8-final2
#   cp-rung16  the same emit + a 16384 rung
# Both arms run RAGGED, because the padded DP will not use a 16384 rung below
# 16384 tokens (padding 8k of dead rows is correctly worse than a second launch)
# so a padded A/B would be a null everywhere the rung could help.
#
# VRAM is read from rocm-smi with the server LOADED and IDLE -- the memory
# question is answered by the card, not by the tensor table.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${PORT:-8195}"
BIN="${PLOWRT_BIN:-/tmp/plowrt_mc16}"
OUT="${OUT:-$WT/perf-data/plow-gfx942/chunk-policy-raw}"
LENS="${LENS:-8192,8193,12345,16386,24576}"
REPS="${REPS:-3}"
export ROCM_LIB="${ROCM_LIB:-/opt/rocm-7.2.4/lib:/opt/amdgpu/lib/x86_64-linux-gnu}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.2.4}" HIP_PATH="${HIP_PATH:-/opt/rocm-7.2.4}"
mkdir -p "$OUT"

SPGID=""
cleanup() { [ -n "$SPGID" ] && kill -TERM "-$SPGID" 2>/dev/null; sleep 5; SPGID=""; }
trap 'cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

for arm in ctl16 rung16; do
  A="/workspace/assets/gfx942/cp-$arm"
  echo "=== $arm ($A)"
  setsid env PLOW_MLA_PF_V2=1 PLOW_RAGGED_CHUNK=1 nix develop "$WT" --command bash -c \
      'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_LIB"; exec "$0" serve --assets "$1" --port "$2"' \
      "$BIN" "$A" "$PORT" > "/tmp/cp_mc_${arm}_$PORT.log" 2>&1 &
  SPID=$!; SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"
  for _ in $(seq 1 1800); do
    curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SPID" 2>/dev/null || { echo "FAIL: server died"; tail -30 "/tmp/cp_mc_${arm}_$PORT.log"; exit 1; }
    sleep 1
  done
  MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  echo "  model: $MODEL"
  curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":32,\"temperature\":0}" \
    | grep -qi paris || { echo ">>> coherence gate FAIL ($arm)"; exit 1; }
  echo ">>> coherence gate: PASS ($arm)"
  grep -iE "prefill chunk policy|PLOW_RAGGED_CHUNK" "/tmp/cp_mc_${arm}_$PORT.log" | head -3 || true
  echo "--- VRAM, loaded and idle:"
  rocm-smi --showmeminfo vram 2>/dev/null | grep -iE "used|GPU\[" | head -20 | tee "$OUT/vram_$arm.txt"
  python3 "$WT/perf-data/probes/chunk_policy_battery.py" --port "$PORT" --arm "$arm" \
    --mode ttft --lens "$LENS" --reps "$REPS" --out "$OUT/ttft_$arm.json" || exit 1
  python3 "$WT/perf-data/probes/chunk_policy_battery.py" --port "$PORT" --arm "$arm" \
    --mode ident --lens 8193,12345 --max-tokens 256 --questions essay,gold,sky \
    --out "$OUT/ident_$arm.json" || exit 1
  cleanup
done
echo "MAXCHUNK ARMS DONE -> $OUT"
