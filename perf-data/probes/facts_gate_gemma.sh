#!/usr/bin/env bash
# CROSS-MODEL exposure check for the `ragged_chunk` DEFAULT flip.
#
# `plan_chunks` and `rebase_chunk_rows` live in the SHARED AMD engine, so making
# ragged-M the default turns it on for every AMD model, not just GLM-5.2 — and
# the entire evidence base (`glm52-chunk-policy.md`, `glm52-facts-gate.md`) is
# GLM-5.2, a FULL-CAUSAL MLA model. Gemma-4 is the case that differs in kind: a
# SLIDING-WINDOW model whose KV cache is a ring sized `window + chunk - 1`.
#
# On Gemma's [128, 512, 1024] ladder the padded DP and the ragged cover pick the
# SAME rungs at every length, so ragged changes only the last chunk's EXECUTED
# ROW COUNT -- which by `glm52-chunk-policy.md` §2.2 is precisely the determinant
# of the output. So the text is expected to move; what this checks is that it
# stays COHERENT and CORRECT.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS="${ASSETS:-/workspace/assets/gfx942/g12b-64k-mergefix}"
PORT="${PORT:-8195}"
OUT="${OUT:-$WT/perf-data/plow-gfx942/facts-gate-raw}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
# Off-rung for the [128, 512, 1024] ladder: every one of these runs a last chunk
# whose executed row count differs between the arms.
LENS="${LENS:-600,1025,1500,2100}"
ITEMS="${ITEMS:-hop_train,hop_money,hop_avg,ndl_early,ndl_late}"
# The Gemma-4 packet declares fp8 weights and the loader binds their twins from
# PLOW_FP8_DIR; without it the load fails outright. Single rank, like the rest of
# the Gemma harness. Neither is an arm variable -- both are "serve this blob".
FP8_DIR="${FP8_DIR:-/workspace/models/gemma-4-12B-it-fp8}"
DEV="${ROCR_VISIBLE_DEVICES:-0}"
export PATH="/root/.nix-profile/bin:/nix/var/nix/profiles/default/bin:$PATH"
export ROCM_LIB="${ROCM_LIB:-/opt/rocm-7.2.4/lib:/opt/amdgpu/lib/x86_64-linux-gnu}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.2.4}" HIP_PATH="${HIP_PATH:-/opt/rocm-7.2.4}"
mkdir -p "$OUT"

SPGID=""
stop_server() { [ -n "$SPGID" ] && kill -TERM "-$SPGID" 2>/dev/null; sleep 8; SPGID=""; }
trap 'stop_server; exit 130' INT TERM
trap 'stop_server' EXIT

for arm in gem-ctl gem-rag; do
  case "$arm" in
    gem-ctl) ENVV=(PLOW_RAGGED_CHUNK=0 PLOW_FP8_DIR="$FP8_DIR" ROCR_VISIBLE_DEVICES="$DEV") ;;
    gem-rag) ENVV=(PLOW_RAGGED_CHUNK=1 PLOW_FP8_DIR="$FP8_DIR" ROCR_VISIBLE_DEVICES="$DEV") ;;
  esac
  echo "=== arm $arm : ${ENVV[*]}"
  LOG="/tmp/gem_${arm}_$PORT.log"
  setsid env "${ENVV[@]}" nix develop "$WT" --command bash -c \
      'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_LIB"; exec "$0" serve --assets "$1" --port "$2"' \
      "$BIN" "$ASSETS" "$PORT" > "$LOG" 2>&1 &
  sleep 3
  SPGID="$(ps -o pgid= -p "$(pgrep -f "serve --assets $ASSETS --port $PORT" | head -1)" 2>/dev/null | tr -d ' ')"
  for _ in $(seq 1 600); do
    curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
  grep -q "backend ready — GPU accelerated" "$LOG" || { echo ">>> NOT ON THE GPU ($arm)"; tail -20 "$LOG"; exit 1; }
  MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  echo "  model: $MODEL"
  curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":32,\"temperature\":0}" \
    | grep -qi paris || { echo ">>> coherence gate FAIL ($arm)"; exit 1; }
  echo ">>> coherence gate: PASS ($arm)"
  python3 "$WT/perf-data/probes/facts_gate.py" run --port "$PORT" --arm "$arm" \
    --lens "$LENS" --items "$ITEMS" --max-tokens 288 --out "$OUT/facts_${arm}.json" || exit 1
  grep -iE "prefill chunk policy|PLOW_RAGGED_CHUNK:" "$LOG" | head -2
  stop_server
done
echo "GEMMA CROSS-CHECK DONE -> $OUT"
