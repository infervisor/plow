#!/usr/bin/env bash
# Sequential-EXCLUSIVE multi-arm showdown template (docs/bringup/07-perf-campaign.md): every arm on
# the same box with the same client, one server at a time, medians over >=5
# rounds. Wrap the whole script in gpulease:
#   perf-data/tools/gpulease showdown perf-data/tools/bringup_showdown.sh
#
# Edit the CONFIG block and the arm list at the bottom for your model/box.
set -u
# ---- CONFIG ------------------------------------------------------------------
HERE="$(cd "$(dirname "$0")" && pwd)"
PLOWRT="${PLOWRT:-$HERE/../../target/release/plowrt}"
SNAP="${SNAP:?set SNAP=<hf snapshot dir> (tokenizer source)}"
MODEL_ID="${MODEL_ID:?set MODEL_ID=<served model id / hf id>}"
BUNDLES="${BUNDLES:?set BUNDLES=<dir holding plow bundles>}"
CUBINS="${CUBINS:?set CUBINS=<segmented cubin dir for PLOW_PF_SEG_DIR>}"
export IN_LENS="${IN_LENS:-1024 4096}" NPROMPT="${NPROMPT:-9}"
ROUND="${ROUND:-showdown}"
PORT_PLOW=8093 PORT_VLLM=8085
# ------------------------------------------------------------------------------

plow_arm() { # <bundle-name> <tag> <PLOW_PF_SEG_PURE value: fp8|1>
  export PLOW_PF_SEG_DIR="$CUBINS"
  export PLOW_PF_SEG_PURE="$3" PLOW_PF_SEG_FA512=all PLOW_PF_SEG_GRAPH=1
  local LOG="${BRINGUP_OUT:-/tmp/bringup-$USER}/serve-$2.log"
  mkdir -p "$(dirname "$LOG")"
  "$PLOWRT" serve --assets "$BUNDLES/$1" --port $PORT_PLOW >"$LOG" 2>&1 &
  local SPID=$!
  for i in $(seq 1 600); do
    curl -sf --max-time 2 "http://127.0.0.1:$PORT_PLOW/v1/models" >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { echo "PLOW DIED: $2"; tail -4 "$LOG"; return 1; }
    sleep 1
  done
  "$HERE/bringup_bench.sh" "$2" "http://127.0.0.1:$PORT_PLOW" "$MODEL_ID" "$SNAP" "$ROUND"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 3
}

vllm_arm() { # <tag> <extra vllm args...>
  local TAG="$1"; shift
  local LOG="${BRINGUP_OUT:-/tmp/bringup-$USER}/serve-$TAG.log"
  mkdir -p "$(dirname "$LOG")"
  vllm serve "$MODEL_ID" --dtype bfloat16 --tensor-parallel-size 1 \
    --max-model-len 8192 --gpu-memory-utilization 0.50 \
    --scheduling-policy fcfs --port $PORT_VLLM "$@" >"$LOG" 2>&1 &
  local SPID=$!
  for i in $(seq 1 900); do
    curl -sf --max-time 2 "http://127.0.0.1:$PORT_VLLM/v1/models" >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { echo "VLLM DIED: $TAG"; tail -20 "$LOG"; return 1; }
    sleep 2
  done
  "$HERE/bringup_bench.sh" "$TAG" "http://127.0.0.1:$PORT_VLLM" "$MODEL_ID" "$SNAP" "$ROUND"
  kill $SPID 2>/dev/null; sleep 6; kill -9 $SPID 2>/dev/null; sleep 4
}

# ---- ARMS (edit per bring-up) ------------------------------------------------
# plow_arm <bundle> plow-fp8 fp8
# plow_arm <bundle> plow-bf16 1
# vllm_arm vllm-bf16
# vllm_arm vllm-fp8 --quantization fp8
echo "edit the arm list at the bottom of $0, then re-run"
