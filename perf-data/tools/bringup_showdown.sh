#!/usr/bin/env bash
# Sequential-EXCLUSIVE multi-arm showdown template (docs/bringup/07-perf-campaign.md): every arm on
# the same box with the same client, one server at a time, medians over >=5
# rounds. Wrap the whole script in gpulease:
#   perf-data/tools/gpulease showdown perf-data/tools/bringup_showdown.sh
#
# Edit the CONFIG block and the arm list at the bottom for your model/box.
set -euo pipefail
# ---- CONFIG ------------------------------------------------------------------
HERE="$(cd "$(dirname "$0")" && pwd)"
PLOWRT="${PLOWRT:-$HERE/../../target/release/plowrt}"
SNAP="${SNAP:?set SNAP=<hf snapshot dir> (tokenizer source)}"
MODEL_ID="${MODEL_ID:?set MODEL_ID=<served model id / hf id>}"
BUNDLES="${BUNDLES:?set BUNDLES=<dir holding plow bundles>}"
CUBINS="${CUBINS:?set CUBINS=<segmented cubin dir for PLOW_PF_SEG_DIR>}"
export IN_LENS="${IN_LENS:-1024 4096}" NPROMPT="${NPROMPT:-9}"
ROUND_PREFIX="${ROUND_PREFIX:-${ROUND:-showdown}}"
ROUNDS="${ROUNDS:-5}"
case "$ROUNDS" in *[!0-9]*|"") echo "ROUNDS must be an integer >= 5" >&2; exit 2;; esac
[ "$ROUNDS" -ge 5 ] || { echo "ROUNDS must be >= 5" >&2; exit 2; }
PORT_PLOW=8093 PORT_VLLM=8085
# ------------------------------------------------------------------------------

plow_arm() { # <bundle-name> <tag> <PLOW_PF_SEG_PURE value: fp8|1>
  python3 - "$BUNDLES/$1/build.json" <<'EOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"missing build manifest: {p}")
d = json.loads(p.read_text())
lean = d.get("lean", {})
if d.get("schema") != 1 or lean.get("verified") is not True or lean.get("oracle") is not True:
    raise SystemExit(f"unverified build manifest: {p}")
if not d.get("pairing", {}).get("hash"):
    raise SystemExit(f"build manifest has no packet/object pairing hash: {p}")
EOF
  export PLOW_PF_SEG_DIR="$CUBINS"
  export PLOW_PF_SEG_PURE="$3" PLOW_PF_SEG_FA512=all PLOW_PF_SEG_GRAPH=1
  local LOG="${BRINGUP_OUT:-/tmp/bringup-$USER}/serve-$2.log"
  mkdir -p "$(dirname "$LOG")"
  "$PLOWRT" serve --assets "$BUNDLES/$1" --port $PORT_PLOW >"$LOG" 2>&1 &
  local SPID=$!
  local READY=0
  for i in $(seq 1 "${BRINGUP_READY_ATTEMPTS:-600}"); do
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT_PLOW/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
    kill -0 $SPID 2>/dev/null || { echo "PLOW DIED: $2"; tail -4 "$LOG"; return 1; }
    sleep "${BRINGUP_READY_SLEEP:-1}"
  done
  if [ "$READY" -ne 1 ] || ! grep -q "backend ready.*GPU accelerated" "$LOG"; then
    echo "PLOW GPU READINESS FAILED: $2" >&2
    tail -30 "$LOG" >&2
    kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true
    return 1
  fi
  local rc=0
  for round in $(seq 1 "$ROUNDS"); do
    "$HERE/bringup_bench.sh" "$2" "http://127.0.0.1:$PORT_PLOW" "$MODEL_ID" "$SNAP" "$ROUND_PREFIX-$round" || { rc=$?; break; }
  done
  kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true; sleep 3
  return "$rc"
}

vllm_arm() { # <tag> <extra vllm args...>
  local TAG="$1"; shift
  local LOG="${BRINGUP_OUT:-/tmp/bringup-$USER}/serve-$TAG.log"
  mkdir -p "$(dirname "$LOG")"
  vllm serve "$MODEL_ID" --dtype bfloat16 --tensor-parallel-size 1 \
    --max-model-len 8192 --gpu-memory-utilization 0.50 \
    --scheduling-policy fcfs --port $PORT_VLLM "$@" >"$LOG" 2>&1 &
  local SPID=$!
  local READY=0
  for i in $(seq 1 "${VLLM_READY_ATTEMPTS:-900}"); do
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT_VLLM/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
    kill -0 $SPID 2>/dev/null || { echo "VLLM DIED: $TAG"; tail -20 "$LOG"; return 1; }
    sleep "${VLLM_READY_SLEEP:-2}"
  done
  if [ "$READY" -ne 1 ]; then
    echo "VLLM READINESS TIMEOUT: $TAG" >&2
    tail -30 "$LOG" >&2
    kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true
    return 1
  fi
  local rc=0
  for round in $(seq 1 "$ROUNDS"); do
    "$HERE/bringup_bench.sh" "$TAG" "http://127.0.0.1:$PORT_VLLM" "$MODEL_ID" "$SNAP" "$ROUND_PREFIX-$round" || { rc=$?; break; }
  done
  kill $SPID 2>/dev/null || true; sleep 6; kill -9 $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true; sleep 4
  return "$rc"
}

# ---- ARMS (edit per bring-up) ------------------------------------------------
# plow_arm <bundle> plow-fp8 fp8
# plow_arm <bundle> plow-bf16 1
# vllm_arm vllm-bf16
# vllm_arm vllm-fp8 --quantization fp8
echo "no showdown arms configured; edit the arm list at the bottom of $0" >&2
exit 2
