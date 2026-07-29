#!/usr/bin/env bash
# =============================================================================
# scripts/bench_vllm_all.sh — run bench_vllm_rocm.sh over the model set.
# =============================================================================
# Every model runs under its own GPU lease (bench_vllm_rocm.sh takes it), so a
# TP=1 run occupies one card and leaves the other seven leasable.
#
# SEQUENTIAL BY DEFAULT. Concurrency-1 TPOT is a latency number; a neighbour
# model hammering the other half of the box perturbs it through power and
# thermal headroom even though the cards are disjoint. PARALLEL=1 overlaps the
# TP=1 models when you want box utilization more than clean latency.
#
# Usage:
#   scripts/bench_vllm_all.sh                 # all ready models
#   MODELS="google/gemma-4-12B-it:1" scripts/bench_vllm_all.sh
#   PARALLEL=1 scripts/bench_vllm_all.sh
#
# TP sizing for MI355X (288 GiB/card), weights only:
#   gemma-4-12B    24 GB -> TP1      gemma-4-26B-A4B  52 GB -> TP1
#   gemma-4-31B    63 GB -> TP1      GLM-5.2-FP8     756 GB -> TP4
#   Kimi-K2.7-Code 595 GB -> TP4
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PARALLEL="${PARALLEL:-0}"
LOGDIR="${LOGDIR:-$REPO/perf-data/vllm-rocm}"
mkdir -p "$LOGDIR"

# "<repo-id>:<tp>:<dtype-args>"
DEFAULT_MODELS="\
google/gemma-4-12B-it:1:--dtype bfloat16
google/gemma-4-26B-A4B-it:1:--dtype bfloat16
google/gemma-4-31B-it:1:--dtype bfloat16
zai-org/GLM-5.2-FP8:4:--dtype auto
moonshotai/Kimi-K2.7-Code:4:--dtype auto"
MODELS="${MODELS:-$DEFAULT_MODELS}"

# A repo is benchable only if its snapshot has no *.incomplete blobs left.
# NB: `local id=$1 d=...$id...` does NOT work — bash expands every word before
# the builtin assigns, so $id is still empty in d. Assign on separate lines.
ready() {
  local id="$1"
  local d="$HF_CACHE/hub/models--${id//\//--}"
  [ -d "$d" ] || return 1
  [ -z "$(find "$d/blobs" -name '*.incomplete' -print -quit 2>/dev/null)" ]
}

run_one() {
  local id="$1" tp="$2" dt="$3"
  local slug="${id//\//_}"
  local log="$LOGDIR/run_${slug}_tp${tp}.log"
  echo ">>> [$(date -Is)] START $id TP=$tp"
  DTYPE_ARGS="$dt" "$REPO/scripts/bench_vllm_rocm.sh" "$id" "$tp" > "$log" 2>&1
  local rc=$?
  case $rc in
    0)  echo ">>> [$(date -Is)] OK   $id TP=$tp" ;;
    76) echo ">>> [$(date -Is)] CONTENDED $id TP=$tp — numbers discarded, re-run" ;;
    75) echo ">>> [$(date -Is)] LEASE TIMEOUT $id TP=$tp" ;;
    5)  echo ">>> [$(date -Is)] SERVER DIED MID-RUN $id TP=$tp — see ${LOGDIR}/*_serve.log" ;;
    4)  echo ">>> [$(date -Is)] NO DATA $id TP=$tp — served, but bench points produced no measurement"
        grep -m3 -iE "error|refus|trust_remote_code" "$log" 2>/dev/null ;;
    *)  echo ">>> [$(date -Is)] FAIL $id TP=$tp rc=$rc — see $log"; tail -20 "$log" ;;
  esac
  return $rc
}

pids=()
while IFS= read -r spec; do
  [ -z "$spec" ] && continue
  id="${spec%%:*}"; rest="${spec#*:}"; tp="${rest%%:*}"; dt="${rest#*:}"
  if ! ready "$id"; then
    echo ">>> SKIP $id — checkpoint not fully downloaded yet"
    continue
  fi
  if [ "$PARALLEL" = "1" ] && [ "$tp" = "1" ]; then
    run_one "$id" "$tp" "$dt" & pids+=($!)
  else
    run_one "$id" "$tp" "$dt"
  fi
done <<<"$MODELS"

for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

echo ">>> ALL RUNS COMPLETE — CSVs in $LOGDIR"
ls -1 "$LOGDIR"/*.csv 2>/dev/null || echo "    (no CSVs produced)"
