#!/usr/bin/env bash
# =============================================================================
# bench_vllm_tp.sh — vLLM tensor-parallel decode baseline for Gemma-4-31B
# =============================================================================
# TP-capable copy of bench_vllm_docker.sh. Adds a --tensor-parallel-size knob
# and shards the 57 GiB Gemma weights across N GPUs by setting
# HIP_VISIBLE_DEVICES to N distinct indices. Goal: get a vLLM TP decode
# baseline (TPOT ms/tok, batch 1, bf16) to compare against plow TP decode.
#
# DOCKER: `sudo -n docker` (invoking user not in docker group; passwordless sudo).
# GPU SELECTION: HIP_VISIBLE_DEVICES=comma list (NOT --gpus); all DRI/KFD via --device.
#
# PARITY WITH plow (batch-1 single-user bf16):
#   --dtype bfloat16 ; --max-concurrency 1 ; --random-output-len 128 ; TP=N.
#
# Usage:
#   TP=4 GPUS=0,1,2,3 IMAGE=vllm-rocm-gemma4:latest bash perf-data/bench_vllm_tp.sh
#   SERVE_ONLY=1 ...   # just try to serve + sanity, no benchmark sweep
# =============================================================================
set -uo pipefail

DOCKER="${DOCKER:-sudo -n docker}"
IMAGE="${IMAGE:-vllm-rocm-gemma4:latest}"
TP="${TP:-4}"
GPUS="${GPUS:-0,1,2,3}"                 # N distinct HIP indices, N==TP
MODEL="${MODEL:-gemma-4-31B-it}"
MAXLEN="${MAXLEN:-66560}"               # >= max(ctx)+output; native cap is huge
CTXS="${CTXS:-1024,4096,8192,16384,32768,65536}"
HOST_MODELS="${HOST_MODELS:-/home/lava/models}"
PORT="${PORT:-8000}"
CNAME="${CNAME:-vllm_tp}"
OUTDIR="${OUTDIR:-$(dirname "$0")/vllm_tp_logs}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
SERVE_ONLY="${SERVE_ONLY:-0}"

mkdir -p "$OUTDIR"

serve() {
  echo ">>> [TP=$TP img=$IMAGE] serving /models/$MODEL (max-model-len=$MAXLEN) on GPUs $GPUS"
  $DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
  $DOCKER run --rm -d --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    --ipc=host --shm-size=32g \
    -e HIP_VISIBLE_DEVICES="$GPUS" \
    -v "$HOST_MODELS":/models \
    -p "$PORT":8000 \
    --entrypoint vllm \
    "$IMAGE" \
    serve "/models/$MODEL" \
      --dtype bfloat16 \
      --max-num-batched-tokens 8192 \
      --max-model-len "$MAXLEN" \
      ${QUANT_ARGS:-} \
      ${MEM_UTIL:+--gpu-memory-utilization "$MEM_UTIL"} \
      --tensor-parallel-size "$TP" >/dev/null
}
# fp8 knob (matched-TP fp8kv matchup): QUANT=fp8 adds --quantization fp8 (dynamic e4m3 weight quant
# off the bf16 checkpoint); KVFP8=1 adds --kv-cache-dtype fp8 (halves the decode KV read). The summary
# CSV / logs get a QTAG suffix so fp8 runs don't clobber the bf16 baseline.
QUANT="${QUANT:-bf16}"; KVFP8="${KVFP8:-0}"; QUANT_ARGS=""
[ "$QUANT" = "fp8" ] && QUANT_ARGS="--quantization fp8"
[ "$KVFP8" = "1" ]   && QUANT_ARGS="$QUANT_ARGS --kv-cache-dtype fp8"
QTAG="$QUANT"; [ "$KVFP8" = "1" ] && QTAG="${QUANT}kv"

wait_health() {
  local t=0
  echo ">>> waiting for :$PORT/health (<= ${HEALTH_TIMEOUT}s)"
  while [ "$t" -lt "$HEALTH_TIMEOUT" ]; do
    # bail early if the container already died
    if ! $DOCKER ps --format '{{.Names}}' | grep -q "^${CNAME}$"; then
      echo "!!! container exited during startup (t=${t}s)"; return 2
    fi
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health")" = "200" ]; then
      echo ">>> healthy after ${t}s"; return 0
    fi
    sleep 10; t=$((t+10))
  done
  echo "!!! endpoint never became healthy"; return 1
}

sanity() {
  local out
  out=$(curl -s "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"/models/$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}")
  echo ">>> sanity raw: $out"
  echo "$out" | python3 -c 'import sys,json;print(">>> completion:",repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>/dev/null
}

bench_point() {
  local L="$1" log="$OUTDIR/${MODEL}_${QTAG}_tp${TP}_in${L}.log"
  $DOCKER exec "$CNAME" vllm bench serve \
    --model "/models/$MODEL" --dataset-name random \
    --random-input-len "$L" --random-output-len "$OUTPUT_LEN" \
    --max-concurrency 1 --num-prompts "$NUM_PROMPTS" --port 8000 \
    > "$log" 2>&1
  python3 - "$L" "$log" <<'PY'
import re,sys
L=int(sys.argv[1]); t=open(sys.argv[2]).read()
def g(p):
    m=re.search(p+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
ttft=g(r"Mean TTFT \(ms\):"); tpot=g(r"Mean TPOT \(ms\):"); itl=g(r"Mean ITL \(ms\):")
pf = L/(ttft/1000.0) if ttft==ttft and ttft>0 else float('nan')
dc = 1000.0/tpot if tpot==tpot and tpot>0 else float('nan')
print(f"{L},{ttft:.2f},{pf:.1f},{tpot:.3f},{itl:.3f},{dc:.2f}")
PY
}

serve || { echo "!!! docker run failed"; exit 1; }
if ! wait_health; then
  rc=$?
  echo "===== DOCKER LOGS (last 60) ====="
  $DOCKER logs "$CNAME" 2>&1 | tail -60
  $DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
  echo ">>> SERVE FAILED at TP=$TP (rc=$rc)"
  exit 3
fi

VER=$($DOCKER exec "$CNAME" vllm --version 2>/dev/null | tail -1)
echo ">>> vLLM version: $VER"
sanity

if [ "$SERVE_ONLY" = "1" ]; then
  echo ">>> SERVE_ONLY=1 — stopping after sanity."
  $DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
  exit 0
fi

csv="$OUTDIR/${MODEL}_${QTAG}_tp${TP}_summary.csv"
echo "input_len,ttft_ms,prefill_tok_s,tpot_ms,itl_ms,decode_tok_s" > "$csv"
IFS=',' read -ra L_ARR <<<"$CTXS"
for L in "${L_ARR[@]}"; do
  echo ">>> bench $MODEL tp=$TP input_len=$L output_len=$OUTPUT_LEN"
  row=$(bench_point "$L"); echo "    $row"; echo "$row" >> "$csv"
done
echo ">>> done TP=$TP ; summary: $csv"
$DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
echo ">>> vLLM version: $VER"
