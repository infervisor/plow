#!/usr/bin/env bash
# DeepSeek-V4.1 on 4x H200 through `plowrt dsv41-serve`: start the server, then `smoke` (a few greedy
# completions) or `sweep` (the same vllm bench grid as the vLLM TP4 baseline, /root/dsv41/run_baseline.sh).
#
# Run under the GPU lease, all four GPUs:
#   perf-data/tools/gpulease -n 4 dsv41-plowrt scripts/dsv41_nv/run_plowrt.sh smoke|sweep <tag>
set -uo pipefail
MODE=${1:?smoke|sweep}
TAG=${2:?tag}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PLOWRT=${PLOWRT:?path to a plowrt binary built with --features cuda}
CUBIN=${CUBIN:?path to dsv41_sm90a.cubin}
CKPT=${CKPT:-/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277}
VENV=${VENV:-/root/tts-work/venv-vllm/bin}
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
PORT=${PORT:-8211}
OUT=${OUT:-/root/dsv41/results/plowrt-$TAG}
mkdir -p "$OUT"
export HF_HOME=${HF_HOME:-/root/dsv41/hf} HF_HUB_OFFLINE=1

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee "$OUT/env.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee -a "$OUT/env.txt"
"$PLOWRT" dsv41-serve --ckpt "$CKPT" --cubin "$CUBIN" --port "$PORT" \
  --max-len "${MAX_LEN:-20480}" --max-slots "${MAX_SLOTS:-64}" --arena-gib "${ARENA_GIB:-24}" \
  --served-model-name "$MODEL" > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for i in $(seq 1 900); do
  if curl -sf localhost:$PORT/health >/dev/null; then READY=1; break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "SERVER DIED"; tail -40 "$OUT/server.log"; exit 3; fi
  sleep 2
done
[ "${READY:-0}" = 1 ] || { echo "SERVER NOT READY"; exit 4; }
echo "server ready after ~$((i*2))s"
nvidia-smi --query-gpu=index,memory.used --format=csv | tee -a "$OUT/env.txt"

complete() { # prompt max_tokens
  curl -s localhost:$PORT/v1/completions -H 'content-type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"m","prompt":sys.argv[1],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$1" "$2")"
  echo
}

run() { # name in out conc
  local name=$1 inl=$2 outl=$3 c=$4
  local n=${NPROMPT:-$(( c * 4 ))}; [ -z "${NPROMPT:-}" ] && [ $n -lt 16 ] && n=16
  $VENV/vllm bench serve --model $MODEL --port $PORT --backend openai --endpoint /v1/completions \
    --dataset-name random --random-input-len $inl --random-output-len $outl \
    --ignore-eos --num-prompts $n --max-concurrency $c --num-warmups 2 --seed 1 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir "$OUT" --result-filename "${name}_c${c}.json" \
    > "$OUT/${name}_c${c}.log" 2>&1
  echo "[$name c=$c] rc=$?"
  grep -E "Output token throughput|Total token throughput|Median TTFT|Median TPOT|Failed requests" "$OUT/${name}_c${c}.log"
}

if [ "$MODE" = streamtest ]; then  # client-side SSE delivery timing under concurrency
  if [ -z "${BENCH_ONLY:-}" ]; then
    echo "== token-id prompts"; $VENV/python "$ROOT/scripts/dsv41_nv/stream_probe.py" "$PORT" 8 32 256 | tee "$OUT/stream_probe.txt"
    echo "== text prompts"; $VENV/python "$ROOT/scripts/dsv41_nv/stream_probe.py" "$PORT" 8 32 256 text | tee -a "$OUT/stream_probe.txt"
  fi
  echo "== vllm bench"; NPROMPT=8 run probe_1k_32 1024 32 8
  exit 0
fi

if [ "$MODE" = smoke ]; then
  complete "The capital of France is" 16 | tee -a "$OUT/smoke.txt"
  complete "def fibonacci(n):" 64 | tee -a "$OUT/smoke.txt"
  complete "Q: What is 17 * 23? A:" 24 | tee -a "$OUT/smoke.txt"
  exit 0
fi

if [ "$MODE" = quick ]; then  # a short profile pass: few prompts, 64 output tokens
  NPROMPT=4 run quick_1k_64 1024 64 1
  NPROMPT=16 run quick_1k_64 1024 64 16
  exit 0
fi
for c in ${CONCS:-1 4 16 64}; do run chat_1k_256 1024 256 $c; done
for c in ${CONCS_MID:-1 4 16}; do run mid_4k_512 4096 512 $c; done
for c in ${CONCS_LONG:-1 4}; do run prefill_16k_128 16384 128 $c; done
echo SWEEP_DONE
