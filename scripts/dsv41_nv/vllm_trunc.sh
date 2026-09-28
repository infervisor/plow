#!/usr/bin/env bash
# vllm_trunc.sh <nlayers> <tag>: vLLM TP4 on the first N layers of V4.1 (same flags as run_baseline.sh),
# chat 1k/256 at c=1 and c=4. Run under gpulease -n 4.
set -uo pipefail
NL=${1:?nlayers}
TAG=${2:?tag}
ROOT=/root/dsv41
VENV=/root/tts-work/venv-vllm/bin
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export CUDA_HOME=/root/dsv41/cuda_home PLOW_VTRUNC_LAYERS=$NL PYTHONPATH=$(cd "$(dirname "$0")" && pwd)/vtrunc_py
export PATH=$CUDA_HOME/bin:$VENV:$PATH
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
PORT=${PORT:-8131}
OUT=$ROOT/results/$TAG
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "$OUT/env.txt"
$VENV/vllm serve $MODEL --port $PORT --tensor-parallel-size 4 \
  --max-model-len 4096 --gpu-memory-utilization ${GMU:-0.1} --language-model-only --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --hf-overrides "{\"num_hidden_layers\": $NL}" \
  > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for i in $(seq 1 900); do
  if curl -sf localhost:$PORT/health >/dev/null; then READY=1; break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "SERVER DIED"; tail -40 "$OUT/server.log"; exit 3; fi
  sleep 3
done
[ "${READY:-0}" = 1 ] || { echo "SERVER NOT READY"; exit 4; }
echo "server ready after ~$((i*3))s"
for c in 1 4; do
  $VENV/vllm bench serve --model $MODEL --port $PORT --backend openai --endpoint /v1/completions \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --ignore-eos --temperature 0 --num-prompts 16 --max-concurrency $c --num-warmups 2 --seed 1 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "chat_c${c}.json" > "$OUT/chat_c${c}.log" 2>&1
  echo "[c=$c] rc=$?"
  grep -E "Median TTFT|Median TPOT|Output token throughput" "$OUT/chat_c${c}.log"
done
echo DONE
