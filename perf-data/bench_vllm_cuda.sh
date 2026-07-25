#!/usr/bin/env bash
# =============================================================================
# bench_vllm_cuda.sh — vLLM single-user baseline sweep, NATIVE (venv) on NVIDIA
# =============================================================================
# CUDA/sm_120 counterpart of `bench_vllm_docker.sh` (which is ROCm-in-docker).
# Same methodology so the numbers are directly comparable to the committed AMD
# baselines in perf-data/:
#
#   --max-concurrency 1           batch 1, single user
#   --tensor-parallel-size 1      single GPU
#   --max-num-batched-tokens 8192 chunked-prefill token budget = 8192
#   --random-output-len 128       128 output tokens / request
#   --num-prompts 3               3 prompts / point, random dataset
#   CUDA graphs: DEFAULT (NOT --enforce-eager), i.e. graph capture ON.
#
# Headline metric: tpot_ms (Mean TPOT = decode ms/token, lower is better).
# TTFT is recorded too but decode is the campaign target.
#
# -----------------------------------------------------------------------------
# DIFFERENCES vs bench_vllm_docker.sh (and why):
#   * No docker. vLLM runs from a venv on /workspace (the host `/` overlay is
#     only 30 GB; a vLLM install + torch does not fit there).
#   * GPU is shared with other agents on this box, so this script benches
#     exactly ONE config per invocation and is meant to be wrapped in a single
#     `gpulease` hold covering serve -> bench -> shutdown. It refuses to leave a
#     server behind: the EXIT trap kills it and waits for VRAM to be released.
# -----------------------------------------------------------------------------
# USAGE (one lease per config):
#   export GPU_LEASE_TIMEOUT=3600
#   gpulease gemma-12b env QUANT=bf16 KVFP8=0 \
#     MODEL_DIR=/workspace/models/gemma-4-12B-it \
#     CTXS=4096,8192,16384 MAXLEN=16512 GPU_UTIL=0.95 \
#     bash perf-data/bench_vllm_cuda.sh
# =============================================================================
set -uo pipefail

VENV="${VENV:-/workspace/venvs/vllm}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/gemma-4-12B-it}"
TAG="${TAG:-$(basename "$MODEL_DIR")}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
PORT="${PORT:-8000}"
OUTDIR="${OUTDIR:-$(dirname "$0")/vllm_cuda_logs}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
CTXS="${CTXS:-4096,8192,16384}"
MAXLEN="${MAXLEN:-33792}"
GPU_UTIL="${GPU_UTIL:-0.90}"

# ---- quantization config ----------------------------------------------------
QUANT="${QUANT:-bf16}"
KVFP8="${KVFP8:-0}"
QUANT_ARGS=""
[ "$QUANT" = "fp8" ] && QUANT_ARGS="--quantization fp8"
[ "$KVFP8" = "1" ]   && QUANT_ARGS="$QUANT_ARGS --kv-cache-dtype fp8"
QTAG="$QUANT"; [ "$KVFP8" = "1" ] && QTAG="${QUANT}kv"

export PATH=/usr/local/cuda-12.8/bin:$PATH
export VLLM_LOGGING_LEVEL=INFO
mkdir -p "$OUTDIR"

SRVLOG="$OUTDIR/serve_${QTAG}.log"
SRVPID=""

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

cleanup() {
  if [ -n "$SRVPID" ]; then
    echo ">>> shutting down vLLM (pid $SRVPID)"
    kill -TERM "$SRVPID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 2; done
    kill -KILL "$SRVPID" 2>/dev/null
    pkill -f "vllm serve $MODEL_DIR" 2>/dev/null
    pkill -f "VLLM::EngineCore" 2>/dev/null
  fi
  # Do not release the lease until the card is actually free for the next agent.
  for _ in $(seq 1 30); do
    u=$(vram_used); echo ">>> VRAM used: ${u} MiB"
    [ "$u" -lt 1000 ] && break
    sleep 3
  done
}
trap cleanup EXIT

# ---- serve ------------------------------------------------------------------
echo ">>> config: QUANT=$QUANT KVFP8=$KVFP8 (serve args: --dtype bfloat16 $QUANT_ARGS)"
echo ">>> serving $MODEL_DIR (max-model-len=$MAXLEN, gpu-util=$GPU_UTIL)"
"$VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name bench \
  --dtype bfloat16 $QUANT_ARGS \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAXLEN" $EXTRA_SERVE_ARGS \
  --port "$PORT" > "$SRVLOG" 2>&1 &
SRVPID=$!

t=0
echo ">>> waiting for :$PORT/health (<= ${HEALTH_TIMEOUT}s)"
while [ "$t" -lt "$HEALTH_TIMEOUT" ]; do
  if ! kill -0 "$SRVPID" 2>/dev/null; then
    echo "!!! server process died during startup; last 40 log lines:"; tail -40 "$SRVLOG"; exit 1
  fi
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health")" = "200" ]; then
    echo ">>> healthy after ${t}s"; break
  fi
  sleep 5; t=$((t+5))
done
if [ "$t" -ge "$HEALTH_TIMEOUT" ]; then
  echo "!!! endpoint never became healthy"; tail -40 "$SRVLOG"; exit 1
fi

# ---- VRAM + KV-cache headroom ----------------------------------------------
echo ">>> VRAM used after load: $(vram_used) MiB / $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MiB"
grep -iE "GPU KV cache size|Available KV cache memory|model weights take|Maximum concurrency" "$SRVLOG" | tail -5

# ---- correctness gate: coherent chat output --------------------------------
echo ">>> sanity[chat]: What is the capital of France?"
curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"bench","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}' \
  | python3 -c 'import sys,json;print(">>> GENERATED:",repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))' 2>&1

# ---- sweep ------------------------------------------------------------------
csv="$OUTDIR/${TAG}_${QTAG}_summary.csv"
echo "input_len,ttft_ms,prefill_tok_s,tpot_ms,itl_ms,decode_tok_s" > "$csv"
IFS=',' read -ra L_ARR <<<"$CTXS"
for L in "${L_ARR[@]}"; do
  echo ">>> bench input_len=$L output_len=$OUTPUT_LEN"
  log="$OUTDIR/${TAG}_${QTAG}_in${L}.log"
  "$VENV/bin/vllm" bench serve \
    --model "$MODEL_DIR" --served-model-name bench --dataset-name random \
    --random-input-len "$L" --random-output-len "$OUTPUT_LEN" \
    --max-concurrency 1 --num-prompts "$NUM_PROMPTS" --port "$PORT" \
    > "$log" 2>&1
  row=$(python3 - "$L" "$log" <<'PY'
import re,sys
L=int(sys.argv[1]); t=open(sys.argv[2]).read()
def g(p):
    m=re.search(p+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
ttft=g(r"Mean TTFT \(ms\):"); tpot=g(r"Mean TPOT \(ms\):"); itl=g(r"Mean ITL \(ms\):")
pf = L/(ttft/1000.0) if ttft==ttft and ttft>0 else float('nan')
dc = 1000.0/tpot if tpot==tpot and tpot>0 else float('nan')
print(f"{L},{ttft:.2f},{pf:.1f},{tpot:.3f},{itl:.3f},{dc:.2f}")
PY
)
  echo "    $row"; echo "$row" >> "$csv"
done

echo ">>> vLLM version: $("$VENV/bin/vllm" --version 2>/dev/null | tail -1)"
echo ">>> done; summary: $csv"
