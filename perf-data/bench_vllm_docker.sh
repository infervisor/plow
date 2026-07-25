#!/usr/bin/env bash
# =============================================================================
# bench_vllm_docker.sh — Reproducible vLLM-in-Docker single-user baseline sweep
# =============================================================================
# Measures TTFT (prefill), TPOT/ITL (decode-per-token), and derived prefill/
# decode throughput for a served model on ONE AMD GPU, batch 1 (single user),
# across a range of input context lengths. Output 128 tokens per request.
#
# This is the EXACT harness — re-run it verbatim to reproduce the committed
# perf-data/*-vllm-docker.json results.
#
# -----------------------------------------------------------------------------
# DOCKER ACCESS (recorded working method on this host, 2026-07-15):
#   The invoking user is NOT in the `docker` group, so a bare `docker ...` gives
#   "permission denied ... /var/run/docker.sock". The host grants PASSWORDLESS
#   sudo, so every docker call is prefixed with `sudo -n`. Verify with:
#       sudo -n docker ps          # must print the container table, no prompt
#   Override with  DOCKER="docker"  if your user is in the docker group.
#
# GPU SELECTION:
#   AMD ROCm/HIP device is chosen via HIP_VISIBLE_DEVICES (NOT --gpus). Here we
#   pin GPU index 7 (all 8 gfx950 / MI350X on this host were idle; verify with
#   `rocm-smi --showuse`). The container gets ALL DRI/KFD nodes via --device,
#   and HIP_VISIBLE_DEVICES restricts vLLM to the one physical GPU.
#
# IMAGE: rocm/vllm:latest  ->  vLLM 0.11.2.dev673+g839868462.rocm700
#        (confirm present:  sudo -n docker images | grep vllm ; pull if absent:
#         sudo -n docker pull rocm/vllm:latest)
#
# PARITY WITH plow:
#   --dtype bfloat16              bf16, no quantization
#   --max-num-batched-tokens 8192 chunked-prefill token budget = 8192
#   --tensor-parallel-size 1      single GPU
#   --max-concurrency 1           batch 1, single user
#   CUDA/HIP graphs: DEFAULT (NOT --enforce-eager), i.e. graph capture ON.
# =============================================================================
set -uo pipefail

# ---- knobs (override via env) ----------------------------------------------
DOCKER="${DOCKER:-sudo -n docker}"
IMAGE="${IMAGE:-rocm/vllm:latest}"
GPU="${GPU:-7}"                       # HIP_VISIBLE_DEVICES index (comma-sep for TP>1)
TP="${TP:-1}"                         # --tensor-parallel-size
HOST_MODELS="${HOST_MODELS:-/home/lava/models}"
PORT="${PORT:-8000}"
CNAME="${CNAME:-vllm_bench}"
OUTDIR="${OUTDIR:-$(dirname "$0")/vllm_docker_logs}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

# ---- quantization (fp8 baseline) -------------------------------------------
# QUANT=bf16 (default) reproduces the committed bf16 baseline. QUANT=fp8 adds
# --quantization fp8 (weight-only dynamic e4m3 from the bf16 checkpoint — no
# pre-quantized weights needed). KVFP8=1 additionally quantizes the KV cache to
# fp8 (--kv-cache-dtype fp8), targeting the decode KV-read bandwidth.
QUANT="${QUANT:-bf16}"
KVFP8="${KVFP8:-0}"
QUANT_ARGS=""
[ "$QUANT" = "fp8" ] && QUANT_ARGS="--quantization fp8"
[ "$KVFP8" = "1" ]   && QUANT_ARGS="$QUANT_ARGS --kv-cache-dtype fp8"
QTAG="$QUANT"; [ "$KVFP8" = "1" ] && QTAG="${QUANT}kv"

mkdir -p "$OUTDIR"

# ---- models: "modeldir:serve_max_model_len:ctx1,ctx2,..." ------------------
# serve_max_model_len is the --max-model-len passed to `vllm serve`. It MUST be
# >= max(swept ctx) + OUTPUT_LEN, otherwise the largest point's prompt+output
# overruns the context window and the server rejects it with HTTP 400 Bad
# Request (the initial single-prompt probe then aborts the whole point). So we
# set it to each model's NATIVE cap, which leaves headroom past the top ctx.
#
# Qwen3-4B  : max_position_embeddings=40960 -> 64K is NOT native. Sweep tops out
#             at 32768 (native); serve at 40960 so 32768+128 fits. NO rope
#             extension is applied (rope_scaling=null in the model config).
# Llama-3.1 : max_position_embeddings=131072 -> 64K native. Sweep to 65536;
#             serve at 66560 (=65536+1024 headroom) so 65536+128 fits. Matches
#             the committed in-process baseline's max_model_len.
MODELS=(
  "Qwen3-4B:40960:1024,4096,8192,16384,32768"
  "gemma-4-31B-it:131072:1024,4096,8192,16384,32768"
  "Llama-3.1-8B-Instruct:66560:1024,4096,8192,16384,32768,65536"
)
# MODELS_OVERRIDE="Qwen3-4B:40960:1024,4096" runs a single model — lets two invocations
# (distinct GPU/CNAME/PORT/OUTDIR) sweep different models in parallel without editing this array.
[ -n "${MODELS_OVERRIDE:-}" ] && MODELS=("$MODELS_OVERRIDE")

serve() {  # $1=model dir  $2=max_model_len
  local model="$1" maxlen="$2"
  echo ">>> serving /models/$model  (max-model-len=$maxlen) on GPU $GPU"
  $DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
  # --entrypoint vllm normalizes across images: rocm/vllm:latest (entrypoint
  # null, cmd /bin/bash) -> runs `vllm serve ...`; vllm/vllm-openai-rocm:latest
  # (entrypoint ["vllm","serve"]) -> override to just `vllm`, then `serve ...`.
  # Either way the container runs `vllm serve /models/<M> ...`.
  $DOCKER run --rm -d --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    --ipc=host --shm-size=16g \
    -e HIP_VISIBLE_DEVICES="$GPU" \
    -v "$HOST_MODELS":/models \
    -p "$PORT":8000 \
    --entrypoint vllm \
    "$IMAGE" \
    serve "/models/$model" \
      --dtype bfloat16 $QUANT_ARGS \
      --tensor-parallel-size "$TP" \
      --max-num-batched-tokens 8192 \
      --max-model-len "$maxlen" >/dev/null
}

wait_health() {
  local t=0
  echo ">>> waiting for :$PORT/health (<= ${HEALTH_TIMEOUT}s)"
  while [ "$t" -lt "$HEALTH_TIMEOUT" ]; do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health")" = "200" ]; then
      echo ">>> healthy after ${t}s"; return 0
    fi
    sleep 10; t=$((t+10))
  done
  echo "!!! endpoint never became healthy"; $DOCKER logs "$CNAME" 2>&1 | tail -30; return 1
}

sanity() {  # $1=model dir — model must answer coherently before we trust timings
  local model="$1" out
  out=$(curl -s "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"/models/$model\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}")
  echo ">>> sanity[completions]: The capital of France is =>$(echo "$out" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null)"
  # Instruction-tuned models (e.g. Gemma-it) degenerate on raw completion; the
  # chat endpoint with the model's template is the real coherence gate.
  local cout
  cout=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"/models/$model\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":12,\"temperature\":0}")
  echo ">>> sanity[chat]: capital of France? =>$(echo "$cout" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)"
}

bench_point() {  # $1=model dir  $2=input_len ; writes log, echoes CSV row
  local model="$1" L="$2" log="$OUTDIR/${1//\//_}_in${2}.log"
  $DOCKER exec "$CNAME" vllm bench serve \
    --model "/models/$model" --dataset-name random \
    --random-input-len "$L" --random-output-len "$OUTPUT_LEN" \
    --max-concurrency 1 --num-prompts "$NUM_PROMPTS" --port 8000 \
    > "$log" 2>&1
  # derive: prefill_tok_s = L / (ttft_ms/1000) ; decode_tok_s = 1000/tpot_ms
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

VER=""

for entry in "${MODELS[@]}"; do
  IFS=':' read -r model maxlen ctxs <<<"$entry"
  serve "$model" "$maxlen" || { echo "serve failed for $model"; continue; }
  wait_health || { $DOCKER rm -f "$CNAME" >/dev/null 2>&1; continue; }
  [ -z "${VER:-}" ] && VER=$($DOCKER exec "$CNAME" vllm --version 2>/dev/null | tail -1)
  echo ">>> config: QUANT=$QUANT KVFP8=$KVFP8  (serve args: --dtype bfloat16 $QUANT_ARGS)"
  sanity "$model"
  csv="$OUTDIR/${model//\//_}_${QTAG}_summary.csv"
  echo "input_len,ttft_ms,prefill_tok_s,tpot_ms,itl_ms,decode_tok_s" > "$csv"
  IFS=',' read -ra L_ARR <<<"$ctxs"
  for L in "${L_ARR[@]}"; do
    echo ">>> bench $model input_len=$L output_len=$OUTPUT_LEN"
    row=$(bench_point "$model" "$L"); echo "    $row"; echo "$row" >> "$csv"
  done
  echo ">>> done $model ; summary: $csv"
  $DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
done

echo ">>> vLLM version: $VER"
echo ">>> all logs + summaries in: $OUTDIR"
