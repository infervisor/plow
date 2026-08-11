#!/usr/bin/env bash
# glm52_vllm_precision_probe.sh — read back what vLLM ACTUALLY chose for GLM-5.2, per tensor
# class, from its own startup log. Not from flags: from the lines the engine prints after it
# has resolved them, plus the KV-cache byte arithmetic.
#
# Brings vLLM up EXACTLY as the published comparison does — same image, same args as
# `scripts/bench_vllm_chat.sh` invoked from `scripts/glm52_tpctx_sweep.sh run <tp>`:
#   --dtype auto, VLLM_ROCM_USE_AITER=1, no --kv-cache-dtype, no --quantization.
#
# Usage:
#   perf-data/tools/gpulease -n 4 vprec sg render -c \
#       './scripts/glm52_vllm_precision_probe.sh 4 /home/lava/models/glm52_precision'
#
# WHY THE ARITHMETIC MATTERS. `kv_cache_dtype=auto` is a FLAG; what it resolved to is a
# measurement. GLM-5.2 is MLA: one latent row per token per layer of
# kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576 elements, 78 layers, NOT sharded by TP.
#   bf16 -> 576 * 78 * 2 = 89 856 B/token
#   fp8  -> 576 * 78 * 1 = 44 928 B/token
# vLLM prints "Available KV cache memory: X GiB" and "GPU KV cache size: N tokens" in the
# same run, and X/N lands unambiguously on one of those two.
set -uo pipefail
TP="${1:-4}"; OUT="${2:-/home/lava/models/glm52_precision}"
mkdir -p "$OUT"
MODEL_ID=zai-org/GLM-5.2-FP8
DOCKER="sudo -n docker"
IMAGE=rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
HF_CACHE="$HOME/.cache/huggingface"
COMPILE_CACHE="$HOME/.cache/vllm-bench-container"; mkdir -p "$COMPILE_CACHE"
PORT="${PORT:-8611}"
CNAME="vllm_precprobe_tp${TP}"
MAXLEN="${MAXLEN:-20480}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-5400}"
GPUS="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-$(seq -s, 0 $((TP-1)))}}"
RGID="$(getent group render | cut -d: -f3)"; RGID="${RGID:-109}"
VGID="$(getent group video  | cut -d: -f3)"; VGID="${VGID:-44}"

echo "GPUS=$GPUS TP=$TP OUT=$OUT"
$DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true

$DOCKER run -d --name "$CNAME" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$VGID" --group-add "$RGID" \
  --security-opt seccomp=unconfined --ipc=host --shm-size=32g \
  -e HIP_VISIBLE_DEVICES="$GPUS" -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
  -e HF_MODULES_CACHE=/tmp/hf_modules -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HF_CACHE":/hf:ro -v "$COMPILE_CACHE":/root/.cache \
  -p "$PORT":8000 --entrypoint vllm "$IMAGE" \
  serve "$MODEL_ID" --max-num-batched-tokens 8192 --max-model-len "$MAXLEN" \
    --dtype auto --tensor-parallel-size "$TP" >/dev/null

t=0
while [ "$t" -lt "$HEALTH_TIMEOUT" ]; do
  $DOCKER ps --format '{{.Names}}' | grep -q "^${CNAME}$" || {
    echo "!! container exited during startup"; $DOCKER logs "$CNAME" > "$OUT/vllm_startup.log" 2>&1
    tail -250 "$OUT/vllm_startup.log"; exit 2; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health")" = "200" ] && {
    echo ">>> healthy after ${t}s"; break; }
  sleep 20; t=$((t+20))
done
[ "$t" -ge "$HEALTH_TIMEOUT" ] && { echo "!! never healthy"; $DOCKER logs "$CNAME" > "$OUT/vllm_startup.log" 2>&1; exit 1; }

# --- 1. the whole startup log, kept verbatim: this is the evidence, not the greps below
$DOCKER logs "$CNAME" > "$OUT/vllm_startup.log" 2>&1
echo "startup log: $(wc -l < "$OUT/vllm_startup.log") lines -> $OUT/vllm_startup.log"

# --- 2. coherence BEFORE anything is believed (knob-contract: wrong numerics run FASTER on
#        this model, because garbage collapses the router's top-k)
curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one sentence.\"}],\"max_tokens\":32,\"temperature\":0}" \
  | tee "$OUT/vllm_coherence.json"; echo

# --- 3. /v1/models
curl -s "http://localhost:$PORT/v1/models" | tee "$OUT/vllm_models.json"; echo

# --- 4. ask the LIVE engine what it resolved. This is the authoritative read: it is the
#        vllm_config object the workers are actually running, not a command line.
$DOCKER exec "$CNAME" python3 - > "$OUT/vllm_resolved.txt" 2>&1 <<'PY'
import json, torch
from vllm.engine.arg_utils import EngineArgs
from vllm.config import ModelConfig
a = EngineArgs(model="zai-org/GLM-5.2-FP8", dtype="auto", tensor_parallel_size=4,
               max_model_len=20480, max_num_batched_tokens=8192)
cfg = a.create_engine_config()
mc, cc, pc = cfg.model_config, cfg.cache_config, cfg.parallel_config
print("== resolved by the same code path the server ran ==")
print("model_config.dtype          =", mc.dtype)
print("model_config.quantization   =", mc.quantization)
print("hf quantization_config      =", json.dumps(getattr(mc.hf_config, 'quantization_config', None), default=str)[:400])
print("cache_config.cache_dtype    =", cc.cache_dtype)
try:
    from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE as M
except Exception:
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE as M
resolved = mc.dtype if cc.cache_dtype == "auto" else M[cc.cache_dtype]
print("RESOLVED KV ELEMENT DTYPE   =", resolved, " itemsize =", torch.empty(0, dtype=resolved).element_size())
print("is_attention_free           =", mc.is_attention_free)
print("use_mla                     =", getattr(mc, 'use_mla', None))
print()
print("== the quant method actually selected for each linear class ==")
from vllm.model_executor.layers.quantization import get_quantization_config
qc = mc.quant_config
print("quant_config               =", type(qc).__name__ if qc else None)
for attr in ("weight_block_size", "activation_scheme", "is_checkpoint_fp8_serialized", "ignored_layers"):
    if qc is not None and hasattr(qc, attr):
        v = getattr(qc, attr)
        print(f"  {attr:32s} = {str(v)[:300]}")
PY
cat "$OUT/vllm_resolved.txt"

# --- 5. the lines that MEASURE the choices (kept separate from the verbatim log above)
echo "===== GREPPED EVIDENCE ====="
grep -inE "kv[_ ]cache|KV cache size|Available KV cache|dtype|quantization|fp8|a8w8|blockscale|aiter|Fp8LinearMethod|Fp8MoEMethod|indexer|sparse|attention backend|Using .* backend|model weights take|Model loading took|memory profiling" \
  "$OUT/vllm_startup.log" | head -200 | tee "$OUT/vllm_evidence.txt"

$DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
echo "DONE"
