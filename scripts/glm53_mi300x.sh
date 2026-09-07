#!/usr/bin/env bash
# GLM-5.3-FP8 on 8x MI300X (gfx942) — emit / serve / bench at TP4 and TP8.
#
# GLM-5.3 is `model_type: glm_moe_dsa`, structurally IDENTICAL to GLM-5.2-FP8 (same 118,629
# tensor names, same dims; config.json differs only in transformers_version), so it serves on
# the existing `glm_main` emit path with no devgen change. Precision is native block-fp8 e4m3
# with a [128,128] `weight_scale_inv` grid, so PLOW_FP8=1 selects MoeEnc::Fp8Blk and no
# precision FLAG is inferred — the checkpoint's quantization_config is the authority.
#
#   ./scripts/glm53_mi300x.sh emit 8            # no GPU, no lease
#   ./scripts/glm53_mi300x.sh serve 8 8100      # takes an N-GPU lease itself
#   ./scripts/glm53_mi300x.sh bench 8 8100      # client only, no lease (server holds it)
#   ./scripts/glm53_mi300x.sh vllm 8 8200       # vLLM 0.28 reference, own lease
#
# EVERY GPU process goes through perf-data/tools/gpulease. The client does not (no GPU).
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${PLOW_CKPT:-/workspace/models/GLM-5.3-plow-lite}"     # PREPPED dir (derived bf16 + fp8 experts)
RAW="${GLM_RAW:-/workspace/models/GLM-5.3-FP8}"              # raw HF: vLLM + tokenizer
OBJ="${PLOW_HSACO:-$WT/build-glm53/hsaco}"
OUT="${GLM53_DIR:-$WT/build-glm53}"
BIN="${PLOW_BIN_DIR:-$WT/target-glm53/release}"
MAXCTX="${MAXCTX:-18432}"
LADDER="${LADDER:-full:128,512,2048,8192}"
BATCH_LADDER="${BATCH_LADDER:-1,2,4}"
LEASE="$WT/perf-data/tools/gpulease"
bundle () { echo "$OUT/tp$1"; }

case "${1:?emit|serve|bench|vllm|smoke}" in

# ---------------------------------------------------------------- EMIT (no GPU)
emit)
  tp="${2:?tp}"; b="$(bundle "$tp")"; mkdir -p "$b"
  # The RECIPE-gfx942 knob set from the GLM-5.2 campaign, minus PLOW_GLM_FUSE_QNORM: its M>1
  # staging is unvalidated and the emit refuses it alongside a decode-batch ladder.
  env GLM_FULL=1 PLOW_FP8=1 \
      PLOW_MLA_PREFILL="$LADDER" PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=2 PLOW_GLM_DSA=0 \
      GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 \
      PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1 PLOW_GLM_FUSE_B1=1 \
      PLOW_MOE_PF_DET=1 \
      PLOW_DECODE_BATCH_LADDER="$BATCH_LADDER" \
    "$BIN/plowc" --hf-dir "$CKPT" --emit devblob --gpu MI300X --arch gfx942 \
      --max-ctx "$MAXCTX" --n-cu "${NCU:-0}" --num-gpus "$tp" --out "$b/model.pkt" || exit 1
  # `checkpoint` MUST be the PREPPED dir: the blob binds derived.q_absorb/v_absorb/kv_a_latent
  # and bf16 q_a_proj/o_proj/shared_experts, which only the prep writes. A raw-HF symlink either
  # refuses on a shard-size mismatch or (with folds armed) faults the GPU at load.
  ln -sfn "$CKPT" "$b/checkpoint"
  ln -sfn "$OBJ"  "$b/hsaco"
  # EVERY tokenizer-side file, not just tokenizer.json. The bundle used to carry
  # tokenizer.json alone, which meant:
  #   * `tokenizer_config.json` was absent, so the Qwen2 pre-tokenizer fix-up in
  #     `text/tokenizer.rs` could never fire for a Qwen-family bundle built this
  #     way, and it would have tokenized against the raw regex and silently
  #     mismatched the reference;
  #   * `generation_config.json` was absent, so the eos set came from the
  #     `config.json` FALLBACK — identical for GLM-5.3 by luck, not by design;
  #   * `chat_template.jinja` was absent, which nothing reads today but which
  #     any template-driven prompt build will need.
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
    [ -e "$RAW/$f" ] && ln -sfn "$RAW/$f" "$b/$f"
  done
  # plowc --emit devblob writes model.pkt + build.json but NOT weights.json, which
  # `plowrt serve` opens unconditionally.
  cat > "$b/weights.json" <<JSON
{ "network": "glm-5.3", "gpu": "mi300x", "num_gpus": $tp, "parallel": "tp",
  "weight_shared": false, "weight": null, "kv": null, "fusion": null, "buckets": [],
  "static_tensors": [], "static_tensors_file_emitted": false, "weight_tiling": null }
JSON
  ls -l "$b/model.pkt" | awk '{printf "   model.pkt %.1f MB\n", $5/1048576}'
  ;;

# ---------------------------------------------------------------- SERVE (leases tp GPUs)
serve)
  tp="${2:?tp}"; port="${3:?port}"; b="$(bundle "$tp")"
  [ -f "$b/model.pkt" ] || { echo "!! $b/model.pkt missing — run 'emit $tp' first"; exit 1; }
  grep -aq libhsa-runtime64 "$BIN/plowrt" || { echo "!! plowrt has no HSA — it would serve CPU garbage"; exit 1; }
  exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-7200}" "$LEASE" -n "$tp" "glm53-tp$tp" \
    nix develop "$WT" --command bash "$WT/scripts/glm53_serve_inner.sh" "$b" "$port" "$OBJ" "$BIN"
  ;;

# ---------------------------------------------------------------- SMOKE (readiness + coherence)
smoke)
  port="${2:?port}"
  for i in $(seq 1 "${SMOKE_TIMEOUT:-900}"); do
    curl -sf --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
  m=$(curl -s "http://127.0.0.1:$port/v1/models" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  [ -n "$m" ] || { echo "!! never ready on $port"; exit 4; }
  echo "model: $m"
  curl -s --max-time 600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":48,\"temperature\":0}"
  echo
  ;;

# ---------------------------------------------------------------- BENCH (client only, no GPU)
bench)
  tp="${2:?tp}"; port="${3:?port}"; label="${4:-plow}"
  res="$OUT/bench/$label-tp$tp"; mkdir -p "$res"
  m=$(curl -s "http://127.0.0.1:$port/v1/models" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  [ -n "$m" ] || { echo "!! no server on $port"; exit 1; }
  for inlen in ${IN_LENS:-1024 4096}; do
    for conc in ${CONCS:-1 4}; do
      echo "===== $label tp$tp in=$inlen conc=$conc out=${OUTLEN:-128}"
      env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 \
        LD_LIBRARY_PATH="/nix/store/8kvxvr3pmsypxiypq4g8zy13glnfr7nx-glibc-2.42-67/lib:/nix/store/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib:/opt/rocm/core-7.14/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu" \
        "$WT/.venv-vllm028/bin/python" -m vllm.entrypoints.cli.main bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$port" --model "$m" --tokenizer "$RAW" \
        --dataset-name random --random-input-len "$inlen" --random-output-len "${OUTLEN:-128}" \
        --num-prompts "${NPROMPT:-8}" --max-concurrency "$conc" \
        --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$res" \
        --result-filename "in${inlen}_c${conc}.json" 2>&1 | tail -32
    done
  done
  ;;

# ---------------------------------------------------------------- vLLM 0.28 reference (own lease)
# ROCM_PATH/ROCM_HOME are load-bearing above TP1 and only above TP1: each multiproc worker
# re-derives the ROCm version at init_device and reads <root>/.info/version. /opt/rocm has no
# .info directory on this box, so every TP4 worker died with "RuntimeError: ROCm version file
# not found" while the TP1 Gemma runs never noticed. /opt/rocm/core-7.14/.info/version is 7.14.0.
# GPU_MEM_UTIL matters at TP4 and only there: the raw checkpoint is 703.7 GiB, so a TP4 rank
# holds 175.9 GiB = 188.9 GB of a 206.1 GB card. vLLM's default 0.9 budget is 185.5 GB — LESS
# than the weights — so TP4 refuses before it reaches the KV cache. TP8 (94.4 GB/rank) is fine
# at any of these. Also note the ATTENTION ASYMMETRY when reading any result: vLLM serves this
# checkpoint with DSA armed (index_topk 2048) while plow emits PLOW_GLM_DSA=0 and reads EVERY
# KV row, so above ~2k context plow is doing strictly more attention work per token.
vllm)
  tp="${2:?tp}"; port="${3:?port}"
  exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-7200}" "$LEASE" -n "$tp" "vllm028-glm53-tp$tp" \
    env VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib HF_HUB_OFFLINE=1 \
        ROCM_PATH="${VLLM_ROCM_ROOT:-/opt/rocm/core-7.14}" \
        ROCM_HOME="${VLLM_ROCM_ROOT:-/opt/rocm/core-7.14}" \
        VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}" \
    "$WT/build-gemma31/vllm-python" -m vllm.entrypoints.openai.api_server \
      --model "$RAW" --served-model-name glm-5.3 --tensor-parallel-size "$tp" \
      --max-model-len "$MAXCTX" --max-num-seqs "${VLLM_SEQS:-8}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.95}" \
      --no-enable-prefix-caching --trust-remote-code --port "$port"
  ;;

# ---------------------------------------------------------------- STOP (clean teardown)
# Kill the plowrt ITSELF, not the gpulease wrapper. gpulease runs the child and releases the
# lease when it returns; killing the wrapper instead orphans a live plowrt that keeps the port
# and the cards, which is exactly what happened here on the first attempt.
stop)
  pat="${2:-/app/plow/build-glm53/tp}"
  pids=$(pgrep -f "plowrt serve --assets $pat" || true)
  [ -z "$pids" ] && { echo "no plowrt serving $pat"; exit 0; }
  echo "stopping: $pids"
  kill -TERM $pids 2>/dev/null
  for i in $(seq 1 60); do
    pgrep -f "plowrt serve --assets $pat" >/dev/null 2>&1 || break
    sleep 2
  done
  pgrep -f "plowrt serve --assets $pat" >/dev/null 2>&1 && { echo "escalating to KILL"; kill -KILL $pids 2>/dev/null; sleep 5; }
  echo "stopped"
  ;;

*) echo "usage: $0 {emit|serve|smoke|bench|vllm|stop} ..."; exit 2 ;;
esac
