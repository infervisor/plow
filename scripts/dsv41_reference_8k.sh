#!/usr/bin/env bash
# DeepSeek-V4.1-Flash, SHIPPED REFERENCE implementation, 8k prefill on 8x MI300X.
#
# This is not plow. It runs `inference/generate.py` from the checkpoint itself,
# and exists for two reasons:
#
#   1. it is the first end-to-end execution of V4.1 available on this box at
#      all -- no vLLM here registers `DeepseekV41ForCausalLM`, and ROCm/ATOM
#      has no V4.1 path either (docs/amd/deepseek-v41-flash-mi300x.md §4);
#   2. Stage 5 of docs/bringup needs an ORACLE, and the reference is it. A
#      block sweep has nothing to compare against until this runs.
#
# It is a REFERENCE, so its latency is a ceiling, not the 300 ms target: it is
# eager PyTorch with tilelang kernels, no CUDA graphs, no paged KV, batch 1.
#
#   ./scripts/dsv41_reference_8k.sh convert      # CPU only, no lease, ~475 GB RAM
#   ./scripts/dsv41_reference_8k.sh run          # takes an 8-GPU lease
#
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF="${DSV41_HF:-/workspace/models/DeepSeek-V4.1-Flash}"
TP8="${DSV41_TP8:-/workspace/models/DeepSeek-V4.1-Flash-TP8}"
PROMPT="${DSV41_PROMPT:-/workspace/models/dsv41-8k.txt}"
OUT="${DSV41_OUT:-$WT/build-dsv41-ref}"
LEASE="$WT/perf-data/tools/gpulease"
MP="${MP:-8}"
NEW_TOKENS="${NEW_TOKENS:-16}"

# The reference's torch links nix glibc/gcc and ROCm off the default loader
# path; without this `import torch` dies on libffi.so.8 / libroctx64.so.4.
find_vllm_root () {
  local d="$WT"
  while [ "$d" != "/" ]; do
    [ -x "$d/.venv-vllm028/bin/python" ] && { echo "$d"; return 0; }
    d="$(dirname "$d")"
  done
  return 1
}
ROOT="${VLLM_ROOT:-$(find_vllm_root)}"
[ -n "${ROOT:-}" ] || { echo "!! no .venv-vllm028 at or above $WT"; exit 1; }
PY="${VLLM_PYTHON:-$ROOT/build-gemma31/vllm-python}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm/core-7.14}"
export VLLM_ROCM_LIB="${VLLM_ROCM_LIB:-$ROCM_PATH/lib}"

case "${1:?convert|run|check}" in

# ---------------------------------------------------------------- CHECK
check)
  echo "hf ckpt : $HF"
  ls "$HF"/*.safetensors 2>/dev/null | wc -l | sed 's/^/  shards: /'
  "$PY" - <<'PY'
import json, os
hf = os.environ.get("DSV41_HF", "/workspace/models/DeepSeek-V4.1-Flash")
idx = json.load(open(f"{hf}/model.safetensors.index.json"))["weight_map"]
want = set(idx.values())
have = {f for f in want if os.path.exists(f"{hf}/{f}")}
print(f"  files  : {len(have)}/{len(want)} present")
print("  COMPLETE" if have == want else f"  MISSING : {sorted(want - have)[:5]}")
PY
  ;;

# ---------------------------------------------------------------- CONVERT (CPU)
# convert.py holds every rank's state dict in RAM at once, so this needs about
# as much memory as the checkpoint is big. It also asserts the source is
# complete, which is the point: a mid-download tree fails here, loudly, rather
# than producing short shards that only fail at load.
convert)
  [ -d "$HF" ] || { echo "!! $HF missing"; exit 1; }
  mkdir -p "$OUT"
  echo "converting $HF -> $TP8 (mp=$MP), log $OUT/convert.log"
  "$PY" "$HF/inference/convert.py" \
    --hf-ckpt-path "$HF" --save-path "$TP8" --model-parallel "$MP" \
    2>&1 | tee "$OUT/convert.log" | tail -5
  ;;

# ---------------------------------------------------------------- RUN (leases MP GPUs)
run)
  [ -f "$PROMPT" ] || { echo "!! prompt $PROMPT missing"; exit 1; }
  for r in $(seq 0 $((MP - 1))); do
    [ -f "$TP8/model${r}-mp${MP}.safetensors" ] || {
      echo "!! $TP8 missing model${r}-mp${MP}.safetensors — run 'convert' first"; exit 1; }
  done
  mkdir -p "$OUT"
  echo "8k reference run, mp=$MP, log $OUT/run.log"
  exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-43200}" "$LEASE" -n "$MP" "dsv41-ref-8k" \
    env ROCM_PATH="$ROCM_PATH" VLLM_ROCM_LIB="$VLLM_ROCM_LIB" \
        PYTHONPATH="$HF/inference" HF_HUB_OFFLINE=1 \
    "$PY" -m torch.distributed.run --nproc-per-node "$MP" \
      "$HF/inference/generate.py" \
      --ckpt-path "$TP8" --config "$HF/inference/config.json" \
      --input-file "$PROMPT" --max-new-tokens "$NEW_TOKENS" --temperature 0
  ;;

# ---------------------------------------------------------------- BENCH (leases MP GPUs)
# Same load path as `run`, but times prefill and decode separately. The
# reference's generate() reports no timing and is the Stage-5 oracle, so it is
# imported rather than edited.
bench)
  [ -f "$PROMPT" ] || { echo "!! prompt $PROMPT missing"; exit 1; }
  for r in $(seq 0 $((MP - 1))); do
    [ -f "$TP8/model${r}-mp${MP}.safetensors" ] || {
      echo "!! $TP8 missing model${r}-mp${MP}.safetensors — run 'convert' first"; exit 1; }
  done
  mkdir -p "$OUT"
  exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-43200}" "$LEASE" -n "$MP" "dsv41-ref-bench-8k" \
    env ROCM_PATH="$ROCM_PATH" VLLM_ROCM_LIB="$VLLM_ROCM_LIB" \
        PYTHONPATH="$HF/inference" HF_HUB_OFFLINE=1 \
    "$PY" -m torch.distributed.run --nproc-per-node "$MP" \
      "$WT/scripts/dsv41_ref_bench.py" \
      --ckpt-path "$TP8" --config "$HF/inference/config.json" \
      --prompt-file "$PROMPT" --prefill-len "${PREFILL:-8192}" \
      --decode-steps "${DECODE_STEPS:-8}" \
      --json-out "$OUT/ref-8k.json"
  ;;

*) echo "usage: $0 {check|convert|run|bench}"; exit 2 ;;
esac
