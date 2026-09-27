#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 on 8x MI300X (gfx942) — the vLLM 8k reference number.
#
# This is the STAGE 4-7 target parameter for the DeepSeek-V4.1-Flash bringup
# (docs/bringup/target.md). V4.1 itself cannot be served here: it needs vLLM
# >= 0.30, which ships only as a container, and this box has no container
# runtime. V4-Flash-0731 IS served by the installed vLLM 0.28.0+rocm723 —
# `DeepseekV4ForCausalLM` resolves to its out-of-tree `vllm/models/deepseek_v4`
# package — so it stands in as the closest measurable reference. It is the
# PREVIOUS generation, not the target model; see the caveat in the results
# header that `report` writes.
#
#   ./scripts/dsv4_vllm_8k.sh serve 8 8300      # takes an N-GPU lease itself
#   ./scripts/dsv4_vllm_8k.sh smoke 8300        # readiness + one coherent answer
#   ./scripts/dsv4_vllm_8k.sh bench 8 8300      # client only, no lease
#   ./scripts/dsv4_vllm_8k.sh all 8 8300        # serve in background, smoke, bench, stop
#
# EVERY GPU process goes through perf-data/tools/gpulease. The client does not.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${DSV4_RAW:-/workspace/models/DeepSeek-V4-Flash-0731}"
OUT="${DSV4_DIR:-$WT/build-dsv4-vllm}"
LEASE="$WT/perf-data/tools/gpulease"

# The vLLM venv is a gitignored build artifact, so it lives in the MAIN checkout
# even when this script runs from a worktree. Walk up for it rather than
# assuming $WT has one, and synthesize the LD_LIBRARY_PATH wrapper its python
# needs: the venv links nix glibc/gcc and ROCm, none of which are on the default
# loader path, so a bare `bin/python -c "import torch"` dies on libffi.so.8.
find_vllm_root () {
  local d="$WT"
  while [ "$d" != "/" ]; do
    [ -x "$d/.venv-vllm028/bin/python" ] && { echo "$d"; return 0; }
    d="$(dirname "$d")"
  done
  return 1
}
VLLM_ROOT="${VLLM_ROOT:-$(find_vllm_root)}"
[ -n "${VLLM_ROOT:-}" ] || { echo "!! no .venv-vllm028 found at or above $WT"; exit 1; }

PYWRAP="${VLLM_PYTHON:-}"
if [ -z "$PYWRAP" ] && [ -x "$VLLM_ROOT/build-gemma31/vllm-python" ]; then
  PYWRAP="$VLLM_ROOT/build-gemma31/vllm-python"
fi
if [ -z "$PYWRAP" ]; then
  PYWRAP="$OUT/vllm-python"
  mkdir -p "$OUT"
  glibc=$(ls -d /nix/store/*-glibc-2.42-*/lib 2>/dev/null | head -1)
  gcclib=$(ls -d /nix/store/*-gcc-*-lib/lib 2>/dev/null | head -1)
  [ -n "$glibc" ] && [ -n "$gcclib" ] || { echo "!! no nix glibc/gcc-lib in /nix/store"; exit 1; }
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo 'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES'
    echo "export LD_LIBRARY_PATH=$glibc:$gcclib:\${VLLM_ROCM_LIB:-\${ROCM_PATH:?}/lib}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
    echo "exec \"$VLLM_ROOT/.venv-vllm028/bin/python\" \"\$@\""
  } > "$PYWRAP"
  chmod +x "$PYWRAP"
fi

# 8k is the point of this script: one input length, a concurrency sweep across
# it. OUTLEN stays at the harness default so TPOT is comparable to the other
# campaigns in perf-data/.
IN_LEN="${IN_LEN:-8192}"
OUTLEN="${OUTLEN:-128}"
CONCS="${CONCS:-1 4 16 32}"
NPROMPT="${NPROMPT:-16}"
# Headroom over IN_LEN+OUTLEN: vLLM refuses a request that would exceed
# max-model-len, and the random dataset's lengths are approximate.
MAXCTX="${MAXCTX:-10240}"

case "${1:?serve|smoke|bench|all|stop}" in

# ---------------------------------------------------------------- SERVE (leases tp GPUs)
# ROCM_PATH/ROCM_HOME are load-bearing above TP1: each multiproc worker re-derives
# the ROCm version at init_device from <root>/.info/version, and /opt/rocm has no
# .info on this box. /opt/rocm/core-7.14/.info/version is 7.14.0.
serve)
  tp="${2:?tp}"; port="${3:?port}"
  [ -f "$RAW/config.json" ] || { echo "!! $RAW/config.json missing"; exit 1; }
  exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-21600}" "$LEASE" -n "$tp" "vllm028-dsv4-tp$tp" \
    env VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib HF_HUB_OFFLINE=1 \
        ROCM_PATH="${VLLM_ROCM_ROOT:-/opt/rocm/core-7.14}" \
        ROCM_HOME="${VLLM_ROCM_ROOT:-/opt/rocm/core-7.14}" \
        VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}" \
    "$PYWRAP" -m vllm.entrypoints.openai.api_server \
      --model "$RAW" --served-model-name deepseek-v4-flash --tensor-parallel-size "$tp" \
      --max-model-len "$MAXCTX" --max-num-seqs "${VLLM_SEQS:-32}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
      --no-enable-prefix-caching --trust-remote-code --port "$port"
  ;;

# ---------------------------------------------------------------- SMOKE
smoke)
  port="${2:?port}"
  for i in $(seq 1 "${SMOKE_TIMEOUT:-1800}"); do
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
  tp="${2:?tp}"; port="${3:?port}"
  res="$OUT/bench/tp$tp"; mkdir -p "$res"
  m=$(curl -s "http://127.0.0.1:$port/v1/models" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  [ -n "$m" ] || { echo "!! no server on $port"; exit 1; }
  for conc in $CONCS; do
    echo "===== vllm028 dsv4 tp$tp in=$IN_LEN conc=$conc out=$OUTLEN"
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 \
      VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib ROCM_PATH=/opt/rocm/core-7.14 \
      "$PYWRAP" -m vllm.entrypoints.cli.main bench serve \
      --backend openai-chat --endpoint /v1/chat/completions \
      --base-url "http://127.0.0.1:$port" --model "$m" --tokenizer "$RAW" \
      --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUTLEN" \
      --num-prompts "$NPROMPT" --max-concurrency "$conc" \
      --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$res" \
      --result-filename "in${IN_LEN}_c${conc}.json" 2>&1 | tail -32
  done
  ;;

# ---------------------------------------------------------------- ALL (serve, smoke, bench, stop)
all)
  tp="${2:?tp}"; port="${3:?port}"
  mkdir -p "$OUT"
  "$0" serve "$tp" "$port" > "$OUT/serve-tp$tp.log" 2>&1 &
  srv=$!
  echo "serve pid=$srv log=$OUT/serve-tp$tp.log (queues for $tp GPUs)"
  "$0" smoke "$port" || { echo "!! smoke failed"; kill -TERM $srv 2>/dev/null; exit 4; }
  "$0" bench "$tp" "$port"
  "$0" stop "$port"
  ;;

# ---------------------------------------------------------------- STOP
# Kill the api_server itself, not the gpulease wrapper: killing the wrapper
# orphans a live server that keeps the port and the cards.
stop)
  port="${2:-}"
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port:-}" || true)
  [ -z "$pids" ] && { echo "no vllm api_server on port ${port:-*}"; exit 0; }
  echo "stopping: $pids"
  kill -TERM $pids 2>/dev/null
  for i in $(seq 1 60); do
    pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port:-}" >/dev/null 2>&1 || break
    sleep 2
  done
  pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port:-}" >/dev/null 2>&1 && {
    echo "escalating to KILL"; kill -KILL $pids 2>/dev/null; sleep 5; }
  echo "stopped"
  ;;

*) echo "usage: $0 {serve|smoke|bench|all|stop} ..."; exit 2 ;;
esac
