#!/usr/bin/env bash
# The SYMMETRIC vLLM point: same client binary, same `--backend openai-chat`,
# same dataset/ctx/concurrency as `bench_plowrt_serve.sh`.
#
# Why not reuse the recorded baselines: those ran the default `openai` backend
# against /v1/completions. plowrt implements /v1/chat/completions only, and the
# chat backend's TTFT is the time to the first chunk carrying `choices` — which
# for BOTH servers is the immediate role frame, not the first token. Comparing a
# chat-backend plow number against a completions-backend vLLM number therefore
# compares two different quantities. This re-measures vLLM on plow's terms.
#
# $1 hf-repo-id  $2 tp
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${1:?repo-id}"; TP="${2:-1}"
DOCKER="sudo -n docker"
IMAGE=rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
HF_CACHE="$HOME/.cache/huggingface"
COMPILE_CACHE="$HOME/.cache/vllm-bench-container"; mkdir -p "$COMPILE_CACHE"
PORT="${PORT:-8500}"
SLUG="$(echo "$MODEL_ID" | tr '/' '_')"
CNAME="vllm_chatbench_${SLUG}_tp${TP}"
IN_LENS="${IN_LENS:-1024}"; CONCS="${CONCS:-1}"
NPROMPT="${NPROMPT:-8}"; OUTLEN="${OUTLEN:-128}"
MAXLEN="${MAXLEN:-8192}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3600}"
GPUS="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-$(seq -s, 0 $((TP-1)))}}"
RGID="$(getent group render | cut -d: -f3)"; RGID="${RGID:-109}"
VGID="$(getent group video  | cut -d: -f3)"; VGID="${VGID:-44}"

echo "GPUS=$GPUS"
$DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
trap '$DOCKER rm -f "$CNAME" >/dev/null 2>&1; sleep 3' EXIT

$DOCKER run -d --name "$CNAME" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$VGID" --group-add "$RGID" \
  --security-opt seccomp=unconfined --ipc=host --shm-size=32g \
  -e HIP_VISIBLE_DEVICES="$GPUS" -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  ${EXTRA_ENV:+$(for kv in $EXTRA_ENV; do printf -- '-e %s ' "$kv"; done)} \
  -v "$HF_CACHE":/hf:ro -v "$COMPILE_CACHE":/root/.cache \
  -p "$PORT":8000 --entrypoint vllm "$IMAGE" \
  serve "$MODEL_ID" --max-num-batched-tokens 8192 --max-model-len "$MAXLEN" \
    ${DTYPE_ARGS:---dtype bfloat16} ${SERVE_EXTRA_ARGS:-} \
    --tensor-parallel-size "$TP" >/dev/null

t=0
while [ "$t" -lt "$HEALTH_TIMEOUT" ]; do
  $DOCKER ps --format '{{.Names}}' | grep -q "^${CNAME}$" || {
    # 40 lines is not enough: vLLM's engine-core failure prints a ~40-line client-side
    # traceback AFTER the root cause, so a 40-line tail shows only
    # "Engine core initialization failed. See root cause above." with the cause cut off.
    echo "!! container exited during startup"; $DOCKER logs --tail 250 "$CNAME"; exit 2; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health")" = "200" ] && {
    echo ">>> healthy after ${t}s"; break; }
  sleep 10; t=$((t+10))
done
[ "$t" -ge "$HEALTH_TIMEOUT" ] && { echo "!! never healthy"; exit 1; }

# Per-input-length prompt / warm-up counts — see the same block in bench_plowrt_serve.sh.
# The caller builds ONE map and passes the identical string to both scripts, so the two
# engines are always compared at the same sample count for a given input length. Unset =
# previous behaviour.
pick () { # <map> <key> <default>
  local kv; for kv in ${1:-}; do [ "${kv%%:*}" = "$2" ] && { echo "${kv##*:}"; return; }; done
  echo "$3"
}

echo "input_len,concurrency,ttft_ms,ttft_med,tpot_ms,tpot_med,itl_ms,itl_med,out_tok_s"
for L in $IN_LENS; do for C in $CONCS; do
  NP="$(pick "${NPROMPT_MAP:-}" "$L" "$NPROMPT")"
  NW="$(pick "${NWARM_MAP:-}" "$L" "")"
  WARM="${BENCH_EXTRA_ARGS:-}"; [ -n "$NW" ] && WARM="--num-warmups $NW"
  blog="/tmp/vllmchat_${SLUG}_in${L}_c${C}.log"
  $DOCKER run --rm --network host -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
    -v "$HF_CACHE":/hf:ro --entrypoint vllm "$IMAGE" \
    bench serve --backend openai-chat \
    --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions \
    --model "$MODEL_ID" --dataset-name random \
    --random-input-len "$L" --random-output-len "$OUTLEN" \
    --max-concurrency "$C" --num-prompts "$NP" \
    $WARM > "$blog" 2>&1
  python3 - "$L" "$C" "$blog" <<'PY'
import re,sys
L,C,p=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
t=open(p).read()
def g(pat):
    m=re.search(pat+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
print(f"{L},{C},{g(r'Mean TTFT .ms.:'):.2f},{g(r'Median TTFT .ms.:'):.2f},"
      f"{g(r'Mean TPOT .ms.:'):.3f},{g(r'Median TPOT .ms.:'):.3f},"
      f"{g(r'Mean ITL .ms.:'):.3f},{g(r'Median ITL .ms.:'):.3f},"
      f"{g(r'Output token throughput .tok/s.:'):.1f}")
PY
done; done

# PREFIX CACHE — verify, do not assume (task #21). `--dataset-name random` with no
# `--random-prefix-len` should give prompts that share nothing, so the cache should have
# nothing to hit. The client's own arg dump confirms `random_prefix_len=0`, but that is the
# REQUEST side; the authority is what the SERVER reports it actually reused.
#
# This has to run BEFORE the EXIT trap removes the container — once `docker rm -f` has run the
# logs are gone and the check can never be made retrospectively.
echo "== prefix cache (server-side) =="
$DOCKER logs "$CNAME" 2>&1 | grep -oiE "[Pp]refix cache hit rate[^,)]*" | tail -5 \
  || echo "  (no prefix-cache line logged by this vLLM build)"
