#!/usr/bin/env bash
# =============================================================================
# scripts/bench_vllm_rocm.sh — vLLM benchmark on ROCm, one model, under a lease.
# =============================================================================
# Descends from the deleted perf-data/bench_vllm_tp.sh (commit ebfb2a6^), which
# produced perf-data/vllm-tp-baseline.md. Same serve flags and same parse, plus:
#   - takes a GPU LEASE (perf-data/tools/gpulease) instead of hard-coding GPUS,
#     so concurrent agents cannot land on the same cards mid-measurement;
#   - two phases: `general` (concurrency sweep) and `ctxsweep` (concurrency 1).
#
# Models are read straight out of the HF cache (mounted read-only, offline), so
# there is no second copy of a 700 GB checkpoint on disk.
#
# GPU SELECTION: HIP_VISIBLE_DEVICES=comma list (NOT --gpus); all DRI/KFD via --device.
# DOCKER: `sudo -n docker` (invoking user's docker group membership needs a re-login).
#
# Usage:
#   scripts/bench_vllm_rocm.sh <hf-repo-id> [TP]
#   PHASES=ctxsweep scripts/bench_vllm_rocm.sh google/gemma-4-12B-it 1
#   SERVE_ONLY=1 scripts/bench_vllm_rocm.sh google/gemma-4-31B-it 2
#
# Env:
#   IMAGE        container image (default: the AMD ROCm vLLM release)
#   PHASES       comma list of general,ctxsweep (default both)
#   CTXS         ctxsweep input lengths (default 1024..65536)
#   CONCS        general-phase concurrencies (default 1,4,16,64)
#   GEN_CTX      general-phase input length (default 1024)
#   OUTPUT_LEN   decode tokens per request (default 128)
#   MAXLEN       --max-model-len (default: max(CTXS)+OUTPUT_LEN+2048)
#   QUANT/KVFP8  fp8 knobs, as in the original harness
#   NO_LEASE=1   skip gpulease (NOT for publishable numbers)
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${1:?usage: bench_vllm_rocm.sh <hf-repo-id> [TP]}"
TP="${2:-1}"

DOCKER="${DOCKER:-sudo -n docker}"
IMAGE="${IMAGE:-rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8000}"
SLUG="$(echo "$MODEL_ID" | tr '/' '_')"
CNAME="${CNAME:-vllm_bench_${SLUG}_tp${TP}}"
OUTDIR="${OUTDIR:-$REPO/perf-data/vllm-rocm}"
PHASES="${PHASES:-general,ctxsweep}"
CTXS="${CTXS:-1024,4096,8192,16384,32768,65536}"
CONCS="${CONCS:-1,4,16,64}"
GEN_CTX="${GEN_CTX:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-7200}"
# torch.compile + AITER JIT for a big MoE takes tens of minutes, and the cache
# normally dies with the container, so every run repays it. Persist it on the
# host: /root/.cache covers both vllm's torch_compile_cache and aiter's.
COMPILE_CACHE="${COMPILE_CACHE:-$HOME/.cache/vllm-bench-container}"
mkdir -p "$COMPILE_CACHE"
SERVE_ONLY="${SERVE_ONLY:-0}"
GPULEASE="${GPULEASE:-$REPO/perf-data/tools/gpulease}"

QUANT="${QUANT:-bf16}"; KVFP8="${KVFP8:-0}"; QUANT_ARGS=""
[ "$QUANT" = "fp8" ] && QUANT_ARGS="--quantization fp8"
[ "$KVFP8" = "1" ]   && QUANT_ARGS="$QUANT_ARGS --kv-cache-dtype fp8"
QTAG="$QUANT"; [ "$KVFP8" = "1" ] && QTAG="${QUANT}kv"

# --- take the lease first, then re-enter. gpulease exports HIP_VISIBLE_DEVICES
# for a partial lease; PLOW_LEASED marks the second entry so we don't recurse.
if [ -z "${PLOW_LEASED:-}" ] && [ "${NO_LEASE:-0}" != "1" ]; then
  [ -x "$GPULEASE" ] || { echo "!!! gpulease missing at $GPULEASE" >&2; exit 1; }
  export PLOW_LEASED=1
  exec "$GPULEASE" -n "$TP" "vllm-${SLUG}-tp${TP}" "$0" "$@"
fi

# Cards to shard across. A full-box lease leaves HIP_VISIBLE_DEVICES unset.
GPUS="${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((TP - 1)))}"

# max_model_len must cover the longest ctx we will actually ask for.
if [ -z "${MAXLEN:-}" ]; then
  MAXCTX=$(echo "$CTXS" | tr ',' '\n' | sort -n | tail -1)
  MAXLEN=$((MAXCTX + OUTPUT_LEN + 2048))
fi

mkdir -p "$OUTDIR"

# EXTRA_ENV="K=V K2=V2" -> "-e K=V -e K2=V2". Needed because some models only
# have a ROCm kernel behind a flag: GLM-5.2's DSA sparse-attention indexer
# hard-fails with "Sparse attention indexer ROCm path is only supported on
# AITER" unless VLLM_ROCM_USE_AITER=1.
ENV_FLAGS=""
for kv in ${EXTRA_ENV:-}; do ENV_FLAGS="$ENV_FLAGS -e $kv"; done

# The ROCm images have no `render`/`video` group in /etc/group, so --group-add
# by NAME fails outright ("Unable to find group render"). Pass host GIDs.
RGID="$(getent group render | cut -d: -f3)"; RGID="${RGID:-109}"
VGID="$(getent group video  | cut -d: -f3)"; VGID="${VGID:-44}"

serve() {
  echo ">>> [TP=$TP img=$IMAGE] serving $MODEL_ID (max-model-len=$MAXLEN) on GPUs $GPUS"
  $DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
  # NOT --rm: a crashed container must survive long enough to read its logs.
  # With --rm the daemon reaps it the moment it exits and the startup failure
  # is unrecoverable ("No such container"). cleanup() removes it explicitly.
  $DOCKER run -d --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add "$VGID" --group-add "$RGID" \
    --security-opt seccomp=unconfined \
    --ipc=host --shm-size=32g \
    -e HIP_VISIBLE_DEVICES="$GPUS" \
    -e HF_HUB_OFFLINE=1 \
    -e HF_HOME=/hf \
    -e HF_MODULES_CACHE=/tmp/hf_modules \
    $ENV_FLAGS \
    -v "$HF_CACHE":/hf:ro \
    -v "$COMPILE_CACHE":/root/.cache \
    -p "$PORT":8000 \
    --entrypoint vllm \
    "$IMAGE" \
    serve "$MODEL_ID" \
      --max-num-batched-tokens 8192 \
      --max-model-len "$MAXLEN" \
      ${DTYPE_ARGS:---dtype bfloat16} \
      ${QUANT_ARGS:-} \
      ${MEM_UTIL:+--gpu-memory-utilization "$MEM_UTIL"} \
      ${SERVE_EXTRA_ARGS:-} \
      --tensor-parallel-size "$TP" >/dev/null
}

wait_health() {
  local t=0
  echo ">>> waiting for :$PORT/health (<= ${HEALTH_TIMEOUT}s)"
  while [ "$t" -lt "$HEALTH_TIMEOUT" ]; do
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

# Coherence gate. MUST go through /v1/chat/completions: these are -it
# checkpoints, and a raw /v1/completions continuation with no chat template
# returns garbage ("The capital of France is" -> "111.1......11111") even
# though the weights are fine. The old harness gated on the raw endpoint,
# which reads as a broken model for every instruct checkpoint.
# Reasoning models (GLM-5.2 opens a <think> block in its chat template by
# default) spend the first tokens thinking, so a short cap truncates before the
# answer and the gate reads FAIL on a perfectly healthy model. Ask for thinking
# off, give it room, and scan the whole reply including any reasoning field.
ask() {
  curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":256,\"temperature\":0${1:-}}"
}
sanity() {
  local out txt
  out=$(ask ',"chat_template_kwargs":{"enable_thinking":false}')
  echo "$out" | grep -q '"choices"' || out=$(ask)      # template rejected the kwarg
  txt=$(echo "$out" | python3 -c 'import sys,json
try:
    m=json.load(sys.stdin)["choices"][0]["message"]
    print((m.get("content") or "")+" "+(m.get("reasoning_content") or ""))
except Exception: print("")' 2>/dev/null)
  echo ">>> sanity completion: $(echo "$txt" | tr -d '\n')"
  if echo "$txt" | grep -qi "paris"; then
    echo ">>> coherence gate: PASS"
  else
    echo ">>> coherence gate: FAIL — model is not answering coherently; numbers below are suspect" >&2
    COHERENT=0
  fi
}
COHERENT=1

# One `vllm bench serve` point. $1=input_len $2=concurrency $3=num_prompts $4=tag
# Emits: ctx,conc,ttft_ms,prefill_tok_s,tpot_ms,itl_ms,itl_med_ms,itl_p99_ms,decode_tok_s,req_throughput,out_tok_s
bench_point() {
  local L="$1" C="$2" N="$3" tag="$4"
  local log="$OUTDIR/${SLUG}_${QTAG}_tp${TP}_${tag}_in${L}_c${C}.log"
  # Run the client in its OWN container over HTTP rather than `docker exec` into
  # the server. exec into the big MoE server died with "OCI runtime exec failed:
  # error executing setns process" while the server itself was healthy, and a
  # separate client cannot perturb the server's process state anyway. No GPU
  # devices here — the client only tokenizes and drives HTTP.
  # BENCH_EXTRA_ARGS: `vllm bench serve` builds its own tokenizer client-side,
  # so a trust-remote-code model needs the flag HERE too, not just on serve.
  $DOCKER run --rm --network host \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf -e HF_MODULES_CACHE=/tmp/hf_modules \
    -v "$HF_CACHE":/hf:ro \
    --entrypoint vllm "$IMAGE" \
    bench serve \
    --model "$MODEL_ID" --dataset-name random \
    --random-input-len "$L" --random-output-len "$OUTPUT_LEN" \
    --max-concurrency "$C" --num-prompts "$N" --port "$PORT" \
    ${BENCH_EXTRA_ARGS:-} \
    > "$log" 2>&1
  python3 - "$L" "$C" "$log" <<'PY'
import re,sys
L=int(sys.argv[1]); C=int(sys.argv[2]); t=open(sys.argv[3]).read()
def g(p):
    m=re.search(p+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
ttft=g(r"Mean TTFT \(ms\):"); tpot=g(r"Mean TPOT \(ms\):"); itl=g(r"Mean ITL \(ms\):")
# ITL median and P99 beside the mean. A mean ITL hides the stalls that decide
# whether a stream reads as smooth: one 400 ms hitch in 128 tokens moves the
# mean by 3 ms and the P99 by 400. The tail is the number to compare engines on.
itl_med=g(r"Median ITL \(ms\):"); itl_p99=g(r"P99 ITL \(ms\):")
rps=g(r"Request throughput \(req/s\):"); ots=g(r"Output token throughput \(tok/s\):")
# `Successful requests` counts a REJECTED request as a success, so it cannot
# gate a point alone: a 131072 input-len point against a max_ctx=131072 plow
# blob had every prefill refused and still reported 4 successful requests at
# 99.1 tok/s with ITL 0.00. `gen_toks` must equal num_prompts x OUTPUT_LEN.
ok=g(r"Successful requests:"); gen=g(r"Total generated tokens:")
pf = L/(ttft/1000.0) if ttft==ttft and ttft>0 else float('nan')
dc = 1000.0/tpot if tpot==tpot and tpot>0 else float('nan')
print(f"{L},{C},{ttft:.2f},{pf:.1f},{tpot:.3f},{itl:.3f},{itl_med:.3f},{itl_p99:.3f},{dc:.2f},{rps:.3f},{ots:.1f},{ok:.0f},{gen:.0f}")
PY
}

# Wait for the container to actually be gone. `docker rm -f` returns before the
# process dies, so the lease's post-run audit could still see our own vLLM on
# the cards and report false contention.
cleanup() {
  # ALWAYS keep the server log. Saving it only on health failure meant a server
  # that started fine and then died mid-benchmark (Kimi: gate passed, then every
  # request refused) left no evidence at all once the container was removed.
  $DOCKER logs "$CNAME" > "$OUTDIR/${SLUG}_tp${TP}_serve.log" 2>&1 || true
  $DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
  local i
  for i in $(seq 1 30); do
    $DOCKER ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CNAME}$" || break
    sleep 1
  done
}
trap cleanup EXIT

serve || { echo "!!! docker run failed"; exit 1; }
if ! wait_health; then
  rc=$?
  echo "===== DOCKER LOGS (last 120) ====="
  $DOCKER logs "$CNAME" 2>&1 | tail -120
  $DOCKER logs "$CNAME" > "$OUTDIR/${SLUG}_tp${TP}_serve_fail.log" 2>&1 || true
  echo ">>> SERVE FAILED $MODEL_ID TP=$TP (rc=$rc)"
  exit 3
fi

VER=$($DOCKER exec "$CNAME" vllm --version 2>/dev/null | tail -1)
echo ">>> vLLM version: $VER"
sanity

if [ "$SERVE_ONLY" = "1" ]; then echo ">>> SERVE_ONLY=1 — stopping after sanity."; exit 0; fi

HDR="input_len,concurrency,ttft_ms,prefill_tok_s,tpot_ms,itl_ms,itl_med_ms,itl_p99_ms,decode_tok_s,req_per_s,out_tok_s,ok_reqs,gen_toks"

# A row of nan means `vllm bench serve` itself died (bad flag, tokenizer refusal,
# OOM) and produced NO measurement. Silently writing that to the CSV and exiting
# 0 reported a fully-NaN Kimi run as OK. Count them and fail the run instead.
BAD=0
record() {  # $1=row $2=csv
  echo "    $1"; echo "$1" >> "$2"
  case "$1" in *nan*) BAD=$((BAD + 1));
    echo "    !!! bench point produced no measurement — see the .log beside $2" >&2
    # If the server is gone there is no point running the rest of the matrix
    # against a dead endpoint; stop and keep its logs.
    if ! $DOCKER ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CNAME}$"; then
      echo "    !!! SERVER CONTAINER IS GONE — it died mid-run; aborting remaining points" >&2
      exit 5
    fi ;;
  esac
}

# Warm-up. The FIRST measured point after a cold server absorbs lazy graph
# capture and JIT, which lands entirely on whichever arm runs first: GLM's
# general conc=1 read TTFT 1930 ms while its clean conc=1 point was 37 ms.
# Burn one discarded bench point so no reported row carries that cost.
echo ">>> warm-up (discarded)"
bench_point "$GEN_CTX" 1 4 warmup >/dev/null 2>&1 || true

# --- Phase 1: general. Fixed ctx, concurrency sweep — the throughput/latency
# curve. num_prompts scales with concurrency so every arm sees a full pipe.
if [[ ",$PHASES," == *",general,"* ]]; then
  csv="$OUTDIR/${SLUG}_${QTAG}_tp${TP}_general.csv"; echo "$HDR" > "$csv"
  IFS=',' read -ra C_ARR <<<"$CONCS"
  for C in "${C_ARR[@]}"; do
    N=$((C * 8)); [ "$N" -lt 8 ] && N=8; [ "$N" -gt 256 ] && N=256
    echo ">>> general $MODEL_ID tp=$TP ctx=$GEN_CTX conc=$C prompts=$N"
    record "$(bench_point "$GEN_CTX" "$C" "$N" general)" "$csv"
  done
  echo ">>> general summary: $csv"
fi

# --- Phase 2: ctxsweep at concurrency 1. Single-user latency vs context —
# TPOT here is the decode number that compares against plow.
if [[ ",$PHASES," == *",ctxsweep,"* ]]; then
  csv="$OUTDIR/${SLUG}_${QTAG}_tp${TP}_ctxsweep_c1.csv"; echo "$HDR" > "$csv"
  IFS=',' read -ra L_ARR <<<"$CTXS"
  for L in "${L_ARR[@]}"; do
    echo ">>> ctxsweep $MODEL_ID tp=$TP input_len=$L conc=1"
    record "$(bench_point "$L" 1 "$NUM_PROMPTS" ctxsweep)" "$csv"
  done
  echo ">>> ctxsweep summary: $csv"
fi

if [ "$BAD" -gt 0 ]; then
  echo ">>> FAILED $MODEL_ID TP=$TP — $BAD bench point(s) produced no measurement" >&2
  exit 4
fi
[ "$COHERENT" = "1" ] || echo ">>> WARNING $MODEL_ID produced incoherent output; numbers are suspect" >&2
echo ">>> done $MODEL_ID TP=$TP ; vLLM $VER ; gpus=$GPUS"
