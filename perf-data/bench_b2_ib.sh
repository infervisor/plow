#!/usr/bin/env bash
# =============================================================================
# bench_b2_ib.sh — campaign B2-ib-12b: multi-user concurrency/capacity sweep
# =============================================================================
# Drives huggingface/inference-benchmarker (pinned rev via bench_ib.sh — the
# tool's own binary) against EITHER vLLM or `plowrt serve`, with an identical
# prompt profile, identical durations, identical tokenizer and identical tool
# rev, so the two engines' numbers are directly comparable.
#
# Profile (the head-to-head point): ~4k input tokens / 128 output tokens,
# greedy (temperature 0), prompts drawn from the hlarcher/inference-benchmarker
# github_code.json dataset truncated to 4000 gemma tokens (2370 distinct
# eligible entries — no trivial prefix-cache repeats).
#
# Per engine config this runs:
#   1. fixed-VU points  : throughput mode (constant N virtual users) for
#                         N in $VUS, 15s warmup + 120s measure each
#   2. sweep            : ib sweep mode — auto max-throughput detection at
#                         $SWEEP_MAX_VUS VUs, then $NUM_RATES constant-arrival
#                         rates up to 1.2x the detected max
#
# TTFT here INCLUDES any server-side queueing — that is the point of a
# capacity benchmark (plow's mux queues arrivals beyond its B slots; vLLM
# admits into its continuous batch).
#
# One invocation = one server config = wrap in ONE gpulease:
#   GPU_LEASE_TIMEOUT=7200 gpulease b2-vllm \
#     env ENGINE=vllm bash perf-data/bench_b2_ib.sh
#   GPU_LEASE_TIMEOUT=7200 gpulease b2-plow-b4 \
#     env ENGINE=plow ASSETS=/root/gpu-assets-b4/b4 TAG=plow-b4 VUS="1 2 4 8" \
#     SWEEP_MAX_VUS=4 bash perf-data/bench_b2_ib.sh
# =============================================================================
set -uo pipefail

ENGINE="${ENGINE:?set ENGINE=vllm|plow}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/workspace/models/gemma-4-12B-it}"
# Served-model id and tokenizer default to the 12B campaign but are OVERRIDABLE,
# so one harness drives the whole Gemma-4 family (12B / 31B dense, 26B-A4B MoE).
# Both engines must use the SAME served id and the SAME tokenizer rev or the
# numbers are not comparable.
MODEL_NAME="${MODEL_NAME:-gemma-4-12b-it}"
TOKENIZER="${TOKENIZER:-google/gemma-4-12B-it}" # hub id; cache pre-seeded with the LOCAL
                                     # tokenizer.json (sha256-identical to hub)
PORT="${PORT:-8091}"
TAG="${TAG:-$ENGINE}"
VUS="${VUS:-1 2 4 8 16 32}"
SWEEP_MAX_VUS="${SWEEP_MAX_VUS:-32}"
NUM_RATES="${NUM_RATES:-8}"
DUR="${DUR:-120s}"
WARM="${WARM:-15s}"
GPU_UTIL="${GPU_UTIL:-0.90}"         # vLLM only; plow sizes KV by blob B
MAXLEN="${MAXLEN:-8192}"             # matches the plow blob's per-slot ctx
CAMPAIGN="${CAMPAIGN:-B2-ib-12b}"    # stamps meta + run-ids (family reuse)
PROMPT_TOKS="${PROMPT_TOKS:-4000}"   # profile (a)=4000; profile (b)=16000
DO_SWEEP="${DO_SWEEP:-1}"            # 0 skips the sweep stage

OUTDIR="$ROOT/perf-data/harness/b2-ib/$TAG"
SRVLOG="$OUTDIR/server.log"
mkdir -p "$OUTDIR"
cd "$OUTDIR"                          # ib writes results/*.json under cwd

PROFILE_ARGS=(
  --dataset-file github_code.json
  --prompt-options "num_tokens=$PROMPT_TOKS,min_tokens=$PROMPT_TOKS,max_tokens=$PROMPT_TOKS,variance=0"
  --decode-options num_tokens=128,min_tokens=128,max_tokens=128,variance=0
)

SRVPID=""
vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
cleanup() {
  if [ -n "$SRVPID" ]; then
    echo ">>> shutting down $ENGINE server (pid $SRVPID)"
    kill -TERM "$SRVPID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 2; done
    kill -KILL "$SRVPID" 2>/dev/null
    pkill -f "vllm serve $MODEL_DIR" 2>/dev/null
    pkill -f "VLLM::EngineCore" 2>/dev/null
  fi
  # Hold the lease until VRAM is actually released for the next agent.
  for _ in $(seq 1 30); do
    u=$(vram_used); echo ">>> VRAM used: ${u} MiB"
    [ "${u:-99999}" -lt 1000 ] && break
    sleep 3
  done
}
trap cleanup EXIT

# ---- serve ------------------------------------------------------------------
case "$ENGINE" in
  vllm)
    export PATH=/workspace/venvs/vllm/bin:/usr/local/cuda/bin:$PATH
    echo ">>> vllm $(vllm --version 2>/dev/null | tail -1)"
    vllm serve "$MODEL_DIR" \
      --served-model-name "$MODEL_NAME" \
      --port "$PORT" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-model-len "$MAXLEN" \
      --tensor-parallel-size 1 \
      ${VLLM_EXTRA:-} \
      > "$SRVLOG" 2>&1 &
    SRVPID=$!
    for i in $(seq 1 900); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
      kill -0 "$SRVPID" 2>/dev/null || { echo "vLLM died"; tail -30 "$SRVLOG"; exit 1; }
      sleep 1
    done
    ;;
  plow)
    ASSETS="${ASSETS:?set ASSETS=<plow assets dir>}"
    BIN="$ROOT/target/release/plowrt"
    NO_COLOR=1 RUST_LOG=info,plowrt=debug "$BIN" serve \
      --assets "$ASSETS" --port "$PORT" > "$SRVLOG" 2>&1 &
    SRVPID=$!
    for i in $(seq 1 900); do
      grep -q "serving OpenAI API over TCP" "$SRVLOG" && break
      kill -0 "$SRVPID" 2>/dev/null || { echo "plowrt died"; tail -30 "$SRVLOG"; exit 1; }
      sleep 1
    done
    ;;
  *) echo "unknown ENGINE=$ENGINE"; exit 1;;
esac
echo ">>> $ENGINE serving on :$PORT"

# Correctness gate before any perf number.
GATE=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0}")
echo ">>> gate: $GATE"
echo "$GATE" | grep -qi paris || { echo "GATE FAILED"; exit 1; }

# ---- benchmark --------------------------------------------------------------
IB=("$ROOT/perf-data/bench_ib.sh"
  --tokenizer-name "$TOKENIZER"
  --model-name "$MODEL_NAME"
  --url "http://127.0.0.1:$PORT"
  --no-console
  --warmup "$WARM"
  --duration "$DUR")

# Engine build provenance stamped into every run's metadata: for plow this is
# the workspace commit the served binary was built from (verify the binary is
# up-to-date with `cargo build --release -p plowrt --features cuda,hf-tokenizer`
# reporting no work before benching).
ENGINE_COMMIT=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
META="campaign=$CAMPAIGN,engine=$ENGINE,tag=$TAG,engine_commit=$ENGINE_COMMIT"

for N in $VUS; do
  echo ">>> [$TAG] fixed-VU point: $N users"
  "${IB[@]}" "${PROFILE_ARGS[@]}" \
    --benchmark-kind throughput --max-vus "$N" \
    --extra-meta "$META,point=vu$N" \
    --run-id "$CAMPAIGN-$TAG-vu$N" || echo ">>> vu$N FAILED"
done

if [ "$DO_SWEEP" = "1" ]; then
echo ">>> [$TAG] sweep (max_vus=$SWEEP_MAX_VUS, num_rates=$NUM_RATES)"
"${IB[@]}" "${PROFILE_ARGS[@]}" \
  --benchmark-kind sweep --max-vus "$SWEEP_MAX_VUS" --num-rates "$NUM_RATES" \
  --extra-meta "$META,point=sweep" \
  --run-id "$CAMPAIGN-$TAG-sweep" || echo ">>> sweep FAILED"
fi

echo ">>> [$TAG] done; results in $OUTDIR/results/"
