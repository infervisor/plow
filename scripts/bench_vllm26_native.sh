#!/usr/bin/env bash
# The vLLM 0.26 + AITER arm of the decode batch-size ladder comparison, run NATIVELY from
# the /workspace/vllm26 venv (no docker on this box).
#
# Why native and not scripts/bench_vllm_chat.sh: that harness runs the
# rocm/vllm:...vllm_0.23.0 CONTAINER, and this box has no docker daemon. The campaign's own
# vLLM 0.26 baselines (fusion-review-and-crossover-sweep.md §15/§17) were all taken from this
# same venv, so this matches the recorded methodology rather than the container script's.
#
# It drives the server with BOTH clients on purpose:
#   1. scripts/sweep_client.py    — the client the plow ladder numbers were taken with, so the
#                                   two engines' rows are commensurable.
#   2. `vllm bench serve --backend openai-chat` — the reference client, at the same points, so
#                                   the sweep_client numbers are CALIBRATED rather than trusted.
# If (1) and (2) disagree, the cross-engine table is a client artifact and says so.
#
#   $1 tag (asset-name-like label used in output paths)
#   QUANT     fp8 | none            (default fp8 — the plow arm is fp8 w8a8)
#   PREFIX    off | on              (default off — see below)
#   PORT      default 8477
#   CONCS NPROMPT OUTLEN IN_LENS REPS   as sweep_client.py
#   CALIB     1 to also run `vllm bench serve` at the same points (default 1)
#
# PREFIX CACHING defaults OFF, and that is a methodology decision, not a tuning one.
# sweep_client.py sends the SAME prompt to all NPROMPT requests of a cell (bench_speed.sh
# always did). plow has no prefix cache, so for plow that is 16 independent prefills. vLLM's
# default APC would serve 15 of the 16 out of cache, collapsing its TTFT and inflating its
# throughput against an engine that cannot do the same thing. `PREFIX=on` re-runs with vLLM's
# default so both numbers are on the record.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VLLM_VENV:-/workspace/vllm26}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/gemma-4-12B-it}"
SERVED="${SERVED:-gemma-4-12B-it}"
TAG="${1:?tag}"
PORT="${PORT:-8477}"
QUANT="${QUANT:-fp8}"
PREFIX="${PREFIX:-off}"
IN_LENS="${IN_LENS:-1024}"; CONCS="${CONCS:-1 2 4 8 16}"
NPROMPT="${NPROMPT:-16}"; OUTLEN="${OUTLEN:-128}"; REPS="${REPS:-3}"
MAXLEN="${MAXLEN:-4096}"
MNBT="${MNBT:-8192}"
GPUMEM="${GPUMEM:-0.85}"
CALIB="${CALIB:-1}"
CALIB_CONCS="${CALIB_CONCS:-$CONCS}"
READY="${READY:-1800}"
OUTDIR="${OUTDIR:-/tmp/vllm26_$TAG}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/serve.log"

QARGS=(); [ "$QUANT" != none ] && QARGS=(--quantization "$QUANT")
PARGS=(--no-enable-prefix-caching); [ "$PREFIX" = on ] && PARGS=()

# --- provenance, recorded before anything is measured -------------------------------------
{
  echo "== provenance =="
  "$VENV/bin/python" -c "import vllm;print('vllm',vllm.__version__)"
  "$VENV/bin/pip" list 2>/dev/null | grep -iE '^(vllm|amd-aiter|torch|triton) ' || true
  echo "rocm: $(cat /opt/rocm/.info/version 2>/dev/null || ls -d /opt/rocm-* 2>/dev/null | tr '\n' ' ')"
  echo "model_dir: $MODEL_DIR  quant: ${QUANT}  prefix_cache: ${PREFIX}"
} | tee "$OUTDIR/provenance.txt"

# --- server -------------------------------------------------------------------------------
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}" \
HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
setsid "$VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAXLEN" \
  --max-num-batched-tokens "$MNBT" \
  --gpu-memory-utilization "$GPUMEM" \
  "${QARGS[@]}" "${PARGS[@]}" \
  --port "$PORT" > "$LOG" 2>&1 &
SPID=$!
SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"
cleanup() {
  [ -n "${SPGID:-}" ] && kill -TERM "-$SPGID" 2>/dev/null
  sleep 8
  [ -n "${SPGID:-}" ] && kill -KILL "-$SPGID" 2>/dev/null
}
# The trap MUST exit: a trap that returns leaves the caller running with the lock released.
trap 'cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

echo "starting vLLM 0.26 on :$PORT (quant=$QUANT prefix=$PREFIX)"
ok=0
for _ in $(seq 1 "$READY"); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health")" = 200 ] \
    && { ok=1; break; }
  kill -0 "$SPID" 2>/dev/null || { echo "FAIL: vLLM died"; tail -60 "$LOG"; exit 1; }
  sleep 2
done
[ "$ok" = 1 ] || { echo "FAIL: never healthy"; tail -60 "$LOG"; exit 1; }
echo ">>> healthy"

# What vLLM actually captured as decode graph sizes — the closest thing it has to the ladder's
# rungs, and the number a "rung quantisation" claim has to be read against.
grep -oiE "cudagraph (capture|sizes)[^)]*" "$LOG" | tail -3 || true
grep -oiE "Capturing.*decode.*" "$LOG" | tail -2 || true

# --- arm 1: the plow ladder's own client ---------------------------------------------------
BASE_URL="http://127.0.0.1:$PORT" MODEL="$SERVED" TAG="$TAG" \
IN_LENS="$IN_LENS" CONCS="$CONCS" NPROMPT="$NPROMPT" OUTLEN="$OUTLEN" REPS="$REPS" \
CSV="$OUTDIR/sweep_client.csv" \
  python3 "$WT/scripts/sweep_client.py" 2>&1 | tee "$OUTDIR/sweep_client.log"

# --- arm 2: the reference client at the same points ----------------------------------------
if [ "$CALIB" = 1 ]; then
  echo "in_len,conc,ttft_ms,tpot_ms,itl_ms,itl_p99,out_tok_s,ok_reqs,gen_toks" \
    | tee "$OUTDIR/vllm_bench.csv"
  for L in $IN_LENS; do for C in $CALIB_CONCS; do
    b="$OUTDIR/vb_in${L}_c${C}.log"
    HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
    "$VENV/bin/vllm" bench serve --backend openai-chat \
      --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions \
      --model "$SERVED" --tokenizer "$MODEL_DIR" \
      --dataset-name random --random-input-len "$L" --random-output-len "$OUTLEN" \
      --max-concurrency "$C" --num-prompts "$NPROMPT" --ignore-eos \
      > "$b" 2>&1
    python3 - "$L" "$C" "$b" <<'PY' | tee -a "$OUTDIR/vllm_bench.csv"
import re,sys
L,C,p=sys.argv[1],sys.argv[2],sys.argv[3]
t=open(p).read()
def g(pat):
    m=re.search(pat+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
print(f"{L},{C},{g(r'Mean TTFT .ms.:'):.2f},{g(r'Mean TPOT .ms.:'):.3f},"
      f"{g(r'Mean ITL .ms.:'):.3f},{g(r'P99 ITL .ms.:'):.3f},"
      f"{g(r'Output token throughput .tok/s.:'):.1f},"
      f"{g(r'Successful requests:'):.0f},{g(r'Total generated tokens:'):.0f}")
PY
  done; done
fi

# --- what the SCHEDULER actually did --------------------------------------------------------
# vLLM's batching is token-budget based and continuous; the ladder's is quantised to a rung.
# The server's own "Running: N reqs" lines are the only direct evidence of the batch shape it
# ran, and they must be read BEFORE the container/process is torn down.
echo "== scheduler running-batch trace (server-side) =="
grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs" "$LOG" | tail -40 \
  | tee "$OUTDIR/running_batch.txt" || echo "  (no engine stats lines)"
echo "== prefix cache (server-side) =="
grep -oiE "[Pp]refix cache hit rate[^,)]*" "$LOG" | tail -5 || echo "  (none logged)"

echo "results in $OUTDIR"
