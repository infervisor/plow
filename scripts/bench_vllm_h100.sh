#!/usr/bin/env bash
# The NVIDIA-side symmetric vLLM point: the SAME `vllm bench serve` client, the same
# backend/dataset/flags as scripts/bench_plowrt_serve.sh, against a natively-served vLLM
# instead of a plowrt endpoint. Prints the identical CSV header, so a result from here
# drops straight into perf-data/campaign/*.reference-*.csv.
#
# bench_vllm_chat.sh is the ROCm/docker twin; it cannot serve this box (no /dev/kfd).
# Everything here goes through the venv at VLLM_VENV, which is vLLM 0.28 built for CUDA.
#
#   $1 model-dir (also the tokenizer)  $2 port  $3 ready-timeout
#
#   IN_LENS / CONCS / NPROMPT / OUTLEN   the sweep, as in bench_plowrt_serve.sh
#   MAXLEN        vLLM --max-model-len (default: max IN_LENS + OUTLEN + 512)
#   MAXSEQS       vLLM --max-num-seqs (default: max CONCS)
#   VLLM_VENV     default /opt/pytorch
#   OUTDIR        raw client logs + JSON (default /tmp/vllm_bench_<port>)
#   PREFIX_PCT    percent of each input_len sent as a prefix shared by every request of a cell
#                 (--random-prefix-len); total input stays input_len. Default 0.
#   MEM_SAMPLE_MS GPU memory sampling period per cell, as in bench_plowrt_serve.sh (default 1000; 0 = off)
#   DATASET_ARGS  as in bench_plowrt_serve.sh (the two sides must be given the same string)
#   GATE_PROMPT   raw-completion coherence prompt, as in bench_plowrt_serve.sh; set => one greedy
#                 completion is checked for GATE_EXPECT (default "paris") before any cell runs
#   VLLM_SERVE_EXTRA_ARGS  extra `vllm serve` flags (quantization, kv-cache-dtype, ...)
#   PREFIX_CACHE=1 serve WITH prefix caching (vLLM's default); unset keeps --no-enable-prefix-caching
#
# Prefix caching is DISABLED. vllm-bench's random prompts share a leading prefix, so a
# cache-on server silently benches cache-hit suffixes; the plow side sets
# PLOW_PREFIX_CACHE=0 for the same reason. Both halves must agree or the comparison is void.
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${1:?model-dir}"; PORT="${2:-8600}"; READY="${3:-1800}"
IN_LENS="${IN_LENS:-1024}"; CONCS="${CONCS:-1}"; NPROMPT="${NPROMPT:-32}"
OUTLEN="${OUTLEN:-128}"
BENCH_BACKEND="${BENCH_BACKEND:-openai}"
VLLM_VENV="${VLLM_VENV:-/opt/pytorch}"
OUTDIR="${OUTDIR:-/tmp/vllm_bench_$PORT}"
RESULT_DIR="$OUTDIR/json"
mkdir -p "$OUTDIR" "$RESULT_DIR"
VLLM="$VLLM_VENV/bin/vllm"
test -x "$VLLM" || { echo "no vllm at $VLLM (set VLLM_VENV)" >&2; exit 2; }

maxof () { local m=0 v; for v in $1; do [ "$v" -gt "$m" ] && m=$v; done; echo "$m"; }
MAXLEN="${MAXLEN:-$(( $(maxof "$IN_LENS") + OUTLEN + 512 ))}"
MAXSEQS="${MAXSEQS:-$(maxof "$CONCS")}"
ENDPOINT="/v1/completions"; [ "$BENCH_BACKEND" = "openai-chat" ] && ENDPOINT="/v1/chat/completions"

PC_FLAG="--no-enable-prefix-caching"; if [ -n "${PREFIX_CACHE:-}" ]; then PC_FLAG=""; fi
# setsid + process-group teardown: `vllm serve` forks engine workers that outlive a plain
# kill of the pid we waited on, and a survivor holds the card past the lease.
setsid "$VLLM" serve "$MODEL_DIR" \
  --port "$PORT" --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
  $PC_FLAG \
  ${VLLM_SERVE_EXTRA_ARGS:-} > "$OUTDIR/server.log" 2>&1 &
SRV=$!
cleanup () { [ -z "${mempid:-}" ] || kill "$mempid" 2>/dev/null || true; kill -TERM -"$SRV" 2>/dev/null || true; sleep 5; kill -KILL -"$SRV" 2>/dev/null || true; }
trap cleanup EXIT

t=0
while [ "$t" -lt "$READY" ]; do
  kill -0 "$SRV" 2>/dev/null || { echo "!! vllm died during startup"; tail -40 "$OUTDIR/server.log"; exit 2; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)" = "200" ] && break
  sleep 5; t=$((t+5))
done
[ "$t" -ge "$READY" ] && { echo "!! vllm never healthy"; tail -40 "$OUTDIR/server.log"; exit 1; }
echo ">>> vllm healthy after ${t}s (max_model_len=$MAXLEN max_num_seqs=$MAXSEQS)" >&2

# Coherence gate BEFORE any timing, as in bench_plowrt_serve.sh: a fast wrong server is not a
# result, and a quantized checkpoint that loads is not yet a checkpoint that answers. Opt-in
# (GATE_PROMPT unset = no gate) and entirely on stderr, because stdout here IS the reference CSV.
if [ -n "${GATE_PROMPT:-}" ]; then
  GATE_BODY=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1], "prompt":sys.argv[2], "max_tokens":32, "temperature":0}))' \
    "$MODEL_DIR" "$GATE_PROMPT")
  GATE=$(curl -s --max-time 300 "http://127.0.0.1:$PORT$ENDPOINT" \
    -H 'Content-Type: application/json' --data-binary "$GATE_BODY")
  echo "== coherence gate ==" >&2; echo "$GATE" >&2
  echo "$GATE" | grep -qi "${GATE_EXPECT:-paris}" && echo ">>> coherence gate: PASS" >&2 || {
    echo ">>> coherence gate: FAIL — numbers below would be meaningless" >&2; exit 1; }
fi

echo "input_len,concurrency,ttft_ms,ttft_med,tpot_ms,tpot_med,itl_ms,itl_med,itl_p99,out_tok_s,req_per_s,ok_reqs,gen_toks"
for L in $IN_LENS; do
  for C in $CONCS; do
    blog="$OUTDIR/in${L}_c${C}.log"
    if [ -n "${DATASET_ARGS:-}" ]; then read -r -a DATASET_ARGV <<< "$DATASET_ARGS"; else
      PFX=$(( L * ${PREFIX_PCT:-0} / 100 ))
      DATASET_ARGV=(--dataset-name random --random-prefix-len "$PFX" --random-input-len "$(( L - PFX ))" --random-output-len "$OUTLEN" --random-range-ratio 0); fi
    # Peak GPU memory of the server's processes over this cell (NVIDIA only). Both engines
    # preallocate their pools, so this is the configured footprint plus any transient workspace.
    memlog="$OUTDIR/in${L}_c${C}.mem"; mempid=
    if [ "${MEM_SAMPLE_MS:-1000}" != 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits \
        -lms "${MEM_SAMPLE_MS:-1000}" > "$memlog" 2>/dev/null &
      mempid=$!
    fi
    "$VLLM" bench serve --backend "$BENCH_BACKEND" \
      --base-url "http://127.0.0.1:$PORT" --endpoint "$ENDPOINT" \
      --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
      "${DATASET_ARGV[@]}" --request-rate inf --ignore-eos --temperature 0 \
      --max-concurrency "$C" --num-prompts "$NPROMPT" ${BENCH_EXTRA_ARGS:-} \
      --save-result --save-detailed --result-dir "$RESULT_DIR" --result-filename "in${L}_c${C}.json" \
      > "$blog" 2>&1 || true
    if [ -n "$mempid" ]; then
      kill "$mempid" 2>/dev/null || true; wait "$mempid" 2>/dev/null || true; mempid=
      # The server was `setsid`'d: its session is exactly its process tree.
      # shellcheck disable=SC2046
      peak="$(python3 "$WT/scripts/bench/gpu_peak_mem.py" "$memlog" $(ps -o pid= -s "$SRV") || true)"
      # Its own line, not a 14th column: other sweeps parse the 13-column row.
      if [ -n "$peak" ]; then echo "peak_mem_mib,$L,$C,$peak"; fi
    fi
    python3 - "$L" "$C" "$blog" <<'PY'
import math, re, sys
L, C, p = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t = open(p).read()
def g(pat):
    m = re.search(pat + r"\D*([\d.]+)", t)
    return float(m.group(1)) if m else float('nan')
vals = [g(r'Mean TTFT .ms.:'), g(r'Median TTFT .ms.:'), g(r'Mean TPOT .ms.:'),
        g(r'Median TPOT .ms.:'), g(r'Mean ITL .ms.:'), g(r'Median ITL .ms.:'),
        g(r'P99 ITL .ms.:'), g(r'Output token throughput .tok/s.:'),
        g(r'Request throughput .req/s.:'), g(r'Successful requests:'),
        g(r'Total generated tokens:')]
if not all(math.isfinite(v) for v in vals):
    print(f"# {L},{C} INCOMPLETE — see {p}", file=sys.stderr)
    sys.exit(0)
ttft, ttft_m, tpot, tpot_m, itl, itl_m, itl99, tps, rps, ok, gen = vals
print(f"{L},{C},{ttft:.2f},{ttft_m:.2f},{tpot:.3f},{tpot_m:.3f},{itl:.3f},{itl_m:.3f},"
      f"{itl99:.3f},{tps:.1f},{rps:.3f},{int(ok)},{int(gen)}")
PY
  done
done
