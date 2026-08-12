#!/usr/bin/env bash
# =============================================================================
# scripts/bench_plow_rocm.sh — plowrt baseline sweep on ROCm, one model.
# =============================================================================
# The plow-side MIRROR of scripts/bench_vllm_rocm.sh. Deliberately the same
# CLIENT (`vllm bench serve`), the same PARSE, the same PHASES and the same CSV
# COLUMNS as the vLLM harness that produced perf-data/vllm-rocm/*.csv, so a plow
# row and a vLLM row can be subtracted without a translation step. Only the
# server differs.
#
# Two deltas from the vLLM harness, both forced:
#   - the client runs NATIVELY from a venv, not in the ROCm vLLM container. It
#     only tokenizes and drives HTTP, and the container's own tokenizer path
#     resolves HF repo ids while plow registers a model under the SLUG from its
#     weights.json. --tokenizer takes the local checkpoint dir instead.
#   - the coherence gate asks over /v1/chat/completions with the SAME question
#     ("capital of France") for the same reason: these are -it checkpoints.
#
# ONE UNAVOIDABLE METHODOLOGY DELTA, and it is not a knob: `vllm bench serve`
# defaults to --backend openai (POST /v1/completions), which is what the vLLM
# baselines were measured with, but plowrt implements ONLY /v1/chat/completions
# (serve/mod.rs routes exactly one completion path). So this harness must pass
# --backend openai-chat. The cost is NOT negligible on TTFT and the reason is
# bucket granularity, not the token count: the template adds ~14 tokens, and
# 1024+14 does not fit the 1024 bucket, so the engine plans chunks=[1024, 128]
# and prefills 1152 tokens' worth of work for 1038 real ones. Measured in the
# serve log: every in1024 request replans as [1024, 128], ~11% more prefill than
# a raw 1024-token /v1/completions prompt would have cost. TPOT is untouched.
# Quoted TTFT gaps should be read with that 11% credited to plow.
# Verified not to truncate: gen_toks comes back at exactly num_prompts x
# OUTPUT_LEN, so no random prompt trips a stop token early and shortens a decode.
#
# AMD serve is BATCH=1 today, so every concurrency above 1 measures QUEUEING and
# not batching. The general phase still runs the vLLM concurrency ladder because
# the comparison is against a vLLM row that DID batch — that gap is the finding,
# not a flaw in the measurement. It is restated in the CSV footer.
#
# Usage:
#   scripts/bench_plow_rocm.sh <assets-dir> <tokenizer-dir> [TP]
#   PHASES=ctxsweep scripts/bench_plow_rocm.sh /workspace/assets/gfx942/g12b ...
#
# Env: PHASES, CTXS, CONCS, GEN_CTX, OUTPUT_LEN, NUM_PROMPTS, QTAG, NO_LEASE=1
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:?usage: bench_plow_rocm.sh <assets-dir> <tokenizer-dir> [TP]}"
TOKZ="${2:?usage: bench_plow_rocm.sh <assets-dir> <tokenizer-dir> [TP]}"
TP="${3:-1}"

PORT="${PORT:-8137}"
OUTDIR="${OUTDIR:-$REPO/perf-data/plow-gfx942}"
PHASES="${PHASES:-general,ctxsweep}"
CTXS="${CTXS:-1024,4096,8192,16384,32768,65536}"
CONCS="${CONCS:-1,4,16,64}"
GEN_CTX="${GEN_CTX:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
QTAG="${QTAG:-bf16}"
GPULEASE="${GPULEASE:-$REPO/perf-data/tools/gpulease}"
BENCH="${BENCH:-/workspace/rocm7-bench-venv/bin/vllm}"
# PLOWRT_BIN exists because target/release/plowrt is SHARED. A concurrent agent
# running `cargo build -p plowrt` with no features replaces it mid-benchmark
# with a build that has no `hsa` — and that build does NOT fail, it serves from
# the CPU reference interpreter through the byte-fallback tokenizer, i.e. fluent
# garbage at "ready in 2s" instead of a 22 GiB weight upload.
PLOWRT_BIN="${PLOWRT_BIN:-$REPO/target/release/plowrt}"
# The slug plow registers the model under is weights.json's `network` field
# (orch/registry.rs) — NOT the directory name, and it is what clients pass as
# "model". Read it rather than making the caller repeat it.
SLUG="${SLUG:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["network"])' "$ASSETS/weights.json")}"
NAME="$(basename "$ASSETS")"

[ -x "$PLOWRT_BIN" ] || { echo "!!! $PLOWRT_BIN missing — cargo build --release -p plowrt --features hsa" >&2; exit 1; }
[ -x "$BENCH" ]      || { echo "!!! bench client $BENCH missing" >&2; exit 1; }
[ -d "$ASSETS/hsaco" ] || { echo "!!! $ASSETS/hsaco missing — run scripts/build_gfx942.sh and link it" >&2; exit 1; }

# --- lease first, then re-enter (same protocol as the vLLM harness).
if [ -z "${PLOW_LEASED:-}" ] && [ "${NO_LEASE:-0}" != "1" ]; then
  [ -x "$GPULEASE" ] || { echo "!!! gpulease missing at $GPULEASE" >&2; exit 1; }
  export PLOW_LEASED=1
  exec "$GPULEASE" -n "$TP" "plow-${NAME}-tp${TP}" "$0" "$@"
fi

mkdir -p "$OUTDIR"
LOG="$OUTDIR/${NAME}_${QTAG}_tp${TP}_serve.log"

# setsid + kill by PROCESS GROUP: killing the pid we waited on leaves the real
# server holding the cards, and the next lease then measures under contention.
setsid "$PLOWRT_BIN" serve --assets "$ASSETS" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
cleanup() {
  kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null
  sleep 2; kill -KILL -"$SRV" 2>/dev/null
  pkill -f "plowrt serve --assets $ASSETS" 2>/dev/null
  sleep 2
}
trap cleanup EXIT

echo ">>> serving $SLUG from $ASSETS on :$PORT (tp=$TP)"
ready=0
for i in $(seq 1 "$READY_TIMEOUT"); do
  kill -0 "$SRV" 2>/dev/null || { echo "!!! server died during startup"; tail -40 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo ">>> ready after ${i}s"; ready=1; break; }
  sleep 1
done
[ "$ready" = 1 ] || { echo "!!! endpoint never came up"; tail -40 "$LOG"; exit 1; }

# Coherence gate. Numbers from an incoherent server are worse than no numbers:
# a wrong-but-fast engine looks like a win. Same question as the vLLM harness.
COHERENT=1
txt=$(curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SLUG\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":64,\"temperature\":0}" \
      | python3 -c 'import sys,json
try:
    m=json.load(sys.stdin)["choices"][0]["message"]
    print(((m.get("content") or "")+" "+(m.get("reasoning_content") or "")).replace("\n"," "))
except Exception: print("")' 2>/dev/null)
echo ">>> sanity completion: $txt"
if echo "$txt" | grep -qi paris; then echo ">>> coherence gate: PASS"
else echo ">>> coherence gate: FAIL — numbers below are suspect" >&2; COHERENT=0; fi

# One bench point. Identical arg set and identical regex parse to the vLLM
# harness, so the two CSVs mean the same thing column for column.
bench_point() {  # $1=input_len $2=concurrency $3=num_prompts $4=tag
  local L="$1" C="$2" N="$3" tag="$4"
  local log="$OUTDIR/${NAME}_${QTAG}_tp${TP}_${tag}_in${L}_c${C}.log"
  HF_HUB_OFFLINE=1 "$BENCH" bench serve \
    --model "$SLUG" --tokenizer "$TOKZ" --dataset-name random \
    --random-input-len "$L" --random-output-len "$OUTPUT_LEN" \
    --max-concurrency "$C" --num-prompts "$N" --port "$PORT" \
    --backend openai-chat --endpoint /v1/chat/completions \
    ${BENCH_EXTRA_ARGS:-} > "$log" 2>&1
  python3 - "$L" "$C" "$log" <<'PY'
import re,sys
L=int(sys.argv[1]); C=int(sys.argv[2]); t=open(sys.argv[3]).read()
def g(p):
    m=re.search(p+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
ttft=g(r"Mean TTFT \(ms\):"); tpot=g(r"Mean TPOT \(ms\):"); itl=g(r"Mean ITL \(ms\):")
itl_med=g(r"Median ITL \(ms\):"); itl_p99=g(r"P99 ITL \(ms\):")
rps=g(r"Request throughput \(req/s\):"); ots=g(r"Output token throughput \(tok/s\):")
# `Successful requests` counts a REJECTED request as a success, so it cannot
# gate a point alone: a 131072 input-len point against a max_ctx=131072 blob had
# every prefill refused and still reported 4 successful requests at ITL 0.00.
# gen_toks must equal num_prompts x OUTPUT_LEN for the row to mean anything.
ok=g(r"Successful requests:"); gen=g(r"Total generated tokens:")
pf = L/(ttft/1000.0) if ttft==ttft and ttft>0 else float('nan')
dc = 1000.0/tpot if tpot==tpot and tpot>0 else float('nan')
print(f"{L},{C},{ttft:.2f},{pf:.1f},{tpot:.3f},{itl:.3f},{itl_med:.3f},{itl_p99:.3f},{dc:.2f},{rps:.3f},{ots:.1f},{ok:.0f},{gen:.0f}")
PY
}

HDR="input_len,concurrency,ttft_ms,prefill_tok_s,tpot_ms,itl_ms,itl_med_ms,itl_p99_ms,decode_tok_s,req_per_s,out_tok_s,ok_reqs,gen_toks"
BAD=0
record() {  # $1=row $2=csv
  echo "    $1"; echo "$1" >> "$2"
  case "$1" in *nan*) BAD=$((BAD+1))
    echo "    !!! bench point produced no measurement — see the .log beside $2" >&2
    kill -0 "$SRV" 2>/dev/null || { echo "    !!! SERVER IS GONE — aborting remaining points" >&2; exit 5; } ;;
  esac
}

# Warm-up, discarded. The first measured point after a cold server absorbs the
# lazy first-touch of every KV page and the first dispatch of each segment
# class; on the vLLM side that cost showed up as a 1930 ms TTFT on a model whose
# clean number was 37 ms.
echo ">>> warm-up (discarded)"
bench_point "$GEN_CTX" 1 4 warmup >/dev/null 2>&1 || true

if [[ ",$PHASES," == *",general,"* ]]; then
  csv="$OUTDIR/${NAME}_${QTAG}_tp${TP}_general.csv"; echo "$HDR" > "$csv"
  IFS=',' read -ra C_ARR <<<"$CONCS"
  for C in "${C_ARR[@]}"; do
    N=$((C * 8)); [ "$N" -lt 8 ] && N=8; [ "$N" -gt 256 ] && N=256
    echo ">>> general $SLUG tp=$TP ctx=$GEN_CTX conc=$C prompts=$N"
    record "$(bench_point "$GEN_CTX" "$C" "$N" general)" "$csv"
  done
  echo ">>> general summary: $csv"
fi

if [[ ",$PHASES," == *",ctxsweep,"* ]]; then
  csv="$OUTDIR/${NAME}_${QTAG}_tp${TP}_ctxsweep_c1.csv"; echo "$HDR" > "$csv"
  IFS=',' read -ra L_ARR <<<"$CTXS"
  for L in "${L_ARR[@]}"; do
    echo ">>> ctxsweep $SLUG tp=$TP input_len=$L conc=1"
    record "$(bench_point "$L" 1 "$NUM_PROMPTS" ctxsweep)" "$csv"
  done
  echo ">>> ctxsweep summary: $csv"
fi

[ "$COHERENT" = 1 ] || { echo ">>> RUN INCOHERENT — do not quote these numbers" >&2; exit 4; }
[ "$BAD" = 0 ] || { echo ">>> $BAD bench point(s) produced no measurement" >&2; exit 6; }
echo ">>> done: $OUTDIR"
