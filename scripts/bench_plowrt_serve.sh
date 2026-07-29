#!/usr/bin/env bash
# `vllm bench serve` against a plowrt OpenAI endpoint (§0-BENCH: same client
# binary, same metric definitions, different base-url).
#
# Runs the server and the client inside ONE leased command, so the server can
# never outlive the lease. The server is `setsid`'d and torn down by PROCESS
# GROUP: `nix develop -c` execs a shell that forks plowrt, so killing the pid we
# waited on leaves the real server holding the cards.
#
# $1 assets  $2 port  $3 model-slug  $4 tokenizer HF repo-id  $5 ready-timeout
#
# The tokenizer is named by REPO ID and resolved from the mounted HF cache, not
# bind-mounted by path: a snapshot dir is all symlinks into `../../blobs`, so
# mounting the snapshot alone gives the client a directory of dangling links and
# it dies in `convert_slow_tokenizer`.
#   IN_LENS  space-separated input lengths (default 1024)
#   CONCS    space-separated concurrencies (default 1) — AMD serve is batch=1,
#            so anything above 1 measures QUEUEING, not batching.
#   NPROMPT  prompts per point (default 8)
#   OUTLEN   output tokens per request (default 128)
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:?assets}"; PORT="${2:?port}"; MODEL="${3:?model}"; TOKZ="${4:?tokenizer}"
READY="${5:-1200}"
IN_LENS="${IN_LENS:-1024}"; CONCS="${CONCS:-1}"; NPROMPT="${NPROMPT:-8}"
OUTLEN="${OUTLEN:-128}"
IMAGE=rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
LOG="${LOG:-/tmp/plowrt_bench_$PORT.log}"
DOCKER="sudo -n docker"

echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
cd "$WT" || exit 1

setsid nix develop -c ./target/release/plowrt serve --assets "$ASSETS" --port "$PORT" \
  >"$LOG" 2>&1 &
SRV=$!
cleanup() {
  # Whole process group — see the header.
  kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null
  sleep 2
  kill -KILL -"$SRV" 2>/dev/null
  pkill -f "plowrt serve --assets $ASSETS" 2>/dev/null
  sleep 2
}
trap cleanup EXIT

for i in $(seq 1 "$READY"); do
  if ! kill -0 $SRV 2>/dev/null; then
    echo "!! server died during load; last 40 lines:"; tail -40 "$LOG"; exit 1
  fi
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && {
    echo "== server ready after ${i}s"; break; }
  sleep 1
done
curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" || {
  echo "!! never became ready"; tail -40 "$LOG"; exit 1; }
echo

# Coherence gate BEFORE any timing — a fast wrong server is not a result.
echo "== coherence gate =="
GATE=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0}")
echo "$GATE"
echo "$GATE" | grep -qi paris && echo ">>> coherence gate: PASS" || {
  echo ">>> coherence gate: FAIL — numbers below would be meaningless"; exit 1; }
echo

# plowrt implements /v1/chat/completions only, so the client runs vLLM's
# `openai-chat` backend. Same binary, same TTFT/TPOT/ITL definitions; the vLLM
# side must be re-measured with THIS backend before the two are tabled together.
# MEDIANS ARE REPORTED BESIDE THE MEANS, and `BENCH_EXTRA_ARGS` exists so both
# sides can be given the SAME client flags (`--num-warmups` above all).
#
# Why: at `NPROMPT=8` with no warm-up, one cold request owns the mean. Two runs
# of the IDENTICAL vLLM GLM-5.2 config measured Mean TTFT 1880.55 ms and 573.00
# ms — 3.3x apart — while their whole-run output throughputs agreed to 3%
# (27.03 vs 27.85 tok/s). The second run's own spread says why: median TTFT
# 137.20 ms, P99 3156.63 ms. A mean over 8 samples with a cold first request is
# not a number two engines can be compared on, and the fix is on the CLIENT, so
# it has to be available to both sides identically.
# Per-input-length prompt / warm-up counts. `NPROMPT_MAP` and `NWARM_MAP` are space-separated
# `<input_len>:<n>` pairs; anything not listed falls back to $NPROMPT / $BENCH_EXTRA_ARGS, so
# leaving them unset reproduces the previous behaviour exactly.
#
# Why this exists: prefill cost is superlinear in prompt length, so a flat prompt count makes
# the long-ctx points dominate the wall clock (measured GLM-5.2 TP4: TTFT 4.8 s @ 4k but
# 72.3 s @ 32k). The count MUST still be identical between the two engines at a given length,
# which is why the caller builds one map and passes the same string to both bench scripts
# rather than each script choosing for itself.
pick () { # <map> <key> <default>
  local kv; for kv in ${1:-}; do [ "${kv%%:*}" = "$2" ] && { echo "${kv##*:}"; return; }; done
  echo "$3"
}

echo "input_len,concurrency,ttft_ms,ttft_med,tpot_ms,tpot_med,itl_ms,itl_med,out_tok_s,req_per_s"
for L in $IN_LENS; do
  for C in $CONCS; do
    NP="$(pick "${NPROMPT_MAP:-}" "$L" "$NPROMPT")"
    NW="$(pick "${NWARM_MAP:-}" "$L" "")"
    WARM="${BENCH_EXTRA_ARGS:-}"; [ -n "$NW" ] && WARM="--num-warmups $NW"
    blog="/tmp/vllmbench_${MODEL}_in${L}_c${C}.log"
    $DOCKER run --rm --network host \
      -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
      -v "$HOME/.cache/huggingface":/hf:ro \
      --entrypoint vllm "$IMAGE" \
      bench serve --backend openai-chat \
      --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions \
      --model "$MODEL" --tokenizer "$TOKZ" \
      --dataset-name random --random-input-len "$L" --random-output-len "$OUTLEN" \
      --max-concurrency "$C" --num-prompts "$NP" $WARM \
      > "$blog" 2>&1
    python3 - "$L" "$C" "$blog" <<'PY'
import re,sys
L,C,p=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
t=open(p).read()
def g(pat):
    m=re.search(pat+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
print(f"{L},{C},{g(r'Mean TTFT .ms.:'):.2f},{g(r'Median TTFT .ms.:'):.2f},"
      f"{g(r'Mean TPOT .ms.:'):.3f},{g(r'Median TPOT .ms.:'):.3f},"
      f"{g(r'Mean ITL .ms.:'):.3f},{g(r'Median ITL .ms.:'):.3f},"
      f"{g(r'Output token throughput .tok/s.:'):.1f},"
      f"{g(r'Request throughput .req/s.:'):.3f}")
PY
    tail -3 "$blog" | sed 's/^/    | /'
  done
done
echo
echo "== server log tail =="
tail -8 "$LOG"
