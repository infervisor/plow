#!/usr/bin/env bash
# `vllm bench serve` against a plowrt OpenAI endpoint (§0-BENCH: same client
# binary, same metric definitions, different base-url). Set BENCH_BACKEND=openai
# for raw `/v1/completions`; the default remains `openai-chat`.
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
#   CONCS    space-separated concurrencies (default 1). The packet's compiled
#            ladder determines whether wider points batch or queue.
#   NPROMPT  prompts per point (default 8)
#   OUTLEN   output tokens per request (default 128)
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:?assets}"; PORT="${2:?port}"; MODEL="${3:?model}"; TOKZ="${4:?tokenizer}"
READY="${5:-1200}"
IN_LENS="${IN_LENS:-1024}"; CONCS="${CONCS:-1}"; NPROMPT="${NPROMPT:-8}"
OUTLEN="${OUTLEN:-128}"
BENCH_BACKEND="${BENCH_BACKEND:-openai-chat}"
BENCH_TRUST_REMOTE_CODE="${BENCH_TRUST_REMOTE_CODE:-0}"
# PLOWRT_BIN exists because `target/release/plowrt` is SHARED. A concurrent agent
# running `cargo build -p plowrt` (no features) replaces the binary mid-benchmark
# with one built without `hsa`, and that build does not fail — it serves from the
# CPU reference interpreter through the byte-fallback tokenizer, i.e. fluent
# garbage at "ready after 2s" instead of a 12 s weight upload. Point this at a
# private copy so another agent's build cannot silently change what is measured.
PLOWRT_BIN="${PLOWRT_BIN:-./target/release/plowrt}"
# TOKZ_MOUNT adds one more read-only bind for models whose HF snapshot has no
# loadable fast tokenizer. Kimi-K3 ships `tiktoken.model` plus a custom
# `tokenization_kimi.py` and declares `tokenizer_class=TikTokenTokenizer` with
# an `auto_map`, so pointing the client at the snapshot needs trust_remote_code
# AND tiktoken inside the image. Build a plain dir (tokenizer.json +
# tokenizer_class=PreTrainedTokenizerFast) and pass
#   TOKZ_MOUNT='-v /path/k3_tokz:/tokz:ro' TOKZ=/tokz
# The tokenizer MUST still be the one the server uses: input-len control and
# every token count in the report are computed with it.
TOKZ_MOUNT="${TOKZ_MOUNT:-}"
IMAGE="${IMAGE:-rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0}"
LOG="${LOG:-/tmp/plowrt_bench_$PORT.log}"
DOCKER="${DOCKER:-docker}"

case "$BENCH_BACKEND" in
  openai-chat) ENDPOINT=/v1/chat/completions ;;
  openai) ENDPOINT=/v1/completions ;;
  *) echo "FAIL: BENCH_BACKEND must be openai-chat or openai" >&2; exit 2 ;;
esac
case "$BENCH_TRUST_REMOTE_CODE" in
  0) TRUST_ARGS="" ;;
  1) TRUST_ARGS="--trust-remote-code" ;;
  *) echo "FAIL: BENCH_TRUST_REMOTE_CODE must be 0 or 1" >&2; exit 2 ;;
esac

echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
cd "$WT" || exit 1

setsid nix develop -c "$PLOWRT_BIN" serve --assets "$ASSETS" --port "$PORT" \
  >"$LOG" 2>&1 &
SRV=$!
cleanup() {
  # Whole process group — see the header.
  kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null || true
  sleep 2
  kill -KILL -"$SRV" 2>/dev/null || true
  pkill -f "plowrt serve --assets $ASSETS" 2>/dev/null || true
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
if [ "$BENCH_BACKEND" = openai ]; then
  GATE=$(curl -s --max-time 300 "http://127.0.0.1:$PORT$ENDPOINT" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0}")
else
  GATE=$(curl -s --max-time 300 "http://127.0.0.1:$PORT$ENDPOINT" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0}")
fi
echo "$GATE"
echo "$GATE" | grep -qi paris && echo ">>> coherence gate: PASS" || {
  echo ">>> coherence gate: FAIL — numbers below would be meaningless"; exit 1; }
echo

# The two endpoint modes let the client contract match the comparison server:
# never table an `openai-chat` result against an `openai` result.
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

echo "input_len,concurrency,ttft_ms,ttft_med,tpot_ms,tpot_med,itl_ms,itl_med,itl_p99,out_tok_s,req_per_s,ok_reqs,gen_toks"
for L in $IN_LENS; do
  for C in $CONCS; do
    NP="$(pick "${NPROMPT_MAP:-}" "$L" "$NPROMPT")"
    NW="$(pick "${NWARM_MAP:-}" "$L" "")"
    WARM="${BENCH_EXTRA_ARGS:-}"; [ -n "$NW" ] && WARM="--num-warmups $NW"
    blog="/tmp/vllmbench_${MODEL}_in${L}_c${C}.log"
    $DOCKER run --rm --network host \
      -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf -e HF_MODULES_CACHE=/tmp/hf_modules \
      -v "$HOME/.cache/huggingface":/hf:ro $TOKZ_MOUNT \
      --entrypoint vllm "$IMAGE" \
      bench serve --backend "$BENCH_BACKEND" \
      --base-url "http://127.0.0.1:$PORT" --endpoint "$ENDPOINT" \
      --model "$MODEL" --tokenizer "$TOKZ" $TRUST_ARGS \
      --dataset-name random --random-input-len "$L" --random-output-len "$OUTLEN" \
      --random-range-ratio 0 --request-rate inf --ignore-eos --temperature 0 \
      --max-concurrency "$C" --num-prompts "$NP" $WARM \
      > "$blog" 2>&1
    python3 - "$L" "$C" "$blog" "$NP" "$OUTLEN" <<'PY'
import math,re,sys
L,C,p,expected,outlen=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
t=open(p).read()
def g(pat):
    m=re.search(pat+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
ok=g(r'Successful requests:')
failed=g(r'Failed requests:')
gen=g(r'Total generated tokens:')
required=[g(r'Mean TTFT .ms.:'),g(r'Median TTFT .ms.:'),g(r'Mean TPOT .ms.:'),
          g(r'Median TPOT .ms.:'),g(r'Mean ITL .ms.:'),g(r'Median ITL .ms.:'),
          g(r'P99 ITL .ms.:'),g(r'Output token throughput .tok/s.:'),
          g(r'Request throughput .req/s.:'),ok,failed,gen]
if not all(math.isfinite(v) for v in required):
    raise SystemExit(f"FAIL: incomplete vllm bench output in {p}")
if int(ok) != expected or int(failed) != 0 or int(gen) != expected*outlen:
    raise SystemExit(f"FAIL: incomplete cell: ok={ok:g}/{expected} failed={failed:g} generated={gen:g}/{expected*outlen}")
# P99 ITL beside mean/median: the tail is what a stream is judged on, and it is
# the metric the two engines are tabled against.
print(f"{L},{C},{g(r'Mean TTFT .ms.:'):.2f},{g(r'Median TTFT .ms.:'):.2f},"
      f"{g(r'Mean TPOT .ms.:'):.3f},{g(r'Median TPOT .ms.:'):.3f},"
      f"{g(r'Mean ITL .ms.:'):.3f},{g(r'Median ITL .ms.:'):.3f},"
      f"{g(r'P99 ITL .ms.:'):.3f},"
      f"{g(r'Output token throughput .tok/s.:'):.1f},"
      f"{g(r'Request throughput .req/s.:'):.3f},"
      # `Successful requests` COUNTS A REJECTED REQUEST AS A SUCCESS, so it
      # cannot gate a point on its own. Measured: a 131072 input-len point
      # against a max_ctx=131072 blob had every prefill refused ("prompt is
      # 131085 tokens" — the chat template adds ~13) and still reported
      # 4 successful requests, 99.1 tok/s and 3.42 req/s with ITL 0.00.
      # `gen_toks` is the honest check: it must equal num_prompts x OUTLEN.
      f"{ok:.0f},"
      f"{gen:.0f}")
PY
    tail -3 "$blog" | sed 's/^/    | /'
  done
done
echo
echo "== server log tail =="
tail -8 "$LOG"
