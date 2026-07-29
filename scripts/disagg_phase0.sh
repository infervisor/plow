#!/usr/bin/env bash
# §DISAGG phase 0 — price the prefill/decode wall split BEFORE building anything.
#
# `scripts/bench_plowrt_serve.sh` with `PLOW_PF_PACKLOG=1` and the server log
# sliced per concurrency cell, so each cell reports how much of the SERVER's
# wall went to prefill ticks (during which every live decode stream is stalled)
# vs decode ticks.
#
# On gfx950 the mux tick is EITHER a prefill OR a decode (`serve/mux.rs`, the
# AMD arm returns before the decode launch), so `mixed_decode_ns` is zero by
# construction and the disaggregation ceiling is `prefill/(prefill+decode)`.
#
# THE GO/NO-GO: a 1:1 prefill:decode split over N GPUs is only better than N
# co-located replicas if the prefill share exceeds 50%. Below that, disagg is a
# throughput LOSS at matched GPU count.
#
# $1 assets  $2 port  $3 tokenizer HF repo-id
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:?assets}"; PORT="${2:?port}"; TOKZ="${3:?tokenizer}"
READY="${READY:-1800}"
IN_LENS="${IN_LENS:-1024}"; CONCS="${CONCS:-1 4 16 64}"; NPROMPT="${NPROMPT:-64}"
OUTLEN="${OUTLEN:-128}"
IMAGE=rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
LOG="${LOG:-/tmp/disagg_phase0_$PORT.log}"
DOCKER="sudo -n docker"

echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
echo "PLOW_HSACO=${PLOW_HSACO:-<unset>}"
cd "$WT" || exit 1
: >"$LOG"

PLOW_PF_PACKLOG=1 setsid nix develop -c ./target/release/plowrt serve \
  --assets "$ASSETS" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
cleanup() {
  kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null
  sleep 2; kill -KILL -"$SRV" 2>/dev/null
  pkill -f "plowrt serve --assets $ASSETS" 2>/dev/null
  sleep 2
}
trap cleanup EXIT

for i in $(seq 1 "$READY"); do
  kill -0 $SRV 2>/dev/null || { echo "!! server died during load"; tail -40 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && {
    echo "== server ready after ${i}s"; break; }
  sleep 1
done
MODELS=$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models") || {
  echo "!! never became ready"; tail -40 "$LOG"; exit 1; }
MODEL=$(printf '%s' "$MODELS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "== slug: $MODEL"

# Coherence BEFORE timing (knob-contract, standing rule 3).
GATE=$(curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0}")
echo "$GATE"
echo "$GATE" | grep -qi paris && echo ">>> coherence gate: PASS" || {
  echo ">>> coherence gate: FAIL"; exit 1; }
echo

echo "input_len,conc,ttft_ms,ttft_med,ttft_p99,tpot_ms,itl_ms,itl_med,itl_p99,out_tok_s,pf_ns,dec_ns,pf_share,mixed_ns,pf_ticks,dec_ticks,dec_rows"
for L in $IN_LENS; do
  for C in $CONCS; do
    # Cumulative packlog BEFORE this cell.
    PRE=$(grep '^PACKLOG WALL' "$LOG" | tail -1)
    blog="/tmp/disagg0_${MODEL}_in${L}_c${C}.log"
    $DOCKER run --rm --network host \
      -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
      -v "$HOME/.cache/huggingface":/hf:ro \
      --entrypoint vllm "$IMAGE" \
      bench serve --backend openai-chat \
      --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions \
      --model "$MODEL" --tokenizer "$TOKZ" \
      --dataset-name random --random-input-len "$L" --random-output-len "$OUTLEN" \
      --max-concurrency "$C" --num-prompts "$NPROMPT" --percentile-metrics ttft,tpot,itl \
      --metric-percentiles 50,99 ${BENCH_EXTRA_ARGS:-} \
      > "$blog" 2>&1
    POST=$(grep '^PACKLOG WALL' "$LOG" | tail -1)
    python3 - "$L" "$C" "$blog" "$PRE" "$POST" <<'PY'
import re,sys
L,C,p,pre,post=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
t=open(p).read()
def g(pat):
    m=re.search(pat+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
def kv(s):
    return {k:int(v) for k,v in re.findall(r"(\w+)=(\d+)",s)}
a,b=kv(pre),kv(post)
d=lambda k: b.get(k,0)-a.get(k,0)
pf,dec,mx=d('prefill_ns'),d('decode_ns'),d('mixed_decode_ns')
share=pf/(pf+dec) if pf+dec else float('nan')
print(f"{L},{C},{g(r'Mean TTFT .ms.:'):.1f},{g(r'Median TTFT .ms.:'):.1f},{g(r'P99 TTFT .ms.:'):.1f},"
      f"{g(r'Mean TPOT .ms.:'):.2f},{g(r'Mean ITL .ms.:'):.2f},{g(r'Median ITL .ms.:'):.2f},"
      f"{g(r'P99 ITL .ms.:'):.2f},{g(r'Output token throughput .tok/s.:'):.1f},"
      f"{pf},{dec},{share:.4f},{mx},{d('prefill_ticks')},{d('decode_ticks')},{d('decode_rows')}")
PY
    tail -3 "$blog" | sed 's/^/    | /'
  done
done
echo
echo "== packlog (cumulative, final) =="
grep '^PACKLOG WALL' "$LOG" | tail -1
echo "== server log tail =="
tail -8 "$LOG"
