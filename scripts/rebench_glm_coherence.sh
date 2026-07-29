#!/usr/bin/env bash
# CORRECTNESS BEFORE SPEED (knob-contract §5). The stacked blob (prefill buckets +
# vocab-parallel lm_head + co-resident shared expert) has never been run, and a sharded
# lm_head on a PREFILL bucket has never been run at TP4 at all — that failure shows as
# WRONG TOKENS, not a crash, so it has to be read.
#
# Short prompt  -> exercises the T=128 bucket.
# Long prompt   -> exercises the T=1024 bucket (the one the benchmark uses).
# Streaming     -> checks the first SSE chunk carries CONTENT (the 63f9957 TTFT artefact
#                  must not come back: role must ride the first token).
#
# $1 assets dir  $2 port  $3 model slug
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:?assets}"; PORT="${2:?port}"; MODEL="${3:?model}"
READY="${READY:-900}"
LOG="${LOG:-/tmp/glm_coherence_$PORT.log}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
cd "$WT" || exit 1

setsid nix develop -c ./target/release/plowrt serve --assets "$ASSETS" --port "$PORT" \
  >"$LOG" 2>&1 &
SRV=$!
cleanup() {
  kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null
  sleep 2; kill -KILL -"$SRV" 2>/dev/null
  pkill -f "plowrt serve --assets $ASSETS" 2>/dev/null
  sleep 2
}
trap cleanup EXIT

for i in $(seq 1 "$READY"); do
  kill -0 $SRV 2>/dev/null || { echo "!! server died during load:"; tail -40 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && {
    echo "== server ready after ${i}s"; break; }
  sleep 1
done
curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "!! never became ready"; tail -60 "$LOG"; exit 1; }

grep -E 'progs=|decode_only|prefill=|n_kvrow|prefill object|STALE' "$LOG" | tail -8
echo

ask() { # <label> <prompt> <max_tokens>
  echo "---- $1"
  python3 - "$PORT" "$MODEL" "$3" <<'PY' "$2"
import json,sys,urllib.request
port,model,mt=sys.argv[1],sys.argv[2],int(sys.argv[3]); prompt=sys.argv[4]
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],
                     "max_tokens":mt,"temperature":0}).encode(),
    headers={"Content-Type":"application/json"})
d=json.load(urllib.request.urlopen(req,timeout=900))
print(repr(d["choices"][0]["message"]["content"]))
print("  usage:",d.get("usage"))
PY
}

ask "short / T=128 bucket" "What is the capital of France? Answer in one short sentence." 32
ask "reasoning / T=128 bucket" "Compute 17 * 23 and then state the result in words." 48
# ~1100 words -> ~1024+ prompt tokens, the bucket the benchmark actually uses.
LONG="$(python3 -c "print(('The quick brown fox jumps over the lazy dog near the riverbank at dawn while the farmer counts his sheep. ')*95 + 'Ignoring the repeated sentence above, what is the capital of Japan? Answer in one short sentence.')")"
ask "long / T=1024 bucket" "$LONG" 40

echo "---- streaming: first chunk must carry CONTENT (63f9957 TTFT artefact)"
curl -sN --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colours.\"}],\"max_tokens\":24,\"stream\":true}" \
  | head -3

echo
echo "== server log tail =="
tail -12 "$LOG"
