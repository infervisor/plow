#!/usr/bin/env bash
# Token-identity gate for the GLM-5.2 prefill program.
#
# Runs ONE prompt through a plowrt endpoint and dumps the completion. Point it at
# the decode-only bundle and at the prefill bundle in turn: the two must produce
# the SAME text. The decode path is the trusted oracle — it is token-identical to
# `runtime/tests/glm52_decode.c` (24/24, commit 52631b2) — so a prefill that
# changes the tokens is a REGRESSION, not a win, however much faster it is.
#
# Greedy by construction: the device samples and the host never sees the logit
# row, so temperature/top_p are ignored and the comparison is deterministic.
#
#   $1 bundle dir   $2 port   $3 output json
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="${1:?bundle}"; PORT="${2:?port}"; OUT="${3:?out}"
LOG="/tmp/srv_$PORT.log"
cd "$WT" || exit 1

echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
# setsid + process-group teardown: `nix develop -c` execs a shell that forks
# plowrt, so killing the pid we waited on would leave the server holding cards.
setsid nix develop -c ./target/release/plowrt serve --assets "$BUNDLE" --port "$PORT" \
  >"$LOG" 2>&1 &
SRV=$!
cleanup() {
  kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null
  sleep 2
  kill -KILL -"$SRV" 2>/dev/null
}
trap cleanup EXIT

for i in $(seq 1 900); do
  kill -0 "$SRV" 2>/dev/null || { echo "!! server died during load"; tail -40 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && {
    echo "== ready after ${i}s"; break; }
  sleep 1
done
curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "!! never became ready"; tail -40 "$LOG"; exit 1; }

curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-plow","messages":[{"role":"user","content":"Explain in one paragraph why tensor parallelism reduces per-GPU memory for large language models."}],"max_tokens":48,"stream":false}' \
  >"$OUT"

echo "== completion =="
head -c 2000 "$OUT"; echo
echo "== load/dispatch evidence =="
grep -iE "decode_only|dense-FFN|prefill|bucket|n_prog|arms|AMD serve engine ready" "$LOG" | tail -15
