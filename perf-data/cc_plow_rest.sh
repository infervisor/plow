#!/usr/bin/env bash
# Remaining plow GPU work: occ2-vs-main decode delta at matched short ctx=128
# (occ2 has no matching prefill cubin -> decode-only prompt consumption, so keep
# ctx small), plus the plow Paris correctness gate via plowrt serve.
set -uo pipefail
export PLOW_LIBCUDA=/usr/local/cuda-13.0/compat/libcuda.so.1
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
SB=./target/release/examples/step_bench
CTX=128
echo "=== occ2-vs-main decode, ctx=$CTX ==="
for tag in main occ2; do
  for B in 1 4 8 16 32; do
    d=/workspace/assets/cc/b$B; [ "$tag" = occ2 ] && d=/workspace/assets/cc/occ2-b$B
    out=$($SB $d $B $CTX 128 2>/dev/null | grep RAW_STEP)
    agg=$(echo "$out" | sed -n 's/.*aggregate_tok_s=\([0-9.]*\).*/\1/p')
    pu=$(echo "$out" | sed -n 's/.*per_user_tok_s=\([0-9.]*\).*/\1/p')
    echo "$tag B=$B agg=$agg per_user=$pu"
  done
done
echo "=== PLOW PARIS GATE (plowrt serve, b1) ==="
PORT=8188
./target/release/plowrt serve --assets /workspace/assets/cc/b1 --port $PORT > /dev/shm/plow-serve.log 2>&1 &
SPID=$!
trap 'kill -9 $SPID 2>/dev/null; pkill -9 -f "plowrt serve" 2>/dev/null' EXIT
t=0; while [ $t -lt 120 ]; do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ] && break
  kill -0 $SPID 2>/dev/null || { echo "serve died"; tail -20 /dev/shm/plow-serve.log; break; }
  sleep 3; t=$((t+3))
done
echo ">>> served after ${t}s; models:"; curl -s http://localhost:$PORT/v1/models 2>/dev/null
MID=$(curl -s http://localhost:$PORT/v1/models 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
echo ">>> model id=$MID; asking capital of France:"
curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":16,\"temperature\":0}" \
  2>/dev/null | python3 -c 'import sys,json
try:
  d=json.load(sys.stdin); print("  PLOW >>>",repr(d["choices"][0]["message"]["content"]))
except Exception as e: print("  RAW:",sys.stdin.read() if False else e)' 2>&1
echo "=== DONE ==="
