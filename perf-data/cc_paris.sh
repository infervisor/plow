#!/usr/bin/env bash
set -uo pipefail
export PLOW_LIBCUDA=/usr/local/cuda-13.0/compat/libcuda.so.1
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
PORT=8188
./target/release/plowrt serve --assets /workspace/assets/cc/b1 --port $PORT > /dev/shm/plow-serve.log 2>&1 &
SPID=$!
trap 'kill -9 $SPID 2>/dev/null; pkill -9 -f "plowrt serve" 2>/dev/null; for _ in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$u" -lt 1000 ]&&break; sleep 2; done' EXIT
t=0; while [ $t -lt 180 ]; do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ] && { echo "healthy ${t}s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "SERVE DIED"; tail -30 /dev/shm/plow-serve.log; exit 1; }
  sleep 3; t=$((t+3))
done
MID=$(curl -s http://localhost:$PORT/v1/models 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
echo "model id=$MID"
echo "== chat: capital of France (greedy) =="
curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":16,\"temperature\":0}" \
  2>/dev/null | python3 -c 'import sys,json
d=json.load(sys.stdin); print("PLOW >>>",repr(d["choices"][0]["message"]["content"]))' 2>&1
echo "PARIS-DONE"
