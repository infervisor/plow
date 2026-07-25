#!/usr/bin/env bash
# Matched-concurrency vLLM sweep for the plow concurrent-decode comparison.
# Serve ONCE (max-num-seqs 32, no prefix cache), bench at concurrency B in {4,8,16,32}
# with input=1024 output=128 (== plow step_bench). Metrics: Output token throughput
# (aggregate tok/s) + Mean TPOT (per-user). ignore-eos + temp0 -> exactly 128 out toks.
set -uo pipefail
VENV=/workspace/venvs/vllm-blk
MODEL=/workspace/models/gemma-4-26B-A4B-it
PORT=8137
OUTDIR=/dev/shm/cc-vllm
mkdir -p "$OUTDIR"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
export PATH=/workspace/venvs/vllm-blk/bin:/usr/local/cuda/bin:$PATH
export VLLM_LOGGING_LEVEL=WARNING
SRVLOG="$OUTDIR/serve.log"; SRVPID=""
vram_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1; }
cleanup(){
  [ -n "$SRVPID" ] && { kill -TERM "$SRVPID" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$SRVPID" 2>/dev/null||break; sleep 2; done; kill -KILL "$SRVPID" 2>/dev/null; pkill -f "vllm serve $MODEL" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null; }
  for _ in $(seq 1 40); do u=$(vram_used); [ "$u" -lt 1000 ]&&break; sleep 3; done
}
trap cleanup EXIT

echo ">>> serving $MODEL"
"$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name bench --dtype bfloat16 --tensor-parallel-size 1 \
  --max-num-seqs 32 --no-enable-prefix-caching \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
  --max-model-len 2048 --port "$PORT" > "$SRVLOG" 2>&1 &
SRVPID=$!
t=0
while [ "$t" -lt 900 ]; do
  kill -0 "$SRVPID" 2>/dev/null || { echo "!!! server died"; tail -40 "$SRVLOG"; exit 1; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health)" = "200" ] && { echo ">>> healthy ${t}s"; break; }
  sleep 5; t=$((t+5))
done
[ "$t" -ge 900 ] && { echo "!!! never healthy"; tail -40 "$SRVLOG"; exit 1; }
echo ">>> VRAM after load: $(vram_used) MiB"
grep -iE "GPU KV cache size|Maximum concurrency|model weights take" "$SRVLOG" | tail -3

echo ">>> PARIS gate:"
curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"bench","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":16,"temperature":0}' \
  | python3 -c 'import sys,json;print("  >>>",repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))' 2>&1

echo "concurrency,num_prompts,dur_s,agg_tok_s,mean_tpot_ms,per_user_tok_s,mean_ttft_ms" > "$OUTDIR/summary.csv"
for B in 4 8 16 32; do
  NP=$((B*4)); [ "$NP" -lt 32 ] && NP=32
  log="$OUTDIR/b${B}.log"
  echo ">>> bench concurrency=$B num_prompts=$NP"
  "$VENV/bin/vllm" bench serve \
    --model "$MODEL" --served-model-name bench --dataset-name random \
    --random-input-len 1024 --random-output-len 128 --ignore-eos --temperature 0 \
    --max-concurrency "$B" --num-prompts "$NP" --port "$PORT" --endpoint /v1/completions \
    > "$log" 2>&1
  row=$(python3 - "$B" "$NP" "$log" <<'PY'
import re,sys
B,NP,f=sys.argv[1],sys.argv[2],sys.argv[3]; t=open(f).read()
def g(p):
    m=re.search(p+r"\D*([\d.]+)",t); return float(m.group(1)) if m else float('nan')
agg=g(r"Output token throughput \(tok/s\):"); tpot=g(r"Mean TPOT \(ms\):")
dur=g(r"Benchmark duration \(s\):"); ttft=g(r"Mean TTFT \(ms\):")
pu=1000.0/tpot if tpot==tpot and tpot>0 else float('nan')
print(f"{B},{NP},{dur:.2f},{agg:.1f},{tpot:.3f},{pu:.1f},{ttft:.2f}")
PY
)
  echo "    $row"; echo "$row" >> "$OUTDIR/summary.csv"
done
echo ">>> DONE"; cat "$OUTDIR/summary.csv"
