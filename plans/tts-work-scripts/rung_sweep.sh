#!/usr/bin/env bash
# rung_sweep.sh <plowrt> <assets> <tag> — greedy decode ITL per concurrency (one load each).
RT=$1; A=$2; TAG=$3
for c in ${CONCS:-1 2 4 8 16}; do
  $RT bench --assets $A --random-input-len 64 --output-len ${OUTLEN:-384} --concurrency $c \
     --requests $((c*2)) --warmup-requests $c 2>/dev/null > /tmp/rung_$$.json
  /root/tts-work/venv-ref/bin/python -c "
import json,sys
d=json.load(open('/tmp/rung_$$.json'))
it=d['itl_ms']; n=d['output_tokens']; dur=d['duration_ms']
# multi-step emits K tokens per sync: mean ITL = step time; also report tok/s.
print(f\"$TAG c=$c itl_mean={it['mean']:.3f}ms out_tok_s={d['output_token_throughput']:.0f} step_ms_est={1000*$c/d['output_token_throughput']:.3f}\")
"
done
