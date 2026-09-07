#!/usr/bin/env bash
# Interleaved A/B of PLOW_MOE_INT8 on GPT-OSS-20B summarize c=1 TTFT, the one remaining
# non-fusion cell where vLLM wins (1829 ms). PLOW_MOE_INT8 is a RUNTIME gate touching only
# x_moe_glu_mx_pf / x_moe_down_mx_pf, so both arms share one blob and decode is untouched.
# Interleaved because this box drifts ~5% over hours; same-session pairs only.
set -u
cd /home/lava/plow/.claude/worktrees/cpu-backend
T=/home/lava/.claude/jobs/c77c21db/tmp
BLOB=$T/gptoss-16-mx4
TOK=/home/lava/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
LOG=$T/i8_ab.log
: > "$LOG"

one() { # arm(0|1) rep
  local arm=$1 rep=$2 tag="i8${1}r${2}"
  {
    echo "==== PLOW_MOE_INT8=$arm rep=$rep  $(date +%T)"
    env PLOW_MOE_INT8="$arm" PLOW_MXFP4_DIR=/home/lava/models/gpt-oss-20b-mxfp4-dense \
      setsid nohup ./target/release/plowrt serve --assets "$BLOB" --port 8096 \
      > "$T/serve_$tag.log" 2>&1 < /dev/null &
    echo $! > "$T/i8.pid"
    local ok=0
    for i in $(seq 1 240); do
      curl -sf localhost:8096/v1/models >/dev/null 2>&1 && { ok=1; break; }
      sleep 5
      kill -0 "$(cat "$T/i8.pid")" 2>/dev/null || break
    done
    [ "$ok" = 1 ] || { echo "SERVER DIED"; tail -5 "$T/serve_$tag.log"; return; }
    local M
    M=$(curl -s localhost:8096/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
    echo "ready $(date +%T) model=$M"
    env LD_LIBRARY_PATH=/home/lava/vllm-cpu/gcc-lib/lib /home/lava/vllm-cpu/venv/bin/python \
      tools/bench-api/bench.py --base-url http://localhost:8096 --model "$M" \
      --workload summarize --concurrency 1 --requests 8 --max-tokens 64 --fresh-prompts \
      --tokenizer "$TOK" --out "$T/$tag.json" --md "$T/$tag.md" 2>&1 |
      grep -E "^\| summarize|Error|error"
    kill "$(cat "$T/i8.pid")" 2>/dev/null
    until ! pgrep -x plowrt >/dev/null; do sleep 2; done
    sleep 5
  } >> "$LOG" 2>&1
}

for rep in 1 2; do
  one 0 "$rep"
  one 1 "$rep"
done
echo "==== DONE $(date +%T)" >> "$LOG"
