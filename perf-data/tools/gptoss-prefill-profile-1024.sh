#!/usr/bin/env bash
# Fresh per-op prefill profile at 1024 tokens on current code, to compute the AMX MoE prefill
# dot's efficiency against this part's measured TMUL ceiling. The 928 ms/thr MoE figure on record
# predates the epilogue transpose, the L2 token chunking and the M-split.
set -u
cd /home/lava/plow/.claude/worktrees/cpu-backend
T=/home/lava/.claude/jobs/c77c21db/tmp
LOG=$T/prof1024.log
: > "$LOG"
GK=/home/lava/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
env PLOW_MXFP4_DIR=/home/lava/models/gpt-oss-20b-mxfp4-dense \
  timeout 1800 ./target/release/examples/cpu_profile "$T/gptoss-16-mx4/model.pkt" "$GK" \
  --threads 16 --prefill --prompt-tokens 1024 >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
tail -45 "$LOG"
