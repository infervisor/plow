#!/usr/bin/env bash
# emit one line per completed cell; exit on "MODE $M done" or server death
set -u
M="${1:-1}"
DIR=/root/plow/.claude/worktrees/agent-a3c067815da63ce31/perf-data/harness/rtx12
RUNLOG="$DIR/mode$M.runlog"
prev=0
while true; do
  n=$(grep -c "CELL .* done" "$RUNLOG" 2>/dev/null || echo 0)
  if [ "$n" -gt "$prev" ]; then
    grep -E "CELL .* done|FAILED" "$RUNLOG" | tail -n $((n-prev))
    prev=$n
  fi
  if grep -q "MODE $M done" "$RUNLOG" 2>/dev/null; then echo "MODE${M}_COMPLETE"; break; fi
  if ! pgrep -f "serve --assets /root/gpu-assets-px1s2b8 --port 8097" >/dev/null 2>&1; then
    if grep -q "MODE $M done" "$RUNLOG" 2>/dev/null; then echo "MODE${M}_COMPLETE"; else echo "SERVER_GONE"; fi
    break
  fi
  sleep 20
done
