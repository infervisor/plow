#!/usr/bin/env bash
# Final end-to-end check on the RE-EMITTED B=8 blob: server up, 6 concurrent
# chat requests, identical questions must give identical answers.
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
B=8; PORT=8125
ASSETS=/home/lava/plow/build-amd/g31b-db$B
LOG=/tmp/plow_final_$PORT.log
cd "$WT" || exit 1
PLOW_HSACO=/home/lava/plow/build-amd/hsaco-b$B \
setsid nix develop -c ./target/release/plowrt serve --assets "$ASSETS" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
cleanup() { kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null; sleep 2
            kill -KILL -"$SRV" 2>/dev/null; pkill -f "plowrt serve --assets $ASSETS" 2>/dev/null; sleep 2; }
trap cleanup EXIT
for i in $(seq 1 600); do
  kill -0 $SRV 2>/dev/null || { echo "!! server died"; tail -20 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "ready ${i}s"; break; }
  sleep 1
done
PORT=$PORT bash "$WT/scratchpad_bench/concur_check.sh"
echo "== engine log =="
grep -E "batch=|KV slot" "$LOG" | tail -3
