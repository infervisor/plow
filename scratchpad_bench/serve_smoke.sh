#!/usr/bin/env bash
# Smoke: batched AMD serve answers N CONCURRENT requests coherently.
# Two identical prompts must give identical text (the slot table must not mix
# sequences), and a third different one must answer its own question.
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
B="${B:-4}"
PORT="${PORT:-8123}"
ASSETS=/home/lava/plow/build-amd/g31b-db$B
SLUG=842da3794eaa0b77d5f08bae87a17459d91ff475
LOG=/tmp/plow_smoke_$PORT.log
cd "$WT" || exit 1

PLOW_HSACO=/home/lava/plow/build-amd/hsaco-b$B \
setsid nix develop -c ./target/release/plowrt serve --assets "$ASSETS" --port "$PORT" \
  >"$LOG" 2>&1 &
SRV=$!
cleanup() { kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null; sleep 2
            kill -KILL -"$SRV" 2>/dev/null; pkill -f "plowrt serve --assets $ASSETS" 2>/dev/null; sleep 2; }
trap cleanup EXIT

for i in $(seq 1 600); do
  kill -0 $SRV 2>/dev/null || { echo "!! server died"; tail -30 "$LOG"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "ready after ${i}s"; break; }
  sleep 1
done

ask() { # <n> <question>
  curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SLUG\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":24,\"temperature\":0}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('  [$1]', repr(d['choices'][0]['message']['content']))" 2>&1
}

echo "== $B concurrent requests: 2 identical + 2 different =="
ask A "What is the capital of France? Answer in one short sentence." &
ask B "What is the capital of France? Answer in one short sentence." &
ask C "What is 2+2? Answer with just the number." &
ask D "Name the largest planet in the solar system in one word." &
wait
echo
echo "== server log tail =="
tail -12 "$LOG"
