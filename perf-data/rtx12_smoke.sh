#!/usr/bin/env bash
# rtx12_smoke.sh — quick end-to-end check that PLOW_PF_PACKLOG emits per-launch
# PACKLOG lines and that concurrent short prompts pack R>1. Brings the B=8 server
# up, fires N concurrent completions, greps PACKLOG, tears down. Run under flock.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN=/root/plow/target/release/plowrt
ASSETS=/root/gpu-assets-px1s2b8
PORT=8097
LOG=/root/plow/.claude/worktrees/agent-a3c067815da63ce31/perf-data/harness/rtx12/smoke.log
mkdir -p "$(dirname "$LOG")"
vram_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
SRVPID=""
cleanup(){ [ -n "$SRVPID" ] && { kill -TERM "$SRVPID" 2>/dev/null; for _ in $(seq 1 20); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 2; done; kill -KILL "$SRVPID" 2>/dev/null; }
  for _ in $(seq 1 20); do u=$(vram_used); echo "VRAM ${u}"; [ "${u:-99999}" -lt 34000 ] && break; sleep 3; done; }
trap cleanup EXIT
echo ">>> smoke: starting server, VRAM now $(vram_used) MiB"
NO_COLOR=1 RUST_LOG=info PLOW_PF_BATCH=1 PLOW_PF_PACKLOG=1 "$BIN" serve --assets "$ASSETS" --port "$PORT" > "$LOG" 2>&1 &
SRVPID=$!
for i in $(seq 1 900); do grep -q "serving OpenAI API over TCP" "$LOG" && break
  kill -0 "$SRVPID" 2>/dev/null || { echo "died"; tail -30 "$LOG"; exit 1; }; sleep 1; done
echo ">>> server up (pid $SRVPID), VRAM $(vram_used) MiB. Firing 8 concurrent short prompts."
for i in $(seq 1 8); do
  curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"gemma-4-12b-it\",\"prompt\":\"Write a short paragraph about the number $i and its history in mathematics and why it matters.\",\"max_tokens\":40,\"temperature\":0}" >/dev/null &
done
wait
sleep 2
echo "=== PACKLOG lines (launch) ==="; grep -c "PACKLOG R=" "$LOG"; grep "PACKLOG R=" "$LOG" | head -20
echo "=== R>=2 launches ==="; grep "PACKLOG R=" "$LOG" | grep -v "R=1 " | head
echo "=== WALL lines ==="; grep "PACKLOG WALL" "$LOG" | tail -3
