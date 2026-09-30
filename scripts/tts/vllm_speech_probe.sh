#!/usr/bin/env bash
# vllm_speech_probe.sh <resdir> [bench-arm ...] — vLLM+SNAC speech server + tts_bench.py arms, ONE lease.
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
RES="${1:?resdir}"; shift
PYV="${VLLM_PY:-/root/tts-work/venv-vllm/bin/python}"
PY="${TTS_PY:-python3}"
mkdir -p "$RES"
PORT=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
setsid "$PYV" "$HERE/scripts/tts/vllm_speech_server.py" --port "$PORT" > "$RES/server.log" 2>&1 &
SP=$!
trap 'kill -TERM -- -$SP 2>/dev/null; sleep 3; kill -KILL -- -$SP 2>/dev/null' EXIT
for i in $(seq 1 900); do curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { tail -n 20 "$RES/server.log"; exit 1; }; sleep 1; done
for arm in "$@"; do
  # shellcheck disable=SC2086
  "$PY" "$HERE/scripts/tts/tts_bench.py" --url "http://127.0.0.1:$PORT" --model veena --out "$RES" $arm || exit 1
done
