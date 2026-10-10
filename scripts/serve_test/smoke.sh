#!/usr/bin/env bash
# smoke.sh <plowrt> <outdir> <plowrt serve args...>: start one `plowrt serve` in a clean
# environment (no PLOW_*, no LD_LIBRARY_PATH: serve settings come from the bundles' serve.json),
# run smoke_client.py against every endpoint it advertises, stop it. GPU: run under the queue,
#   scripts/bench/gpuq.py submit smoke 1 scripts/serve_test/smoke.sh <plowrt> <out> \
#       --assets <bundle>[,checkpoint=<hf dir>] [--assets ...] [--asr-vad-packet <vad.pkt>]
# Env: PY (python3 for the clients), ASR_MANIFEST ([{path, text}] 16 kHz WAVs; ASR legs skip
# without it), VAD_WAV, SMOKE_ARGS (extra smoke_client.py args, e.g. "--tts veena=kavya"),
# TTS_CHECK=1 (Whisper CER of the TTS WAVs via scripts/tts/asr_check.py, --max-cer TTS_MAX_CER),
# EVAL=asr|tts|llm|all (also run eval.py into <outdir>/eval), READY_S (1200).
set -u
[ $# -ge 3 ] || { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
plowrt=$1 out=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-python3}
rm -rf "$out"; mkdir -p "$out"
source "$REPO/scripts/bench/plowbench.sh"
PB_SERVER_PORT=$(pb_free_port); PB_SERVER_LOG=$out/server.log
URL=http://127.0.0.1:$PB_SERVER_PORT
echo "env -i PATH=/usr/bin:/bin HOME=$HOME $plowrt serve $* --port $PB_SERVER_PORT" > "$out/cmd.txt"
env -i PATH=/usr/bin:/bin HOME="$HOME" "$plowrt" serve "$@" --port "$PB_SERVER_PORT" > "$PB_SERVER_LOG" 2>&1 &
PB_SERVER_PID=$!
trap pb_serve_stop EXIT
if ! pb_serve_wait "${READY_S:-1200}"; then echo "SMOKE rc=3 (not ready)"; tail -30 "$PB_SERVER_LOG"; exit 3; fi
grep -o '/[^ ]*libcublasLt[^ ]*' "/proc/$PB_SERVER_PID/maps" 2>/dev/null | sort -u > "$out/cublaslt_loaded.txt"
# shellcheck disable=SC2086
"$PY" "$HERE/smoke_client.py" "$URL" "$out" ${SMOKE_ARGS:-}
rc=$?
if [ -n "${EVAL:-}" ]; then
    PLOW_URL=$URL "$PY" "$HERE/eval.py" "$EVAL" --out "$out/eval" ${ASR_MANIFEST:+--manifest "$ASR_MANIFEST"} || rc=5
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$out/gpu_mem.txt"
pb_serve_stop
if [ "${TTS_CHECK:-0}" = 1 ] && [ -f "$out/texts.json" ]; then
    "$PY" "$REPO/scripts/tts/asr_check.py" "$out"/smoke_*.wav --texts "$out/texts.json" \
        --max-cer "${TTS_MAX_CER:-0.3}" > "$out/asr_check.txt" 2> "$out/asr_check.log" || rc=6
    cat "$out/asr_check.txt"
fi
echo "SMOKE rc=$rc"
exit $rc
