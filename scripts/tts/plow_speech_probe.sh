#!/usr/bin/env bash
# plow_speech_probe.sh <assets> <resdir> [bench-arm ...]
# Serves a speech packet with plowrt and runs tts_bench.py arms against /v1/audio/speech, all in
# ONE lease:  perf-data/tools/gpulease -n 1 tts-plow scripts/tts/plow_speech_probe.sh A R "--conc 1 --stream"
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
source "$HERE/scripts/bench/plowbench.sh"
ASSETS="${1:?assets}"; RES="${2:?resdir}"; shift 2
RT="${PLOWRT_BIN:-$HERE/target/release/plowrt}"
PY="${TTS_PY:-python3}"
mkdir -p "$RES"
PORT=$(pb_free_port)
pb_serve_start "$RT" "$ASSETS" "$ASSETS" "$PORT" "$RES/serve.log" 7200
trap pb_serve_stop EXIT
pb_serve_wait 600 || exit 1
MODEL=$(pb_model_id)
for arm in "$@"; do
  # shellcheck disable=SC2086
  "$PY" "$HERE/scripts/tts/tts_bench.py" --url "http://127.0.0.1:$PORT" --model "$MODEL" --out "$RES" $arm || exit 1
done
