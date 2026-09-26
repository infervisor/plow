#!/usr/bin/env bash
# veena_serve_probe.sh <assets> <resdir> [client args...]
# Serves a Veena packet with plowrt and runs veena_plow_client.py against it, in ONE lease:
#   perf-data/tools/gpulease -n 1 veena-plow scripts/tts/veena_serve_probe.sh ...
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
source "$HERE/scripts/bench/plowbench.sh"
ASSETS="${1:?assets}"; RES="${2:?resdir}"; shift 2
RT="${PLOWRT_BIN:-$HERE/target/release/plowrt}"
PY="${TTS_PY:-python3}"
mkdir -p "$RES"
PORT=$(pb_free_port)
pb_serve_start "$RT" "$ASSETS" "$ASSETS" "$PORT" "$RES/serve.log" 3600
trap pb_serve_stop EXIT
pb_serve_wait 600 || exit 1
"$PY" "$HERE/scripts/tts/veena_plow_client.py" --port "$PORT" --out "$RES" "$@"
