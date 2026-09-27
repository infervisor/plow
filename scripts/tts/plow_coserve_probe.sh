#!/usr/bin/env bash
# plow_coserve_probe.sh <resdir> <asr-assets> <veena-assets> <chatterbox-assets>
# One plowrt serving all three speech models on one GPU, in ONE lease:
#   perf-data/tools/gpulease -n 1 coserve scripts/tts/plow_coserve_probe.sh R A V C
# Phases: each model alone (gates + throughput), switch latency (switch_bench.py), then all
# three under concurrent load. Env: PLOWRT_BIN, TTS_PY, MANIFEST (served_bench manifest),
# SERVE_ARGS (extra plowrt serve flags, e.g. "--co-sched rr").
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
source "$HERE/scripts/bench/plowbench.sh"
RES="${1:?resdir}"; A="${2:?asr}"; V="${3:?veena}"; C="${4:?chatterbox}"
RT="${PLOWRT_BIN:-$HERE/target/release/plowrt}"
PY="${TTS_PY:-python3}"
MANIFEST="${MANIFEST:?served_bench manifest}"
mkdir -p "$RES"
PB_SERVER_PORT=$(pb_free_port); PB_SERVER_LOG="$RES/serve.log"
# shellcheck disable=SC2086
timeout --foreground --kill-after=10s -s TERM 7200 \
    "$RT" serve --assets "$A" --assets "$V" --assets "$C" --port "$PB_SERVER_PORT" ${SERVE_ARGS:-} > "$PB_SERVER_LOG" 2>&1 &
PB_SERVER_PID=$!
trap pb_serve_stop EXIT
pb_serve_wait 900 || exit 1
URL="http://127.0.0.1:$PB_SERVER_PORT"
curl -fsS "$URL/v1/models" > "$RES/models.json"
AM=$(basename "$A"); VM=$(basename "$V"); CM=$(basename "$C")
CLIP=$("$PY" -c "import json;print(json.load(open('$MANIFEST'))[0]['path'])")
asr() { "$PY" "$HERE/scripts/asr/nvidia/served_bench.py" --url "$URL" --model "$AM" --manifest "$MANIFEST" "$@"; }
vtts() { "$PY" "$HERE/scripts/tts/tts_bench.py" --url "$URL" --model "$VM" --out "$RES/veena" "$@"; }
ctts() { "$PY" "$HERE/scripts/tts/tts_bench.py" --url "$URL" --model "$CM" --out "$RES/chatterbox" --prompt-set chatterbox --voice default "$@"; }

echo "== solo"
asr --conc 1,16 --tag solo || exit 1
vtts --conc 1 --stream --wav --n 16 --tag solo_stream_c1 || exit 1
vtts --conc 8 --stream --n 32 --tag solo_stream_c8 || exit 1
ctts --conc 1 --wav --n 16 --tag solo_full_c1 || exit 1
ctts --conc 8 --n 32 --tag solo_full_c8 || exit 1

echo "== switch"
"$PY" "$HERE/scripts/tts/switch_bench.py" --url "$URL" --asr "$AM" --clip "$CLIP" \
    --tts "$VM:kavya" --tts "$CM:default" --rounds 12 | tee "$RES/switch.jsonl" || exit 1

echo "== mixed"
asr --conc 4 --tag mixed > "$RES/mixed_asr.log" 2>&1 & P1=$!
vtts --conc 8 --stream --n 32 --tag mixed_stream_c8 > "$RES/mixed_veena.log" 2>&1 & P2=$!
ctts --conc 4 --n 16 --tag mixed_full_c4 > "$RES/mixed_cbx.log" 2>&1 & P3=$!
rc=0; for p in $P1 $P2 $P3; do wait $p || rc=1; done
grep -h '^{' "$RES"/mixed_*.log
[ $rc = 0 ] || { echo "mixed phase failed"; tail -5 "$RES"/mixed_*.log; exit 1; }
grep -ciE "panic|fault|illegal" "$PB_SERVER_LOG" | sed 's/^/server faults: /'
echo COSERVE_DONE
