#!/usr/bin/env bash
# serve_voice_agent.sh — the voice-agent co-serve: Qwen3-ASR + Gemma 4 E4B + Chatterbox-MTL in ONE
# `plowrt serve` on one GPU, and the call_sim load against it.
#
#   scripts/voice/serve_voice_agent.sh serve <resdir>           # foreground server until killed / VA_RUN_S
#   scripts/voice/serve_voice_agent.sh calls <resdir> [N ...]   # serve, call_sim per call count, SLO table
#
# Both take the GPU through gpulease (-n 1) with `timeout` on the wait and on the run; set
# VA_LEASE=0 only when already inside a lease. Assets: ASR_ASSETS / LLM_ASSETS / TTS_ASSETS, else
# $VA_BUILD_ROOT/{qwen3-asr,gemma-4-e4b,chatterbox-mtl}/assets, i.e. the `--out` dirs of
#   campaign.py build recipes/infervisor/<model>/sm90a-h100-tp1.toml --out $VA_BUILD_ROOT/<model>
#
# Env (defaults): PLOWRT_BIN (<repo>/target/release/plowrt, copied into <resdir> before serving),
# PY (python3; needs aiohttp numpy soundfile), MANIFEST (ASR clips, [{path,text,dur}] 16 kHz),
# VA_CO_SCHED (deadline), VA_LIVE_CTX (qwen3-asr=768,chatterbox-mtl=512), VA_SESSION_TTL_MS
# (60000), SERVE_ARGS (extra plowrt serve flags), PORT (free port), TURNS (3), CALL_ARGS
# (--language en), VA_RUN_S (run timeout, 3600), GPU_LEASE_TIMEOUT (lease wait, 1800).
# docs/runtime/gemma4-e4b-h100.md "Voice co-serving" has the memory and scheduling rationale.
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
MODE=${1:-}; RES=${2:-}
case "$MODE" in
    serve|calls) [ -n "$RES" ] || { echo "usage: $0 serve|calls <resdir> [calls...]" >&2; exit 2; } ;;
    -h|--help|help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "usage: $0 serve|calls <resdir> [calls...]   (--help)" >&2; exit 2 ;;
esac
RUN_S=${VA_RUN_S:-3600}
if [ "${VA_LEASE:-1}" = 1 ]; then
    GL=${GPULEASE:-$HERE/perf-data/tools/gpulease}
    # A linked worktree has no perf-data/; the main checkout does.
    [ -x "$GL" ] || GL="$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)/../perf-data/tools/gpulease"
    [ -x "$GL" ] || GL=/app/plow/perf-data/tools/gpulease
    [ -x "$GL" ] || { echo "no gpulease (set GPULEASE)" >&2; exit 2; }
    exec timeout --kill-after=30s $(( ${GPU_LEASE_TIMEOUT:-1800} + RUN_S + 60 )) \
        "$GL" -n 1 "voice-$MODE" env VA_LEASE=0 timeout --kill-after=30s "$RUN_S" "$0" "$@"
fi
shift 2
source "$HERE/scripts/bench/plowbench.sh"
ROOT=${VA_BUILD_ROOT:-}
A=${ASR_ASSETS:-${ROOT:+$ROOT/qwen3-asr/assets}}
L=${LLM_ASSETS:-${ROOT:+$ROOT/gemma-4-e4b/assets}}
S=${TTS_ASSETS:-${ROOT:+$ROOT/chatterbox-mtl/assets}}
for d in "$A" "$L" "$S"; do
    [ -n "$d" ] && [ -e "$d/model.pkt" ] || { echo "missing assets '$d' (set VA_BUILD_ROOT or ASR/LLM/TTS_ASSETS)" >&2; exit 2; }
done
PY=${PY:-python3}
[ "$MODE" = serve ] || [ -e "${MANIFEST:-}" ] || { echo "MANIFEST: ASR clip manifest (json list of {path,text,dur})" >&2; exit 2; }
mkdir -p "$RES"
RT="$RES/plowrt"
cp "${PLOWRT_BIN:-$HERE/target/release/plowrt}" "$RT" || exit 2
PB_SERVER_PORT=${PORT:-$(pb_free_port)}; PB_SERVER_LOG="$RES/serve.log"
# LLM last: its KV admission budget is sampled from what the speech models leave.
# shellcheck disable=SC2086
PLOW_LIVE_CTX_MODELS=${VA_LIVE_CTX:-qwen3-asr=768,chatterbox-mtl=512} \
    "$RT" serve --assets "$A" --assets "$S" --assets "$L" --port "$PB_SERVER_PORT" \
    --co-sched "${VA_CO_SCHED:-deadline}" --session-ttl-ms "${VA_SESSION_TTL_MS:-60000}" ${SERVE_ARGS:-} \
    > "$PB_SERVER_LOG" 2>&1 &
PB_SERVER_PID=$!
trap pb_serve_stop EXIT
pb_serve_wait 900 || exit 3
URL=http://127.0.0.1:$PB_SERVER_PORT
curl -fsS "$URL/v1/models" > "$RES/models.json"
echo "serving $URL: $("$PY" -c 'import json,sys; print(" ".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' "$RES/models.json")"
if [ "$MODE" = serve ]; then
    wait "$PB_SERVER_PID"
    exit $?
fi
for n in "${@:-10}"; do
    curl -fsS "$URL/metrics" > "$RES/metrics-$n.before" 2>/dev/null
    # shellcheck disable=SC2086
    "$PY" "$HERE/scripts/voice/call_sim.py" --url "$URL" --calls "$n" --turns "${TURNS:-3}" \
        --asr-model qwen3-asr --llm-model gemma-4-e4b --tts-model chatterbox-mtl \
        --manifest "$MANIFEST" ${CALL_ARGS:---language en} --out "$RES/calls$n.json" 2>&1 | tail -2
    curl -fsS "$URL/metrics" > "$RES/metrics-$n.after" 2>/dev/null
done
echo "server faults: $(grep -ciE 'panic|fault|illegal' "$PB_SERVER_LOG")"
"$PY" "$HERE/scripts/voice/slo_table.py" "$RES"/calls*.json | tee "$RES/slo.md"
