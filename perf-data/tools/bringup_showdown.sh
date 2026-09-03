#!/usr/bin/env bash
# Alternating, sequential-exclusive Plow/vLLM serving showdown. Run the WHOLE command under one
# lease so server startup, warmups, and measured clients cannot overlap another campaign:
#   perf-data/tools/gpulease -n "$TP" showdown perf-data/tools/bringup_showdown.sh
# The lease is advisory; gpulease also audits foreign processes before/after. A run outside that
# contract is not publishable timing evidence.
set -euo pipefail

: "${PLOWRT:?set PLOWRT to the explicit private plowrt executable}"
: "${PLOW_ASSETS:=${BUNDLES:-}}"
: "${PLOW_ASSETS:?set PLOW_ASSETS to the served Plow bundle}"
: "${PLOW_ARTIFACTS:?set PLOW_ARTIFACTS to whitespace-separated Plow files/directories to freeze}"
: "${VLLM_MODEL:=${MODEL_ID:-}}"
: "${VLLM_MODEL:?set VLLM_MODEL to the vLLM model path/id}"
: "${MODEL_ID:=$VLLM_MODEL}"
: "${SNAP:?set SNAP to the tokenizer snapshot used by the shared client}"

if [ -z "${VLLM_ARTIFACTS:-}" ]; then
  : "${VLLM_IMAGE_DIGEST:?set VLLM_ARTIFACTS, or an immutable VLLM_IMAGE_DIGEST}"
  : "${VLLM_MODEL_IDENTITY:?set VLLM_ARTIFACTS, or an immutable VLLM_MODEL_IDENTITY}"
fi

case "$PLOWRT" in /*) ;; *) echo "PLOWRT must be an absolute path" >&2; exit 2;; esac
[ -f "$PLOWRT" ] && [ -x "$PLOWRT" ] || { echo "PLOWRT is not an executable file: $PLOWRT" >&2; exit 2; }
[ -d "$PLOW_ASSETS" ] || { echo "missing PLOW_ASSETS: $PLOW_ASSETS" >&2; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
PLOW_REQUIRE_TUNED=${PLOW_REQUIRE_TUNED:-0}
case "$PLOW_REQUIRE_TUNED" in 0|1) ;; *) echo "PLOW_REQUIRE_TUNED must be 0 or 1" >&2; exit 2;; esac
ROUNDS=${ROUNDS:-5}
TP=${TP:-1}
DTYPE=${DTYPE:-bfloat16}
PLOW_TAG=${PLOW_TAG:-plow}
VLLM_TAG=${VLLM_TAG:-vllm}
PORT_PLOW=${PORT_PLOW:-8093}
PORT_VLLM=${PORT_VLLM:-8085}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
ROUND_PREFIX=${ROUND_PREFIX:-showdown}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
BRINGUP_OUT=${BRINGUP_OUT:-/tmp/bringup-$USER/$RUN_ID}
export BRINGUP_OUT

positive() { case "$2" in *[!0-9]*|0|'') echo "$1 must be a positive integer" >&2; exit 2;; esac; }
positive ROUNDS "$ROUNDS"; positive TP "$TP"
[ -z "$MAX_MODEL_LEN" ] || positive MAX_MODEL_LEN "$MAX_MODEL_LEN"
[ "$ROUNDS" -ge 3 ] || { echo "ROUNDS must be >= 3 alternating whole-server rounds" >&2; exit 2; }
mkdir -p "$BRINGUP_OUT"
[ ! -e "$BRINGUP_OUT/cells.tsv" ] || { echo "result directory already contains a run: $BRINGUP_OUT" >&2; exit 2; }

read -r -a PLOW_EXTRA <<<"${PLOW_ENGINE_ARGV:-}"
read -r -a VLLM_EXTRA <<<"${VLLM_ENGINE_ARGV:-}"
read -r -a PLOW_COMMAND <<<"${PLOW_SERVER_COMMAND_ARGV:-$PLOWRT}"
read -r -a VLLM_COMMAND <<<"${VLLM_SERVER_COMMAND_ARGV:-vllm}"
[ "${#PLOW_COMMAND[@]}" -gt 0 ] || { echo "PLOW_SERVER_COMMAND_ARGV is empty" >&2; exit 2; }
[ "${#VLLM_COMMAND[@]}" -gt 0 ] || { echo "VLLM_SERVER_COMMAND_ARGV is empty" >&2; exit 2; }
VLLM_CONTEXT_ARGV=()
[ -z "$MAX_MODEL_LEN" ] || VLLM_CONTEXT_ARGV=(--max-model-len "$MAX_MODEL_LEN")

if [ "${PLOW_REQUIRE_VERIFIED:-1}" = 1 ]; then
  python3 - "$PLOW_ASSETS/build.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file(): raise SystemExit(f"missing build manifest: {p}")
d = json.loads(p.read_text())
lean = d.get("lean", {})
if d.get("schema") != 1 or lean.get("verified") is not True or lean.get("oracle") is not True:
    raise SystemExit(f"unverified build manifest: {p}")
if not d.get("pairing", {}).get("hash"):
    raise SystemExit(f"build manifest has no packet/object pairing hash: {p}")
PY
fi

TUNING_RECORD=$("$HERE/bringup_tuning_profile.py" "$PLOW_ASSETS/build.json" "$PLOW_REQUIRE_TUNED")
IFS=$'\t' read -r TILE_MEASURED TILE_SOURCE TUNING_PROFILE <<<"$TUNING_RECORD"
if [ "$TUNING_PROFILE" = measured ]; then
  echo "tuning profile: measured ($TILE_MEASURED selections, source=$TILE_SOURCE)" >&2
else
  echo "WARNING: analytical tuning fallback (tile_measured=$TILE_MEASURED, source=$TILE_SOURCE); baseline evidence only" >&2
fi

artifact_manifest() {
  python3 - "$@" <<'PY'
import hashlib, pathlib, sys
files=[]
for raw in sys.argv[1:]:
    p=pathlib.Path(raw)
    if not p.exists(): raise SystemExit(f"declared artifact does not exist: {p}")
    if p.is_dir(): files.extend(x for x in p.rglob("*") if x.is_file())
    elif p.is_file(): files.append(p)
for p in sorted(set(files), key=lambda x: str(x.resolve())):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20), b""): h.update(chunk)
    print(f"{h.hexdigest()}  {p.resolve()}")
PY
}
artifact_manifest "$PLOWRT" $PLOW_ARTIFACTS >"$BRINGUP_OUT/plow-artifacts.sha256"
PLOW_ARTIFACT_DIGEST=$(sha256sum "$BRINGUP_OUT/plow-artifacts.sha256" | awk '{print $1}')
if [ -n "${VLLM_ARTIFACTS:-}" ]; then
  artifact_manifest $VLLM_ARTIFACTS >"$BRINGUP_OUT/vllm-artifacts.sha256"
  VLLM_ARTIFACT_DIGEST=$(sha256sum "$BRINGUP_OUT/vllm-artifacts.sha256" | awk '{print $1}')
else
  VLLM_ARTIFACT_DIGEST=$(python3 - "$VLLM_IMAGE_DIGEST" "$VLLM_MODEL_IDENTITY" <<'PY'
import hashlib, sys
print(hashlib.sha256("\0".join(sys.argv[1:]).encode()).hexdigest())
PY
)
fi
printf '%s\n' "$PLOW_ARTIFACT_DIGEST" >"$BRINGUP_OUT/plow-artifact-set.sha256"
printf '%s\n' "$VLLM_ARTIFACT_DIGEST" >"$BRINGUP_OUT/vllm-artifact-set.sha256"
printf 'tag\tround\tinlen\tttft_mean\tttft_median\tttft_p99\ttpot_median\tconcurrency\tprompts\toutlen\tinput_tokens\toutput_tokens\treq_s\tout_tok_s\tartifact_digest\ttpot_mean\ttpot_p99\titl_mean\titl_median\titl_p99\te2el_mean\te2el_median\te2el_p99\n' >"$BRINGUP_OUT/cells.tsv"
{
  printf 'rounds=%s\ntp=%s\ndtype=%s\nmodel_id=%s\nvllm_model=%s\n' "$ROUNDS" "$TP" "$DTYPE" "$MODEL_ID" "$VLLM_MODEL"
  printf 'max_model_len=%s\n' "${MAX_MODEL_LEN:-<model-default>}"
  printf 'plowrt=%s\nplow_assets=%s\nplow_server_command_argv=%s\nplow_engine_argv=%s\n' "$PLOWRT" "$PLOW_ASSETS" "${PLOW_SERVER_COMMAND_ARGV:-$PLOWRT}" "${PLOW_ENGINE_ARGV:-}"
  printf 'vllm_server_command_argv=%s\nvllm_client_command_argv=%s\nvllm_engine_argv=%s\n' "${VLLM_SERVER_COMMAND_ARGV:-vllm}" "${VLLM_CLIENT_COMMAND_ARGV:-vllm}" "${VLLM_ENGINE_ARGV:-}"
  printf 'client_argv=%s\nseed=%s\n' "${BRINGUP_CLIENT_ARGV:-}" "${BRINGUP_SEED:-42}"
  printf 'require_tuned=%s\ntuning_profile=%s\ntile_measured=%s\ntile_source=%s\n' \
    "$PLOW_REQUIRE_TUNED" "$TUNING_PROFILE" "$TILE_MEASURED" "$TILE_SOURCE"
  printf 'plow_artifact_digest=%s\nvllm_artifact_digest=%s\nvllm_image_digest=%s\nvllm_model_identity=%s\n' "$PLOW_ARTIFACT_DIGEST" "$VLLM_ARTIFACT_DIGEST" "${VLLM_IMAGE_DIGEST:-}" "${VLLM_MODEL_IDENTITY:-}"
  printf 'input_lens=%s\nconcurrency_map=%s\nprompt_map=%s\nwarmup_map=%s\noutlen_map=%s\n' \
    "${INPUT_MAP:-${INPUT_LENS:-${IN_LENS:-128 1024 4096}}}" "${CONCURRENCY_MAP:-default=1}" \
    "${PROMPT_MAP:-default=${NPROMPT:-6}}" "${WARMUP_MAP:-default=1}" \
    "${OUTLEN_MAP:-default=${OUTLEN:-8}}"
} >"$BRINGUP_OUT/config.txt"

check_artifacts() {
  local now="$BRINGUP_OUT/plow-artifacts.now"
  artifact_manifest "$PLOWRT" $PLOW_ARTIFACTS >"$now"
  cmp -s "$BRINGUP_OUT/plow-artifacts.sha256" "$now" || {
    diff -u "$BRINGUP_OUT/plow-artifacts.sha256" "$now" >&2 || true
    echo "Plow artifact set changed during showdown" >&2
    return 1
  }
  if [ -n "${VLLM_ARTIFACTS:-}" ]; then
    now="$BRINGUP_OUT/vllm-artifacts.now"
    artifact_manifest $VLLM_ARTIFACTS >"$now"
    cmp -s "$BRINGUP_OUT/vllm-artifacts.sha256" "$now" || {
      diff -u "$BRINGUP_OUT/vllm-artifacts.sha256" "$now" >&2 || true
      echo "vLLM artifact set changed during showdown" >&2
      return 1
    }
  fi
}

stop_server() {
  local pid=$1
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return; }; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

ACTIVE_PID=
cleanup() {
  [ -z "$ACTIVE_PID" ] || stop_server "$ACTIVE_PID"
}
trap cleanup EXIT INT TERM

wait_ready() { # <pid> <port> <log> <attempts> <sleep> <model>
  local pid=$1 port=$2 log=$3 attempts=$4 delay=$5 model=$6 body advertised="<endpoint unavailable>"
  for _ in $(seq 1 "$attempts"); do
    if body=$(curl -sf --max-time 2 "http://127.0.0.1:$port/v1/models" 2>/dev/null); then
      if advertised=$(python3 -c '
import json, sys
try:
    ids = [item["id"] for item in json.load(sys.stdin)["data"]]
    if not all(isinstance(model_id, str) for model_id in ids):
        raise TypeError
except (KeyError, TypeError, ValueError):
    print("<invalid /v1/models response>")
    raise SystemExit(2)
print(", ".join(ids) if ids else "<none>")
raise SystemExit(0 if sys.argv[1] in ids else 1)
' "$model" <<<"$body"); then
        return 0
      fi
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "endpoint on port $port exited before advertising required model '$model' (last advertised: $advertised)" >&2
      tail -40 "$log" >&2
      return 1
    fi
    sleep "$delay"
  done
  echo "endpoint on port $port did not advertise required model '$model' (last advertised: $advertised)" >&2
  tail -40 "$log" >&2
  return 1
}

run_plow() { # <round>
  local round=$1
  local log="$BRINGUP_OUT/serve-$PLOW_TAG-r$round.log" pid
  check_artifacts
  "${PLOW_COMMAND[@]}" serve --assets "$PLOW_ASSETS" --port "$PORT_PLOW" \
    --executors "$TP" "${PLOW_EXTRA[@]}" >"$log" 2>&1 & pid=$!
  ACTIVE_PID=$pid
  if ! wait_ready "$pid" "$PORT_PLOW" "$log" "${BRINGUP_READY_ATTEMPTS:-600}" "${BRINGUP_READY_SLEEP:-1}" "$MODEL_ID"; then
    stop_server "$pid"; ACTIVE_PID=; return 1
  fi
  if ! grep -q "backend ready.*GPU accelerated" "$log"; then
    echo "Plow did not report a GPU-accelerated backend" >&2; stop_server "$pid"; ACTIVE_PID=; return 1
  fi
  local rc=0
  export BRINGUP_ARTIFACT_DIGEST=$PLOW_ARTIFACT_DIGEST
  "$HERE/bringup_bench.sh" "$PLOW_TAG" "http://127.0.0.1:$PORT_PLOW" "$MODEL_ID" "$SNAP" "$ROUND_PREFIX-$round" || rc=$?
  stop_server "$pid"; ACTIVE_PID=
  check_artifacts || rc=$?
  return "$rc"
}

run_vllm() { # <round>
  local round=$1
  local log="$BRINGUP_OUT/serve-$VLLM_TAG-r$round.log" pid
  check_artifacts
  "${VLLM_COMMAND[@]}" serve "$VLLM_MODEL" --served-model-name "$MODEL_ID" --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" "${VLLM_CONTEXT_ARGV[@]}" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --scheduling-policy fcfs \
    --port "$PORT_VLLM" "${VLLM_EXTRA[@]}" >"$log" 2>&1 & pid=$!
  ACTIVE_PID=$pid
  if ! wait_ready "$pid" "$PORT_VLLM" "$log" "${VLLM_READY_ATTEMPTS:-900}" "${VLLM_READY_SLEEP:-2}" "$MODEL_ID"; then
    stop_server "$pid"; ACTIVE_PID=; return 1
  fi
  local rc=0
  export BRINGUP_ARTIFACT_DIGEST=$VLLM_ARTIFACT_DIGEST
  "$HERE/bringup_bench.sh" "$VLLM_TAG" "http://127.0.0.1:$PORT_VLLM" "$MODEL_ID" "$SNAP" "$ROUND_PREFIX-$round" || rc=$?
  stop_server "$pid"; ACTIVE_PID=
  check_artifacts || rc=$?
  return "$rc"
}

for round in $(seq 1 "$ROUNDS"); do
  if [ $((round % 2)) -eq 1 ]; then
    run_plow "$round"; run_vllm "$round"
  else
    run_vllm "$round"; run_plow "$round"
  fi
done
check_artifacts
touch "$BRINGUP_OUT/complete"
printf 'showdown complete: %s\n' "$BRINGUP_OUT"
