#!/usr/bin/env bash
# Campaign R2 SPEED runner -- serves ONE engine, gates it at LONG context, runs r2speed.py.
#
#   $1 engine (plow|vllm)   $2 port   $3 label   [$4 assets, plow only]
#
# WHY THE GATE IS LONG HERE. vLLM routes MLA prefill with `prefill_max_seq_len <= topk_tokens`
# (2048) down the dense MHA path and anything longer down the sparse path. A SHORT gate therefore
# certifies an attention path the long-context numbers never execute. It is also the path that
# on ROCm is `raise NotImplementedError`, so short prompts kill the engine outright -- and the
# documented workaround (sparse_mla_force_mqa) runs the sparse path out of spec and MEASURED
# GSM8K 0.175 vs plow's 0.970. Above 2048 BOTH engines run their intended path and both are
# correct, so every prompt in this bench is > 2048 and the gate is too.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="${PLOW_REPO:-$(cd "$HERE/../.." && pwd)}"
ENGINE="${1:?plow|vllm}"; PORT="${2:?port}"; LABEL="${3:?label}"; ASSETS="${4:-}"
OUT="${OUT:-${TMPDIR:-/tmp}/twoengine}"; mkdir -p "$OUT"
LOG="$OUT/speed_serve_$LABEL.log"
LOCK=/tmp/plow_gpu.lock
HAVE_LOCK=0
MAXLEN="${MAXLEN:-73728}"
GATE_TOK="${GATE_TOK:-3000}"

release() {
  [ -n "${SPGID:-}" ] && kill -TERM "-$SPGID" 2>/dev/null
  sleep 5
  [ -n "${SPGID:-}" ] && kill -KILL "-$SPGID" 2>/dev/null
  [ "$HAVE_LOCK" = 1 ] && rm -rf "$LOCK"
  return 0
}
trap 'release; exit 143' INT TERM
trap 'release' EXIT

for i in $(seq 1 720); do
  mkdir "$LOCK" 2>/dev/null && { HAVE_LOCK=1; break; }
  sleep 5
done
[ "$HAVE_LOCK" = 1 ] || { echo "FAIL: no GPU lock"; exit 1; }
echo "$$ $LABEL" > "$LOCK/owner" 2>/dev/null
pgrep '^plowrt' >/dev/null 2>&1 && { echo "FAIL: plowrt already running"; pgrep -a '^plowrt'; exit 1; }

cd "$WT"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

if [ "$ENGINE" = plow ]; then
  [ -n "$ASSETS" ] || { echo "FAIL: plow needs assets"; exit 1; }
  NIX="${PLOW_NIX:-/nix/var/nix/profiles/default/bin/nix}"
  ROCM_LIB="${PLOW_ROCM_LIB:-/opt/rocm-7.2.4/lib}"
  SERVE_ENV="${SERVE_ENV:-PLOW_MLA_PF_V2=1}"
  echo "=== [$LABEL] plow serving $ASSETS on :$PORT  ($SERVE_ENV) ==="
  setsid "$NIX" develop "$WT" -c bash -c \
    "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:$ROCM_LIB\"; export $SERVE_ENV; \
     exec ./target/release/plowrt serve --assets '$ASSETS' --port '$PORT'" > "$LOG" 2>&1 &
else
  echo "=== [$LABEL] vLLM 0.26 serving GLM-5.2-FP8 tp8 on :$PORT (NO force_mqa) ==="
  setsid env VLLM_ROCM_USE_AITER=1 HF_HUB_OFFLINE=1 \
    "${VLLM_VENV:-/workspace/vllm26}/bin/vllm" serve /workspace/models/GLM-5.2-FP8 \
      --served-model-name glm-5.2-fp8 --tensor-parallel-size 8 \
      --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 \
      --no-enable-prefix-caching --trust-remote-code --port "$PORT" > "$LOG" 2>&1 &
fi
SPID=$!
SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"

for i in $(seq 1 3600); do
  kill -0 "$SPID" 2>/dev/null || { echo "FAIL: server died during load"; tail -30 "$LOG"; exit 1; }
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "FAIL: never ready"; tail -30 "$LOG"; exit 1; }
echo "  ready"

if [ "$ENGINE" = plow ]; then
  grep -q "CPU reference backend active" "$LOG" && {
    echo "FAIL: CPU reference backend -- every number would be fiction"; exit 1; }
  grep -qE "HSA backend selected|hsa=true" "$LOG" && echo "  HSA backend: OK" \
    || echo "  >>> WARN: no HSA banner"
fi

PORT="$PORT" MODEL=auto NTOK="$GATE_TOK" python3 $HERE/needle_gate.py || {
  echo ">>> LONG GATE FAILED -- refusing to report speed for a degraded server."; exit 1; }

PORT="$PORT" MODEL=auto LABEL="$LABEL" OUT="$OUT" \
  CONCS="${CONCS:-1 4 8 16 32}" INLEN="${INLEN:-4096}" OUTLEN="${OUTLEN:-128}" \
  NMULT="${NMULT:-4}" CTXS="${CTXS:-4096 8192 16384 32768}" LC_OUTLEN="${LC_OUTLEN:-32}" \
  python3 $HERE/speed.py
echo "=== [$LABEL] speed done ==="
