#!/usr/bin/env bash
# The vLLM arm of the R2 baseline. Same client (r2client.py), same box, same session.
#
# `perf-data/plow-gfx942/README.md` calls re-baselining BOTH engines in one session "the single
# most valuable missing measurement" -- every ratio in that directory rests on a vLLM number
# taken hours apart with an install that has since been deleted, and the stored CSV already
# failed to reproduce once (7.57 -> 10.03, a 33% move). This closes that.
#
# PREFIX CACHING IS OFF, DELIBERATELY, and the comparison is invalid without saying so:
#   * the TTFT ladder sends the SAME prompt 3x -- with caching, reps 2 and 3 are ~free;
#   * GSM8K shares one 8-shot preamble across every question -- with caching vLLM pays for it
#     once and plow pays 100 times.
# plow has no prefix cache at all, so leaving it on would measure a FEATURE gap and report it as
# a kernel gap. Off is like-for-like. vLLM would gain further with it on; that is a real and
# separate advantage, and it belongs in the writeup as one.
#
# $1 port
set -uo pipefail
# `$HERE` is used at the bottom to locate client.py and was never defined -- the vLLM arm
# loaded the model, passed the coherence gate, and THEN died on `HERE: unbound variable`
# without running a single measurement. Under `set -u` that is a hard exit at the last line,
# i.e. the most expensive possible place to discover it (measured 2026-08-09: ~6 min of model
# load thrown away).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8241}"
VENV="${VLLM_VENV:-/workspace/vllm26}/bin"
MODEL="${MODEL_PATH:-/workspace/models/GLM-5.2-FP8}"
TP="${TP:-8}"
MAXLEN="${MAXLEN:-32768}"
OUT="${OUT:-${TMPDIR:-/tmp}/twoengine}"; mkdir -p "$OUT"
LOG="$OUT/serve_vllm.log"
LOCK=/tmp/plow_gpu.lock
HAVE_LOCK=0

release() {
  [ -n "${SPGID:-}" ] && kill -TERM "-$SPGID" 2>/dev/null
  sleep 5
  [ -n "${SPGID:-}" ] && kill -KILL "-$SPGID" 2>/dev/null
  [ "$HAVE_LOCK" = 1 ] && rm -rf "$LOCK"
  return 0
}
trap 'release; exit 143' INT TERM
trap 'release' EXIT

for i in $(seq 1 600); do
  mkdir "$LOCK" 2>/dev/null && { HAVE_LOCK=1; break; }
  sleep 5
done
[ "$HAVE_LOCK" = 1 ] || { echo "FAIL: no GPU lock"; exit 1; }
echo "$$ vllm" > "$LOCK/owner" 2>/dev/null

if pgrep '^plowrt' >/dev/null 2>&1; then
  echo "FAIL: plowrt still running"; pgrep -a '^plowrt'; exit 1
fi

echo "=== vLLM 0.26 serving $MODEL tp=$TP on :$PORT ==="
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
setsid env VLLM_ROCM_USE_AITER=1 HF_HUB_OFFLINE=1 \
  "$VENV/vllm" serve "$MODEL" \
    --served-model-name glm-5.2-fp8 \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --attention-config '{"sparse_mla_force_mqa":true}' \
    --port "$PORT" > "$LOG" 2>&1 &
# sparse_mla_force_mqa IS REQUIRED HERE, and it is a vLLM limitation rather than a tuning choice.
# GLM-5.2 is a DSA model, so vLLM picks ROCM_AITER_MLA_SPARSE (the log shows it as the ONLY
# candidate). mla_attention.py:756 then routes prefills with
# `prefill_max_seq_len <= topk_tokens` (2048) down the DENSE MHA path -- and that backend's
# `forward_mha` is `raise NotImplementedError` (v1/attention/backend.py:1040). So ANY prompt
# shorter than 2048 tokens kills the engine: the coherence gate ("capital of France") took the
# whole server down with EngineDeadError on the first request.
# Forcing MQA sends every token through the sparse path, which IS implemented. Flagged in the
# writeup because it means the <=2048 rungs of the vLLM ladder run a DIFFERENT attention path
# than they would by default -- one vLLM would have used only above 2048.
SPID=$!
SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"

for i in $(seq 1 3600); do
  kill -0 "$SPID" 2>/dev/null || { echo "FAIL: vllm died during load"; tail -40 "$LOG"; exit 1; }
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "FAIL: vllm never ready"; tail -40 "$LOG"; exit 1; }
echo "  ready"

# Same coherence gate as the plow arm. A fast wrong server is not a result on either side.
GATE=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"glm-5.2-fp8\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":200,\"temperature\":0}")
echo "$GATE" | grep -qi paris || { echo ">>> vLLM COHERENCE GATE FAIL"; echo "$GATE" | head -c 500; exit 1; }
echo "  coherence gate: PASS"
grep -iE "aiter" "$LOG" | head -3

PORT="$PORT" MODEL=auto LABEL="${LABEL:-vllm}" OUT="$OUT" \
  N="${N:-100}" SHOTS="${SHOTS:-8}" MAXTOK="${MAXTOK:-320}" CONC="${CONC:-1}" \
  CTXS="${CTXS:-1024 4096 8192 16384}" GSM="${GSM:-${GSM8K_DIR:-$HOME/.cache/gsm8k}}" \
  python3 $HERE/client.py
echo "=== vllm arm done ==="
