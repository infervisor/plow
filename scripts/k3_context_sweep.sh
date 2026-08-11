#!/usr/bin/env bash
# Kimi-K3 B8 context sweep against an already-running OpenAI-chat endpoint.
#
# Usage:
#   nix develop .#vllm --command scripts/k3_context_sweep.sh
#
# The server is deliberately outside this script. Run the same pinned client,
# seeds, and output directory layout against each endpoint being compared.
set -euo pipefail

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

require_nix_tool() {
  local tool="$1" path
  path="$(command -v "$tool" 2>/dev/null || true)"
  case "$path" in
    /nix/store/*) ;;
    *) fail "$tool must come from nix develop .#vllm, got ${path:-missing}" ;;
  esac
}

[ -n "${IN_NIX_SHELL:-}" ] || fail "run through nix develop .#vllm"
require_nix_tool vllm
require_nix_tool python3
require_nix_tool jq
require_nix_tool curl

VLLM_VERSION="$(python3 -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')"
[ "$VLLM_VERSION" = 0.27.0 ] || fail "expected flake-pinned vLLM 0.27.0, got $VLLM_VERSION"

BASE_URL="${PLOW_K3_BASE_URL:-http://127.0.0.1:8018}"
MODEL="${PLOW_K3_MODEL:-k3_farm}"
TOKENIZER="${PLOW_K3_TOKENIZER:-/home/lava/models/k3_tokz}"
CONTEXTS="${PLOW_K3_CONTEXTS:-128 512 1024 2048 4096 8192 16000 32000}"
SEEDS="${PLOW_K3_SEEDS:-0 1 2}"
OUTLEN="${PLOW_K3_OUTLEN:-128}"
NWARM="${PLOW_K3_NWARM:-1}"
OUTDIR="${PLOW_K3_OUTDIR:-/tmp/k3_context_sweep}"
TAG="${PLOW_K3_TAG:-plow}"
MAX_CTX=32768
MAX_CHAT_OVERHEAD="${PLOW_K3_MAX_CHAT_OVERHEAD:-64}"

[[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || fail "PLOW_K3_TAG must be filename-safe"
[[ "$OUTLEN" =~ ^[0-9]+$ ]] && [ "$OUTLEN" -gt 0 ] || fail "PLOW_K3_OUTLEN must be positive"
[[ "$NWARM" =~ ^[0-9]+$ ]] || fail "PLOW_K3_NWARM must be non-negative"
[[ "$MAX_CHAT_OVERHEAD" =~ ^[0-9]+$ ]] || fail "PLOW_K3_MAX_CHAT_OVERHEAD must be non-negative"
[ -d "$TOKENIZER" ] || fail "tokenizer directory does not exist: $TOKENIZER"

# Accept commas for convenience, but keep the actual sweep a plain finite list.
CONTEXTS="${CONTEXTS//,/ }"
SEEDS="${SEEDS//,/ }"
[ -n "${CONTEXTS// /}" ] || fail "PLOW_K3_CONTEXTS is empty"
[ -n "${SEEDS// /}" ] || fail "PLOW_K3_SEEDS is empty"
for ctx in $CONTEXTS; do
  [[ "$ctx" =~ ^[0-9]+$ ]] || fail "invalid context: $ctx"
  [ "$ctx" -gt 0 ] || fail "context must be positive"
  [ "$ctx" -ne 32768 ] || fail "32768 is forbidden: chat framing reaches/passes max_ctx"
  [ "$ctx" -le 32000 ] || fail "context $ctx exceeds the validated 32000-token ceiling"
done
for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || fail "invalid seed: $seed"
done

models="$(curl --fail --silent --show-error --max-time 10 "$BASE_URL/v1/models")" ||
  fail "endpoint is not ready: $BASE_URL"
jq -e --arg model "$MODEL" 'any(.data[]?; .id == $model)' <<<"$models" >/dev/null ||
  fail "model $MODEL is not advertised by $BASE_URL/v1/models"

echo "vllm       $VLLM_VERSION ($(command -v vllm))"
echo "endpoint   $BASE_URL"
echo "model      $MODEL"
echo "tokenizer  $TOKENIZER"
echo "contexts   $CONTEXTS"
echo "seeds      $SEEDS"
echo "output     $OUTLEN tokens, concurrency 8, warmups $NWARM"
echo "results    $OUTDIR"

for ctx in $CONTEXTS; do
  if [ "$ctx" -le 4096 ]; then
    n=32
  elif [ "$ctx" -le 8192 ]; then
    n=16
  else
    n=8
  fi

  for seed in $SEEDS; do
    result="$OUTDIR/${TAG}_ctx${ctx}_c8_n${n}_seed${seed}.json"
    echo
    echo "== ctx=$ctx concurrency=8 prompts=$n seed=$seed =="
    vllm bench serve \
      --backend openai-chat \
      --base-url "$BASE_URL" \
      --endpoint /v1/chat/completions \
      --model "$MODEL" \
      --served-model-name "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --tokenizer-mode hf \
      --dataset-name random \
      --random-input-len "$ctx" \
      --random-output-len "$OUTLEN" \
      --random-range-ratio 0 \
      --request-rate inf \
      --max-concurrency 8 \
      --num-prompts "$n" \
      --num-warmups "$NWARM" \
      --ignore-eos \
      --temperature 0 \
      --seed "$seed" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,99 \
      --save-result \
      --save-detailed \
      --result-dir "$OUTDIR" \
      --result-filename "${result##*/}" \
      --label "$TAG" \
      --metadata "requested_context=$ctx" "expected_output=$OUTLEN" "seed=$seed"

    [ -s "$result" ] || fail "vllm did not write $result"
    jq -e \
      --argjson n "$n" \
      --argjson out "$OUTLEN" \
      --argjson ctx "$ctx" \
      --argjson max_ctx "$MAX_CTX" \
      --argjson max_overhead "$MAX_CHAT_OVERHEAD" '
        .completed == $n and
        .failed == 0 and
        .total_output_tokens == ($n * $out) and
        (.output_lens | length) == $n and
        all(.output_lens[]; . == $out) and
        (.errors | length) == $n and
        all(.errors[]; . == "") and
        (.generated_texts | length) == $n and
        all(.generated_texts[]; (ascii_downcase | contains("[error:")) | not) and
        (.input_lens | length) == $n and
        .total_input_tokens == ([.input_lens[]] | add) and
        all(.input_lens[];
          . >= $ctx and
          . <= ($ctx + $max_overhead) and
          (. + $out) < $max_ctx)
      ' "$result" >/dev/null || {
        jq '{completed,failed,total_input_tokens,total_output_tokens,input_lens,output_lens,errors}' "$result" >&2
        fail "hard gate failed for ctx=$ctx seed=$seed"
      }

    jq -r '
      "PASS ctx=\(.requested_context) completed=\(.completed) " +
      "input=[\(.input_lens|min),\(.input_lens|max)] output=\(.total_output_tokens) " +
      "tok/s=\(.output_throughput)"
    ' "$result"
  done
done

echo
echo ">>> K3 context sweep PASS: every detailed result satisfied the hard gates"
