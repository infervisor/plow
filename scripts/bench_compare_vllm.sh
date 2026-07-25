#!/usr/bin/env bash
# scripts/bench_compare_vllm.sh — Side-by-side plowrt vs vLLM benchmark.
#
# Runs the same prompt/context/concurrency matrix against both systems and
# produces a JSON for side-by-side plotting. Requires both servers running:
#   - plowrt on $PLOW_SOCKET (UDS)
#   - vLLM on $VLLM_URL (HTTP)
#
# Usage:
#   ./scripts/bench_compare_vllm.sh
#
# Environment:
#   PLOW_SOCKET  — plowrt UDS path (default: /tmp/plowrt.sock)
#   VLLM_URL     — vLLM base URL (default: http://localhost:8000)
#   MODEL        — model slug (default: gemma-4-12b-it)

set -euo pipefail

MODEL="${MODEL:-gemma-4-12b-it}"
PLOW_SOCKET="${PLOW_SOCKET:-/tmp/plowrt.sock}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
OUTPUT="perf-data/compare-plow-vllm-$(date +%Y%m%d-%H%M%S).json"
MAX_TOKENS=128

CONTEXTS=(128 512 2048 8192)
CONCURRENCIES=(1 4 8)

mkdir -p perf-data

echo "=== plowrt vs vLLM benchmark ==="
echo "  model: $MODEL"
echo "  plow:  $PLOW_SOCKET"
echo "  vllm:  $VLLM_URL"
echo "  max_tokens: $MAX_TOKENS"
echo ""

# Benchmark one system. Args: $1=curl_prefix, $2=ctx, $3=conc
bench_one() {
    local curl_cmd="$1"
    local ctx="$2"
    local conc="$3"
    local prompt
    prompt=$(head -c "$ctx" /dev/urandom | base64 | tr -d '\n' | head -c "$ctx")

    local total_tokens=0
    local t_start t_end elapsed_ms

    t_start=$(date +%s%N)
    for _ in $(seq 1 "$conc"); do
        local resp
        resp=$(eval "$curl_cmd" -s \
            -X POST /v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL\",
                \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"stream\": false
            }" 2>/dev/null || echo '{}')
        local toks
        toks=$(echo "$resp" | jq -r '.choices[0].message.content // ""' | wc -c)
        total_tokens=$((total_tokens + toks))
    done
    t_end=$(date +%s%N)
    elapsed_ms=$(( (t_end - t_start) / 1000000 ))

    if [ "$elapsed_ms" -gt 0 ] && [ "$total_tokens" -gt 0 ]; then
        echo "scale=1; $total_tokens * 1000 / $elapsed_ms" | bc
    else
        echo "0"
    fi
}

results="["
first=true

for ctx in "${CONTEXTS[@]}"; do
    for conc in "${CONCURRENCIES[@]}"; do
        echo -n "  ctx=$ctx conc=$conc ... "

        # plowrt (UDS)
        plow_tps=$(bench_one "curl --unix-socket $PLOW_SOCKET http://localhost" "$ctx" "$conc")

        # vLLM (TCP)
        vllm_tps=$(bench_one "curl $VLLM_URL" "$ctx" "$conc")

        ratio="0"
        if [ "$(echo "$vllm_tps > 0" | bc)" -eq 1 ]; then
            ratio=$(echo "scale=2; $plow_tps / $vllm_tps" | bc)
        fi

        echo "plow=${plow_tps} tok/s  vllm=${vllm_tps} tok/s  ratio=${ratio}x"

        if [ "$first" = true ]; then
            first=false
        else
            results="$results,"
        fi
        results="$results{\"model\":\"$MODEL\",\"ctx\":$ctx,\"conc\":$conc,\"max_tokens\":$MAX_TOKENS,\"plow_tps\":$plow_tps,\"vllm_tps\":$vllm_tps,\"ratio\":$ratio}"
    done
done

results="$results]"
echo "$results" | jq '.' > "$OUTPUT"
echo ""
echo "Results written to $OUTPUT"
echo ""
echo "Summary:"
echo "$results" | jq -r '.[] | "  ctx=\(.ctx) conc=\(.conc) plow=\(.plow_tps)tok/s vllm=\(.vllm_tps)tok/s ratio=\(.ratio)x"'
