#!/usr/bin/env bash
# scripts/bench_plowrt_sweep.sh — Load-sweep benchmark for plowrt.
#
# Same methodology as perf-data/bench_vllm_docker.sh: single-user sequential
# requests at varying context lengths. Measures time-per-output-token (TPOT)
# and throughput (tok/s) for comparison.
#
# Usage:
#   ./scripts/bench_plowrt_sweep.sh [MODEL] [SOCKET]
#
# Requires: curl, jq, plowrt running on the socket.

set -euo pipefail

MODEL="${1:-gemma-4-12b-it}"
SOCKET="${2:-/tmp/plowrt.sock}"
OUTPUT="perf-data/plowrt-sweep-$(date +%Y%m%d-%H%M%S).json"
MAX_TOKENS=128

CONTEXTS=(128 512 2048 4096 8192 16384)
CONCURRENCIES=(1 4 8 16)

mkdir -p perf-data

echo "plowrt sweep: model=$MODEL socket=$SOCKET max_tokens=$MAX_TOKENS"
echo "contexts: ${CONTEXTS[*]}"
echo "concurrencies: ${CONCURRENCIES[*]}"
echo ""

results="["
first=true

for ctx in "${CONTEXTS[@]}"; do
  for conc in "${CONCURRENCIES[@]}"; do
    # Generate a random prompt of ctx tokens (base64-encoded random bytes).
    prompt=$(head -c "$ctx" /dev/urandom | base64 | tr -d '\n' | head -c "$ctx")

    # Sequential requests for this (ctx, conc) pair.
    total_tokens=0
    t_start=$(date +%s%N)

    for _ in $(seq 1 "$conc"); do
      resp=$(curl -s --unix-socket "$SOCKET" \
        -X POST http://localhost/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"$MODEL\",
          \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
          \"max_tokens\": $MAX_TOKENS,
          \"stream\": false
        }" 2>/dev/null || echo '{}')

      # Extract completion tokens from usage (if present) or count from content.
      toks=$(echo "$resp" | jq -r '.choices[0].message.content // ""' | wc -c)
      total_tokens=$((total_tokens + toks))
    done

    t_end=$(date +%s%N)
    elapsed_ms=$(( (t_end - t_start) / 1000000 ))
    elapsed_s=$(echo "scale=3; $elapsed_ms / 1000" | bc)

    if [ "$elapsed_ms" -gt 0 ]; then
      tps=$(echo "scale=1; $total_tokens / $elapsed_s" | bc)
      tpot_ms=$(echo "scale=2; $elapsed_ms / $total_tokens" | bc 2>/dev/null || echo "0")
    else
      tps="0"
      tpot_ms="0"
    fi

    echo "  ctx=$ctx conc=$conc → ${tps} tok/s (TPOT=${tpot_ms}ms)"

    if [ "$first" = true ]; then
      first=false
    else
      results="$results,"
    fi
    results="$results{\"model\":\"$MODEL\",\"ctx\":$ctx,\"conc\":$conc,\"max_tokens\":$MAX_TOKENS,\"total_tokens\":$total_tokens,\"elapsed_ms\":$elapsed_ms,\"tps\":$tps,\"tpot_ms\":$tpot_ms}"
  done
done

results="$results]"
echo "$results" | jq '.' > "$OUTPUT"
echo ""
echo "Results written to $OUTPUT"
