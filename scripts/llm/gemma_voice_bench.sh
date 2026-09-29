#!/usr/bin/env bash
# gemma_voice_bench.sh plow|vllm <resdir> [assets]
#
# Voice-agent LLM serving sweep for Gemma 4 E4B on one GPU: `vllm bench serve` (same client for both
# servers, random dataset, completions backend, ignore_eos) at CONCS, then the multi-turn session
# client (scripts/llm/session_bench.py). Run the whole script inside ONE lease:
#
#   gpulease -n 1 gemma-bench timeout 3600 scripts/llm/gemma_voice_bench.sh plow <res> <assets>
#
# Env: PYREF (python with vllm + transformers), HF (checkpoint dir), PLOWRT (plowrt binary),
# CONCS (default "1 8 32 64 128 200"), ISL/OSL (1000/128), PORT, VLLM_MEM (0.85), SERVE_ARGS,
# SESSION_CALLS (call counts for the session client, e.g. "64 200"; 0 = skip; default 64),
# BENCH_ARGS (extra `vllm bench serve` args, e.g. "--temperature 0"; unset = the model's sampling defaults).
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
source "$HERE/scripts/bench/plowbench.sh"
SIDE=${1:?plow|vllm} RES=${2:?resdir} ASSETS=${3:-}
: "${PYREF:?python with vllm}" "${HF:?hf checkpoint dir}"
CONCS=${CONCS:-1 8 32 64 128 200} ISL=${ISL:-1000} OSL=${OSL:-128}
PORT=${PORT:-$(pb_free_port)}
mkdir -p "$RES"
if [ "$SIDE" = plow ]; then
    PLOW_HSACO="$ASSETS" "${PLOWRT:?plowrt binary}" serve --assets "$ASSETS" --port "$PORT" ${SERVE_ARGS:-} \
        > "$RES/server.log" 2>&1 &
else
    "$PYREF" -m vllm.entrypoints.cli.main serve "$HF" --served-model-name gemma-4-e4b --port "$PORT" \
        --max-model-len 8192 --gpu-memory-utilization "${VLLM_MEM:-0.85}" --max-num-seqs 256 \
        --limit-mm-per-prompt '{"image":0,"audio":0}' ${SERVE_ARGS:-} > "$RES/server.log" 2>&1 &
fi
PB_SERVER_PID=$! PB_SERVER_PORT=$PORT PB_SERVER_LOG=$RES/server.log
pb_serve_wait 900 || { pb_serve_stop; exit 3; }
MODEL=$(pb_model_id)
export PB_VLLM=$PYREF PB_TOKENIZER=$HF
for c in $CONCS; do
    np=$(( c < 4 ? 16 : c * 3 ))
    pb_bench "$RES" "c$c" "$MODEL" "$c" "$np" "$ISL" "$OSL" ${BENCH_ARGS:-}
    f=$(pb_result "$RES" "c$c") && python3 - "$f" "$c" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"c={sys.argv[2]:>3} ttft p50 {d['median_ttft_ms']:8.1f} p99 {d['p99_ttft_ms']:8.1f}  tpot p50 {d['median_tpot_ms']:6.2f} "
      f"p99 {d['p99_tpot_ms']:6.2f}  out tok/s {d['output_throughput']:8.1f}  req/s {d['request_throughput']:.2f}")
EOF
done
for n in ${SESSION_CALLS:-64}; do
    [ "$n" -gt 0 ] || continue
    "$PYREF" "$HERE/scripts/llm/session_bench.py" --port "$PORT" --model "$MODEL" --tokenizer "$HF" \
        --calls "$n" --out "$RES/session-$n.json" 2>&1 | tail -20
done
pb_serve_stop
