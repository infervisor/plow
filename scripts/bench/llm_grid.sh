#!/usr/bin/env bash
# llm_grid.sh plow|vllm <resdir> — the matched LLM serving grid, one side per call, run inside ONE
# lease (the same client, `vllm bench serve` via pb_bench, against both servers):
#
#   GL=perf-data/tools/gpulease
#   $GL -n 1 grid-plow timeout 3600 env ASSETS=... HF=... scripts/bench/llm_grid.sh plow <res>/plow
#   $GL -n 1 grid-vllm timeout 3600 env HF=... scripts/bench/llm_grid.sh vllm <res>/vllm
#   scripts/bench/waterfall.py <res>/plow <res>/vllm          # grid table, spread, waterfall
#
# Hygiene it enforces: every cell and repeat draws unique prompts (explicit per-cell seed), so a
# prefix cache cannot replay earlier cells; temperature is always pinned (greedy cells
# `--temperature 0`, sampled cells $SAMPLED); REPS runs per cell for the spread; the vLLM side
# records /metrics (prefix-cache hit rate, engine steps) and the plow side PLOW_PF_PACKLOG ticks.
#
# Cells: g<c>.r<k> greedy and s<c>.r<k> sampled at ISL/OSL; p<c> prefill-only (OSL=1);
# d<B>x<ctx> decode-only (B prompts of ctx-128 tokens, OSL 256: the decode phase spans ctx±128).
# Env (defaults): HF (checkpoint + tokenizer dir), PYREF (python with vllm: client + reference
# server), ASSETS, PLOWRT (<repo>/target/release/plowrt, copied into <resdir>), MODEL_NAME (served
# name for vLLM; basename of HF), SERVE_ARGS (plowrt serve flags), VLLM_ARGS (vllm serve flags,
# e.g. "--max-model-len 8192"), VLLM_MEM (0.85), CONCS (1 8 32 64 128), ISL/OSL (1000/128),
# REPS (2), SAMPLED ("--temperature 1 --top-p 0.95"; empty = skip), PREFILL_CONCS (64),
# DECODE ("1x1024 64x1024 128x1024"), PACKLOG (1), PORT.
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
source "$HERE/scripts/bench/plowbench.sh"
case "${1:-}" in plow|vllm) ;; *) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;; esac
SIDE=$1 RES=${2:?resdir}
: "${PYREF:?python with vllm}" "${HF:?checkpoint dir}"
CONCS=${CONCS:-1 8 32 64 128} ISL=${ISL:-1000} OSL=${OSL:-128} REPS=${REPS:-2}
SAMPLED=${SAMPLED---temperature 1 --top-p 0.95}
mkdir -p "$RES"
PB_SERVER_PORT=${PORT:-$(pb_free_port)} PB_SERVER_LOG=$RES/server.log
if [ "$SIDE" = plow ]; then
    : "${ASSETS:?plow assets dir}"
    cp "${PLOWRT:-$HERE/target/release/plowrt}" "$RES/plowrt"
    # shellcheck disable=SC2086
    PLOW_PF_PACKLOG=${PACKLOG:-1} PLOW_HSACO="$ASSETS" "$RES/plowrt" serve --assets "$ASSETS" \
        --port "$PB_SERVER_PORT" ${SERVE_ARGS:-} > "$PB_SERVER_LOG" 2>&1 &
else
    # shellcheck disable=SC2086
    "$PYREF" -m vllm.entrypoints.cli.main serve "$HF" --served-model-name "${MODEL_NAME:-$(basename "$HF")}" \
        --port "$PB_SERVER_PORT" --gpu-memory-utilization "${VLLM_MEM:-0.85}" --max-num-seqs 256 \
        ${VLLM_ARGS:-} > "$PB_SERVER_LOG" 2>&1 &
fi
PB_SERVER_PID=$!
trap 'pb_metrics_stop; pb_serve_stop' EXIT
pb_serve_wait 900 || exit 3
MODEL=$(pb_model_id)
export PB_VLLM=$PYREF PB_TOKENIZER=$HF
[ "$SIDE" = vllm ] && pb_metrics_start "$RES"
: > "$RES/cells.log"
# Distinct seed per (kind, conc, isl, osl, repeat): no cell's prompts are a prefix of another's.
cell() { # tag kind conc np isl osl rep [client args...]
    local tag=$1 kind=$2 c=$3 np=$4 isl=$5 osl=$6 rep=$7; shift 7
    PB_SEED=$(( 8193 + kind * 1000003 + c * 131 + isl * 7 + osl + rep * 7919 )) \
        pb_cell "$RES" "$tag" "$MODEL" "$c" "$np" "$isl" "$osl" "$@"
    local f; f=$(pb_result "$RES" "$tag") && python3 - "$f" "$tag" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]:12s} ttft p50 {d['median_ttft_ms']:8.1f}  tpot p50 {d['median_tpot_ms']:6.2f}  "
      f"out tok/s {d['output_throughput']:8.1f}  done {d['completed']}")
EOF
}
for rep in $(seq 1 "$REPS"); do
    for c in $CONCS; do
        np=$(( c < 4 ? 16 : c * 3 ))
        cell "g$c.r$rep" 1 "$c" "$np" "$ISL" "$OSL" "$rep" --temperature 0
        # shellcheck disable=SC2086
        [ -n "$SAMPLED" ] && cell "s$c.r$rep" 2 "$c" "$np" "$ISL" "$OSL" "$rep" $SAMPLED
    done
done
for c in ${PREFILL_CONCS-64}; do cell "p$c" 3 "$c" $(( c * 3 )) "$ISL" 1 1 --temperature 0; done
for bc in ${DECODE-1x1024 64x1024 128x1024}; do
    b=${bc%x*} ctx=${bc#*x}
    cell "d$bc" 4 "$b" $(( b < 4 ? 4 : b )) $(( ctx - 128 )) 256 1 --temperature 0
done
pb_metrics_stop
if [ "$SIDE" = vllm ]; then python3 "$HERE/scripts/bench/vllm_metrics.py" cells "$RES"
else python3 "$HERE/scripts/bench/packlog_audit.py" "$PB_SERVER_LOG" > "$RES/packlog.txt"; fi
