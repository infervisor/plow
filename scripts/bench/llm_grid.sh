#!/usr/bin/env bash
# llm_grid.sh plow|vllm <resdir> — the matched LLM serving grid, one side per call, run inside ONE
# lease (the same client, `vllm bench serve` via pb_bench, against both servers):
#
#   Q="scripts/bench/gpuq.py submit"
#   $Q grid-plow 1 timeout 3600 env ASSETS=... HF=... scripts/bench/llm_grid.sh plow <res>/plow
#   $Q grid-vllm 1 timeout 3600 env HF=... scripts/bench/llm_grid.sh vllm <res>/vllm
#   scripts/bench/waterfall.py <res>/plow <res>/vllm          # grid table, spread, waterfall
#
# Hygiene it enforces: every cell and repeat draws unique prompts (explicit per-cell seed), so a
# prefix cache cannot replay earlier cells; temperature is always pinned (greedy cells
# `--temperature 0`, sampled cells $SAMPLED); REPS runs per cell for the spread; the vLLM side
# records engine steps; both sides record /metrics cache rates, and Plow also PACKLOG ticks.
#
# Cells: g<c>.r<k> greedy and s<c>.r<k> sampled at ISL/OSL; p<c> prefill-only (OSL=1);
# d<B>x<ctx> decode-only (B prompts of ctx-128 tokens, OSL 256: the decode phase spans ctx±128).
# Env (defaults): HF (checkpoint + tokenizer dir), PYREF (python with vllm: client + reference
# server), ASSETS, PLOWRT (<repo>/target/release/plowrt, copied into <resdir>), MODEL_NAME (served
# name for vLLM; basename of HF), SERVE_ARGS (plowrt serve flags), VLLM_ARGS (vllm serve flags,
# e.g. "--max-model-len 8192"), VLLM_MEM (0.85), CONCS (1 8 32 64 128), ISL/OSL (1000/128),
# REPS (2), SAMPLED ("--temperature 1 --top-p 0.95"; empty = skip), PREFILL_CONCS (64),
# DECODE ("1x1024 64x1024 128x1024"), PACKLOG (1), PORT.
# Add --quality <corpus files...> after resdir to capture natural-text completions instead.
# QUALITY_LENGTHS ("128 1024 4096 7000"), QUALITY_PER_LENGTH (2), QUALITY_CONCURRENCY (1).
# Add --needle for exact-length retrieval; NEEDLE_LENGTHS ("128,1024,4096,7000").
# NEEDLE_CONCURRENCY/NEEDLE_REPEATS (1), NEEDLE_MAX_TOKENS (32), NEEDLE_IGNORE_EOS (unset).
# NEEDLE_CHAT_TEMPLATE_HF optionally supplies the local checkpoint's instruction template.
# Add --agentic for the agentic16k profile (scripts/bench/agentic_turns.py): AGENTIC_CONCS
# (32 64 128) concurrent sessions x AGENTIC_TURNS (10), history growing to AGENTIC_TARGET (15600)
# tokens, AGENTIC_MAX_TOKENS (128) per reply, AGENTIC_API (chat), REPS repeats, greedy + SAMPLED.
# Prefix caching stays ON on both sides (vLLM APC default; plow X-Session-Id + prefix cache).
# Add --prod for the open-loop production mix (agentic_turns.py --open-loop): Poisson session
# arrivals at each PROD_RATES (sessions/s; required) for PROD_DURATION (300) s, PROD_WARMUP (75) /
# PROD_COOLDOWN (25) s unmeasured, PROD_ARGS (extra client args: distributions, SLOs); cells q<1000*rate>.
# Records <resdir>/provenance.json for the strict report (serving_comparison.py render): plow needs
# KV_DTYPE (or PRECISION) and, for a PLOWRT outside a git checkout, PLOWRT_GIT_SHA.
set -u
# Bash otherwise reads later commands from a file that a long campaign may edit.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    grid_source=$(cat -- "${BASH_SOURCE[0]}") || exit
    exec bash -c "$grid_source" "$0" "$@"
fi
HERE=$(cd "$(dirname "$0")/../.." && pwd)
source "$HERE/scripts/bench/plowbench.sh"
case "${1:-}" in plow|vllm) ;; *) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;; esac
SIDE=$1 RES=${2:?resdir}
if [ "$#" -gt 2 ] && ! { { [ "$3" = --quality ] && [ "$#" -ge 4 ]; } || { [ "$3" = --needle ] && [ "$#" -eq 3 ]; } \
        || { { [ "$3" = --agentic ] || [ "$3" = --prod ]; } && [ "$#" -eq 3 ]; }; }; then
    echo 'usage: llm_grid.sh plow|vllm resdir [--quality corpus files... | --needle | --agentic | --prod]' >&2
    exit 2
fi
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
if [ "$SIDE" = plow ]; then server_args=${SERVE_ARGS:-}
else server_args="--gpu-memory-utilization ${VLLM_MEM:-0.85} --max-num-seqs 256 ${VLLM_ARGS:-}"; fi
python3 "$HERE/scripts/bench/serving_comparison.py" record "$RES" --side "$SIDE" --hf "$HF" --reps "$REPS" \
    --server-args "$server_args" --sampled "$SAMPLED" --pyref "$PYREF" --assets "${ASSETS:-}" \
    --plowrt "${PLOWRT:-}" --repo "$HERE"
memory_sampler_pid=
memory_start() { # tag
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-compute-apps=timestamp,pid,used_memory \
            --format=csv,noheader,nounits -lms 100 > "$RES/$1.memory.csv" 2> "$RES/$1.memory.log" &
        memory_sampler_pid=$!
    fi
}
memory_stop() {
    if [ -n "$memory_sampler_pid" ]; then
        kill "$memory_sampler_pid" 2>/dev/null || true
        wait "$memory_sampler_pid" 2>/dev/null || true
        memory_sampler_pid=
    fi
}
memory_finish() { # tag
    memory_stop
    if [ -f "$RES/$1.memory.csv" ]; then
        python3 "$HERE/scripts/bench/gpu_peak_mem.py" "$RES/$1.memory.csv" \
            --root-pid "$PB_SERVER_PID" > "$RES/$1.peak_gpu_memory_mib.txt"
    fi
}
trap 'memory_stop; pb_metrics_stop; pb_serve_stop' EXIT
pb_serve_wait 900 || exit 3
if [ "${3:-}" = --needle ]; then
    needle_template_args=()
    if [ -n "${NEEDLE_CHAT_TEMPLATE_HF:-}" ]; then
        needle_template_args=(--chat-template-hf "$NEEDLE_CHAT_TEMPLATE_HF")
    fi
    "$PYREF" "$HERE/scripts/glm53_needle_probe.py" \
        --url "http://127.0.0.1:$PB_SERVER_PORT" --arm "$SIDE" --out "$RES/needle.json" \
        --lens "${NEEDLE_LENGTHS:-128,1024,4096,7000}" --exact-lengths \
        --concurrency "${NEEDLE_CONCURRENCY:-1}" --repeats "${NEEDLE_REPEATS:-1}" \
        --max-tokens "${NEEDLE_MAX_TOKENS:-32}" ${NEEDLE_IGNORE_EOS:+--ignore-eos} "${needle_template_args[@]}"
    exit $?
fi
if [ "${3:-}" = --quality ]; then
    shift 3
    # shellcheck disable=SC2086
    "$PYREF" "$HERE/scripts/gemma4_greedy_quality.py" capture \
        --url "http://127.0.0.1:$PB_SERVER_PORT" --label "$SIDE" --out "$RES/quality.jsonl" \
        --tokenizer "$HF/tokenizer.json" --corpus "$@" \
        --lengths ${QUALITY_LENGTHS:-128 1024 4096 7000} \
        --per-length "${QUALITY_PER_LENGTH:-2}" --concurrency "${QUALITY_CONCURRENCY:-1}"
    exit $?
fi
MODEL=$(pb_model_id)
export PB_VLLM=$PYREF PB_TOKENIZER=$HF
pb_metrics_start "$RES"
: > "$RES/cells.log"
if [ "${3:-}" = --prod ]; then : "${PROD_RATES:?sessions/s per load level}"; fi
if [ "${3:-}" = --agentic ] || [ "${3:-}" = --prod ]; then
    # Distinct seed per (mode, sessions, repeat): no cell replays another's sessions or system prompt.
    agentic() { # tag mode c rep [client args...]
        local tag=$1 mode=$2 c=$3 rep=$4; shift 4
        echo "CELL_BEGIN $tag $(date +%s.%N)" >> "$RES/cells.log"
        memory_start "$tag"
        "$PYREF" "$HERE/scripts/bench/agentic_turns.py" --url "http://127.0.0.1:$PB_SERVER_PORT" \
            --model "$MODEL" --tokenizer "$HF" --sessions "$c" --turns "${AGENTIC_TURNS:-10}" \
            --target-tokens "${AGENTIC_TARGET:-15600}" --max-tokens "${AGENTIC_MAX_TOKENS:-128}" \
            --api "${AGENTIC_API:-chat}" --seed $(( 7001 + mode * 1000003 + c * 131 + rep * 7919 )) \
            --out "$RES/$tag.json" "$@" > "$RES/$tag.log" 2>&1
        memory_finish "$tag"
        echo "CELL_END $tag $(date +%s.%N)" >> "$RES/cells.log"
        tail -1 "$RES/$tag.log" | sed "s/^/$tag /"
    }
    # Unrecorded warm-up (own seed), so the first cell does not pay first-touch costs.
    "$PYREF" "$HERE/scripts/bench/agentic_turns.py" --url "http://127.0.0.1:$PB_SERVER_PORT" --model "$MODEL" \
        --tokenizer "$HF" --sessions 16 --turns 3 --target-tokens "${AGENTIC_TARGET:-15600}" \
        --max-tokens "${AGENTIC_MAX_TOKENS:-128}" --api "${AGENTIC_API:-chat}" --seed 1 --temperature 0 \
        --out "$RES/warmup.json" > "$RES/warmup.log" 2>&1
    # Same seed per (mode, rate, repeat) on both stacks; distinct across cells and repeats.
    prod() { # tag mode rate rep [client args...]
        local tag=$1 mode=$2 rate=$3 rep=$4 m; shift 4
        m=$(python3 -c "print(round(1000 * $rate))")
        echo "CELL_BEGIN $tag $(date +%s.%N)" >> "$RES/cells.log"
        memory_start "$tag"
        # shellcheck disable=SC2086
        "$PYREF" "$HERE/scripts/bench/agentic_turns.py" --url "http://127.0.0.1:$PB_SERVER_PORT" \
            --model "$MODEL" --tokenizer "$HF" --open-loop --rate "$rate" --duration "${PROD_DURATION:-300}" \
            --warmup "${PROD_WARMUP:-75}" --cooldown "${PROD_COOLDOWN:-25}" --api "${AGENTIC_API:-chat}" \
            --seed $(( 9001 + mode * 1000003 + m * 131 + rep * 7919 )) ${PROD_ARGS:-} \
            --out "$RES/$tag.json" "$@" > "$RES/$tag.log" 2>&1
        memory_finish "$tag"
        echo "CELL_END $tag $(date +%s.%N)" >> "$RES/cells.log"
        tail -1 "$RES/$tag.log" | sed "s/^/$tag /"
    }
    for rep in $(seq 1 "$REPS"); do
        if [ "$3" = --prod ]; then
            for rate in $PROD_RATES; do
                m=$(python3 -c "print(round(1000 * $rate))")
                prod "q$m.g.r$rep" 1 "$rate" "$rep" --temperature 0
                # shellcheck disable=SC2086
                [ -n "$SAMPLED" ] && prod "q$m.s.r$rep" 2 "$rate" "$rep" $SAMPLED
            done
            continue
        fi
        for c in ${AGENTIC_CONCS:-32 64 128}; do
            agentic "a$c.g.r$rep" 1 "$c" "$rep" --temperature 0
            # shellcheck disable=SC2086
            [ -n "$SAMPLED" ] && agentic "a$c.s.r$rep" 2 "$c" "$rep" $SAMPLED
        done
    done
    pb_metrics_stop
    python3 "$HERE/scripts/bench/vllm_metrics.py" cells "$RES" --cache-only --max-prefix-hit 1 > "$RES/cache.json"
    [ "$SIDE" = plow ] && python3 "$HERE/scripts/bench/packlog_audit.py" "$PB_SERVER_LOG" > "$RES/packlog.txt"
    exit 0
fi
# Distinct seed per (kind, conc, isl, osl, repeat): no cell's prompts are a prefix of another's.
cell() { # tag kind conc np isl osl rep [client args...]
    local tag=$1 kind=$2 c=$3 np=$4 isl=$5 osl=$6 rep=$7; shift 7
    memory_start "$tag"
    PB_SEED=$(( 8193 + kind * 1000003 + c * 131 + isl * 7 + osl + rep * 7919 )) \
        pb_cell "$RES" "$tag" "$MODEL" "$c" "$np" "$isl" "$osl" "$@"
    memory_finish "$tag"
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
else
    python3 "$HERE/scripts/bench/packlog_audit.py" "$PB_SERVER_LOG" > "$RES/packlog.txt"
    python3 "$HERE/scripts/bench/vllm_metrics.py" cells "$RES" --cache-only > "$RES/cache.json"
fi
