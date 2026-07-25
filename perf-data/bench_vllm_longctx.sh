#!/usr/bin/env bash
# =============================================================================
# bench_vllm_longctx.sh — vLLM LONG-CONTEXT (48k-128k) TP sweep for Gemma-4-31B
# =============================================================================
# Thin driver over bench_vllm_tp.sh for plow's strongest regime: long context.
# Serves Gemma-4-31B at --max-model-len 131200 (128k + headroom; Gemma native
# max_position_embeddings=262144 supports it) and sweeps the long contexts
# {48k,64k,72k,96k,128k}, batch 1, bf16, TTFT (prefill) + TPOT (decode ms/tok).
#
# Everything is delegated to bench_vllm_tp.sh via env overrides — no logic here.
#
#   TP=4 GPUS=4,5,6,7 CNAME=vllm_lc_tp4 PORT=8104 bash perf-data/bench_vllm_longctx.sh
#   TP=8 GPUS=0,1,2,3,4,5,6,7 CNAME=vllm_lc_tp8 PORT=8108 bash perf-data/bench_vllm_longctx.sh
#
# KV-cache OOM watch: at 128k a batch-1 KV cache is large; if a TP level OOMs at
# serve/health, the container exits and the driver reports it (see the harness's
# "container exited during startup" path). TP=8 shards KV 8-way (least pressure).
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-vllm/vllm-openai-rocm:latest}"
export MODEL="${MODEL:-gemma-4-31B-it}"
export MAXLEN="${MAXLEN:-131200}"                         # 128k + headroom
export CTXS="${CTXS:-49152,65536,73728,98304,131072}"     # 48k,64k,72k,96k,128k
export OUTPUT_LEN="${OUTPUT_LEN:-128}"
export NUM_PROMPTS="${NUM_PROMPTS:-3}"
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"            # 31B multi-GPU load is slow
export TP="${TP:-4}"
export GPUS="${GPUS:-4,5,6,7}"
export CNAME="${CNAME:-vllm_lc_tp${TP}}"
export PORT="${PORT:-810${TP}}"
export OUTDIR="${OUTDIR:-$HERE/vllm_longctx_logs}"

exec bash "$HERE/bench_vllm_tp.sh"
