#!/usr/bin/env bash
# Served showdown (Plow vs pinned vLLM 0.28, alternating rounds) on a bundle. Usage: run_showdown.sh <bundle-dir> <run-id>
set -euo pipefail
export GPU_LEASE_TIMEOUT=14400
B=${1:?bundle}; RUN=${2:?run id}
cd /home/lava/plow
export GPU_LEASE_DIR=/tmp/gpulease
export PLOWRT="$B/bin/plowrt"
export PLOW_ASSETS="$B/assets"
export PLOW_HSACO="$B/hsaco"
export PLOW_ARTIFACTS="$B/assets $B/hsaco $B/source-head.txt /home/shaswot/models/Kimi-K3/config.json /home/shaswot/models/Kimi-K3/tokenizer.json"
export PLOW_SERVER_COMMAND_ARGV="nix develop /home/lava/plow --command $B/bin/plowrt"
export PLOW_ENGINE_ARGV="--rt-checkpoint /tmp/k3-farm.dvzmZN --rt-hsaco $B/hsaco --amd-tp-no-audit --max-hold-ms 8 --slo-ms 250 --max-queued-requests 0"
export PLOW_REQUIRE_VERIFIED=1
export PLOW_REQUIRE_TUNED=1
export VLLM_MODEL=/model_weights
export MODEL_ID=kimi-k3
export SNAP=/model_weights
export VLLM_IMAGE_DIGEST="vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032"
export VLLM_MODEL_IDENTITY="Kimi-K3 config=9710e121 tokenizer=9ca6299a local-96-shard-mxfp4"
export VLLM_SERVER_COMMAND_ARGV="docker run --rm --network host --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined --ipc=host --shm-size=32g -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e VLLM_ROCM_USE_AITER=1 -e SAFETENSORS_FAST_GPU=1 -e VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1 -e AITER_SITUV2_A8W4=1 -e AITER_BF16_FP8_MOE_BOUND=0 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -v /home/shaswot/models/Kimi-K3:/model_weights:ro --entrypoint vllm vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032"
export VLLM_CLIENT_COMMAND_ARGV="docker run --rm --network host --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined -v /home/shaswot/models/Kimi-K3:/model_weights:ro --entrypoint vllm vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032"
export VLLM_ENGINE_ARGV="--trust-remote-code --no-enable-prefix-caching --load-format auto --moe-backend auto --mm-encoder-tp-mode data --max-num-seqs 128 --max-num-batched-tokens 4096 --reasoning-parser kimi_k3 --language-model-only --disable-uvicorn-access-log"
export BRINGUP_CLIENT_ARGV="--trust-remote-code"
export BRINGUP_SEED=0
export DTYPE=auto
export INPUT_MAP=8192
export OUTLEN_MAP=default=1024
export CONCURRENCY_MAP=default=1
export PROMPT_MAP=default=10
export WARMUP_MAP=default=1
export ROUNDS=2
export TP=8
export ROUND_PREFIX=showdown
export RUN_ID="$RUN"
export BRINGUP_OUT="/tmp/$RUN"
unset PLOW_TRACE_RAW PLOW_PREFILL_SEG_TIMING PLOW_PF_TRACE_LOG PLOW_MOE_DECODE_STANDALONE PLOW_PHASE_OBJECTS
exec sg docker -c "perf-data/tools/gpulease -n 8 $RUN perf-data/tools/bringup_showdown.sh"
