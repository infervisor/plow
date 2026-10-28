#!/usr/bin/env bash
# bench_gemma26b_voice50.sh — Reproduce 30 & 50 concurrent voice calls on 1x H100 SXM5
# Co-serving Qwen3-ASR (1.7B) + Gemma-4-26B-A4B-it FP8 + Chatterbox-MTL (T3 + S3Gen v3)
set -euo pipefail

HERE=$(cd "$(dirname "$0")/../.." && pwd)
RES=${1:-/opt/dlami/nvme/lava-tts/g26-voice50/res}

export PY=${PY:-/opt/pytorch/bin/python3}
export PYTHONPATH=/opt/dlami/nvme/lava-tts/pyextra:/opt/dlami/nvme/lava-tts/pyextra-librosa
export MANIFEST=${MANIFEST:-/opt/dlami/nvme/lava-tts/audio/manifest.json}
export ASR_ASSETS=${ASR_ASSETS:-/opt/dlami/nvme/lava-tts/asrpackets/stage/qwen3-asr}
export TTS_ASSETS=${TTS_ASSETS:-/opt/dlami/nvme/lava-tts/kit-tts-b451c8c6/chatterbox-mtl/assets}
export LLM_ASSETS=${LLM_ASSETS:-/opt/dlami/nvme/lava-tts/g26kit-20261020-k2/assets}
export GPULEASE=${GPULEASE:-/home/lava/plow/perf-data/tools/gpulease}
export GPU_LEASE_TIMEOUT=7200
export VA_RUN_S=7200
export LD_LIBRARY_PATH=/opt/dlami/nvme/lava-tts/g26kit-20261020-k2/lib:/usr/local/cuda-12.9/lib64
export PLOWRT_BIN=${PLOWRT_BIN:-/opt/dlami/nvme/lava-tts/target-main/release/plowrt}

# Gemma-26B FP8 optimal serving environment
export PLOW_DECODE_MAX_RUNG=128
export PLOW_DECODE_PIPELINE=1
export PLOW_LT_FP8_FAST_ACCUM_MAX_K=8192
export PLOW_LT_RUNG_ALGOS=true
export PLOW_MULTISTEP=0
export PLOW_SPEC=1
export PLOW_PF_ADMIT_OVERLAP=1
export PLOW_PF_INTERLEAVE=0
export PLOW_PF_ITL_BUDGET_MS=60
export PLOW_PF_ITL_FORCE=0
export PLOW_PF_SEG_FA512=1
export PLOW_PF_SEG_PURE=fp8
export PLOW_PREFIX_CACHE=1
export PLOW_TERMINAL_LT=0
export PLOW_VMM_LIVE_RINGS=1
export PLOW_VMM_PREFIX=1
export PLOW_VMM_RING_PARK=8
export PLOW_VMM_STRICT_PUBLISH=1

# Multi-tenant voice knobs
export PLOW_TTS_TURN_BATCH=16
export VA_LIVE_CTX="qwen3-asr=768,chatterbox-mtl=512"
export VA_CO_SCHED=deadline
export TURNS=3
export CALL_ARGS="--language en --slo-asr-ms 1000 --slo-ttft-ms 1000 --slo-ttfa-ms 1500 --slo-underrun-frac 0.05"

cd "$HERE"
bash scripts/voice/serve_voice_agent.sh calls "$RES" 30 50
cat "$RES/slo.md"
