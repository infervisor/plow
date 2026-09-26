#!/usr/bin/env bash
# build_sampler.sh <out.cubin> — the sampler cubin exactly as runtime/CMakeLists.txt builds it.
source /root/tts-work/cuda-env.sh
cd /root/plow/.claude/worktrees/tts-veena-chatterbox
bash runtime/cmake/nvcc_cubin.sh "$PLOW_NVCC" "$1" plow_sample required -arch=sm_90a -O3 -cubin runtime/nvidia/sample_sm120.cu
