#!/usr/bin/env bash
set -euo pipefail
gemma_dir=plans/gemma3-12b-roofline
mkdir -p "$gemma_dir/cubin-gemma3-fa256"
cp "$gemma_dir"/cubin-gemma3-ws384/interp_sm90a*.cubin "$gemma_dir/cubin-gemma3-fa256/"
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia \
  -DPLOW_NV_PREFILL=1 -DPLOW_NV_SEGMENTS=1 -DPLOW_NV_FA_ONLY=1 \
  -DPLOW_NV_PACKED_REQUEST=1 -DPLOW_NV_FA_ONLY_HD256=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_GEMMA3=1 \
  -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_NV_MLA=0 \
  -DPLOW_NV_MAMBA=0 -DPLOW_NV_DSA=0 -DPLOW_NV_GEMV_RB=1 \
  -DPLOW_MOE_DOWN_LANESPLIT=1 -DPLOW_NV_FA_WPR=1 -DPLOW_NV_FP8_RB=4 \
  -DPLOW_NV_TMA_GEMM=1 \
  -o "$gemma_dir/cubin-gemma3-fa256/interp_sm90a_pfpackedfa.cubin" runtime/nvidia/interp_sm90a.cu
