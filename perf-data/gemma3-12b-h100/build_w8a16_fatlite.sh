#!/usr/bin/env bash
set -euo pipefail
gemma_dir=plans/gemma3-12b-roofline
gemma_objects="$gemma_dir/cubin-gemma3-w8a16-fatlite"
mkdir -p "$gemma_objects"
cp "$gemma_dir"/cubin-gemma3-fa256/*.cubin "$gemma_objects/"
gemma_flags=(
  -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia
  -DPLOW_NV_PREFILL=1 -DPLOW_NV_SEGMENTS=1 -DPLOW_NV_W8A16_WGMMA=1 -DPLOW_NV_FATLITE=1
  -DPLOW_NV_GEMMA=1 -DPLOW_NV_GEMMA3=1 -DPLOW_NV_FA_GF=2
  -DPLOW_NV_EMBED_SMEM=1 -DPLOW_NV_MLA=0 -DPLOW_NV_MAMBA=0 -DPLOW_NV_DSA=0
  -DPLOW_NV_GEMV_RB=1 -DPLOW_MOE_DOWN_LANESPLIT=1 -DPLOW_NV_FA_WPR=1
  -DPLOW_NV_FP8_RB=4 -DPLOW_NV_TMA_GEMM=1
)
for gemma_packed in 0 1; do
  gemma_suffix=pfseg
  if [ "$gemma_packed" = 1 ]; then gemma_suffix=pfpackedseg; fi
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    "${gemma_flags[@]}" -DPLOW_NV_PACKED_REQUEST="$gemma_packed" \
    -o "$gemma_objects/interp_sm90a_$gemma_suffix.cubin" runtime/nvidia/interp_sm90a.cu
done
