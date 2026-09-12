#!/usr/bin/env bash
set -euo pipefail
gemma_base=${1:?usage: build_sm90a_gemma4_segments.sh base-cubin-dir output-dir}
gemma_out=${2:?usage: build_sm90a_gemma4_segments.sh base-cubin-dir output-dir}
if [ "$gemma_base" = "$gemma_out" ]; then
  echo 'Use a separate output directory for the candidate objects.' >&2
  exit 2
fi
for gemma_file in interp_sm90a.cubin interp_sm90a_pf.cubin interp_sm90a_pfseg.cubin interp_sm90a_pfpackedseg.cubin; do
  test -f "$gemma_base/$gemma_file"
done
mkdir -p "$gemma_out"
cp "$gemma_base"/*.cubin "$gemma_out/"
gemma_flags=(
  -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia
  -DPLOW_NV_PREFILL=1 -DPLOW_NV_SEGMENTS=1
  -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1
  -DPLOW_NV_MLA=0 -DPLOW_NV_MAMBA=0 -DPLOW_NV_DSA=0
  -DPLOW_NV_GEMV_RB=1 -DPLOW_MOE_DOWN_LANESPLIT=1 -DPLOW_NV_FA_WPR=1
  -DPLOW_NV_FP8_RB=4 -DPLOW_NV_TMA_GEMM=1
)
for gemma_packed in 0 1; do
  gemma_prefix=pf
  if [ "$gemma_packed" = 1 ]; then gemma_prefix=pfpacked; fi
  gemma_padding_flags=()
  if [ "$gemma_packed" = 1 ] && [ "${PLOW_BUILD_MASKED_PADDING:-0}" = 1 ]; then
    gemma_padding_flags=(-DPLOW_NV_MASKED_PADDING=1)
  fi
  for gemma_role in gemm fa; do
    if [ "$gemma_role" = gemm ]; then
      gemma_role_flags=(
        -DPLOW_NV_SEG_WS384=1 -DPGM90_UNI_BN256=1 -DPLOW_NV_SEG_GEMM=1
        -DPLOW_NV_GEMM_ONLY=1 -DPGM90_TMA_STAGES=3
        -DPGM90_WS384_PREFETCH=1 -DPGM90_WS384_ISSUE_CURSOR=1
        -DPGM90_WS384_SMEPI="${PLOW_BUILD_GEMM_SMEPI:-0}"
      )
    else
      gemma_role_flags=(
        -DPLOW_NV_FA_ONLY=1 -DPLOW_NV_FA_ONLY_HD256=1
        -DPLOW_NV_FA512_WG=1 -DPLOW_NV_FA512_BKV=32
        -DPLOW_NV_PACKED_FA_WGMMA=1 -DPLOW_NV_PACKED_FA_TMA=1
      )
    fi
    env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
      "${gemma_flags[@]}" "${gemma_role_flags[@]}" "${gemma_padding_flags[@]}" -DPLOW_NV_PACKED_REQUEST="$gemma_packed" \
      -o "$gemma_out/interp_sm90a_$gemma_prefix$gemma_role.cubin" runtime/nvidia/interp_sm90a.cu
  done
  # Packed light occupancy needs matching PLOW_SEG_SLICE_ALL=1 packets.
  if [ "$gemma_packed" = 1 ] && { [ "${PLOW_BUILD_FATLITE:-0}" = 1 ] || [ "${PLOW_BUILD_MASKED_PADDING:-0}" = 1 ]; }; then
    env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
      "${gemma_flags[@]}" "${gemma_padding_flags[@]}" -DPLOW_NV_PACKED_REQUEST=1 \
      -DPLOW_NV_FATLITE="${PLOW_BUILD_FATLITE:-0}" -DPGM90_TMA_STAGES=3 \
      -o "$gemma_out/interp_sm90a_pfpackedseg.cubin" runtime/nvidia/interp_sm90a.cu
  fi
done
if [ "${PLOW_BUILD_FA_GQA2_PAIR:-0}" = 1 ]; then
  gemma_gqa2_padding_flags=()
  if [ "${PLOW_BUILD_MASKED_PADDING:-0}" = 1 ]; then
    gemma_gqa2_padding_flags=(-DPLOW_NV_MASKED_PADDING=1)
  fi
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    "${gemma_flags[@]}" "${gemma_gqa2_padding_flags[@]}" -DPLOW_NV_PACKED_REQUEST=1 \
    -DPLOW_NV_FA_ONLY=1 -DPLOW_NV_FA_ONLY_HD256=1 \
    -DPLOW_NV_FA_ONLY_HD256_EXACT=1 -DPLOW_NV_FA_WGITEM=1 \
    -DPLOW_NV_FA_GQA2_PAIR=1 -DPLOW_NV_PACKED_FA_WGMMA=1 \
    -DPLOW_NV_PACKED_FA_TMA=1 \
    -o "$gemma_out/interp_sm90a_pfpackedfa256_gqa2.cubin" \
    runtime/nvidia/interp_sm90a.cu
  /usr/local/cuda/bin/cuobjdump -symbols \
    "$gemma_out/interp_sm90a_pfpackedfa256_gqa2.cubin" | \
    grep -q plow_attention_sm90_hd256_gqa2_abi
fi
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v \
  -I runtime/common -I runtime/nvidia \
  -o "$gemma_out/interp_sm90a_pfgemm_w8a16_m1.cubin" \
  runtime/nvidia/interp_sm90a_pfgemm_w8a16_m1.cu
/usr/local/cuda/bin/cuobjdump -symbols "$gemma_out/interp_sm90a_pfgemm_w8a16_m1.cubin" | \
  grep -q plow_sm90a_pfgemm_w8a16_m1
if [ "${PLOW_BUILD_PFATTN_HD256_BKV64:-0}" = 1 ]; then
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia \
    -o "$gemma_out/interp_sm90a_pfattn_hd256_bkv64.cubin" \
    runtime/nvidia/interp_sm90a_pfattn_hd256_bkv64.cu
fi
if [ "${PLOW_BUILD_MASKED_PADDING:-0}" = 1 ]; then
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia \
    -DPLOW_NV_FA512_WG=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_WPR=1 \
    -DPLOW_NV_FA_QK_UNROLL="${PLOW_BUILD_PFATTN_QK_UNROLL:-32}" \
    -DPLOW_NV_FA512_KV64="${PLOW_BUILD_PFATTN_KV64:-1}" \
    -DPLOW_NV_FA512_N_SPLIT="${PLOW_BUILD_PFATTN_N_SPLIT:-1}" \
    -DPLOW_NV_FA512_FIXED_HEADS="${PLOW_BUILD_PFATTN_FIXED_HEADS:-0}" \
    -DPLOW_NV_PACKED_REQUEST=1 -DPLOW_NV_PACKED_FA_WGMMA=1 -DPLOW_NV_PACKED_FA_TMA=1 \
    -DPLOW_NV_MASKED_PADDING=1 -o "$gemma_out/interp_sm90a_pfattn_hd512.cubin" \
    runtime/nvidia/interp_sm90a_pfattn_hd512.cu
fi
