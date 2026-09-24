#!/usr/bin/env bash
set -euo pipefail
gemma_base=${1:?usage: build_sm90a_gemma4_segments.sh base-cubin-dir output-dir}
gemma_out=${2:?usage: build_sm90a_gemma4_segments.sh base-cubin-dir output-dir}
if [ "$gemma_base" = "$gemma_out" ]; then
  echo 'Use a separate output directory for the candidate objects.' >&2
  exit 2
fi
gemma_required=(interp_sm90a.cubin interp_sm90a_pf.cubin interp_sm90a_pfseg.cubin)
# Packed-request objects exist only for a packet that HAS packed-prefill topology. A Gemma MoE FP8
# packet legitimately has none (PLOW_EMIT_PACKED_PREFILL panics for it at devgen lib.rs:9774), so
# the base emit produces no pfpacked* object and interp_sm120.cu:203 hard-#errors if one is built
# anyway. Requiring it here unconditionally made `set -e` abort with no message at all.
gemma_has_packed=1
if [ -f "$gemma_base/plow_config.h" ] &&
   ! grep -qx '#define PLOW_PACKET_HAS_PACKED_PREFILL_TOPOLOGY 1' "$gemma_base/plow_config.h"; then
  gemma_has_packed=0
fi
if [ "$gemma_has_packed" = 1 ]; then
  gemma_required+=(interp_sm90a_pfpackedseg.cubin)
fi
for gemma_file in "${gemma_required[@]}"; do
  test -f "$gemma_base/$gemma_file" || {
    echo "missing base object: $gemma_base/$gemma_file" >&2
    exit 2
  }
done
mkdir -p "$gemma_out"
cp "$gemma_base"/*.cubin "$gemma_out/"
gemma_config_flags=()
if [ -n "${PLOW_CUBIN_CONFIG:-}" ]; then
  test -f "$PLOW_CUBIN_CONFIG" || {
    echo "missing PLOW_CUBIN_CONFIG: $PLOW_CUBIN_CONFIG" >&2
    exit 2
  }
  gemma_config_dir=$(dirname -- "$PLOW_CUBIN_CONFIG")
  gemma_config_name=$(basename -- "$PLOW_CUBIN_CONFIG")
  gemma_config_flags=(
    -I "$gemma_config_dir"
    "-DPLOW_CONFIG=\"$gemma_config_name\""
    -DPLOW_BUCKET_DECODE=0
  )
fi
gemma_flags=(
  -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia
  -DPLOW_NV_PREFILL=1 -DPLOW_NV_SEGMENTS=1
  -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1
  -DPLOW_NV_MLA=0 -DPLOW_NV_MAMBA=0 -DPLOW_NV_DSA=0
  -DPLOW_NV_GEMV_RB=1 -DPLOW_MOE_DOWN_LANESPLIT=1 -DPLOW_NV_FA_WPR=1
  -DPLOW_NV_FP8_RB=4 -DPLOW_NV_TMA_GEMM=1 -DPLOW_NV_QUANT_WPR=1
)
gemma_w8a8=0
if [ "${PLOW_BUILD_W8A8:-0}" = 1 ] ||
   { [ -n "${PLOW_CUBIN_CONFIG:-}" ] && grep -qx '#define PLOW_HAS_QUANT_FP8 1' "$PLOW_CUBIN_CONFIG"; }; then
  gemma_w8a8=1
  gemma_flags+=(
    -DPLOW_NV_W8A8=1
    -DPGM90_FP8_PROMOTE="${PLOW_W8A8_PROMOTE:-1}"
  )
fi
# BF16 packets default to the H100 recipe's object set; each PLOW_BUILD_* below still overrides.
# FATLITE: the light-op object (flash arms out, 128-reg cap, 2 blocks/SM; 12B -2.8 ms @1024,
# -4.8 ms @4096, bit-identical). A Gemma MoE packet keeps its grouped prefill bodies in it
# (FATLITE_MOE): no other object implements them. MASKED_PADDING also builds the WGMMA BQ64 hd512
# role object (interp_sm90a_pfattn_hd512.cubin).
gemma_bf16=$((1 - gemma_w8a8))
gemma_moe_pf=0
if [ -n "${PLOW_CUBIN_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_HAS_MOE_GROUP_GLU_GEMMA_PF 1' "$PLOW_CUBIN_CONFIG"; then
  gemma_moe_pf=1
fi
gemma_fatlite=${PLOW_BUILD_FATLITE:-$gemma_bf16}
gemma_fatlite_moe=${PLOW_BUILD_FATLITE_MOE:-$((gemma_fatlite == 1 ? gemma_moe_pf : 0))}
gemma_masked=${PLOW_BUILD_MASKED_PADDING:-$gemma_bf16}
# The masked-padding GUARD is a correctness requirement in every object that writes KV on a
# packed prefill, and it is orthogonal to precision. op_norm.cuh:780 skips a pfslot[t] < 0 row
# before the unguarded (unsigned)pfslot[t] cast at :806, so without it a negative slot becomes
# a huge obase -- CUDA_ERROR_ILLEGAL_ADDRESS on the first KV write. Defaulting it to
# $gemma_bf16 compiled it out of every FP8 packet, which is why all three faulted there and no
# bf16 packet ever did. $gemma_masked still gates the hd512/hd256 ROLE OBJECTS below, which
# are a bf16-only default and a separate decision.
gemma_masked_def=${PLOW_BUILD_MASKED_PADDING:-1}
# Raw extra nvcc flags for every segment object: the A/B arm of a kernel default on a block packet.
if [ -n "${PLOW_BUILD_SEG_EXTRA_DEFINES:-}" ]; then
  read -r -a gemma_extra_flags <<<"$PLOW_BUILD_SEG_EXTRA_DEFINES"
  gemma_flags+=("${gemma_extra_flags[@]}")
fi
for gemma_packed in 0 1; do
  # No packed-prefill topology -> no packed-request objects to build (interp_sm120.cu:203).
  if [ "$gemma_packed" = 1 ] && [ "$gemma_has_packed" = 0 ]; then continue; fi
  gemma_prefix=pf
  if [ "$gemma_packed" = 1 ]; then gemma_prefix=pfpacked; fi
  gemma_padding_flags=()
  if [ "$gemma_packed" = 1 ] && [ "$gemma_masked_def" = 1 ]; then
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
        -DPLOW_NV_FA_ONLY_HD256_ONLY="${PLOW_BUILD_FA_HD256_ONLY:-0}"
        -DPLOW_NV_FA512_WG=1 -DPLOW_NV_FA512_BKV=32
        -DPLOW_NV_PACKED_FA_WGMMA=1 -DPLOW_NV_PACKED_FA_TMA=1
      )
    fi
    env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
      "${gemma_flags[@]}" "${gemma_config_flags[@]}" "${gemma_role_flags[@]}" "${gemma_padding_flags[@]}" -DPLOW_NV_PACKED_REQUEST="$gemma_packed" \
      -o "$gemma_out/interp_sm90a_$gemma_prefix$gemma_role.cubin" runtime/nvidia/interp_sm90a.cu
  done
  # A packet config binds the packed light object to that packet even when its
  # implementation flags are unchanged.
  if [ "$gemma_packed" = 1 ] && { [ -n "${PLOW_CUBIN_CONFIG:-}" ] || [ "$gemma_fatlite" = 1 ] || [ "$gemma_masked" = 1 ]; }; then
    env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
      "${gemma_flags[@]}" "${gemma_config_flags[@]}" "${gemma_padding_flags[@]}" -DPLOW_NV_PACKED_REQUEST=1 \
      -DPLOW_NV_FATLITE="$gemma_fatlite" \
      -DPLOW_NV_FATLITE_MOE="$gemma_fatlite_moe" -DPGM90_TMA_STAGES=3 \
      -o "$gemma_out/interp_sm90a_pfpackedseg.cubin" runtime/nvidia/interp_sm90a.cu
  fi
done
if [ "${PLOW_BUILD_FA_GQA2_PAIR:-$gemma_bf16}" = 1 ] && [ "$gemma_has_packed" = 1 ]; then
  gemma_gqa2_padding_flags=()
  if [ "$gemma_masked_def" = 1 ]; then
    gemma_gqa2_padding_flags=(-DPLOW_NV_MASKED_PADDING=1)
  fi
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    "${gemma_flags[@]}" "${gemma_config_flags[@]}" "${gemma_gqa2_padding_flags[@]}" -DPLOW_NV_PACKED_REQUEST=1 \
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
# Causal softmax of the vendor-GEMM attention route (PLOW_PF_ATTN_GEMM): its presence turns the route on, KV budget permitting.
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v \
  -I runtime/common -I runtime/nvidia \
  -o "$gemma_out/attn_softmax_sm90a.cubin" \
  runtime/nvidia/attn_softmax_sm90a.cu
/usr/local/cuda/bin/cuobjdump -symbols "$gemma_out/attn_softmax_sm90a.cubin" | \
  grep -q plow_attn_softmax_abi
gemma_glu_log=$(mktemp)
if ! env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v \
  -I runtime/common -I runtime/nvidia \
  -o "$gemma_out/interp_sm90a_pfgemm_glu_gemma4.cubin" \
  runtime/nvidia/interp_sm90a_pfgemm_glu_gemma4.cu 2>&1 | tee "$gemma_glu_log"; then
  rm -f "$gemma_glu_log"
  exit 1
fi
if grep -Eq '[1-9][0-9]* bytes (stack frame|spill stores|spill loads)' "$gemma_glu_log"; then
  echo 'Gemma-4 fused GLU object uses stack or spills.' >&2
  rm -f "$gemma_glu_log"
  exit 1
fi
rm -f "$gemma_glu_log"
gemma_glu_symbols=$(/usr/local/cuda/bin/cuobjdump -symbols \
  "$gemma_out/interp_sm90a_pfgemm_glu_gemma4.cubin")
for gemma_glu_symbol in \
  plow_sm90a_pfgemm_glu_gemma4 \
  plow_pfgemm_glu_gemma4_abi \
  plow_pfgemm_glu_gemma4_min_rows \
  plow_pfgemm_glu_gemma4_max_rows \
  plow_pfgemm_glu_gemma4_n \
  plow_pfgemm_glu_gemma4_k \
  plow_pfgemm_glu_gemma4_stages \
  plow_pfgemm_glu_gemma4_bm \
  plow_pfgemm_glu_gemma4_bn \
  plow_pfgemm_glu_gemma4_bk \
  plow_block_pfgemm_glu_gemma4 \
  plow_arena_bytes_pfgemm_glu_gemma4 \
  plow_pf_request_abi \
  plow_pf_masked_padding_abi \
  plow_pf_fp8_request_abi \
  plow_pf_fp8_masked_padding_abi; do
  grep -q "$gemma_glu_symbol" <<<"$gemma_glu_symbols" || {
    echo "missing Gemma-4 fused GLU symbol: $gemma_glu_symbol" >&2
    exit 1
  }
done
gemma_w8a8_glu_log=$(mktemp)
if ! env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v \
  -I runtime/common -I runtime/nvidia \
  -o "$gemma_out/interp_sm90a_pfgemm_glu_w8a8_gemma4.cubin" \
  runtime/nvidia/interp_sm90a_pfgemm_glu_w8a8_gemma4.cu 2>&1 | tee "$gemma_w8a8_glu_log"; then
  rm -f "$gemma_w8a8_glu_log"
  exit 1
fi
if grep -Eq '[1-9][0-9]* bytes (stack frame|spill stores|spill loads)' "$gemma_w8a8_glu_log"; then
  echo 'Gemma-4 W8A8 fused GLU object uses stack or spills.' >&2
  rm -f "$gemma_w8a8_glu_log"
  exit 1
fi
rm -f "$gemma_w8a8_glu_log"
gemma_w8a8_glu_symbols=$(/usr/local/cuda/bin/cuobjdump -symbols \
  "$gemma_out/interp_sm90a_pfgemm_glu_w8a8_gemma4.cubin")
for gemma_w8a8_glu_symbol in \
  plow_sm90a_pfgemm_glu_w8a8_gemma4 \
  plow_sm90a_pfgemm_glu_w8a8_gemma4_direct \
  plow_pfgemm_glu_w8a8_gemma4_abi \
  plow_pfgemm_glu_w8a8_gemma4_min_rows \
  plow_pfgemm_glu_w8a8_gemma4_max_rows \
  plow_pfgemm_glu_w8a8_gemma4_n \
  plow_pfgemm_glu_w8a8_gemma4_k \
  plow_pfgemm_glu_w8a8_gemma4_stages \
  plow_pfgemm_glu_w8a8_gemma4_bm \
  plow_pfgemm_glu_w8a8_gemma4_bn \
  plow_pfgemm_glu_w8a8_gemma4_bk \
  plow_pfgemm_glu_w8a8_gemma4_tile_band \
  plow_pfgemm_glu_w8a8_gemma4_direct_entry \
  plow_block_pfgemm_glu_w8a8_gemma4 \
  plow_arena_bytes_pfgemm_glu_w8a8_gemma4 \
  plow_pf_request_abi \
  plow_pf_masked_padding_abi \
  plow_pf_fp8_request_abi \
  plow_pf_fp8_masked_padding_abi; do
  grep -q "$gemma_w8a8_glu_symbol" <<<"$gemma_w8a8_glu_symbols" || {
    echo "missing Gemma-4 W8A8 fused GLU symbol: $gemma_w8a8_glu_symbol" >&2
    exit 1
  }
done
if [ "${PLOW_BUILD_PFATTN_HD256_BKV64:-0}" = 1 ]; then
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia \
    -o "$gemma_out/interp_sm90a_pfattn_hd256_bkv64.cubin" \
    runtime/nvidia/interp_sm90a_pfattn_hd256_bkv64.cu
fi
if [ "${PLOW_BUILD_PFATTN_HD256_BKV32:-$gemma_bf16}" = 1 ]; then
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia \
    -o "$gemma_out/interp_sm90a_pfattn_hd256_bkv32.cubin" \
    runtime/nvidia/interp_sm90a_pfattn_hd256_bkv32.cu
fi
if [ "${PLOW_BUILD_PFATTN_HD256_GQA2_BKV32:-$gemma_bf16}" = 1 ]; then
  gemma_gqa2_log=$(mktemp)
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia \
    -DPLOW_NV_GQA2_ROLE_BKV="${PLOW_BUILD_PFATTN_GQA2_BKV:-32}" \
    -o "$gemma_out/interp_sm90a_pfattn_hd256_gqa2_bkv32.cubin" \
    runtime/nvidia/interp_sm90a_pfattn_hd256_gqa2_bkv32.cu 2> >(tee "$gemma_gqa2_log" >&2)
  if grep -Eq '[1-9][0-9]* bytes (stack frame|spill stores|spill loads)' "$gemma_gqa2_log"; then
    echo 'paired HD256/GQA2 object uses stack or spills.' >&2
    rm -f "$gemma_gqa2_log"
    exit 1
  fi
  rm -f "$gemma_gqa2_log"
  gemma_gqa2_symbols=$(/usr/local/cuda/bin/cuobjdump -symbols \
    "$gemma_out/interp_sm90a_pfattn_hd256_gqa2_bkv32.cubin")
  for gemma_gqa2_symbol in \
    plow_sm90a_pfattn_hd256_gqa2_bkv32 \
    plow_sm90a_pfattn_hd256_gqa2_bkv32_direct \
    plow_attention_sm90_hd256_gqa2_bkv32_abi \
    plow_attention_direct_entry; do
    grep -q "$gemma_gqa2_symbol" <<<"$gemma_gqa2_symbols" || {
      echo "missing paired HD256/GQA2 symbol: $gemma_gqa2_symbol" >&2
      exit 1
    }
  done
fi
if [ "$gemma_masked" = 1 ]; then
  pfattn_wg=${PLOW_BUILD_PFATTN_WG:-1}
  pfattn_kv16=${PLOW_BUILD_PFATTN_KV16:-0}
  pfattn_kv64=${PLOW_BUILD_PFATTN_KV64:-0}
  pfattn_qk_unroll=${PLOW_BUILD_PFATTN_QK_UNROLL:-4}
  pfattn_tma=${PLOW_BUILD_PFATTN_TMA:-$pfattn_wg}
  if [ "$pfattn_wg" = 0 ] && { [ "$pfattn_kv16" != 0 ] || [ "$pfattn_kv64" != 0 ]; }; then
    echo 'BQ32/BKV16 px4 requires PLOW_BUILD_PFATTN_KV16=0 and PLOW_BUILD_PFATTN_KV64=0.' >&2
    exit 1
  fi
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia \
    -DPLOW_NV_FA512_WG="$pfattn_wg" -DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_WPR=1 \
    -DPLOW_NV_FA_TMA="$pfattn_tma" \
    -DPLOW_NV_FA_QK_UNROLL="$pfattn_qk_unroll" \
    -DPLOW_NV_FA512_KV16="$pfattn_kv16" \
    -DPLOW_NV_FA512_KV64="$pfattn_kv64" \
    -DPLOW_NV_FA512_N_SPLIT="${PLOW_BUILD_PFATTN_N_SPLIT:-1}" \
    -DPLOW_NV_FA512_FIXED_HEADS="${PLOW_BUILD_PFATTN_FIXED_HEADS:-0}" \
    -DPLOW_NV_PACKED_REQUEST=1 -DPLOW_NV_PACKED_FA_WGMMA=1 -DPLOW_NV_PACKED_FA_TMA=1 \
    -DPLOW_NV_MASKED_PADDING=1 -o "$gemma_out/interp_sm90a_pfattn_hd512.cubin" \
    runtime/nvidia/interp_sm90a_pfattn_hd512.cu
fi
if [ "${PLOW_BUILD_PFATTN_HD512_PX4_BQ64:-0}" = 1 ]; then
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia \
    -DPLOW_NV_FA512_PX4_BQ64=1 -DPLOW_NV_FA512_WG=0 -DPLOW_NV_FA_TMA=1 \
    -DPLOW_NV_FA_TMA_ROW_WARP=1 -DPLOW_NV_FA_TMA_DESC=1 -DPLOW_NV_FA_SCORE_SWIZZLE=1 \
    -DPLOW_NV_FA_CTA_SNAKE="${PLOW_BUILD_PFATTN_CTA_SNAKE:-1}" \
    -DPLOW_NV_PACKED_REQUEST=1 -DPLOW_NV_MASKED_PADDING=1 \
    -o "$gemma_out/interp_sm90a_pfattn_hd512_px4_bq64.cubin" \
    runtime/nvidia/interp_sm90a_pfattn_hd512.cu
fi
# Glue kernels of the cuBLASLt grouped-GEMM MoE prefill route (PLOW_MOE_PF_LT): only for a packet
# that carries the grouped expert GEMMs.
if [ -n "${PLOW_CUBIN_CONFIG:-}" ] &&
   grep -qx '#define PLOW_PACKET_HAS_MOE_GROUP_GLU_GEMMA_PF 1' "$PLOW_CUBIN_CONFIG"; then
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
    -std=c++17 -arch=sm_90a -O3 -cubin -Xptxas=-v -I runtime/common -I runtime/nvidia \
    -o "$gemma_out/interp_sm90a_moe_lt.cubin" runtime/nvidia/moe_lt_sm90.cu
fi
