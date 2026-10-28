# gemma-4-26b-a4b — sm90a-h100-tp1-fp8

Status: **validated**. Reference: `infervisor/gemma-4-26b-a4b:sm90a-h100-tp1-fp8`.

The command blocks below are generated from the recipe TOML by
`scripts/render_recipe.py`; edit the TOML, not the blocks. Prose outside
the markers is hand-written.

## 1. Prepare the checkpoint

<!-- plow:recipe:prepare -->
```bash
# [spec] drafter gemma4_assistant @ hub 6e5aaaf4; model.safetensors sha256 006c8713b29b046e296c0f0b7c524aa9264ae07ae8b0c3888b9bc062242297ff: re-key it beside the target shards
nix develop --command python3 scripts/gemma4_mtp_rekey.py "$G26_MTP_ASSISTANT" "$CKPT"
nix develop --command python3 scripts/gemma4_mtp_fp8.py "$CKPT"
# produces: $CKPT/model-mtp.safetensors (sha256 9e899298473493a366df2548e5f4398e17237394df3b9715583457718048b314)
```
<!-- /plow:recipe:prepare -->

## 2. Build the interpreter objects

<!-- plow:recipe:objects -->
```bash
nix develop --command env \
  PLOW_BUILD_FATLITE=1 \
  PLOW_BUILD_FA_GQA2_PAIR=0 \
  PLOW_BUILD_FP8KV_GEMM=1 \
  PLOW_BUILD_GEN_KERNELS=attn_pf_hd256_sliding_fp8kv,attn_pf_hd512_fp8kv \
  PLOW_BUILD_PFATTN_HD256_BKV32=0 \
  PLOW_BUILD_PFATTN_HD256_GQA2_BKV32=0 \
  PLOW_BUILD_PFATTN_HD512_PX4_BQ64=0 \
  PLOW_BUILD_W8A8=1 \
  scripts/build_sm90a_gemma4_segments.sh \
  $OBJDIR
```
<!-- /plow:recipe:objects -->

## 3. Emit the packet

<!-- plow:recipe:emit -->
```bash
nix develop --command env \
  NVCC_APPEND_FLAGS=-DPLOW_MOE_FP8_DEDUP=1 -DPLOW_NV_FA_RG8=2 -DPLOW_NV_RAW_NARROW=1 -DPLOW_NV_INST_PRELOAD=1 -DPLOW_NV_ROUTER_V8=1 -DPLOW_NV_TOPK_V4=1 -DPLOW_NV_ROUTER_V9=1 -DPLOW_NV_FA_RG8S=1 \
  PLOW_ATTENTION_DECODE_BALANCE_GF=4 \
  PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32,64,128 \
  PLOW_EMIT_DECODE_CUBLASLT=1 \
  PLOW_EMIT_DECODE_CUBLASLT_HEAD=1 \
  PLOW_EMIT_DECODE_CUBLASLT_MIN_ROWS=32 \
  PLOW_EMIT_MOE_PF_LT=1 \
  PLOW_EMIT_PACKED_PREFILL=1 \
  PLOW_EMIT_PREFILL_CUBLASLT=1 \
  PLOW_EMIT_SPAN_VERIFY=1 \
  PLOW_EMIT_SPEC=1 \
  PLOW_FA_MMAQK=3 \
  PLOW_FP8=1 \
  PLOW_FP8_HEAD=1 \
  PLOW_FP8_KV=1 \
  PLOW_FUSE_ARGMAX=1 \
  PLOW_GEMMA_MOE_ROUTER_EXACT=0 \
  PLOW_MAX_CHUNK=4096 \
  PLOW_MAX_REQUEST_CHUNK=4096 \
  PLOW_NO_GLU_FUSE=1 \
  PLOW_PF_LADDER_APPEND=256,1088,1152,2112,4160,4224 \
  PLOW_SEG_FA256_GQA2=1 \
  PLOW_SEG_FA512=1 \
  PLOW_SEG_PURE_GEMM=fp8 \
  PLOW_SLIDING_NS_CAP=1 \
  PLOW_SLIDING_NS_GRID=1 \
  PLOW_SPEC_DRAFT_DIR={env:G26_MTP_ASSISTANT} \
  PLOW_SPEC_DRAFT_FP8=all \
  PLOW_SPEC_KMAX=7 \
  PLOW_SPEC_ROWS=1,2,4,8,16 \
  PLOW_SPEC_SHARD_SHA256=9e899298473493a366df2548e5f4398e17237394df3b9715583457718048b314 \
  PLOW_TMA_GEMM=1 \
  PLOW_W8A8=1 \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --segmented --served-name \
  gemma-4-26b-a4b-it-fp8 \
  --out $ASSETS
```
<!-- /plow:recipe:emit -->

## 4. Gates

<!-- plow:recipe:gates -->
```bash
# gates.python = {env:PYREF}
# llm_fp32_ref
#   reference = {env:G26_FP8_FP32_REF}
#   vllm_capture = {env:G26_FP8_FP32_VLLM}
#   concurrency = 16
#   timeout_s = 2400
#   kl_over_slack = 0.02
#   cont_decisive_drop_max = 0.045
#   top1_drop_max = 0.015

# llm_fp32_ref_long
#   reference = {env:G26_FP8_FP32_REF_LONG}
#   vllm_capture = {env:G26_FP8_FP32_VLLM_LONG}
#   concurrency = 4
#   timeout_s = 7200
#   cont_decisive_drop_max = 0.125
#   kl_over_slack = 0.01
```
<!-- /plow:recipe:gates -->

## 5. Serve

<!-- plow:recipe:serve -->
```bash
# From the distribution — resolves the variant for this machine:
plowrt load infervisor/gemma-4-26b-a4b:sm90a-h100-tp1-fp8
plowrt serve --model infervisor/gemma-4-26b-a4b:sm90a-h100-tp1-fp8 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_DECODE_MAX_RUNG=128 \
  PLOW_DECODE_PIPELINE=1 \
  PLOW_KV_POOL_MIB=2048 \
  PLOW_LT_FP8_FAST_ACCUM_MAX_K=8192 \
  PLOW_LT_RUNG_ALGOS=true \
  PLOW_MULTISTEP=0 \
  PLOW_PF_ADMIT_OVERLAP=1 \
  PLOW_PF_INTERLEAVE=0 \
  PLOW_PF_ITL_BUDGET_MS=60 \
  PLOW_PF_ITL_FORCE=0 \
  PLOW_PF_SEG_FA512=1 \
  PLOW_PF_SEG_PURE=fp8 \
  PLOW_PREFIX_CACHE=1 \
  PLOW_SPEC=1 \
  PLOW_TERMINAL_LT=0 \
  PLOW_VMM_LIVE_RINGS=1 \
  PLOW_VMM_PREFIX=1 \
  PLOW_VMM_RING_PARK=8 \
  PLOW_VMM_STRICT_PUBLISH=1 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080
```
<!-- /plow:recipe:serve -->
