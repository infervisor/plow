# glm-5.3-flash — gfx950-mi350x-tp8-fp8

Status: **validated**. Reference: `infervisor/glm-5.3-flash:gfx950-mi350x-tp8-fp8`.

The command blocks below are generated from the recipe TOML by
`scripts/render_recipe.py`; edit the TOML, not the blocks. Prose outside
the markers is hand-written.

## 1. Prepare the checkpoint

<!-- plow:recipe:prepare -->
```bash
nix develop .#quantize --command python3 scripts/glm53_prep.py \
  --model $CKPT --out $WORK/glm53-flash-prepped
# produces: zz-plow-glm53-derived.safetensors
```
<!-- /plow:recipe:prepare -->

## 2. Build the interpreter objects

<!-- plow:recipe:objects -->
```bash
nix develop --command env \
  FA_MLA_PF2_DEFER=1 \
  JOBS=32 \
  PLOW_DECODE_BATCH=128 \
  PLOW_DECODE_TIERS=1,8 \
  PLOW_GEMV_MM=4 \
  PLOW_GEMV_WALK=1 \
  PLOW_K3_PF_XR_SCHED=aiter \
  PLOW_KDA_CARRY_KTR16=1 \
  PLOW_KDA_CARRY_MULTI=1 \
  PLOW_KDA_CARRY_PIPE=1 \
  PLOW_KDA_CARRY_SPLIT=4 \
  PLOW_KDA_DECODE_LANES=1 \
  PLOW_MLA_DEC_HPACK=1 \
  PLOW_MLA_PF2_DPP=1 \
  PLOW_MLA_PF2_PCVT=1 \
  PLOW_MLA_PF_DENSE=1 \
  PLOW_XR_TAG_SLOT=90112 \
  scripts/build_gfx950.sh \
  $OBJDIR
```
<!-- /plow:recipe:objects -->

## 3. Emit the packet

<!-- plow:recipe:emit -->
```bash
nix develop --command env \
  GLM_SHARD_HEAD=1 \
  PLOW_DECODE_BATCH_LADDER=1,8,32,64,128 \
  PLOW_GLM53_GEMM_LT_DEC=q_a,kv_a,q_absorb,mla_o,kda_o,kda_qkv,kda_gfa,kda_gb,kda_b,lm_head \
  PLOW_GLM53_GEMM_LT_DEC_RUNGS=8,32,64,128 \
  PLOW_GLM53_L2_PLACE=1 \
  PLOW_GLM53_MLA_HPACK_GATHER=1 \
  PLOW_GLM53_MLA_HPACK_ROWS=32 \
  PLOW_GLM53_PACKED_LEAN_KDA=1 \
  PLOW_GLM53_SP_KDA_NORM=1 \
  PLOW_GLM53_SP_ROUTER=1 \
  PLOW_GLM_DECODE_GEMM_ROWS=32 \
  PLOW_GLM_DEC_SPLITK=16 \
  PLOW_GLM_DEC_SPLITK_2P=1 \
  PLOW_GLM_DSA=1 \
  PLOW_GLM_DSA_PF=1 \
  PLOW_GLM_HC_AITER=1 \
  PLOW_GLM_HC_FOLD_MLA=1 \
  PLOW_GLM_HC_FUSE_DEC=1 \
  PLOW_GLM_HC_FUSE_R1=1 \
  PLOW_GLM_HC_FUSE_VARIANT=3 \
  PLOW_GLM_HC_MIX_RAW=1 \
  PLOW_GLM_HC_NORM_FOLD=1 \
  PLOW_GLM_HC_POST_FOLD=1 \
  PLOW_GLM_HC_PRE_TAIL=1 \
  PLOW_GLM_KDA_PROJ_GROUP=1 \
  PLOW_GLM_KDA_QKVG_R1=1 \
  PLOW_GLM_MOE_AITER=1 \
  PLOW_GLM_MOE_RESIDENT=1 \
  PLOW_GLM_MOE_SHARED_FOLD=1 \
  PLOW_GLM_PF_NS=2 \
  PLOW_GLM_ROUTER_OVERLAP=2 \
  PLOW_GLM_SEQ_PAR=1 \
  PLOW_GLM_SEQ_PAR_PROJ=0 \
  PLOW_KDA_DECODE_FUSED_ARM=1 \
  PLOW_KDA_FB_FOLD=1 \
  PLOW_KDA_PF_GROUP_GEMM=1 \
  PLOW_MLA_PF_V2=1 \
  PLOW_MLA_PREFILL=full:128,512,2048,8192 \
  PLOW_MOE_DEC_W8A8=1 \
  PLOW_MOE_FP8_PF_LEAN=1 \
  PLOW_MOE_FP8_PF_PACKED=1 \
  PLOW_MOE_PF_W8A8=1 \
  PLOW_XR_DEC_CUS=64 \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --emit devblob \
  --arch gfx950 \
  --gpu MI350X \
  --num-gpus 8 \
  --parallel tp \
  --max-ctx 65536 \
  --n-cu 256 \
  --out $ASSETS
```
<!-- /plow:recipe:emit -->

## 4. Gates

<!-- plow:recipe:gates -->
```bash
# smoke
scripts/serve_test/smoke.sh

# llm_fp32_ref
scripts/llm/fp32_ref_gate.py
# expect: PASS

# needle-long-context
scripts/glm53_needle_probe.py
# expect: 36/36
```
<!-- /plow:recipe:gates -->

## 5. Serve

<!-- plow:recipe:serve -->
```bash
# From the distribution — resolves the variant for this machine:
plowrt load infervisor/glm-5.3-flash:gfx950-mi350x-tp8-fp8
plowrt serve --model infervisor/glm-5.3-flash:gfx950-mi350x-tp8-fp8 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_AMD_DENSE_NATIVE=1 \
  PLOW_AMD_KARG_CHAIN_FLUSH=1 \
  PLOW_AMD_LT_CONCURRENT=1 \
  PLOW_AMD_LT_FUSE=1 \
  PLOW_AMD_MOE_DEC_TILES=1 \
  PLOW_AMD_PACKED_LEAN_KDA_BATCH=1 \
  PLOW_AMD_PACKED_SEG_MAJOR=1 \
  PLOW_AMD_ROUTER_FUSED=1 \
  PLOW_AMD_ROUTER_NATIVE=1 \
  PLOW_AMD_SNAP_KERNEL=1 \
  PLOW_AMD_XR_NATIVE=131072 \
  PLOW_DECODE_MIN_RUNG=1 \
  PLOW_GLM_ROWBAND=1 \
  PLOW_KDA_CARRY_SPLIT=4 \
  PLOW_MLA_PF_DENSE=1 \
  PLOW_PF_BACK_TO_BACK=1 \
  PLOW_PF_HOLD=8192 \
  PLOW_PF_INTERLEAVE=8192 \
  PLOW_PREFIX_ARM_END=1 \
  PLOW_STATE_CLEAR_DEVICE=1 \
  PLOW_TP_AGREE_EVERY=1 \
  PLOW_TP_PAR_HOST=1 \
  PLOW_TP_RANK_POOL=1 \
  PLOW_XR_TAG_SLOT=90112 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080
```
<!-- /plow:recipe:serve -->
