# kimi-k3 — gfx942-mi325x-tp8-32k-fp8kv-mxfp4

Status: **validated**. Reference: `infervisor/kimi-k3:gfx942-mi325x-tp8-32k-fp8kv-mxfp4`.

The command blocks below are generated from the recipe TOML by
`scripts/render_recipe.py`; edit the TOML, not the blocks. Prose outside
the markers is hand-written.

## 1. Prepare the checkpoint

<!-- plow:recipe:prepare -->
```bash
nix develop .#quantize --command python3 scripts/kimi_k3_tokenizer.py \
  --model $CKPT --out $WORK/k3_tokz --verify
# produces: tokenizer.json

nix develop .#quantize --command python3 scripts/kimi_k3_prep.py \
  --model $CKPT --out $WORK/k3_derived --derived --farm $WORK/k3_farm
# produces: model-idx-derived-00001.safetensors
```
<!-- /plow:recipe:prepare -->

## 2. Build the interpreter objects

<!-- plow:recipe:objects -->
```bash
nix develop --command env \
  JOBS=8 \
  PLOW_DECODE_BATCH=32 \
  PLOW_GEMV_MM=16 \
  PLOW_GEMV_WALK=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  scripts/build_gfx942.sh \
  $OBJDIR

# rung <= 1: a PARTIAL directory, valid only as an override
nix develop --command env \
  JOBS=2 \
  PLOW_DECODE_BATCH=1 \
  PLOW_K3_DECODE_GROUPED=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 \
  scripts/build_gfx942.sh \
  $OBJDIR/lowrung1

# rung <= 2: a PARTIAL directory, valid only as an override
nix develop --command env \
  JOBS=2 \
  PLOW_DECODE_BATCH=2 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 \
  scripts/build_gfx942.sh \
  $OBJDIR/lowrung2

# rung <= 4: a PARTIAL directory, valid only as an override
nix develop --command env \
  JOBS=2 \
  PLOW_DECODE_BATCH=4 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 \
  scripts/build_gfx942.sh \
  $OBJDIR/lowrung4

# rung <= 8: a PARTIAL directory, valid only as an override
nix develop --command env \
  JOBS=2 \
  PLOW_DECODE_BATCH=8 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 \
  scripts/build_gfx942.sh \
  $OBJDIR/lowrung8
```
<!-- /plow:recipe:objects -->

## 3. Emit the packet

<!-- plow:recipe:emit -->
```bash
nix develop --command env \
  K3_FULL=1 \
  PLOW_DECODE_BATCH=32 \
  PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 \
  PLOW_FP8_KV=1 \
  PLOW_GEMV_MM=16 \
  PLOW_GEMV_WALK=1 \
  PLOW_GLM_GEMV_WG=128 \
  PLOW_L2_PLACE=1 \
  PLOW_MLA_PF_V2=1 \
  PLOW_MXFP4=1 \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --emit devblob \
  --arch gfx942 \
  --gpu MI325X \
  --num-gpus 8 \
  --parallel tp \
  --max-ctx 32768 \
  --n-cu 304 \
  --out $ASSETS
```
<!-- /plow:recipe:emit -->

## 4. Gates

<!-- plow:recipe:gates -->
```bash
# tp-logit-equivalence
env PLOW_K3_LAYERS=1,2 scripts/k3_tp_equivalence.sh

# gsm8k
# expect: 197/200
```
<!-- /plow:recipe:gates -->

## 5. Serve

<!-- plow:recipe:serve -->
```bash
# From the distribution — resolves the variant for this machine:
plowrt load infervisor/kimi-k3:gfx942-mi325x-tp8-32k-fp8kv-mxfp4
plowrt serve --model infervisor/kimi-k3:gfx942-mi325x-tp8-32k-fp8kv-mxfp4 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_CTR_DBUF=1 \
  PLOW_L2_PLACE_DISPATCH=1 \
  PLOW_MLA_PF_V2=1 \
  PLOW_STATE_CLEAR_DEVICE=1 \
  PLOW_TP_AUDIT_COMPACT=1 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080

# The rung overrides travel in the bundle; `serve --model` wires them
# automatically, so PLOW_HSACO_LOWRUNG needs no absolute build-host path.
```
<!-- /plow:recipe:serve -->
