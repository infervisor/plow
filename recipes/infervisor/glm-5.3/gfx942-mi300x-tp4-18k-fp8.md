# glm-5.3 — gfx942-mi300x-tp4-18k-fp8

Status: **emits**. Reference: `infervisor/glm-5.3:gfx942-mi300x-tp4-18k-fp8`.

The command blocks below are generated from the recipe TOML by
`scripts/render_recipe.py`; edit the TOML, not the blocks. Prose outside
the markers is hand-written.

## 1. Prepare the checkpoint

<!-- plow:recipe:prepare -->
```bash
nix develop .#quantize --command python3 scripts/glm52_prep_lite.py \
  --model $CKPT --out $WORK/glm53-lite
# produces: zz-derived-00001.safetensors
```
<!-- /plow:recipe:prepare -->

## 2. Build the interpreter objects

<!-- plow:recipe:objects -->
```bash
nix develop --command env \
  JOBS=8 \
  PLOW_DECODE_TIERS=1,2,4,8 \
  scripts/build_gfx942.sh \
  $OBJDIR

# rung <= 1: a PARTIAL directory, valid only as an override
nix develop --command env \
  PLOW_DECODE_TIER=1 \
  PLOW_ROWS_ONLY==interp_decode \
  scripts/build_gfx942.sh \
  $OBJDIR/lowrung1

# rung <= 2: a PARTIAL directory, valid only as an override
nix develop --command env \
  PLOW_DECODE_TIER=2 \
  PLOW_ROWS_ONLY==interp_decode \
  scripts/build_gfx942.sh \
  $OBJDIR/lowrung2
```
<!-- /plow:recipe:objects -->

## 3. Emit the packet

<!-- plow:recipe:emit -->
```bash
nix develop --command env \
  GLM_FULL=1 \
  GLM_MOE_CORESIDENT=2 \
  GLM_SHARD_HEAD=1 \
  GLM_SHARED_CUS=48 \
  PLOW_DECODE_BATCH_LADDER=1,2,4 \
  PLOW_FP8=1 \
  PLOW_GLM_DSA=0 \
  PLOW_GLM_FUSE_B1=1 \
  PLOW_GLM_FUSE_ROPE=1 \
  PLOW_GLM_FUSE_SEAM=1 \
  PLOW_GLM_PF_NS=2 \
  PLOW_MLA_PF_V2=1 \
  PLOW_MLA_PREFILL=full:128,512,2048,8192 \
  PLOW_MOE_PF_DET=1 \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --emit devblob \
  --arch gfx942 \
  --gpu MI300X \
  --num-gpus 4 \
  --parallel tp \
  --max-ctx 18432 \
  --out $ASSETS
```
<!-- /plow:recipe:emit -->

## 4. Gates

<!-- plow:recipe:gates -->
```bash
# smoke
scripts/glm53_mi300x.sh smoke 18930
```
<!-- /plow:recipe:gates -->

## 5. Serve

<!-- plow:recipe:serve -->
```bash
# From the distribution — resolves the variant for this machine:
plowrt load infervisor/glm-5.3:gfx942-mi300x-tp4-18k-fp8
plowrt serve --model infervisor/glm-5.3:gfx942-mi300x-tp4-18k-fp8 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_L2_PLACE_DISPATCH=1 \
  PLOW_MLA_PF_V2=1 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080

# The rung overrides travel in the bundle; `serve --model` wires them
# automatically, so PLOW_HSACO_LOWRUNG needs no absolute build-host path.
```
<!-- /plow:recipe:serve -->
