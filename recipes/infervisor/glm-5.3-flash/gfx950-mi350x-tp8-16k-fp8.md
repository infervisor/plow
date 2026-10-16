# glm-5.3-flash — gfx950-mi350x-tp8-16k-fp8

Status: **emits**. Reference: `infervisor/glm-5.3-flash:gfx950-mi350x-tp8-16k-fp8`.

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
  JOBS=32 \
  PLOW_DECODE_BATCH=1 \
  scripts/build_gfx950.sh \
  $OBJDIR
```
<!-- /plow:recipe:objects -->

## 3. Emit the packet

<!-- plow:recipe:emit -->
```bash
nix develop --command env \
  PLOW_DECODE_BATCH_LADDER=1 \
  PLOW_GLM_DSA=0 \
  PLOW_MLA_PREFILL=full:128,512,2048,8192 \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --emit devblob \
  --arch gfx950 \
  --gpu MI350X \
  --num-gpus 8 \
  --parallel tp \
  --max-ctx 16384 \
  --n-cu 256 \
  --out $ASSETS
```
<!-- /plow:recipe:emit -->

## 4. Gates

<!-- plow:recipe:gates -->
```bash
# smoke
scripts/serve_test/smoke.sh
```
<!-- /plow:recipe:gates -->

## 5. Serve

<!-- plow:recipe:serve -->
```bash
# From the distribution — resolves the variant for this machine:
plowrt load infervisor/glm-5.3-flash:gfx950-mi350x-tp8-16k-fp8
plowrt serve --model infervisor/glm-5.3-flash:gfx950-mi350x-tp8-16k-fp8 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_PREFIX_CACHE=0 \
  PLOW_STATE_CLEAR_DEVICE=1 \
  PLOW_TP_AGREE_EVERY=1 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080
```
<!-- /plow:recipe:serve -->
