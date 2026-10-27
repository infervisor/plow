# muse-spark-open — gfx950-mi350x-tp8-32k-mxfp4

Status: **emits**. Reference: `infervisor/muse-spark-open:gfx950-mi350x-tp8-32k-mxfp4`.

The command blocks below are generated from the recipe TOML by
`scripts/render_recipe.py`; edit the TOML, not the blocks. Prose outside
the markers is hand-written.

## 1. Prepare the checkpoint

<!-- plow:recipe:prepare -->
```bash
# This model needs no preparation step.
```
<!-- /plow:recipe:prepare -->

## 2. Build the interpreter objects

<!-- plow:recipe:objects -->
```bash
nix develop --command env \
  PLOW_FA_MERGE_LDS=1 \
  PLOW_MUSE_CMB_ILP=1 \
  PLOW_XR_TAG_SLOT=90112 \
  scripts/build_gfx950.sh \
  $OBJDIR
```
<!-- /plow:recipe:objects -->

## 3. Emit the packet

<!-- plow:recipe:emit -->
```bash
nix develop --command env \
  PLOW_DECODE_BATCH_LADDER=1,8,32 \
  PLOW_MUSE_NGN=1 \
  PLOW_MUSE_QKVG=1 \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --emit devblob \
  --arch gfx950 \
  --gpu MI350X \
  --num-gpus 8 \
  --parallel tp \
  --max-ctx 32768 \
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
plowrt load infervisor/muse-spark-open:gfx950-mi350x-tp8-32k-mxfp4
plowrt serve --model infervisor/muse-spark-open:gfx950-mi350x-tp8-32k-mxfp4 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_DECODE_MIN_RUNG=1 \
  PLOW_XR_TAG_SLOT=90112 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080
```
<!-- /plow:recipe:serve -->
