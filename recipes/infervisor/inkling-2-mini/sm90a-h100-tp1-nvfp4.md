# inkling-2-mini — sm90a-h100-tp1-nvfp4

Status: **emits**. Reference: `infervisor/inkling-2-mini:sm90a-h100-tp1-nvfp4`.

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
# This target builds no interpreter objects (CPU has no emit target).
```
<!-- /plow:recipe:objects -->

## 3. Emit the packet

<!-- plow:recipe:emit -->
```bash
nix develop --command env \
  PLOW_UNISEG=0 \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --served-name inkling-2-mini \
  --out $ASSETS
```
<!-- /plow:recipe:emit -->

## 4. Gates

<!-- plow:recipe:gates -->
```bash
# This recipe declares no gates.
```
<!-- /plow:recipe:gates -->

## 5. Serve

<!-- plow:recipe:serve -->
```bash
# From the distribution — resolves the variant for this machine:
plowrt load infervisor/inkling-2-mini:sm90a-h100-tp1-nvfp4
plowrt serve --model infervisor/inkling-2-mini:sm90a-h100-tp1-nvfp4 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_PREFIX_CACHE=1 \
  PLOW_VMM_PREFIX=1 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080
```
<!-- /plow:recipe:serve -->
