# chatterbox — sm89-l4-tp1

Status: **emits**. Reference: `infervisor/chatterbox:sm89-l4-tp1`.

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
  PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 \
  PLOW_EMIT_DECODE_CUBLASLT=0 \
  PLOW_EMIT_PREFILL_CUBLASLT=1 \
  PLOW_FUSE_KV_HNR=1 \
  PLOW_MAX_CHUNK=512 \
  PLOW_NO_GLU_FUSE=1 \
  PLOW_PF_LADDER_APPEND=256,384 \
  PLOW_TTS_VOCODER_DIR={out}/prep/s3gen \
  PLOW_VERIFY_BIN={repo}/lean-plow/.lake/build/bin/plow_verify \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --served-name chatterbox \
  --out $ASSETS
```
<!-- /plow:recipe:emit -->

## 4. Gates

<!-- plow:recipe:gates -->
```bash
# gates.python = {env:PYREF}
# tts_cer
#   arms = ['--conc 1 --stream --wav --n 16 --prompt-set chatterbox --voice default', '--conc 8 --stream --wav --n 16 --prompt-set chatterbox --voice default', '--conc 16 --stream --wav --n 32 --prompt-set chatterbox --voice default']
#   cer_median_max = 0.05

# s3gen_rel_l2
#   python = {env:CBX_PY}
#   env = {'PYTHONPATH': ''}
#   rel_l2_max = 0.001
#   timeout_s = 1800
```
<!-- /plow:recipe:gates -->

## 5. Serve

<!-- plow:recipe:serve -->
```bash
# From the distribution — resolves the variant for this machine:
plowrt load infervisor/chatterbox:sm89-l4-tp1
plowrt serve --model infervisor/chatterbox:sm89-l4-tp1 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_VMM_PREFIX=1 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080
```
<!-- /plow:recipe:serve -->
