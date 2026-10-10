# gemma-4-31b — xeon6-amx-bf16

Status: **emits**. Reference: `infervisor/gemma-4-31b:xeon6-amx-bf16`.

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
  PLOW_DENSE_PF_NS_MIN=2 \
  PLOW_MAX_CHUNK=2048 \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --out $ASSETS
```
<!-- /plow:recipe:emit -->

## 4. Gates

<!-- plow:recipe:gates -->
```bash
# gates.python = {env:PYREF}
# llm_fp32_ref
#   reference = {env:GEMMA4_31B_FP32_REF}
#   vllm_capture = {env:GEMMA4_31B_FP32_VLLM}
#   concurrency = 8
```
<!-- /plow:recipe:gates -->

## 5. Serve

<!-- plow:recipe:serve -->
```bash
# From the distribution — resolves the variant for this machine:
plowrt load infervisor/gemma-4-31b:xeon6-amx-bf16
plowrt serve --model infervisor/gemma-4-31b:xeon6-amx-bf16 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_CPU_SRAM=1 \
  PLOW_CPU_WEIGHT_AFFINE=1 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080
```
<!-- /plow:recipe:serve -->
