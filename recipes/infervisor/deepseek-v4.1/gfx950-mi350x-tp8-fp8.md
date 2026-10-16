# deepseek-v4.1 — gfx950-mi350x-tp8-fp8

Status: **validated**. Reference: `infervisor/deepseek-v4.1:gfx950-mi350x-tp8-fp8`.

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
  JOBS=32 \
  PLOW_MOE_GLU_KW=1 \
  PLOW_MOE_PF_DOWN_SWEEP=1 \
  PLOW_MOE_PREFILL=1 \
  PLOW_MXFP4=1 \
  PLOW_UNISEG=0 \
  scripts/build_gfx950.sh \
  $OBJDIR
```
<!-- /plow:recipe:objects -->

## 3. Emit the packet

<!-- plow:recipe:emit -->
```bash
nix develop --command env \
  PLOW_UNISEG=0 \
  ./target/release/plowc \
  --hf-dir $CKPT \
  --emit devblob \
  --arch gfx950 \
  --gpu MI350X \
  --num-gpus 8 \
  --parallel tp \
  --max-ctx 8192 \
  --n-cu 256 \
  --out $ASSETS
```
<!-- /plow:recipe:emit -->

## 4. Gates

<!-- plow:recipe:gates -->
```bash
# rung_run_e2e
./target/release/examples/rung_run /home/lava/bench/dsv41/pkt/dsv41_tp8_full.pkt /home/lava/bench/dsv41/obj --checkpoint /home/lava/models/DeepSeek-V4.1-Flash --tp 8 --iters 1 --entry act.hc_residual_a --exit act.hc_residual_a
# expect: zero: 0, NaN: 0, Inf: 0
```
<!-- /plow:recipe:gates -->

## 5. Serve

<!-- plow:recipe:serve -->
```bash
# From the distribution — resolves the variant for this machine:
plowrt load infervisor/deepseek-v4.1:gfx950-mi350x-tp8-fp8
plowrt serve --model infervisor/deepseek-v4.1:gfx950-mi350x-tp8-fp8 --port 8080

# Equivalently, from a locally built asset directory:
nix develop --command env \
  PLOW_AMD_KARG_CHAIN_FLUSH=1 \
  PLOW_AMD_PACKED_SEG_MAJOR=1 \
  PLOW_AMD_SNAP_KERNEL=1 \
  PLOW_DECODE_MIN_RUNG=1 \
  PLOW_PF_BACK_TO_BACK=1 \
  PLOW_PF_HOLD=8192 \
  PLOW_PF_INTERLEAVE=8192 \
  PLOW_PREFIX_ARM_END=1 \
  PLOW_STATE_CLEAR_DEVICE=1 \
  PLOW_TP_AGREE_EVERY=1 \
  PLOW_TP_PAR_HOST=1 \
  PLOW_TP_RANK_POOL=1 \
  ./target/release/plowrt serve --assets $ASSETS --port 8080
```
<!-- /plow:recipe:serve -->
