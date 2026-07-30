#!/usr/bin/env bash
# k3_block_sweep.sh — the FAST K3 decode iteration loop: a 5-layer TP8 asset swept over ctx.
#
# WHY A BLOCK AND NOT THE MODEL. A full 93-layer emit + load + 32 steps is ~5 minutes a point,
# almost all of it weight-table allocation and bind. `K3_NLAYERS=5` is the smallest span that
# contains BOTH mixers (4 KDA + 1 MLA, because layer 3 is the first MLA layer), so it exercises the
# ctx-invariant half of a layer AND the ctx-scaling half. It loads in ~1 s and sweeps three context
# points in under two minutes.
#
# WHAT IT CANNOT TELL YOU, and this is the same caveat `tune_decode_block_sweep.sh` carries:
#   1. MAGNITUDE. Block ms does not scale to model ms — there is one embed and one lm_head in both,
#      so the fixed term is amortised over 5 layers instead of 93.
#   2. Anything with FEW INSTANCES PER BLOCK. A block has ~4 MoE layers, so ablating `MoeRouterTopk`
#      or `MoeCombine` here lands inside noise; those need the full network. Measured: base 3.184,
#      router-ablated 3.152, combine-ablated 3.230 ms/token — all one spread apart.
#   3. Register/occupancy differences, if the block instantiates a different arm set.
# It IS reliable for: ctx scaling, and ranking any knob whose effect is per-layer and uniform.
#
#   scripts/k3_block_sweep.sh                      # baseline sweep
#   scripts/k3_block_sweep.sh PLOW_K3_FUSE_A=1     # any emit knob, applied to the emit
#
# Env: PLOW_K3_HSACO (default build-amd/hsaco), PLOW_K3_STEPS (64), PLOW_K3_CTX (8000,16000,32000).
# `sg render` must be OUTSIDE nix; see plans/k3-decode-perf.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CK="${PLOW_K3_CKPT:-$(ls -d "$HOME"/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/*/ | head -1)}"
HS="${PLOW_K3_HSACO:-build-amd/hsaco}"
STEPS="${PLOW_K3_STEPS:-64}"
OUT="${PLOW_K3_OUT:-/home/lava/models/k3blk_sweep}"
IFS=, read -ra CTXS <<< "${PLOW_K3_CTX:-8000,16000,32000}"

echo "=== emit (K3_NLAYERS=5, K3_PREFILL=0, TP8, fp8 KV) ${*:-no extra knobs}"
nix develop --command bash -c \
  "K3_FULL=1 K3_NLAYERS=5 K3_PREFILL=0 PLOW_FP8_KV=1 PLOW_MXFP4=1 $* ./target/release/plowc \
     --hf-dir '$CK' --emit devblob --arch gfx950 --gpu mi350 --num-gpus 8 --parallel tp \
     --max-ctx 32768 --n-cu 256 --out '$OUT'" 2>&1 | grep -aE "emitted 5 layers|decode instructions|ABLATE"

echo "=== sweep"
for C in "${CTXS[@]}"; do
  # `pos` starts at ctx and increments, so ctx must leave room for --steps under max_ctx.
  printf 'ctx=%-6s ' "$C"
  sg render -c "cd '$ROOT' && PLOW_TP_NO_AUDIT=1 nix develop --command ./target/release/plowrt \
     amd-bench --blob '$OUT/model.pkt' --hsaco '$HS' --steps $STEPS --ctx $C --tp 8" 2>&1 \
    | grep -aoE "[0-9.]+ ms/token \([0-9.]+ tok/s\)" || echo FAILED
done
