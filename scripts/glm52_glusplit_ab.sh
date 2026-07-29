#!/usr/bin/env bash
# Emit the three GLM-5.2 TP4 decode blobs of the shared-gate/up A/B, with EVERY other knob held
# fixed (knob-contract §6b-STALE: a knob measured against one interpreter does not transfer, so the
# control has to be contemporaneous and byte-comparable):
#
#   base        bf16 shared expert          — GemvGlu (19)             , /home/lava/models/GLM-5.2-plow
#   linfp8_old  GLM_LINEAR_FP8=1            — DenseGluFp8Blk (47)      , -q weight dir
#   linfp8_split GLM_LINEAR_FP8=1 + split   — 2x GemvFp8Blk (44) + Glu , -q weight dir
#
# CPU only, inside `nix develop`. $1 = out dir.
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?out dir}"
BF16="${PLOW_CKPT:-/home/lava/models/GLM-5.2-plow}"
FP8="${PLOW_CKPT_Q:-/home/lava/models/GLM-5.2-plow-q}"
CTX="${GLM_CTX:-4096}"
mkdir -p "$OUT"
cd "$WT"

common=(--emit devblob --max-ctx "$CTX" --n-cu 256 --num-gpus 4)
export GLM_FULL=1 PLOW_FP8=1
export GLM_MOE_CORESIDENT="${GLM_MOE_CORESIDENT:-1}"

echo "== base (bf16 shared expert)"
env -u GLM_LINEAR_FP8 -u GLM_SHARED_GLU_SPLIT \
    ./target/release/plowc --hf-dir "$BF16" "${common[@]}" --out "$OUT/base.pkt"

echo "== linfp8_old (op 47 — the DEFAULT under GLM_LINEAR_FP8)"
env -u GLM_SHARED_GLU_SPLIT GLM_LINEAR_FP8=1 \
    ./target/release/plowc --hf-dir "$FP8" "${common[@]}" --out "$OUT/linfp8_old.pkt"

echo "== linfp8_split (2x op 44 + Glu)"
env GLM_LINEAR_FP8=1 GLM_SHARED_GLU_SPLIT=1 \
    ./target/release/plowc --hf-dir "$FP8" "${common[@]}" --out "$OUT/linfp8_split.pkt"

ls -la "$OUT"
