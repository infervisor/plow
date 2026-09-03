#!/usr/bin/env bash
# Emit the TWO **STACKED** GLM-5.2 TP4 blobs of the GLM_LINEAR_FP8 re-evaluation.
#
# WHY STACKED, AND WHY THIS IS A DIFFERENT EXPERIMENT FROM `glm52_linfp8_ab.sh`.
# That script emits DECODE-ONLY blobs, and it had to: `declare_glm_rows` REFUSED the knob on any
# emit carrying prefill buckets, because the prefill emitters put a bf16 `Gemm` on the fp8 handles.
# Its own header says so — "these blobs measure what it WOULD be worth if a `GemmFp8Blk` existed".
# `GemmFp8Blk` (107) now exists and both prefill emitters route to it, so the knob can be measured
# on the blob that would actually SHIP: `scripts/rebench_emit_glm.sh`'s stacked configuration.
#
# The arms are the shipping decode knobs plus the prefill ladder, with and without the knob:
#   stk_base   bf16 checkpoint, the shipping stacked emit         (the contemporaneous control)
#   stk_lfp8   -q checkpoint + GLM_LINEAR_FP8=1                   (o_proj + shared expert at 1 B/elt)
#
# Both arms load the SAME checkpoint dir at run time: `-q` symlinks the base shards and ADDS the
# `.weight_fp8` / `.weight_scale_inv` ones, so the page cache is hot across arms and the 4-minute
# load is not part of the comparison. The bf16 arm simply never binds the extra tensors.
#
# ONE pair, not the four of the decode-only script, and deliberately: that script's second axis was
# GLM_MOE_CORESIDENT, which exists to ask whether the shared expert is already hidden. The stacked
# question is narrower — does the knob pay on the SHIPPING configuration now that it can be set at
# all — so it gets the shipping configuration and its own control.
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

# EXACTLY `scripts/rebench_emit_glm.sh`'s shipping configuration. Kept as literals rather than by
# sourcing it because that script `exec`s plowc with one output path; the two must not drift, so if
# the ladder there moves this must move with it (and the tile campaign with both — see its header).
export GLM_FULL=1 PLOW_FP8=1
export PLOW_MLA_PREFILL="${PLOW_MLA_PREFILL:-full:512,2048,8192,32768}"
common=(--emit devblob --max-ctx "$CTX" --n-cu 256 --num-gpus 4)

emit() { # name  ckpt  env...
  local name="$1" ckpt="$2"; shift 2
  echo "== $name  ($*)"
  rm -f "$OUT/build.json" "$OUT/$name.build.json"
  env -u GLM_LINEAR_FP8 -u GLM_SHARED_GLU_SPLIT \
      GLM_SHARD_HEAD=1 GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 \
      "$@" ./target/release/plowc --hf-dir "$ckpt" "${common[@]}" --out "$OUT/$name.pkt"
  test -s "$OUT/build.json" || {
    echo "FAIL: $name emit did not produce $OUT/build.json" >&2
    exit 1
  }
  mv "$OUT/build.json" "$OUT/$name.build.json"
}

emit stk_base "$BF16"
emit stk_lfp8 "$FP8" GLM_LINEAR_FP8=1

ls -la "$OUT"/*.pkt "$OUT"/*.build.json
