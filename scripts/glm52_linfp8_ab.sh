#!/usr/bin/env bash
# Emit the FOUR decode-only GLM-5.2 TP4 blobs of the GLM_LINEAR_FP8 re-evaluation.
#
# Why four and not two (knob-contract §6b-STALE): the same knob has measured −1.09 / +1.32 / −0.79
# on three successive interpreters, and −0.05 / +0.39 / −0.44 on three successive objects, each
# time CORRECT against its own contemporaneous control. So a single pair answers "does it pay HERE",
# not "does it pay". Two configurations, each with its OWN bf16 control, separate the two questions:
#
#   c1_*    GLM_MOE_CORESIDENT=1                      — the config `glm52-moe-tail-ab.md` §3.1
#                                                       measured −0.44 ms on. Reproduces or refutes.
#   ship_*  GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48    — the SHIPPING decode knobs
#           GLM_SHARD_HEAD=1                            (`scripts/rebench_emit_glm.sh`). This is the
#                                                       decision-relevant number, and nothing has
#                                                       ever measured the knob against it.
#
# CORESIDENT is not an innocent bystander here: it overlaps the shared expert with the routed
# experts, and the shared expert is three of the four tensors this knob converts. If the shared
# expert is already hidden under CORESIDENT=2 then converting it cannot pay, whatever it does at
# CORESIDENT=1 — that is a mechanism, and this pair is how you tell.
#
# DECODE-ONLY, and that is now a CHOICE rather than a constraint. It used to be enforced:
# `declare_glm_rows` refused GLM_LINEAR_FP8 on any emit carrying prefill buckets, because the
# prefill emitters put a bf16 Gemm on the fp8 handles, so these blobs could only measure what the
# knob WOULD be worth if a `GemmFp8Blk` existed. It exists (opcode 107) and the refusal is gone.
# These four arms stay decode-only because their question is decode-internal — they isolate the knob
# against GLM_MOE_CORESIDENT with no prefill packets in the blob at all. For the SHIPPING question,
# use `scripts/glm52_linfp8_stacked_ab.sh` + `glm52_linfp8_stacked_run.sh`, which emit and measure
# the stacked pair.
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
# PLOW_MLA_PREFILL is deliberately unset for every arm: decode-only, see the header.
unset PLOW_MLA_PREFILL || true

emit() { # name  ckpt  env...
  local name="$1" ckpt="$2"; shift 2
  echo "== $name  ($*)"
  env -u GLM_LINEAR_FP8 -u GLM_SHARED_GLU_SPLIT -u GLM_SHARD_HEAD \
      -u GLM_MOE_CORESIDENT -u GLM_SHARED_CUS -u PLOW_MLA_PREFILL \
      "$@" ./target/release/plowc --hf-dir "$ckpt" "${common[@]}" --out "$OUT/$name.pkt"
}

emit c1_base   "$BF16" GLM_MOE_CORESIDENT=1
emit c1_lfp8   "$FP8"  GLM_MOE_CORESIDENT=1 GLM_LINEAR_FP8=1
emit ship_base "$BF16" GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1
emit ship_lfp8 "$FP8"  GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 GLM_LINEAR_FP8=1

ls -la "$OUT"/*.pkt
