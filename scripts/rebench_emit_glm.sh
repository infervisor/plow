#!/usr/bin/env bash
# Emit the STACKED GLM-5.2 TP4 serving blob for the re-benchmark:
#   prefill bucket ladder + decode   (PLOW_MLA_PREFILL=full:...)   — the TTFT lever
#   vocab-parallel lm_head           (GLM_SHARD_HEAD=1)            — −0.26 ms, bit-identical
#   co-resident shared expert        (GLM_MOE_CORESIDENT=2, GLM_SHARED_CUS=48) — −0.81 ms
#
# NOT set: GLM_GROUP (§6g-KNOBS, +2.88 ms).
#
# NOT set, but no longer REFUSED: GLM_LINEAR_FP8. It used to be structurally impossible here —
# this emit is STACKED (PLOW_MLA_PREFILL below), the knob re-declares o_proj and the three
# shared_experts projections at 1 B/elt, and only the DECODE emitters routed them to a block-fp8
# opcode, so the prefill emitters would have put a bf16 Gemm on fp8 bytes. `GemmFp8Blk` (107) is the
# dense prefill block-fp8 GEMM that was missing; both prefill emitters route to it now and
# `declare_glm_rows` no longer asserts `rows == 1`. It also needs the weight dir
# `scripts/glm52_prep_fp8_linear.py` publishes (PLOW_CKPT=.../GLM-5.2-plow-q), which is why it is
# not simply switched on here — this script's default checkpoint has no `.weight_fp8` shards.
#
# WHETHER IT PAYS IS A SEPARATE QUESTION AND ITS RECORDED VALUE IS NOT EVIDENCE. The same knob has
# measured −0.05 / +0.39 / −0.44 / −0.31 ms on four successive interpreters against blobs that did
# not change (knob-contract §6b-STALE; perf-data/glm52-linear-fp8-reeval.md). Re-derive it against
# the object you are about to ship, with interleaved controls:
#     scripts/glm52_linfp8_stacked_ab.sh  <dir>     # emit the stacked pair
#     scripts/glm52_linfp8_stacked_run.sh           # interleaved A/B under a lease
#
# Runs INSIDE `nix develop` (plowc is nix-linked). No GPU, no lease.
#   $1 out.pkt   $2.. extra plowc args
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?out.pkt}"; shift
CKPT="${PLOW_CKPT:-/home/lava/models/GLM-5.2-plow}"

# SWEEP ON A TRUNCATED MODEL; SPEND FULL-NETWORK RUNS ONLY ON THE WINNER.
#
# This script hardcodes GLM_FULL=1, i.e. all 78 layers, and that makes it the WRONG vehicle for
# searching a knob space. glm52_decode.c:224 states the cost: "the 4-minute 183 GiB/rank weight
# load is the whole cost of a run". Any emit-time knob (PLOW_MLA_NS, PLOW_GLM_GF, the prefill
# ladder, GLM_LINEAR_FP8) needs a fresh blob AND a fresh 4-minute load per arm, so an 8-arm sweep
# is over an hour of pure loading before a single useful number.
#
# Instead, for the SEARCH pass:
#     GLM_FULL=1 GLM_NLAYERS=4 ./target/release/plowc --hf-dir ... --num-gpus 4 ...
# `GLM_NLAYERS` (crates/devgen/src/mla.rs:3569) truncates to the first N layers while KEEPING the
# serving structure and the TP degree. At N=4 the load is 4/78 of 183 GiB/rank — seconds.
#
# Do NOT use the single-layer validation gate (GLM_FULL unset) for this: mla.rs:3860 asserts
# `tp == 1` ("GLM TP sharding is milestone-3"), so it cannot sweep a TP4/TP8 knob at all.
#
# WHAT TRANSFERS AND WHAT DOES NOT: per-layer quantities transfer — a per-layer op time, and any
# trade between two per-layer ops (e.g. flash-decode saving vs O(nsplit) merge growth). What does
# NOT transfer is anything that is a property of the whole 78-layer stream: CU contention, chain
# depth, the CU-0 pileup where narrow ops serialise, and TTFT (prefill over the full stack).
# So the truncated run RANKS the arms; the full run PRICES the winner. A sign disagreement between
# the two is itself a finding, not noise.
export GLM_FULL=1
export PLOW_FP8=1
# LADDER EXTENDED 2026-07-29: was full:128,512,1024,2048, which TOPPED OUT AT 2048.
#
# Two reasons. (1) Launch overhead: prefill cost is 139 ms FIXED + 0.943 ms/row, so a 128k prompt
# on 2048-buckets is 64 launches = ~8.9 s of pure fixed cost; on this ladder its top bucket is
# 32768, i.e. 4 launches = ~0.56 s. (2) GEMM efficiency: the bucket IS the M dimension of every
# prefill GEMM (M = rows in the chunk = batch*seq), and a 2048-row M leaves the wide shapes far
# from their efficient regime.
#
# THE LADDER AND THE TILE CAMPAIGN ARE COUPLED — do not change one alone. Each bucket M generates
# its own set of (M,N,K) tile lookups, and scripts/rebench_tune_gemm.sh must carry exactly those
# shapes or selection silently falls back to the analytical model for the new M values. That is
# not hypothetical: before 2026-07-29 GLM had ZERO measured prefill shapes and every lookup missed.
# After changing this line, re-derive:
#     PLOW_TUNE_DUMP=1 GLM_CTX=131072 PLOW_MLA_PREFILL=<new ladder> ./scripts/rebench_emit_glm.sh /tmp/x.pkt
# and re-sync the GLM block in rebench_tune_gemm.sh from the TUNEDUMP lines.
# CORRECTED: the top rung is 8192, NOT 32768. `MAX_CHUNK = 8192`
# (crates/plowrt/src/exec/amd.rs:718) and `plan_chunks` (:735) filters `b <= MAX_CHUNK`, so any
# bucket above it is DROPPED AT RUNTIME while still costing `act.part` device memory at emit
# (T*top_k*hidden f32 — 3.2 GiB at T=16384, ~6.4 at 32768). scripts/glm52_tpctx_sweep.sh's header
# records a previous harness making exactly this mistake with a 16384 rung; I repeated it at
# 32768 and am undoing it. A ladder rung above MAX_CHUNK is pure cost with zero dispatch.
#
# The launch-overhead argument for a wider top rung still holds, but it is CAPPED by MAX_CHUNK:
# a 128k prompt is 16 chunks of 8192, ~2.2 s of fixed cost, and no ladder change can beat that
# without raising MAX_CHUNK itself (which is bounded by the KV ring: RING >= window + MAX_CHUNK - 1).
# So the rungs below the cap are the only free variable — 4096 is added for a tighter fit.
export PLOW_MLA_PREFILL="${PLOW_MLA_PREFILL:-full:512,2048,4096,8192}"
export GLM_SHARD_HEAD="${GLM_SHARD_HEAD:-1}"
export GLM_MOE_CORESIDENT="${GLM_MOE_CORESIDENT:-2}"
export GLM_SHARED_CUS="${GLM_SHARED_CUS:-48}"

cd "$WT"
echo "PLOW_MLA_PREFILL=$PLOW_MLA_PREFILL GLM_SHARD_HEAD=$GLM_SHARD_HEAD" \
     "GLM_MOE_CORESIDENT=$GLM_MOE_CORESIDENT GLM_SHARED_CUS=$GLM_SHARED_CUS"
exec ./target/release/plowc --hf-dir "$CKPT" --emit devblob \
    --max-ctx "${GLM_CTX:-4096}" --n-cu 256 --num-gpus 4 --out "$OUT" "$@"
