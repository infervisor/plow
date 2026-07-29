#!/usr/bin/env bash
# INTERLEAVED A/B for GLM_LINEAR_FP8 on a **STACKED** blob and the CURRENT interpreter.
#
# The twin of `scripts/glm52_linfp8_run.sh`, on the blobs `glm52_linfp8_stacked_ab.sh` emits. That
# script could only measure the knob DECODE-ONLY, because `declare_glm_rows` refused it on any emit
# carrying prefill buckets. `GemmFp8Blk` (107) removed that refusal, so this measures the knob on
# the configuration that would actually ship.
#
# knob-contract §6b-STALE is the whole reason this exists as a fresh measurement: the same knob has
# measured −0.05 / +0.39 / −0.44 / −0.31 ms on four successive interpreters against blobs that did
# not change. A recorded number is evidence about the object it ran on and nothing else. Report the
# interpreter this ran against.
#
# amd-bench, not the served endpoint, and that is §0-BENCH-compatible rather than a violation: this
# is an A/B of plow AGAINST ITSELF at ~1% of the token, and §6b-WIDTH measured that a served A/B at
# a lease-sized sample cannot resolve it (its two interleaved CONTROLS differed by 1.77 ms, 4x the
# effect). These numbers must never be placed next to a vLLM number.
#
# Run INSIDE `nix develop`, UNDER a lease, with plowrt built `--features hsa` (without it plowrt
# silently serves CPU-reference garbage). gpulease exports ROCR_ and HIP_VISIBLE_DEVICES to the same
# absolute id and they COMPOSE, so HIP_ is unset here (§0a).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${PLOW_AB_DIR:-/tmp/glmlfp8_stk}"
CKPT="${PLOW_CKPT_Q:-/home/lava/models/GLM-5.2-plow-q}"
# FRESH objects built from THIS branch's runtime source. Measuring a knob against an interpreter
# that is not the one under test is exactly the §6b-STALE error.
OBJS="${PLOW_HSACO:-$REPO/build-amd/lfp8-stk-objs}"
RT="$REPO/target/release/plowrt"
STEPS="${STEPS:-65}"
CTX="${CTX:-1024}"
TP="${TP:-4}"
FOLDS="${FOLDS:-3}"
# Fold numbering starts here. A SECOND LEASE must set FOLD0 past the first lease's last fold, or it
# silently overwrites `out/t.<arm>.<n>.txt` and the pooled sample is half the size it looks. The
# recorded methodology for this knob is n=6 over two leases precisely because n=3 could not separate
# the effect from the fold noise (`glm52-linear-fp8-reeval.md` §3.2, sd 0.349 ms on the delta).
FOLD0="${FOLD0:-1}"
ARMS="${ARMS:-stk_base stk_lfp8}"
PROMPT="${PROMPT:-100,264,6722,315,9822,374}"

unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$D/out"

for f in $(seq "$FOLD0" $((FOLD0 + FOLDS - 1))); do
  echo "########## fold $f ##########"
  for arm in $ARMS; do
    echo "--- $arm (timing, ctx=$CTX) ---"
    "$RT" amd-bench --blob "$D/$arm.pkt" --hsaco "$OBJS" --checkpoint "$CKPT" \
        --steps "$STEPS" --ctx "$CTX" --tp "$TP" 2>&1 \
      | tee "$D/out/t.$arm.$f.txt" \
      | grep -Ei 'ms/token|token-identical|DISAGREE|error|panic' | sed "s/^/[t.$arm.$f] /"
  done
done

# Cross-rank agreement, fold 1 only. NOT a precision gate: greedy decode on this checkpoint forks
# within 3 tokens between every arm INCLUDING the bf16 control, so the token stream carries no
# signal about a weight-encoding change (`glm52-moe-tail-ab.md` §3.2). What it DOES prove is that
# all 4 ranks agreed at every step, which a skipped or mis-shaped collective would fail.
for arm in $ARMS; do
  echo "--- $arm (cross-rank agreement) ---"
  "$RT" amd-bench --blob "$D/$arm.pkt" --hsaco "$OBJS" --checkpoint "$CKPT" \
      --prompt "$PROMPT" --steps 24 --tp "$TP" 2>&1 \
    | tee "$D/out/id.$arm.txt" \
    | grep -Ei 'ms/token|token-identical|DISAGREE|^  \[|error|panic' | sed "s/^/[id.$arm] /"
done

echo "===== summary ====="
grep -H 'ms/token' "$D"/out/t.*.txt || true
