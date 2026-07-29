#!/usr/bin/env bash
# INTERLEAVED A/B for GLM_LINEAR_FP8 on the CURRENT interpreter (knob-contract §6b-STALE).
#
# Four arms, two independent pairs, each pair carrying its OWN contemporaneous bf16 control:
#   c1_base / c1_lfp8      GLM_MOE_CORESIDENT=1                          (the −0.44 ms config)
#   ship_base / ship_lfp8  CORESIDENT=2 SHARED_CUS=48 SHARD_HEAD=1       (the SHIPPING decode knobs)
#
# The arms alternate INSIDE each fold, so a drift in machine state (thermals, a foreign process
# arriving) lands on the arm and its control alike instead of on one of them. Reporting the
# per-fold delta rather than the difference of two medians is what makes that pay off.
#
# ALL FOUR load the SAME checkpoint dir: `-q` symlinks the base shards and ADDS the `.weight_fp8`
# / `.weight_scale_inv` ones, so the page cache is hot across arms and the 4-minute load is not
# part of the comparison. The bf16 arms simply never bind the extra tensors.
#
# Run INSIDE `nix develop`, UNDER a lease, with `--features hsa` built in (without it plowrt
# silently serves CPU-reference garbage). gpulease exports ROCR_ and HIP_VISIBLE_DEVICES to the
# same absolute id and they COMPOSE, so HIP_ is unset here (§0a).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${PLOW_AB_DIR:-/tmp/glmlfp8}"
CKPT="${PLOW_CKPT_Q:-/home/lava/models/GLM-5.2-plow-q}"
# FRESH objects built from THIS branch's runtime source. `build-amd/hsaco-abi144` is 11.5 KB
# smaller and predates two commits that touched runtime/amd — measuring a knob against a stale
# interpreter is the exact error §6b-STALE is about, so it is not used here.
OBJS="${PLOW_HSACO:-$REPO/build-amd/lfp8-objs}"
RT="$REPO/target/release/plowrt"
STEPS="${STEPS:-65}"
CTX="${CTX:-1024}"
TP="${TP:-4}"
FOLDS="${FOLDS:-3}"
ARMS="${ARMS:-c1_base c1_lfp8 ship_base ship_lfp8}"
PROMPT="${PROMPT:-100,264,6722,315,9822,374}"

unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$D/out"

for f in $(seq 1 "$FOLDS"); do
  echo "########## fold $f ##########"
  for arm in $ARMS; do
    echo "--- $arm (timing, ctx=$CTX) ---"
    "$RT" amd-bench --blob "$D/$arm.pkt" --hsaco "$OBJS" --checkpoint "$CKPT" \
        --steps "$STEPS" --ctx "$CTX" --tp "$TP" 2>&1 \
      | tee "$D/out/t.$arm.$f.txt" \
      | grep -Ei 'ms/token|token-identical|DISAGREE|error|panic' | sed "s/^/[t.$arm.$f] /"
  done
done

# Token streams, fold 1 only (identity does not vary run to run). NOT the gate for this change:
# greedy decode on this checkpoint forks within 3 tokens between every arm INCLUDING the bf16
# control (`glm52-moe-tail-ab.md` §3.2). What these DO prove is that all 4 ranks agreed on every
# step, which is the check a skipped collective would fail.
for arm in $ARMS; do
  echo "--- $arm (token identity) ---"
  "$RT" amd-bench --blob "$D/$arm.pkt" --hsaco "$OBJS" --checkpoint "$CKPT" \
      --prompt "$PROMPT" --steps 24 --tp "$TP" 2>&1 \
    | tee "$D/out/id.$arm.txt" \
    | grep -Ei 'ms/token|token-identical|DISAGREE|^  \[|error|panic' | sed "s/^/[id.$arm] /"
done

echo "===== summary ====="
grep -H 'ms/token' "$D"/out/t.*.txt || true
