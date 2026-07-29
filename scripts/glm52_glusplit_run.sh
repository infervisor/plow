#!/usr/bin/env bash
# INTERLEAVED A/B for the GLM-5.2 shared gate/up arm (knob-contract §6b-STALE: a knob measured
# against one interpreter does not transfer, so every arm runs against its own CONTEMPORANEOUS
# control, in the same lease, alternating).
#
#   base          bf16 shared expert, GemvGlu (19)                — the shipped default
#   linfp8_old    GLM_LINEAR_FP8=1, DenseGluFp8Blk (47)           — the +0.39 ms regression
#   linfp8_split  GLM_LINEAR_FP8=1, 2x GemvFp8Blk (44) + Glu (5)  — the change under test
#
# ALL THREE load the SAME checkpoint dir (`-q` symlinks the base shards and adds the `.weight_fp8`
# ones), so the page cache is hot across arms and the load is not part of the comparison.
#
# TWO invocations per arm, because they answer different questions:
#   * timing   — no `--prompt`: decode at ctx=1024, and the run prints how many of the `steps`
#                had ranks DISAGREE (a rank that skipped a collective still emits fluent ids).
#   * identity — with `--prompt`: prefill + greedy decode, `agree()` HARD-FAILS on any step where
#                the ranks differ, and the stream is compared to the §6g-KNOBS reference. The fp8
#                arms change PRECISION, so they are read against the oracle tolerance, not required
#                to be bit-identical to bf16. Fold 1 only — identity does not vary run to run.
#
# Run INSIDE `nix develop`, UNDER a lease, with `--features hsa` built in (without it plowrt serves
# CPU-reference garbage). gpulease exports ROCR_ and HIP_VISIBLE_DEVICES to the same absolute id and
# they COMPOSE, so HIP_ is unset here (§0a).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${PLOW_AB_DIR:-/tmp/glmab}"
CKPT="${PLOW_CKPT_Q:-/home/lava/models/GLM-5.2-plow-q}"
# hsaco-abi144 + the two interp_prefill_mla objects it lacks. GLM's DECODE program uses
# `MlaMergeFold`, so `PrefillArm::detect` classifies even a decode-only packet as `Mla` and the
# loader refuses to start without them (`scripts/build_glm52_mla_pf_obj.sh`). Never dispatched.
OBJS="${PLOW_HSACO:-$REPO/build-amd/glusplit-objs}"
RT="$REPO/target/release/plowrt"
STEPS="${STEPS:-65}"
CTX="${CTX:-1024}"
TP="${TP:-4}"
FOLDS="${FOLDS:-3}"
ARMS="${ARMS:-base linfp8_old linfp8_split}"
# §6g reference prompt + the bf16 reference stream it produces:
#   264 5777 9125 1948 498 323 279 6372 315 264 3162 2025 429 6147 498 311 653 3654 2513 429 ...
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

for arm in $ARMS; do
  echo "--- $arm (token identity) ---"
  "$RT" amd-bench --blob "$D/$arm.pkt" --hsaco "$OBJS" --checkpoint "$CKPT" \
      --prompt "$PROMPT" --steps 24 --tp "$TP" 2>&1 \
    | tee "$D/out/id.$arm.txt" \
    | grep -Ei 'ms/token|token-identical|DISAGREE|^  \[|error|panic' | sed "s/^/[id.$arm] /"
done

echo "===== summary ====="
grep -H 'ms/token' "$D"/out/t.*.txt || true
