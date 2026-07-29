#!/usr/bin/env bash
# COHERENCE gate for the stacked GLM_LINEAR_FP8 arm — the check that has to pass BEFORE any ms is
# believed, and the one the MoE confound makes non-optional.
#
# WHY A TIMING A/B CANNOT DOUBLE AS THIS CHECK. On GLM-5.2 an arm with wrong numerics OVER-reports:
# routing is data-dependent, so garbage activations collapse the router's top-k, the expert ops do
# LESS work, and the token gets FASTER. `PLOW_XR_SHUFFLE` captured 45% of a measured "ceiling" while
# DELETING NOTHING. A subtly-wrong fp8 path looks like a win, so speed is not evidence of anything
# until the output is known to be right.
#
# WHY NOT TOKEN IDENTITY. Greedy decode on this checkpoint forks within 3 tokens between every arm
# INCLUDING the bf16 control (`glm52-moe-tail-ab.md` §3.2), so identity carries no signal about a
# weight-encoding change. What DOES carry signal is (a) the text staying fluent and on-topic, which
# a collapsed router or a mis-indexed scale block destroys immediately, and (b) all 4 ranks agreeing
# at every step, which a mis-shaped collective would fail.
#
# THE PROMPT IS 600 TOKENS ON PURPOSE. Below the smallest bucket (512) a stacked blob still has a
# prefill program, but the point is to drive `GemmFp8Blk` over real weights at a real M — a six-token
# prompt does not. It also makes `amd-bench`'s "prefill: N tokens in X ms" line a usable TTFT
# comparison, which is the only place the prefill-side cost of this change shows up at all.
#
# Run INSIDE `nix develop`, UNDER a lease. $1 = A/B dir (must hold stk_base.pkt / stk_lfp8.pkt and
# prompt_ids.txt).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${1:-${PLOW_AB_DIR:-/tmp/glmlfp8_stk}}"
CKPT="${PLOW_CKPT_Q:-/home/lava/models/GLM-5.2-plow-q}"
OBJS="${PLOW_HSACO:-$REPO/build-amd/lfp8-stk-objs}"
RT="$REPO/target/release/plowrt"
TP="${TP:-4}"
STEPS="${STEPS:-48}"
IDS="$(cat "${PROMPT_IDS:-$D/prompt_ids.txt}")"

unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$D/coh"

for arm in stk_base stk_lfp8; do
  echo "--- $arm ---"
  "$RT" amd-bench --blob "$D/$arm.pkt" --hsaco "$OBJS" --checkpoint "$CKPT" \
      --prompt "$IDS" --steps "$STEPS" --tp "$TP" 2>&1 | tee "$D/coh/$arm.txt" \
    | grep -Ei 'prefill:|ms/token|token-identical|DISAGREE|unbound|error|panic' \
    | sed "s/^/[$arm] /"
done

echo "===== generated ids (detokenize with the model tokenizer to read them) ====="
grep -h '^  \[' "$D"/coh/*.txt || true
