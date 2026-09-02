#!/usr/bin/env bash
# SCOPE-3 CORRECTNESS GATE for batched decode on gfx950.
#
# (a) B copies of ONE prompt must produce ONE stream.
# (b) B DIFFERENT prompts (different LENGTHS => ragged positions) must each
#     produce what that prompt produces alone on a batch-1 blob.
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT=/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
RT="nix develop --command $WT/target/release/plowrt"

# Four prompts of four different lengths. Gemma ids: 2 = <bos>.
P1="2,106,1645"
P2="2,106,1645,236764,3689"
P3="2,3689,506,7534,529,6427,236761"
P4="2,106,1645,236764"

run() { # <blob-batch> <hsaco-batch> <prompt-spec> <label>
  echo "======== $4"
  $RT amd-bench \
    --blob /home/lava/plow/build-amd/g31b-db$1/model.pkt \
    --hsaco /home/lava/plow/build-amd/hsaco-b$2 --checkpoint "$CKPT" \
    --steps 4 --batched --prompt "$3" 2>&1 \
    | grep -E "slot [0-9]|agree|chain|tpot|Error|error"
}

echo "#### (b) reference: each prompt ALONE on the batch-1 blob"
for P in "$P1" "$P2" "$P3" "$P4"; do run 1 1 "$P" "B=1  prompt=$P"; done

echo
echo "#### (a) B=4, four copies of ONE prompt"
run 4 4 "$P1;$P1;$P1;$P1" "B=4  identical prompts"

echo
echo "#### (b) B=4, four DIFFERENT prompts of DIFFERENT lengths (ragged)"
run 4 4 "$P1;$P2;$P3;$P4" "B=4  ragged prompts"

echo
echo "#### (b) B=8, same four prompts cycled"
run 8 8 "$P1;$P2;$P3;$P4" "B=8  ragged prompts"
