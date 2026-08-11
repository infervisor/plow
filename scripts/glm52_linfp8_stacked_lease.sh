#!/usr/bin/env bash
# Lease 4 GPUs (GLM is TP4) and run the STACKED GLM_LINEAR_FP8 A/B. Blocks until 4 are free rather
# than contending — a contended run silently invalidates every number (knob-contract §5).
#
# A LEASE IS NOT ISOLATION ON THIS BOX. It serialises access to the four cards it holds; it does not
# stop another agent's job on the OTHER four from moving the shared power budget. That is not
# hypothetical: the first lease of this A/B ran alongside a `plowrt serve` on cards 0-3 for its whole
# duration, the control arm drifted 25.36 -> 25.88 ms across three folds, and the effect being
# measured is ~0.3 ms. Interleaving the arms INSIDE each fold is what makes that survivable — the
# drift lands on the arm and its control alike — and it is why the statistic to report is the paired
# per-fold delta and never the difference of two medians.
#
# Run a SECOND lease with FOLD0 past the first one's last fold and pool the deltas. n=3 could not
# separate this knob from the fold noise on the decode-only blobs either
# (`glm52-linear-fp8-reeval.md` §3.2 needed n=6).
#
#   scripts/glm52_linfp8_stacked_lease.sh              # folds 1-3
#   FOLD0=4 scripts/glm52_linfp8_stacked_lease.sh      # folds 4-6, pooled with the above
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-21600}"
export FOLD0="${FOLD0:-1}"
exec "$WT/perf-data/tools/gpulease" -n 4 lin-fp8-stacked sg render -c \
  "cd '$WT' && unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES && nix develop -c bash scripts/glm52_linfp8_stacked_run.sh"
