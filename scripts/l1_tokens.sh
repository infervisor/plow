#!/usr/bin/env bash
# TOKEN IDENTITY GATE. flash_merge folds softmax partials: a wrong D-chunk bound
# corrupts tokens SILENTLY, so a perf number without this check is worthless.
#
# Three arms, greedy, same prompt, same checkpoint:
#   base : pre-change objects  + pre-change blob      (the reference)
#   d1   : post-change objects + dsplit=1 blob        (regression: the default path)
#   d8   : post-change objects + dsplit=8 blob        (the widened path)
# All three token lists must be character-for-character equal.
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CK=/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
STEPS="${STEPS:-32}"
# A MEANINGFUL prompt, not random ids. A random-id prompt makes the model emit one
# constant token forever, and a constant stream would compare equal even if the merge
# were subtly wrong — the gate has to be able to fail.
PROMPT=$(python3 "$W/scripts/l1_prompt.py")

one() { # <label> <blob-dir> <hsaco-dir>
  local log; log=$(mktemp)
  "$W/perf-data/tools/gpulease" -n 1 "l1-tok-$1" \
      "$W/target/release/plowrt" amd-bench \
      --blob "$2/model.pkt" --hsaco "$3" --checkpoint "$CK" \
      --prompt "$PROMPT" --steps "$STEPS" >"$log" 2>&1 || { cat "$log"; exit 1; }
  grep -A1 'greedy decode:' "$log" | tail -1 | tr -d ' ' > "/tmp/l1-tok-$1.txt"
  echo "$1: $(head -c 120 /tmp/l1-tok-$1.txt)..."
  rm -f "$log"
}

one base /home/lava/plow/build-amd/l1-basepkt /home/lava/plow/build-amd/l1-base
one d1   /home/lava/plow/build-amd/l1-d1      /home/lava/plow/build-amd/l1-gfx950
one d8   /home/lava/plow/build-amd/l1-d8      /home/lava/plow/build-amd/l1-gfx950

fail=0
cmp -s /tmp/l1-tok-base.txt /tmp/l1-tok-d1.txt && echo "d1 == base  TOKEN-IDENTICAL" || { echo "d1 != base  MISMATCH"; fail=1; }
cmp -s /tmp/l1-tok-base.txt /tmp/l1-tok-d8.txt && echo "d8 == base  TOKEN-IDENTICAL" || { echo "d8 != base  MISMATCH"; fail=1; }
exit $fail
