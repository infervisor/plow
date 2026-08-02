#!/usr/bin/env bash
# SCOPE-1: does device-level decode throughput SCALE with PLOW_DECODE_BATCH on gfx950?
# One leased GPU, one blob per B, hsaco built at the matching PLOW_GEMV_MM.
# Run under: perf-data/harness/gpulease -n 1 batch-ceiling sg render -c '<this>'
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT=/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
CTXS="${CTXS:-1024 4096}"
STEPS="${STEPS:-64}"
for B in ${BS:-1 2 4 8}; do
  HS=/home/lava/plow/build-amd/hsaco-b$B
  for C in $CTXS; do
    echo "======== B=$B ctx=$C hsaco=$HS"
    nix develop --command "$WT/target/release/plowrt" amd-bench \
      --blob /home/lava/plow/build-amd/g31b-db$B/model.pkt \
      --hsaco "$HS" --checkpoint "$CKPT" \
      --steps "$STEPS" --ctx "$C" --batched 2>&1 | grep -v '^\s*program '
  done
done
