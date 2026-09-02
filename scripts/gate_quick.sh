#!/usr/bin/env bash
# One B=4 gate run with the KV-geometry log visible.
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT=/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
RUST_LOG="info,plowrt::exec::amd=debug" nix develop --command "$WT/target/release/plowrt" amd-bench \
  --blob /home/lava/plow/build-amd/g31b-db${B:-4}/model.pkt \
  --hsaco /home/lava/plow/build-amd/hsaco-b${B:-4} --checkpoint "$CKPT" \
  --steps 4 --batched --prompt "${P:-2,106,1645;2,106,1645;2,106,1645;2,106,1645}" 2>&1 \
  | grep -E "rebase readback|KV slot|slot [0-9]|agree|chain|tpot|rror"
