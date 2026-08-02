#!/usr/bin/env bash
# B=1 code objects from THIS tree, so the B sweep compares same-source objects.
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
/usr/bin/env -i PATH=/opt/rocm/bin:/usr/bin:/bin HOME=/home/lava PLOW_DECODE_BATCH=1 \
  bash "$WT/scripts/build_gfx950.sh" /home/lava/plow/build-amd/hsaco-b1 2>&1 | tail -25
