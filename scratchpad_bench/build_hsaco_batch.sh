#!/usr/bin/env bash
# Build gfx950 code objects at PLOW_DECODE_BATCH = 2,4,8 into separate hsaco dirs.
# MUST run OUTSIDE nix (see knob-contract §0a: nix's glibc breaks system ROCm).
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
for B in 16 32; do
  OUT=/home/lava/plow/build-amd/hsaco-b$B
  echo "######## PLOW_DECODE_BATCH=$B -> $OUT"
  /usr/bin/env -i PATH=/opt/rocm/bin:/usr/bin:/bin HOME=/home/lava \
    PLOW_DECODE_BATCH=$B \
    bash "$WT/scripts/build_gfx950.sh" "$OUT" 2>&1 | tail -30
  echo "rc=$? elf-count=$(ls $OUT/*.elf 2>/dev/null | wc -l)"
done
