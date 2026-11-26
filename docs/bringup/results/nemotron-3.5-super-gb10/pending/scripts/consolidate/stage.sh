#!/usr/bin/env bash
# stage.sh: export HEAD + the shared tree's uncommitted changes into a scratch dir, minus the
# hunks of patches given as args (reverse-applied), for committing a gated subset.
set -e
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
S=/home/ifv/plow-scratch/consolidate/stage
rm -rf "$S"; mkdir -p "$S"
git -C "$W" archive HEAD | tar -x -C "$S"
git -C "$W" diff HEAD --binary > /home/ifv/plow-scratch/consolidate/full.patch
cd "$S"
git apply /home/ifv/plow-scratch/consolidate/full.patch
for f in runtime/nvidia/op_flash_pf256.cuh runtime/tests/flashpre_hd256_sm120.cu runtime/ubench/fp4_tile_sm120.cu; do
  cp "$W/$f" "$f"
done
for p in "$@"; do
  patch -p1 -R --no-backup-if-mismatch < "$p"
done
rm -f runtime/ubench/fp4_tile_sm120.cu.orig
echo STAGED
