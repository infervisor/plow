#!/usr/bin/env bash
# stage_files.sh FILE...: export HEAD into the stage dir and overlay the shared worktree's copy of FILE...
set -e
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
S=/home/ifv/plow-scratch/consolidate/stage
rm -rf "$S"; mkdir -p "$S"
git -C "$W" archive HEAD | tar -x -C "$S"
for f in "$@"; do
  mkdir -p "$S/$(dirname "$f")"
  cp "$W/$f" "$S/$f"
done
echo STAGED "$@"
