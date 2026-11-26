#!/usr/bin/env bash
# commit_stage.sh <msgfile>: amend HEAD's single commit with the staged export's tree (temp index),
# leaving the shared worktree's files and index untouched. Prints the new commit id.
set -e
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
S=/home/ifv/plow-scratch/consolidate/stage
MSG=$1
export GIT_INDEX_FILE=/home/ifv/plow-scratch/consolidate/stage.index
rm -f "$GIT_INDEX_FILE"
git -C "$W" read-tree HEAD
git -C "$W" --work-tree="$S" add -A .
git -C "$W" --work-tree="$S" status --short | grep -v '^??' | head -60
TREE=$(git -C "$W" write-tree)
PARENT=$(git -C "$W" rev-parse HEAD^)
NEW=$(git -C "$W" -c user.name=ifv -c user.email="$GIT_EMAIL" commit-tree "$TREE" -p "$PARENT" -F "$MSG")
echo "NEW=$NEW"
