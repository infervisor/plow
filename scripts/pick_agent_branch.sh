#!/usr/bin/env bash
# Land an agent worktree branch onto a SQUASHED integration branch.
#
# WHY NOT `git merge`. glm53-mi300x was squashed to a single commit before these agents
# reported. A merge then diffs the whole squashed commit against the agent's base and
# manufactures conflicts in every file the squash touched — 32 across 12 files in the one
# case it was tried, none of them a real disagreement. Cherry-picking replays only the
# agent's own commits, so a conflict means two lanes genuinely edited the same lines.
#
#   scripts/pick_agent_branch.sh <agent-id-or-branch> [base]
#
# `base` defaults to the squash commit the agents branched from; pass the agent's actual
# base when it branched earlier (check `git log --oneline <base>..<branch>` first — if that
# lists commits you do not recognise, the base is wrong).
set -euo pipefail
arg="${1:?agent id or branch}"
branch="$arg"; case "$arg" in worktree-agent-*) ;; *) branch="worktree-agent-$arg";; esac
git rev-parse --verify -q "$branch" >/dev/null || { echo "no such branch: $branch" >&2; exit 1; }
# `HEAD..branch` rather than a fixed base. A base that is too early replays commits already
# in HEAD as EMPTY patches, and cherry-pick then stops on each one with "nothing to commit"
# — which reads like a conflict and is not one. `HEAD..` asks git for exactly the commits
# this branch has and HEAD does not, which is the question being asked.
base="${2:-HEAD}"
mapfile -t picks < <(git rev-list --reverse "$base..$branch")
[ "${#picks[@]}" -gt 0 ] || { echo "nothing to pick: $branch adds no commit HEAD lacks"; exit 0; }
echo ">>> $branch: ${#picks[@]} commit(s) to pick"
git log --oneline --reverse "$base..$branch" | cat
git cherry-pick "${picks[@]}"
