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
base="${2:-$(git rev-parse 3c7699df 2>/dev/null || true)}"
[ -n "$base" ] || { echo "pass the base commit explicitly" >&2; exit 1; }
echo ">>> $branch: $(git rev-list --count "$base..$branch") commit(s) to pick"
git log --oneline "$base..$branch" | cat
git cherry-pick "$base..$branch"
