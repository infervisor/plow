#!/usr/bin/env bash
# The symmetric vLLM point for the concurrency sweep: same client binary, same
# chat backend, same dataset shape as bench_batched.sh.
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IN_LENS="${IN_LENS:-1024}" CONCS="${CONCS:-1 4 16 64}" NPROMPT="${NPROMPT:-64}" \
OUTLEN="${OUTLEN:-128}" MAXLEN="${MAXLEN:-16384}" \
  bash "$WT/scripts/bench_vllm_chat.sh" google/gemma-4-31B-it 1
