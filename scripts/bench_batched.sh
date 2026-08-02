#!/usr/bin/env bash
# SCOPE-4: §0-BENCH-legal concurrency sweep for the batched AMD serve path.
# `vllm bench serve`, chat backend, against a plowrt endpoint. Same client as
# the vLLM side.
#
#   B      compiled PLOW_DECODE_BATCH of the blob to serve
#   CONCS  client concurrencies
set -u
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
B="${B:-8}"
export PLOW_HSACO=/home/lava/plow/build-amd/hsaco-b$B
IN_LENS="${IN_LENS:-1024}" CONCS="${CONCS:-1 4 16 64}" NPROMPT="${NPROMPT:-64}" OUTLEN="${OUTLEN:-128}" \
  bash "$WT/scripts/bench_plowrt_serve.sh" \
    /home/lava/plow/build-amd/g31b-db$B \
    "${PORT:-8123}" \
    842da3794eaa0b77d5f08bae87a17459d91ff475 \
    google/gemma-4-31B-it \
    1200
