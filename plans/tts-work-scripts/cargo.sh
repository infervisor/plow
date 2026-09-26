#!/usr/bin/env bash
# Run cargo in the CUDA-only env from the worktree: cargo.sh <cargo args...>
source /root/tts-work/cuda-env.sh
cd /root/plow/.claude/worktrees/tts-veena-chatterbox
exec cargo "$@"
