#!/usr/bin/env bash
source /root/tts-work/cuda-env.sh
bash /root/plow/.claude/worktrees/tts-veena-chatterbox/runtime/nvidia/s3gen/build.sh "$1"
