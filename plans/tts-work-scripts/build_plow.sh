#!/usr/bin/env bash
# Build plowc + plowrt(cuda) in the CUDA-only env, private target dir.
source /root/tts-work/cuda-env.sh
cd /root/plow/.claude/worktrees/tts-veena-chatterbox
set -x
cargo --version; nvcc --version | tail -n 1
cargo build --release -p plowc 2>&1 | tail -n 20
cargo build --release -p plowrt --features cuda 2>&1 | tail -n 20
ls -la $CARGO_TARGET_DIR/release/plowc $CARGO_TARGET_DIR/release/plowrt
echo BUILD_DONE
