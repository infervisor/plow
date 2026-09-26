#!/usr/bin/env bash
# Stage 3 toolchain: elan installs the pinned Lean, then lake builds Plow + plow_verify.
source /root/tts-work/cuda-env.sh
export PATH=/nix/store/vdwa17hx00gfzhz0kbaw5x3cp781mgx0-elan-4.2.3/bin:$HOME/.elan/bin:$PATH
cd /root/plow/.claude/worktrees/tts-veena-chatterbox/lean-plow
elan --version
time lake build 2>&1 | tail -n 15
ls -la .lake/build/bin/ 2>&1
echo LEAN_DONE
