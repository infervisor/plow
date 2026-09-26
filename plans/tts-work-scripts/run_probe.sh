#!/usr/bin/env bash
# run_probe.sh <label> <script> [args...] — CUDA env + gpulease -n 1 wrapper for worktree scripts.
source /root/tts-work/cuda-env.sh
export HF_HOME=/root/tts-work/hf TTS_PY=/root/tts-work/venv-ref/bin/python
export PLOWRT_BIN=${PLOWRT_BIN:-/root/tts-work/plowrt-probe1}
cd /root/plow/.claude/worktrees/tts-veena-chatterbox
L=$1; shift
exec perf-data/tools/gpulease -n ${NGPU_LEASE:-1} "$L" "$@"
