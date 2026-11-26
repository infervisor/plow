#!/usr/bin/env bash
# r.sh <args...>: the vLLM venv python in the flake env, CUDA 13 for library JIT (as lease_grid_vllm_nemo.sh).
cd /home/ifv/plow/.claude/worktrees/gb10-sm121a || exit 1
exec /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh bash -c '
  export CC="$(command -v cc)" CXX="$(command -v c++)" MAX_JOBS=3
  unset NVCC_PREPEND_FLAGS CUDA_PATH CUDA_HOME PLOW_NVCC
  PATH=$(printf "%s" "$PATH" | tr ":" "\n" | grep -v cuda | paste -sd:)
  C13=/home/ifv/plow-scratch/cuda13
  export CUDA_HOME=$C13 PATH="$C13/bin:/home/ifv/venvs/vllm-gb10/bin:$PATH" LIBRARY_PATH="$C13/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
  exec /home/ifv/venvs/vllm-gb10/bin/python "$@"' _ "$@"
