#!/usr/bin/env bash
set -euo pipefail
gemma_dir=plans/gemma3-12b-roofline
mkdir -p "$gemma_dir/cubin-gemma3-decode-mma"
PLOW_BUILD_DECODE_ONLY=1 \
PLOW_EXTRA_DEFINES='-DPLOW_NV_GEMMA3=1 -DPLOW_NV_FP8_DECODE_MMA=1' \
  bash scripts/build_sm90a_cubin.sh "$gemma_dir/cubin-gemma3-decode-mma/interp_sm90a.cubin"
