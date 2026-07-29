#!/usr/bin/env bash
# Run the blockIdx -> XCD probe on one leased gfx950. See runtime/tests/xcd_map_gfx950_test.hip.
#
# `gpulease` exports BOTH ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES to the same absolute id
# and they COMPOSE, so a HIP binary reports "no ROCm-capable device" on a correctly leased GPU.
# Unset HIP_/CUDA_ inside the `sg render` shell and keep ROCR_ (knob-contract §0a).
set -euo pipefail
BIN="${1:-/tmp/xcd_map}"
exec sg render -c "unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; $BIN"
