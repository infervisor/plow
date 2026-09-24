#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix
pb_hazard_env
run=$(realpath "${1:?frozen replay directory required}")
cd "$run"
sha256sum --check SHA256SUMS
export PLOW_TEST_AITER_DIR="$run"
timeout --foreground --kill-after=10s 120 "$run/plowrt-tests" \
    device::hsa::tests::kernarg_admission_gpu_regression --exact --ignored --nocapture
