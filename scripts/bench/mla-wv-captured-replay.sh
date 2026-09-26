#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix
pb_hazard_env
run=$(realpath "${1:?frozen replay directory required}")
case "${2:-wv}" in
    wv) test_name=bf16_strided_wv_captured_weights ;;
    attention-chain) test_name=bf16_attention_strided_wv_captured_chain ;;
    *) echo "unsupported replay mode" >&2; exit 2 ;;
esac
if [[ -f "$run/SHA256SUMS" ]]; then
    (cd "$run" && sha256sum --check SHA256SUMS)
fi
sha256sum "$run/plowrt-tests" "$run/test_kernels.elf" "$run/captured/mla-comparison.json"
export PLOW_TEST_AITER_DIR="$run"
timeout --foreground --kill-after=10s 180 "$run/plowrt-tests" \
    "exec::amd_mla_bf16::tests::$test_name" --exact --ignored --nocapture
