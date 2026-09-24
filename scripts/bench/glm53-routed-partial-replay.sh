#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix
pb_hazard_env
native=$(realpath "${1:?frozen native replay directory required}")
reference=$(realpath "${2:?partial fixture directory required}")
cd "$native"
sha256sum --check SHA256SUMS
sha256sum --check REFERENCE_SHA256SUMS
if test "${3:-}" = oproj || test "${3:-}" = oproj-m16; then
    replay_mode=a8w8-replay
    if test "$3" = oproj-m16; then replay_mode=a8w8-m16-replay; fi
    jq -e '(.vllm_version == "0.29.0") and (.cases | length == 8) and
        ([.cases[].rank] | sort == [0,1,2,3,4,5,6,7]) and
        all(.cases[]; (.isolated_partials | length == 4) and
            all(.isolated_partials[]; .finite and .repeat_bitwise))' "$reference/comparison.json" >/dev/null
    status=0
    while IFS= read -r fixture; do
        if ! timeout --foreground --kill-after=10s 60 "$native/block_fp8_test" "$native/test_kernels.elf" \
            "$replay_mode" "$reference/$fixture" "$native/${fixture%.bin}.output.bf16"; then
            status=1
        fi
    done < <(jq -r '.cases[].isolated_partials[].file' "$reference/comparison.json")
    exit "$status"
elif test "${3:-}" = grouped; then
    jq -e '(.cases | length == 8) and all(.cases[]; .replicated_rows == false)' "$reference/comparison.json" >/dev/null
    status=0
    while IFS= read -r fixture; do
        if ! timeout --foreground --kill-after=10s 60 "$native/block_fp8_test" "$native/test_kernels.elf" \
            routed-glu-replay "$reference/$fixture" "$native/${fixture%.bin}.output.bf16"; then
            status=1
        fi
    done < <(jq -r '.cases[].file' "$reference/comparison.json")
    : "${4:?prior grouped BF16 fixtures required}"
    regression_index=0
    for regression_dir in "${@:4}"; do
        regression=$(realpath "$regression_dir")
        manifest="$regression/replay.json"
        if test ! -f "$manifest"; then manifest="$regression/comparison.json"; fi
        while IFS= read -r fixture; do
            if ! timeout --foreground --kill-after=10s 60 "$native/block_fp8_test" "$native/test_kernels.elf" \
                routed-glu-replay "$regression/$fixture" "$native/regression${regression_index}-${fixture%.bin}.output.bf16"; then
                status=1
            fi
        done < <(jq -r '.cases[].file' "$manifest")
        regression_index=$((regression_index + 1))
    done
    exit "$status"
elif test -n "${3:-}"; then
    echo 'third argument must be grouped, oproj, oproj-m16 or omitted' >&2
    exit 2
fi
jq -e '(.cases | length == 8) and all(.cases[]; (.exports | length > 0) and
    all(.exports[]; .output_dtype == "float16"))' "$reference/comparison.json" >/dev/null
mkdir "$native/bf16-regression"
timeout --foreground --kill-after=10s 60 "$native/block_fp8_test" \
    "$native/test_kernels.elf" a8w8-m16 "$native/bf16-regression"
operands=()
while IFS= read -r fixture; do
    operands+=("$reference/$fixture" "$native/${fixture%.bin}.output.fp16")
done < <(jq -r '.cases[].exports[].file' "$reference/comparison.json")
timeout --foreground --kill-after=10s 300 "$native/block_fp8_test" \
    "$native/test_kernels.elf" a8w8-m16-f16-replay "${operands[@]}"
