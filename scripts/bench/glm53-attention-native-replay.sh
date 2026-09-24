#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix
pb_hazard_env
native=$(realpath "${1:?frozen native replay directory required}")
reference=$(realpath "${2:?model fixture directory required}")
cd "$native"
sha256sum --check SHA256SUMS
sha256sum --check REFERENCE_SHA256SUMS
jq -e '.passed and (.cases | length == 4) and
    ([.cases[].native_fixtures[]] | length == 20) and
    all(.cases[].native_fixtures[]; .finite and .repeat_bitwise and .serving_reference_bitwise)' \
    "$reference/comparison.json" >/dev/null
while IFS=$'\t' read -r fixture reduction; do
    timeout --foreground --kill-after=10s 60 "$native/block_fp8_test" \
        "$native/mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co" attention-pipeline-replay \
        "$reference/$fixture" "$native/${fixture%.bin}.f32" \
        "$reference/$reduction" "$native/mla_sparse_adapter_gfx950.elf" \
        "$native/metadata_gfx950.elf" "$native/reduce_gfx950.elf"
done < <(jq -r '.cases[].native_fixtures[] | [.file, .reduce_file] | @tsv' "$reference/comparison.json")
