#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix
pb_hazard_env
replay=$(realpath "${1:?frozen replay directory required}")
capture=$(realpath "${2:?audited model capture directory required}")
export_args=()
mode=(--replay-model-attention)
extra_mounts=()
if test "${3:-}" = export; then
    export_args=(--export-attention-ps)
elif test "${3:-}" = attention-counterfactual; then
    native=$(realpath "${4:?connected native run directory required}")
    mode=(--replay-model-attention --reference-json "$native/run-record.json")
    extra_mounts=(-v "$native:$native:ro")
elif test "${3:-}" = query; then
    native=$(realpath "${4:?connected native run directory required}")
    checkpoint=$(realpath "${5:?original model checkpoint required}")
    mode=(--replay-model-query --reference-json "$native/run-record.json" --checkpoint "$checkpoint")
    extra_mounts=(-v "$native:$native:ro" -v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = query-model; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--replay-model-query --checkpoint "$checkpoint")
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = oproj; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--block-oproj --oproj-diagnostic --live-prefix --checkpoint "$checkpoint")
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = attention-native; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--block-attention --live-prefix --checkpoint "$checkpoint" --precision-inventory "$replay/precision.json")
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = mla-bmm; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--block-mla-bmm --live-prefix --checkpoint "$checkpoint" --precision-inventory "$replay/precision.json")
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = rope; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--block-rope --live-prefix --checkpoint "$checkpoint" --precision-inventory "$replay/precision.json")
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = qkva; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--export-qkva --checkpoint "$checkpoint" --precision-inventory "$replay/precision.json")
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = qanorm || test "${3:-}" = qanorm-sweep; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--block-qanorm --live-prefix --checkpoint "$checkpoint" --native-rmsnorm "$replay/test_kernels.elf")
    if test "$3" = qanorm-sweep; then mode+=(--qanorm-sweep); fi
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = qb || test "${3:-}" = mla-chain; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    reference=$(realpath "${5:?QKV-A reference directory required}")
    mode=(--export-qb --checkpoint "$checkpoint" --precision-inventory "$replay/precision.json" --reference-json "$reference/comparison.json")
    if test "$3" = mla-chain; then
        mode=(--block-mla --live-prefix --checkpoint "$checkpoint" --precision-inventory "$replay/precision.json" --qb-reference "$reference/comparison.json")
    fi
    extra_mounts=(-v "$checkpoint:$checkpoint:ro" -v "$reference:$reference:ro")
elif test "${3:-}" = router; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--block-router --live-prefix --checkpoint "$checkpoint" --precision-inventory "$replay/precision.json")
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = shared; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--block-shared --live-prefix --checkpoint "$checkpoint")
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test "${3:-}" = routed || test "${3:-}" = routed-stage1 || test "${3:-}" = routed-router; then
    checkpoint=$(realpath "${4:?original model checkpoint required}")
    mode=(--block-routed --routed-w8a8 --routed-down-isolate --live-prefix --checkpoint "$checkpoint")
    if test "$3" = routed-stage1; then
        mode+=(--routed-stage1-diagnostic)
    fi
    if test "$3" = routed-router; then
        mode+=(--router-reference "$replay/router-reference.json")
    fi
    extra_mounts=(-v "$checkpoint:$checkpoint:ro")
elif test -n "${3:-}"; then
    echo 'third argument must be export, attention-counterfactual, query, query-model, oproj, attention-native, mla-bmm, rope, qkva, qanorm, qanorm-sweep, qb, mla-chain, router, shared, routed, routed-stage1, routed-router or omitted' >&2
    exit 2
fi
cd "$replay"
sha256sum --check SHA256SUMS
selection=()
if test -f "$replay/indexer_select_gfx950.elf"; then
    selection=(--native-indexer-selection "$replay/indexer_select_gfx950.elf")
fi
cidfile="$replay/container.id"
cleanup() {
    if test -s "$cidfile"; then
        sudo -n docker stop --time 30 "$(<"$cidfile")" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
visibility=()
if test -n "${ROCR_VISIBLE_DEVICES:-}"; then
    visibility+=(-e "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES")
fi
timeout --foreground --kill-after=30s 600 sudo -n docker run --rm --network none --ipc host \
    --device /dev/kfd --device /dev/dri --cidfile "$cidfile" "${visibility[@]}" \
    -e VLLM_ROCM_USE_AITER=1 -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
    -v "$capture:$capture:ro" -v "$replay:$replay" "${extra_mounts[@]}" --entrypoint python3 \
    vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1 \
    "$replay/block_fp8_aiter_compare.py" "$capture" "${mode[@]}" \
    "${selection[@]}" "${export_args[@]}" --output "$replay/comparison.json"
