#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix

args=("$@")
mounts=(-v /opt/models:/opt/models:ro)
for ((i=0; i<${#args[@]}; i++)); do
    if [ "${args[i]}" = --result-dir ]; then
        result_dir="$(realpath -m -- "${args[i+1]:?missing result directory}")"
        mkdir -p -- "$result_dir"
        mounts+=(-v "$result_dir:$result_dir")
        args[i+1]="$result_dir"
    fi
done
exec sudo -n docker run --rm --network host \
    -e HF_HUB_OFFLINE=1 -e HF_MODULES_CACHE=/tmp/hf_modules \
    "${mounts[@]}" --entrypoint python3 \
    vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1 \
    "${args[@]}"
