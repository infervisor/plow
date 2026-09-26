#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix

args=()
exact_latencies=0
for arg in "$@"; do
    if [ "$arg" = --plow-exact-latencies ]; then
        exact_latencies=1
    else
        args+=("$arg")
    fi
done
mounts=(-v /opt/models:/opt/models:ro)
result_dir=
for ((i=0; i<${#args[@]}; i++)); do
    if [ "${args[i]}" = --result-dir ]; then
        result_dir="$(realpath -m -- "${args[i+1]:?missing result directory}")"
        mkdir -p -- "$result_dir"
        mounts+=(-v "$result_dir:$result_dir")
        args[i+1]="$result_dir"
    fi
done
if [ "$exact_latencies" = 1 ]; then
    if [ -z "$result_dir" ] || [[ " ${args[*]} " != *" --save-result "* ]] || [[ " ${args[*]} " != *" --save-detailed "* ]]; then
        echo 'exact client latency export requires --result-dir, --save-result and --save-detailed' >&2
        exit 2
    fi
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    exporter="$script_dir/client_latency.py"
    if [ ! -f "$exporter" ]; then exporter="$script_dir/../campaign/client_latency.py"; fi
    python3 "$exporter" --output-dir "$result_dir/client-overlay"
    mounts+=(-v "$result_dir/client-overlay/serve.py:/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py:ro")
fi
exec sudo -n docker run --rm --network host \
    -e HF_HUB_OFFLINE=1 -e HF_MODULES_CACHE=/tmp/hf_modules \
    "${mounts[@]}" --entrypoint python3 \
    vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1 \
    "${args[@]}"
