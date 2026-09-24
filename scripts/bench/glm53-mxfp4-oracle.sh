#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/plowbench.sh"
pb_require_nix
pb_hazard_env

out=$(realpath -m -- "${1:?usage: glm53-mxfp4-oracle.sh OUT CASES [MAX_CTX]}")
cases=$(realpath -- "${2:?missing cases JSON}")
max_ctx=${3:-8320}
capture=${4:-}
[[ "$max_ctx" =~ ^[1-9][0-9]*$ ]] || exit 2
checkpoint=/opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b
patch_file="$script_dir/vllm029_glm53_mxfp4.patch"
oracle="$script_dir/../vllm_logit_oracle.py"
image=vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1
test -f "$checkpoint/model.safetensors.index.json"
test -f "$cases"
test -f "$patch_file"
test -f "$oracle"
mkdir -p "$out"
cidfile="$out/container.id"
test ! -e "$cidfile"
inputs=("$checkpoint/config.json" "$checkpoint/model.safetensors.index.json" "$patch_file" "$cases" "$oracle")
capture_env=()
capture_mounts=()
if test -n "$capture"; then
    capture=$(realpath -- "$capture")
    capture_site=$(realpath -- "$script_dir/../vllm_capture_site")
    test -f "$capture"
    inputs+=("$capture" "$capture_site/sitecustomize.py" "$capture_site/vllm_forward_capture.py")
    capture_env=(-e PYTHONPATH=/tmp/capture-site -e PLOW_VLLM_CAPTURE_CONFIG=/tmp/capture.json)
    capture_mounts=(-v "$capture:/tmp/capture.json:ro" -v "$capture_site:/tmp/capture-site:ro")
fi
sha256sum "${inputs[@]}" > "$out/inputs.sha256"
cleanup() {
    if test -s "$cidfile"; then
        sudo -n docker stop --time 30 "$(<"$cidfile")" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

visibility=()
if test -n "${ROCR_VISIBLE_DEVICES:-}"; then visibility+=(-e "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"); fi
if test -n "${HIP_VISIBLE_DEVICES:-}"; then visibility+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"); fi
timeout --foreground --kill-after=30s 1800 sudo -n docker run --rm --network none --ipc host \
    --device /dev/kfd --device /dev/dri --cidfile "$cidfile" "${visibility[@]}" \
    -e HF_HUB_OFFLINE=1 -e VLLM_ROCM_USE_AITER=1 -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 \
    "${capture_env[@]}" "${capture_mounts[@]}" \
    -v /opt/models:/opt/models:ro -v "$out:$out" -v "$cases:/tmp/cases.json:ro" \
    -v "$patch_file:/tmp/glm53.patch:ro" -v "$oracle:/tmp/vllm_logit_oracle.py:ro" \
    --entrypoint bash "$image" \
    -lc 'cd /usr/local/lib/python3.12/dist-packages && patch --batch -p1 < /tmp/glm53.patch && exec python3 /tmp/vllm_logit_oracle.py "$@"' bash \
    --model "$checkpoint" --cases /tmp/cases.json --output "$out/reference" \
    --max-output-tokens 1 --tp 8 --max-model-len "$max_ctx" \
    --max-num-batched-tokens "$max_ctx" --request-batch-size 1 \
    --gpu-memory-utilization 0.80 --enforce-eager --trust-remote-code \
    > "$out/oracle.log" 2>&1
