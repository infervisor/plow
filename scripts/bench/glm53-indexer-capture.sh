#!/usr/bin/env bash
set -euo pipefail
source /home/lava/plow/scripts/bench/plowbench.sh
pb_require_nix
pb_hazard_env
capture=$(realpath "${1:?usage: glm53-indexer-capture.sh frozen-capture-directory}")
for file in capture.json cases.json sitecustomize.py vllm_forward_capture.py vllm_logit_oracle.py; do
    test -f "$capture/$file"
done
cidfile="$capture/container.id"
test ! -e "$cidfile"
cleanup() {
    if test -s "$cidfile"; then
        sudo -n docker stop --time 30 "$(<"$cidfile")" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
visibility=()
if test -n "${ROCR_VISIBLE_DEVICES:-}"; then visibility+=(-e "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"); fi
timeout --foreground --kill-after=30s 1800 sudo -n docker run --rm --network none --ipc host \
    --device /dev/kfd --device /dev/dri --cidfile "$cidfile" "${visibility[@]}" \
    -e HF_HUB_OFFLINE=1 -e VLLM_ROCM_USE_AITER=1 -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 \
    -e "PYTHONPATH=$capture" -e "PLOW_VLLM_CAPTURE_CONFIG=$capture/capture.json" \
    -v /opt/models:/opt/models:ro -v "$capture:$capture" --entrypoint python3 \
    vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1 \
    "$capture/vllm_logit_oracle.py" --model /opt/models/GLM-5.3-full-aca966e4 \
    --cases "$capture/cases.json" --output "$capture/reference" --precision-report \
    --max-output-tokens 5 --tp 8 --max-model-len 73728 --max-num-batched-tokens 8192 \
    --enforce-eager --trust-remote-code > "$capture/oracle.log" 2>&1
