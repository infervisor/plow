#!/usr/bin/env bash
set -euo pipefail

if ! docker info >/dev/null 2>&1; then
  if sg docker -c "docker info >/dev/null" 2>/dev/null; then
    printf -v reexec "%q " "$0" "$@"
    exec sg docker -c "$reexec"
  fi
  echo "docker access is required" >&2
  exit 1
fi

if [[ $# -ne 1 ]]; then
  echo "usage: $0 BUILD_DIR" >&2
  exit 2
fi

root=$(cd "$(dirname "$0")/../../../.." && pwd)
here="$root/runtime/bench/amd/lean_moe_combine_ref"
out=$(realpath "$1")
image='vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032'

docker run --rm --entrypoint python3 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined \
  -e "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:?run under gpulease}" \
  -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:?run under gpulease}" \
  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?run under gpulease}" \
  -v "$here/gate.py:/opt/plow/gate.py:ro" \
  -v "$out:/opt/plow/out:ro" \
  "$image" /opt/plow/gate.py \
  /opt/plow/out/kernel.co /opt/plow/out/manifest.json --tokens 8192
