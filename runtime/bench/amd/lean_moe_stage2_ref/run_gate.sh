#!/usr/bin/env bash
set -euo pipefail

if ! docker info >/dev/null 2>&1 && ! sudo -n docker info >/dev/null 2>&1; then
  if sg docker -c "docker info >/dev/null" 2>/dev/null; then
    printf -v REEXEC "%q " "$0" "$@"
    exec sg docker -c "$REEXEC"
  fi
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/runtime/bench/amd/lean_moe_stage2_ref"
OUT=${1:-/tmp/plow-moe2-lean}
if [ "$#" -gt 0 ]; then
  shift
fi
IMAGE='vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032'
OUT=$(realpath "$OUT")

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
  DOCKER=(sudo -n docker)
else
  echo 'docker access is required (directly, docker group, or sudo -n)' >&2
  exit 1
fi

"${DOCKER[@]}" run --rm \
  --entrypoint python3 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined \
  -v "$HERE/gate.py:/opt/plow/gate.py:ro" \
  -v "$OUT:/opt/plow/out:ro" \
  "$IMAGE" \
  /opt/plow/gate.py /opt/plow/out/kernel.co /opt/plow/out/manifest.json "$@"
