#!/usr/bin/env bash
# vllm-rocm-client.sh <python3 args...> — PYREF for an AMD box whose only vLLM is the ROCm Docker
# image: `python3` inside VLLM_IMAGE (vllm/vllm-openai-rocm:nightly-rocm100), so the reference
# server, the `vllm bench serve` client and `serving_comparison.py record`'s version query all run
# the same image.
#
#   PYREF=scripts/bench/vllm-rocm-client.sh HF=/home/lava/models/<name> MODEL_NAME=<name> \
#       READY_S=3600 scripts/bench/llm_grid.sh vllm <res>/vllm
#
# With VLLM_CONTAINER set (llm_grid.sh does, for the server) the container is named so the caller
# can `docker rm -f` it, gets the GPUs (/dev/kfd, /dev/dri, host video/render gids,
# ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES forwarded), host IPC and VLLM_CACHE_DIR
# ($HOME/.cache/vllm-rocm-container) at /root/.cache: HF, torch.compile and AITER JIT caches persist
# across runs. Without it (client calls) no devices or cache are attached and any --result-dir is
# mounted read-write at its host path. VLLM_MOUNTS: space list of host:container[:ro] (default
# /home/lava/models:/home/lava/models:ro); VLLM_DOCKER_ENV: space list of K=V forwarded with -e.
# vllm029-client.sh's --plow-exact-latencies overlay is not supported (its serve.py path is pinned
# to that image).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix

image=${VLLM_IMAGE:-vllm/vllm-openai-rocm:nightly-rocm100}
args=("$@")
docker_args=(--rm --network host --ipc host --shm-size 32g
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface -e HF_MODULES_CACHE=/tmp/hf_modules)
for m in ${VLLM_MOUNTS:-/home/lava/models:/home/lava/models:ro}; do docker_args+=(-v "$m"); done
for kv in ${VLLM_DOCKER_ENV:-}; do docker_args+=(-e "$kv"); done
if [ -n "${VLLM_CONTAINER:-}" ]; then
    cache=${VLLM_CACHE_DIR:-$HOME/.cache/vllm-rocm-container}
    mkdir -p -- "$cache"
    docker_args+=(--name "$VLLM_CONTAINER" --device /dev/kfd --device /dev/dri
        --security-opt seccomp=unconfined -v "$cache:/root/.cache")
    # The image has no video/render groups, so --group-add by name fails; pass host gids.
    for g in video render; do
        gid=$(getent group "$g" | cut -d: -f3)
        [ -n "$gid" ] && docker_args+=(--group-add "$gid")
    done
    for var in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; do
        [ -n "${!var:-}" ] && docker_args+=(-e "$var=${!var}")
    done
else
    # A host script (agentic_turns.py, needle/quality probes) runs from its checkout: mount the
    # repo root read-only at its own path so sibling imports resolve.
    if [[ "${args[0]:-}" == *.py && -f "${args[0]}" ]]; then
        repo="$(cd -- "$(dirname -- "${args[0]}")" && git rev-parse --show-toplevel 2>/dev/null || true)"
        [ -n "$repo" ] && docker_args+=(-v "$repo:$repo:ro")
    fi
    for ((i=0; i<${#args[@]}; i++)); do
        case "${args[i]}" in
        --result-dir)
            result_dir="$(realpath -m -- "${args[i+1]:?missing result directory}")"
            mkdir -p -- "$result_dir"
            docker_args+=(-v "$result_dir:$result_dir")
            args[i+1]="$result_dir" ;;
        --out|--output)
            out="$(realpath -m -- "${args[i+1]:?missing output path}")"
            out_dir=$out; [[ "$out" == *.json || "$out" == *.jsonl ]] && out_dir=$(dirname -- "$out")
            mkdir -p -- "$out_dir"
            docker_args+=(-v "$out_dir:$out_dir")
            args[i+1]="$out" ;;
        esac
    done
fi
exec sudo -n docker run "${docker_args[@]}" --entrypoint python3 "$image" "${args[@]}"
