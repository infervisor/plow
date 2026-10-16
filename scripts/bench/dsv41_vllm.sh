#!/usr/bin/env bash
# DeepSeek-V4.1-Flash vLLM runner on 8x MI350X
# Usage:
#   scripts/bench/dsv41_vllm.sh OUT MAXCTX CONC [smoke|bench] [INPUT_LEN] [OUTPUT_LEN]
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/plowbench.sh"
pb_require_nix
pb_hazard_env

out=$(realpath -m -- "${1:?usage: dsv41_vllm.sh OUT MAXCTX CONC [smoke|bench] [INPUT_LEN] [OUTPUT_LEN]}")
maxctx=${2:-16384}
conc=${3:-1}
mode=${4:-smoke}
input_len=${5:-1024}
output_len=${6:-128}

checkpoint=/home/lava/models/DeepSeek-V4.1-Flash
image=vllm/vllm-openai-rocm:nightly-rocm100
test -f "$checkpoint/config.json"
test -f "$checkpoint/model.safetensors.index.json"

mkdir -p "$out"
cidfile="$out/container.id"
rm -f "$cidfile"

cleanup() {
    if test -s "$cidfile"; then
        sudo -n docker stop --time 30 "$(<"$cidfile")" >/dev/null 2>&1 || true
        sudo -n docker rm -f "$(<"$cidfile")" >/dev/null 2>&1 || true
    fi
    if test -n "${PB_SERVER_PID:-}"; then wait "$PB_SERVER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

visibility=()
if test -n "${ROCR_VISIBLE_DEVICES:-}"; then visibility+=(-e "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"); fi
if test -n "${HIP_VISIBLE_DEVICES:-}"; then visibility+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"); fi

PB_SERVER_PORT=$(pb_free_port)
PB_SERVER_LOG="$out/server.log"
tp=${TP:-4}

cache=${VLLM_CACHE_DIR:-$HOME/.cache/vllm-rocm-container}
mkdir -p "$cache"

docker_groups=()
for g in video render; do
    gid=$(getent group "$g" | cut -d: -f3 || true)
    [ -n "$gid" ] && docker_groups+=(--group-add "$gid")
done

docker_env=(-e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1)
if [ "$tp" -le 4 ]; then
    docker_env+=(-e VLLM_ROCM_MONO_DECODE=1)
fi

echo "Starting vLLM server for DeepSeek-V4.1-Flash (TP=$tp, port=$PB_SERVER_PORT, maxctx=$maxctx)..."
timeout --foreground --kill-after=30s 7200 sudo -n docker run --rm --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --cidfile "$cidfile" "${visibility[@]}" "${docker_groups[@]}" \
    "${docker_env[@]}" \
    -v /home/lava/models:/home/lava/models:ro \
    -v "$cache:/root/.cache" \
    --entrypoint vllm "$image" serve \
    "$checkpoint" --host 127.0.0.1 --port "$PB_SERVER_PORT" --served-model-name deepseek-v4.1 \
    --tensor-parallel-size "$tp" --max-model-len "$maxctx" --max-num-seqs 128 \
    --language-model-only --no-swa-bounded-replay \
    --gpu-memory-utilization 0.90 --no-enable-prefix-caching --trust-remote-code \
    > "$PB_SERVER_LOG" 2>&1 &
PB_SERVER_PID=$!

echo "Waiting for vLLM server readiness on port $PB_SERVER_PORT (up to 3600s)..."
pb_serve_wait 3600

model=$(pb_model_id)
echo "Server is ready! Model: $model"

echo "Running smoke check..."
curl -fsS --max-time 600 "http://127.0.0.1:$PB_SERVER_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],"max_tokens":48,"temperature":0}))' "$model")" \
    > "$out/smoke.json"
cat "$out/smoke.json"
echo

if test "$mode" = bench; then
    tag="in${input_len}_c${conc}"
    echo "Running benchmark for $tag (input=$input_len, output=$output_len, conc=$conc)..."
    PB_VLLM="$script_dir/vllm-rocm-client.sh" PB_TOKENIZER="$checkpoint" \
        pb_bench "$out" "$tag" "$model" "$conc" "$conc" "$input_len" "$output_len" --temperature 0
    pb_validate_result "$(pb_result "$out" "$tag")" "$conc" "$output_len"
elif test "$mode" = grid; then
    grid_isls=${ISLS:-"1024 4096 8192 16384"}
    grid_concs=${CONCS:-"1 8 32 64 128"}
    echo "Running full grid sweep over ISLs: $grid_isls and CONCs: $grid_concs..."
    for isl in $grid_isls; do
        for c in $grid_concs; do
            np=$(( c * 2 ))
            [ "$np" -lt 8 ] && np=8
            [ "$np" -gt 64 ] && np=64
            tag="in${isl}_c${c}"
            echo "===== BENCH $tag (isl=$isl, conc=$c, n=$np, out=$output_len) ====="
            PB_VLLM="$script_dir/vllm-rocm-client.sh" PB_TOKENIZER="$checkpoint" \
                pb_bench "$out" "$tag" "$model" "$c" "$np" "$isl" "$output_len" --temperature 0
            pb_validate_result "$(pb_result "$out" "$tag")" "$np" "$output_len"
        done
    done
fi
