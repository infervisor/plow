#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/plowbench.sh"
pb_require_nix
pb_hazard_env

out=$(realpath -m -- "${1:?usage: glm53-mxfp4-vllm.sh OUT MAXCTX CONC [smoke|bench|dual] [INPUT_LEN] [OUTPUT_LEN]}")
maxctx=${2:?missing max context}
conc=${3:?missing concurrency}
mode=${4:-smoke}
input_len=${5:-8192}
output_len=${6:-128}
[[ "$maxctx" =~ ^[1-9][0-9]*$ && "$conc" =~ ^[1-9][0-9]*$ \
    && "$input_len" =~ ^[1-9][0-9]*$ && "$output_len" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$mode" = smoke || "$mode" = bench || "$mode" = dual ]] || exit 2
if test "$mode" = bench || test "$mode" = dual; then (( input_len + output_len <= maxctx )) || exit 2; fi
if test "$mode" = dual; then (( conc % 2 == 0 )) || exit 2; fi
checkpoint=/opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b
patch_file="$script_dir/vllm029_glm53_mxfp4.patch"
image=vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1
test -f "$checkpoint/model.safetensors.index.json"
test -f "$patch_file"
mkdir -p "$out"
sha256sum "$checkpoint/config.json" "$checkpoint/model.safetensors.index.json" "$patch_file" > "$out/inputs.sha256"
cidfile="$out/container.id"
test ! -e "$cidfile"
cleanup() {
    if test -s "$cidfile"; then
        sudo -n docker stop --time 30 "$(<"$cidfile")" >/dev/null 2>&1 || true
    fi
    if test -n "${PB_SERVER_PID:-}"; then wait "$PB_SERVER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

visibility=()
if test -n "${ROCR_VISIBLE_DEVICES:-}"; then visibility+=(-e "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"); fi
if test -n "${HIP_VISIBLE_DEVICES:-}"; then visibility+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"); fi
PB_SERVER_PORT=$(pb_free_port)
PB_SERVER_LOG="$out/server.log"
timeout --foreground --kill-after=30s 3600 sudo -n docker run --rm --network host --ipc host \
    --device /dev/kfd --device /dev/dri --cidfile "$cidfile" "${visibility[@]}" \
    -e HF_HUB_OFFLINE=1 -e VLLM_ROCM_USE_AITER=1 -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 \
    -v /opt/models:/opt/models:ro -v "$patch_file:/tmp/glm53.patch:ro" --entrypoint bash "$image" \
    -lc 'cd /usr/local/lib/python3.12/dist-packages && patch --batch -p1 < /tmp/glm53.patch && exec vllm serve "$@"' bash \
    "$checkpoint" --host 127.0.0.1 --port "$PB_SERVER_PORT" --served-model-name glm-5.3-mxfp4 \
    --tensor-parallel-size 8 --max-model-len "$maxctx" --max-num-seqs "$conc" \
    --gpu-memory-utilization 0.80 --no-enable-prefix-caching --trust-remote-code \
    > "$PB_SERVER_LOG" 2>&1 &
PB_SERVER_PID=$!
pb_serve_wait 1800
model=$(pb_model_id)
curl -fsS --max-time 600 "http://127.0.0.1:$PB_SERVER_PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    --data "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":"What is the capital of France? Answer:","max_tokens":32,"temperature":0}))' "$model")" \
    > "$out/smoke.json"
if test "$mode" = bench; then
    tag="in${input_len}_c${conc}"
    PB_VLLM="$script_dir/vllm029-client.sh" PB_TOKENIZER="$checkpoint" \
        pb_bench "$out" "$tag" "$model" "$conc" "$conc" "$input_len" "$output_len" --temperature 0
    pb_validate_result "$(pb_result "$out" "$tag")" "$conc" "$output_len"
elif test "$mode" = dual; then
    half=$((conc / 2))
    mkdir -p "$out/replica0" "$out/replica1"
    start_ns=$(date +%s%N)
    ( PB_VLLM="$script_dir/vllm029-client.sh" PB_TOKENIZER="$checkpoint" PB_SEED=8193 \
        pb_bench "$out/replica0" dual "$model" "$half" "$half" "$input_len" "$output_len" --temperature 0 ) &
    client0=$!
    ( PB_VLLM="$script_dir/vllm029-client.sh" PB_TOKENIZER="$checkpoint" PB_SEED=8293 \
        pb_bench "$out/replica1" dual "$model" "$half" "$half" "$input_len" "$output_len" --temperature 0 ) &
    client1=$!
    wait "$client0"
    wait "$client1"
    end_ns=$(date +%s%N)
    result0=$(pb_result "$out/replica0" dual)
    result1=$(pb_result "$out/replica1" dual)
    pb_validate_result "$result0" "$half" "$output_len"
    pb_validate_result "$result1" "$half" "$output_len"
    python3 -c 'import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:3]); wall=(int(sys.argv[4])-int(sys.argv[3]))/1e9; start=min(min(a["start_times"]),min(b["start_times"])); end=max(min(a["start_times"])+a["duration"],min(b["start_times"])+b["duration"]); total=a["total_output_tokens"]+b["total_output_tokens"]; completed=a["completed"]+b["completed"]; print(f"dual completed={completed} output_tokens={total} bench_window_s={end-start:.3f} bench_tok_s={total/(end-start):.3f} launch_wall_s={wall:.3f} wall_tok_s={total/wall:.3f}")' "$result0" "$result1" "$start_ns" "$end_ns"
fi
