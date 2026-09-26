#!/usr/bin/env bash
set -euo pipefail
script_dir=$(dirname -- "${BASH_SOURCE[0]}")
source "$script_dir/plowbench.sh"
pb_require_nix
pb_hazard_env

out=$(realpath -m -- "${1:?usage: glm53-mxfp4-plow.sh OUT ASSETS OBJECTS CONC INPUT_LEN OUTPUT_LEN}")
assets=$(realpath -- "${2:?missing assets}")
objects=$(realpath -- "${3:?missing objects}")
conc=${4:?missing concurrency}
input_len=${5:?missing input length}
output_len=${6:?missing output length}
[[ "$conc" =~ ^[1-9][0-9]*$ && "$input_len" =~ ^[1-9][0-9]*$ \
    && "$output_len" =~ ^[1-9][0-9]*$ ]] || exit 2
runtime=${PLOWRT_BIN:-/home/lava/plow/build-glm53-mxfp4/tp8-8k-v2/plowrt}
tokenizer=/opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
test -f "$assets/model.pkt"
test -f "$assets/weights.json"
test -x "$runtime"
mkdir -p "$out"
PB_SERVER_PORT=$(pb_free_port)
trap pb_serve_stop EXIT
pb_serve_start "$runtime" "$assets" "$objects" "$PB_SERVER_PORT" "$out/server.log" 3600
pb_serve_wait 600
model=$(pb_model_id)
curl -fsS --max-time 600 "http://127.0.0.1:$PB_SERVER_PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    --data "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":"What is the capital of France? Answer:","max_tokens":32,"temperature":0}))' "$model")" \
    > "$out/smoke.json"
tag="in${input_len}_c${conc}"
PB_VLLM="$script_dir/vllm029-client.sh" PB_TOKENIZER="$tokenizer" \
    pb_bench "$out" "$tag" "$model" "$conc" "$conc" "$input_len" "$output_len" --temperature 0
pb_validate_result "$(pb_result "$out" "$tag")" "$conc" "$output_len"
