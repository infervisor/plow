#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix
pb_hazard_env

assets=$(realpath -- "${1:?usage: glm53-mxfp4-dp2.sh ASSETS OBJECTS RUNTIME OUT CONC_PER_REPLICA INPUT_LEN OUTPUT_LEN}")
objects=$(realpath -- "${2:?missing objects}")
runtime=$(realpath -- "${3:?missing runtime}")
out=$(realpath -m -- "${4:?missing output dir}")
conc=${5:?missing concurrency per replica}
input_len=${6:?missing input length}
output_len=${7:?missing output length}
[[ "$conc" =~ ^[1-9][0-9]*$ && "$input_len" =~ ^[1-9][0-9]*$ \
    && "$output_len" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -f "$assets/model.pkt" && -f "$assets/weights.json" && -x "$runtime" ]] || exit 2

IFS=, read -r -a devices <<< "${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ ${#devices[@]} == 8 ]] || { echo "DP2 needs exactly eight leased GPUs" >&2; exit 2; }
mask0=${devices[0]},${devices[1]},${devices[2]},${devices[3]}
mask1=${devices[4]},${devices[5]},${devices[6]},${devices[7]}
port0=$(pb_free_port)
port1=$(pb_free_port)
while [[ "$port1" == "$port0" ]]; do port1=$(pb_free_port); done
mkdir -p "$out/replica0" "$out/replica1"
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1

server0= server1=
stop_all() {
    if [[ -n "$server1" ]]; then PB_SERVER_PID=$server1; pb_serve_stop; fi
    if [[ -n "$server0" ]]; then PB_SERVER_PID=$server0; pb_serve_stop; fi
}
trap stop_all EXIT

ROCR_VISIBLE_DEVICES=$mask0 pb_serve_start "$runtime" "$assets" "$objects" "$port0" "$out/replica0/server.log" 3600
server0=$PB_SERVER_PID
ROCR_VISIBLE_DEVICES=$mask1 pb_serve_start "$runtime" "$assets" "$objects" "$port1" "$out/replica1/server.log" 3600
server1=$PB_SERVER_PID
PB_SERVER_PID=$server0 PB_SERVER_PORT=$port0 PB_SERVER_LOG="$out/replica0/server.log" pb_serve_wait 600
PB_SERVER_PID=$server1 PB_SERVER_PORT=$port1 PB_SERVER_LOG="$out/replica1/server.log" pb_serve_wait 600

export PB_VLLM="$(dirname -- "${BASH_SOURCE[0]}")/vllm029-client.sh"
export PB_TOKENIZER=/opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b
model=$(PB_SERVER_PORT=$port0 pb_model_id)
start_ns=$(date +%s%N)
( PB_SERVER_PORT=$port0 PB_SEED=8193 pb_bench "$out/replica0" dp2 "$model" "$conc" "$conc" "$input_len" "$output_len" --temperature 0 ) &
client0=$!
( PB_SERVER_PORT=$port1 PB_SEED=8293 pb_bench "$out/replica1" dp2 "$model" "$conc" "$conc" "$input_len" "$output_len" --temperature 0 ) &
client1=$!
wait "$client0"
wait "$client1"
end_ns=$(date +%s%N)
result0=$(pb_result "$out/replica0" dp2)
result1=$(pb_result "$out/replica1" dp2)
pb_validate_result "$result0" "$conc" "$output_len"
pb_validate_result "$result1" "$conc" "$output_len"
python3 -c 'import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:3]); wall=(int(sys.argv[4])-int(sys.argv[3]))/1e9; start=min(min(a["start_times"]),min(b["start_times"])); end=max(min(a["start_times"])+a["duration"],min(b["start_times"])+b["duration"]); total=a["total_output_tokens"]+b["total_output_tokens"]; completed=a["completed"]+b["completed"]; print(f"DP2 completed={completed} output_tokens={total} bench_window_s={end-start:.3f} bench_tok_s={total/(end-start):.3f} launch_wall_s={wall:.3f} wall_tok_s={total/wall:.3f}")' "$result0" "$result1" "$start_ns" "$end_ns"
