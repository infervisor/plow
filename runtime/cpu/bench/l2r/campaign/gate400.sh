#!/usr/bin/env bash
# gate400.sh: start the n_cu 90 E2B packet as the gate's run.sh does and print one request's error body.
source /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/scripts/bench/plowbench.sh
export PLOW_CPU_WEIGHT_AFFINE=1
PB_SERVER_PORT=$(pb_free_port)
PB_SERVER_LOG=/tmp/g4c/l2r/gate400.serve.log
A=/tmp/g4c/l2r/pk90/gemma-4-E2B-it
PLOW_HSACO=$A taskset -c 2-31,34-63,66-95 /tmp/g4c/l2r/bin/plowrt serve --assets $A --port "$PB_SERVER_PORT" > "$PB_SERVER_LOG" 2>&1 &
PB_SERVER_PID=$!
trap pb_serve_stop EXIT
pb_serve_wait 300 || exit 3
MODEL=$(pb_model_id); echo "model=$MODEL"
python3 /tmp/g4c/l2r/gate400.py http://127.0.0.1:$PB_SERVER_PORT /tmp/g4c/gate/gemma-4-E2B-it.ref.json
: curl -s -X POST http://127.0.0.1:$PB_SERVER_PORT/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":[2,818,5279,529,7001,563],\"max_tokens\":4,\"logprobs\":20,\"return_tokens_as_token_ids\":true,\"ignore_eos\":true,\"temperature\":0}"; echo
curl -s -X POST http://127.0.0.1:$PB_SERVER_PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"temperature\":0}"; echo
