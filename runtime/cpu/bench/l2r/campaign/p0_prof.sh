#!/usr/bin/env bash
# p0_prof.sh <tag> <assets> <wrapper> <cpus>: perf record (no callgraph) on <cpus> for 4 s during a long c1 decode.
set -u
tag=$1 A=$2 RT=$3 C=$4
O=/tmp/g4c/l2r/results/p0_prof; mkdir -p $O
source /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/scripts/bench/plowbench.sh
PB_SERVER_PORT=$(pb_free_port); PB_SERVER_LOG=$O/$tag.serve.log
PLOW_CPU_WEIGHT_AFFINE=1 PLOW_HSACO=$A $RT serve --assets $A --port $PB_SERVER_PORT \
  --rt-checkpoint /tmp/models/google/gemma-4-E2B-it > $PB_SERVER_LOG 2>&1 &
PB_SERVER_PID=$!
trap pb_serve_stop EXIT
pb_serve_wait 600 || exit 3
REQ='{"model":"gemma-4-e2b-it","prompt":"Write a long story about a lighthouse keeper.","max_tokens":1500,"temperature":0,"ignore_eos":true}'
curl -s -X POST http://127.0.0.1:$PB_SERVER_PORT/v1/completions -H 'Content-Type: application/json' -d "$REQ" > /dev/null &
CP=$!
sleep 3
sudo perf record -C $C -o $O/$tag.data -- sleep 4 2> $O/$tag.rec.log || true
wait $CP
sudo perf report -i $O/$tag.data --sort sym --stdio 2>/dev/null | grep -v "^#" | grep -v "^$" | head -30 > $O/$tag.top.txt
cat $O/$tag.top.txt
