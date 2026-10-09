#!/usr/bin/env bash
# p0_noise.sh: P0.2 checks under load. (1) context switches / migrations / IRQs on the 90 inference cpus during
# a 20 s AMX-BF16 L2 stream; (2) NUMA placement of a running E4B plowrt (numastat -p, numa_maps summary).
O=/tmp/g4c/l2r/results/p0_noise; mkdir -p $O
INF=2-31,34-63,66-95
sudo perf stat -a -C $INF -x, -e context-switches,cpu-migrations,irq_vectors:local_timer_entry,page-faults \
  -o $O/perf_stream.csv -- env L2R_HUGE=1 /tmp/g4c/l2r/l2r_bw 1048576 1 20 1 > $O/stream.jsonl 2>&1
source /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/scripts/bench/plowbench.sh
PB_SERVER_PORT=$(pb_free_port); PB_SERVER_LOG=$O/serve.log
A=/tmp/g4c/l2r/pk90/gemma-4-E4B-it
PLOW_CPU_WEIGHT_AFFINE=1 PLOW_HSACO=$A /tmp/g4c/l2r/bin/plowrt-iso serve --assets $A --port $PB_SERVER_PORT \
  --rt-checkpoint /tmp/models/google/gemma-4-E4B-it > $PB_SERVER_LOG 2>&1 &
PB_SERVER_PID=$!
trap pb_serve_stop EXIT
pb_serve_wait 600 || exit 3
P=$(pgrep -f "l2r/bin/plowrt serve" | head -1)
nix develop -c numastat -p $P > $O/numastat.txt 2>&1
awk '{for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/){split($i,a,"="); n[a[1]]+=a[2]}} END {for(k in n) print k, n[k]*4/1048576 " GiB (4K-page units)"}' /proc/$P/numa_maps > $O/numa_maps_sum.txt
grep -c huge /proc/$P/numa_maps >> $O/numa_maps_sum.txt
sudo perf stat -a -C $INF -x, -e context-switches,cpu-migrations -o $O/perf_serve.csv -- \
  curl -s -X POST http://127.0.0.1:$PB_SERVER_PORT/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e4b-it","prompt":"Write a long story about a lighthouse keeper.","max_tokens":256,"temperature":0,"ignore_eos":true}' > $O/serve_req.json
echo DONE
