#!/usr/bin/env bash
# p1_kvstep.sh: AMX weight-phase overhead after a KV phase vs KV step size (plain, 1.25 MiB, scenario C, 10 s).
O=/tmp/g4c/l2r/results/p1_kvstep; mkdir -p $O
for k in 4096 16384 65536 262144 1048576; do
  L2R_KV_STEP=$k L2R_LOCK=0 L2R_KERNEL=amx /tmp/g4c/l2r/l2r_lock 1310720 10 C > $O/kv$k.jsonl 2> $O/kv$k.err
  echo "kvstep=$k $(grep -o '"per_core_mean":[0-9.]*,"min_gbs":[0-9.]*' $O/kv$k.jsonl) $(grep -o '"worst_p99_us":[0-9.]*' $O/kv$k.jsonl)"
done
