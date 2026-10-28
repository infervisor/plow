#!/usr/bin/env bash
# p0_ctx.sh: attribute the context switches on the inference cpus during one 256-token E4B request.
# Per-thread voluntary/nonvoluntary ctxt deltas + comm + affinity for plowrt, and perf sched-free per-cpu
# context-switch counts with/without the plowrt pid filter.
O=/tmp/g4c/l2r/results/p0_noise; mkdir -p $O
INF=2-31,34-63,66-95
source /home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/scripts/bench/plowbench.sh
PB_SERVER_PORT=$(pb_free_port); PB_SERVER_LOG=$O/serve_ctx.log
A=/tmp/g4c/l2r/pk90/gemma-4-E4B-it
PLOW_CPU_WEIGHT_AFFINE=1 PLOW_HSACO=$A /tmp/g4c/l2r/bin/plowrt-iso serve --assets $A --port $PB_SERVER_PORT \
  --rt-checkpoint /tmp/models/google/gemma-4-E4B-it > $PB_SERVER_LOG 2>&1 &
PB_SERVER_PID=$!
trap pb_serve_stop EXIT
pb_serve_wait 600 || exit 3
P=$(pgrep -f "l2r/bin/plowrt serve" | head -1)
snap() {
  for t in /proc/$P/task/*; do
    tid=${t##*/}
    printf '%s %s %s %s %s\n' "$tid" "$(tr ' ' '_' < $t/comm)" \
      "$(awk '/^voluntary_ctxt/{print $2}' $t/status)" "$(awk '/^nonvoluntary_ctxt/{print $2}' $t/status)" \
      "$(awk '/^Cpus_allowed_list/{print $2}' $t/status)"
  done
}
REQ='{"model":"gemma-4-e4b-it","prompt":"Write a long story about a lighthouse keeper.","max_tokens":256,"temperature":0,"ignore_eos":true}'
curl -s -X POST http://127.0.0.1:$PB_SERVER_PORT/v1/completions -H 'Content-Type: application/json' -d "$REQ" > /dev/null
snap > $O/ctx_before.txt
sudo perf stat -a -C $INF -x, -e context-switches,cpu-migrations -o $O/perf_serve_all.csv -- \
  curl -s -X POST http://127.0.0.1:$PB_SERVER_PORT/v1/completions -H 'Content-Type: application/json' -d "$REQ" > /dev/null
snap > $O/ctx_after.txt
sudo perf stat -a -C $INF -x, -e context-switches -o $O/perf_idle.csv -- sleep 10
python3 - "$O" <<'EOF'
import sys, collections
o = sys.argv[1]
def rd(f):
    d = {}
    for l in open(f):
        tid, comm, v, n, aff = l.split()
        d[tid] = (comm, int(v), int(n), aff)
    return d
a, b = rd(o + "/ctx_before.txt"), rd(o + "/ctx_after.txt")
g = collections.defaultdict(lambda: [0, 0, 0, set()])
for tid, (comm, v, n, aff) in b.items():
    v0, n0 = (a[tid][1], a[tid][2]) if tid in a else (0, 0)
    k = comm.rstrip("0123456789-_")
    g[k][0] += 1; g[k][1] += v - v0; g[k][2] += n - n0; g[k][3].add(aff)
with open(o + "/ctx_summary.txt", "w") as f:
    for k, (cnt, v, n, affs) in sorted(g.items(), key=lambda x: -(x[1][1] + x[1][2])):
        f.write(f"{k:24s} threads={cnt:4d} vol={v:8d} nonvol={n:6d} aff={','.join(sorted(affs))}\n")
print(open(o + "/ctx_summary.txt").read())
EOF
cat $O/perf_serve_all.csv $O/perf_idle.csv
