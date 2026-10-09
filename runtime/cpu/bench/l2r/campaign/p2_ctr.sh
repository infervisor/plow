#!/usr/bin/env bash
# p2_ctr.sh <ref> <steps> <tag>: DRAM CAS reads (all IMCs) and L2 lines in on the worker cpus during an l2r_layer run
# (no all-gather). Prints per-step DRAM bytes and per-worker L2 lines in per step.
O=/tmp/g4c/l2r/results/p2_ctr; mkdir -p $O
ref=$1 n=$2 tag=$3
sudo perf stat -x, -a -o $O/$tag.imc.csv -e uncore_imc/cas_count_read_sch0/,uncore_imc/cas_count_read_sch1/ -- \
  sudo perf stat -x, -C 2-31,34-63,66-95 -o $O/$tag.core.csv -e r1f25 -- \
  sudo -u $(id -un) env L2R_NOBCAST=1 ${L2R_ENV:-} /tmp/g4c/l2r/l2r_layer $ref $n > $O/$tag.json
python3 - $O/$tag $n <<'EOF'
import json, sys
t, n = sys.argv[1], int(sys.argv[2])
def rd(f):
    s = 0
    for l in open(f):
        p = l.split(",")
        if len(p) > 2 and p[0].replace(".", "").isdigit(): s += float(p[0])
    return s
cas, lin = rd(t + ".imc.csv"), rd(t + ".core.csv")
j = json.loads(open(t + ".json").read().strip().splitlines()[-1])
print(f"{t.split('/')[-1]}: step p50 {j['step_us']['p50']:.1f} us, DRAM read {cas * 64 / n / 2**20:.2f} MiB/step "
      f"(incl. load phase), L2 lines in {lin / n / 90:,.0f} per worker per step")
EOF
