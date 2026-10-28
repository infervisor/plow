#!/usr/bin/env bash
# p5_srvctr.sh: E2B 2K serving (recipe packet + [serve.env], HEAD plowrt) at c1, ISL 1000 / OSL 512, under system-wide
# L2 lines in (r1f25), LLC misses (r412e) and DRAM CAS reads; per output token (prefill included, OSL 512 so decode
# dominates).
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
M=gemma-4-E2B-it
O=/tmp/g4c/l2r/results/p5_srvctr; rm -rf $O; mkdir -p $O
senv=$(python3.11 -c 'import sys,tomllib; print(" ".join(f"{k}={v}" for k,v in tomllib.load(open(sys.argv[1],"rb")).get("serve",{}).get("env",{}).items()))' $WT/recipes/infervisor/gemma-4-e2b/xeon6-amx-bf16.toml)
sudo perf stat -a -x, -e r1f25,r412e,uncore_imc/cas_count_read_sch0/,uncore_imc/cas_count_read_sch1/ -o $O/ctr.csv -- \
  sudo -u "$(id -un)" env $senv ASSETS=/tmp/g4c/build/$M/assets PLOWRT=/tmp/g4c/l2r/combtgt/release/plowrt \
  PLOWRT_GIT_SHA=$(cd $WT && git rev-parse --short=12 HEAD) CONCS=1 ISL=1000 OSL=512 REPS=1 /tmp/g4c/grid.sh plow $M $O/run > $O/run.log 2>&1
python3 - $O <<'EOF'
import json, sys, glob
o = sys.argv[1]
c = {}
for l in open(o + "/ctr.csv"):
    p = l.strip().split(",")
    if len(p) > 3 and p[0].replace(".", "").isdigit():
        k = "cas" if "cas_count" in p[2] else p[2]
        c[k] = c.get(k, 0) + float(p[0])
b = json.load(open(glob.glob(o + "/run/g1.r1/bench.json")[0]))
n = b["total_output_tokens"]
print(json.dumps(dict(out_tokens=n, tpot_p50=b["median_tpot_ms"], tpot_p99=b["p99_tpot_ms"],
                      l2_lines_in_per_tok=c["r1f25"] / n, llc_miss_per_tok=c["r412e"] / n, dram_mib_per_tok=c["cas"] / n)))
EOF
echo P5_SRVCTR_DONE
