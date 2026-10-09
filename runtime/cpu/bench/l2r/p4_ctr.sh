#!/usr/bin/env bash
# p4_ctr.sh <outdir> <tag> <refdir> <path A|B> <seqs> [ENV=VAL ...]: p4_run.sh under counters, counted over the timed
# steps only (l2r_layer drives `perf stat --control` through L2R_PERF_CTL). DRAM CAS reads (all IMCs) and L2 lines in
# (r1f25) on the 90 worker cpus (perf reports CAS in MiB). Prints per step: DRAM MiB read, L2 lines in per worker, and the excess over the
# worker's KV lines (an upper bound on weight / activation refetch, as a share of the worker's weight slice).
O=$1 tag=$2; shift 2
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"
fifo=$O/$tag.ctl
rm -f "$fifo"; mkfifo "$fifo"
sudo perf stat -a -A -x, -D -1 --control "fifo:$fifo" -o "$O/$tag.ctr.csv" \
  -e r1f25,uncore_imc/cas_count_read_sch0/,uncore_imc/cas_count_read_sch1/ -- \
  sudo -u "$(id -un)" env L2R_PERF_CTL="$fifo" L2R_BIN="${L2R_BIN:-/tmp/g4c/l2r/l2r_layer}" \
  L2R_COLD_MIB="${L2R_COLD_MIB:-1536}" bash "$S/p4_run.sh" "$O" "$tag" "$@"
rm -f "$fifo"
python3 - "$O/$tag" <<'EOF'
import json, sys
t = sys.argv[1]
workers = set(range(2, 32)) | set(range(34, 64)) | set(range(66, 96))
cas = lin = 0.0
for l in open(t + ".ctr.csv"):
    p = l.strip().split(",")
    if len(p) < 4 or not p[0].startswith("CPU") or not p[1].replace(".", "").isdigit():
        continue
    if "cas_count" in p[3]:
        cas += float(p[1])
    elif p[3] == "r1f25" and int(p[0][3:]) in workers:
        lin += float(p[1])
j = json.loads(open(t + ".json").read().strip().splitlines()[-1])
n = j["steps"] - j["steps"] // 10
kv_lines = j["kv"]["bytes_per_step"] / 64 / j["workers"]
w_lines = j["weight_bytes_per_worker"] / 64
l2 = lin / n / j["workers"]
r = dict(tag=t.split("/")[-1], step_p50=j["step_us"]["p50"], dram_mib_step=cas / n,
         kv_mib_step=j["kv"]["bytes_per_step"] / 2**20, l2_lines_in=l2, kv_lines=kv_lines,
         excess_pct_of_weights=(l2 - kv_lines) / w_lines * 100)
json.dump(r, open(t + ".ctr.json", "w"))
print(f"{r['tag']}: step p50 {r['step_p50']:.1f} us, DRAM {r['dram_mib_step']:.1f} MiB/step (KV {r['kv_mib_step']:.1f}), "
      f"L2 lines in {l2:,.0f}/worker/step (KV {kv_lines:,.0f}), excess {r['excess_pct_of_weights']:.1f}% of the weight slice")
EOF
