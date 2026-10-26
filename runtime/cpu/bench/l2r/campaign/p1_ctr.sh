#!/usr/bin/env bash
# p1_ctr.sh: L2 fill/evict and CHA snoop-filter eviction counters for l2r_lock, 1.25 MiB slice, 12 lock ways,
# plain vs locked, scenarios A / C / D, 10 s. Raw core events (perf 6.1 has no GNR JSON):
#   L2_LINES_IN.ALL r1f25, L2_LINES_OUT.SILENT r0126, .NON_SILENT r0226, .USELESS_HWPF r0426; CHA SF_EVICTION 0x3d/0x07.
set -u
O=/tmp/g4c/l2r/results/p1_ctr; mkdir -p $O
INF=2-31,34-63,66-95
SZ=${SZ:-1310720}
for sc in A C D; do
  for lock in 0 1; do
    t=$O/s$SZ.l$lock.$sc
    sudo perf stat -x, -o $t.core.csv -C $INF -e r1f25,r0126,r0226,r0426,cycles \
      -- sudo perf stat -x, -o $t.cha.csv -a -e uncore_cha/event=0x3d,umask=0x07/ \
      -- sudo -u $(id -un) env L2R_LOCK=$lock L2R_KERNEL=amx /tmp/g4c/l2r/l2r_lock $SZ 10 $sc > $t.jsonl 2> $t.err
    echo "$sc lock=$lock rc=$?"
  done
done
