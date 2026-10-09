#!/usr/bin/env bash
# l3tier.sh: AMX-BF16 stream rate (mode 1) on the 90 workers at the per-core footprints an L2+L3 expert socket needs
for mib in ${MIBS:-3 4.5 5.5 6.5}; do
  b=$(python3 -c "print(int($mib * 2**20))")
  echo "== $mib MiB/core"
  L2R_HUGE=1 /tmp/g4c/l2r/l2r_bw $b 1 2 2 | tail -2
done
