#!/usr/bin/env bash
# Token-identity gate for the L2-placement A/B: every arm of every fold must emit the SAME ids.
# Placement moves which workgroup runs which packet and nothing else, so any divergence means the
# workgroup->domain mapping is wrong.
set -uo pipefail
D="${1:-/tmp/l2place}"
for f in A1 B1 A2 B2 A3 B3; do
  [ -f "$D/raw.$f.txt" ] || continue
  ids=$(grep -A 2 -i 'greedy decode' "$D/raw.$f.txt" | sed -n '2p')
  printf "%-3s %s  %s\n" "$f" "$(printf '%s' "$ids" | md5sum | cut -c1-16)" "$(printf '%s' "$ids" | cut -c1-70)"
done
