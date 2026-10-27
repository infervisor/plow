#!/usr/bin/env bash
# p1_long.sh: P1 long runs, scenario E (DRAM KV stream + 64 KiB q block), 1.25 MiB slice, 12 lock ways, AMX.
#   plain 600 s, plain 1800 s, locked 600 s; per-core first/last 10% step time gives drift.
set -u
O=/tmp/g4c/l2r/results/p1_long; mkdir -p $O
SZ=1310720
L2R_LOCK=0 L2R_KERNEL=amx /tmp/g4c/l2r/l2r_lock $SZ 600 E > $O/l0.600.jsonl 2> $O/l0.600.err; echo "plain 600 rc=$?"
L2R_LOCK=1 L2R_KERNEL=amx /tmp/g4c/l2r/l2r_lock $SZ 600 E > $O/l1.600.jsonl 2> $O/l1.600.err; echo "locked 600 rc=$?"
L2R_LOCK=0 L2R_KERNEL=amx /tmp/g4c/l2r/l2r_lock $SZ 1800 E > $O/l0.1800.jsonl 2> $O/l0.1800.err; echo "plain 1800 rc=$?"
echo LONG_DONE
