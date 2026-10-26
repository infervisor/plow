#!/usr/bin/env bash
# p1_kill.sh: P1.1 cleanup after interruption. SIGKILL a locked l2r_lock mid-run; the driver must drop every
# region when the fd closes and remove the CAT partition (lazy: CLOS 0 mask back to full when no lock is live).
O=/tmp/g4c/l2r/results/p1_kill; mkdir -p $O
L2R_LOCK=1 L2R_KERNEL=amx /tmp/g4c/l2r/l2r_lock 1310720 30 C > $O/run.jsonl 2> $O/run.err &
P=$!
sleep 8
echo "--- regions while running (count)"; grep -vc '^#' /sys/class/misc/pseudo_lock/regions
echo "--- L2 mask CLOS0 cpu40 while running: $(sudo python3 /tmp/g4c/l2r/rdmsr.py 40 0xd10) PQR=$(sudo python3 /tmp/g4c/l2r/rdmsr.py 40 0xc8f)"
kill -9 $P; wait $P 2>/dev/null; echo "killed rc=$?"
sleep 1
echo "--- regions after kill (count)"; grep -vc '^#' /sys/class/misc/pseudo_lock/regions
echo "--- L2 mask CLOS0 cpu40 after: $(sudo python3 /tmp/g4c/l2r/rdmsr.py 40 0xd10) PQR=$(sudo python3 /tmp/g4c/l2r/rdmsr.py 40 0xc8f)"
cat /sys/class/misc/pseudo_lock/caps
dmesg | tail -5
