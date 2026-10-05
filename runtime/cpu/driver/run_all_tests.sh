#!/bin/bash
set -e

echo "========================================================================"
echo "  INTEL RDT CACHE PSEUDO-LOCKING (L2 & L3 SRAM) VERIFICATION SUITE"
echo "  Target: Intel(R) Xeon(R) 6975P-C (Granite Rapids, 96 Cores / 192 Threads)"
echo "========================================================================"

MODULE_DIR="/home/ec2-user/pseudo_lock_module"
cd "$MODULE_DIR"

echo "[STEP 1] Checking Kernel Module Status..."
if ! lsmod | grep -q pseudo_lock_sram; then
    echo "  Loading pseudo_lock_sram.ko..."
    sudo insmod "$MODULE_DIR/pseudo_lock_sram.ko"
else
    echo "  pseudo_lock_sram.ko is already loaded."
fi

echo "[STEP 2] Verifying Character Devices..."
for dev in /dev/pseudo_lock_l2 /dev/pseudo_lock_l3; do
    if [ -c "$dev" ]; then
        echo "  [OK] Character device exists: $dev"
    else
        echo "  [FAIL] Missing character device: $dev"
        exit 1
    fi
done

echo "[STEP 3] Inspecting Sysfs Properties..."
echo "  --- L2 Pseudo-Lock Device ---"
echo "    Size:    $(cat /sys/class/misc/pseudo_lock_l2/size) bytes ($(($(cat /sys/class/misc/pseudo_lock_l2/size)/1024)) KB)"
echo "    CBM:     $(cat /sys/class/misc/pseudo_lock_l2/cbm)"
echo "    CPU:     $(cat /sys/class/misc/pseudo_lock_l2/cpu)"
echo "    Latency: $(cat /sys/class/misc/pseudo_lock_l2/latency_cycles)"

echo "  --- L3 Pseudo-Lock Device ---"
echo "    Size:    $(cat /sys/class/misc/pseudo_lock_l3/size) bytes ($(($(cat /sys/class/misc/pseudo_lock_l3/size)/1024/1024)) MB)"
echo "    CBM:     $(cat /sys/class/misc/pseudo_lock_l3/cbm)"
echo "    CPU:     $(cat /sys/class/misc/pseudo_lock_l3/cpu)"
echo "    Latency: $(cat /sys/class/misc/pseudo_lock_l3/latency_cycles)"

echo "[STEP 4] Compiling and Executing Userspace SRAM Benchmark..."
gcc -O2 -pthread "$MODULE_DIR/sram_bench.c" -o "$MODULE_DIR/sram_bench"
sudo "$MODULE_DIR/sram_bench"

echo "========================================================================"
echo "  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!"
echo "========================================================================"
