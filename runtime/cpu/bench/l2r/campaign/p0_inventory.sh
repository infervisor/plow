#!/usr/bin/env bash
# p0_inventory.sh <outdir>: P0.1 host inventory (topology, flags, RDT, kernel, tools). Read-only.
O=${1:-/tmp/g4c/l2r/results/p0}
mkdir -p "$O"
{
  echo "## run"; date -u +%FT%TZ; hostname
  echo "## uname"; uname -a
  echo "## cmdline"; cat /proc/cmdline
  echo "## microcode"; grep -m1 microcode /proc/cpuinfo
  echo "## lscpu"; lscpu
  echo "## numactl -H"; numactl -H
  echo "## lscpu -C"; lscpu -C
  echo "## flags of interest"
  grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -xE 'amx_tile|amx_bf16|amx_int8|amx_fp16|avx512_bf16|avx512f|avx512_vnni|cat_l2|cat_l3|cdp_l2|cdp_l3|mba|cqm|cqm_llc|cqm_mbm_total|rdt_a|prfchw|cldemote|movdir64b|serialize|waitpkg' | tr '\n' ' '; echo
  echo "## resctrl"; mount | grep resctrl || echo "not mounted"; ls /sys/fs/resctrl 2>/dev/null
  grep -i resctrl /proc/filesystems || echo "resctrl fs not in kernel"
  echo "## isolation"; for f in isolated nohz_full; do echo "$f: $(cat /sys/devices/system/cpu/$f 2>/dev/null)"; done
  echo "## smt"; cat /sys/devices/system/cpu/smt/control /sys/devices/system/cpu/smt/active 2>/dev/null
  echo "## governor / pstate"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null
  cat /sys/devices/system/cpu/intel_pstate/no_turbo /sys/devices/system/cpu/intel_pstate/status 2>/dev/null
  echo "## numa_balancing"; cat /proc/sys/kernel/numa_balancing
  echo "## thp"; cat /sys/kernel/mm/transparent_hugepage/enabled
  echo "## hugepages"; grep -i huge /proc/meminfo
  echo "## per-node meminfo"; grep -E "MemTotal|MemFree|Shmem:" /sys/devices/system/node/node*/meminfo
  echo "## caches (cpu0)"; for i in /sys/devices/system/cpu/cpu0/cache/index*; do echo "$(cat $i/level) $(cat $i/type) $(cat $i/size) ways=$(cat $i/ways_of_associativity) shared=$(cat $i/shared_cpu_list)"; done
  echo "## l3 sharing sample"; for c in 0 32 64 95; do echo "cpu$c L3 shared: $(cat /sys/devices/system/cpu/cpu$c/cache/index3/shared_cpu_list)"; done
  echo "## smt siblings sample"; for c in 0 1 95; do echo "cpu$c siblings: $(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list) die=$(cat /sys/devices/system/cpu/cpu$c/topology/die_id 2>/dev/null) pkg=$(cat /sys/devices/system/cpu/cpu$c/topology/physical_package_id)"; done
  echo "## tools"; for t in perf turbostat taskset numactl numastat pqos rdmsr wrmsr cpupower; do printf "%s: " $t; command -v $t || echo missing; done
  perf --version 2>/dev/null
  echo "## modules"; lsmod | grep -iE "msr|resctrl|pseudo|plow" || echo "none matching"
  echo "## sudo"; sudo -n true 2>/dev/null && echo "sudo: passwordless" || echo "sudo: no"
  echo "## irq count"; ls /proc/irq | wc -l
  echo "## running load"; uptime; ps -eo pid,pcpu,comm --sort=-pcpu | head -8
} > "$O/inventory.txt" 2>&1
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE > "$O/lscpu_e.txt" 2>&1
echo "$O"
