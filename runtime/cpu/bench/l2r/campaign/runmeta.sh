#!/usr/bin/env bash
# runmeta.sh <run-dir> [key=value...]: write meta.json for a run (checklist global rules).
D=$1; shift; mkdir -p "$D"
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
python3 - "$D" "$@" <<'EOF'
import json, os, subprocess, sys, time, platform
d = sys.argv[1]
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
wt = '/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16'
m = {
    'run_id': os.path.basename(d.rstrip('/')),
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_sha': sh(f'git -C {wt} rev-parse HEAD'),
    'git_dirty': bool(sh(f'git -C {wt} status --porcelain')),
    'kernel': platform.release(),
    'microcode': sh("grep -m1 microcode /proc/cpuinfo | awk '{print $3}'"),
    'cpu': sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(),
    'governor': sh('cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor'),
    'no_turbo': sh('cat /sys/devices/system/cpu/intel_pstate/no_turbo'),
    'smt': sh('cat /sys/devices/system/cpu/smt/control'),
    'numa_balancing': sh('cat /proc/sys/kernel/numa_balancing'),
    'thp': sh('cat /sys/kernel/mm/transparent_hugepage/enabled'),
    'isolation': {
        'housekeeping': '0,1,32,33,64,65,96,97,128,129,160,161',
        'workqueue_cpumask': sh('cat /sys/devices/virtual/workqueue/cpumask'),
        'system_slice': sh('systemctl show -p AllowedCPUs --value system.slice'),
        'irqbalance': sh('systemctl is-active irqbalance'),
        'boot_isolcpus': sh('cat /sys/devices/system/cpu/isolated'),
    },
    'pseudo_lock_module': sh('lsmod | grep pseudo_lock_sram'),
    'qos_holder': bool(sh("pgrep -f 'qos0 sleep'")),
}
for kv in sys.argv[2:]:
    k, _, v = kv.partition('=')
    m[k] = v
json.dump(m, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
print(os.path.join(d, 'meta.json'))
EOF
