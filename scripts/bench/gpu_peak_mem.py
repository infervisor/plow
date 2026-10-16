#!/usr/bin/env python3
"""Peak GPU memory (MiB) of a server's processes over one bench cell.

    gpu_peak_mem.py <samples-file> <pid>...
    gpu_peak_mem.py <samples-file> --root-pid <server-pid>
    gpu_peak_mem.py --amd --out FILE [--interval-ms 100]

The samples are `nvidia-smi --query-compute-apps=timestamp,pid,used_memory
--format=csv,noheader,nounits -lms N` output. Rows of one sample share a timestamp: the listed
pids are summed per timestamp (a TP>1 server is several processes) and the largest sum is
printed. Prints nothing when no sample names one of the pids.
--root-pid includes descendants still alive at collection end. This is sampled process
allocation, not total board usage or an allocator high-water mark; short peaks may be missed.
--amd has no per-process attribution (amd-smi lists the same pids under every GPU): it samples
`amd-smi metric --mem-usage --json` (~0.7 s a call) for the GPUs ROCR_VISIBLE_DEVICES /
HIP_VISIBLE_DEVICES name (all when unset) until SIGTERM, then writes the largest per-sample sum
of used VRAM to --out, like cpu_peak_mem.py. Board usage: the idle ~300 MiB per card is included.
"""
import argparse
import json
import os
import signal
import sys
import subprocess
import time
from collections import defaultdict


def descendants(root, process_table):
    pids = {str(root)}
    while True:
        children = {pid for pid, parent in process_table if parent in pids}
        if children <= pids:
            return pids
        pids |= children


def peak(samples, pids):
    by_sample = defaultdict(int)
    for line in samples:
        f = [x.strip() for x in line.split(",")]
        if len(f) == 3 and f[1] in pids and f[2].isdigit():
            by_sample[f[0]] += int(f[2])
    return max(by_sample.values()) if by_sample else None


def amd_used_mib(gpus):
    cmd = ["amd-smi", "metric", "--mem-usage", "--json"] + (["-g", *gpus] if gpus else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return sum(g["mem_usage"]["used_vram"]["value"] for g in json.loads(r.stdout)["gpu_data"])
    except (ValueError, KeyError, TypeError):
        return None


def amd_sample(out, interval_ms):
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    visible = os.environ.get("ROCR_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES") or ""
    gpus = [g for g in visible.split(",") if g]
    best, n = 0, 0
    while not stop:
        value = amd_used_mib(gpus)
        if value is not None:
            best, n = max(best, value), n + 1
        time.sleep(interval_ms / 1000)
    if n:
        with open(out, "w") as f:
            f.write(f"{best}\n")


if __name__ == '__main__':
    if sys.argv[1:2] == ['--amd']:
        ap = argparse.ArgumentParser()
        ap.add_argument("--amd", action="store_true")
        ap.add_argument("--out", required=True)
        ap.add_argument("--interval-ms", type=int, default=100)
        a = ap.parse_args()
        amd_sample(a.out, a.interval_ms)
        sys.exit(0)
    if sys.argv[2:3] == ['--root-pid']:
        table = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
        pids = descendants(sys.argv[3], [line.split() for line in table.splitlines()])
    else:
        pids = set(sys.argv[2:])
    with open(sys.argv[1], errors='replace') as samples:
        value = peak(samples, pids)
    if value is not None:
        print(value)
