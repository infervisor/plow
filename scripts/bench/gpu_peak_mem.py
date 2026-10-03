#!/usr/bin/env python3
"""Peak GPU memory (MiB) of a server's processes over one bench cell.

    gpu_peak_mem.py <samples-file> <pid>...
    gpu_peak_mem.py <samples-file> --root-pid <server-pid>

The samples are `nvidia-smi --query-compute-apps=timestamp,pid,used_memory
--format=csv,noheader,nounits -lms N` output. Rows of one sample share a timestamp: the listed
pids are summed per timestamp (a TP>1 server is several processes) and the largest sum is
printed. Prints nothing when no sample names one of the pids.
--root-pid includes descendants still alive at collection end. This is sampled process
allocation, not total board usage or an allocator high-water mark; short peaks may be missed.
"""
import sys
import subprocess
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


if __name__ == '__main__':
    if sys.argv[2:3] == ['--root-pid']:
        table = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
        pids = descendants(sys.argv[3], [line.split() for line in table.splitlines()])
    else:
        pids = set(sys.argv[2:])
    with open(sys.argv[1], errors='replace') as samples:
        value = peak(samples, pids)
    if value is not None:
        print(value)
