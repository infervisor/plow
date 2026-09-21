#!/usr/bin/env python3
"""Peak GPU memory (MiB) of a server's processes over one bench cell.

    gpu_peak_mem.py <samples-file> <pid>...

The samples are `nvidia-smi --query-compute-apps=timestamp,pid,used_memory
--format=csv,noheader,nounits -lms N` output. Rows of one sample share a timestamp: the listed
pids are summed per timestamp (a TP>1 server is several processes) and the largest sum is
printed. Prints nothing when no sample names one of the pids.
"""
import sys
from collections import defaultdict

pids = set(sys.argv[2:])
by_sample = defaultdict(int)
for line in open(sys.argv[1], errors="replace"):
    f = [x.strip() for x in line.split(",")]
    if len(f) == 3 and f[1] in pids and f[2].isdigit():
        by_sample[f[0]] += int(f[2])
if by_sample:
    print(max(by_sample.values()))
