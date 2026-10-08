#!/usr/bin/env python3
"""Peak host memory (MiB) of a CPU server over one bench cell: the CPU twin of gpu_peak_mem.py.

    cpu_peak_mem.py --out FILE --root-pid PID [--container NAME] [--interval-ms 100]

Samples until SIGTERM/SIGINT, then writes the largest per-sample sum of Pss (smaps_rollup) over
the root pid's live descendants plus every process in the named docker container's cgroup (a
containerised server is not a descendant of the `docker run` client). Pss splits shared pages
(a mmap'd checkpoint, forked workers) so a page is counted once across the set. Writes nothing
when no sample saw a process.
"""
import argparse
import os
import signal
import subprocess
import time


def children(pid):
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            return [int(c) for c in f.read().split()]
    except OSError:
        return []


def tree(root):
    out, todo = set(), [root]
    while todo:
        p = todo.pop()
        if p not in out:
            out.add(p)
            todo += children(p)
    return out


def container_id(name):
    r = subprocess.run(["sudo", "-n", "docker", "inspect", "-f", "{{.Id}}", name], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def in_cgroup(cid):
    pids = set()
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/cgroup") as f:
                    if cid in f.read():
                        pids.add(int(d))
            except OSError:
                pass
    return pids


def pss_kib(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--container")
    ap.add_argument("--interval-ms", type=int, default=100)
    a = ap.parse_args()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    cid, peak, n = None, 0, 0
    while not stop:
        pids = tree(a.root_pid)
        if a.container:
            cid = cid or container_id(a.container)
            if cid:
                pids |= in_cgroup(cid)
        vals = [v for v in map(pss_kib, pids) if v is not None]
        if vals:
            peak, n = max(peak, sum(vals)), n + 1
        time.sleep(a.interval_ms / 1000)
    if n:
        with open(a.out, "w") as f:
            f.write(f"{peak / 1024:.0f}\n")


if __name__ == "__main__":
    main()
