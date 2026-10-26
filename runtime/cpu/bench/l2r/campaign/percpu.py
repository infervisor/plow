import subprocess, sys, collections
data = sys.argv[1]
out = subprocess.run(["sudo", "perf", "report", "-i", data, "--sort", "cpu,sym", "--stdio"],
                     capture_output=True, text=True).stdout
g = collections.defaultdict(lambda: collections.defaultdict(float))
for l in out.splitlines():
    p = l.split()
    if len(p) < 4 or not p[0].endswith("%"):
        continue
    cpu = int(p[1]); sym = " ".join(p[3:])
    k = "gemv" if sym.startswith("gemv_rows") else "wait" if "WorkerPool>::spawn" in sym else "idle" if "poll_idle" in sym else "other"
    g[cpu][k] += float(p[0][:-1])
for cpu in sorted(g):
    t = sum(g[cpu].values())
    print(cpu, " ".join(f"{k}={100*g[cpu][k]/t:.0f}" for k in ("gemv", "wait", "other", "idle")))
