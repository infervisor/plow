"""prof_edge.py <dump.csv> <producer inst> <consumer inst>: last producer slices (worker, end) and consumer start
distribution (us after the last producer end)."""
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
p, c = int(sys.argv[2]), int(sys.argv[3])
P = sorted([r for r in rows if int(r["inst"]) == p], key=lambda r: int(r["t1_ns"]))
C = sorted([r for r in rows if int(r["inst"]) == c], key=lambda r: int(r["t0_ns"]))
last = int(P[-1]["t1_ns"])
print("last producers (worker, slice, end us rel last, dur us):",
      [(int(r["worker"]), int(r["slice"]), round((int(r["t1_ns"]) - last) / 1e3, 1), round((int(r["t1_ns"]) - int(r["t0_ns"])) / 1e3, 1)) for r in P[-5:]])
d = [(int(r["t0_ns"]) - last) / 1e3 for r in C]
print("consumer start rel last producer end: first %.1f, 10%% %.1f, median %.1f, 90%% %.1f, last %.1f us"
      % (d[0], d[len(d) // 10], d[len(d) // 2], d[len(d) * 9 // 10], d[-1]))
print("first consumers (worker, start):", [(int(r["worker"]), round(x, 1)) for r, x in zip(C[:5], d[:5])])
# what were the first consumer workers doing right before (their previous packet)
for r in C[:3]:
    w = r["worker"]
    prev = max((x for x in rows if x["worker"] == w and int(x["t1_ns"]) <= int(r["t0_ns"])), key=lambda x: int(x["t1_ns"]))
    print(f"  worker {w}: previous packet inst {prev['inst']} {prev['op']} ended {(int(prev['t1_ns']) - last) / 1e3:.1f} us rel")
