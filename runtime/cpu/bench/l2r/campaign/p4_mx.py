import re, sys
rows = {}
for l in open("/tmp/g4c/l2r/results/p4_matrix/summary.md"):
    m = re.match(r"\| (\S+)\.s(\d+)\.([ABC]) \| \d+ \| ([\d.]+) / ([\d.]+) \(([\d.]+)%\) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \|", l)
    if m:
        ref, s, p = m.group(1), int(m.group(2)), m.group(3)
        rows.setdefault((ref, s), {})[p] = (float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7)), float(m.group(8)), float(m.group(9)), int(m.group(10)))
def k(x):
    r, s = x; a = r.split(".c"); return (a[0], int(a[1]), s)
print("| layer | ctx | seqs | KV MiB/step | B p50/p99 | A p50/p99 | C p50/p99 | C attn µs (GB/s) | C GEMV µs | max spread |")
print("|---|---:|---:|---:|---|---|---|---|---:|---:|")
for key in sorted(rows, key=k):
    v = rows[key]; ref, s = key; a = ref.split(".c")
    f = lambda p: f"{v[p][0]:.0f} / {v[p][1]:.0f}" if p in v else "= B"
    c = v["C"]
    print(f"| {a[0]} | {int(a[1])//1024}K | {s} | {c[5]:.0f} | {f('B')} | {f('A')} | {f('C')} | {c[3]:.0f} ({c[6]}) | {c[4]:.0f} | {max(x[2] for x in v.values()):.1f}% |")
print()
for key in sorted(rows, key=k):
    v = rows[key]
    if "A" in v and ".L0" not in key[0]:
        print(key, "C vs A p50 %.1f%%  p99 C<A %s" % ((1 - v["C"][0] / v["A"][0]) * 100, v["C"][1] < v["A"][1]))
