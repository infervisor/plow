"""quiet.py [secs]: interrupts + context switches per CPU over a window; summary for housekeeping vs inference cores."""
import sys, time, re
HK = {0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161}
INF = [c for c in range(96) if c not in HK]
SIB = [c + 96 for c in INF]


def irqs():
    lines = open('/proc/interrupts').read().splitlines()
    ncpu = len(lines[0].split())
    tot = [0] * ncpu
    for l in lines[1:]:
        f = l.split()
        for i in range(ncpu):
            if i + 1 < len(f) and f[i + 1].isdigit():
                tot[i] += int(f[i + 1])
    return tot


def ctxt():
    out = {}
    for c in range(192):
        try:
            s = open(f'/sys/devices/system/cpu/cpu{c}/../../../../proc/schedstat').read()
        except Exception:
            break
    for l in open('/proc/schedstat'):
        m = re.match(r'cpu(\d+) (.*)', l)
        if m:
            f = list(map(int, m.group(2).split()))
            out[int(m.group(1))] = f[2]  # sched_switch-ish: yld_count, sched_count field 2
    return out


T = float(sys.argv[1]) if len(sys.argv) > 1 else 10
a, ca = irqs(), ctxt()
time.sleep(T)
b, cb = irqs(), ctxt()
d = [y - x for x, y in zip(a, b)]
dc = {c: cb[c] - ca[c] for c in cb}
for name, s in (('housekeeping', sorted(HK)), ('inference', INF), ('inference SMT siblings', SIB)):
    v = [d[c] / T for c in s]
    w = [dc.get(c, 0) / T for c in s]
    print(f"{name:24s} irq/s mean {sum(v) / len(v):8.1f} max {max(v):8.1f} (cpu {s[v.index(max(v))]})   sched/s mean {sum(w) / len(w):7.1f} max {max(w):7.1f}")
