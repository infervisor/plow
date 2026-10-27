"""turbo_sum.py <turbostat.txt> [label] [workers]: worker-core Bzy_MHz, package/DRAM watts, temperature."""
import statistics as st, sys


def summarize(f, label='', workers=None):
    W = set(workers or [c for c in range(96) if c not in (0, 1, 32, 33, 64, 65)])
    blocks, cur = [], None
    for l in open(f):
        p = l.split()
        if not p:
            continue
        if p[0] == 'CPU':
            cur = {'mhz': [], 'tmp': []}
            blocks.append(cur)
            continue
        if cur is None:
            continue
        if p[0] == '-':
            cur['pkg'] = float(p[5]) if len(p) > 5 else None
            cur['ram'] = float(p[6]) if len(p) > 6 else None
            continue
        c = int(p[0])
        if c in W:
            cur['mhz'].append(float(p[3]))
            if len(p) > 4:
                cur['tmp'].append(float(p[4]))
    b = blocks[1:-1] or blocks
    mhz = sorted(v for x in b for v in x['mhz'])
    tmp = [v for x in b for v in x['tmp']]
    pk = [x['pkg'] for x in b if x.get('pkg')]
    ram = [x['ram'] for x in b if x.get('ram')]
    return (f"{label:24s} worker Bzy_MHz mean {st.mean(mhz):.0f} p5 {mhz[len(mhz) // 20]:.0f} min {mhz[0]:.0f} max {mhz[-1]:.0f}"
            f" | PkgWatt {st.mean(pk):.0f} RAMWatt {st.mean(ram):.1f} | CoreTmp max {max(tmp):.0f} | intervals {len(b)}")


if __name__ == '__main__':
    print(summarize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]))
