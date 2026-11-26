#!/usr/bin/env python3
"""res_diff.py <objs> <name>...: per cubin, per entry REG/STACK/LOCAL (cuobjdump --dump-resource-usage)
base vs hard, plus ptxas -v spill bytes per function from each emit's build.log. Prints every
increase, a count of unchanged/decreased entries, and cubins present on one side only."""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
FN = re.compile(r"^\s*Function (\S+):")
RES = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")


def res(path):
    out, fn = {}, None
    for line in path.read_text(errors="replace").splitlines():
        if m := FN.match(line):
            fn = m.group(1)
        elif fn and (m := RES.search(line)):
            out[fn] = tuple(int(x) for x in m.groups())
            fn = None
    return out


PROPS = re.compile(r"Function properties for (\S+)")
SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
GEN = re.compile(r"Generating .*?(\w+\.cubin)|-o \S*?/(\w+\.cubin)")


def spills(log):
    """{(cubin or '?', function): (stack, spill stores, spill loads)} from ptxas -v lines."""
    out, fn, obj = {}, None, "?"
    if not log.exists():
        return out
    for line in log.read_text(errors="replace").splitlines():
        if m := GEN.search(line):
            obj = m.group(1) or m.group(2)
        if m := PROPS.search(line):
            fn = m.group(1)
        elif fn and (m := SPILL.search(line)):
            out[(obj, fn)] = tuple(int(x) for x in m.groups())
            fn = None
    return out


for name in sys.argv[2:]:
    a_dir, b_dir = root / "base" / name / "assets", root / "hard" / name / "assets"
    print(f"== {name}")
    a_c = {p.name for p in a_dir.glob("*.cubin")}
    b_c = {p.name for p in b_dir.glob("*.cubin")}
    for c in sorted(a_c ^ b_c):
        print(f"  {c}: only in {'base' if c in a_c else 'hard'}")
    for c in sorted(a_c & b_c):
        a, b = res(a_dir / f"{c}.res"), res(b_dir / f"{c}.res")
        up, down, same = [], 0, 0
        for fn in sorted(set(a) | set(b)):
            if fn not in a or fn not in b:
                print(f"  {c} {fn}: only in {'base' if fn in a else 'hard'} {a.get(fn) or b.get(fn)}")
                continue
            ra, rb = a[fn], b[fn]
            if ra == rb:
                same += 1
            elif rb[0] > ra[0] or rb[1] > ra[1] or rb[3] > ra[3]:
                up.append((fn, ra, rb))
            else:
                down += 1
        print(f"  {c}: entries {len(a)}/{len(b)}, same {same}, lower {down}, higher {len(up)}")
        for fn, ra, rb in up:
            print(f"    UP {fn}: REG {ra[0]}->{rb[0]} STACK {ra[1]}->{rb[1]} SHARED {ra[2]}->{rb[2]} LOCAL {ra[3]}->{rb[3]}")
    sa, sb = spills(root / "base" / name / "build.log"), spills(root / "hard" / name / "build.log")
    print(f"  ptxas -v: functions base {len(sa)} hard {len(sb)}")
    for k in sorted(set(sa) & set(sb)):
        if sb[k] != sa[k] and (sb[k][1] > sa[k][1] or sb[k][2] > sa[k][2] or sb[k][0] > sa[k][0]):
            print(f"    UP {k[0]} {k[1]}: stack/spill-st/spill-ld {sa[k]} -> {sb[k]}")
    tot = lambda s: (sum(v[1] for v in s.values()), sum(v[2] for v in s.values()))
    print(f"  ptxas -v total spill stores/loads base {tot(sa)} hard {tot(sb)}")
