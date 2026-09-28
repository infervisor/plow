#!/usr/bin/env python3
"""packlog_audit.py server.log [segment...]: tick accounting from PLOW_PF_PACKLOG=1 lines.
Per segment (split at >1 s idle): tokens, device time by tick kind (mixed / prefill-only / decode),
host gap and idle, launch padding, decode steps and rows per step, riders priced against the
pure-launch baseline, and time-weighted slot occupancy. See docs/runtime/throughput-audit.md.
Importable: segments(path) and account(segment) feed scripts/bench/waterfall.py."""
import re, sys, collections
tickre = re.compile(r"PACKLOG TICK t_ms=([\d.]+) prefill_ms=([\d.]+) decode_ms=([\d.]+) did_prefill=(\d) decode_rows=(\d+)(?: steps=(\d+) tokens=(\d+) live=(\d+) prefilling=(\d+))?")
packre = re.compile(r"PACKLOG PACK reqs=(\d+) rows=(\d+) decode_feeds=(\d+) unified=(\w+)")
rre = re.compile(r"PACKLOG R=(\d+) rows=(\d+) bucket=(\d+) chunks=\[([^\]]*)\]")


def segments(path):
    segs = []; cur = None; prev = None; pend_pack = []; pend_r = []
    for line in open(path, errors="replace"):
        m = packre.search(line)
        if m: pend_pack.append((int(m[2]), int(m[3]), m[4] == "true")); continue
        m = rre.search(line)
        if m:
            ch = [int(x) for x in m[4].split(",") if x]
            pend_r.append(dict(n=int(m[1]), rows=int(m[2]), bucket=int(m[3]), riders=sum(1 for c in ch if c == 1), pf=sum(c for c in ch if c > 1)))
            continue
        m = tickre.search(line)
        if not m: continue
        t, pf, dc = float(m[1]), float(m[2]), float(m[3])
        tk = dict(t=t, pf=pf, dc=dc, did=int(m[4]), rows=int(m[5]), steps=int(m[6]) if m[6] else None,
                  tokens=int(m[7]) if m[7] else None, live=int(m[8]) if m[8] else None,
                  prefilling=int(m[9]) if m[9] else None, packs=pend_pack, launches=pend_r)
        pend_pack, pend_r = [], []
        g = None if prev is None else t - prev - pf - dc
        prev = t
        if g is None or g > 1000:
            cur = dict(ticks=[], gap=0.0, idle=0.0, t0=t - pf - dc); segs.append(cur)
        elif g >= 5: cur["idle"] += g
        else: cur["gap"] += max(g, 0)
        cur["ticks"].append(tk); cur["t1"] = t
    return segs


def account(s):
    """One segment's accounting; times in ms."""
    T = s["ticks"]
    wall = s["t1"] - s["t0"]
    # pure launch baseline per bucket (prefill launches with no riders)
    pure = collections.defaultdict(list)
    for tk in T:
        if len(tk["launches"]) == 1 and tk["launches"][0]["riders"] == 0 and tk["pf"] > 0:
            pure[tk["launches"][0]["bucket"]].append(tk["pf"])
    pmed = {b: sorted(v)[len(v)//2] for b, v in pure.items()}
    acc = collections.defaultdict(lambda: collections.Counter())
    pad_ms = 0.0; ride_extra = 0.0; riders_tot = 0; pf_rows = 0; launch_ms = 0.0
    dec_hist = collections.Counter(); tokens = 0
    occ = []
    for tk in T:
        L = tk["launches"]; ride = sum(l["riders"] for l in L)
        kind = "mixed" if ride else ("prefill" if L else None)
        if kind:
            a = acc[kind]; a["n"] += 1; a["ms"] += tk["pf"]; a["pf_rows"] += sum(l["pf"] for l in L); a["riders"] += ride
            for l in L:
                pf_rows += l["pf"]
            # padding: share of the launch spent on bucket rows nobody used (cost ~ bucket)
            for l in L:
                share = tk["pf"] / len(L)
                pad_ms += share * (l["bucket"] - l["rows"]) / l["bucket"]
                launch_ms += share
                if l["riders"] and l["bucket"] in pmed:
                    ride_extra += max(share - pmed[l["bucket"]] * 1.0, 0.0); riders_tot += l["riders"]
        if tk["dc"] > 0.01 or tk["rows"]:
            a = acc["decode"]; a["n"] += 1; a["ms"] += tk["dc"]; a["rows"] += tk["rows"]
            if tk["steps"] is not None:
                a["steps"] += tk["steps"]; a["rowsteps"] += tk["rows"] * tk["steps"]
                dec_hist[(tk["rows"] // 8 * 8, tk["steps"])] += 1
        if tk["tokens"] is not None: tokens += tk["tokens"]
        if tk["live"] is not None: occ.append((tk["pf"] + tk["dc"], tk["live"], tk["prefilling"]))
    return dict(ticks=len(T), wall=wall, tokens=tokens, acc=acc, gap=s["gap"], idle=s["idle"],
                pf_rows=pf_rows, launch_ms=launch_ms, pad_ms=pad_ms, pmed=pmed, riders=riders_tot,
                ride_extra=ride_extra, occ=occ, dec_hist=dec_hist)


def report(si, d):
    wall, tokens, acc = d["wall"], d["tokens"], d["acc"]
    print(f"== segment {si}: {d['ticks']} ticks, wall {wall/1e3:.3f} s, tokens {tokens} ({tokens/wall*1e3:.0f} tok/s)")
    for k in ("mixed", "prefill", "decode"):
        a = acc[k]
        if not a["n"]: continue
        extra = ""
        if k == "decode" and a["steps"]:
            extra = f" steps {a['steps']} ms/step {a['ms']/a['steps']:.2f} rows/step {a['rowsteps']/a['steps']:.1f} ms/row-step {a['ms']/a['rowsteps']*1e3:.1f}us"
        if k != "decode":
            extra = f" pf_rows/t {a['pf_rows']/a['n']:.0f} riders/t {a['riders']/a['n']:.1f}"
        print(f"  {k:8s} n={a['n']:5d} {a['ms']/1e3:7.3f} s {100*a['ms']/wall:5.1f}% mean {a['ms']/a['n']:6.2f} ms{extra}")
    print(f"  host gap {d['gap']/1e3:.3f} s ({100*d['gap']/wall:.1f}%)  idle {d['idle']/1e3:.3f} s ({100*d['idle']/wall:.1f}%)")
    print(f"  prefill rows {d['pf_rows']}; launches {d['launch_ms']/1e3:.3f} s, padding (bucket-rows)/bucket {d['pad_ms']/1e3:.3f} s ({100*d['pad_ms']/wall:.1f}% of wall)")
    print(f"  pure launch median ms by bucket {dict(sorted(d['pmed'].items()))}")
    if d["riders"]: print(f"  riders {d['riders']}: launch time over pure {d['ride_extra']/1e3:.3f} s = {d['ride_extra']/d['riders']*1e3:.1f} us/rider")
    occ = d["occ"]
    if occ:
        tw = sum(o[0] for o in occ) or 1
        print(f"  time-weighted live slots {sum(o[0]*o[1] for o in occ)/tw:.1f}, prefilling {sum(o[0]*o[2] for o in occ)/tw:.1f}")
    if d["dec_hist"]:
        print("  decode ticks (rows//8*8, steps): n", sorted(d["dec_hist"].items()))


if __name__ == "__main__":
    segs = segments(sys.argv[1])
    for si in [int(a) for a in sys.argv[2:]] or range(len(segs)):
        if len(segs[si]["ticks"]) < 20: continue
        report(si, account(segs[si]))
