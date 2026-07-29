#!/usr/bin/env python3
"""glm52_stall_attrib.py -- WHERE THE GATE STALL GOES.

Companion to scripts/glm52_token_attrib.py, which attributes the token to the ops that are
RUNNING.  This one attributes the complement: the per-CU spin (`t_ready - t_arrive`) that
glm52_token_attrib.py reports as one number ("gate-stall 16.77 ms") and never breaks down.

Same inputs (PLOW_TRACE_RAW `.insts.txt` sidecar + `PlowTraceRec[n_stream]` dumps), but it
takes ONE OR MORE rank dumps -- `PLOW_TRACE_ALLRANKS=1` writes `<prefix>.rk<R>.tp<N>.ctx<C>.bin`.

Three questions, three sections, and they are the three candidate mechanisms:

  (1) CHIP-OCCUPANCY DECOMPOSITION.  A workgroup spinning at a closed gate is either
      (a) waiting while NOTHING runs anywhere (true idle -- a serial-chain bubble), or
      (b) waiting while a NARROW op runs (starvation: the op on the critical path cannot
          use the CU), or
      (c) waiting while the chip is FULL (genuine queueing -- its own input is not ready
          and every CU is busy producing something).
      Only (b) is addressable by widening ops; only (c) could ever be addressed by finer
      gating, and (a) by shortening the chain.  The stall is charged to the ops that are
      live during it, so the table reads "op X made 255 other CUs spin for Y ms".

  (2) THE XREDUCE RENDEZVOUS.  d_xreduce_mega waits for its N-1 peers INSIDE the packet
      body, between t_ready and t_end -- so cross-rank arrival skew is invisible in the
      body/stall split above: it is counted as BODY.  Per collective this reconstructs
        t_sig   = t_ready of slice 0        (this rank announces its arrival)
        R       = min body over workgroups  (a workgroup that arrived after the gate opened
                                             does the reduce and nothing else)
        t_open  = min(t_end) - R            (the instant the N-th rank's signal landed)
      and reports peer-wait = t_open - t_sig, plus the spin every OTHER workgroup of the
      packet does inside its own body.

  (3) CROSS-RANK SKEW, measured WITHOUT cross-device clock sync.  The 156 collectives hard-
      synchronise all N ranks, so the work each rank does BETWEEN two consecutive collectives
      is a duration inside that rank's OWN s_memrealtime domain and is directly comparable.
      W[r,k] = t_sig[r,k+1] - t_open[r,k].  The token advances at max_r W[r,k], so the price
      of imbalance is sum_k (max_r W[r,k] - mean_r W[r,k]).
"""
import sys, struct, argparse, re
from collections import defaultdict

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from glm52_token_attrib import OPNAME, parse_insts, eff_wg   # noqa: E402

XREDUCE_OPS = {24, 25, 26, 29}
REC = struct.Struct('<IIIHHQQQ')


def load(path):
    """-> (per_inst, cu_pkts, gmin, gmax).  per_inst[inst] = dict of arrays over workgroups."""
    data = open(path, 'rb').read()
    per = defaultdict(lambda: dict(ta=[], tr=[], te=[], cu=[], slice=[], op=0))
    cu_pkts = defaultdict(list)
    gmin = gmax = None
    for r in range(len(data) // REC.size):
        cu, pc, inst, op, slc, ta, tr, te = REC.unpack_from(data, r * REC.size)
        if ta == 0 and tr == 0 and te == 0:
            continue
        d = per[inst]
        d['ta'].append(ta); d['tr'].append(tr); d['te'].append(te)
        d['cu'].append(cu); d['slice'].append(slc); d['op'] = op
        cu_pkts[cu].append((pc, inst, op, ta, tr, te))
        gmin = ta if gmin is None else min(gmin, ta)
        gmax = te if gmax is None else max(gmax, te)
    for c in cu_pkts:
        cu_pkts[c].sort()
    return per, cu_pkts, gmin, gmax


def occupancy_decomposition(per, cu_pkts, gmin, gmax, ncu, tick_ms, insts, topn=14):
    """Sweep the timeline once.  At every instant we know how many workgroups are in a BODY
    (`nb`) and how many are STALLED at a gate (`ns`); nb + ns + (not yet started / finished)
    = ncu.  Charge each elementary interval's ns*dt of spin to the bucket named by nb, and
    to the op instances that are live in it (split equally, the same convention
    glm52_token_attrib.py uses for the positive side of the ledger)."""
    ev = []                                            # (t, kind, inst)  kind: body/stall +-1
    for inst, d in per.items():
        for ta, tr, te in zip(d['ta'], d['tr'], d['te']):
            if tr > ta:
                ev.append((ta, 0, +1, inst)); ev.append((tr, 0, -1, inst))
            if te > tr:
                ev.append((tr, 1, +1, inst)); ev.append((te, 1, -1, inst))
    ev.sort(key=lambda x: (x[0], -x[2]))

    nb = ns = 0
    live = defaultdict(int)                            # inst -> workgroups currently in body
    buckets = [(0, 0), (1, 1), (2, 4), (5, 32), (33, 128), (129, 224), (225, 10**9)]
    bkt = [0.0] * len(buckets)
    blame = defaultdict(float)                         # op -> stall CU-ticks caused
    # The "exactly 1 CU busy" bucket has TWO tenants with completely different fixes: a packet
    # that is intrinsically 1-workgroup (widen the emit) and the last straggler of a 256-wide
    # packet (§6a's +-19% tail, which widening makes WORSE, §6b-i). Split them.
    solo = defaultdict(float)                          # (op, is_1wg_packet) -> stall CU-ticks
    npkt_wg = {i: len(d['tr']) for i, d in per.items()}
    prev = ev[0][0]
    for t, kind, delta, inst in ev:
        if t > prev and ns:
            dt = t - prev
            spin = ns * dt
            for i, (lo, hi) in enumerate(buckets):
                if lo <= nb <= hi:
                    bkt[i] += spin; break
            if live:
                share = spin / len(live)
                for j in live:
                    blame[per[j]['op']] += share
                if nb == 1:
                    j = next(iter(live))
                    solo[(per[j]['op'], npkt_wg[j] <= 1)] += spin
            else:
                blame[-1] += spin                      # -1 = nothing live: true idle bubble
        if t > prev:
            prev = t
        if kind == 0:
            ns += delta
        else:
            nb += delta
            live[inst] += delta
            if live[inst] == 0:
                del live[inst]

    tot = sum(bkt)
    print("\n### (1) THE GATE STALL, BY WHAT THE REST OF THE CHIP WAS DOING")
    print(f"# total spin {tot*tick_ms/ncu:8.3f} ms/CU   ({tot*tick_ms:.1f} CU-ms over {ncu} CUs)")
    print(f"{'CUs in a body':<22} {'ms/CU':>9} {'% of stall':>11}")
    names = ['0  (TRUE IDLE)', '1', '2-4', '5-32', '33-128', '129-224', '225-256 (chip full)']
    for nmm, v in zip(names, bkt):
        print(f"{nmm:<22} {v*tick_ms/ncu:>9.3f} {100*v/tot if tot else 0:>10.1f}%")

    print("\n### (1b) WHICH OP WAS RUNNING WHILE THE OTHERS SPUN  (the starvation bill)")
    print(f"{'live op':<22} {'stall ms/CU':>12} {'%stall':>8} {'pkts':>6} {'effwg':>7} {'ownms':>8}")
    own = defaultdict(float); npk = defaultdict(int); ewg = defaultdict(float)
    for inst, d in per.items():
        op = d['op']
        own[op] += (max(d['te']) - min(d['tr']))
        npk[op] += 1
        b = [te - tr for tr, te in zip(d['tr'], d['te'])]
        m = max(b) if b else 0
        ewg[op] += (sum(b) / m) if m else len(b)
    rows = sorted(blame.items(), key=lambda kv: -kv[1])
    for op, v in rows[:topn]:
        nm = 'IDLE (nothing live)' if op == -1 else OPNAME.get(op, '?%d' % op)
        n = npk.get(op, 0)
        print(f"{nm:<22} {v*tick_ms/ncu:>12.3f} {100*v/tot if tot else 0:>7.1f}% {n:>6} "
              f"{(ewg[op]/n if n else 0):>7.1f} {(own[op]*tick_ms/n if n else 0)*1000:>7.1f}u")

    st = sum(solo.values())
    if st:
        print("\n### (1c) THE '1 CU BUSY' BUCKET: intrinsically narrow packet, or straggler tail?")
        print(f"{'live op':<22} {'kind':<28} {'ms/CU':>8} {'%':>6}")
        for (op, is1), v in sorted(solo.items(), key=lambda kv: -kv[1])[:10]:
            print(f"{OPNAME.get(op,'?%d'%op):<22} "
                  f"{'1-WORKGROUP PACKET' if is1 else 'straggler tail of a wide op':<28} "
                  f"{v*tick_ms/ncu:>8.3f} {100*v/st:>5.1f}%")
        s1 = sum(v for (op, is1), v in solo.items() if is1)
        print(f"  1-workgroup packets (widen the emit) : {s1*tick_ms/ncu:.3f} ms/CU ({100*s1/st:.1f}%)")
        print(f"  straggler tail of a WIDE packet      : {(st-s1)*tick_ms/ncu:.3f} ms/CU "
              f"({100*(st-s1)/st:.1f}%)  <- widening makes this WORSE (§6b-i)")
    return tot


def per_cu_split(cu_pkts, gmin, gmax, tick_ms, ncu):
    body = stall = gap = 0
    for cu, pkts in cu_pkts.items():
        body += sum(te - tr for _, _, _, ta, tr, te in pkts)
        stall += sum(tr - ta for _, _, _, ta, tr, te in pkts)
        last = None
        for _, _, _, ta, tr, te in pkts:
            if last is not None and ta > last:
                gap += ta - last
            last = te
        gap += (pkts[0][3] - gmin) + (gmax - pkts[-1][5])
    print(f"# per-CU mean: body {body*tick_ms/ncu:.3f} ms | gate-stall {stall*tick_ms/ncu:.3f} ms"
          f" | gap/launch {gap*tick_ms/ncu:.3f} ms   (span {(gmax-gmin)*tick_ms:.3f} ms)")
    return body, stall, gap


def xreduce_detail(per, tick_ms, ncu, label=''):
    """Reconstruct each collective's rendezvous.  Returns {inst: (t_sig, t_open, R)}."""
    out = {}
    tot_peer = tot_spin = tot_body = 0.0
    rows = []
    for inst, d in sorted(per.items()):
        if d['op'] not in XREDUCE_OPS:
            continue
        body = [te - tr for tr, te in zip(d['tr'], d['te'])]
        R = min(body)
        t_open = min(d['te']) - R
        sig = [tr for tr, s in zip(d['tr'], d['slice']) if s == 0]
        t_sig = sig[0] if sig else min(d['tr'])
        peer = t_open - t_sig
        spin = sum(max(0, t_open - tr) for tr in d['tr'])
        tot_peer += peer; tot_spin += spin; tot_body += sum(body)
        rows.append((inst, peer, R, spin, max(d['tr']) - min(d['tr'])))
        out[inst] = (t_sig, t_open, R)
    if not rows:
        return out
    n = len(rows)
    pk = sorted(r[1] for r in rows)
    print(f"\n### (2) THE XREDUCE RENDEZVOUS {label} -- {n} collectives")
    print(f"# body total          {tot_body*tick_ms/ncu:8.3f} ms/CU   (counted as BODY, not stall)")
    print(f"#   of which spin     {tot_spin*tick_ms/ncu:8.3f} ms/CU   waiting for the N-th rank")
    print(f"#   of which reduce   {(tot_body-tot_spin)*tick_ms/ncu:8.3f} ms/CU")
    print(f"# peer-wait on the CRITICAL PATH (t_open - t_sig, summed): {tot_peer*tick_ms:8.3f} ms/token")
    print(f"#   per collective: median {pk[n//2]*tick_ms*1e3:7.2f} us  p10 {pk[n//10]*tick_ms*1e3:7.2f}"
          f"  p90 {pk[9*n//10]*tick_ms*1e3:7.2f}  max {pk[-1]*tick_ms*1e3:7.2f}")
    rr = sorted(r[2] for r in rows)
    print(f"#   local reduce  R: median {rr[n//2]*tick_ms*1e3:7.2f} us   (bare all-reduce is 0.626 us @TP4)")
    ls = sorted(r[4] for r in rows)
    print(f"#   local arrival spread max(t_ready)-min(t_ready): median {ls[n//2]*tick_ms*1e3:7.2f} us")
    # Independent check on the estimator: if a real peer wait existed, EARLY workgroups would
    # carry it in their body and late ones would not, so max(body) >> min(body).  A flat body
    # distribution means the packet is doing the local reduce and nothing else.
    bmin = bmed = bmax = []
    allb = []
    for inst, d in sorted(per.items()):
        if d['op'] not in XREDUCE_OPS:
            continue
        b = sorted(te - tr for tr, te in zip(d['tr'], d['te']))
        allb.append((b[0], b[len(b) // 2], b[-1]))
    if allb:
        k = len(allb) // 2
        s = sorted(allb, key=lambda x: x[1])[k]
        print(f"#   body over workgroups (median collective): min {s[0]*tick_ms*1e3:6.2f}  "
              f"med {s[1]*tick_ms*1e3:6.2f}  max {s[2]*tick_ms*1e3:6.2f} us "
              f"-- flat => no peer wait, skewed => the early arrivals are paying it")
    return out


def cross_rank(ranks, tick_ms):
    """W[r,k] = t_sig[r,k+1] - t_open[r,k]: the work rank r does between collective k and k+1,
    entirely inside rank r's own clock domain.  The token advances at max_r W[r,k]."""
    insts = sorted(set.intersection(*[set(x) for x in ranks]))
    if len(ranks) < 2 or len(insts) < 2:
        return
    N = len(ranks)
    print(f"\n### (3) CROSS-RANK ARRIVAL SKEW -- {N} ranks, {len(insts)-1} inter-collective windows")
    tot_max = tot_mean = 0.0
    straggler = defaultdict(int); leader = defaultdict(int)
    worst = []
    for k in range(len(insts) - 1):
        a, b = insts[k], insts[k + 1]
        W = [ranks[r][b][0] - ranks[r][a][1] for r in range(N)]     # t_sig(k+1) - t_open(k)
        if min(W) < 0:
            continue
        mx, mn = max(W), sum(W) / N
        tot_max += mx; tot_mean += mn
        straggler[W.index(mx)] += 1; leader[W.index(min(W))] += 1
        worst.append((mx - mn, a, b, W))
    print(f"# sum_k max_r W  = {tot_max*tick_ms:8.3f} ms      (what the token actually pays)")
    print(f"# sum_k mean_r W = {tot_mean*tick_ms:8.3f} ms      (a perfectly balanced TP4 rank)")
    print(f"# SKEW COST      = {(tot_max-tot_mean)*tick_ms:8.3f} ms/token "
          f"({100*(tot_max-tot_mean)/tot_max if tot_max else 0:.1f}% of the inter-collective time)")
    print(f"# straggler rank counts {dict(sorted(straggler.items()))}  "
          f"leader {dict(sorted(leader.items()))}   (uniform => jitter, concentrated => structural)")
    worst.sort(reverse=True)
    print(f"{'window':<16} {'skew us':>9}   per-rank W (us)")
    for dv, a, b, W in worst[:8]:
        print(f"{str(a)+'->'+str(b):<16} {dv*tick_ms*1e3:>9.2f}   "
              + ' '.join(f'{w*tick_ms*1e3:8.2f}' for w in W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('insts')
    ap.add_argument('bins', nargs='+')
    ap.add_argument('--traced-ms', type=float, default=0.0)
    ap.add_argument('--ghz', type=float, default=0.1)
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    insts = parse_insts(a.insts)
    loaded = [load(p) for p in a.bins]
    per0, cu0, gmin0, gmax0 = loaded[0]
    span = gmax0 - gmin0
    ghz = span / (a.traced_ms * 1e6) if a.traced_ms > 0 else a.ghz
    tick_ms = 1.0 / (ghz * 1e6)                        # ticks -> ms
    ncu = len(cu0)

    print(f"\n############ GLM-5.2 decode GATE-STALL attribution {a.label}")
    print(f"# {len(a.bins)} rank dump(s), {ncu} workgroups, {len(per0)} op instances on rank 0")
    print(f"# span {span*tick_ms:.3f} ms (clock {ghz*1000:.1f} MHz"
          + (f", calibrated on traced-ms={a.traced_ms}" if a.traced_ms > 0 else "") + ")")
    per_cu_split(cu0, gmin0, gmax0, tick_ms, ncu)
    tot_body = sum(sum(te - tr for tr, te in zip(d['tr'], d['te'])) for d in per0.values())
    print(f"# PERFECT-PACKING FLOOR sum(body)/{ncu} = {tot_body*tick_ms/ncu:.3f} ms "
          f"({100*tot_body/ncu/span:.1f}% of the step) -- the token if every packet were "
          f"balanced over all {ncu} CUs and no dependency ever stalled one")

    occupancy_decomposition(per0, cu0, gmin0, gmax0, ncu, tick_ms, insts)
    rz = [xreduce_detail(p, tick_ms, len(c), label=f'rank{i}' if len(loaded) > 1 else '')
          for i, (p, c, _, _) in enumerate(loaded)]
    cross_rank(rz, tick_ms)


if __name__ == '__main__':
    main()
