#!/usr/bin/env python3
"""glm52_token_attrib.py — WHOLE-TOKEN per-opcode attribution of a GLM-5.2 decode step.

Companion to scripts/glm52_trace_analyze.py (which profiles ONE block). Same inputs:
the PLOW_TRACE_RAW `.insts.txt` sidecar + the rank-0 `PlowTraceRec[n_stream]` dump written
by runtime/tests/glm52_decode.c.

Produces the §7-shaped budget for GLM:
  1. per-opcode ms/token that SUMS TO THE TRACED STEP (fractional timeline attribution),
  2. the §7b starvation census (what fraction of packets/time runs on <=4 of 256 CUs),
  3. HBM bytes + achieved GB/s per opcode against the 6200 GB/s measured ceiling.

Attribution method (stated because it is the whole point):
  * Each op INSTANCE (one `inst`) owns the wall interval [min(t_ready), max(t_end)] --
    the window in which some workgroup of that op is doing body work.
  * The step span [min(t_arrive), max(t_end)] is swept; every elementary interval is
    divided EQUALLY among the op instances live in it. Intervals with nothing live are
    charged to IDLE/GATE-STALL.
  * Therefore sum(per-op ms) + idle == traced step span, exactly. No double counting,
    and concurrency (GLM_MOE_CORESIDENT) shows up as a real reduction, not as overlap.

Clock: s_memrealtime, 100 MHz on gfx950 (10 ns/tick) -- calibrated in glm52_trace_analyze.py
against traced-ms. --ghz overrides; --traced-ms recalibrates from this run's own span.
"""
import sys, struct, argparse
from collections import defaultdict

OPNAME = {
 0:"NOP",1:"RmsNorm",2:"RowRms",3:"HeadNormRope",4:"Residual",5:"GLU",6:"Embed",7:"Softcap",
 8:"Gemm",9:"GemmNorm",10:"Gemv",11:"FlashPrefill",12:"FlashDecode",13:"FlashMerge",
 14:"GemmSmall",15:"GemmMed",16:"NormResidual",17:"Argmax",18:"ArgmaxFin",19:"GemvGlu",
 20:"GemmGlu",21:"AddNorm",22:"GemvQkv",23:"NormResidualNorm",24:"XReduce",25:"XReduceScatter",
 26:"XAllGather",27:"XFlashMerge",28:"XArgmaxFin",29:"XReduce2",30:"GemvFp8",31:"GemvGluFp8",
 32:"QuantFp8",33:"GemmFp8",34:"GemmMedFp8",35:"GemmSmallFp8",36:"GemmGluFp8",37:"HeadNormRopeFp8",
 38:"FlashDecodeFp8",39:"FlashPrefillFp8",40:"MoeRouter",41:"MoeExpertGlu",42:"MoeExpertDown",
 43:"MoeCombine",44:"GemvFp8Blk",45:"MoeExpertGluFp8Blk",46:"MoeExpertDownFp8Blk",47:"DenseGluFp8Blk",
 48:"MoeGroupGluFp8Blk",49:"MoeGroupDownFp8Blk",50:"FlashMlaDecode",51:"FlashMlaPrefill",
 52:"OUvFold",53:"AttnSelect",54:"FlashGatherDecode",55:"FlashGatherPrefill",56:"MoeRouterTopk",
 57:"MlaMergeFold",58:"IndexScore",59:"IndexSelect",60:"LayerNorm",78:"GemvSz",79:"GemvGluSz",
 80:"GemvArgmax",91:"GemvMxfp4",92:"GemvGluMxfp4",
}
PEAK_GBs = 6200.0   # knob-contract §5: MEASURED HBM ceiling on this part, not the 8 TB/s spec.

EXPERT_OPS   = {45, 46}        # per-expert block-fp8 (i[1],i[2] are the two dims, 1 expert)
GROUP_OPS    = {48, 49}        # grouped: i[0]=top_k experts inside one packet
FLASH_LATENT = {12, 38, 50, 54}
MLA_ROW_B    = 1152            # kv_lora 512 + qk_rope 64, bf16 => 576*2 B per position


def parse_insts(path):
    insts = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        head, rest = line.split('|', 1)
        parts = head.split()
        inst = int(parts[0]); op = int(parts[1]); blocks = int(parts[2])
        opnd_str, tail = rest.rsplit('|', 1)
        opnds = []
        for tok in opnd_str.split():
            if not tok.startswith('t'):
                continue
            k, val = tok.split('=', 1)
            idx, name, nb = val.split(':')
            opnds.append((k, int(idx), name, int(nb)))
        i = [int(x) for x in tail.split()]
        insts[inst] = dict(op=op, blocks=blocks, opnds=opnds, i=i)
    return insts


def inst_layer(rec):
    lyr = -1
    for _, _, name, _ in rec['opnds']:
        if name.startswith('model.layers.'):
            lyr = max(lyr, int(name.split('.')[2]))
        elif name.startswith('kv.'):
            lyr = max(lyr, int(name.split('.')[1]))
    return lyr


def eff_wg(d):
    """EFFECTIVE workgroup width of a packet.

    Counting workgroups that merely *recorded* the packet overstates occupancy badly: a
    256-workgroup MLA_MERGE_FOLD has 240 workgroups that fall straight out of the work loop
    in ~4 us while 16 grind for ~107 us. sum(body)/max(body) is the width a perfectly
    balanced packet of the same total work would need -- 16, not 256."""
    b = d['body']
    m = max(b)
    return (sum(b) / m) if m > 0 else float(len(b))


def blk_scale_bytes(n, k, blk=128):
    """f32 [128,128] block-scale grid bytes for an (n,k) fp8 weight."""
    return ((n + blk - 1) // blk) * ((k + blk - 1) // blk) * 4


def hbm_bytes(rec, ctx, tp):
    """HBM bytes this op instance streams. Weight operands are read in full; activations
    are excluded (they are KB-scale and live in cache across the serial chain).

    Special cases, because the declared tensor size is NOT what is read:
      * expert ops 45/46 (+48/49 grouped): t[3]/t[4] are [E][3] u64 POINTER TABLES
        (6 KB), not weights. Real bytes come off i[1],i[2] (and i[0]=top_k when grouped).
      * flash MLA: kv.* is declared at max_ctx; only `ctx` rows are touched.
    """
    op = rec['op']
    i = rec['i']
    if op in EXPERT_OPS or op in GROUP_OPS:
        n, k = i[1], i[2]                       # glu: (imoe_e, h)   down: (h, imoe_e)
        per = n * k * (2 if op in (45, 48) else 1)   # glu reads BOTH gate and up
        per += blk_scale_bytes(n, k) * (2 if op in (45, 48) else 1)
        mult = i[0] if op in GROUP_OPS else 1
        return per * mult
    if op in FLASH_LATENT:
        return ctx * MLA_ROW_B
    if op in {13, 27}:                          # merge of nsplit partials: fixed-cost bound
        return 0
    if op == 6:                                 # Embed: ONE row of the table, not the table
        return i[1] * 2 if len(i) > 1 and i[1] else 0
    if op in {24, 25, 26, 29}:                  # collectives: FABRIC bytes, not HBM. Priced apart.
        return 0
    b = 0
    for _, _, name, nb in rec['opnds']:
        if name.startswith('act.') or name.startswith('in.') or name.startswith('kv.'):
            continue
        b += nb
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('insts'); ap.add_argument('bin')
    ap.add_argument('--ctx', type=int, default=0)
    ap.add_argument('--tp', type=int, default=4)
    ap.add_argument('--ghz', type=float, default=0.1)
    ap.add_argument('--traced-ms', type=float, default=0.0,
                    help='traced-ms printed by glm52_decode for THIS dump; recalibrates the clock')
    ap.add_argument('--label', default='')
    ap.add_argument('--csv', default='')
    a = ap.parse_args()
    if a.ctx == 0 and 'ctx' in a.bin:
        try: a.ctx = int(a.bin.split('ctx')[1].split('.')[0])
        except Exception: a.ctx = 1024

    insts = parse_insts(a.insts)
    data = open(a.bin, 'rb').read()
    n = len(data) // 40
    per = defaultdict(lambda: dict(ready=[], end=[], arrive=[], cus=set(), body=[]))
    cu_pkts = defaultdict(list)
    gmin = None; gmax = None
    nrec = 0
    for r in range(n):
        cu, pc, inst, op, slc, ta, tr, te = struct.unpack_from('<IIIHHQQQ', data, r*40)
        if ta == 0 and tr == 0 and te == 0:
            continue
        nrec += 1
        d = per[inst]
        d['ready'].append(tr); d['end'].append(te); d['arrive'].append(ta); d['cus'].add(cu)
        d['body'].append(te - tr)
        d['op'] = op
        cu_pkts[cu].append((pc, inst, op, ta, tr, te))
        gmin = ta if gmin is None else min(gmin, ta)
        gmax = te if gmax is None else max(gmax, te)
    span = gmax - gmin

    ghz = a.ghz
    if a.traced_ms > 0:
        ghz = span / (a.traced_ms * 1e6)
    tick_us = 1000.0 * ghz
    def us(t): return t / tick_us
    def ms(t): return t / (tick_us * 1000.0)

    # ---------- fractional timeline attribution ----------
    ev = []
    for inst, d in per.items():
        t0 = min(d['ready']); t1 = max(d['end'])
        if t1 <= t0: t1 = t0 + 1        # a zero-length packet still occupies one tick
        ev.append((t0, +1, inst)); ev.append((t1, -1, inst))
    ev.sort(key=lambda x: (x[0], -x[1]))
    live = set(); attr = defaultdict(float); idle = 0.0
    prev = ev[0][0] if ev else 0
    for t, delta, inst in ev:
        if t > prev:
            dt = t - prev
            if live:
                share = dt / len(live)
                for j in live: attr[j] += share
            else:
                idle += dt
            prev = t
        if delta > 0: live.add(inst)
        else: live.discard(inst)
    # anything before the first t_ready is pure launch/gate stall
    lead = (min(e[0] for e in ev) - gmin) if ev else 0
    idle += lead

    # ---------- per-CU body / stall / gap (the §7 decomposition) ----------
    body_cu = []; stall_cu = []; gap_cu = []
    for cu, pkts in cu_pkts.items():
        pkts.sort()
        body = sum(te - tr for _, _, _, ta, tr, te in pkts)
        stall = sum(tr - ta for _, _, _, ta, tr, te in pkts)
        gap = 0
        last = None
        for _, _, _, ta, tr, te in pkts:
            if last is not None and ta > last: gap += ta - last
            last = te
        head = pkts[0][3] - gmin
        tail = gmax - pkts[-1][5]
        body_cu.append(body); stall_cu.append(stall); gap_cu.append(gap + head + tail)
    ncu = len(cu_pkts)

    # ---------- aggregate by opcode ----------
    agg = defaultdict(lambda: dict(npk=0, t=0.0, wall=0.0, wg=0, ewg=0.0, bytes=0, dur=[]))
    for inst, d in per.items():
        rec = insts.get(inst)
        if rec is None: continue
        op = d['op']
        t0 = min(d['ready']); t1 = max(d['end'])
        g = agg[op]
        g['npk'] += 1
        g['t'] += attr[inst]
        g['wall'] += max(t1 - t0, 1)
        g['wg'] += len(d['cus'])
        g['ewg'] += eff_wg(d)
        g['bytes'] += hbm_bytes(rec, a.ctx, a.tp)
        g['dur'].append(max(t1 - t0, 1))

    tot_attr = sum(g['t'] for g in agg.values())
    print(f"\n############ GLM-5.2 decode token attribution {a.label}")
    print(f"# ctx={a.ctx} TP={a.tp} | {nrec} wg-packets, {len(per)} op instances, {ncu} workgroups")
    print(f"# step span {ms(span):.3f} ms  (clock {ghz*1000:.1f} MHz"
          + (f", calibrated on traced-ms={a.traced_ms}" if a.traced_ms > 0 else ")") + ")")
    print(f"# attributed {ms(tot_attr):.3f} ms + idle/gate {ms(idle):.3f} ms = {ms(tot_attr+idle):.3f} ms")
    print(f"# per-CU mean: body {ms(sum(body_cu)/ncu):.3f} ms | gate-stall {ms(sum(stall_cu)/ncu):.3f} ms"
          f" | gap/launch {ms(sum(gap_cu)/ncu):.3f} ms")

    print(f"\n{'opcode':<22} {'pkts':>5} {'ms':>7} {'%tok':>6} {'us/pkt':>8} {'wg':>5} "
          f"{'effwg':>6} {'GB':>7} {'pktGB/s':>8} {'%peak':>6} {'tokGB/s':>8} {'%peak':>6}")
    rows = []
    for op, g in sorted(agg.items(), key=lambda kv: -kv[1]['t']):
        # pkt-GB/s  = what ONE packet achieves (bytes/wall)  -> judge the kernel against its roofline
        # tok-GB/s  = machine utilisation while this op owns the token (bytes/attributed ms)
        #             -> differs only where packets of the op run CONCURRENTLY (co-resident experts)
        gbs = (g['bytes'] / (us(g['wall']) * 1e3)) if g['wall'] > 0 and g['bytes'] else 0.0
        tgbs = (g['bytes'] / (ms(g['t']) * 1e6)) if g['t'] > 0 and g['bytes'] else 0.0
        rows.append((OPNAME.get(op, '?%d' % op), op, g['npk'], ms(g['t']), 100*g['t']/(tot_attr+idle),
                     us(g['wall'])/g['npk'], g['wg']/g['npk'], g['ewg']/g['npk'],
                     g['bytes']/1e9, gbs, 100*gbs/PEAK_GBs, tgbs, 100*tgbs/PEAK_GBs))
    for nm, op, npk, mst, pct, uspk, wg, ewg, gb, gbs, pk, tgbs, tpk in rows:
        print(f"{nm:<22} {npk:>5} {mst:>7.3f} {pct:>5.1f}% {uspk:>8.2f} {wg:>5.0f} {ewg:>6.1f} "
              f"{gb:>7.3f} {gbs:>8.0f} {pk:>5.1f}% {tgbs:>8.0f} {tpk:>5.1f}%")
    print(f"{'IDLE / GATE-STALL':<22} {'':>5} {ms(idle):>7.3f} {100*idle/(tot_attr+idle):>5.1f}%")
    tot_gb = sum(r[8] for r in rows)
    print(f"{'TOTAL':<22} {sum(r[2] for r in rows):>5} {ms(tot_attr+idle):>7.3f} {100.0:>5.1f}% "
          f"{'':>8} {'':>5} {'':>6} {tot_gb:>7.3f} {'':>8} {'':>6} "
          f"{tot_gb/ms(tot_attr+idle)/1e6:>8.0f} {100*tot_gb/ms(tot_attr+idle)/1e6/PEAK_GBs:>5.1f}%")
    print(f"# HBM roofline: {tot_gb:.2f} GB/GPU/token @ {PEAK_GBs:.0f} GB/s = "
          f"{tot_gb/PEAK_GBs*1e3:.2f} ms floor;  measured {ms(tot_attr+idle):.2f} ms = "
          f"{ms(tot_attr+idle)/(tot_gb/PEAK_GBs*1e3):.1f}x the floor")

    # ---------- §7b starvation census ----------
    for tag, width in (("DISPATCHED workgroups", lambda d: float(len(d['cus']))),
                       ("EFFECTIVE width  sum(body)/max(body)", eff_wg)):
        print(f"\n### starvation census -- {tag} (of 256)")
        buckets = [(0,1),(1,4),(4,32),(32,128),(128,255),(255,256)]
        print(f"{'wgs':<12} {'pkts':>6} {'%pkts':>7} {'ms':>8} {'%time':>7}")
        for lo, hi in buckets:
            sel = [inst for inst, d in per.items() if lo < width(d) <= hi or (lo == 0 and width(d) <= 1)]
            if not sel: continue
            t = sum(attr[j] for j in sel)
            lbl = "1" if lo == 0 else f"{lo}-{hi}"
            print(f"{lbl:<12} {len(sel):>6} {100*len(sel)/len(per):>6.1f}% {ms(t):>8.3f} "
                  f"{100*t/(tot_attr+idle):>6.1f}%")
        for thr in (4, 32, 128):
            sel = [inst for inst, d in per.items() if width(d) <= thr]
            t = sum(attr[j] for j in sel)
            print(f"  <= {thr:<3} wg : {len(sel):>4} pkts ({100*len(sel)/len(per):>4.1f}%)  "
                  f"{ms(t):>6.3f} ms ({100*t/(tot_attr+idle):>4.1f}% of token)")

    # ---------- per-instance detail for the worst offenders ----------
    print(f"\n### per-op-instance shapes (one representative inst per opcode)")
    print(f"{'opcode':<22} {'inst':>5} {'blocks':>6} {'wg':>4} {'i0':>6} {'i1':>6} {'i2':>6} {'i3':>6}  operands")
    seen = set()
    for inst in sorted(per):
        op = per[inst]['op']
        if op in seen: continue
        seen.add(op)
        rec = insts.get(inst)
        if rec is None: continue
        ops = ' '.join(f"{k}={nm}({nb//1024}K)" for k, _, nm, nb in rec['opnds'])
        print(f"{OPNAME.get(op,'?%d'%op):<22} {inst:>5} {rec['blocks']:>6} {len(per[inst]['cus']):>4} "
              f"{rec['i'][0]:>6} {rec['i'][1]:>6} {rec['i'][2]:>6} {rec['i'][3]:>6}  {ops[:110]}")

    # ---------- per-layer-class split (dense L0-2 vs MoE L3-77 vs head) ----------
    # Ops whose operands are all act./in. (HeadNormRope, XReduce, Residual, MoeCombine) carry no
    # layer name; the stream is emitted layer-sequentially, so carry the last tagged layer forward.
    lyr_of = {}; cur = -1; last_tagged = max(i for i in insts if inst_layer(insts[i]) >= 0)
    for inst in sorted(insts):
        l = inst_layer(insts[inst])
        if l >= 0: cur = l
        lyr_of[inst] = cur if inst <= last_tagged else -1
    cls_t = defaultdict(float); cls_n = defaultdict(int)
    for inst, d in per.items():
        rec = insts.get(inst)
        if rec is None: continue
        l = lyr_of[inst]
        c = 'pre/post (embed+head)' if l < 0 else ('dense L0-2' if l < 3 else 'MoE L3-77')
        cls_t[c] += attr[inst]; cls_n[c] += 1
    print(f"\n### by layer class")
    for c, t in sorted(cls_t.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<14} {cls_n[c]:>5} pkts  {ms(t):>7.3f} ms  {100*t/(tot_attr+idle):>5.1f}%")

    if a.csv:
        with open(a.csv, 'w') as f:
            f.write("op,opcode,pkts,ms,pct,us_per_pkt,wg,GB,GBs,pct_peak\n")
            for r in rows:
                f.write(','.join(str(x) for x in r) + "\n")
        with open(a.csv.replace('.csv', '.inst.csv'), 'w') as f:
            f.write("inst,op,opname,layer,wg,effwg,attr_us,wall_us,bytes\n")
            for inst in sorted(per):
                rec = insts.get(inst)
                if rec is None: continue
                d = per[inst]
                w0 = max(max(d['end']) - min(d['ready']), 1)
                f.write(f"{inst},{d['op']},{OPNAME.get(d['op'],'?')},{inst_layer(rec)},"
                        f"{len(d['cus'])},{eff_wg(d):.1f},{us(attr[inst]):.3f},{us(w0):.3f},"
                        f"{hbm_bytes(rec,a.ctx,a.tp)}\n")


if __name__ == '__main__':
    main()
