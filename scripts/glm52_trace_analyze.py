#!/usr/bin/env python3
"""glm52_trace_analyze.py — per-kernel timing breakdown of ONE GLM MoE block + one dense block
from a PlowTraceRec dump (PLOW_TRACE_RAW) + the .insts.txt sidecar written by glm52_decode.c.

Usage:
  glm52_trace_analyze.py <prefix.insts.txt> <prefix.ctxCTX.bin> [--ctx CTX] [--tp N] [--ghz G]

PlowTraceRec (dev_isa.h, 40 B): u32 cu, u32 pc, u32 inst, u16 op, u16 slice,
                                u64 t_arrive, u64 t_ready, u64 t_end.
Per the campaign method each op's span is min(t_ready) -> p90(t_end) over its packets.
Clock: s_memrealtime. dev_isa.h header says ~100 MHz (10 ns/tick); tp_decode.c/tp-transport
say gfx950 = 1 ns/tick (1 GHz). We DON'T trust either blindly: --ghz overrides, and the tool
prints the full-step tick span so it can be calibrated against the measured traced-ms (span_ticks
/ (traced_ms*1e6) = GHz). %-of-block and GB/s ratios that matter are computed from the same clock
so relative numbers are clock-independent; only absolute us scale with --ghz.
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
}
PEAK_GBs = 8000.0  # MI350X HBM3e ~8 TB/s

def parse_insts(path):
    insts = {}
    for line in open(path):
        if line.startswith('#') or not line.strip(): continue
        head, rest = line.split('|', 1)
        parts = head.split()
        inst = int(parts[0]); op = int(parts[1]); blocks = int(parts[2])
        opnd_str, tail = rest.rsplit('|', 1)
        opnds = []  # (slot, idx, name, bytes)
        for tok in opnd_str.split():
            if not tok.startswith('t'): continue
            k, val = tok.split('=', 1)
            idx, name, nb = val.split(':')
            opnds.append((k, int(idx), name, int(nb)))
        i = [int(x) for x in tail.split()]
        insts[inst] = dict(op=op, blocks=blocks, opnds=opnds, i=i)
    return insts

def inst_layer(rec):
    """layer index this inst belongs to (max over 'model.layers.N.' / 'kv.N.' operands)."""
    lyr = -1
    for _, _, name, _ in rec['opnds']:
        if name.startswith('model.layers.'):
            lyr = max(lyr, int(name.split('.')[2]))
        elif name.startswith('kv.'):
            lyr = max(lyr, int(name.split('.')[1]))
    return lyr

def hbm_bytes(rec, ctx):
    """Estimate HBM bytes streamed by this op. Weight-bound GEMVs: sum non-activation weight
    operands. Flash MLA/gather: latent depth = ctx (declared kv buffer is max_ctx -> overstated),
    latent row = (kv_lora 512 + rope 64)*2B = 1152 B; reread once here (achieved-effective)."""
    op = rec['op']
    FLASH = {12,38,50,54}
    if op in FLASH:
        return ctx * 1152          # single-stream latent footprint (reread factor noted separately)
    if op in {13,27}:              # FlashMerge: reads nsplit partials of the latent-out
        return 0                   # small; dominated by fixed cost, report us only
    b = 0
    for _, _, name, nb in rec['opnds']:
        if name.startswith('act.') or name.startswith('in.') or name.startswith('kv.'):
            continue
        b += nb
    return b

def p90(xs):
    if not xs: return 0
    s = sorted(xs); k = int(0.9*(len(s)-1)+0.5)
    return s[min(k,len(s)-1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('insts'); ap.add_argument('bin')
    ap.add_argument('--ctx', type=int, default=0)
    ap.add_argument('--tp', type=int, default=1)
    # CALIBRATED: gfx950 s_memrealtime = 100 MHz (10 ns/tick). Verified span_ticks*10ns == traced-ms
    # (ctx1024 TP1: 404772 ticks -> 4.048 ms == traced 4.098 ms). dev_isa.h "~100 MHz" is correct;
    # tp_decode.c's "1 ns/tick" note is WRONG for this GPU. --ghz overrides.
    ap.add_argument('--ghz', type=float, default=0.1)
    ap.add_argument('--dense', type=int, default=1)     # dense layer to profile
    ap.add_argument('--moe', type=int, default=-1)      # MoE layer to profile (-1 = first MoE found)
    a = ap.parse_args()
    if a.ctx == 0 and 'ctx' in a.bin:
        try: a.ctx = int(a.bin.split('ctx')[1].split('.')[0])
        except: a.ctx = 2048

    insts = parse_insts(a.insts)
    data = open(a.bin, 'rb').read()
    n = len(data)//40
    per = defaultdict(lambda: dict(ready=[], end=[], arrive=[], slices=0))
    gmin=None; gmax=None
    for i in range(n):
        cu,pc,inst,op,slc,ta,tr,te = struct.unpack_from('<IIIHHQQQ', data, i*40)
        if ta==0 and tr==0 and te==0: continue   # untouched slot
        d = per[inst]; d['ready'].append(tr); d['end'].append(te); d['arrive'].append(ta); d['slices']+=1
        d['op']=op
        gmin = ta if gmin is None else min(gmin, ta)
        gmax = te if gmax is None else max(gmax, te)
    span_ticks = (gmax-gmin) if gmin is not None else 0

    # assign layers
    firstmoe = None
    for inst in sorted(insts):
        if insts[inst]['op'] in {40,56}:   # a router marks a MoE layer
            firstmoe = inst_layer(insts[inst]); break
    moe_layer = a.moe if a.moe>=0 else (firstmoe if firstmoe is not None else 3)

    tick_us = 1000.0*a.ghz    # ticks per us
    def us(t): return t/tick_us

    print(f"# GLM-5.2 decode trace  |  ctx={a.ctx}  TP={a.tp}  |  {n} recs, "
          f"full-step span={span_ticks} ticks ({us(span_ticks):.1f} us @ {a.ghz}GHz)")
    print(f"#   (calibrate: span_ticks / (traced_ms*1e6) = actual GHz)")

    for tag, lyr in (("DENSE block  L%d"%a.dense, a.dense), ("MoE block  L%d"%moe_layer, moe_layer)):
        rows = []
        blk_start=None; blk_end=None
        for inst in sorted(insts):
            if inst_layer(insts[inst]) != lyr: continue
            if inst not in per: continue
            d = per[inst]; rec = insts[inst]
            t0 = min(d['ready']); t1 = p90(d['end']); ta0 = min(d['arrive'])
            dur = t1 - t0
            stall = t0 - ta0
            hb = hbm_bytes(rec, a.ctx)
            gbs = (hb/ (us(dur)*1e3)) if dur>0 and hb>0 else 0.0   # bytes/(us*1e3)=GB/s
            rows.append(dict(inst=inst, op=rec['op'], blocks=rec['blocks'], slices=d['slices'],
                             dur=dur, stall=stall, gbs=gbs, hb=hb, t0=t0, t1=t1))
            blk_start = t0 if blk_start is None else min(blk_start, t0)
            blk_end   = t1 if blk_end   is None else max(blk_end, t1)
        blk_dur = (blk_end-blk_start) if blk_start is not None else 0
        # wall = sum of busy on the critical path ~ block span; %-of-block vs block span
        print(f"\n### {tag}  ({a.ctx}ctx TP{a.tp})   block span {us(blk_dur):.2f} us "
              f"(sum-of-ops {us(sum(r['dur'] for r in rows)):.2f} us)")
        print(f"  {'inst':>4} {'op':<20} {'blk':>4} {'slc':>4} {'us':>9} {'%blk':>6} {'stall_us':>9} {'GB/s':>8}")
        for r in sorted(rows, key=lambda r:-r['dur']):
            pct = 100.0*r['dur']/blk_dur if blk_dur else 0
            print(f"  {r['inst']:>4} {OPNAME.get(r['op'],'?'+str(r['op'])):<20} {r['blocks']:>4} "
                  f"{r['slices']:>4} {us(r['dur']):>9.3f} {pct:>5.1f}% {us(r['stall']):>9.3f} "
                  f"{r['gbs']:>8.0f}")

if __name__ == '__main__':
    main()
