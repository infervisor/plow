#!/usr/bin/env python3
"""Aggregate the MFMA candidate's full-model numerics from paired --dump-logits runs.

    python3 scripts/mfma_numerics_report.py <outdir>

<outdir>/{ctl,cand}/<tag>/logits_*.bin, as written by scripts/mfma_numerics_ab.sh.

`--dump-logits` writes the WHOLE `act.logits` tensor, which is the decode bucket's [B, vocab] --
four rows on this blob, of which only the live slots mean anything. Row 0 is the sequence
amd-bench reports its greedy ids for, so every comparison here is row 0.

Three separable numbers, because they answer different questions:
  * `pf ident` -- the prefill row, which must be BIT-IDENTICAL: prefill runs the GEMM, not the
                  GEMV, so nothing here should touch it.
  * `step0`    -- one decode step off a KV cache both arms wrote identically. The PURE per-step
                  kernel difference, with no history in it.
  * `med`/`max`-- over all decode steps, where the arms decode the same tokens but write slightly
                  different K/V rows, so the difference carries accumulated history.
Agreement is greedy-argmax agreement; `first_div` is the first step whose argmax differs, after
which the two arms are decoding DIFFERENT histories and later rows describe different states.
"""
import os
import sys

import numpy as np

VOCAB = int(os.environ.get("PLOW_VOCAB", "262144"))


def bf16(path):
    v = (np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return v[:VOCAB]


def steps(d):
    out = []
    p = os.path.join(d, "logits_prefill.bin")
    if os.path.exists(p):
        out.append(("prefill", p))
    for f in sorted(os.listdir(d)):
        if f.startswith("logits_") and f != "logits_prefill.bin":
            out.append((f[len("logits_"):-len(".bin")], os.path.join(d, f)))
    return out


def compare(da, db):
    sa, sb = steps(da), steps(db)
    n = min(len(sa), len(sb))
    if n == 0:
        return None
    rels, relhs = [], []
    pre_ident = None
    agree = 0
    first_div = None
    for i in range(n):
        a, b = bf16(sa[i][1]), bf16(sb[i][1])
        ident = np.array_equal(a.view(np.uint32), b.view(np.uint32))
        r = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
        k = np.argsort(a)[::-1][:64]
        rh = float(np.linalg.norm(a[k] - b[k]) / (np.linalg.norm(a[k]) + 1e-30))
        if sa[i][0] == "prefill":
            pre_ident = ident
            continue
        rels.append(r)
        relhs.append(rh)
        if int(a.argmax()) == int(b.argmax()):
            agree += 1
        elif first_div is None:
            first_div = sa[i][0]
    if not rels:
        return None
    return dict(n=len(rels), pre_ident=pre_ident, step0=rels[0], step0h=relhs[0],
                med=float(np.median(rels)), mx=max(rels), medh=float(np.median(relhs)),
                mxh=max(relhs), agree=agree / len(rels), first_div=first_div)


def row(t, r):
    print(f"{t:<14} {r['n']:>4} {str(r['pre_ident']):>9} {r['step0']:>10.3e} {r['step0h']:>10.3e} "
          f"{r['med']:>10.3e} {r['mx']:>10.3e} {r['medh']:>10.3e} {r['mxh']:>10.3e} "
          f"{r['agree']*100:>8.1f}% {str(r['first_div']):>10}")


def main():
    root = sys.argv[1]
    tags = sorted(t for t in os.listdir(os.path.join(root, "ctl"))
                  if os.path.isdir(os.path.join(root, "ctl", t)))
    print(f"{'tag':<14} {'dec':>4} {'pf ident':>9} {'step0':>10} {'step0 h64':>10} "
          f"{'med':>10} {'max':>10} {'med h64':>10} {'max h64':>10} {'agree':>9} {'first_div':>10}")
    for t in tags:
        a, b = os.path.join(root, "cand", t), os.path.join(root, "ctl", t)
        if not (os.path.isdir(a) and os.path.isdir(b)):
            continue
        r = compare(a, b)
        if r:
            row(t, r)

    print()
    for arm in ("ctl", "cand"):
        a = os.path.join(root, arm, "ragged4")
        b = os.path.join(root, arm, "solo1024p0")
        if os.path.isdir(a) and os.path.isdir(b):
            r = compare(a, b)
            if r:
                print(f"batch-width independence {arm}: ragged-batch-4 slot 0 vs the same "
                      f"sequence alone -- step0 rel={r['step0']:.3e} max={r['mx']:.3e} "
                      f"agree={r['agree']*100:.1f}% first_div={r['first_div']}")


main()
