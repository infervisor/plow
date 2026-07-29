#!/usr/bin/env python3
"""glm52_logit_cmp.py — compare two `plowrt amd-bench --dump-logits` runs as VECTORS.

    python3 scripts/glm52_logit_cmp.py <dir-A> <dir-B> [label-A label-B]

The A/B this exists for is PREFILL vs the DECODE ORACLE: the same prompt, the same weights, the
same binary — the arm is chosen only by whether the blob carries a prefill bucket ladder
(`crates/plowrt/src/main.rs`, `if g.rank(0).n_programs() == 1`). Comparing the two by their greedy
IDS answers the wrong question. Greedy argmax is a step function: on a near-tie a 1e-3 perturbation
flips the reported token and looks exactly like a broken kernel, and on a wide margin a 1e-1 error
changes nothing and looks exactly like a healthy one. Neither reading is evidence.

So this prints, per step, the things that ARE evidence:
  * rel   — ||A-B|| / ||B||, the residual on the whole logit row;
  * gap   — (top1 - top2) / |top1| in the reference arm, i.e. how close that step was to a tie;
  * a FLIP is then classified: `gap <= rel-scale` means the two arms disagree about a coin flip,
    `gap >> rel` with a flip would mean the logit row itself moved and IS a fault.
"""
import sys, os
import numpy as np


def bf16(path):
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32)


def steps(d):
    out = []
    if os.path.exists(os.path.join(d, "logits_prefill.bin")):
        out.append(("prefill", os.path.join(d, "logits_prefill.bin")))
    for f in sorted(os.listdir(d)):
        if f.startswith("logits_") and f != "logits_prefill.bin":
            out.append((f[len("logits_"):-len(".bin")], os.path.join(d, f)))
    return out


def main():
    da, db = sys.argv[1], sys.argv[2]
    la = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(da.rstrip("/"))
    lb = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(db.rstrip("/"))
    sa, sb = steps(da), steps(db)
    n = min(len(sa), len(sb))
    print(f"{la} vs {lb}   ({n} steps dumped)\n")
    print(f"{'step':>8} {'argmax A':>9} {'argmax B':>9} {'rel(all)':>9} {'rel(head)':>10} "
          f"{'head set':>9} {'gap A':>8} {'gap B':>8} {'maxabs':>8} {'verdict':>22}")
    HEAD = 64
    for i in range(n):
        (ta, pa), (tb, pb) = sa[i], sb[i]
        a, b = bf16(pa), bf16(pb)
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
        # THE HEAD IS WHAT DECIDES. rel(all) is dominated by the ~150k low-magnitude tail entries
        # (measured on GLM-5.2: ranks >16k hold 87% of the squared difference at mean |logit| 1.7),
        # which no sampler ever reaches. rel(head) and the top-64 SET overlap are the numbers that
        # say whether the two arms agree about the distribution.
        kb = np.argsort(b)[::-1][:HEAD]
        relh = float(np.linalg.norm(a[kb] - b[kb]) / (np.linalg.norm(b[kb]) + 1e-30))
        ov = len(set(np.argsort(a)[::-1][:HEAD].tolist()) & set(kb.tolist())) / HEAD
        mx = float(np.abs(a - b).max())
        ia, ib = int(a.argmax()), int(b.argmax())
        sra, srb = np.sort(a)[::-1], np.sort(b)[::-1]
        ga, gb = float(sra[0] - sra[1]), float(srb[0] - srb[1])
        if ia == ib:
            v = "same token"
        elif min(ga, gb) <= mx:
            v = "FLIP on a NEAR-TIE"
        else:
            v = "*** FLIP, gap > maxabs ***"
        print(f"{ta:>8} {ia:>9} {ib:>9} {rel:>9.5f} {relh:>10.5f} {ov:>9.3f} "
              f"{ga:>8.4f} {gb:>8.4f} {mx:>8.4f} {v:>22}")
        if ia != ib:
            print("         ^ FIRST DIVERGENCE. Every LATER row compares two arms that are decoding "
                  "DIFFERENT\n           token histories, so those rows describe different states "
                  "and mean nothing.")
            break


if __name__ == "__main__":
    main()
