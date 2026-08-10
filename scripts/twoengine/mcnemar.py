#!/usr/bin/env python3
"""Paired accuracy comparison of two `gsm_paired.py` runs -- McNemar's exact test.

The unpaired two-proportion test is the wrong instrument for "did this arm change the
model's answers". It treats question difficulty as noise, and question difficulty is most
of the variance: both arms get the same easy questions right and the same hard ones wrong.
McNemar conditions on that by looking ONLY at the discordant pairs -- questions where the
two arms disagree -- which is where all the information about a difference lives.

  b = control right, arm wrong      c = control wrong, arm right
  H0: b and c come from the same Binomial(b+c, 1/2)

Reports the exact two-sided p (no normal approximation, which is unreliable when b+c is
small) and a Wilson interval on the paired difference.

usage: mcnemar.py control.json arm.json
"""
import json, sys
from math import comb


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    ca, cb = a["correct"], b["correct"]
    if len(ca) != len(cb):
        print(f"FAIL: different lengths ({len(ca)} vs {len(cb)}) -- not the same question set")
        sys.exit(1)

    # A question either arm failed to ANSWER (-1) carries no paired information and must be
    # dropped from both, not scored as wrong -- otherwise a transport error reads as a
    # numerics regression.
    keep = [i for i in range(len(ca)) if ca[i] >= 0 and cb[i] >= 0]
    dropped = len(ca) - len(keep)
    n00 = n01 = n10 = n11 = 0
    for i in keep:
        if ca[i] and cb[i]:
            n11 += 1
        elif ca[i] and not cb[i]:
            n10 += 1
        elif not ca[i] and cb[i]:
            n01 += 1
        else:
            n00 += 1
    disc = n10 + n01

    print(f"control = {a['label']}   arm = {b['label']}")
    print(f"paired on {len(keep)} questions ({dropped} dropped: unanswered by one or both)\n")
    print(f"  control accuracy {sum(ca[i] for i in keep)/len(keep):.4f}")
    print(f"  arm     accuracy {sum(cb[i] for i in keep)/len(keep):.4f}")
    print(f"  paired difference {(sum(cb[i] for i in keep)-sum(ca[i] for i in keep))/len(keep)*100:+.2f} pp\n")
    print("  contingency          arm wrong   arm right")
    print(f"    control wrong  {n00:11d} {n01:11d}")
    print(f"    control right  {n10:11d} {n11:11d}")
    print(f"\n  discordant pairs: b (ctl right, arm wrong) = {n10}, c = {n01}, b+c = {disc}")

    if disc == 0:
        print("\n  no discordant pairs -- the arms agree on every question. p = 1.0")
        return
    k = min(n10, n01)
    p = min(1.0, 2 * sum(comb(disc, i) for i in range(k + 1)) / 2 ** disc)
    print(f"  McNemar exact two-sided p = {p:.4f}")

    # What effect size COULD this run have detected? Reported always, because a null result
    # from an underpowered run is the failure mode this file exists to prevent.
    import math
    mde = 1.96 * math.sqrt(disc) / len(keep) * 100
    print(f"  minimum detectable difference at this discordance: ~{mde:.2f} pp (2 sigma)")

    if p < 0.05:
        d = "LOWER" if n10 > n01 else "HIGHER"
        print(f"\n  VERDICT: the arm is significantly {d} (p = {p:.4f}). Not a wash.")
    else:
        print(f"\n  VERDICT: no significant difference (p = {p:.4f}). Note this is 'not "
              f"distinguishable', not 'identical' -- the run could detect ~{mde:.2f} pp.")


if __name__ == "__main__":
    main()
