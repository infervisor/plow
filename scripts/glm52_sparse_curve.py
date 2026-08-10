#!/usr/bin/env python3
"""What DENSE attention costs today, and what an IDEALLY sparse (top-2048) one would.

Fits the traced FlashMlaPrefill span to `a + b * causal_pairs` over the four
measured chunks, then re-evaluates the same fit at the top-2048 pair count.
`b` is the marginal cost of one (query,key) pair; `a` is the per-packet floor.

The counterfactual is IDEAL: flash busy exactly proportional to the number of
selected pairs, i.e. a per-query membership-skipping kernel. It is NOT the
union-of-8 route, which perf-data/plow-gfx942/glm52-dsa-sparse-b3.md closed
(adjacent queries share 89% of their top-2048, so a union-of-8 is 0.30 of causal
against a per-query ideal of 2048/mean_causal).
"""
import json
import numpy as np

TOPK = 2048
CHUNK = 8192
NLAY = 78
D = "/tmp/tc"
# (json key, chunk width, ctx at chunk start)
MEAS = [(1024, 1024, 0), (4096, 4096, 0), (8192, 8192, 0), (16384, 8192, 8192)]
TTFT = {1024: 343.3, 4096: 973.4, 8192: 1677.0, 16384: 3627.4}
# indexer price, measured on the DSA-armed objects: op 117 IndexScorePf materialises a
# T x ctx f32 score matrix -> 3.7 ms/layer at 16k (glm52-dsa-sparse-b2.md).  It is
# quadratic in the same variable as the flash, so scale by pair count.
IDX_MS_AT = (3.7, 8192 * 16384)     # (ms/layer, pairs) reference point


def pairs(w, ctx0):
    d = sum(ctx0 + i + 1 for i in range(w))
    s = sum(min(ctx0 + i + 1, TOPK) for i in range(w))
    return d, s


def main():
    rows = []
    for key, w, ctx0 in MEAS:
        j = json.load(open(f"{D}/pf_moe_{key}.json"))
        d, s = pairs(w, ctx0)
        rows.append((key, w, ctx0, d, s, j['ops']['FlashMlaPrefill']['span'],
                     j['ops']['MlaMergeFold']['span']))

    P = np.array([r[3] for r in rows], float)
    Y = np.array([r[5] for r in rows], float)
    A = np.vstack([np.ones(len(P)), P]).T
    (a, b), *_ = np.linalg.lstsq(A, Y, rcond=None)
    print(f"fit: flash span/layer = {a:.0f} us + {b*1e6:.4f} us per 1e6 causal pairs "
          f"(R^2 {1-((Y-A@[a,b])**2).sum()/((Y-Y.mean())**2).sum():.3f})")

    print(f"\n{'chunk':>16}{'causal pairs':>14}{'top2048':>12}{'ratio':>7}"
          f"{'flash meas':>11}{'sparse ideal':>13}{'saved us/lay':>13}{'merge (flat)':>13}")
    save = {}
    for (key, w, ctx0, d, s, fl, mg) in rows:
        ideal = a + b * s
        sv = b * (d - s)
        save[key] = sv
        print(f"{('T=%d c%d' % (w, ctx0//CHUNK)):>16}{d:>14.4g}{s:>12.4g}{s/d:>7.3f}"
              f"{fl:>11.0f}{ideal:>13.0f}{sv:>13.0f}{mg:>13.0f}")

    print(f"\n== whole-prompt effect, x{NLAY} layers")
    print(f"{'T':>7}{'flash ms':>10}{'sparse ms':>11}{'saved ms':>10}{'% TTFT':>8}"
          f"{'indexer ms':>12}{'NET ms':>9}{'% TTFT':>8}")
    for T in (4096, 8192, 16384, 32768):
        fl = sv = idx = 0.0
        rem, ctx = T, 0
        while rem > 0:
            w = min(rem, CHUNK)
            d, s = pairs(w, ctx)
            fl += (a + b * d) * NLAY / 1e3
            sv += b * (d - s) * NLAY / 1e3
            idx += IDX_MS_AT[0] * (d / IDX_MS_AT[1]) * NLAY
            ctx += w
            rem -= w
        net = sv - idx
        t = TTFT.get(T, float('nan'))
        print(f"{T:>7}{fl:>10.0f}{fl-sv:>11.0f}{sv:>10.0f}{100*sv/t:>8.1f}"
              f"{idx:>12.0f}{net:>9.0f}{100*net/t:>8.1f}")


main()
