#!/usr/bin/env python3
"""Attribute plow's TTFT, and its gap to vLLM, to named traced components.

Reads the per-layer JSON censuses written by scripts/glm52_layer_census.py and
turns per-layer busy CU-us into ms of TTFT:

    ms(component, T) = busy_CUus/layer / 304 CU  x  n_layers  /  1000

That is the component's PERFECT-PACK cost. What the layer span does NOT give
back to the components (gate wait, idle CUs, packet-boundary gaps) is reported
as one explicit "schedule overhead" row so the columns sum to the measured wall.
"""
import json

D = "/tmp/tc"
NCU = 304
N_MOE, N_DENSE = 75, 3
CHUNK = 8192
TS = [1024, 4096, 8192, 16384]
TTFT = {1024: 343.3, 4096: 973.4, 8192: 1677.0, 16384: 3627.4}   # served, round-3 gate
WALL = {1024: 343.5, 4096: 731.2, 8192: 1390.0, 16384: 3239.4}   # amd-bench device prefill
VLLM = {1024: 69.0, 4096: 566.0, 8192: 672.0, 16384: 1631.0}
TAIL = 203.0     # measured: prefill(1025) - prefill(1024) = 546.5 - 343.5

GROUP = {'Gemm': 'attn+shared GEMM', 'GemmSmall': 'attn+shared GEMM',
         'GemmMed': 'attn+shared GEMM', 'GemmWide': 'attn+shared GEMM',
         'GemmGlu': 'attn+shared GEMM',
         'FlashMlaPrefill': 'flash attention', 'MlaMergeFold': 'flash merge/fold',
         'MoeGroupGluPf': 'MoE GLU', 'MoeGroupDownPf': 'MoE DOWN',
         'MoeCombinePf': 'MoE combine', 'MoeRouterTopkPf': 'MoE router',
         'MoeAlignPf': 'MoE router', 'XReduceTwoShot': 'collectives (XR)',
         'RmsNorm': 'norms/residual', 'Residual': 'norms/residual',
         'HeadNormRope': 'norms/residual'}
ROLLUP = {'MoE GLU': 'MoE', 'MoE DOWN': 'MoE', 'MoE combine': 'MoE', 'MoE router': 'MoE',
          'flash attention': 'attention', 'flash merge/fold': 'attention'}


def chunks(T):
    out, rem, ctx = [], T, 0
    while rem > 0:
        c = min(rem, CHUNK)
        ctx += c
        out.append((c, ctx))
        rem -= c
    return out


def per_chunk_ms(key):
    """{component: ms} for ONE chunk whose trace is keyed `key`, whole 78-layer model."""
    out = {}
    for kind, n in (("pf_moe", N_MOE), ("pf_dense", N_DENSE)):
        j = json.load(open(f"{D}/{kind}_{key}.json"))
        for op, v in j['ops'].items():
            c = GROUP.get(op, op)
            out[c] = out.get(c, 0.0) + v['busy'] / NCU * n / 1e3
        span_ms = j['layer_span_us'] * n / 1e3
        busy_ms = sum(v['busy'] for v in j['ops'].values()) / NCU * n / 1e3
        out['schedule overhead'] = out.get('schedule overhead', 0.0) + span_ms - busy_ms
    return out


def main():
    cols = {}
    for T in TS:
        tot = {}
        for (w, ctxend) in chunks(T):
            key = T if (T > CHUNK and ctxend == T) else w
            for c, v in per_chunk_ms(key).items():
                tot[c] = tot.get(c, 0.0) + v
        cols[T] = tot

    order = sorted(cols[8192], key=lambda c: -cols[8192][c])
    print("== plow TTFT attributed, ms (busy CU-us / 304 x n_layers; whole 78-layer model)")
    print(f"{'component':<22}" + "".join(f"{('T=%d' % T):>10}" for T in TS))
    for c in order:
        print(f"{c:<22}" + "".join(f"{cols[T].get(c, 0.0):>10.1f}" for T in TS))
    print(f"{'--- traced total':<22}" +
          "".join(f"{sum(cols[T].values()):>10.1f}" for T in TS))
    print(f"{'device prefill wall':<22}" + "".join(f"{WALL[T]:>10.1f}" for T in TS))
    print(f"{'served TTFT':<22}" + "".join(f"{TTFT[T]:>10.1f}" for T in TS))

    print("\n== the served-vs-device delta, and what it is")
    print(f"{'T':>7}{'served':>9}{'device':>9}{'delta':>8}{'ragged tail chunk':>19}{'rest':>8}")
    for T in TS:
        d = TTFT[T] - WALL[T]
        tail = 0.0 if T == 1024 else TAIL
        print(f"{T:>7}{TTFT[T]:>9.1f}{WALL[T]:>9.1f}{d:>8.1f}{tail:>19.1f}{d-tail:>8.1f}")

    print("\n== GAP TO vLLM, attributed (ms). plow column scaled so it sums to served TTFT.")
    print(f"{'component':<22}" + "".join(f"{('T=%d' % T):>10}" for T in TS))
    scale = {T: TTFT[T] / sum(cols[T].values()) for T in TS}
    for c in order:
        print(f"{c:<22}" + "".join(f"{cols[T].get(c, 0.0)*scale[T]:>10.1f}" for T in TS))
    print(f"{'plow TTFT':<22}" + "".join(f"{TTFT[T]:>10.1f}" for T in TS))
    print(f"{'vLLM TTFT':<22}" + "".join(f"{VLLM[T]:>10.1f}" for T in TS))
    print(f"{'GAP':<22}" + "".join(f"{TTFT[T]-VLLM[T]:>10.1f}" for T in TS))


main()
