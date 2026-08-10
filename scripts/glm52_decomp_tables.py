#!/usr/bin/env python3
"""Turn the per-T layer censuses into the tables of
perf-data/plow-gfx942/glm52-current-cost-decomposition.md.

Inputs are the --json outputs of scripts/glm52_layer_census.py:
    /tmp/tc/pf_moe_<T>.json  pf_dense_<T>.json   (prefill)
    /tmp/tc/dec_moe_<T>.json                     (decode)

Produces: per-op span at each T, the fitted scaling exponent in T, the
TTFT reconciliation (3 dense + 75 MoE layers), and the vLLM gap attribution.
"""
import json, math, sys, os

D = "/tmp/tc"
TS = [1024, 4096, 8192, 16384]
CHUNK = 8192
N_MOE, N_DENSE = 75, 3
TTFT = {1024: 343.3, 4096: 973.4, 8192: 1677.0, 16384: 3627.4}     # round-3 gate, served
VLLM = {1024: 69.0, 4096: 566.0, 8192: 672.0, 16384: 1631.0}       # vLLM 0.26 AITER, same box


def load(kind, T):
    p = f"{D}/{kind}_{T}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def chunks(T):
    """(width, ctx_at_chunk_end) for the runtime's plan_chunks cover."""
    out, rem, ctx = [], T, 0
    while rem > 0:
        c = min(rem, CHUNK)
        ctx += c
        out.append((c, ctx))
        rem -= c
    return out


def fit_exponent(xs, ys):
    """least-squares slope of log y vs log x; 0=flat, 1=linear, 2=quadratic."""
    xs = [x for x, y in zip(xs, ys) if y > 0]
    ys = [y for y in ys if y > 0]
    if len(xs) < 2:
        return float('nan')
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else float('nan')


def op_table(kind, label):
    js = {T: load(kind, T) for T in TS}
    js = {T: j for T, j in js.items() if j}
    if not js:
        print(f"(no data for {kind})")
        return {}
    ops = sorted({o for j in js.values() for o in j['ops']},
                 key=lambda o: -max(j['ops'].get(o, {}).get('span', 0) for j in js.values()))
    print(f"\n### {label}: packet span (us) per layer, by chunk width")
    hdr = "".join(f"{('T=%d' % T):>12}" for T in js)
    print(f"{'op':<24}{hdr}{'exp(T)':>9}")
    for o in ops:
        row = [js[T]['ops'].get(o, {}).get('span', 0.0) for T in js]
        # x axis is the CHUNK width actually dispatched, not the prompt length
        xs = [min(T, CHUNK) for T in js]
        # at T=16384 the traced chunk is chunk 2 (ctx 8192..16384); exclude from the
        # exponent fit because its chunk width is the same 8192
        fx = [x for x, T in zip(xs, js) if T <= CHUNK]
        fy = [v for v, T in zip(row, js) if T <= CHUNK]
        print(f"{o:<24}" + "".join(f"{v:>12.1f}" for v in row) + f"{fit_exponent(fx, fy):>9.2f}")
    print(f"{'LAYER SPAN (median)':<24}" +
          "".join(f"{js[T]['layer_span_us']:>12.1f}" for T in js))
    print(f"{'sum packet spans':<24}" +
          "".join(f"{js[T]['sum_span_us']:>12.1f}" for T in js))
    print(f"{'busy CU-us':<24}" + "".join(f"{js[T]['busy_cuus']:>12.0f}" for T in js))
    print(f"{'gate-wait CU-us':<24}" + "".join(f"{js[T]['wait_cuus']:>12.0f}" for T in js))
    print(f"{'gate-wait % of in-pkt':<24}" +
          "".join(f"{100*js[T]['wait_cuus']/(js[T]['busy_cuus']+js[T]['wait_cuus']):>12.1f}"
                  for T in js))
    print(f"{'packing efficiency %':<24}" +
          "".join(f"{100*js[T]['busy_cuus']/304/js[T]['layer_span_us']:>12.1f}" for T in js))
    print(f"{'pkt-boundary gap us':<24}" + "".join(f"{js[T]['gap_us']:>12.1f}" for T in js))
    return js


def reconcile(moe, dense):
    print("\n### TTFT reconciliation (3 dense + 75 MoE layers, per chunk)")
    print(f"{'T':>7}{'chunks':>8}{'dense ms':>10}{'MoE ms':>10}{'model ms':>11}"
          f"{'served TTFT':>13}{'residual':>11}{'resid %':>9}")
    for T in TS:
        if T not in moe:
            continue
        # every chunk of width w costs the layer spans measured at that width;
        # the T=16384 trace IS chunk 2, so chunk 1 is priced from the T=8192 trace
        tot = 0.0
        for (w, ctxend) in chunks(T):
            key = T if (T > CHUNK and ctxend == T) else w
            if key not in moe:
                key = w
            tot += (N_DENSE * dense[key]['layer_span_us'] +
                    N_MOE * moe[key]['layer_span_us']) / 1e3
        d = N_DENSE * dense[T]['layer_span_us'] / 1e3 if T in dense else 0
        m = N_MOE * moe[T]['layer_span_us'] / 1e3
        r = TTFT[T] - tot
        print(f"{T:>7}{len(chunks(T)):>8}{d:>10.1f}{m:>10.1f}{tot:>11.1f}"
              f"{TTFT[T]:>13.1f}{r:>11.1f}{100*r/TTFT[T]:>9.1f}")


def main():
    moe = op_table("pf_moe", "PREFILL — one MoE layer (median of L6..L74)")
    dense = op_table("pf_dense", "PREFILL — one DENSE layer (L0..L2)")
    if moe and dense:
        reconcile(moe, dense)
    op_table("dec_moe", "DECODE — one MoE layer (median of L6..L74)")


main()
