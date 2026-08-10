#!/usr/bin/env python3
"""Turn measured per-chunk indexer costs into the sparse-prefill NET table.

Consumes the (T, ctx) -> (score_us, select_us) measurements from dsa_pf_indexer_bench and
rebuilds §4 of glm52-current-cost-decomposition.md with plow's own indexer repriced.

Two things this fixes about the original table's indexer column:
  * it priced a whole prompt by scaling ONE chunk's cost by pair count, but plan_chunks covers a
    16k prompt as [8192, 8192] -- two chunks with very different causal pair counts, so the whole
    prompt is the SUM over chunks, not a single scaled cell;
  * it used op 117 alone; the chain the flash actually waits on is 117 + 118.

usage: dsa_indexer_net.py            (edit MEAS below with the bench's numbers)
"""
LAYERS = 78
MAX_CHUNK = 8192

# TTFT baselines, round-3 consolidation gate (ms) @ 4k / 8k / 16k
TTFT = {4096: 973.0, 8192: 1677.0, 16384: 3627.0, 32768: None}
# Deliverable 4, whole-prompt x78: dense flash ms, ideal top-2048 flash ms
FLASH = {4096: (77.0, 63.0), 8192: (240.0, 118.0), 16384: (915.0, 249.0), 32768: (3570.0, 511.0)}

# MEASURED per-chunk, per-layer microseconds, gfx942 / 304 WGs / 9 reps / median
# (dsa_pf_indexer_bench, 2026-08-08, GPU lock held, rocm-smi 0% on acquire):
#   "shipped" = index_score_pf_128 + index_select_pf_k
#   "rebuilt" = index_score_pf_row64_128 + index_select_pf_fast_k   (both gated EXACT)
MEAS = {
    "shipped": {(4096, 4096): (1129, 634), (8192, 8192): (3689, 1827), (8192, 16384): (8800, 3048)},
    "rebuilt": {(4096, 4096): (321, 361), (8192, 8192): (1279, 981), (8192, 16384): (2568, 1729)},
}


def pairs(T, ctx):
    """causal (query, key) pairs in a chunk of T queries whose KV ends at ctx."""
    q0 = ctx - T
    return T * q0 + T * (T + 1) // 2


def chunks(prompt):
    """plan_chunks: MAX_CHUNK-wide chunks, each attending over everything before it."""
    out, done = [], 0
    while done < prompt:
        t = min(MAX_CHUNK, prompt - done)
        done += t
        out.append((t, done))
    return out


def fit(meas_arm):
    """cost per layer is linear in causal pair count (the kernel is MFMA-bound); fit us/pair."""
    xs = [(pairs(T, c), v) for (T, c), v in meas_arm.items()]
    sxx = sum(x * x for x, _ in xs)
    sxy = sum(x * y for x, y in xs)
    return sxy / sxx


def chain_cost(arm, prompt):
    """whole-prompt indexer ms for one arm: sum over chunks, x78 layers."""
    per_pair = fit(arm)
    return sum(pairs(t, c) * per_pair for t, c in chunks(prompt)) * LAYERS / 1000.0


def table(arms):
    print(f"{'T':>6} {'flash dense':>11} {'ideal sparse':>12} {'gross':>7} "
          f"{'%TTFT':>6} | " + " | ".join(f"{n:>9} {'NET':>7} {'%TTFT':>6}" for n in arms))
    for T in (4096, 8192, 16384, 32768):
        d, s = FLASH[T]
        gross = d - s
        tt = TTFT[T]
        pg = f"{gross / tt * 100:5.1f}%" if tt else "    -"
        cells = []
        for n in arms:
            idx = chain_cost(arms[n], T)
            net = gross - idx
            pn = f"{net / tt * 100:5.1f}%" if tt else "    -"
            cells.append(f"{idx:9.0f} {net:7.0f} {pn:>6}")
        print(f"{T:>6} {d:11.0f} {s:12.0f} {gross:7.0f} {pg:>6} | " + " | ".join(cells))


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) > 1:
        MEAS = {n: {tuple(int(x) for x in k.split(",")): tuple(v) for k, v in a.items()}
                for n, a in json.load(open(sys.argv[1])).items()}
    if not MEAS:
        print("no measurements loaded — pass the bench JSON"); sys.exit(1)
    arms = {n: {k: v[0] + v[1] for k, v in a.items()} for n, a in MEAS.items()}
    for n, a in arms.items():
        print(f"# {n}: {fit(a) * 1e6:.4f} us per 1e6 causal pairs per layer")
    print()
    table(arms)
