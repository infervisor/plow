#!/usr/bin/env python3
"""#30: price the ONE surviving DSA-sparse lever -- TP-sharding the indexer query axis.

`glm52-dsa-indexer-rebuild.md` sec 5 closed indexer optimisation ("a perfect indexer still
lands under 15% at 16k") and named exactly one thing that composes with it:

    The indexer is REPLICATED on all 8 TP ranks (all 32 index heads on every rank), while
    the flash is SHARDED 8 ways (8 of 64 attention heads per rank). That replication is why
    the per-rank ratio is 47% rather than 6%.

That row was written as "scope it before you build it". This script is the scoping. It
differs from `dsa_indexer_net.py` in four ways, each of which moves the answer:

  1. the indexer compute is divided by the TP degree;
  2. an all-gather of the SELECTED INDICES is added (the scores are a T x ctx matrix and
     un-gatherable, so the query axis is the only shardable one) and priced at the MEASURED
     fabric rate, not a spec number;
  3. u16 positions are offered as an arm -- top_k positions index a context that fits in 16
     bits at every length in this table, so the gather bytes halve;
  4. an OVERLAP arm hides the gather behind the next query sub-chunk's scoring, which is the
     "double buffer" the directive asks for and which plow's packet model can express
     without a separate stream.

It also reports the two things the sec-5 row did not: the NET in absolute ms (so it does not
inherit one TTFT basis), and the distance to vLLM, which is the actual campaign goal.

usage: dsa_tp_shard_net.py
"""

LAYERS = 78
MAX_CHUNK = 8192
TP = 8
TOPK = 2048

# --- measured inputs, each with its provenance -------------------------------------------
# Whole-prompt x78 flash, dense vs an IDEAL per-query sparse walk.
# glm52-current-cost-decomposition.md sec 4 / glm52-dsa-indexer-rebuild.md sec 5.
FLASH = {4096: (77.0, 63.0), 8192: (240.0, 118.0), 16384: (915.0, 249.0), 32768: (3570.0, 511.0)}

# REBUILT indexer chain (op 117 row-resident + op 118 FAST_EXIT), us per layer per chunk,
# gfx942 / 304 WGs / 9 reps / median, both arms gated EXACT. glm52-dsa-indexer-rebuild.md sec 4.
MEAS_REBUILT = {(4096, 4096): 318 + 361, (8192, 8192): 1262 + 981, (8192, 16384): 2500 + 1729}

# Peer-to-peer all-gather rate at the FULL machine width (304 workgroups), 8 devices
# concurrent. op_collective.h PLOW_XR_MLP note, probe section [E]: the rate is DEAD LINEAR in
# workgroup count (18.4 / 36.5 / 72.6 / 141.8 / 240.5 GB/s at 19/38/76/152/304 WGs) -- the
# links never saturate. That linearity is why the OVERLAP arm is not free: a gather that runs
# concurrently with compute gets a fraction of the CUs and slows in proportion.
XGMI_GBS = 240.5

# TTFT baselines. Two different bases exist in this directory and they must not be mixed:
#   R3 consolidation gate (the basis FLASH above was decomposed against)
#   R2 re-baseline (task #26, the current served numbers)
TTFT_R3 = {4096: 973.0, 8192: 1677.0, 16384: 3627.0}
TTFT_R2 = {1024: 319.0, 4096: 712.0, 8192: 1372.0, 16384: 3245.0}
# vLLM 0.26, GLM-5.2 tp8, same box (task #16).
VLLM = {1024: 69.0, 4096: 566.0, 8192: 672.0, 16384: 1631.0, 32768: 3493.0}


def pairs(T, ctx):
    """causal (query, key) pairs in a chunk of T queries whose KV ends at ctx."""
    return T * (ctx - T) + T * (T + 1) // 2


def chunks(prompt):
    out, done = [], 0
    while done < prompt:
        t = min(MAX_CHUNK, prompt - done)
        done += t
        out.append((t, done))
    return out


def per_pair_us(meas):
    """cost per layer is linear in causal pair count (the kernel is MFMA-bound)."""
    sxx = sum(pairs(*k) ** 2 for k in meas)
    sxy = sum(pairs(*k) * v for k, v in meas.items())
    return sxy / sxx


RATE = per_pair_us(MEAS_REBUILT)


def indexer_ms(prompt, tp=1):
    """whole-prompt indexer ms, x78 layers, divided by the TP degree if sharded."""
    return sum(pairs(t, c) * RATE for t, c in chunks(prompt)) * LAYERS / 1000.0 / tp


def gather_ms(prompt, elem_bytes):
    """all-gather of `iidx` [T, top_k]: each rank RECEIVES (TP-1)/TP of the table, per layer."""
    per_layer_B = prompt * TOPK * elem_bytes * (TP - 1) / TP
    return per_layer_B / (XGMI_GBS * 1e9) * 1e3 * LAYERS


def row(T):
    dense, ideal = FLASH[T]
    gross = dense - ideal
    idx_repl = indexer_ms(T, 1)
    idx_shard = indexer_ms(T, TP)
    g32, g16 = gather_ms(T, 4), gather_ms(T, 2)
    return {
        "T": T,
        "gross": gross,
        # as the ledger closed it: replicated indexer, no gather
        "net_repl": gross - idx_repl,
        # sharded + u32 gather, serial (the sec-5 row)
        "net_shard_u32": gross - idx_shard - g32,
        # sharded + u16 gather, serial
        "net_shard_u16": gross - idx_shard - g16,
        # sharded + gather fully hidden behind the next sub-chunk's scoring (the ceiling:
        # a gather overlapped with compute steals CUs, so this is optimistic BY DESIGN)
        "net_overlap": gross - idx_shard,
        "idx_repl": idx_repl,
        "idx_shard": idx_shard,
        "g32": g32,
        "g16": g16,
    }


def main():
    print(f"indexer rebuilt rate: {RATE * 1e6:.4f} us per 1e6 causal pairs per layer "
          f"(fit over {len(MEAS_REBUILT)} measured configs)\n")

    print("=== NET saving in ms (whole prompt, x78 layers). Positive = sparse is cheaper. ===")
    hdr = (f"{'T':>6} {'flash':>7} {'ideal':>7} {'gross':>7} | {'idx x1':>7} {'NET':>7} | "
           f"{'idx /8':>7} {'gath32':>7} {'NET':>7} | {'gath16':>7} {'NET':>7} | {'NET ovl':>8}")
    print(hdr)
    rows = {}
    for T in (4096, 8192, 16384, 32768):
        r = row(T)
        rows[T] = r
        d, i = FLASH[T]
        print(f"{T:>6} {d:7.0f} {i:7.0f} {r['gross']:7.0f} | {r['idx_repl']:7.0f} "
              f"{r['net_repl']:7.0f} | {r['idx_shard']:7.0f} {r['g32']:7.0f} "
              f"{r['net_shard_u32']:7.0f} | {r['g16']:7.0f} {r['net_shard_u16']:7.0f} | "
              f"{r['net_overlap']:8.0f}")

    print("\n=== as a fraction of TTFT -- BOTH bases, because this directory holds two ===")
    print(f"{'T':>6} | {'R3 TTFT':>8} {'shard u16':>10} {'overlap':>9} | "
          f"{'R2 TTFT':>8} {'shard u16':>10} {'overlap':>9}")
    for T in (4096, 8192, 16384):
        r = rows[T]
        a, b = TTFT_R3.get(T), TTFT_R2.get(T)
        print(f"{T:>6} | {a:8.0f} {r['net_shard_u16'] / a * 100:9.1f}% "
              f"{r['net_overlap'] / a * 100:8.1f}% | {b:8.0f} "
              f"{r['net_shard_u16'] / b * 100:9.1f}% {r['net_overlap'] / b * 100:8.1f}%")

    print("\n=== the question the campaign actually asks: does it beat vLLM? (R2 basis) ===")
    print(f"{'T':>6} {'plow':>7} {'vLLM':>7} {'gap':>7} | {'best NET':>9} {'plow+sparse':>12} "
          f"{'still behind':>13} {'gap closed':>11}")
    for T in (4096, 8192, 16384):
        r, p, v = rows[T], TTFT_R2[T], VLLM[T]
        best = r["net_overlap"]  # the optimistic ceiling
        after = p - best
        print(f"{T:>6} {p:7.0f} {v:7.0f} {p - v:7.0f} | {best:9.0f} {after:12.0f} "
              f"{after - v:13.0f} {best / (p - v) * 100:10.1f}%")
    print("\n32k has no measured plow TTFT on the R2 basis, so it is deliberately NOT "
          "tabled against vLLM here.\n"
          f"For reference the 32k NET ceiling is {rows[32768]['net_overlap']:.0f} ms "
          f"against a vLLM TTFT of {VLLM[32768]:.0f} ms.")


if __name__ == "__main__":
    main()
