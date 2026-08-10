#!/usr/bin/env python3
"""Analytic scaling terms for the GLM-5.2 TP8 prefill cost decomposition.

Pure arithmetic (no measurement): the DRAM and pair-count floors each traced
term is measured against in
perf-data/plow-gfx942/glm52-current-cost-decomposition.md.

  python3 scripts/glm52_cost_model.py
"""
H = 6144            # hidden
TP = 8
E = 256             # routed experts
K = 8               # top-k
I_MOE = 2048 // TP  # 256 per rank
DI = 12288 // TP    # 1536 per rank (dense FFN)
KV_LORA, ROPE = 512, 64
NH_L = 64 // TP     # 8 heads per rank
V = 256
TOPK_IDX = 2048     # GLM DSA index_topk
N_MOE, N_DENSE = 75, 3
CHUNK = 8192        # runtime MAX_CHUNK

MB = 1 << 20
GB = 1 << 30


def moe_bytes(T):
    """DRAM bytes touched by the grouped pair (ops 85+86), one MoE layer, one rank."""
    w_gu = E * 2 * I_MOE * H            # fp8 gate+up
    w_dn = E * H * I_MOE                # fp8 down
    a = T * H * 2                       # bf16 activation, distinct rows
    fu = T * K * I_MOE * 2 * 2          # bf16 intermediate, written + read
    part = T * K * H * 4                # f32 partial scatter (op 86 write)
    return dict(weights=w_gu + w_dn, act=a, fu=fu, part=part,
                total=w_gu + w_dn + a + fu + part)


def flash_pairs(T, ctx0=0):
    """(query,key) pairs a causal chunk of T queries starting at ctx0 must visit."""
    dense = sum(ctx0 + i + 1 for i in range(T))
    sparse = sum(min(ctx0 + i + 1, TOPK_IDX) for i in range(T))
    return dense, sparse


def flash_flops(pairs):
    """bf16 MACs*2 for absorbed MLA: QK over (kv_lora+rope), PV over kv_lora."""
    return 2 * pairs * NH_L * (KV_LORA + ROPE + KV_LORA)


def main():
    print("== MoE grouped pair, DRAM bytes per layer per rank (weights are T-INVARIANT)")
    print(f"{'T':>7}{'weights MB':>12}{'act MB':>9}{'fu MB':>8}{'part MB':>10}"
          f"{'total MB':>11}{'MB/token':>10}")
    for T in (1024, 2048, 4096, 8192):
        b = moe_bytes(T)
        print(f"{T:>7}{b['weights']/MB:>12.0f}{b['act']/MB:>9.1f}{b['fu']/MB:>8.1f}"
              f"{b['part']/MB:>10.0f}{b['total']/MB:>11.0f}{b['total']/MB/T:>10.3f}")

    print("\n== attention pair counts, whole prompt (chunked at %d)" % CHUNK)
    print(f"{'T':>7}{'chunks':>8}{'dense pairs':>14}{'top-2048 pairs':>16}"
          f"{'sparse/dense':>14}{'dense GFLOP/lay/rank':>22}")
    for T in (1024, 2048, 4096, 8192, 16384, 32768):
        d = s = 0
        rem, ctx = T, 0
        while rem > 0:
            c = min(rem, CHUNK)
            dd, ss = flash_pairs(c, ctx)
            d += dd; s += ss; ctx += c; rem -= c
        print(f"{T:>7}{(T+CHUNK-1)//CHUNK:>8}{d:>14.4g}{s:>16.4g}"
              f"{s/d:>14.4f}{flash_flops(d)/1e9:>22.1f}")

    print("\n== per-chunk MoE weight-stream floor, whole model, one rank")
    w = moe_bytes(1024)['weights'] * N_MOE + \
        (E and 0) + 2 * DI * H * N_DENSE + DI * H * N_DENSE
    print(f"routed-expert weights: {moe_bytes(0)['weights']/GB*N_MOE:.1f} GB per chunk "
          f"({N_MOE} MoE layers)")
    for bw in (5.3, 3.5, 1.53, 1.31):
        print(f"  at {bw:>4.2f} TB/s -> {moe_bytes(0)['weights']*N_MOE/(bw*1e12)*1e3:>7.1f} ms")


main()
