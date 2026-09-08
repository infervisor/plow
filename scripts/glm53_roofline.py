#!/usr/bin/env python3
"""Per-op roofline for a compiled plow blob, computed from the packet stream itself.

WHY FROM THE BLOB. A roofline written from the model card is a roofline for a model nobody
runs: it misses that the attention LoRA down-projections stay REPLICATED across ranks, that
the MoE experts are block-fp8 with a scale grid, and that the emitted per-rank shapes are what
they are. `plowrt disasm` prints the shapes that will actually execute, so this reads those.

    plowrt disasm <assets> --program 1 > decode.txt
    python3 scripts/glm53_roofline.py decode.txt --bw 5325 --label "TP4 decode"

`--bw` is GB/s. Pass the MEASURED number when you have one; the MI300X spec sheet says 5325
and no real kernel sees that.
"""
import argparse, re, sys, collections

AP = argparse.ArgumentParser()
AP.add_argument("disasm")
AP.add_argument("--bw", type=float, default=5325.0, help="HBM GB/s used for the memory roof")
AP.add_argument("--flops", type=float, default=1307.0, help="BF16 matrix TFLOP/s for the compute roof")
AP.add_argument("--label", default="")
AP.add_argument("--measured-ms", type=float, default=None, help="measured time for this program")
AP.add_argument("--top", type=int, default=14)
AP.add_argument("--experts", type=int, default=256,
                help="routed experts the grouped MoE prefill touches (all of them at T>=~256)")
AP.add_argument("--tokens-per-expert", type=float, default=None,
                help="mean tokens per expert = T*top_k/n_exp; default derived from --experts")
A = AP.parse_args()
EXPERTS = A.experts
TOK_PER_EXPERT = A.tokens_per_expert or 1.0

LINE = re.compile(r"^#(\d+)\s+(\w+)\s+b=(\d+)\s+(.*)$")


def params(tail):
    seg = tail.split("|", 1)
    if len(seg) < 2:
        return {}
    out = {}
    for tok in seg[1].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
    return out


def blk_scale_bytes(n, k):
    """f32 scale grid for a [128,128]-block-fp8 matrix."""
    return ((n + 127) // 128) * ((k + 127) // 128) * 4


def cost(op, p):
    """(weight+KV bytes moved, FLOPs) for one packet. M=1 decode unless stated."""
    g = p.get
    if op == "Gemv":
        n, k = g("N", 0), g("K", 0)
        return n * k * 2, 2 * n * k
    if op == "GemvFp8Blk":
        n, k = g("N", 0), g("K", 0)
        return n * k + blk_scale_bytes(n, k), 2 * n * k
    if op == "GemvQkv":
        k = g("K", 0)
        n = g("Nq", 0) + g("Nk", 0) + g("Nv", 0)
        return n * k * 2, 2 * n * k
    if op == "GemvGlu":
        n, k = g("N", 0), g("K", 0)
        return 2 * n * k * 2, 2 * 2 * n * k
    if op == "DenseGluFp8Blk":
        n, k = g("N", 0), g("K", 0)
        return 2 * (n * k + blk_scale_bytes(n, k)), 2 * 2 * n * k
    if op == "MoeExpertGluFp8Blk":
        i, h = g("I_moe", 0), g("H", 0)
        return 2 * (i * h + blk_scale_bytes(i, h)), 2 * 2 * i * h
    if op == "MoeExpertDownFp8Blk":
        i, h = g("I_moe", 0), g("H", 0)
        return h * i + blk_scale_bytes(h, i), 2 * h * i
    if op == "FlashMlaDecode":
        # MLA latent cache: one (kv_lora + rope) row per position, SHARED across heads.
        kv = g("kv_stride", 0)
        return kv * (512 + 64) * 2, 2 * g("n_head", 0) * kv * (512 + 64)
    if op == "MlaMergeFold":
        # value absorb, bf16, per head
        return g("n_head", 0) * 512 * g("V", 0) * 2, 2 * g("n_head", 0) * 512 * g("V", 0)
    # ---- prefill ops. `--experts` is the routed-expert count the grouped MoE
    # actually touches; at T >= a few hundred, top-8 over 256 reaches all of them.
    if op in ("Gemm", "GemmMed", "GemmSmall", "GemmWide"):
        m_, n, k = g("M", 0), g("N", 0), g("K", 0)
        return n * k * 2 + (m_ * k + m_ * n) * 2, 2 * m_ * n * k
    if op == "GemmGlu":
        m_, n, k = g("M", 0), g("N", 0), g("K", 0)
        return 2 * n * k * 2 + (m_ * k + 2 * m_ * n) * 2, 2 * 2 * m_ * n * k
    if op == "MoeGroupGluPf":
        i, h = g("I_moe", 0), g("H", 0)
        e = EXPERTS
        return e * 2 * (i * h + blk_scale_bytes(i, h)), 2 * 2 * TOK_PER_EXPERT * i * h * e
    if op == "MoeGroupDownPf":
        i, h = g("I_moe", 0), g("H", 0)
        e = EXPERTS
        return e * (h * i + blk_scale_bytes(h, i)), 2 * TOK_PER_EXPERT * h * i * e
    if op == "FlashMlaPrefill":
        t = g("n_batch", 0) or g("T", 0)
        return t * (512 + 64) * 2 * 2, 2 * g("n_head", 0) * t * t * (512 + 64)
    if op in ("MoeCombinePf", "MoeAlignPf", "MoeRouterTopkPf"):
        t = g("T", 0)
        return t * g("H", g("atom_h", 0)) * 4, t * g("n_exp", 0)
    if op == "XReduceTwoShot":
        return 0, 0
    if op == "Embed":
        return g("hidden", 0) * 2, 0
    if op in ("RmsNorm", "AddNorm"):
        f = g("feat", 0)
        return f * 2 * 3, 5 * f
    if op == "Residual":
        n = g("n", 0)
        return n * 2 * 3, n
    if op == "XReduce":
        # the collective moves H per rank over the fabric, not over HBM; counted apart.
        return 0, 0
    if op == "MoeRouterTopk":
        return g("n_exp", 0) * 4, g("n_exp", 0) * 8
    if op == "MoeCombine":
        return g("H", 0) * g("k", 0) * 4, g("H", 0) * g("k", 0)
    if op == "HeadNormRope":
        return g("hd", 0) * g("nhead", 0) * 2 * 2, g("hd", 0) * g("nhead", 0) * 6
    if op in ("Argmax", "XArgmaxFin"):
        n = g("n", 0) or g("slot", 0)
        return n * 4, n
    return 0, 0


agg = collections.defaultdict(lambda: [0, 0, 0])  # op -> [count, bytes, flops]
for line in open(A.disasm):
    m = LINE.match(line.strip())
    if not m:
        continue
    op = m.group(2)
    b, f = cost(op, params(m.group(4)))
    a = agg[op]
    a[0] += 1
    a[1] += b
    a[2] += f

tot_b = sum(v[1] for v in agg.values())
tot_f = sum(v[2] for v in agg.values())
mem_ms = tot_b / (A.bw * 1e9) * 1e3
cmp_ms = tot_f / (A.flops * 1e12) * 1e3
roof_ms = max(mem_ms, cmp_ms)

print(f"=== {A.label or A.disasm}")
print(f"    HBM roof {A.bw:.0f} GB/s   compute roof {A.flops:.0f} TFLOP/s (bf16 matrix)")
print()
print(f"{'op':<24}{'n':>5}{'GB':>10}{'GFLOP':>10}{'AI':>8}{'mem ms':>9}{'% bytes':>9}")
for op, (n, b, f) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: A.top]:
    if b == 0 and f == 0:
        continue
    ai = f / b if b else float("inf")
    print(f"{op:<24}{n:>5}{b/1e9:>10.3f}{f/1e9:>10.1f}{ai:>8.1f}"
          f"{b/(A.bw*1e9)*1e3:>9.3f}{100*b/tot_b:>9.1f}")
print()
print(f"{'TOTAL':<24}{'':>5}{tot_b/1e9:>10.3f}{tot_f/1e9:>10.1f}"
      f"{tot_f/tot_b:>8.1f}{mem_ms:>9.3f}{100:>9.1f}")
print()
print(f"    arithmetic intensity : {tot_f/tot_b:.2f} FLOP/byte")
print(f"    ridge point          : {A.flops*1e12/(A.bw*1e9):.1f} FLOP/byte  "
      f"=> {'MEMORY' if tot_f/tot_b < A.flops*1e12/(A.bw*1e9) else 'COMPUTE'} bound")
print(f"    memory-roof time     : {mem_ms:.3f} ms")
print(f"    compute-roof time    : {cmp_ms:.3f} ms")
print(f"    ROOFLINE             : {roof_ms:.3f} ms")
if A.measured_ms:
    print(f"    measured             : {A.measured_ms:.3f} ms"
          f"   =>  {A.measured_ms/roof_ms:.1f}x off roofline"
          f"   ({100*roof_ms/A.measured_ms:.1f}% of roof)")
