"""Per-instruction roofline for every packet op, priced at a declared precision contract.

`packet_roofline.analyze` prices weight streams only and skips non-GEMM ops, so it cannot rank
prefill rungs, where activations, attention and collectives dominate. This prices every
instruction as max(HBM, matrix, fabric) with activation traffic, at the dtype the reference
engine runs (`contract="vllm029-mxfp4"`), not at whatever the packet currently implements —
so the floor is the target a matched-precision kernel must hit.

    python3 scripts/campaign/op_roofline.py disasm.txt --ctx 8192 --out roof.json [--md roof.md]
"""
import argparse
import collections
import json
import math
import re
import struct

from packet_roofline import mxfp4_weight_bytes, scale_bytes

LINE = re.compile(r"^#(\d+)\s+(\w+)\s+b=(\d+)\s*(.*)$")
PROGRAM = re.compile(r"^===== program T=(\d+)")
MLA_LATENT, MLA_ROPE = 512, 64

# Weight-name suffix -> precision role under pinned vLLM 0.29 serving the AMD Quark
# MXFP4/AttnFP8 checkpoint (plans/glm53-mxfp4-preparation.md, glm53-precision-parity.md).
ROLES = (
    ("self_attn.q_a_proj", "fp8"), ("self_attn.kv_a_proj", "fp8"),
    ("self_attn.derived.kv_a_latent", "fp8"), ("self_attn.derived.k_rope", "fp8"),
    ("self_attn.derived.q_absorb", "fp8"), ("self_attn.derived.q_rope", "fp8"),
    ("self_attn.derived.v_absorb", "fp8"), ("self_attn.o_proj", "fp8"),
    ("self_attn.indexer.wq_b", "fp8"), ("self_attn.indexer.wk", "bf16"),
    ("self_attn.indexer.weights_proj", "bf16"),
    ("mlp.gate.weight", "bf16"), ("lm_head", "bf16"), ("embed_tokens", "bf16"),
    ("mlp.shared_experts", "mxfp4"), ("mlp.gate_proj", "mxfp4"), ("mlp.up_proj", "mxfp4"),
    ("mlp.down_proj", "mxfp4"), ("mlp.dense_weight_table", "mxfp4"),
    ("mlp.expert_weight_table", "mxfp4"),
)


def role(tail):
    for suffix, r in ROLES:
        if suffix in tail:
            return r
    return None


def w_bytes(r, n, k):
    if r == "mxfp4":
        return mxfp4_weight_bytes(n, k)
    if r == "fp8":
        return n * k + scale_bytes(n, k)
    return 2 * n * k


def a_bytes(r, m, k):
    """Activation operand bytes: A4 (group32 E8M0), A8 (group128 F32), else BF16."""
    if r == "mxfp4":
        return m * ((k + 1) // 2 + (k + 31) // 32)
    if r == "fp8":
        return m * (k + (k + 127) // 128 * 4)
    return 2 * m * k


def union(rows, topk, experts):
    """Expected distinct experts touched by `rows` uniform top-k draws."""
    return experts * (1 - (1 - topk / experts) ** rows)


def attn_keys(t, topk):
    """Sum over causal query rows of attended keys, capped at the sparse top-k."""
    if not topk or t <= topk:
        return t * (t + 1) // 2
    return topk * (topk + 1) // 2 + (t - topk) * topk


def gemm(r, m, n, k, out=2, glu=1):
    return {"hbm": glu * w_bytes(r, n, k) + a_bytes(r, m, k) + out * m * n,
            "flops": glu * 2 * m * n * k, "dtype": r}


def cost(op, p, tail, rows, ctx, heads, topk, n_gpu):
    """(hbm bytes, matrix flops, dtype, fabric bytes) for one instruction, or None."""
    g = p.get
    r = role(tail) or "bf16"
    m = g("M", rows)
    if op in {"Gemm", "GemmMed", "GemmSmall", "GemmWide", "Gemv",
              "GemmMxfp4", "GemmSmallMxfp4", "GemmMedMxfp4", "GemmWideMxfp4", "GemvMxfp4"}:
        return gemm(r, m, g("N"), g("K"))
    if op in {"GemmFp8Block128", "GemmFp8Block128Split4"}:
        return gemm("fp8", m, g("N"), g("K"))
    if op == "QuantFp8Block128":  # bf16 in, e4m3 + one f32 per 128 out
        m, k = g("M", rows), g("K")
        return {"hbm": 2 * m * k + m * k + 4 * m * (k // 128), "flops": 0, "dtype": "bf16"}
    if op == "MlaBmmFp8":  # per-head [M x K] x [K x N], e4m3 weights, A quantized in-kernel
        m, hh, n, k = g("M", rows), g("heads", heads), g("N"), g("K")
        return {"hbm": hh * (n * k + 4) + 2 * m * hh * k + 2 * m * hh * n,
                "flops": 2 * m * hh * n * k, "dtype": "fp8"}
    if op == "FlashMerge":  # normalize f32 latent partials to bf16
        n, hh, ns, hd = g("n_batch", rows), g("n_head", heads), g("nsplit", 1), g("hd", MLA_LATENT)
        return {"hbm": 4 * n * hh * hd * ns + 2 * n * hh * hd, "flops": 0, "dtype": "bf16"}
    if op == "ZeroF32":
        return {"hbm": 4 * g("M", rows) * g("N"), "flops": 0, "dtype": "f32"}
    if op == "GemmF32":
        return gemm("bf16", m, g("N"), g("K"), out=4)
    if op == "GemvQkv":
        return gemm(r, m, g("Nq") + g("Nk") + g("Nv"), g("K"))
    if op in {"GemmGluMxfp4", "GemvGluMxfp4", "GemmGlu", "GemvGlu"}:
        c = gemm(r, m, g("N"), g("K"), glu=2)
        c["hbm"] -= m * g("N") * 2  # two accumulators, one gated output
        return c
    if op in {"MoeGroupGluPf", "MoeGroupDownPf", "MoeExpertGluFp8Blk", "MoeExpertDownFp8Blk"}:
        i, h, e = g("I_moe"), g("H"), g("n_exp")
        t, k = g("T", rows), g("k", 8 if e > 1 else 1)
        if op.startswith("MoeExpert"):
            # decode per-slot instruction: one expert, `rows` tokens routed to it at most
            t, k, e = rows, 1, 1
        used = union(t, k, e) if e > 1 else 1
        slots = t * k
        if "Glu" in op:
            return {"hbm": used * 2 * mxfp4_weight_bytes(i, h) + a_bytes("mxfp4", slots, h) + 2 * slots * i,
                    "flops": 2 * 2 * slots * i * h, "dtype": "mxfp4"}
        return {"hbm": used * mxfp4_weight_bytes(h, i) + a_bytes("mxfp4", slots, i) + 2 * slots * h,
                "flops": 2 * slots * h * i, "dtype": "mxfp4"}
    if op == "FlashMlaPrefill":
        keys = attn_keys(rows, topk)
        qk, pv = MLA_LATENT + MLA_ROPE, MLA_LATENT
        return {"hbm": 2 * rows * (heads * qk + qk + heads * pv),
                "flops": 2 * heads * keys * (qk + pv), "dtype": "bf16"}
    if op == "FlashMlaDecode":
        n = g("n_batch", rows)
        keys = min(ctx, topk) if topk else ctx
        qk = MLA_LATENT + MLA_ROPE
        return {"hbm": 2 * n * keys * qk + 2 * n * heads * (qk + MLA_LATENT),
                "flops": 2 * n * g("n_head", heads) * keys * (qk + MLA_LATENT), "dtype": "bf16"}
    if op == "MlaMergeFold":
        n, hh, v = g("n_batch", rows), g("n_head", heads), g("V")
        split = g("nsplit", 1)
        return {"hbm": w_bytes("fp8", hh * v, MLA_LATENT) + 4 * n * hh * MLA_LATENT * split + 2 * n * hh * v,
                "flops": 2 * n * hh * MLA_LATENT * v, "dtype": "fp8"}
    qk = MLA_LATENT + MLA_ROPE
    kv_row = 2 * qk  # vLLM 0.29 main MLA cache resolves to BF16 for this checkpoint
    if op == "FlashMlaPrefillFp8":
        t, hh = g("n_tok", rows), g("n_head", heads)
        return {"hbm": 2 * t * hh * (qk + MLA_LATENT) + t * kv_row,
                "flops": 2 * hh * attn_keys(t, topk) * (qk + MLA_LATENT), "dtype": "bf16"}
    if op == "FlashMlaDecodeFp8":
        n, keys = g("n_batch", rows), min(ctx, topk or ctx)
        return {"hbm": n * keys * kv_row + 2 * n * heads * (qk + MLA_LATENT),
                "flops": 2 * n * g("n_head", heads) * keys * (qk + MLA_LATENT), "dtype": "bf16"}
    if op in {"IndexScore", "IndexScorePf"}:
        # FP8 key + F32 scale per position (132 B), FP8 per-token-group query, F32 scores
        ih, hd = g("index_heads"), g("index_head_dim")
        if op == "IndexScore":
            n = g("n_batch", rows)
            keys, fresh = n * ctx, n * ctx
        else:
            n = g("n_tok", rows)
            keys, fresh = attn_keys(n, 0), n
        return {"hbm": fresh * (hd + 4) + n * ih * (hd + 4) + 4 * keys,
                "flops": 2 * ih * hd * keys, "dtype": "fp8"}
    if op == "IndexSelectPf":
        n = g("n_tok", rows)
        return {"hbm": 4 * attn_keys(n, 0) + 4 * n * g("top_k"), "flops": 0, "dtype": "f32"}
    if op == "IndexSelect":
        return {"hbm": 4 * ctx + 4 * g("top_k"), "flops": 0, "dtype": "f32"}  # one row each
    if op == "IndexUnionPf":
        return {"hbm": 8 * g("n_tok", rows) * g("top_k"), "flops": 0, "dtype": "i32"}
    if op == "HeadNormRopeFp8":
        return {"hbm": 3 * rows * qk, "flops": 0, "dtype": "bf16"}
    if op == "DcpKvScatter":
        return {"hbm": 2 * g("rows", rows) * kv_row, "flops": 0, "dtype": "bf16"}
    if op == "DcpKvPack":
        keys = min(ctx, topk or ctx)
        return {"hbm": 2 * g("n_batch", 1) * keys * kv_row, "flops": 0, "dtype": "bf16"}
    if op == "XDcpGather":
        moved = g("n_batch", 1) * min(ctx, topk or ctx) * kv_row
        return {"hbm": 2 * moved, "flops": 0, "dtype": "bf16", "fabric": moved * (n_gpu - 1) / n_gpu}
    if op == "LayerNorm":
        return {"hbm": 4 * g("rows") * g("feat") + 4 * g("feat"), "flops": 0, "dtype": "bf16"}
    if op == "XArgmaxFin":
        return {"hbm": 8 * g("n_gpu"), "flops": 0, "dtype": "f32"}
    if op == "RmsNorm":
        return {"hbm": 4 * g("rows") * g("feat") + 2 * g("feat"), "flops": 0, "dtype": "bf16"}
    if op == "Residual":
        return {"hbm": 6 * g("n"), "flops": 0, "dtype": "bf16"}
    if op == "Glu":
        return {"hbm": 6 * g("n"), "flops": 0, "dtype": "bf16"}
    if op == "HeadNormRope":
        return {"hbm": 4 * g("ntok") * g("nhead") * g("hd"), "flops": 0, "dtype": "bf16"}
    if op == "Embed":
        return {"hbm": 4 * g("ntok") * g("hidden"), "flops": 0, "dtype": "bf16"}
    if op in {"MoeRouterTopkPf", "MoeRouterTopk"}:
        return {"hbm": 4 * g("T", rows) * g("n_exp"), "flops": 0, "dtype": "f32"}
    if op == "MoeAlignPf":
        return {"hbm": 8 * g("T") * g("k"), "flops": 0, "dtype": "i32"}
    if op in {"MoeCombinePf", "MoeCombine"}:
        t = g("T", rows)
        return {"hbm": 2 * t * g("k") * g("H") + 2 * t * g("H"), "flops": 0, "dtype": "bf16"}
    if op in {"Argmax", "ArgmaxFin"}:
        return {"hbm": 4 * g("n", 0), "flops": 0, "dtype": "f32"}
    if op == "XReduceTwoShot":
        n = g("n")
        # two-shot all-reduce: reduce-scatter + all-gather, (g-1)/g of the tensor each way
        return {"hbm": 6 * n, "flops": 0, "dtype": "bf16", "fabric": 2 * 2 * n * (n_gpu - 1) / n_gpu}
    if op == "XReduce":
        n = g("H")  # already rows * hidden
        return {"hbm": 6 * n, "flops": 0, "dtype": "bf16", "fabric": 2 * 2 * n * (n_gpu - 1) / n_gpu}
    return None


def analyze(disasm, ctx, bw_gbps, ceilings, fabric_gbps, topk=0):
    programs = collections.OrderedDict()
    cur = None
    heads = 8
    phase = "prefill"
    for line in disasm.splitlines():
        pm = PROGRAM.match(line)
        if pm:
            rows = int(pm.group(1))
            if cur is not None and rows < cur["rows"]:
                phase = "decode"  # plowc orders prefill rungs, then decode rungs, each ascending
            cur = {"rows": rows, "ops": collections.OrderedDict(), "excluded": collections.Counter(),
                   "inst_keys": {}}
            programs[f"{phase}-{rows}"] = cur
            continue
        lm = LINE.match(line.strip())
        if cur is None or not line.startswith("#"):
            continue
        if not lm:
            raise ValueError(f"unparsed instruction line: {line!r}")
        op, tail = lm.group(2), lm.group(4)
        seg = tail.split("|", 1)
        p = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)(?=\s|$)", seg[1] if len(seg) > 1 else "")}
        if op == "MlaMergeFold":
            heads = p.get("n_head", heads)
        c = cost(op, p, seg[0], cur["rows"], ctx, heads, topk, p.get("n_gpu", 8))
        if c is None:
            cur["excluded"][op] += 1
            continue
        key = f"{op}:{role(seg[0]) or c['dtype']}:" + " ".join(
            f"{k}={p[k]}" for k in ("M", "N", "K", "Nq", "I_moe", "H", "n_exp", "rows", "feat", "n", "n_batch") if k in p)
        cur["inst_keys"][int(lm.group(1))] = key
        mem = c["hbm"] / (bw_gbps * 1e3)
        mat = c["flops"] / (ceilings[c["dtype"]] * 1e6) if c["flops"] else 0.0
        fab = c.get("fabric", 0) / (fabric_gbps * 1e3)
        floor = max(mem, mat, fab)
        row = cur["ops"].setdefault(key, {"op": op, "dtype": c["dtype"], "count": 0, "hbm_bytes": 0,
                                          "flops": 0, "fabric_bytes": 0, "floor_us": 0.0,
                                          "bound": None, "one_floor_us": floor})
        row["count"] += 1
        row["hbm_bytes"] += c["hbm"]
        row["flops"] += c["flops"]
        row["fabric_bytes"] += c.get("fabric", 0)
        row["floor_us"] += floor
        row["bound"] = ("fabric" if fab == floor else "matrix" if mat == floor else "hbm")
    out = {}
    for name, prog in programs.items():
        total = sum(r["floor_us"] for r in prog["ops"].values())
        ops = sorted(prog["ops"].items(), key=lambda kv: -kv[1]["floor_us"])
        out[name] = {"rows": prog["rows"], "floor_us": total,
                     "tokens_per_s_floor": prog["rows"] / total * 1e6 if total else None,
                     "ops": dict(ops), "excluded": dict(prog["excluded"]),
                     "inst_keys": prog["inst_keys"]}
    return {"scope": "per-instruction max(HBM, matrix, fabric) at the vLLM 0.29 MXFP4/AttnFP8 dtype "
                     "contract; serialized sum, no overlap, no launch/sync floor, weights streamed once",
            "ctx": ctx, "sparse_topk": topk, "bandwidth_gbps": bw_gbps, "fabric_gbps": fabric_gbps,
            "matrix_tflops": ceilings, "programs": out}


def attach_trace(prog, data, clock_hz):
    """Per-op measured time from a PLOW_TRACE_RAW dump of exactly this program.

    Body envelope per instruction = last workgroup end - first workgroup ready. Instructions
    overlap, so the per-op sums are attribution, not a serialized wall-time breakdown."""
    rec = struct.Struct("<IIIHHQQQ")
    if not data or len(data) % rec.size:
        raise ValueError("incomplete trace")
    first, last, arrive = {}, {}, None
    for _, _, inst, _, _, a, ready, end in rec.iter_unpack(data):
        if a == ready == end == 0:
            continue
        first[inst] = min(first.get(inst, ready), ready)
        last[inst] = max(last.get(inst, end), end)
        arrive = a if arrive is None else min(arrive, a)
    keys = prog["inst_keys"]
    if not set(first) <= set(keys) | set(range(max(keys, default=0) + 1)):
        raise ValueError("trace instructions outside the program")
    tick_us = 1e6 / clock_hz
    for row in prog["ops"].values():
        row["measured_us"] = 0.0
    for inst, key in keys.items():
        if inst in first:
            prog["ops"][key]["measured_us"] += (last[inst] - first[inst]) * tick_us
    prog["trace_wall_us"] = (max(last.values()) - arrive) * tick_us if last else None
    prog["traced_insts"] = len(first)
    return prog


def markdown(result, top):
    lines = []
    for name, prog in result["programs"].items():
        lines.append(f"### {name}  floor {prog['floor_us'] / 1e3:.3f} ms  "
                     f"({prog['tokens_per_s_floor']:.0f} tok/s/rank-step)")
        traced = prog.get("trace_wall_us") is not None
        if traced:
            lines.append(f"trace wall {prog['trace_wall_us'] / 1e3:.3f} ms = "
                         f"{100 * prog['floor_us'] / prog['trace_wall_us']:.1f}% of floor-sum roof")
        lines.append("| op | dtype | n | bound | floor ms | share |" + (" measured ms | % roof |" if traced else ""))
        lines.append("|---|---|---:|---|---:|---:|" + ("---:|---:|" if traced else ""))
        rows = list(prog["ops"].items())
        if traced:
            rows.sort(key=lambda kv: -kv[1].get("measured_us", 0))
        for key, r in rows[:top]:
            extra = ""
            if traced:
                m = r.get("measured_us", 0)
                extra = f" {m / 1e3:.3f} | {100 * r['floor_us'] / m if m else 0:.1f}% |"
            lines.append(f"| `{key}` | {r['dtype']} | {r['count']} | {r['bound']} | "
                         f"{r['floor_us'] / 1e3:.3f} | {100 * r['floor_us'] / prog['floor_us']:.1f}% |" + extra)
        if prog["excluded"]:
            lines.append(f"\nexcluded: {prog['excluded']}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("disasm")
    ap.add_argument("--ctx", type=int, required=True, help="decode live context per row")
    ap.add_argument("--sparse-topk", type=int, default=0, help="DSA top-k; 0 = dense attention")
    ap.add_argument("--bw", type=float, default=6200.0, help="measured HBM GB/s")
    ap.add_argument("--fabric", type=float, default=1075.0, help="xGMI GB/s per GPU")
    ap.add_argument("--bf16", type=float, default=2300.0)
    ap.add_argument("--fp8", type=float, default=4600.0)
    ap.add_argument("--mxfp4", type=float, default=9200.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--trace", action="append", default=[], metavar="PROGRAM=FILE",
                    help="attach a PLOW_TRACE_RAW dump to a program, e.g. prefill-8192=T8192.trace.prefill")
    ap.add_argument("--trace-clock-hz", type=float, default=100e6)
    a = ap.parse_args()
    ceilings = {"bf16": a.bf16, "fp8": a.fp8, "mxfp4": a.mxfp4, "f32": a.bf16, "i32": a.bf16}
    result = analyze(open(a.disasm).read(), a.ctx, a.bw, ceilings, a.fabric, a.sparse_topk)
    for spec in a.trace:
        name, path = spec.split("=", 1)
        attach_trace(result["programs"][name], open(path, "rb").read(), a.trace_clock_hz)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    if a.md:
        with open(a.md, "w") as f:
            f.write(markdown(result, a.top))
    for name, prog in result["programs"].items():
        wall = prog.get("trace_wall_us")
        print(f"{name:22s} floor {prog['floor_us'] / 1e3:9.3f} ms"
              + (f"   traced {wall / 1e3:9.3f} ms ({100 * prog['floor_us'] / wall:5.1f}% of roof)" if wall else "")
              + f"   excluded {prog['excluded']}")


if __name__ == "__main__":
    main()
