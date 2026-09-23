"""Optimistic per-rank decode traffic floors from the dispatched packet geometry."""
import collections
import math
import re
import struct

NATIVE_FP8_GEMMS = {"GemmFp8Block128", "GemmFp8Block128Split4"}
NATIVE_ROUTED_FP8 = {"MoeGluFp8Block128", "MoeDownFp8Block128"}
NATIVE_MLA_FP8 = {"MlaBmmFp8"}
NATIVE_INDEXER_FP8 = {"IndexFp8Decode"}


def routed_experts(table, rows, topk, experts):
    if len(table) != rows * topk * 8:
        raise ValueError("router table must contain exactly T*k {u32 id, f32 gate} entries")
    selected = set()
    for row in range(rows):
        ids = set()
        for column in range(topk):
            expert, gate = struct.unpack_from("<If", table, (row * topk + column) * 8)
            if expert >= experts or not math.isfinite(gate) or gate < 0 or expert in ids:
                raise ValueError("invalid expert id/gate or duplicate expert within a row")
            ids.add(expert)
        selected.update(ids)
    return len(selected)


def scale_bytes(n, k):
    return ((n + 127) // 128) * ((k + 127) // 128) * 4


def decode_cost(op, p, ctx):
    m, n, k = p.get("M", 1), p.get("N", 0), p.get("K", 0)
    if op in NATIVE_INDEXER_FP8:
        if m not in (1, 8, 16, 32, 64) or p["ctx"] % 16 or not 1 <= ctx <= p["ctx"] <= 131072:
            raise ValueError("invalid native indexer geometry/live context")
        return m * ctx * 132, 2 * m * ctx * 32 * 128
    if op in NATIVE_MLA_FP8:
        return p["heads"] * n * k + 4, 2 * m * p["heads"] * n * k
    if op in {"Gemv", "Gemm", "GemmMed", "GemmSmall", "GemmWide", "GemmF32"}:
        # GemmF32 is the BF16 router with FP32 output/accumulation.
        return 2 * n * k, 2 * m * n * k
    if op == "GemvQkv":
        n = sum(p.get(x, 0) for x in ("Nq", "Nk", "Nv"))
        return 2 * n * k, 2 * m * n * k
    if op in {"GemvGlu", "GemmGlu"}:
        return 4 * n * k, 4 * m * n * k
    if op in {"GemvFp8Blk", "DenseGluFp8Blk"} | NATIVE_FP8_GEMMS:
        factor = 2 if op == "DenseGluFp8Blk" else 1
        return factor * (n * k + scale_bytes(n, k)), factor * 2 * m * n * k
    if op in {"MoeExpertGluFp8Blk", "MoeExpertDownFp8Blk"}:
        n, k = p["I_moe"], p["H"]
        factor = 2 if op == "MoeExpertGluFp8Blk" else 1
        return factor * (n * k + scale_bytes(n, k)), factor * 2 * n * k
    if op in {"MoeGroupGluPf", "MoeGroupDownPf"} | NATIVE_ROUTED_FP8:
        n, k = p["I_moe"], p["H"]
        rows, topk, experts = p["T"], p["k"], p["n_exp"]
        # Distribution-free optimistic union: every row may route to the same experts.
        union = p.get("routed_experts", min(topk, experts))
        factor = 2 if op in {"MoeGroupGluPf", "MoeGluFp8Block128"} else 1
        return factor * union * (n * k + scale_bytes(n, k)), factor * 2 * rows * topk * n * k
    if op in {"FlashMlaDecode", "FlashGatherDecode"}:
        keys = min(ctx, p["top_k"]) if op == "FlashGatherDecode" else ctx
        rows = p.get("n_batch", 1)
        # GLM MLA latent=512, rotary=64. These op ABIs fix the latent width.
        return rows * keys * 576 * 2, 4 * rows * p["n_head"] * keys * 512 + 2 * rows * p["n_head"] * keys * 64
    if op == "MlaMergeFold":
        heads, v, rows = p["n_head"], p["V"], p.get("n_batch", 1)
        return heads * 512 * v * 2, 2 * rows * heads * 512 * v
    return None


def analyze(disasm, ctx, bandwidth_gbps, tflops, router_table=None, fp8_tflops=None):
    if ctx < 1 or bandwidth_gbps <= 0 or tflops <= 0:
        raise ValueError("context and hardware ceilings must be positive")
    if fp8_tflops is not None and (not math.isfinite(fp8_tflops) or fp8_tflops <= 0):
        raise ValueError("FP8 ceiling must be finite and positive")
    parts = collections.defaultdict(lambda: {"count": 0, "bytes": 0, "flops": 0})
    excluded = collections.Counter()
    programs = re.findall(r"^===== program T=(\d+)", disasm, re.M)
    if len(programs) != 1:
        raise ValueError("roofline requires exactly one disassembled program")
    grouped_rows = None
    union = None
    router_count = 0
    for line in disasm.splitlines():
        match = re.match(r"^#\d+\s+(\w+)\s+b=\d+\s+.*?\|\s*(.*)$", line)
        if not match:
            continue
        op, fields = match.groups()
        params = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)(?=\s|$)", fields)}
        if op in NATIVE_INDEXER_FP8 and params.get("M") != int(programs[0]):
            raise ValueError("native indexer rows disagree with decode rung")
        if op == "MoeAlignPf":
            grouped_rows = params
            router_count += 1
            if router_table is not None:
                if router_count != 1 or params["T"] != int(programs[0]):
                    raise ValueError("captured routing requires one matching decode router")
                union = routed_experts(router_table, params["T"], params["k"], params["n_exp"])
        if op in NATIVE_ROUTED_FP8:
            if (grouped_rows is None or params["T"] != grouped_rows["T"]
                    or params.get("topk", grouped_rows["k"]) != grouped_rows["k"]):
                raise ValueError("native routed shape disagrees with preceding MoeAlignPf")
            params.update(I_moe=params["I"], n_exp=params["E"])
        if op in {"MoeGroupGluPf", "MoeGroupDownPf"} | NATIVE_ROUTED_FP8:
            if grouped_rows is None or grouped_rows["n_exp"] != params["n_exp"]:
                raise ValueError("grouped expert shape requires its preceding MoeAlignPf")
            params.update(T=grouped_rows["T"], k=grouped_rows["k"])
            if union is not None:
                params["routed_experts"] = union
        cost = decode_cost(op, params, ctx)
        if cost is None:
            excluded[op] += 1
            continue
        row = parts[op]
        row["count"] += 1
        row["bytes"] += cost[0]
        row["flops"] += cost[1]
    if not parts:
        raise ValueError("no costed packet instructions")
    if router_table is not None and union is None:
        raise ValueError("captured router table has no grouped-expert packet consumer")
    for op, row in parts.items():
        if op in NATIVE_FP8_GEMMS | NATIVE_ROUTED_FP8 | NATIVE_MLA_FP8 | NATIVE_INDEXER_FP8 and fp8_tflops is None:
            raise ValueError("native FP8 GEMM requires an explicit FP8 compute ceiling")
        ceiling = fp8_tflops if op in NATIVE_FP8_GEMMS | NATIVE_ROUTED_FP8 | NATIVE_MLA_FP8 | NATIVE_INDEXER_FP8 else tflops
        row["matrix_tflops"] = ceiling
        row["memory_floor_us"] = row["bytes"] / (bandwidth_gbps * 1e3)
        row["compute_floor_us"] = row["flops"] / (ceiling * 1e6)
        row["arithmetic_intensity_flop_per_byte"] = row["flops"] / row["bytes"]
    total_bytes = sum(p["bytes"] for p in parts.values())
    total_flops = sum(p["flops"] for p in parts.values())
    return {
        "scope": "optimistic per-rank decode HBM/compute lower bound",
        "batch": int(programs[0]), "ctx": ctx,
        "routed_experts": union,
        "bandwidth_gbps": bandwidth_gbps, "matrix_tflops": tflops,
        "fp8_matrix_tflops": fp8_tflops,
        "bytes": total_bytes, "flops": total_flops,
        "lower_bound_us": max(total_bytes / (bandwidth_gbps * 1e3),
                              sum(p["compute_floor_us"] for p in parts.values())),
        "components": dict(sorted(parts.items(), key=lambda item: -item[1]["memory_floor_us"])),
        "excluded_ops": dict(excluded),
        "limitations": ["No launch, fabric, reduction, or activation-traffic floor",
                        "Split-K partial stores/reduction traffic excluded; partitions do not multiply weights or GEMM FLOPs",
                        "Weights streamed once; no cache reuse or repeated tile reads modeled",
                        ("Grouped MoE assumes maximum expert reuse across rows" if union is None else
                         "Grouped MoE streams captured selected experts once; repeated tile reads excluded"),
                        "Configured ceilings are not a same-session hardware measurement"] + (
                            ["Indexer assumes all rows live at the supplied context; ragged/parked rows need per-row lengths",
                             "Indexer counts each logical FP8 key+F32 scale once, assuming a cold cache; physical HBM transactions/reloads are not measured",
                             "Indexer append, query preparation, page-table, score-output and scalar reduction costs excluded"]
                            if NATIVE_INDEXER_FP8.intersection(parts) else []),
    }
