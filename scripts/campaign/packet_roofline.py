"""Conditional logical-traffic roofs and observed counter-chain priorities."""
import collections
import hashlib
import json
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
    if type(ctx) is not int or ctx < 1 or any(not math.isfinite(v) or v <= 0 for v in (bandwidth_gbps, tflops)):
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
        "scope": "conditional per-rank logical operand traffic / matrix compute roof; not a physical HBM lower bound",
        "batch": int(programs[0]), "ctx": ctx,
        "routed_experts": union,
        "bandwidth_gbps": bandwidth_gbps, "matrix_tflops": tflops,
        "fp8_matrix_tflops": fp8_tflops,
        "bytes": total_bytes, "flops": total_flops,
        "logical_operand_bytes": total_bytes, "physical_hbm_bytes": None,
        "spill_hbm_bytes": None, "performance_qualified": False,
        "lower_bound_us": max(total_bytes / (bandwidth_gbps * 1e3),
                              sum(p["compute_floor_us"] for p in parts.values())),
        "components": dict(sorted(parts.items(), key=lambda item: -item[1]["memory_floor_us"])),
        "excluded_ops": dict(excluded),
        "limitations": ["Legacy lower_bound_us is conditional on the stated traffic/ceiling assumptions, not unconditional optimality",
                        "Logical operand reads need not reach HBM; cache residency can invalidate an HBM-floor interpretation",
                        "No launch, fabric, reduction, or activation-traffic floor",
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


def trace_priorities(data, program, clock_hz):
    """Observed body-envelope chain on explicit full-completion counter edges.

    Never invent a host barrier for a missing edge or sum overlapping packet spans.
    Trace provenance and the supplied clock calibration remain external assumptions.
    """
    rec = struct.Struct("<IIIHHQQQ")
    if type(clock_hz) not in (int, float) or not math.isfinite(clock_hz) or clock_hz <= 0 or not data or len(data) % rec.size:
        raise ValueError("positive calibrated clock and complete trace records required")
    insts = program["insts"]
    n = program["n_inst"]
    if type(n) is not int or n <= 0 or len(insts) != n \
            or any(type(d["idx"]) is not int for d in insts) \
            or [d["idx"] for d in insts] != list(range(n)):
        raise ValueError("complete instruction domain required")
    if any(type(d["blocks"]) is not int or d["blocks"] <= 0 for d in insts):
        raise ValueError("invalid instruction workgroup count")
    observed = [dict() for _ in insts]
    for _, _, inst, op, part, arrive, ready, end in rec.iter_unpack(data):
        if arrive == ready == end == 0:
            continue
        if inst >= n or op != insts[inst]["op"] or part >= insts[inst]["blocks"] \
                or not 0 < arrive <= ready < end or part in observed[inst]:
            raise ValueError("invalid, repeated, mismatched or unfinished trace workgroup")
        observed[inst][part] = (arrive, ready, end)
    if any(set(rows) != set(range(d["blocks"])) for rows, d in zip(observed, insts)):
        raise ValueError("trace does not cover every packet workgroup exactly once")
    starts = [min(row[1] for row in rows.values()) for rows in observed]
    ends = [max(row[2] for row in rows.values()) for rows in observed]
    durations = [end - start for start, end in zip(starts, ends)]
    predecessors = [set() for _ in insts]
    seen_counters = set()
    n_counter = program["n_counter"]
    if type(n_counter) is not int or n_counter < 0:
        raise ValueError("invalid counter domain")
    for counter in program["counters"]["per_counter"]:
        source = counter["producer"]
        if type(source) is not int or source not in range(n) \
                or type(counter["id"]) is not int or counter["id"] not in range(n_counter) \
                or counter["id"] in seen_counters or type(counter["threshold"]) is not int \
                or counter["threshold"] != insts[source]["blocks"]:
            raise ValueError("ambiguous or fine-grained counter is not a full instruction edge")
        seen_counters.add(counter["id"])
        for target in counter["consumers"]:
            if type(target) is not int or target not in range(n) or target == source:
                raise ValueError("invalid counter dependency")
            if ends[source] > starts[target]:
                raise ValueError("trace violates declared full-completion dependency")
            predecessors[target].add(source)
    if seen_counters != set(range(n_counter)):
        raise ValueError("incomplete counter inventory")
    successors = [set() for _ in insts]
    indegree = [len(pred) for pred in predecessors]
    for target, sources in enumerate(predecessors):
        for source in sources:
            successors[source].add(target)
    ready = collections.deque(i for i, degree in enumerate(indegree) if not degree)
    order = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for target in sorted(successors[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != n:
        raise ValueError("counter dependency cycle")
    prefix, previous = [0] * n, [None] * n
    for node in order:
        if predecessors[node]:
            previous[node] = max(sorted(predecessors[node]), key=lambda source: prefix[source])
        prefix[node] = durations[node] + (prefix[previous[node]] if previous[node] is not None else 0)
    suffix = [0] * n
    for node in reversed(order):
        suffix[node] = durations[node] + max((suffix[target] for target in successors[node]), default=0)
    last = max(range(n), key=lambda node: prefix[node])
    chain = prefix[last]
    path = []
    while last is not None:
        path.append(last)
        last = previous[last]
    path.reverse()
    ns = lambda ticks: ticks * 1e9 / clock_hz
    rows = [{"instruction": i, "op": d["op_name"], "body_envelope_ns": ns(durations[i]),
             "chain_slack_ns": ns(chain - (prefix[i] + suffix[i] - durations[i])),
             "on_selected_longest_chain": i in path} for i, d in enumerate(insts)]
    rows.sort(key=lambda row: (row["chain_slack_ns"], -row["body_envelope_ns"], row["instruction"]))
    wall = max(ends) - min(row[0] for records in observed for row in records.values())
    if chain > wall:
        raise ValueError("counter chain exceeds observed wall span")
    return {"scope": "observed full-counter body-envelope chain; not end-to-end critical path or predicted speedup",
            "trace_sha256": hashlib.sha256(data).hexdigest(),
            "program_sha256": hashlib.sha256(json.dumps(program, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "trace_clock_hz": clock_hz, "observed_wall_ns": ns(wall),
            "counter_chain_ns": ns(chain), "chain": path, "priorities": rows,
            "performance_qualified": False,
            "assumptions": ["Trace is from exactly this packet invocation and supplied clock is calibrated",
                            "Instrumented trace is diagnostic, not uninstrumented campaign timing",
                            "Only explicit full-completion counter edges; dropped segment/host/queue edges are not invented",
                            "Body envelope includes staggered workgroups; no attribution of gaps to launch overhead",
                            "Slack ranks a fixed observed DAG; optimization may change costs and scheduling"]}
