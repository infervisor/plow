#!/usr/bin/env python3
"""op_roof.py <disasm.txt> --ctx N [--sweep B=sweep.jsonl ...] [--segtime server.log --segtime-rows T]

Per-op roofline for a dense packet (Llama / Gemma / Qwen shapes; GLM/MLA/MoE packets: see
scripts/campaign/op_roofline.py), as markdown: bytes and FLOPs per op, the floor max(bytes/BW,
FLOPs/peak), the measured time and the floor as % of it.

  disasm      `plowrt disasm <model.pkt>` output (every program; prefill rungs first, then decode)
  --sweep     decode rung B measured with `step_bench <assets> B <ctx> 10 --warmup 4 --sweep 0..<n>`
              (lines {"cap":k,"ms":x}; cap k runs instructions 0..k-1, so inst i costs ms[i+1]-ms[i];
              cap 0 is the interpreter skeleton, cap -1 the full step). Raw step_bench logs work.
  --prefill-past  prefix tokens before one contiguous prefill chunk; overrides emitted
              q_pos0/n_kv for prefill pricing only. Does not model packed request mixtures.
  --segtime   prefill measured with PLOW_PF_SEG_TIME=1 (segment-site lines, per chunk); the chunk
              rows are --segtime-rows. SEG_TIME drains per segment: shares, not latency.
  --nsys      map CUDA graph correlations to --program-index in JSON disasm
              (produced with --format json --stream --no-analysis). Use --nsys-correlation for
              an explicit prefill graph, or --graphs for the final decode graphs. Reports measured
              fused groups, not individual-op efficiencies. Verify graph selection excludes warmup.
  no measurement: the floor of every program (one row per prefill/decode rung).

Generic weights are priced at --wbytes (2 = bf16), KV at --kvbytes; FP8 GEMVs use one-byte
weights plus FP32 row scales. FP8 GEMMs read FP8 activations and FP32 activation/weight
scales; QuantFp8 includes BF16 input and FP8 output traffic. These are compulsory-byte floors,
not measured HBM traffic (cache reuse and repeated reads are not modeled). GemvArgmax uses BF16
weights. Other activations use BF16; FP8 attention uses one-byte KV regardless of --kvbytes,
which applies to BF16 attention. Attention reads min(ctx, window) keys per row. Ceilings:
--gpu (scripts/campaign/roofline.py registry), --bw / --tflops / --fp8-tflops override. Prefer
calibrated bandwidth; an assumed or registry ceiling does not establish measured efficiency.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "campaign"))
from roofline import lookup_gpu  # noqa: E402

LINE = re.compile(r"^#(\d+)\s+(\w+)\s+b=(\d+)\s*(.*)$")
PROGRAM = re.compile(r"^===== program T=(\d+)")
WEIGHT = re.compile(r"(?:W|B|Wg|W_gate|W_q|table)<-(\S+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SITE = re.compile(r'sites="([^"]+)" count=(\d+) elapsed_ms="([0-9.]+)"')


def parse(text):
    """[(phase, rows, [(idx, op, name, params)])] in disasm order."""
    progs, cur, phase = [], None, "prefill"
    for line in text.splitlines():
        m = PROGRAM.match(line)
        if m:
            rows = int(m.group(1))
            if cur is not None and rows < cur[1]:
                phase = "decode"  # plowc orders prefill rungs, then decode rungs, each ascending
            cur = (phase, rows, [])
            progs.append(cur)
            continue
        m = LINE.match(line.strip())
        if cur is None or not m:
            continue
        head, _, tail = m.group(4).partition(" | ")
        p = {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+)(?=\s|$)", tail)}
        if m.group(2) == "GemmGluFp8":
            scale = re.search(r"a_scale<-(\S+)", head)
            p["fp8_activation"] = bool(scale and scale.group(1) != "—")
        if m.group(2) == "QuantFp8":
            gate = re.search(r"gate<-(\S+)", head)
            p["fused_glu"] = bool(gate and gate.group(1) != "—")
        if m.group(2) == "HeadNormRopeFp8":
            out = re.search(r"out<-(\S+)", head)
            p["output"] = out.group(1) if out else ""
        if m.group(2) in ("FlashPrefillFp8", "FlashDecodeFp8"):
            for slot in ("K", "V"):
                cache = re.search(rf"\b{slot}<-(\S+)", head)
                if cache:
                    p[f"{slot.lower()}_cache"] = cache.group(1)
        w = WEIGHT.search(head)
        name = re.sub(r"layers\.\d+\.", "", w.group(1)).replace("model.language_model.", "").replace("model.", "") if w and not w.group(1).startswith("act.") else ""
        cur[2].append((int(m.group(1)), m.group(2), name, p))
    for _, rows, insts in progs:
        cache_shapes = {}
        for _, op, _, p in insts:
            if op not in ("FlashPrefillFp8", "FlashDecodeFp8"):
                continue
            ntok = p.get("n_q", rows) if op == "FlashPrefillFp8" else p.get("n_batch", rows)
            shape = (ntok, p.get("n_kv_head"), p.get("hd"))
            for slot in ("k_cache", "v_cache"):
                if p.get(slot):
                    cache_shapes[p[slot]] = shape
        for _, op, _, p in insts:
            if op == "HeadNormRopeFp8" and p.get("output") in cache_shapes:
                p["ntok"], p["nhead"], p["hd"] = cache_shapes[p["output"]]
    return progs


def cost(op, p, rows, ctx, wb, kvb):
    """(bytes, flops) of one instruction, or None when the op is not priced."""
    g = p.get
    m = g("M", rows)
    if op == "Nop":  # dependency anchor (a block packet's entry)
        return 0, 0
    if op in ("Gemm", "Gemv", "GemmMed", "GemmSmall", "GemmWide"):
        n, k = g("N"), g("K")
        return wb * n * k + 2 * m * k + 2 * m * n, 2 * m * n * k
    if op in ("GemvFp8", "GemvGluFp8"):
        n, k = g("N"), g("K")
        matrices = 2 if op == "GemvGluFp8" else 1
        return matrices * (n * k + 4 * n) + 2 * m * (k + n), 2 * matrices * m * n * k
    if op in ("GemmFp8", "GemmMedFp8", "GemmSmallFp8"):
        n, k = g("N"), g("K")
        return n * k + m * k + 4 * (m + n) + 2 * m * n, 2 * m * n * k
    if op == "GemmGluFp8":
        n, k = g("N"), g("K")
        activation = m * k + 4 * m if g("fp8_activation", False) else 2 * m * k
        return 2 * (n * k + 4 * n) + activation + 2 * m * n, 4 * m * n * k
    if op == "QuantFp8":
        # Fused GLU reads gate/up and materializes BF16 x as well as FP8 xq.
        return (7 if g("fused_glu", False) else 3) * m * g("K") + 4 * m, 0
    if op == "GemvArgmax":
        n, k = g("N"), g("K")
        return 2 * n * k + 2 * m * (k + n), 2 * m * n * k
    if op == "GemvQkv":
        n, k = g("Nq") + g("Nk") + g("Nv"), g("K")
        return wb * n * k + 2 * m * k + 2 * m * n, 2 * m * n * k
    if op in ("GemvGlu", "GemmGlu"):
        n, k = g("N"), g("K")
        return 2 * wb * n * k + 2 * m * k + 2 * m * n, 4 * m * n * k
    if op in ("FlashDecode", "FlashDecodeFp8"):
        nb, hq, hk, hd, win = g("n_batch", rows), g("n_head"), g("n_kv_head"), g("hd"), g("window", 0)
        keys = min(ctx, win) if win else ctx
        scale_bytes = 8 if op == "FlashDecodeFp8" else 0
        kv_elem = 1 if op == "FlashDecodeFp8" else kvb
        return nb * keys * hk * (hd * 2 * kv_elem + scale_bytes) + 4 * nb * hq * hd, 4 * nb * hq * keys * hd
    if op in ("FlashPrefill", "FlashPrefillFp8"):
        nq, nkv, hq, hk, hd, win = g("n_q"), g("n_kv"), g("n_head"), g("n_kv_head"), g("hd"), g("window", 0)
        past = g("q_pos0", 0)
        ramp = min(nq, max(win - past, 0)) if win else nq
        pairs = ramp * past + ramp * (ramp + 1) / 2 + (nq - ramp) * win
        kv_rows = min(nkv, win + nq - 1) if win else nkv
        scale_bytes = 8 if op == "FlashPrefillFp8" else 0
        kv_elem = 1 if op == "FlashPrefillFp8" else kvb
        return 4 * nq * hq * hd + kv_rows * hk * (hd * 2 * kv_elem + scale_bytes), 4 * hq * pairs * hd
    if op == "FlashMerge":
        n, hh, ns, hd = g("n_batch", rows), g("n_head"), g("nsplit", 1), g("hd")
        return 4 * n * hh * hd * ns + 2 * n * hh * hd, 0
    if op in ("RmsNorm", "NormResidual", "NormResidualNorm", "AddNorm"):
        # AddNorm: reads a and b, writes the residual sum and its norm.
        tensors = {"RmsNorm": 2, "NormResidual": 3, "NormResidualNorm": 4, "AddNorm": 4}[op]
        return 2 * tensors * g("rows", rows) * g("feat"), 0
    if op == "GluStrided":
        return 6 * g("rows", rows) * g("width"), 0
    if op in ("Glu", "Residual"):
        return 6 * g("n"), 0
    if op == "HeadNormRope":
        return 4 * g("ntok") * g("nhead") * g("hd"), 0
    if op == "HeadNormRopeFp8" and all(g(k) is not None for k in ("ntok", "nhead", "hd")):
        row_count = g("ntok") * g("nhead")
        # BF16 input, FP8 cache output, and one FP32 scale per cache row.
        return 3 * row_count * g("hd") + 4 * row_count, 0
    if op == "EmbedOverlayBf16":
        # Per row: a token-table or overlay (encoder) row read, BF16 row written.
        return 4 * g("rows", rows) * g("width"), 0
    if op == "EmbedPosBf16":
        # Token row + position row read, BF16 row written.
        return 6 * g("rows", rows) * g("width"), 0
    if op == "Embed":
        return 4 * g("ntok") * g("hidden"), 0
    if op == "SoftCap":
        return 8 * g("n"), 0
    if op in ("Argmax", "ArgmaxFin"):
        return 4 * g("n", 0), 0
    # Qwen3.5 hybrid (qwen35.rs): B = decode rows, T = prefill rows; elementwise ops are byte floors.
    if op == "QwenRmsNorm":
        return 4 * g("B", rows) * g("width"), 0
    if op == "QwenHeadNormRope":
        return 4 * g("rows", rows) * g("H") * g("D"), 0
    if op == "QwenQGateSplit":  # packed [q | gate] read, both halves written
        return 4 * g("B", rows) * 2 * g("H") * g("D"), 0
    if op == "QwenSigmoidGate":
        return 6 * g("B", rows) * g("width"), 0
    if op == "QwenGatedNorm":  # core and z read, gated written
        return 6 * g("B", rows) * g("HV") * g("V"), 0
    if op in ("QwenGdnConv", "QwenGdnConvPrefill"):
        n = g("B", rows) if op == "QwenGdnConv" else g("T", rows)
        return 4 * n * g("C") + 2 * (g("W") - 1) * g("C") * (1 if op == "QwenGdnConvPrefill" else n), 0
    if op == "QwenGdnQkvPrep":  # packed conv output read, q/k/v written
        return 4 * g("T", rows) * (2 * g("HK") * g("K") + g("HV") * g("V")), 0
    if op == "QwenGdnGatePrep":
        return 16 * g("T", rows) * g("HV"), 0
    if op == "QwenGdnStep":  # per row: FP32 state read + written, q/k/v read; 4 HV K V FLOPs
        hv, k, v = g("HV"), g("K"), g("V")
        n = g("B", rows)
        return n * (8 * hv * k * v + 2 * (2 * g("HK") * k + hv * v)), 4 * n * hv * k * v
    if op == "QwenGdnPrefill":  # chunked delta rule: state in/out, q/k/v/o rows, ~6 HV K V per token
        hv, k, v, t = g("HV"), g("K"), g("V"), g("T", rows)
        return 8 * hv * k * v + 2 * t * (2 * g("HK") * k + 2 * hv * v), 6 * t * hv * k * v
    return moe_cost(op, p, rows, wb)


GEMMA_MOE_EXPERT = {"MoeExpertGluGemma", "MoeExpertGluNormGemma", "MoeExpertGluGemmaFp8",
                    "MoeExpertDownGemma", "MoeExpertDownGemmaFp8",
                    "MoeGroupGluGemmaPf", "MoeGroupGluGemmaPfW8a8",
                    "MoeGroupDownGemmaPf", "MoeGroupDownGemmaPfW8a8"}


def moe_cost(op, p, rows, wb):
    """Gemma-4 MoE (NVIDIA). Expert weights are charged for `experts` touched experts (`price`
    injects it from the router-table or uniform-routing model); FP8/W8A8 experts carry one-byte
    weights plus an FP32 scale per output row. The partial sums are FP32 (`act.moe.part`)."""
    g = p.get
    if op in ("MoeRouterGemmaScoreFast", "MoeRouterGemmaPf"):
        t, h, n = (g("T", rows) if op == "MoeRouterGemmaPf" else rows), g("H"), g("n_exp")
        out = t * g("k") * 8 if op == "MoeRouterGemmaPf" else t * n * 4
        return 2 * n * h + 4 * h + 2 * t * h + out, 2 * t * n * h
    if op == "MoeRouterGemmaTopk":
        return rows * (4 * g("n_exp") + 8 * g("k")) + 2 * g("n_exp"), 0
    if op == "MoeAlignGemmaPf":
        return 20 * g("T", rows) * g("k") + 12 * g("n_exp"), 0
    if op in ("MoeCombineNormGemma", "MoeCombineNormGemmaPf"):
        t = g("T", rows) if op == "MoeCombineNormGemmaPf" else rows
        return t * g("H") * (4 * g("k") + 4) + 2 * g("H"), 0
    if op not in GEMMA_MOE_EXPERT:
        return None
    t, k, h = g("T", rows), g("k"), g("H")
    i = g("I_moe") if g("I_moe") is not None else g("I")
    if None in (t, k, h, i, g("experts")):
        raise ValueError(f"{op}: expert shape needs T, k, H, I and the touched-expert count")
    quantized = op.endswith(("Fp8", "W8a8"))
    elem = 1 if quantized else wb
    glu = "Glu" in op
    n_out, k_in = (2 * i, h) if glu else (h, i)
    weight = g("experts") * (n_out * k_in * elem + (4 * n_out if quantized else 0))
    act_in = t * k * k_in if "W8a8" in op else 2 * t * (h if glu and "Group" not in op else k * k_in)
    act_out = t * k * (2 * i if glu else 4 * h)
    return weight + act_in + act_out, 2 * t * k * n_out * k_in


def uniform_union(experts, topk, rows):
    """Expected experts touched by `rows` independent uniformly routed rows."""
    return experts * (1.0 - (1.0 - topk / experts) ** rows)


class RouterTable:
    """A captured `act.moe.table` ({u32 expert, f32 gate} x k per row, real traffic): the
    touched-expert count of `rows` rows is the mean over evenly spaced row samples."""

    def __init__(self, raw, topk):
        import struct
        if not raw or len(raw) % (8 * topk):
            raise ValueError("router table must hold whole rows of k {u32 id, f32 gate} entries")
        self.topk = topk
        self.rows = []
        for r in range(len(raw) // (8 * topk)):
            ids = [struct.unpack_from("<I", raw, (r * topk + c) * 8)[0] for c in range(topk)]
            if len(set(ids)) != topk:
                raise ValueError(f"router table row {r} repeats an expert")
            self.rows.append(frozenset(ids))

    def union(self, experts, topk, rows):
        if topk != self.topk or any(e >= experts for r in self.rows for e in r):
            raise ValueError("router table does not match the packet's expert count / top-k")
        n = len(self.rows)
        if rows >= n:
            return float(len(frozenset().union(*self.rows)))
        stride = n // rows
        samples = [frozenset().union(*(self.rows[(o + j * stride) % n] for j in range(rows)))
                   for o in range(min(stride, 16))]
        return sum(map(len, samples)) / len(samples)


def price(prog, ctx, wb, kvb, bw, tflops, fp8_tflops=None, prefill_past=None, router=None):
    """Per instruction: key, bytes, flops, floor ms, bound. `router` (a RouterTable) replaces
    uniform routing when charging touched experts."""
    out = {}
    routed = None  # (T, k, n_exp) of the program's latest router
    for idx, op, name, p in prog[2]:
        if prog[0] == "prefill" and op in ("FlashPrefill", "FlashPrefillFp8") and prefill_past is not None:
            p = dict(p, q_pos0=prefill_past, n_kv=prefill_past + p["n_q"])
        if op in ("MoeRouterGemmaPf", "MoeAlignGemmaPf", "MoeRouterGemmaTopk"):
            routed = (p.get("T", prog[1]), p["k"], p["n_exp"])
        if op in GEMMA_MOE_EXPERT:
            if routed is None or routed[2] != p["n_exp"]:
                raise ValueError(f"{op} at #{idx} has no preceding router of its expert count")
            t, k, n = routed
            touched = (router.union if router else uniform_union)(n, k, t)
            p = dict(p, T=t, k=k, experts=touched)
        c = cost(op, p, prog[1], ctx, wb, kvb)
        if c is None:
            out[idx] = (f"{op}", None, None, 0.0, "unpriced")
            continue
        fp8_compute = op in ("GemmFp8", "GemmMedFp8", "GemmSmallFp8") or (
            op == "GemmGluFp8" and p.get("fp8_activation", False))
        peak = fp8_tflops if fp8_tflops is not None and fp8_compute else tflops
        mem, mat = c[0] / (bw * 1e6), c[1] / (peak * 1e9)
        key = f"{op}:{name}" if name else op
        out[idx] = (key, c[0], c[1], max(mem, mat), "matrix" if mat > mem else "hbm")
    return out


def ceilings(a, hw):
    """(GB/s, BF16 TFLOP/s, FP8 TFLOP/s, provenance). A ceilings file from
    calibrate_ceilings.py is measured on the box; the registry is a datasheet hypothesis."""
    if getattr(a, "ceilings", None):
        with open(a.ceilings) as f:
            cal = json.load(f)
        missing = [k for k in ("bw_gbps", "bf16_tflops", "fp8_tflops") if not cal.get(k)]
        if missing:
            raise ValueError(f"{a.ceilings}: missing {', '.join(missing)}")
        bw, tf, f8 = cal["bw_gbps"], cal["bf16_tflops"], cal["fp8_tflops"]
        source = f"calibrated {cal.get('device', '?')} {cal.get('date', '')}".strip()
    else:
        bw, tf, f8 = hw.bandwidth_for_bound_gbps, hw.bf16_tflops_dense, hw.fp8_tflops_dense
        source = f"registry {hw.name}, NOT calibrated"
    if a.bw or a.tflops or a.fp8_tflops:
        source += " + explicit overrides"
    return a.bw or bw, a.tflops or tf, a.fp8_tflops or f8, source


def load_router(path, progs):
    if not path:
        return None
    ks = {p["k"] for _, _, insts in progs for _, op, _, p in insts
          if op in ("MoeRouterGemmaPf", "MoeRouterGemmaTopk")}
    if len(ks) != 1:
        raise ValueError("--router-table needs a packet with one MoE top-k")
    with open(path, "rb") as f:
        return RouterTable(f.read(), int(ks.pop()))


def require_priced(prog, priced):
    unpriced = sorted({entry[0] for entry in priced.values() if entry[4] == "unpriced"})
    if unpriced:
        raise ValueError(f"{prog[0]} T={prog[1]} has unpriced ops: {', '.join(unpriced)}")


def sweep_deltas(path):
    caps = {}
    with open(path) as source:
        for ln in source:
            ln = ln.strip()
            if ln.startswith('{"cap"'):
                d = json.loads(ln)
                caps[d["cap"]] = d["ms"]
    full = caps.pop(-1, None)
    ks = sorted(caps)
    if not ks:
        raise ValueError(f"no instruction caps in {path}; library-routed decode needs segment timing")
    if ks != list(range(ks[-1] + 1)) or full is None:
        raise ValueError(f"incomplete instruction caps in {path}: need 0..{ks[-1]} and cap -1")
    return caps[ks[0]], {k - 1: caps[k] - caps[k - 1] for k in ks[1:]}, full


def table(title, rows, measured_total, extra_lines, top):
    """rows: key -> [n, bytes, flops, floor, measured|None, bound]"""
    out = [f"### {title}", ""]
    floor = sum(r[3] for r in rows.values())
    have = any(r[4] is not None for r in rows.values())
    out.append(f"floor {floor:.3f} ms" + (f", measured {measured_total:.3f} ms = {100 * floor / measured_total:.0f}% of roofline"
                                          if measured_total else ""))
    out += extra_lines + [""]
    out.append("| op | n | MB | GFLOP | bound | floor ms |" + (" measured ms | % roof |" if have else ""))
    out.append("|---|---:|---:|---:|---|---:|" + ("---:|---:|" if have else ""))
    order = sorted(rows.items(), key=lambda kv: -(kv[1][4] if have and kv[1][4] is not None else kv[1][3]))
    for key, (n, by, fl, fm, ms, bound) in order[:top]:
        line = (f"| `{key}` | {n} | {'-' if by is None else f'{by / 1e6:.1f}'} | "
                f"{'-' if fl is None else f'{fl / 1e9:.2f}'} | {bound} | {fm:.3f} |")
        if have:
            line += f" {'-' if ms is None else f'{ms:.3f}'} | {f'{100 * fm / ms:.0f}%' if ms and ms > 0 and fm else '-'} |"
        out.append(line)
    if len(order) > top:
        rest = order[top:]
        out.append(f"| (other {len(rest)} ops) | {sum(r[0] for _, r in rest)} | | | | {sum(r[3] for _, r in rest):.3f} |"
                   + (f" {sum(r[4] or 0 for _, r in rest):.3f} | |" if have else ""))
    out.append("")
    return "\n".join(out)


def segment_launches(program, launches):
    """Match SM90 prefill and decode segment routes in packet order."""
    segments = collections.defaultdict(set)
    for entry in program["stream"]:
        segments[entry["seg"]].add(entry["inst"])
    if sorted(segments) != list(range(len(segments))):
        raise ValueError("packet segments must be contiguous")
    for index, launch in enumerate(launches):
        if launch["end"] <= launch["start"]:
            raise ValueError("nonpositive kernel duration")
        if index and launch["start"] < launches[index - 1]["end"]:
            raise ValueError("overlapping kernels cannot be matched by serial segment order")
    insts = {inst["idx"]: inst for inst in program["insts"]}
    matched, cursor = [], 0
    for segment in range(len(segments)):
        ops = [insts[i] for i in sorted(segments[segment])]
        names = [op["op_name"] for op in ops]
        if cursor >= len(launches):
            raise ValueError("trace ends before all packet segments")
        launch = launches[cursor]
        count = 1
        key = "+".join(names)
        fp8 = any(name in ("GemmFp8", "GemmMedFp8", "GemmSmallFp8") for name in names)
        if fp8:
            if len(ops) != 1 or not launch["name"].startswith("nvjet_"):
                raise ValueError(f"segment {segment}: expected an isolated FP8 cuBLASLt kernel")
            weight = next(t["tensor"] for t in ops[0]["tensors"] if t["name"] == "B")
            key = re.sub(r"layers\.\d+\.", "", weight)
        elif names == ["Gemv"] and launch["name"].startswith("nvjet_tst_"):
            weight = next(t["tensor"] for t in ops[0]["tensors"] if t["slot"] == "t2")
            if not weight.endswith("embed_tokens.weight"):
                raise ValueError("BF16 Lt mapping is restricted to the output head")
            key = "BF16 Lt head"
        elif names == ["Gemm"] and launch["name"].startswith("interp_sm90a_pfpackedgemm("):
            pass
        elif names == ["QuantFp8"] and launch["name"] == "plow_glu_quant_cached_pfpackedseg":
            tensors = {t["name"]: t for t in ops[0]["tensors"]}
            if not tensors.get("gate", {}).get("present") or not tensors.get("up", {}).get("present"):
                raise ValueError(f"segment {segment}: cached GLU route requires gate and up")
            key += " [cached GLU]"
        elif names == ["NormResidualNorm"] and launch["name"] == "plow_sm90a_light":
            pass
        elif names == ["NormResidualNorm", "QuantFp8"] and launch["name"] == "plow_sm90a_light_norm_quant":
            key += " [direct]"
        elif names == ["SoftCap", "Argmax", "ArgmaxFin"] and launch["name"] == "plow_sm90a_light_capmax":
            count = 2
            if cursor + 1 >= len(launches) or launches[cursor + 1]["name"] != "plow_sm90a_light_tail":
                raise ValueError("light capmax segment requires its tail kernel")
        elif any(name in ("FlashDecode", "FlashDecodeFp8") for name in names) and launch["name"] in ("plow_sm90a_light_attn_s", "plow_sm90a_light_attn"):
            count = 6
            expected = [launch["name"]] * 4 + ["plow_sm90a_light"] * 2
            actual = [k["name"] for k in launches[cursor:cursor + count]]
            mixed = [launch["name"]] + ["plow_sm90a_light_attn"] * 3 + ["plow_sm90a_light"] * 2
            if actual not in (expected, mixed):
                raise ValueError(f"segment {segment}: incomplete light attention route")
            key += " [light attention]"
        else:
            if launch["name"].startswith("interp_sm90a_pfpackedseg("):
                pass
            elif not launch["name"].startswith(("interp_sm90a_gw(", "interp_sm90a(")):
                raise ValueError(f"segment {segment}: expected the packet interpreter")
            for op in ops:
                if op["op_name"] == "FlashDecode":
                    shape = {v["name"]: v["value"] for v in op["ints"]}
                    key += " [" + ", ".join(f"{k}={shape[k]}" for k in
                                             ("hd", "n_head", "n_kv_head", "window")) + "]"
                elif op["op_name"] in ("FlashPrefill", "FlashPrefillFp8"):
                    shape = {v["name"]: v["value"] for v in op.get("ints", [])}
                    fields = ("hd", "n_head", "n_kv_head", "window")
                    if all(k in shape for k in fields):
                        key += " [" + ", ".join(f"{k}={shape[k]}" for k in fields) + "]"
        kernels = launches[cursor:cursor + count]
        matched.append({"segment": segment, "instructions": sorted(segments[segment]),
                        "group": key, "kernel": "+".join(k["name"] for k in kernels),
                        "kernels": [k["name"] for k in kernels],
                        "ms": sum(k["end"] - k["start"] for k in kernels) / 1e6})
        cursor += count
    if cursor != len(launches):
        raise ValueError("trace has unmatched trailing kernels")
    return matched


def nsys_segments(report, program_index, path, graphs=None, correlations=None):
    import sqlite3
    from pathlib import Path
    if correlations is None and (graphs is None or graphs < 1):
        raise ValueError("--graphs must be positive without --nsys-correlation")
    if correlations is not None and (not correlations or graphs is not None):
        raise ValueError("select either --graphs or --nsys-correlation")
    program = report["programs"][program_index]
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        if correlations is None:
            correlations = [r[0] for r in db.execute(
                "select correlationId from CUPTI_ACTIVITY_KIND_KERNEL group by correlationId "
                "order by max(end) desc limit ?", (graphs,))]
            if len(correlations) != graphs or None in correlations:
                raise ValueError("trace lacks the requested complete graph correlations")
            correlations.reverse()
        elif len(set(correlations)) != len(correlations):
            raise ValueError("duplicate --nsys-correlation")
        runs = []
        for correlation in correlations:
            launches = [dict(start=start, end=end, name=name) for start, end, name in db.execute(
                "select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                "join StringIds s on s.id=k.demangledName where k.correlationId=? order by k.start",
                (correlation,))]
            if not launches:
                raise ValueError(f"trace lacks correlation {correlation}")
            runs.append({"correlation_id": correlation,
                         "segments": segment_launches(program, launches)})
    groups = collections.defaultdict(lambda: {"launches": 0, "mean_ms_per_graph": 0.0})
    for run in runs:
        for segment in run["segments"]:
            group = groups[segment["group"]]
            group["launches"] += len(segment["kernels"])
            group["mean_ms_per_graph"] += segment["ms"] / len(runs)
    return {"scope": "selected graph correlations, validated packet routes; instrumented GPU time; "
                     "caller must verify these are representative steps, not warmup",
            "program_index": program_index, "rows": program["t"], "graphs": runs,
            "groups": dict(sorted(groups.items(), key=lambda item: -item[1]["mean_ms_per_graph"]))}


def select_segtime_chunk(lines, chunk):
    if chunk is None:
        return lines
    groups = []
    for line in lines:
        if "seg-class wall time (chunk)" in line:
            groups.append([])
        if groups:
            groups[-1].append(line)
    if chunk < 0 or chunk >= len(groups):
        raise ValueError(f"segtime chunk {chunk} outside {len(groups)} recorded chunks")
    return groups[chunk]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("disasm")
    ap.add_argument("--nsys", help="Nsight SQLite trace; disasm must be JSON with --stream")
    ap.add_argument("--nsys-correlation", type=int, action="append", help="explicit graph correlation ID; repeat to average")
    ap.add_argument("--program-index", type=int, help="exact index in JSON programs (prefill/decode may share T)")
    ap.add_argument("--graphs", type=int, help="number of final measured decode graph launches")
    ap.add_argument("--ctx", type=int, help="decode live context per row")
    ap.add_argument("--prefill-past", type=int,
                    help="prefix tokens before one contiguous prefill chunk; default uses emitted operands")
    ap.add_argument("--sweep", action="append", default=[], metavar="B=FILE")
    ap.add_argument("--segtime")
    ap.add_argument("--segtime-rows", type=int)
    ap.add_argument("--segtime-chunk", type=int, help="zero-based chunk index; default averages all chunks")
    ap.add_argument("--gpu", default="h100")
    ap.add_argument("--bw", type=float, help="GB/s (default: the --gpu registry)")
    ap.add_argument("--tflops", type=float, help="dense BF16 matrix TFLOP/s")
    ap.add_argument("--fp8-tflops", type=float, help="dense FP8 matrix TFLOP/s for W8A8 GEMMs")
    ap.add_argument("--ceilings", help="calibrated ceilings JSON (scripts/bench/calibrate_ceilings.py); "
                    "--bw/--tflops/--fp8-tflops override its fields")
    ap.add_argument("--router-table", help="captured act.moe.table (scripts/bench/router_table.py); "
                    "default: uniform routing")
    ap.add_argument("--wbytes", type=float, default=2.0)
    ap.add_argument("--kvbytes", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    if a.prefill_past is not None and a.prefill_past < 0:
        ap.error("--prefill-past must be nonnegative")
    if a.segtime_chunk is not None and (a.segtime_chunk < 0 or not a.segtime):
        ap.error("--segtime-chunk requires --segtime and a nonnegative index")
    if a.nsys:
        if a.program_index is None or a.program_index < 0 or (a.graphs is None) == (a.nsys_correlation is None):
            ap.error("--nsys requires --program-index and exactly one of --graphs or --nsys-correlation")
        print(json.dumps(nsys_segments(json.load(open(a.disasm)), a.program_index, a.nsys,
                                       a.graphs, a.nsys_correlation), indent=2))
        return
    if a.ctx is None:
        ap.error("--ctx is required for theoretical roofline analysis")
    hw = lookup_gpu(a.gpu)
    bw, tf, fp8_tf, source = ceilings(a, hw)
    progs = parse(open(a.disasm).read())
    router = load_router(a.router_table, progs)
    if not progs:
        sys.exit(f"{a.disasm}: no programs (expected `plowrt disasm` output)")
    print(f"# Op roofline: {os.path.basename(a.disasm)}, ctx {a.ctx}, {bw:.0f} GB/s, {tf:.0f} BF16 / "
          f"{fp8_tf:.0f} FP8 TFLOP/s ({source})\n")
    if a.prefill_past is not None:
        print(f"Prefill scenario: one contiguous chunk after {a.prefill_past} prefix tokens; "
              "not a packed mixture of requests.\n")
    else:
        print("Prefill uses emitted q_pos0/n_kv; --ctx affects decode only.\n")
    find = lambda phase, rows: next((p for p in progs if p[0] == phase and p[1] == rows), None)

    for spec in a.sweep:
        b, path = spec.split("=", 1)
        prog = find("decode", int(b))
        if prog is None:
            sys.exit(f"no decode program T={b} in {a.disasm}")
        pr = price(prog, a.ctx, a.wbytes, a.kvbytes, bw, tf, fp8_tf, a.prefill_past, router)
        require_priced(prog, pr)
        skel, delta, full = sweep_deltas(path)
        if delta.keys() != pr.keys():
            raise ValueError(f"sweep {path} does not cover every decode instruction in B={b}")
        rows = collections.OrderedDict()
        for idx, (key, by, fl, fm, bound) in pr.items():
            r = rows.setdefault(key, [0, 0, 0, 0.0, 0.0, bound])
            r[0] += 1
            r[1] = None if by is None else r[1] + by
            r[2] = None if fl is None else r[2] + fl
            r[3] += fm
            r[4] += delta.get(idx, 0.0)
        cls = collections.Counter()
        for key, r in rows.items():
            c = ("lm_head" if "embed_tokens" in key and key.startswith(("Gemv", "Gemm")) else
                 "projections" if key.startswith(("Gemv", "Gemm")) else
                 "attention" if key.startswith("Flash") else "other")
            cls[c] += r[4]
        extra = [f"skeleton (cap 0) {skel:.3f} ms; by class: " +
                 ", ".join(f"{k} {v:.3f}" for k, v in sorted(cls.items(), key=lambda kv: -kv[1]))]
        print(table(f"decode B={b} ctx {a.ctx} (step_bench sweep)", rows, full, extra, a.top))

    if a.segtime:
        if not a.segtime_rows:
            sys.exit("--segtime needs --segtime-rows (the chunk's prefill rows)")
        prog = find("prefill", a.segtime_rows)
        if prog is None:
            sys.exit(f"no prefill program T={a.segtime_rows} in {a.disasm}")
        pr = price(prog, a.ctx, a.wbytes, a.kvbytes, bw, tf, fp8_tf, a.prefill_past, router)
        require_priced(prog, pr)
        rows, chunks = collections.OrderedDict(), 0
        with open(a.segtime, errors="replace") as source:
            try:
                lines = select_segtime_chunk(list(source), a.segtime_chunk)
            except ValueError as error:
                sys.exit(str(error))
        if a.segtime_chunk is not None:
            print(f"Selected prefill chunk index {a.segtime_chunk}; rows and prefix must match that chunk.\n")
        for ln in lines:
            ln = ANSI.sub("", ln)
            if "seg-class wall time (chunk)" in ln:
                chunks += 1
            m = SITE.search(ln)
            if not m:
                continue
            pcs = [int(x.split(":")[0][2:]) for x in m.group(1).split("+")]
            key = "+".join(sorted({pr[i][0] for i in pcs if i in pr}))
            r = rows.setdefault(key, [0, 0, 0, 0.0, 0.0, "hbm"])
            r[4] += float(m.group(3))
            if chunks <= 1:
                r[0] += len(pcs)
                for i in pcs:
                    _, by, fl, fm, bound = pr.get(i, ("", None, None, 0.0, "unpriced"))
                    r[1] = None if by is None or r[1] is None else r[1] + by
                    r[2] = None if fl is None or r[2] is None else r[2] + fl
                    r[3] += fm
                    r[5] = bound if fm else r[5]
        for r in rows.values():
            r[4] /= max(chunks, 1)
        tot = sum(r[4] for r in rows.values())
        print(table(f"prefill T={a.segtime_rows} (PLOW_PF_SEG_TIME, mean of {chunks} chunk(s); SEG_TIME drains per "
                    "segment: attribution, not latency)", rows, tot, [], a.top))

    if not a.sweep and not a.segtime:
        print("| program | rows | floor ms | floor ms/row | top op (floor) |")
        print("|---|---:|---:|---:|---|")
        for prog in progs:
            pr = price(prog, a.ctx, a.wbytes, a.kvbytes, bw, tf, fp8_tf, a.prefill_past, router)
            require_priced(prog, pr)
            by = collections.Counter()
            for key, _, _, fm, _ in pr.values():
                by[key] += fm
            f = sum(by.values())
            top = by.most_common(1)[0]
            print(f"| {prog[0]} | {prog[1]} | {f:.3f} | {f / prog[1]:.4f} | `{top[0]}` {top[1]:.3f} |")


if __name__ == "__main__":
    main()
