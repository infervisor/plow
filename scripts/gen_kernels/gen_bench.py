"""Shared measurement code of the generated-kernel benches (bench_cfm.py, bench_decode.py,
gen_search.py): CUDA-graph timing over rotating cold buffers, error metrics, calibrated roofline
floors and the block-vs-roofline table. torch is imported only by the functions that touch a GPU,
so the roofline and table code runs on a CPU-only box (test_gen_search.py).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "campaign"))
import speech_roofline  # noqa: E402

# Rotating inputs must not fit in L2 (H100: 50 MB); bench_decode's bound.
COLD_BYTES = 160e6
# Calibrated H100 SXM ceilings (docs/bringup/isa/sm90a.md, calibrate_ceilings.py 2026-10-29);
# another box passes its own ceilings.json. TF32 / FP32 are speech_roofline.py's measured peaks.
CEILINGS = {"bw_gbps": 3094.0, "bf16_tflops": 771.3, "fp8_tflops": 1407.0}
# int16-RMS PCM bound: per-case RMS error of full-scale float PCM <= one int16 quantisation step's
# RMS.
PCM_RMS_BOUND = 1.0 / 32768.0 / math.sqrt(12.0)
# kernel-generators.md policy: a generated kernel ships as a role object only above this in-model
# gain; below it the schedule is a reference for a hand port.
ROLE_GAIN = 0.05


def graph_time(fns, reps=7):
    """Median us per call of `fns` replayed as one CUDA graph."""
    import torch
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for f in fns:
                f()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / len(fns))
    ts.sort()
    return ts[len(ts) // 2]


def rel_err(out, ref):
    """(max|d| / rms(ref), rel. L2) of `out` against an fp64 reference; an all-zero reference
    gives 0 for an exact output and inf otherwise."""
    d = out.double() - ref.double()
    ratio = lambda num, den: num / den if den > 0 else (0.0 if num == 0 else math.inf)
    return (ratio(d.abs().max().item(), ref.double().pow(2).mean().sqrt().item()),
            ratio(d.norm().item(), ref.double().norm().item()))


def pcm_rms(out, ref):
    """Worst per-case (dim 0) RMS error of full-scale float PCM; compare with PCM_RMS_BOUND."""
    d = (out.double() - ref.double()).reshape(out.shape[0], -1)
    return d.pow(2).mean(1).sqrt().max().item()


COLD_SETS_MAX = 64


def cold_sets(rotating_bytes):
    """Number of rotating input sets whose total exceeds L2, capped at COLD_SETS_MAX graph nodes;
    under the cap's reach (sets x bytes < COLD_BYTES) the timing is cache-warm and the caller
    says so."""
    return max(2, min(COLD_SETS_MAX, math.ceil(COLD_BYTES / max(1, rotating_bytes))))


def load_ceilings(path=None):
    c = dict(CEILINGS, source="calibrated H100 SXM5 (docs/bringup/isa/sm90a.md)")
    if path:
        c.update(json.load(open(path)), source=path)
    return {"bw": c["bw_gbps"] * 1e9,
            "peak": {"bf16": c["bf16_tflops"] * 1e12, "fp8": c["fp8_tflops"] * 1e12,
                     "tf32": speech_roofline.PEAK["tf32"], "fp32": speech_roofline.PEAK["fp32"]},
            "source": c["source"]}


def floor_us(cost, ceil):
    """Roofline floor of one call: max(bytes / HBM read, passes * flops / the arithmetic's peak)."""
    if cost["peak"] not in ceil["peak"]:
        raise ValueError(f"cost peak {cost['peak']!r}: one of {sorted(ceil['peak'])}")
    flops = cost["flops"] * cost.get("passes", 1)
    return max(cost["bytes"] / ceil["bw"], flops / ceil["peak"][cost["peak"]]) * 1e6


def verdict(cur_us, gen_us, step=None):
    """Policy verdict for the best valid generated candidate against the current plow kernel."""
    if gen_us is None:
        return "no valid candidate"
    if cur_us is None:
        return "no current time"
    if gen_us >= cur_us:
        return "leave: plow kernel ahead"
    if not step:
        return "reference: hand-port (in-model gain unmeasured)"
    gain = (cur_us - gen_us) * step.get("calls", 1) / step["us"]
    kind = "role-object candidate" if gain > ROLE_GAIN else "reference: hand-port"
    return f"{kind} ({gain * 100:.1f}% of step)"


def _us(v):
    return "—" if v is None else f"{v:.1f}"


def _pct(floor, v):
    return "—" if v is None else f"{100 * floor / v:.0f}%"


def table(rows):
    """Block-vs-roofline table (agent-tools.md HARD RULE) with the best generated candidate as a
    reference row. rows: dicts with block, floor_us, cur_us, cur_note, gen_us, gen_label, rel_l2,
    verdict."""
    out = ["| block (op, class) | current plow us | best generated us | generated config | "
           "roofline floor us | % of floor (current) | % of floor (generated) | "
           "gap (generated vs current) | rel-L2 | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cur, gen = r.get("cur_us"), r.get("gen_us")
        gap = "—" if cur is None or gen is None else f"{(gen - cur) / cur * 100:+.1f}%"
        l2 = "—" if r.get("rel_l2") is None else f"{r['rel_l2']:.1e}"
        out.append(f"| {r['block']} | {_us(cur)}{r.get('cur_note', '')} | {_us(gen)} | "
                   f"{r.get('gen_label') or '—'} | {r['floor_us']:.1f} | "
                   f"{_pct(r['floor_us'], cur)} | {_pct(r['floor_us'], gen)} | {gap} | {l2} | "
                   f"{r['verdict']} |")
    return "\n".join(out)
