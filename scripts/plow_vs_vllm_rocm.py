#!/usr/bin/env python3
"""Join the plow gfx942 sweep CSVs against the vLLM MI300X baselines.

Both sides were produced by the SAME client (`vllm bench serve`) and the SAME
regex parse -- scripts/bench_plow_rocm.sh is a deliberate mirror of
scripts/bench_vllm_rocm.sh -- so the columns mean the same thing and a ratio is
meaningful without a translation step.

Ratios are stated as plow/vLLM for latency (lower is better, >1 = plow slower)
and vLLM/plow for throughput, so that in EVERY column ">1 means plow is behind".

  usage: plow_vs_vllm_rocm.py [--out FILE]
"""

import argparse
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLOW = ROOT / "perf-data" / "plow-gfx942"
VLLM = ROOT / "perf-data" / "vllm-rocm"

# plow asset dir stem -> (vLLM slug, tp, label). The vLLM `_bf16_` in a filename
# is NOT a measured precision for every row -- see vllm-rocm/PRECISION-LABELS.md
# -- so the label carries what actually ran on each side.
PAIRS = [
    ("gemma4-12b-bf16", "google_gemma-4-12B-it", 1, "Gemma-4 12B", "bf16", "bf16"),
    ("gemma-4-26b-a4b-it-bf16", "google_gemma-4-26B-A4B-it", 1, "Gemma-4 26B-A4B", "bf16", "bf16"),
    ("gemma-4-31b-it-bf16", "google_gemma-4-31B-it", 1, "Gemma-4 31B", "bf16", "bf16"),
    ("glm52-fp8-tp4", "zai-org_GLM-5.2-FP8", 4, "GLM-5.2", "block-fp8", "block-fp8"),
]

PHASES = [("general", "concurrency"), ("ctxsweep_c1", "input_len")]


def load(path):
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def ratio(a, b, invert=False):
    """plow/vLLM for latency; vLLM/plow for throughput. >1 always = plow behind."""
    try:
        a, b = float(a), float(b)
        if a <= 0 or b <= 0:
            return "-"
        return f"{(b / a if invert else a / b):.2f}x"
    except (TypeError, ValueError, ZeroDivisionError):
        return "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="-")
    args = ap.parse_args()
    out = sys.stdout if args.out == "-" else open(args.out, "w")

    print("# plow vs vLLM on MI300X (gfx942)\n", file=out)
    print(
        "Same client (`vllm bench serve`), same parse, same points. Ratio columns are "
        "oriented so **>1 always means plow is behind**.\n",
        file=out,
    )

    for stem, vslug, tp, label, plow_prec, vllm_prec in PAIRS:
        prec = plow_prec if plow_prec == vllm_prec else f"plow {plow_prec} vs vLLM {vllm_prec}"
        print(f"\n## {label} — tp{tp}, {prec}\n", file=out)
        for phase, key in PHASES:
            p = load(PLOW / f"{stem}_{'bf16' if plow_prec=='bf16' else 'fp8'}_tp{tp}_{phase}.csv")
            v = load(VLLM / f"{vslug}_bf16_tp{tp}_{phase}.csv")
            if not p:
                print(f"*{phase}: no plow measurement*\n", file=out)
                continue
            vmap = {r[key]: r for r in (v or [])}
            print(f"### {phase}\n", file=out)
            if phase == "general":
                print(
                    "> Only the **concurrency-1 row compares kernels**. plow's AMD serve runs "
                    "`batch=1` (the blob's compiled `PLOW_DECODE_BATCH`), so every row below it "
                    "measures REQUESTS QUEUEING behind one another while vLLM batches them "
                    "continuously. That is why plow's tok/s stays flat as concurrency rises and "
                    "its TTFT grows linearly — those ratios are the absence of continuous "
                    "batching, not a kernel deficit.\n",
                    file=out,
                )
            print(
                f"| {key} | plow TTFT ms | vLLM TTFT ms | TTFT | plow TPOT ms | "
                "vLLM TPOT ms | TPOT | plow tok/s | vLLM tok/s | tok/s |",
                file=out,
            )
            print("|--:" * 10 + "|", file=out)
            for r in p:
                w = vmap.get(r[key], {})
                print(
                    f"| {r[key]} | {fmt(r['ttft_ms'])} | {fmt(w.get('ttft_ms'))} | "
                    f"{ratio(r['ttft_ms'], w.get('ttft_ms'))} | "
                    f"{fmt(r['tpot_ms'], 3)} | {fmt(w.get('tpot_ms'), 3)} | "
                    f"{ratio(r['tpot_ms'], w.get('tpot_ms'))} | "
                    f"{fmt(r['out_tok_s'], 1)} | {fmt(w.get('out_tok_s'), 1)} | "
                    f"{ratio(r['out_tok_s'], w.get('out_tok_s'), invert=True)} |",
                    file=out,
                )
            print("", file=out)

    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
