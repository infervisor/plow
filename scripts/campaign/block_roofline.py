#!/usr/bin/env python3
"""Block-vs-roofline table for a single-block asset (the HARD RULE report, agent-tools.md).

    block_roofline.py <block asset dir | model.pkt | disasm.txt> --ceilings ceilings.json \
        --plow <block_run sweep.json> [--current <sweep.json of the production block>] \
        [--router-table table.bin] [--vllm layer.json] [--json out.json]

The floor is priced from the block's own packet (`plowrt disasm`), instruction by instruction, by
`scripts/bench/op_roof.py`: real element sizes (BF16, FP8 weights + scales, W8A8 activations, FP8
KV + scales) and every layer kind the packet contains (sliding/full attention, GDN/recurrent, MoE,
KV-shared layers); library-attention and cuBLASLt segments are priced at the op they serve. An
op without a price fails the run instead of being skipped. MoE expert reads use a captured router
table (`scripts/bench/router_table.py`) when given, else uniform routing. Ceilings come from
`scripts/bench/calibrate_ceilings.py` (`--ceilings`); without it the registry numbers are used
and the output says NOT calibrated.

Measured points come from `block_run` (crates/plowrt/examples/block_run.rs):
  bench          decode rows {batch, ctx, latency_us_median}: decode rung B at live context ctx
  prefill-bench  {phase: "prefill", requests, query_rows, prior, riders, rider_ctx, ms_median}:
                 one packed launch of `requests` chunks of `query_rows` rows after `prior` prefix
                 rows each, plus `riders` one-row decode requests at `rider_ctx`
The floor of a prefill point prices the program the launch ran (the smallest prefill bucket that
holds all rows), with attention charged per request at its own prefix.

`--current` is the production packet's same block measured in the same lease: the table's
"delta vs current" column. Block times are host-observed synchronous step times, so a few
microseconds of launch overhead sit above the device floor at small rungs.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import op_roof  # noqa: E402
from roofline import lookup_gpu  # noqa: E402

ATTENTION_PREFILL = ("FlashPrefill", "FlashPrefillFp8")
REPO = Path(__file__).resolve().parents[2]


def disassemble(path, plowrt):
    path = Path(path)
    if path.is_file() and path.suffix == ".txt":
        return path.read_text(), None
    blob = path / "model.pkt" if path.is_dir() else path
    out = subprocess.run([plowrt, "disasm", str(blob)], capture_output=True, text=True)
    if out.returncode:
        sys.exit(out.stderr or out.stdout)
    desc = blob.parent / "block.json"
    return out.stdout, json.loads(desc.read_text()) if desc.is_file() else None


def label(desc):
    if not desc:
        return "block"
    kinds = desc.get("kind", [])
    if len(kinds) <= 4:
        return f"L{desc['layer']} {'+'.join(kinds)}"
    counts = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return f"L{desc['layer']}+ " + ", ".join(f"{n}x {k}" for k, n in counts.items())


def program(progs, phase, rows):
    return next((p for p in progs if p[0] == phase and p[1] == rows), None)


def bucket(progs, rows):
    fits = sorted(p[1] for p in progs if p[0] == "prefill" and p[1] >= rows)
    return program(progs, "prefill", fits[0]) if fits else None


def floor_ms(priced):
    return sum(entry[3] for entry in priced.values())


def decode_floor(progs, b, ctx, ceil, wb, kvb, router):
    prog = program(progs, "decode", b)
    if prog is None:
        raise ValueError(f"block packet has no decode rung B={b}")
    priced = op_roof.price(prog, ctx, wb, kvb, *ceil, router=router)
    op_roof.require_priced(prog, priced)
    return floor_ms(priced), priced


def packed_floor(progs, requests, rows, prior, riders, rider_ctx, ceil, wb, kvb, router):
    """Floor of one packed prefill launch: `requests` chunks of `rows` rows at prefix `prior`
    plus `riders` one-row requests at context `rider_ctx`."""
    total = requests * rows + riders
    prog = bucket(progs, total)
    if prog is None:
        raise ValueError(f"no prefill bucket holds {total} rows")
    bw, tflops, fp8_tflops = ceil
    priced = op_roof.price(prog, rider_ctx, wb, kvb, bw, tflops, fp8_tflops, router=router)
    op_roof.require_priced(prog, priced)
    total_ms = 0.0
    for idx, op, _, p in prog[2]:
        if op not in ATTENTION_PREFILL:
            total_ms += priced[idx][3]
            continue
        for q, past in [(rows, prior)] * requests + [(1, rider_ctx)] * riders:
            one = dict(p, n_q=q, q_pos0=past, n_kv=past + q)
            nbytes, flops = op_roof.cost(op, one, q, past, wb, kvb)
            total_ms += max(nbytes / (bw * 1e6), flops / (tflops * 1e9))
    return total_ms, priced


def measured_points(sweep):
    rows = sweep.get("sweep", []) if isinstance(sweep, dict) else sweep
    out = {}
    for r in rows:
        if r.get("phase", "decode") == "decode" and "latency_us_median" in r:
            out[("decode", r["batch"], r["ctx"])] = r["latency_us_median"] / 1e3
        elif r.get("phase") == "prefill":
            key = ("prefill", r["requests"], r["query_rows"], r["prior"], r.get("riders", 0),
                   r.get("rider_ctx", r["prior"]))
            out[key] = r["ms_median"]
    return out


def load_sweep(path):
    if not path:
        return {}
    with open(path) as f:
        return measured_points(json.load(f))


def table(rows, have_vllm=False):
    head = "| block (layer kind, rung, ctx) | block time | roofline floor | % of floor | delta vs current |"
    sep = "|---|---:|---:|---:|---:|"
    if have_vllm:
        head += " vLLM layer |"
        sep += "---:|"
    out = [head, sep]
    for r in rows:
        t, f, cur = r["time_ms"], r["floor_ms"], r.get("current_ms")
        pct = f"{100 * f / t:.0f}%" if t else "-"
        delta = f"{t - cur:+.3f} ms ({100 * (t - cur) / cur:+.1f}%)" if t and cur else "-"
        line = f"| {r['block']}, {r['point']} | {f'{t:.3f} ms' if t else '-'} | {f:.3f} ms | {pct} | {delta} |"
        if have_vllm:
            v = r.get("vllm_ms")
            line += f" {f'{v:.3f} ms' if v else '-'} |"
        out.append(line)
    return "\n".join(out)


def build_rows(progs, desc, plow, current, vllm, ceil, wb, kvb, router):
    rows = []
    name = label(desc)
    for key in sorted(set(plow) | set(current), key=str):
        if key[0] == "decode":
            _, b, ctx = key
            floor, priced = decode_floor(progs, b, ctx, ceil, wb, kvb, router)
            point = f"decode B={b}, ctx {ctx}"
        else:
            _, n, q, prior, riders, rctx = key
            floor, priced = packed_floor(progs, n, q, prior, riders, rctx, ceil, wb, kvb, router)
            point = f"prefill {n}x{q} rows, prior {prior}" + (f", {riders} riders @ {rctx}" if riders else "")
        top = max(priced.values(), key=lambda e: e[3])
        rows.append({"block": name, "point": point, "key": list(key), "floor_ms": floor,
                     "time_ms": plow.get(key), "current_ms": current.get(key), "vllm_ms": vllm.get(key),
                     "top_op": top[0], "top_op_floor_ms": top[3]})
    return rows


def parse_point(spec):
    parts = spec.split(":")
    if parts[0] == "decode" and len(parts) == 3:
        return ("decode", int(parts[1]), int(parts[2]))
    if parts[0] == "prefill" and len(parts) in (4, 6):
        n, q, prior = map(int, parts[1:4])
        riders, rctx = (int(parts[4]), int(parts[5])) if len(parts) == 6 else (0, prior)
        return ("prefill", n, q, prior, riders, rctx)
    raise ValueError(f"bad point {spec!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("block", help="block asset dir (model.pkt + block.json), model.pkt, or a disasm .txt")
    ap.add_argument("--plowrt", default=os.environ.get("PLOWRT_BIN", str(REPO / "target/release/plowrt")))
    ap.add_argument("--plow", help="block_run sweep.json of the candidate")
    ap.add_argument("--current", help="block_run sweep.json of the production block, same lease")
    ap.add_argument("--vllm", help="block_layer_bench.py layer.json (decode rows joined on B, ctx)")
    ap.add_argument("--ceilings", help="calibrate_ceilings.py output")
    ap.add_argument("--gpu", default="h100", help="registry fallback when no --ceilings")
    ap.add_argument("--bw", type=float)
    ap.add_argument("--tflops", type=float)
    ap.add_argument("--fp8-tflops", type=float)
    ap.add_argument("--router-table", help="router_table.py output; default uniform routing")
    ap.add_argument("--wbytes", type=float, default=2.0, help="generic (non-FP8-op) weight bytes")
    ap.add_argument("--kvbytes", type=float, default=2.0, help="BF16-attention KV bytes")
    ap.add_argument("--point", action="append", default=[],
                    metavar="decode:B:CTX | prefill:N:ROWS:PRIOR[:RIDERS:RIDER_CTX]",
                    help="floor-only point (no measurement); repeatable")
    ap.add_argument("--json", help="write the rows (with each point's top op) here")
    a = ap.parse_args()

    text, desc = disassemble(a.block, a.plowrt)
    progs = op_roof.parse(text)
    if not progs:
        sys.exit(f"{a.block}: no programs in the disassembly")
    bw, tf, f8, source = op_roof.ceilings(a, lookup_gpu(a.gpu))
    router = op_roof.load_router(a.router_table, progs)
    plow, current = load_sweep(a.plow), load_sweep(a.current)
    for spec in a.point:
        try:
            plow.setdefault(parse_point(spec), None)
        except ValueError as error:
            ap.error(str(error))
    if not plow and not current:
        ap.error("give --plow/--current measurements or --point")
    vllm = load_sweep(a.vllm)
    rows = build_rows(progs, desc, plow, current, vllm, (bw, tf, f8), a.wbytes, a.kvbytes, router)
    routing = f"router table {a.router_table}" if router else "uniform routing"
    print(f"Ceilings: {bw:.0f} GB/s, {tf:.0f} BF16 / {f8:.0f} FP8 TFLOP/s ({source}); MoE: {routing}\n")
    print(table(rows, bool(vllm)))
    if a.json:
        Path(a.json).write_text(json.dumps({"ceilings": {"bw_gbps": bw, "bf16_tflops": tf,
                                                         "fp8_tflops": f8, "source": source},
                                            "router": a.router_table, "block": desc, "rows": rows},
                                           indent=2) + "\n")


if __name__ == "__main__":
    main()
