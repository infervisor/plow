#!/usr/bin/env python3
"""Split TP XReduceTwoShot gate1/RS/gate2/AG phase traces."""

import argparse
import json
import statistics
import struct
from collections import defaultdict
from pathlib import Path


REC = struct.Struct("<IIIHHQQQ")
TPUS = 100.0


def load_rank(path, xr_op):
    data = Path(path).read_bytes()
    if len(data) % REC.size:
        raise ValueError(f"{path}: size is not a multiple of {REC.size}")
    out = defaultdict(list)
    for rec in REC.iter_unpack(data):
        rs_delta, gate2_delta, inst, op, sl, entry, ready, done = rec
        if op == xr_op and done:
            if sl & 0xc000 != 0xc000:
                raise ValueError(f"{path}: XReduceTwoShot record lacks v2 phase-trace marker")
            if rs_delta == 0xffffffff or gate2_delta == 0xffffffff:
                raise ValueError(f"{path}: XReduceTwoShot phase delta saturated")
            out[inst].append(
                (sl & 0x3fff, entry, ready, entry + rs_delta, entry + gate2_delta, done)
            )
    return out


def median(values):
    return statistics.median(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round(q * (len(ordered) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("disasm", help="pure JSON from `plowrt disasm ... --format json`")
    ap.add_argument("traces", nargs="+", help="one PLOW_TRACE_ALLRANKS dump per rank")
    ap.add_argument("--top", type=int, default=24)
    args = ap.parse_args()

    doc = json.loads(Path(args.disasm).read_text())
    if len(doc["programs"]) != 1:
        raise ValueError("disasm JSON must contain exactly one selected program")
    insts = doc["programs"][0]["insts"]
    by_inst = {i["idx"]: i for i in insts}
    xr_ops = {i["op"] for i in insts if i["op_name"] == "XReduceTwoShot"}
    if len(xr_ops) != 1:
        raise ValueError(f"expected one XReduceTwoShot opcode, found {sorted(xr_ops)}")
    xr_op = xr_ops.pop()
    ranks = [load_rank(path, xr_op) for path in args.traces]
    common = set.intersection(*(set(rank) for rank in ranks))
    expected = {i["idx"] for i in insts if i["op"] == xr_op}
    if common != expected:
        missing = sorted(expected - common)
        extra = sorted(common - expected)
        raise ValueError(f"trace coverage mismatch: missing={missing[:8]} extra={extra[:8]}")

    rows = []
    families = defaultdict(list)
    for inst in sorted(common):
        xr = by_inst[inst]
        raw = xr["raw"]["i"]
        nbytes = raw[0] * 2
        shape = "gather" if raw[6] and raw[7] else "full"
        if shape != "gather" and nbytes * 2 == doc.get("tp", {}).get("slot_bytes", -1):
            shape = "half"
        producer = by_inst.get(inst - 1, {})
        family = producer.get("op_name", "START")
        producer_dims = "x".join(str(v) for v in producer.get("raw", {}).get("i", [])[:3])
        phases = {name: [] for name in ("gate1", "rs", "gate2", "ag", "envelope")}
        for rank, records in enumerate(ranks):
            slice0 = [r for r in records[inst] if r[0] == 0]
            if len(slice0) != 1:
                raise ValueError(f"inst {inst} rank {rank}: expected one slice-0 record")
            for _, entry, ready, rs_done, gate2_ready, done in records[inst]:
                if not entry <= ready <= rs_done <= gate2_ready <= done:
                    raise ValueError(f"inst {inst} rank {rank}: non-monotonic phase timestamps")
                phases["gate1"].append((ready - entry) / TPUS)
                phases["rs"].append((rs_done - ready) / TPUS)
                phases["gate2"].append((gate2_ready - rs_done) / TPUS)
                phases["ag"].append((done - gate2_ready) / TPUS)
                phases["envelope"].append((done - entry) / TPUS)
        row = {
            "inst": inst,
            "family": family,
            "producer_dims": producer_dims,
            "shape": shape,
            "mib": nbytes / (1 << 20),
            "phases": phases,
        }
        rows.append(row)
        families[(family, shape)].append(row)

    print(f"ranks={len(ranks)} collectives={len(rows)} tick/us={TPUS:g}")
    print("all phase durations are same-workgroup deltas; values below are microseconds")
    print(f"{'inst':>5} {'producer':<20} {'producer i0:i2':<22} {'shape':<7} {'MiB':>7} "
          f"{'gate1p90':>9} {'RSp90':>9} {'gate2p90':>9} {'AGp90':>9} {'env-max':>9}")
    for r in sorted(rows, key=lambda x: -max(x["phases"]["envelope"]))[: args.top]:
        p = r["phases"]
        print(f"{r['inst']:5d} {r['family']:<20.20} {r['producer_dims']:<22.22} "
              f"{r['shape']:<7} {r['mib']:7.1f} "
              f"{percentile(p['gate1'], 0.90):9.2f} {percentile(p['rs'], 0.90):9.2f} "
              f"{percentile(p['gate2'], 0.90):9.2f} {percentile(p['ag'], 0.90):9.2f} "
              f"{max(p['envelope']):9.2f}")

    print("\nproducer-family summary")
    print(f"{'producer':<20} {'shape':<7} {'n':>4} {'gate1-maxΣ':>11} {'RS-maxΣ':>10} "
          f"{'gate2-maxΣ':>11} {'AG-maxΣ':>10} {'env-maxΣ':>10}")
    for (family, shape), rs in sorted(
        families.items(), key=lambda x: -sum(max(r["phases"]["envelope"]) for r in x[1])
    ):
        sums = {
            phase: sum(max(r["phases"][phase]) for r in rs)
            for phase in ("gate1", "rs", "gate2", "ag", "envelope")
        }
        print(f"{family:<20.20} {shape:<7} {len(rs):4d} {sums['gate1']:11.2f} "
              f"{sums['rs']:10.2f} {sums['gate2']:11.2f} {sums['ag']:10.2f} "
              f"{sums['envelope']:10.2f}")

    print("\npooled workgroup phase percentiles")
    print(f"{'producer':<20} {'shape':<7} {'phase':<9} {'p50':>9} {'p90':>9} {'max':>9}")
    for (family, shape), rs in sorted(families.items()):
        for phase in ("gate1", "rs", "gate2", "ag"):
            values = [value for r in rs for value in r["phases"][phase]]
            print(f"{family:<20.20} {shape:<7} {phase:<9} {median(values):9.2f} "
                  f"{percentile(values, 0.90):9.2f} {max(values):9.2f}")


if __name__ == "__main__":
    main()
