#!/usr/bin/env python3
"""Rank TP XReduceTwoShot producer skew from PLOW_XR_TRACE_PHASES traces."""

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
        cu, delta, inst, op, sl, entry, ready, done = rec
        if op == xr_op and done:
            if not sl & 0x8000:
                raise ValueError(f"{path}: XReduceTwoShot record lacks phase-trace marker")
            out[inst].append((cu, sl & 0x7fff, entry, entry + delta, ready, done))
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
        rank_phase = []
        ready_spreads = []
        wg_durations = []
        envelopes = []
        for rank, records in enumerate(ranks):
            slice0 = [r for r in records[inst] if r[1] == 0]
            if len(slice0) != 1:
                raise ValueError(f"inst {inst} rank {rank}: expected one slice-0 record")
            _, _, entry, publish, ready, _ = slice0[0]
            done = max(r[5] for r in records[inst])
            rank_phase.append(
                ((publish - entry) / TPUS, (ready - publish) / TPUS,
                 (done - ready) / TPUS)
            )
            readies = [r[4] for r in records[inst]]
            dones = [r[5] for r in records[inst]]
            ready_spreads.append((max(readies) - min(readies)) / TPUS)
            wg_durations.extend((r[5] - r[4]) / TPUS for r in records[inst])
            envelopes.append((max(dones) - min(readies)) / TPUS)
        waits = [p[1] for p in rank_phase]
        latest = min(range(len(waits)), key=waits.__getitem__)
        skew = max(waits) - min(waits)
        row = {
            "inst": inst,
            "family": family,
            "producer_dims": producer_dims,
            "shape": shape,
            "mib": nbytes / (1 << 20),
            "rank0_wait": waits[0],
            "skew": skew,
            "latest": latest,
            "publish": median([p[0] for p in rank_phase]),
            "rest": max(p[2] for p in rank_phase),
            "ready_spread": max(ready_spreads),
            "wg_durations": wg_durations,
            "envelope": max(envelopes),
        }
        rows.append(row)
        families[(family, shape)].append(row)

    print(f"ranks={len(ranks)} collectives={len(rows)} tick/us={TPUS:g}")
    print("skew=max(publish->ready)-min(publish->ready); latest=min wait (same-rank deltas only)")
    print(f"{'inst':>5} {'producer':<20} {'producer i0:i2':<22} {'shape':<7} {'MiB':>7} {'r0wait':>9} "
          f"{'skew':>9} {'latest':>7} {'publish':>9} {'ready-spr':>9} {'wg-p90':>9} {'wg-max':>9} {'envelope':>9}")
    for r in sorted(rows, key=lambda x: -x["skew"])[: args.top]:
        print(f"{r['inst']:5d} {r['family']:<20.20} {r['producer_dims']:<22.22} "
              f"{r['shape']:<7} {r['mib']:7.1f} "
              f"{r['rank0_wait']:9.2f} {r['skew']:9.2f} {r['latest']:7d} "
              f"{r['publish']:9.2f} {r['ready_spread']:9.2f} "
              f"{percentile(r['wg_durations'], 0.90):9.2f} {max(r['wg_durations']):9.2f} "
              f"{r['envelope']:9.2f}")

    print("\nproducer-family summary")
    print(f"{'producer':<20} {'shape':<7} {'n':>4} {'skew-sum':>10} {'skew-med':>10} "
          f"{'ready-sum':>10} {'wg-p50':>9} {'wg-p90':>9} {'wg-max':>9} {'env-sum':>10}")
    for (family, shape), rs in sorted(families.items(), key=lambda x: -sum(r["skew"] for r in x[1])):
        skews = [r["skew"] for r in rs]
        durations = [d for r in rs for d in r["wg_durations"]]
        print(f"{family:<20.20} {shape:<7} {len(rs):4d} {sum(skews):10.2f} "
              f"{median(skews):10.2f} {sum(r['ready_spread'] for r in rs):10.2f} "
              f"{median(durations):9.2f} {percentile(durations, 0.90):9.2f} "
              f"{max(durations):9.2f} {sum(r['envelope'] for r in rs):10.2f}")


if __name__ == "__main__":
    main()
