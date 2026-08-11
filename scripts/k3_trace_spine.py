#!/usr/bin/env python3
"""Attribute a K3 raw trace along its observed latest-predecessor spine.

Unlike an instruction-order envelope sum, this follows the packet counter DAG.
At every join it selects the predecessor that actually finished last, then
charges the elapsed time from that predecessor's completion to the consumer's
completion. The charges telescope to the traced step span while preserving
overlap on sibling branches.

`pre_body` is the spine charge minus this packet's traced work envelope. At a
join it includes work on the hidden sibling subtree; it is not a packet-local
queue-stall estimate.
"""

import argparse
import json
import struct
from collections import defaultdict


REC = struct.Struct("<IIIHHQQQ")
TICKS_PER_US = 100.0


def fold_trace(path):
    packets = {}
    data = open(path, "rb").read()
    if len(data) % REC.size:
        raise SystemExit(f"{path}: size is not a multiple of {REC.size}")
    for offset in range(0, len(data), REC.size):
        _, _, inst, op, _, arrive, ready, end = REC.unpack_from(data, offset)
        if not end:
            continue
        packet = packets.get(inst)
        if packet is None:
            packets[inst] = [op, 1, arrive, ready, end]
        else:
            packet[1] += 1
            packet[2] = min(packet[2], arrive)
            packet[3] = max(packet[3], ready)
            packet[4] = max(packet[4], end)
    return packets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("disasm_json")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--gemv-shapes", action="store_true")
    parser.add_argument("--gemv-instances", type=int, default=0)
    args = parser.parse_args()

    packets = fold_trace(args.trace)
    with open(args.disasm_json) as stream:
        document = json.load(stream)
    if len(document["programs"]) != 1:
        raise SystemExit("disassembly must contain exactly one program")
    program = document["programs"][0]
    insts = {inst["idx"]: inst for inst in program["insts"]}
    predecessors = defaultdict(set)
    for counter in program["counters"]["per_counter"]:
        producer = counter["producer"]
        if producer is None:
            continue
        for consumer in counter["consumers"]:
            predecessors[consumer].add(producer)

    missing = sorted(set(insts) - set(packets))
    if missing:
        raise SystemExit(f"trace is missing {len(missing)} packets, first={missing[:8]}")
    for idx, packet in packets.items():
        if idx not in insts:
            raise SystemExit(f"trace packet {idx} is absent from disassembly")
        if packet[0] != insts[idx]["op"]:
            raise SystemExit(
                f"packet {idx}: trace opcode {packet[0]} != disassembly {insts[idx]['op']}"
            )

    current = max(packets)
    reverse_spine = []
    seen = set()
    while True:
        if current in seen:
            raise SystemExit(f"cycle while backtracking at packet {current}")
        seen.add(current)
        reverse_spine.append(current)
        available = [idx for idx in predecessors[current] if idx in packets]
        if not available:
            break
        current = max(available, key=lambda idx: packets[idx][4])
    spine = list(reversed(reverse_spine))

    start = min(packet[2] for packet in packets.values())
    final_end = packets[spine[-1]][4]
    aggregate = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    gemv_shapes = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    gemv_instances = []
    previous_end = start
    total_charge = 0.0
    for idx in spine:
        op, _, _, ready, end = packets[idx]
        charge = (end - previous_end) / TICKS_PER_US
        body = (end - ready) / TICKS_PER_US
        pre_body = charge - body
        if charge < -1e-6:
            raise SystemExit(f"packet {idx}: negative spine charge {charge:.6f} us")
        name = insts[idx]["op_name"]
        row = aggregate[(op, name)]
        row[0] += 1
        row[1] += charge
        row[2] += pre_body
        row[3] += body
        if name == "Gemv":
            values = {value["name"]: value["value"] for value in insts[idx]["ints"]}
            shape = (
                values["N"],
                values["K"],
                insts[idx]["blocks"],
                values["norm"],
            )
            shape_row = gemv_shapes[shape]
            shape_row[0] += 1
            shape_row[1] += charge
            shape_row[2] += pre_body
            shape_row[3] += body
            weight = next(
                value.get("tensor", "")
                for value in insts[idx]["tensors"]
                if value["name"] == "W"
            )
            predecessor = predecessors[idx]
            latest = max(predecessor, key=lambda value: packets[value][4]) if predecessor else None
            gemv_instances.append(
                (
                    charge,
                    idx,
                    values["N"],
                    values["K"],
                    insts[idx]["blocks"],
                    pre_body,
                    body,
                    latest,
                    insts[latest]["op_name"] if latest is not None else "-",
                    weight,
                )
            )
        total_charge += charge
        previous_end = end

    span = (final_end - start) / TICKS_PER_US
    if abs(total_charge - span) > 1e-3:
        raise SystemExit(f"charges {total_charge:.6f} us do not telescope to span {span:.6f} us")

    print(
        f"trace_packets={len(packets)} graph_cp={program['counters']['aggregate']['critical_path']} "
        f"observed_spine={len(spine)} span_us={span:.3f} charge_us={total_charge:.3f}"
    )
    print(f"{'op':<30}{'n':>6}{'charge_us':>13}{'pre_body':>13}{'body_us':>13}")
    for (_, name), (count, charge, pre_body, body) in sorted(
        aggregate.items(), key=lambda item: -item[1][1]
    )[: args.top]:
        print(f"{name:<30}{count:>6}{charge:>13.3f}{pre_body:>13.3f}{body:>13.3f}")
    print(f"{'TOTAL':<30}{len(spine):>6}{total_charge:>13.3f}")
    if args.gemv_shapes:
        print()
        print(f"{'N':>8}{'K':>8}{'grid':>8}{'norm':>7}{'n':>6}{'charge_us':>13}{'pre_body':>13}{'body_us':>13}")
        for (n, k, grid, norm), (count, charge, pre_body, body) in sorted(
            gemv_shapes.items(), key=lambda item: -item[1][1]
        ):
            print(
                f"{n:>8}{k:>8}{grid:>8}{norm:>7}{count:>6}"
                f"{charge:>13.3f}{pre_body:>13.3f}{body:>13.3f}"
            )
    if args.gemv_instances:
        print()
        print(
            f"{'idx':>6}{'N':>8}{'K':>8}{'grid':>8}{'charge':>11}{'pre_body':>11}"
            f"{'body':>11}{'pred':>8} {'pred_op':<20} weight"
        )
        for charge, idx, n, k, grid, pre_body, body, latest, pred_op, weight in sorted(
            gemv_instances, reverse=True
        )[: args.gemv_instances]:
            print(
                f"{idx:>6}{n:>8}{k:>8}{grid:>8}{charge:>11.3f}{pre_body:>11.3f}"
                f"{body:>11.3f}{('-' if latest is None else latest):>8} {pred_op:<20} {weight}"
            )


if __name__ == "__main__":
    main()
