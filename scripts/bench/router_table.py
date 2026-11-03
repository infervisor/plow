#!/usr/bin/env python3
"""Extract a recorded MoE router table (real routing) from a plowrt activation dump.

The NVIDIA equivalent of the AMD capture's `rank0.act.tab.bin`: serve the FULL production packet,
send one real prompt of length L, and dump the routing of the layer the block was cut from:

    env PLOW_NV_DUMP_KV=<dir> PLOW_NV_DUMP_ACT=<L>:<pc> plowrt serve --assets <full assets> ...
    # <pc>: a prefill-program instruction index at or after that layer's MoeRouterGemmaPf
    #       (`plowrt disasm <full>/model.pkt`); the dump runs the chain through its segment.
    router_table.py <dir>/act-p<L>-pc<pc>-c<c0>.json --out <scratch>/router-L<layer>.bin

The output is the request's rows of `act.moe.table` ({u32 expert, f32 gate} x top-k per row),
validated, for `op_roof.py --router-table` / `block_roofline.py --router-table`.
"""
import argparse
import json
import math
import struct
from pathlib import Path


def extract(meta, data):
    entry = next((t for t in meta["tensors"] if t["name"] == "act.moe.table"), None)
    if entry is None:
        raise ValueError("dump has no act.moe.table (packet without MoE, or a plowrt before the table was dumped)")
    row_bytes, rows, row0 = entry["row_bytes"], meta["rows"], meta["row0"]
    if row_bytes % 8 or not row_bytes:
        raise ValueError(f"act.moe.table row of {row_bytes} bytes is not whole {{id, gate}} entries")
    begin = entry["offset"] + row0 * row_bytes
    table = data[begin:begin + rows * row_bytes]
    if len(table) != rows * row_bytes:
        raise ValueError("dump is shorter than its table rows")
    topk = row_bytes // 8
    for r in range(rows):
        ids = set()
        for c in range(topk):
            expert, gate = struct.unpack_from("<If", table, (r * topk + c) * 8)
            if not math.isfinite(gate) or gate < 0 or expert in ids:
                raise ValueError(f"row {r}: invalid gate or repeated expert (pc before the router?)")
            ids.add(expert)
    return table, topk, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta", help="act-p<L>-pc<pc>-c<c0>.json written by PLOW_NV_DUMP_ACT")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta_path = Path(a.meta)
    meta = json.loads(meta_path.read_text())
    table, topk, rows = extract(meta, meta_path.with_suffix(".bin").read_bytes())
    Path(a.out).write_bytes(table)
    print(f"{a.out}: {rows} rows x top-{topk} (prompt {meta['prompt_len']}, pc {meta['pc']}, c0 {meta['c0']})")


if __name__ == "__main__":
    main()
