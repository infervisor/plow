#!/usr/bin/env python3
"""Per-opcode prefill attribution from a PLOW_PF_SEG_TIME=1 server log.

    segtime_table.py <server.log>

plowrt emits one "seg-class wall time (chunk)" line per prefill chunk followed by that
chunk's "segment-site wall time (chunk)" lines (sites="pc12:Gemm+pc13:Gemm" count=N
elapsed_ms="..."). This groups the site lines per chunk, collapses the pc indices to an
opcode signature, and prints one share table per chunk.

The totals are ATTRIBUTION SHARES, never latency: SEG_TIME drains per segment and
inflates wall time ~3.7x. Compare shares within a chunk; do not quote ms as TTFT.
"""
import collections
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SITE = re.compile(r'sites="([^"]+)" count=(\d+) elapsed_ms="([0-9.]+)"')
MOE = {"73": "MoeRouterPf", "74": "MoeAlignPf", "75": "MoeGroupGluPf",
       "76": "MoeGroupDownPf", "77": "MoeCombineNormPf"}


def signature(sites: str) -> str:
    ops = []
    for part in sites.split("+"):
        op = re.sub(r"pc\d+:", "", part)
        ops.append(MOE.get(op, op))
    return "+".join(sorted(set(ops)))


def main() -> None:
    chunks, cur = [], None
    for raw in open(sys.argv[1], errors="replace"):
        line = ANSI.sub("", raw)
        if "seg-class wall time (chunk)" in line:
            cur = collections.defaultdict(lambda: [0, 0.0])
            chunks.append(cur)
        elif "segment-site wall time (chunk)" in line and cur is not None:
            m = SITE.search(line)
            if m:
                sig = signature(m.group(1))
                cur[sig][0] += int(m.group(2))
                cur[sig][1] += float(m.group(3))
    for i, agg in enumerate(chunks):
        tot = sum(v[1] for v in agg.values()) or 1.0
        print(f"\n== chunk {i}: {tot:.1f} ms attributed (shares, not latency)")
        print(f"{'opcode signature':52} {'launches':>8} {'ms':>9} {'share':>6}")
        for sig, (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:10]:
            print(f"{sig[:52]:52} {n:8d} {ms:9.2f} {100 * ms / tot:5.1f}%")


if __name__ == "__main__":
    main()
