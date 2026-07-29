#!/usr/bin/env python3
"""Collate glm52_ctx_sweep.sh logs into the TP x ctx table.

Both engines are measured by the SAME client (`vllm bench serve`), so both logs carry
the same header line and the same CSV columns; the only difference is which server the
client was pointed at. That is the whole point of §0-BENCH and it is why one parser
serves both.

Reads   <dir>/plow_tp<N>.log  and  <dir>/vllm_tp<N>.log
Writes  the markdown table on stdout.
"""
import re
import sys
import glob
import os

HDR = "input_len,concurrency,ttft_ms"


def rows(path):
    """CSV lines emitted by the bench scripts, keyed by input_len."""
    out = {}
    if not os.path.exists(path):
        return out
    seen_hdr = False
    for line in open(path):
        line = line.strip()
        if line.startswith(HDR):
            seen_hdr = True
            continue
        if not seen_hdr:
            continue
        f = line.split(",")
        # ttft_ms,ttft_med,tpot_ms,tpot_med,itl_ms,itl_med,out_tok_s[,req_per_s]
        if len(f) < 9 or not f[0].isdigit():
            continue
        try:
            out[int(f[0])] = [float(x) for x in f[2:]]
        except ValueError:
            continue
    return out


def main(d):
    tps = sorted(
        {int(m.group(1))
         for p in glob.glob(os.path.join(d, "*_tp*.log"))
         if (m := re.search(r"_tp(\d+)\.log$", p))}
    )
    print("| TP | ctx | engine | TTFT mean ms | TTFT med ms | TPOT mean ms | TPOT med ms | out tok/s |")
    print("|---|--:|---|--:|--:|--:|--:|--:|")
    for tp in tps:
        p = rows(os.path.join(d, f"plow_tp{tp}.log"))
        v = rows(os.path.join(d, f"vllm_tp{tp}.log"))
        for ctx in sorted(set(p) | set(v)):
            for name, r in (("plow", p.get(ctx)), ("vLLM", v.get(ctx))):
                if r is None:
                    print(f"| {tp} | {ctx} | {name} | — | — | — | — | not run |")
                    continue
                print(f"| {tp} | {ctx} | {name} | {r[0]:.1f} | {r[1]:.1f} | "
                      f"{r[2]:.2f} | {r[3]:.2f} | {r[6]:.1f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/lava/models/glm52_ctxsweep")
