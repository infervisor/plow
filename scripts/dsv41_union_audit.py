#!/usr/bin/env python3
"""How clustered is the DSA selection?

A gathered prefill cannot use a query-row-tiled MFMA arm, because a tile of query rows shares no
KV range (interp.hip says so at the FLASH_GATHER_PREFILL dispatch). The one way out is to attend
over the UNION of a tile's selections and mask each query down to its own top-k -- which costs
|union| / top_k times the arithmetic. This prices that multiplier from the SHIPPED table rather
than from an assumption about locality.

    rung_run <pkt> <hsaco> --checkpoint <ckpt> --tp 8 --iters 1 \\
        --dump act.index_idx=/tmp/idx.bin
    scripts/dsv41_union_audit.py /tmp/idx.bin --tokens 8192 --topk 512

Layer 2 at T=8192 (a 4096-row compressed cache) measured mean |U| = 2059 at a 64-query tile --
half the cache, 4.29x the work. See docs/amd/deepseek-v41-flash-mi300x.md 12.31.
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("idx", help="raw i32 [tokens][topk] dump of act.index_idx")
ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--topk", type=int, required=True)
ap.add_argument("--tiles", type=int, nargs="+", default=[16, 32, 64, 128, 256])
a = ap.parse_args()

t = np.fromfile(a.idx, dtype=np.int32).reshape(a.tokens, a.topk)
# -1 is the PAD d_index_select_pf writes when a query has fewer than topk candidates.
live = t >= 0
ideal = int(live.sum())
print(f"queries {a.tokens}  slots {a.topk}  live {100.0 * live.mean():.2f}%  "
      f"rows named 0..{t.max()}  sparse ideal {ideal} query-kv pairs")
print()
print("  tile   tiles   mean|U|    p50    p95    max        work    vs ideal")
for tile in a.tiles:
    us, work = [], 0
    for t0 in range(0, a.tokens, tile):
        blk = t[t0:t0 + tile]
        u = np.unique(blk[blk >= 0]).size
        us.append(u)
        work += blk.shape[0] * u
    us = np.array(us)
    print(f"  {tile:4d}  {len(us):6d}   {us.mean():7.1f} {np.percentile(us, 50):6.0f} "
          f"{np.percentile(us, 95):6.0f} {us.max():6d}  {work:10d}     {work / ideal:5.2f}x")
