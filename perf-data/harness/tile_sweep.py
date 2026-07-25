#!/usr/bin/env python3
"""SplitZip lossless: compression ratio vs TILE SIZE, on real weights (CPU).

Extends expo_hist.py (single tile = 8192 elems) to the tile ladder, grounded by
plow's SRAM page size (costmodel::DEFAULT_PAGE_BYTES = 8 KiB):

    tile elems | bf16 bytes | pages (8 KiB)
    4096       |  8 KiB     | 1
    8192       | 16 KiB     | 2
    16384      | 32 KiB     | 4
    32768      | 64 KiB     | 8

Format (tile-local code table, tile-local escape list):

    bits/elem = code_bits
              + payload_bits                       # sign+mantissa: bf16 8, e4m3 4
              + esc_rate * (POS_BITS + exp_bits)   # escape: position + raw exponent
              + meta/tile_elems                    # code table + escape count

CORRECTION vs expo_hist.py: that script pins POS_BITS=10 (a 1024-elem chunk) while
tiling at 8192 elems. A tile-local escape position needs ceil(log2(tile_elems)) bits,
so escape cost there is understated. Here POS_BITS = log2(tile_elems), which is the
whole point of a tile-size sweep (bigger tile => cheaper meta but pricier escapes).

meta = 2^code_bits * exp_bits (table) + 16 (escape count), matching expo_hist.py's
reported meta at tile=8192 (top-8: 80/8192 = 0.0098 b/elem).
"""
import argparse, glob, os
import numpy as np
from safetensors import safe_open
import torch

TILES = [4096, 8192, 16384, 32768]
CODES = [(2, 4), (3, 8), (4, 16), (5, 32)]  # (code_bits, top-K)
PAGE_BYTES = 8 * 1024


def tensor_iter(model_dir):
    """2-D GEMM weights only, matching expo_hist.py's selection."""
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for name in sorted(sf.keys()):
                if len(sf.get_slice(name).get_shape()) != 2:
                    continue
                if "embed" in name or "lm_head" in name or "scale" in name:
                    continue
                yield name, sf.get_tensor(name)


def exponents(tt, dtype):
    if dtype == "bf16":
        u = tt.view(torch.int16).numpy().view(np.uint16)
        return ((u >> 7) & 0xFF).astype(np.uint8), 8, 8, 16.0  # exp_bits, payload, raw
    u = tt.view(torch.int8).numpy().view(np.uint8)
    return ((u >> 3) & 0x0F).astype(np.uint8), 4, 4, 8.0


def tile_escapes(e_flat, tile, topk):
    """Per-tile escape counts using a tile-local top-K exponent table."""
    n = (e_flat.size // tile) * tile
    if n == 0:
        return None
    g = e_flat[:n].reshape(-1, tile)
    vals = np.unique(g)                      # few distinct exponents in practice
    counts = np.empty((g.shape[0], vals.size), dtype=np.int32)
    for i, v in enumerate(vals):
        counts[:, i] = (g == v).sum(axis=1)
    if vals.size > topk:
        part = np.partition(counts, vals.size - topk, axis=1)[:, vals.size - topk:]
        covered = part.sum(axis=1)
    else:
        covered = counts.sum(axis=1)
    return tile - covered                    # escapes per tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--dtype", choices=["bf16", "e4m3"], default="bf16")
    ap.add_argument("--limit", type=int, default=0, help="max tensors (0=all)")
    a = ap.parse_args()

    # acc[(tile, code_bits)] = [sum_bits, n_tiles, worst_ratio, n_lose, sum_esc]
    acc = {(t, cb): [0.0, 0, 9e9, 0, 0] for t in TILES for cb, _ in CODES}
    total_elems = 0

    for i, (name, tt) in enumerate(tensor_iter(a.model_dir)):
        if a.limit and i >= a.limit:
            break
        e, exp_bits, payload_bits, raw_bits = exponents(tt, a.dtype)
        flat = e.ravel()
        total_elems += flat.size
        for tile in TILES:
            pos_bits = int(np.log2(tile))
            for code_bits, topk in CODES:
                esc = tile_escapes(flat, tile, topk)
                if esc is None:
                    continue
                meta = (2 ** code_bits) * exp_bits + 16
                bits = (code_bits + payload_bits
                        + (esc / tile) * (pos_bits + exp_bits)
                        + meta / tile)
                r = acc[(tile, code_bits)]
                r[0] += bits.sum()
                r[1] += bits.size
                r[2] = min(r[2], raw_bits / bits.max())
                r[3] += int((bits >= raw_bits).sum())
                r[4] += int(esc.sum())

    raw_bits = 16.0 if a.dtype == "bf16" else 8.0
    print(f"\n{'='*86}")
    print(f"SplitZip tile-size sweep — {a.dtype} — {a.model_dir}")
    print(f"elements: {total_elems:.4g}   page={PAGE_BYTES//1024} KiB")
    print(f"{'='*86}")
    hdr = (f"{'tile':>7} {'KiB':>5} {'pg':>3} {'code':>6} {'pos':>4} "
           f"{'meta b/el':>10} {'esc/tile':>9} {'RATIO':>8} {'worst':>7} {'lose':>6}")
    for tile in TILES:
        print()
        print(hdr if tile == TILES[0] else "")
        kib = tile * (2 if a.dtype == "bf16" else 1) // 1024
        for code_bits, topk in CODES:
            s, n, worst, lose, esc = acc[(tile, code_bits)]
            if n == 0:
                continue
            ratio = raw_bits / (s / n)
            meta = ((2 ** code_bits) * (8 if a.dtype == "bf16" else 4) + 16) / tile
            print(f"{tile:>7} {kib:>5} {kib*1024//PAGE_BYTES:>3} "
                  f"top{topk:<2}/{code_bits}b {int(np.log2(tile)):>4} "
                  f"{meta:>10.4f} {esc/n:>9.1f} {ratio:>8.4f}x {worst:>6.4f}x {lose:>6}")


if __name__ == "__main__":
    main()
