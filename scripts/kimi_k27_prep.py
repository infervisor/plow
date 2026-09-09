#!/usr/bin/env python3
"""kimi_k27_prep.py — expert re-encoding for Kimi-K2.7-Code, the piece that gates serving.

`plowc` refuses this checkpoint with `ckpt_quant_compressed-tensors`
(crates/devgen/src/mla.rs): its routed experts are int4, group_size 32, symmetric,
pack-quantized, and `MoeEnc` has Bf16 / Fp8Blk / Mxfp4 and no int4-group-32 arm. Every other
Kimi gate is downstream of that one. See docs/amd/tp-bringup-mi300x.md §5b for the three routes
and why this script implements the fp8 one.

WHAT IS ON DISK (read from a shard header, not assumed):

    …experts.N.{gate,up,down}_proj.weight_packed   I32   [N, K/8]   8 x int4 per int32
    …experts.N.{gate,up,down}_proj.weight_scale    BF16  [N, K/32]  one scale per group of 32
    …experts.N.{gate,up,down}_proj.weight_shape    I32   [2]

The unpack order is NOT guessed — it mirrors `compressed_tensors.compressors.unpack_from_int32`
exactly: nibble `i` of packed word `c` is column `8c + i`, and the stored nibble is unsigned with
an offset of 8, so the value is `(nibble - 8)`. Dequantization is then

    w[n, k] = (nibble(n, k) - 8) * weight_scale[n, k // 32]

WHAT IT WRITES, the block-fp8 contract the shipped arm already reads (ops 45/46/48/49, and
`GemmFp8Blk` 107):

    …weight             F8_E4M3  [N, K]
    …weight_scale_inv   F32      [ceil(N/128), ceil(K/128)]

indexed by the kernel as `S[(n >> 7) * ceil(K/128) + (k >> 7)]` and used as a MULTIPLIER, so
`w ~= fp8 * scale_inv`. That is the same grid GLM-5.3 serves on in this tree today.

THE ACCURACY QUESTION THIS SCRIPT EXISTS TO ANSWER. Going int4 -> fp8 is a WIDENING of the value
grid (e4m3 carries more precision than 16 levels), but it is a COARSENING of the scale grid: one
f32 covers a [128,128] block where the source had a separate bf16 scale per 32 columns — 4 along
K times 128 rows = 512 source scales per destination scale. If those 512 disagree enough, the
block scale cannot serve them all and the widening is undone. `--verify` measures exactly that,
per expert, against the exact dequantization, and reports the error and the in-block scale spread
that drives it. Run it before converting 555 GB.
"""
import argparse
import json
import os
import struct
import sys

BLOCK = 128  # the [128,128] weight_scale_inv grid; not a parameter anywhere in the emitter
GROUP = 32  # compressed-tensors group_size for this checkpoint
E4M3_MAX = 448.0  # largest finite magnitude in e4m3 (OCP fn); the fnuz variant differs


def _torch():
    import torch

    return torch


def shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load_tensor(root, index, name):
    """One tensor, read from whichever shard holds it. safetensors' own reader, mmap'd."""
    from safetensors import safe_open

    path = os.path.join(root, index[name])
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(name)


def dequantize_expert(packed, scale, shape):
    """int4 group-32 -> exact float32, mirroring compressed_tensors.unpack_from_int32."""
    torch = _torch()
    n, k = int(shape[0]), int(shape[1])
    pack_factor = 8
    unpacked = torch.zeros((packed.shape[0], packed.shape[1] * pack_factor), dtype=torch.int32)
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (packed >> (4 * i)) & 0xF
    unpacked = unpacked[:, :k]
    # Stored unsigned with an offset; symmetric int4 is -8..7.
    vals = (unpacked - 8).to(torch.float32)
    s = scale.to(torch.float32)  # [n, k/GROUP]
    # Expand each group scale across its 32 columns.
    s = s.repeat_interleave(GROUP, dim=1)[:, :k]
    assert vals.shape == (n, k) and s.shape == (n, k), (vals.shape, s.shape, (n, k))
    return vals * s


def to_block_fp8(w, block=BLOCK):
    """Exact-dequantized float32 -> (e4m3 bytes, f32 [ceil(N/128), ceil(K/128)] scale_inv).

    Per block: scale_inv = max|w| / E4M3_MAX, so the block's largest magnitude lands on the
    format's largest finite value and nothing in the block saturates. A block that is entirely
    zero gets scale_inv = 1.0 rather than 0, so the dequantization stays well defined.
    """
    torch = _torch()
    n, k = w.shape
    nb, kb = (n + block - 1) // block, (k + block - 1) // block
    scale_inv = torch.ones((nb, kb), dtype=torch.float32)
    q = torch.zeros_like(w)
    for bi in range(nb):
        r0, r1 = bi * block, min((bi + 1) * block, n)
        for bj in range(kb):
            c0, c1 = bj * block, min((bj + 1) * block, k)
            blk = w[r0:r1, c0:c1]
            amax = blk.abs().max().item()
            s = (amax / E4M3_MAX) if amax > 0 else 1.0
            scale_inv[bi, bj] = s
            q[r0:r1, c0:c1] = blk / s
    fp8 = q.to(torch.float8_e4m3fn)
    return fp8, scale_inv


def verify(root, index, layer, experts, proj, block=BLOCK):
    torch = _torch()
    print(f"# layer {layer} proj {proj}: int4 g32 -> block-fp8 [{block},{block}]")
    print(
        f"{'expert':>6} {'rel_err_mean':>13} {'rel_err_max':>12} "
        f"{'scale_spread':>13} {'w_amax':>10}"
    )
    worst = 0.0
    for e in experts:
        base = f"language_model.model.layers.{layer}.mlp.experts.{e}.{proj}_proj"
        packed = load_tensor(root, index, base + ".weight_packed")
        scale = load_tensor(root, index, base + ".weight_scale")
        shape = load_tensor(root, index, base + ".weight_shape")
        w = dequantize_expert(packed, scale, shape)
        fp8, sinv = to_block_fp8(w, block)
        # Round-trip through the SAME arithmetic the kernel does: fp8 value * block scale.
        back = fp8.to(torch.float32) * sinv.repeat_interleave(block, 0).repeat_interleave(
            block, 1
        )[: w.shape[0], : w.shape[1]]
        denom = w.abs().mean().item()
        err = (back - w).abs()
        rel_mean = err.mean().item() / denom
        rel_max = err.max().item() / (w.abs().max().item() or 1.0)
        # The spread that decides whether one block scale can serve 512 source groups.
        s = scale.to(torch.float32).abs()
        spread = (s.max() / s.clamp(min=1e-30).min()).item()
        worst = max(worst, rel_mean)
        print(
            f"{e:6d} {rel_mean:13.5f} {rel_max:12.5f} {spread:13.1f} "
            f"{w.abs().max().item():10.4f}"
        )
    print(f"\nworst mean relative error over {len(experts)} expert(s): {worst:.5f}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/models/kimi_k27_code")
    ap.add_argument("--layer", type=int, default=3, help="0-based; layer 0 is dense")
    ap.add_argument("--experts", default="0,1,2,3", help="comma list")
    ap.add_argument("--proj", default="gate", choices=["gate", "up", "down"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument(
        "--block",
        type=int,
        default=BLOCK,
        help="scale-grid side. 128 is the shipped arm's grid; smaller isolates how much of the "
        "error is the COARSENING of the scale grid rather than e4m3's own precision.",
    )
    a = ap.parse_args()

    index = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
    if a.verify:
        experts = [int(x) for x in a.experts.split(",") if x != ""]
        verify(a.ckpt, index, a.layer, experts, a.proj, a.block)
        return
    print("nothing to do: pass --verify (conversion of the full 555 GB is a separate run)")


if __name__ == "__main__":
    sys.exit(main())
