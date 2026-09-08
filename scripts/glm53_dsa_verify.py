#!/usr/bin/env python3
"""Check plowrt's DSA-indexer fp8 -> bf16 upcast against a torch dequantisation.

The loader's upcast is the one place a block-fp8 GLM-5.3 can go silently wrong: a
scale-grid indexing error produces an indexer of exactly the right SIZE and the right
general magnitude that selects the wrong KV rows, and no serving smoke test would catch
it -- sparse attention is expected to change the tokens, so a plausible answer proves
nothing. So the bytes are compared directly, against the reference's own arithmetic
(vLLM `scaled_dequantize` with `GroupShape(128, 128)`: `(fp8.float() * scale).bfloat16()`).

Run the Rust half first -- it writes one raw bf16 file per tensor:

    PLOW_DSA_VERIFY_CKPT=/workspace/models/GLM-5.3-plow-lite \
    PLOW_DSA_VERIFY_OUT=/tmp/dsa \
      nix develop /app/plow --command cargo test --release -p plowrt --features hsa \
        --lib -- --ignored --nocapture dsa_indexer

then this half (nix's torch is broken on this host; use the qualified interpreter):

    build-gemma31/vllm-python scripts/glm53_dsa_verify.py \
      --ckpt /workspace/models/GLM-5.3-plow-lite --dir /tmp/dsa

Exit status is 0 only if every tensor matches BIT FOR BIT. The bf16 narrowing is
round-to-nearest-even on both sides, so an exact match is the right bar; a max/rms is
reported anyway, because "how far off" is the useful number when it is not zero.
"""

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np
import torch


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def find(ckpt, name):
    """(file, header entry) for `name`, or None. No index.json in the prepped dir."""
    for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        h, _ = read_header(f)
        if name in h:
            return f, h[name]
    return None


DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
}


def load(ckpt, name):
    hit = find(ckpt, name)
    if hit is None:
        return None
    path, e = hit
    _, base = read_header(path)
    lo, hi = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + lo)
        raw = f.read(hi - lo)
    t = torch.frombuffer(bytearray(raw), dtype=DTYPE[e["dtype"]])
    return t.reshape(e["shape"])


def dequant(w_fp8, scale):
    """The reference arithmetic, block-index gather, ceil grid tolerated.

    vLLM: `(x_q.to(f32) * group_broadcast(x_s)).to(bf16)` with GroupShape(128, 128).
    Written as an index gather rather than a repeat_interleave so a ragged last block
    (N or K not a multiple of 128) is handled the same way the C kernels handle it.
    """
    w = w_fp8.float()
    N, K = w.shape
    rows = torch.arange(N) // 128
    cols = torch.arange(K) // 128
    assert scale.shape == (
        (N + 127) // 128,
        (K + 127) // 128,
    ), f"scale grid {tuple(scale.shape)} is not the [128,128] grid over {(N, K)}"
    return (w * scale.float()[rows][:, cols]).bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", default="/tmp/dsa", help="where the Rust half wrote its .bf16 files")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.bf16")))
    if not files:
        print(f"!! no *.bf16 in {a.dir} — run the Rust half first", file=sys.stderr)
        return 2

    worst = 0.0
    bad = 0
    for f in files:
        name = os.path.basename(f)[: -len(".bf16")]
        w = load(a.ckpt, name)
        s = load(a.ckpt, name + "_scale_inv")
        if w is None or s is None:
            print(f"!! {name}: not in the checkpoint (or no scale)", file=sys.stderr)
            bad += 1
            continue
        ref = dequant(w, s)
        got = torch.frombuffer(bytearray(open(f, "rb").read()), dtype=torch.bfloat16)
        if got.numel() != ref.numel():
            print(f"!! {name}: loader gave {got.numel()} elements, reference {ref.numel()}")
            bad += 1
            continue
        got = got.reshape(ref.shape)

        same = torch.equal(got.view(torch.int16), ref.view(torch.int16))
        d = (got.float() - ref.float()).abs()
        # Relative to the reference's own scale, so the numbers mean something across
        # two tensors whose magnitudes differ.
        rms_ref = ref.float().pow(2).mean().sqrt().item()
        mx, rms = d.max().item(), d.pow(2).mean().sqrt().item()
        worst = max(worst, mx)
        if not same:
            bad += 1
        n_diff = int((d > 0).sum())
        print(
            f"{'OK  ' if same else 'FAIL'} {name}  shape={tuple(ref.shape)} "
            f"rms(ref)={rms_ref:.5f}  max|d|={mx:.3e}  rms|d|={rms:.3e}  "
            f"differing={n_diff}/{ref.numel()}"
        )

    print(f"\n{len(files) - bad}/{len(files)} bit-identical; worst max|d| = {worst:.3e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
