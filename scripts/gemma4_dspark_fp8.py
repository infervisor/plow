#!/usr/bin/env python3
"""Write FP8 twins of the DSpark drafter's projection weights as `model-dspark-fp8.safetensors`.

Reads `model-dspark.safetensors` (scripts/gemma4_dspark_rekey.py) from <checkpoint_dir> and writes,
for the context projections (`dspark.fc.<j>`, k_proj) and every draft-program GEMM/GEMV weight
(q/o, gate/up/down, lm_head), the per-output-row e4m3 twin in the quantize_fp8_head.py convention
(scale = amax(row)/448, W8 = e4m3(W/scale)):
  fp8/<name>        F8_E4M3 [N, K]
  fp8/<name>_scale  F32     [N]
The bf16 originals stay in model-dspark.safetensors (norms, embeddings and the Markov head read them).

Usage:
  python3 scripts/gemma4_dspark_fp8.py <checkpoint_dir>
"""
import hashlib
import json
import os
import re
import struct
import sys

import torch

TWINS = re.compile(
    r"^dspark\.(fc\.\d+|lm_head|layers\.\d+\.(self_attn\.[qko]_proj|mlp\.(gate|up|down)_proj))\.weight$"
)
ROWS = 16384


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], "model-dspark.safetensors")
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in sorted(header.items()):
            if name == "__metadata__" or not TWINS.match(name):
                continue
            if info["dtype"] != "BF16" or len(info["shape"]) != 2:
                sys.exit(f"{name}: {info['dtype']} {info['shape']} (expected 2-D BF16)")
            a, _ = info["data_offsets"]
            rows, cols = info["shape"]
            w8s, scs = [], []
            for r0 in range(0, rows, ROWS):
                nr = min(ROWS, rows - r0)
                f.seek(base + a + 2 * r0 * cols)
                w = torch.frombuffer(bytearray(f.read(2 * nr * cols)), dtype=torch.bfloat16)
                w = w.reshape(nr, cols).float()
                if not torch.isfinite(w).all():
                    sys.exit(f"{name}: nonfinite weights")
                sc = (w.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
                w8s.append((w / sc[:, None]).to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes())
                scs.append(sc.numpy().tobytes())
            out[f"fp8/{name}"] = ("F8_E4M3", [rows, cols], b"".join(w8s))
            out[f"fp8/{name}_scale"] = ("F32", [rows], b"".join(scs))
    if not any(k.startswith("fp8/dspark.fc.") for k in out):
        sys.exit(f"{src}: no dspark.fc.<j>.weight")
    meta = {"source_shard_sha256": sha256(src), "scheme": "e4m3 per-output-row, scale=amax/448"}
    hdr = {"__metadata__": meta}
    ofs = 0
    for name, (dtype, shape, data) in out.items():
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [ofs, ofs + len(data)]}
        ofs += len(data)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    target = os.path.join(sys.argv[1], "model-dspark-fp8.safetensors")
    tmp = target + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, _, data in out.values():
            f.write(data)
    os.replace(tmp, target)
    print(f"wrote {target}: {len(out)} tensors, {ofs} bytes, sha256 {sha256(target)}")


if __name__ == "__main__":
    main()
