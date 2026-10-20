#!/usr/bin/env python3
"""Write FP8 twins of the Gemma-4 assistant's GEMV weights as `model-mtp-fp8.safetensors`.

Reads `model-mtp.safetensors` (scripts/gemma4_mtp_rekey.py) from <checkpoint_dir> and writes, for the
tied draft head (`mtp.model.embed_tokens.weight`) and every draft GEMV weight (pre/post projections,
q_proj, o_proj, gate/up/down), the per-output-row e4m3 twin the PLOW_SPEC_DRAFT_FP8 draft programs
bind, in the quantize_fp8_head.py convention (scale = amax(row)/448, W8 = e4m3(W/scale)):
  fp8/<name>        F8_E4M3 [N, K]
  fp8/<name>_scale  F32     [N]
The bf16 originals stay in model-mtp.safetensors (the draft embedding lookup reads the bf16 table).

Usage:
  python3 scripts/gemma4_mtp_fp8.py <checkpoint_dir>
"""
import hashlib
import json
import os
import re
import struct
import sys

import torch

TWINS = re.compile(
    r"^mtp\.(model\.embed_tokens|pre_projection_[eh]|post_projection"
    r"|model\.layers\.\d+\.(self_attn\.[qo]_proj|mlp\.(gate|up|down)_proj))\.weight$"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], "model-mtp.safetensors")
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
            a, b = info["data_offsets"]
            f.seek(base + a)
            rows, cols = info["shape"]
            w = torch.frombuffer(bytearray(f.read(b - a)), dtype=torch.bfloat16).reshape(rows, cols).float()
            if not torch.isfinite(w).all():
                sys.exit(f"{name}: nonfinite weights")
            sc = (w.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
            w8 = (w / sc[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
            out[f"fp8/{name}"] = ("F8_E4M3", [rows, cols], w8.numpy().tobytes())
            out[f"fp8/{name}_scale"] = ("F32", [rows], sc.numpy().tobytes())
    if not any(k == "fp8/mtp.model.embed_tokens.weight" for k in out):
        sys.exit(f"{src}: no mtp.model.embed_tokens.weight")
    meta = {"source_shard_sha256": sha256(src), "scheme": "e4m3 per-output-row, scale=amax/448"}
    hdr = {"__metadata__": meta}
    ofs = 0
    for name, (dtype, shape, data) in out.items():
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [ofs, ofs + len(data)]}
        ofs += len(data)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    target = os.path.join(sys.argv[1], "model-mtp-fp8.safetensors")
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
