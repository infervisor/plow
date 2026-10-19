#!/usr/bin/env python3
"""Write the Gemma-4 assistant (MTP drafter) weights as `model-mtp.safetensors` for PLOW_EMIT_SPEC=1.

plowrt maps every *.safetensors in the checkpoint dir, so the drafter binds beside the target
shards (the model-head.safetensors precedent). Every tensor is renamed `mtp.<name>` and kept
bf16, except `pre_projection.weight` [hidden, 2*backbone], split by columns into
`mtp.pre_projection_e.weight` (the token-embedding half) and `mtp.pre_projection_h.weight`
(the target-hidden half), so the draft step needs no concatenation. `layer_scalar` stays in the
shard (devgen reads it as an immediate). Header metadata records the source config and tokenizer
sha256; devgen pins the shard itself.

Usage:
  python3 scripts/gemma4_mtp_rekey.py <assistant_dir> <target_checkpoint_dir>
"""
import hashlib
import json
import os
import struct
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(src, "config.json")))
    if cfg.get("model_type") != "gemma4_assistant":
        sys.exit(f"{src}: model_type {cfg.get('model_type')!r} is not gemma4_assistant")
    backbone = cfg["backbone_hidden_size"]
    path = os.path.join(src, "model.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            if info["dtype"] != "BF16":
                sys.exit(f"{name}: dtype {info['dtype']} (expected BF16)")
            a, b = info["data_offsets"]
            f.seek(base + a)
            data = f.read(b - a)
            if name == "pre_projection.weight":
                rows, cols = info["shape"]
                if cols != 2 * backbone:
                    sys.exit(f"pre_projection.weight shape {info['shape']} (expected [*, {2 * backbone}])")
                row = cols * 2
                half = backbone * 2
                e = b"".join(data[r * row:r * row + half] for r in range(rows))
                h = b"".join(data[r * row + half:(r + 1) * row] for r in range(rows))
                out["mtp.pre_projection_e.weight"] = ([rows, backbone], e)
                out["mtp.pre_projection_h.weight"] = ([rows, backbone], h)
            else:
                out["mtp." + name] = (info["shape"], data)
    meta = {
        "source": os.path.abspath(src),
        "config_sha256": sha256(os.path.join(src, "config.json")),
        "tokenizer_sha256": sha256(os.path.join(src, "tokenizer.json")),
        "pre_projection_split": str(backbone),
    }
    hdr = {"__metadata__": meta}
    ofs = 0
    for name, (shape, data) in out.items():
        hdr[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [ofs, ofs + len(data)]}
        ofs += len(data)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    target = os.path.join(dst, "model-mtp.safetensors")
    tmp = target + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, data in out.values():
            f.write(data)
    os.replace(tmp, target)
    print(f"wrote {target}: {len(out)} tensors, {ofs} bytes, sha256 {sha256(target)}")


if __name__ == "__main__":
    main()
