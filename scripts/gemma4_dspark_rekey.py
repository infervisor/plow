#!/usr/bin/env python3
"""Write the DSpark block drafter's weights as `model-dspark.safetensors` for
PLOW_EMIT_SPEC=1 PLOW_SPEC_KIND=gemma4_dspark.

plowrt maps every *.safetensors in the checkpoint dir, so the drafter binds beside the target
shards (the model-mtp.safetensors precedent). Every tensor is renamed `dspark.<name>` and kept
bf16, except `fc.weight` [hidden, taps*hidden], split by input columns into
`dspark.fc.<j>.weight` [hidden, hidden] (tap j's block), so the context tail projects each target
tap without concatenating them. The drafter is refused by name wherever it differs from the
target it was trained against. Header metadata records the source config sha256.

Usage:
  python3 scripts/gemma4_dspark_rekey.py <dspark_dir> <target_checkpoint_dir>
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


def text(cfg):
    return cfg.get("text_config", cfg)


def check(cfg, target):
    """Refusals: the drafter must describe the target layers it taps and reads like."""
    t = text(target)
    want = {
        "model_type": cfg.get("model_type") == "gemma4_text",
        "architectures": cfg.get("architectures") == ["Gemma4DSparkModel"],
        "hidden_size == target": cfg.get("hidden_size") == t.get("hidden_size"),
        "vocab_size == target": cfg.get("vocab_size") == t.get("vocab_size"),
        "num_target_layers == target layers": cfg.get("num_target_layers") == t.get("num_hidden_layers"),
        "every layer full_attention": set(cfg.get("layer_types") or ["?"]) == {"full_attention"},
        "attention_k_eq_v": cfg.get("attention_k_eq_v") is True,
        "num_global_key_value_heads == target": cfg.get("num_global_key_value_heads")
        == t.get("num_global_key_value_heads"),
        "global_head_dim == target": cfg.get("global_head_dim") == t.get("global_head_dim"),
        "full-attention rope == target": (cfg.get("rope_parameters") or {}).get("full_attention")
        == (t.get("rope_parameters") or {}).get("full_attention"),
        "no per-layer inputs": not cfg.get("hidden_size_per_layer_input"),
        "markov_head_type vanilla": cfg.get("markov_head_type") == "vanilla",
    }
    taps = cfg.get("target_layer_ids") or []
    want["target_layer_ids ascending below target layers"] = (
        len(taps) > 0
        and taps == sorted(set(taps))
        and all(isinstance(x, int) and 0 <= x < t.get("num_hidden_layers", 0) for x in taps)
    )
    bad = [k for k, ok in want.items() if not ok]
    if bad:
        sys.exit("DSpark drafter refused: " + "; ".join(bad))
    return taps


def layout(header, hidden, n_taps):
    """[(out name, shape, source name, column block or None)] in output order."""
    out = []
    for name, info in sorted(header.items()):
        if name == "__metadata__":
            continue
        if info["dtype"] != "BF16":
            sys.exit(f"{name}: dtype {info['dtype']} (expected BF16)")
        if name == "fc.weight":
            if info["shape"] != [hidden, n_taps * hidden]:
                sys.exit(f"fc.weight shape {info['shape']} (expected [{hidden}, {n_taps * hidden}])")
            out += [(f"dspark.fc.{j}.weight", [hidden, hidden], name, j) for j in range(n_taps)]
        else:
            out.append(("dspark." + name, info["shape"], name, None))
    if not any(src == "fc.weight" for _, _, src, _ in out):
        sys.exit("checkpoint has no fc.weight")
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(src, "config.json")))
    taps = check(cfg, json.load(open(os.path.join(dst, "config.json"))))
    hidden = cfg["hidden_size"]
    path = os.path.join(src, "model.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        plan = layout(header, hidden, len(taps))
        hdr = {
            "__metadata__": {
                "source": os.path.abspath(src),
                "config_sha256": sha256(os.path.join(src, "config.json")),
                "fc_split": str(len(taps)),
            }
        }
        ofs = 0
        for name, shape, _, _ in plan:
            size = 2 * shape[0] * (shape[1] if len(shape) > 1 else 1)
            hdr[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [ofs, ofs + size]}
            ofs += size
        hb = json.dumps(hdr, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        target = os.path.join(dst, "model-dspark.safetensors")
        tmp = target + ".tmp"
        with open(tmp, "wb") as out:
            out.write(struct.pack("<Q", len(hb)))
            out.write(hb)
            for _, shape, name, block in plan:
                a, b = header[name]["data_offsets"]
                f.seek(base + a)
                data = f.read(b - a)
                if block is not None:
                    row, part = 2 * len(taps) * hidden, 2 * hidden
                    data = b"".join(
                        data[r * row + block * part:r * row + (block + 1) * part] for r in range(shape[0])
                    )
                out.write(data)
    os.replace(tmp, target)
    print(f"wrote {target}: {len(plan)} tensors, {ofs} bytes, sha256 {sha256(target)}")


if __name__ == "__main__":
    main()
