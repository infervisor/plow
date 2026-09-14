#!/usr/bin/env python3
"""Compare native BF16 instruction traces with PyTorch decoder activations."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(native, reference, name):
    if native.shape != reference.shape or not np.isfinite(native).all() or not np.isfinite(reference).all():
        raise ValueError(f"invalid activation: {name}")
    a, b = native.astype(np.float64), reference.astype(np.float64)
    delta = a-b
    return {"name": name, "relative_l2": float(np.linalg.norm(delta)/max(np.linalg.norm(b), 1e-20)),
            "max_abs": float(np.abs(delta).max()), "equal_fraction": float(np.mean(a == b))}


def read_native(path, count):
    words = np.fromfile(path, dtype="<u2", count=count)
    if len(words) != count:
        raise ValueError(f"short activation: {path}")
    return (words.astype(np.uint32) << 16).view(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Also isolate the first RMSNorm rounding boundary")
    args = parser.parse_args()
    instructions = json.loads((args.native / "metadata.json").read_text())
    rows = []
    pairs = [
        ("input_layernorm.input", "RmsNorm", "act.x", 0),
        ("input_layernorm.output", "RmsNorm", "act.hn", 0),
        ("self_attn.q_proj.output", "GemmMed", "act.qg", 0),
        ("self_attn.k_proj.output", "GemmMed", "act.kg", 0),
        ("self_attn.v_proj.output", "GemmMed", "act.vg", 0),
        ("self_attn.o_proj.input", "GemmMed", "act.at", 0),
        ("self_attn.o_proj.output", "GemmMed", "act.og", 0),
        ("post_attention_layernorm.input", "Residual", "act.x", 0),
        ("post_attention_layernorm.output", "RmsNorm", "act.hn", 1),
        ("mlp.down_proj.input", "GemmGlu", "act.fu", 0),
        ("mlp.down_proj.output", "GemmMed", "act.dg", 0),
    ]
    for suffix, op, tensor, occurrence in pairs:
        selected = [i for i in instructions if i["layer"] == 0 and i["op"] == op and tensor in i["saved"]]
        record = selected[occurrence]
        name = f"layers.0.{suffix}"
        expected = np.fromfile(args.reference / f"{name}.f32", dtype="<f4")
        actual = read_native(args.native / record["saved"][tensor], len(expected))
        rows.append(compare(actual, expected, name))
    layers = sorted({i["layer"] for i in instructions})
    for layer in layers:
        record = [i for i in instructions if i["layer"] == layer and i["op"] == "Residual"][-1]
        name = f"layers.{layer}.output"
        expected = np.fromfile(args.reference / f"{name}.f32", dtype="<f4")
        actual = read_native(args.native / record["saved"]["act.x"], len(expected))
        rows.append(compare(actual, expected, name))
    expected = np.fromfile(args.reference / "logits.f32", dtype="<f4")
    rows.append(compare(read_native(args.native / "logits.bin", len(expected)), expected, "logits"))
    if args.checkpoint:
        import torch
        from safetensors import safe_open
        weight_name = "thinker.model.layers.0.input_layernorm.weight"
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
        config = json.loads((args.checkpoint / "config.json").read_text())["thinker_config"]["text_config"]
        with safe_open(args.checkpoint / index["weight_map"][weight_name], framework="pt", device="cpu") as weights:
            gamma = weights.get_tensor(weight_name)
        values = np.fromfile(args.reference / "layers.0.input_layernorm.input.f32", dtype="<f4").reshape(-1, gamma.numel())
        x = torch.from_numpy(values)
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + config["rms_norm_eps"])
        rounded = ((x*inv).to(torch.bfloat16)*gamma).float().numpy().reshape(-1)
        fused = (x*inv*gamma.float()).to(torch.bfloat16).float().numpy().reshape(-1)
        reference = np.fromfile(args.reference / "layers.0.input_layernorm.output.f32", dtype="<f4")
        record = next(i for i in instructions if i["layer"] == 0 and i["op"] == "RmsNorm")
        native = read_native(args.native / record["saved"]["act.hn"], len(reference))
        rows.append(compare(rounded, reference, "RMSNorm: explicit BF16 intermediate vs reference"))
        rows.append(compare(fused, native, "RMSNorm: fused FP32 intermediate vs native"))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
