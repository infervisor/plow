#!/usr/bin/env python3
"""Check sampled rows of a block_run GLU capture against its own FP8 inputs (CPU)."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def projection(x, weight, activation_scale, weight_scale):
    dot = x.double() @ weight.double().T
    return (dot.float() * activation_scale[:, None] * weight_scale[None, :]).to(torch.bfloat16).float()


def glu(gate, up):
    return (0.5 * gate * (1 + torch.tanh(0.7978845608 *
            (gate + 0.044715 * gate * gate * gate))) * up).to(torch.bfloat16).float()


def metrics(got, reference):
    if got.shape != reference.shape or not torch.isfinite(got).all() or not torch.isfinite(reference).all():
        raise ValueError("shape mismatch or non-finite values")
    delta = got.double() - reference.double()
    return {
        "rel_l2": (delta.norm() / reference.double().norm().clamp_min(1e-30)).item(),
        "max_abs": delta.abs().max().item(),
        "equal_fraction": (got == reference).double().mean().item(),
    }


def read_rows(directory, item, dtype, width, rows, input_rows):
    size = torch.empty((), dtype=dtype).element_size()
    path = directory / item["file"]
    if path.stat().st_size != item["bytes"] or item["bytes"] < input_rows * width * size:
        raise ValueError(f"invalid allocation size: {path}")
    result = []
    with path.open("rb") as source:
        for row in rows:
            source.seek(row * width * size)
            raw = source.read(width * size)
            result.append(torch.frombuffer(bytearray(raw), dtype=dtype).clone())
    return torch.stack(result).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", default="0,63,64,127,128,1023,2047,4095")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-probe-inputs", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [int(value) for value in args.rows.split(",")]
    count = manifest["input_rows"]
    if not rows or len(set(rows)) != len(rows) or any(row < 0 or row >= count for row in rows):
        raise ValueError("rows must be distinct indices within input_rows")
    tensors = {item["name"]: item for item in manifest["tensors"]}
    wanted = {f"fp8/model.language_model.layers.{args.layer}.mlp.{part}_proj.{suffix}"
              for part in ("gate", "up") for suffix in ("weight", "weight_scale")}
    weights, sources = {}, {}
    for path in sorted(args.checkpoint.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in wanted.intersection(source.keys()):
                if name in weights:
                    raise ValueError(f"duplicate checkpoint tensor: {name}")
                weights[name] = source.get_tensor(name)
                raw = weights[name].contiguous().view(torch.uint8).numpy().tobytes()
                sources[name] = {"file": str(path), "sha256_tensor_bytes": hashlib.sha256(raw).hexdigest()}
    if weights.keys() != wanted:
        raise ValueError(f"missing checkpoint tensors: {wanted - weights.keys()}")
    prefix = f"fp8/model.language_model.layers.{args.layer}.mlp."
    gate_w, up_w = (weights[prefix + part + "_proj.weight"] for part in ("gate", "up"))
    if gate_w.dtype != torch.float8_e4m3fn or up_w.dtype != torch.float8_e4m3fn or gate_w.shape != up_w.shape:
        raise ValueError("expected matching E4M3FN gate/up matrices")
    n, k = gate_w.shape
    if args.export_probe_inputs:
        directory = args.export_probe_inputs
        directory.mkdir(parents=True, exist_ok=False)
        exports = {}
        for name, tensor_name, width in (("activation", "act.xqh", k), ("activation_scale", "act.ash", 4)):
            item = tensors[tensor_name]
            path = args.capture / item["file"]
            if path.stat().st_size != item["bytes"] or item["bytes"] < count * width:
                raise ValueError(f"invalid allocation size: {path}")
            with path.open("rb") as source:
                raw = source.read(count * width)
            (directory / f"{name}.bin").write_bytes(raw)
            exports[name] = {"source": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for part in ("gate", "up"):
            for suffix, label in (("weight", part), ("weight_scale", part + "_scale")):
                value = weights[prefix + part + "_proj." + suffix]
                if suffix == "weight_scale":
                    if value.numel() != n:
                        raise ValueError("expected one weight scale per output channel")
                    value = value.float()
                raw = value.contiguous().view(torch.uint8).numpy().tobytes()
                (directory / f"{label}.bin").write_bytes(raw)
                exports[label] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        (directory / "manifest.json").write_text(json.dumps({
            "m": count, "n": n, "k": k, "layer": args.layer,
            "capture_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "checkpoint_sources": sources, "files": exports,
        }, indent=2) + "\n")
    captured = {}
    def read(name, dtype, width):
        value = read_rows(args.capture, tensors[name], dtype, width, rows, count)
        captured[name] = {
            "file": str(args.capture / tensors[name]["file"]),
            "dtype": str(dtype), "width": width,
            "sha256_selected_float32": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }
        return value
    x = read("act.xqh", torch.float8_e4m3fn, k)
    scale = read("act.ash", torch.float32, 1).flatten()
    projections = []
    for part, weight in (("gate", gate_w), ("up", up_w)):
        ws = weights[prefix + part + "_proj.weight_scale"].float().flatten()
        if ws.numel() != n:
            raise ValueError("expected one weight scale per output channel")
        projections.append(projection(x, weight.float(), scale, ws))
    reference = glu(*projections)
    report = {
        "scope": "Sampled-row CPU float64 dot products, FP32 scaling, BF16 projection and GLU boundaries. CPU tanh is not a bit-exact CUDA activation oracle. No vLLM parity or full-rung quality claim.",
        "rows": rows, "input_rows": count, "layer": args.layer,
        "capture_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "weights": sources,
        "glu": metrics(read("act.fu", torch.bfloat16, n), reference),
    }
    for name, ref in zip(("act.gt", "act.ut"), projections):
        if name in tensors:
            report[name] = metrics(read(name, torch.bfloat16, n), ref)
    report["captured_rows"] = captured
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key in ("glu", "act.gt", "act.ut")}, indent=2))


if __name__ == "__main__":
    main()
