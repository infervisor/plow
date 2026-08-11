#!/usr/bin/env python3
"""Create the absorbed MLA-weight sidecar required by the Kimi-K3 DSpark block emitter."""

import argparse
import json
import mmap
import os
import struct
import sys

import numpy as np


SIDECAR = "model-idx-derived-dspark.safetensors"


class Checkpoint:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "rb")
        n = struct.unpack("<Q", self.file.read(8))[0]
        self.header = json.loads(self.file.read(n))
        self.data0 = 8 + n
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)

    def load_bf16(self, name, shape):
        ent = self.header.get(name)
        if ent is None:
            raise SystemExit(f"checkpoint has no {name}")
        if ent["dtype"] != "BF16" or ent["shape"] != list(shape):
            raise SystemExit(
                f"{name}: expected BF16 {list(shape)}, got {ent['dtype']} {ent['shape']}"
            )
        lo, hi = ent["data_offsets"]
        raw = np.frombuffer(
            self.mm, dtype="<u2", count=(hi - lo) // 2, offset=self.data0 + lo
        )
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def bf16_bytes(values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise SystemExit("derived tensor contains non-finite values")
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2", copy=False).tobytes()


def layer_entries(cfg, layer):
    h = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    dk = cfg["kv_lora_rank"]
    dr = cfg["qk_rope_head_dim"]
    vd = cfg["v_head_dim"]
    ql = cfg["q_lora_rank"]
    p = f"layers.{layer}.self_attn."
    return [
        (p + "derived.q_absorb.weight", [nh * dk, ql]),
        (p + "derived.q_rope.weight", [nh * dr, ql]),
        (p + "derived.kv_a_latent.weight", [dk, h]),
        (p + "derived.k_rope.weight", [dr, h]),
        (p + "derived.v_absorb.weight", [nh * dk, vd]),
    ]


def derived_layer(checkpoint, cfg, layer):
    h = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    dk = cfg["kv_lora_rank"]
    dr = cfg["qk_rope_head_dim"]
    qn = cfg["qk_nope_head_dim"]
    vd = cfg["v_head_dim"]
    ql = cfg["q_lora_rank"]
    p = f"layers.{layer}.self_attn."

    q_b = checkpoint.load_bf16(p + "q_b_proj.weight", [nh * (qn + dr), ql])
    q_b = q_b.reshape(nh, qn + dr, ql)
    kv_b = checkpoint.load_bf16(p + "kv_b_proj.weight", [nh * (qn + vd), dk])
    kv_b = kv_b.reshape(nh, qn + vd, dk)
    kv_a = checkpoint.load_bf16(p + "kv_a_proj_with_mqa.weight", [dk + dr, h])

    q_absorb = np.empty((nh, dk, ql), dtype=np.float32)
    for head in range(nh):
        np.matmul(
            kv_b[head, :qn].T,
            q_b[head, :qn],
            out=q_absorb[head],
        )
    return [
        q_absorb.reshape(nh * dk, ql),
        np.ascontiguousarray(q_b[:, qn:]).reshape(nh * dr, ql),
        kv_a[:dk],
        kv_a[dk : dk + dr],
        np.ascontiguousarray(kv_b[:, qn:].transpose(0, 2, 1)).reshape(nh * dk, vd),
    ]


def write_sidecar(path, checkpoint, cfg, layers):
    entries = [entry for layer in layers for entry in layer_entries(cfg, layer)]
    header = {}
    offset = 0
    for name, shape in entries:
        nbytes = int(np.prod(shape)) * 2
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-((8 + len(blob)) % 8) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for layer in layers:
            tensors = derived_layer(checkpoint, cfg, layer)
            for (name, shape), tensor in zip(layer_entries(cfg, layer), tensors):
                if list(tensor.shape) != shape:
                    raise SystemExit(f"{name}: derived {list(tensor.shape)}, expected {shape}")
                out.write(bf16_bytes(tensor))
                print(f"  {name} {shape}", flush=True)
    return offset, len(entries)


def link_input(src, dst):
    if os.path.lexists(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            return
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            raise SystemExit(f"refusing to replace non-symlink {dst}")
    os.symlink(os.path.realpath(src), dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Inferact/Kimi-K3-DSpark checkpoint")
    parser.add_argument("--out", required=True, help="Plow checkpoint farm")
    parser.add_argument(
        "--layers",
        default="0",
        help="comma-separated draft layers, or 'all' (default: 0)",
    )
    args = parser.parse_args()

    model = os.path.realpath(args.model)
    out = os.path.realpath(args.out)
    if model == out:
        raise SystemExit("--out must differ from --model")
    with open(os.path.join(model, "config.json"), "rb") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "k3_dspark":
        raise SystemExit(f"expected model_type=k3_dspark, got {cfg.get('model_type')!r}")
    n_layers = cfg["num_hidden_layers"]
    layers = (
        list(range(n_layers))
        if args.layers == "all"
        else sorted({int(value) for value in args.layers.split(",")})
    )
    if not layers or layers[0] < 0 or layers[-1] >= n_layers:
        raise SystemExit(f"--layers must be within 0..{n_layers - 1}")

    os.makedirs(out, exist_ok=True)
    link_input(os.path.join(model, "config.json"), os.path.join(out, "config.json"))
    link_input(
        os.path.join(model, "model.safetensors"),
        os.path.join(out, "model.safetensors"),
    )
    checkpoint = Checkpoint(os.path.join(model, "model.safetensors"))
    sidecar = os.path.join(out, SIDECAR)
    nbytes, count = write_sidecar(sidecar, checkpoint, cfg, layers)
    print(
        f"wrote {sidecar}: {count} BF16 tensors, {nbytes / (1024 ** 2):.2f} MiB; "
        f"farm={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
