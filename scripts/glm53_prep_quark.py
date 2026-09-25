#!/usr/bin/env python3
"""Prepare GLM-5.3 Quark MXFP4/attention-FP8 for Plow without copying MXFP4 weights."""

import argparse
import hashlib
import json
import mmap
import os
import struct

import torch

import glm52_prep as P
import glm52_prep_fp8_linear as F


ATTN_PROJS = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
)
DERIVED = (
    "q_a_proj.weight",
    "derived.q_absorb.weight",
    "derived.q_rope.weight",
    "derived.kv_a_latent.weight",
    "derived.k_rope.weight",
    "derived.v_absorb.weight",
    "o_proj.weight",
)


def layers_arg(value):
    layers = set()
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            layers.update(range(first, last + 1))
        else:
            layers.add(int(part))
    assert layers and min(layers) >= 0 and max(layers) < 78
    return sorted(layers)


def config(model):
    with open(os.path.join(model, "config.json")) as source:
        cfg = json.load(source)
    q = cfg["quantization_config"]
    assert cfg["model_type"] == "glm_moe_dsa" and cfg["num_hidden_layers"] == 78
    assert q["quant_method"] == "quark"
    global_weight = q["global_quant_config"]["weight"]
    assert (global_weight["dtype"], global_weight["group_size"], global_weight["scale_format"]) == (
        "fp4", 32, "e8m0"
    )
    return cfg


def require_fp8(cfg, layer, proj):
    name = f"model.layers.{layer}.self_attn.{proj}"
    weight = cfg["quantization_config"]["layer_quant_config"][name]["weight"]
    assert weight["dtype"] == "fp8_e4m3" and weight["block_size"] == [128, 128], name


def header(path):
    try:
        with open(path, "rb") as shard:
            size = struct.unpack("<Q", shard.read(8))[0]
            tensors = json.loads(shard.read(size))
        tensors.pop("__metadata__", None)
        end = 0
        for record in sorted(tensors.values(), key=lambda v: v["data_offsets"][0]):
            start, next_end = record["data_offsets"]
            if start != end or next_end < start:
                return None
            end = next_end
        return tensors if os.path.getsize(path) == 8 + size + end else None
    except (OSError, ValueError, KeyError, struct.error):
        return None


def expected_layout(cfg, raw, layer):
    prefix = f"model.layers.{layer}.self_attn."
    h, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    dk, dr, vd, ql = cfg["kv_lora_rank"], cfg["qk_rope_head_dim"], cfg["v_head_dim"], cfg["q_lora_rank"]
    layout = dict(zip((prefix + suffix for suffix in DERIVED), (
        ("BF16", [ql, h]),
        ("BF16", [nh * dk, ql]),
        ("BF16", [nh * dr, ql]),
        ("BF16", [dk, h]),
        ("BF16", [dr, h]),
        ("BF16", [nh * dk, vd]),
        ("BF16", [h, nh * vd]),
    )))
    if cfg["indexer_types"][layer] == "full":
        for proj in ("wq_b", "wk"):
            name = prefix + f"indexer.{proj}.weight"
            layout[name + "_scale_inv"] = ("F32", raw[name + "_scale"][4])
    return layout


def valid_sidecar(path, layout):
    found = header(path)
    return found is not None and set(found) == set(layout) and all(
        (record["dtype"], record["shape"]) == layout[name]
        for name, record in found.items()
    )


def verify_values(raw, cfg, layer, path, aliases=False):
    writer = P.STWriter()
    if aliases:
        build_oproj_fp8_alias(raw, cfg, writer, layer)
    else:
        build_layer(raw, cfg, writer, layer)

    class HashSink:
        def __init__(self):
            self.digest = hashlib.sha256()

        def write(self, data):
            self.digest.update(data)

    records = header(path)
    with open(path, "rb") as shard:
        mapped = mmap.mmap(shard.fileno(), 0, access=mmap.ACCESS_READ)
        base = 8 + struct.unpack("<Q", mapped[:8])[0]
        for name, _, _, size, producer in writer.entries:
            sink = HashSink()
            producer(sink)
            start, end = records[name]["data_offsets"]
            assert end - start == size
            actual = hashlib.sha256(memoryview(mapped)[base + start:base + end]).digest()
            assert actual == sink.digest.digest(), f"layer {layer}: tensor bytes differ: {name}"
        mapped.close()


def add_bf16(writer, name, value):
    writer.add(name, "BF16", value.shape, value.numel() * 2, P.p_bf16(value))


def oproj_fp8_layout(raw, cfg, layer):
    require_fp8(cfg, layer, "o_proj")
    source = f"model.layers.{layer}.self_attn.o_proj.weight"
    weight = raw[source]
    scale = raw[source + "_scale"]
    assert weight[3] == "F8_E4M3" and scale[3] == "F32"
    return {
        source + "_fp8": (weight[3], weight[4]),
        source + "_scale_inv": (scale[3], scale[4]),
    }


def build_oproj_fp8_alias(raw, cfg, writer, layer):
    layout = oproj_fp8_layout(raw, cfg, layer)
    source = f"model.layers.{layer}.self_attn.o_proj.weight"
    for alias, original in ((source + "_fp8", source),
                            (source + "_scale_inv", source + "_scale")):
        dtype, shape, size, producer = P.p_raw(raw, original)
        assert (dtype, shape) == layout[alias]
        writer.add(alias, dtype, shape, size, producer)


def qkva_fp8_entries(raw, cfg, layer):
    for proj in ("q_a_proj", "kv_a_proj_with_mqa"):
        require_fp8(cfg, layer, proj)
    prefix = f"model.layers.{layer}.self_attn."
    aliased = raw.copy()
    for proj in ("q_a_proj", "kv_a_proj_with_mqa"):
        weight = prefix + proj + ".weight"
        aliased[weight + "_scale_inv"] = raw[weight + "_scale"]
    return F.qkva_entries(aliased, cfg, layer)


def verify_qkva_values(path, entries):
    records = header(path)
    with open(path, "rb") as shard:
        mapped = mmap.mmap(shard.fileno(), 0, access=mmap.ACCESS_READ)
        base = 8 + struct.unpack("<Q", mapped[:8])[0]
        for name, parts in entries:
            digest = hashlib.sha256()
            for source, start, end, _, _ in parts:
                digest.update(F.mm(source)[start:end])
            start, end = records[name]["data_offsets"]
            assert hashlib.sha256(memoryview(mapped)[base + start:base + end]).digest() == digest.digest(), name
        mapped.close()


def build_layer(raw, cfg, writer, layer):
    h, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    dk, dr, qn = cfg["kv_lora_rank"], cfg["qk_rope_head_dim"], cfg["qk_nope_head_dim"]
    vd, ql = cfg["v_head_dim"], cfg["q_lora_rank"]
    prefix = f"model.layers.{layer}.self_attn."
    aliased = raw.copy()
    for proj in ATTN_PROJS:
        require_fp8(cfg, layer, proj)
        weight = prefix + proj + ".weight"
        assert raw[weight][3] == "F8_E4M3" and raw[weight + "_scale"][3] == "F32"
        aliased[weight + "_scale_inv"] = raw[weight + "_scale"]

    q_b = P.dequant_blockfp8(aliased, prefix + "q_b_proj.weight").view(nh, qn + dr, ql)
    kv_b = P.dequant_blockfp8(aliased, prefix + "kv_b_proj.weight").view(nh, qn + vd, dk)
    q_nope, q_rope = q_b[:, :qn, :], q_b[:, qn:, :]
    k_nope, values = kv_b[:, :qn, :], kv_b[:, qn:, :]
    q_absorb = torch.einsum("hpl,hpk->hlk", k_nope, q_nope).contiguous()
    v_absorb = values.transpose(-1, -2).contiguous()
    kv_a = P.dequant_blockfp8(aliased, prefix + "kv_a_proj_with_mqa.weight")
    assert kv_a.shape == (dk + dr, h)

    add_bf16(writer, prefix + "q_a_proj.weight", P.dequant_blockfp8(aliased, prefix + "q_a_proj.weight"))
    add_bf16(writer, prefix + "derived.q_absorb.weight", q_absorb.reshape(nh * dk, ql))
    add_bf16(writer, prefix + "derived.q_rope.weight", q_rope.reshape(nh * dr, ql))
    add_bf16(writer, prefix + "derived.kv_a_latent.weight", kv_a[:dk])
    add_bf16(writer, prefix + "derived.k_rope.weight", kv_a[dk:])
    add_bf16(writer, prefix + "derived.v_absorb.weight", v_absorb.reshape(nh * dk, vd))
    add_bf16(writer, prefix + "o_proj.weight", P.dequant_blockfp8(aliased, prefix + "o_proj.weight"))

    if cfg["indexer_types"][layer] == "full":
        for proj in ("wq_b", "wk"):
            require_fp8(cfg, layer, "indexer." + proj)
            name = prefix + f"indexer.{proj}.weight"
            assert raw[name][3] == "F8_E4M3"
            dtype, shape, size, producer = P.p_raw(raw, name + "_scale")
            assert dtype == "F32"
            writer.add(name + "_scale_inv", dtype, shape, size, producer)


def link_snapshot(model, out):
    assert os.path.realpath(model) != os.path.realpath(out)
    os.makedirs(out, exist_ok=True)
    for filename in os.listdir(model):
        if not (filename.endswith((".safetensors", ".json", ".jinja")) or filename == "LICENSE"):
            continue
        destination = os.path.join(out, filename)
        if not os.path.lexists(destination):
            os.symlink(os.path.join(model, filename), destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="0-77")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-values", action="store_true")
    parser.add_argument("--fp8-oproj-alias", action="store_true")
    parser.add_argument("--fp8-qkva-alias", action="store_true")
    args = parser.parse_args()
    cfg = config(args.model)
    layers = layers_arg(args.layers)
    if not args.verify_only:
        link_snapshot(args.model, args.out)
    raw = P._index_shards(args.model)
    for layer in layers:
        path = os.path.join(args.out, f"zz-quark-mla-{layer:05d}.safetensors")
        expected = expected_layout(cfg, raw, layer)
        if valid_sidecar(path, expected):
            if args.verify_values:
                verify_values(raw, cfg, layer, path)
            print(f"[quark] layer {layer}: verified existing sidecar", flush=True)
        else:
            assert not args.verify_only, f"layer {layer}: missing or incomplete sidecar"
            writer = P.STWriter()
            build_layer(raw, cfg, writer, layer)
            assert {entry[0] for entry in writer.entries} == set(expected)
            temporary = f"{path}.{os.getpid()}.tmp"
            writer.flush(temporary)
            os.replace(temporary, path)
            assert valid_sidecar(path, expected), f"layer {layer}: invalid sidecar"
            if args.verify_values:
                verify_values(raw, cfg, layer, path)
            print(f"[quark] layer {layer}: {len(expected)} tensors, {os.path.getsize(path)} bytes", flush=True)
        if args.fp8_oproj_alias:
            alias_path = os.path.join(args.out, f"zz-quark-oproj-fp8-{layer:05d}.safetensors")
            alias_layout = oproj_fp8_layout(raw, cfg, layer)
            if not valid_sidecar(alias_path, alias_layout):
                assert not args.verify_only, f"layer {layer}: missing FP8 o-projection aliases"
                aliases = P.STWriter()
                build_oproj_fp8_alias(raw, cfg, aliases, layer)
                temporary = f"{alias_path}.{os.getpid()}.tmp"
                aliases.flush(temporary)
                os.replace(temporary, alias_path)
            assert valid_sidecar(alias_path, alias_layout)
            if args.verify_values:
                verify_values(raw, cfg, layer, alias_path, aliases=True)
            print(f"[quark] layer {layer}: verified FP8 o-projection aliases", flush=True)
        if args.fp8_qkva_alias:
            alias_path = os.path.join(args.out, f"zz-quark-qkva-fp8-{layer:05d}.safetensors")
            entries = qkva_fp8_entries(raw, cfg, layer)
            if not F.shard_ok(alias_path, entries):
                assert not args.verify_only, f"layer {layer}: missing FP8 QKV-A aliases"
                F.write_shard(alias_path, entries)
            assert F.shard_ok(alias_path, entries)
            if args.verify_values:
                verify_qkva_values(alias_path, entries)
            print(f"[quark] layer {layer}: verified FP8 QKV-A aliases", flush=True)


if __name__ == "__main__":
    main()
