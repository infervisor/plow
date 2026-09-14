#!/usr/bin/env python3
"""Assemble a vLLM-servable FP8 checkpoint from plow's `fp8/`-keyed PTPC export.

The export holds only the projection matrices, each keyed `fp8/<name>` with an
`fp8/<name>_scale` twin, because that is the contract plowc/plowrt read. vLLM needs
the bare `<name>` / `<name>.weight_scale` spelling, plus every tensor the export does
not carry (embeddings, norms, layer scalars, vision tower). Nothing is requantized
here: shard 1 is a byte-for-byte copy of the export's data region under a rewritten
header, and shard 2 copies the remaining BF16 tensors out of the source checkpoint.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys

HDR_ALIGN = 8


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def dumps_header(header):
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(blob)) % HDR_ALIGN
    return blob + b" " * pad


def copy_range(src, dst, offset, length, chunk=64 << 20):
    src.seek(offset)
    remaining = length
    while remaining:
        block = src.read(min(chunk, remaining))
        if not block:
            raise EOFError(f"short read at {offset}, {remaining} bytes left")
        dst.write(block)
        remaining -= len(block)


def write_fp8_shard(export, out):
    """Rewrite the export header without the `fp8/` prefix; copy the data region once."""
    header, data_start = read_header(export)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    order = sorted(tensors, key=lambda k: tensors[k]["data_offsets"][0])
    span = tensors[order[-1]]["data_offsets"][1]
    if tensors[order[0]]["data_offsets"][0] != 0:
        raise ValueError("export data region does not start at 0")
    for lo, hi in zip(order, order[1:]):
        if tensors[lo]["data_offsets"][1] != tensors[hi]["data_offsets"][0]:
            raise ValueError(f"export has a hole between {lo} and {hi}")

    new = {}
    for name in order:
        if not name.startswith("fp8/"):
            raise ValueError(f"unexpected unprefixed tensor {name}")
        # The twin key is `fp8/<module>.weight` + `fp8/<module>.weight_scale`, so
        # stripping the prefix already yields the HuggingFace spelling for both.
        bare = name[len("fp8/"):]
        info = dict(tensors[name])
        if bare.endswith(".weight_scale"):
            if len(info["shape"]) != 1:
                raise ValueError(f"{name}: expected a one-per-channel scale row")
            # vLLM's ChannelQuantScaleParameter is (out_features, 1), which is also
            # what llm-compressor writes. The bytes are the same either way; only
            # the declared shape moves, so this stays a header-only rewrite.
            info["shape"] = [info["shape"][0], 1]
        elif not bare.endswith(".weight"):
            raise ValueError(f"unexpected twin tensor {name}")
        new[bare] = info
    if len(new) != len(tensors):
        raise ValueError("renaming collapsed two tensors onto one name")

    blob = dumps_header(new)
    with open(export, "rb") as src, open(out, "wb") as dst:
        dst.write(struct.pack("<Q", len(blob)))
        dst.write(blob)
        copy_range(src, dst, data_start, span)
    return new, span


def write_bf16_shard(source_dir, index, names, out):
    """Copy the tensors the export does not carry, in source-file/offset order."""
    headers = {}
    for shard in sorted(set(index[n] for n in names)):
        headers[shard] = read_header(os.path.join(source_dir, shard))

    def key(name):
        shard = index[name]
        return (shard, headers[shard][0][name]["data_offsets"][0])

    new = {}
    cursor = 0
    plan = []
    for name in sorted(names, key=key):
        shard = index[name]
        info = headers[shard][0][name]
        lo, hi = info["data_offsets"]
        length = hi - lo
        new[name] = {"dtype": info["dtype"], "shape": info["shape"],
                     "data_offsets": [cursor, cursor + length]}
        plan.append((shard, headers[shard][1] + lo, length))
        cursor += length

    blob = dumps_header(new)
    handles = {}
    try:
        with open(out, "wb") as dst:
            dst.write(struct.pack("<Q", len(blob)))
            dst.write(blob)
            for shard, offset, length in plan:
                if shard not in handles:
                    handles[shard] = open(os.path.join(source_dir, shard), "rb")
                copy_range(handles[shard], dst, offset, length)
    finally:
        for handle in handles.values():
            handle.close()
    return new, cursor


QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "actorder": None, "block_structure": None, "dynamic": True,
                "group_size": None, "num_bits": 8, "observer": None,
                "observer_kwargs": {}, "strategy": "token", "symmetric": True,
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None, "block_structure": None, "dynamic": False,
                "group_size": None, "num_bits": 8, "observer": "minmax",
                "observer_kwargs": {}, "strategy": "channel", "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "float-quantized",
    "global_compression_ratio": None,
    "ignore": [
        "lm_head",
        "re:.*vision_tower.*",
        "re:.*embed_vision.*",
        "re:.*audio_tower.*",
        "re:.*embed_audio.*",
    ],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}

SIDECARS = ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
            "chat_template.jinja", "processor_config.json", "preprocessor_config.json",
            "special_tokens_map.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="plow fp8 export directory")
    ap.add_argument("--source", required=True, help="original BF16 checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight-only", action="store_true",
                    help="keep BF16 activations (W8A16) instead of dynamic FP8 activations")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out):
        if not args.force:
            sys.exit(f"{args.out} exists; pass --force to rebuild")
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    quantization = json.load(open(os.path.join(args.export, "quantization.json")))
    if not quantization.get("complete"):
        sys.exit("export is not marked complete")
    exported = set(quantization["sources"])

    index_path = os.path.join(args.source, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
    else:
        header, _ = read_header(os.path.join(args.source, "model.safetensors"))
        weight_map = {name: "model.safetensors" for name in header
                      if name != "__metadata__"}
    missing = exported - set(weight_map)
    if missing:
        sys.exit(f"export names absent from the source index: {sorted(missing)[:3]}")
    remainder = sorted(set(weight_map) - exported)

    fp8_shard = "model-00001-of-00002.safetensors"
    bf16_shard = "model-00002-of-00002.safetensors"
    print(f"fp8 shard: {len(exported)} projections + scales", flush=True)
    fp8_map, fp8_bytes = write_fp8_shard(
        os.path.join(args.export, "model.safetensors"), os.path.join(args.out, fp8_shard))
    print(f"  wrote {fp8_bytes/2**30:.1f} GiB, {len(fp8_map)} tensors", flush=True)
    print(f"bf16 shard: {len(remainder)} tensors", flush=True)
    bf16_map, bf16_bytes = write_bf16_shard(
        args.source, weight_map, remainder, os.path.join(args.out, bf16_shard))
    print(f"  wrote {bf16_bytes/2**30:.1f} GiB, {len(bf16_map)} tensors", flush=True)

    new_index = {
        "metadata": {"total_size": fp8_bytes + bf16_bytes},
        "weight_map": {**{n: fp8_shard for n in fp8_map},
                       **{n: bf16_shard for n in bf16_map}},
    }
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    config = json.load(open(os.path.join(args.source, "config.json")))
    quant_config = json.loads(json.dumps(QUANT_CONFIG))
    if args.weight_only:
        quant_config["config_groups"]["group_0"]["input_activations"] = None
    config["quantization_config"] = quant_config
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for name in SIDECARS:
        src = os.path.join(args.source, name)
        if os.path.exists(src):
            os.symlink(os.path.abspath(src), os.path.join(args.out, name))

    provenance = {
        "built_by": os.path.basename(__file__),
        "export": os.path.abspath(args.export),
        "export_quantization_sha256": hashlib.sha256(
            open(os.path.join(args.export, "quantization.json"), "rb").read()).hexdigest(),
        "source": os.path.abspath(args.source),
        "scale_mode": quantization["scale_mode"],
        "weight_dtype": quantization["weight_dtype"],
        "activation_scheme": "bf16" if args.weight_only else "dynamic-token-fp8",
        "renaming": "fp8/<name> -> <name>; fp8/<name>_scale -> <name>.weight_scale",
        "requantized": False,
        "note": "shard 1 data region is a byte-for-byte copy of the export; only the "
                "safetensors header was rewritten.",
    }
    with open(os.path.join(args.out, "plow-provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
