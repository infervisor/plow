#!/usr/bin/env python3
"""Seal and transform model-generic MLA boundary captures.

Payloads are raw little-endian dense tensors.  Packing happens before replay,
so its cost is never included in an attention timing interval.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "plow.mla-boundary.v1"
DTYPE_BYTES = {"bf16": 2, "float32": 4}
REQUIRED_SOURCE = {
    "latent.q",
    "latent.kv",
    "rope.k",
    "weight.q_projection",
    "weight.kv_projection",
    "weight.output_projection",
}
RESIDUAL_SEAM_INPUTS = {
    "residual.prefix",
    "residual.delta",
    "residual.ring",
    "weight.residual_norm",
    "weight.residual_projection",
    "weight.post_attention_norm",
}


def product(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return out


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolved(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def checked_tensor(base, row):
    item = dict(row)
    path = resolved(base, item["file"])
    dtype = item["dtype"]
    shape = [int(x) for x in item["shape"]]
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"{item['semantic']}: unsupported dtype {dtype}")
    expected = product(shape) * DTYPE_BYTES[dtype]
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{item['semantic']}: payload is {actual} bytes, expected {expected}")
    actual_hash = digest(path)
    if item.get("sha256", actual_hash) != actual_hash:
        raise ValueError(f"{item['semantic']}: stale payload hash")
    item.update(file=str(path), shape=shape, sha256=actual_hash)
    return item


def validate(data, manifest_path, require_source=False, require_qkv=False):
    if data.get("schema") != SCHEMA:
        raise ValueError(f"unsupported schema {data.get('schema')!r}")
    contract = data["contract"]
    dims = contract["dimensions"]
    for name in ("tokens", "heads", "qk_nope", "qk_rope", "v_head"):
        if int(dims[name]) <= 0:
            raise ValueError(f"dimension {name} must be positive")
    if contract.get("causal") is not True:
        raise ValueError("only causal MLA boundary captures are qualified")
    if contract.get("layout") != "token-head-dense":
        raise ValueError("layout must be token-head-dense")
    tensors = [checked_tensor(manifest_path.parent, x) for x in data["tensors"]]
    keys = [(x["semantic"], x.get("layer", 0), x.get("rank", 0)) for x in tensors]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate semantic/layer/rank tensor")
    semantics = {x["semantic"] for x in tensors}
    if require_source and not REQUIRED_SOURCE <= semantics:
        raise ValueError(f"missing source semantics {sorted(REQUIRED_SOURCE - semantics)}")
    if require_qkv and not {"q", "k", "v"} <= semantics:
        raise ValueError("materialized replay requires q, k, and v")
    by_semantic = {x["semantic"]: x for x in tensors}
    t, h = int(dims["tokens"]), int(dims["heads"])
    nope, rope, dv = int(dims["qk_nope"]), int(dims["qk_rope"]), int(dims["v_head"])
    if require_source:
        q_latent, kv_latent = by_semantic["latent.q"], by_semantic["latent.kv"]
        if len(q_latent["shape"]) != 2 or q_latent["shape"][0] != t:
            raise ValueError("latent.q must have shape [tokens,q_lora]")
        if len(kv_latent["shape"]) != 2 or kv_latent["shape"][0] != t:
            raise ValueError("latent.kv must have shape [tokens,kv_lora]")
        expected = {
            "rope.k": [t, rope],
            "weight.q_projection": [h, nope + rope, q_latent["shape"][1]],
            "weight.kv_projection": [h, nope + dv, kv_latent["shape"][1]],
        }
        output_weight = by_semantic["weight.output_projection"]
        if len(output_weight["shape"]) != 2 or output_weight["shape"][1] != h * dv:
            raise ValueError("weight.output_projection must have shape [hidden,heads*v_head]")
        for semantic, shape in expected.items():
            if by_semantic[semantic]["shape"] != shape:
                raise ValueError(f"{semantic} must have shape {shape}")
    if require_qkv:
        expected = {"q": [t, h, nope + rope], "k": [t, h, nope + rope], "v": [t, h, dv]}
        for semantic, shape in expected.items():
            if by_semantic[semantic]["dtype"] != "bf16" or by_semantic[semantic]["shape"] != shape:
                raise ValueError(f"{semantic} must be bf16 with shape {shape}")
    seam_present = RESIDUAL_SEAM_INPUTS & semantics
    if seam_present:
        missing = RESIDUAL_SEAM_INPUTS - semantics
        if missing:
            raise ValueError(f"incomplete residual seam, missing {sorted(missing)}")
        seam = contract.get("residual_seam")
        if not isinstance(seam, dict):
            raise ValueError("residual seam tensors require contract.residual_seam")
        required_state = {
            "operation", "prefix_delta_rounding", "ring_layout", "num_blocks",
            "block_capacity", "block_write_idx", "score_epsilon",
            "output_norm_epsilon", "output_norm_input",
        }
        if missing_state := required_state - set(seam):
            raise ValueError(f"residual seam state missing {sorted(missing_state)}")
        state = {k: v for k, v in seam.items() if k != "state_sha256"}
        if seam.get("state_sha256") != canonical_hash(state):
            raise ValueError("residual seam state hash is missing or stale")
        if seam["operation"] != "softmax-rms-residual-mix":
            raise ValueError("unsupported residual seam operation")
        if seam["prefix_delta_rounding"] != "add-f32-round-bf16":
            raise ValueError("unsupported prefix/delta rounding")
        if seam["ring_layout"] != "token-block-hidden":
            raise ValueError("unsupported residual ring layout")
        if seam["output_norm_input"] not in {"mixed-f32", "mixed-bf16"}:
            raise ValueError("unsupported residual output norm input")
        num_blocks, capacity = int(seam["num_blocks"]), int(seam["block_capacity"])
        if num_blocks < 0 or capacity < num_blocks:
            raise ValueError("invalid residual block count/capacity")
        hidden = by_semantic["weight.output_projection"]["shape"][0]
        expected_seam = {
            "residual.prefix": [t, hidden],
            "residual.delta": [t, hidden],
            "residual.ring": [t, capacity, hidden],
            "weight.residual_norm": [hidden],
            "weight.residual_projection": [hidden],
            "weight.post_attention_norm": [hidden],
        }
        for semantic, shape in expected_seam.items():
            item = by_semantic[semantic]
            if item["dtype"] != "bf16" or item["shape"] != shape:
                raise ValueError(f"{semantic} must be bf16 with shape {shape}")
    history = data["prompt"]
    token_path = resolved(manifest_path.parent, history["u32le_file"])
    if token_path.stat().st_size % 4:
        raise ValueError("prompt token payload is not u32le")
    token_hash = digest(token_path)
    if history.get("sha256", token_hash) != token_hash:
        raise ValueError("stale prompt token hash")
    return tensors, str(token_path), token_hash


def write_manifest(spec_path, output, require_source=False):
    data = json.loads(spec_path.read_text())
    data["schema"] = SCHEMA
    tensors = [checked_tensor(spec_path.parent, x) for x in data["tensors"]]
    token_path = resolved(spec_path.parent, data["prompt"]["u32le_file"])
    data["prompt"] = {
        "u32le_file": str(token_path),
        "sha256": digest(token_path),
        "tokens": token_path.stat().st_size // 4,
    }
    data["prompt_sha256_u32le"] = data["prompt"]["sha256"]
    seam = data.get("contract", {}).get("residual_seam")
    if seam is not None:
        state = {k: v for k, v in seam.items() if k != "state_sha256"}
        seam["state_sha256"] = canonical_hash(state)
    data["tensors"] = tensors
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    validate(data, output, require_source=require_source)


def row(data, semantic):
    found = [x for x in data["tensors"] if x["semantic"] == semantic]
    if len(found) != 1:
        raise ValueError(f"expected exactly one {semantic}, found {len(found)}")
    return found[0]


def pack_bf16(data, manifest, output_dir):
    tensors, _, _ = validate(data, manifest, require_source=True)
    data["tensors"] = tensors
    dims = data["contract"]["dimensions"]
    t, h = int(dims["tokens"]), int(dims["heads"])
    nope, rope, dv = int(dims["qk_nope"]), int(dims["qk_rope"]), int(dims["v_head"])
    qrow = row(data, "projected.q")
    kvrow = row(data, "projected.kv")
    rrow = row(data, "rope.k")
    expected = {
        "projected.q": [t, h, nope + rope],
        "projected.kv": [t, h, nope + dv],
        "rope.k": [t, rope],
    }
    for item in (qrow, kvrow, rrow):
        if item["dtype"] != "bf16" or item["shape"] != expected[item["semantic"]]:
            raise ValueError(
                f"{item['semantic']}: expected bf16 {expected[item['semantic']]}, "
                f"got {item['dtype']} {item['shape']}"
            )
    qraw = Path(qrow["file"]).read_bytes()
    kvraw = Path(kvrow["file"]).read_bytes()
    rraw = Path(rrow["file"]).read_bytes()
    qstride, kvstride, rstride = (nope + rope) * 2, (nope + dv) * 2, rope * 2
    kparts, vparts = [], []
    for token in range(t):
        rp = rraw[token * rstride : (token + 1) * rstride]
        for head in range(h):
            off = (token * h + head) * kvstride
            kparts.append(kvraw[off : off + nope * 2] + rp)
            vparts.append(kvraw[off + nope * 2 : off + (nope + dv) * 2])
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {"q": qraw, "k": b"".join(kparts), "v": b"".join(vparts)}
    shapes = {"q": [t, h, nope + rope], "k": [t, h, nope + rope], "v": [t, h, dv]}
    base = dict(qrow)
    for semantic, raw in payloads.items():
        path = (output_dir / f"{semantic}.bf16").resolve()
        path.write_bytes(raw)
        item = {
            "semantic": semantic,
            "layer": base.get("layer", 0),
            "rank": base.get("rank", 0),
            "dtype": "bf16",
            "source_dtype": "bf16",
            "shape": shapes[semantic],
            "file": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        data["tensors"] = [x for x in data["tensors"] if x["semantic"] != semantic] + [item]
    data["capture"] = dict(data.get("capture", {}), packing="outside-timed-region")
    out = output_dir / "manifest.json"
    out.write_text(json.dumps(data, indent=2) + "\n")
    validate(data, out, require_source=True, require_qkv=True)
    return out


def replay_materialized(manifest, binary, object_, gpulease, output_dir):
    data = json.loads(manifest.read_text())
    tensors, _, _ = validate(data, manifest, require_source=True, require_qkv=True)
    data["tensors"] = tensors
    dims = data["contract"]["dimensions"]
    dq = int(dims["qk_nope"]) + int(dims["qk_rope"])
    dv = int(dims["v_head"])
    # This is an implementation capability, not a model predicate.
    if (dq, dv) != (192, 128):
        raise ValueError(f"replay object supports (qk,v)=(192,128), got ({dq},{dv})")
    inputs = {x["semantic"]: x for x in tensors if x["semantic"] in {"q", "k", "v"}}
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "attention.output"
    command = [
        str(gpulease), "-n", "1", "mla-boundary-replay", str(binary), str(object_),
        inputs["q"]["file"], inputs["k"]["file"], inputs["v"]["file"], str(prefix),
        str(dims["tokens"]), str(dims["heads"]), str(dq), str(dv),
        str(data["contract"]["softmax_scale"]),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    timing = json.loads(result.stdout.strip().splitlines()[-1])
    manifests = []
    for repeat in range(3):
        payload_path = (output_dir / f"attention.output.repeat-{repeat}.bf16").resolve()
        item = {
            "semantic": "attention.output", "layer": inputs["q"].get("layer", 0),
            "rank": inputs["q"].get("rank", 0), "dtype": "bf16", "source_dtype": "bf16",
            "shape": [int(dims["tokens"]), int(dims["heads"]), dv],
            "file": str(payload_path), "sha256": digest(payload_path),
        }
        arm = dict(data)
        arm["tensors"] = [x for x in tensors if x["semantic"] != "attention.output"] + [item]
        arm["replay"] = {
            "implementation": "plow-aiter-opus-materialized",
            "object_sha256": digest(object_),
            "capture_and_pack_outside_timing": True,
            "adjacent_repeat": repeat,
            **timing,
        }
        out = output_dir / f"manifest.repeat-{repeat}.json"
        out.write_text(json.dumps(arm, indent=2) + "\n")
        validate(arm, out, require_source=True, require_qkv=True)
        manifests.append(out)
    return manifests


def replay_absorbed(manifest, binary, object_, gpulease, output_dir):
    data = json.loads(manifest.read_text())
    tensors, _, _ = validate(data, manifest, require_source=True)
    data["tensors"] = tensors
    dims = data["contract"]["dimensions"]
    by_name = {x["semantic"]: x for x in tensors}
    ql = by_name["latent.q"]["shape"][1]
    kl = by_name["latent.kv"]["shape"][1]
    dq = int(dims["qk_nope"]) + int(dims["qk_rope"])
    capability = (ql, kl, int(dims["qk_rope"]), dq, int(dims["v_head"]))
    if capability != (1536, 512, 64, 192, 128):
        raise ValueError(f"absorbed replay object does not support dimensions {capability}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "attention.output"
    command = [
        str(gpulease), "-n", "1", "mla-boundary-replay-absorbed", str(binary), str(object_),
        by_name["latent.q"]["file"], by_name["latent.kv"]["file"], by_name["rope.k"]["file"],
        by_name["weight.q_projection"]["file"], by_name["weight.kv_projection"]["file"],
        str(prefix), str(dims["tokens"]), str(dims["heads"]), str(ql), str(kl),
        str(dims["qk_rope"]), str(dims["v_head"]), str(data["contract"]["softmax_scale"]),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    timing = json.loads(result.stdout.strip().splitlines()[-1])
    manifests = []
    for repeat in range(3):
        payload_path = (output_dir / f"attention.output.repeat-{repeat}.bf16").resolve()
        item = {
            "semantic": "attention.output", "layer": by_name["latent.q"].get("layer", 0),
            "rank": by_name["latent.q"].get("rank", 0), "dtype": "bf16",
            "source_dtype": "bf16",
            "shape": [int(dims["tokens"]), int(dims["heads"]), int(dims["v_head"])],
            "file": str(payload_path), "sha256": digest(payload_path),
        }
        arm = dict(data)
        arm["tensors"] = [x for x in tensors if x["semantic"] != "attention.output"] + [item]
        arm["replay"] = {
            "implementation": "plow-absorbed-latent",
            "object_sha256": digest(object_),
            "capture_and_weight_derivation_outside_timing": True,
            "adjacent_repeat": repeat,
            **timing,
        }
        out = output_dir / f"manifest.repeat-{repeat}.json"
        out.write_text(json.dumps(arm, indent=2) + "\n")
        validate(arm, out, require_source=True)
        manifests.append(out)
    return manifests


def attach_tensor(manifest, output, semantic, payload_path, dtype, shape, layer, rank):
    data = json.loads(manifest.read_text())
    tensors, _, _ = validate(data, manifest)
    item = checked_tensor(
        Path.cwd(),
        {
            "semantic": semantic,
            "layer": layer,
            "rank": rank,
            "dtype": dtype,
            "source_dtype": dtype,
            "shape": [int(x) for x in shape.split(",") if x],
            "file": str(payload_path.resolve()),
        },
    )
    key = (semantic, layer, rank)
    data["tensors"] = [
        x for x in tensors if (x["semantic"], x.get("layer", 0), x.get("rank", 0)) != key
    ] + [item]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    validate(data, output)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--spec", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--require-source", action="store_true")
    pack = sub.add_parser("pack-materialized")
    pack.add_argument("--manifest", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--require-source", action="store_true")
    check.add_argument("--require-qkv", action="store_true")
    replay = sub.add_parser("replay-materialized")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--binary", type=Path, required=True)
    replay.add_argument("--object", type=Path, required=True)
    replay.add_argument("--gpulease", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    absorbed = sub.add_parser("replay-absorbed")
    absorbed.add_argument("--manifest", type=Path, required=True)
    absorbed.add_argument("--binary", type=Path, required=True)
    absorbed.add_argument("--object", type=Path, required=True)
    absorbed.add_argument("--gpulease", type=Path, required=True)
    absorbed.add_argument("--output-dir", type=Path, required=True)
    attach = sub.add_parser("attach-tensor")
    attach.add_argument("--manifest", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)
    attach.add_argument("--semantic", required=True)
    attach.add_argument("--file", type=Path, required=True)
    attach.add_argument("--dtype", choices=sorted(DTYPE_BYTES), required=True)
    attach.add_argument("--shape", required=True)
    attach.add_argument("--layer", type=int, default=0)
    attach.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    if args.command == "seal":
        write_manifest(args.spec, args.output, args.require_source)
    elif args.command == "pack-materialized":
        data = json.loads(args.manifest.read_text())
        print(pack_bf16(data, args.manifest, args.output_dir))
    elif args.command == "validate":
        data = json.loads(args.manifest.read_text())
        validate(data, args.manifest, args.require_source, args.require_qkv)
    elif args.command == "replay-materialized":
        for path in replay_materialized(
            args.manifest, args.binary, args.object, args.gpulease, args.output_dir
        ):
            print(path)
    elif args.command == "replay-absorbed":
        for path in replay_absorbed(
            args.manifest, args.binary, args.object, args.gpulease, args.output_dir
        ):
            print(path)
    else:
        attach_tensor(
            args.manifest, args.output, args.semantic, args.file, args.dtype,
            args.shape, args.layer, args.rank
        )


if __name__ == "__main__":
    main()
