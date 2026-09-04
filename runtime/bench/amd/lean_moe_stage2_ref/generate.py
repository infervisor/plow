#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path


SYMBOL = "mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x256x128_vscale_fix3_fp4opt_v1_pm1"
EXACT_SHA256 = "3034c6cf087a0229cd723f226b74df4763f05f0c3cdf07b194bc03649a7899f5"


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--model-dim", type=int, default=3584)
    p.add_argument("--inter-dim", type=int, default=384)
    p.add_argument("--experts", type=int, default=896)
    p.add_argument("--tile-m", type=int, default=32)
    p.add_argument("--tile-n", type=int, default=256)
    p.add_argument("--tile-k", type=int, default=128)
    return p


def extract_binary(cache_root: Path) -> bytes:
    candidates = sorted(cache_root.rglob("*.pkl"), key=lambda p: p.stat().st_mtime_ns)
    if not candidates:
        raise RuntimeError("FlyDSL did not create a cache artifact")
    with candidates[-1].open("rb") as handle:
        artifact = pickle.load(handle)
    text = artifact._ir_text
    start = text.index("bin = \"") + len("bin = \"")
    end = text.index('"', start)
    encoded = text[start:end]
    result = bytearray()
    i = 0
    escapes = {"n": 10, "r": 13, "t": 9, "\\": 92, '"': 34}
    while i < len(encoded):
        if encoded[i] == "\\" and i + 2 < len(encoded) and all(
            c in "0123456789abcdefABCDEF" for c in encoded[i + 1 : i + 3]
        ):
            result.append(int(encoded[i + 1 : i + 3], 16))
            i += 3
        elif encoded[i] == "\\" and i + 1 < len(encoded):
            result.append(escapes[encoded[i + 1]])
            i += 2
        else:
            result.extend(encoded[i].encode())
            i += 1
    return bytes(result)


def main():
    args = parser().parse_args()
    if (args.tile_m, args.tile_n, args.tile_k) != (32, 256, 128):
        raise SystemExit("phase-1 schedule requires tile 32x256x128")
    if args.model_dim % args.tile_n or args.inter_dim % args.tile_k:
        raise SystemExit("model/inter dimensions must be divisible by tile N/K")

    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "flydsl-cache"
    cache.mkdir(exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(cache)

    import torch
    from aiter.ops.flydsl.moe_kernels import (
        _run_compiled,
        _s2_args_fp4,
        compile_flydsl_moe_stage2,
    )
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "") or torch.cuda.get_device_name(0)
    if "gfx950" not in arch:
        raise SystemExit(f"phase-1 object requires gfx950, found {arch}")

    exe = compile_flydsl_moe_stage2(
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        doweight_stage2=True,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="bf16",
        accumulate=True,
        persist_m=1,
        sort_block_m=args.tile_m,
        b_nt=2,
    )

    dev = torch.device("cuda")
    rows = args.tile_m
    out = torch.zeros((args.tokens, args.model_dim), dtype=torch.bfloat16, device=dev)
    act = torch.zeros((args.tokens, args.topk, args.inter_dim // 2), dtype=torch.uint8, device=dev)
    weight = shuffle_weight(
        torch.zeros((args.experts, args.model_dim, args.inter_dim // 2), dtype=torch.uint8, device=dev),
        layout=(16, 16),
    )
    act_scale = shuffle_scale(
        torch.full((rows, args.inter_dim // 32), 127, dtype=torch.uint8, device=dev)
    )
    weight_scale = shuffle_scale(
        torch.full((args.experts * args.model_dim, args.inter_dim // 32), 127, dtype=torch.uint8, device=dev)
    )
    sorted_ids = torch.arange(rows, dtype=torch.int32, device=dev)
    expert_ids = torch.zeros(1, dtype=torch.int32, device=dev)
    gates = torch.ones(rows, dtype=torch.float32, device=dev)
    num_valid = torch.tensor([rows], dtype=torch.int32, device=dev)
    bias = torch.empty(0, dtype=torch.float32, device=dev)
    launch_args = _s2_args_fp4(
        out, act, weight, act_scale, weight_scale, sorted_ids, expert_ids, gates,
        num_valid, args.tokens, args.model_dim, args.inter_dim, 1, dev, bias=bias,
    )
    _run_compiled(exe, launch_args)
    torch.cuda.synchronize()

    binary = extract_binary(cache)
    digest = hashlib.sha256(binary).hexdigest()
    exact_geometry = (
        args.tokens, args.model_dim, args.inter_dim, args.experts, args.topk
    ) == (1024, 3584, 384, 896, 16)
    if exact_geometry and digest != EXACT_SHA256:
        raise SystemExit(f"reproducibility failure: expected {EXACT_SHA256}, got {digest}")
    (args.output / "kernel.co").write_bytes(binary)

    manifest = {
        "schema": 1,
        "implementation": "reference-aiter-generated",
        "status": "gate-only-not-routed",
        "generator": {
            "source": "AITER 0.1.19 / FlyDSL 0.2.4 (MIT)",
            "image": "vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032",
        },
        "object": {"file": "kernel.co", "sha256": digest, "symbol": SYMBOL},
        "capability": {"arch": "gfx950", "wavefront": 64, "workgroup": 256},
        "geometry": {
            "tokens": args.tokens, "topk": args.topk, "model_dim": args.model_dim,
            "inter_dim": args.inter_dim, "experts": args.experts,
            "tile_m": args.tile_m, "tile_n": args.tile_n, "tile_k": args.tile_k,
            "sort_block_m": args.tile_m,
        },
        "encoding": {
            "activation": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
            "weight": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
            "output": "bf16-atomic-accumulate",
            "weight_layout": "E,N/16,Kbytes/32,2,16,16-permute-0,1,3,4,2,5",
            "scale_layout": "pad256x8-view-sm/32,2,16,sn/8,2,4-permute-0,3,5,2,4,1",
        },
        "abi": {
            "kernarg_bytes": 96,
            "arguments": [
                "out*", "activation*", "weight*", "activation_scale*", "weight_scale*",
                "sorted_token_ids*", "expert_ids*", "sorted_weights*", "num_valid_ids*",
                "bias*", "tokens:i32", "model_dim:i32", "inter_dim:i32", "expert_blocks:i32",
            ],
        },
        "resources": {
            "vgpr": 98, "sgpr": 46, "lds_bytes": 16640,
            "fixed_lds_bytes": 16640, "dynamic_lds_bytes": 0,
            "private_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"object": str(args.output / "kernel.co"), "sha256": digest, "bytes": len(binary)}))


if __name__ == "__main__":
    main()
