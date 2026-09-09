#!/usr/bin/env python3
"""Compare plow sparse MLA with AITER on identical BF16 inputs; requires a GPU lease."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics

import aiter
from aiter.mla import mla_decode_fwd
import torch


def bench(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    # Keep capture outputs alive through replay.
    del result
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 128, 8192])
    parser.add_argument("--context", type=int, default=81920)
    args = parser.parse_args()
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        parser.error("the library and reference are qualified for gfx942")
    if any(r <= 0 or args.context - r < 2048 for r in args.rows):
        parser.error("each row count must leave at least 2048 past KV rows")

    lib = ctypes.CDLL(str(args.library.resolve()))
    ptr, uint, flt = ctypes.c_void_p, ctypes.c_uint, ctypes.c_float
    lib.plow_union.argtypes = [ptr] * 4 + [uint] * 3 + [ptr]
    lib.plow_attend.argtypes = [ptr] * 8 + [uint] * 3 + [flt, ptr]
    lib.plow_pack.argtypes = [ptr] * 3 + [uint, ptr]

    def launch(fn, tensors, *scalars):
        status = fn(
            *(t.data_ptr() for t in tensors),
            *scalars,
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"HIP launch failed: {status}")

    metadata = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "aiter": importlib.metadata.version("amd-aiter"),
        "aiter_path": aiter.__file__,
        "gpu": torch.cuda.get_device_name(0),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "context": args.context,
        "top_k": 2048,
        "splits": 2,
        "seed": 17,
    }
    code_object = Path(aiter.__file__).parent.parent / (
        "aiter_meta/hsa/gfx942/mla/mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co"
    )
    if code_object.exists():
        metadata["aiter_object_sha256"] = hashlib.sha256(code_object.read_bytes()).hexdigest()
    records = []
    torch.manual_seed(metadata["seed"])
    for rows in args.rows:
        for overlap in ("shared", "half", "distinct"):
            heads, width, value, ctx, k = 8, 576, 512, args.context, 2048
            live = ctx - rows
            step = next(s for s in range(37, live) if math.gcd(live, s) == 1)
            q = torch.randn(rows, heads, width, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(ctx, 1, 1, width, device="cuda", dtype=torch.bfloat16)
            query = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None]
            slot = torch.arange(k, device="cuda", dtype=torch.int64)[None, :]
            if overlap == "shared":
                slot = slot + query * 0
            elif overlap == "half":
                slot = torch.where(
                    slot < k // 2, slot,
                    k // 2 + (slot - k // 2 + query * (k // 2)) % (live - k // 2),
                )
            else:
                slot = slot + query * k
            idx = ((slot * step) % live).to(torch.int32).contiguous()
            sorted_idx = idx.sort(dim=1).values
            assert (sorted_idx[:, 1:] != sorted_idx[:, :-1]).all()
            qp = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
            kp = qp * k
            last = torch.ones(rows, device="cuda", dtype=torch.int32)
            out = torch.empty(rows, heads, value, device="cuda", dtype=torch.bfloat16)
            qa, qr = q[:, :, :value].contiguous(), q[:, :, value:].contiguous()
            ck = kv.view(ctx, width)[:, :value].contiguous()
            kr = kv.view(ctx, width)[:, value:].contiguous()
            packed_q, packed_kv = torch.empty_like(q), torch.empty_like(kv)
            length = torch.tensor([ctx], device="cuda", dtype=torch.int32)
            tiles, cap = (rows + 7) // 8, min(k * 8, ctx)
            header = (tiles * 4 + 255) // 256 * 256
            uni = torch.empty(header + tiles * cap * 12, device="cuda", dtype=torch.uint8)
            mask = torch.empty(min(tiles, 304) * ctx, device="cuda", dtype=torch.int64)
            part = torch.empty(rows, heads, value, device="cuda", dtype=torch.float32)
            ml = torch.empty(rows, heads, 2, device="cuda", dtype=torch.float32)

            def union():
                launch(lib.plow_union, (uni, mask, idx, length), rows, ctx, cap)

            def plow():
                launch(lib.plow_attend, (part, ml, qa, qr, ck, kr, length, uni),
                       rows, ctx, cap, width**-0.5)
                return part / ml[:, :, 1, None]

            def pack():
                launch(lib.plow_pack, (packed_q, qa, qr), rows * heads)
                launch(lib.plow_pack, (packed_kv, ck, kr), ctx)

            def reference():
                return mla_decode_fwd(
                    packed_q, packed_kv, out, qp, kp, idx.flatten(), last, 1,
                    page_size=1, sm_scale=width**-0.5, num_kv_splits=2,
                )

            def adapter():
                pack()
                reference()
                return out.float()

            union()
            actual_plow, actual_aiter = plow(), adapter()
            torch.cuda.synchronize()
            assert torch.equal(packed_q, q) and torch.equal(packed_kv, kv)
            picks = list(dict.fromkeys([0, rows // 2, rows - 1]))
            selected = kv.view(ctx, width)[idx[picks].long()].float()
            scores = torch.einsum("bhd,bkd->bhk", q[picks].float(), selected) * width**-0.5
            gold = torch.einsum("bhk,bkd->bhd", scores.softmax(-1), selected[:, :, :value])
            errors = {}
            for name, actual in (("plow", actual_plow), ("aiter", actual_aiter)):
                delta = actual[picks] - gold
                rel = (delta.square().sum() / gold.square().sum()).sqrt().item()
                assert torch.isfinite(actual).all() and rel < 0.02, (name, rows, overlap, rel)
                errors[name] = rel
            record = {
                "rows": rows,
                "overlap": overlap,
                "union_mean": uni[:tiles * 4].view(torch.int32).float().mean().item(),
                "relative_l2": errors,
                "union_ms": bench(union),
                "plow_ms": bench(plow),
                "pack_ms": bench(pack),
                "aiter_ms": bench(reference),
                "adapter_ms": bench(adapter),
            }
            print(json.dumps(record), flush=True)
            records.append(record)
            args.out.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
