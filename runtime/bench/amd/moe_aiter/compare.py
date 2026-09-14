#!/usr/bin/env python3
"""Matched GLM TP8 grouped MoE screening; run inside a GPU lease."""

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path

import aiter
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight
import torch

spec = importlib.util.spec_from_file_location(
    "mla_compare", Path(__file__).resolve().parent.parent / "mla_sparse_aiter/compare.py"
)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("plow", "aiter", "ck"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 2048, 8192])
    args = parser.parse_args()
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        parser.error("this comparison is qualified for gfx942")
    if any(rows <= 0 for rows in args.rows):
        parser.error("--rows must be positive")
    moe_module = importlib.import_module("aiter.fused_moe")
    if args.arm == "ck":
        # Force the installed wheel's two-stage dispatch before its first cached lookup.
        moe_module.fused_moe_1stage_dict["gfx942"] = {}
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.plow_moe.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_uint, ctypes.c_void_p]
    lib.plow_moe.restype = ctypes.c_int
    metadata = {
        "arm": args.arm,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "aiter": importlib.metadata.version("amd-aiter"),
        "aiter_dispatch_sha256": hashlib.sha256(Path(moe_module.__file__).read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0),
        "seed": 31,
        "oracle_rows": "first, middle, last",
        "scale_distribution": "independent log2-uniform [-1,1] per 128x128 weight block",
    }
    records = []
    for rows in args.rows:
        torch.manual_seed(metadata["seed"])
        hidden, intermediate, experts, topk = 6144, 256, 256, 8
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randint(-8, 9, (experts, 2*intermediate, hidden), device="cuda",
                           dtype=torch.int8).to(torch.float8_e4m3fnuz)
        w2 = torch.randint(-8, 9, (experts, hidden, intermediate), device="cuda",
                           dtype=torch.int8).to(torch.float8_e4m3fnuz)
        s1 = torch.exp2(torch.rand(experts, 2*intermediate//128, hidden//128,
                                  device="cuda") * 2 - 1) / (5 * math.sqrt(hidden))
        s2 = torch.exp2(torch.rand(experts, hidden//128, intermediate//128,
                                  device="cuda") * 2 - 1) / (5 * math.sqrt(intermediate))
        logits, ids = torch.randn(rows, experts, device="cuda").topk(topk, dim=-1)
        ids, weights = ids.to(torch.int32), logits.softmax(-1)

        if args.arm == "plow":
            # Convert values, not bytes: OCP and FNUZ encode the same number differently.
            wp1, wp2 = w1.float().to(torch.float8_e4m3fn), w2.float().to(torch.float8_e4m3fn)
            wt = torch.tensor([
                [wp1.data_ptr() + e*2*intermediate*hidden,
                 wp1.data_ptr() + (e*2+1)*intermediate*hidden,
                 wp2.data_ptr() + e*hidden*intermediate] for e in range(experts)
            ], device="cuda", dtype=torch.int64)
            scale_stride = intermediate//128 * (hidden//128) * 4
            st = torch.tensor([
                [s1.data_ptr() + e*2*scale_stride, s1.data_ptr() + (e*2+1)*scale_stride,
                 s2.data_ptr() + e*scale_stride] for e in range(experts)
            ], device="cuda", dtype=torch.int64)
            table = torch.empty(rows, topk, 2, device="cuda", dtype=torch.int32)
            table[:, :, 0], table[:, :, 1] = ids, weights.view(torch.int32)
            capacity = rows*topk + experts*63
            meta = torch.empty(3*experts+1+64*experts, device="cuda", dtype=torch.int32)
            rt = torch.empty(capacity, device="cuda", dtype=torch.int32)
            rp, rg = torch.empty_like(rt), torch.empty(capacity, device="cuda")
            fu = torch.empty(capacity, intermediate, device="cuda", dtype=torch.bfloat16)
            part = torch.empty(rows, hidden, device="cuda", dtype=torch.float64)
            out = torch.empty_like(x)

            def run():
                status = lib.plow_moe(
                    *(t.data_ptr() for t in (out, x, wt, st, table, meta, rt, rp, rg, fu, part)),
                    rows, torch.cuda.current_stream().cuda_stream,
                )
                if status:
                    raise RuntimeError(f"HIP launch failed: {status}")
                return out
        else:
            wp1, wp2 = shuffle_weight(w1, (16, 16)), shuffle_weight(w2, (16, 16))

            def run():
                return fused_moe(
                    x, wp1, wp2, weights, ids, quant_type=aiter.QuantType.per_1x128,
                    activation=aiter.ActivationType.Silu, w1_scale=s1, w2_scale=s2,
                )

        out = run()
        torch.cuda.synchronize()
        picks = list(dict.fromkeys([0, rows//2, rows-1]))
        gold = torch.zeros(len(picks), hidden, device="cuda", dtype=torch.float32)
        for j, row in enumerate(picks):
            for slot in range(topk):
                expert = int(ids[row, slot])
                scale1 = s1[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
                scale2 = s2[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
                gate, up = ((w1[expert].float() * scale1) @ x[row].float()).chunk(2)
                activation = torch.nn.functional.silu(gate) * up
                gold[j] += ((w2[expert].float() * scale2) @ activation) * weights[row, slot]
        rel = ((out[picks].float()-gold).square().sum() / gold.square().sum()).sqrt().item()
        # FP8 activation quantization is screened separately from model-level quality.
        threshold = 0.01 if args.arm == "plow" else 0.1
        assert torch.isfinite(out).all() and rel < threshold, (args.arm, rows, rel)
        record = {
            "rows": rows, "hidden": hidden, "intermediate": intermediate,
            "experts": experts, "topk": topk, "relative_l2": rel,
            "ms": timing.bench(run),
        }
        print(json.dumps(record), flush=True)
        records.append(record)
        args.out.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
