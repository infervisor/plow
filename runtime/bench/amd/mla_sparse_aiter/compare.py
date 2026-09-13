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

NATIVE_OBJECT_SHA256 = "cd8fa62e18abada15beeeedd49357bcc1e9e2eee7353d038533f30cac93c3607"


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
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--capture", type=Path, help="Directory containing idx/qa/qr/ck/kr/len.bin")
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 128, 8192])
    parser.add_argument("--context", type=int, default=81920)
    parser.add_argument("--scale", type=float, default=576**-0.5)
    args = parser.parse_args()
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        parser.error("the library and reference are qualified for gfx942")
    if any(r <= 0 or args.context - r < 2048 for r in args.rows):
        parser.error("each row count must leave at least 2048 past KV rows")
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive")
    if args.capture and len(args.rows) != 1:
        parser.error("--capture requires a single --rows value")

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
        "scale": args.scale,
        "seed": 17,
    }
    code_object = Path(aiter.__file__).parent.parent / (
        "aiter_meta/hsa/gfx942/mla/mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co"
    )
    if code_object.exists():
        metadata["aiter_object_sha256"] = hashlib.sha256(code_object.read_bytes()).hexdigest()
    native = None
    if args.native_library:
        if metadata.get("aiter_object_sha256") != NATIVE_OBJECT_SHA256:
            parser.error("native argument layout requires the qualified AITER code object")
        native = ctypes.CDLL(str(args.native_library.resolve()))
        native.native_load.argtypes = [ctypes.c_char_p]
        native.native_attention.argtypes = [ptr] * 9 + [uint, flt, ptr]
        native.native_reduce.argtypes = [ptr] * 4 + [uint, ptr]
        status = native.native_load(str(code_object).encode())
        if status:
            raise RuntimeError(f"native module load failed: {status}")
        metadata["native_library_sha256"] = hashlib.sha256(args.native_library.read_bytes()).hexdigest()
    if args.capture:
        metadata["capture_sha256"] = {
            name: hashlib.sha256((args.capture / f"{name}.bin").read_bytes()).hexdigest()
            for name in ("idx", "qa", "qr", "ck", "kr", "len")
        }

    def captured(name, dtype, shape):
        data = bytearray((args.capture / f"{name}.bin").read_bytes())
        return torch.frombuffer(data, dtype=dtype)[:math.prod(shape)].reshape(shape).cuda()

    records = []
    torch.manual_seed(metadata["seed"])
    for rows in args.rows:
        for overlap in (("model",) if args.capture else ("shared", "half", "distinct")):
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
            kv_len = ctx
            if args.capture:
                qa = captured("qa", torch.bfloat16, (rows, heads, value))
                qr = captured("qr", torch.bfloat16, (rows, heads, width-value))
                ck = captured("ck", torch.bfloat16, (ctx, value))
                kr = captured("kr", torch.bfloat16, (ctx, width-value))
                q = torch.cat((qa, qr), dim=-1)
                kv = torch.cat((ck, kr), dim=-1).view(ctx, 1, 1, width)
                idx = captured("idx", torch.int32, (rows, k))
                kv_len = int.from_bytes((args.capture / "len.bin").read_bytes()[:4], "little")
                assert rows + k - 1 <= kv_len <= ctx, kv_len
                assert (idx >= 0).all() and (idx <= kv_len-rows+query).all()
                metadata["capture"] = str(args.capture)
                metadata["kv_len"] = kv_len
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
            length = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
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
                       rows, ctx, cap, args.scale)
                return part / ml[:, :, 1, None]

            def pack():
                launch(lib.plow_pack, (packed_q, qa, qr), rows * heads)
                launch(lib.plow_pack, (packed_kv, ck, kr), ctx)

            def reference():
                return mla_decode_fwd(
                    packed_q, packed_kv, out, qp, kp, idx.flatten(), last, 1,
                    page_size=1, sm_scale=args.scale, num_kv_splits=2,
                )

            def adapter():
                pack()
                reference()
                return out.float()

            if native:
                splits = qp * 2
                native_part = torch.empty(rows, 2, heads, value, device="cuda", dtype=torch.float32)
                native_lse = torch.empty(rows, 2, heads, device="cuda", dtype=torch.float32)
                native_out = torch.empty_like(part)
                native_ml = torch.empty_like(ml)

                def native_adapter():
                    pack()
                    launch(native.native_attention,
                           (packed_q, packed_kv, idx, qp, kp, last, splits, native_part, native_lse),
                           rows, args.scale)
                    launch(native.native_reduce, (native_out, native_ml, native_part, native_lse), rows)
                    return native_out
            union()
            actual_plow, actual_aiter = plow(), adapter()
            torch.cuda.synchronize()
            assert torch.equal(packed_q, q) and torch.equal(packed_kv, kv)
            picks = list(dict.fromkeys([0, rows // 2, rows - 1]))
            selected = kv.view(ctx, width)[idx[picks].long()].float()
            scores = torch.einsum("bhd,bkd->bhk", q[picks].float(), selected) * args.scale
            gold = torch.einsum("bhk,bkd->bhd", scores.softmax(-1), selected[:, :, :value])
            errors = {}
            actuals = [("plow", actual_plow), ("aiter", actual_aiter)]
            if native:
                actuals.append(("native", native_adapter()))
                torch.cuda.synchronize()
                assert (native_ml[:, :, 0] == 0).all() and (native_ml[:, :, 1] == 1).all()
            for name, actual in actuals:
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
            if native:
                record["native_adapter_ms"] = bench(native_adapter)
            print(json.dumps(record), flush=True)
            records.append(record)
            args.out.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
