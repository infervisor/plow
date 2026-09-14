#!/usr/bin/env python3
"""Check sparse FP8 prefill scale addressing against sampled FP32 attention."""

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path

import torch

from compare import bench, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 129, 4464])
    parser.add_argument("--record-failure", action="store_true")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--scale", type=float, default=576**-0.5)
    args = parser.parse_args()
    assert all(0 < r <= 8192 for r in args.rows)
    assert math.isfinite(args.scale) and args.scale > 0
    assert not args.capture or len(args.rows) == 1
    assert "gfx942" in torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(54)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = ctypes.CDLL(str(args.library.resolve()))
    ptr, uint, flt = ctypes.c_void_p, ctypes.c_uint, ctypes.c_float
    lib.plow_union.argtypes = [ptr] * 4 + [uint] * 3 + [ptr]
    lib.plow_prefill.argtypes = [ptr] * 9 + [uint] * 4 + [flt, ptr]

    def launch(fn, tensors, *scalars):
        status = fn(*(t.data_ptr() for t in tensors), *scalars,
                    torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"HIP launch failed: {status}")

    records = []
    for rows in args.rows:
        ctx, length, k, cap = 81920, 70000, 2048, 16384
        qa = torch.randn(rows, 8, 512, device="cuda", dtype=torch.bfloat16)
        qr = torch.randn(rows, 8, 64, device="cuda", dtype=torch.bfloat16)
        ck = torch.randn(ctx, 512, device="cuda", dtype=torch.bfloat16)
        kr = torch.randn(ctx, 64, device="cuda", dtype=torch.bfloat16)
        ck *= ((torch.arange(ctx, device="cuda") % 17 + 1) / 8)[:, None].to(torch.bfloat16)
        if args.capture:
            def captured(name, dtype, shape):
                data = bytearray((args.capture / f"{name}.bin").read_bytes())
                return torch.frombuffer(data, dtype=dtype)[:math.prod(shape)].reshape(shape).cuda()
            qa = captured("qa", torch.bfloat16, (rows, 8, 512))
            qr = captured("qr", torch.bfloat16, (rows, 8, 64))
            ck = captured("ck", torch.bfloat16, (ctx, 512))
            kr = captured("kr", torch.bfloat16, (ctx, 64))
            length = int.from_bytes((args.capture / "len.bin").read_bytes()[:4], "little")
            assert rows + k - 1 <= length <= ctx
        scales = ck.float().abs().amax(-1) / 448
        ck8 = (ck.float() / scales[:, None]).to(torch.float8_e4m3fn)
        # Poison the inactive cache; a union tail must never use these scales.
        scales[length:] = float("nan")
        ck8.view(torch.uint8)[length:] = 0x7f
        live = length - rows + torch.arange(rows, device="cuda", dtype=torch.int64) + 1
        idx = torch.empty(rows, k, device="cuda", dtype=torch.int32)
        for q in range(rows):
            n = int(live[q])
            step = next(s for s in range(37, 101) if math.gcd(s, n) == 1)
            idx[q] = (torch.arange(k, device="cuda") * step + q * 313) % n
        if args.capture:
            idx = captured("idx", torch.int32, (rows, k))
            assert (idx >= 0).all() and (idx < live[:, None]).all()
        lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
        tiles = (rows + 7) // 8
        header = (tiles * 4 + 255) // 256 * 256
        uni = torch.empty(header + tiles * cap * 12, device="cuda", dtype=torch.uint8)
        mask = torch.empty(min(tiles, 304) * ctx, device="cuda", dtype=torch.int64)
        launch(lib.plow_union, [uni, mask, idx, lengths], rows, ctx, cap)
        sampled = sorted(set([0, min(1, rows - 1), min(7, rows - 1), rows // 2, rows - 1]))
        original, reference = [], []
        for q in sampled:
            selected = idx[q].long()
            rope = kr[selected].float()
            latent = ck8.view(torch.uint8)[selected].view(torch.float8_e4m3fn).float() * scales[selected, None]
            for values, outputs in ((ck[selected].float(), original), (latent, reference)):
                logits = (qa[q].float() @ values.T + qr[q].float() @ rope.T) * args.scale
                outputs.append(logits.softmax(-1) @ values)
        original, reference = torch.stack(original), torch.stack(reference)
        out = torch.empty(rows, 8, 512, device="cuda")
        ml = torch.empty(rows, 8, 2, device="cuda")
        record = {"rows": rows, "length": length, "sampled_rows": sampled}
        for fp8 in (False, True):
            def call():
                launch(lib.plow_prefill, [out, ml, qa, qr, ck8 if fp8 else ck,
                                         kr, lengths, uni, scales],
                       rows, ctx, cap, int(fp8), args.scale)
            call()
            got = out[sampled] / ml[sampled, :, 1, None]
            measured = error(got, reference if fp8 else original)
            key = "fp8" if fp8 else "bf16"
            record[key + "_kernel_error"] = measured
            if not args.record_failure:
                assert measured["relative_l2"] < .01, record
            record[key + "_ms"] = bench(call)
            if fp8:
                record["fp8_vs_original_error"] = error(got, original)
        records.append(record)
        print(json.dumps(record), flush=True)
    args.out.write_text(json.dumps({
        "metadata": {
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "hip": torch.version.hip, "seed": 54,
            "scale": args.scale,
            "capture": str(args.capture) if args.capture else None,
            "capture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in args.capture.glob("*.bin")} if args.capture else None,
            "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
            "scope": "standalone HIP; no FP8 serving qualification",
            "timing": "median of 11 graph replays; attention only, excludes union and quantization",
        }, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
