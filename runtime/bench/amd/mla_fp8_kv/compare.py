#!/usr/bin/env python3
"""Qualify existing FP8 gathered MLA templates on gfx942; requires a GPU lease."""

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


def bench(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    samples = []
    for _ in range(11):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples)


def error(got, want):
    assert torch.isfinite(got).all(), "nonfinite output"
    delta = (got - want).float()
    return {
        "relative_l2": (delta.norm() / want.norm().clamp_min(1e-12)).item(),
        "max_abs": delta.abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 20])
    args = parser.parse_args()
    assert all(0 < b <= 20 for b in args.batch)
    assert "gfx942" in torch.cuda.get_device_properties(0).gcnArchName
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(53)
    fp8 = torch.float8_e4m3fn
    max_fp8 = torch.finfo(fp8).max
    lib = ctypes.CDLL(str(args.library.resolve()))
    ptr, uint, flt = ctypes.c_void_p, ctypes.c_uint, ctypes.c_float
    lib.plow_gather.argtypes = [ptr] * 9 + [uint] * 4 + [flt, ptr]
    lib.plow_write.argtypes = [ptr] * 5 + [uint] * 3 + [ptr]

    def launch(fn, tensors, *scalars):
        status = fn(*(t.data_ptr() for t in tensors), *scalars,
                    torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"HIP launch failed: {status}")

    metadata = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0), "seed": 53,
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "context": 81920, "heads": 8, "latent": 512, "rope": 64,
        "topk": 2048, "gf": 4, "fp8_format": str(fp8),
        "scope": "standalone HIP templates; no packet, HSA adapter or model qualification",
        "timing": "median of 11 HIP graph replays, attention only; excludes merge and quantization",
    }
    writers, attention = [], []
    for batch in args.batch:
        # Rung 1 must still target slot zero's current position in a larger allocation.
        slots, ctx = max(batch, 8), 81920
        for mask in (0xffffffff, 65535):
            cache = torch.full((slots, ctx, 512), 0x7f, device="cuda", dtype=torch.uint8)
            scales = torch.full((slots, ctx), -12345., device="cuda")
            x = torch.randn(batch, 512, device="cuda", dtype=torch.bfloat16)
            if mask != 0xffffffff:
                x[0].zero_()
            gamma = torch.randn(512, device="cuda", dtype=torch.bfloat16)
            positions = (torch.arange(batch, device="cuda", dtype=torch.int32) * 7919 + 70000) % ctx
            target = positions.to(torch.int64) & mask
            expected_rows = torch.zeros((slots, ctx), device="cuda", dtype=torch.bool)
            expected_rows[torch.arange(batch, device="cuda"), target] = True
            norm = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            norm *= gamma.float()
            expected_scale = norm.abs().amax(-1) / max_fp8
            expected_q = (norm / expected_scale[:, None].clamp_min(1e-30)).to(fp8)
            # The CDNA3 writer canonicalizes negative zero to positive zero.
            expected_q.view(torch.uint8).masked_fill_(expected_q.view(torch.uint8) == 0x80, 0)
            for _ in range(3):
                launch(lib.plow_write, [cache, scales, x, gamma, positions], batch, ctx, mask)
            got_scale = scales[torch.arange(batch, device="cuda"), target]
            got_q = cache[torch.arange(batch, device="cuda"), target].view(fp8)
            assert torch.allclose(got_scale, expected_scale, rtol=2e-6, atol=1e-8)
            mismatch = got_q.view(torch.uint8) != expected_q.view(torch.uint8)
            # Different FP32 reduction orders can round opposite ways at an FP8 midpoint.
            scaled = norm / expected_scale[:, None].clamp_min(1e-30)
            midpoint = (got_q.float() + expected_q.float()) * .5
            assert torch.all((scaled - midpoint).abs()[mismatch] <=
                             2e-5 * scaled.abs()[mismatch].clamp_min(1e-3))
            assert torch.all(scales[~expected_rows] == -12345.)
            # Test byte sentinels in chunks to avoid a second full cache allocation.
            for s in range(slots):
                assert torch.all(cache[s][~expected_rows[s]] == 0x7f)
            writers.append({"batch": batch, "allocated_slots": slots, "mask": mask,
                            "midpoint_rounding_differences": int(mismatch.sum()),
                            "untouched_tails": True})
            del cache, scales

        ctx, k = 81920, 2048
        qa = torch.randn(batch, 8, 512, device="cuda", dtype=torch.bfloat16)
        qr = torch.randn(batch, 8, 64, device="cuda", dtype=torch.bfloat16)
        ck = torch.randn(batch, ctx, 512, device="cuda", dtype=torch.bfloat16)
        kr = torch.randn(batch, ctx, 64, device="cuda", dtype=torch.bfloat16)
        # Different scales by slot and position expose a missing batch or gathered-row offset.
        amplitude = (torch.arange(ctx, device="cuda") % 17 + 1).float() / 8
        ck *= amplitude[None, :, None].to(torch.bfloat16)
        scales = ck.float().abs().amax(-1) / max_fp8
        ck8 = (ck.float() / scales[..., None]).to(fp8)
        for mixed in (False, True):
            live = [70000] * batch if not mixed else [
                [0, 1, 129, 2047, 2048, 65537, 70000, 79800][b % 8] for b in range(batch)]
            lengths = torch.tensor(live, device="cuda", dtype=torch.int32)
            idx = torch.full((batch, k), -8192, device="cuda", dtype=torch.int32)
            for b, length in enumerate(live):
                if length:
                    step = next(s for s in range(37, 101) if math.gcd(s, length) == 1)
                    idx[b, :min(k, length)] = (
                        (torch.arange(min(k, length), device="cuda") * step + b * 313) % length)

            reference, original = [], []
            for b, length in enumerate(live):
                chosen = idx[b, :min(k, length)].long()
                rope = kr[b, chosen].float()
                latent = ck8[b].view(torch.uint8)[chosen].view(fp8).float() * scales[b, chosen, None]
                for values, outputs in ((latent, reference), (ck[b, chosen].float(), original)):
                    logits = (qa[b].float() @ values.T + qr[b].float() @ rope.T) / math.sqrt(576)
                    outputs.append(logits.softmax(-1) @ values)
            reference, original = torch.stack(reference), torch.stack(original)
            for splits in (1, 16):
                out = torch.empty(batch, 8, splits, 512, device="cuda")
                ml = torch.empty(batch, 8, splits, 2, device="cuda")

                def call(use_fp8):
                    launch(lib.plow_gather, [out, ml, qa, qr, ck8 if use_fp8 else ck,
                                           kr, lengths, idx, scales],
                           batch, ctx, splits, int(use_fp8), 576**-0.5)

                def merged():
                    m, l = ml[..., 0], ml[..., 1]
                    maximum = m.max(-1, keepdim=True).values
                    weights = torch.where(l > 0, (m - maximum).exp2(), 0.)
                    return ((out * weights[..., None]).sum(-2) /
                            (l * weights).sum(-1)[..., None].clamp_min(1e-30))

                call(False)
                bf16_error = error(merged(), original)
                assert bf16_error["relative_l2"] < .005, bf16_error
                call(True)
                fp8_error = error(merged(), reference)
                assert fp8_error["relative_l2"] < .005, fp8_error
                record = {"batch": batch, "mixed_lengths": mixed, "lengths": live,
                          "splits": splits, "bf16_kernel_error": bf16_error,
                          "fp8_kernel_error": fp8_error,
                          "fp8_vs_original_error": error(merged(), original)}
                # Alternate timing order across cells to reduce a fixed warm-cache bias.
                for use_fp8 in ((False, True) if splits == 1 else (True, False)):
                    record["fp8_ms" if use_fp8 else "bf16_ms"] = bench(lambda: call(use_fp8))
                attention.append(record)
                print(json.dumps(record), flush=True)
        del ck, ck8, kr, scales
    args.out.write_text(json.dumps({"metadata": metadata, "writers": writers,
                                    "attention": attention}, indent=2) + "\n")


if __name__ == "__main__":
    main()
