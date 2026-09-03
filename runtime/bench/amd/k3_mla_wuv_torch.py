#!/usr/bin/env python3
# Run this in the pinned vLLM ROCm image under perf-data/tools/gpulease -n 1.
import statistics

import torch


def timed(run, repeats: int = 100) -> float:
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(21):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(repeats):
            run()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / repeats)
    return statistics.median(samples)


def measure(batch: int) -> None:
    heads, latent, value = 12, 512, 128
    x = torch.randn((heads, batch, latent), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((heads, latent, value), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((batch, heads, value), dtype=torch.bfloat16, device="cuda")

    def run() -> None:
        torch.bmm(x, weight, out=out.transpose(0, 1))

    print(f"B{batch},torch_bf16_wuv_us={timed(run):.3f}")

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.quantization.quark.utils import (
        quark_quantize_weight_to_mxfp4,
    )

    raw = torch.randn((latent, heads, value), dtype=torch.bfloat16, device="cuda")
    packed, scale = quark_quantize_weight_to_mxfp4(raw.permute(1, 2, 0))

    def run_fp4() -> None:
        rocm_aiter_ops.batched_gemm_a16wfp4(
            x, packed, scale, out, transpose_bm=True, prequant=True, y_scale=None
        )

    print(f"B{batch},aiter_mxfp4_wuv_us={timed(run_fp4):.3f}")


measure(1)
measure(8)
