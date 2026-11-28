"""The vLLM bar for DeepSeek-V4.1's [32,32] ue8m0 fp8 linears on Hopper: vLLM serves them with
MarlinMxfp8LinearKernel (weight-only fp8 -> bf16, bf16 activations; see its server log "Using
MarlinMxfp8LinearKernel for MXFP8 GEMM"). Times that kernel, and cuBLAS bf16 for reference, on the
shapes of scripts/dsv41_nv/test_gemm_fp8mx.py.

  perf-data/tools/gpulease -n 1 vllm-mxfp8 /root/tts-work/venv-vllm/bin/python scripts/dsv41_nv/bench_vllm_mxfp8.py
"""
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_mxfp8_marlin_linear, prepare_mxfp8_layer_for_marlin

torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
SHAPES = [
    ("q_a 1k", 1024, 1280, 5120), ("q_b tp4 1k", 1024, 16384, 1280), ("wkv 1k", 1024, 512, 5120), ("sh_w1 1k", 1024, 2304, 5120),
    ("sh_w2 1k", 1024, 5120, 2304), ("idx_wq_b 1k", 1024, 4096, 1280), ("q_b tp4 4k", 4096, 16384, 1280), ("sh_w1 4k", 4096, 2304, 5120),
    ("sh_w2 4k", 4096, 5120, 2304), ("q_a 8k", 8192, 1280, 5120), ("q_b tp4 8k", 8192, 16384, 1280),
]


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return sorted(ts)[len(ts) // 2]


print(f"{'shape':16s} {'T':>5s} {'N':>6s} {'K':>5s} | {'marlin us':>9s} {'TF':>5s} | {'cublas bf16 us':>14s} {'TF':>5s} | rel")
for name, T, N, K in SHAPES:
    layer = torch.nn.Module()
    w = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
    ws = torch.randint(118, 124, (N // 32, K // 32), device=dev, dtype=torch.uint8)
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(ws.repeat_interleave(32, 0).contiguous(), requires_grad=False)
    layer.output_size_per_partition, layer.input_size_per_partition = N, K
    prepare_mxfp8_layer_for_marlin(layer)
    x = torch.randn(T, K, device=dev)
    wd = (w.float() * (2.0 ** (ws.float() - 127)).repeat_interleave(32, 0).repeat_interleave(32, 1)).bfloat16()
    f = lambda: apply_mxfp8_marlin_linear(x, layer.weight, layer.weight_scale, layer.workspace, N, K)
    y = f()
    ref = x @ wd.T
    r = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
    us_m = timed(f)
    us_c = timed(lambda: x @ wd.T)
    fl = 2 * T * N * K
    print(f"{name:16s} {T:5d} {N:6d} {K:5d} | {us_m:9.1f} {fl / us_m / 1e6:5.0f} | {us_c:14.1f} {fl / us_c / 1e6:5.0f} | {r:.1e}", flush=True)
