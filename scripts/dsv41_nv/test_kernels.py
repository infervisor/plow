"""DeepSeek-V4.1 sm_90a kernels vs the checkpoint's reference kernels (inference/kernel.py) and torch.

Run on one leased GPU:
  perf-data/tools/gpulease -n 1 dsv41-kern <venv-python> scripts/dsv41_nv/test_kernels.py <cubin> [names...]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cudrv import Cubin, f32, i32, i64  # noqa: E402

REF = os.environ.get(
    "DSV41_REF",
    "/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/"
    "dba1be0a40aa45a94ad051997016db3960a90277/inference",
)
sys.path.insert(0, REF)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype(torch.bfloat16)  # the reference GEMMs size their output from it
torch.manual_seed(0)
dev = "cuda"
K = Cubin(sys.argv[1])
only = set(sys.argv[2:])
RESULTS = []


def ref_kernels():
    import kernel  # reference tilelang kernels

    return kernel


def check(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def want(name):
    return not only or name in only


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# ------------------------------------------------------------------------------------------ rmsnorm
if want("rmsnorm"):
    for D in (5120, 1280, 512, 128):
        x = torch.randn(37, D, device=dev, dtype=torch.bfloat16) * 3
        w = torch.randn(D, device=dev, dtype=torch.bfloat16)
        y = torch.empty_like(x)
        K.launch("dsv_rmsnorm", (37,), (256,), [y, x, w, i32(D), i64(D), i64(D), f32(1e-20)])
        xf = x.float()
        ref = (w * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-20))).to(torch.bfloat16)
        diff = (y.float() - ref.float()).abs().max().item()
        mism = (y != ref).sum().item()
        check(f"rmsnorm D={D}", mism <= y.numel() * 1e-3, f"maxdiff={diff:.3g} mismatches={mism}")

# ------------------------------------------------------------------------------------------ act_quant
if want("act_quant"):
    kr = ref_kernels()
    for M, Kd in ((1, 5120), (77, 1280), (256, 8192)):
        x = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16) * torch.logspace(-3, 2, Kd, device=dev).to(torch.bfloat16)
        q = torch.empty(M, Kd, device=dev, dtype=torch.uint8)
        s = torch.empty(M, Kd // 32, device=dev, dtype=torch.uint8)
        groups = M * Kd // 32
        K.launch("dsv_act_quant_fp8", ((groups + 7) // 8,), (256,), [q, s, None, x, i32(M), i32(Kd), i64(Kd)])
        rq, rs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        eq = (q == rq.view(torch.uint8)).float().mean().item()
        es = (s == rs.view(torch.uint8)).float().mean().item()
        check(f"act_quant codes M={M} K={Kd}", eq == 1.0 and es == 1.0, f"codes={eq:.6f} scales={es:.6f}")
        fq = x.clone()
        K.launch("dsv_act_quant_fp8", ((groups + 7) // 8,), (256,), [None, None, fq, fq, i32(M), i32(Kd), i64(Kd)])
        rfq = kr.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
        check(f"act_quant fake M={M} K={Kd}", torch.equal(fq, rfq), f"mismatch={(fq != rfq).sum().item()}")

# ------------------------------------------------------------------------------------------ fp4 fake quant
if want("fp4"):
    kr = ref_kernels()
    for gs, e4 in ((32, 0), (16, 1)):
        x = torch.randn(64, 512, device=dev, dtype=torch.bfloat16) * 2
        x[3] = 0  # all-zero groups exercise the amax floor
        ours = x.clone()
        n = ours.numel()
        K.launch("dsv_fp4_fakequant", ((n + 255) // 256,), (256,), [ours, i64(n), i32(gs), i32(e4)])
        sd = torch.float8_e4m3fn if e4 else torch.float8_e8m0fnu
        ref = kr.fp4_act_quant(x.clone(), gs, True, scale_dtype=sd)
        check(f"fp4 fakequant gs={gs} e4m3_scale={e4}", torch.equal(ours, ref), f"mismatch={(ours != ref).sum().item()}")

# ------------------------------------------------------------------------------------------ gemm_w8a8
if want("gemm_w8a8"):
    kr = ref_kernels()
    for M, N, Kd in ((1, 1280, 5120), (5, 32768, 1280), (200, 5120, 8192), (129, 512, 5120)):
        x = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16)
        w = (torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ws = (torch.randint(118, 124, ((N + 31) // 32, Kd // 32), device=dev, dtype=torch.uint8))
        xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        ref = kr.fp8_gemm(xq, xs, w, ws.view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
        c = torch.empty(M, N, device=dev, dtype=torch.float32)
        K.launch(
            "dsv_gemm_w8a8",
            ((N + 127) // 128, (M + 63) // 64),
            (128,),
            [c, xq.view(torch.uint8), xs.view(torch.uint8), w.view(torch.uint8), ws, i32(M), i32(N), i32(Kd), i64(N), i32(1)],
        )
        r = rel(c, ref)
        check(f"gemm_w8a8 M={M} N={N} K={Kd}", r < 2e-3, f"rel={r:.3g}")

# ------------------------------------------------------------------------------------------ gemm_bf16w
if want("gemm_bf16w"):
    for M, N, Kd, fp8 in ((3, 1024, 4096, 1), (300, 512, 5120, 0), (64, 128, 512, 0)):
        x = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16)
        if fp8:
            w = (torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn)
            ws = torch.randint(118, 124, ((N + 31) // 32, Kd // 32), device=dev, dtype=torch.uint8)
            wd = (w.float() * torch.pow(2.0, ws.float() - 127).repeat_interleave(32, 0)[:N].repeat_interleave(32, 1)).to(torch.bfloat16)
            wp, wsp = w.view(torch.uint8), ws
        else:
            wd = torch.randn(N, Kd, device=dev, dtype=torch.bfloat16)
            wp, wsp = wd, None
        ref = x.float() @ wd.float().T
        c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        K.launch(
            "dsv_gemm_bf16w",
            ((N + 127) // 128, (M + 63) // 64, 1),
            (128,),
            [c, x, wp, wsp, i32(M), i32(N), i32(Kd), i64(Kd), i64(N), i32(fp8), i32(0), i64(0), i64(0), i64(0), i64(0)],
        )
        r = rel(c, ref)
        check(f"gemm_bf16w M={M} N={N} K={Kd} fp8={fp8}", r < 5e-3, f"rel={r:.3g}")

# ------------------------------------------------------------------------------------------ gemm_f32
if want("gemm_f32"):
    for M, N, Kd, abf, wbf in ((7, 384, 5120, 0, 1), (100, 24, 20480, 0, 0), (33, 512, 5120, 0, 1)):
        a = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16 if abf else torch.float32)
        w = torch.randn(N, Kd, device=dev, dtype=torch.bfloat16 if wbf else torch.float32)
        ref = a.float() @ w.float().T
        c = torch.empty(M, N, device=dev, dtype=torch.float32)
        K.launch("dsv_gemm_f32", ((N + 63) // 64, (M + 63) // 64), (256,), [c, a, w, i32(M), i32(N), i32(Kd), i64(Kd), i64(N), i32(abf), i32(wbf)])
        r = rel(c, ref)
        check(f"gemm_f32 M={M} N={N} K={Kd}", r < 1e-5, f"rel={r:.3g}")

torch.cuda.synchronize()
bad = [n for n, ok in RESULTS if not ok]
print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} passed" + (f"; FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
