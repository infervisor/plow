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
            [c, xq.view(torch.uint8), xs.view(torch.uint8), w.view(torch.uint8), ws, i32(M), i32(N), i32(Kd), i64(N), i32(1), i32(1), None],
            smem=3 * (64 + 128) * 144,
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
            [c, x, wp, wsp, i32(M), i32(N), i32(Kd), i64(Kd), i64(N), i32(fp8), i32(0), i64(0), i64(0), i64(0), i64(0), i32(1), None],
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
        K.launch("dsv_gemm_f32", ((N + 63) // 64, (M + 63) // 64), (256,), [c, a, w, i32(M), i32(N), i32(Kd), i64(Kd), i64(N), i32(abf), i32(wbf)], smem=3 * (64 + 128) * 144)
        r = rel(c, ref)
        check(f"gemm_f32 M={M} N={N} K={Kd}", r < 1e-5, f"rel={r:.3g}")

# ------------------------------------------------------------------------------------------ fused mHC mix
if want("hc_mix"):
    import math
    for T in (1, 9, 1024):
        Kd = 20480
        x = torch.randn(T, Kd, device=dev, dtype=torch.bfloat16)
        fn = torch.randn(24, Kd, device=dev, dtype=torch.float32) * 0.01
        scale = torch.tensor([0.9, 1.1, 0.7], device=dev, dtype=torch.float32)
        base = torch.randn(24, device=dev, dtype=torch.float32) * 0.1
        S = max(1, min(math.ceil(264 / math.ceil(T / 4)), Kd // 1024))
        part = torch.empty(S, T, 25, device=dev, dtype=torch.float32)
        K.launch("dsv_hc_mix_partial", (S, (T + 3) // 4), (256,), [part, x, fn, i32(T), i32(Kd)])
        pre = torch.empty(T, 4, device=dev, dtype=torch.float32)
        post = torch.empty(T, 4, device=dev, dtype=torch.float32)
        comb = torch.empty(T, 4, 4, device=dev, dtype=torch.float32)
        K.launch("dsv_hc_mix_finish", ((T + 7) // 8,), (128,),
                 [pre, post, comb, part, i32(S), i32(T), i32(Kd), scale, base, i32(20), f32(1e-20), f32(1e-6)])
        # reference: model.py Block.hc_mixes + kernel.py hc_split_sinkhorn, in torch
        xf = x.float()
        mixes = (xf @ fn.T) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-20)
        rp = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1e-6
        rq = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
        c = (mixes[:, 8:] * scale[2] + base[8:]).view(T, 4, 4)
        c = c.softmax(-1) + 1e-6
        c = c / (c.sum(-2, keepdim=True) + 1e-6)
        for _ in range(19):
            c = c / (c.sum(-1, keepdim=True) + 1e-6)
            c = c / (c.sum(-2, keepdim=True) + 1e-6)
        e = max(rel(pre, rp), rel(post, rq), rel(comb, c))
        check(f"hc_mix T={T} splits={S}", e < 1e-4, f"rel={e:.3g}")

# ------------------------------------------------------------------------------------------ split-K
if want("wg"):
    # wgmma prefill GEMMs (dsv41_wg.cu): dense fp8 against fp8_gemm with ragged M / N edges and f32 or
    # bf16 out, the batched wo_a form against per-group launches, grouped fp4 in both A modes
    kr = ref_kernels()
    WG_SMEM = 3 * (128 * 64 * 2 + 256 * 64 * 2 + 256 * 80) + 3 * 3 * 8 + 128 * 4 + 1024

    def wg_fp8(c, a, w, ws, M, N, Kd, lda, ldc, f32o, batch=1, strides=(0, 0, 0, 0)):
        K.launch("dsv_gemm_wg_fp8", ((N + 255) // 256, (M + 127) // 128, batch), (384,),
                 [c, a, w, ws, i32(M), i32(N), i32(Kd), i64(lda), i64(ldc), i32(f32o)] + [i64(v) for v in strides], smem=WG_SMEM)

    for M, N, Kd, f32o in ((300, 640, 1024, 1), (129, 256, 5120, 0), (1000, 1280, 4096, 1)):
        x = torch.randn(M, Kd, device=dev)
        w = (torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ws = torch.randint(118, 124, ((N + 31) // 32, Kd // 32), device=dev, dtype=torch.uint8)
        xq, xs = kr.act_quant(x.bfloat16(), 32, "ue8m0", torch.float8_e8m0fnu)
        fq = (xq.float() * xs.float().repeat_interleave(32, 1)).bfloat16()
        ref = kr.fp8_gemm(xq, xs, w, ws.view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32).float()
        c = torch.empty(M, N, device=dev, dtype=torch.float32 if f32o else torch.bfloat16)
        wg_fp8(c, fq, w.view(torch.uint8), ws, M, N, Kd, Kd, N, f32o)
        r = rel(c.float(), ref)
        check(f"wg_fp8 M={M} N={N} K={Kd} f32={f32o}", r < 3e-3, f"rel vs fp8_gemm={r:.3g}")
    # batched: G groups of (T x KG) . (OR x KG)^T side by side, as wo_a
    T, G, OR, KG = 300, 4, 512, 1024
    o = torch.randn(T, G * KG, device=dev).bfloat16()
    wa = (torch.randn(G * OR, KG, device=dev) * 0.05).to(torch.float8_e4m3fn)
    was = torch.randint(118, 124, (G * OR // 32, KG // 32), device=dev, dtype=torch.uint8)
    cb = torch.empty(T, G * OR, device=dev, dtype=torch.bfloat16)
    wg_fp8(cb, o, wa.view(torch.uint8), was, T, OR, KG, G * KG, G * OR, 0, G, (OR * KG, (OR // 32) * (KG // 32), KG, OR))
    worst = 0.0
    for g in range(G):
        cg = torch.empty(T, OR, device=dev, dtype=torch.bfloat16)
        wg_fp8(cg, o[:, g * KG:(g + 1) * KG].contiguous(), wa.view(torch.uint8)[g * OR:(g + 1) * OR].contiguous(),
               was[g * OR // 32:(g + 1) * OR // 32].contiguous(), T, OR, KG, KG, OR, 0)
        worst = max(worst, rel(cb[:, g * OR:(g + 1) * OR].float(), cg.float()))
    check(f"wg_fp8 batched G={G}", worst == 0.0, f"rel vs per-group={worst:.3g}")
    # grouped fp4: 3 experts with 5 / 130 / 0 / 260 rows, A gathered through rows vs pre-gathered
    E, N, Kd, BM = 4, 512, 1024, 128
    counts = [5, 130, 0, 260]
    n = sum(counts)
    offs = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.int32, device=dev)
    tl = []
    for e in range(E):
        for r0 in range(int(offs[e]), int(offs[e + 1]), BM):
            tl += [e, r0]
    tiles = torch.tensor(tl, dtype=torch.int32, device=dev)
    meta = torch.tensor([len(tl) // 2], dtype=torch.int32, device=dev)
    ntok = 200
    rows = torch.randint(0, ntok, (n,), dtype=torch.int32, device=dev)
    x = torch.randn(ntok, Kd, device=dev).bfloat16()
    xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    fq = (xq.float() * xs.float().repeat_interleave(32, 1)).bfloat16()
    w4 = torch.randint(0, 256, (E, N, Kd // 2), dtype=torch.uint8, device=dev)
    s4 = torch.randint(118, 124, (E, N, Kd // 32), dtype=torch.uint8, device=dev)
    C = torch.full((n, N), float("nan"), device=dev, dtype=torch.bfloat16)
    C2 = torch.full((n, N), float("nan"), device=dev, dtype=torch.bfloat16)
    for out, a, by_row in ((C, fq, 0), (C2, fq[rows.long()].contiguous(), 1)):
        K.launch("dsv_moe_gemm_wg_fp4", ((N + 255) // 256, len(tl) // 2), (384,),
                 [out, a, w4, s4, tiles, meta, offs, rows, i32(by_row), i32(N), i32(Kd), i64(N * Kd // 2), i64(N * Kd // 32)], smem=WG_SMEM)
    worst = 0.0
    for e in range(E):
        p0, p1 = int(offs[e]), int(offs[e + 1])
        if p1 == p0:
            continue
        toks = rows[p0:p1].long()
        ref = kr.fp4_gemm(xq[toks].contiguous(), xs[toks].contiguous(), w4[e].view(torch.float4_e2m1fn_x2),
                          s4[e].view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
        worst = max(worst, rel(C[p0:p1].float(), ref.float()))
    check("wg_fp4 grouped (gathered A)", worst < 3e-3 and not C.isnan().any().item(), f"rel vs fp4_gemm={worst:.3g}")
    check("wg_fp4 grouped (a_by_row == gathered)", torch.equal(C, C2))

if want("gemv"):
    # decode W8A8 GEMV (swap-AB) against fp8_gemm, unsplit and split-K, ragged N
    kr = ref_kernels()
    for M, N, Kd, ks in ((1, 1280, 5120, 1), (5, 1000, 2304, 4), (16, 4608, 1280, 8), (12, 512, 5120, 16)):
        x = torch.randn(M, Kd, device=dev).bfloat16()
        w = (torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ws = torch.randint(118, 124, ((N + 31) // 32, Kd // 32), device=dev, dtype=torch.uint8)
        xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        ref = kr.fp8_gemm(xq, xs, w, ws.view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32).float()
        c = torch.empty(M, N, device=dev, dtype=torch.float32)
        part = torch.empty(ks, M, N, device=dev, dtype=torch.float32) if ks > 1 else None
        K.launch("dsv_gemv_w8a8_t8" if M <= 8 else "dsv_gemv_w8a8_t16", ((N + 63) // 64, ks), (128,),
                 [c, xq.view(torch.uint8), xs.view(torch.uint8), w.view(torch.uint8), ws, i32(M), i32(N), i32(Kd), i64(N), i32(1), part])
        if ks > 1:
            K.launch("dsv_splitk_reduce", ((M * N + 255) // 256,), (256,), [c, part, i32(ks), i32(1), i32(M), i32(N), i64(N), i64(0), i32(1)])
        r = rel(c, ref)
        check(f"gemv_w8a8 M={M} N={N} K={Kd} ks={ks}", r < 3e-3, f"rel vs fp8_gemm={r:.3g}")

if want("sparse_split"):
    # split-KV sparse attention + merge against the unsplit kernel: decode rows over a window ring
    # and compressed picks, with -1 holes and one row that has no valid index (sink-only zeros)
    SA_SMEM = (64 * 520 + 64 * 520 + 64 * 72) * 2 + 4 * 64 * 4 * 2 + 64 * 4
    H, D, Wn, Nc = 64, 512, 128, 512
    for T in (1, 3):
        n_idx = Wn + Nc
        q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
        win = [torch.randn(Wn, D, device=dev, dtype=torch.bfloat16) for _ in range(T)]
        cmp = [torch.randn(4096, D, device=dev, dtype=torch.bfloat16) for _ in range(T)]
        idx = torch.cat([torch.arange(Wn, device=dev).repeat(T, 1),
                         Wn + torch.randint(0, 4096, (T, Nc), device=dev)], 1).int()
        idx[:, 5:40] = -1
        if T > 1:
            idx[1] = -1
        wp = torch.tensor([w.data_ptr() for w in win], dtype=torch.uint64, device=dev)
        cp = torch.tensor([c.data_ptr() for c in cmp], dtype=torch.uint64, device=dev)
        sink = torch.randn(H, device=dev, dtype=torch.float32)
        base = torch.empty(T, H, D, device=dev, dtype=torch.bfloat16)
        args = [idx, i32(n_idx), wp, cp, i32(Wn), i32(1), sink, f32(D ** -0.5)]
        K.launch("dsv_sparse_attn", (T, 1), (512,), [base, q] + args + [None], smem=SA_SMEM)
        for S in (2, 4, 11):
            part = torch.empty(T, S, H, D + 4, device=dev, dtype=torch.float32)
            o = torch.empty(T, H, D, device=dev, dtype=torch.bfloat16)
            K.launch("dsv_sparse_attn", (T, S), (512,), [o, q] + args + [part], smem=SA_SMEM)
            K.launch("dsv_sparse_attn_merge", (T, H), (128,), [o, part, i32(S), sink])
            r = rel(o.float(), base.float())
            check(f"sparse_attn split-KV T={T} S={S}", r < 1e-2 and torch.isfinite(o.float()).all().item(), f"rel vs unsplit={r:.3g}")

if want("splitk"):
    kr = ref_kernels()
    for M, N, Kd, ks in ((9, 5120, 8192, 4), (1, 1280, 5120, 8)):
        x = torch.randn(M, Kd, device=dev)
        w = (torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ws = torch.randint(118, 124, ((N + 31) // 32, Kd // 32), device=dev, dtype=torch.uint8)
        xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        base = torch.empty(M, N, device=dev, dtype=torch.float32)
        K.launch("dsv_gemm_w8a8", ((N + 127) // 128, (M + 63) // 64, 1), (128,),
                 [base, xq.view(torch.uint8), xs.view(torch.uint8), w.view(torch.uint8), ws, i32(M), i32(N), i32(Kd), i64(N), i32(1), i32(1), None], smem=3 * (64 + 128) * 144)
        part = torch.empty(ks, M, N, device=dev, dtype=torch.float32)
        c = torch.empty(M, N, device=dev, dtype=torch.float32)
        K.launch("dsv_gemm_w8a8", ((N + 127) // 128, (M + 63) // 64, ks), (128,),
                 [c, xq.view(torch.uint8), xs.view(torch.uint8), w.view(torch.uint8), ws, i32(M), i32(N), i32(Kd), i64(N), i32(1), i32(ks), part], smem=3 * (64 + 128) * 144)
        K.launch("dsv_splitk_reduce", ((M * N + 255) // 256,), (256,), [c, part, i32(ks), i32(1), i32(M), i32(N), i64(N), i64(0), i32(1)])
        r = rel(c, base)
        check(f"w8a8 split-K {ks} M={M} N={N} K={Kd}", r < 1e-5, f"rel vs unsplit={r:.3g}")
    # bf16w batched (the wo_a shape): 8 groups x (T x 4096 -> 1024)
    T, G, OR, KG = 3, 8, 1024, 4096
    o = torch.randn(T, G * KG, device=dev)
    wa = (torch.randn(G * OR, KG, device=dev) * 0.05).to(torch.float8_e4m3fn)
    was = torch.randint(118, 124, (G * OR // 32, KG // 32), device=dev, dtype=torch.uint8)
    def woa(ks):
        c = torch.empty(T, G * OR, device=dev)
        part = torch.empty(ks, G, T, OR, device=dev, dtype=torch.float32) if ks > 1 else None
        K.launch("dsv_gemm_bf16w", ((OR + 127) // 128, (T + 63) // 64, G * ks), (128,),
                 [c, o, wa.view(torch.uint8), was, i32(T), i32(OR), i32(KG), i64(G * KG), i64(G * OR), i32(1), i32(0),
                  i64(KG), i64(OR * KG), i64((OR // 32) * (KG // 32)), i64(OR), i32(ks), part])
        if ks > 1:
            K.launch("dsv_splitk_reduce", ((G * T * OR + 255) // 256,), (256,), [c, part, i32(ks), i32(G), i32(T), i32(OR), i64(G * OR), i64(OR), i32(0)])
        return c
    a1, a4 = woa(1), woa(4)
    wd = (wa.float() * torch.pow(2.0, was.float() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)).view(G, OR, KG)
    ref = torch.einsum("tgk,grk->tgr", o.float().view(T, G, KG), wd).reshape(T, G * OR)
    r1, r4 = rel(a1, ref), rel(a4, ref)
    check("bf16w batched split-K 4 (wo_a)", r1 < 5e-3 and r4 < 5e-3, f"rel unsplit={r1:.3g} split={r4:.3g}")

# ------------------------------------------------------------------------------------------ fp32 dot / rows forms
if want("gemm_f32_small"):
    for name, M, N, Kd, abf, wbf, grid in (
        ("dsv_gemm_f32_rows", 1024, 24, 20480, 1, 0, None),
        ("dsv_gemm_f32_rows", 130, 24, 20480, 1, 0, None),
        ("dsv_gemm_f32_dot", 9, 384, 5120, 1, 1, None),
        ("dsv_gemm_f32_dot", 1, 24, 20480, 1, 0, None),
    ):
        a = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16 if abf else torch.float32)
        w = torch.randn(N, Kd, device=dev, dtype=torch.bfloat16 if wbf else torch.float32)
        ref = a.float() @ w.float().T
        c = torch.empty(M, N, device=dev, dtype=torch.float32)
        g = ((M + 3) // 4,) if name.endswith("rows") else (N, M)
        K.launch(name, g, (256,), [c, a, w, i32(M), i32(N), i32(Kd), i64(Kd), i64(N), i32(abf), i32(wbf)])
        r = rel(c, ref)
        check(f"{name} M={M} N={N} K={Kd}", r < 1e-5, f"rel={r:.3g}")

torch.cuda.synchronize()
bad = [n for n, ok in RESULTS if not ok]
print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} passed" + (f"; FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
