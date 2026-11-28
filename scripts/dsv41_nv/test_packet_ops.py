"""Packet op bodies for DeepSeek-V4.1 on sm_90a (runtime/nvidia/dsv41/packet_ops_test.cu wrappers)
against the reference (inference/kernel.py) and the verified dsv41 kernels.

  perf-data/tools/gpulease -n 1 dsv41-pktops <venv-python> scripts/dsv41_nv/test_packet_ops.py \
      <packet_ops_test.cubin> <dsv41 cubin> [test ...]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cudrv import Cubin, i32, i64  # noqa: E402

REF = os.environ.get(
    "DSV41_REF",
    "/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/inference",
)
sys.path.insert(0, REF)
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)
dev = "cuda"
P = Cubin(sys.argv[1])
D = Cubin(sys.argv[2])
only = set(sys.argv[3:])
results = []


def want(name):
    return not only or name in only


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


if want("act_quant_mx"):
    import kernel as kr
    for rows, k, nblk in ((1, 5120, 7), (37, 1024, 132), (300, 2304, 64)):
        x = torch.randn(rows, k, device=dev) * torch.logspace(-12, 4, rows, device=dev).unsqueeze(1)
        x[0, :32] = 0  # an all-zero block: the 1e-4 amax floor
        x[-1, 64:96] = 3e-8  # deep in the e4m3 subnormal range after scaling
        out = torch.empty_like(x)
        P.launch("t_act_quant_mx", (nblk,), (256,), [out, x, i32(rows), i32(k)])
        xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        ref = (xq.float() * xs.float().repeat_interleave(32, 1)).bfloat16()
        fq = torch.empty_like(x)
        D.launch("dsv_act_quant_fp8", (((rows * k // 32) + 7) // 8,), (256,), [None, None, fq, x, i32(rows), i32(k), i64(k)])
        inplace = x.clone()
        P.launch("t_act_quant_mx", (nblk,), (256,), [inplace, inplace, i32(rows), i32(k)])
        ok = torch.equal(out, ref) and torch.equal(out, fq) and torch.equal(inplace, out)
        check(f"act_quant_mx rows={rows} k={k} nblk={nblk}", ok,
              f"mismatch vs kernel.py={(out != ref).sum().item()} vs dsv41={(out != fq).sum().item()}")
        q8 = torch.empty(rows, k, dtype=torch.uint8, device=dev)
        s8 = torch.empty(rows, k // 32, dtype=torch.uint8, device=dev)
        P.launch("t_act_quant_mx8", (nblk,), (256,), [q8, s8, x, i32(rows), i32(k)])
        ok = torch.equal(q8, xq.view(torch.uint8)) and torch.equal(s8, xs.view(torch.uint8))
        check(f"act_quant_mx e4m3 rows={rows} k={k} nblk={nblk}", ok,
              f"q mismatch={(q8 != xq.view(torch.uint8)).sum().item()} s mismatch={(s8 != xs.view(torch.uint8)).sum().item()}")

if want("gemv_fp8mx"):
    # GemmFp8Mx at decode rows (op_gemv_fp8mx.cuh): fp8 activations (act_quant pair) and bf16 (wo_a)
    def q32(v, rows_blk):
        r, c = v.shape
        s = (v.abs().view(r // rows_blk, rows_blk, c // 32, 32).amax((1, 3)) / 448).clamp(min=1e-4 / 448)
        e = torch.ceil(torch.log2(s))
        q = (v / torch.exp2(e).repeat_interleave(rows_blk, 0).repeat_interleave(32, 1)).to(torch.float8_e4m3fn)
        return q, (e + 127).to(torch.uint8), torch.exp2(e).repeat_interleave(rows_blk, 0).repeat_interleave(32, 1)
    for n, k, groups in ((1280, 5120, 1), (512, 5120, 1), (1024, 4096, 2)):
        wq, ws, wsx = q32(torch.randn(groups * n, k, device=dev, dtype=torch.float32) * 0.05, 32)
        wd = wq.float() * wsx
        for fp8 in ((1, 0) if groups == 1 else (0,)):
            for T in (1, 5, 16, 64):
                x = torch.randn(T, groups * k, device=dev, dtype=torch.float32)
                if fp8:
                    xq, xs, xsx = q32(x, 1)
                    xin, xsin, xd = xq.view(torch.uint8), xs, xq.float() * xsx
                else:
                    xin = x.bfloat16()
                    xsin, xd = torch.zeros(1, dtype=torch.uint8, device=dev), xin.float()
                ref = torch.cat([xd[:, g * k:(g + 1) * k] @ wd[g * n:(g + 1) * n].T for g in range(groups)], 1)
                c = torch.empty(T, groups * n, device=dev, dtype=torch.bfloat16)
                arena = 8 * 2 * 16 * 144  # the minimum: plow_gv8::arena_floats(2)
                P.launch("t_gemv_fp8mx", (132,), (256,), [c, xin, xsin, wq.view(torch.uint8), ws, i32(T), i32(n), i32(k), i32(groups),
                                                          i32(fp8), i32(1), i32(1), i32(arena // 4)], smem=arena)
                r = ((c.float() - ref).norm() / ref.norm()).item()
                check(f"gemv_fp8mx fp8={fp8} T={T} N={n} K={k} groups={groups}", r < 4e-3, f"rel={r:.1e}")

# ---------------------------------------------------------------- inside a real packet (packet_op)
import subprocess
import tempfile

PACKET_OP = os.environ.get("PACKET_OP")
ISA = os.environ.get("ISA_CUBIN")  # stock sm_90a interpreter
ISA_PF = os.environ.get("ISA_PF_CUBIN")  # its prefill object
ISA_PF_CFG = os.environ.get("ISA_PF_CFG_CUBIN")  # a prefill object built with a V4.1 block's plow_config.h (MoE arms)
FP8MX = os.environ.get("FP8MX_CUBIN")  # interp_sm90a_pfgemm_fp8mx.cubin
tmp = tempfile.mkdtemp(prefix="pktops_")


def packet_op(obj, symbol, block, arena, op, ints, tensors, outs, n_cu=132, iters=1, floats=()):
    """tensors: [(name, bytes, tensor-or-None)]; outs: [(name, torch dtype, shape)] -> [tensor], median_us"""
    cmd = [PACKET_OP, "--object", obj, "--symbol", symbol, "--block", str(block), "--arena-symbol", arena, "--op", op,
           "--n-cu", str(n_cu), "--i", ",".join(map(str, ints)), "--iters", str(iters)]
    if floats:
        cmd += ["--f", ",".join(repr(float(f)) for f in floats)]
    for k, (name, nbytes, t) in enumerate(tensors):
        spec = f"{name}:{nbytes}"
        if t is not None:
            f = os.path.join(tmp, f"in{k}.bin")
            t.contiguous().cpu().view(torch.uint8).numpy().tofile(f)
            spec += f":{f}"
        cmd += ["--t", spec]
    files = []
    for k, (name, dt, shape) in enumerate(outs):
        f = os.path.join(tmp, f"out{k}.bin")
        files.append((f, dt, shape))
        cmd += ["--out", f"{name}={f}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"packet_op failed: {r.stderr[-2000:]}")
    us = float(next(l.split()[1] for l in r.stdout.splitlines() if l.startswith("median_us")))
    got = []
    for f, dt, shape in files:
        raw = torch.frombuffer(bytearray(open(f, "rb").read()), dtype=torch.uint8)
        n = 1
        for d in shape:
            n *= d
        got.append(raw.view(dt)[:n].view(*shape).to(dev))
    return got, us


if want("pkt_act_quant_mx") and PACKET_OP and ISA:
    import kernel as kr
    for rows, k in ((3, 5120), (129, 2304)):
        x = torch.randn(rows, k, device=dev) * 4
        (out,), us = packet_op(ISA, "_Z12interp_sm90a11PlowProgram", 256, "plow_arena_bytes", "ActQuantMx", [rows, k],
                               [("out", rows * k * 2, None), ("x", rows * k * 2, x)], [("out", torch.bfloat16, (rows, k))])
        xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        ref = (xq.float() * xs.float().repeat_interleave(32, 1)).bfloat16()
        check(f"pkt ActQuantMx rows={rows} k={k} (sm_90a interpreter)", torch.equal(out, ref), f"mismatch={(out != ref).sum().item()} {us:.1f} us")

if want("pkt_gemm_fp8mx") and PACKET_OP and FP8MX:
    import kernel as kr
    for T, N, K, blocks in ((300, 1280, 5120, 132), (1024, 5120, 8192, 132), (4096, 4608, 5120, 132), (129, 512, 1024, 7)):
        x = torch.randn(T, K, device=dev)
        w = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ws = torch.randint(118, 124, ((N + 31) // 32, K // 32), device=dev, dtype=torch.uint8)
        xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        fq = (xq.float() * xs.float().repeat_interleave(32, 1)).bfloat16()
        (out,), us = packet_op(FP8MX, "plow_sm90a_pfgemm_fp8mx", 384, "plow_arena_bytes_pfgemm_fp8mx", "GemmFp8Mx", [T, N, K],
                               [("out", T * N * 2, None), ("x", T * K * 2, fq), ("w", N * K, w.view(torch.uint8)), ("scale", ws.numel(), ws)],
                               [("out", torch.bfloat16, (T, N))], n_cu=blocks, iters=5)
        ref = kr.fp8_gemm(xq, xs, w, ws.view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
        r = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        tf = 2 * T * N * K / (us * 1e-6) / 1e12
        check(f"pkt GemmFp8Mx T={T} N={N} K={K} blocks={blocks} (role object)", r < 3e-3, f"rel vs fp8_gemm={r:.2e} {us:.1f} us {tf:.0f} TFLOP/s")

if want("pkt_gemm_fp8mx_grouped") and PACKET_OP and FP8MX:
    # i3 = groups (the output LoRA below TP8) and i4/i5 = a_row0/c_row0 (a row band)
    import kernel as kr
    for T, G, N, K, row0, total in ((300, 2, 1024, 4096, 0, 300), (128, 8, 1024, 4096, 0, 128), (128, 2, 1024, 4096, 100, 300)):
        x = torch.randn(total, G * K, device=dev)
        w = (torch.randn(G * N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ws = torch.randint(118, 124, (G * N // 32, K // 32), device=dev, dtype=torch.uint8)
        xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        fq = (xq.float() * xs.float().repeat_interleave(32, 1)).bfloat16()
        (out,), us = packet_op(FP8MX, "plow_sm90a_pfgemm_fp8mx", 384, "plow_arena_bytes_pfgemm_fp8mx", "GemmFp8Mx", [T, N, K, G, row0, row0],
                               [("out", total * G * N * 2, None), ("x", total * G * K * 2, fq), ("w", G * N * K, w.view(torch.uint8)), ("scale", ws.numel(), ws)],
                               [("out", torch.bfloat16, (total, G * N))], iters=3)
        worst = 0.0
        for g in range(G):
            ref = kr.fp8_gemm(xq[row0:row0 + T, g * K:(g + 1) * K].contiguous(), xs[row0:row0 + T, g * K // 32:(g + 1) * K // 32].contiguous(),
                              w[g * N:(g + 1) * N].contiguous(), ws[g * N // 32:(g + 1) * N // 32].contiguous().view(torch.float8_e8m0fnu),
                              torch.float8_e8m0fnu, 32)
            got = out[row0:row0 + T, g * N:(g + 1) * N]
            worst = max(worst, ((got.float() - ref.float()).norm() / ref.float().norm()).item())
        untouched = total == T or bool((out[:row0] == 0).all() and (out[row0 + T:] == 0).all())
        check(f"pkt GemmFp8Mx grouped G={G} T={T} rows {row0}..{row0 + T} of {total}", worst < 3e-3 and untouched,
              f"rel vs per-group fp8_gemm={worst:.2e} band-only={untouched} {us:.1f} us")

def isa_op(op, ints, tensors, outs, floats=(), iters=1, pf=False):
    """on the interpreter object the phase would use: the prefill object (larger arena) from 64 rows"""
    if (pf or ints[0] >= 64) and ISA_PF:
        return packet_op(ISA_PF, "_Z15interp_sm90a_pf11PlowProgram", 256, "plow_arena_bytes_pf", op, ints, tensors, outs, floats=floats,
                         iters=iters)
    return packet_op(ISA, "_Z12interp_sm90a11PlowProgram", 256, "plow_arena_bytes", op, ints, tensors, outs, floats=floats, iters=iters)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def mhc_ref(res, mixes, scale, base, rms_eps, hc_eps, repeat):
    """model.py Block.hc_pre / hc_split_sinkhorn in fp32: pre [T,4], post [T,4], comb [T,4,4]"""
    r = res.float()
    inv = torch.rsqrt(r.flatten(1).square().mean(-1, keepdim=True) + rms_eps)
    m = mixes * inv
    pre = torch.sigmoid(m[:, :4] * scale[0] + base[:4]) + hc_eps
    post = 2 * torch.sigmoid(m[:, 4:8] * scale[1] + base[4:8])
    c = (m[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4)
    c = c.softmax(-1) + hc_eps
    c = c / (c.sum(-2, keepdim=True) + hc_eps)
    for _ in range(repeat - 1):
        c = c / (c.sum(-1, keepdim=True) + hc_eps)
        c = c / (c.sum(-2, keepdim=True) + hc_eps)
    return pre, post, c


if want("pkt_hyperconn") and PACKET_OP and ISA:
    torch.manual_seed(1)
    H, rep = 5120, 20
    for T, mode in ((1, 0), (37, 1), (300, 2)):
        res = (torch.randn(T, 4, H, device=dev) * 3).bfloat16()
        mixes = torch.randn(T, 24, device=dev, dtype=torch.float32) * 50
        scale = torch.tensor([0.9, 1.1, 0.7], device=dev, dtype=torch.float32)
        base = torch.randn(24, device=dev, dtype=torch.float32) * 0.1
        pair = torch.rand(2, T, 4, device=dev, dtype=torch.float32)
        in_half = 1
        outs, us = isa_op("HyperConnPre", [T, 4, H, rep, in_half, mode],
                          [("post", T * 16, None), ("comb", T * 64, None), ("li", T * H * 2, None), ("mixes", T * 96, mixes),
                           ("res", T * 4 * H * 2, res), ("scale", 12, scale), ("base", 96, base), ("pair", 2 * T * 16, pair)],
                          [("post", torch.float32, (T, 4)), ("comb", torch.float32, (T, 4, 4)), ("li", torch.bfloat16, (T, H)),
                           ("pair", torch.float32, (2, T, 4))], floats=(1e-6, 1e-6))
        post, comb, li, pair_out = outs
        pre_r, post_r, comb_r = mhc_ref(res, mixes, scale, base, 1e-6, 1e-6, rep)
        gate = pre_r if mode == 0 else (torch.tensor([1.0, 0, 0, 0], device=dev).expand(T, 4) if mode == 1 else pair[in_half])
        li_r = (gate.unsqueeze(-1) * res.float()).sum(1)
        e = {"post": rel(post, post_r), "comb": rel(comb, comb_r), "li": rel(li, li_r)}
        if mode != 0:
            e["pre_pair"] = rel(pair_out[in_half ^ 1], pre_r)
            e["kept"] = rel(pair_out[in_half], pair[in_half])
        ok = e["post"] < 1e-5 and e["comb"] < 1e-5 and e["li"] < 4e-3 and e.get("pre_pair", 0) < 1e-5 and e.get("kept", 0) == 0
        check(f"pkt HyperConnPre T={T} pre_mode={mode}", ok, " ".join(f"{k}={v:.2e}" for k, v in e.items()) + f" {us:.1f} us")
    for T in (1, 300):
        res = (torch.randn(T, 4, H, device=dev) * 3).bfloat16()
        x = torch.randn(T, H, device=dev).bfloat16()
        post = torch.rand(T, 4, device=dev, dtype=torch.float32) * 2
        comb = torch.rand(T, 4, 4, device=dev, dtype=torch.float32)
        (new,), us = isa_op("HyperConnPost", [T, 4, H, 0],
                            [("new", T * 4 * H * 2, None), ("x", T * H * 2, x), ("res", T * 4 * H * 2, res), ("post", T * 16, post), ("comb", T * 64, comb)],
                            [("new", torch.bfloat16, (T, 4, H))])
        ref = (comb.unsqueeze(-1) * res.float().unsqueeze(2)).sum(1) + post.unsqueeze(-1) * x.float().unsqueeze(1)
        r = rel(new, ref)
        check(f"pkt HyperConnPost T={T}", r < 4e-3, f"rel={r:.2e} {us:.1f} us")

if want("pkt_gemm_f32") and PACKET_OP and ISA:
    for M, N, K in ((1, 24, 20480), (301, 24, 20480), (1024, 24, 20480), (7, 40, 1000)):
        x = torch.randn(M, K, device=dev).bfloat16()
        w = torch.randn(N, K, device=dev, dtype=torch.float32) * 0.01
        (c,), us = isa_op("GemvF32", [M, N, K], [("c", M * N * 4, None), ("x", M * K * 2, x), ("w", N * K * 4, w)],
                          [("c", torch.float32, (M, N))], iters=3)
        r = rel(c, x.float() @ w.T)
        check(f"pkt GemvF32 M={M} N={N} K={K}", r < 1e-5, f"rel={r:.2e} {us:.1f} us")
    for M, N, K in ((1, 384, 5120), (300, 384, 5120), (1024, 384, 5120), (1024, 256, 5120), (1024, 512, 5120), (4096, 512, 5120)):
        a = torch.randn(M, K, device=dev).bfloat16()
        w = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
        (c,), us = isa_op("GemmF32", [M, N, K], [("c", M * N * 4, None), ("a", M * K * 2, a), ("w", N * K * 2, w)],
                          [("c", torch.float32, (M, N))], iters=3)
        r = rel(c, a.float() @ w.float().T)
        check(f"pkt GemmF32 M={M} N={N} K={K}", r < 1e-5, f"rel={r:.2e} {us:.1f} us")

def freqs(npos, rd):
    inv = 1.0 / (10000 ** (torch.arange(0, rd, 2, device=dev, dtype=torch.float32) / rd))
    f = torch.polar(torch.ones(npos, rd // 2, device=dev, dtype=torch.float32), torch.outer(torch.arange(npos, device=dev, dtype=torch.float32), inv))
    return f, f.real.contiguous(), f.imag.contiguous()


if want("pkt_compress") and PACKET_OP and ISA:
    import kernel as kr
    import model as mr
    torch.manual_seed(2)
    # CompressPool arm 2: softmax pool over `ratio` slots, bf16, RMSNorm(gamma), bf16 (Compressor.forward)
    for n_pools, ratio, d in ((1, 4, 512), (75, 4, 512), (16, 128, 512)):
        T = n_pools * ratio
        kv = torch.randn(T, d, device=dev).bfloat16()
        sc = (torch.randn(T, d, device=dev) * 3).bfloat16()
        g = (torch.rand(d, device=dev) + 0.5).bfloat16()
        (out,), us = isa_op("CompressPool", [n_pools, ratio, 1, d, 0, 0, 0, 2],
                            [("out", n_pools * d * 2, None), ("kv", T * d * 2, kv), ("score", T * d * 2, sc), ("-", 0, None), ("gamma", d * 2, g)],
                            [("out", torch.bfloat16, (n_pools, d))], floats=(1e-6,), iters=3)
        pooled = (kv.float().view(n_pools, ratio, d) * sc.float().view(n_pools, ratio, d).softmax(1)).sum(1).bfloat16()
        norm = mr.RMSNorm(d, 1e-6).to(dev)
        norm.weight.data.copy_(g)
        ref = norm(pooled)
        bad = (out != ref).sum().item()
        check(f"pkt CompressPool n_pools={n_pools} ratio={ratio} d={d}", bad <= n_pools * d // 1000, f"mismatch={bad} rel={rel(out, ref):.2e} {us:.1f} us")
    # CompressRopeQuant: the cache row (fp4/16, E4M3 scale), the index keys and queries (fp4/32, E8M0)
    for name, rows, heads, d, rd, qblk, ratio, qmode, sdt in (("cache", 75, 1, 512, 64, 16, 4, 2, torch.float8_e4m3fn),
                                                             ("keys", 75, 1, 128, 64, 32, 4, 1, torch.float8_e8m0fnu),
                                                             ("queries", 300, 32, 128, 64, 32, 1, 1, torch.float8_e8m0fnu),
                                                             ("fp8", 40, 2, 512, 64, 32, 1, 0, None)):
        f, cs, sn = freqs(rows * ratio, rd)
        x = (torch.randn(rows, heads, d, device=dev) * torch.logspace(-3, 2, rows, device=dev).view(-1, 1, 1)).bfloat16()
        x[0, 0, :qblk] = 0
        (out,), us = isa_op("CompressRopeQuant", [rows, d, rd, qblk, ratio, 0, qmode, heads],
                            [("out", x.numel() * 2, None), ("src", x.numel() * 2, x), ("cos", cs.numel() * 4, cs), ("sin", sn.numel() * 4, sn)],
                            [("out", torch.bfloat16, (rows, heads, d))], iters=3)
        ref = x.clone().unsqueeze(0)
        mr.apply_rotary_emb(ref[..., -rd:], f[::ratio][:rows])
        ref = ref.squeeze(0)
        if qmode:
            kr.fp4_act_quant(ref.view(-1, d), qblk, True, scale_dtype=sdt)
        else:
            ref = kr.act_quant(ref.reshape(-1, d).contiguous(), qblk, "ue8m0", torch.float8_e8m0fnu, True).view(rows, heads, d)
        bad = (out != ref).sum().item()
        check(f"pkt CompressRopeQuant {name} rows={rows} heads={heads} d={d} qblk={qblk} qmode={qmode}", bad <= x.numel() // 10000,
              f"mismatch={bad} rel={rel(out, ref):.2e} {us:.1f} us")
    # RopeInverseO: de-rotate the attention output by the query position (apply_rotary_emb inverse=True)
    for T, H, D, rd, pos0 in ((1, 64, 512, 64, 777), (300, 64, 512, 64, 0)):
        f, cs, sn = freqs(pos0 + T, rd)
        o = torch.randn(T, H, D, device=dev).bfloat16()
        (out,), us = isa_op("RopeInverseO", [T, H, D, rd, pos0],
                            [("o", o.numel() * 2, o), ("cos", cs.numel() * 4, cs), ("sin", sn.numel() * 4, sn)],
                            [("o", torch.bfloat16, (T, H, D))], iters=1)
        ref = o.clone().unsqueeze(0)
        mr.apply_rotary_emb(ref[..., -rd:], f[pos0:pos0 + T], inverse=True)
        ref = ref.squeeze(0)
        bad = (out != ref).sum().item()
        check(f"pkt RopeInverseO T={T} H={H} pos0={pos0}", bad <= o.numel() // 1000, f"mismatch={bad} rel={rel(out, ref):.2e} {us:.1f} us")

if want("pkt_index") and PACKET_OP and ISA_PF:
    torch.manual_seed(3)
    kl = lambda n: torch.tensor([n], dtype=torch.int32, device=dev)
    # op 117: f32 score = scale * sum_h w * relu(q . k), columns are pools, causal bound (q_pos0+t+1)//pool
    for T, prior, pool in ((64, 0, 4), (300, 100, 4), (2048, 0, 4)):
        L = prior + T
        S = L // pool
        q = torch.randn(T, 32, 128, device=dev).bfloat16()
        k = torch.randn(S, 128, device=dev).bfloat16()
        w = torch.randn(T, 32, device=dev).bfloat16()
        scale = 128 ** -0.5 * 32 ** -0.5
        stride = S + 16
        (sc,), us = isa_op("IndexScorePf", [T, 32, stride, 128, pool],
                           [("score", T * stride * 4, torch.zeros(T, stride, device=dev, dtype=torch.float32)), ("q", q.numel() * 2, q),
                            ("k", k.numel() * 2, k), ("w", w.numel() * 2, w), ("kv_len", 4, kl(L))],
                           [("score", torch.float32, (T, stride))], floats=(scale,), iters=3, pf=True)
        ref = (torch.einsum("thd,sd->ths", q.double(), k.double()).relu() * w.double().unsqueeze(-1)).sum(1) * scale
        bound = (prior + torch.arange(T, device=dev) + 1) // pool
        valid = torch.arange(S, device=dev).unsqueeze(0) < bound.unsqueeze(1)
        err = ((sc[:, :S] - ref) * valid).abs().max().item() / ref.abs().max().item()
        untouched = bool((sc[:, :S][~valid] == 0).all()) and bool((sc[:, S:] == 0).all())
        gf = 2 * valid.sum().item() * 32 * 128 / (us * 1e-6) / 1e9
        check(f"pkt IndexScorePf T={T} prior={prior} pool={pool}", err < 1e-5 and untouched, f"max_err/max={err:.2e} bounded={untouched} {us:.1f} us {gf:.0f} GFLOP/s")

    # op 118: exact top-k, score desc then lowest position, ascending emit, identity + -1 when short
    def topk_ref(sc, T, prior, pool, k):
        out = torch.full((T, k), -1, dtype=torch.int32)
        scc = sc.cpu()
        for t in range(T):
            n = (prior + t + 1) // pool
            if n <= k:
                out[t, :n] = torch.arange(n)
                continue
            r = scc[t, :n].double()
            order = sorted(range(n), key=lambda i: (-r[i].item(), i))[:k]
            out[t] = torch.tensor(sorted(order), dtype=torch.int32)
        return out
    for T, prior, pool, k in ((64, 0, 4, 8), (300, 2000, 4, 512), (1024, 0, 1, 512)):
        stride = (prior + T) // pool + 8
        sc = torch.randn(T, stride, device=dev, dtype=torch.float32)
        sc[:, ::7] = 0.5  # exact ties across the threshold
        sc[:, 3::11] = -0.0
        (ix,), us = isa_op("IndexSelectPf", [T, k, stride, pool],
                           [("idx", T * k * 4, None), ("score", sc.numel() * 4, sc), ("kv_len", 4, kl(prior + T))],
                           [("idx", torch.int32, (T, k))], iters=3, pf=True)
        ref = topk_ref(sc, T, prior, pool, k)
        bad = (ix.cpu() != ref).any(1).sum().item()
        check(f"pkt IndexSelectPf T={T} prior={prior} pool={pool} k={k}", bad == 0, f"bad_rows={bad} {us:.1f} us")

    # op 119: per-P-query union, ascending positions + membership masks
    for T, prior, k, P, stride in ((300, 2000, 512, 8, 8192), (64, 0, 16, 64, 64)):
        cap = min(P * k, stride)
        idx = torch.full((T, k), -1, dtype=torch.int32, device=dev)
        for t in range(T):
            n = min((prior + t + 1) // 4, stride)
            m = min(n, k)
            idx[t, :m] = torch.randperm(n, device=dev)[:m].sort().values.int()
        n_qt = (T + P - 1) // P
        hdr = (n_qt * 4 + 255) // 256 * 256
        nbytes = hdr + n_qt * cap * 12 + 8
        (uni,), us = isa_op("IndexUnionPf", [T, k, stride, cap, P],
                            [("uni", nbytes, None), ("umask", 132 * stride * 8, None), ("idx", idx.numel() * 4, idx), ("kv_len", 4, kl(prior + T))],
                            [("uni", torch.uint8, (nbytes,))], iters=1, pf=True)
        ok = True
        cnt = uni[:n_qt * 4].view(torch.int32).cpu()
        for qt in range(n_qt):
            m = {}
            for ql in range(P):
                t = qt * P + ql
                if t >= T:
                    break
                for s_ in idx[t].tolist():
                    if s_ >= 0:
                        m[s_] = m.get(s_, 0) | (1 << ql)
            keys = sorted(m)
            base = hdr + qt * cap * 12
            blk = uni[base:base + cap * 12].view(torch.int32).cpu()
            c = cnt[qt].item()
            pos = blk[:c].tolist()
            lo = blk[cap:cap + c].tolist()
            hi = blk[2 * cap:2 * cap + c].tolist()
            got = [(p_, (l_ & 0xffffffff) | ((h_ & 0xffffffff) << 32)) for p_, l_, h_ in zip(pos, lo, hi)]
            if c != len(keys) or got != [(s_, m[s_]) for s_ in keys]:
                ok = False
                break
        check(f"pkt IndexUnionPf T={T} k={k} P={P}", ok, f"{us:.1f} us")

def union_table(idx, P, cap):
    """IndexUnionPf's table for per-query selections idx [T][k] (-1 pad)"""
    T = idx.shape[0]
    n_qt = (T + P - 1) // P
    hdr = (n_qt * 4 + 255) // 256 * 256
    buf = torch.zeros(hdr + n_qt * cap * 12 + 8, dtype=torch.uint8)
    cnt = buf[:n_qt * 4].view(torch.int32)
    ic = idx.cpu()
    for qt in range(n_qt):
        m = {}
        for ql in range(P):
            t = qt * P + ql
            if t < T:
                for s_ in ic[t].tolist():
                    if s_ >= 0:
                        m[s_] = m.get(s_, 0) | (1 << ql)
        keys = sorted(m)[:cap]
        cnt[qt] = len(keys)
        blk = buf[hdr + qt * cap * 12:hdr + (qt + 1) * cap * 12].view(torch.int32)
        blk[:len(keys)] = torch.tensor(keys, dtype=torch.int32)
        blk[cap:cap + len(keys)] = torch.tensor([m[k_] & 0xffffffff for k_ in keys], dtype=torch.int64).to(torch.int32)
    return buf


if want("pkt_flash") and PACKET_OP and ISA_PF:
    import kernel as kr
    torch.manual_seed(4)
    D, W = 512, 128
    for T, H, n_cmp, k, gsplit in ((64, 16, 0, 0, 1), (300, 16, 75, 64, 2), (1024, 16, 256, 256, 2), (129, 64, 32, 32, 1), (2048, 16, 512, 512, 2)):
        q = (torch.randn(T, H, D, device=dev) * 0.5).bfloat16()
        kv = torch.randn(T, D, device=dev).bfloat16()
        sink = torch.randn(H, device=dev, dtype=torch.float32)
        scale = D ** -0.5
        nsplit = 1 + (gsplit if n_cmp else 0)
        pos = torch.arange(T, device=dev)
        widx = pos.unsqueeze(1) - torch.arange(W - 1, -1, -1, device=dev).unsqueeze(0)
        widx = torch.where(widx >= 0, widx, torch.full_like(widx, -1)).int()
        kl = torch.tensor([T], dtype=torch.int32, device=dev)
        osz, msz = T * H * nsplit * D * 4, T * H * nsplit * 8
        word = (nsplit << 8) if nsplit > 1 else 0
        (op_, ml_), us_w = isa_op("FlashMlaPrefill", [1, H, T, (1 << 31) | W, T, 0xFFFFFFFF, 0, word],
                                  [("opart", osz, None), ("ml", msz, None), ("q", q.numel() * 2, q), ("-", 0, None), ("kv", kv.numel() * 2, kv),
                                   ("-", 0, None), ("kv_len", 4, kl)],
                                  [("opart", torch.float32, (osz // 4,)), ("ml", torch.float32, (msz // 4,))], floats=(scale,), iters=3, pf=True)
        us_g = 0.0
        full_kv, full_idx = kv, widx
        if n_cmp:
            cache = torch.randn(n_cmp, D, device=dev).bfloat16()
            cidx = torch.full((T, k), -1, dtype=torch.int32, device=dev)
            for t in range(T):
                n = min((t + 1) // 4, n_cmp)
                m = min(n, k)
                if m:
                    cidx[t, :m] = torch.randperm(n, device=dev)[:m].sort().values.int()
            cap = min(8 * k, n_cmp)
            uni = union_table(cidx, 8, cap).to(dev)
            (op_, ml_), us_g = isa_op("FlashMlaPrefill", [1, H, n_cmp, 1 << 31, T, 0xFFFFFFFF, cap, (gsplit << 16) | (nsplit << 8) | 1],
                                      [("opart", osz, op_), ("ml", msz, ml_), ("q", q.numel() * 2, q), ("-", 0, None),
                                       ("cache", cache.numel() * 2, cache), ("-", 0, None), ("kv_len", 4, kl), ("uni", uni.numel(), uni)],
                                      [("opart", torch.float32, (osz // 4,)), ("ml", torch.float32, (msz // 4,))], floats=(scale,), iters=3, pf=True)
            full_kv = torch.cat([kv, cache])
            full_idx = torch.cat([widx, torch.where(cidx >= 0, cidx + T, cidx)], 1)
        (o,), us_m = isa_op("FlashMerge", [T, H, nsplit, D, 1],
                            [("o", T * H * D * 2, None), ("opart", osz, op_), ("ml", msz, ml_), ("sink", H * 4, sink)],
                            [("o", torch.bfloat16, (T, H, D))], iters=3, pf=True)
        ref = kr.sparse_attn(q.unsqueeze(0), full_kv.unsqueeze(0), sink, full_idx.unsqueeze(0).contiguous(), scale).squeeze(0)
        r = rel(o, ref)
        nk = (full_idx >= 0).sum().item()
        tf = 4 * nk * H * D / ((us_w + us_g) * 1e-6) / 1e12
        check(f"pkt FlashMlaPrefill+FlashMerge T={T} H={H} window={W} cmp={n_cmp} k={k} gsplit={gsplit}", r < 5e-3,
              f"rel vs sparse_attn={r:.2e} window {us_w:.1f} us gather {us_g:.1f} us merge {us_m:.1f} us {tf:.0f} TFLOP/s")

if want("pkt_misc") and PACKET_OP and ISA_PF:
    import model as mr
    torch.manual_seed(5)
    # Glu act 4: DeepSeek's clamped SwiGLU, silu(min(g, 10)) * clamp(u, -10, 10) (model.py Expert)
    n = 1024 * 2304 + 5
    g = (torch.randn(n, device=dev) * 8).bfloat16()
    u = (torch.randn(n, device=dev) * 8).bfloat16()
    (o,), us = isa_op("Glu", [n, 4], [("o", n * 2, None), ("g", n * 2, g), ("u", n * 2, u)], [("o", torch.bfloat16, (n,))],
                      floats=(0.0, 10.0), iters=3, pf=True)
    ref = (torch.nn.functional.silu(g.float().clamp(max=10)) * u.float().clamp(-10, 10)).bfloat16()
    bad = (o != ref).sum().item()
    check(f"pkt Glu act=4 n={n}", bad <= n // 10000, f"mismatch={bad} {us:.1f} us {n * 6 / us / 1e3:.0f} GB/s")
    # QwenHeadNormRope in V4.1's form: interleaved rope of [448, 512) at pos[t], no norm
    for T, H in ((1024, 64), (37, 1)):
        x = torch.randn(T, H, 512, device=dev).bfloat16()
        pos = (torch.arange(T, device=dev, dtype=torch.int32) + 3)
        f, cs, sn = freqs(T + 3, 64)
        (o,), us = isa_op("QwenHeadNormRope", [H, 512, (1 << 31) | 64, T, 0, 0, 1, 448],
                          [("o", x.numel() * 2, None), ("x", x.numel() * 2, x), ("-", 0, None), ("cos", cs.numel() * 4, cs),
                           ("sin", sn.numel() * 4, sn), ("pos", T * 4, pos)], [("o", torch.bfloat16, (T, H, 512))], iters=3, pf=True)
        ref = x.clone().unsqueeze(0)
        mr.apply_rotary_emb(ref[..., -64:], f[3:3 + T])
        ref = ref.squeeze(0)
        bad = (o != ref).sum().item()
        check(f"pkt QwenHeadNormRope interleaved suffix T={T} H={H}", bad <= x.numel() // 10000, f"mismatch={bad} {us:.1f} us")

if want("pkt_engram") and PACKET_OP and ISA:
    torch.manual_seed(7)
    # EngramEmbed: 24 ids per token into an fp8 [rows][256] table with ue8m0 per 32; ids past this
    # shard write zeros
    T, ncol, hd, rows = 37, 24, 256, 5000
    tab = (torch.randn(rows, hd, device=dev) * 2).to(torch.float8_e4m3fn)
    sc = torch.randint(110, 135, (rows, hd // 32), device=dev, dtype=torch.uint8)
    ids = torch.randint(0, rows + 100, (T, ncol), device=dev, dtype=torch.int32)
    (o,), _ = isa_op("EngramEmbed", [T, ncol, hd, 32, 0, rows], [("o", T * ncol * hd * 2, None), ("tab", rows * hd, tab.view(torch.uint8)),
                     ("sc", rows * hd // 32, sc), ("ids", T * ncol * 4, ids)], [("o", torch.bfloat16, (T, ncol * hd))])
    valid = (ids < rows).unsqueeze(-1)
    li = ids.clamp(max=rows - 1).long()
    ref = (tab.float()[li] * torch.exp2(sc.float()[li] - 127).repeat_interleave(32, -1)) * valid
    ref = ref.flatten(1).bfloat16()
    check(f"pkt EngramEmbed T={T}", torch.equal(o, ref), f"mismatch={(o != ref).sum().item()}")
    # EngramGate: x [T][4][H] += sigmoid(signed sqrt dot) * value, in place
    T, n, H = 37, 4, 5120
    x = torch.randn(T, n, H, device=dev).bfloat16()
    kv = torch.randn(T, (n + 1) * H, device=dev).bfloat16()
    qw = (1 + 0.1 * torch.randn(n, H, device=dev)).bfloat16()
    kw = (1 + 0.1 * torch.randn(n, H, device=dev)).bfloat16()
    (o,), _ = isa_op("EngramGate", [T, n, H], [("x", x.numel() * 2, x), ("kv", kv.numel() * 2, kv), ("qw", qw.numel() * 2, qw),
                     ("kw", kw.numel() * 2, kw)], [("x", torch.bfloat16, (T, n, H))], floats=(1e-20,))
    key, value = kv.float().split([n * H, H], -1)
    key = key.unflatten(-1, (n, H))
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(key.square().mean(-1) + 1e-20)
    dot = (h * (qw.float() * kw.float()) * key).sum(-1) * rstd * H ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    ref = (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).bfloat16()
    check(f"pkt EngramGate T={T}", rel(o, ref) < 2e-3, f"rel={rel(o, ref):.1e} gate_mean={gate.mean().item():.3f}")
    # ArgmaxF32 over a vocab row; Embed with rep=4 (the hc copies)
    V = 129280
    lg = torch.randn(3, V, device=dev, dtype=torch.float32)
    lg[1, 777] = lg[1, 778] = 100.0
    (am,), _ = isa_op("ArgmaxF32", [3, V], [("ids", 12, None), ("x", 3 * V * 4, lg)], [("ids", torch.int32, (3,))])
    check("pkt ArgmaxF32", am.tolist() == [lg[0].argmax().item(), 777, lg[2].argmax().item()], f"{am.tolist()}")
    tabe = torch.randn(1000, 5120, device=dev).bfloat16()
    tok = torch.randint(0, 1000, (9,), device=dev, dtype=torch.int32)
    (e,), _ = isa_op("Embed", [9, 5120, 4], [("o", 9 * 4 * 5120 * 2, None), ("tab", tabe.numel() * 2, tabe), ("ids", 36, tok)],
                     [("o", torch.bfloat16, (9, 4, 5120))], floats=(1.0,))
    check("pkt Embed rep=4", torch.equal(e, tabe[tok.long()].unsqueeze(1).expand(9, 4, 5120)), "")

if want("pkt_router") and PACKET_OP and ISA_PF_CFG:
    torch.manual_seed(6)
    for T in (1, 1024, 4096):
        E, k = 384, 6
        logit = torch.randn(T, E, device=dev, dtype=torch.float32) * 2
        bias = torch.randn(E, device=dev, dtype=torch.float32) * 0.1
        (tab,), us = packet_op(ISA_PF_CFG, "_Z15interp_sm90a_pf11PlowProgram", 256, "plow_arena_bytes_pf", "MoeRouterTopkPf",
                               [0, E, k, 46, T, 0], [("tab", T * k * 8, None), ("logit", T * E * 4, logit), ("-", 0, None), ("bias", E * 4, bias)],
                               [("tab", torch.int32, (T, k, 2))], floats=(1.5,), iters=3)
        sc = torch.nn.functional.softplus(logit.double()).sqrt()
        idx = (sc + bias.double()).topk(k, dim=-1)[1]
        w = sc.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * 1.5
        gi = tab[..., 0]
        gw = tab[..., 1].view(torch.float32).double()
        same = (gi.sort(1).values == idx.int().sort(1).values).all().item()
        werr = (gw.gather(1, gi.long().argsort(1)) - w.gather(1, idx.argsort(1))).abs().max().item() if same else 1.0
        check(f"pkt MoeRouterTopkPf V4.1 (sqrtsoftplus, f32 logit, bias) T={T}", same and werr < 1e-5, f"indices_equal={same} max_w_err={werr:.1e} {us:.1f} us")

if want("pkt_align") and PACKET_OP and ISA_PF_CFG:
    torch.manual_seed(7)
    BM = 64
    for T, E, k, npart in ((1024, 384, 6, 64), (4096, 384, 6, 64), (3, 384, 6, 64)):
        ns = T * k
        ex = torch.randint(0, E, (T, k), dtype=torch.int32)
        ex[: T // 3, 0] = 5  # a hot expert
        tab = torch.stack([ex, torch.rand(T, k, dtype=torch.float32).view(torch.int32)], -1).contiguous()
        msz = (3 * E + 1 + npart * E) * 4
        cnt = torch.bincount(ex.flatten().long(), minlength=E)
        tiles = (cnt + BM - 1) // BM
        rsz = int(tiles.sum()) * BM + BM
        state = [("meta", msz, torch.zeros(msz // 4, dtype=torch.int32, device=dev)), ("tab", ns * 8, tab.to(dev)),
                 ("rowtok", rsz * 4, torch.zeros(rsz, dtype=torch.int32, device=dev)), ("rowpart", rsz * 4, torch.zeros(rsz, dtype=torch.int32, device=dev)),
                 ("rowgate", rsz * 4, torch.zeros(rsz, dtype=torch.float32, device=dev))]
        tot_us = 0.0
        for ph in (1, 2, 3, 4):
            outs, us = packet_op(ISA_PF_CFG, "_Z15interp_sm90a_pf11PlowProgram", 256, "plow_arena_bytes_pf", "MoeAlignPf", [T, E, k, ph, npart],
                                 state, [("meta", torch.int32, (msz // 4,)), ("rowtok", torch.int32, (rsz,)), ("rowpart", torch.int32, (rsz,)),
                                         ("rowgate", torch.float32, (rsz,))], n_cu=1 if ph == 2 else npart)
            tot_us += us
            state = [("meta", msz, outs[0]), state[1], ("rowtok", rsz * 4, outs[1]), ("rowpart", rsz * 4, outs[2]), ("rowgate", rsz * 4, outs[3])]
        meta, rt, rp, rg = [x.cpu() for x in outs]
        tp = torch.cat([torch.zeros(1, dtype=torch.long), tiles.cumsum(0)])
        ok = torch.equal(meta[E:2 * E], cnt.int()) and torch.equal(meta[2 * E:3 * E + 1], tp.int()) and torch.equal(meta[:E], (tp[:E] * BM).int())
        flat = ex.flatten()
        gates = tab[..., 1].flatten().view(torch.float32)
        exp_tok = torch.full((int(tp[-1]) * BM,), -1, dtype=torch.int32)
        exp_part = exp_tok.clone()
        exp_gate = torch.zeros(int(tp[-1]) * BM, dtype=torch.float32)
        for e in range(E):
            sl = (flat == e).nonzero().flatten()
            o = int(tp[e]) * BM
            exp_part[o:o + len(sl)] = sl.int()
            exp_tok[o:o + len(sl)] = (sl // k).int()
            exp_gate[o:o + len(sl)] = gates[sl]
        n = int(tp[-1]) * BM
        det = (f"cnt={torch.equal(meta[E:2 * E], cnt.int())} tilep={torch.equal(meta[2 * E:3 * E + 1], tp.int())} "
               f"rowoff={torch.equal(meta[:E], (tp[:E] * BM).int())} tok={torch.equal(rt[:n], exp_tok)} part={torch.equal(rp[:n], exp_part)} "
               f"gate={torch.equal(rg[:n], exp_gate)}")
        ok = ok and torch.equal(rt[:n], exp_tok) and torch.equal(rp[:n], exp_part) and torch.equal(rg[:n], exp_gate)
        check(f"pkt MoeAlignPf phased T={T} E={E} k={k} npart={npart}", ok, f"{det} {tot_us:.1f} us over 4 packets")

print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
