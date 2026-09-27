"""DeepSeek-V4.1 routed experts on sm_90a: MoeGroupGluPf / MoeGroupDownPf (ops 85/86, MXFP4) in the
prefill role object runtime/nvidia/interp_sm90a_pfmoe_fp4.cu, inside a real packet (packet_op),
against the reference Expert.forward (inference/model.py + kernel.py act_quant / fp4_gemm).

  PACKET_OP=... PFMOE_CUBIN=... perf-data/tools/gpulease -n 1 dsv41-pfmoe <venv-python> \
      scripts/dsv41_nv/test_packet_moe.py [test ...]      tests: glu down perf
"""
import os
import subprocess
import sys
import tempfile

import torch

REF = os.environ.get(
    "DSV41_REF",
    "/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/inference",
)
sys.path.insert(0, REF)
import kernel as kr  # noqa: E402

torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
PACKET_OP = os.environ["PACKET_OP"]
OBJ = os.environ["PFMOE_CUBIN"]
only = set(sys.argv[1:])
tmp = tempfile.mkdtemp(prefix="pfmoe_", dir=os.environ.get("PFMOE_TMP"))
results = []
UNUSED = 0xFFFFFFFF
LIM = 10.0  # config.json swiglu_limit
BM = 64  # MoeAlignPf's per-expert padding on NVIDIA (op_moe.cuh d_moe_align_pf_nv)


def want(name):
    return not only or name in only


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def dump(t, name):
    f = os.path.join(tmp, name)
    t.contiguous().cpu().view(torch.uint8).numpy().tofile(f)
    return f


def packet_op(op, ints, floats, operands, extra, ptrs, outs, blocks=132, iters=1):
    cmd = [PACKET_OP, "--object", OBJ, "--symbol", "plow_sm90a_pfmoe_fp4", "--block", "384", "--arena-symbol",
           "plow_arena_bytes_pfmoe_fp4", "--op", op, "--n-cu", str(blocks), "--i", ",".join(map(str, ints)),
           "--iters", str(iters)]
    if floats:
        cmd += ["--f", ",".join(repr(float(f)) for f in floats)]
    for name, nbytes, f in operands:
        cmd += ["--t", f"{name}:{nbytes}" + (f":{f}" if f else "")]
    for name, nbytes, f in extra:
        cmd += ["--x", f"{name}:{nbytes}:{f}"]
    for name, lst in ptrs:
        cmd += ["--ptrs", f"{name}={','.join(lst)}"]
    files = []
    for k, (name, dt, shape) in enumerate(outs):
        f = os.path.join(tmp, f"out{k}.bin")
        files.append((f, dt, shape))
        cmd += ["--out", f"{name}={f}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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


class Case:
    """T tokens routed top-k over E experts (I = per-rank intermediate), aligned like MoeAlignPf."""

    def __init__(self, T, E, k, H, I, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.T, self.E, self.k, self.H, self.I = T, E, k, H, I
        idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])  # [T, k]
        wt = torch.rand(T, k, generator=g, dtype=torch.float32) * 0.5 + 0.05
        cnt = torch.bincount(idx.flatten(), minlength=E)
        padded = (cnt + BM - 1) // BM * BM
        rowoff = torch.cumsum(padded, 0) - padded
        self.rows = int(padded.sum())
        tok = torch.full((self.rows,), UNUSED, dtype=torch.int64)
        part = torch.full((self.rows,), UNUSED, dtype=torch.int64)
        gate = torch.zeros(self.rows, dtype=torch.float32)
        fill = rowoff.clone()
        flat = idx.flatten()
        for s in range(T * k):  # slot order within an expert, as the align op scans
            e = int(flat[s])
            at = int(fill[e])
            fill[e] += 1
            tok[at], part[at], gate[at] = s // k, s, wt.flatten()[s]
        tilep = torch.zeros(E + 1, dtype=torch.int64)
        tilep[1:] = torch.cumsum(padded // BM, 0)
        self.meta = torch.cat([rowoff, cnt, tilep]).to(torch.int32)
        self.tok, self.part, self.gate = tok.to(torch.uint32), part.to(torch.uint32), gate
        self.cnt, self.rowoff = cnt, rowoff
        # weights: fp4 bytes (all 16 codes) and ue8m0 scales near 2^-6, one packed tensor per matrix
        self.w13 = torch.randint(0, 256, (E, 2, I, H // 2), generator=g, dtype=torch.uint8)
        self.s13 = torch.randint(119, 124, (E, 2, I, H // 32), generator=g, dtype=torch.uint8)
        self.w2 = torch.randint(0, 256, (E, H, I // 2), generator=g, dtype=torch.uint8)
        self.s2 = torch.randint(119, 124, (E, H, I // 32), generator=g, dtype=torch.uint8)
        x = torch.randn(T, H, generator=g).bfloat16()
        xq, xs = kr.act_quant(x.to(dev), 32, "ue8m0", torch.float8_e8m0fnu)
        self.xq, self.xs = xq, xs
        self.fq = (xq.float() * xs.float().repeat_interleave(32, 1)).bfloat16()  # ActQuantMx's output

    def ptr_tables(self):
        E, I, H = self.E, self.I, self.H
        wb, sb = I * H // 2, I * H // 32
        wt = [f"w13+{(2 * e + m) * wb}" for e in range(E) for m in range(2)]
        st = [f"s13+{(2 * e + m) * sb}" for e in range(E) for m in range(2)]
        wtab = sum(([wt[2 * e], wt[2 * e + 1], f"w2+{e * H * I // 2}"] for e in range(E)), [])
        stab = sum(([st[2 * e], st[2 * e + 1], f"s2+{e * H * I // 32}"] for e in range(E)), [])
        return [("wtab", wtab), ("stab", stab)]

    def extra(self):
        if not hasattr(self, "_files"):
            self._files = [("w13", self.w13.numel(), dump(self.w13, "w13.bin")), ("s13", self.s13.numel(), dump(self.s13, "s13.bin")),
                           ("w2", self.w2.numel(), dump(self.w2, "w2.bin")), ("s2", self.s2.numel(), dump(self.s2, "s2.bin"))]
        return self._files

    def fp4(self, x_q, x_s, w, s):
        return kr.fp4_gemm(x_q.contiguous(), x_s.contiguous(), w.to(dev).contiguous().view(torch.float4_e2m1fn_x2),
                           s.to(dev).contiguous().view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, act_block_size=32)

    def ref_glu(self):
        """fu per gathered row (Expert.forward up to w2's act_quant), zeros on pad rows"""
        fu = torch.zeros(self.rows, self.I, device=dev)
        for e in range(self.E):
            n = int(self.cnt[e])
            if n == 0:
                continue
            r0 = int(self.rowoff[e])
            toks = self.tok[r0:r0 + n].to(torch.int64).to(dev)
            gq = self.xq.view(torch.uint8)[toks].view(torch.float8_e4m3fn)
            gs = self.xs.view(torch.uint8)[toks].view(torch.float8_e8m0fnu)
            g = self.fp4(gq, gs, self.w13[e, 0], self.s13[e, 0]).float()
            u = self.fp4(gq, gs, self.w13[e, 1], self.s13[e, 1]).float()
            u = torch.clamp(u, min=-LIM, max=LIM)
            g = torch.clamp(g, max=LIM)
            h = (self.gate[r0:r0 + n].to(dev).unsqueeze(1) * (torch.nn.functional.silu(g) * u)).bfloat16()
            fu[r0:r0 + n] = kr.act_quant(h.contiguous(), 32, "ue8m0", torch.float8_e8m0fnu, True)
        return fu

    def ref_down(self, fu):
        part = torch.zeros(self.T * self.k, self.H, dtype=torch.float32, device=dev)
        for e in range(self.E):
            n = int(self.cnt[e])
            if n == 0:
                continue
            r0 = int(self.rowoff[e])
            hq, hs = kr.act_quant(fu[r0:r0 + n].contiguous(), 32, "ue8m0", torch.float8_e8m0fnu)
            part[self.part[r0:r0 + n].to(torch.int64).to(dev)] = self.fp4(hq, hs, self.w2[e], self.s2[e]).float()
        return part

    def run_glu(self, blocks=132, iters=1, gate=True):
        c = self
        ops = [("fu", c.rows * c.I * 2, None), ("x", c.T * c.H * 2, dump(c.fq, "x.bin")), ("wtab", c.E * 24, None), ("stab", c.E * 24, None),
               ("meta", c.meta.numel() * 4, dump(c.meta, "meta.bin")), ("rowtok", c.rows * 4, dump(c.tok.view(torch.int32), "tok.bin")),
               ("-", 0, None), ("rowgate", c.rows * 4, dump(c.gate, "gate.bin")) if gate else ("-", 0, None)]
        return packet_op("MoeGroupGluPf", [c.I, c.H, c.E, 2, 0, 4], [0.0, LIM], ops, c.extra(), c.ptr_tables(),
                         [("fu", torch.bfloat16, (c.rows, c.I))], blocks, iters)

    def run_down(self, fu, blocks=132, iters=1):
        c = self
        ops = [("part", c.T * c.k * c.H * 4, None), ("fu", c.rows * c.I * 2, dump(fu.bfloat16(), "fu.bin")), ("wtab", c.E * 24, None),
               ("stab", c.E * 24, None), ("meta", c.meta.numel() * 4, dump(c.meta, "meta.bin")), ("-", 0, None),
               ("rowpart", c.rows * 4, dump(c.part.view(torch.int32), "part.bin")), ("-", 0, None)]
        return packet_op("MoeGroupDownPf", [c.H, c.I, c.E, 2], [], ops, c.extra(), c.ptr_tables(),
                         [("part", torch.float32, (c.T * c.k, c.H))], blocks, iters)

    def live(self):
        m = torch.zeros(self.rows, dtype=torch.bool)
        for e in range(self.E):
            m[int(self.rowoff[e]):int(self.rowoff[e]) + int(self.cnt[e])] = True
        return m.to(dev)


if want("glu") or want("down"):
    # rows per expert: ~14 (N=16 skinny tile), ~56 (N=64 skinny, some 128-row), ~100 and ~256 (128-row tiles)
    for T, E, k, H, I in ((37, 16, 6, 5120, 2304), (300, 32, 6, 5120, 576), (200, 12, 6, 2048, 512), (1024, 24, 6, 1024, 256)):
        c = Case(T, E, k, H, I, seed=T)
        fu_ref = c.ref_glu()
        if want("glu"):
            (fu,), us = c.run_glu()
            lv = c.live()
            r = rel(fu[lv], fu_ref[lv])
            bad = (fu[lv] != fu_ref[lv].bfloat16()).float().mean().item()
            check(f"pkt MoeGroupGluPf T={T} E={E} H={H} I={I}", r < 1e-2, f"rel={r:.2e} mismatch={bad:.2%} {us:.1f} us")
        if want("down"):
            (part,), us = c.run_down(fu_ref)
            ref = c.ref_down(fu_ref)
            r = rel(part, ref)
            check(f"pkt MoeGroupDownPf T={T} E={E} H={H} I={I}", r < 3e-3, f"rel={r:.2e} {us:.1f} us")

if want("perf"):
    # V4.1 per rank: 384 experts, top-6; I = 2304 / tp (576 at TP4). PFMOE_PERF="T:I,..."
    for spec in os.environ.get("PFMOE_PERF", "1024:576,4096:576").split(","):
        T, I = map(int, spec.split(":"))
        c = Case(T, 384, 6, 5120, I, seed=7)
        n = T * c.k
        w13b = c.E * 2 * c.I * c.H * (0.5 + 1 / 32)
        w2b = c.E * c.H * c.I * (0.5 + 1 / 32)
        (fu,), us_g = c.run_glu(iters=10)
        (part,), us_d = c.run_down(fu, iters=10)
        for name, us, fl, wb in (("GLU", us_g, 2 * n * 2 * c.I * c.H, w13b), ("DOWN", us_d, 2 * n * c.H * c.I, w2b)):
            mem, f8, b16 = wb / 4.8e12 * 1e6, fl / 1979e12 * 1e6, fl / 989e12 * 1e6
            print(f"  T={T} I={I} {name}: {us:.1f} us  {fl / us / 1e6:.0f} TFLOP/s  weights {wb / us / 1e3:.0f} GB/s  "
                  f"floor mem {mem:.0f} us fp8 {f8:.0f} us bf16 {b16:.0f} us -> {max(mem, f8) / us:.0%} of fp8 roofline, "
                  f"{max(mem, b16) / us:.0%} of bf16 roofline", flush=True)

print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
