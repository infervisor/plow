#!/usr/bin/env python3
"""Mamba-2 interpreter ops (runtime/nvidia/op_mamba.cuh) vs a torch fp32 reference.

Runs the ops from the standalone cubin (runtime/tests/mamba2_ops_kernels.cu) on one real
Nemotron-H mamba layer's weights (conv, A_log, D, dt_bias, gated-norm gamma, FP8 in_proj for
realistic activations) and checks, per case, the conv output, the SSM output y, the gated-norm
output and the final conv history / SSM state:

  prefill T = 1, 7, 128 (bucket 128) and 1000 (bucket 1024, 24 pad rows), fresh state;
  continuation: 600 rows then 400 rows (bucket 1024 each) == 1000 rows at once;
  decode after prefill: three B=1 steps continuing the T=1000 state;
  decode B=4 from random slot states with one inactive row;
  stochastic rounding: the step's Philox-4x32-5 words and fp16 SR bit-exact against a numpy model
  of the peer's operator (FlashInfer mamba/conversion.cuh), and unbiased over 256 seeds;
  drift: 64 decode steps with the fp16 SR state vs the f32-state recurrence (and vs an RNE fp16
  state, the deterministic alternative).

The reference is the sequential recurrence (the definition), rounded where vLLM rounds:
conv output, y and the norm output in bf16, math in f32; the slot state is fp16 (prefill stores
it round-to-nearest-even, decode with stochastic rounding: compared against the f32 update of the
same fp16 input state). dt = softplus(dt + dt_bias) clamped to config time_step_limit.

  python3 runtime/tests/mamba2_oracle.py --cubin mamba2_ops.cubin --ckpt <dir> --layer 0
"""
import argparse
import ctypes
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

H, P, N, G, W, L = 128, 64, 128, 8, 4, 128
DI = H * P
CONV = DI + 2 * G * N
PROJ = DI + CONV + H
GSIZE = DI // G


class Ckpt:
    def __init__(self, root):
        self.root = root
        self.map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]

    def get(self, name):
        with safe_open(os.path.join(self.root, self.map[name]), "pt") as f:
            return f.get_tensor(name)


def layer_weights(ck, layer):
    p = f"language_model.backbone.layers.{layer}.mixer."
    w = {
        "conv_w": ck.get(p + "conv1d.weight").reshape(CONV, W).contiguous(),
        "conv_b": ck.get(p + "conv1d.bias"),
        "A_log": ck.get(p + "A_log"),
        "D": ck.get(p + "D"),
        "dt_bias": ck.get(p + "dt_bias"),
        "gamma": ck.get(p + "norm.weight"),
        "in_w": ck.get(p + "in_proj.weight"),
        "in_s": ck.get(p + "in_proj.weight_scale"),
        "pre_norm": ck.get(f"language_model.backbone.layers.{layer}.norm.weight"),
    }
    for k in ("conv_w", "conv_b", "A_log", "D", "dt_bias", "gamma", "pre_norm"):
        assert w[k].dtype == torch.bfloat16, (k, w[k].dtype)
    return {k: v.cuda() for k, v in w.items()}


def make_proj(w, rows, gen):
    """Realistic in_proj outputs: RMSNorm(randn) * gamma through the dequantized FP8 weight."""
    x = torch.randn(rows, 4096, device="cuda", generator=gen)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w["pre_norm"].float()
    wf = w["in_w"].float() * w["in_s"].float()
    return (x.to(torch.bfloat16).float() @ wf.t()).to(torch.bfloat16)


# ---------------- reference ----------------
def ref_forward(w, proj, conv_hist, state, lo, hi):
    """Sequential reference over proj rows [R, PROJ] (bf16). Returns conv out, y, norm, hist, state."""
    R = proj.shape[0]
    z = proj[:, :DI].float()
    xin = proj[:, DI:DI + CONV]
    dt_raw = proj[:, DI + CONV:].float()
    full = torch.cat([conv_hist.float().t(), xin.float()], 0)
    cw = w["conv_w"].float()
    conv = torch.zeros(R, CONV, device="cuda")
    for k in range(W):
        conv += full[k:k + R] * cw[:, k]
    conv = F.silu(conv + w["conv_b"].float()).to(torch.bfloat16)
    new_hist = full[-(W - 1):].t().to(torch.bfloat16).contiguous()
    x = conv[:, :DI].float().view(R, H, P)
    Bm = conv[:, DI:DI + G * N].float().view(R, G, N)
    Cm = conv[:, DI + G * N:].float().view(R, G, N)
    dt = F.softplus(dt_raw + w["dt_bias"].float()).clamp(lo, hi)
    A = -torch.exp(w["A_log"].float())
    Dh = w["D"].float()
    s = state.clone()
    y = torch.zeros(R, H, P, device="cuda")
    hpg = H // G
    for t in range(R):
        dA = torch.exp(dt[t] * A)  # [H]
        Bt = Bm[t].repeat_interleave(hpg, 0)  # [H, N]
        Ct = Cm[t].repeat_interleave(hpg, 0)
        s = s * dA[:, None, None] + (dt[t][:, None] * x[t])[:, :, None] * Bt[:, None, :]
        y[t] = torch.einsum("hpn,hn->hp", s, Ct) + Dh[:, None] * x[t]
    y = y.reshape(R, DI).to(torch.bfloat16)
    v = y.float() * F.silu(z)
    vg = v.view(R, G, GSIZE)
    vg = vg * torch.rsqrt(vg.pow(2).mean(-1, keepdim=True) + 1e-5)
    norm = (w["gamma"] * vg.reshape(R, DI).to(torch.bfloat16)).to(torch.bfloat16)
    return conv, y, norm, new_hist, s


# ---------------- stochastic rounding model (FlashInfer mamba/conversion.cuh) ----------------
M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF


def layer_seed(seed, layer):
    z = (seed ^ (0x9E3779B97F4A7C15 * (layer + 1))) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def philox4(key, ctr, rounds=5):
    """Philox-4x32 (Triton tl.randint4x): key (k0, k1) = key lo/hi, counter (ctr lo, ctr hi, 0, 0)."""
    m = np.uint64(M32)
    ctr = ctr.astype(np.uint64)
    c0, c1 = ctr & m, ctr >> np.uint64(32)
    c2, c3 = np.zeros_like(c0), np.zeros_like(c0)
    k0, k1 = np.uint64(key & M32), np.uint64(key >> 32)
    ra, rb = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
    for _ in range(rounds):
        a, b = c0, c2
        pb, pa = rb * b, ra * a
        c0 = (pb >> np.uint64(32)) ^ c1 ^ k0
        c2 = (pa >> np.uint64(32)) ^ c3 ^ k1
        c1, c3 = pb & m, pa & m
        k0, k1 = (k0 + np.uint64(0x9E3779B9)) & m, (k1 + np.uint64(0xBB67AE85)) & m
    return c0, c1, c2, c3


def sr_f16_bits(x, r):
    """cvt_rs_f16_sw: |x| bits += r & 0x1FFF, truncate to f16; < f16 normal -> 0, > max -> inf."""
    bits = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    a = (bits & np.uint64(0x7FFFFFFF)) + (r & np.uint64(0x1FFF))
    e, mant = (a >> np.uint64(23)) & np.uint64(0xFF), a & np.uint64(0x7FFFFF)
    h = np.where(e == 0xFF, np.where(mant != 0, 0x7E00, 0x7C00),
                 np.where(e > 142, 0x7C00,
                          np.where(e < 113, 0, ((e - np.uint64(112)) << np.uint64(10)) | (mant >> np.uint64(13)))))
    return ((bits >> np.uint64(16)) & np.uint64(0x8000)) | h.astype(np.uint64)


# ---------------- device ----------------
class Ops:
    """`pf="tc"`: the prefill object's tensor-core bodies (op_mamba_pf.cuh) for conv and the
    three chunk phases; "simt": op_mamba.cuh's."""

    def __init__(self, cubin, pf="simt"):
        torch.zeros(1, device="cuda")
        self.cuda = ctypes.CDLL("libcuda.so.1")
        mod = ctypes.c_void_p()
        assert self.cuda.cuModuleLoadData(ctypes.byref(mod), open(cubin, "rb").read()) == 0
        self.fn = {}
        self.smem = 48 * 1024 if pf == "simt" else 55296
        for k in ("conv", "step", "norm", "chunk_state", "chunk_pass", "chunk_out", "sr"):
            sym = f"mamba2_{k}_tc_k" if pf == "tc" and k not in ("step", "norm", "sr") else f"mamba2_{k}_k"
            f = ctypes.c_void_p()
            assert self.cuda.cuModuleGetFunction(ctypes.byref(f), mod, sym.encode()) == 0, sym
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
            assert self.cuda.cuFuncSetAttribute(f, 8, self.smem) == 0, sym
            self.fn[k] = f
        if pf == "tc":
            for k in ("conv", "chunk_state", "chunk_pass", "chunk_out"):
                f = ctypes.c_void_p()
                assert self.cuda.cuModuleGetFunction(ctypes.byref(f), mod, f"mamba2_{k}_pk".encode()) == 0, k
                assert self.cuda.cuFuncSetAttribute(f, 8, self.smem) == 0, k
                self.fn[k + "_pk"] = f
            # CHUNK_STATE + CHUNK_PASS in one walk (PLOW_NV_MAMBA_SCAN_FUSE).
            for k, sym in (("chunk_scan", "mamba2_chunk_scan_tc_k"), ("chunk_scan_pk", "mamba2_chunk_scan_pk"),
                           ("chunk_scan_bf", "mamba2_chunk_scan_bf_tc_k"), ("chunk_scan_bf_pk", "mamba2_chunk_scan_bf_pk"),
                           ("chunk_out_bf", "mamba2_chunk_out_bf_tc_k"), ("chunk_out_bf_pk", "mamba2_chunk_out_bf_pk")):
                f = ctypes.c_void_p()
                assert self.cuda.cuModuleGetFunction(ctypes.byref(f), mod, sym.encode()) == 0, sym
                assert self.cuda.cuFuncSetAttribute(f, 8, self.smem) == 0, sym
                self.fn[k] = f
        self.scan = False
        self.stream = torch.cuda.current_stream().cuda_stream
        self.grid = 48

    def launch(self, name, *args, smem=0):
        cargs = []
        for a in args:
            if isinstance(a, torch.Tensor):
                cargs.append(ctypes.c_uint64(a.data_ptr()))
            elif a is None:
                cargs.append(ctypes.c_uint64(0))
            elif isinstance(a, float):
                cargs.append(ctypes.c_float(a))
            else:
                cargs.append(ctypes.c_uint32(a))
        arr = (ctypes.c_void_p * len(cargs))(*[ctypes.addressof(c) for c in cargs])
        rc = self.cuda.cuLaunchKernel(self.fn[name], self.grid, 1, 1, 256, 1, 1, smem,
                                      ctypes.c_void_p(self.stream), arr, None)
        assert rc == 0, f"{name} launch rc={rc}"

    def prefill(self, w, proj_bucket, R, hist, state, lo, hi):
        """proj_bucket [T, PROJ] (rows >= R are pad); hist [C, W-1] bf16, state [H,P,N] fp16 updated."""
        T = proj_bucket.shape[0]
        conv = torch.full((T, CONV), float("nan"), device="cuda").to(torch.bfloat16)
        y = torch.full((T, DI), float("nan"), device="cuda").to(torch.bfloat16)
        out = torch.empty(T, DI, device="cuda", dtype=torch.bfloat16)
        cstate = torch.empty(T // L, H, P, N, device="cuda")
        dtv = torch.empty(H, T, device="cuda")
        cum = torch.empty(H, T, device="cuda")
        src = PROJ | (DI << 16)
        dsrc = PROJ | ((DI + CONV) << 16)
        zsrc = PROJ | (0 << 16)
        self.launch("conv", conv, proj_bucket, w["conv_w"], w["conv_b"], hist, None, CONV, W, R, T, 1, src)
        bf = "_bf" if self.scan == "bf16" else ""
        if self.scan:
            self.launch("chunk_scan" + bf, cstate, conv, proj_bucket, w["A_log"], w["dt_bias"], dtv, cum, state,
                        H, P, R, T, N, G, lo, hi, dsrc, smem=self.smem)
        else:
            self.launch("chunk_state", cstate, conv, proj_bucket, w["A_log"], w["dt_bias"], dtv, cum,
                        H, P, R, T, N, G, lo, hi, dsrc, smem=self.smem)
            self.launch("chunk_pass", cstate, cum, state, H, P, R, T, N)
        self.launch("chunk_out" + bf, y, conv, cstate, cum, dtv, w["D"], H, P, R, T, N, G, smem=self.smem)
        self.launch("norm", out, y, proj_bucket, w["gamma"], None, G, GSIZE, T, 1e-5, zsrc)
        torch.cuda.synchronize()
        self.last = (cstate, dtv, cum)
        return conv, y, out

    def prefill_packed(self, w, proj_bucket, req, hist, state, lo, hi):
        """Packed: `req` int32 [1 + 4n] request table; hist [slots, C, W-1], state [slots, H, P, N]."""
        T = proj_bucket.shape[0]
        n = int(req[0])
        R = int(req[1 + 4 * (n - 1)] + req[2 + 4 * (n - 1)])
        conv = torch.full((T, CONV), float("nan"), device="cuda").to(torch.bfloat16)
        y = torch.full((T, DI), float("nan"), device="cuda").to(torch.bfloat16)
        out = torch.empty(T, DI, device="cuda", dtype=torch.bfloat16)
        cstate = torch.full((T // L + n, H, P, N), float("nan"), device="cuda")
        dtv = torch.full((H, T + L * n), float("nan"), device="cuda")
        cum = torch.full((H, T + L * n), float("nan"), device="cuda")
        src = PROJ | (DI << 16)
        dsrc = PROJ | ((DI + CONV) << 16)
        zsrc = PROJ | (0 << 16)
        self.launch("conv_pk", conv, proj_bucket, w["conv_w"], w["conv_b"], hist, req, CONV, W, R, T, src)
        bf = "_bf" if self.scan == "bf16" else ""
        if self.scan:
            self.launch("chunk_scan" + bf + "_pk", cstate, conv, proj_bucket, w["A_log"], w["dt_bias"], dtv, cum,
                        state, req, H, P, R, T, N, G, lo, hi, dsrc, smem=self.smem)
        else:
            self.launch("chunk_state_pk", cstate, conv, proj_bucket, w["A_log"], w["dt_bias"], dtv, cum, req,
                        H, P, R, T, N, G, lo, hi, dsrc, smem=self.smem)
            self.launch("chunk_pass_pk", cstate, cum, state, req, H, P, R, T, N)
        self.launch("chunk_out" + bf + "_pk", y, conv, cstate, cum, dtv, w["D"], req, H, P, R, T, N, G, smem=self.smem)
        self.launch("norm", out, y, proj_bucket, w["gamma"], None, G, GSIZE, T, 1e-5, zsrc)
        torch.cuda.synchronize()
        return conv, y, out

    def decode(self, w, proj, hist, state, active, lo, hi, seed=0x1234567, layer=0):
        """active [B] i32; the step reads its seed from the word after the slots (in.active tail)."""
        B = proj.shape[0]
        word = torch.from_numpy(np.array([seed], dtype=np.uint32).view(np.int32)).cuda()
        active = torch.cat([active, word])
        conv = torch.empty(B, CONV, device="cuda", dtype=torch.bfloat16)
        y = torch.zeros(B, DI, device="cuda", dtype=torch.bfloat16)
        out = torch.zeros(B, DI, device="cuda", dtype=torch.bfloat16)
        src = PROJ | (DI << 16)
        dsrc = PROJ | ((DI + CONV) << 16)
        self.launch("conv", conv, proj, w["conv_w"], w["conv_b"], hist, active, CONV, W, B, 0, 0, src)
        self.launch("step", y, conv, proj, w["A_log"], w["D"], w["dt_bias"], state, active, H, P, N, G, B,
                    dsrc, lo, hi, B, layer)
        self.launch("norm", out, y, proj, w["gamma"], active, G, GSIZE, B, 1e-5, PROJ)
        torch.cuda.synchronize()
        return conv, y, out


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item(), (a - b).abs().max().item()


LIMITS = {"conv": 2e-3, "y": 1e-2, "norm": 1e-2, "hist": 0.0, "state": 2e-3, "drift y": 2e-2,
          "drift state": 1e-2}


def check(rows, case, name, got, want):
    r, m = rel(got, want)
    bad = not math.isfinite(r) or r > LIMITS[name] or not torch.isfinite(got.float()).all()
    rows.append((case, name, r, m, "FAIL" if bad else "ok"))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--json", help="write the result rows here")
    ap.add_argument("--pf", choices=["simt", "tc"], default="tc", help="prefill bodies under test")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    tc = cfg.get("llm_config", cfg)
    lo, hi = (float(v) if not isinstance(v, dict) else math.inf for v in tc["time_step_limit"])
    w = layer_weights(Ckpt(a.ckpt), a.layer)
    ops = Ops(a.cubin, a.pf)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    rows, ok = [], True
    zero_hist = lambda: torch.zeros(CONV, W - 1, device="cuda", dtype=torch.bfloat16)
    zero_state = lambda: torch.zeros(H, P, N, device="cuda", dtype=torch.float16)

    def bucket_of(R):
        return 128 if R <= 128 else 1024 if R <= 1024 else 8192

    keep = {}
    for R in (1, 7, 128, 1000):
        T = bucket_of(R)
        proj = make_proj(w, R, gen)
        pad = torch.randn(T - R, PROJ, device="cuda", generator=gen).to(torch.bfloat16) * 4
        pb = torch.cat([proj, pad], 0).contiguous()
        hist, state = zero_hist(), zero_state()
        conv, y, out = ops.prefill(w, pb, R, hist, state, lo, hi)
        rc, ry, rn, rh, rs = ref_forward(w, proj, zero_hist(), zero_state().float(), lo, hi)
        case = f"prefill T={R} (bucket {T})"
        ok &= check(rows, case, "conv", conv[:R], rc)
        ok &= check(rows, case, "y", y[:R], ry)
        ok &= check(rows, case, "norm", out[:R], rn)
        ok &= check(rows, case, "hist", hist, rh)
        ok &= check(rows, case, "state", state, rs)
        pad_ok = bool((conv[R:] == 0).all() and (y[R:] == 0).all())
        rows.append((case, "pad rows zero", 0.0, 0.0, "ok" if pad_ok else "FAIL"))
        ok &= pad_ok
        if R == 1000:
            keep = {"proj": proj, "hist": hist, "state": state}

    # Continuation: 600 + 400 rows == 1000 rows.
    proj = keep["proj"]
    hist, state = zero_hist(), zero_state()
    outs = []
    for lo_r, hi_r in ((0, 600), (600, 1000)):
        part = proj[lo_r:hi_r]
        pb = torch.cat([part, torch.zeros(1024 - part.shape[0], PROJ, device="cuda", dtype=torch.bfloat16)], 0)
        conv, y, out = ops.prefill(w, pb.contiguous(), part.shape[0], hist, state, lo, hi)
        outs.append((conv[:part.shape[0]], y[:part.shape[0]], out[:part.shape[0]]))
    # The reference stores the 600-row state fp16 (RNE) as the device does between launches.
    rc0, ry0, rn0, rh, rs = ref_forward(w, proj[:600], zero_hist(), zero_state().float(), lo, hi)
    rc1, ry1, rn1, rh, rs = ref_forward(w, proj[600:], rh, rs.half().float(), lo, hi)
    rc, ry, rn = torch.cat([rc0, rc1]), torch.cat([ry0, ry1]), torch.cat([rn0, rn1])
    case = "continuation 600+400"
    ok &= check(rows, case, "conv", torch.cat([o[0] for o in outs]), rc)
    ok &= check(rows, case, "y", torch.cat([o[1] for o in outs]), ry)
    ok &= check(rows, case, "norm", torch.cat([o[2] for o in outs]), rn)
    ok &= check(rows, case, "hist", hist, rh)
    ok &= check(rows, case, "state", state, rs)

    # Decode after prefill: three single-row steps from the T=1000 device state; each step's
    # reference is the f32 update of the device's fp16 state before it (SR error <= 1 fp16 ulp).
    hist, state = keep["hist"].clone(), keep["state"].clone()
    one = torch.ones(1, device="cuda", dtype=torch.int32)
    for step in range(3):
        pr = make_proj(w, 1, gen)
        rhist, rprev = hist.clone(), state.float()
        conv, y, out = ops.decode(w, pr, hist, state, one, lo, hi, seed=0xA5A5A5A5 + step)
        rc, ry, rn, rhist, rstate = ref_forward(w, pr, rhist, rprev, lo, hi)
        case = f"decode-after-prefill step {step}"
        ok &= check(rows, case, "conv", conv, rc)
        ok &= check(rows, case, "y", y, ry)
        ok &= check(rows, case, "norm", out, rn)
        ok &= check(rows, case, "hist", hist, rhist)
        ok &= check(rows, case, "state", state, rstate)

    # Decode B=4 from random slot states; row 2 inactive (state and history untouched).
    B = 4
    hist = (torch.randn(B, CONV, W - 1, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    state = (torch.randn(B, H, P, N, device="cuda", generator=gen) * 0.1).half()
    h0, s0 = hist.clone(), state.clone()
    active = torch.tensor([1, 1, 0, 1], device="cuda", dtype=torch.int32)
    pr = make_proj(w, B, gen)
    conv, y, out = ops.decode(w, pr, hist, state, active, lo, hi)
    for b in range(B):
        case = f"decode B=4 row {b}" + (" (inactive)" if b == 2 else "")
        if b == 2:
            same = bool((hist[b] == h0[b]).all() and (state[b] == s0[b]).all())
            rows.append((case, "untouched", 0.0, 0.0, "ok" if same else "FAIL"))
            ok &= same
            continue
        rc, ry, rn, rhb, rsb = ref_forward(w, pr[b:b + 1], h0[b], s0[b].float(), lo, hi)
        ok &= check(rows, case, "conv", conv[b:b + 1], rc)
        ok &= check(rows, case, "y", y[b:b + 1], ry)
        ok &= check(rows, case, "norm", out[b:b + 1], rn)
        ok &= check(rows, case, "hist", hist[b], rhb)
        ok &= check(rows, case, "state", state[b], rsb)

    # Packed prefill: 8 requests of mixed lengths (1- and 2-row requests, chunk boundaries inside
    # and on 128, continuations from random slot history / state) in one 1024-row launch against
    # each request prefilled alone from the same slot: bit-identical rows, history and state.
    if a.pf == "tc":
        lens = (1, 2, 130, 7, 300, 127, 129, 33)
        slot_of = (5, 0, 3, 7, 1, 6, 2, 4)
        S = 8
        hist0 = (torch.randn(S, CONV, W - 1, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
        state0 = (torch.randn(S, H, P, N, device="cuda", generator=gen) * 0.1).half()
        projs = [make_proj(w, r, gen) for r in lens]
        total = sum(lens)
        T = bucket_of(total)
        pad = torch.randn(T - total, PROJ, device="cuda", generator=gen).to(torch.bfloat16) * 4
        pb = torch.cat(projs + [pad], 0).contiguous()
        table, row0 = [len(lens)], 0
        for r, s in zip(lens, slot_of):
            table += [row0, r, s, r]
            row0 += r
        req = torch.tensor(table, device="cuda", dtype=torch.int32)
        hist, state = hist0.clone(), state0.clone()
        conv, y, out = ops.prefill_packed(w, pb, req, hist, state, lo, hi)
        pb_packed, y_packed, out_packed, state_packed = pb, y.clone(), out.clone(), state.clone()
        row0, exact = 0, True
        for i, (r, s) in enumerate(zip(lens, slot_of)):
            hb, sb = hist0[s].clone(), state0[s].clone()
            ta = bucket_of(r)
            pa = torch.cat([projs[i], torch.zeros(ta - r, PROJ, device="cuda", dtype=torch.bfloat16)], 0)
            c1, y1, o1 = ops.prefill(w, pa.contiguous(), r, hb, sb, lo, hi)
            same = bool((conv[row0:row0 + r] == c1[:r]).all() and (y[row0:row0 + r] == y1[:r]).all()
                        and (out[row0:row0 + r] == o1[:r]).all() and (hist[s] == hb).all()
                        and (state[s] == sb).all())
            rows.append((f"packed req {i} len {r} slot {s}", "bit-exact", 0.0, 0.0, "ok" if same else "FAIL"))
            exact &= same
            row0 += r
        pad_ok = bool((conv[total:] == 0).all() and (y[total:] == 0).all())
        rows.append((f"packed {len(lens)} reqs T={T}", "pad rows zero", 0.0, 0.0, "ok" if pad_ok else "FAIL"))
        ok &= exact and pad_ok

        # Fused CHUNK_STATE + CHUNK_PASS vs the two ops: entering states, dt, cumsum, slot state and the
        # outputs bit-identical (serial buckets incl. 8192 rows from a random state; the packed table above).
        for R in (1, 7, 128, 1000, 8192):
            T = bucket_of(R) if R <= 1024 else 8192
            proj = make_proj(w, R, gen)
            pb = torch.cat([proj, torch.zeros(T - R, PROJ, device="cuda", dtype=torch.bfloat16)], 0).contiguous()
            h0 = (torch.randn(CONV, W - 1, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
            s0 = (torch.randn(H, P, N, device="cuda", generator=gen) * 0.1).half()
            res = []
            for scan in (False, True, "bf16"):
                ops.scan = scan
                hb, sb = h0.clone(), s0.clone()
                outs = ops.prefill(w, pb, R, hb, sb, lo, hi)
                nc = (R + L - 1) // L
                cs, dv, cm = ops.last
                if scan == "bf16":  # bf16(fp16(entering)) in the first half of the buffer
                    cs = cs.view(-1).view(torch.bfloat16)[: cs.numel()].view(cs.shape)
                else:
                    cs = cs.half().float().bfloat16()
                res.append((outs, hb, sb, cs[:nc].clone(), dv[:, :nc * L].clone(), cm[:, :nc * L].clone()))
            ops.scan = False
            for name, r1 in (("fused", res[1]), ("fused bf16 enter", res[2])):
                (o0, h0_, s0_, c0, d0, m0), (o1, h1, s1, c1, d1, m1) = res[0], r1
                same = all(bool(torch.equal(a, b)) for a, b in zip(o0, o1)) and all(
                    bool(torch.equal(a, b)) for a, b in ((s0_, s1), (c0, c1), (d0, d1), (m0, m1)))
                rows.append((f"scan {name} vs state+pass T={R}", "bit-exact", 0.0, 0.0, "ok" if same else "FAIL"))
                ok &= same
        for scan in (True, "bf16"):
            ops.scan = scan
            hist, state = hist0.clone(), state0.clone()
            conv2, y2, out2 = ops.prefill_packed(w, pb_packed, req, hist, state, lo, hi)
            ops.scan = False
            same = bool(torch.equal(y2, y_packed) and torch.equal(out2, out_packed) and torch.equal(state, state_packed))
            rows.append((f"scan fused{' bf16' if scan == 'bf16' else ''} packed {len(lens)} reqs", "bit-exact",
                         0.0, 0.0, "ok" if same else "FAIL"))
            ok &= same

    # Stochastic rounding operator, bit-exact: the step's Philox words and fp16 results.
    n = 1 << 16
    xs = torch.randn(n, device="cuda", generator=gen) * torch.exp2(
        torch.randint(-30, 20, (n,), device="cuda", generator=gen).float())
    xs[:8] = torch.tensor([0.0, -0.0, 65504.0, -1e-7, float("inf"), float("-inf"), float("nan"), 6.1e-5],
                          device="cuda")
    out16 = torch.empty(n, device="cuda", dtype=torch.int16)
    bits = torch.empty(n, device="cuda", dtype=torch.int32)
    for seed, layer in ((0, 0), (0xDEADBEEF, 7), (12345, 87)):
        ops.launch("sr", out16, bits, xs, n, seed, layer)
        torch.cuda.synchronize()
        ctr = np.arange(n, dtype=np.uint64) & ~np.uint64(3)
        words = np.stack(philox4(layer_seed(seed, layer), ctr), 0)[np.arange(n) % 4, np.arange(n)]
        dev_bits = bits.cpu().numpy().view(np.uint32).astype(np.uint64)
        want = sr_f16_bits(xs.cpu().numpy(), words)
        got = out16.cpu().numpy().view(np.uint16).astype(np.uint64)
        bad = int((dev_bits != words).sum() + (got != want).sum())
        rows.append((f"SR seed {seed:#x} layer {layer}", "bit-exact", float(bad), 0.0,
                     "ok" if bad == 0 else "FAIL"))
        ok &= bad == 0
    # Unbiased: the mean of SR(x) over 256 seeds is x (in fp16 ulps of x); every result is one of
    # the two fp16 neighbours.
    xu = (1.0 + torch.rand(n, device="cuda", generator=gen)) * torch.exp2(
        torch.randint(-10, 10, (n,), device="cuda", generator=gen).float())
    ulp = torch.exp2(torch.floor(torch.log2(xu)) - 10)
    down = torch.floor(xu / ulp) * ulp
    acc = torch.zeros(n, device="cuda", dtype=torch.float64)
    two_sided = True
    for seed in range(256):
        ops.launch("sr", out16, bits, xu, n, seed * 0x9E3779B9 & M32, 3)
        torch.cuda.synchronize()
        v = out16.view(torch.float16).float()
        two_sided &= bool(((v == down) | (v == down + ulp)).all())
        acc += v.double()
    bias = ((acc / 256 - xu.double()) / ulp.double()).mean().item()
    unb = abs(bias) < 0.01 and two_sided
    rows.append(("SR unbiased, 256 seeds", "mean err/ulp", abs(bias), 0.0, "ok" if unb else "FAIL"))
    ok &= unb

    # Drift: 64 single-row decode steps from the T=1000 state, fp16 SR state vs the f32 state
    # recurrence; the RNE fp16 recurrence is reported beside it.
    hist, state = keep["hist"].clone(), keep["state"].clone()
    rh32, rs32 = hist.clone(), state.float()
    rhn, rsn = hist.clone(), state.float()
    for step in range(64):
        pr = make_proj(w, 1, gen)
        conv, y, out = ops.decode(w, pr, hist, state, one, lo, hi, seed=(step + 1) * 0x9E3779B9 & M32)
        _, ry, _, rh32, rs32 = ref_forward(w, pr, rh32, rs32, lo, hi)
        _, yn, _, rhn, rsn = ref_forward(w, pr, rhn, rsn, lo, hi)
        rsn = rsn.half().float()
    ok &= check(rows, "drift 64 steps (SR)", "drift y", y, ry)
    ok &= check(rows, "drift 64 steps (SR)", "drift state", state, rs32)
    rows.append(("drift 64 steps (RNE ref)", "drift y", *rel(yn, ry), "info"))
    rows.append(("drift 64 steps (RNE ref)", "drift state", *rel(rsn, rs32), "info"))

    print(f"{'case':38s} {'tensor':14s} {'rel-L2':>10s} {'max-abs':>10s}  verdict")
    for c, n, r, m, v in rows:
        print(f"{c:38s} {n:14s} {r:10.3e} {m:10.3e}  {v}")
    print(f"{sum(r[4] == 'ok' for r in rows)}/{sum(r[4] != 'info' for r in rows)} checks pass; layer {a.layer}, "
          f"time_step_limit=({lo}, {hi})")
    if a.json:
        json.dump([dict(zip(("case", "tensor", "rel_l2", "max_abs", "verdict"), r)) for r in rows],
                  open(a.json, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
