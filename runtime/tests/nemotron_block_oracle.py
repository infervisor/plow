#!/usr/bin/env python3
"""Nemotron-H block packet (mamba / attention layers) on the GPU vs a torch reference.

Drives `examples/block_run` on a `--block` asset of mamba + attention layers and compares its
`act.x` output with a torch model of the same layers on the real checkpoint weights:

  prefill   `block_run check --in x.npy` (T rows, fresh state, off-bucket T in several chunks), T in
            --prefill (default 7,128,1000 and the off-bucket 1500,1537,2048,3000,6000)
  decode    `block_run decode-check --in x.npy --rows r.npy --batch B --steps S` (decode after a
            prefill of T rows; B slots share the prompt, then decode distinct rows)

The reference models the device arithmetic: RMSNorm (f32, one bf16 round), FP8 per-tensor
linears with static per-tensor e4m3 activations (the checkpoint input_scale, `--fp8-act static`,
the peer's arithmetic) or per-row dynamic ones (`--fp8-act row`) in prefill and bf16
activations in decode (W8A16), the Mamba-2 mixer of runtime/tests/mamba2_oracle.py (fp16 slot state), NoPE GQA with bf16 KV (or unit-scale e4m3, `--kv fp8`), bf16
residual stream. A second "ideal" reference (bf16 activations into every FP8 linear, i.e. no
activation quantization) reports how far the prefill's dynamic FP8 activations move the block.
Metric: rel-L2 of the block delta (y - x), the part the layers compute.

  python3 runtime/tests/nemotron_block_oracle.py --asset <emit dir> --block-run <bin> \
      --ckpt <dir> --layers 6,7
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mamba2_oracle as mo  # noqa: E402

BF = torch.bfloat16
FP8_ACT = "static"  # prefill activation quant of the FP8 linears (--fp8-act)
MOE_ACT = "nvfp4"  # prefill activations of the NVFP4 experts (--moe-act): nvfp4 = W4A4, bf16 = W4A16
KV = "bf16"  # KV cache elements (--kv): fp8 = sat_e4m3(x / 1.0), vLLM's unit k/v scale


def bf(x):
    return x.to(BF)


def rmsnorm(x, w, eps):
    xf = x.float()
    return bf(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float())


def fp8_quant_rows(x):
    """op_gemm.cuh d_quant_fp8: as = max(amax/448, 1e-12), q = e4m3(x / as)."""
    xf = x.float()
    s = torch.clamp(xf.abs().amax(-1, keepdim=True) / 448.0, min=1e-12)
    q = (xf * (1.0 / s)).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return q, s


def fp8_quant_static(x, s):
    """op_gemm.cuh d_quant_fp8_tensor: q = e4m3(clamp(x * (1/s), +-448))."""
    return (x.float() * (1.0 / s)).clamp(-448, 448).to(torch.float8_e4m3fn).float()


class Linear:
    def __init__(self, ck, module):
        self.w = ck.get(module + ".weight").cuda().float()
        self.s = ck.get(module + ".weight_scale").cuda().float()
        self.si = ck.get(module + ".input_scale").cuda().float().reshape(())

    def __call__(self, x, quant_act):
        if quant_act and FP8_ACT == "static":
            return bf((fp8_quant_static(x, self.si) @ self.w.t()) * (self.si * self.s))
        if quant_act:
            q, s = fp8_quant_rows(x)
            return bf((q @ self.w.t()) * s * self.s)
        return bf((x.float() @ self.w.t()) * self.s)


class Mamba:
    def __init__(self, ck, layer, eps, lo, hi):
        self.w = mo.layer_weights(ck, layer)
        self.inp = Linear(ck, f"language_model.backbone.layers.{layer}.mixer.in_proj")
        self.out = Linear(ck, f"language_model.backbone.layers.{layer}.mixer.out_proj")
        self.norm = ck.get(f"language_model.backbone.layers.{layer}.norm.weight").cuda()
        self.eps, self.lo, self.hi = eps, lo, hi
        self.reset()

    def reset(self):
        self.hist = torch.zeros(mo.CONV, mo.W - 1, device="cuda", dtype=BF)
        self.state = torch.zeros(mo.H, mo.P, mo.N, device="cuda")

    def __call__(self, x, quant_act):
        hn = rmsnorm(x, self.norm, self.eps)
        proj = self.inp(hn, quant_act)
        _, _, gated, self.hist, self.state = mo.ref_forward(
            self.w, proj, self.hist, self.state, self.lo, self.hi)
        self.state = self.state.half().float()  # the fp16 slot state (device decode rounds it SR)
        return self.out(gated, quant_act)


class Attention:
    def __init__(self, ck, layer, eps, heads, kvh, hd):
        m = f"language_model.backbone.layers.{layer}.mixer"
        self.q, self.k, self.v, self.o = (Linear(ck, f"{m}.{p}_proj") for p in "qkvo")
        self.norm = ck.get(f"language_model.backbone.layers.{layer}.norm.weight").cuda()
        self.eps, self.heads, self.kvh, self.hd = eps, heads, kvh, hd
        self.reset()

    def reset(self):
        self.kc = torch.zeros(0, self.kvh, self.hd, device="cuda", dtype=BF)
        self.vc = torch.zeros(0, self.kvh, self.hd, device="cuda", dtype=BF)

    def __call__(self, x, quant_act):
        T = x.shape[0]
        hn = rmsnorm(x, self.norm, self.eps)
        q = self.q(hn, quant_act).view(T, self.heads, self.hd)
        k = self.k(hn, quant_act).view(T, self.kvh, self.hd)
        v = self.v(hn, quant_act).view(T, self.kvh, self.hd)
        p0 = self.kc.shape[0]
        if KV == "fp8":  # every reader (prefill and decode) sees the stored e4m3 values
            k, v = (bf(t.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()) for t in (k, v))
        self.kc = torch.cat([self.kc, k])
        self.vc = torch.cat([self.vc, v])
        g = self.heads // self.kvh
        if T * self.kc.shape[0] > 1 << 26:  # long context: query chunks over the causal keys only
            o = torch.empty(T, self.kvh, g, self.hd, device="cuda")
            qc = q.float().view(T, self.kvh, g, self.hd)
            for t0 in range(0, T, 64):
                t1, n = min(t0 + 64, T), p0 + min(t0 + 64, T)
                kk, vv = self.kc[:n].float(), self.vc[:n].float()
                s = torch.einsum("tkgd,skd->kgts", qc[t0:t1], kk) / math.sqrt(self.hd)
                pos_q = torch.arange(p0 + t0, p0 + t1, device="cuda")[:, None]
                s = s.masked_fill((torch.arange(n, device="cuda")[None, :] > pos_q)[None, None], float("-inf"))
                o[t0:t1] = torch.einsum("kgts,skd->tkgd", torch.softmax(s, -1), vv)
                del s
            return self.o(bf(o.reshape(T, -1)), quant_act)
        kk = self.kc.float().repeat_interleave(g, 1)  # [S, H, D]
        vv = self.vc.float().repeat_interleave(g, 1)
        s = torch.einsum("thd,shd->hts", q.float(), kk) / math.sqrt(self.hd)
        pos_q = torch.arange(p0, p0 + T, device="cuda")[:, None]
        pos_k = torch.arange(self.kc.shape[0], device="cuda")[None, :]
        s = s.masked_fill((pos_k > pos_q)[None], float("-inf"))
        o = torch.einsum("hts,shd->thd", torch.softmax(s, -1), vv)
        return self.o(bf(o.reshape(T, -1)), quant_act)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def nvfp4(ck, module, split=False):
    """Dequantized NVFP4 linear `module` ([N, K] f32): e2m1(nib) * e4m3(S[n][k/16]) * gs
    (low nibble = even k, the vLLM `break_fp4_bytes` order). `split`: (bf16 e2m1 * S, which
    is exact in bf16, and gs)."""
    w = ck.get(module + ".weight").cuda()
    s = ck.get(module + ".weight_scale").cuda().float()
    gs = ck.get(module + ".weight_scale_2").float().item()
    lut = E2M1.cuda()
    v = torch.stack([lut[(w & 15).long()], lut[(w >> 4).long()]], -1).reshape(w.shape[0], -1)
    v = v * s.repeat_interleave(16, 1)
    return (v.to(BF), gs) if split else v * gs


def e2m1_round(x):
    """Round to the e2m1 grid, ties to even (vLLM cast_to_fp4 / cvt.rn.satfinite.e2m1x2)."""
    a = x.abs()
    q = torch.zeros_like(a)
    for lo, v, inc in ((0.25, 0.5, False), (0.75, 1.0, True), (1.25, 1.5, False), (1.75, 2.0, True),
                       (2.5, 3.0, False), (3.5, 4.0, True), (5.0, 6.0, False)):
        q = torch.where(a >= lo if inc else a > lo, torch.full_like(a, v), q)
    return q * torch.sign(x)


def nvfp4_qdq(x, input_scale):
    """vLLM ref_nvfp4_quant + dequant, global scale 1 / input_scale, blocks of 16."""
    gs = 1.0 / input_scale
    xf = x.float().reshape(x.shape[0], -1, 16)
    amax = xf.abs().amax(-1, keepdim=True)
    sf = (gs * amax / 6.0).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    mul = torch.where(sf != 0, 1.0 / (sf * (1.0 / gs)), torch.zeros_like(sf))
    q = e2m1_round((xf * mul).clamp(-6, 6))
    return (q * sf / gs).reshape(x.shape)


class Moe:
    """Latent sigmoid top-k MoE in the device's rounding order (ABI: f32 parts, one bf16 round at
    the latent combine, bf16(routed + shared)). Prefill activations per MOE_ACT: nvfp4 (W4A4, the
    peer: xl / hn and relu^2 quantized with the layer's input_scales, bf16 GEMM outputs) or bf16
    (W4A16: fu bf16). Decode is W4A16."""

    def __init__(self, ck, layer, t):
        m = f"language_model.backbone.layers.{layer}.mixer"
        self.ck, self.m = ck, m
        self.norm = ck.get(f"language_model.backbone.layers.{layer}.norm.weight").cuda()
        self.eps = t["layer_norm_epsilon"]
        self.gate = ck.get(m + ".gate.weight").cuda().float()
        self.bias = ck.get(m + ".gate.e_score_correction_bias").cuda().float()
        self.fc1 = ck.get(m + ".fc1_latent_proj.weight").cuda().float()
        self.fc2 = ck.get(m + ".fc2_latent_proj.weight").cuda().float()
        self.k, self.scale = t["num_experts_per_tok"], t["routed_scaling_factor"]
        self.norm_topk = t["norm_topk_prob"]
        self.su = nvfp4(ck, m + ".shared_experts.up_proj")
        self.sd = nvfp4(ck, m + ".shared_experts.down_proj")
        self.cache = {}
        isc = lambda n: ck.get(n).float().reshape(-1).max().item()
        self.in_up = max(isc(f"{m}.experts.{e}.up_proj.input_scale") for e in range(t["n_routed_experts"]))
        self.in_dn = max(isc(f"{m}.experts.{e}.down_proj.input_scale") for e in range(t["n_routed_experts"]))
        self.in_su = isc(m + ".shared_experts.up_proj.input_scale")
        self.in_sd = isc(m + ".shared_experts.down_proj.input_scale")

    def expert(self, e):
        if e not in self.cache:
            self.cache[e] = (nvfp4(self.ck, f"{self.m}.experts.{e}.up_proj", True),
                             nvfp4(self.ck, f"{self.m}.experts.{e}.down_proj", True))
        (up, gu), (down, gd) = self.cache[e]
        return up.float() * gu, down.float() * gd

    def reset(self):
        pass

    def __call__(self, x, quant_act):
        hn = rmsnorm(x, self.norm, self.eps)
        h = hn.float()
        s = torch.sigmoid(h @ self.gate.t())
        top = torch.topk(s + self.bias, self.k + 1, -1).values
        idx = torch.topk(s + self.bias, self.k, -1).indices
        if getattr(self, "force", None) is not None:  # --route-check: the device's expert sets
            idx, self.force = self.force, None
        # Selection margin (k-th minus (k+1)-th biased score) per row, for --route-check.
        self.last = (idx, (top[:, -2] - top[:, -1]))
        w = s.gather(-1, idx)
        if self.norm_topk:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * self.scale
        xl = bf(h @ self.fc1.t()).float()
        w4a4 = quant_act and MOE_ACT == "nvfp4"
        if w4a4:
            xl = nvfp4_qdq(xl, self.in_up)
        lat = torch.zeros(x.shape[0], self.fc1.shape[0], device="cuda")
        for e in idx.unique().tolist():
            rows, slot = (idx == e).nonzero(as_tuple=True)
            up, down = self.expert(e)
            if w4a4:
                fu = nvfp4_qdq(torch.relu(bf(xl[rows] @ up.t()).float()).square(), self.in_dn)
            else:
                fu = bf(torch.relu(xl[rows] @ up.t()).square()).float()
            y = fu @ down.t()
            if w4a4:  # bf16 expert outputs (down mode 2), gated in the f32 combine
                y = bf(y).float()
            lat.index_add_(0, rows, y * w[rows, slot][:, None])
        rout = bf(bf(lat).float() @ self.fc2.t())
        if w4a4:
            sfu = nvfp4_qdq(torch.relu(bf(nvfp4_qdq(h, self.in_su) @ self.su.t()).float()).square(),
                            self.in_sd)
        else:
            sfu = bf(torch.relu(h @ self.su.t()).square()).float()
        sout = bf(sfu @ self.sd.t())
        return bf(rout.float() + sout.float())


class Block:
    def __init__(self, ck, cfg, layers):
        t = cfg.get("llm_config", cfg)
        lo, hi = (float(v) if not isinstance(v, dict) else math.inf for v in t["time_step_limit"])
        eps = t["layer_norm_epsilon"]
        self.mixers = []
        for l in layers:
            kind = t["layers_block_type"][l]
            if kind == "mamba":
                self.mixers.append(Mamba(ck, l, eps, lo, hi))
            elif kind == "attention":
                self.mixers.append(Attention(ck, l, eps, t["num_attention_heads"],
                                             t["num_key_value_heads"], t["head_dim"]))
            elif kind == "moe":
                self.mixers.append(Moe(ck, l, t))
            else:
                raise SystemExit(f"layer {l} is {kind}")

    def reset(self):
        for m in self.mixers:
            m.reset()

    def __call__(self, x, quant_act):
        for m in self.mixers:
            x = bf(x.float() + m(x, quant_act).float())
        return x


def run(cmd, env):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"{cmd[1:3]} failed rc={r.returncode}")
    return r.stdout


def rel(a, b):
    a, b = torch.as_tensor(a).float().cuda(), torch.as_tensor(b).float().cuda()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--block-run", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layers", required=True, help="comma list, the block's layers in order")
    ap.add_argument("--prefill", default="7,128,1000,1500,1537,2048,3000,6000",
                    help="prefill rows; off-bucket T run as several chunks (buckets 128/1024/4096/8192)")
    ap.add_argument("--decode-ctx", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--limit", type=float, default=2e-2, help="rel-L2 of the block delta")
    ap.add_argument("--fp8-act", choices=["static", "row"], default="static",
                    help="prefill FP8 activation quant the packet runs (static = input_scale)")
    ap.add_argument("--moe-act", choices=["nvfp4", "bf16"], default="nvfp4",
                    help="prefill activations of the NVFP4 experts the packet runs (nvfp4 = W4A4)")
    ap.add_argument("--kv", choices=["bf16", "fp8"], default="bf16",
                    help="KV cache elements the packet stores (fp8 = PLOW_FP8_KV, unit-scale e4m3)")
    ap.add_argument("--route-check", action="store_true",
                    help="compare the device's top-k sets of the last MoE layer with the reference's "
                         "(prefill T <= 1024; decode: the last step of every slot)")
    a = ap.parse_args()
    global FP8_ACT, MOE_ACT, KV
    FP8_ACT, MOE_ACT, KV = a.fp8_act, a.moe_act, a.kv
    os.makedirs(a.scratch, exist_ok=True)
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    ck = mo.Ckpt(a.ckpt)
    layers = [int(v) for v in a.layers.split(",")]
    blk = Block(ck, cfg, layers)
    env = dict(os.environ, PLOW_CHECKPOINT=a.ckpt)
    gen = torch.Generator(device="cuda").manual_seed(7)
    rows, ok = [], True

    def x_rows(n):
        return bf(torch.randn(n, 4096, device="cuda", generator=gen))

    def save(name, x):
        p = os.path.join(a.scratch, name)
        np.save(p, x.float().cpu().numpy().astype(np.float32))
        return p

    moe = [m for m in blk.mixers if isinstance(m, Moe)]
    route = a.route_check and bool(moe)

    def dump_args(name):
        return ["--dump-tensors", "act.moe.table", "--dump-dir", os.path.join(a.scratch, name)] if route else []

    def route_diff(name, n, ref_idx, margin, label):
        """Device top-k sets (the last MoE layer's act.moe.table) vs the reference's, rows [0, n)."""
        d = os.path.join(a.scratch, name)
        f = next(os.path.join(d, p) for p in sorted(os.listdir(d)) if p.endswith(".bin"))
        k = moe[-1].k
        dev = torch.from_numpy(np.fromfile(f, dtype=np.uint32).reshape(-1, 2)[:n * k, 0].astype(np.int64))
        dev = dev.view(n, k).sort(-1).values.cuda()
        diff = (dev != ref_idx.sort(-1).values).any(-1)
        m = margin[diff]
        print(f"route {label}: {int(diff.sum())}/{n} rows route to a different expert set; margin "
              f"(k-th - (k+1)-th biased score) min {margin.min().item():.2e}, on differing rows "
              f"{'-' if m.numel() == 0 else f'max {m.max().item():.2e}'}")
        return diff

    for T in [int(v) for v in a.prefill.split(",")]:
        x = x_rows(T)
        xin, yout = save(f"x{T}.npy", x), os.path.join(a.scratch, f"y{T}.npy")
        run([a.block_run, a.asset, "check", "--in", xin, "--out", yout] + dump_args(f"route-pf{T}"), env)
        y = torch.from_numpy(np.load(yout)).cuda()
        blk.reset()
        ref = blk(x, True)
        if route and T <= 1024:  # one launch: the dumped table is every row's
            diff = route_diff(f"route-pf{T}", T, *moe[-1].last, f"prefill T={T}")
            if bool(diff.any()) and len(moe) == 1:
                # The same reference on the device's expert sets: what the near-tie flips cost.
                d = os.path.join(a.scratch, f"route-pf{T}")
                f = next(os.path.join(d, p) for p in sorted(os.listdir(d)) if p.endswith(".bin"))
                dev = np.fromfile(f, dtype=np.uint32).reshape(-1, 2)[:T * moe[0].k, 0].astype(np.int64)
                moe[0].force = torch.from_numpy(dev).view(T, moe[0].k).cuda()
                blk.reset()
                rd = rel(y - x.float(), blk(x, True).float() - x.float())
                rows.append((f"prefill T={T} (device routing)", rd, float("nan"), "info"))
        blk.reset()
        ideal = blk(x, False)
        r = rel(y - x.float(), ref.float() - x.float())
        ri = rel(y - x.float(), ideal.float() - x.float())
        rows.append((f"prefill T={T}", r, ri, "ok" if r <= a.limit else "FAIL"))
        ok &= r <= a.limit

    T, B, S = a.decode_ctx, a.batch, a.steps
    x = x_rows(T)
    r_rows = x_rows(S * B)
    yout = os.path.join(a.scratch, "ydec.npy")
    run([a.block_run, a.asset, "decode-check", "--in", save("xdec.npy", x), "--rows",
         save("rdec.npy", r_rows), "--batch", str(B), "--steps", str(S), "--out", yout]
        + dump_args("route-dec"), env)
    y = torch.from_numpy(np.load(yout)).cuda().view(S, B, 4096)
    last_idx, last_margin = [], []
    for b in range(B):
        blk.reset()
        blk(x, True)  # the prefill the device ran for this slot
        for s in range(S):
            xr = r_rows[s * B + b:s * B + b + 1]
            ref = blk(xr, False)
            if route and s == S - 1:
                last_idx.append(moe[-1].last[0])
                last_margin.append(moe[-1].last[1])
            r = rel(y[s, b] - xr[0].float(), ref[0].float() - xr[0].float())
            rows.append((f"decode B={B} slot {b} step {s} (after T={T})", r, float("nan"),
                         "ok" if r <= a.limit else "FAIL"))
            ok &= r <= a.limit
    if route:
        diff = route_diff("route-dec", B, torch.cat(last_idx), torch.cat(last_margin),
                          f"decode B={B} step {S - 1}")
        print(f"  slots with a different set: {diff.nonzero().flatten().tolist()}")

    print(f"{'case':42s} {'rel-L2 dev':>11s} {'rel-L2 ideal':>13s}  verdict")
    for c, r, ri, v in rows:
        print(f"{c:42s} {r:11.3e} {ri:13.3e}  {v}")
    print(f"{sum(r[3] == 'ok' for r in rows)}/{len(rows)} pass (limit {a.limit} on the block delta; "
          f"'ideal' = no activation quantization in the reference)")
    json.dump([dict(zip(("case", "rel_l2_device_model", "rel_l2_ideal", "verdict"), r)) for r in rows],
              open(os.path.join(a.scratch, "block_oracle.json"), "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
