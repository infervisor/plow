"""gen_search block spec: Nemotron-H prefill routed MoE, NVFP4 W4A4 grouped GEMM. Geometry from
$NEMOTRON35_CHECKPOINT/config.json (the recipes' checkpoint variable; `llm_config` of an omni
checkpoint): E = n_routed_experts, top-k = num_experts_per_tok, up N = moe_intermediate_size, up
K = moe_latent_size (hidden_size without a latent space). Unset: Nemotron-3.5-Super (E 512, top-22). Per expert e: C[rows_e][N] = alpha_e * A_e[rows_e][K] . W_e[N][K]^T, alpha_e = a_scale *
gs[e], f32 accumulate. Operands NVFP4: e2m1 bytes [R][K/2] (low nibble = even k) and one e4m3 scale
per 16 k, row-major [R][K/16]. Rows sorted by expert; the up GEMM gathers A by row_token from the
[T][K] activation, the down GEMM reads the sorted rows.

  up   N I K L (gathered A)  |  down N L K I (sorted A)       R = T x top-k rows
  (3.5-Super: I 2688, L 1024, top-22)

Generated body: Triton tl.dot_scaled e2m1 x e2m1 with e4m3 per-16 scales, which Triton 3.8 lowers on
sm_121a to mma.sync m16n8k64 kind::mxf4nvf4 block_scale scale_vec::4X ue4m3 (the plow kernel's
instruction). TileLang 0.1.12 has block-scaled GEMM only as tcgen05 (sm_100, E8M0 scales), so it
has no entry. Inputs are synthetic: uniform random routing, random e2m1 codes, e4m3 scales in
[2^-3, 2^3]. Reference: f32 GEMM (IEEE, no TF32) of the dequantized operands, whose products are
exact in f32, so a correct kernel differs only in summation order (rel. L2 ~1e-7). The bf16-output
classes round the output (~1e-3), hence the bound.

The plow kernel fuses more than this GEMM: up gathers, applies relu^2 and requantizes to NVFP4 (fq,
272 MB at T 8192 against 1.9 GB of f32 here); down applies the router gate and scatters an f32 part
in token order. Floors use GB10 ceilings with fp4_tflops (pass --ceilings).
"""
import json
import os
from pathlib import Path

# Nemotron-3.5-Super: the geometry `current` was measured at.
SUPER = (512, 22, 2688, 1024)


def _geometry():
    """(experts, top-k, moe_intermediate_size, routed-expert input width)."""
    ckpt = os.environ.get("NEMOTRON35_CHECKPOINT")
    if not ckpt:
        return SUPER
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    cfg = cfg.get("llm_config", cfg)
    return (cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"],
            cfg.get("moe_latent_size") or cfg["hidden_size"])


E, KSEL, _I, _L = _geometry()
SHAPES = {"up": (_I, _L), "down": (_L, _I)}  # kind: (N, K)
ROUTE_SEED = 1234


def _classes():
    out = {}
    for t in (8192, 1024):
        for kind in ("up", "down"):
            for dt in (("f32", "bf16") if t == 8192 else ("f32",)):
                n, k = SHAPES[kind]
                out[f"{kind}_t{t}" + ("" if dt == "f32" else "_bf16")] = {
                    "kind": kind, "T": t, "R": t * KSEL, "N": n, "K": k, "out": dt}
    return out


def _cost(c):
    n, k, r = c["N"], c["K"], c["R"]
    a_rows = c["T"] if c["kind"] == "up" else r  # up: the gathered rows re-read an L2-sized tensor
    return {"bytes": E * n * k * 9 // 16 + a_rows * k * 9 // 16 + r * n * (4 if c["out"] == "f32" else 2),
            "flops": 2 * r * n * k, "peak": "fp4"}


def _routing(c):
    import torch
    g = torch.Generator(device="cuda").manual_seed(ROUTE_SEED + c["T"])
    eid = torch.rand(c["T"], E, device="cuda", generator=g).topk(KSEL, -1).indices.reshape(-1)
    order = torch.sort(eid, stable=True).indices
    rows = (order // KSEL).int()
    off = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    off[1:] = torch.bincount(eid, minlength=E).cumsum(0).int()
    return rows, off


def _e4m3(shape, g):
    import torch
    # positive normal e4m3 with exponent field 4..10: 2^-3 .. 2^3 (x 1.875)
    return torch.randint(32, 88, shape, device="cuda", dtype=torch.uint8, generator=g).view(torch.float8_e4m3fn)


def _inputs(c, seed):
    import torch
    n, k = c["N"], c["K"]
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + n + c["T"])
    a_rows = c["T"] if c["kind"] == "up" else c["R"]
    rows, off = _routing(c)
    return {
        "a": torch.randint(0, 256, (a_rows, k // 2), device="cuda", dtype=torch.uint8, generator=g),
        "as": _e4m3((a_rows, k // 16), g),
        "w": torch.randint(0, 256, (E, n, k // 2), device="cuda", dtype=torch.uint8, generator=g),
        "ws": _e4m3((E, n, k // 16), g),
        "gs": torch.rand(E, device="cuda", generator=g) * 1e-3 + 5e-4,
        "a_scale": torch.tensor([0.03125], device="cuda"),
        "rows": rows, "off": off,
    }


def _dequant(codes, scales):
    import torch
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
    v = torch.stack([lut[(codes & 15).long()], lut[(codes >> 4).long()]], -1).reshape(codes.shape[0], -1)
    return v * scales.float().repeat_interleave(16, 1)


def _reference(c, inp):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    off = inp["off"].tolist()
    out = torch.empty(c["R"], c["N"], device="cuda")
    a_all = _dequant(inp["a"], inp["as"]) if c["kind"] == "up" else None
    a_scale = inp["a_scale"]
    for e in range(E):
        r0, r1 = off[e], off[e + 1]
        if r0 == r1:
            continue
        if a_all is not None:
            a = a_all[inp["rows"][r0:r1].long()]
        else:
            a = _dequant(inp["a"][r0:r1], inp["as"][r0:r1])
        w = _dequant(inp["w"][e], inp["ws"][e])
        out[r0:r1] = (a @ w.T) * (a_scale * inp["gs"][e])
    return out.double()


_TR = []


def _kernel():
    if _TR:
        return _TR[0]
    import triton
    import triton.language as tl

    @triton.jit
    def moe_w4a4(a_ptr, as_ptr, w_ptr, ws_ptr, gs_ptr, sc_ptr, rows_ptr, off_ptr, te_ptr, tr_ptr, c_ptr,
                 N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 GATHER: tl.constexpr, BF16: tl.constexpr, STAGES: tl.constexpr):
        NTN: tl.constexpr = N // BN
        pid = tl.program_id(0)
        pm = pid // NTN
        pn = pid % NTN
        e = tl.load(te_ptr + pm)
        r0 = tl.load(tr_ptr + pm)
        r1 = tl.load(off_ptr + e + 1)
        om = r0 + tl.arange(0, BM)
        mok = om < r1
        if GATHER:
            src = tl.load(rows_ptr + om, mask=mok, other=0)
        else:
            src = tl.where(mok, om, 0)
        src = src.to(tl.int64)
        on = pn * BN + tl.arange(0, BN)
        k2 = tl.arange(0, BK // 2)
        ks = tl.arange(0, BK // 16)
        ap = a_ptr + src[:, None] * (K // 2) + k2[None, :]
        asp = as_ptr + src[:, None] * (K // 16) + ks[None, :]
        wb = e.to(tl.int64) * N + on
        wp = w_ptr + wb[:, None] * (K // 2) + k2[None, :]
        wsp = ws_ptr + wb[:, None] * (K // 16) + ks[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in tl.range(0, K // BK, num_stages=STAGES):
            a = tl.load(ap)
            sa = tl.load(asp)
            b = tl.load(wp)
            sb = tl.load(wsp)
            acc = tl.dot_scaled(a, sa, "e2m1", b.T, sb, "e2m1", acc)
            ap += BK // 2
            asp += BK // 16
            wp += BK // 2
            wsp += BK // 16
        acc = acc * (tl.load(sc_ptr) * tl.load(gs_ptr + e))
        cp = c_ptr + om.to(tl.int64)[:, None] * N + on[None, :]
        if BF16:
            tl.store(cp, acc.to(tl.bfloat16), mask=mok[:, None])
        else:
            tl.store(cp, acc, mask=mok[:, None])

    _TR.append(moe_w4a4)
    return moe_w4a4


def _tiles(off, bm):
    """Per M tile (expert-major, one expert per tile): its expert and first row."""
    import torch
    te, tr = [], []
    o = off.tolist()
    for e in range(E):
        for r in range(o[e], o[e + 1], bm):
            te.append(e)
            tr.append(r)
    return (torch.tensor(te, dtype=torch.int32, device="cuda"),
            torch.tensor(tr, dtype=torch.int32, device="cuda"))


def _build(cfg, c):
    return _kernel(), cfg


def _bind(built, c, inp, out):
    kern, cfg = built
    te, tr = _tiles(inp["off"], cfg["bm"])
    grid = (te.numel() * (c["N"] // cfg["bn"]),)
    args = (inp["a"], inp["as"], inp["w"], inp["ws"], inp["gs"], inp["a_scale"], inp["rows"], inp["off"],
            te, tr, out)
    kw = dict(N=c["N"], K=c["K"], BM=cfg["bm"], BN=cfg["bn"], BK=cfg["bk"], GATHER=c["kind"] == "up",
              BF16=c["out"] == "bf16", STAGES=cfg["stages"], num_warps=cfg["warps"])
    return lambda: kern[grid](*args, **kw)


def _valid(cfg, c):
    return c["N"] % cfg["bn"] == 0 and c["K"] % cfg["bk"] == 0


# (bm, bn, bk, warps, stages); smem per stage = (bm + bn) * bk * 9/16 bytes, sm_121a has 99 KB.
SWEEP = [dict(bm=bm, bn=bn, bk=bk, warps=w, stages=s) for bm, bn, bk, w, s in (
    (128, 128, 128, 8, 3), (128, 128, 128, 4, 3), (128, 128, 128, 8, 2), (128, 128, 128, 8, 4),
    (128, 128, 256, 8, 2), (128, 64, 128, 4, 3), (64, 128, 128, 4, 3), (64, 128, 256, 4, 2),
    (64, 64, 128, 4, 4), (128, 256, 128, 8, 2), (256, 128, 128, 8, 2), (64, 256, 128, 8, 3))]


SPEC = {
    "name": "nemotron_moe_w4a4",
    "op": f"routed MoE NVFP4 grouped GEMM E{E} top{KSEL}",
    "dtype": "e2m1 x e2m1, e4m3/16 scales -> f32 acc x alpha -> f32 | bf16",
    "classes": _classes(),
    "cost": _cost,
    "inputs": _inputs,
    "rotate": ["a", "as", "w", "ws"],
    "out": lambda c: ((c["R"], c["N"]), "float32" if c["out"] == "f32" else "bfloat16"),
    "reference": _reference,
    "tol": {"rel_l2": 4e-3},
    "generators": {
        "triton": {"sweep": SWEEP, "build": _build, "bind": _bind, "valid": _valid},
    },
    # Measured on the 3.5-Super geometry only.
    "current": None if (E, KSEL, _I, _L) != SUPER else {
        "label": "plow op_moe_nvfp4_pf.cuh W4A4 v2 nf4v_tiled (mma.sync mxf4nvf4), w4a4_test.py on checkpoint "
                 "layer 1 with its own router (skewed, not this spec's uniform routing); up includes the "
                 "gather, relu^2 and NVFP4 requant of fq, down the gate and the f32 (bf16: dmode 2) scatter",
        "measured_us": {"up_t8192": 7612.3, "down_t8192": 10102.6, "down_t8192_bf16": 8622.8,
                        "up_t1024": 3800.8, "down_t1024": 4615.9},
        "source": "w4a4_test.py gpuq job w4v2a, GB10 jg01, 2026-11-22",
    },
}
