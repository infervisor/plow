"""gen_search block spec: Maya 2 DepthV2 projections (GemvF32Out mma arm: f16 W [N][K], x rounded
to f16, f32 accumulate) at the depth stage's shapes: qkv / f1 (N 3072, K 1024), f2 (N 1024,
K 3072), o (1024 x 1024), head (N 2048, K 1024). Row classes: m1 = one stream's cached level,
m8 / m32 = 8 / 32 streams, m832 = a full level (26 prefix rows) at 32 streams. Generated bodies:
gemm_f16.py (TileLang), gemm_f16_triton.py (Triton). Current kernel: recorded in-model per-op time (packet_bench --sweep of the standalone
depth program, RC9 speech object, H100); the generated number is a standalone launch, so the
comparison is a reference for the schedule, not an in-model gain.
"""

SHAPES = {"qkv": (3072, 1024), "f2": (1024, 3072), "o": (1024, 1024), "head": (2048, 1024)}
# B=1 frame: 16.73 ms; qkv + f1 share the shape (4 layers x 20 cached levels x 2), f2 and o 80, head 31.
STEP_B1 = {"qkv": 160, "f2": 80, "o": 80, "head": 31}
CURRENT_US = {"qkv_m1": 15.1, "f2_m1": 28.6, "o_m1": 12.7, "head_m1": 18.2, "qkv_m8": 20.7, "f2_m8": 43.4}


def _classes():
    out = {}
    for s, (n, k) in SHAPES.items():
        for m in (1, 8, 32, 832):
            if s == "head" and m == 832:
                continue  # the head reads the last prefix row only
            c = {"shape": s, "M": m, "N": n, "K": k}
            if m == 1:
                c["step"] = {"us": 16730.0, "calls": STEP_B1[s]}
            out[f"{s}_m{m}"] = c
    return out


def _cost(c):
    m, n, k = c["M"], c["N"], c["K"]
    return {"bytes": n * k * 2 + m * k * 2 + m * n * 4, "flops": 2 * m * n * k, "peak": "bf16", "passes": 1}


def _inputs(c, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(c["M"], c["K"], device="cuda", generator=g).half()
    w = (torch.randn(c["N"], c["K"], device="cuda", generator=g) * 0.05).half()
    return {"x": x, "w": w}


def _reference(c, inp):
    return inp["x"].double() @ inp["w"].double().T


def _tl_build(cfg, c):
    import gemm_f16
    return gemm_f16.compile(c["M"], c["N"], c["K"], cfg["bm"], cfg["bn"], cfg["bk"], cfg["threads"], cfg["stages"])


def _tl_valid(cfg, c):
    # one row tile for the single-stream classes; 64+ row tiles only where there are rows to fill them
    if c["M"] <= 16:
        return cfg["bm"] == 16
    if c["M"] <= 32:
        return cfg["bm"] == 32
    return cfg["bm"] >= 64


def _tr_bind(cfg, c, inp, out):
    import gemm_f16_triton
    return gemm_f16_triton.launcher(cfg, c["M"], c["N"], c["K"], inp["x"], inp["w"], out)


SPEC = {
    "name": "maya2_depth_gemv",
    "op": "GemvF32Out depth projection",
    "dtype": "f16 W x f16 x -> f32 (mma)",
    "classes": _classes(),
    "cost": _cost,
    "inputs": _inputs,
    "rotate": ["w"],
    "out": lambda c: ((c["M"], c["N"]), "float32"),
    "reference": _reference,
    "tol": {"rel_l2": 1e-5},
    "generators": {
        "tilelang": {
            "grid": {"bm": [16, 32, 64, 128], "bn": [64, 128], "bk": [32, 64], "threads": [128, 256],
                     "stages": [2, 3]},
            "build": _tl_build,
            "bind": lambda kern, c, inp, out: (lambda: kern(inp["x"], inp["w"], out)),
            "valid": _tl_valid,
            "static_shapes": True,
        },
        # Triton compiles at first launch: `build` hands the config through and `bind` jits.
        "triton": {
            "grid": {"bm": [16, 32, 64, 128], "bn": [64, 128], "bk": [32, 64], "warps": [4, 8], "stages": [2, 3]},
            "build": lambda cfg, c: cfg,
            "bind": _tr_bind,
            "valid": _tl_valid,
        },
    },
    "current": {
        "label": "gemv_f16w_mma in the persistent interpreter (per-op delta of packet_bench --sweep)",
        "measured_us": CURRENT_US,
        "source": "tts-perf/tp-maya2-depth/baseline/sweep.b1.jsonl (lease tp-maya2-depth-baseline, RC9 speech object)",
    },
}
