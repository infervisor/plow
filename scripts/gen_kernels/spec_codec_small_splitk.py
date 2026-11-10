"""gen_search block spec: the 12 Hz codec transformer's small-M pointwise convs (y = x W^T + b,
1024->512 x17 and 4096->1024 x2 per step) with fixed split-K parts (P partial tiles, ordered fold:
the reduction order depends on P alone, never on rows). Current plow: wgmma pointwise tile, unsplit.
"""
import codec_spec_common as cc


def _cls(role, m, n, k, calls, us):
    return {"M": m, "N": n, "K": k, "step": {"us": cc.STEP_US[role], "calls": calls}, "us": us}


CLASSES = {
    "d1024x512_b1": _cls("b1", 1, 512, 1024, 17, 33.8),
    "d1024x512_b8": _cls("b8", 136, 512, 1024, 17, 35.4),
    "d1024x512_b32": _cls("b32", 544, 512, 1024, 17, 39.4),
    "d4096x1024_b1": _cls("b1", 2, 1024, 4096, 2, 117.0),
    "d4096x1024_b8": _cls("b8", 272, 1024, 4096, 2, 134.0),
    "d4096x1024_b32": _cls("b32", 1088, 1024, 4096, 2, 198.0),
}


def _cost(c):
    return {"bytes": 4 * (c["M"] * c["K"] + c["M"] * c["N"] + c["N"] * c["K"]),
            "flops": 2 * c["M"] * c["N"] * c["K"], "passes": 3, "peak": "tf32"}


def _inputs(c, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    gw = torch.Generator(device="cuda").manual_seed(1234)
    w = torch.randn(c["N"], c["K"], device="cuda", generator=gw) / c["K"] ** 0.5
    wh = cc._hi(w)
    return {"x": torch.randn(c["M"], c["K"], device="cuda", generator=g), "w": w, "wh": wh,
            "wl": (w - wh).contiguous(), "bias": torch.randn(c["N"], device="cuda", generator=gw) * 0.1}


def _reference(c, inp):
    return inp["x"].double() @ inp["w"].double().T + inp["bias"].double()


def _tl_build(cfg, c):
    import codec_blocks as cb
    d, f = cb.build_dense(c["M"], c["N"], c["K"], cfg["bm"], cfg["bn"], 64, cfg["threads"], cfg["stages"], cfg["parts"])
    return cb.compile(d), cb.compile(f)


def _tl_bind(kern, c, inp, out):
    import torch
    parts = _tl_bind.parts[id(kern)]
    part = torch.empty(parts, c["M"], c["N"], device="cuda")

    def call():
        kern[0](inp["x"], inp["wh"], inp["wl"], part)
        kern[1](part, inp["bias"], out)
    return call


_tl_bind.parts = {}


def _tl_build_tracked(cfg, c):
    k = _tl_build(cfg, c)
    _tl_bind.parts[id(k)] = cfg["parts"]
    return k


def _tr_bind(cfg, c, inp, out):
    import torch
    import triton
    import codec_blocks_triton as tb
    m, n, k, p = c["M"], c["N"], c["K"], cfg["parts"]
    chunk = -(-k // 64 // p) * 64
    part = torch.empty(p, m, n, device="cuda")
    grid = (triton.cdiv(m, cfg["bm"]), triton.cdiv(n, cfg["bn"]), p)

    def call():
        tb.tdense[grid](inp["x"], inp["w"], part, m, n, k, chunk, cfg["bm"], cfg["bn"], 64, cfg["prec"],
                        num_warps=cfg["warps"], num_stages=3)
        tb.tfold[(triton.cdiv(m * n, 1024),)](part, inp["bias"], out, m * n, n, p, 1024)
    return call


SPEC = {
    "name": "codec_small_splitk",
    "op": "pointwise Conv1dF32 small M, fixed split-K parts",
    "dtype": "f32 (3xTF32; plow: split-bf16 wgmma)",
    "classes": CLASSES,
    "cost": _cost,
    "inputs": _inputs,
    "rotate": ["x"],
    "out": lambda c: ((c["M"], c["N"]), "float32"),
    "reference": _reference,
    # the served split-bf16 tiles are gated at 1e-4 against the golden (speech_f32_op_test)
    "tol": {"rel_l2": 1e-4},
    "generators": {
        "tilelang": {"grid": {"bm": [64], "bn": [64], "threads": [128], "stages": [3], "parts": [1, 2, 4]},
                     "build": _tl_build_tracked, "bind": _tl_bind, "static_shapes": True},
        "triton": {"grid": {"prec": ["tf32x3", "tf32"], "bm": [16, 64], "bn": [64, 128], "warps": [4], "parts": [1, 2, 4]},
                   "build": lambda cfg, c: cfg, "bind": _tr_bind},
    },
    "current": {
        "label": "sp_wg_gemm pointwise tile (unsplit), packet_bench --sweep per call",
        "measured_us": {n: c["us"] for n, c in CLASSES.items()},
        "source": "tts-perf/tp-codec/verify/l7a (qwen3-tts codec step, tp-codec 14759885b)",
    },
}
