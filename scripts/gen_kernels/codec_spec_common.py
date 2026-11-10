"""Shared pieces of the 12 Hz codec gen_search specs (spec_codec_*.py): inputs, fp64 references and
generator bindings for the causal k-tap conv (optional SnakeBeta prologue / epilogue) and the
small-M dense with fixed split-K parts. Current-kernel times are recorded packet_bench --sweep
medians of the qwen3-tts codec step (H100, tp-codec 14759885b), per call."""

STEP_US = {"b1": 4786.0, "b8": 20507.0, "b32": 69721.0}


def _hi(t):
    import torch
    u = t.contiguous().view(torch.int32)
    return ((u + 0x1000) & -0x2000).view(torch.float32)


def _snake_beta(v, ab, n):
    import torch
    return v + ab[n:] * torch.sin(v * ab[:n]) ** 2


def conv_cost(c):
    m, kk = c["B"] * c["T"], c["k"] * c["cin"]
    return {"bytes": 4 * (m * c["cin"] + m * c["cout"] + c["cout"] * kk),
            "flops": 2 * m * c["cout"] * kk, "passes": 3, "peak": "tf32"}


def conv_inputs(c, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    cin, cout, k = c["cin"], c["cout"], c["k"]
    x = torch.randn(c["B"], c["T"], cin, device="cuda", generator=g)
    # weights and parameters are the model's: the same at every seed
    gw = torch.Generator(device="cuda").manual_seed(1234)
    w = torch.randn(cout, cin, k, device="cuda", generator=gw) / (cin * k) ** 0.5
    wt = w.permute(0, 2, 1).reshape(cout, k * cin).contiguous()
    wh = _hi(wt)
    return {"x": x, "w": w, "wt": wt, "wh": wh, "wl": (wt - wh).contiguous(),
            "bias": torch.randn(cout, device="cuda", generator=gw) * 0.1,
            "apre": torch.rand(2 * cin, device="cuda", generator=gw) * 0.8 + 0.2,
            "apost": torch.rand(2 * cout, device="cuda", generator=gw) * 0.8 + 0.2}


def conv_reference(c, inp):
    import torch
    x = inp["x"].double()
    if c.get("pre"):
        x = _snake_beta(x, inp["apre"].double(), c["cin"])
    xp = torch.nn.functional.pad(x.transpose(1, 2), (c["dil"] * (c["k"] - 1), 0))
    y = torch.nn.functional.conv1d(xp, inp["w"].double(), inp["bias"].double(), dilation=c["dil"]).transpose(1, 2)
    if c.get("post"):
        y = _snake_beta(y, inp["apost"].double(), c["cout"])
    return y


def conv_tl_build(cfg, c):
    import codec_blocks as cb
    return cb.compile(cb.build_conv(c["B"], c["T"], c["cin"], c["cout"], c["k"], c["dil"], cfg["bm"], cfg["bn"],
                                    cfg["bk"], cfg["threads"], cfg["stages"], bool(c.get("pre")), bool(c.get("post"))))


def conv_tl_bind(kern, c, inp, out):
    return lambda: kern(inp["x"], inp["wh"], inp["wl"], inp["bias"], inp["apre"], inp["apost"], out)


def conv_tl_valid(cfg, c):
    return cfg["bn"] <= max(c["cout"], 64) and (cfg["bn"] != 96 or c["cout"] % 96 == 0)


def conv_tr_bind(cfg, c, inp, out):
    import triton
    import codec_blocks_triton as tb
    m, kk = c["B"] * c["T"], c["k"] * c["cin"]
    grid = (triton.cdiv(m, cfg["bm"]), triton.cdiv(c["cout"], cfg["bn"]))
    return lambda: tb.tconv[grid](inp["x"], inp["wt"], inp["bias"], inp["apre"], inp["apost"], out, m, c["T"], c["T"],
                                  c["cin"], c["cout"], kk, c["dil"], c["dil"] * (c["k"] - 1), bool(c.get("pre")),
                                  bool(c.get("post")), cfg["bm"], cfg["bn"], cfg["bk"],
                                  num_warps=cfg["warps"], num_stages=cfg["stages"])


def conv_generators(tl_grid, tr_grid):
    return {
        "tilelang": {"grid": tl_grid, "build": conv_tl_build, "bind": conv_tl_bind, "valid": conv_tl_valid,
                     "static_shapes": True},
        "triton": {"grid": tr_grid, "build": lambda cfg, c: cfg, "bind": conv_tr_bind},
    }


def conv_class(role, cin, t, dil, us, pre=False, post=False):
    batch = {"b1": 1, "b8": 8, "b32": 32}[role]
    return {"B": batch, "T": t, "cin": cin, "cout": cin, "k": 7, "dil": dil, "pre": pre, "post": post,
            "step": {"us": STEP_US[role], "calls": 3}, "us": us}
