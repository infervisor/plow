"""Shared pieces of the 12 Hz codec gen_search specs (spec_codec_*.py): inputs, fp64 references and
generator bindings for the causal k-tap conv (optional SnakeBeta prologue / epilogue) and the
small-M dense with fixed split-K parts. The current plow kernel is measured in the same lease
through the probe codec_conv.cu: the interpreter's own arms (op_speech_f32.cuh) launched one op at
a time on 132 blocks with the flags word the production Qwen3-TTS recipe emits (disassembly of the
tts-perf-next packet: pinned, split-bf16 wgmma taps tile, ragged, 64-wide, SnakeBeta epilogue)."""

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
                                  bool(c.get("post")), cfg["bm"], cfg["bn"], cfg["bk"], cfg["prec"],
                                  num_warps=cfg["warps"], num_stages=cfg["stages"])


def conv_generators(tl_grid, tr_grid):
    return {
        "tilelang": {"grid": tl_grid, "build": conv_tl_build, "bind": conv_tl_bind, "valid": conv_tl_valid,
                     "static_shapes": True},
        "triton": {"grid": tr_grid, "build": lambda cfg, c: cfg, "bind": conv_tr_bind},
    }


def conv_class(role, cin, t, dil, us, pre=False, post=False):
    """`t` = rows per item as the step program carries them (b8.f17: 17 frames + the stream tail)."""
    batch = {"b1": 1, "b8": 8, "b32": 32}[role]
    return {"B": batch, "T": t, "cin": cin, "cout": cin, "k": 7, "dil": dil, "pre": pre, "post": post,
            "step": {"us": STEP_US[role], "calls": 3}, "us": us}


# Conv1dF32 flags of the production Qwen3-TTS codec packet (plowrt disasm, tts-perf-next recipe):
# c1 k7 convs 0x26ec0c = SnakeBeta output (2) | ragged (3) | post snake | 3xTF32 (13) | split-bf16
# (14) | tap-major (15) | wgmma (17) | pre-split weights (18) | pinned (21), 64-wide tile, unsplit;
# the same word without the output activation for the plain conv; pointwise denses 0x266000.
CONV_FLAGS = 0x26E008
CONV_BETA_FLAGS = 0x26EC0C
DENSE_FLAGS = 0x266000
CURRENT_LIB = {"src": "scripts/gen_kernels/codec_conv.cu",
               "flags": ["-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-DPLOW_NV_HOPPER=1",
                         "-DPLOW_SPEECH_CONV_WIDE_PADDED=0", "-DPLOW_SPEECH_CONV_POSTS=0x81001ull",
                         "-I", "runtime/common", "-I", "runtime/nvidia"]}


def _abi(lib):
    import ctypes
    vp, u = ctypes.c_void_p, ctypes.c_uint
    lib.cc_weights.restype = vp
    lib.cc_weights.argtypes = [vp, u, u, ctypes.c_int]
    lib.cc_conv.restype = vp
    lib.cc_conv.argtypes = [vp] * 7 + [u] * 8
    lib.cc_unary.restype = vp
    lib.cc_unary.argtypes = [vp] * 3 + [u] * 3
    lib.cc_run.argtypes = [vp, ctypes.c_int, vp]


_WEIGHTS = {}


def _weights(lib, w2d):
    """Device buffer of the packet's weight tensor: f32 [n][k] + the pre-split wgmma image (one
    per weight tensor: the cold sets rotate x only)."""
    key = (w2d.data_ptr(), tuple(w2d.shape))
    if key not in _WEIGHTS:
        host = w2d.detach().cpu().contiguous()
        _WEIGHTS[key] = lib.cc_weights(host.data_ptr(), host.shape[0], host.shape[1], 1)
    return _WEIGHTS[key]


def _launcher(lib, handles, keep=None):
    import torch

    def call(keep=keep):
        s = torch.cuda.current_stream().cuda_stream
        for h in handles:
            rc = lib.cc_run(h, 132, s)
            assert rc == 0, f"cc_run rc {rc}"
    return call


def conv_current_bind(lib, c, inp, out):
    """UnaryF32 SnakeBeta (when the class has the input activation; out of place so x survives the
    replays) then the c1 conv as emitted: zero pad dil*(k-1) before, SnakeBeta in the epilogue."""
    import torch
    _abi(lib)
    x, handles, keep = inp["x"], [], None
    if c.get("pre"):
        s = keep = torch.empty_like(x)
        handles.append(lib.cc_unary(s.data_ptr(), x.data_ptr(), inp["apre"].data_ptr(), c["B"] * c["T"], c["cin"], 19))
        x = s
    post = bool(c.get("post"))
    handles.append(lib.cc_conv(out.data_ptr(), x.data_ptr(), _weights(lib, inp["wt"]), inp["bias"].data_ptr(),
                               inp["apost"].data_ptr() if post else None, None, None, c["B"], c["T"], c["cin"],
                               c["cout"], c["k"], c["dil"], c["dil"] * (c["k"] - 1),
                               CONV_BETA_FLAGS if post else CONV_FLAGS))
    return _launcher(lib, handles, keep)


def dense_current_bind(lib, c, inp, out):
    _abi(lib)
    h = lib.cc_conv(out.data_ptr(), inp["x"].data_ptr(), _weights(lib, inp["w"]), inp["bias"].data_ptr(), None, None,
                    None, 1, c["M"], c["K"], c["N"], 1, 1, 0, DENSE_FLAGS)
    return _launcher(lib, [h])
