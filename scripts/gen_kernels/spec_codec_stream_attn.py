"""gen_search: the one-frame codec's legacy gathered-window AttentionF32 stream arm."""
import ctypes
import weakref


def visible_range(position, query, history, window, length):
    return range(max(history - min(position, history), history + query - window + 1),
                 min(length, history + query + 1))


def cost(c):
    b, t, l, h, d = (c[k] for k in ["B", "T", "L", "H", "D"])
    return {"bytes": 4 * b * h * d * (2 * (l + t) + 2 * t),
            "flops": 4 * b * h * d * t * min(c["W"], l + t), "peak": "fp32"}


def inputs(c, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    b, t, l, h, d = (c[k] for k in ["B", "T", "L", "H", "D"])
    qkv = torch.randn(b, l + t, 3 * h * d, device="cuda", generator=g) * .5
    positions = torch.full((b,), 256, dtype=torch.int32, device="cuda")
    lengths = torch.full((b,), l + t, dtype=torch.int32, device="cuda")
    if c.get("edge"):
        positions.copy_(torch.tensor([0, 1, 63, l, 9999], device="cuda").repeat((b + 4) // 5)[:b])
        lengths[::7] = 0
    return {"qkv": qkv, "positions": positions, "lengths": lengths}


def reference(c, inp):
    import torch
    b, t, l, h, d = (c[k] for k in ["B", "T", "L", "H", "D"])
    z = inp["qkv"].double().reshape(b, l + t, 3, h, d)
    q, k, v = z[:, l:, 0].transpose(1, 2), z[:, :, 1].transpose(1, 2), z[:, :, 2].transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) * d ** -.5
    j = torch.arange(l + t, device=z.device)[None, None, :]
    r = l + torch.arange(t, device=z.device)[None, :, None]
    left = l - inp["positions"].clamp(max=l)[:, None, None]
    valid = (j >= left) & (j < inp["lengths"][:, None, None]) & (j <= r) & (j > r - c["W"])
    scores.masked_fill_(~valid[:, None], float("-inf"))
    p = torch.softmax(scores, -1).nan_to_num()
    return (p @ v).transpose(1, 2).reshape(b, t, h * d)


def current(lib, c, inp, out):
    import torch
    lib.ca_make.restype = ctypes.c_void_p
    lib.ca_make.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 6
    lib.ca_run.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
    lib.ca_free.argtypes = [ctypes.c_void_p]
    handle = lib.ca_make(out.data_ptr(), inp["qkv"].data_ptr(), inp["positions"].data_ptr(),
                         inp["lengths"].data_ptr() if c.get("edge") else None,
                         *(c[k] for k in ["B", "T", "L", "H", "D", "W"]))
    assert handle, "ca_make failed"

    def call(keep=(inp, out, lib)):
        blocks = min(132, c["B"] * c["H"] * ((c["T"] + 63) // 64))
        rc = lib.ca_run(handle, blocks, torch.cuda.current_stream().cuda_stream)
        assert rc == 0, f"ca_run rc {rc}"
    weakref.finalize(call, lib.ca_free, handle)
    return call


def generated(cfg, c, inp, out):
    import codec_stream_attn
    return codec_stream_attn.launcher(cfg, c, inp, out)


SPEC = {
    "name": "codec_stream_attn", "op": "AttentionF32 stream history249 window250 h8x64 q2",
    "dtype": "f32 FMA", "classes": {
        **{f"b{b}_q2": dict(B=b, T=2, L=249, H=8, D=64, W=250) for b in [32, 64, 128, 200, 224]},
        "edges_b33_q2": dict(B=33, T=2, L=249, H=8, D=64, W=250, edge=True),
    },
    "cost": cost, "inputs": inputs, "rotate": ["qkv"],
    "out": lambda c: ((c["B"], c["T"], c["H"] * c["D"]), "float32"),
    "reference": reference, "tol": {"rel_l2": 1e-5},
    "generators": {"triton": {
        "grid": {"warps": [4, 8, 16]}, "build": lambda cfg, c: cfg, "bind": generated,
    }},
    "current": {"label": "legacy sp_attention_f32<64,128>, 132 persistent blocks, exact packet flags",
                "lib": {"src": "scripts/gen_kernels/codec_stream_attn.cu",
                        "flags": ["-gencode", "arch=compute_90a,code=sm_90a", "-O3",
                                  "-DPLOW_NV_HOPPER=1", "-I", "runtime/common", "-I", "runtime/nvidia"]},
                "bind": current},
}
