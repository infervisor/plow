"""gen_search block spec: Maya 2 DepthV2 projections (GemvF32Out mma arm: f16 W [N][K], x rounded
to f16, f32 accumulate) at the depth stage's shapes: qkv (N 3072, K 1024, LayerNorm prologue), f2
(N 1024, K 3072, bias + residual), o (1024 x 1024, residual), head (N 2048, K 1024, LayerNorm).
Row classes: m1 = one stream's cached level, m8 / m32 = 8 / 32 streams, m832 = a full level (26
prefix rows) at 32 streams. Generated bodies: gemm_f16.py (TileLang), gemm_f16_triton.py (Triton).

Current kernel: the standalone probe depth_gemv.cu, which launches the interpreter's own arm
(d_gemv_f32out_mode) with the flags and block count maya2_depth.rs emits for that projection, in
the same lease and on the same cold weight sets as the generators. The arm does more than the
generated GEMM: below 16 rows it standardizes x in the LayerNorm prologue. At M >= 16 it receives
materialized f16 LayerNorm output; that separate op is timed by the depth-block harness. The
batched o input remains unrounded f32 and is converted by the GEMM. The native route writes f1's
GELU output as f16 and uses f2's per-stream grouped bias. Bias/residual inputs default to zeros;
the f1 generators include GELU and the output cast. `IN_MODEL_US` keeps older in-situ timings.
"""

# flags i[7] and blocks as emitted (packet_bench --sweep of the production depth program)
EMIT = {"qkv": (1283, 132), "f1": (1299, 132), "f2": (1039, 128), "o": (1035, 128), "head": (1283, 132)}
SHAPES = {"qkv": (3072, 1024), "f1": (3072, 1024), "f2": (1024, 3072), "o": (1024, 1024), "head": (2048, 1024)}
# B=1 frame: 16.73 ms; qkv + f1 share the shape (4 layers x 20 cached levels x 2), f2 and o 80, head 31.
STEP_B1 = {"qkv": 80, "f1": 80, "f2": 80, "o": 80, "head": 31}
IN_MODEL_US = {"qkv_m1": 15.1, "f2_m1": 28.6, "o_m1": 12.7, "head_m1": 18.2, "qkv_m8": 20.7, "f2_m8": 43.4}


def _classes():
    out = {}
    for s, (n, k) in SHAPES.items():
        for m in (1, 8, 20, 32, 64, 128, 200, 224, 832, 1664, 3328, 5200, 5824):
            if s == "head" and m > 224:
                continue  # the head reads the last prefix row only
            c = {"shape": s, "M": m, "N": n, "K": k, "prefix_rows": 26 if m >= 832 else 1}
            if m == 1:
                c["step"] = {"us": 16730.0, "calls": STEP_B1[s]}
            out[f"{s}_m{m}"] = c
    return out


def _cost(c):
    m, n, k = c["M"], c["N"], c["K"]
    output_bytes = 2 if c["shape"] == "f1" and m >= 16 else 4
    input_bytes = 4 if m < 16 or c["shape"] == "o" else 2
    epilogue_bytes = m * n * 4 if c["shape"] in ("o", "f2") else 0
    if c["shape"] == "f2":
        epilogue_bytes += m // c.get("prefix_rows", 1) * n * 4
    return {"bytes": n * k * 2 + m * k * input_bytes + m * n * output_bytes + epilogue_bytes,
            "flops": 2 * m * n * k, "peak": "bf16", "passes": 1}


def _inputs(c, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    x32 = torch.randn(c["M"], c["K"], device="cuda", generator=g)
    if EMIT[c["shape"]][0] & 256:
        # what the arm's LayerNorm prologue (gamma 1, beta 0) feeds the product
        xn = (x32 - x32.mean(1, keepdim=True)) * torch.rsqrt(x32.var(1, unbiased=False, keepdim=True) + 1e-5)
        if c["M"] >= 16:
            x32 = xn  # materialized LayerNorm output, as emitted for batched projections
    else:
        if c["shape"] != "o":
            x32 = x32.half().float()
        xn = x32
    w = (torch.randn(c["N"], c["K"], device="cuda", generator=g) * 0.05).half()
    return {"x": xn.half(), "w": w, "x32": x32}


def _reference(c, inp):
    value = inp["x"].double() @ inp["w"].double().T
    if "bias" in inp:
        bias = inp["bias"]
        if bias.ndim == 2 and bias.shape[0] > 1:
            bias = bias.repeat_interleave(c.get("prefix_rows", 1), dim=0)[:c["M"]]
        value = value.float() + bias
    if "residual" in inp:
        value = value.float() + inp["residual"]
    if c["shape"] == "f1":
        import torch
        value = torch.nn.functional.gelu(value, approximate="none")
        if c["M"] >= 16:
            value = value.half().double()
    return value


def _cur_bind(lib, c, inp, out, input_f16=None, variant=0, nblk_override=None):
    import ctypes
    import torch
    flags, nblk = EMIT[c["shape"]]
    if nblk_override is not None:
        nblk = nblk_override
    if input_f16 is None:
        input_f16 = c["M"] >= 16 and c["shape"] != "o"
    if c["M"] >= 16:
        flags &= ~256
        if c["shape"] == "f1":
            flags |= 16384
    if input_f16:
        flags |= 8192
    group = c.get("prefix_rows", 1) if c["shape"] == "f2" and "bias" not in inp else 0
    if "bias" in inp:
        flags |= 4
        if inp["bias"].ndim == 2 and inp["bias"].shape[0] > 1:
            group = c.get("prefix_rows", 1)
    if "residual" in inp:
        flags |= 8
    if group:
        flags |= 512
    lib.dg_make_grouped.restype = ctypes.c_void_p
    lib.dg_make_grouped.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_uint] * 5
    lib.dg_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    if variant:
        lib.dg_run_variant.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    zeros = torch.zeros(max(c["M"] * c["N"], c["K"]), device="cuda")
    ones = torch.ones(c["K"], device="cuda")
    bias = inp.get("bias", zeros)
    residual = inp.get("residual", zeros)
    h = lib.dg_make_grouped(out.data_ptr(), inp["x" if input_f16 else "x32"].data_ptr(), inp["w"].data_ptr(), bias.data_ptr(), residual.data_ptr(),
                            ones.data_ptr(), zeros.data_ptr(), flags, c["M"], c["N"], c["K"], group)

    def call(keep=(zeros, ones, inp, bias, residual)):
        stream = torch.cuda.current_stream().cuda_stream
        rc = lib.dg_run_variant(h, nblk, stream, variant) if variant else lib.dg_run(h, nblk, stream)
        assert rc == 0, f"dg_run rc {rc}"
    return call


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
    return _generated_bind(c, out, lambda dest: gemm_f16_triton.launcher(
        cfg, c["M"], c["N"], c["K"], inp["x"], inp["w"], dest))


def _generated_bind(c, out, bind):
    if c["shape"] != "f1":
        return bind(out)
    import torch
    tmp = torch.empty(out.shape, device=out.device, dtype=torch.float32)
    product = bind(tmp)

    def call():
        product()
        if out.dtype == torch.float32:
            torch.ops.aten.gelu.out(tmp, approximate="none", out=out)
        else:
            torch.ops.aten.gelu.out(tmp, approximate="none", out=tmp)
            out.copy_(tmp)
    return call


def _vendor_bind(cfg, c, inp, out):
    import torch
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    product = out if out.dtype == torch.float32 else torch.empty_like(out, dtype=torch.float32)
    operand = torch.empty_like(inp["x"]) if c["shape"] == "o" else inp["x"]
    bias = inp.get("bias")
    if bias is not None and bias.ndim == 2 and bias.shape[0] > 1:
        bias = bias.repeat_interleave(c.get("prefix_rows", 1), dim=0)[:c["M"]]

    def call():
        torch.backends.cuda.preferred_blas_library(cfg["preferred"])
        if c["shape"] == "o":
            operand.copy_(inp["x32"])
        torch.mm(operand, inp["w"].T, out_dtype=torch.float32, out=product)
        if bias is not None:
            product.add_(bias)
        if "residual" in inp:
            product.add_(inp["residual"])
        if c["shape"] == "f1":
            torch.ops.aten.gelu.out(product, approximate="none", out=product)
        if product is not out:
            out.copy_(product)
    return call


SPEC = {
    "name": "maya2_depth_gemv",
    "op": "GemvF32Out depth projection",
    "dtype": "f16 W, served f16/f32 operands, f32 accumulate; f1 GELU output f16 at M>=16",
    "classes": _classes(),
    "cost": _cost,
    "inputs": _inputs,
    "rotate": ["w"],
    "out": lambda c: ((c["M"], c["N"]), "float16" if c["shape"] == "f1" and c["M"] >= 16 else "float32"),
    "reference": _reference,
    # the arm standardizes in f32 with its own summation order: an x element can land one f16 ulp off
    "tol": {"rel_l2": 2e-4},
    "generators": {
        "vendor_blas": {
            "grid": {"preferred": ["cublaslt", "cublas"]},
            "build": lambda cfg, c: cfg,
            "bind": _vendor_bind,
        },
        "tilelang": {
            "grid": {"bm": [16, 32, 64, 128], "bn": [64, 128], "bk": [32, 64], "threads": [128, 256],
                     "stages": [2, 3]},
            "build": _tl_build,
            "bind": lambda kern, c, inp, out: _generated_bind(c, out, lambda dest: lambda: kern(inp["x"], inp["w"], dest)),
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
        "label": "d_gemv_f32out_mode native half operands/GELU and per-stream bias (LayerNorm excluded)",
        "lib": {"src": "scripts/gen_kernels/depth_gemv.cu",
                "flags": ["-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-DPLOW_NV_HOPPER=1", "-I", "runtime/common", "-I", "runtime/nvidia"]},
        "bind": _cur_bind,
    },
}
