"""gen_search block spec: S3Gen CFM attention (AttentionF32, 3xTF32, 306-row key prefix, 8 heads
x 64) at the cached-prompt `csynth` shapes. Generated bodies: attn_cfm.py (TileLang),
attn_cfm_triton.py (Triton); current plow kernel:
the interpreter path of the standalone probe cfm_attn.cu (which 0, 132 blocks); inputs and the
fp64 reference are bench_cfm.py's.
"""

SCALE = 0.125
H, HW, PRE = 8, 64, 306


def _cost(c):
    kv = PRE + c["Tq"]
    return {
        # qkv read + out write per item, the four shared prefix blocks once
        "bytes": c["B"] * c["Tq"] * 4 * H * HW * 4 + 4 * PRE * 2 * H * HW * 4,
        "flops": 4 * c["B"] * H * c["Tq"] * kv * HW,
        # three split products per GEMM; 3xFP16 holds the bound (cfm_attn.cu h16m), so the
        # tensor-core peak is the 16-bit one
        "passes": 3,
        "peak": "bf16",
    }


def _inputs(c, seed):
    import bench_cfm
    qkv, prefix, pidx, klen = bench_cfm.inputs(c["B"], c["Tq"], seed)
    return {"qkv": qkv, "prefix": prefix, "pidx": pidx, "klen": klen}


def _reference(c, inp):
    import bench_cfm
    return bench_cfm.reference(inp["qkv"], inp["prefix"], inp["pidx"], inp["klen"], SCALE)


def _tl_build(cfg, c):
    import attn_cfm
    return attn_cfm.compile(cfg["bm"], cfg["bn"], cfg["threads"], cfg["stages"], H, PRE)


def _tl_bind(kern, c, inp, out):
    return lambda: kern(inp["qkv"], inp["prefix"], inp["pidx"], inp["klen"], out, SCALE)


def _tr_build(cfg, c):
    import attn_cfm_triton
    return attn_cfm_triton.launcher(cfg["bm"], cfg["bn"], cfg["warps"], cfg["stages"], cfg["prec"],
                                    H, HW, PRE)


def _cur_bind(lib, c, inp, out):
    import ctypes
    import torch
    lib.cfm_make.restype = ctypes.c_void_p
    lib.cfm_make.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint] * 3
    lib.cfm_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    h = lib.cfm_make(out.data_ptr(), inp["qkv"].data_ptr(), inp["prefix"].data_ptr(),
                     inp["pidx"].data_ptr(), inp["klen"].data_ptr(), c["B"], c["Tq"], PRE)

    def call():
        rc = lib.cfm_run(h, 0, 132, torch.cuda.current_stream().cuda_stream)
        assert rc == 0, f"cfm_run rc {rc}"
    return call


SPEC = {
    "name": "s3gen_cfm_attn",
    "op": "AttentionF32 prefix306 h8x64",
    "dtype": "f32 (3xTF32)",
    # csynth.b32.t64 is 64 CFG items x 136 rows; b64.t32 is 128 x 72.
    "classes": {
        "b64_q136": {"B": 64, "Tq": 136},
        "b128_q136": {"B": 128, "Tq": 136},
        "b64_q72": {"B": 64, "Tq": 72},
        "b128_q72": {"B": 128, "Tq": 72},
    },
    "cost": _cost,
    "inputs": _inputs,
    "rotate": ["qkv"],
    "out": lambda c: ((c["B"], c["Tq"], H * HW), "float32"),
    "reference": _reference,
    "tol": {"rel_l2": 1e-5},
    "generators": {
        "tilelang": {
            "grid": {"bm": [16, 32, 64], "bn": [32, 64], "threads": [128, 256], "stages": [1, 2]},
            "build": _tl_build,
            "bind": _tl_bind,
        },
        # prec is tl.dot's input_precision: tf32x3 = the plow kernel's hi/lo split, tf32 = one
        # truncated tensor-core product, ieee = FP32 FMA.
        "triton": {
            "grid": {"prec": ["tf32x3", "tf32", "ieee"], "bm": [16, 32, 64, 128], "bn": [32, 64],
                     "warps": [2, 4, 8], "stages": [1, 3]},
            "build": _tr_build,
            "bind": _tl_bind,
        },
    },
    "current": {
        "label": "sp_attention interpreter path (cfm_attn.cu which 0, 132 blocks)",
        "lib": {"src": "scripts/gen_kernels/cfm_attn.cu",
                "flags": ["-arch=sm_90a", "-O3", "-I", "runtime/common", "-I", "runtime/nvidia"]},
        "bind": _cur_bind,
    },
}
