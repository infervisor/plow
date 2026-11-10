"""Generator bodies and block data for the Qwen3-TTS code-predictor depth stage's small-M GEMMs
(spec_qtts_depth_{mm,glu,head}.py): C[M][N] = x[M][K] . W[N][K]^T, bf16 operands, f32 accumulate,
M = B rows in {1, 8, 16, 32} padded to the mma M tile (16 / 32, pad rows zero).

  proj K=2048 N=1024 | qkv K=1024 N=4096 | o K=2048 N=1024 | down K=3072 N=1024      (bf16 out)
  glu  K=1024 N=3072, two weights, silu(gate) * up                                   (bf16 out)
  head K=1024 N=2048                                                                 (f32 out)
"""
OPS = {  # op: (K, N, kind, calls per frame)
    "proj": (2048, 1024, "mm", 16), "qkv": (1024, 4096, "mm", 80), "o": (2048, 1024, "mm", 80),
    "down": (3072, 1024, "mm", 80), "glu": (1024, 3072, "glu", 80), "head": (1024, 2048, "f32", 15),
}
BS = (1, 8, 16, 32)
# fish_gen depth step per frame (us), packet abc-inv (bb7709d66), H100.
STEP_US = {1: 3630, 8: 3870, 16: 4340, 32: 5470}
# fish_gen --depth-sweep 1: mean body marginal per instruction (us).
BODY_US = {
    "proj": {1: 1.36, 8: 4.83, 16: 4.55, 32: 8.14}, "qkv": {1: 1.99, 8: 2.76, 16: 4.35, 32: 4.90},
    "o": {1: 1.44, 8: 2.32, 16: 2.56, 32: 6.37}, "down": {1: 2.67, 8: 2.42, 16: 3.86, 32: 6.51},
    "glu": {1: 5.54, 8: 4.83, 16: 6.19, 32: 8.49}, "head": {1: 7.38, 8: 8.73, 16: 17.86, 32: 38.78},
}
# cap-0 time / 641 instructions: the gate / signal walk, which also carries the weight prefetch.
WALK_US = {1: 2.35, 8: 2.46, 16: 2.43, 32: 2.40}
# The arms the Qwen3-TTS recipe ships (recipes/infervisor/qwen3-tts-1.7b-customvoice, decode objects
# interp_sm90a / interp_sm90a_gw with NVCC_APPEND_FLAGS PLOW_NV_GEMV_K8_MIN=1 PLOW_NV_GEMV_K8_UNB1=2
# PLOW_NV_GEMV_MMA_B1=1 PLOW_NV_GEMV_K8_MAX_MAIN=64 PLOW_NV_INST_PRELOAD=1 PLOW_NV_RAW_NARROW=1,
# PLOW_DEPTH_ATTN_FOLD=1): Gemv / GemvQkv / GemvGlu on the K8 tensor-core route, GemvF32Out on
# d_gemv_f32out (scalar fma, arena-staged x). Measured inside the frame's one cooperative launch,
# weights prefetched during the gate wait: NOT like for like with a standalone cold launch.
CURRENT_LABEL = ("shipped Qwen3-TTS depth arms, in situ (recorded): fish_gen --depth-sweep body "
                 "marginal + mean per-instruction walk (carries the weight prefetch); a generated "
                 "row is a standalone cold launch, so the comparison is indicative only")
CURRENT_SOURCE = "tts-perf/W5/STATUS.md, S/run/dsw10-{1,8,16,32}.jsonl (packet abc-inv)"


def classes(kind):
    return {
        f"{op}_b{b}": {"op": op, "B": b, "M": 16 if b <= 16 else 32, "K": k, "N": n,
                       "step": {"us": STEP_US[b], "calls": calls}}
        for op, (k, n, kd, calls) in OPS.items() if kd == kind for b in BS
    }


def measured(kind):
    return {f"{op}_b{b}": round(us + WALK_US[b], 2)
            for op, (_, _, kd, _) in OPS.items() if kd == kind for b, us in BODY_US[op].items()}


def cost(c):
    nw = 2 if c["op"] == "glu" else 1
    out = 4 if c["op"] == "head" else 2
    return {"bytes": nw * c["N"] * c["K"] * 2 + c["B"] * c["K"] * 2 + c["B"] * c["N"] * out,
            "flops": 2 * nw * c["B"] * c["N"] * c["K"], "peak": "bf16", "passes": 1}


def inputs(c, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + c["K"] + c["N"] + c["B"])
    x = torch.zeros(c["M"], c["K"], device="cuda", dtype=torch.bfloat16)
    x[:c["B"]] = torch.randn(c["B"], c["K"], device="cuda", generator=g).to(torch.bfloat16)
    w = lambda: (torch.randn(c["N"], c["K"], device="cuda", generator=g) * c["K"] ** -0.5).to(torch.bfloat16)
    inp = {"x": x, "w": w()}
    if c["op"] == "glu":
        inp["u"] = w()
    return inp


def reference(c, inp):
    import torch
    x = inp["x"].double()
    r = x @ inp["w"].double().T
    if c["op"] == "glu":
        r = torch.nn.functional.silu(r) * (x @ inp["u"].double().T)
    return r


def out(c):
    return (c["M"], c["N"]), "float32" if c["op"] == "head" else "bfloat16"


def valid(cfg, c):
    return cfg["bm"] == c["M"]


TL_SWEEP = [dict(bm=bm, bn=bn, bk=bk, stages=st, threads=th) for bm in (16, 32) for (bn, bk, st, th) in (
    (16, 64, 2, 128), (16, 128, 2, 128), (32, 64, 2, 128), (32, 128, 2, 128), (32, 64, 3, 128),
    (64, 64, 2, 128), (64, 32, 3, 128), (64, 128, 2, 128), (128, 32, 2, 128), (64, 64, 2, 256))]
TR_GRID = {"bm": [16, 32], "bn": [32, 64, 128], "bk": [32, 64], "warps": [4, 8]}


def tl_build(kind, cfg):
    import tilelang
    import tilelang.language as T
    bm, bn, bk, stages, threads = (cfg[k] for k in ("bm", "bn", "bk", "stages", "threads"))
    M, N, K = T.dynamic("M, N, K")
    dt = "bfloat16"
    odt = "float32" if kind == "f32" else dt

    if kind == "glu":
        @T.prim_func
        def k(X: T.Tensor((M, K), dt), Wg: T.Tensor((N, K), dt), Wu: T.Tensor((N, K), dt),
              C: T.Tensor((M, N), dt)):
            with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
                A = T.alloc_shared((bm, bk), dt)
                Bg = T.alloc_shared((bn, bk), dt)
                Bu = T.alloc_shared((bn, bk), dt)
                G = T.alloc_fragment((bm, bn), "float")
                U = T.alloc_fragment((bm, bn), "float")
                T.clear(G)
                T.clear(U)
                for kk in T.Pipelined(T.ceildiv(K, bk), num_stages=stages):
                    T.copy(X[by * bm, kk * bk], A)
                    T.copy(Wg[bx * bn, kk * bk], Bg)
                    T.copy(Wu[bx * bn, kk * bk], Bu)
                    T.gemm(A, Bg, G, transpose_B=True)
                    T.gemm(A, Bu, U, transpose_B=True)
                for i, j in T.Parallel(bm, bn):
                    G[i, j] = G[i, j] / (1.0 + T.exp(-G[i, j])) * U[i, j]
                T.copy(G, C[by * bm, bx * bn])
    else:
        @T.prim_func
        def k(X: T.Tensor((M, K), dt), W: T.Tensor((N, K), dt), C: T.Tensor((M, N), odt)):
            with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
                A = T.alloc_shared((bm, bk), dt)
                Bs = T.alloc_shared((bn, bk), dt)
                Cl = T.alloc_fragment((bm, bn), "float")
                T.clear(Cl)
                for kk in T.Pipelined(T.ceildiv(K, bk), num_stages=stages):
                    T.copy(X[by * bm, kk * bk], A)
                    T.copy(W[bx * bn, kk * bk], Bs)
                    T.gemm(A, Bs, Cl, transpose_B=True)
                T.copy(Cl, C[by * bm, bx * bn])
    return tilelang.compile(k, target={"kind": "cuda", "arch": "sm_90a"}, pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })


def tl_bind(kern, c, inp, out):
    if c["op"] == "glu":
        return lambda: kern(inp["x"], inp["w"], inp["u"], out)
    return lambda: kern(inp["x"], inp["w"], out)


_TR = []


def _triton_kernel():
    if _TR:
        return _TR[0]
    import triton
    import triton.language as tl

    @triton.jit
    def mm(x_ptr, w_ptr, u_ptr, c_ptr, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           GLU: tl.constexpr, F32: tl.constexpr):
        om = tl.program_id(1) * BM + tl.arange(0, BM)
        on = tl.program_id(0) * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        acu = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(x_ptr + om[:, None] * K + (k + ok)[None, :])
            b = tl.load(w_ptr + on[None, :] * K + (k + ok)[:, None])
            acc = tl.dot(a, b, acc)
            if GLU:
                bu = tl.load(u_ptr + on[None, :] * K + (k + ok)[:, None])
                acu = tl.dot(a, bu, acu)
        if GLU:
            acc = acc * tl.sigmoid(acc) * acu
        cp = c_ptr + om[:, None] * N + on[None, :]
        if F32:
            tl.store(cp, acc)
        else:
            tl.store(cp, acc.to(tl.bfloat16))

    _TR.append(mm)
    return mm


def tr_build(kind, cfg):
    return (_triton_kernel(), kind, dict(cfg))


def tr_bind(kern, c, inp, out):
    mm, kind, cfg = kern
    grid = (c["N"] // cfg["bn"], c["M"] // cfg["bm"])
    u = inp.get("u", inp["w"])
    return lambda: mm[grid](inp["x"], inp["w"], u, out, c["N"], c["K"], BM=cfg["bm"], BN=cfg["bn"],
                            BK=cfg["bk"], GLU=kind == "glu", F32=kind == "f32",
                            num_warps=cfg["warps"], num_stages=3)


def spec(name, kind, op, dtype, tol):
    return {
        "name": name, "op": op, "dtype": dtype,
        "classes": classes(kind), "cost": cost, "inputs": inputs,
        "rotate": ["w", "u"] if kind == "glu" else ["w"],
        "out": out, "reference": reference, "tol": {"rel_l2": tol},
        "generators": {
            "tilelang": {"sweep": TL_SWEEP, "build": lambda cfg, c: tl_build(kind, cfg),
                         "bind": tl_bind, "valid": valid},
            "triton": {"grid": TR_GRID, "build": lambda cfg, c: tr_build(kind, cfg),
                       "bind": tr_bind, "valid": valid},
        },
        "current": {"label": CURRENT_LABEL, "measured_us": measured(kind), "source": CURRENT_SOURCE},
    }
