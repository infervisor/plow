"""Generated-kernel catalog: role objects keyed by op signature, built from a checked-in tuning table.

  build_catalog.py build OUT [--entries a,b]   no GPU: regenerate each entry's tuned config, nvcc -> OUT
  build_catalog.py tune [--entries a,b]        GPU (gpulease): sweep configs per shape class, rewrite
                                               the table (the only command that changes it)
  build_catalog.py bench OBJDIR [--px4 CUBIN]  GPU: standalone us / rel-L2 of built objects, packed
                                               multi-request contract check
  build_catalog.py check [--entries a,b]       no TileLang: ENTRIES' signature/object == table

The table (TABLE) records each entry's signature, the chosen config, the generated body's sha256
and per-shape-class measurements. It is the one source of the signatures: devgen
`gen_kernels.rs` binds packet ops from it, and `build` refuses entries that disagree with it.
`build` fails when the generator no longer reproduces the recorded body (generator drift): the
body digest, not a commit, is the key, so any checkout reproducing it builds the entry.
Python runs only here; the runtime loads the cubin as a packet role object (plowrt role path).
`nix develop .#tilelang` provides the pinned generator (TileLang 0.1.12) and nvcc 12.9 for
`build`; tune/bench also need a CUDA torch (that shell's torch is CPU-only).
"""
import argparse, ctypes, hashlib, json, math, os, re, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "tuning/nvidia/sm_90a/h100-sxm5/gen_kernels.json"
GEN_ATTEMPTS = 12
WRAPPERS = {"flash_prefill": REPO / "runtime/nvidia/gen_flash_prefill.cu",
            "flash_decode": REPO / "runtime/nvidia/gen_flash_decode.cu"}
# A persistent role object must hold its SM alone (plowrt: role occupancy == packet grid).
MIN_ARENA = 116 * 1024
BF16_PEAK_TFLOPS = 989.0
HBM_BYTES_PER_S = 3.35e12
H100_SMS = 132


def entry_op(name):
    return ENTRIES[name]["signature"]["op"]


def pick_nsplit(rows, kvh, tiles, nsplit_max, sms=H100_SMS):
    """Splits per (row, kv head) for a flash-decode launch: enough items to cover the SMs, at
    most two tiles per split. plowrt mirrors this (exec/gpu/riders.rs)."""
    return max(1, min(nsplit_max, -(-sms // max(1, rows * kvh)), max(1, tiles // 2)))


def flash_prefill_body(D, BM, BN, stages, threads):
    """Causal flash prefill over one (query tile, head): Q/O [qlen][heads][D], one KV head's K/V
    [kvlen][D]; query row i sits at position kvlen - qlen + i (chunked prefill)."""
    import tilelang.language as T
    qlen, kvlen, heads = T.dynamic("qlen, kvlen, heads")
    dt, acc = "bfloat16", "float"
    log2e = 1.4426950408889634

    @T.prim_func
    def attn_pf(Q: T.Tensor((qlen, heads, D), dt), K: T.Tensor((kvlen, D), dt),
                V: T.Tensor((kvlen, D), dt), O: T.Tensor((qlen, heads, D), dt),
                scale: T.float32):
        with T.Kernel(T.ceildiv(qlen, BM), heads, threads=threads) as (qb, h):
            Q_s = T.alloc_shared((BM, D), dt)
            K_s = T.alloc_shared((BN, D), dt)
            V_s = T.alloc_shared((BN, D), dt)
            S = T.alloc_fragment((BM, BN), acc)
            P = T.alloc_shared((BM, BN), dt)
            Oacc = T.alloc_fragment((BM, D), acc)
            m = T.alloc_fragment((BM,), acc)
            mp = T.alloc_fragment((BM,), acc)
            sc = T.alloc_fragment((BM,), acc)
            ssum = T.alloc_fragment((BM,), acc)
            l = T.alloc_fragment((BM,), acc)
            sl = scale * log2e
            pos0 = kvlen - qlen
            T.copy(Q[qb * BM:(qb + 1) * BM, h, :], Q_s)
            T.fill(Oacc, 0)
            T.fill(l, 0)
            T.fill(m, -T.infinity(acc))
            nt = T.min(T.ceildiv(kvlen, BN), T.ceildiv(pos0 + (qb + 1) * BM, BN))
            for k in T.Pipelined(nt, num_stages=stages):
                T.copy(K[k * BN:(k + 1) * BN, :], K_s)
                for i, j in T.Parallel(BM, BN):
                    S[i, j] = T.if_then_else(pos0 + qb * BM + i >= k * BN + j, 0,
                                             -T.infinity(acc))
                T.gemm(Q_s, K_s, S, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(m, mp)
                T.reduce_max(S, m, dim=1, clear=False)
                for i in T.Parallel(BM):
                    sc[i] = T.exp2(mp[i] * sl - m[i] * sl)
                for i, j in T.Parallel(BM, BN):
                    S[i, j] = T.exp2(S[i, j] * sl - m[i] * sl)
                T.reduce_sum(S, ssum, dim=1)
                for i in T.Parallel(BM):
                    l[i] = l[i] * sc[i] + ssum[i]
                T.copy(S, P)
                for i, j in T.Parallel(BM, D):
                    Oacc[i, j] *= sc[i]
                T.copy(V[k * BN:(k + 1) * BN, :], V_s)
                T.gemm(P, V_s, Oacc, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(BM, D):
                Oacc[i, j] /= l[i]
            T.copy(Oacc, O[qb * BM:(qb + 1) * BM, h, :])

    return attn_pf


def flash_prefill_body_fp8kv(D, BM, BN, stages, threads):
    """flash_prefill_body over an e4m3 KV cache with one f32 scale per row (FlashPrefillFp8): K/V
    tiles are staged as e4m3 and converted exactly to fp16 (one cvt per pair on sm_90; bf16 has no
    e4m3 cvt and ran at 0.7x); k_scale multiplies the f32 score columns before masking, v_scale the
    V tile rows during the convert, so P stays in [0, 1] (P * v_scale in fp16 would flush at long
    context). That multiply runs in f32 and rounds once to fp16: v_scale itself cast to fp16 is
    subnormal below amax 0.027 (real V rows are that small; rel-L2 4.8e-3 at amax 0.002 against
    2.8e-4 for the f32 product, 7% slower). The exp2 carries +15, so the fp16 P tile holds P * 2^15
    (<= 32768, subnormal only below P = 2e-9); the row sum l carries the same factor and the final
    O = Oacc / l cancels it. fp16 bounds: |Q| and |V| < 65504."""
    import tilelang.language as T
    qlen, kvlen, heads = T.dynamic("qlen, kvlen, heads")
    dt, mt, acc, f8 = "bfloat16", "float16", "float", "float8_e4m3"
    log2e = 1.4426950408889634
    p_exp2 = 15.0

    @T.prim_func
    def attn_pf(Q: T.Tensor((qlen, heads, D), dt), K: T.Tensor((kvlen, D), f8),
                V: T.Tensor((kvlen, D), f8), KS: T.Tensor((kvlen,), acc),
                VS: T.Tensor((kvlen,), acc), O: T.Tensor((qlen, heads, D), dt),
                scale: T.float32):
        with T.Kernel(T.ceildiv(qlen, BM), heads, threads=threads) as (qb, h):
            Q_s = T.alloc_shared((BM, D), mt)
            K8 = T.alloc_shared((BN, D), f8)
            V8 = T.alloc_shared((BN, D), f8)
            K_s = T.alloc_shared((BN, D), mt)
            V_s = T.alloc_shared((BN, D), mt)
            ks = T.alloc_shared((BN,), acc)
            vs = T.alloc_shared((BN,), acc)
            S = T.alloc_fragment((BM, BN), acc)
            P = T.alloc_shared((BM, BN), mt)
            Oacc = T.alloc_fragment((BM, D), acc)
            m = T.alloc_fragment((BM,), acc)
            mp = T.alloc_fragment((BM,), acc)
            sc = T.alloc_fragment((BM,), acc)
            ssum = T.alloc_fragment((BM,), acc)
            l = T.alloc_fragment((BM,), acc)
            sl = scale * log2e
            pos0 = kvlen - qlen
            T.copy(Q[qb * BM:(qb + 1) * BM, h, :], Q_s)
            T.fill(Oacc, 0)
            T.fill(l, 0)
            T.fill(m, -T.infinity(acc))
            nt = T.min(T.ceildiv(kvlen, BN), T.ceildiv(pos0 + (qb + 1) * BM, BN))
            for k in T.Pipelined(nt, num_stages=stages):
                T.copy(K[k * BN:(k + 1) * BN, :], K8)
                T.copy(KS[k * BN:(k + 1) * BN], ks)
                T.copy(VS[k * BN:(k + 1) * BN], vs)
                for i, j in T.Parallel(BN, D):
                    K_s[i, j] = K8[i, j]
                T.clear(S)
                T.gemm(Q_s, K_s, S, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BM, BN):
                    S[i, j] = T.if_then_else(pos0 + qb * BM + i >= k * BN + j, S[i, j] * ks[j],
                                             -T.infinity(acc))
                T.copy(m, mp)
                T.reduce_max(S, m, dim=1, clear=False)
                for i in T.Parallel(BM):
                    sc[i] = T.exp2(mp[i] * sl - m[i] * sl)
                for i, j in T.Parallel(BM, BN):
                    S[i, j] = T.exp2(S[i, j] * sl - m[i] * sl + p_exp2)
                T.reduce_sum(S, ssum, dim=1)
                for i in T.Parallel(BM):
                    l[i] = l[i] * sc[i] + ssum[i]
                T.copy(S, P)
                for i, j in T.Parallel(BM, D):
                    Oacc[i, j] *= sc[i]
                T.copy(V[k * BN:(k + 1) * BN, :], V8)
                for i, j in T.Parallel(BN, D):
                    V_s[i, j] = V8[i, j] * vs[i]
                T.gemm(P, V_s, Oacc, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(BM, D):
                Oacc[i, j] /= l[i]
            T.copy(Oacc, O[qb * BM:(qb + 1) * BM, h, :])

    return attn_pf


# Catalog entries. `signature` is what devgen matches packet ops against (`tune` writes it to the
# table, which crates/devgen/src/gen_kernels.rs reads); `classes` are the tuning shape classes
# (heads, kv heads, rows).
ENTRIES = {
    "attn_pf_hd512": {
        "signature": {"op": "flash_prefill", "head_dim": 512, "mask": "causal", "window": 0,
                      "gqa": [1, 16], "dtype": "bf16", "kv_dtype": "bf16", "arch": "sm_90a"},
        "object": "gen_sm90a_attn_pf_hd512.cubin",
        "body": lambda c: flash_prefill_body(512, c["bm"], c["bn"], c["stages"], c["threads"]),
        "sweep": [{"bm": bm, "bn": bn, "stages": st, "threads": thr}
                  for bm, bn, st, thr in [(64, 64, 1, 256), (64, 32, 1, 256), (64, 32, 2, 256),
                                          (64, 16, 2, 256), (64, 16, 3, 256)]],
        "classes": [(16, 1, 1024), (16, 1, 4096), (16, 1, 8192)],
    },
    # FlashPrefillFp8 global attention (crates/devgen/src/gen_kernels.rs KvDtype::Fp8). The
    # extra staging tiles leave room for BN <= 48.
    "attn_pf_hd512_fp8kv": {
        "signature": {"op": "flash_prefill", "head_dim": 512, "mask": "causal", "window": 0,
                      "gqa": [1, 16], "dtype": "bf16", "kv_dtype": "fp8_e4m3_rowscale",
                      "arch": "sm_90a"},
        "object": "gen_sm90a_attn_pf_hd512_fp8kv.cubin",
        "fp8": True,
        # fp16 V tile times the per-row V scale: the hd256 FP8-KV entry's 2.3e-3 class.
        "tol": 4e-3,
        "body": lambda c: flash_prefill_body_fp8kv(512, c["bm"], c["bn"], c["stages"],
                                                   c["threads"]),
        "sweep": [{"bm": bm, "bn": bn, "stages": st, "threads": thr}
                  for bm, bn, st, thr in [(64, 32, 1, 256), (64, 32, 2, 256), (64, 48, 1, 256),
                                          (64, 16, 2, 256), (64, 16, 3, 256)]],
        "classes": [(16, 1, 1024), (16, 1, 4096), (16, 1, 1024, 7168), (16, 1, 1024, 14336)],
    },
}

# Further entries live in their own modules: scripts/gen_kernels/catalog_<name>.py exporting
# ENTRIES in the same shape (TileLang `body`, signature, object, sweep, classes), or with custom
# `build(config, out_cubin) -> source digest` and `tune() -> row` hooks for hand-parameterized
# kernels that bring their own wrapper.
for _path in sorted(Path(__file__).resolve().parent.glob("catalog_*.py")):
    import importlib.util
    _spec = importlib.util.spec_from_file_location(_path.stem, _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    for _name, _entry in _mod.ENTRIES.items():
        assert _name not in ENTRIES, f"duplicate catalog entry {_name}"
        ENTRIES[_name] = _entry

PASS_CONFIGS_KEYS = ("TL_DISABLE_TMA_LOWER", "TL_DISABLE_WARP_SPECIALIZED")


def tilelang_compile(func, extra=()):
    import tilelang
    cfg = {getattr(tilelang.PassConfigKey, k): True for k in (*PASS_CONFIGS_KEYS, *extra)}
    return tilelang.compile(func, target={"kind": "cuda", "arch": "sm_90a"}, pass_configs=cfg)


def generate(name, cfg):
    """TileLang source for one config -> the wrapper's body header and the arena it claims."""
    import tilelang
    os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
    kern = tilelang_compile(ENTRIES[name]["body"](cfg), ENTRIES[name].get("pass_configs", ()))
    src = kern.get_kernel_source()
    funcs = list(kern.adapter.device_mod.functions.items())
    assert len(funcs) == 1, "one device kernel per entry"
    arena = int(funcs[0][1].attrs["dyn_shared_memory_buf"])
    for banned in ("blockIdx.z", "gridDim", "CUtensorMap", "__grid_constant__"):
        assert banned not in src, f"generated body uses {banned}: not a persistent-loop body"
    head = re.compile(r'extern "C" __global__ void (?:__launch_bounds__\((\d+), 1\) )?(\w+)\(([^)]*)\)')
    decls = list(head.finditer(src))
    defn = [d for d in decls if src[d.end():d.end() + 3].strip().startswith("{")]
    assert len(defn) == 1 and int(defn[0].group(1)) == cfg["threads"], "unexpected kernel header"
    params = [p.strip().split()[-1] for p in defn[0].group(3).split(",")]
    body = src[:defn[0].start()] + (
        "static __device__ __forceinline__ void plow_gen_body(int plow_bx, int plow_by, "
        + defn[0].group(3) + ")") + src[defn[0].end():]
    body = "\n".join(l for l in body.splitlines() if not head.match(l.strip()) or "{" in l)
    body = body.replace("blockIdx.x", "plow_bx").replace("blockIdx.y", "plow_by")
    sig = ENTRIES[name]["signature"]
    if sig["op"] == "flash_decode":
        # gen_flash_decode.cu: a plain (kv heads x nsplit, rows) grid, no persistent loop.
        names = ["K", "ML", "O", "Opart", "Q", "V", "ctr", "kv_len", "slot", "batch", "kv_mask",
                 "kv_stride", "kvh", "nsplit", "scale", "window"]
        args = {n: n + "_" for n in names}
        assert set(params) <= set(args), f"unexpected body params {params}"
        hdr = [f"// generated: {name} {json.dumps(cfg, sort_keys=True)} tilelang {tilelang.__version__}",
               f"#define PLOW_GEN_HEAD_DIM {sig['head_dim']}",
               f"#define PLOW_GEN_GQA {sig['gqa'][0]}", f"#define PLOW_GEN_BN {cfg['bn']}",
               f"#define PLOW_GEN_THREADS {cfg['threads']}", f"#define PLOW_GEN_ARENA {arena}",
               f"#define PLOW_GEN_NSPLIT_MAX {sig['nsplit_max']}"]
        call = (f"#define plow_gen_call_decode(bx, by, {', '.join(args[n] for n in names)}) "
                f"plow_gen_body(bx, by, {', '.join(args[p] for p in params)})")
        return "\n".join(hdr) + "\n" + body + "\n" + call + "\n", arena, tilelang.__version__
    args = {"Q": "Q_", "K": "K_", "V": "V_", "O": "O_", "heads": "heads_", "qlen": "qlen_",
            "kvlen": "kvlen_", "scale": "scale_", "window": "window_", "kv_mask": "kv_mask_"}
    fp8 = ENTRIES[name].get("fp8", False)
    if fp8:
        args.update(KS="KS_", VS="VS_")
    assert set(params) <= set(args), f"unexpected body params {params}"
    hdr = [f"// generated: {name} {json.dumps(cfg, sort_keys=True)} tilelang {tilelang.__version__}",
           f"#define PLOW_GEN_HEAD_DIM {sig['head_dim']}",
           f"#define PLOW_GEN_BM {cfg['bm']}", f"#define PLOW_GEN_BN {cfg['bn']}",
           f"#define PLOW_GEN_THREADS {cfg['threads']}",
           f"#define PLOW_GEN_ARENA {max(arena, MIN_ARENA)}"]
    if fp8:
        # The wrapper's FP8-KV mode (gen_flash_prefill.cu): e4m3 K/V plus per-row scales.
        hdr.append("#define PLOW_GEN_FP8_KV 1")
        call = ("#define plow_gen_call_fp8(bx, by, Q_, K_, V_, KS_, VS_, O_, heads_, qlen_, "
                "kvlen_, scale_, window_, kv_mask_) "
                f"plow_gen_body(bx, by, {', '.join(args[p] for p in params)})")
    else:
        call = ("#define plow_gen_call(bx, by, Q_, K_, V_, O_, heads_, qlen_, kvlen_, scale_, "
                "window_, kv_mask_) "
                f"plow_gen_body(bx, by, {', '.join(args[p] for p in params)})")
    text = "\n".join(hdr) + "\n" + body + "\n" + call + "\n"
    return text, max(arena, MIN_ARENA), tilelang.__version__


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def body_digest(header):
    """sha256 of the body modulo reduction-scratch naming: TileLang assigns its `workspace_N`
    buffers in an unordered pass, so two runs differ only in which scratch slot a reduction uses."""
    return sha256(re.sub(r"workspace(_\d+)?( = .*)?", "workspace", header))


def nvcc_compile(name, header, out):
    from tilelang import env
    nvcc = os.environ.get("PLOW_NVCC", "nvcc")
    wrapper = WRAPPERS[ENTRIES[name]["signature"]["op"]]
    with tempfile.TemporaryDirectory() as tmp:
        hpath = Path(tmp) / f"{name}_body.cuh"
        hpath.write_text(header)
        cmd = [nvcc, "-std=c++20", "-arch=sm_90a", "-O3", "-cubin", "-w", "-Xcudafe",
               "--diag_suppress=177", "-Xptxas=-v", f"-I{env.TILELANG_TEMPLATE_PATH}",
               f"-I{env.CUTLASS_INCLUDE_DIR}", f"-I{REPO / 'runtime/common'}",
               f"-I{REPO / 'runtime/nvidia'}", f'-DPLOW_GEN_BODY="{hpath}"',
               *ENTRIES[name].get("defines", ()), "-o", str(out), str(wrapper)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode:
            sys.exit(f"nvcc failed for {name}:\n{res.stderr[-4000:]}")
    spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", res.stderr)
    return res.stderr, spills


def load_table():
    return json.loads(TABLE.read_text()) if TABLE.exists() else {"version": 1, "entries": {}}


def table_mismatches(table, names):
    """Entries whose signature or object differs from the table row (devgen gen_kernels.rs binds
    from the table, so the two must agree)."""
    return [f"{n}: {k}" for n in names if n in table["entries"]
            for k in ("signature", "object") if ENTRIES[n][k] != table["entries"][n][k]]


def cmd_check(a):
    bad = table_mismatches(load_table(), a.entries)
    if bad:
        sys.exit("catalog entries disagree with the table (retune): " + ", ".join(bad))
    print(f"{len(a.entries)} entries agree with {TABLE.relative_to(REPO)}")


def cmd_build(a):
    table = load_table()
    bad = table_mismatches(table, a.entries)
    if bad:
        sys.exit("catalog entries disagree with the table (retune): " + ", ".join(bad))
    # The body digest is the build key; a different generator release cannot be expected to
    # reproduce it (`nix develop .#tilelang` pins the tuned one). Hand-parameterized entries
    # (their own `build` hook) need only nvcc.
    if any("build" not in ENTRIES[n] for n in a.entries):
        import tilelang
        pinned = table.get("generator", {}).get("version")
        if tilelang.__version__ != pinned:
            sys.exit(f"tilelang {tilelang.__version__} != table generator {pinned}; "
                     "run under `nix develop .#tilelang`")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.entries:
        row = table["entries"].get(name)
        if row is None:
            sys.exit(f"{name}: no tuned config in {TABLE}; run `tune` first")
        if "build" in ENTRIES[name]:
            digest = ENTRIES[name]["build"](row["config"], out / row["object"])
            if digest != row["body_sha256"]:
                sys.exit(f"{name}: source drift ({digest} != table {row['body_sha256']}); retune")
            print(f"{name}: {out / row['object']} sha256="
                  f"{sha256((out / row['object']).read_bytes())}")
            continue
        # TileLang's codegen varies run to run under one hash seed; the pinned body is the modal
        # variant, so regenerate before calling it drift.
        for _ in range(GEN_ATTEMPTS):
            header, arena, _ = generate(name, row["config"])
            if body_digest(header) == row["body_sha256"]:
                break
        else:
            sys.exit(f"{name}: generator drift (body sha256 {body_digest(header)} != table "
                     f"{row['body_sha256']}); retune with `tune` or pin the generator version "
                     f"{table['generator']}")
        nvcc_compile(name, header, out / row["object"])
        print(f"{name}: {out / row['object']} arena={arena} sha256="
              f"{sha256((out / row['object']).read_bytes())}")


# ---- GPU side (tune / bench) -------------------------------------------------------------------

class Driver:
    def __init__(self):
        import torch
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        self.cu = ctypes.CDLL(os.environ.get("PLOW_LIBCUDA", "libcuda.so.1"))
        self.torch = torch

    def check(self, rc, what):
        if rc:
            name = ctypes.c_char_p()
            self.cu.cuGetErrorName(rc, ctypes.byref(name))
            raise RuntimeError(f"{what}: {name.value}")

    def function(self, image, entry, smem):
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.cu.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")
        self.check(self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, entry.encode()), entry)
        self.check(self.cu.cuFuncSetAttribute(fn, 8, ctypes.c_int(smem)), "max dynamic smem")
        n = ctypes.c_int()
        self.check(self.cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(n), fn, ctypes.c_int(256 if "px4" not in entry else 512),
            ctypes.c_size_t(smem)), "occupancy")
        return mod, fn

    def launch(self, fn, grid, block, smem, args):
        buf = ctypes.create_string_buffer(bytes(args), len(args))
        params = (ctypes.c_void_p * 1)(ctypes.addressof(buf))
        stream = ctypes.c_void_p(self.torch.cuda.current_stream().cuda_stream)
        gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
        self.check(self.cu.cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem, stream, params,
                                          None), "cuLaunchKernel")
        self._keep = (buf, params)

    def tmap_kv3(self, base, rows, hd, heads, box_rows):
        enc = self.cu.cuTensorMapEncodeTiled
        m = (ctypes.c_uint8 * 128)()
        gd = (ctypes.c_uint64 * 3)(hd, rows, heads)
        gs = (ctypes.c_uint64 * 2)(hd * 2, rows * hd * 2)
        bd = (ctypes.c_uint32 * 3)(64, box_rows, 1)
        es = (ctypes.c_uint32 * 3)(1, 1, 1)
        self.check(enc(m, 9, 3, ctypes.c_void_p(base), gd, gs, bd, es, 0, 3, 2, 0), "tmap")
        return bytes(m)


def pack_args(fields):
    import struct
    fmt = "<" + "".join("Q" if k == "p" else ("f" if k == "f" else "I") for k, _ in fields)
    return struct.pack(fmt, *[v for _, v in fields])


def graph_time(torch, fns, reps=5):
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for f in fns:
                f()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / len(fns))
    return sorted(ts)[len(ts) // 2]


class Case:
    """Packet-shaped buffers: Q/O [seq_q][H][D], K/V [slots][KVH][kv_stride][D], a request table."""

    def __init__(self, drv, H, KVH, D, seq_q, kv_stride, slots, requests, seed=0, window=0,
                 fp8=False):
        torch = drv.torch
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.H, self.KVH, self.D, self.seq_q, self.kv_stride = H, KVH, D, seq_q, kv_stride
        self.window = window
        self.requests = requests
        bf = torch.bfloat16
        # Pre-scaled so the kernels run Gemma-4's scale 1.0 on standard attention logits.
        self.q = (torch.randn(seq_q, H, D, device="cuda", generator=g) * D ** -0.5).to(bf)
        self.k = torch.randn(slots, KVH, kv_stride, D, device="cuda", dtype=bf, generator=g)
        self.v = torch.randn(slots, KVH, kv_stride, D, device="cuda", dtype=bf, generator=g)
        self.fp8 = fp8
        if fp8:
            # The packet's cache write (one scale per row, amax / 448); the reference reads the
            # dequantized cache.
            def quant(x):
                xf = x.float()
                sc = (xf.abs().amax(-1) / 448.0).clamp_min(1e-12)
                q = (xf / sc[..., None]).to(torch.float8_e4m3fn)
                return q.view(torch.uint8).contiguous(), sc.contiguous(), q.float() * sc[..., None]
            self.k8, self.ks, self.k = quant(self.k)
            self.v8, self.vs, self.v = quant(self.v)
        self.o = torch.full((seq_q, H, D), float("nan"), device="cuda", dtype=bf)
        flat = [len(requests)] + [x for r in requests for x in r]
        self.req = torch.tensor(flat, device="cuda", dtype=torch.int32)
        self.entries = torch.zeros(2048 * 24, device="cuda", dtype=torch.uint8)
        self.counters = torch.zeros(1024, device="cuda", dtype=torch.int32)
        self.scratch = torch.zeros(1 << 20, device="cuda", dtype=torch.float32)

    def gen_args(self):
        p = lambda t: ("p", t.data_ptr())
        if self.fp8:
            # ABI 2: k / v scales ride the opart / mlpart slots.
            return pack_args([p(self.req), p(self.ks), p(self.vs), p(self.q), p(self.k8),
                              p(self.v8), p(self.o), ("p", 0), p(self.entries), p(self.counters),
                              p(self.counters), ("u", self.seq_q), ("u", 0), ("u", 0),
                              ("u", self.kv_stride), ("u", 0xFFFFFFFF), ("f", 1.0),
                              ("u", self.H), ("u", self.KVH), ("u", self.window), ("u", 0)])
        return pack_args([p(self.req), p(self.scratch), p(self.scratch), p(self.q), p(self.k),
                          p(self.v), p(self.o), ("p", 0), p(self.entries), p(self.counters),
                          p(self.counters), ("u", self.seq_q), ("u", 0), ("u", 0),
                          ("u", self.kv_stride), ("u", 0xFFFFFFFF), ("f", 1.0),
                          ("u", self.H), ("u", self.KVH), ("u", self.window), ("u", 0)])

    def reference(self, rows_per_req=3):
        """fp32 reference on sampled rows of every request (scale 1.0)."""
        import torch
        out, got = [], []
        for q0, qlen, slot, kvlen in self.requests:
            for i in sorted({0, qlen // 3, qlen - 1})[:rows_per_req]:
                pos = kvlen - qlen + i
                lo = max(0, pos + 1 - self.window) if self.window else 0
                for h in range(self.H):
                    kv = h // (self.H // self.KVH)
                    k = self.k[slot, kv, lo:pos + 1].float()
                    v = self.v[slot, kv, lo:pos + 1].float()
                    s = k @ self.q[q0 + i, h].float()
                    out.append(s.softmax(0) @ v)
                    got.append(self.o[q0 + i, h].float())
        r, o = torch.stack(out), torch.stack(got)
        return ((o - r).norm() / r.norm()).item()


def run_gen(drv, fn, smem, block, case):
    drv.launch(fn, 132, block, smem, case.gen_args())


class DecodeCase:
    """Packet-shaped decode buffers: Q/O [B][H][D], K/V [slots][KVH][kv_stride][D], kv_len and a
    slot table (a permutation, so the table is exercised), split-KV scratch at NSPLIT_MAX."""

    def __init__(self, drv, H, KVH, D, B, ctx, ns_max, seed=0, window=0, ring=False):
        torch = drv.torch
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.H, self.KVH, self.D, self.B, self.window = H, KVH, D, B, window
        # The body stages whole KV tiles (rows past kv_len are masked after the load): the slot
        # stride is a multiple of the widest tile, as the packet's page-granular KV is (plowrt
        # rejects a stride the object's tile does not divide).
        self.kv_stride = 1 << (ctx - 1).bit_length() if ring else -(-ctx // 64) * 64
        self.kv_mask = self.kv_stride - 1 if ring else 0xFFFFFFFF
        bf = torch.bfloat16
        self.q = (torch.randn(B, H, D, device="cuda", generator=g) * D ** -0.5).to(bf)
        self.k = torch.randn(B, KVH, self.kv_stride, D, device="cuda", dtype=bf, generator=g)
        self.v = torch.randn(B, KVH, self.kv_stride, D, device="cuda", dtype=bf, generator=g)
        self.lens = torch.tensor([ctx - (b % 7) for b in range(B)], device="cuda",
                                 dtype=torch.int32)
        self.slot = torch.tensor([(b * 5 + 3) % B for b in range(B)], device="cuda",
                                 dtype=torch.int32)
        self.o = torch.full((B, H, D), float("nan"), device="cuda", dtype=bf)
        self.opart = torch.zeros(B, H, ns_max, D, device="cuda", dtype=torch.float32)
        self.ml = torch.zeros(B, H, ns_max, 2, device="cuda", dtype=torch.float32)
        self.ctr = torch.zeros(B, KVH, device="cuda", dtype=torch.int32)

    def args(self, nsplit):
        p = lambda t: ("p", t.data_ptr())
        return pack_args([p(self.opart), p(self.ml), p(self.q), p(self.k), p(self.v),
                          p(self.lens), p(self.slot), p(self.o), p(self.ctr), ("u", self.B),
                          ("u", self.H), ("u", self.KVH), ("u", self.kv_stride),
                          ("u", self.window), ("u", nsplit), ("u", self.kv_mask), ("u", self.D),
                          ("f", 1.0), ("u", 0)])

    def reference(self):
        import torch
        out, got = [], []
        for b in range(0, self.B, max(1, self.B // 6)):
            n = int(self.lens[b])
            lo = max(0, n - self.window) if self.window else 0
            rows = torch.arange(lo, n, device="cuda") & self.kv_mask
            s = int(self.slot[b])
            for h in range(self.H):
                kv = h // (self.H // self.KVH)
                k = self.k[s, kv, rows].float()
                v = self.v[s, kv, rows].float()
                sc = k @ self.q[b, h].float()
                out.append(sc.softmax(0) @ v)
                got.append(self.o[b, h].float())
        r, o = torch.stack(out), torch.stack(got)
        return ((o - r).norm() / r.norm()).item()


def bench_decode_entry(drv, name, image, arena, block, classes, check_pack=True):
    """Flash-decode entries: us per launch at each (heads, kv heads, rows, context) class with
    the runtime's split policy; `packed_check` = a ring KV with a window at a permuted slot table."""
    torch = drv.torch
    _, fn = drv.function(image, "plow_gen_flash_decode", arena)
    sig = ENTRIES[name]["signature"]
    D, ns_max, bn = sig["head_dim"], sig["nsplit_max"], None
    row = load_table()["entries"].get(name)
    res = {}
    for H, KVH, B, ctx in classes:
        per = B * KVH * ctx * D * 2 * 2
        n = max(2, min(8, math.ceil(160e6 / per)))
        cases = [DecodeCase(drv, H, KVH, D, B, ctx, ns_max, seed=s) for s in range(n)]
        nsplit = pick_nsplit(B, KVH, -(-ctx // 64), ns_max)
        launch = lambda c, ns=nsplit: drv.launch(fn, (KVH * ns, B), block, arena, c.args(ns))
        launch(cases[0])
        torch.cuda.synchronize()
        err = cases[0].reference()
        us = graph_time(torch, [(lambda c=c: launch(c)) for c in cases])
        res[f"h{H}kv{KVH}_b{B}_ctx{ctx}"] = {"us": round(us, 1), "rel_l2": float(f"{err:.2e}"),
                                             "nsplit": nsplit,
                                             "floor_us": round(per / HBM_BYTES_PER_S * 1e6, 1)}
        del cases
        torch.cuda.empty_cache()
    if check_pack:
        H, KVH = classes[0][:2]
        c = DecodeCase(drv, H, KVH, D, 5, 3000, ns_max, seed=7, window=1024, ring=True)
        drv.launch(fn, (KVH * ns_max, 5), block, arena, c.args(ns_max))
        torch.cuda.synchronize()
        res["packed_check"] = {"rel_l2": float(f"{c.reference():.2e}"),
                               "tail_zero": bool((c.ctr == 0).all().item())}
    return res


def class_key(name, cls):
    if entry_op(name) == "flash_decode":
        H, KVH, B, ctx = cls
        return f"h{H}kv{KVH}_b{B}_ctx{ctx}"
    H, KVH, rows, *past = cls
    return f"h{H}kv{KVH}_rows{rows}" + (f"_past{past[0]}" if past else "")


def bench_any(drv, name, image, arena, block, classes):
    bench = bench_decode_entry if entry_op(name) == "flash_decode" else bench_entry
    return bench(drv, name, image, arena, block, classes)


def bench_entry(drv, name, image, arena, block, classes, check_pack=True):
    torch = drv.torch
    _, fn = drv.function(image, "plow_gen_flash_prefill_direct", arena)
    res = {}
    fp8 = ENTRIES[name].get("fp8", False)
    # A class is (heads, kv heads, rows[, past]): one chunk of `rows` after `past` cached rows.
    for H, KVH, rows, *rest in classes:
        past = rest[0] if rest else 0
        D = ENTRIES[name]["signature"]["head_dim"]
        per = (rows * 2 * H + (rows + past) * 2 * KVH) * D * 2
        n = max(2, min(8, math.ceil(160e6 / per)))
        win = ENTRIES[name]["signature"]["window"]
        cases = [Case(drv, H, KVH, D, rows, rows + past, 1, [(0, rows, 0, rows + past)], seed=s,
                      window=win, fp8=fp8)
                 for s in range(n)]
        run_gen(drv, fn, arena, block, cases[0])
        torch.cuda.synchronize()
        err = cases[0].reference()
        us = graph_time(torch, [(lambda c=c: run_gen(drv, fn, arena, block, c)) for c in cases])
        flop = 4 * H * D * sum(min(past + r + 1, win) if win else past + r + 1
                               for r in range(rows))
        res[f"h{H}kv{KVH}_rows{rows}" + (f"_past{past}" if past else "")] = {"us": round(us, 1), "rel_l2": float(f"{err:.2e}"),
                                         "floor_us": round(flop / BF16_PEAK_TFLOPS / 1e6, 1),
                                         "tflops": round(flop / us / 1e6)}
        del cases
        torch.cuda.empty_cache()
    if check_pack:
        # Two chunked requests in one launch, padded rung: slot order, kv offset, zeroed tail.
        H, KVH, D = classes[0][0], classes[0][1], ENTRIES[name]["signature"]["head_dim"]
        reqs = [(0, 1000, 2, 3000), (1000, 1500, 0, 1500), (2500, 77, 1, 2125)]
        c = Case(drv, H, KVH, D, 2688, 4096, 3, reqs, seed=7,
                 window=ENTRIES[name]["signature"]["window"], fp8=fp8)
        run_gen(drv, fn, arena, block, c)
        torch.cuda.synchronize()
        tail = c.o[2577:].float()
        res["packed_check"] = {"rel_l2": float(f"{c.reference():.2e}"),
                               "tail_zero": bool((tail == 0).all().item())}
    return res


def bench_px4(drv, cubin, rows):
    """The role-15 direct entry on the same packet-shaped buffers (descriptor TMA, box 16)."""
    torch = drv.torch
    image = Path(cubin).read_bytes()
    smem = 110592
    _, fn = drv.function(image, "plow_sm90a_pfattn_hd512_px4_bq64_direct", smem)
    per = rows * 34 * 512 * 2
    n = max(2, min(8, math.ceil(160e6 / per)))
    cases = []
    for s in range(n):
        c = Case(drv, 16, 1, 512, rows, rows, 1, [(0, rows, 0, rows)], seed=s)
        desc = torch.zeros(256 + 128, device="cuda", dtype=torch.uint8)
        base = (desc.data_ptr() + 127) // 128 * 128
        off = base - desc.data_ptr()
        km = drv.tmap_kv3(c.k.data_ptr(), rows, 512, 1, 16)
        vm = drv.tmap_kv3(c.v.data_ptr(), rows, 512, 1, 16)
        desc[off:off + 256] = torch.tensor(list(km + vm), dtype=torch.uint8, device="cuda")
        c.table = torch.tensor([base], dtype=torch.int64, device="cuda")
        c.desc = desc
        p = lambda t: ("p", t.data_ptr())
        c.px4 = pack_args([p(c.req), p(c.scratch), p(c.scratch), p(c.q), p(c.k), p(c.v),
                           p(c.o), p(c.table), p(c.entries), p(c.counters), p(c.counters),
                           ("u", rows), ("u", rows)])
        cases.append(c)
    drv.launch(fn, 132, 512, smem, cases[0].px4)
    torch.cuda.synchronize()
    err = cases[0].reference()
    us = graph_time(torch, [(lambda c=c: drv.launch(fn, 132, 512, smem, c.px4)) for c in cases])
    return {"us": round(us, 1), "rel_l2": float(f"{err:.2e}")}


def cmd_tune(a):
    import tilelang
    drv = Driver()
    table = load_table()
    table["generator"] = {"tool": "tilelang", "version": tilelang.__version__,
                          "nvcc": subprocess.run([os.environ.get("PLOW_NVCC", "nvcc"), "--version"],
                                                 capture_output=True, text=True).stdout.split()[-1]}
    for name in a.entries:
        e = ENTRIES[name]
        if "tune" in e:
            # Custom entries tune themselves and return the row's config, body_sha256 and
            # classes in this table's shape.
            table["entries"][name] = {"signature": e["signature"], "object": e["object"],
                                      **e["tune"]()}
            continue
        trials = []
        for cfg in e["sweep"]:
            try:
                header, arena, _ = generate(name, cfg)
                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / e["object"]
                    log, spills = nvcc_compile(name, header, out)
                    image = out.read_bytes()
                res = bench_any(drv, name, image, arena, cfg["threads"], e["classes"])
                spill = max(int(s) + int(l) for s, l in spills)
                res["spill_bytes"] = spill
                ok = (res["packed_check"]["tail_zero"]
                      and all(v["rel_l2"] < e.get("tol", 1e-3)
                              for k, v in res.items() if k != "spill_bytes"))
                trials.append((cfg, header, res, ok))
                print(f"{name} {cfg}: {json.dumps(res)} spill={spill}", flush=True)
            except Exception as ex:  # a config TileLang or nvcc rejects is just not a candidate
                print(f"{name} {cfg}: FAIL {str(ex).splitlines()[-1][:200]}", flush=True)
        good = [t for t in trials if t[3]]
        if not good:
            sys.exit(f"{name}: no valid config")
        total = lambda t: sum(v["us"] for k, v in t[2].items() if k.startswith("h"))
        best = min(good, key=total)
        classes = {}
        for cls in e["classes"]:
            key = class_key(name, cls)
            meas = [{"config": c, **r[key]} for c, _, r, ok in good]
            classes[key] = {"best": min(meas, key=lambda m: m["us"])["config"],
                            "measured": meas}
        table["entries"][name] = {"signature": e["signature"], "object": e["object"],
                                  "config": best[0], "body_sha256": body_digest(best[1]),
                                  "selection": "min total us over classes", "classes": classes,
                                  "packed_check": best[2]["packed_check"],
                                  "spill_bytes": best[2]["spill_bytes"]}
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(f"wrote {TABLE}")


def cmd_bench(a):
    drv = Driver()
    table = load_table()
    for name in a.entries:
        if "build" in ENTRIES[name]:
            print(f"{name}: custom entry, benched by its own harness (see its catalog module)")
            continue
        row = table["entries"][name]
        image = (Path(a.objdir) / row["object"]).read_bytes()
        header, arena, _ = generate(name, row["config"])
        H, KVH = ENTRIES[name]["classes"][0][:2]
        classes = ENTRIES[name]["classes"]
        if entry_op(name) != "flash_decode":
            classes = classes + [(H, KVH, int(x)) for x in a.rows]
        rows = [c[2] for c in classes]
        print(name, json.dumps(bench_any(drv, name, image, arena, row["config"]["threads"],
                                         classes), indent=1))
        if a.px4:
            for r in rows:
                if r % 64 == 0:
                    print(f"px4 rows{r}", json.dumps(bench_px4(drv, a.px4, r)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out")
    t = sub.add_parser("tune")
    be = sub.add_parser("bench")
    be.add_argument("objdir")
    be.add_argument("--px4")
    be.add_argument("--rows", nargs="*", default=[])
    c = sub.add_parser("check")
    for p in (b, t, be, c):
        p.add_argument("--entries", default=",".join(ENTRIES),
                       type=lambda s: [x for x in s.split(",") if x])
    a = ap.parse_args()
    for name in a.entries:
        if name not in ENTRIES:
            sys.exit(f"unknown catalog entry {name}; known: {', '.join(ENTRIES)}")
    {"build": cmd_build, "tune": cmd_tune, "bench": cmd_bench, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
