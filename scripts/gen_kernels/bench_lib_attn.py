"""Library attention (FlashAttention-3/4, FlashInfer) vs the packet's native role objects, at serving
shapes, reading plowrt's own KV layout.

  bench_lib_attn.py prefill CUBIN --classes H,KVH,W,ROWS,PAST[,COUNT] ... [--ilv] [--libs ...]
  bench_lib_attn.py decode  CUBIN --classes H,KVH,B,CTX ... [--ilv] [--libs ...]
  bench_lib_attn.py hd512   CUBIN --classes H,KVH,ROWS,PAST ... [--libs fa4,fa4_dq,fi2_dq]
  bench_lib_attn.py vmm     CUBIN|- ARM H,KVH,B,CTX      one arm per process (a fault kills it)

Native = the packet's object through its direct entry, with the runtime's launch policy (hd256
wgmma template: attn_pf_hd256_sliding.py Kernel, TMA staging for bf16 KV as in-model; TileLang
bodies: build_catalog.py; decode: pick_nsplit). Library arms read the SAME K/V memory through a
page table of page_size P over plowrt's slot layout, no copy: head-major [slot][kvh][stride][hd]
pages are (P rows, kvh) with head stride stride*hd; row-interleaved [slot][stride][kvh][hd] pages
are plain NHD; a ring maps logical page j to ring page (j*P & mask)/P. FP8-KV objects (one f32
scale per (position, kv head) row) have no library equivalent (library FP8 takes per-tensor or
per-(sequence, head) descales): `*_dq` arms time a dequantizing adapter (Triton, the request's
needed rows into a bf16 scratch of the same layout) plus the bf16 library kernel.

Timing: CUDA graph over rotating distinct cases (> 160 MB), median of 7 replays, us per call.
Error: rel-L2 vs fp32 attention. Library planning (FlashInfer plan, FA3 scheduler metadata) is
host work outside the graph; its host time is reported separately. Every GPU run goes through
gpulease. Needs /opt/pytorch (torch, flashinfer, vllm_flash_attn).
"""
import argparse, ctypes, json, math, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_pf_hd256_sliding as hd256
import build_catalog as bc

HBM = 3.35e12
PEAK = 989.0


def libs_arg(s):
    return [x for x in s.split(",") if x]


# ---- library plumbing --------------------------------------------------------------------------

def page_view(t, ilv, kvh, stride, hd, page):
    """(pages, page, kvh, hd) view of a plowrt KV tensor; page ids from page_id()."""
    torch = __import__("torch")
    slots = t.shape[0]
    assert stride % page == 0, f"slot stride {stride} is not a multiple of page {page}"
    if ilv:
        return t.view(slots * stride // page, page, kvh, hd)
    n = (slots * kvh - kvh + 1) * stride // page
    return torch.as_strided(t, (n, page, kvh, hd), (page * hd, hd, stride * hd, 1))


def page_id(ilv, kvh, stride, page, slot, row):
    return (slot * stride + row) // page if ilv else (slot * kvh * stride + row) // page


def page_table(reqs, ilv, kvh, stride, mask, page):
    """reqs = [(slot, kvlen)] -> (block table [n][max pages], flat indices, indptr, last_len)."""
    torch = __import__("torch")
    rows, flat, indptr, last = [], [], [0], []
    for slot, kvlen in reqs:
        n = -(-kvlen // page)
        ids = [page_id(ilv, kvh, stride, page, slot, (j * page) & mask) for j in range(n)]
        rows.append(ids)
        flat += ids
        indptr.append(len(flat))
        last.append(kvlen - (n - 1) * page)
    width = max(len(r) for r in rows)
    bt = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.int32, device="cuda")
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device="cuda")
    return bt, i32(flat), i32(indptr), i32(last)


def fa(version):
    from vllm.vllm_flash_attn import flash_attn_interface as fi
    if not fi.is_fa_version_supported(version):
        raise RuntimeError(f"FA{version}: {fi.fa_version_unsupported_reason(version)}")
    return fi


_dq_kernels = {}
DQ_ROWS = 16


def dequant_rows(k8, ks, out, ids, hd):
    """out.flat[row] = e4m3 k8.flat[row] * ks.flat[row] for every row id in `ids` (rows of hd
    elements; one f32 scale per row), rounded to out's dtype (bf16 or fp16): the adapter a
    library kernel needs in front of plowrt's per-row-scaled FP8 KV. One Triton launch."""
    import triton
    import triton.language as tl
    torch = __import__("torch")
    if not _dq_kernels:
        @triton.jit
        def _dq(k8, ks, out, ids, n, HD: tl.constexpr, R: tl.constexpr, F16: tl.constexpr):
            i = tl.program_id(0) * R + tl.arange(0, R)
            m = i < n
            r = tl.load(ids + i, mask=m, other=0).to(tl.int64)
            c = tl.arange(0, HD)
            off = r[:, None] * HD + c[None, :]
            x = tl.load(k8 + off, mask=m[:, None], other=0).to(tl.float8e4nv, bitcast=True)
            y = x.to(tl.float32) * tl.load(ks + r, mask=m, other=0.0)[:, None]
            tl.store(out + off, y.to(tl.float16 if F16 else tl.bfloat16), mask=m[:, None])
        _dq_kernels["k"] = _dq
    n = ids.numel()
    _dq_kernels["k"][(triton.cdiv(n, DQ_ROWS),)](k8, ks, out, ids, n, HD=hd, R=DQ_ROWS,
                                                 F16=out.dtype == torch.float16, num_warps=4)


def needed_rows(reqs, kvh, stride, mask, window, ilv=False):
    """Flat row ids ((slot, head, ring row) in plowrt's layout) a prefill/decode request set
    reads: [kvlen - qlen - window + 1, kvlen) per request (all rows without a window)."""
    torch = __import__("torch")
    ids = []
    for slot, qlen, kvlen in reqs:
        lo = max(0, kvlen - qlen - window + 1) if window else 0
        pos = torch.arange(lo, kvlen, device="cuda", dtype=torch.int64) & mask
        for h in range(kvh):
            if ilv:
                ids.append((slot * stride + pos) * kvh + h)
            else:
                ids.append((slot * kvh + h) * stride + pos)
    return torch.cat(ids).to(torch.int64)


# ---- prefill -----------------------------------------------------------------------------------

class PrefillLib:
    """Library arms over one hd256 Case (attn_pf_hd256_sliding.Case)."""

    def __init__(self, case, page, fp8=False, half=False):
        torch = __import__("torch")
        c = case
        self.c, self.page, self.fp8, self.half = c, page, fp8, half
        self.reqs = c.requests
        real = c.requests[-1][0] + c.requests[-1][1]
        self.real = real
        self.ilv = getattr(c, "ilv", False)
        if fp8:
            # The adapter's bf16 scratch, same layout as the cache.
            # fp16 scratch (`half`): e4m3 x scale keeps more mantissa; Q and O then pass through
            # fp16 too (casts timed with the adapter).
            dt = torch.float16 if half else torch.bfloat16
            self.kb = torch.empty(c.k8.shape, dtype=dt, device="cuda")
            self.vb = torch.empty_like(self.kb)
            if half:
                self.q16 = torch.empty(real, c.H, hd256.D, dtype=dt, device="cuda")
                self.o16 = torch.empty_like(self.q16)
            self.ids = needed_rows([(s, q, kv) for _, q, s, kv in c.requests], c.KVH,
                                   c.kv_stride, c.kv_mask, c.window)
            src_k, src_v = self.kb, self.vb
        else:
            src_k, src_v = c.k, c.v
        self.kp = page_view(src_k, self.ilv, c.KVH, c.kv_stride, hd256.D, page)
        self.vp = page_view(src_v, self.ilv, c.KVH, c.kv_stride, hd256.D, page)
        self.bt, self.flat, self.kv_indptr, self.last = page_table(
            [(s, kv) for _, _, s, kv in c.requests], self.ilv, c.KVH, c.kv_stride, c.kv_mask, page)
        self.cu_q = torch.tensor([0] + [q0 + ql for q0, ql, _, _ in c.requests],
                                 dtype=torch.int32, device="cuda")
        self.seqused = torch.tensor([kv for *_, kv in c.requests], dtype=torch.int32,
                                    device="cuda")
        self.max_q = max(ql for _, ql, _, _ in c.requests)
        self.max_k = max(kv for *_, kv in c.requests)

    def adapt(self):
        if self.fp8:
            dequant_rows(self.c.k8, self.c.ks, self.kb, self.ids, hd256.D)
            dequant_rows(self.c.v8, self.c.vs, self.vb, self.ids, hd256.D)
            if self.half:
                self.q16.copy_(self.c.q[:self.real])

    def io(self):
        c = self.c
        return (self.q16, self.o16) if self.half else (c.q[:self.real], c.o[:self.real])

    def finish(self):
        if self.half:
            self.c.o[:self.real].copy_(self.o16)

    def fa3(self, scale=1.0, version=3):
        fi = fa(version)
        c = self.c
        win = [c.window - 1, 0] if c.window else None

        def run():
            self.adapt()
            q, o = self.io()
            fi.flash_attn_varlen_func(
                q, self.kp, self.vp, self.max_q, self.cu_q, self.max_k,
                seqused_k=self.seqused, causal=True, window_size=win, softmax_scale=scale,
                block_table=self.bt, out=o, fa_version=version)
            self.finish()
        return run

    def flashinfer(self, backend, scale=1.0):
        import flashinfer
        torch = __import__("torch")
        c = self.c
        ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
        dt = torch.float16 if self.half else torch.bfloat16
        plan = lambda: w.plan(self.cu_q, self.kv_indptr, self.flat, self.last, c.H, c.KVH,
                              hd256.D, self.page, causal=True,
                              window_left=(c.window - 1) if c.window else -1,
                              q_data_type=dt, kv_data_type=dt, sm_scale=scale)
        plan()  # JIT/load outside the timed plan
        t0 = time.perf_counter()
        plan()
        self.plan_us = (time.perf_counter() - t0) * 1e6

        def run():
            self.adapt()
            q, o = self.io()
            w.run(q, (self.kp, self.vp), out=o)
            self.finish()
        run._keep = (ws, w)
        return run


def cmd_prefill(a):
    drv = hd256.Driver()
    torch = drv.torch
    kern = hd256.Kernel(drv, a.cubin)
    fp8 = kern.fp8
    if not fp8:
        kern = hd256.Kernel(drv, a.cubin, tma=True)  # the bf16 role stages through KV maps
    out = {}
    for spec in a.classes:
        H, KVH, W, rows, past, *rest = [int(x) for x in spec.split(",")]
        count = rest[0] if rest else 1
        reqs = hd256.class_requests(rows, past, count)
        span = max((W or kv) + ql - 1 for _, ql, _, kv in reqs) if W else max(kv for *_, kv in reqs)
        per = rows * count * 2 * H * hd256.D * 2 + 2 * KVH * count * span * hd256.D * (1 if fp8 else 2)
        n = max(2, min(8, math.ceil(160e6 / per)))
        cases = []
        for s in range(n):
            lin = {} if W else {"ring": False, "kv_stride": -(-(rows + past) // 256) * 256}
            c = hd256.Case(drv, H, KVH, W, reqs, seed=s, fp8=fp8, ilv=a.ilv, **lin)
            cases.append(c)
        fl = hd256.flop(H, W, reqs)
        key = f"h{H}kv{KVH}_w{W}_rows{rows}_past{past}" + (f"_x{count}" if count > 1 else "")
        res = {"floor_us": round(fl / PEAK / 1e6, 1), "gflop": round(fl / 1e9, 2)}
        kern(cases[0])
        torch.cuda.synchronize()
        err = cases[0].reference_error()[0]
        us = hd256.graph_time(torch, [(lambda c=c: kern(c)) for c in cases])
        res["native"] = {"us": round(us, 1), "rel_l2": float(f"{err:.2e}"),
                         "tflops": round(fl / us / 1e6)}
        for lib in a.libs:
            try:
                half = lib.endswith("_h")
                arms = [PrefillLib(c, a.page, fp8, half) for c in cases]
                if lib.startswith("fa3"):
                    fns = [x.fa3() for x in arms]
                elif lib.startswith("fa4"):
                    fns = [x.fa3(version=4) for x in arms]
                else:
                    fns = [x.flashinfer({"fi3": "fa3", "fi2": "fa2"}[lib.split("_")[0]])
                           for x in arms]
                cases[0].o.fill_(float("nan"))
                fns[0]()
                torch.cuda.synchronize()
                err = cases[0].reference_error()[0]
                us = hd256.graph_time(torch, fns)
                r = {"us": round(us, 1), "rel_l2": float(f"{err:.2e}"),
                     "tflops": round(fl / us / 1e6)}
                if hasattr(arms[0], "plan_us"):
                    r["plan_host_us"] = round(arms[0].plan_us, 1)
                if fp8:
                    # Adapter alone.
                    r["adapter_us"] = round(hd256.graph_time(torch, [x.adapt for x in arms]), 1)
                res[lib.removesuffix("_h") + ("_dq" if fp8 else "") + ("16" if half else "")] = r
                del arms, fns
            except Exception as e:  # report and keep going: a blocker is a result
                res[lib] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            torch.cuda.empty_cache()
        print(key, json.dumps(res), flush=True)
        out[key] = res
        del cases
        torch.cuda.empty_cache()
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


# ---- hd512 (global layers; TileLang FP8-KV object) ---------------------------------------------

class Fa4Lib:
    """libplow_fa4_hd512.so (scripts/build_sm90a_fa4_hd512.py): the AOT-exported FA4 entry
    plowrt would load, called through its C ABI on torch's current stream."""

    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        self.h = ctypes.c_void_p()
        rc = self.lib.plow_fa4_create(0, ctypes.byref(self.h))
        if rc:
            raise RuntimeError(f"plow_fa4_create rc={rc}")
        assert self.lib.plow_fa4_abi() == 2
        f = self.lib.plow_fa4_hd512_run
        f.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint] + [ctypes.c_void_p] * 4 + \
            [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 3 + [
                ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_float, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        self.f = f
        torch = __import__("torch")
        self.ws = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")

    @staticmethod
    def splits(H, KVH, nreq, max_q, max_kv, rows, ws_bytes):
        """plowrt lib_attention.rs Launch::splits (vLLM's num_splits_heuristic)."""
        n = -(-max_kv // 64)
        # GQA packed only at 2 KV heads (export_fa4_hd512_sm90.py).
        m = nreq * KVH * (-(-(max_q * (H // KVH)) // 64)) if KVH == 2 else nreq * H * (-(-max_q // 64))
        if n <= 4 or m == 0:
            return 1
        s = max(1, min(132 // m, 128, n))
        while s > 1 and s * rows * 16 * (512 * 4 + 4) > ws_bytes:
            s -= 1
        return s

    def run(self, H, KVH, q, kp, vp, o, total_q, cu_q, used, bt, head_stride, scale=1.0,
            splits=1):
        torch = __import__("torch")
        rc = self.f(self.h, H, KVH, q.data_ptr(), kp.data_ptr(), vp.data_ptr(), o.data_ptr(),
                    total_q, used.numel(), cu_q.data_ptr(), used.data_ptr(), bt.data_ptr(),
                    bt.shape[1], kp.shape[0], head_stride, scale, splits, self.ws.data_ptr(),
                    self.ws.numel(), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"plow_fa4_hd512_run rc={rc}")


_fa4_lib = None


class PagePool:
    """The runtime route's FP8 path: kv_dequant_sm90.cubin gathers each request's rows [0,
    kv_len) into a bf16 pool [kvh][pages][64][hd] (a page table over consecutive pool pages),
    then libplow_fa4_hd512.so reads the pool."""

    def __init__(self, drv, cubin, reqs, k8, ks, v8, vs, kvh, stride, hd=512):
        torch = drv.torch
        self.drv = drv
        mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        drv.check(drv.cu.cuModuleLoadData(ctypes.byref(mod), Path(cubin).read_bytes()), "load")
        drv.check(drv.cu.cuModuleGetFunction(ctypes.byref(self.fn), mod,
                                             b"plow_kv_dequant_pages"), "fn")
        self._mod = mod
        pages, bt, n = [], [], 0
        for slot, kvlen in reqs:
            ids = []
            for p in range(-(-kvlen // 64)):
                pages.append([slot, 64 * p, min(64, kvlen - 64 * p)])
                ids.append(n)
                n += 1
            bt.append(ids)
        w = max(len(r) for r in bt)
        self.bt = torch.tensor([r + [0] * (w - len(r)) for r in bt], dtype=torch.int32,
                               device="cuda")
        self.pages = torch.tensor(pages, dtype=torch.int32, device="cuda")
        self.n = n
        self.ko = torch.empty(kvh, n, 64, hd, dtype=torch.bfloat16, device="cuda")
        self.vo = torch.empty_like(self.ko)
        self.kvh, self.stride = kvh, stride
        self.kp = self.ko.permute(1, 2, 0, 3)  # (pages, 64, kvh, hd), head stride n*64*hd
        self.vp = self.vo.permute(1, 2, 0, 3)
        self.args = bc.pack_args([("p", k8.data_ptr()), ("p", ks.data_ptr()), ("p", v8.data_ptr()),
                                  ("p", vs.data_ptr()), ("p", self.ko.data_ptr()),
                                  ("p", self.vo.data_ptr()), ("p", self.pages.data_ptr()),
                                  ("u", n), ("u", kvh), ("u", stride), ("u", n)])

    def adapt(self):
        buf = ctypes.create_string_buffer(self.args, len(self.args))
        params = (ctypes.c_void_p * 1)(ctypes.addressof(buf))
        stream = ctypes.c_void_p(self.drv.torch.cuda.current_stream().cuda_stream)
        self.drv.check(self.drv.cu.cuLaunchKernel(self.fn, self.n, self.kvh, 2, 256, 1, 1, 0,
                                                  stream, params, None), "dequant")
        self._keep = (buf, params)


def fa4_lib(path):
    global _fa4_lib
    if _fa4_lib is None:
        _fa4_lib = Fa4Lib(path)
    return _fa4_lib


def cmd_hd512(a):
    drv = bc.Driver()
    torch = drv.torch
    image = Path(a.cubin).read_bytes()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    drv.check(drv.cu.cuModuleLoadData(ctypes.byref(mod), image), "load")
    drv.check(drv.cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"plow_gen_flash_prefill_direct"),
              "fn")
    g = hd256.Driver.global_u32
    block, arena = g(drv, mod, "plow_gen_block"), g(drv, mod, "plow_gen_arena_bytes")
    fp8 = g(drv, mod, "plow_gen_flash_prefill_abi") == 2
    drv.check(drv.cu.cuFuncSetAttribute(fn, 8, ctypes.c_int(arena)), "smem")
    D = g(drv, mod, "plow_attention_head_dim")
    out = {}
    for spec in a.classes:
        H, KVH, rows, past = [int(x) for x in spec.split(",")]
        kvlen = rows + past
        stride = -(-kvlen // 64) * 64
        per = (rows * 2 * H + kvlen * 2 * KVH) * D * 2
        n = max(2, min(6, math.ceil(160e6 / per)))
        cases = [bc.Case(drv, H, KVH, D, rows, stride, 1, [(0, rows, 0, kvlen)], seed=s, fp8=fp8)
                 for s in range(n)]
        fl = 4 * H * D * sum(past + r + 1 for r in range(rows))
        key = f"hd{D}_h{H}kv{KVH}_rows{rows}_past{past}"
        res = {"floor_us": round(fl / PEAK / 1e6, 1)}
        launch = lambda c: drv.launch(fn, 132, block, arena, c.gen_args())
        launch(cases[0])
        torch.cuda.synchronize()
        res["native"] = {"us": round(bc.graph_time(torch, [(lambda c=c: launch(c)) for c in cases]), 1),
                         "rel_l2": float(f"{cases[0].reference():.2e}")}
        res["native"]["tflops"] = round(fl / res["native"]["us"] / 1e6)
        for lib in a.libs:
            try:
                fns, adapts = [], []
                half = lib.endswith("_h")
                dt = torch.float16 if half else torch.bfloat16
                for c in cases:
                    # Gemma-4 global layers share K and V (attention_k_eq_v): V = K here too
                    # would change the reference, so the library reads the case's own V.
                    if fp8:
                        kb = torch.empty(c.k8.shape, dtype=dt, device="cuda")
                        vb = torch.empty_like(kb)
                        ids = needed_rows([(0, rows, kvlen)], KVH, stride, 0xFFFFFFFF, 0)
                    else:
                        kb, vb, ids = c.k, c.v, None
                    kp = page_view(kb, False, KVH, stride, D, a.page)
                    vp = page_view(vb, False, KVH, stride, D, a.page)
                    bt, flat, indptr, last = page_table([(0, kvlen)], False, KVH, stride,
                                                        0xFFFFFFFF, a.page)
                    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
                    used = torch.tensor([kvlen], dtype=torch.int32, device="cuda")

                    q16 = torch.empty_like(c.q, dtype=dt) if half else c.q
                    o16 = torch.empty_like(c.o, dtype=dt) if half else c.o

                    def adapt(c=c, kb=kb, vb=vb, ids=ids, q16=q16):
                        if ids is not None:
                            dequant_rows(c.k8, c.ks, kb, ids, D)
                            dequant_rows(c.v8, c.vs, vb, ids, D)
                        if half:
                            q16.copy_(c.q)
                    adapts.append(adapt)
                    if lib == "fa4rt":
                        aot = fa4_lib(a.fa4lib)
                        pool = PagePool(drv, a.dqcubin, [(0, kvlen)], c.k8, c.ks, c.v8, c.vs,
                                        KVH, stride)
                        adapts[-1] = pool.adapt

                        ns = Fa4Lib.splits(H, KVH, 1, rows, kvlen, rows, aot.ws.numel())
                        res.setdefault("fa4rt_splits", ns)

                        def run(c=c, pool=pool, cu_q=cu_q, used=used, ns=ns):
                            pool.adapt()
                            aot.run(H, KVH, c.q, pool.kp, pool.vp, c.o, rows, cu_q, used,
                                    pool.bt, pool.kp.stride(2), splits=ns)
                    elif lib.startswith("fa4aot"):
                        aot = fa4_lib(a.fa4lib)
                        head_stride = kp.stride(2)

                        def run(c=c, kp=kp, vp=vp, bt=bt, cu_q=cu_q, used=used, adapt=adapt,
                                q16=q16, o16=o16, hs=head_stride):
                            adapt()
                            aot.run(H, KVH, q16, kp, vp, o16, rows, cu_q, used, bt, hs)
                    elif lib.startswith("fa4"):
                        fi = fa(4)

                        def run(c=c, kp=kp, vp=vp, bt=bt, cu_q=cu_q, used=used, adapt=adapt,
                                q16=q16, o16=o16):
                            adapt()
                            fi.flash_attn_varlen_func(q16, kp, vp, rows, cu_q, kvlen,
                                                      seqused_k=used, causal=True,
                                                      softmax_scale=1.0, block_table=bt,
                                                      out=o16, fa_version=4)
                            if half:
                                c.o.copy_(o16)
                    else:
                        import flashinfer
                        ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
                        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
                        w.plan(cu_q, indptr, flat, last, H, KVH, D, a.page, causal=True,
                               q_data_type=dt, kv_data_type=dt, sm_scale=1.0)

                        def run(c=c, kp=kp, vp=vp, w=w, ws=ws, adapt=adapt, q16=q16, o16=o16):
                            adapt()
                            w.run(q16, (kp, vp), out=o16)
                            if half:
                                c.o.copy_(o16)
                    fns.append(run)
                cases[0].o.fill_(float("nan"))
                fns[0]()
                torch.cuda.synchronize()
                err = cases[0].reference()
                us = bc.graph_time(torch, fns)
                name = lib.removesuffix("_h") + ("_dq" if fp8 else "") + ("16" if half else "")
                res[name] = {"us": round(us, 1), "rel_l2": float(f"{err:.2e}"),
                             "tflops": round(fl / us / 1e6)}
                if fp8:
                    res[name]["adapter_us"] = round(bc.graph_time(torch, adapts), 1)
            except Exception as e:
                res[lib] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            torch.cuda.empty_cache()
        print(key, json.dumps(res), flush=True)
        out[key] = res
        del cases
        torch.cuda.empty_cache()
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


# ---- decode ------------------------------------------------------------------------------------

class DecodeObj:
    def __init__(self, drv, cubin):
        image = Path(cubin).read_bytes()
        self.drv = drv
        self.mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        drv.check(drv.cu.cuModuleLoadData(ctypes.byref(self.mod), image), "load")
        drv.check(drv.cu.cuModuleGetFunction(ctypes.byref(self.fn), self.mod,
                                             b"plow_gen_flash_decode"), "fn")
        g = lambda n: hd256.Driver.global_u32(drv, self.mod, n)
        self.block, self.arena = g("plow_gen_block"), g("plow_gen_arena_bytes")
        self.hd, self.gqa = g("plow_attention_head_dim"), g("plow_gen_decode_gqa")
        self.ns_max, self.bn = g("plow_gen_decode_nsplit_max"), g("plow_attention_kv_tile")
        try:
            self.ilv = g("plow_gen_kv_layout") == 1
        except RuntimeError:
            self.ilv = False
        drv.check(drv.cu.cuFuncSetAttribute(self.fn, 8, ctypes.c_int(self.arena)), "smem")

    def nsplit(self, B, KVH, ctx):
        return bc.pick_nsplit(B, KVH, -(-ctx // self.bn), self.ns_max)

    def launch(self, c, ns):
        self.drv.launch(self.fn, (c.KVH * ns, c.B), self.block, self.arena, c.args(ns))


class DecodeLib:
    def __init__(self, c, page):
        torch = __import__("torch")
        self.c, self.page = c, page
        slots = [int(x) for x in c.slot.tolist()]
        lens = [int(x) for x in c.lens.tolist()]
        self.kp = page_view(c.k, c.ilv, c.KVH, c.kv_stride, c.D, page)
        self.vp = page_view(c.v, c.ilv, c.KVH, c.kv_stride, c.D, page)
        self.bt, self.flat, self.indptr, self.last = page_table(
            list(zip(slots, lens)), c.ilv, c.KVH, c.kv_stride, c.kv_mask, page)
        self.cu_q = torch.arange(c.B + 1, dtype=torch.int32, device="cuda")
        self.max_k = max(lens)

    def fa3(self, sched):
        fi = fa(3)
        c = self.c
        meta = None
        self.plan_us = 0.0
        if sched:
            from vllm.vllm_flash_attn import get_scheduler_metadata
            sched_args = lambda: get_scheduler_metadata(
                batch_size=c.B, max_seqlen_q=1, max_seqlen_k=self.max_k, num_heads_q=c.H,
                num_heads_kv=c.KVH, headdim=c.D, cache_seqlens=c.lens, qkv_dtype=c.q.dtype,
                page_size=self.page, cu_seqlens_q=self.cu_q, causal=True,
                window_size=(c.window - 1, 0) if c.window else (-1, -1))
            sched_args()
            t0 = time.perf_counter()
            meta = sched_args()
            self.plan_us = (time.perf_counter() - t0) * 1e6
        win = [c.window - 1, 0] if c.window else None

        def run():
            fi.flash_attn_varlen_func(c.q, self.kp, self.vp, 1, self.cu_q, self.max_k,
                                      seqused_k=c.lens, causal=True, window_size=win,
                                      softmax_scale=1.0, block_table=self.bt, out=c.o,
                                      scheduler_metadata=meta, fa_version=3)
        return run

    def flashinfer(self, tc):
        import flashinfer
        torch = __import__("torch")
        c = self.c
        ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
        plan = lambda: w.plan(self.indptr, self.flat, self.last, c.H, c.KVH, c.D, self.page,
                              window_left=(c.window - 1) if c.window else -1,
                              q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                              sm_scale=1.0)
        plan()
        t0 = time.perf_counter()
        plan()
        self.plan_us = (time.perf_counter() - t0) * 1e6

        def run():
            w.run(c.q, (self.kp, self.vp), out=c.o)
        run._keep = (ws, w)
        return run


def cmd_decode(a):
    drv = bc.Driver()
    torch = drv.torch
    obj = DecodeObj(drv, a.cubin)
    out = {}
    for spec in a.classes:
        H, KVH, B, ctx = [int(x) for x in spec.split(",")]
        assert H == KVH * obj.gqa and obj.hd > 0
        per = B * KVH * ctx * obj.hd * 2 * 2
        n = max(2, min(8, math.ceil(160e6 / per)))
        cases = [bc.DecodeCase(drv, H, KVH, obj.hd, B, ctx, obj.ns_max, seed=s, ilv=obj.ilv)
                 for s in range(n)]
        ns = obj.nsplit(B, KVH, ctx)
        key = f"h{H}kv{KVH}_b{B}_ctx{ctx}"
        res = {"floor_us": round(per / HBM * 1e6, 1), "nsplit": ns}
        obj.launch(cases[0], ns)
        torch.cuda.synchronize()
        err = cases[0].reference()
        us = bc.graph_time(torch, [(lambda c=c: obj.launch(c, ns)) for c in cases], reps=7)
        res["native"] = {"us": round(us, 1), "rel_l2": float(f"{err:.2e}"),
                         "tb_s": round(per / us / 1e6, 2)}
        for lib in a.libs:
            try:
                arms = [DecodeLib(c, a.page) for c in cases]
                if lib == "fa3":
                    fns = [x.fa3(False) for x in arms]
                elif lib == "fa3s":
                    fns = [x.fa3(True) for x in arms]
                elif lib == "fi":
                    fns = [x.flashinfer(False) for x in arms]
                else:
                    fns = [x.flashinfer(True) for x in arms]
                cases[0].o.fill_(float("nan"))
                fns[0]()
                torch.cuda.synchronize()
                err = cases[0].reference()
                us = bc.graph_time(torch, fns, reps=7)
                res[lib] = {"us": round(us, 1), "rel_l2": float(f"{err:.2e}"),
                            "tb_s": round(per / us / 1e6, 2),
                            "plan_host_us": round(arms[0].plan_us, 1)}
                del arms, fns
            except Exception as e:
                res[lib] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            torch.cuda.empty_cache()
        print(key, json.dumps(res), flush=True)
        out[key] = res
        del cases
        torch.cuda.empty_cache()
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


# ---- live-KV over-read (VMM) -------------------------------------------------------------------

class VmmKV:
    """[slots][stride][kvh][hd] bf16 (row-interleaved) per-slot VA reservations with only the
    first ceil(kv_len / 4096) 4096-row blocks mapped, like plowrt's live KV."""

    def __init__(self, drv, B, stride, kvh, hd, lens):
        cu = drv.cu
        self.cu, self.handles = cu, []
        row_bytes = kvh * hd * 2
        slot_bytes = stride * row_bytes
        gran = ctypes.c_size_t()

        class Prop(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                        ("locType", ctypes.c_int), ("locId", ctypes.c_int),
                        ("win32", ctypes.c_void_p), ("compressionType", ctypes.c_ubyte),
                        ("gpuDirectRDMACapable", ctypes.c_ubyte), ("usage", ctypes.c_ushort),
                        ("reserved", ctypes.c_ubyte * 4)]

        class Access(ctypes.Structure):
            _fields_ = [("locType", ctypes.c_int), ("locId", ctypes.c_int), ("flags", ctypes.c_int)]
        prop = Prop(type=1, locType=1, locId=0)
        drv.check(cu.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0),
                  "granularity")
        blk = 4096 * row_bytes
        assert blk % gran.value == 0 and slot_bytes % blk == 0
        total = B * slot_bytes
        self.base = ctypes.c_uint64()
        drv.check(cu.cuMemAddressReserve(ctypes.byref(self.base), ctypes.c_size_t(total),
                                         ctypes.c_size_t(0), ctypes.c_uint64(0),
                                         ctypes.c_uint64(0)), "reserve")
        acc = Access(locType=1, locId=0, flags=3)
        for s in range(B):
            for b in range(-(-lens[s] // 4096)):
                h = ctypes.c_uint64()
                drv.check(cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(blk), ctypes.byref(prop),
                                         ctypes.c_uint64(0)), "create")
                va = self.base.value + s * slot_bytes + b * blk
                drv.check(cu.cuMemMap(ctypes.c_uint64(va), ctypes.c_size_t(blk), ctypes.c_size_t(0),
                                      h, ctypes.c_uint64(0)), "map")
                drv.check(cu.cuMemSetAccess(ctypes.c_uint64(va), ctypes.c_size_t(blk),
                                            ctypes.byref(acc), ctypes.c_size_t(1)), "access")
                self.handles.append(h)
        torch = drv.torch
        self.slot_bytes, self.total = slot_bytes, total

        class Iface:
            def __init__(s, ptr):
                s.__cuda_array_interface__ = {"shape": (B, stride, kvh, hd), "typestr": "<i2",
                                              "data": (ptr, False), "version": 3}
        self.t = torch.as_tensor(Iface(self.base.value), device="cuda").view(torch.bfloat16)
        for s in range(B):
            self.t[s, :lens[s]].normal_()


def cmd_vmm(a):
    """One arm at one point: native object or a library over KV whose unmapped tail faults."""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    drv = bc.Driver()
    torch = drv.torch
    H, KVH, B, ctx = [int(x) for x in a.point.split(",")]
    hd = 256
    stride = -(-ctx // 4096) * 4096 + 4096
    lens = [ctx - (b % 3) for b in range(B)]
    c = bc.DecodeCase.__new__(bc.DecodeCase)
    c.H, c.KVH, c.D, c.B, c.window, c.ilv = H, KVH, hd, B, 0, True
    c.kv_stride, c.kv_mask = stride, 0xFFFFFFFF
    kv_k, kv_v = VmmKV(drv, B, stride, KVH, hd, lens), VmmKV(drv, B, stride, KVH, hd, lens)
    c.k, c.v = kv_k.t, kv_v.t
    c.q = (torch.randn(B, H, hd, device="cuda") * hd ** -0.5).to(torch.bfloat16)
    c.lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    c.slot = torch.arange(B, dtype=torch.int32, device="cuda")
    c.o = torch.full((B, H, hd), float("nan"), device="cuda", dtype=torch.bfloat16)
    if a.arm == "native":
        obj = DecodeObj(drv, a.cubin)
        c.opart = torch.zeros(B, H, obj.ns_max, hd, device="cuda", dtype=torch.float32)
        c.ml = torch.zeros(B, H, obj.ns_max, 2, device="cuda", dtype=torch.float32)
        c.ctr = torch.zeros(B, KVH, device="cuda", dtype=torch.int32)
        obj.launch(c, obj.nsplit(B, KVH, ctx))
    else:
        x = DecodeLib(c, a.page)
        {"fa3": lambda: x.fa3(False), "fa3s": lambda: x.fa3(True),
         "fi": lambda: x.flashinfer(False), "fitc": lambda: x.flashinfer(True)}[a.arm]()()
    torch.cuda.synchronize()
    print(f"VMM {a.arm} {a.point} OK rel_l2 {c.reference():.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefill")
    p.add_argument("cubin")
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--ilv", action="store_true")
    p.add_argument("--libs", type=libs_arg, default=["fa3", "fi3", "fi2"])
    p.add_argument("--page", type=int, default=64)
    p.add_argument("--json")
    h = sub.add_parser("hd512")
    h.add_argument("cubin")
    h.add_argument("--classes", nargs="+", required=True)
    h.add_argument("--libs", type=libs_arg, default=["fa4", "fi2"])
    h.add_argument("--page", type=int, default=64)
    h.add_argument("--dqcubin", default="/opt/dlami/nvme/lava-tts/lib-attn/kv_dequant_sm90.cubin")
    h.add_argument("--fa4lib", default="/opt/dlami/nvme/lava-tts/lib-attn/fa4lib2/libplow_fa4_hd512.so")
    h.add_argument("--json")
    d = sub.add_parser("decode")
    d.add_argument("cubin")
    d.add_argument("--classes", nargs="+", required=True)
    d.add_argument("--libs", type=libs_arg, default=["fa3", "fa3s", "fi", "fitc"])
    d.add_argument("--page", type=int, default=64)
    d.add_argument("--json")
    v = sub.add_parser("vmm")
    v.add_argument("cubin")
    v.add_argument("arm")
    v.add_argument("point")
    v.add_argument("--page", type=int, default=64)
    a = ap.parse_args()
    {"prefill": cmd_prefill, "hd512": cmd_hd512, "decode": cmd_decode, "vmm": cmd_vmm}[a.cmd](a)


if __name__ == "__main__":
    main()
