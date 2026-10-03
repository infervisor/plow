"""Standalone GQA decode attention: the TileLang kernel (attn_decode.py) vs FlashInfer, E4B heads.
Timing: a CUDA graph over rotating cold K/V buffers (> L2), us per call; error = rel. L2 vs an
fp32 reference on sampled rows. Needs torch, tilelang, flashinfer and one leased GPU.
  bench_decode.py HD WINDOW B CTX BLOCK_N STAGES [NSPLIT] [THREADS] [HSPLIT]
  bench_decode.py grid fi|nofi HD,WINDOW,B,CTX,BLOCK_N,STAGES[,NSPLIT,THREADS,HSPLIT] ...
"""
import sys, math, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_decode as ad

dev = "cuda"
H, KVH = 8, 2


def graph_time(fns, reps=5):
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
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / len(fns))
    ts.sort()
    return ts[len(ts) // 2]


def ref_out(q, k, v, lens, b, window, mask):
    n = int(lens[b])
    lo = n - window if window and n > window else 0
    pos = torch.arange(lo, n, device=dev)
    rows = pos & mask if mask != -1 else pos
    out = []
    for h in range(H):
        hk = h // (H // KVH)
        kk = k[b, hk, rows].float()
        vv = v[b, hk, rows].float()
        s = kk @ q[b, h].float()
        p = torch.softmax(s, 0)
        out.append(p @ vv)
    return torch.stack(out)


def run(hd, window, B, ctx, bn, stages, nsplit=1, threads=128, hsplit=1, do_fi=True):
    S = 1024 if window else ctx
    mask = S - 1 if window else -1
    span = min(ctx, window) if window else ctx
    kv_bytes = B * KVH * span * hd * 2 * 2
    n = max(2, min(16, math.ceil(160e6 / kv_bytes)))
    kern = ad.compile(hd, H // KVH, bn, stages, threads=threads, nsplit=nsplit, hsplit=hsplit)
    lens = torch.tensor([ctx - (b % 7) for b in range(B)], device=dev, dtype=torch.int32)
    bufs = []
    for i in range(n):
        q = torch.randn(B, H, hd, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, KVH, S, hd, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, KVH, S, hd, device=dev, dtype=torch.bfloat16)
        bufs.append((q, k, v))
    out = torch.empty(B, H, hd, device=dev, dtype=torch.bfloat16)
    opart = torch.empty(B, H, nsplit, hd, device=dev, dtype=torch.float32)
    ml = torch.empty(B, H, nsplit, 2, device=dev, dtype=torch.float32)
    ctr = torch.zeros(B, KVH, device=dev, dtype=torch.int32)
    scale = 1.0
    call = lambda q, k, v: kern(q, k, v, lens, out, opart, ml, ctr, window, mask, scale)
    q, k, v = bufs[0]
    call(q, k, v)
    torch.cuda.synchronize()
    err = 0.0
    for b in range(0, B, max(1, B // 6)):
        r = ref_out(q, k, v, lens, b, window, mask)
        g = out[b].float()
        err = max(err, ((g - r).norm() / r.norm()).item())
    t = graph_time([(lambda q=q, k=k, v=v: call(q, k, v)) for q, k, v in bufs])
    res = f"gen {t:.2f} us relL2 {err:.1e}"
    if do_fi:
        try:
            import flashinfer
            page = 16
            npages = (ctx + page - 1) // page
            wsb = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(wsb, "NHD", use_tensor_cores=True)
            indptr = torch.arange(0, (B + 1) * npages, npages, device=dev, dtype=torch.int32)
            last = torch.full((B,), (ctx - 1) % page + 1, device=dev, dtype=torch.int32)
            w.plan(indptr, torch.arange(B * npages, device=dev, dtype=torch.int32), last, H, KVH,
                   hd, page, window_left=(window - 1) if window else -1,
                   q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, sm_scale=scale)
            del bufs
            torch.cuda.empty_cache()
            fb = []
            for i in range(n):
                fb.append((torch.randn(B, H, hd, device=dev, dtype=torch.bfloat16),
                           torch.randn(B * npages, page, KVH, hd, device=dev, dtype=torch.bfloat16),
                           torch.randn(B * npages, page, KVH, hd, device=dev, dtype=torch.bfloat16)))
            tf = graph_time([(lambda q=q, kc=kc, vc=vc: w.run(q, (kc, vc), out=out))
                             for q, kc, vc in fb])
            res += f" | flashinfer {tf:.2f} us"
            del fb
        except Exception as e:
            res += f" | flashinfer err {str(e)[:80]}"
    floor = kv_bytes / 3.35e12 * 1e6
    print(f"hd{hd} win{window} B{B} ctx{ctx} bn{bn} st{stages} ns{nsplit} thr{threads} hs{hsplit}: "
          f"floor {floor:.2f} | {res}", flush=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if sys.argv[1] == "grid":
        fi = len(sys.argv) < 3 or sys.argv[2] != "nofi"
        for spec in sys.argv[3:] if len(sys.argv) > 3 else []:
            a = [int(x) for x in spec.split(",")]
            run(*a, do_fi=fi)
    else:
        a = [int(x) for x in sys.argv[1:]]
        run(*a)
