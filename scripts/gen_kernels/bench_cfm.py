"""Standalone S3Gen CFM attention (attn_cfm.py) at the csynth shapes: us per call (CUDA graph over
rotating inputs) and error vs an fp64 reference (max|d| / rms(ref), rel. L2).
  bench_cfm.py ITEMS T BM BN THREADS STAGES [src]   (TileLang)
  bench_cfm.py so LIB ITEMS T WHICH[:NBLK] ...      (cfm_attn.cu: 0 = interpreter path, 1.. candidates)
Needs torch, tilelang and one leased GPU.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cfm as ac

dev = "cuda"
H, HW, PRE = 8, 64, 306


def graph_time(fns, reps=7):
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


def inputs(B, Tq, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    qkv = torch.randn(B, Tq, 3 * H * HW, device=dev, generator=g)
    prefix = torch.randn(4, PRE, 2 * H * HW, device=dev, generator=g)
    pidx = (torch.arange(B, device=dev, dtype=torch.int32) % 4)
    # Real windows: the last items' own rows partly masked.
    klen = torch.tensor([PRE + Tq - (b % 5) * 3 for b in range(B)], device=dev, dtype=torch.int32)
    return qkv, prefix, pidx, klen


def reference(qkv, prefix, pidx, klen, scale):
    B, Tq, _ = qkv.shape
    q = qkv[:, :, :H * HW].double().view(B, Tq, H, HW).transpose(1, 2)
    k = torch.cat([prefix[pidx.long(), :, :H * HW], qkv[:, :, H * HW:2 * H * HW]], 1).double()
    v = torch.cat([prefix[pidx.long(), :, H * HW:], qkv[:, :, 2 * H * HW:]], 1).double()
    k = k.view(B, -1, H, HW).transpose(1, 2)
    v = v.view(B, -1, H, HW).transpose(1, 2)
    s = q @ k.transpose(-1, -2) * scale
    j = torch.arange(s.shape[-1], device=dev)
    s = s.masked_fill(j[None, None, None, :] >= klen.long()[:, None, None, None], float("-inf"))
    o = torch.softmax(s, -1) @ v
    return o.transpose(1, 2).reshape(B, Tq, H * HW)


def err(out, ref):
    d = (out.double() - ref)
    rms = ref.pow(2).mean().sqrt()
    return (d.abs().max() / rms).item(), (d.norm() / ref.norm()).item()


def run(B, Tq, bm, bn, threads, stages, src=False):
    kern = ac.compile(bm, bn, threads, stages)
    if src:
        print(kern.get_kernel_source())
    scale = 0.125
    qkv, prefix, pidx, klen = inputs(B, Tq)
    out = torch.empty(B, Tq, H * HW, device=dev)
    kern(qkv, prefix, pidx, klen, out, scale)
    torch.cuda.synchronize()
    e = err(out, reference(qkv, prefix, pidx, klen, scale))
    bufs = [inputs(B, Tq, s)[0] for s in range(4)]
    us = graph_time([lambda x=x: kern(x, prefix, pidx, klen, out, scale) for x in bufs])
    print(f"tilelang B{B} T{Tq} bm{bm} bn{bn} thr{threads} st{stages}: {us:.1f} us  max|d|/rms {e[0]:.2e}  relL2 {e[1]:.2e}",
          flush=True)


def run_so(lib, B, Tq, specs):
    import ctypes
    so = ctypes.CDLL(lib)
    so.cfm_make.restype = ctypes.c_void_p
    so.cfm_make.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint] * 3
    so.cfm_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    scale = 0.125
    qkv, prefix, pidx, klen = inputs(B, Tq)
    ref = reference(qkv, prefix, pidx, klen, scale)
    out = torch.empty(B, Tq, H * HW, device=dev)
    bufs = [inputs(B, Tq, s)[0] for s in range(4)]
    mk = lambda x: so.cfm_make(out.data_ptr(), x.data_ptr(), prefix.data_ptr(), pidx.data_ptr(), klen.data_ptr(), B, Tq, PRE)
    h0 = mk(qkv)
    hs = [mk(x) for x in bufs]
    for spec in specs:
        which, nblk = (int(v) for v in (spec.split(":") + ["132"])[:2])
        out.fill_(float("nan"))
        stream = torch.cuda.current_stream().cuda_stream
        assert so.cfm_run(h0, which, nblk, stream) == 0
        torch.cuda.synchronize()
        e = err(out, ref)
        us = graph_time([lambda h=h: so.cfm_run(h, which, nblk, torch.cuda.current_stream().cuda_stream) for h in hs])
        print(f"cfm_attn B{B} T{Tq} which{which} nblk{nblk}: {us:.1f} us  max|d|/rms {e[0]:.2e}  relL2 {e[1]:.2e}",
              flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "so":
        run_so(a[1], int(a[2]), int(a[3]), a[4:])
    else:
        run(*[int(x) for x in a[:6]], src=len(a) > 6)
