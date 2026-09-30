#!/usr/bin/env python3
"""Time `plow_sample` from a sampler cubin on synthetic logits (device events), and check a
candidate cubin draws the same tokens as the control for identical inputs.

  gpulease -n 1 smp python scripts/tts/sample_kernel_bench.py CTRL.cubin [CAND.cubin] --V 156951
"""
import argparse, ctypes, statistics, sys

import torch


def load(path, cuda):
    mod = ctypes.c_void_p()
    img = open(path, "rb").read()
    assert cuda.cuModuleLoadData(ctypes.byref(mod), img) == 0, "cuModuleLoadData"
    fn = ctypes.c_void_p()
    assert cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"plow_sample") == 0, "plow_sample"
    ptr, size = ctypes.c_uint64(), ctypes.c_size_t()
    threads = 256
    if cuda.cuModuleGetGlobal_v2(ctypes.byref(ptr), ctypes.byref(size), mod, b"plow_sample_threads") == 0:
        v = ctypes.c_uint32()
        assert cuda.cuMemcpyDtoH_v2(ctypes.byref(v), ptr, 4) == 0
        threads = v.value
    return fn, threads


def launch(cuda, fn, B, args, stream):
    fn, threads = fn
    arr = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    rc = cuda.cuLaunchKernel(fn, B, 1, 1, threads, 1, 1, 0, ctypes.c_void_p(stream), arr, None)
    assert rc == 0, f"launch rc={rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cubins", nargs="+")
    ap.add_argument("--V", type=int, default=156951)
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scale", type=float, default=2.0, help="logit noise scale (smaller = broader)")
    args = ap.parse_args()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    cuda = ctypes.CDLL("libcuda.so.1")
    stream = torch.cuda.current_stream().cuda_stream
    fns = [load(p, cuda) for p in args.cubins]
    g = torch.Generator(device="cuda").manual_seed(0)
    for B in (1, 8, 16, 32):
        V = args.V
        # Audio-LM-like logits: a peaked head over a broad tail.
        logits = (torch.randn(B, V, device="cuda", generator=g) * args.scale)
        logits[:, 128266:128266 + 7 * 4096] += 6.0
        logits = logits.to(torch.bfloat16).contiguous()
        f32 = lambda v: torch.full((B,), v, device="cuda", dtype=torch.float32)
        temp, topp, minp = f32(args.temp), f32(args.top_p), f32(0.0)
        topk = torch.zeros(B, device="cuda", dtype=torch.int32)
        esc = torch.empty(B, V, device="cuda", dtype=torch.float32)
        res = []
        for fn in fns:
            ids = torch.zeros(B, device="cuda", dtype=torch.int32)
            times, draws = [], []
            for it in range(args.iters + 10):
                rng = torch.rand(B, device="cuda", generator=torch.Generator(device="cuda").manual_seed(it))
                a = [ctypes.c_uint64(x.data_ptr()) for x in (logits, ids, temp, topk, topp, minp, rng, esc)]
                a += [ctypes.c_uint32(V), ctypes.c_uint32(B)]
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                launch(cuda, fn, B, a, stream)
                e.record()
                torch.cuda.synchronize()
                if it >= 10:
                    times.append(s.elapsed_time(e) * 1e3)
                    draws.append(ids.clone())
            res.append((statistics.median(times), torch.stack(draws)))
        line = f"B={B:3d} " + "  ".join(f"{p.split('/')[-1]}: {t:8.1f} us" for p, (t, _) in zip(args.cubins, res))
        if len(res) > 1:
            same = (res[0][1] == res[1][1]).float().mean().item()
            line += f"  token agreement {same:.4f}"
        print(line)
    # Distribution: one fixed row, many uniforms, per-cubin token histograms, TVD vs the first.
    V, D = args.V, 20000
    logits = (torch.randn(1, V, device="cuda", generator=g) * args.scale)
    logits[:, 128266:128266 + 7 * 4096] += 6.0
    logits = logits.to(torch.bfloat16).contiguous()
    f32 = lambda v: torch.full((1,), v, device="cuda", dtype=torch.float32)
    temp, topp, minp = f32(args.temp), f32(args.top_p), f32(0.0)
    topk = torch.zeros(1, device="cuda", dtype=torch.int32)
    esc = torch.empty(1, V, device="cuda", dtype=torch.float32)
    hists = []
    for fn in fns:
        ids = torch.zeros(1, device="cuda", dtype=torch.int32)
        us = torch.rand(D, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
        out = []
        for d in range(D):
            rng = us[d:d + 1]
            a = [ctypes.c_uint64(x.data_ptr()) for x in (logits, ids, temp, topk, topp, minp, rng, esc)]
            a += [ctypes.c_uint32(V), ctypes.c_uint32(1)]
            launch(cuda, fn, 1, a, stream)
            out.append(ids.clone())
        torch.cuda.synchronize()
        hists.append(torch.bincount(torch.cat(out).long(), minlength=V).float() / D)
    # Exact kept distribution (the sampler contract): e = exp((l - max)/t), floor = largest v with
    # kept mass > top_p * total, p = e[e >= floor] / sum.
    l = logits[0].float()
    e = torch.exp((l - l.max()) / args.temp)
    srt = torch.sort(e, descending=True).values
    csum = torch.cumsum(srt, 0)
    cut = int(torch.searchsorted(csum, args.top_p * e.sum()).item())
    floor = srt[min(cut, len(srt) - 1)]
    exact = torch.where(e >= floor, e, torch.zeros_like(e))
    exact /= exact.sum()
    ref = torch.multinomial(exact, D, replacement=True, generator=torch.Generator(device="cuda").manual_seed(11))
    noise = 0.5 * (torch.bincount(ref, minlength=V).float() / D - exact).abs().sum().item()
    print(f"kept support {(exact > 0).sum().item()} tokens, D={D}; multinomial noise floor TVD {noise:.4f}")
    for p, h in zip(args.cubins, hists):
        print(f"TVD {p.split('/')[-1]} vs exact: {0.5 * (h - exact).abs().sum().item():.4f}")


if __name__ == "__main__":
    main()
