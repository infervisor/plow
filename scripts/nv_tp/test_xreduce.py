"""TP collectives of the NVIDIA interpreter (runtime/nvidia/op_collective.cuh) on 2..N GPUs:
bit-exact against the r = 0..N-1 f32 sum, and the bus bandwidth of the prefill two-shot.

  perf-data/tools/gpulease -n 4 nv-xreduce <venv-python> scripts/nv_tp/test_xreduce.py <xreduce_test.cubin> [n_gpu]
"""
import ctypes
import sys
import time

import torch

_cu = ctypes.CDLL("libcuda.so.1")
STRIDE = 128  # PLOW_CTR_STRIDE bytes


def chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: CUDA error {rc}")


def main():
    path = sys.argv[1]
    n_gpu = int(sys.argv[2]) if len(sys.argv) > 2 else torch.cuda.device_count()
    image = ctypes.create_string_buffer(open(path, "rb").read())
    fns, ctxs = [], []
    for d in range(n_gpu):
        with torch.cuda.device(d):
            torch.empty(1, device=f"cuda:{d}")
            ctx = ctypes.c_void_p()
            chk(_cu.cuCtxGetCurrent(ctypes.byref(ctx)), "ctx")
            ctxs.append(ctx)
            mod, f = ctypes.c_void_p(), ctypes.c_void_p()
            chk(_cu.cuModuleLoadData(ctypes.byref(mod), image), "load")
            chk(_cu.cuModuleGetFunction(ctypes.byref(f), mod, b"t_xreduce"), "fn")
            fns.append(f)
    for a in range(n_gpu):
        chk(_cu.cuCtxSetCurrent(ctxs[a]), "set")
        for b in range(n_gpu):
            if a != b:
                rc = _cu.cuCtxEnablePeerAccess(ctxs[b], 0)
                if rc not in (0, 704):
                    chk(rc, "peer")

    ok = True
    for two in (0, 1):
        for n in (5120, 1000 * 5120 + 24, 1024 * 5120, 4096 * 5120, 8192 * 5120):
            reps = 50
            xbytes = 2 * (reps + 1) * STRIDE
            regions, tables, outs, parts = [], [], [], []
            for d in range(n_gpu):
                dev = f"cuda:{d}"
                regions.append(torch.zeros(n * 2 + xbytes, dtype=torch.uint8, device=dev))
                parts.append((torch.randn(n, device=dev) * (d + 1)).bfloat16())
                outs.append(torch.empty(n, dtype=torch.bfloat16, device=dev))
            for d in range(n_gpu):
                tables.append(torch.tensor([r.data_ptr() for r in regions], dtype=torch.int64, device=f"cuda:{d}"))
            ref = torch.zeros(n, dtype=torch.float32, device="cuda:0")
            for d in range(n_gpu):
                ref += parts[d].to("cuda:0").float()
            ref = ref.bfloat16()

            def run(it, reps=1):
                for d in range(n_gpu):
                    with torch.cuda.device(d):
                        vals = [ctypes.c_uint64(outs[d].data_ptr()), ctypes.c_uint64(tables[d].data_ptr()),
                                ctypes.c_uint64(regions[d].data_ptr() + n * 2), ctypes.c_uint(d), ctypes.c_uint(n_gpu),
                                ctypes.c_uint(n), ctypes.c_uint(0), ctypes.c_uint(2 * it), ctypes.c_uint(2 * it + 1),
                                ctypes.c_uint(two), ctypes.c_uint(reps)]
                        ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
                        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
                        chk(_cu.cuLaunchKernel(fns[d], 132, 1, 1, 256, 1, 1, 0, stream, ptrs, None), "launch")

            def publish():
                for d in range(n_gpu):
                    regions[d][: n * 2].copy_(parts[d].view(torch.uint8))
                for d in range(n_gpu):
                    torch.cuda.synchronize(d)

            publish()
            run(0)
            for d in range(n_gpu):
                torch.cuda.synchronize(d)
            good = all(torch.equal(outs[d].to("cuda:0"), ref) for d in range(n_gpu))
            ok &= good
            # timing: one launch of `reps` back-to-back collectives (values drift; only the clock matters)
            t = time.perf_counter()
            run(1, reps)
            for d in range(n_gpu):
                torch.cuda.synchronize(d)
            us = (time.perf_counter() - t) / reps * 1e6
            msg = n * 2
            bus = msg * 2 * (n_gpu - 1) / n_gpu / us / 1e3  # nccl-style bus bandwidth, GB/s
            print(f"[{'PASS' if good else 'FAIL'}] {'twoshot' if two else 'oneshot'} n_gpu={n_gpu} n={n}: "
                  f"{us:.1f} us/op, busbw {bus:.0f} GB/s", flush=True)
    print("ALL PASS" if ok else "FAILURES")


main()
