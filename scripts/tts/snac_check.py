#!/usr/bin/env python3
"""Numerics + latency harness for libplow_snac.so (native SNAC-24kHz decoder).

  perf-data/tools/gpulease -n 1 snac-check env HF_HOME=/root/tts-work/hf \
      /root/tts-work/venv-ref/bin/python scripts/tts/snac_check.py \
      --lib /root/tts-work/snac/libplow_snac.so --weights /root/tts-work/snac/snac24k.bin

(a) numerics vs PyTorch SNAC fp32 (TF32 disabled in torch, NoiseBlock patched to
    identity, native seed 0) for B in {1,4}, F in {4,16,64}. GATE: the default 3xTF32 mode
    (fp32-accurate) must reach rel-L2 < 1e-5. The optional 1-pass TF32 mode is reported for
    information only (it measures ~1.0-1.4e-3, i.e. it would NOT pass a 1e-3 TF32 gate).
    Also checks graph == direct launch bit-exactly and noise determinism.
(b) latency: median of 50 CUDA-event-timed decodes after warmup, noise ON everywhere.
"""
import argparse, ctypes, os, statistics, sys

import torch

SHAPES_NUM = [(b, f) for b in (1, 4) for f in (4, 16, 64)]
SHAPES_LAT = [(1, 4), (1, 8), (8, 4), (32, 4), (64, 4), (1, 64), (8, 64)]


class Native:
    def __init__(self, lib, weights, max_batch, max_frames, tf32, graph=True):
        os.environ["PLOW_SNAC_PREC"] = "tf32" if tf32 else "3xtf32"
        os.environ["PLOW_SNAC_GRAPH"] = "1" if graph else "0"
        self.lib = ctypes.CDLL(lib)
        self.lib.plow_snac_create.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                              ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.plow_snac_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_int, ctypes.c_void_p, ctypes.c_ulonglong,
                                              ctypes.c_void_p]
        self.lib.plow_snac_destroy.argtypes = [ctypes.c_void_p]
        self.lib.plow_snac_decode_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                   ctypes.c_int, ctypes.c_void_p, ctypes.c_ulonglong]
        h = ctypes.c_void_p()
        rc = self.lib.plow_snac_create(torch.cuda.current_device(), weights.encode(), max_batch,
                                       max_frames, ctypes.byref(h))
        assert rc == 0, f"create rc={rc}"
        self.h = h

    def decode(self, codes, seed=0, out=None):
        B, F, _ = codes.shape
        if out is None:
            out = torch.empty(B, F * 2048, device=codes.device, dtype=torch.float32)
        rc = self.lib.plow_snac_decode(self.h, codes.data_ptr(), B, F, out.data_ptr(), seed,
                                       torch.cuda.current_stream().cuda_stream)
        assert rc == 0, f"decode rc={rc}"
        return out

    def decode_host(self, codes_np, seed=0, out_np=None):
        import numpy as np
        B, F, _ = codes_np.shape
        codes_np = np.ascontiguousarray(codes_np, dtype=np.int32)
        if out_np is None:
            out_np = np.empty((B, F * 2048), dtype=np.float32)
        rc = self.lib.plow_snac_decode_host(self.h, codes_np.ctypes.data, B, F, out_np.ctypes.data, seed)
        assert rc == 0, f"decode_host rc={rc}"
        return out_np

    def close(self):
        self.lib.plow_snac_destroy(self.h)


def ref_codes(codes):
    c = codes.long()
    l0 = c[:, :, 0]
    l1 = torch.stack([c[:, :, 1], c[:, :, 4]], -1).flatten(1)
    l2 = torch.stack([c[:, :, 2], c[:, :, 3], c[:, :, 5], c[:, :, 6]], -1).flatten(1)
    return [l0, l1, l2]


def time_fn(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()
    from snac import SNAC
    from snac import layers as snac_layers

    dev = torch.device("cuda")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(dev)
    g = torch.Generator(device=dev).manual_seed(1234)
    fp32 = Native(args.lib, args.weights, 64, 64, tf32=False)
    tf32 = Native(args.lib, args.weights, 64, 64, tf32=True)
    nog = Native(args.lib, args.weights, 8, 16, tf32=False, graph=False)

    # ---------------- (a) numerics ----------------
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    orig_noise = snac_layers.NoiseBlock.forward
    snac_layers.NoiseBlock.forward = lambda self, x: x + 0.0 * x
    print("== numerics (noise off both sides; torch fp32, TF32 disabled) ==")
    print(f"{'B':>3} {'F':>4} | {'3xtf32 maxabs':>13} {'3xtf32 relL2':>12} | {'tf32 maxabs':>11} {'tf32 relL2':>10}")
    ok = True
    for B, F in SHAPES_NUM:
        codes = torch.randint(0, 4096, (B, F, 7), device=dev, dtype=torch.int32, generator=g)
        with torch.no_grad():
            ref = model.decode(ref_codes(codes))[:, 0]
        row = []
        for nat, gated in ((fp32, True), (tf32, False)):
            out = nat.decode(codes)
            torch.cuda.synchronize()
            d = (out - ref).double()
            mx = d.abs().max().item()
            rel = (d.norm() / ref.double().norm()).item()
            if gated:
                ok &= rel < 1e-5
            row += [mx, rel]
        print(f"{B:>3} {F:>4} | {row[0]:13.3e} {row[1]:12.3e} | {row[2]:11.3e} {row[3]:10.3e}")
    # direct-launch (no graph) path must match the graph path bit-exactly
    codes = torch.randint(0, 4096, (4, 16, 7), device=dev, dtype=torch.int32, generator=g)
    a, b = fp32.decode(codes, seed=7), nog.decode(codes, seed=7)
    torch.cuda.synchronize()
    same = torch.equal(a, b)
    ok &= same
    print(f"graph vs direct launch (noise on) bit-identical: {same}")
    # noise: deterministic per seed; magnitude comparable to the reference's noise effect
    a2 = fp32.decode(codes, seed=7)
    c = fp32.decode(codes, seed=8)
    z = fp32.decode(codes, seed=0)
    torch.cuda.synchronize()
    snac_layers.NoiseBlock.forward = orig_noise
    with torch.no_grad():
        rn = model.decode(ref_codes(codes))[:, 0]
    r0 = z  # native no-noise ~= reference no-noise
    print(f"noise: same seed identical={torch.equal(a, a2)}, seeds differ={not torch.equal(a, c)}; "
          f"rms(noise effect) native={(a - z).pow(2).mean().sqrt().item():.4f} "
          f"torch={(rn - r0).pow(2).mean().sqrt().item():.4f} (signal rms {z.pow(2).mean().sqrt().item():.4f})")
    ok &= torch.equal(a, a2)
    # host-memory entry (internal stream + pinned staging) == device-pointer path, bit-exact
    same_host = True
    for B, F, seed in ((1, 4, 0), (8, 4, 5), (64, 4, 9), (4, 16, 0)):
        c = torch.randint(0, 4096, (B, F, 7), device=dev, dtype=torch.int32, generator=g)
        dv = fp32.decode(c, seed=seed)
        torch.cuda.synchronize()
        hv = fp32.decode_host(c.cpu().numpy(), seed=seed)
        same_host &= torch.equal(dv.cpu(), torch.from_numpy(hv))
    ok &= same_host
    print(f"decode_host vs device-pointer decode bit-identical: {same_host}")
    print("NUMERICS GATE:", "PASS" if ok else "FAIL", "(default 3xtf32: relL2 < 1e-5; tf32 column informational)")
    if args.skip_latency:
        return 0 if ok else 1

    # ---------------- (b) latency ----------------
    torch.backends.cudnn.allow_tf32 = True  # torch defaults
    torch.backends.cuda.matmul.allow_tf32 = False
    print("\n== latency, ms (median of 50, CUDA events, noise on) ==")
    hdr = f"{'B':>3} {'F':>3} | {'nat 3xtf32':>11} {'native tf32':>11} {'3xtf32 nograph':>16} | {'torch eager':>11} {'torch cudagraph':>15} | {'x eager(3x/1x)':>15}"
    print(hdr)
    for B, F in SHAPES_LAT:
        codes = torch.randint(0, 4096, (B, F, 7), device=dev, dtype=torch.int32, generator=g)
        out = torch.empty(B, F * 2048, device=dev)
        t_f = time_fn(lambda: fp32.decode(codes, 1, out))
        t_t = time_fn(lambda: tf32.decode(codes, 1, out))
        t_ng = time_fn(lambda: nog.decode(codes, 1, out)) if B <= 8 and F <= 16 else float("nan")
        rc = ref_codes(codes)
        with torch.no_grad():
            t_e = time_fn(lambda: model.decode(rc))
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        model.decode(rc)
                torch.cuda.current_stream().wait_stream(s)
                cg = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cg):
                    model.decode(rc)
                t_g = time_fn(cg.replay)
                del cg
            except Exception as e:  # noqa
                print("  torch cudagraph failed:", e)
                t_g = float("nan")
        print(f"{B:>3} {F:>3} | {t_f:11.3f} {t_t:11.3f} {t_ng:16.3f} | {t_e:11.3f} {t_g:15.3f} | {t_e / t_f:6.1f}x /{t_e / t_t:5.1f}x")
    print("\n== plow_snac_decode_host latency (3xtf32, wall clock incl. H2D/D2H + sync, median of 50) ==")
    import time
    for B, F in ((1, 4), (8, 4), (32, 4), (64, 4)):
        c = torch.randint(0, 4096, (B, F, 7), dtype=torch.int32).numpy()
        out = None
        for _ in range(10):
            out = fp32.decode_host(c, 1, out)
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            fp32.decode_host(c, 1, out)
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"{B:>3} {F:>3} | {statistics.median(ts):7.3f} ms")
    for n in (fp32, tf32, nog):
        n.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
