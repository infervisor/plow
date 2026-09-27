"""GemmFp8Mx (op 198) role object in a real packet: the bf16-decode arm (i6 = 0, x = act_quant's
fake-quant bf16) and the fp8 arm (i6 = 1, x = e4m3 + t4 ue8m0 scales) against kernel.py fp8_gemm.

  PACKET_OP=<packet_op> FP8MX_CUBIN=<interp_sm90a_pfgemm_fp8mx.cubin> \\
  perf-data/tools/gpulease -n 1 fp8mx <venv-python> scripts/dsv41_nv/test_gemm_fp8mx.py [--fast] [shape filter]
"""
import os
import subprocess
import sys
import tempfile

import torch

REF = os.environ.get(
    "DSV41_REF",
    "/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/inference",
)
sys.path.insert(0, REF)
import kernel as kr  # noqa: E402

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)
dev = "cuda"
PACKET_OP = os.environ["PACKET_OP"]
OBJ = os.environ["FP8MX_CUBIN"]
tmp = tempfile.mkdtemp(prefix="fp8mx_")
FAST = "--fast" in sys.argv
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]
PEAK = 1979.0  # H100/H200 SXM dense fp8, TFLOP/s
results = []


def run(ints, tensors, outs, n_cu=132, iters=5):
    cmd = [PACKET_OP, "--object", OBJ, "--symbol", "plow_sm90a_pfgemm_fp8mx", "--block", "384", "--arena-symbol",
           "plow_arena_bytes_pfgemm_fp8mx", "--op", "GemmFp8Mx", "--n-cu", str(n_cu), "--i", ",".join(map(str, ints)), "--iters", str(iters)]
    for k, (name, nbytes, t) in enumerate(tensors):
        spec = f"{name}:{nbytes}"
        if t is not None:
            f = os.path.join(tmp, f"in{k}.bin")
            t.contiguous().cpu().view(torch.uint8).numpy().tofile(f)
            spec += f":{f}"
        cmd += ["--t", spec]
    files = []
    for k, (name, dt, shape) in enumerate(outs):
        f = os.path.join(tmp, f"out{k}.bin")
        files.append((f, dt, shape))
        cmd += ["--out", f"{name}={f}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    us = float(next(l.split()[1] for l in r.stdout.splitlines() if l.startswith("median_us")))
    got = []
    for f, dt, shape in files:
        raw = torch.frombuffer(bytearray(open(f, "rb").read()), dtype=torch.uint8)
        n = 1
        for d in shape:
            n *= d
        got.append(raw.view(dt)[:n].view(*shape).to(dev))
    return got, us


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# (name, T, N, K, groups, row0, total)
SHAPES = [
    ("q_a 1k", 1024, 1280, 5120, 1, 0, 1024),
    ("q_b tp4 1k", 1024, 16384, 1280, 1, 0, 1024),
    ("wkv 1k", 1024, 512, 5120, 1, 0, 1024),
    ("sh_w1 1k", 1024, 2304, 5120, 1, 0, 1024),
    ("sh_w2 1k", 1024, 5120, 2304, 1, 0, 1024),
    ("idx_wq_b 1k", 1024, 4096, 1280, 1, 0, 1024),
    ("q_b tp4 4k", 4096, 16384, 1280, 1, 0, 4096),
    ("sh_w1 4k", 4096, 2304, 5120, 1, 0, 4096),
    ("sh_w2 4k", 4096, 5120, 2304, 1, 0, 4096),
    ("q_a 8k", 8192, 1280, 5120, 1, 0, 8192),
    ("q_b tp4 8k", 8192, 16384, 1280, 1, 0, 8192),
    ("wo_a G2 1k", 1024, 1024, 4096, 2, 0, 1024),
    ("band 300 of 1k", 300, 1024, 4096, 2, 100, 1024),
    ("tail 129", 129, 512, 1024, 1, 0, 129),
]
if FAST:
    SHAPES = [s for s in SHAPES if s[0] in ("q_b tp4 1k", "sh_w1 1k", "wo_a G2 1k", "band 300 of 1k", "tail 129")]
if ONLY:
    SHAPES = [s for s in SHAPES if any(o in s[0] for o in ONLY)]

# FP8MX_AB=<cubin>,...: extra objects timed on the fp8 arm in the same lease, interleaved (A/B)
AB = [o for o in os.environ.get("FP8MX_AB", "").split(",") if o]
print(f"{'shape':16s} {'T':>5s} {'N':>6s} {'K':>5s} {'G':>2s} | {'bf16 us':>8s} {'TF':>5s} {'rel':>8s} | {'fp8 us':>8s} {'TF':>5s} {'%pk':>4s} {'rel':>8s}"
      + "".join(f" | ab{i} us" for i in range(len(AB))))
for name, T, N, K, G, row0, total in SHAPES:
    x = torch.randn(total, G * K, device=dev)
    w = (torch.randn(G * N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
    ws = torch.randint(118, 124, (G * N // 32, K // 32), device=dev, dtype=torch.uint8)
    xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    fq = (xq.float() * xs.float().repeat_interleave(32, 1)).bfloat16()
    ref = torch.zeros(total, G * N, device=dev)
    for g in range(G):
        ref[row0:row0 + T, g * N:(g + 1) * N] = kr.fp8_gemm(
            xq[row0:row0 + T, g * K:(g + 1) * K].contiguous(), xs[row0:row0 + T, g * K // 32:(g + 1) * K // 32].contiguous(),
            w[g * N:(g + 1) * N].contiguous(), ws[g * N // 32:(g + 1) * N // 32].contiguous().view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
    flops = 2 * T * N * K * G
    common = [("out", total * G * N * 2, None)]
    wt = [("w", G * N * K, w.view(torch.uint8)), ("scale", ws.numel(), ws)]
    (o16,), us16 = run([T, N, K, G, row0, row0], common + [("x", fq.numel() * 2, fq)] + wt, [("out", torch.bfloat16, (total, G * N))])
    (o8,), us8 = run([T, N, K, G, row0, row0, 1], common + [("x", xq.numel(), xq.view(torch.uint8))] + wt + [("xs", xs.numel(), xs.view(torch.uint8))],
                     [("out", torch.bfloat16, (total, G * N))])
    band = slice(row0, row0 + T)
    r16, r8 = rel(o16[band], ref[band]), rel(o8[band], ref[band])
    untouched = total == T or bool((o8[:row0] == 0).all() and (o8[row0 + T:] == 0).all())
    ok = r16 < 3e-3 and r8 < 3e-3 and untouched
    results.append(ok)
    ab = []
    for o in AB:
        OBJ, main = o, OBJ
        (oab,), usab = run([T, N, K, G, row0, row0, 1], common + [("x", xq.numel(), xq.view(torch.uint8))] + wt + [("xs", xs.numel(), xs.view(torch.uint8))],
                           [("out", torch.bfloat16, (total, G * N))])
        OBJ = main
        ab.append(f" | {usab:6.1f}{'' if rel(oab[band], ref[band]) < 3e-3 else '!'}")
    tf16, tf8 = flops / us16 / 1e6, flops / us8 / 1e6
    print(f"{name:16s} {T:5d} {N:6d} {K:5d} {G:2d} | {us16:8.1f} {tf16:5.0f} {r16:8.1e} | {us8:8.1f} {tf8:5.0f} {100 * tf8 / PEAK:4.0f} {r8:8.1e} "
          f"{'PASS' if ok else 'FAIL'}" + "".join(ab), flush=True)
print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
