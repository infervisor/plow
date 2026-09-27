"""DeepSeek-V4.1 sparse attention on the sm_90a role object (runtime/nvidia/interp_sm90a_pfflash_v41.cu)
against the reference (inference/kernel.py sparse_attn), inside real packets (packet_op).

  PACKET_OP=... FLASH_CUBIN=interp_sm90a_pfflash_v41.cubin ISA_PF_CUBIN=interp_sm90a_pf.cubin \\
  perf-data/tools/gpulease -n 1 dsv41-flash <venv-python> scripts/dsv41_nv/test_packet_flash.py [--perf] [--interp]

--interp also times the interpreter arm (FlashMlaPrefill on ISA_PF_CUBIN) at the same shapes.
"""
import os
import struct
import subprocess
import sys
import tempfile

import torch

REF = os.environ.get(
    "DSV41_REF",
    "/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/inference",
)
sys.path.insert(0, REF)
torch.set_default_dtype(torch.bfloat16)
dev = "cuda"
PACKET_OP = os.environ["PACKET_OP"]
FLASH = os.environ["FLASH_CUBIN"]
ISA_PF = os.environ["ISA_PF_CUBIN"]
PERF = "--perf" in sys.argv
INTERP = "--interp" in sys.argv
INDEX = "--index" in sys.argv  # only the indexer score (op 117) on the interpreter
tmp = tempfile.mkdtemp(prefix="pktflash_")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def packet_op(obj, symbol, block, arena, op, ints, tensors, outs, n_cu=132, iters=1, floats=()):
    cmd = [PACKET_OP, "--object", obj, "--symbol", symbol, "--block", str(block), "--arena-symbol", arena, "--op", op,
           "--n-cu", str(n_cu), "--i", ",".join(map(str, ints)), "--iters", str(iters)]
    if floats:
        cmd += ["--f", ",".join(repr(float(f)) for f in floats)]
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
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"packet_op failed: {r.stderr[-2000:]}")
    us = float(next(l.split()[1] for l in r.stdout.splitlines() if l.startswith("median_us")))
    got = []
    for f, dt, shape in files:
        raw = torch.frombuffer(bytearray(open(f, "rb").read()), dtype=torch.uint8)
        n = 1
        for d in shape:
            n *= d
        got.append(raw.view(dt)[:n].view(*shape).to(dev))
    return got, us


def flash(obj, *a, **k):
    if obj == "role":
        return packet_op(FLASH, "plow_sm90a_pfflash_v41", 384, "plow_arena_bytes_pfflash_v41", "FlashMlaPrefill", *a, **k)
    return packet_op(ISA_PF, "_Z15interp_sm90a_pf11PlowProgram", 256, "plow_arena_bytes_pf", "FlashMlaPrefill", *a, **k)


def merge(*a, **k):
    return packet_op(ISA_PF, "_Z15interp_sm90a_pf11PlowProgram", 256, "plow_arena_bytes_pf", "FlashMerge", *a, **k)


def union_table(idx, P, cap):
    T = idx.shape[0]
    n_qt = (T + P - 1) // P
    hdr = (n_qt * 4 + 255) // 256 * 256
    ic = idx.cpu()
    cnt = torch.zeros(n_qt, dtype=torch.int32)
    pos = torch.zeros(n_qt, cap, dtype=torch.int32)
    msk = torch.zeros(n_qt, cap, dtype=torch.int64)
    for qt in range(n_qt):
        rows = ic[qt * P:min(T, qt * P + P)]
        keys, inv = torch.unique(rows[rows >= 0], return_inverse=True)
        keys = keys[:cap]
        cnt[qt] = len(keys)
        pos[qt, :len(keys)] = keys.int()
        m = torch.zeros(len(keys), dtype=torch.int64)
        for ql in range(rows.shape[0]):
            r = rows[ql]
            sel = torch.searchsorted(keys, r[r >= 0])
            m[sel] |= 1 << ql
        msk[qt, :len(keys)] = m
    buf = torch.zeros(hdr + n_qt * cap * 12 + 8, dtype=torch.uint8)
    buf[:n_qt * 4] = cnt.view(torch.uint8)
    body = torch.cat([pos, (msk & 0xFFFFFFFF).int(), (msk >> 32).int()], 1)  # [n_qt][3 cap] i32
    buf[hdr:hdr + n_qt * cap * 12] = body.contiguous().view(torch.uint8).flatten()
    return buf


def selections(T, n_cmp, k, seed=0):
    """per-query top-k over compressed rows < (t+1)//4, smooth in t (neighbours overlap, as a real indexer's do)"""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(n_cmp, generator=g)
    idx = torch.full((T, k), -1, dtype=torch.int32)
    for t in range(T):
        n = min((t + 1) // 4, n_cmp)
        m = min(n, k)
        if m:
            sc = base[:n] + 0.15 * torch.rand(n, generator=g)
            idx[t, :m] = sc.topk(m).indices.sort().values.int()
    return idx


import kernel as kr

if INDEX:
    # op 117: f32 score = scale * sum_h w * relu(q . k), pools, causal (t + 1) // 4
    kl = lambda n: torch.tensor([n], dtype=torch.int32, device=dev)
    for T in (1024, 4096, 8192):
        S = T // 4
        q = torch.randn(T, 32, 128, device=dev).bfloat16()
        k = torch.randn(S, 128, device=dev).bfloat16()
        w = torch.randn(T, 32, device=dev).bfloat16()
        scale = 128 ** -0.5 * 32 ** -0.5
        ref = (torch.einsum("thd,sd->ths", q.float(), k.float()).relu() * w.float().unsqueeze(-1)).sum(1) * scale
        valid = torch.arange(S, device=dev).unsqueeze(0) < ((torch.arange(T, device=dev) + 1) // 4).unsqueeze(1)
        ts = [("score", T * S * 4, torch.zeros(T, S, device=dev, dtype=torch.float32)), ("q", q.numel() * 2, q), ("k", k.numel() * 2, k),
              ("w", w.numel() * 2, w), ("kv_len", 4, kl(T))]
        for name, obj, sym, block, arena in (("interp", ISA_PF, "_Z15interp_sm90a_pf11PlowProgram", 256, "plow_arena_bytes_pf"),
                                             ("role", FLASH, "plow_sm90a_pfflash_v41", 384, "plow_arena_bytes_pfflash_v41")):
            (sc,), us = packet_op(obj, sym, block, arena, "IndexScorePf", [T, 32, S, 128, 4], ts, [("score", torch.float32, (T, S))],
                                  floats=(scale,), iters=5)
            err = ((sc - ref) * valid).abs().max().item() / ref.abs().max().item()
            untouched = bool((sc[~valid] == 0).all())
            tf = 2 * valid.sum().item() * 32 * 128 / (us * 1e-6) / 1e12
            check(f"IndexScorePf {name} T={T}", err < 1e-5 and untouched, f"err={err:.1e} bounded={untouched} {us:.1f} us {tf:.0f} TFLOP/s")
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)

torch.manual_seed(0)
D, W = 512, 128
shapes = [(64, 16, 0, 0, 1, 8), (300, 16, 75, 64, 2, 4), (1024, 16, 256, 256, 1, 4), (129, 64, 32, 32, 1, 1), (1024, 64, 256, 256, 1, 1),
          (512, 16, 128, 128, 2, 8), (256, 8, 64, 64, 1, 8)]
if PERF:
    shapes = [(1024, 16, 256, 256, 1, 4), (4096, 16, 1024, 512, 1, 4), (8192, 16, 2048, 512, 1, 4), (4096, 64, 1024, 512, 1, 1),
              (8192, 64, 2048, 512, 1, 1)]
for T, H, n_cmp, k, gsplit, P in shapes:
    q = (torch.randn(T, H, D, device=dev) * 0.5).bfloat16()
    kv = torch.randn(T, D, device=dev).bfloat16()
    sink = torch.randn(H, device=dev, dtype=torch.float32)
    scale = D ** -0.5
    nsplit = 1 + (gsplit if n_cmp else 0)
    kl = torch.tensor([T], dtype=torch.int32, device=dev)
    osz, msz = T * H * nsplit * D * 4, T * H * nsplit * 8
    word = (nsplit << 8) if nsplit > 1 else 0
    it = 5 if PERF else 1
    win_t = [("opart", osz, None), ("ml", msz, None), ("q", q.numel() * 2, q), ("-", 0, None), ("kv", kv.numel() * 2, kv), ("-", 0, None),
             ("kv_len", 4, kl)]
    outs = [("opart", torch.float32, (osz // 4,)), ("ml", torch.float32, (msz // 4,))]
    (op_, ml_), us_w = flash("role", [1, H, T, (1 << 31) | W, T, 0xFFFFFFFF, 0, word], win_t, outs, floats=(scale,), iters=it)
    us_wi = flash("interp", [1, H, T, (1 << 31) | W, T, 0xFFFFFFFF, 0, word], win_t, outs, floats=(scale,), iters=it)[1] if INTERP else 0
    us_g = us_gi = 0.0
    pos = torch.arange(T)
    widx = pos.unsqueeze(1) - torch.arange(W - 1, -1, -1).unsqueeze(0)
    widx = torch.where(widx >= 0, widx, torch.full_like(widx, -1)).int()
    full_kv, full_idx = kv, widx.to(dev)
    nsel = (widx >= 0).sum().item()
    if n_cmp:
        cache = torch.randn(n_cmp, D, device=dev).bfloat16()
        cidx = selections(T, n_cmp, k)
        cap = min(P * k, n_cmp)
        uni = union_table(cidx, P, cap).to(dev)
        n_qt = (T + P - 1) // P
        urows = uni[:n_qt * 4].view(torch.int32).sum().item()
        pbits = struct.unpack("<f", struct.pack("<I", P))[0]  # j0 = union tile, riding f1's slot
        g_t = [("opart", osz, op_), ("ml", msz, ml_), ("q", q.numel() * 2, q), ("-", 0, None), ("cache", cache.numel() * 2, cache),
               ("-", 0, None), ("kv_len", 4, kl), ("uni", uni.numel(), uni)]
        (op_, ml_), us_g = flash("role", [1, H, n_cmp, 1 << 31, T, 0xFFFFFFFF, cap, (gsplit << 16) | (nsplit << 8) | 1], g_t, outs,
                                 floats=(scale, pbits), iters=it)
        if INTERP and P == 8:
            us_gi = flash("interp", [1, H, n_cmp, 1 << 31, T, 0xFFFFFFFF, cap, (gsplit << 16) | (nsplit << 8) | 1], g_t, outs,
                          floats=(scale,), iters=it)[1]
        full_kv = torch.cat([kv, cache])
        full_idx = torch.cat([widx, torch.where(cidx >= 0, cidx + T, cidx)], 1).to(dev)
        nsel = (full_idx >= 0).sum().item()
    (o,), us_m = merge([T, H, nsplit, D, 1], [("o", T * H * D * 2, None), ("opart", osz, op_), ("ml", msz, ml_), ("sink", H * 4, sink)],
                       [("o", torch.bfloat16, (T, H, D))], iters=it)
    ref = kr.sparse_attn(q.unsqueeze(0), full_kv.unsqueeze(0), sink, full_idx.unsqueeze(0).contiguous(), scale).squeeze(0)
    r = ((o.float() - ref.float()).norm() / ref.float().norm()).item()
    useful = 4 * nsel * H * D
    tf = useful / ((us_w + us_g) * 1e-6) / 1e12
    extra = ""
    if n_cmp:
        extra = f" union_rows/query={urows / T:.0f} (waste {urows * P / max(1, (cidx >= 0).sum().item()):.2f}x)"
    if INTERP:
        extra += f" | interp window {us_wi:.1f} us gather {us_gi:.1f} us"
    check(f"role FlashMlaPrefill T={T} H={H} cmp={n_cmp} k={k} P={P} gsplit={gsplit}", r < 5e-3,
          f"rel={r:.2e} window {us_w:.1f} us gather {us_g:.1f} us merge {us_m:.1f} us useful {tf:.0f} TFLOP/s{extra}")
    # FUSED: window (t5) + union (t4) in one pass, sink folded, bf16 O -- no partials, no merge
    if not n_cmp:
        cache = torch.zeros(1, D, device=dev).bfloat16()
        cap = 1
        uni = torch.zeros((((T + P - 1) // P) * 4 + 255) // 256 * 256 + ((T + P - 1) // P) * 12 + 8, dtype=torch.uint8, device=dev)
    pbits = struct.unpack("<f", struct.pack("<I", P))[0]
    f_t = [("o", T * H * D * 2, None), ("-", 0, None), ("q", q.numel() * 2, q), ("sink", H * 4, sink), ("cache", cache.numel() * 2, cache),
           ("kv", kv.numel() * 2, kv), ("kv_len", 4, kl), ("uni", uni.numel(), uni)]
    (of,), us_f = flash("role", [1, H, max(n_cmp, 1), (1 << 31) | W, T, 0xFFFFFFFF, cap, 0], f_t, [("o", torch.bfloat16, (T, H, D))],
                        floats=(scale, pbits), iters=it)
    rf = ((of.float() - ref.float()).norm() / ref.float().norm()).item()
    check(f"role FUSED T={T} H={H} cmp={n_cmp} k={k} P={P}", rf < 5e-3,
          f"rel={rf:.2e} {us_f:.1f} us useful {useful / (us_f * 1e-6) / 1e12:.0f} TFLOP/s (vs {us_w + us_g + us_m:.1f} us window+gather+merge)")

print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
