# V4.1 compressor decode step + batched CompressRopeQuant / RopeInverseO (op_compress.cuh).
# Usage: python probe_compress_decode.py <packet_ops_test.cubin>  (under gpulease)
import sys, ctypes, torch
sys.path.insert(0, "/root/plow/.claude/worktrees/dsv41-on-main/scripts/dsv41_nv")
from cudrv import Cubin, i32
P = Cubin(sys.argv[1])
dev = "cuda"
D, RD, EPS = 512, 64, 1e-20
torch.manual_seed(0)
ok = True
NUL = ctypes.c_void_p(0)


def check(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"[{'PASS' if good else 'FAIL'}] {name} {detail}", flush=True)


# ---- decode step vs model.py Compressor.forward (start_pos > 0) ----
for ratio in (2, 4):
    B = 37
    pos = torch.randint(100, 5000, (B,), device=dev, dtype=torch.int32)
    pos[:4] = torch.tensor([ratio - 1, ratio, 2 * ratio - 1, 7], device=dev)
    st_kv = torch.randn(B, ratio, D, device=dev)
    st_sc = torch.randn(B, ratio, D, device=dev)
    kv = torch.randn(B, D, device=dev)
    sc = torch.randn(B, D, device=dev)
    gamma = (1 + 0.1 * torch.randn(D, device=dev)).bfloat16()
    lat = torch.zeros(B, D, device=dev).bfloat16()
    rk, rs = st_kv.clone(), st_sc.clone()
    ref = lat.clone()
    for b in range(B):
        p = int(pos[b])
        rk[b, p % ratio], rs[b, p % ratio] = kv[b], sc[b]
        if (p + 1) % ratio == 0:
            pooled = (rk[b] * rs[b].softmax(0)).sum(0).bfloat16().float()
            ref[b] = (pooled * torch.rsqrt(pooled.square().mean() + EPS) * gamma.float()).bfloat16()
    P.launch("t_compress_decode_step", (132,), (256,), [lat, st_kv, st_sc, kv, sc, gamma, pos, i32(B), i32(ratio), i32(D), ctypes.c_float(EPS), i32(0)])
    torch.cuda.synchronize()
    fired = ((pos + 1) % ratio == 0)
    bad = (lat.float() - ref.float()).abs().max().item()
    check(f"compress_decode_step ratio={ratio} fired={int(fired.sum())}/{B}", torch.equal(st_kv, rk) and torch.equal(st_sc, rs) and bad <= 1e-2 * ref.float().abs().max().item(),
          f"state_eq={torch.equal(st_kv, rk) and torch.equal(st_sc, rs)} lat_maxerr={bad:.2e}")

# ---- batched CompressRopeQuant vs the legacy single-slot decode path, bitwise ----
maxpos = 8192
ang = torch.rand(maxpos, RD // 2, device=dev) * 6.28
cosb, sinb = ang.cos().contiguous(), ang.sin().contiguous()
for (ratio, qblk, qmode, n_head, stride) in ((2, 16, 2, 1, 4096), (1, 16, 2, 1, 8192), (1, 32, 1, 1, 8192), (1, 32, 1, 32, 0)):
    B = 13
    pos = torch.randint(0, 8000, (B,), device=dev, dtype=torch.int32)
    pos[0] = 0
    src = torch.randn(B, n_head, D, device=dev).bfloat16()
    rows = max(stride, 1)
    out = torch.zeros(B, rows, n_head, D, device=dev).bfloat16() if stride else src.clone()
    exp = out.clone()
    P.launch("t_compress_rope_quant", (132,), (256,), [out, src, cosb, sinb, pos, i32(B), i32(D), i32(RD), i32(qblk), i32(ratio), i32(0), i32(qmode),
                                                     i32(n_head), i32(1), i32(stride), i32(0), NUL])
    for b in range(B):
        p = int(pos[b])
        if (p + 1) % ratio:
            continue
        if stride:  # legacy: out/src indexed by pos/ratio -> stage src at that row of a scratch cache
            cache = exp[b].clone()
            srcc = torch.zeros_like(cache)
            srcc[p // ratio] = src[b]
            P.launch("t_compress_rope_quant", (132,), (256,), [cache, srcc, cosb, sinb, pos[b:b + 1], i32(1), i32(D), i32(RD), i32(qblk), i32(ratio),
                                                             i32(0), i32(qmode), i32(n_head), i32(0), i32(0), i32(0), NUL])
            torch.cuda.synchronize()
            exp[b] = cache
        else:  # indexer query: legacy row_base form, one row at position p
            big = torch.zeros(p + 1, n_head, D, device=dev).bfloat16()
            big[p] = src[b]
            P.launch("t_compress_rope_quant", (132,), (256,), [big, big, cosb, sinb, ctypes.c_void_p(0), i32(1), i32(D), i32(RD), i32(qblk), i32(ratio),
                                                             i32(p), i32(qmode), i32(n_head), i32(0), i32(0), i32(0), NUL])
            torch.cuda.synchronize()
            exp[b] = big[p]
    torch.cuda.synchronize()
    check(f"rope_quant batched ratio={ratio} qblk={qblk} qmode={qmode} heads={n_head} stride={stride}", torch.equal(out, exp),
          f"mismatch={(out != exp).sum().item()}")

# ---- window ring write: ratio 1, stride 128, mask 127, fp8 pow2 / 32 == row p of a flat cache, wrapped ----
B = 9
pos = torch.randint(0, 8000, (B,), device=dev, dtype=torch.int32)
src = torch.randn(B, 1, D, device=dev).bfloat16()
ring = torch.zeros(B, 128, 1, D, device=dev).bfloat16()
P.launch("t_compress_rope_quant", (132,), (256,), [ring, src, cosb, sinb, pos, i32(B), i32(D), i32(RD), i32(32), i32(1), i32(0), i32(0),
                                                 i32(1), i32(1), i32(128), i32(127), NUL])
flat = torch.zeros(B, 8192, 1, D, device=dev).bfloat16()
P.launch("t_compress_rope_quant", (132,), (256,), [flat, src, cosb, sinb, pos, i32(B), i32(D), i32(RD), i32(32), i32(1), i32(0), i32(0),
                                                 i32(1), i32(1), i32(8192), i32(0), NUL])
torch.cuda.synchronize()
good = all(torch.equal(ring[b, int(pos[b]) % 128], flat[b, int(pos[b])]) for b in range(B)) and int((ring.float().abs().sum((2, 3)) > 0).sum()) == B
check("ring write (mask 127)", good)

# ---- prefill seeds: ring rows [L-128, L) at p % 128; compressor tail rows [L - L%r, L) -> state ----
T, L = 300, 261
cpos = torch.arange(T, device=dev, dtype=torch.int32)
kvl = torch.tensor([L], device=dev, dtype=torch.int32)
src = torch.randn(T, 1, D, device=dev).bfloat16()
ring = torch.zeros(128, 1, D, device=dev).bfloat16()
flat = torch.zeros(T, 1, D, device=dev).bfloat16()
P.launch("t_compress_rope_quant", (132,), (256,), [ring, src, cosb, sinb, cpos, i32(T), i32(D), i32(RD), i32(32), i32(1), i32(0), i32(0),
                                                 i32(1), i32(1), i32(0), i32(127), kvl])
P.launch("t_compress_rope_quant", (132,), (256,), [flat, src, cosb, sinb, cpos, i32(T), i32(D), i32(RD), i32(32), i32(1), i32(0), i32(0),
                                                 i32(1), i32(1), i32(0), i32(0), kvl])
torch.cuda.synchronize()
exp = torch.zeros_like(ring)
for p in range(L - 128, L):
    exp[p % 128] = flat[p]
check("ring seed from prefill chunk", torch.equal(ring, exp) and flat[L:].abs().sum().item() == 0, f"mismatch={(ring != exp).sum().item()}")
for ratio in (2, 4):
    kvc, scc = torch.randn(T, D, device=dev), torch.randn(T, D, device=dev)
    sk, ss = torch.zeros(ratio, D, device=dev), torch.zeros(ratio, D, device=dev)
    P.launch("t_compress_decode_step", (132,), (256,), [lat, sk, ss, kvc, scc, gamma, kvl, i32(1), i32(ratio), i32(D), ctypes.c_float(EPS), i32(1)])
    torch.cuda.synchronize()
    tail = L % ratio
    ek, es = torch.zeros_like(sk), torch.zeros_like(ss)
    ek[:tail], es[:tail] = kvc[L - tail:L], scc[L - tail:L]
    check(f"state seed ratio={ratio} tail={tail}", torch.equal(sk, ek) and torch.equal(ss, es))

# ---- RopeInverseO per-row positions vs legacy per token ----
B, H = 11, 16
pos = torch.randint(0, 8000, (B,), device=dev, dtype=torch.int32)
o = torch.randn(B, H, D, device=dev).bfloat16()
exp = o.clone()
for b in range(B):
    row = exp[b:b + 1].clone()
    P.launch("t_rope_inverse_o", (132,), (256,), [row, cosb, sinb, pos[b:b + 1], i32(1), i32(H), i32(D), i32(RD), i32(0)])
    torch.cuda.synchronize()
    exp[b] = row[0]
P.launch("t_rope_inverse_o", (132,), (256,), [o, cosb, sinb, pos, i32(B), i32(H), i32(D), i32(RD), i32(1)])
torch.cuda.synchronize()
check("rope_inverse_o per_row", torch.equal(o, exp), f"mismatch={(o != exp).sum().item()}")
print("ALL PASS" if ok else "SOME FAIL")
