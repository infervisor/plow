"""Triton versions of the 12 Hz codec blocks (codec_blocks.py): tf32x3 dots, the same shapes and semantics as
codec_blocks.py. Batch-invariant: one (BM, BN, BK) per class, the k loop serial."""
import triton
import triton.language as tl


@triton.jit
def _snake_beta(v, a, r):
    s = tl.sin(v * a)
    return v + r * s * s


@triton.jit
def tconv(x, w, bias, apre, apost, y, M, Tout, Tin, CIN: tl.constexpr, COUT: tl.constexpr, KK: tl.constexpr,
          DIL: tl.constexpr, PAD: tl.constexpr, PRE: tl.constexpr, POST: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    b = rm // Tout
    t = rm % Tout
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, KK, BK):
        kk = k0 + tl.arange(0, BK)
        tap = kk // CIN
        ci = kk % CIN
        u = t[:, None] + tap[None, :] * DIL - PAD
        am = (rm[:, None] < M) & (kk[None, :] < KK) & (u >= 0)
        a = tl.load(x + (b[:, None] * Tin + u) * CIN + ci[None, :], mask=am, other=0.0)
        if PRE:
            al = tl.load(apre + ci, mask=kk < KK, other=0.0)
            rr = tl.load(apre + CIN + ci, mask=kk < KK, other=0.0)
            a = _snake_beta(a, al[None, :], rr[None, :])
        wm = (rn[None, :] < COUT) & (kk[:, None] < KK)
        wv = tl.load(w + rn[None, :] * KK + kk[:, None], mask=wm, other=0.0)
        acc = tl.dot(a, wv, acc, input_precision="tf32x3")
    nm = rn < COUT
    v = acc + tl.load(bias + rn, mask=nm, other=0.0)[None, :]
    if POST:
        al = tl.load(apost + rn, mask=nm, other=0.0)
        rr = tl.load(apost + COUT + rn, mask=nm, other=0.0)
        v = _snake_beta(v, al[None, :], rr[None, :])
    tl.store(y + rm[:, None] * COUT + rn[None, :], v, mask=(rm[:, None] < M) & nm[None, :])


@triton.jit
def tdense(x, w, part, M, N: tl.constexpr, K: tl.constexpr, CHUNK: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    p = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, CHUNK, BK):
        kk = p * CHUNK + k0 + tl.arange(0, BK)
        a = tl.load(x + rm[:, None] * K + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        wv = tl.load(w + rn[None, :] * K + kk[:, None], mask=(rn[None, :] < N) & (kk[:, None] < K), other=0.0)
        acc = tl.dot(a, wv, acc, input_precision="tf32x3")
    tl.store(part + (p * M + rm[:, None]) * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def tfold(part, bias, y, MN, N: tl.constexpr, P: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < MN
    acc = tl.load(part + i, mask=m, other=0.0)
    for q in tl.static_range(1, P):
        acc += tl.load(part + q * MN + i, mask=m, other=0.0)
    tl.store(y + i, acc + tl.load(bias + i % N, mask=m, other=0.0), mask=m)
