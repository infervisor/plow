"""12 Hz codec blocks for gen_search (spec_codec_*.py): TileLang, f32-accurate 3xTF32 (hi/lo split,
lo*lo dropped), batch-invariant schedules (one tile config per shape class, k order fixed by bk).

Blocks
  conv  : causal Conv1d k taps, dilation d (pad = d*(k-1) before), tap-major weights [cout][k][cin]
          pre-split hi/lo on the host (as plow's bit-18 image); optional SnakeBeta prologue
          (input, [a | r] per cin) and/or SnakeBeta epilogue ([a | r] per cout).
  dense : y = x @ W^T + b, M small; fixed split-K parts P (partials [P][M][N], ordered fold).
"""
import os
import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}
TARGET = {"kind": "cuda", "arch": "sm_90a"}
f32 = "float32"


def _hi(x):
    u = T.reinterpret("uint32", x)
    return T.reinterpret(f32, (u + T.uint32(0x1000)) & T.uint32(0xFFFFE000))


def _snake_beta(v, a, r):
    s = T.sin(v * a)
    return v + r * s * s


def build_conv(B, Tin, cin, cout, k, dil, bm, bn, bk, threads, stages, pre=False, post=False):
    pad = dil * (k - 1)
    Tout = Tin
    M, KK = B * Tout, k * cin
    G = T.gemm

    @T.prim_func
    def conv(x: T.Tensor((B, Tin, cin), f32), wh: T.Tensor((cout, KK), f32), wl: T.Tensor((cout, KK), f32),
             bias: T.Tensor((cout,), f32), apre: T.Tensor((2 * cin,), f32), apost: T.Tensor((2 * cout,), f32),
             y: T.Tensor((B, Tout, cout), f32)):
        with T.Kernel(T.ceildiv(M, bm), T.ceildiv(cout, bn), threads=threads) as (mt, nt):
            Ah = T.alloc_shared((bm, bk), f32)
            Al = T.alloc_shared((bm, bk), f32)
            Bh = T.alloc_shared((bn, bk), f32)
            Bl = T.alloc_shared((bn, bk), f32)
            C = T.alloc_fragment((bm, bn), f32)
            T.clear(C)
            for kt in T.Pipelined(T.ceildiv(KK, bk), num_stages=stages):
                for i, j in T.Parallel(bm, bk):
                    r = mt * bm + i
                    kk = kt * bk + j
                    tap = kk // cin
                    ci = kk % cin
                    rr = T.min(r, M - 1)
                    b = rr // Tout
                    t = rr % Tout
                    u = t + tap * dil - pad
                    v = T.if_then_else((r < M) & (kk < KK) & (u >= 0), x[b, T.max(u, 0), T.min(ci, cin - 1)], T.float32(0))
                    if pre:
                        v = _snake_beta(v, apre[T.min(ci, cin - 1)], apre[cin + T.min(ci, cin - 1)])
                    Ah[i, j] = _hi(v)
                    Al[i, j] = v - _hi(v)
                T.copy(wh[nt * bn, kt * bk], Bh)
                T.copy(wl[nt * bn, kt * bk], Bl)
                G(Al, Bh, C, transpose_B=True)
                G(Ah, Bl, C, transpose_B=True)
                G(Ah, Bh, C, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                r = mt * bm + i
                n = nt * bn + j
                if (r < M) & (n < cout):
                    v = C[i, j] + bias[n]
                    if post:
                        v = _snake_beta(v, apost[n], apost[cout + n])
                    y[r // Tout, r % Tout, n] = v
    return conv


def build_dense(M, N, K, bm, bn, bk, threads, stages, P):
    chunk = -(-K // bk // P) * bk
    G = T.gemm

    @T.prim_func
    def dense(x: T.Tensor((M, K), f32), wh: T.Tensor((N, K), f32), wl: T.Tensor((N, K), f32),
              part: T.Tensor((P, M, N), f32)):
        with T.Kernel(T.ceildiv(M, bm), T.ceildiv(N, bn), P, threads=threads) as (mt, nt, p):
            Ah = T.alloc_shared((bm, bk), f32)
            Al = T.alloc_shared((bm, bk), f32)
            Bh = T.alloc_shared((bn, bk), f32)
            Bl = T.alloc_shared((bn, bk), f32)
            C = T.alloc_fragment((bm, bn), f32)
            T.clear(C)
            for kt in T.Pipelined(chunk // bk, num_stages=stages):
                for i, j in T.Parallel(bm, bk):
                    r = mt * bm + i
                    kk = p * chunk + kt * bk + j
                    v = T.if_then_else((r < M) & (kk < K), x[T.min(r, M - 1), T.min(kk, K - 1)], T.float32(0))
                    Ah[i, j] = _hi(v)
                    Al[i, j] = v - _hi(v)
                for i, j in T.Parallel(bn, bk):
                    kk = p * chunk + kt * bk + j
                    ok = kk < K
                    Bh[i, j] = T.if_then_else(ok, wh[nt * bn + i, T.min(kk, K - 1)], T.float32(0))
                    Bl[i, j] = T.if_then_else(ok, wl[nt * bn + i, T.min(kk, K - 1)], T.float32(0))
                G(Al, Bh, C, transpose_B=True)
                G(Ah, Bl, C, transpose_B=True)
                G(Ah, Bh, C, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                r = mt * bm + i
                if r < M:
                    part[p, r, nt * bn + j] = C[i, j]

    @T.prim_func
    def fold(part: T.Tensor((P, M, N), f32), bias: T.Tensor((N,), f32), y: T.Tensor((M, N), f32)):
        with T.Kernel(T.ceildiv(M * N, 256), threads=256) as bx:
            for e in T.Parallel(256):
                idx = bx * 256 + e
                if idx < M * N:
                    r = idx // N
                    n = idx % N
                    acc = T.alloc_var(f32)
                    acc = part[0, r, n]
                    for q in T.serial(P - 1):
                        acc = acc + part[q + 1, r, n]
                    y[r, n] = acc + bias[n]
    return dense, fold


def compile(fn):
    return tilelang.compile(fn, target=TARGET, pass_configs=PASS_CONFIGS)
