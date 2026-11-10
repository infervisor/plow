"""TileLang f16 x f16 -> f32 projection C[m][n] = sum_k X[m][k] * W[n][k] (W row-major by output,
as plow's GemvF32Out mma arm streams it). One block per (BN output tile, BM row tile); static
shapes, K pipelined in BK steps."""
import os

import tilelang
import tilelang.language as T

PASS_CONFIGS = {} if os.environ.get("TL_WS") else {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def compile(M, N, K, bm, bn, bk, threads, stages):
    f16, f32 = "float16", "float32"

    @T.prim_func
    def gemm_f16(X: T.Tensor((M, K), f16), W: T.Tensor((N, K), f16), C: T.Tensor((M, N), f32)):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            A = T.alloc_shared((bm, bk), f16)
            B = T.alloc_shared((bn, bk), f16)
            Cl = T.alloc_fragment((bm, bn), f32)
            T.clear(Cl)
            for kt in T.Pipelined(T.ceildiv(K, bk), num_stages=stages):
                T.copy(X[by * bm, kt * bk], A)
                T.copy(W[bx * bn, kt * bk], B)
                T.gemm(A, B, Cl, transpose_B=True)
            T.copy(Cl, C[by * bm, bx * bn])

    return tilelang.compile(gemm_f16, out_idx=None, target={"kind": "cuda", "arch": "sm_90a"},
                            pass_configs=PASS_CONFIGS)
