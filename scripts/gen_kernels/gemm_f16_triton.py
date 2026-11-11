"""Triton f16 x f16 -> f32 projection C[m][n] = sum_k X[m][k] * W[n][k] (the arithmetic of
gemm_f16.py): one program per (BLOCK_M row tile, BLOCK_N output tile), K in BLOCK_K steps."""
import triton
import triton.language as tl


@triton.jit
def gemm_f16(x_ptr, w_ptr, c_ptr, M, N, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_K: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(w_ptr + rk[:, None] + rn[None, :] * K, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def launcher(cfg, M, N, K, x, w, out):
    grid = (triton.cdiv(M, cfg["bm"]), triton.cdiv(N, cfg["bn"]))
    return lambda: gemm_f16[grid](x, w, out, M, N, K, cfg["bm"], cfg["bn"], cfg["bk"],
                                  num_warps=cfg["warps"], num_stages=cfg["stages"])
