"""CSA2 compressor oracle: the V4.1 reference against a model of what op 180 computes.

The reference half is transcribed from the checkpoint's own `inference/model.py` (`class
Compressor`, 429-486; `RMSNorm`, 281; `apply_rotary_emb`, 392) and `inference/kernel.py`
(`fp4_act_quant`, 184). The op-180 half is transcribed from `runtime/amd/op_compress.h` --
its header states the math and `cmp_fake_quant_block` states the epilogue.

It answers three questions that doc 12.10 currently answers by READING:
  1. does V4.1's compressor equal op 180's with ape = 0 and coff = 1?
  2. is the per-block scale format the ONLY remaining difference?
  3. what tolerance does a hardware test get to assert?

Checks 7-10 then cover the SPLIT those answers forced. `Compressor.forward` returns the latent
before RoPE, so op 180 gained an arm that stops after the norm (`PLOW_CMP_EPI_NORM`) and the rope
and the quant moved to op 185 (`d_compress_rope_quant`), which is where `_compress_kv` and the
indexer's `wk` both do them. Those checks price the pair against the reference end to end, and
they pin the E4M3 amax floor, which is the one constant in the epilogue that is chosen rather
than conservative.

Run: PYTHONPATH=/workspace/oracle-venv/site python3 csa2_oracle.py
"""
import sys

import torch

torch.manual_seed(0)

D = 64          # head_dim, scaled down from 512; the math is per-channel so width is free
RD = 16         # rope_head_dim, scaled from 64
DIM = 96        # model dim
EPS = 1e-20     # rms_norm_eps, the real one
QBLK = 16       # fp4_act_quant block_size for compressed KV
FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0
FLOOR_E4M3 = 6 * 2.0**-9     # kernel.py:160
FLOOR_POW2 = 6 * 2.0**-126   # kernel.py:163
IDX_QBLK = 32                # fp4_block_size, the indexer's key quant


def apply_rotary_emb(x, cos, sin):
    """model.py:392-405, interleaved (GPT-J) pairs: x viewed as complex over adjacent channels."""
    y = x.float().clone()
    x0, x1 = y[..., 0::2], y[..., 1::2]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    y[..., 0::2], y[..., 1::2] = r0.to(torch.bfloat16).float(), r1.to(torch.bfloat16).float()
    return y.to(torch.bfloat16)


def reference_compress_kv(latent, cos, sin):
    """model.py:751-761: rope the tail at the group's FIRST token, then fp4 at 16 with E4M3."""
    out = latent.float().clone()
    out[..., -RD:] = apply_rotary_emb(latent[..., -RD:], cos, sin).float()
    return fake_quant(out.to(torch.bfloat16), QBLK, "e4m3")


def op185(src, cos, sin, qblk, scale_mode):
    """op_compress.h d_compress_rope_quant: rope the last rd, then fake-quant the WHOLE row."""
    out = src.float().clone()
    out[..., -RD:] = apply_rotary_emb(src[..., -RD:], cos, sin).float()
    return fake_quant(out.to(torch.bfloat16), qblk, scale_mode)


def rmsnorm(x, gamma, eps=EPS):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (gamma * x).to(dt)


def quant_fp4(q):
    """Nearest e2m1 magnitude, sign preserved -- the LUT op_compress.h's cmp_dequant_fp4 uses."""
    s = torch.sign(q)
    a = q.abs().unsqueeze(-1)
    i = (a - FP4_LUT).abs().argmin(-1)
    return s * FP4_LUT[i]


def fake_quant(x, blk, scale_mode):
    """kernel.py fp4_act_quant(inplace=True): quantize then dequantize, back to bf16.

    `scale_mode` is the whole question: op 180 rounds the scale to a POWER OF TWO
    (plow_round_scale); V4.1 asks for an E4M3 scale.
    """
    y = x.float().clone()
    n = y.shape[-1]
    assert n % blk == 0
    v = y.unflatten(-1, (n // blk, blk))
    # The floor is part of the format, not a guard: kernel.py:159-163 uses 6*2**-9 on the E4M3
    # branch and 6*2**-126 on the E8M0 one, and they are not interchangeable -- see check [8].
    floor = FLOOR_E4M3 if scale_mode == "e4m3" else FLOOR_POW2
    amax = v.abs().amax(-1, keepdim=True).clamp_min(floor)
    s = amax / FP4_MAX
    if scale_mode == "pow2":
        s = torch.exp2(torch.ceil(torch.log2(s)))
    elif scale_mode == "e4m3":
        s = s.to(torch.float8_e4m3fn).float()
    else:
        raise ValueError(scale_mode)
    q = quant_fp4((v / s).clamp(-FP4_MAX, FP4_MAX))
    return (q * s).flatten(-2).to(torch.bfloat16)


def reference_compressor(x, wkv, wgate, gamma, ratio):
    """model.py:458-486, prefill (start_pos == 0). Returns the latent BEFORE RoPE."""
    if ratio == 1:
        return rmsnorm(x @ wkv.T.to(x.dtype), gamma)
    xf = x.float()
    kv, score = xf @ wkv.T, xf @ wgate.T
    seqlen = xf.shape[1]
    cutoff = seqlen - seqlen % ratio
    kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
    score = score[:, :cutoff].unflatten(1, (-1, ratio))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
    return rmsnorm(kv.to(x.dtype), gamma)


def op180_compressor(x, wkv, wgate, gamma, ratio, ape=None, coff=1):
    """op_compress.h: out[c] = SUM_s kv[s][c] * softmax_s(score[s][c] + ape[s][c]), then RMSNorm.

    coff == 1 and ape == 0 is the claim under test. The softmax is over the SLOT axis
    independently per channel, which is what the reference's `softmax(dim=2)` also is.
    """
    assert coff == 1, "V4.1 has no overlap transform"
    xf = x.float()
    kv, score = xf @ wkv.T, xf @ wgate.T
    seqlen = xf.shape[1]
    cutoff = seqlen - seqlen % ratio
    kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
    score = score[:, :cutoff].unflatten(1, (-1, ratio))
    if ape is not None:
        score = score + ape
    w = torch.softmax(score, dim=2)
    return rmsnorm((kv * w).sum(dim=2).to(x.dtype), gamma)


def main():
    B, T, ratio = 1, 12, 2
    x = torch.randn(B, T, DIM, dtype=torch.bfloat16)
    wkv = torch.randn(D, DIM)
    wgate = torch.randn(D, DIM)
    gamma = torch.randn(D).abs() + 0.5

    ref = reference_compressor(x, wkv, wgate, gamma, ratio)
    got = op180_compressor(x, wkv, wgate, gamma, ratio, ape=None, coff=1)
    d = (ref.float() - got.float()).abs().max().item()
    print(f"[1] ratio-2 pool+norm, op180(ape=0, coff=1) vs reference : max|err| = {d:.3e}")
    ok1 = d == 0.0

    # ape is NOT zero-equivalent: a nonzero one changes the answer, so "V4.1 has no ape" is a
    # real claim about the emit and not a detail that washes out.
    ape = torch.randn(ratio, D)
    got_ape = op180_compressor(x, wkv, wgate, gamma, ratio, ape=ape, coff=1)
    d_ape = (ref.float() - got_ape.float()).abs().max().item()
    print(f"[2] same, but with a nonzero ape                          : max|err| = {d_ape:.3e}")
    ok2 = d_ape > 1e-3

    # ratio 1: no gate at all, and the gate weights must not be consulted.
    ref1 = reference_compressor(x, wkv, None, gamma, 1)
    got1 = rmsnorm(x @ wkv.T.to(x.dtype), gamma)
    d1 = (ref1.float() - got1.float()).abs().max().item()
    print(f"[3] ratio-1 (layer 20): plain projection + norm           : max|err| = {d1:.3e}")
    ok3 = d1 == 0.0

    # The epilogue: the scale format is the difference, and it is worth real error.
    lat = ref
    q_e4m3 = fake_quant(lat, QBLK, "e4m3")
    q_pow2 = fake_quant(lat, QBLK, "pow2")
    d_scale = (q_e4m3.float() - q_pow2.float()).abs().max().item()
    rel = d_scale / lat.float().abs().max().item()
    print(f"[4] fake-quant e4m3-scale vs pow2-scale (both fp4, blk 16): max|err| = {d_scale:.3e}"
          f"  ({rel * 100:.1f}% of latent amax)")
    ok4 = d_scale > 1e-3

    # And the quantization itself is lossy, so a hardware test must not assert bit-equality
    # against the pre-quant latent -- it asserts against the quantized reference.
    d_q = (q_e4m3.float() - lat.float()).abs().max().item()
    print(f"[5] fake-quant(e4m3) vs unquantized latent                : max|err| = {d_q:.3e}")

    # THE QUANT SPAN. op 180's attention arm quantizes [0, d-rd) only -- "rope dims stay bf16
    # for positional precision", citing V4's model.py:510. V4.1 quantizes the WHOLE latent:
    # `fp4_act_quant(latent, 16, True, ...)` with no slice (model.py:760), and `_window_kv`'s
    # own docstring says the window K is "quantized over the whole post-RoPE vector, RoPE tail
    # included" (701). Rope dims left in bf16 are simply not quantized, so they differ by the
    # full quantization error on that tail.
    full = fake_quant(lat, QBLK, "e4m3")
    partial = lat.float().clone()
    partial[..., : D - RD] = fake_quant(lat[..., : D - RD], QBLK, "e4m3").float()
    d_span = (full.float() - partial).abs().max().item()
    print(f"[6] quant span d vs d-rd (rope tail left bf16)            : max|err| = {d_span:.3e}")
    ok5 = d_span > 1e-3

    # ---- the split: op 180 arm 2 (stop after norm) + op 185 -------------------
    # A latent stands for the first token of its group, so group j ropes at position j * ratio
    # (model.py:753-756). The tables here are arbitrary; what is under test is that the two
    # halves compose to exactly the reference's one expression.
    n_pool = ref.shape[1]
    pos = torch.arange(n_pool) * ratio
    ang = pos.unsqueeze(-1).float() * torch.randn(RD // 2).abs()
    cos, sin = torch.cos(ang), torch.sin(ang)

    ref_cache = reference_compress_kv(ref, cos, sin)
    got_cache = op185(ref, cos, sin, QBLK, "e4m3")
    d7 = (ref_cache.float() - got_cache.float()).abs().max().item()
    print(f"[7] op180(stop after norm) + op185 vs _compress_kv        : max|err| = {d7:.3e}")
    ok7 = d7 == 0.0

    # THE E4M3 FLOOR IS CHOSEN, NOT CONSERVATIVE. 6*2**-9 makes amax/6 exactly 2**-9, e4m3's
    # smallest subnormal, so an all-zero block gets the smallest NONZERO scale. The E8M0 branch's
    # 6*2**-126 rounds to zero in e4m3 and the dequant would divide by it.
    s_e4m3 = torch.tensor(FLOOR_E4M3 / FP4_MAX).to(torch.float8_e4m3fn).float().item()
    s_wrong = torch.tensor(FLOOR_POW2 / FP4_MAX).to(torch.float8_e4m3fn).float().item()
    print(f"[8] all-zero block: scale at the E4M3 floor = {s_e4m3:.6g}, "
          f"at the E8M0 floor = {s_wrong:.6g}")
    ok8 = s_e4m3 == 2.0**-9 and s_wrong == 0.0

    # The indexer takes the SAME op at a different block and scale format -- `fp4_act_quant(k,
    # fp4_block_size, True)` defaults to E8M0, which is `fast_round_scale`, which is
    # `plow_round_scale`. One op, two call sites, and they must not be interchangeable.
    k = torch.randn(B, n_pool, D, dtype=torch.bfloat16)
    idx_ref = op185(k, cos, sin, IDX_QBLK, "pow2")
    idx_as_kv = op185(k, cos, sin, QBLK, "e4m3")
    d9 = (idx_ref.float() - idx_as_kv.float()).abs().max().item()
    print(f"[9] index keys (blk 32, E8M0) vs KV settings (blk 16, E4M3): max|err| = {d9:.3e}")
    ok9 = d9 > 1e-3

    # And the tap itself: what op 180 arm 2 writes is what the indexer's `wk` consumes, which is
    # the pre-RoPE latent and NOT the cache row. If those were interchangeable the split would
    # have been unnecessary.
    d10 = (ref.float() - ref_cache.float()).abs().max().item()
    print(f"[10] pre-RoPE tap vs the cache row it becomes             : max|err| = {d10:.3e}")
    ok10 = d10 > 1e-3

    print()
    checks = {
        "op180 with ape=0, coff=1 IS V4.1's compressor": ok1,
        "ape is not zero-equivalent (so it must be omitted, not passed)": ok2,
        "ratio 1 is a plain projection with no gate": ok3,
        "the scale format is a REAL difference, not a rounding detail": ok4,
        "the quant SPAN differs too: V4.1 quantizes the rope tail, V4 does not": ok5,
        "op 180 arm 2 + op 185 IS _compress_kv, exactly": ok7,
        "the E4M3 amax floor is the one that keeps a zero block's scale nonzero": ok8,
        "the index-key and compressed-KV quant settings are NOT interchangeable": ok9,
        "the pre-RoPE tap is not the cache row, so the split is load-bearing": ok10,
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
