"""CSA2 compressor oracle: the V4.1 reference against a model of what op 180 computes.

The reference half is transcribed from the checkpoint's own `inference/model.py` (`class
Compressor`, 429-486; `RMSNorm`, 281; `apply_rotary_emb`, 392) and `inference/kernel.py`
(`fp4_act_quant`, 184). The op-180 half is transcribed from `runtime/amd/op_compress.h` --
its header states the math and `cmp_fake_quant_block` states the epilogue.

It answers three questions that doc 12.10 currently answers by READING:
  1. does V4.1's compressor equal op 180's with ape = 0 and coff = 1?
  2. is the per-block scale format the ONLY remaining difference?
  3. what tolerance does a hardware test get to assert?

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
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-6)
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

    print()
    checks = {
        "op180 with ape=0, coff=1 IS V4.1's compressor": ok1,
        "ape is not zero-equivalent (so it must be omitted, not passed)": ok2,
        "ratio 1 is a plain projection with no gate": ok3,
        "the scale format is a REAL difference, not a rounding detail": ok4,
        "the quant SPAN differs too: V4.1 quantizes the rope tail, V4 does not": ok5,
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
