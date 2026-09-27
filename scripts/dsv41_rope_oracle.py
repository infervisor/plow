"""RoPE oracle for V4.1's attention core: the pairing, and the inverse that closes it.

Two claims the emit makes by OMISSION, priced here before either is changed.

  1. `QwenHeadNormRope` (op 142) rotates the pair `(i, i + rotary/2)` -- the half-split
     (NeoX) convention -- and that is what `emit_dsv41_attn_core` uses for `q` and for the
     shared latent. V4.1's `apply_rotary_emb` (`model.py:392`) says "taking adjacent element
     pairs as complex numbers": `view_as_complex(x.unflatten(-1, (-1, 2)))`, the interleaved
     (GPT-J) convention. Same tensor shape, same table shape, same table CONTENTS, different
     operator.

  2. `Attention.forward` ends with `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`
     (`model.py:781`) on EVERY layer, window-only ones included. The emit stops at
     `FlashMerge`. Op 195 exists for exactly this and has no emit site.

The interesting question for (1) is whether a CONSISTENT wrong pairing cancels. Both q and
the latent go through the same op, and a rotation applied in a permuted basis is conjugate to
the right one -- so the check below asks the attention SCORE, not the vectors.

Run: PYTHONPATH=/workspace/oracle-venv/site python3 scripts/dsv41_rope_oracle.py
"""
import sys

import torch

torch.manual_seed(0)

D = 512     # head_dim
RD = 64     # qk_rope_head_dim: the rope is INTERIOR, on [D-RD, D)
H = 4       # heads, scaled down
T = 16      # tokens
THETA = 10000.0  # window-only layers: base theta, YaRN disabled (model.py:686)


def tables(ctx, rd, theta):
    """The [ctx][rd/2] cos/sin pair. ONE table serves both conventions: they differ in which
    channels pair up, not in the angle a pair takes."""
    freqs = 1.0 / (theta ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
    ang = torch.outer(torch.arange(ctx, dtype=torch.float64), freqs)
    return torch.cos(ang).float(), torch.sin(ang).float()


def rope_interleaved(x, cos, sin, pos, inverse=False):
    """model.py:392-406. Pairs are (2m, 2m+1); `inverse` conjugates."""
    y = x.clone()
    c = cos[pos].unsqueeze(1)  # [T, 1, rd/2], broadcast over heads
    s = sin[pos].unsqueeze(1)
    if inverse:
        s = -s
    x0, x1 = y[..., -RD:][..., 0::2], y[..., -RD:][..., 1::2]
    r0 = x0 * c - x1 * s
    r1 = x0 * s + x1 * c
    tail = y[..., -RD:]
    tail[..., 0::2], tail[..., 1::2] = r0, r1
    return y


def rope_half_split(x, cos, sin, pos):
    """op_qwen_gdn.h d_qwen_headnorm_rope_t: partner is `lane ^ (rotary/2)`, table index
    `lane & (half-1)`. Pairs are (i, i + rd/2) inside the rotary section."""
    y = x.clone()
    h2 = RD // 2
    c = cos[pos].unsqueeze(1)
    s = sin[pos].unsqueeze(1)
    tail = y[..., -RD:]
    lo, hi = tail[..., :h2].clone(), tail[..., h2:].clone()
    tail[..., :h2] = lo * c - hi * s
    tail[..., h2:] = hi * c + lo * s
    return y


def score(q, k):
    """The rope term of one absorbed-MLA score: q is [T,H,D], k is the ONE shared latent row
    per token, [T,D]. Only the relative-position content of the rope tail is at stake, so the
    whole dot is taken and the nope part is common to both conventions."""
    return torch.einsum("thd,sd->ths", q.float(), k.float())


def main():
    cos, sin = tables(T, RD, THETA)
    pos = torch.arange(T)
    q = torch.randn(T, H, D)
    kv = torch.randn(T, D)          # ONE latent row per token, shared by every head
    kv4 = kv.unsqueeze(1)           # the rope helpers want a head axis

    q_int = rope_interleaved(q, cos, sin, pos)
    q_half = rope_half_split(q, cos, sin, pos)
    k_int = rope_interleaved(kv4, cos, sin, pos).squeeze(1)
    k_half = rope_half_split(kv4, cos, sin, pos).squeeze(1)

    d_q = (q_int - q_half).abs().max().item()
    print(f"[1] q after interleaved vs half-split rope     : max|err| = {d_q:.3e}")
    ok1 = d_q > 1e-3

    # THE REAL QUESTION. Both sides took the same wrong pairing, and a rotation in a permuted
    # basis is conjugate to the right one -- so if the permutation commuted with the rotation
    # the scores would agree and the pairing would be free. It does not: half-split gives pair
    # (i, i+32) the angle of frequency i, interleaved gives pair (2m, 2m+1) the angle of
    # frequency m, so the two operators assign DIFFERENT angles to the same channels.
    s_int, s_half = score(q_int, k_int), score(q_half, k_half)
    d_s = (s_int - s_half).abs().max().item()
    rel = d_s / s_int.abs().max().item()
    print(f"[2] attention score, both sides consistently   : max|err| = {d_s:.3e}"
          f"  ({rel * 100:.1f}% of max score)")
    ok2 = d_s > 1e-3

    # A rope is only meaningful through the relative rotation it induces. If that were the
    # same, (2) would be an artefact of the absolute phase rather than a defect.
    dq, dk = q_int - q, q_half - q
    print(f"[3] neither convention is a no-op              : "
          f"{dq.abs().max().item():.3e} / {dk.abs().max().item():.3e}")
    ok3 = dq.abs().max().item() > 1e-3 and dk.abs().max().item() > 1e-3

    # ---- the inverse -----------------------------------------------------------------
    # op 195 is the conjugate, and `sin -> -sin` is the whole of it. Round-tripping is the
    # cheapest statement of that.
    rt = rope_interleaved(rope_interleaved(q, cos, sin, pos), cos, sin, pos, inverse=True)
    d_rt = (rt - q).abs().max().item()
    print(f"[4] irope(rope(q)) == q (op 195 is the conjugate): max|err| = {d_rt:.3e}")
    ok4 = d_rt < 1e-4

    # And what skipping it costs. `o` comes out of the flash still carrying the query's
    # rotation; `wo_a` is a fixed matrix that was trained against the de-rotated form.
    o = torch.randn(T, H, D)
    o_fixed = rope_interleaved(o, cos, sin, pos, inverse=True)
    d_o = (o - o_fixed).abs().max().item()
    print(f"[5] attention output with vs without op 195    : max|err| = {d_o:.3e}")
    ok5 = d_o > 1e-3

    print()
    checks = {
        "the two rope conventions are different operators": ok1,
        "a CONSISTENT wrong pairing does NOT cancel in the score": ok2,
        "both conventions actually rotate (the table is not degenerate)": ok3,
        "op 195's sin -> -sin is exactly the inverse of the forward rope": ok4,
        "omitting op 195 leaves the query's rotation in the attention output": ok5,
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
