"""Indexer oracle: V4.1's `Indexer.forward` against ops 117/118.

The 38 reader layers select their compressed positions with this, and it is the piece op 55's
own doc calls "the SELECTOR, not the flash ... a real design problem". Four questions, all of
which decide what the emit is allowed to leave out:

  1. is op 117's `score[t][s] = Σ_h w[t][h]·ReLU(q[t][h]·k[s]) · scale` the reference's
     `(index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)`?
  2. is op 117/118's new `pool_size` bound exactly `compress_lens = arange(1, s+1) // ratio`?
  3. IS THE TWO-LEVEL CANDIDATE STAGE INERT AT 8k? `candidate_topk_blocks` is 2048 and
     `candidate_block_size` is 8, so the level-one mask covers 16384 compressed positions --
     more than an 8k context has at either ratio. If it is inert the emit may skip it; if not,
     every layer after 20 needs a second selection pass.
  4. does the top-k ORDER matter? The reference re-sorts into position order; op 118 emits in
     radix order. The gather flash sums over the selected set, so only the SET should matter.

Run: PYTHONPATH=/workspace/oracle-venv/site python3 scripts/dsv41_indexer_oracle.py
"""
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)

DI = 128       # index_head_dim
HI = 32        # index_n_heads -- replicated, which is what the MFMA A-tile wants
TOPK = 8       # index_topk, scaled down from 512
CAND_BLOCKS = 2048   # candidate_topk_blocks, the real one
CAND_BLK = 8         # candidate_block_size, the real one


def reference_score(q, k, w, scale):
    """model.py:556-559, minus the all_reduce (one rank holds all 32 heads here)."""
    s = torch.einsum("shd,td->sht", q.float(), k.float())
    return (s.relu_() * (w.float() * scale).unsqueeze(-1)).sum(dim=1)


def op117(q, k, w, scale, ratio, seqlen):
    """op_attention_common.h d_index_score_pf with `pool_size = ratio`: the same weighted-ReLU
    sum, with `scale` in the EPILOGUE and the column axis in pools."""
    n_pool = seqlen // ratio
    out = torch.full((seqlen, n_pool), 0.0)
    for t in range(seqlen):
        row_end = (t + 1) // ratio  # the causal bound, in COLUMNS
        for p in range(n_pool):
            if p >= row_end:
                continue
            part = 0.0
            for h in range(HI):
                d = float(q[t, h].float() @ k[p].float())
                part += float(w[t, h]) * (d if d > 0.0 else 0.0)
            out[t, p] = part * scale
    return out


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """model.py:583-612, transcribed."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def main():
    seqlen, ratio = 24, 2
    n_pool = seqlen // ratio
    scale = DI**-0.5 * HI**-0.5
    q = torch.randn(seqlen, HI, DI, dtype=torch.bfloat16)
    k = torch.randn(n_pool, DI, dtype=torch.bfloat16)
    w = torch.randn(seqlen, HI, dtype=torch.bfloat16)

    ref = reference_score(q, k, w, scale)
    # the reference's own reachability mask
    compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
    ref_masked = ref.masked_fill(torch.arange(n_pool) >= compress_lens, -torch.inf)

    got = op117(q, k, w, scale, ratio, seqlen)
    live = torch.arange(n_pool) < compress_lens
    d1 = (ref[live] - got[live]).abs().max().item()
    rel = d1 / ref[live].abs().max().item()
    print(f"[1] op117 vs the reference score, on reachable pools : max|err| = {d1:.3e}"
          f"  ({rel:.2e} relative)")
    ok1 = rel < 1e-5

    # [2] the bound. op 117 writes NOTHING past `(t+1)//ratio`, and op 118 scans the same
    # count -- so an unreachable column can never be selected. The reference reaches the same
    # place by writing -inf. Equal as SETS, which is what the selector produces.
    unreachable = (~live) & (got != 0.0)
    print(f"[2] columns written past (t+1)//ratio               : {int(unreachable.sum())}")
    ok2 = int(unreachable.sum()) == 0 and bool((ref_masked[~live] == -torch.inf).all())

    # [3] THE CANDIDATE STAGE AT 8k. Level one keeps `min(topk_blocks, num_blocks)` blocks, and
    # drops the picks that came back -inf -- i.e. the unreachable ones. When num_blocks does not
    # exceed topk_blocks that is every reachable block, so the mask a level-two layer applies is
    # the reachability mask it already has.
    for ctx, r in ((8192, 2), (8192, 1)):
        nb = -(-(ctx // r) // CAND_BLK)
        print(f"[3] ctx={ctx} ratio={r}: {ctx // r} pools -> {nb} blocks vs "
              f"candidate_topk_blocks={CAND_BLOCKS}  -> {'INERT' if nb <= CAND_BLOCKS else 'ACTIVE'}")
    ok3 = all(-(-(8192 // r) // CAND_BLK) <= CAND_BLOCKS for r in (1, 2))

    # ... and prove the inertness on a small case rather than only by counting. The mask is NOT
    # equal to reachability -- it is block-granular, so a block holding one reachable position
    # keeps all `candidate_block_size` of them -- it is a SUPERSET of it. That is what makes the
    # stage inert: `Indexer.forward` applies the reachability mask FIRST (model.py:563-565) and
    # only then masks with `~candidates`, so every position the candidate mask would remove is
    # already -inf. Masking with a superset removes nothing.
    keep = select_candidate_blocks(ref_masked, compress_lens, CAND_BLOCKS, CAND_BLK)
    dropped = int((live & ~keep).sum())
    extra = int((keep & ~live).sum())
    after = ref_masked.masked_fill(~keep, -torch.inf)
    changed = int((after != ref_masked).sum())
    print(f"[4] candidate mask vs reachability: drops {dropped} reachable, keeps {extra} "
          f"unreachable; masking a reachability-masked score changes {changed}")
    ok4 = dropped == 0 and changed == 0

    # [5] ORDER. The reference re-sorts the top-k into position order; op 118 emits in radix
    # order. The gather flash sums over the selected rows, so the SET is what has to match.
    topk = min(TOPK, n_pool)
    ref_idx = ref_masked.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    got_idx = got.masked_fill(~live, -torch.inf).topk(topk, dim=-1).indices
    same = all(
        set(ref_idx[t].tolist()) == set(got_idx[t].tolist())
        for t in range(seqlen)
        if int(compress_lens[t]) >= topk
    )
    print(f"[5] selected SETS agree (order does not)            : {same}")
    ok5 = same

    print()
    checks = {
        "op 117's weighted-ReLU sum IS the reference's index_score": ok1,
        "pool_size makes the bound exactly compress_lens": ok2,
        "the two-level candidate stage is INERT at 8k, so the emit may skip it": ok3,
        "the candidate mask is a SUPERSET of reachability, so level two is a no-op": ok4,
        "the top-k SET is what matters; op 118's radix order is free": ok5,
    }
    for k_, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k_}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
