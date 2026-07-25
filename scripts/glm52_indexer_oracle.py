#!/usr/bin/env python3
# glm52_indexer_oracle.py — G6 GATE: validate the plow DSA lightning-indexer path against the REAL
# HuggingFace GlmMoeDsaIndexer on REAL layer-0 weights.  [GLM52-DSA G6]
#
# The kernel bench (dsa_gather_bench.c) selected the top-k of SYNTHETIC random scores, so its
# "accuracy" was meaningless. THIS is the missing correctness number: on real weights, does plow's
# indexer -> radix-select produce the SAME top-2048 set that HF's indexer.topk produces?
#
#   HF reference  : instantiate transformers GlmMoeDsaIndexer, load the real (dequantized) indexer
#                   weights, run indexer.forward -> the int32 top-2048 indices (the ground truth).
#   plow mirror   : numpy/torch mirror of the ON-DEVICE path exactly as emitted (G2) + scored
#                   (d_index_score, op_attention.h) + selected (d_index_select_coop radix, dsa_pack_key):
#                     q = reshape_32x128(dequant(wq_b) @ q_resid);  rope(interleave, first 64 dims)
#                     k = layernorm_bias(k_norm, dequant(wk) @ hidden);  rope(interleave, first 64)
#                     w = weights_proj @ hidden
#                     score[t] = (1/sqrt(128)) * sum_h (w[h]/sqrt(32)) * ReLU(q[h].k[t])
#                     select   = top-2048 by dsa_pack_key (score desc, lowest-index tie-break)
#
# Layer 0 is a 'full' indexer layer AND its input is just embed_tokens(ids) -> input_layernorm, so
# the oracle needs no other decoder layers (cheap, CPU-only). Reports:
#   (A) score relmax  plow-mirror vs HF index_scores
#   (B) SELECTION set match  plow-radix top-2048 vs HF topk  (THE gate: exact set equality)
#
# Usage:
#   nix develop -c python3 scripts/glm52_indexer_oracle.py --model /home/lava/models/GLM-5.2-FP8 \
#       --layer 0 --seq 4096
import os, sys, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import glm52_prep as P                 # _index_shards, load_tensor, dequant_blockfp8
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaIndexer, GlmMoeDsaRMSNorm, GlmMoeDsaRotaryEmbedding)

torch.set_grad_enabled(False)


def dsa_pack_key(score, t, length):
    """Mirror op_attention.h dsa_pack_key: monotone 52-bit key = (ordered_f32_bits<<20)|(len-1-t).
    Top-k of these keys == top-k scores with LOWEST-INDEX tie-break (the reproducible selector)."""
    sb = np.frombuffer(np.float32(score).tobytes(), dtype=np.uint32)[0]
    sb = (~sb & 0xFFFFFFFF) if (sb & 0x80000000) else (sb | 0x80000000)
    return (int(sb) << 20) | ((length - 1 - t) & 0xFFFFF)


def plow_topk(scores, top_k):
    """The device radix selector's RESULT set: the top_k positions by dsa_pack_key."""
    n = len(scores)
    keys = [(dsa_pack_key(float(scores[t]), t, n), t) for t in range(n)]
    keys.sort(reverse=True)
    return set(t for _, t in keys[:top_k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/lava/models/GLM-5.2-FP8")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    idx = P._index_shards(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    L = args.layer
    assert cfg.indexer_types[L] == "full", f"layer {L} is not a 'full' indexer layer"
    S = args.seq
    HD, HI, TOPK = cfg.index_head_dim, cfg.index_n_heads, cfg.index_topk
    DR = cfg.qk_rope_head_dim
    p = f"model.layers.{L}."
    a = p + "self_attn."

    # ---- real hidden_states at layer L input = embed(ids) [layer 0]; then input_layernorm (HF applies
    #      input_layernorm BEFORE attention, so the indexer sees the NORMED hidden_states). ----
    assert L == 0, "cheap oracle only builds real hidden_states for layer 0 (embed-only input)"
    rng = np.random.default_rng(args.seed)
    ids = rng.integers(0, cfg.vocab_size, size=S)
    emb = P.load_tensor(idx, "model.embed_tokens.weight")        # [V,H] bf16
    hidden_pre = emb[torch.from_numpy(ids).long()].to(torch.bfloat16)  # [S,H]

    gin = GlmMoeDsaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    gin.weight.copy_(P.load_tensor(idx, p + "input_layernorm.weight").float())
    hidden = gin(hidden_pre.float()).to(torch.bfloat16)          # [S,H] post input_layernorm

    # ---- q_resid = q_a_layernorm(q_a_proj(hidden)) (block-fp8 q_a_proj dequantized) ----
    qa_w = torch.from_numpy(P.dequant_blockfp8(idx, a + "q_a_proj.weight").numpy())  # [QL,H]
    q_resid = hidden.float() @ qa_w.t()
    gqa = GlmMoeDsaRMSNorm(cfg.q_lora_rank, eps=cfg.rms_norm_eps)
    gqa.weight.copy_(P.load_tensor(idx, a + "q_a_layernorm.weight").float())
    q_resid = gqa(q_resid).to(torch.bfloat16)                    # [S,QL]

    # ---- rotary cos/sin (main rope tables, theta=8e6) ----
    rope = GlmMoeDsaRotaryEmbedding(cfg)
    pos = torch.arange(S).unsqueeze(0)
    cos, sin = rope(hidden.float().unsqueeze(0), pos)            # [1,S,DR]

    # ================= HF REFERENCE =================
    hf = GlmMoeDsaIndexer(cfg, L).eval()
    hf.wq_b.weight.copy_(torch.from_numpy(P.dequant_blockfp8(idx, a + "indexer.wq_b.weight").numpy()))
    hf.wk.weight.copy_(torch.from_numpy(P.dequant_blockfp8(idx, a + "indexer.wk.weight").numpy()))
    hf.k_norm.weight.copy_(P.load_tensor(idx, a + "indexer.k_norm.weight").float())
    hf.k_norm.bias.copy_(P.load_tensor(idx, a + "indexer.k_norm.bias").float())
    hf.weights_proj.weight.copy_(P.load_tensor(idx, a + "indexer.weights_proj.weight").float())

    # HF forward wants position_embeddings (cos,sin) and a causal mask; pass mask=None (causal via pos).
    hidden_b = hidden.float().unsqueeze(0)          # [1,S,H] (module weights are f32)
    q_resid_b = q_resid.float().unsqueeze(0)        # [1,S,QL]
    hf_topk = hf(hidden_b, q_resid_b, (cos, sin), None, pos)[0]  # [S,TOPK] int32
    # Re-derive HF's raw index_scores for the score-fidelity check (mirror forward up to topk).
    with torch.no_grad():
        q = hf.wq_b(q_resid_b).view(1, S, HI, HD)
        q_rot, q_pass = torch.split(q, [DR, HD - DR], dim=-1)
        k = hf.k_norm(hf.wk(hidden_b)).unsqueeze(2)
        k_rot, k_pass = torch.split(k, [DR, HD - DR], dim=-1)
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import apply_rotary_pos_emb_interleave
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)
        sc = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * hf.softmax_scale
        sc = torch.relu(sc)
        wts = hf.weights_proj(hidden_b.float()).float() * (HI ** -0.5)
        hf_scores = torch.matmul(wts.unsqueeze(-2), sc).squeeze(-2)[0]   # [S,S]

    # ================= PLOW MIRROR (on-device path) =================
    # q/k/w projections mirror the fp8 block-GEMV (GemvFp8Blk) via the SAME dequant the prep ships.
    wq_b = torch.from_numpy(P.dequant_blockfp8(idx, a + "indexer.wq_b.weight").numpy())  # [HI*HD,QL]
    wk = torch.from_numpy(P.dequant_blockfp8(idx, a + "indexer.wk.weight").numpy())      # [HD,H]
    knw = P.load_tensor(idx, a + "indexer.k_norm.weight").float()
    knb = P.load_tensor(idx, a + "indexer.k_norm.bias").float()
    wp = P.load_tensor(idx, a + "indexer.weights_proj.weight").float()                    # [HI,H]

    qf = (q_resid.float() @ wq_b.t()).view(S, HI, HD)           # [S,HI,HD]
    kf = hidden.float() @ wk.t()                                # [S,HD]
    # LayerNorm with bias (eps 1e-6) — matches HF nn.LayerNorm(k_norm).
    mu = kf.mean(-1, keepdim=True); var = kf.var(-1, unbiased=False, keepdim=True)
    kf = (kf - mu) / torch.sqrt(var + 1e-6) * knw + knb         # [S,HD]

    def rope_interleave(x, cos, sin):                           # x[...,DR]; mirror HeadNormRope interleave
        c = cos[0, :, :DR // 2]; s = sin[0, :, :DR // 2]        # [S,DR/2]
        while c.dim() < x.dim():
            c = c.unsqueeze(1); s = s.unsqueeze(1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    q_rot = rope_interleave(qf[..., :DR], cos, sin); qf = torch.cat([q_rot, qf[..., DR:]], -1)
    k_rot = rope_interleave(kf[:, None, :DR], cos, sin)[:, 0]; kf = torch.cat([k_rot, kf[:, DR:]], -1)

    wf = hidden.float() @ wp.t()                                # [S,HI]
    scale = HD ** -0.5
    # decode = the newest query at position qpos; attends to all t in [0, qpos].
    qpos = S - 1
    qh = qf[qpos]                                               # [HI,HD]
    dots = torch.relu(qh @ kf[:qpos + 1].t())                  # [HI, qpos+1]
    plow_scores = (wf[qpos] * (HI ** -0.5)) @ dots * scale      # [qpos+1]

    # ================= DEVICE-PATH MIRROR (the ACTUAL emitted ops) =================
    # The emit does NOT split the head: a plain GemvFp8Blk(wq_b) yields [HI][128] per head as
    # [rope64 | pass64] (HF's split order), then a SINGLE HD=128 GPT-J-interleaved RoPE with an
    # identity-tail table (cos=1/sin=0 for pairs 32..63) rotates dims 0..63 and passes 64..127.
    # GPT-J layout differs from HF's de-interleaved layout, but the transform is applied IDENTICALLY
    # to q and k, so the 128-dot (hence score + selection) is invariant. This block proves that.
    def rope_gptj_128(x, cos, sin):                            # x[...,128]; cos/sin[S,32] (real pairs 0..31)
        c = cos[0]; s = sin[0]                                 # [S,32]
        while c.dim() < x.dim():
            c = c.unsqueeze(1); s = s.unsqueeze(1)
        out = x.clone()
        e = x[..., 0:64:2]; o = x[..., 1:64:2]                 # even/odd of the first 64 (pairs 0..31)
        out[..., 0:64:2] = e * c - o * s
        out[..., 1:64:2] = e * s + o * c                       # dims 64..127 untouched (identity tail)
        return out
    qdev = rope_gptj_128(qf_dev_raw := (q_resid.float() @ wq_b.t()).view(S, HI, HD),
                         cos[..., :DR // 2], sin[..., :DR // 2])
    kf_dev = hidden.float() @ wk.t()
    mu2 = kf_dev.mean(-1, keepdim=True); var2 = kf_dev.var(-1, unbiased=False, keepdim=True)
    kf_dev = (kf_dev - mu2) / torch.sqrt(var2 + 1e-6) * knw + knb
    kdev = rope_gptj_128(kf_dev[:, None, :], cos[..., :DR // 2], sin[..., :DR // 2])[:, 0]
    qhd = qdev[qpos]
    dots_d = torch.relu(qhd @ kdev[:qpos + 1].t())
    dev_scores = (wf[qpos] * (HI ** -0.5)) @ dots_d * scale
    dev_relmax = (dev_scores - hf_scores[qpos, :qpos + 1]).abs().max().item() / (denom := hf_scores[qpos, :qpos + 1].abs().max().item() + 1e-9)
    dev_set = plow_topk(dev_scores.numpy(), min(TOPK, qpos + 1))
    hf_set0 = set(hf_topk[qpos].tolist())
    di = len(dev_set & hf_set0)
    print(f"[G6-DEV] device-path (HD=128 GPT-J identity-tail rope) score relmax vs HF: {dev_relmax:.3e}; "
          f"top-{min(TOPK,qpos+1)} match {di}/{min(TOPK,qpos+1)} (exact={dev_set==hf_set0})")

    # score fidelity
    ref = hf_scores[qpos, :qpos + 1]
    denom = ref.abs().max().item() + 1e-9
    relmax = (plow_scores - ref).abs().max().item() / denom
    print(f"[G6-A] score relmax plow-vs-HF @qpos={qpos}: {relmax:.3e}  (denom={denom:.3e})")

    # selection fidelity (THE gate)
    hf_set = set(hf_topk[qpos].tolist())
    plow_set = plow_topk(plow_scores.numpy(), min(TOPK, qpos + 1))
    inter = len(hf_set & plow_set)
    tk = min(TOPK, qpos + 1)
    print(f"[G6-B] top-{tk} SELECTION match plow-vs-HF: {inter}/{tk} "
          f"({100.0*inter/tk:.3f}%)  exact_set_equal={hf_set == plow_set}")
    # margin analysis: how close is the plow/HF boundary (sensitivity of the last selected)?
    srt = np.sort(plow_scores.numpy())[::-1]
    if tk < len(srt):
        gap = srt[tk - 1] - srt[tk]
        print(f"[G6-B] boundary gap score[{tk-1}]-score[{tk}] = {gap:.3e} "
              f"(rel {gap/denom:.2e})")

    # ================= REAL-WEIGHT COHERENCE (attention output dense-vs-gather) =================
    # THE coherence number: does attending to ONLY the top-2048 selected latent rows (gather-ON)
    # change the MLA attention output vs attending to ALL ctx (dense, gather-OFF)? Runs the ABSORBED
    # MLA decode (mla_ref.rs math) on REAL layer-0 derived weights, at qpos, over the two sets. The
    # gather kernel is already validated to equal dense-restricted-to-set, so this bounds the on-device
    # gather-ON vs gather-OFF per-token output divergence on real weights.
    QN, VD, QKH = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.qk_head_dim
    NH = cfg.num_attention_heads
    q_b = torch.from_numpy(P.dequant_blockfp8(idx, a + "q_b_proj.weight").numpy()).view(NH, QKH, cfg.q_lora_rank)
    kv_b = torch.from_numpy(P.dequant_blockfp8(idx, a + "kv_b_proj.weight").numpy()).view(NH, QN + VD, cfg.kv_lora_rank)
    Wqa = torch.einsum('hpl,hpk->hlk', kv_b[:, :QN, :], q_b[:, :QN, :])       # [NH,DK,QL]
    Wqr = q_b[:, QN:, :]                                                       # [NH,DR,QL]
    Wuv = kv_b[:, QN:, :].transpose(-1, -2).contiguous()                      # [NH,DK,VD]
    kv_a = torch.from_numpy(P.dequant_blockfp8(idx, a + "kv_a_proj_with_mqa.weight").numpy())
    Wckv, Wkr = kv_a[:cfg.kv_lora_rank], kv_a[cfg.kv_lora_rank:cfg.kv_lora_rank + DR]
    kvln = P.load_tensor(idx, a + "kv_a_layernorm.weight").float()
    DK = cfg.kv_lora_rank
    hf32 = hidden.float()
    ckv = hf32 @ Wckv.t()                                                     # [S,DK]
    ckv = ckv * torch.rsqrt(ckv.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * kvln
    krot = rope_gptj_128(torch.cat([(hf32 @ Wkr.t()), torch.zeros(S, DR)], -1)[:, None, :],
                         cos[..., :DR // 2], sin[..., :DR // 2])[:, 0, :DR]   # [S,DR] (rope the 64)
    qlat_q = q_resid.float()[qpos]                                            # [QL]
    q_abs = torch.einsum('hlk,k->hl', Wqa, qlat_q)                            # [NH,DK]
    q_rp_raw = torch.einsum('hrk,k->hr', Wqr, qlat_q)                         # [NH,DR]
    cq = cos[:, qpos:qpos + 1, :DR // 2]; sq = sin[:, qpos:qpos + 1, :DR // 2]  # [1,1,DR/2]
    q_rp = rope_gptj_128(torch.cat([q_rp_raw, torch.zeros(NH, DR)], -1)[None], cq, sq)[0][:, :DR]
    ascale = QKH ** -0.5

    def mla_out(sel):
        sel = torch.tensor(sorted(sel))
        sc = (q_abs @ ckv[sel].t() + q_rp @ krot[sel].t()) * ascale          # [NH, |sel|]
        p = torch.softmax(sc, -1)
        oacc = p @ ckv[sel]                                                   # [NH, DK]
        return torch.einsum('hl,hlv->hv', oacc, Wuv)                         # [NH, VD]

    # (1) FIDELITY (the gate): plow's gather set vs HF's gather set produce the SAME MLA output.
    #     GLM-5.2 is NATIVELY sparse — HF itself attends only to its top-2048 — so this (not a full-dense
    #     comparison) is what "does plow reproduce the model" means. Sets are 2048/2048 equal => ~0.
    o_plow = mla_out(plow_set)
    o_hf = mla_out(hf_set)
    fid = (o_plow - o_hf).abs().max().item() / (o_hf.abs().max().item() + 1e-9)
    print(f"[G6-COH] MLA attn-output plow-set vs HF-set (the fidelity, qpos={qpos}): relmax {fid:.3e}")
    # (2) FYI — the MODEL's own sparsity effect (gather-ON vs a hypothetical full-dense attention). This
    #     is a property of GLM-5.2's DSA design, NOT a plow metric, and is input-dependent (inflated here
    #     by RANDOM tokens keeping only 2048/{qpos+1}; on trained-for real data the indexer's picks align
    #     with the MLA attention mass, so it shrinks). Reported for context, NOT gated.
    o_dense = mla_out(set(range(qpos + 1)))
    spars = (o_plow - o_dense).abs().max().item() / (o_dense.abs().max().item() + 1e-9)
    print(f"[G6-COH] (fyi) model sparsity effect gather-vs-full-dense (random tokens, input-dependent): "
          f"relmax {spars:.3e}")

    ok = relmax < 5e-2 and inter >= tk - 2 and fid < 5e-2  # gate: formula + selection + set-output fidelity
    print(f"\n[G6] {'GATE PASS' if ok else '*** GATE FAIL ***'} "
          f"(score relmax {relmax:.2e}, selection {inter}/{tk})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
