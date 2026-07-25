#!/usr/bin/env python3
# glm52_oracle.py — HF-transformers ORACLE for the GLM-5.2 (GlmMoeDsa) single-block B1 bring-up.
#
# Instantiates ONE GlmMoeDsaDecoderLayer at REAL dims (hidden 6144, nh 64, qk 192+64, v 256,
# kv_lora 512, q_lora 2048) with SYNTHETIC SEEDED bf16 weights, runs the HF forward over an
# L<=2048 dense context (layer_idx=3 => "shared" indexer => the DSA indexer is a no-op; we feed
# a dense prev_topk_indices so attention is full causal), and DUMPS a binary fixture the C test
# (glm52_block_gfx950_test.c) diffs plow against:
#   - the absorbed MLA weights plow's kernel wants (Wqa = W_uk_nope^T @ q_b_nope, W_uv = value^T),
#     with the query RoPE folded into Wqr and the k-rope cache pre-rotated (INTERLEAVED rope),
#   - the two lora down-projs + their RMSNorm gains, o_proj, both block layernorms,
#   - the router weight + e_score_correction_bias, the E experts + the shared expert,
#   - the c_kv / k_rot latent caches, and the HF block output (+ attn_out / xn2 / h1 substeps).
#
# MEMORY: the box is RAM-limited (~3GB), so the real 256 experts (19GB) are infeasible for a
# single-layer synthetic de-risk; we use E=16, top_k=8 (a TIGHTER 8-of-16 selection than 8-of-256,
# which stresses the fp32-router-vs-bf16 flip risk harder). All other dims are the real model.
import os, struct, sys
import numpy as np
import torch

torch.manual_seed(1234)
np.random.seed(1234)

MODEL_DIR = os.environ.get("GLM_CONFIG_DIR", "/home/lava/models/GLM-5.2-FP8")
OUT = sys.argv[1] if len(sys.argv) > 1 else "glm52_fixture.bin"
L = int(os.environ.get("GLM_L", "512"))       # dense context length (<=2048 -> indexer no-op)
E = int(os.environ.get("GLM_E", "16"))         # experts (RAM-forced reduction from 256)
LAYER = 3                                       # sparse MLP + "shared" indexer

from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer, GlmMoeDsaRotaryEmbedding, apply_rotary_pos_emb_interleave,
)

cfg = AutoConfig.from_pretrained(MODEL_DIR)
cfg.n_routed_experts = E
cfg.num_local_experts = E
cfg._attn_implementation = "eager"
assert cfg.mlp_layer_types[LAYER] == "sparse" and cfg.indexer_types[LAYER] == "shared", \
    (cfg.mlp_layer_types[LAYER], cfg.indexer_types[LAYER])

H   = cfg.hidden_size            # 6144
NH  = cfg.num_attention_heads    # 64
DK  = cfg.kv_lora_rank           # 512
DR  = cfg.qk_rope_head_dim       # 64
QN  = cfg.qk_nope_head_dim       # 192
VD  = cfg.v_head_dim             # 256
QL  = cfg.q_lora_rank            # 2048
QKH = cfg.qk_head_dim            # 256
IMOE= cfg.moe_intermediate_size  # 2048
TOPK= cfg.num_experts_per_tok    # 8
EPS = cfg.rms_norm_eps
SCALE = QKH ** -0.5              # 1/sqrt(256) = 0.0625
RSCALE = cfg.routed_scaling_factor
THETA = cfg.rope_parameters["rope_theta"]
qpos = L - 1
print(f"dims H={H} NH={NH} DK={DK} DR={DR} QN={QN} VD={VD} QL={QL} IMOE={IMOE} "
      f"E={E} TOPK={TOPK} L={L} scale={SCALE} eps={EPS} theta={THETA}")

torch.set_grad_enabled(False)
BF16 = torch.bfloat16

# ---- build the layer + rotary; overwrite params with seeded bf16 weights ----
layer = GlmMoeDsaDecoderLayer(cfg, LAYER).eval()
rot = GlmMoeDsaRotaryEmbedding(cfg)

g = torch.Generator().manual_seed(0xB1)
def rw(*shape, s=0.02):   # small seeded bf16 weight
    return (torch.randn(*shape, generator=g) * s).to(BF16)

sd = layer.self_attn
sd.q_a_proj.weight.copy_(rw(QL, H))
sd.q_a_layernorm.weight.copy_((1.0 + 0.05 * torch.randn(QL, generator=g)).to(BF16))
sd.q_b_proj.weight.copy_(rw(NH * QKH, QL))
sd.kv_a_proj_with_mqa.weight.copy_(rw(DK + DR, H))
sd.kv_a_layernorm.weight.copy_((1.0 + 0.05 * torch.randn(DK, generator=g)).to(BF16))
sd.kv_b_proj.weight.copy_(rw(NH * (QN + VD), DK))
sd.o_proj.weight.copy_(rw(H, NH * VD))
layer.input_layernorm.weight.copy_((1.0 + 0.05 * torch.randn(H, generator=g)).to(BF16))
layer.post_attention_layernorm.weight.copy_((1.0 + 0.05 * torch.randn(H, generator=g)).to(BF16))

mlp = layer.mlp
mlp.gate.weight.copy_(rw(E, H, s=0.03))
mlp.gate.e_score_correction_bias.copy_((0.1 * torch.randn(E, generator=g)).to(torch.float32))
# experts: gate_up_proj [E, 2*IMOE, H] (rows 0..IMOE=gate, IMOE..2IMOE=up); down_proj [E, H, IMOE]
mlp.experts.gate_up_proj.copy_(rw(E, 2 * IMOE, H))
mlp.experts.down_proj.copy_(rw(E, H, IMOE))
mlp.shared_experts.gate_proj.weight.copy_(rw(IMOE, H))
mlp.shared_experts.up_proj.weight.copy_(rw(IMOE, H))
mlp.shared_experts.down_proj.weight.copy_(rw(H, IMOE))

layer = layer.to(BF16)

# ---- inputs ----
hidden = (0.3 * torch.randn(1, L, H, generator=g)).to(BF16)
position_ids = torch.arange(L).unsqueeze(0)
cos, sin = rot(hidden.float(), position_ids)          # [1,L,DR]
# dense causal mask [1,1,L,L]
mask = torch.full((L, L), torch.finfo(torch.float32).min)
mask = torch.triu(mask, diagonal=1).view(1, 1, L, L)
# dense prev_topk_indices for the "shared" layer: every query selects all L keys -> full attention
prev_topk = torch.arange(L).view(1, 1, L).expand(1, L, L).to(torch.int32).contiguous()

# ---- HF forward, replicating GlmMoeDsaDecoderLayer.forward but exposing router picks ----
residual = hidden
hn = layer.input_layernorm(hidden)
attn_out, _, _ = layer.self_attn(
    hidden_states=hn, position_embeddings=(cos.to(BF16), sin.to(BF16)),
    attention_mask=mask.to(BF16), position_ids=position_ids, prev_topk_indices=prev_topk)
h1 = residual + attn_out
xn2 = layer.post_attention_layernorm(h1)
_, topk_w, topk_idx = mlp.gate(xn2)                    # [L,TOPK]
expert_out = mlp.experts(xn2.view(-1, H), topk_idx, topk_w).view(1, L, H)
shared_out = mlp.shared_experts(xn2)
# f32 accumulation of the three bf16 terms, mirroring plow's f32 MOE_COMBINE (single rounding)
block_out = (h1.float() + expert_out.float() + shared_out.float()).to(BF16)

# cross-check against the real layer.forward
ref_out, _ = layer(hidden_states=hidden, position_embeddings=(cos.to(BF16), sin.to(BF16)),
                   attention_mask=mask.to(BF16), position_ids=position_ids, prev_topk_indices=prev_topk)
d = (ref_out.float() - block_out.float()).abs().max().item()
print(f"layer.forward vs manual block: max abs diff {d:.3e}  (should be ~0)")

sel_ids = topk_idx[qpos].to(torch.int32).cpu().numpy()
sel_gates = topk_w[qpos].float().cpu().numpy()
print("last-token router pick:", list(zip(sel_ids.tolist(), np.round(sel_gates, 4).tolist())))

# ---- absorbed MLA weights (f32 fold, then bf16 for plow) ----
def bf(x): return x.to(BF16)
q_b = sd.q_b_proj.weight.float().view(NH, QKH, QL)               # [NH,256,QL]
q_b_nope = q_b[:, :QN, :]                                        # [NH,192,QL]
q_b_rope = q_b[:, QN:, :]                                        # [NH,64,QL]
kv_b = sd.kv_b_proj.weight.float().view(NH, QN + VD, DK)         # [NH,448,512]
k_nope_w = kv_b[:, :QN, :]                                       # [NH,192,512]  (512->192)
value_w  = kv_b[:, QN:, :]                                       # [NH,256,512]  (512->256)
# Wqa[h] = W_uk_nope[h]^T @ q_b_nope[h] : [512,192]@[192,QL] -> [512,QL]
Wqa = torch.einsum('hpl,hpk->hlk', k_nope_w, q_b_nope).contiguous()   # [NH,DK,QL]
Wuv = value_w.transpose(-1, -2).contiguous()                          # [NH,DK,VD] (l-major)

# interleaved-RoPE rotation folded at position qpos into q_b_rope -> Wqr, and into k-rope down-proj
c = cos[0, qpos, :DR // 2].float().numpy()   # [32]
s = sin[0, qpos, :DR // 2].float().numpy()   # [32]
def fold_rope_rows(Wraw):  # Wraw [DR, K] -> RoPE'd [DR, K] at qpos (interleave formula)
    Wr = np.empty_like(Wraw)
    half = DR // 2
    for i in range(half):
        Wr[i]        = c[i] * Wraw[2 * i]     - s[i] * Wraw[2 * i + 1]
        Wr[half + i] = s[i] * Wraw[2 * i]     + c[i] * Wraw[2 * i + 1]
    return Wr
Wqr = np.stack([fold_rope_rows(q_b_rope[h].numpy()) for h in range(NH)], 0)  # [NH,DR,QL]
kv_a_w = sd.kv_a_proj_with_mqa.weight.float().numpy()          # [576,H]
W_ckv_down = kv_a_w[:DK]                                       # [512,H]
W_krot_folded = fold_rope_rows(kv_a_w[DK:DK + DR])             # [64,H] RoPE'd at qpos

# ---- c_kv / k_rot caches from the NORMED hidden sequence (hn = input_layernorm(hidden)),
# matching HF: the attention projects the input-layernorm output, not the raw residual ----
compressed = (hn.float() @ sd.kv_a_proj_with_mqa.weight.float().T)[0]        # [L,576]
ckv_raw = compressed[:, :DK]
krot_raw = compressed[:, DK:DK + DR]
var = ckv_raw.pow(2).mean(-1, keepdim=True)
c_kv = (ckv_raw * torch.rsqrt(var + EPS) * sd.kv_a_layernorm.weight.float())  # [L,512]
# per-position interleaved rope on k_rot_raw
kr = krot_raw.view(1, 1, L, DR)
_, kr_rot = apply_rotary_pos_emb_interleave(kr, kr, cos.float(), sin.float())
k_rot = kr_rot[0, 0]                                          # [L,64]

# ---- write fixture ----
def w_bf(f, t): f.write(np.ascontiguousarray(t.to(BF16).view(torch.uint16).cpu().numpy()).tobytes())
def w_bf_np(f, a): f.write(np.ascontiguousarray(
        torch.from_numpy(np.ascontiguousarray(a)).to(BF16).view(torch.uint16).numpy()).tobytes())
def w_f32(f, a): f.write(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes())
def w_i32(f, a): f.write(np.ascontiguousarray(np.asarray(a, np.int32)).tobytes())

with open(OUT, "wb") as f:
    f.write(struct.pack("<13i", 0x474C4D35, L, H, NH, DK, DR, QN, VD, QL, E, TOPK, IMOE, qpos))
    f.write(struct.pack("<3f", EPS, SCALE, RSCALE))
    w_bf(f, hidden[0, qpos])                                   # x_last [H]
    w_bf(f, layer.input_layernorm.weight)                     # g_input [H]
    w_bf(f, sd.q_a_proj.weight)                               # [QL,H]
    w_bf(f, sd.q_a_layernorm.weight)                          # [QL]
    w_bf_np(f, Wqa.reshape(NH * DK, QL).numpy())              # [NH*DK,QL]
    w_bf_np(f, Wqr.reshape(NH * DR, QL))                      # [NH*DR,QL]
    w_bf_np(f, W_ckv_down)                                    # [DK,H]
    w_bf(f, sd.kv_a_layernorm.weight)                        # [DK]
    w_bf_np(f, W_krot_folded)                                 # [DR,H]
    w_bf_np(f, Wuv.reshape(NH * DK, VD).numpy())             # [NH*DK,VD]
    w_bf(f, sd.o_proj.weight)                                 # [H,NH*VD]
    w_bf(f, layer.post_attention_layernorm.weight)           # g_post [H]
    w_bf(f, c_kv)                                             # [L,DK]
    w_bf(f, k_rot)                                            # [L,DR]
    w_bf(f, mlp.gate.weight)                                  # [E,H]
    w_f32(f, mlp.gate.e_score_correction_bias.float().cpu().numpy())  # [E]
    gup = mlp.experts.gate_up_proj.float()                    # [E,2IMOE,H]
    w_bf(f, gup[:, :IMOE, :].reshape(E * IMOE, H))            # gate [E*IMOE,H]
    w_bf(f, gup[:, IMOE:, :].reshape(E * IMOE, H))            # up   [E*IMOE,H]
    w_bf(f, mlp.experts.down_proj.float().reshape(E * H, IMOE))  # down [E*H,IMOE]
    w_bf(f, mlp.shared_experts.gate_proj.weight)             # [IMOE,H]
    w_bf(f, mlp.shared_experts.up_proj.weight)               # [IMOE,H]
    w_bf(f, mlp.shared_experts.down_proj.weight)             # [H,IMOE]
    # diff targets / substeps (last token)
    w_bf(f, block_out[0, qpos])                              # block_out_hf [H]
    w_i32(f, sel_ids)                                        # [TOPK]
    w_f32(f, sel_gates)                                      # [TOPK]
    w_bf(f, attn_out[0, qpos])                               # attn_out_hf [H]
    w_bf(f, xn2[0, qpos])                                    # xn2_hf [H]
    w_bf(f, h1[0, qpos])                                     # h1_hf [H]

print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
