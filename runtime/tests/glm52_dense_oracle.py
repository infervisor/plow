#!/usr/bin/env python3
# glm52_dense_oracle.py — HF-transformers reference for a GLM-5.2 DENSE decoder layer (0-2).  [GLM52-B3]
#
# The dense layers (first_k_dense_replace=3) run the SAME MLA attention as the MoE layers but a plain
# block-fp8 SwiGLU MLP (intermediate 12288) instead of the router+experts. This dumps a fixture the
# GPU loader (glm52_run.c, dense mode) diffs the EMITTED dense block against: input hidden, the pre-
# rotated latent/rope cache, and the HF references (attn_out, xn2, the dense-FFN output in f32, and
# the block output). The dense-FFN output is diffed DIRECTLY in f32 (like the MoE expert_sum) so the
# block-fp8 quant error is visible rather than hidden under the residual.
#
# Reuses glm52_real_oracle.py's shard reader + block-fp8 dequant. Runs CPU-only, low memory (the dense
# MLP is ~0.45 GB bf16). Fixture magic "GLM7" (0x474C4D37).
import os, struct, sys, json, mmap
import numpy as np
import torch

torch.manual_seed(1234); np.random.seed(1234); torch.set_grad_enabled(False)
MODEL_DIR = os.environ.get("GLM_MODEL_DIR", "/home/lava/models/GLM-5.2-FP8")
OUT = sys.argv[1] if len(sys.argv) > 1 else "glm52_dense_fixture.bin"
L = int(os.environ.get("GLM_L", "512"))
LAYER = int(os.environ.get("GLM_LAYER", "0"))   # dense layer
BF16 = torch.bfloat16

_DT = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
def _index_shards(md):
    idx = {}
    for fn in sorted(os.listdir(md)):
        if not (fn.startswith("model-") and fn.endswith(".safetensors")): continue
        path = os.path.join(md, fn)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__": continue
            idx[k] = (path, base + v["data_offsets"][0], base + v["data_offsets"][1], v["dtype"], v["shape"])
    return idx
_MM = {}; _FH = {}
def _mm(p):
    if p not in _MM:
        _FH[p] = open(p, "rb")                       # keep the file alive; mmap needs a live fd
        _MM[p] = mmap.mmap(_FH[p].fileno(), 0, prot=mmap.PROT_READ)
    return _MM[p]
def load_tensor(idx, name):
    path, a, b, dt, shape = idx[name]
    return torch.frombuffer(bytearray(_mm(path)[a:b]), dtype=_DT[dt]).view(*shape)
def dequant_blockfp8(idx, wn):
    w = load_tensor(idx, wn).float(); s = load_tensor(idx, wn + "_scale_inv").float()
    N, K = w.shape
    return w * s[torch.arange(N) // 128][:, torch.arange(K) // 128]

from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer, GlmMoeDsaRotaryEmbedding, apply_rotary_pos_emb_interleave)

cfg = AutoConfig.from_pretrained(MODEL_DIR)
H, NH, DK, DR, QN, VD, QL = (cfg.hidden_size, cfg.num_attention_heads, cfg.kv_lora_rank,
    cfg.qk_rope_head_dim, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.q_lora_rank)
QKH, DI, EPS = cfg.qk_head_dim, cfg.intermediate_size, cfg.rms_norm_eps
SCALE = QKH ** -0.5; qpos = L - 1
assert cfg.mlp_layer_types[LAYER] == "dense", cfg.mlp_layer_types[LAYER]
idx = _index_shards(MODEL_DIR); P = f"model.layers.{LAYER}."
print(f"[dense] layer={LAYER} H={H} NH={NH} DI={DI} L={L} scale={SCALE}")

cfg_i = AutoConfig.from_pretrained(MODEL_DIR); cfg_i._attn_implementation = "eager"
layer = GlmMoeDsaDecoderLayer(cfg_i, LAYER).eval()
rot = GlmMoeDsaRotaryEmbedding(cfg)
def cp(param, t): param.copy_(t.to(param.dtype).reshape(param.shape))
sd = layer.self_attn
cp(sd.q_a_proj.weight,           dequant_blockfp8(idx, P+"self_attn.q_a_proj.weight"))
cp(sd.q_a_layernorm.weight,      load_tensor(idx, P+"self_attn.q_a_layernorm.weight"))
cp(sd.q_b_proj.weight,           dequant_blockfp8(idx, P+"self_attn.q_b_proj.weight"))
cp(sd.kv_a_proj_with_mqa.weight, dequant_blockfp8(idx, P+"self_attn.kv_a_proj_with_mqa.weight"))
cp(sd.kv_a_layernorm.weight,     load_tensor(idx, P+"self_attn.kv_a_layernorm.weight"))
cp(sd.kv_b_proj.weight,          dequant_blockfp8(idx, P+"self_attn.kv_b_proj.weight"))
cp(sd.o_proj.weight,             dequant_blockfp8(idx, P+"self_attn.o_proj.weight"))
cp(layer.input_layernorm.weight,          load_tensor(idx, P+"input_layernorm.weight"))
cp(layer.post_attention_layernorm.weight, load_tensor(idx, P+"post_attention_layernorm.weight"))
layer = layer.to(BF16)

g = torch.Generator().manual_seed(0xB3)
hidden = (0.3 * torch.randn(1, L, H, generator=g)).to(BF16)
pids = torch.arange(L).unsqueeze(0)
cos, sin = rot(hidden.float(), pids)
mask = torch.triu(torch.full((L, L), torch.finfo(torch.float32).min), 1).view(1, 1, L, L)
prev_topk = torch.arange(L).view(1, 1, L).expand(1, L, L).to(torch.int32).contiguous()

hn = layer.input_layernorm(hidden)
attn_out, _, _ = layer.self_attn(hidden_states=hn, position_embeddings=(cos.to(BF16), sin.to(BF16)),
    attention_mask=mask.to(BF16), position_ids=pids, prev_topk_indices=prev_topk)
h1 = hidden + attn_out
xn2 = layer.post_attention_layernorm(h1)
xq = xn2[0, qpos].to(BF16).float()                       # dense FFN input (bf16 activation, w8a16)

# dense FFN reference (f32 accumulate over dequantised block-fp8 weights) — the direct fp8 de-risk
Wg = dequant_blockfp8(idx, P+"mlp.gate_proj.weight")     # [DI,H]
Wu = dequant_blockfp8(idx, P+"mlp.up_proj.weight")       # [DI,H]
Wd = dequant_blockfp8(idx, P+"mlp.down_proj.weight")     # [H,DI]
act = torch.nn.functional.silu(xq @ Wg.T) * (xq @ Wu.T)  # [DI]
dffn = (act @ Wd.T)                                      # [H] f32
block_out = (h1[0, qpos].float() + dffn).to(BF16)

# latent/rope cache (same construction as the MoE oracle; k_rot half-split HF layout)
W_ckv = sd.kv_a_proj_with_mqa.weight.float()
compressed = (hn.float() @ W_ckv.T)[0]
ckv_raw, krot_raw = compressed[:, :DK], compressed[:, DK:DK+DR]
c_kv = ckv_raw * torch.rsqrt(ckv_raw.pow(2).mean(-1, keepdim=True) + EPS) * sd.kv_a_layernorm.weight.float()
_, kr = apply_rotary_pos_emb_interleave(krot_raw.view(1,1,L,DR), krot_raw.view(1,1,L,DR), cos.float(), sin.float())
k_rot = kr[0, 0]

def w_bf(f, t): f.write(np.ascontiguousarray(t.to(BF16).view(torch.uint16).cpu().numpy()).tobytes())
def w_f32(f, a): f.write(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes())
with open(OUT, "wb") as f:
    f.write(struct.pack("<11i", 0x474C4D37, L, H, NH, DK, DR, QN, VD, QL, DI, qpos))
    f.write(struct.pack("<2f", EPS, SCALE))
    w_bf(f, hidden[0, qpos]); w_bf(f, c_kv); w_bf(f, k_rot)
    w_bf(f, attn_out[0, qpos]); w_bf(f, xn2[0, qpos]); w_bf(f, block_out)
    w_f32(f, dffn.numpy())
    sz = f.tell()
print(f"[dense] wrote {OUT} ({sz/1e6:.1f} MB): attn+xn2+block_out(bf16)+dffn(f32) refs, cache, input")
