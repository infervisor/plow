#!/usr/bin/env python3
# glm52_real_oracle.py — REAL-WEIGHT HF-transformers ORACLE for the GLM-5.2 (GlmMoeDsa) single
# MoE layer B4-core de-risk.  [GLM52-B4]
#
# The B1 oracle used SYNTHETIC bf16 weights and E=16 experts. This one loads the REAL
# zai-org/GLM-5.2-FP8 [128,128]-block-fp8 weights for ONE real MoE layer (layer 3: sparse MLP +
# "shared" indexer -> DSA indexer is a no-op, dense causal attention for L<=2048) with the REAL
# 256 experts, and dumps a fixture the GPU harness (glm52_real_block_gfx950_test.c) diffs plow
# against. This RECONFIRMS what B1 simplified:
#   (a) the 8-of-256 top-k SELECTION (bf16 device dot vs HF fp32) — the flip margin is tighter at
#       256 than at 16; we compute BOTH a fp32 and a bf16 router arm and REPORT any flip + margin,
#   (b) the REAL block-fp8 scale layout (HF weight_scale_inv [N/128,K/128] grids fed verbatim to
#       plow's Sg/Sd tables — no transpose; scale is a direct multiplier).
#
# MEMORY: the box is RAM-limited (~3GB) and torch here is CPU-only, so the 256 experts (~19GB bf16)
# can NEVER be materialised at once. We:
#   - instantiate GlmMoeDsaDecoderLayer with n_routed_experts=1 (the big expert module stays tiny)
#     and copy the REAL (dequantised) attention + layernorm weights into it -> HF self_attn is the
#     TRUSTED attention reference at real weights,
#   - compute the router MANUALLY over the real [256,6144] gate weight (fp32 AND bf16 arms),
#   - dequantise ONLY the top-8 selected experts + shared for the reference block output,
#   - STREAM all 256 real fp8 experts + real scale grids from the safetensors straight into the
#     fixture (copy fp8 bytes; never dequantise them on host).
# INPUT is a seeded synthetic hidden state (we have no real layer-3 activations without running
# layers 0-2); the WEIGHTS are real. The de-risk is about weight layout + 256-wide selection, not
# prompt-specific activations — documented, not hidden.
import os, struct, sys, json, mmap
import numpy as np
import torch

torch.manual_seed(1234)
np.random.seed(1234)
torch.set_grad_enabled(False)

MODEL_DIR = os.environ.get("GLM_MODEL_DIR", "/home/lava/models/GLM-5.2-FP8")
OUT = sys.argv[1] if len(sys.argv) > 1 else "glm52_real_fixture.bin"
L = int(os.environ.get("GLM_L", "512"))          # dense context (<=2048 -> indexer no-op)
LAYER = int(os.environ.get("GLM_LAYER", "3"))    # first sparse layer, "shared" indexer
BF16 = torch.bfloat16

# ---------------------------------------------------------------- safetensors reader (mmap, lazy)
_DT = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F32": torch.float32,
       "F16": torch.float16}
def _index_shards(model_dir):
    idx = {}
    for fn in sorted(os.listdir(model_dir)):
        if not (fn.startswith("model-") and fn.endswith(".safetensors")):
            continue
        path = os.path.join(model_dir, fn)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            idx[k] = (path, base + v["data_offsets"][0], base + v["data_offsets"][1],
                      v["dtype"], v["shape"])
    return idx

_MM = {}
def _mm(path):
    if path not in _MM:
        f = open(path, "rb")
        _MM[path] = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    return _MM[path]

def raw_bytes(idx, name):
    path, a, b, dt, shape = idx[name]
    return _mm(path)[a:b], dt, shape

def load_tensor(idx, name):
    buf, dt, shape = raw_bytes(idx, name)
    t = torch.frombuffer(bytearray(buf), dtype=_DT[dt]).view(*shape)
    return t

def dequant_blockfp8(idx, wname):
    """Real GLM block-fp8 dequant: w_dequant = fp8.float() * weight_scale_inv[i//128, j//128].
    Block-index gather (robust to the ceil grid, e.g. kv_a's 576 rows -> 5 row-blocks)."""
    w = load_tensor(idx, wname).float()                       # [N,K]
    s = load_tensor(idx, wname + "_scale_inv").float()        # [ceil(N/128), ceil(K/128)]
    N, K = w.shape
    rb = torch.arange(N) // 128
    cb = torch.arange(K) // 128
    sf = s[rb][:, cb]                                          # [N,K]
    return w * sf

# ---------------------------------------------------------------- config + real layer-3 tensors
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer, GlmMoeDsaRotaryEmbedding, apply_rotary_pos_emb_interleave,
)

cfg = AutoConfig.from_pretrained(MODEL_DIR)
E   = cfg.n_routed_experts        # 256
H   = cfg.hidden_size             # 6144
NH  = cfg.num_attention_heads     # 64
DK  = cfg.kv_lora_rank            # 512
DR  = cfg.qk_rope_head_dim        # 64
QN  = cfg.qk_nope_head_dim        # 192
VD  = cfg.v_head_dim              # 256
QL  = cfg.q_lora_rank             # 2048
QKH = cfg.qk_head_dim             # 256
IMOE= cfg.moe_intermediate_size   # 2048
TOPK= cfg.num_experts_per_tok     # 8
EPS = cfg.rms_norm_eps
SCALE = QKH ** -0.5               # 1/sqrt(256) = 0.0625
RSCALE = cfg.routed_scaling_factor
qpos = L - 1
assert cfg.mlp_layer_types[LAYER] == "sparse" and cfg.indexer_types[LAYER] == "shared", \
    (cfg.mlp_layer_types[LAYER], cfg.indexer_types[LAYER])
print(f"[real] E={E} H={H} NH={NH} DK={DK} DR={DR} QN={QN} VD={VD} QL={QL} IMOE={IMOE} "
      f"TOPK={TOPK} L={L} layer={LAYER} scale={SCALE} rscale={RSCALE} eps={EPS}")

idx = _index_shards(MODEL_DIR)
P = f"model.layers.{LAYER}."
need = [P+"self_attn.q_a_proj.weight", P+"mlp.gate.weight", P+"mlp.experts.0.gate_proj.weight",
        P+"mlp.experts.255.down_proj.weight", P+"mlp.shared_experts.gate_proj.weight"]
miss = [n for n in need if n not in idx]
assert not miss, f"real layer-{LAYER} tensors missing (download incomplete): {miss}"

# ---------------------------------------------------------------- instantiate layer (tiny experts)
cfg_i = AutoConfig.from_pretrained(MODEL_DIR)
cfg_i.n_routed_experts = 1
cfg_i.num_experts_per_tok = 1
cfg_i._attn_implementation = "eager"
layer = GlmMoeDsaDecoderLayer(cfg_i, LAYER).eval()
rot = GlmMoeDsaRotaryEmbedding(cfg)

def cp(param, t):
    param.copy_(t.to(param.dtype).reshape(param.shape))

sd = layer.self_attn
cp(sd.q_a_proj.weight,          dequant_blockfp8(idx, P+"self_attn.q_a_proj.weight"))
cp(sd.q_a_layernorm.weight,     load_tensor(idx, P+"self_attn.q_a_layernorm.weight"))
cp(sd.q_b_proj.weight,          dequant_blockfp8(idx, P+"self_attn.q_b_proj.weight"))
cp(sd.kv_a_proj_with_mqa.weight,dequant_blockfp8(idx, P+"self_attn.kv_a_proj_with_mqa.weight"))
cp(sd.kv_a_layernorm.weight,    load_tensor(idx, P+"self_attn.kv_a_layernorm.weight"))
cp(sd.kv_b_proj.weight,         dequant_blockfp8(idx, P+"self_attn.kv_b_proj.weight"))
cp(sd.o_proj.weight,            dequant_blockfp8(idx, P+"self_attn.o_proj.weight"))
cp(layer.input_layernorm.weight,          load_tensor(idx, P+"input_layernorm.weight"))
cp(layer.post_attention_layernorm.weight, load_tensor(idx, P+"post_attention_layernorm.weight"))
layer = layer.to(BF16)

# router weight + correction bias (bf16 / f32 on disk; NOT fp8)
Wr   = load_tensor(idx, P+"mlp.gate.weight").float()                       # [E,H]
bias = load_tensor(idx, P+"mlp.gate.e_score_correction_bias").float()      # [E]

# ---------------------------------------------------------------- inputs (seeded), rope, mask
g = torch.Generator().manual_seed(0xB4)
hidden = (0.3 * torch.randn(1, L, H, generator=g)).to(BF16)
position_ids = torch.arange(L).unsqueeze(0)
cos, sin = rot(hidden.float(), position_ids)                              # [1,L,DR]
mask = torch.triu(torch.full((L, L), torch.finfo(torch.float32).min), 1).view(1, 1, L, L)
prev_topk = torch.arange(L).view(1, 1, L).expand(1, L, L).to(torch.int32).contiguous()

# ---------------------------------------------------------------- HF attention (TRUSTED, real wts)
residual = hidden
hn = layer.input_layernorm(hidden)
attn_out, _, _ = layer.self_attn(
    hidden_states=hn, position_embeddings=(cos.to(BF16), sin.to(BF16)),
    attention_mask=mask.to(BF16), position_ids=position_ids, prev_topk_indices=prev_topk)
h1 = residual + attn_out
xn2 = layer.post_attention_layernorm(h1)
xq = xn2[0, qpos].float()                                                 # [H] last-token router in

# ---------------------------------------------------------------- ROUTER: fp32 vs bf16 (flip test)
def route(x_row, wr, dtype):
    logits = (x_row.to(dtype) @ wr.to(dtype).T).float()                   # [E]
    scores = torch.sigmoid(logits)
    sel = torch.topk(scores + bias, TOPK).indices                        # bias only for SELECTION
    g_sel = scores[sel]                                                   # gate = UNBIASED score
    g_sel = g_sel / g_sel.sum() * RSCALE                                 # norm_topk then scale
    return sel, g_sel, scores

sel32, gate32, sc32 = route(xq, Wr, torch.float32)
selb,  gateb,  scb  = route(xq, Wr, torch.bfloat16)
set32, setb = set(sel32.tolist()), set(selb.tolist())
# selection margin: gap between the 8th-in and 9th-out biased score (fp32)
biased = (sc32 + bias)
srt = torch.sort(biased, descending=True).values
margin = (srt[TOPK - 1] - srt[TOPK]).item()
print("\n=== ROUTER 256-wide flip test (last token) ===")
print(f"  fp32 top-8 (id:gate): {[(int(i), round(float(g),4)) for i,g in zip(sel32, gate32)]}")
print(f"  bf16 top-8 ids      : {sorted(selb.tolist())}")
print(f"  fp32 top-8 ids      : {sorted(sel32.tolist())}")
print(f"  SET flip bf16 vs fp32: {'*** FLIP ***' if set32 != setb else 'MATCH'}"
      f"   (selection margin 8th-9th biased score = {margin:.3e})")

# reference uses the fp32 selection (the ground truth); plow's device router is diffed against it
sel = sel32
sel_np = sel.to(torch.int32).cpu().numpy()
gate_np = gate32.float().cpu().numpy()

# ---------------------------------------------------------------- reference experts (top-8 + shared)
def expert_fwd(x_row, e):
    gw = dequant_blockfp8(idx, P+f"mlp.experts.{e}.gate_proj.weight")     # [IMOE,H]
    uw = dequant_blockfp8(idx, P+f"mlp.experts.{e}.up_proj.weight")       # [IMOE,H]
    dw = dequant_blockfp8(idx, P+f"mlp.experts.{e}.down_proj.weight")     # [H,IMOE]
    a = torch.nn.functional.silu(x_row @ gw.T) * (x_row @ uw.T)           # [IMOE]
    return a @ dw.T                                                       # [H]
def shared_fwd(x_row):
    gw = dequant_blockfp8(idx, P+"mlp.shared_experts.gate_proj.weight")
    uw = dequant_blockfp8(idx, P+"mlp.shared_experts.up_proj.weight")
    dw = dequant_blockfp8(idx, P+"mlp.shared_experts.down_proj.weight")
    return torch.nn.functional.silu(x_row @ gw.T) * (x_row @ uw.T) @ dw.T, (gw, uw, dw)

xq_bf = xn2[0, qpos].to(BF16).float()      # experts see bf16 activations (w8a16)
expert_sum = torch.zeros(H)
for i, e in enumerate(sel.tolist()):
    expert_sum += float(gate_np[i]) * expert_fwd(xq_bf, e)
shared_out, (shg, shu, shd) = shared_fwd(xq_bf)
# f32 accumulate the three bf16 terms (single rounding), matching plow's f32 MOE_COMBINE
block_out = (h1[0, qpos].float() + expert_sum + shared_out).to(BF16)
print(f"\n  ref block_out norm={block_out.float().norm():.3f}  "
      f"shared_out norm={shared_out.norm():.3f}  expert_sum norm={expert_sum.norm():.3f}")

# ---------------------------------------------------------------- absorbed MLA weights + caches
q_b = sd.q_b_proj.weight.float().view(NH, QKH, QL)
q_b_nope, q_b_rope = q_b[:, :QN, :], q_b[:, QN:, :]
kv_b = sd.kv_b_proj.weight.float().view(NH, QN + VD, DK)
k_nope_w, value_w = kv_b[:, :QN, :], kv_b[:, QN:, :]
Wqa = torch.einsum('hpl,hpk->hlk', k_nope_w, q_b_nope).contiguous()       # [NH,DK,QL]
Wuv = value_w.transpose(-1, -2).contiguous()                             # [NH,DK,VD]
c = cos[0, qpos, :DR // 2].float().numpy(); s = sin[0, qpos, :DR // 2].float().numpy()
def fold_rope_rows(Wraw):
    Wr_ = np.empty_like(Wraw); half = DR // 2
    for i in range(half):
        Wr_[i]        = c[i] * Wraw[2*i] - s[i] * Wraw[2*i+1]
        Wr_[half + i] = s[i] * Wraw[2*i] + c[i] * Wraw[2*i+1]
    return Wr_
Wqr = np.stack([fold_rope_rows(q_b_rope[h].numpy()) for h in range(NH)], 0)   # [NH,DR,QL]
kv_a_w = sd.kv_a_proj_with_mqa.weight.float().numpy()
W_ckv_down = kv_a_w[:DK]
W_krot_folded = fold_rope_rows(kv_a_w[DK:DK+DR])
compressed = (hn.float() @ sd.kv_a_proj_with_mqa.weight.float().T)[0]
ckv_raw, krot_raw = compressed[:, :DK], compressed[:, DK:DK+DR]
var = ckv_raw.pow(2).mean(-1, keepdim=True)
c_kv = ckv_raw * torch.rsqrt(var + EPS) * sd.kv_a_layernorm.weight.float()
kr = krot_raw.view(1, 1, L, DR)
_, kr_rot = apply_rotary_pos_emb_interleave(kr, kr, cos.float(), sin.float())
k_rot = kr_rot[0, 0]

# ---------------------------------------------------------------- write fixture v2
def w_bf(f, t):    f.write(np.ascontiguousarray(t.to(BF16).view(torch.uint16).cpu().numpy()).tobytes())
def w_bf_np(f, a): f.write(np.ascontiguousarray(torch.from_numpy(np.ascontiguousarray(a)).to(BF16).view(torch.uint16).numpy()).tobytes())
def w_f32(f, a):   f.write(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes())
def w_i32(f, a):   f.write(np.ascontiguousarray(np.asarray(a, np.int32)).tobytes())

IB, HB = IMOE // 128, H // 128
sz = 0
with open(OUT, "wb") as f:
    f.write(struct.pack("<13i", 0x474C4D36, L, H, NH, DK, DR, QN, VD, QL, E, TOPK, IMOE, qpos))
    f.write(struct.pack("<3f", EPS, SCALE, RSCALE))
    w_bf(f, hidden[0, qpos]); w_bf(f, layer.input_layernorm.weight)
    w_bf(f, sd.q_a_proj.weight); w_bf(f, sd.q_a_layernorm.weight)
    w_bf_np(f, Wqa.reshape(NH*DK, QL).numpy()); w_bf_np(f, Wqr.reshape(NH*DR, QL))
    w_bf_np(f, W_ckv_down); w_bf(f, sd.kv_a_layernorm.weight); w_bf_np(f, W_krot_folded)
    w_bf_np(f, Wuv.reshape(NH*DK, VD).numpy()); w_bf(f, sd.o_proj.weight)
    w_bf(f, layer.post_attention_layernorm.weight)
    w_bf(f, c_kv); w_bf(f, k_rot)
    w_bf(f, Wr.to(BF16)); w_f32(f, bias.cpu().numpy())
    # experts fp8 (streamed, raw bytes, per expert: gate,up,down), then scales (per expert: Sg,Su,Sd)
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            buf, dt, shape = raw_bytes(idx, P+f"mlp.experts.{e}.{proj}.weight")
            assert dt == "F8_E4M3", (e, proj, dt)
            f.write(bytes(buf))
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            s_ = load_tensor(idx, P+f"mlp.experts.{e}.{proj}.weight_scale_inv").float().contiguous()
            f.write(np.ascontiguousarray(s_.numpy(), np.float32).tobytes())
    # shared expert (dequantised to bf16 for the harness bf16 shared-MLP path)
    w_bf(f, shg); w_bf(f, shu); w_bf(f, shd)
    # reference / diff targets
    w_bf(f, block_out); w_i32(f, sel_np); w_f32(f, gate_np)
    w_bf(f, attn_out[0, qpos]); w_bf(f, xn2[0, qpos]); w_bf(f, h1[0, qpos])
    # f32 expert-path references (the fp8 de-risk: MoE contribution is tiny in absolute terms —
    # small layer-3 gammas — so it MUST be diffed directly at f32, never via bf16 xnext-xmid).
    w_f32(f, expert_sum.numpy())          # Σ_j gate_j·expert_j(xn2)  [H] f32
    w_bf(f, shared_out.to(BF16))          # shared_experts(xn2)       [H] bf16
    sz = f.tell()
print(f"\nwrote {OUT} ({sz/1e9:.2f} GB)  IB={IB} HB={HB}  experts fp8 + real weight_scale_inv grids")
