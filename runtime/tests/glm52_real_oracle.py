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
#
# ---------------------------------------------------------------------------------------------
# GLM_T=<T>: the T-ROW **PREFILL** ARM.                                            [GLM52-PF-GATE]
#
# Set it and this script computes the SAME per-stage references for T query rows at positions
# 0..T-1 under a causal mask — attn_out / xmid / post-LN / shared / the f32 FFN contribution /
# block_out, EVERY ROW — and writes a small (~15 MB) companion fixture with magic "GLM8", which
# `glm52_run.c --prefill` diffs a prefill BUCKET program against. Nothing about the single-token
# path changes: with GLM_T unset the GLM6 fixture is written by the code below, byte for byte.
#
# The prefill fixture carries NO weights and NO kv cache. It cannot: a prefill bucket computes its
# own latent/rope cache rows (`out_row0 = 0` on both writers), and `glm52_run.c` binds every weight
# by name from the prepped dir. That also removes the half-split-vs-interleaved k_rope permutation
# the decode harness has to apply to fixture history — the kernel writes the cache it then reads.
#
# It also covers DENSE layers (< first_k_dense_replace), which had no T-row oracle at all: their
# prefill FFN runs on the GROUPED MoE arms with degenerate 1-expert routing, and that path had
# never been checked against a reference.
#
# ---------------------------------------------------------------------------------------------
# GLM_EXTRA_DIRS=<dir>[:<dir>]: extra safetensors dirs merged into the shard index.
#
# The `derived.*` weight-prep output (`scripts/glm52_prep.py`) REPLACES `q_b_proj` / `kv_b_proj`
# with their absorbed folds (q_absorb = k_nope_wᵀ·q_b_nope, v_absorb = value_wᵀ), and a product is
# not invertible — so a plow-prepped checkpoint alone cannot construct HF's `self_attn`, which is
# the whole trusted half of this oracle. Point GLM_EXTRA_DIRS at a directory holding just those two
# tensors (per layer) from the ORIGINAL fp8 checkpoint and the HF module is buildable again while
# every other weight comes from the dir plow itself runs on.
import os, struct, sys, json, mmap, argparse, copy
import numpy as np
import torch

torch.manual_seed(1234)
np.random.seed(1234)
torch.set_grad_enabled(False)

MODEL_DIR = os.environ.get("GLM_MODEL_DIR", "/home/lava/models/GLM-5.2-FP8")
EXTRA_DIRS = [d for d in os.environ.get("GLM_EXTRA_DIRS", "").split(":") if d]
parser = argparse.ArgumentParser()
parser.add_argument("output", nargs="?", default="glm52_real_fixture.bin")
parser.add_argument("--block-inputs", help="also export raw operands and the reference residual for amd-block")
parser.add_argument("--candidate-dir", help="diagnose amd-block rank0 dumps against HF; write no fixture")
parser.add_argument("--batch", type=int, default=1, help="independent decode sequences for block inputs")
parser.add_argument("--inputs-only", action="store_true", help="omit the legacy weight-bearing fixture")
parser.add_argument("--block-context", type=int, help="spread supplied selected keys across this logical context")
parser.add_argument("--shared-w8a8-tp", type=int, default=0,
                    help="use installed vLLM/AITER FP8 shared-expert reference with this TP degree")
parser.add_argument("--routed-w8a8", action="store_true",
                    help="also use pinned AITER routed experts and per-rank BF16 shared addition")
parser.add_argument("--vllm-post-norm", action="store_true",
                    help="use vLLM fused residual/RMSNorm on independent HF attention output")
args = parser.parse_args()
OUT = args.output
T = int(os.environ.get("GLM_T", "0"))            # >0 => the T-row prefill fixture (magic GLM8)
B = args.batch
if B < 1 or B > 64 or (B > 1 and (T > 0 or args.candidate_dir or not args.inputs_only)):
    parser.error("batch must be 1..64; batched capture requires --inputs-only and no prefill/candidate mode")
if args.inputs_only and (not args.block_inputs or T > 0 or args.candidate_dir):
    parser.error("--inputs-only requires --block-inputs and no prefill/candidate mode")
if args.shared_w8a8_tp not in (0, 1, 2, 4, 8) or (args.shared_w8a8_tp and not args.inputs_only):
    parser.error("--shared-w8a8-tp requires --inputs-only and TP 1, 2, 4, or 8")
if args.vllm_post_norm and not args.inputs_only:
    parser.error("--vllm-post-norm requires --inputs-only")
if args.routed_w8a8 and (not args.shared_w8a8_tp or not args.vllm_post_norm):
    parser.error("--routed-w8a8 requires --shared-w8a8-tp and --vllm-post-norm")
L = int(os.environ.get("GLM_L", "512"))          # dense context (<=2048 -> indexer no-op)
if T > 0:
    L = T                                        # prefill: the chunk IS the whole context
BLOCK_CTX = L if args.block_context is None else args.block_context
if BLOCK_CTX < L or (args.block_context is not None and (not args.inputs_only or L < 2)):
    parser.error("--block-context requires --inputs-only, at least two selected keys, and context >= GLM_L")
LAYER = int(os.environ.get("GLM_LAYER", "3"))    # first sparse layer, "shared" indexer
BF16 = torch.bfloat16

# ---------------------------------------------------------------- safetensors reader (mmap, lazy)
_DT = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F32": torch.float32,
       "F16": torch.float16}
def _index_shards(*model_dirs):
    idx = {}
    for model_dir in model_dirs:
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

def dq(idx, wname):
    """Block-fp8 dequant, or a straight f32 read when the tensor is already bf16.

    `scripts/glm52_prep.py` dequantises the PROJECTION weights (q_a_proj, o_proj, the shared
    expert) to bf16 while leaving the routed experts and the dense FFN as fp8 + `weight_scale_inv`.
    So on a plow-prepped dir some of these names have a scale grid and some do not, and the ones
    that do not are ALREADY the exact bytes plow will run — reading them verbatim is what keeps the
    oracle's weights and the harness's weights the same weights."""
    return (dequant_blockfp8(idx, wname) if (wname + "_scale_inv") in idx
            else load_tensor(idx, wname).float())

def kv_a_with_mqa(idx, P, DK, DR):
    """`kv_a_proj_with_mqa` [DK+DR, H] — from the checkpoint, or re-joined from the two halves
    weight-prep splits it into (`derived.kv_a_latent` [DK,H] + `derived.k_rope` [DR,H], both RAW:
    prep applies no position fold, `glm52_prep.py:prep_layer`)."""
    if (P + "self_attn.kv_a_proj_with_mqa.weight") in idx:
        return dq(idx, P + "self_attn.kv_a_proj_with_mqa.weight")
    a = dq(idx, P + "self_attn.derived.kv_a_latent.weight")
    b = dq(idx, P + "self_attn.derived.k_rope.weight")
    assert a.shape[0] == DK and b.shape[0] == DR, (a.shape, b.shape)
    return torch.cat([a, b], 0)

# ---------------------------------------------------------------- config + real layer-3 tensors
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer, GlmMoeDsaRotaryEmbedding,
)
try:
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        apply_rotary_pos_emb_interleave,
    )
except ImportError:
    # transformers 5.5.x dropped the interleaved helper from glm_moe_dsa and rewired BOTH the
    # indexer and the main attention onto the split-half `apply_rotary_pos_emb` -- contradicting
    # that file's own docstring ("the main attention uses interleaved RoPE") and the checkpoint's
    # `indexer_rope_interleave: true`. deepseek_v3 still carries the ORIGINAL function, byte-for-
    # byte, so import it there rather than reimplement a reference from memory. This keeps the
    # oracle's math identical to what the B4 gate was designed against.
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        apply_rotary_pos_emb_interleave,
    )

    # ...AND the layer's own attention must be put back on interleaved rope too, or the "trusted"
    # reference silently rotates 64 of every 256 head dims the wrong way. Measured, layer 3, real
    # weights: leaving 5.5.4's split-half in place fails the gate at attn_out rms 0.188 (tol
    # 1.5e-2) while post-LN / shared-expert / block-fp8 expert_sum / 256-wide router selection ALL
    # pass -- a defect localised to MLA, which is exactly where the rope slice lives.
    #
    # GROUND TRUTH is vLLM, which serves this checkpoint and produced our baseline CSVs.
    # `vllm/model_executor/models/deepseek_v2.py` (GlmMoeDsaForCausalLM maps there):
    #     line 987: is_neox_style=False                                   <- main MLA: INTERLEAVED
    #     line 1007: is_neox_style=not config.indexer_rope_interleave     <- True here: INTERLEAVED
    # So BOTH paths are interleaved for this checkpoint, plow's interp.hip (HD=64 interleaved,
    # indexer i[5]==1) is correct, and transformers 5.5.4 is the thing that regressed.
    import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as _glm

    def _rope_interleaved(x, cos, sin, unsqueeze_dim=1):
        """`apply_rotary_pos_emb` with GPT-J/interleaved pairing, shape-generic.

        Same identity deepseek_v3 uses: de-interleave (x0,x1,x2,x3) -> (x0,x2,x1,x3), then apply
        the ordinary split-half rotation. Kept shape-generic because the indexer calls this with
        [B,S,1,D] and unsqueeze_dim=2 while the main attention uses [B,H,S,D] and 1.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        d = x.shape[-1]
        x = x.reshape(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape)
        return (x * cos) + (_glm.rotate_half(x) * sin)

    _glm.apply_rotary_pos_emb = _rope_interleaved

# NOT bare AutoConfig: `GlmMoeDsaConfig` does not round-trip this checkpoint -- it reports
# qk_rope_head_dim = 192 where config.json (and the tensors: kv_a_proj_with_mqa is [576,6144]
# = 512+64) say 64. That is the same disagreement that killed the full-model prep, so reuse the
# reconciler that already fixed it there instead of writing a second one.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from glm52_prep import load_cfg

cfg = load_cfg(MODEL_DIR)
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
DI  = cfg.intermediate_size       # 12288 — dense-FFN width (layers < first_k_dense_replace)
DENSE = cfg.mlp_layer_types[LAYER] == "dense"
qpos = L - 1
# Full-indexer blocks are valid here only when every live key is selected.
if T == 0:
    assert cfg.mlp_layer_types[LAYER] == "sparse" and (cfg.indexer_types[LAYER] == "shared" \
        or (args.inputs_only and BLOCK_CTX == L and cfg.indexer_types[LAYER] == "full")), \
        (cfg.mlp_layer_types[LAYER], cfg.indexer_types[LAYER])
# T <= index_topk is what makes the DSA indexer a no-op: `GlmMoeDsaIndexer.forward` takes
# `topk = min(index_topk, total_len)`, so at total_len <= index_topk every position is selected and
# the scattered index_mask is all-zero — the combined mask is the plain causal one. Past that the
# reference would be sparse-attending with a randomly-initialised indexer, which is not a reference.
assert L <= cfg.index_topk, f"L={L} exceeds index_topk={cfg.index_topk}: the DSA indexer stops " \
                            f"being a no-op and this oracle's mask would be meaningless"
print(f"[real] E={E} H={H} NH={NH} DK={DK} DR={DR} QN={QN} VD={VD} QL={QL} IMOE={IMOE} DI={DI} "
      f"TOPK={TOPK} L={L} layer={LAYER} {'DENSE' if DENSE else 'sparse'} scale={SCALE} "
      f"rscale={RSCALE} eps={EPS}")

idx = _index_shards(MODEL_DIR, *EXTRA_DIRS)
P = f"model.layers.{LAYER}."
need = [P+"self_attn.q_a_proj.weight", P+"self_attn.q_b_proj.weight", P+"self_attn.kv_b_proj.weight"]
need += ([P+"mlp.gate_proj.weight"] if DENSE else
         [P+"mlp.gate.weight", P+"mlp.experts.0.gate_proj.weight",
          P+f"mlp.experts.{E-1}.down_proj.weight", P+"mlp.shared_experts.gate_proj.weight"])
miss = [n for n in need if n not in idx]
assert not miss, (f"real layer-{LAYER} tensors missing: {miss}. q_b_proj/kv_b_proj are ABSORBED "
                  f"away by weight-prep — point GLM_EXTRA_DIRS at a dir holding them.")

# ---------------------------------------------------------------- instantiate layer (tiny experts)
cfg_i = load_cfg(MODEL_DIR)   # reconciled, same as `cfg` -- a bare AutoConfig here sizes
                              # kv_a_proj_with_mqa as [512+192, H] and the real [576, H] tensor
                              # will not reshape into it.
cfg_i.n_routed_experts = 1
cfg_i.num_experts_per_tok = 1
cfg_i.intermediate_size = 8    # the module's DENSE MLP is never used (the FFN reference below is
                               # computed from the real weights); at 12288 it is 0.9 GB of random
                               # init this script would allocate and throw away.
cfg_i._attn_implementation = "eager"
layer = GlmMoeDsaDecoderLayer(cfg_i, LAYER).eval()
# New Transformers rotary classes use head_dim, while GLM rotates only DR channels.
rope_cfg = copy.deepcopy(cfg)
rope_cfg.head_dim = DR
rot = GlmMoeDsaRotaryEmbedding(rope_cfg)

def cp(param, t):
    param.copy_(t.to(param.dtype).reshape(param.shape))

sd = layer.self_attn
cp(sd.q_a_proj.weight,          dq(idx, P+"self_attn.q_a_proj.weight"))
cp(sd.q_a_layernorm.weight,     load_tensor(idx, P+"self_attn.q_a_layernorm.weight"))
cp(sd.q_b_proj.weight,          dq(idx, P+"self_attn.q_b_proj.weight"))
cp(sd.kv_a_proj_with_mqa.weight,kv_a_with_mqa(idx, P, DK, DR))
cp(sd.kv_a_layernorm.weight,    load_tensor(idx, P+"self_attn.kv_a_layernorm.weight"))
cp(sd.kv_b_proj.weight,         dq(idx, P+"self_attn.kv_b_proj.weight"))
cp(sd.o_proj.weight,            dq(idx, P+"self_attn.o_proj.weight"))
cp(layer.input_layernorm.weight,          load_tensor(idx, P+"input_layernorm.weight"))
cp(layer.post_attention_layernorm.weight, load_tensor(idx, P+"post_attention_layernorm.weight"))
layer = layer.to(BF16)

# router weight + correction bias (bf16 / f32 on disk; NOT fp8). A dense layer has no router.
Wr   = None if DENSE else load_tensor(idx, P+"mlp.gate.weight").float()                  # [E,H]
bias = None if DENSE else load_tensor(idx, P+"mlp.gate.e_score_correction_bias").float() # [E]

# ---------------------------------------------------------------- inputs (seeded), rope, mask
g = torch.Generator().manual_seed(0xB4)
hidden = (0.3 * torch.randn(B, L, H, generator=g)).to(BF16)
logical_positions = torch.arange(L)
if BLOCK_CTX != L:
    logical_positions = logical_positions * (BLOCK_CTX - 1) // (L - 1)
position_ids = logical_positions.unsqueeze(0).expand(B, -1)
cos, sin = rot(hidden.float(), position_ids)                              # [1,L,DR]
mask = torch.triu(torch.full((L, L), torch.finfo(torch.float32).min), 1).view(1, 1, L, L)
prev_topk = torch.arange(L).view(1, 1, L).expand(B, L, L).to(torch.int32).contiguous()

# ---------------------------------------------------------------- HF attention (TRUSTED, real wts)
residual = hidden
hn = layer.input_layernorm(hidden)
# transformers 4.x returned (out, weights, past); 5.x returns (out, weights). Take [0] rather
# than pin an arity, so this reference survives the next signature change.
attn_out = layer.self_attn(
    hidden_states=hn, position_embeddings=(cos.to(BF16), sin.to(BF16)),
    attention_mask=mask.to(BF16), position_ids=position_ids, prev_topk_indices=prev_topk)[0]
h1 = residual + attn_out
xn2 = layer.post_attention_layernorm(h1)
post_norm_version = None
if args.vllm_post_norm:
    from vllm import __version__ as post_norm_version
    from vllm.kernels.aiter_ops import fused_add_rms_norm
    if not post_norm_version.startswith("0.29."):
        raise ValueError("post-attention normalization reference requires vLLM 0.29")
    def post_norm():
        return fused_add_rms_norm.impl_fn(attn_out[:, qpos].contiguous().cuda(),
            residual[:, qpos].contiguous().cuda(), layer.post_attention_layernorm.weight.cuda(), EPS)
    norm, norm_residual = post_norm()
    repeat, repeat_residual = post_norm()
    if (not torch.isfinite(norm).all() or not torch.equal(norm.view(torch.int16), repeat.view(torch.int16))
            or not torch.equal(norm_residual.view(torch.int16), repeat_residual.view(torch.int16))
            or not torch.equal(norm_residual.cpu().view(torch.int16), h1[:, qpos].view(torch.int16))):
        raise ValueError("post-attention norm is unstable or changes the reference residual")
    xn2[:, qpos] = norm.cpu()

# =================================================================== T-ROW PREFILL FIXTURE (GLM8)
# Everything above is row-generic already — `hidden` is [1,L,H] and HF attends over the whole
# causal square — so the prefill arm is the SAME references, sliced at every row instead of at
# `qpos`. What is genuinely new is the FFN at T rows: each token routes independently (a 256-wide
# top-8 per row), and the dense layers have an FFN reference for the first time.
def w_bf_(f, t):  f.write(np.ascontiguousarray(t.to(BF16).view(torch.uint16).cpu().numpy()).tobytes())
def w_f32_(f, a): f.write(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes())
def w_i32_(f, a): f.write(np.ascontiguousarray(np.asarray(a, np.int32)).tobytes())

if T > 0:
    xn2_bf = xn2[0].to(BF16).float()                       # [T,H] — the FFN sees bf16 (w8a16)
    ffn_f32 = torch.zeros(T, H)                            # expert_sum (MoE) / dffn (dense), f32
    shared_out = torch.zeros(T, H)
    sel_np = np.zeros((T, TOPK), np.int32)
    gate_np = np.zeros((T, TOPK), np.float32)
    marg_np = np.zeros(T, np.float32)      # per-token 8th-minus-9th BIASED score, fp32
    if DENSE:
        # Dense SwiGLU MLP. The plow prefill runs this on the GROUPED expert arms with degenerate
        # 1-expert routing (n_exp=1, k=1, gate=1) — a construction with no oracle until now.
        Wg = dq(idx, P+"mlp.gate_proj.weight")              # [DI,H]
        Wu = dq(idx, P+"mlp.up_proj.weight")                # [DI,H]
        Wd = dq(idx, P+"mlp.down_proj.weight")              # [H,DI]
        act = torch.nn.functional.silu(xn2_bf @ Wg.T) * (xn2_bf @ Wu.T)
        ffn_f32 = act @ Wd.T
        del Wg, Wu, Wd, act
    else:
        # Per-token 256-wide routing, fp32 arm (the ground truth the device router is diffed against).
        logits = (xn2_bf @ Wr.T)                            # [T,E] f32
        scores = torch.sigmoid(logits)
        sel_t = torch.topk(scores + bias, TOPK, dim=-1).indices          # bias only for SELECTION
        gate_t = torch.gather(scores, 1, sel_t)                          # gate = UNBIASED score
        gate_t = gate_t / gate_t.sum(-1, keepdim=True) * RSCALE
        sel_np = sel_t.to(torch.int32).numpy(); gate_np = gate_t.float().numpy()
        # The SELECTION MARGIN, per token, shipped in the fixture. plow's router dots in bf16 and
        # this one in fp32, so a token whose 8th and 9th biased scores are within bf16 resolution
        # can legitimately pick either — the harness needs the margin to tell that near-tie apart
        # from a routing FAULT, which is the whole reason this gate compares vectors and not picks.
        srt = torch.sort(scores + bias, dim=-1, descending=True).values
        marg = (srt[:, TOPK-1] - srt[:, TOPK])
        marg_np = marg.float().numpy()
        print(f"\n=== ROUTER (prefill, {T} rows) ===  distinct experts hit: "
              f"{len(set(sel_np.reshape(-1).tolist()))}/{E}   8th-9th margin min={marg.min():.3e} "
              f"med={marg.median():.3e}  ties(<1e-6)={int((marg < 1e-6).sum())}")
        # GROUP BY EXPERT: 256 experts x 3 projections is ~10 GB of fp8 to dequantise, so each is
        # touched ONCE for every row that chose it — the same reuse the grouped prefill kernel is
        # built around, which is also why a per-row loop over top-8 would be 8x the work.
        by_e = {}
        for t_ in range(T):
            for j in range(TOPK):
                by_e.setdefault(int(sel_np[t_, j]), []).append((t_, j))
        for n_done, (e, lst) in enumerate(sorted(by_e.items())):
            gw = dq(idx, P+f"mlp.experts.{e}.gate_proj.weight")
            uw = dq(idx, P+f"mlp.experts.{e}.up_proj.weight")
            dw = dq(idx, P+f"mlp.experts.{e}.down_proj.weight")
            X = xn2_bf[[t_ for t_, _ in lst]]                            # [n,H]
            O = (torch.nn.functional.silu(X @ gw.T) * (X @ uw.T)) @ dw.T # [n,H]
            for i, (t_, j) in enumerate(lst):
                ffn_f32[t_] += float(gate_np[t_, j]) * O[i]
            del gw, uw, dw, X, O
            if n_done % 32 == 0:
                print(f"   expert {n_done}/{len(by_e)} (e{e}, {len(lst)} rows)", flush=True)
        shg = dq(idx, P+"mlp.shared_experts.gate_proj.weight")
        shu = dq(idx, P+"mlp.shared_experts.up_proj.weight")
        shd = dq(idx, P+"mlp.shared_experts.down_proj.weight")
        shared_out = (torch.nn.functional.silu(xn2_bf @ shg.T) * (xn2_bf @ shu.T)) @ shd.T
        del shg, shu, shd
    # f32 accumulate the terms once, matching plow's f32 MOE_COMBINE_PF.
    block_out = (h1[0].float() + ffn_f32 + shared_out).to(BF16)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<15i", 0x474C4D38, T, H, NH, DK, DR, QN, VD, QL, E, TOPK, IMOE, DI,
                            1 if DENSE else 0, LAYER))
        f.write(struct.pack("<3f", EPS, SCALE, RSCALE))
        w_bf_(f, hidden[0]); w_bf_(f, attn_out[0]); w_bf_(f, h1[0]); w_bf_(f, xn2[0])
        w_bf_(f, block_out); w_f32_(f, ffn_f32.numpy()); w_bf_(f, shared_out.to(BF16))
        w_i32_(f, sel_np); w_f32_(f, gate_np); w_f32_(f, marg_np)
        sz = f.tell()
    print(f"\n[prefill] wrote {OUT} ({sz/1e6:.1f} MB) T={T} layer={LAYER} "
          f"{'DENSE' if DENSE else 'sparse'}: hidden/attn/xmid/xn2/block(bf16) + ffn(f32) + "
          f"shared(bf16) + sel/gate, every row")
    print(f"  ref norms: attn={attn_out[0].float().norm():.3f} xn2={xn2[0].float().norm():.3f} "
          f"ffn={ffn_f32.norm():.3f} shared={shared_out.norm():.3f} "
          f"block={block_out.float().norm():.3f}")
    sys.exit(0)
# ================================================================ end T-row arm; GLM6 path below

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

def shared_w8a8_reference(rows, tp):
    from vllm import __version__
    from vllm._aiter_ops import rocm_aiter_ops
    if not __version__.startswith("0.29.") or not rocm_aiter_ops.is_linear_fp8_enabled():
        raise ValueError("shared W8A8 reference requires vLLM 0.29 with AITER FP8 enabled")
    weights, scales = {}, {}
    for proj in ("gate", "up", "down"):
        name = P + f"mlp.shared_experts.{proj}_proj.weight"
        weights[proj] = load_tensor(idx, name)
        scales[proj] = load_tensor(idx, name + "_scale_inv")
    full_inter, hidden = weights["gate"].shape
    if full_inter % (tp * 128) or hidden % 128:
        raise ValueError("shared W8A8 requires aligned TP weight shards")
    inter = full_inter // tp
    for proj in weights:
        shape = (hidden, full_inter) if proj == "down" else (full_inter, hidden)
        if (weights[proj].dtype != torch.float8_e4m3fn or tuple(weights[proj].shape) != shape
                or scales[proj].dtype != torch.float32
                or tuple(scales[proj].shape) != tuple(d // 128 for d in shape)):
            raise ValueError("shared reference requires original FP8 weights and FP32 block scales")
    x = rows.to(BF16).cuda()
    def gemm(a, w, sa, sw):
        n, k = w.shape
        op = (rocm_aiter_ops.triton_gemm_a8w8_blockscale
              if rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n, k)
              else rocm_aiter_ops.gemm_a8w8_blockscale)
        return op(a, w, sa, sw, [128, 128], output_dtype=BF16)
    partials = []
    for rank in range(tp):
        start, end = rank * inter, (rank + 1) * inter
        guw = torch.cat([weights[p].view(torch.uint8)[start:end] for p in ("gate", "up")])
        guw = guw.view(torch.float8_e4m3fn).contiguous().cuda()
        gus = torch.cat([scales[p][start // 128:end // 128] for p in ("gate", "up")]).cuda()
        dw = weights["down"][:, start:end].contiguous().cuda()
        ds = scales["down"][:, start // 128:end // 128].contiguous().cuda()
        def forward():
            q, s = rocm_aiter_ops.group_fp8_quant(x, 128)
            gu = gemm(q, guw, s, gus)
            act = torch.empty((len(rows), inter), dtype=BF16, device=x.device)
            torch.ops._C.silu_and_mul(act, gu)
            q, s = rocm_aiter_ops.group_fp8_quant(act, 128)
            return gemm(q, dw, s, ds).cpu()
        value, repeat = forward(), forward()
        if not torch.isfinite(value).all() or not torch.equal(value.view(torch.int16), repeat.view(torch.int16)):
            raise ValueError(f"shared W8A8 reference rank{rank} is nonfinite or not repeat-bitwise")
        partials.append(value.float())
    return torch.stack(partials).sum(dim=0), __version__, torch.stack(partials)

shared_rows, shared_reference_version, shared_rank_parts = (shared_w8a8_reference(xn2[:, qpos], args.shared_w8a8_tp)
                                        if args.shared_w8a8_tp else (None, None, None))
xq_bf = xn2[0, qpos].to(BF16).float()      # experts see bf16 activations (w8a16)
expert_sum = torch.zeros(H)
for i, e in enumerate(sel.tolist()):
    expert_sum += float(gate_np[i]) * expert_fwd(xq_bf, e)
if shared_rows is None:
    shared_out, (shg, shu, shd) = shared_fwd(xq_bf)
else:
    shared_out = shared_rows[0]
# f32 accumulate the three bf16 terms (single rounding), matching plow's f32 MOE_COMBINE
block_out = (h1[0, qpos].float() + expert_sum + shared_out).to(BF16)
print(f"\n  ref block_out norm={block_out.float().norm():.3f}  "
      f"shared_out norm={shared_out.norm():.3f}  expert_sum norm={expert_sum.norm():.3f}")

block_rows = [block_out]
ffn_rows = [expert_sum + shared_out]
router_rows = [sel.tolist()]
for row in range(1, B):
    row_x = xn2[row, qpos].float()
    row_sel, row_gates, _ = route(row_x, Wr, torch.float32)
    row_experts = torch.zeros(H)
    for expert, gate in zip(row_sel.tolist(), row_gates.tolist()):
        row_experts += gate * expert_fwd(row_x, expert)
    row_shared = (torch.nn.functional.silu(row_x @ shg.T) * (row_x @ shu.T) @ shd.T
                  if shared_rows is None else shared_rows[row])
    ffn_rows.append(row_experts + row_shared)
    block_rows.append((h1[row, qpos].float() + row_experts + row_shared).to(BF16))
    router_rows.append(row_sel.tolist())
    print(f"  row {row} fp32 top-8: {sorted(row_sel.tolist())}", flush=True)

routed_reference = None
if args.routed_w8a8:
    from pathlib import Path
    from block_fp8_aiter_compare import routed_weights, tensor_digest
    from vllm import __version__
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import QuantMethod
    if __version__ != "0.29.0" or not rocm_aiter_ops.is_fused_moe_enabled():
        raise ValueError("routed reference requires pinned vLLM 0.29.0 with AITER MoE")
    ids, gates = zip(*(route(xn2[row, qpos].float(), Wr, torch.float32)[:2] for row in range(B)))
    ids = torch.stack(ids).to(torch.int32).cuda()
    gates = torch.stack(gates).float().cuda()
    x = xn2[:, qpos].to(BF16).contiguous().cuda()
    tp = args.shared_w8a8_tp
    partials, repeats, records = [], [], []
    os.makedirs(args.block_inputs, exist_ok=True)
    for rank in range(tp):
        weights = routed_weights(Path(MODEL_DIR), LAYER, rank, tp)
        hashes = [tensor_digest(v) for v in weights]
        w1, w2 = rocm_aiter_ops.shuffle_weights(weights[0].cuda(), weights[1].cuda())
        w1.is_shuffled = w2.is_shuffled = True
        s1, s2 = weights[2].cuda(), weights[3].cuda()
        del weights
        def routed_forward():
            return rocm_aiter_ops.fused_moe(x, w1, w2, gates, ids,
                quant_method=QuantMethod.BLOCK_128x128.value, w1_scale=s1, w2_scale=s2,
                doweight_stage1=False, output_dtype=BF16,
                moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy()).cpu()
        value, repeat = routed_forward(), routed_forward()
        if not torch.isfinite(value).all() or not torch.isfinite(repeat).all():
            raise ValueError("nonfinite routed reference")
        partials.append((shared_rank_parts[rank] + value.float()).to(BF16).float())
        repeats.append((shared_rank_parts[rank] + repeat.float()).to(BF16).float())
        for tag, tensor in (("reference", value), ("repeat", repeat)):
            with open(os.path.join(args.block_inputs, f"rank{rank}.routed.{tag}.bf16"), "wb") as f:
                w_bf_(f, tensor)
        records.append(dict(rank=rank, weights_sha256=hashes, output_sha256=tensor_digest(value),
            repeat_sha256=tensor_digest(repeat), repeat_max_row_rel_l2=float(
                ((value.double() - repeat.double()).norm(dim=1) / value.double().norm(dim=1).clamp_min(1e-30)).max())))
        del w1, w2, s1, s2
    ffn = torch.stack(partials).sum(dim=0).to(BF16)
    ffn_repeat = torch.stack(repeats).sum(dim=0).to(BF16)
    ffn_rows = list(ffn)
    block_rows = list((h1[:, qpos].float() + ffn.float()).to(BF16))
    routed_reference = dict(tp=tp, vllm_version=__version__, backend="AITER fused_moe BLOCK_128x128",
        input="independent HF attention, vLLM post-norm, FP32 router",
        shared_add="BF16 per rank before FP32 TP sum and BF16 store", ranks=records,
        ffn_repeat_max_row_rel_l2=float(((ffn.double() - ffn_repeat.double()).norm(dim=1)
            / ffn.double().norm(dim=1).clamp_min(1e-30)).max()))
    print("routed W8A8 reference:", json.dumps(routed_reference), flush=True)

if args.candidate_dir:
    def captured(name):
        path = os.path.join(args.candidate_dir, f"rank0.{name}.bin")
        return torch.from_numpy(np.fromfile(path, dtype=np.uint16)).view(BF16).float()

    def compare(name, actual, expected):
        error = (actual - expected.float()).norm() / expected.float().norm()
        print(f"  {name}: rel_l2={error.item():.6g} "
              f"actual_norm={actual.norm().item():.6g} reference_norm={expected.float().norm().item():.6g}")

    candidate_x = captured("act.xn2")
    compare("xmid", captured("act.xmid"), h1[0, qpos])
    compare("xn2", candidate_x, xq_bf)
    compare("FFN end-to-end", captured("act.attn"), expert_sum + shared_out)
    candidate_sel, candidate_gates, _ = route(candidate_x, Wr, torch.float32)
    candidate_experts = torch.zeros(H)
    for e, gate in zip(candidate_sel.tolist(), candidate_gates.tolist()):
        candidate_experts += gate * expert_fwd(candidate_x, e)
    candidate_shared, _ = shared_fwd(candidate_x)
    print(f"  HF router on captured xn2: {candidate_sel.tolist()} gates={candidate_gates.tolist()}")
    compare("FFN conditioned on captured xn2", captured("act.attn"), candidate_experts + candidate_shared)
    sys.exit(0)

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
compressed = hn.float() @ sd.kv_a_proj_with_mqa.weight.float().T
ckv_raw, krot_raw = compressed[..., :DK], compressed[..., DK:DK+DR]
var = ckv_raw.pow(2).mean(-1, keepdim=True)
c_kv = ckv_raw * torch.rsqrt(var + EPS) * sd.kv_a_layernorm.weight.float()
kr = krot_raw.reshape(B, 1, L, DR)
_, kr_rot = apply_rotary_pos_emb_interleave(kr, kr, cos.float(), sin.float())
k_rot = kr_rot[:, 0]

# ---------------------------------------------------------------- write fixture v2
def w_bf(f, t):    f.write(np.ascontiguousarray(t.to(BF16).view(torch.uint16).cpu().numpy()).tobytes())
def w_bf_np(f, a): f.write(np.ascontiguousarray(torch.from_numpy(np.ascontiguousarray(a)).to(BF16).view(torch.uint16).numpy()).tobytes())
def w_f32(f, a):   f.write(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes())
def w_i32(f, a):   f.write(np.ascontiguousarray(np.asarray(a, np.int32)).tobytes())

if args.block_inputs:
    os.makedirs(args.block_inputs, exist_ok=True)
    full_indexer = cfg.indexer_types[LAYER] == "full"
    if full_indexer and BLOCK_CTX != L:
        raise ValueError("full-indexer capture requires short dense context; supplied sparse keys are not a learned-indexer oracle")
    # The HF sequence contains only supplied selected keys, at their original RoPE positions.
    # This tests a shared-index block's carried state, not the learned indexer's selections.
    if BLOCK_CTX != L:
        def scatter_cache(tensor):
            full = torch.zeros((B, BLOCK_CTX, tensor.shape[-1]), dtype=BF16)
            full[:, logical_positions] = tensor.to(BF16)
            return full

        c_kv, k_rot = scatter_cache(c_kv), scatter_cache(k_rot)
    for name, tensor in (("act.x", hidden[:, qpos]),
                         (f"kv.{LAYER}.ckv", c_kv),
                         (f"kv.{LAYER}.krot", k_rot)):
        with open(os.path.join(args.block_inputs, name + ".bin"), "wb") as f:
            w_bf(f, tensor)
    selected = np.full((B, cfg.index_topk), -1, dtype=np.int32)
    selected[:, :L] = logical_positions.numpy()
    with open(os.path.join(args.block_inputs, "act.iidx.bin"), "wb") as f:
        w_i32(f, selected)
    if full_indexer:
        # All live keys are selected at L<=topk; this history does not qualify indexer scoring.
        with open(os.path.join(args.block_inputs, f"kv.{LAYER}.kidx.bin"), "wb") as f:
            w_bf(f, torch.zeros((B, L, cfg.index_head_dim), dtype=BF16))
    with open(os.path.join(args.block_inputs, "reference.bf16"), "wb") as f:
        w_bf(f, torch.stack(block_rows))
    # The residual can hide a broken, low-amplitude FFN, so gate its contribution separately.
    for name, tensor in (("act.xmid", h1[:, qpos]), ("act.xn2", xn2[:, qpos]),
                         ("act.attn", torch.stack(ffn_rows))):
        with open(os.path.join(args.block_inputs, name + ".reference.bf16"), "wb") as f:
            w_bf(f, tensor)
    with open(os.path.join(args.block_inputs, "reference.json"), "w") as f:
        json.dump(dict(model=MODEL_DIR, layer=LAYER, ctx=BLOCK_CTX, batch=B, seed=0xB4,
                       selected_keys=L, selection="supplied evenly spaced logical positions; not learned indexer",
                       indexer_history="zero BF16 keys; all live keys selected" if full_indexer else None,
                       router_topk=router_rows,
                       routed_reference=routed_reference,
                       reference="HF single-layer residual", tolerance_rel_l2=0.03,
                       post_norm_reference=(dict(vllm_version=post_norm_version, backend="aiter fused_add_rms_norm",
                           input="independent HF attention and original residual", repeat_bitwise=True)
                           if args.vllm_post_norm else None),
                       shared_reference=(dict(tp=args.shared_w8a8_tp, vllm_version=shared_reference_version,
                           weights="FP8 E4M3 block128", activation="FP8 E4M3 group128",
                           scales="FP32", gemm_output="BF16", silu_intermediate="BF16",
                           tp_partial_sum="FP32 reference", input="independent HF post-attention norm")
                           if args.shared_w8a8_tp else None),
                       stages=["act.xmid", "act.xn2", "act.attn"]), f, indent=2)

if args.inputs_only:
    print(f"wrote B{B} block operands and per-row references to {args.block_inputs}")
    sys.exit(0)

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
