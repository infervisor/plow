#!/usr/bin/env python3
# glm52_prep.py — HOST WEIGHT-PREP for the GLM-5.2 (GlmMoeDsa) plow serving emitter.  [GLM52-ms1]
#
# Reads the REAL zai-org/GLM-5.2-FP8 checkpoint and writes a NAMED plow-ready weight dir whose
# tensor names match the bound-tensor contract that gemma4.rs::declare_glm emits (Stack B: the
# .pkt binds every weight by name from this dir). It is the production analog of
# glm52_real_oracle.py's absorption block, but it WRITES a safetensors weight dir instead of a
# packed test fixture:
#   - MLA derived bf16 weights (absorbed q_nope Wqa=einsum(k_nope_w,q_b_nope), value Wuv=value_w^T,
#     kv_a latent down, and the rope-FOLDED q_rope/k_rope at a fixed position for milestone-1),
#   - dequantised bf16 q_a_proj / o_proj / shared-expert gate/up/down,
#   - block-fp8 routed experts (gate/up/down .weight + .weight_scale_inv) copied VERBATIM (fp8 bytes
#     + f32 scale grids, no dequant),
#   - bf16 norms + router gate.weight + f32 e_score_correction_bias.
# The rope fold at a FIXED position (index_pos) is the documented milestone-1 simplification
# (plans/glm52-campaign.md): single-token validation. Milestone-3 multi-token decode replaces the
# fold with the dynamic interleaved-RoPE op (kernels branch).
#
# By default it preps ONE MoE layer (layer 3 = first sparse layer, "shared" indexer -> DSA no-op,
# dense causal MLA for L<=2048) — exactly the layer the ms1 vs-HF gate validates. --layers a,b,c
# preps several; --globals also writes embed_tokens/norm/lm_head (needed for the full-model B5, not
# for the single-layer gate).
#
# Usage:
#   nix develop -c python3 scripts/glm52_prep.py \
#       --model /home/lava/models/GLM-5.2-FP8 --out /home/lava/models/glm52_prep --layer 3
#   # writes <out>/config.json + <out>/model-00001-of-00001.safetensors (names per declare_glm).
import os, sys, json, struct, mmap, argparse
import numpy as np
import torch

torch.set_grad_enabled(False)

BF16 = torch.bfloat16
_DT = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F32": torch.float32,
       "F16": torch.float16}

# ---------------------------------------------------------------- safetensors reader (mmap, lazy)
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
    return torch.frombuffer(bytearray(buf), dtype=_DT[dt]).view(*shape)

def dequant_blockfp8(idx, wname):
    """w_dequant = fp8.float() * weight_scale_inv[i//128, j//128] (block-index gather)."""
    w = load_tensor(idx, wname).float()
    s = load_tensor(idx, wname + "_scale_inv").float()
    N, K = w.shape
    sf = s[torch.arange(N) // 128][:, torch.arange(K) // 128]
    return w * sf

# ---------------------------------------------------------------- streaming safetensors WRITER
class STWriter:
    """Two-pass streaming safetensors writer: register (name, dtype, shape, producer) with a byte
    length, then flush() writes the JSON header once and streams each producer straight to disk so
    the ~10 GB of fp8 experts never sit in RAM."""
    def __init__(self):
        self.entries = []   # (name, dtype_str, shape, nbytes, producer)
    def add(self, name, dtype_str, shape, nbytes, producer):
        self.entries.append((name, dtype_str, list(shape), int(nbytes), producer))
    def flush(self, path):
        hdr, off = {}, 0
        for name, dt, shape, nb, _ in self.entries:
            hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
            off += nb
        blob = json.dumps(hdr, separators=(",", ":")).encode()
        pad = (-((8 + len(blob)) % 8)) % 8            # 8-byte align data section
        blob += b" " * pad
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for name, dt, shape, nb, prod in self.entries:
                start = f.tell()
                prod(f)
                wrote = f.tell() - start
                assert wrote == nb, f"{name}: wrote {wrote} != declared {nb}"
        return off

# producers (bind their argument at call time so the loop var isn't captured late)
def p_bf16(t):
    a = np.ascontiguousarray(t.to(BF16).view(torch.uint16).cpu().numpy())
    return lambda f: f.write(a.tobytes())
def p_bf16_np(arr):
    a = np.ascontiguousarray(torch.from_numpy(np.ascontiguousarray(arr)).to(BF16).view(torch.uint16).numpy())
    return lambda f: f.write(a.tobytes())
def p_f32(arr):
    a = np.ascontiguousarray(np.asarray(arr, np.float32))
    return lambda f: f.write(a.tobytes())
def p_raw(idx, name):
    buf, dt, shape = raw_bytes(idx, name)
    b = bytes(buf)
    return (dt, list(shape), len(b), lambda f: f.write(b))

# ---------------------------------------------------------------- config
from transformers import AutoConfig

def add_indexer_tensors(idx, cfg, w, layer):
    """Queue the 7 DSA lightning-indexer tensors for a 'full' layer into writer `w` (used by both the
    full per-layer prep and the incremental indexer-only prep). wq_b/wk are block-fp8 -> copied
    VERBATIM (fp8 bytes + f32 [128,128] scale grid) for the GemvFp8Blk projections; k_norm (LayerNorm,
    has BIAS) + weights_proj are bf16. [DSA G1]"""
    H = cfg.hidden_size
    IX = f"model.layers.{layer}.self_attn.indexer."
    for wn in (IX + "wq_b.weight", IX + "wk.weight"):
        dt, shape, nb, prod = p_raw(idx, wn); assert dt == "F8_E4M3", (wn, dt)
        w.add(wn, "F8_E4M3", shape, nb, prod)
        sdt, sshape, snb, sprod = p_raw(idx, wn + "_scale_inv")
        w.add(wn + "_scale_inv", "F32", sshape, snb, sprod)
    w.add(IX + "k_norm.weight", "BF16", [cfg.index_head_dim], cfg.index_head_dim * 2,
          p_bf16(load_tensor(idx, IX + "k_norm.weight")))
    w.add(IX + "k_norm.bias", "BF16", [cfg.index_head_dim], cfg.index_head_dim * 2,
          p_bf16(load_tensor(idx, IX + "k_norm.bias")))
    w.add(IX + "weights_proj.weight", "BF16", [cfg.index_n_heads, H], cfg.index_n_heads * H * 2,
          p_bf16(load_tensor(idx, IX + "weights_proj.weight")))


def prep_layer(idx, cfg, w, L, layer, pos):
    """Write layer `layer`'s named tensors into writer `w`. RoPE is applied dynamically on device,
    so the rope-slice weights are RAW (unfolded); `pos`/`L` are kept for signature compatibility."""
    H, NH, DK, DR, QN, VD, QL = (cfg.hidden_size, cfg.num_attention_heads, cfg.kv_lora_rank,
        cfg.qk_rope_head_dim, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.q_lora_rank)
    QKH, IMOE, E, EPS = cfg.qk_head_dim, cfg.moe_intermediate_size, cfg.n_routed_experts, cfg.rms_norm_eps
    IB, HB = IMOE // 128, H // 128
    P = f"model.layers.{layer}."
    dense = layer < cfg.first_k_dense_replace

    # --- MLA absorbed (position-independent) + RAW rope-slice derived weights (from dequantised
    #     q_b/kv_b/kv_a). RoPE is applied DYNAMICALLY on device (interleaved HeadNormRope HD=64), so
    #     the q_rope/k_rope down-projections are the RAW (unfolded) weights — no position fold here. ---
    q_b = dequant_blockfp8(idx, P + "self_attn.q_b_proj.weight").view(NH, QKH, QL)
    q_b_nope, q_b_rope = q_b[:, :QN, :], q_b[:, QN:, :]
    kv_b = dequant_blockfp8(idx, P + "self_attn.kv_b_proj.weight").view(NH, QN + VD, DK)
    k_nope_w, value_w = kv_b[:, :QN, :], kv_b[:, QN:, :]
    Wqa = torch.einsum('hpl,hpk->hlk', k_nope_w, q_b_nope).contiguous()      # [NH,DK,QL]
    Wuv = value_w.transpose(-1, -2).contiguous()                            # [NH,DK,VD]
    Wqr = q_b_rope.contiguous().numpy()                                     # RAW [NH,DR,QL]
    kv_a_w = dequant_blockfp8(idx, P + "self_attn.kv_a_proj_with_mqa.weight").numpy()
    W_ckv_down    = kv_a_w[:DK]                                              # [DK,H]
    W_krot_folded = kv_a_w[DK:DK + DR]                                       # RAW [DR,H]

    A = "self_attn."
    w.add(P + "input_layernorm.weight", "BF16", [H], H * 2,
          p_bf16(load_tensor(idx, P + "input_layernorm.weight")))
    w.add(P + A + "q_a_proj.weight", "BF16", [QL, H], QL * H * 2,
          p_bf16(dequant_blockfp8(idx, P + A + "q_a_proj.weight")))
    w.add(P + A + "q_a_layernorm.weight", "BF16", [QL], QL * 2,
          p_bf16(load_tensor(idx, P + A + "q_a_layernorm.weight")))
    w.add(P + A + "derived.q_absorb.weight", "BF16", [NH * DK, QL], NH * DK * QL * 2,
          p_bf16_np(Wqa.reshape(NH * DK, QL).numpy()))
    w.add(P + A + "derived.q_rope.weight", "BF16", [NH * DR, QL], NH * DR * QL * 2,
          p_bf16_np(Wqr.reshape(NH * DR, QL)))
    w.add(P + A + "derived.kv_a_latent.weight", "BF16", [DK, H], DK * H * 2, p_bf16_np(W_ckv_down))
    w.add(P + A + "kv_a_layernorm.weight", "BF16", [DK], DK * 2,
          p_bf16(load_tensor(idx, P + A + "kv_a_layernorm.weight")))
    w.add(P + A + "derived.k_rope.weight", "BF16", [DR, H], DR * H * 2, p_bf16_np(W_krot_folded))
    w.add(P + A + "derived.v_absorb.weight", "BF16", [NH * DK, VD], NH * DK * VD * 2,
          p_bf16_np(Wuv.reshape(NH * DK, VD).numpy()))
    w.add(P + A + "o_proj.weight", "BF16", [H, NH * VD], H * NH * VD * 2,
          p_bf16(dequant_blockfp8(idx, P + A + "o_proj.weight")))
    w.add(P + "post_attention_layernorm.weight", "BF16", [H], H * 2,
          p_bf16(load_tensor(idx, P + "post_attention_layernorm.weight")))

    # --- DSA lightning indexer (ONLY on 'full' layers). ADDITIVE; keeps every existing name contract. ---
    if cfg.indexer_types[layer] == "full":
        add_indexer_tensors(idx, cfg, w, layer)

    if dense:
        # DENSE FFN (layers 0-2): block-fp8 gate/up/down + f32 scale grids, VERBATIM (no dequant).
        for proj in ("gate_proj", "up_proj", "down_proj"):
            wn = P + f"mlp.{proj}.weight"
            dt, shape, nb, prod = p_raw(idx, wn); assert dt == "F8_E4M3", (wn, dt)
            w.add(wn, "F8_E4M3", shape, nb, prod)
            sdt, sshape, snb, sprod = p_raw(idx, wn + "_scale_inv")
            w.add(wn + "_scale_inv", "F32", sshape, snb, sprod)
        return dict(H=H, NH=NH, DK=DK, DR=DR, dense_inter=cfg.intermediate_size, pos=pos,
                    eps=EPS, scale=QKH ** -0.5)

    # --- router (bf16 gate + f32 correction bias) ---
    w.add(P + "mlp.gate.weight", "BF16", [E, H], E * H * 2,
          p_bf16(load_tensor(idx, P + "mlp.gate.weight")))
    w.add(P + "mlp.gate.e_score_correction_bias", "F32", [E], E * 4,
          p_f32(load_tensor(idx, P + "mlp.gate.e_score_correction_bias").float().cpu().numpy()))

    # --- shared expert (dequantised to bf16) ---
    for proj, shape in (("gate_proj", [IMOE, H]), ("up_proj", [IMOE, H]), ("down_proj", [H, IMOE])):
        w.add(P + f"mlp.shared_experts.{proj}.weight", "BF16", shape,
              int(np.prod(shape)) * 2, p_bf16(dequant_blockfp8(idx, P + f"mlp.shared_experts.{proj}.weight")))

    # --- routed experts: block-fp8 weight + f32 scale grid, VERBATIM (no dequant) ---
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            wn = P + f"mlp.experts.{e}.{proj}.weight"
            dt, shape, nb, prod = p_raw(idx, wn)
            assert dt == "F8_E4M3", (wn, dt)
            w.add(wn, "F8_E4M3", shape, nb, prod)
            sn = wn + "_scale_inv"
            sdt, sshape, snb, sprod = p_raw(idx, sn)
            w.add(sn, "F32", sshape, snb, sprod)
    return dict(H=H, NH=NH, DK=DK, DR=DR, QN=QN, VD=VD, QL=QL, IMOE=IMOE, E=E, IB=IB, HB=HB,
                pos=pos, eps=EPS, scale=QKH ** -0.5, rscale=cfg.routed_scaling_factor)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL_DIR", "/home/lava/models/GLM-5.2-FP8"))
    ap.add_argument("--out", default="/home/lava/models/glm52_prep")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--layers", default=None, help="comma list overrides --layer")
    ap.add_argument("--L", type=int, default=512, help="validation context (fold pos = L-1)")
    ap.add_argument("--globals", action="store_true", help="also write embed_tokens/norm/lm_head")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [args.layer]
    pos = args.L - 1

    idx = _index_shards(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    os.makedirs(args.out, exist_ok=True)
    # config.json passthrough so the emitter's cfg_glm parses the same dims.
    with open(os.path.join(args.model, "config.json")) as f:
        cfgj = f.read()
    with open(os.path.join(args.out, "config.json"), "w") as f:
        f.write(cfgj)

    w = STWriter()
    meta = None
    for l in layers:
        meta = prep_layer(idx, cfg, w, args.L, l, pos)
        print(f"[prep] layer {l}: MLA derived + block-fp8 experts queued")
    if args.globals:
        w.add("model.embed_tokens.weight", "BF16", [cfg.vocab_size, cfg.hidden_size],
              cfg.vocab_size * cfg.hidden_size * 2, p_bf16(load_tensor(idx, "model.embed_tokens.weight")))
        w.add("model.norm.weight", "BF16", [cfg.hidden_size], cfg.hidden_size * 2,
              p_bf16(load_tensor(idx, "model.norm.weight")))
        w.add("lm_head.weight", "BF16", [cfg.vocab_size, cfg.hidden_size],
              cfg.vocab_size * cfg.hidden_size * 2, p_bf16(load_tensor(idx, "lm_head.weight")))

    out_st = os.path.join(args.out, "model-00001-of-00001.safetensors")
    total = w.flush(out_st)
    print(f"[prep] wrote {out_st} ({total/1e9:.2f} GB), {len(w.entries)} tensors")
    print(f"[prep] dims {meta}  (fold pos={pos}); names match gemma4.rs::declare_glm")

if __name__ == "__main__":
    main()
