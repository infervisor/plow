#!/usr/bin/env python3
"""Breeze TTS 2's T5Gemma2 text encoder -> a plow-compilable Gemma3 text checkpoint directory.

  HF_HOME=... python scripts/tts/breeze_enc_prep.py OUT_DIR REVISION

The encoder is Gemma3-shaped (sandwich norms, (1 + w) RMSNorm, q/k norm, local/global RoPE) but
bidirectional; config `plow_encoder` makes plowc emit it in encoder mode (every row through the
head). Tensors:
  model.layers.N.{input,post_attention,pre_feedforward,post_feedforward}_layernorm
                       <- text_encoder.layers.N.{pre_self_attn,post_self_attn,pre_feedforward,post_feedforward}_layernorm
  model.layers.N.self_attn.* / mlp.*, model.norm.weight     <- text_encoder.* (same names)
  model.embed_tokens.weight  <- text_encoder.embed_tokens.weight; row 256000 (eoi) = bf16(eoi_embedding / 34),
                                the reference's post-scale replacement up to that rounding
  lm_head.weight             <- text_encoder_proj.weight [2048, 1152] (config `plow_head_rows` 2048)
"""
import argparse, glob, json, os, shutil

import torch
from safetensors.torch import load_file, save_file

REPO = "models--BreezeBlue--Breeze-TTS-2"
RENAME = {
    "pre_self_attn_layernorm": "input_layernorm",
    "post_self_attn_layernorm": "post_attention_layernorm",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("revision")
    args = ap.parse_args()
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    src = os.path.join(hub, REPO, "snapshots", args.revision)
    cfg = json.load(open(f"{src}/config.json"))
    tc = cfg["text_encoder_config"]
    sd = {}
    for f in sorted(glob.glob(f"{src}/model-*.safetensors")):
        sd.update({k: v for k, v in load_file(f).items() if k.startswith("text_encoder")})
    os.makedirs(args.out, exist_ok=True)

    out = {}
    pre = "text_encoder."
    for k, v in sd.items():
        if k.startswith(pre + "layers."):
            parts = k[len(pre):].split(".")
            parts[2] = RENAME.get(parts[2], parts[2])
            out["model." + ".".join(parts)] = v
    out["model.norm.weight"] = sd[pre + "norm.weight"]
    emb = sd[pre + "embed_tokens.weight"].clone()
    hidden = tc["hidden_size"]
    scale = torch.tensor(hidden ** 0.5, dtype=torch.bfloat16).float()
    emb[tc["eoi_token_index"]] = (sd[pre + "embed_tokens.eoi_embedding"].float() / scale).to(emb.dtype)
    out["model.embed_tokens.weight"] = emb
    proj = sd["text_encoder_proj.weight"]
    out["lm_head.weight"] = proj
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}
    save_file(out, f"{args.out}/model.safetensors")

    rp = tc["rope_parameters"]
    types = tc["layer_types"]
    pattern = types.index("full_attention") + 1
    assert all((t == "full_attention") == ((i + 1) % pattern == 0) for i, t in enumerate(types)), types
    gemma = dict(
        architectures=["Gemma3ForCausalLM"], model_type="gemma3_text", hidden_size=hidden,
        intermediate_size=tc["intermediate_size"], num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"], num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"], rms_norm_eps=tc["rms_norm_eps"], hidden_activation=tc["hidden_activation"],
        query_pre_attn_scalar=tc["query_pre_attn_scalar"], sliding_window=tc["sliding_window"],
        sliding_window_pattern=pattern, rope_local_base_freq=rp["sliding_attention"]["rope_theta"],
        rope_theta=rp["full_attention"]["rope_theta"],
        rope_scaling=dict(rope_type="linear", factor=rp["full_attention"]["factor"]),
        attention_bias=False, attn_logit_softcapping=None, final_logit_softcapping=None,
        vocab_size=tc["vocab_size"], max_position_embeddings=tc["max_position_embeddings"],
        tie_word_embeddings=False, torch_dtype="bfloat16", bos_token_id=2, eos_token_id=1,
        plow_head_rows=proj.shape[0], plow_encoder=True,
    )
    json.dump(gemma, open(f"{args.out}/config.json", "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if os.path.exists(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", f"{args.out}/{f}")


if __name__ == "__main__":
    main()
