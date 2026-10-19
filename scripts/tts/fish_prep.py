#!/usr/bin/env python3
"""Fish Audio S2 Pro (fishaudio/s2-pro) -> a plow-compilable checkpoint directory.

  HF_HOME=... python scripts/tts/fish_prep.py OUT_DIR

The slow AR is Qwen3-4B-shaped, so the directory is a Qwen3 checkpoint (the compiler keeps checkpoint
naming; the runtime binds roles):
  model.layers.N.self_attn.{q,k,v,o}_proj / q_norm / k_norm, mlp.{gate,up,down}_proj, *_layernorm
                                          <- text_model.model.layers.N.{attention.wqkv/wo/q_norm/k_norm,
                                             feed_forward.w1/w3/w2, attention_norm, ffn_norm}
  model.embed_tokens.weight              <- text_model.model.embeddings.weight (155776 rows: prompt text)
  lm_head.weight                         <- the 4097 rows the model may emit: semantic tokens
                                            151678..155773 (classes 0..4095), then <|im_end|> 151645
                                            (class 4096, the stop); config `plow_head_rows` = 4097
  fish.codebook_embed.weight             <- audio_decoder.codebook_embeddings.weight [10*4096, 2560]
  fish.depth.layers.N.* / fish.depth.{embed,head,norm}.weight
                                         <- audio_decoder.layers.N.* / embeddings (+ a zero stop row)
                                            / output / norm
S2 Pro rotates interleaved pairs (x[2i], x[2i+1]); plow's kernels rotate half-split pairs
(x[i], x[i+64]). Permuting every head's q and k rows (and the q/k norm weights) to [evens, odds]
makes the half-split rotation compute the same thing: q.k is invariant under a shared permutation
of their dims, and V / O are untouched.
"""
import argparse, glob, json, math, os, shutil

import torch
from safetensors.torch import load_file, save_file

REPO = "models--fishaudio--s2-pro"
SEMANTIC0, SEMANTIC_N, IM_END = 151678, 4096, 151645


def interleaved_to_half(hd):
    return torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])


def permute_heads(w, heads, hd, perm):
    """Rows [heads*hd, ...] (or a [hd] norm weight when heads == 0) in half-split order."""
    if heads == 0:
        return w[perm].contiguous()
    return w.reshape(heads, hd, *w.shape[1:])[:, perm].reshape(w.shape).contiguous()


def block(sd, src, dst, heads, kv_heads, hd, qk_norm, perm, out):
    q, k = heads * hd, kv_heads * hd
    wqkv = sd[f"{src}.attention.wqkv.weight"]
    assert wqkv.shape[0] == q + 2 * k, wqkv.shape
    out[f"{dst}.self_attn.q_proj.weight"] = permute_heads(wqkv[:q], heads, hd, perm)
    out[f"{dst}.self_attn.k_proj.weight"] = permute_heads(wqkv[q:q + k], kv_heads, hd, perm)
    out[f"{dst}.self_attn.v_proj.weight"] = wqkv[q + k:].contiguous()
    out[f"{dst}.self_attn.o_proj.weight"] = sd[f"{src}.attention.wo.weight"]
    if qk_norm:
        out[f"{dst}.self_attn.q_norm.weight"] = permute_heads(sd[f"{src}.attention.q_norm.weight"], 0, hd, perm)
        out[f"{dst}.self_attn.k_norm.weight"] = permute_heads(sd[f"{src}.attention.k_norm.weight"], 0, hd, perm)
    out[f"{dst}.input_layernorm.weight"] = sd[f"{src}.attention_norm.weight"]
    out[f"{dst}.post_attention_layernorm.weight"] = sd[f"{src}.ffn_norm.weight"]
    out[f"{dst}.mlp.gate_proj.weight"] = sd[f"{src}.feed_forward.w1.weight"]
    out[f"{dst}.mlp.up_proj.weight"] = sd[f"{src}.feed_forward.w3.weight"]
    out[f"{dst}.mlp.down_proj.weight"] = sd[f"{src}.feed_forward.w2.weight"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    src = sorted(glob.glob(os.path.join(hub, REPO, "snapshots/*")))[-1]
    cfg = json.load(open(f"{src}/config.json"))
    tc, ac = cfg["text_config"], cfg["audio_decoder_config"]
    sd = {}
    for f in sorted(glob.glob(f"{src}/model-*.safetensors")):
        sd.update(load_file(f))
    os.makedirs(args.out, exist_ok=True)

    hd = tc["head_dim"]
    perm = interleaved_to_half(hd)
    out = {}
    for n in range(tc["n_layer"]):
        block(sd, f"text_model.model.layers.{n}", f"model.layers.{n}", tc["n_head"], tc["n_local_heads"], hd,
              True, perm, out)
    emb = sd["text_model.model.embeddings.weight"]
    out["model.embed_tokens.weight"] = emb
    out["model.norm.weight"] = sd["text_model.model.norm.weight"]
    out["lm_head.weight"] = torch.cat([emb[SEMANTIC0:SEMANTIC0 + SEMANTIC_N], emb[IM_END:IM_END + 1]]).contiguous()
    out["fish.codebook_embed.weight"] = sd["audio_decoder.codebook_embeddings.weight"]
    dhd = ac.get("head_dim", hd)
    dperm = interleaved_to_half(dhd)
    for n in range(ac["n_layer"]):
        block(sd, f"audio_decoder.layers.{n}", f"fish.depth.layers.{n}", ac["n_head"], ac["n_local_heads"], dhd,
              bool(ac.get("attention_qk_norm", False)), dperm, out)
    # A zero row past the codebook: the depth stage embeds the stop class as zeros.
    de = sd["audio_decoder.embeddings.weight"]
    out["fish.depth.embed.weight"] = torch.cat([de, torch.zeros_like(de[:1])])
    out["fish.depth.head.weight"] = sd["audio_decoder.output.weight"]
    out["fish.depth.norm.weight"] = sd["audio_decoder.norm.weight"]
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}
    save_file(out, f"{args.out}/model.safetensors")

    codebooks = ac["num_codebooks"]
    qwen = dict(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=tc["dim"],
        intermediate_size=tc["intermediate_size"], num_hidden_layers=tc["n_layer"],
        num_attention_heads=tc["n_head"], num_key_value_heads=tc["n_local_heads"], head_dim=hd,
        rms_norm_eps=tc["norm_eps"], rope_theta=tc["rope_base"], rope_scaling=None,
        vocab_size=emb.shape[0], max_position_embeddings=tc["max_seq_len"], tie_word_embeddings=False,
        torch_dtype="bfloat16", bos_token_id=None, eos_token_id=IM_END,
        plow_head_rows=SEMANTIC_N + 1,
        fish_s2=dict(
            semantic_base=SEMANTIC0, semantic_tokens=SEMANTIC_N, stop_class=SEMANTIC_N, im_end=IM_END,
            codebooks=codebooks, codebook_size=ac["vocab_size"], frame_scale=1.0 / math.sqrt(codebooks + 1),
            depth=dict(layers=ac["n_layer"], hidden=ac["dim"], heads=ac["n_head"], kv_heads=ac["n_local_heads"],
                       head_dim=dhd, intermediate=ac["intermediate_size"], eps=ac["norm_eps"],
                       rope_theta=ac["rope_base"], positions=codebooks, qk_norm=bool(ac.get("attention_qk_norm", False))),
            sampling=dict(temperature=0.7, top_p=0.7, top_k=30, ras_window=10, ras_temperature=1.0, ras_top_p=0.9),
        ),
    )
    json.dump(qwen, open(f"{args.out}/config.json", "w"), indent=1)
    json.dump({"eos_token_id": [IM_END]}, open(f"{args.out}/generation_config.json", "w"))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(f"{src}/{f}", f"{args.out}/{f}")


if __name__ == "__main__":
    main()
