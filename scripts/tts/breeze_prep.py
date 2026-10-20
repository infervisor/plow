#!/usr/bin/env python3
"""Breeze TTS 2 (BreezeBlue/Breeze-TTS-2) backbone -> a plow-compilable Qwen3 checkpoint directory.

  HF_HOME=... python scripts/tts/breeze_prep.py OUT_DIR REVISION

The backbone is Qwen3-1.7B-shaped with no token embedding of its own: prompt rows are the projected
T5Gemma2 text-encoder rows (host overlay rows), generated frames embed as the sum of their 16
codebook rows. So:
  model.layers.N.* / model.norm.weight    <- backbone_model.layers.N.* / backbone_model.norm.weight
  model.embed_tokens.weight               <- zeros [2049, 2048] (frame token = codebook-0 class)
  lm_head.weight                          <- lm_head rows 0..2047 (codes) then 2051 (EOS, class 2048)
  breeze.frame_embed.weight               <- depth_decoder.model.embed_tokens.weight [16*2051, 2048]
                                             (row k*2051 + code_k; config `plow_frame_embed`)
  breeze.depth.layers.N.* / norm.weight   <- depth_decoder.model.layers.N.* / model.norm.weight
  breeze.depth.proj.weight                <- depth_decoder.model.inputs_embeds_projector.weight [1024, 2048]
  breeze.depth.head.J.weight              <- depth_decoder.codebooks_head.weight[J].T rows 0..2047 [2048, 1024]
                                             (codebook J + 1)
Rows 2048..2050 of the native heads are suppressed by the reference samplers, so they are dropped.
"""
import argparse, glob, json, os, shutil

import torch
from safetensors.torch import load_file, save_file

REPO = "models--BreezeBlue--Breeze-TTS-2"
CODES, EOS_NATIVE = 2048, 2051


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("revision")
    args = ap.parse_args()
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    src = os.path.join(hub, REPO, "snapshots", args.revision)
    cfg = json.load(open(f"{src}/config.json"))
    bc, dc = cfg["backbone_config"], cfg["depth_decoder_config"]
    sd = {}
    for f in sorted(glob.glob(f"{src}/model-*.safetensors")):
        sd.update(load_file(f))
    os.makedirs(args.out, exist_ok=True)

    out = {}
    pre = "backbone_model."
    for k, v in sd.items():
        if k.startswith(pre + "layers.") or k == pre + "norm.weight":
            out["model." + k[len(pre):]] = v
    hidden = bc["hidden_size"]
    out["model.embed_tokens.weight"] = torch.zeros(CODES + 1, hidden, dtype=torch.bfloat16)
    head = sd["lm_head.weight"]
    out["lm_head.weight"] = torch.cat([head[:CODES], head[EOS_NATIVE:EOS_NATIVE + 1]])
    table = sd["depth_decoder.model.embed_tokens.weight"]
    codebooks, cb_size = dc["num_codebooks"], dc["vocab_size"]
    assert table.shape == (codebooks * cb_size, hidden), table.shape
    out["breeze.frame_embed.weight"] = table
    pre = "depth_decoder.model."
    for k, v in sd.items():
        if k.startswith(pre + "layers.") or k == pre + "norm.weight":
            out["breeze.depth." + k[len(pre):]] = v
    out["breeze.depth.proj.weight"] = sd[pre + "inputs_embeds_projector.weight"]
    for j, w in enumerate(sd["depth_decoder.codebooks_head.weight"]):
        out[f"breeze.depth.head.{j}.weight"] = w.T[:CODES]
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}
    save_file(out, f"{args.out}/model.safetensors")

    max_seq = 2048
    qwen = dict(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=hidden,
        intermediate_size=bc["intermediate_size"], num_hidden_layers=bc["num_hidden_layers"],
        num_attention_heads=bc["num_attention_heads"], num_key_value_heads=bc["num_key_value_heads"],
        head_dim=bc["head_dim"], rms_norm_eps=bc["rms_norm_eps"], rope_theta=bc["rope_theta"], rope_scaling=None,
        vocab_size=CODES + 1, max_position_embeddings=max_seq, tie_word_embeddings=False,
        torch_dtype="bfloat16", bos_token_id=None, eos_token_id=CODES,
        plow_frame_embed=dict(
            table="breeze.frame_embed.weight", codebooks=codebooks, codebook_size=cb_size, lo=0, rows=CODES,
            overlay_rows=max_seq,
            depth=dict(layers=dc["num_hidden_layers"], hidden=dc["hidden_size"], heads=dc["num_attention_heads"],
                       kv_heads=dc["num_key_value_heads"], head_dim=dc["head_dim"],
                       intermediate=dc["intermediate_size"], eps=dc["rms_norm_eps"], rope_theta=dc["rope_theta"],
                       rope_scaling=dc["rope_scaling"], positions=codebooks)),
    )
    json.dump(qwen, open(f"{args.out}/config.json", "w"), indent=1)
    json.dump({"eos_token_id": [CODES]}, open(f"{args.out}/generation_config.json", "w"))
    # The speech contract's prompt framing and voice check read the Breeze (Gemma3) tokenizer.
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if os.path.exists(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", f"{args.out}/{f}")


if __name__ == "__main__":
    main()
