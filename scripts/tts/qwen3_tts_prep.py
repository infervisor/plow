#!/usr/bin/env python3
"""Qwen3-TTS 12Hz talker + code predictor (Qwen/Qwen3-TTS-12Hz-*) -> a plow-compilable Qwen3
checkpoint directory.

  HF_HOME=... python scripts/tts/qwen3_tts_prep.py OUT_DIR REVISION [--repo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice]

The talker is a Qwen3 LM whose every input row is a text-side row plus codec-side rows, so:
  model.layers.N.* / model.norm.weight   <- talker.model.layers.N.* / talker.model.norm.weight
  model.embed_tokens.weight              <- T = text_projection(text_embedding) [151936, 2048]:
                                            the SiLU MLP over the bf16 weights, each stage
                                            rounded to bf16 like the reference (a row's text side)
  lm_head.weight                         <- codec_head rows 0..2047 (codes), then row 2150 (EOS,
                                            class 2048); config `plow_head_rows` = 2049. The
                                            reference suppresses the other 1023 rows.
  qwen3tts.frame_embed.weight            <- [16 * 3073, 2048]: block 0 = talker.model.codec_embedding
                                            (3072 rows: codes, control and speaker ids), block k =
                                            code_predictor.model.codec_embedding[k-1] (2048 rows),
                                            every block zero-padded to 3073 rows; row 3072 of each
                                            block is zero (`none`)
  qwen3tts.depth.layers.N.* / norm.weight <- talker.code_predictor.model.layers.N.* / model.norm
  qwen3tts.depth.proj.{weight,bias}      <- talker.code_predictor.small_to_mtp_projection
  qwen3tts.depth.head.J.weight           <- talker.code_predictor.lm_head.J (codebook J + 1)
A row's embedding is T[text id] + sum_k block_k[code_k] (`plow_frame_embed` over the whole
vocabulary): the reference's text-side plus codec-side rows, and a generated frame's 16 code rows
plus the frame's trailing text row. The talker's interleaved MRoPE sees equal position ids on its
three axes (text only), which is plain RoPE, so `rope_scaling` is null.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

CODES, EOS, CB_ROWS = 2048, 2150, 3073


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("revision")
    ap.add_argument("--repo", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    a = ap.parse_args()
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    src = os.path.join(hub, "models--" + a.repo.replace("/", "--"), "snapshots", a.revision)
    cfg = json.load(open(f"{src}/config.json"))
    tc = cfg["talker_config"]
    cc = tc["code_predictor_config"]
    sd = load_file(f"{src}/model.safetensors")
    os.makedirs(a.out, exist_ok=True)
    hidden = tc["hidden_size"]
    assert tc["vocab_size"] < CB_ROWS and cc["vocab_size"] == CODES and tc["codec_eos_token_id"] == EOS

    out = {}
    for k, v in sd.items():
        if k.startswith("talker.model.layers.") or k == "talker.model.norm.weight":
            out["model." + k[len("talker.model."):]] = v
    # T = fc2(silu(fc1(E))), each stage accumulated in fp64 and rounded to bf16 as the bf16
    # reference rounds fc1's output, the SiLU and fc2's output.
    E = sd["talker.model.text_embedding.weight"]
    w1, b1 = sd["talker.text_projection.linear_fc1.weight"].double(), sd["talker.text_projection.linear_fc1.bias"].double()
    w2, b2 = sd["talker.text_projection.linear_fc2.weight"].double(), sd["talker.text_projection.linear_fc2.bias"].double()
    T = torch.empty_like(E)
    for i in range(0, E.shape[0], 8192):
        x = E[i:i + 8192].double()
        bf = lambda t: t.to(torch.bfloat16).double()
        h = bf(torch.nn.functional.silu(bf(x @ w1.T + b1)))
        T[i:i + 8192] = (h @ w2.T + b2).to(torch.bfloat16)
    out["model.embed_tokens.weight"] = T
    head = sd["talker.codec_head.weight"]
    out["lm_head.weight"] = torch.cat([head[:CODES], head[EOS:EOS + 1]])

    cp = "talker.code_predictor."
    groups = tc["num_code_groups"]
    blocks = [sd["talker.model.codec_embedding.weight"]] + [
        sd[f"{cp}model.codec_embedding.{k}.weight"] for k in range(groups - 1)]
    table = torch.zeros(groups * CB_ROWS, hidden, dtype=torch.bfloat16)
    for k, blk in enumerate(blocks):
        table[k * CB_ROWS:k * CB_ROWS + blk.shape[0]] = blk
    out["qwen3tts.frame_embed.weight"] = table
    for k, v in sd.items():
        if k.startswith(cp + "model.layers.") or k == cp + "model.norm.weight":
            out["qwen3tts.depth." + k[len(cp + "model."):]] = v
    out["qwen3tts.depth.proj.weight"] = sd[cp + "small_to_mtp_projection.weight"]
    out["qwen3tts.depth.proj.bias"] = sd[cp + "small_to_mtp_projection.bias"]
    for j in range(groups - 1):
        out[f"qwen3tts.depth.head.{j}.weight"] = sd[f"{cp}lm_head.{j}.weight"]
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}
    save_file(out, f"{a.out}/model.safetensors")

    max_seq = 4096
    qwen = dict(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=hidden,
        intermediate_size=tc["intermediate_size"], num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"], num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"], rms_norm_eps=tc["rms_norm_eps"], rope_theta=tc["rope_theta"], rope_scaling=None,
        vocab_size=E.shape[0], plow_head_rows=CODES + 1, max_position_embeddings=max_seq,
        tie_word_embeddings=False, torch_dtype="bfloat16", bos_token_id=None, eos_token_id=CODES,
        plow_frame_embed=dict(
            table="qwen3tts.frame_embed.weight", codebooks=groups, codebook_size=CB_ROWS, lo=0,
            rows=E.shape[0], classes=CODES, prefix="qwen3tts.depth",
            depth=dict(layers=cc["num_hidden_layers"], hidden=cc["hidden_size"], heads=cc["num_attention_heads"],
                       kv_heads=cc["num_key_value_heads"], head_dim=cc["head_dim"],
                       intermediate=cc["intermediate_size"], eps=cc["rms_norm_eps"], rope_theta=cc["rope_theta"],
                       rope_scaling=None, positions=groups, qk_norm=True, proj_bias=True,
                       min_frames=2, bf16_logits=True)),
    )
    json.dump(qwen, open(f"{a.out}/config.json", "w"), indent=1)
    json.dump({"eos_token_id": [CODES]}, open(f"{a.out}/generation_config.json", "w"))
    # The reference tokenizer (Qwen2 BPE, vocab.json + merges.txt) as tokenizer.json for plowrt.
    from transformers import AutoTokenizer
    # The reference processor loads it with `fix_mistral_regex=True` (qwen3_tts_model.py).
    tok = AutoTokenizer.from_pretrained(src, fix_mistral_regex=True)
    tok.save_pretrained(a.out)
    for f in ("vocab.json", "merges.txt"):
        shutil.copy(f"{src}/{f}", f"{a.out}/{f}")
    # The dual-stream prompt contract (plow_asset::dual_stream; devgen embeds it in the packet).
    ids = lambda *k: [tc[f"codec_{x}_id"] for x in k]
    json.dump(dict(
        text_template="<|im_start|>assistant\n{input}<|im_end|>\n<|im_start|>assistant\n",
        instructions_template="<|im_start|>user\n{instructions}<|im_end|>\n",
        lead=3, drop=5, text_pad=cfg["tts_pad_token_id"], text_bos=cfg["tts_bos_token_id"],
        text_eos=cfg["tts_eos_token_id"], none_code=CB_ROWS - 1, frame_codes=groups,
        codec_language=ids("think", "think_bos", "think_eos"), language_at=2,
        codec_auto=ids("nothink", "think_bos", "think_eos"), codec_pad=tc["codec_pad_id"],
        codec_bos=tc["codec_bos_id"], languages=tc["codec_language_id"],
        voices={k: dict(code=v, dialect=tc["spk_is_dialect"][k] or None) for k, v in tc["spk_id"].items()},
        dialect_languages=["chinese", "auto"]), open(f"{a.out}/dual_stream.json", "w"), indent=1)


if __name__ == "__main__":
    main()
