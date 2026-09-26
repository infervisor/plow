#!/usr/bin/env python3
"""Chatterbox (ResembleAI/chatterbox, English) -> a plow-compilable T3 checkpoint directory.

  HF_HOME=/root/tts-work/hf python scripts/tts/chatterbox_prep.py OUT_DIR

T3 is a Llama-520M whose inputs are embeddings, so the directory is Llama-shaped with the
T3-specific tables renamed into it (the compiler keeps checkpoint naming; the runtime binds roles):
  model.layers.* / model.norm.weight     <- tfmr.layers.* / tfmr.norm.weight
  model.embed_tokens.weight              <- speech_emb.weight      (decode token embedding)
  lm_head.weight                         <- speech_head.weight     (speech logits, 8194)
  model.speech_pos_emb.weight            <- speech_pos_emb.emb.weight (decode EmbedPosBf16)
  t3.text_emb.weight, t3.text_pos_emb.weight, t3.speech_emb_bos (host prefill rows)
config.json is the T3 LlamaConfig with vocab_size = 8194 plus a `chatterbox_t3` block (read by
devgen: overlay rows for the embedding prefill, speech position rows for decode). voices/<name>.f32
holds the voice's conditioning rows (T3CondEnc output, [cond_len, 1024] fp32, emotion_adv baked
at --exaggeration), computed by the reference module so the runtime never re-implements it.
tokenizer.json is the text tokenizer (EnTokenizer: spaces -> [SPACE], punc_norm on the host).
"""
import argparse, glob, json, os, shutil

import torch
from safetensors.torch import load_file, save_file

OVERLAY_ROWS = 512
T3_CONTRACT = dict(start_text=255, stop_text=0, start_speech=6561, stop_speech=6562, text_vocab=704,
                   speech_vocab=8194, max_speech_tokens=1000, cfg_weight=0.5, temperature=0.8,
                   min_p=0.05, top_p=1.0, repetition_penalty=1.2, s3_valid_below=6561)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--exaggeration", type=float, default=0.5)
    args = ap.parse_args()
    src = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                                 "hub/models--ResembleAI--chatterbox/snapshots/*"))[0]
    os.makedirs(os.path.join(args.out, "voices"), exist_ok=True)
    sd = load_file(f"{src}/t3_cfg.safetensors")
    out = {}
    for k, v in sd.items():
        if k.startswith("tfmr.layers.") or k == "tfmr.norm.weight":
            out["model." + k[len("tfmr."):]] = v
    bf = lambda t: t.to(torch.bfloat16).contiguous()
    out = {k: bf(v) for k, v in out.items()}
    out["model.embed_tokens.weight"] = bf(sd["speech_emb.weight"])
    out["lm_head.weight"] = bf(sd["speech_head.weight"])
    out["model.speech_pos_emb.weight"] = bf(sd["speech_pos_emb.emb.weight"])
    out["t3.text_emb.weight"] = sd["text_emb.weight"].float().contiguous()
    out["t3.text_pos_emb.weight"] = sd["text_pos_emb.emb.weight"].float().contiguous()
    # The first decode input (BOS) is a prefill row: speech_emb[start_speech] + speech_pos[0].
    out["t3.speech_bos"] = (sd["speech_emb.weight"][T3_CONTRACT["start_speech"]] +
                            sd["speech_pos_emb.emb.weight"][0]).float().contiguous()
    save_file(out, f"{args.out}/model.safetensors")

    from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
    cfg = dict(LLAMA_CONFIGS["Llama_520M"])
    cfg.pop("attn_implementation", None)
    cfg.update(architectures=["LlamaForCausalLM"], vocab_size=T3_CONTRACT["speech_vocab"],
               tie_word_embeddings=False, torch_dtype="bfloat16", bos_token_id=None, eos_token_id=None,
               chatterbox_t3=dict(overlay_rows=OVERLAY_ROWS, speech_pos_rows=int(sd["speech_pos_emb.emb.weight"].shape[0]),
                                  **T3_CONTRACT))
    json.dump(cfg, open(f"{args.out}/config.json", "w"), indent=1)
    json.dump({"eos_token_id": [T3_CONTRACT["stop_speech"]]}, open(f"{args.out}/generation_config.json", "w"))
    shutil.copy(f"{src}/tokenizer.json", f"{args.out}/tokenizer.json")

    # Voice conditioning rows through the reference conditioning encoder (fp32).
    import perth
    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    t3 = T3()
    t3.load_state_dict(sd)
    t3.eval()
    conds = torch.load(f"{src}/conds.pt", map_location="cpu", weights_only=True)["t3"]
    cond = T3Cond(speaker_emb=conds["speaker_emb"], cond_prompt_speech_tokens=conds["cond_prompt_speech_tokens"],
                  emotion_adv=args.exaggeration * torch.ones(1, 1, 1))
    with torch.no_grad():
        rows = t3.prepare_conditioning(cond)[0].float().contiguous()
    rows.numpy().tofile(f"{args.out}/voices/default.f32")
    json.dump({"default": {"rows": int(rows.shape[0]), "exaggeration": args.exaggeration}},
              open(f"{args.out}/voices/voices.json", "w"), indent=1)
    print(f"wrote {args.out}: {len(out)} tensors, voice rows {tuple(rows.shape)}")


if __name__ == "__main__":
    main()
