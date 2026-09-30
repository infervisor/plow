#!/usr/bin/env python3
"""Chatterbox Multilingual V3 T3 fp32 reference for the plow T3 numerics gate (t3_check.rs).

Per prompt (mtl_prompts.PROMPTS + MILESTONE): language, text ids (upstream punc_norm +
MTLTokenizer), last-prefill logits of the conditional / unconditional rows, and N greedy guided
steps (argmax of cond + w (cond - uncond), no penalty, no sampling). Same JSON as t3_ref.py.

  gpulease -n 1 t3-mtl-ref env PYTHONPATH= <v3 venv python> scripts/tts/t3_mtl_ref.py OUT.json [--steps 60]
"""
import argparse, glob, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mtl_prompts import PROMPTS, MILESTONE


def load_mtl(t3_model="v3", s3gen="s3gen_v3.safetensors", device="cuda"):
    """Upstream ChatterboxMultilingualTTS with the V3 T3; S3Gen weights from `s3gen` (the
    snapshot's s3gen_v3 vocoder unless `s3gen.pt`, which upstream mtl_tts loads)."""
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    src = glob.glob(os.path.join(hf, "hub/models--ResembleAI--chatterbox/snapshots/*/"))[0]
    m = ChatterboxMultilingualTTS.from_local(src, device, t3_model=t3_model)
    if s3gen != "s3gen.pt":
        from safetensors.torch import load_file
        m.s3gen.load_state_dict(load_file(os.path.join(src, s3gen)), strict=False)
        m.s3gen.to(device).eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--milestone", action="store_true", help="also the 23-language demo sentence")
    ap.add_argument("--bf16", action="store_true", help="run T3 in bf16 (the greedy-agreement floor of bf16 numerics)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from chatterbox.mtl_tts import punc_norm
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    m = load_mtl()
    t3, hp = m.t3, m.t3.hp
    if args.bf16:
        t3.to(torch.bfloat16)
        m.conds.t3 = m.conds.t3.to(dtype=torch.bfloat16)
    w = 0.5
    be = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head)
    prompts = list(PROMPTS) + (list(MILESTONE.items()) if args.milestone else [])
    out = []
    for lang, text in prompts:
        norm = punc_norm(text)
        ids = m.tokenizer.text_to_tokens(norm, language_id=lang)[0].tolist()
        tt = torch.tensor([[hp.start_text_token] + ids + [hp.stop_text_token]] * 2, device="cuda")
        with torch.inference_mode():
            embeds, _ = t3.prepare_input_embeds(t3_cond=m.conds.t3, text_tokens=tt,
                                                speech_tokens=hp.start_speech_token * torch.ones_like(tt[:, :1]),
                                                cfg_weight=w)
            bos = t3.speech_emb(torch.tensor([[hp.start_speech_token]], device="cuda")) + \
                t3.speech_pos_emb.get_fixed_embedding(0)
            embeds = torch.cat([embeds, torch.cat([bos, bos])], dim=1)
            r = be(inputs_embeds=embeds, past_key_values=None, use_cache=True, return_dict=True)
            first = r.logits[:, -1, :].float()
            past, logits, toks = r.past_key_values, first, []
            for i in range(args.steps):
                cfg = logits[0:1] + w * (logits[0:1] - logits[1:2])
                tok = int(cfg.argmax(-1))
                toks.append(tok)
                if tok == hp.stop_speech_token:
                    break
                e = t3.speech_emb(torch.tensor([[tok]], device="cuda")) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
                r = be(inputs_embeds=torch.cat([e, e]), past_key_values=past, use_cache=True, return_dict=True)
                past, logits = r.past_key_values, r.logits[:, -1, :].float()
        out.append(dict(language=lang, text=text, norm=norm, text_ids=ids, prefill_rows=int(embeds.shape[1]),
                        cond_logits=first[0].tolist(), uncond_logits=first[1].tolist(), greedy_cfg=toks))
        print(lang, len(ids), tuple(embeds.shape), toks[:8], flush=True)
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
