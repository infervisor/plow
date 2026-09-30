#!/usr/bin/env python3
"""Chatterbox T3 fp32 reference for the plow T3 numerics gate.

For each prompt: the text token ids after punc_norm + [SPACE] (the runtime must produce the same),
the last-prefill-row logits of the conditional and unconditional rows, and N greedy classifier-
free-guidance steps (argmax of cond + w*(cond - uncond), no penalty, no sampling).

  gpulease -n 1 t3-ref python scripts/tts/t3_ref.py OUT.json [--steps 60]
"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from chatterbox_ref import PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    import perth
    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts import ChatterboxTTS, punc_norm
    m = ChatterboxTTS.from_pretrained(device="cuda")
    t3, hp = m.t3, m.t3.hp
    w = 0.5
    out = []
    for text in PROMPTS[: args.n]:
        norm = punc_norm(text)
        ids = m.tokenizer.text_to_tokens(norm)[0].tolist()
        tt = torch.tensor([[hp.start_text_token] + ids + [hp.stop_text_token]] * 2, device="cuda")
        with torch.inference_mode():
            embeds, _ = t3.prepare_input_embeds(t3_cond=m.conds.t3, text_tokens=tt,
                                                speech_tokens=hp.start_speech_token * torch.ones_like(tt[:, :1]),
                                                cfg_weight=w)
            # prepare_input_embeds ends with the BOS speech row; T3.inference appends a second one.
            bos = t3.speech_emb(torch.tensor([[hp.start_speech_token]], device="cuda")) + \
                t3.speech_pos_emb.get_fixed_embedding(0)
            embeds = torch.cat([embeds, torch.cat([bos, bos])], dim=1)
            from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
            be = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb,
                                      speech_head=t3.speech_head, alignment_stream_analyzer=None)
            r = be(inputs_embeds=embeds, past_key_values=None, use_cache=True, return_dict=True)
            first = r.logits[:, -1, :].float()
            past = r.past_key_values
            toks = []
            logits = first
            for i in range(args.steps):
                cfg = logits[0:1] + w * (logits[0:1] - logits[1:2])
                tok = int(cfg.argmax(-1))
                toks.append(tok)
                if tok == hp.stop_speech_token:
                    break
                e = t3.speech_emb(torch.tensor([[tok]], device="cuda")) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
                r = be(inputs_embeds=torch.cat([e, e]), past_key_values=past, use_cache=True, return_dict=True)
                past, logits = r.past_key_values, r.logits[:, -1, :].float()
        out.append(dict(text=text, norm=norm, text_ids=ids, prefill_rows=int(embeds.shape[1]),
                        cond_logits=first[0].tolist(), uncond_logits=first[1].tolist(), greedy_cfg=toks))
        print(len(ids), embeds.shape, toks[:10])
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
