#!/usr/bin/env python3
"""Qwen3-TTS (12Hz, CustomVoice) reference: the official `qwen_tts` model driven by an explicit
streaming-text loop (`non_streaming_mode=False`, the tts-bench request), dumping the per-frame codes,
the talker / code-predictor logits and the decoded PCM that the plow packet is gated against.

  python scripts/tts/qwen3_tts_ref.py MODEL_DIR TEXTS.jsonl OUT_DIR [--voice ryan] [--language english]
      [--dtype bf16|fp32] [--device cuda] [--seed S | greedy] [--logits] [--official] [--limit N]
      [--force CODES_DIR]

TEXTS.jsonl rows: {"id": ..., "text": ...}. Per row OUT_DIR gets `<id>.codes.npy` (int32 [frames][16],
EOS frame excluded), `<id>.wav` (24 kHz), with --logits `<id>.logits.npz` (cb0 [frames+1][3072] after
the reference's suppression, cp [frames][15][2048], prefill [rows][2048] input embeddings), and one
line of OUT_DIR/meta.jsonl (frames, audible onset ms, seconds, wall ms). --official also runs the
package's own `generate` with the same settings and records whether its codes equal the loop's.
--force teacher-forces every frame to CODES_DIR/<id>.codes.npy (then EOS), so logits of two runs
(dtypes, or plow) compare position by position; the logits then also hold `cb0_raw`, the head's
3072 logits before suppression and penalty.
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

EOS, CODES, VOCAB = 2150, 2048, 3072


def onset_ms(pcm):
    """tts-bench AudibleOnsetDetector v1: 20 ms frames, 10 ms hop, DC-removed RMS >= -45 dBFS twice."""
    x = np.clip(np.round(pcm * 32767.0), -32768, 32767).astype(np.int16).astype(np.float32) / 32768.0
    f, h, thr = 480, 240, 10 ** (-45 / 20)
    run, start = 0, None
    for s in range(0, len(x) - f + 1, h):
        w = x[s:s + f] - x[s:s + f].mean()
        if np.sqrt(np.mean(w * w)) >= thr:
            if run == 0:
                start = s
            run += 1
            if run >= 2:
                return start * 1000.0 / 24000
        else:
            run = 0
    return None


class Loop:
    def __init__(self, m, tok, voice, language):
        self.m, self.t, self.cfg = m, m.talker, m.config
        tc = self.cfg.talker_config
        self.tok = tok
        self.dev = next(m.parameters()).device
        self.spk = tc.spk_id[voice.lower()]
        lang = tc.codec_language_id[language.lower()]
        self.codec_prompt = [tc.codec_think_id, tc.codec_think_bos_id, lang, tc.codec_think_eos_id,
                             self.spk, tc.codec_pad_id, tc.codec_bos_id]

    def T(self, ids):
        ids = torch.tensor([ids], device=self.dev)
        return self.t.text_projection(self.t.get_text_embeddings()(ids))

    def C(self, ids):
        return self.t.get_input_embeddings()(torch.tensor([ids], device=self.dev))

    def prompt(self, text):
        ids = self.tok(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")["input_ids"]
        bos, eos, pad = self.T([self.cfg.tts_bos_token_id, self.cfg.tts_eos_token_id,
                                self.cfg.tts_pad_token_id]).chunk(3, dim=1)
        codec = self.C(self.codec_prompt)
        n = codec.shape[1]
        head = torch.cat([pad.expand(-1, n - 2, -1), bos], dim=1) + codec[:, :-1]
        rows = torch.cat([self.T(ids[:3]), head, self.T(ids[3:4]) + codec[:, -1:]], dim=1)
        trailing = torch.cat([self.T(ids[4:-5]), eos], dim=1)
        return ids, rows, trailing, pad

    def sample(self, logits, greedy, gen, top_k=50, temperature=0.9):
        if greedy:
            return int(torch.argmax(logits).item())
        v, i = torch.topk(logits.float() / temperature, top_k)
        p = torch.softmax(v, -1)
        return int(i[torch.multinomial(p, 1, generator=gen)].item())

    @torch.inference_mode()
    def run(self, text, greedy, seed, max_frames, dump, force=None):
        from transformers import DynamicCache
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(seed)
        ids, rows, trailing, pad = self.prompt(text)
        cache = DynamicCache()
        L = rows.shape[1]
        pos = torch.arange(L, device=self.dev).view(1, 1, -1).expand(3, 1, -1)
        out = self.t.model(inputs_embeds=rows, position_ids=pos, past_key_values=cache, use_cache=True)
        h = out.last_hidden_state[:, -1:]
        cp = self.t.code_predictor
        frames, l0s, lcps, hist, raws = [], [], [], [], []
        for step in range(max_frames if force is None else len(force) + 1):
            logits = self.t.codec_head(h)[0, -1].float()
            raws.append(logits.clone().cpu())
            logits[CODES:] = -float("inf")
            if step >= 2:
                logits[EOS] = self.t.codec_head(h)[0, -1, EOS].float()
            for t_ in set(hist):
                logits[t_] = logits[t_] * 1.05 if logits[t_] < 0 else logits[t_] / 1.05
            l0s.append(logits.cpu())
            c0 = self.sample(logits, greedy, gen)
            if force is not None:
                c0 = int(force[step][0]) if step < len(force) else EOS
            if c0 == EOS:
                break
            hist.append(c0)
            codes, lcp = [c0], []
            e0 = self.t.get_input_embeddings()(torch.tensor([[c0]], device=self.dev))
            cpc = DynamicCache()
            x = cp.small_to_mtp_projection(torch.cat([h, e0], dim=1))
            for j in range(15):
                o = cp.model(inputs_embeds=x, past_key_values=cpc, use_cache=True)
                lg = cp.lm_head[j](o.last_hidden_state[:, -1]).float()[0]
                lcp.append(lg.cpu())
                c = self.sample(lg, greedy, gen)
                if force is not None:
                    c = int(force[step][j + 1])
                codes.append(c)
                x = cp.small_to_mtp_projection(cp.model.get_input_embeddings()[j](
                    torch.tensor([[c]], device=self.dev)))
            frames.append(codes)
            lcps.append(torch.stack(lcp))
            # The reference sums the 16 rows in one reduction (`codec_hiddens.sum(1)`).
            emb = torch.cat([e0] + [cp.model.get_input_embeddings()[j](
                torch.tensor([[codes[j + 1]]], device=self.dev)) for j in range(15)], dim=1).sum(1, keepdim=True)
            emb = emb + (trailing[:, step:step + 1] if step < trailing.shape[1] else pad)
            p = torch.full((3, 1, 1), L + step, device=self.dev)
            h = self.t.model(inputs_embeds=emb, position_ids=p, past_key_values=cache,
                             use_cache=True).last_hidden_state
        res = dict(ids=ids, codes=np.array(frames, dtype=np.int32).reshape(-1, 16))
        if dump:
            res["logits"] = dict(cb0=torch.stack(l0s).numpy(), cb0_raw=torch.stack(raws).numpy(),
                                 cp=(torch.stack(lcps).numpy() if lcps else np.zeros((0, 15, CODES), np.float32)),
                                 prefill=rows[0].float().cpu().numpy())
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--voice", default="ryan")
    ap.add_argument("--language", default="english")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None, help="sample with this seed (default greedy)")
    ap.add_argument("--max-frames", type=int, default=2048)
    ap.add_argument("--logits", action="store_true")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", default=None)
    a = ap.parse_args()
    from qwen_tts import Qwen3TTSModel
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    w = Qwen3TTSModel.from_pretrained(a.model, dtype=dt, device_map=a.device)
    tok = w.processor.tokenizer if hasattr(w.processor, "tokenizer") else w.processor
    loop = Loop(w.model, tok, a.voice, a.language)
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.texts) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    greedy = a.seed is None
    meta = open(os.path.join(a.out, "meta.jsonl"), "a")
    for r in rows:
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        force = np.load(os.path.join(a.force, f"{r['id']}.codes.npy")) if a.force else None
        res = loop.run(r["text"], greedy, a.seed or 0, a.max_frames, a.logits, force)
        codes = res["codes"]
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        wavs, sr = w.model.speech_tokenizer.decode([{"audio_codes": torch.from_numpy(codes).long().to(a.device)}])
        pcm = np.asarray(wavs[0], dtype=np.float32)
        sf.write(os.path.join(a.out, f"{r['id']}.wav"), pcm, sr)
        np.save(os.path.join(a.out, f"{r['id']}.codes.npy"), codes)
        if a.logits:
            np.savez_compressed(os.path.join(a.out, f"{r['id']}.logits.npz"), **res["logits"])
        rec = dict(id=r["id"], text=r["text"], ids=res["ids"], frames=int(codes.shape[0]),
                   seconds=len(pcm) / sr, onset_ms=onset_ms(pcm), wall_ms=round(ms, 1),
                   dtype=a.dtype, seed=a.seed, voice=a.voice)
        if a.official:
            if not greedy:
                torch.manual_seed(a.seed)
            off, _ = w.model.generate(
                input_ids=[torch.tensor([res["ids"]], device=a.device)], instruct_ids=[None],
                languages=[a.language], speakers=[a.voice], non_streaming_mode=False,
                do_sample=not greedy, subtalker_dosample=not greedy, top_k=50, top_p=1.0,
                temperature=0.9, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9,
                repetition_penalty=1.05, max_new_tokens=a.max_frames)
            oc = off[0].cpu().numpy().astype(np.int32)
            n = min(len(oc), len(codes))
            rec["official_frames"] = int(len(oc))
            rec["official_prefix_match"] = int(next((i for i in range(n) if not (oc[i] == codes[i]).all()), n))
        meta.write(json.dumps(rec) + "\n")
        meta.flush()
        print(json.dumps({k: rec[k] for k in rec if k != "ids"}), flush=True)


if __name__ == "__main__":
    main()
