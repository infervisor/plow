#!/usr/bin/env python3
"""Chatterbox Multilingual V3 reference TTS on CUDA (upstream mtl_tts, t3_model="v3").

Stock path: T3 (Llama-520M fp32, CFG batch 2, HF eager loop, alignment analyzer removed in V3) ->
S3Gen (s3gen_v3 vocoder weights by default; --s3gen s3gen.pt is what upstream mtl_tts loads).
Timed per stage; the Perth watermark is excluded. Writes wavs, texts.json (language-tagged, for
asr_check.py) and cbx_mtl_stock.json.

  gpulease -n 1 cbx-mtl-ref env PYTHONPATH= <v3 venv python> scripts/tts/chatterbox_mtl_ref.py --out DIR
"""
import argparse, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from mtl_prompts import PROMPTS
from t3_mtl_ref import load_mtl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(PROMPTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s3gen", default="s3gen_v3.safetensors")
    ap.add_argument("--t3-bf16", action="store_true", help="T3 in bf16 (plow's weight precision) instead of fp32")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import torch, soundfile as sf
    m = load_mtl(s3gen=args.s3gen)
    if args.t3_bf16:
        m.t3.to(torch.bfloat16)
        m.conds.t3 = m.conds.t3.to(dtype=torch.bfloat16)
    stage = {}

    def timed(name, fn):
        def w(*a, **k):
            torch.cuda.synchronize(); t = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize(); stage[name] = time.perf_counter() - t
            if name == "t3":
                stage["n_speech_tokens"] = int(r.shape[-1])
            return r
        return w
    m.t3.inference = timed("t3", m.t3.inference)
    m.s3gen.inference = timed("s3gen", m.s3gen.inference)

    m.generate("Warm up the kernels once.", language_id="en")
    rows, texts = [], {}
    for i in range(args.n):
        lang, text = PROMPTS[i % len(PROMPTS)]
        torch.manual_seed(args.seed + i)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        wav = m.generate(text, language_id=lang)
        torch.cuda.synchronize(); tot = time.perf_counter() - t0
        a = wav.squeeze(0).numpy()
        dur = len(a) / m.sr
        name = f"ref_{i:02d}_{lang}.wav"
        sf.write(os.path.join(args.out, name), a, m.sr)
        texts[name] = dict(text=text, language=lang)
        synth = stage["t3"] + stage["s3gen"]
        rows.append(dict(i=i, language=lang, text=text, audio_s=dur, t3_s=stage["t3"], s3gen_s=stage["s3gen"],
                         ntok=stage["n_speech_tokens"], total_s=tot, rtf=synth / dur,
                         t3_ms_per_tok=1000 * stage["t3"] / max(1, stage["n_speech_tokens"])))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summ = dict(engine="chatterbox-mtl-v3-stock", s3gen=args.s3gen, n=len(rows),
                audio_s=sum(r["audio_s"] for r in rows),
                audio_s_per_s=sum(r["audio_s"] for r in rows) / sum(r["total_s"] for r in rows),
                med_rtf=statistics.median(r["rtf"] for r in rows),
                med_t3_s=statistics.median(r["t3_s"] for r in rows),
                med_s3gen_s=statistics.median(r["s3gen_s"] for r in rows),
                med_t3_ms_per_tok=statistics.median(r["t3_ms_per_tok"] for r in rows))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(args.out, "cbx_mtl_stock.json"), "w"), indent=1,
              ensure_ascii=False)
    json.dump(texts, open(os.path.join(args.out, "texts.json"), "w"), ensure_ascii=False, indent=0)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
