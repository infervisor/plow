#!/usr/bin/env python3
"""Chatterbox (ResembleAI/chatterbox, English) reference TTS on CUDA.

Stock `chatterbox-tts` path: T3 (Llama-520M, CFG batch 2, HF eager loop) ->
S3Gen (conformer + CFM 10 steps + HiFT). Timed per stage; the Perth watermark
is excluded from timing (it is post-processing plow does not do).

  gpulease -n 1 cbx-ref python scripts/tts/chatterbox_ref.py --out $RESULTS/cbx
"""
import argparse, json, os, statistics, sys, time

PROMPTS = [
    "Today I learned about a new technology that uses artificial intelligence to generate human-like voices.",
    "The quick brown fox jumps over the lazy dog, and then it takes a long nap in the warm afternoon sun.",
    "Hello, how are you doing today?",
    "Please confirm your appointment for tomorrow at three thirty in the afternoon.",
    "Streaming speech synthesis needs both low latency for the first chunk and high throughput under load.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "Your package has shipped and should arrive within two business days.",
    "Machine learning systems are only as good as the data they are trained on.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-tokens", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import torch, soundfile as sf, perth
    if perth.PerthImplicitWatermarker is None:  # needs pkg_resources; watermark is not timed anyway
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained(device="cuda")

    stage = {}
    t3_inf, s3_inf = m.t3.inference, m.s3gen.inference

    def timed(name, fn):
        def w(*a, **k):
            torch.cuda.synchronize(); t = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize(); stage[name] = time.perf_counter() - t
            if name == "t3":
                stage["n_speech_tokens"] = int(r.shape[-1])
            return r
        return w
    m.t3.inference = timed("t3", t3_inf)
    m.s3gen.inference = timed("s3gen", s3_inf)

    m.generate("Warm up the kernels once.")  # excluded
    rows = []
    for i in range(args.n):
        text = PROMPTS[i % len(PROMPTS)]
        torch.manual_seed(args.seed + i)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        wav = m.generate(text)
        torch.cuda.synchronize(); tot = time.perf_counter() - t0
        a = wav.squeeze(0).numpy()
        dur = len(a) / m.sr
        sf.write(f"{args.out}/cbx_{i:02d}.wav", a, m.sr)
        synth = stage["t3"] + stage["s3gen"]
        rows.append(dict(i=i, text=text, audio_s=dur, t3_s=stage["t3"], s3gen_s=stage["s3gen"],
                         ntok=stage["n_speech_tokens"], total_s=tot, rtf=synth / dur,
                         t3_ms_per_tok=1000 * stage["t3"] / max(1, stage["n_speech_tokens"])))
        print(json.dumps(rows[-1]))
    summ = dict(engine="chatterbox-stock", n=len(rows),
                audio_s=sum(r["audio_s"] for r in rows),
                med_rtf=statistics.median(r["rtf"] for r in rows),
                med_t3_s=statistics.median(r["t3_s"] for r in rows),
                med_s3gen_s=statistics.median(r["s3gen_s"] for r in rows),
                med_t3_ms_per_tok=statistics.median(r["t3_ms_per_tok"] for r in rows))
    json.dump(dict(summary=summ, rows=rows), open(f"{args.out}/cbx_stock.json", "w"), indent=1)
    json.dump({f"cbx_{r['i']:02d}.wav": r["text"] for r in rows}, open(f"{args.out}/texts.json", "w"), indent=0)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
