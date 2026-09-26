#!/usr/bin/env python3
"""Intelligibility gate for TTS output: Whisper round-trip, CER vs input text.

  gpulease -n 1 tts-asr python scripts/tts/asr_check.py <result.json> <wav-dir> <glob-prefix>

The result JSON is what veena_ref.py / chatterbox_ref.py / tts_bench.py write
(rows with i, speaker; texts are looked up from the prompt table the writer used,
stored as rows[*].text when present, else veena_ref.PROMPTS).
Prints per-file CER and the median; exits 1 if median CER > --max-cer.
"""
import argparse, glob, json, os, re, statistics, sys, unicodedata

sys.path.insert(0, os.path.dirname(__file__))


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--texts", required=True, help="json: {basename: text}")
    ap.add_argument("--max-cer", type=float, default=0.15)
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    args = ap.parse_args()
    import torch, jiwer, soundfile as sf, librosa
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=args.model, torch_dtype=torch.float16, device="cuda")
    texts = json.load(open(args.texts))
    cers = []
    for w in sorted(args.wavs):
        ref = texts[os.path.basename(w)]
        a, sr = sf.read(w)
        if sr != 16000:
            a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        hyp = asr({"raw": a.astype("float32"), "sampling_rate": 16000})["text"]
        c = jiwer.cer(norm(ref), norm(hyp))
        cers.append(c)
        print(f"{os.path.basename(w)}\tCER={c:.3f}\t{hyp.strip()[:90]}")
    med = statistics.median(cers)
    print(f"MEDIAN_CER={med:.3f} n={len(cers)}")
    sys.exit(0 if med <= args.max_cer else 1)


if __name__ == "__main__":
    main()
