#!/usr/bin/env python3
"""Intelligibility gate for TTS output: Whisper round-trip, CER vs input text.

  nix develop .#speech-eval --command python3 scripts/bench/gpuq.py submit tts-asr 1 \
      nix develop .#speech-eval --command "$PYREF" scripts/tts/asr_check.py WAV... --texts texts.json

The result JSON is what veena_ref.py / chatterbox_ref.py / tts_bench.py write
(rows with i, speaker; texts are looked up from the prompt table the writer used,
stored as rows[*].text when present, else veena_ref.PROMPTS).
Prints per-file CER and the median; exits 1 if median CER > --max-cer.
A texts.json value may be {"text", "language"} (multilingual): Whisper then transcribes in that
language, punctuation / symbols (Unicode P*, S*) are dropped instead of non-word characters (which
would split Indic words at their vowel signs), zh / ja compare without spaces, and a per-language
median is printed.

--metric seed-tts-wer uses the English path of ByteDance Seed-TTS-eval (752f4297): FP32
Whisper large-v3, scipy Fourier resampling, English forced, punctuation stripped except the
ASCII apostrophe, uncapped mean per-sentence WER. This is a port, not the unmodified upstream
script; ground-truth ASR parity must be checked before comparison. --transcripts JSON re-scores
saved {rows: [{text, hyp, language, ...}]} without ASR and labels that scope in the report.
"""
import argparse, glob, hashlib, json, math, os, re, statistics, string, sys, unicodedata

sys.path.insert(0, os.path.dirname(__file__))


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def norm_lang(s, lang):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return "".join(s.split()) if lang in ("zh", "ja") else " ".join(s.split())


def cer_report(rows, model, limit):
    values = sorted(r["cer"] for r in rows)
    med = statistics.median(values)
    return dict(model=model, n=len(values), median_cer=med, mean_cer=sum(values) / len(values),
                p95_cer=values[math.ceil(0.95 * len(values)) - 1], max_cer=values[-1],
                over_limit_count=sum(c > limit for c in values),
                median_gate=dict(limit=limit, passed=med <= limit), rows=rows)


SEED_TTS_SOURCE = "https://github.com/BytedanceSpeech/seed-tts-eval/blob/752f4297f090c46bb1a55a1f7439e5944ddefe8d/run_wer.py"


def norm_seed_en(text):
    from zhon.hanzi import punctuation
    return text.translate(str.maketrans("", "", (punctuation + string.punctuation).replace("'", ""))).replace("  ", " ").lower()


def seed_wer_report(rows, model, limit=None, scope="saved transcripts; ASR protocol not verified"):
    import jiwer
    if not rows:
        raise ValueError("no transcripts to score")
    scored = []
    for row in rows:
        if row.get("language") not in ("en", "english"):
            raise ValueError("Seed-TTS English scoring requires language=en")
        ref, hyp = norm_seed_en(row["text"]), norm_seed_en(row["hyp"])
        if not ref.split():
            raise ValueError("empty normalized reference")
        result = jiwer.process_words(ref, hyp)
        scored.append(dict(row, normalized_reference=ref, normalized_hypothesis=hyp,
                           wer=result.wer, substitutions=result.substitutions,
                           deletions=result.deletions, insertions=result.insertions,
                           reference_words=result.hits + result.substitutions + result.deletions))
    values = sorted(row["wer"] for row in scored)
    mean = sum(values) / len(values)
    return dict(metric="mean_sentence_wer", protocol="seed-tts-en-port", source=SEED_TTS_SOURCE,
                scope=scope, model=model, n=len(values), mean_wer=mean,
                p95_wer=values[math.ceil(0.95 * len(values)) - 1], max_wer=values[-1],
                over_limit_count=None if limit is None else sum(v > limit for v in values),
                mean_gate=None if limit is None else dict(limit=limit, passed=mean <= limit),
                rows=scored)


def seed_en_asr(model_id, device):
    import torch
    import transformers
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    if (model.config.encoder_layers, model.config.decoder_layers,
            model.config.num_mel_bins, model.config.d_model) not in ((32, 32, 128, 1280), (32, 4, 128, 1280)):
        raise ValueError("Seed-TTS English requires Whisper large-v3 or large-v3-turbo")
    forced = processor.get_decoder_prompt_ids(language="english", task="transcribe")

    @torch.inference_mode()
    def transcribe(path):
        import soundfile as sf
        from scipy.signal import resample
        wav, rate = sf.read(path)
        if wav.ndim != 1 or not len(wav):
            raise ValueError(f"expected nonempty mono audio: {path}")
        if rate != 16000:
            wav = resample(wav, int(len(wav) * 16000 / rate))
        features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        ids = model.generate(features, forced_decoder_ids=forced)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    details = dict(torch=torch.__version__, transformers=transformers.__version__,
                   dtype=str(next(model.parameters()).dtype), device=str(next(model.parameters()).device),
                   forced_decoder_ids=forced, resampling="scipy.signal.resample",
                   generation_config=model.generation_config.to_dict())
    return transcribe, details


def seed_wer_main(args):
    if args.transcripts:
        if args.wavs:
            raise ValueError("choose WAVs or --transcripts")
        with open(args.transcripts, "rb") as file:
            raw = file.read()
        source = json.loads(raw)
        report = seed_wer_report(source["rows"], source.get("model"), args.max_wer)
        report["transcripts_sha256"] = hashlib.sha256(raw).hexdigest()
    else:
        if not args.wavs or not args.texts:
            raise ValueError("WAVs and --texts are required")
        model = args.model or "openai/whisper-large-v3"
        with open(args.texts) as file:
            texts = json.load(file)
        inputs = []
        for wav in sorted(args.wavs):
            ref = texts[os.path.basename(wav)]
            if isinstance(ref, dict):
                if ref.get("language") not in ("en", "english"):
                    raise ValueError("Seed-TTS English scoring requires language=en")
                ref = ref["text"]
            inputs.append((wav, ref))
        transcribe, details = seed_en_asr(model, args.device)
        rows = []
        for wav, ref in inputs:
            hyp = transcribe(wav)
            rows.append(dict(wav=wav, text=ref, language="en", hyp=hyp))
            print(f"{os.path.basename(wav)}\t{hyp[:90]}", flush=True)
        report = seed_wer_report(rows, model, args.max_wer,
            "FP32 Whisper large-v3 English ASR port; ground-truth parity not established by this run")
        report["asr"] = details
    if args.out:
        with open(args.out, "w") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"MEAN_SENTENCE_WER={report['mean_wer']:.9f} n={report['n']}")
    return 0 if report["mean_gate"] is None or report["mean_gate"]["passed"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--texts", help="json: {basename: text}")
    ap.add_argument("--max-cer", type=float, default=0.15)
    ap.add_argument("--model")
    ap.add_argument("--metric", choices=("cer", "seed-tts-wer"), default="cer")
    ap.add_argument("--max-wer", type=float, help="optional gate on uncapped mean per-sentence WER")
    ap.add_argument("--transcripts", help="score saved transcripts without loading ASR (seed-tts-wer only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", help="write full transcripts and exact CER results as JSON")
    args = ap.parse_args()
    if args.max_wer is not None and (not math.isfinite(args.max_wer) or args.max_wer < 0):
        ap.error("--max-wer must be finite and nonnegative")
    if args.metric == "seed-tts-wer":
        sys.exit(seed_wer_main(args))
    if args.transcripts or args.max_wer is not None or not args.wavs or not args.texts:
        ap.error("CER requires WAVs and --texts; --transcripts and --max-wer require seed-tts-wer")
    args.model = args.model or "openai/whisper-large-v3-turbo"
    import torch, jiwer, soundfile as sf, librosa
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=args.model, torch_dtype=torch.float16, device=args.device)
    texts = json.load(open(args.texts))
    cers, by_lang, rows = [], {}, []
    for w in sorted(args.wavs):
        ref = texts[os.path.basename(w)]
        lang = ref.get("language") if isinstance(ref, dict) else None
        ref = ref["text"] if isinstance(ref, dict) else ref
        a, sr = sf.read(w)
        if sr != 16000:
            a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        kw = dict(generate_kwargs=dict(language=lang, task="transcribe")) if lang else {}
        if len(a) > 30 * 16000:
            kw["return_timestamps"] = True  # Whisper's long-form decoding past one 30 s window
        hyp = asr({"raw": a.astype("float32"), "sampling_rate": 16000}, **kw)["text"]
        c = jiwer.cer(norm_lang(ref, lang), norm_lang(hyp, lang)) if lang else jiwer.cer(norm(ref), norm(hyp))
        cers.append(c)
        by_lang.setdefault(lang, []).append(c)
        rows.append(dict(wav=w, text=ref, language=lang, hyp=hyp.strip(), cer=c))
        print(f"{os.path.basename(w)}\tCER={c:.3f}\t{hyp.strip()[:90]}")
    if len(by_lang) > 1 or None not in by_lang:
        for lang, cs in sorted(by_lang.items(), key=lambda kv: str(kv[0])):
            print(f"LANG {lang} median_CER={statistics.median(cs):.3f} mean_CER={sum(cs) / len(cs):.3f} n={len(cs)}")
    med = statistics.median(cers)
    print(f"MEDIAN_CER={med:.3f} n={len(cers)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(cer_report(rows, args.model, args.max_cer), f, ensure_ascii=False, indent=2)
    sys.exit(0 if med <= args.max_cer else 1)


if __name__ == "__main__":
    main()
