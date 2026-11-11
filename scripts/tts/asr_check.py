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


def forbid_hits(hyp, terms, lang=None):
    if not isinstance(terms, list) or any(not isinstance(t, str) or not t.strip() for t in terms):
        raise ValueError("forbid must be a list of nonempty words or phrases")
    normalize = (lambda s: norm_lang(s, lang)) if lang else norm
    text = normalize(hyp)
    word = lambda c: c.isalnum() or c == "_" or unicodedata.category(c).startswith("M")
    hits = []
    for term in terms:
        needle = normalize(term)
        if not needle:
            raise ValueError("forbid term has no normalized characters")
        start = 0
        while (at := text.find(needle, start)) >= 0:
            end = at + len(needle)
            if lang in ("zh", "ja") or ((at == 0 or not word(text[at - 1])) and
                                         (end == len(text) or not word(text[end]))):
                if term not in hits:
                    hits.append(term)
                break
            start = at + 1
    return hits


def silence_runs(samples, rate, dbfs=-60.0):
    if not isinstance(rate, int) or rate <= 0:
        raise ValueError("invalid audio sample rate")
    block = max(1, round(rate * 0.005))
    threshold = 10 ** (dbfs / 10)
    runs, start = [], None
    for at in range(0, len(samples), block):
        part = samples[at:at + block]
        if any(not math.isfinite(float(x)) for x in part):
            raise ValueError("nonfinite audio sample")
        silent = sum(float(x) ** 2 for x in part) / len(part) <= threshold
        if silent and start is None:
            start = at
        elif not silent and start is not None:
            runs.append((start * 1000 / rate, at * 1000 / rate))
            start = None
    if start is not None:
        runs.append((start * 1000 / rate, len(samples) * 1000 / rate))
    return runs


def number(value, label, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return float(value)


def validate_signal(meta):
    pauses = meta.get("pauses", [])
    if not isinstance(pauses, list):
        raise ValueError("pauses must be a list")
    for pause in pauses:
        if not isinstance(pause, dict) or "ms" not in pause:
            raise ValueError("pause requires ms")
        number(pause["ms"], "pause ms")
        position = pause.get("position", "internal")
        if position not in ("leading", "trailing", "internal"):
            raise ValueError("pause position must be leading, trailing or internal")
        if position == "internal" and "window_ms" not in pause:
            raise ValueError("internal pause requires window_ms around its join")
        if "window_ms" in pause:
            window = pause["window_ms"]
            if not isinstance(window, list) or len(window) != 2:
                raise ValueError("window_ms must contain two bounds")
            if number(window[0], "window start") > number(window[1], "window end"):
                raise ValueError("reversed pause window")
    if "duration_delta" in meta:
        spec = meta["duration_delta"]
        if not isinstance(spec, dict) or not isinstance(spec.get("baseline"), str) or not spec["baseline"]:
            raise ValueError("duration_delta requires a baseline WAV")
        number(spec.get("ms"), "duration delta ms", nonnegative=False)


def signal_report(samples, rate, meta, directory, tolerance_ms=80.0):
    import wave
    validate_signal(meta)
    runs = silence_runs(samples, rate)
    duration = len(samples) * 1000 / rate
    checks, used = [], set()
    for spec in meta.get("pauses", []):
        want = float(spec["ms"])
        position = spec.get("position", "internal")
        eligible = []
        for i, (start, end) in enumerate(runs):
            if i in used:
                continue
            if position == "leading" and start > 5:
                continue
            if position == "trailing" and duration - end > 5:
                continue
            if position == "internal" and (start <= 5 or duration - end <= 5):
                continue
            if "window_ms" in spec and not spec["window_ms"][0] <= start <= spec["window_ms"][1]:
                continue
            eligible.append((abs(end - start - want), i, start, end))
        if eligible:
            error, i, start, end = min(eligible)
            used.add(i)
            checks.append(dict(requested_ms=want, start_ms=start, end_ms=end,
                               measured_ms=end - start, error_ms=end - start - want,
                               passed=error <= tolerance_ms))
        else:
            checks.append(dict(requested_ms=want, measured_ms=None, error_ms=None, passed=False))
    delta = None
    if "duration_delta" in meta:
        spec = meta["duration_delta"]
        path = os.path.join(directory, spec["baseline"])
        with wave.open(path, "rb") as wav:
            baseline = wav.getnframes() * 1000 / wav.getframerate()
        observed = duration - baseline
        delta = dict(baseline=spec["baseline"], requested_ms=spec["ms"],
                     measured_ms=observed, error_ms=observed - spec["ms"],
                     passed=abs(observed - spec["ms"]) <= tolerance_ms)
    return dict(duration_ms=duration, silence_dbfs=-60.0, frame_ms=5.0,
                tolerance_ms=tolerance_ms, pauses=checks, duration_delta=delta,
                passed=all(c["passed"] for c in checks) and (delta is None or delta["passed"]))


def relative_cer(row, baseline, margin):
    candidates = baseline["rows"]
    key = row.get("baseline")
    if key:
        candidates = [b for b in candidates if key in (b.get("case_id"), os.path.basename(b.get("wav", "")))]
    else:
        candidates = [b for b in candidates if (b.get("language"), b["text"]) == (row.get("language"), row["text"])]
    if len(candidates) != 1:
        raise ValueError(f"baseline must resolve exactly one row for {row.get('wav')}")
    control = candidates[0]
    if (control.get("language"), control["text"]) != (row.get("language"), row["text"]):
        raise ValueError("baseline language/reference mismatch")
    cer = number(control["cer"], "baseline CER")
    return dict(baseline_cer=cer, margin=margin, limit=cer + margin,
                delta=row["cer"] - cer, passed=row["cer"] <= cer + margin)


def tag_gates(rows, baseline=None, margin=None):
    for row in rows:
        row["forbid_hits"] = forbid_hits(row["hyp"], row.get("forbid", []), row.get("language"))
        if baseline is not None:
            row["relative_cer"] = relative_cer(row, baseline, margin)
    return dict(forbid_hits=sum(len(r["forbid_hits"]) for r in rows),
                passed=all(not r["forbid_hits"] and r.get("signal", {}).get("passed", True)
                           and r.get("relative_cer", {}).get("passed", True) for r in rows))


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
    if model_id == "openai/whisper-large-v3-turbo":
        raise ValueError("Seed-TTS English requires Whisper large-v3, not turbo")
    import torch
    import transformers
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    if (model.config.encoder_layers, model.config.decoder_layers,
            model.config.num_mel_bins, model.config.d_model) != (32, 32, 128, 1280):
        raise ValueError("Seed-TTS English requires Whisper large-v3")
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
    ap.add_argument("--forbid", help="JSON {WAV basename: [forbidden words/phrases]}; also accepted per texts row")
    ap.add_argument("--baseline-report", help="ASR report containing unique tag-free baseline rows")
    ap.add_argument("--cer-margin", type=float, help="maximum per-row CER increase over baseline")
    ap.add_argument("--transcripts", help="score saved transcripts without loading ASR (seed-tts-wer only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", help="write full transcripts and exact CER results as JSON")
    args = ap.parse_args()
    if args.max_wer is not None and (not math.isfinite(args.max_wer) or args.max_wer < 0):
        ap.error("--max-wer must be finite and nonnegative")
    if args.metric == "seed-tts-wer":
        if args.forbid or args.baseline_report or args.cer_margin is not None:
            ap.error("tag gates require --metric cer")
        sys.exit(seed_wer_main(args))
    if args.transcripts or args.max_wer is not None or not args.wavs or not args.texts:
        ap.error("CER requires WAVs and --texts; --transcripts and --max-wer require seed-tts-wer")
    args.model = args.model or "openai/whisper-large-v3-turbo"
    number(args.max_cer, "--max-cer")
    if bool(args.baseline_report) != (args.cer_margin is not None):
        ap.error("--baseline-report and --cer-margin are required together")
    baseline = None
    if args.baseline_report:
        number(args.cer_margin, "--cer-margin")
        with open(args.baseline_report) as file:
            baseline = json.load(file)
        if baseline.get("model") != args.model:
            raise ValueError("baseline ASR model mismatch")
    with open(args.texts) as file:
        texts = json.load(file)
    forbidden = {}
    if args.forbid:
        with open(args.forbid) as file:
            forbidden = json.load(file)
        if not isinstance(forbidden, dict) or set(forbidden) - {os.path.basename(w) for w in args.wavs}:
            raise ValueError("--forbid keys must identify requested WAV basenames")
    inputs = []
    for wav in sorted(args.wavs):
        value = texts[os.path.basename(wav)]
        meta = dict(value) if isinstance(value, dict) else dict(text=value)
        if not isinstance(meta.get("text"), str) or not meta["text"].strip():
            raise ValueError("empty ASR reference")
        terms, extra = meta.get("forbid", []), forbidden.get(os.path.basename(wav), [])
        forbid_hits("", terms, meta.get("language"))
        forbid_hits("", extra, meta.get("language"))
        meta["forbid"] = terms + extra
        validate_signal(meta)
        if baseline is not None:
            relative_cer(dict(meta, wav=wav, cer=0), baseline, args.cer_margin)
        inputs.append((wav, meta))
    import torch, jiwer, soundfile as sf, librosa
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=args.model, torch_dtype=torch.float16, device=args.device)
    cers, by_lang, rows = [], {}, []
    for w, meta in inputs:
        ref, lang = meta["text"], meta.get("language")
        a, sr = sf.read(w)
        if a.ndim != 1 or not len(a):
            raise ValueError(f"expected nonempty mono audio: {w}")
        signal = signal_report(a, sr, meta, os.path.dirname(w))
        if sr != 16000:
            a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        kw = dict(generate_kwargs=dict(language=lang, task="transcribe")) if lang else {}
        if len(a) > 30 * 16000:
            kw["return_timestamps"] = True  # Whisper's long-form decoding past one 30 s window
        hyp = asr({"raw": a.astype("float32"), "sampling_rate": 16000}, **kw)["text"]
        c = jiwer.cer(norm_lang(ref, lang), norm_lang(hyp, lang)) if lang else jiwer.cer(norm(ref), norm(hyp))
        cers.append(c)
        by_lang.setdefault(lang, []).append(c)
        rows.append(dict(meta, wav=w, text=ref, language=lang, hyp=hyp.strip(), cer=c, signal=signal))
        print(f"{os.path.basename(w)}\tCER={c:.3f}\t{hyp.strip()[:90]}")
    if len(by_lang) > 1 or None not in by_lang:
        for lang, cs in sorted(by_lang.items(), key=lambda kv: str(kv[0])):
            print(f"LANG {lang} median_CER={statistics.median(cs):.3f} mean_CER={sum(cs) / len(cs):.3f} n={len(cs)}")
    med = statistics.median(cers)
    print(f"MEDIAN_CER={med:.3f} n={len(cers)}")
    gates = tag_gates(rows, baseline, args.cer_margin)
    report = dict(cer_report(rows, args.model, args.max_cer), tag_gate=gates)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    sys.exit(0 if med <= args.max_cer and gates["passed"] else 1)


if __name__ == "__main__":
    main()
