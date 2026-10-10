#!/usr/bin/env python3
"""Quality evaluation against a running `plowrt serve` (models found by their advertised endpoints).

  eval.py asr  --manifest M [--models a,b]          WER per ASR model
  eval.py tts  [--models a,b] [--whisper-device cuda|cpu]
                                                    Whisper round-trip CER per TTS model and language
  eval.py llm  [--models m]                         functional checks of one chat model
  eval.py all  --manifest M --out DIR               all three; writes eval.json + eval.md

$PLOW_URL (default http://127.0.0.1:8000) names the server, $PLOW_API_KEY a bearer token.
ASR: every clip of the manifest ([{path, text, dur}], paths relative to it; default $ASR_MANIFEST)
is transcribed at concurrency 16; WER uses the release-gate normalization (lowercase,
[a-z0-9'] words). --librispeech <dir with *.trans.txt and .flac> reads a LibriSpeech split instead
(needs soundfile).
TTS: the gate prompt set of the repo harnesses (scripts/tts/{mtl_prompts,veena_ref,chatterbox_ref}.py;
--tts-prompts, default by model name) is synthesized as WAV, then transcribed by Whisper
large-v3-turbo (needs torch, transformers, jiwer); CER after Unicode punctuation stripping, without
spaces for zh/ja.
LLM: deterministic answers, chat template, streaming, stop strings, logprobs, raw <bos> completions,
the context limit (from the card's max_model_len), and error envelopes.
Thresholds are the recipes' [gates] (campaign.py gate): this reports numbers, and exits 1 only on
request errors or a failed LLM check.
"""
import argparse
import concurrent.futures as cf
import io
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tts"))
from chatterbox_ref import PROMPTS as CBX_TEXTS  # noqa: E402
from mtl_prompts import PROMPTS as MTL_PROMPTS  # noqa: E402
from veena_ref import PROMPTS as VEENA_PROMPTS  # noqa: E402

URL = os.environ.get("PLOW_URL", "http://127.0.0.1:8000").rstrip("/")
AUTH = {"Authorization": f"Bearer {os.environ['PLOW_API_KEY']}"} if os.environ.get("PLOW_API_KEY") else {}



def call(method, path, body=None, headers=None, timeout=600):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    h = dict(AUTH)
    if isinstance(body, dict):
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    req = urllib.request.Request(URL + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def models_by_kind():
    st, b = call("GET", "/v1/models")
    cards = [c for c in json.loads(b)["data"] if not c.get("parent")]
    asr = [c["id"] for c in cards if "audio/transcriptions" in c.get("x_plow_endpoints", [])]
    tts = [c["id"] for c in cards if "audio/speech" in c.get("x_plow_endpoints", [])]
    llm = [c["id"] for c in cards if "chat/completions" in c.get("x_plow_endpoints", [])]
    return asr, tts, llm, {c["id"]: c.get("max_model_len") for c in cards}


# ------------------------------------------------------------------ ASR
def norm_en(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def multipart(fields, fname, data):
    boundary = "plowvoiceeval"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def librispeech_manifest(root, limit):
    import soundfile as sf
    items = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.endswith(".trans.txt"):
                for line in open(os.path.join(dirpath, fn)):
                    uid, text = line.strip().split(" ", 1)
                    items.append((os.path.join(dirpath, uid + ".flac"), text))
    items.sort()
    out = []
    for path, text in items[: limit or None]:
        a, sr = sf.read(path, dtype="int16")
        buf = io.BytesIO()
        sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
        out.append(dict(data=buf.getvalue(), text=text, dur=len(a) / sr, name=os.path.basename(path)))
    return out


def eval_asr(a, models):
    if a.librispeech:
        clips = librispeech_manifest(a.librispeech, a.limit)
        src = a.librispeech
    else:
        man = a.manifest
        if not man:
            sys.exit("asr: --manifest (or $ASR_MANIFEST) or --librispeech is required")
        base = os.path.dirname(os.path.abspath(man))
        clips = [dict(data=open(os.path.join(base, m["path"]), "rb").read(), text=m["text"], dur=m["dur"],
                      name=os.path.basename(m["path"])) for m in json.load(open(man))[: a.limit or None]]
        src = man
    import jiwer
    results = {}
    for m in models:
        def one(c):
            body, h = multipart({"model": m, "language": "en"} if a.language_hint else {"model": m}, c["name"], c["data"])
            st, b = call("POST", "/v1/audio/transcriptions", body, h)
            return json.loads(b)["text"] if st == 200 else None, None if st == 200 else f"{st} {b[:200]!r}"
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(16) as ex:
            res = list(ex.map(one, clips))
        wall = time.perf_counter() - t0
        hyps = [r[0] or "" for r in res]
        errs = [r[1] for r in res if r[1]]
        w = jiwer.wer([norm_en(c["text"]) for c in clips], [norm_en(h) for h in hyps])
        results[m] = dict(clips=len(clips), errors=len(errs), first_error=errs[0] if errs else None, wer=round(w, 5),
                          audio_s=round(sum(c["dur"] for c in clips), 1),
                          rtfx=round(sum(c["dur"] for c in clips) / wall, 1), source=src)
        print(f"ASR {m}: WER {100 * w:.3f}% on {len(clips)} clips ({len(errs)} errors)", flush=True)
    return results


# ------------------------------------------------------------------ TTS
def norm_lang(s, lang):
    s = unicodedata.normalize("NFKC", s).lower()
    if lang is None:
        s = re.sub(r"[^\w\s]", " ", s)
        return " ".join(s.split())
    s = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in s)
    return "".join(s.split()) if lang in ("zh", "ja") else " ".join(s.split())


def prompt_set(model, kind="auto", voice=None):
    if kind == "auto":
        kind = "mtl" if "mtl" in model else "veena" if "veena" in model else "orpheus" if "orpheus" in model else "cbx"
    if kind == "mtl":
        return [(voice or "default", t, lang) for lang, t in MTL_PROMPTS]
    if kind == "veena":
        return [(voice or v, t, None) for v, t in VEENA_PROMPTS]
    if kind == "orpheus":
        return [(voice or "tara", t, None) for v, t in VEENA_PROMPTS if all(ord(ch) < 128 for ch in t)]
    return [(voice or "default", t, None) for t in CBX_TEXTS]


def eval_tts(a, models, outdir):
    import numpy as np
    import jiwer
    import torch
    from transformers import pipeline
    dev = a.whisper_device or ("cuda" if torch.cuda.is_available() else "cpu")
    asr = pipeline("automatic-speech-recognition", model=a.whisper, device=dev,
                   torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    results = {}
    for m in models:
        rows = []
        wavdir = os.path.join(outdir, f"tts-{m}")
        os.makedirs(wavdir, exist_ok=True)
        for i, (voice, text, lang) in enumerate(prompt_set(m, a.tts_prompts, a.voice)):
            body = {"model": m, "input": text, "voice": voice, "response_format": "wav", "seed": 1 + i}
            if lang:
                body["language"] = lang
            st, b = call("POST", "/v1/audio/speech", body)
            if st != 200:
                rows.append(dict(i=i, lang=lang, error=f"{st} {b[:200]!r}"))
                continue
            path = os.path.join(wavdir, f"{i:02d}_{lang or voice}.wav")
            open(path, "wb").write(b)
            with wave.open(io.BytesIO(b)) as w:
                sr, pcm = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            x = pcm.astype(np.float32) / 32768.0
            n16 = int(len(x) * 16000 / sr)
            x16 = np.interp(np.linspace(0, len(x) - 1, n16), np.arange(len(x)), x).astype(np.float32)
            kw = dict(generate_kwargs=dict(language=lang, task="transcribe")) if lang else {}
            hyp = asr({"raw": x16, "sampling_rate": 16000}, **kw)["text"]
            cer = jiwer.cer(norm_lang(text, lang), norm_lang(hyp, lang))
            rows.append(dict(i=i, lang=lang, voice=voice, audio_s=round(len(pcm) / sr, 2), cer=round(cer, 4), text=text, hyp=hyp.strip()))
        ok = [r for r in rows if "cer" in r]
        by_lang = {}
        for r in ok:
            by_lang.setdefault(r["lang"] or "default", []).append(r["cer"])
        res = dict(n=len(rows), errors=len(rows) - len(ok), cer_median=round(statistics.median(r["cer"] for r in ok), 4) if ok else None,
                   cer_mean=round(sum(r["cer"] for r in ok) / len(ok), 4) if ok else None,
                   cer_by_language={k: round(statistics.median(v), 4) for k, v in sorted(by_lang.items())},
                   wavs=wavdir, rows=rows)
        json.dump(rows, open(os.path.join(wavdir, "cer.json"), "w"), ensure_ascii=False, indent=1)
        results[m] = res
        print(f"TTS {m}: Whisper CER median {res['cer_median']} mean {res['cer_mean']} over {len(ok)} prompts "
              f"({res['errors']} errors); by language {res['cer_by_language']}", flush=True)
    return results


# ------------------------------------------------------------------ LLM
def eval_llm(a, model, max_len):
    checks = []

    def check(name, ok, detail=""):
        checks.append(dict(check=name, ok=bool(ok), detail=str(detail)[:300]))
        print(f"LLM {'PASS' if ok else 'FAIL'} {name}: {str(detail)[:160]}", flush=True)

    def chat(content, **kw):
        body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 64, "temperature": 0}
        body.update(kw)
        st, b = call("POST", "/v1/chat/completions", body)
        return st, json.loads(b)

    qa = [("What is the capital of France? Answer with one word.", "paris"),
          ("What is 12 times 12? Answer with the number only.", "144"),
          ("Which planet is known as the Red Planet? One word.", "mars"),
          ("Translate 'thank you' into Spanish. Answer with the translation only.", "gracias"),
          ("What color do you get by mixing blue and yellow? One word.", "green"),
          ("Who wrote 'Romeo and Juliet'? Answer with the author's name only.", "shakespeare")]
    for q, want in qa:
        st, j = chat(q)
        text = j["choices"][0]["message"]["content"] if st == 200 else j
        check(f"answer: {q[:40]}", st == 200 and want in str(text).lower(), text)
    st1, j1 = chat("Write one sentence about the ocean.", max_tokens=40)
    st2, j2 = chat("Write one sentence about the ocean.", max_tokens=40)
    check("greedy determinism", st1 == st2 == 200 and j1["choices"][0]["message"]["content"] == j2["choices"][0]["message"]["content"],
          j1["choices"][0]["message"]["content"] if st1 == 200 else j1)
    check("usage reported", st1 == 200 and j1["usage"]["prompt_tokens"] > 0 and j1["usage"]["completion_tokens"] > 0, j1.get("usage"))
    st, j = chat("Count from one to ten in words, separated by spaces.", stop=["five"])
    check("stop string", st == 200 and "five" not in j["choices"][0]["message"]["content"].lower()
          and j["choices"][0]["finish_reason"] == "stop", j["choices"][0] if st == 200 else j)
    st, j = chat("Say hello.", logprobs=True, top_logprobs=3, max_tokens=8)
    lp = (j.get("choices") or [{}])[0].get("logprobs") if st == 200 else None
    good = bool(lp and lp.get("content")) and all(
        t["logprob"] <= 1e-6 and len(t["top_logprobs"]) == 3 and
        all(x["logprob"] >= y["logprob"] for x, y in zip(t["top_logprobs"], t["top_logprobs"][1:])) for t in lp["content"])
    check("logprobs (top 3, sorted, <= 0)", good, lp["content"][0] if lp else j)
    st, b = call("POST", "/v1/completions", {"model": model, "prompt": "<bos>The capital of France is", "max_tokens": 6, "temperature": 0})
    text = json.loads(b)["choices"][0]["text"] if st == 200 else b
    check("raw completion with <bos>", st == 200 and "paris" in str(text).lower(), text)
    body = {"model": model, "messages": [{"role": "user", "content": "Name three primary colors."}], "max_tokens": 32,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **AUTH})
    t0, first, parts = time.perf_counter(), None, []
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                d = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
                if d:
                    first = first or time.perf_counter() - t0
                    parts.append(d)
    check("streamed chat", len(parts) > 1 and first is not None, f"TTFT {1000 * (first or 0):.0f} ms: {''.join(parts)}")
    st, b = call("POST", "/v1/completions", {"model": model, "prompt": "<bos>" + "hello " * ((max_len or 8192) + 512), "max_tokens": 4})
    err = json.loads(b).get("error", {}) if b[:1] == b"{" else {}
    check("context limit -> 400 context_length_exceeded", st == 400 and "context" in json.dumps(err), f"{st} {err}")
    st, b = call("POST", "/v1/chat/completions", {"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]})
    check("unknown model -> 404 with error envelope", st == 404 and b"error" in b, f"{st} {b[:120]!r}")
    return dict(model=model, passed=sum(c["ok"] for c in checks), total=len(checks), checks=checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["asr", "tts", "llm", "all"])
    ap.add_argument("--models", default=None, help="comma-separated (default: every served model of that kind)")
    ap.add_argument("--manifest", default=os.environ.get("ASR_MANIFEST"))
    ap.add_argument("--librispeech", default=None, help="LibriSpeech split dir (e.g. test-clean) instead of a manifest")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only")
    ap.add_argument("--language-hint", action="store_true", help="send language=en with each clip")
    ap.add_argument("--tts-prompts", default="auto", choices=["auto", "mtl", "veena", "orpheus", "cbx"])
    ap.add_argument("--voice", default=None, help="TTS voice for every prompt (default: the prompt set's)")
    ap.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--whisper-device", default=None)
    ap.add_argument("--out", default="results/eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    asr, tts, llm, max_len = models_by_kind()
    pick = lambda found: [m for m in a.models.split(",") if m in found] if a.models else found  # noqa: E731
    report = {"url": URL, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if a.what in ("asr", "all"):
        report["asr"] = eval_asr(a, pick(asr))
    if a.what in ("llm", "all") and llm:
        m = pick(llm)[0] if pick(llm) else llm[0]
        report["llm"] = eval_llm(a, m, max_len.get(m))
    if a.what in ("tts", "all"):
        report["tts"] = eval_tts(a, pick(tts), a.out)
    json.dump(report, open(os.path.join(a.out, "eval.json"), "w"), ensure_ascii=False, indent=1)
    md = [f"# plowrt quality evaluation\n\nServer {URL}, {report['date']}.\n"]
    if "asr" in report:
        md += ["| ASR model | clips | WER | errors |", "|---|---|---|---|"]
        for m, r in report["asr"].items():
            md.append(f"| {m} | {r['clips']} | {100 * r['wer']:.3f}% | {r['errors']} |")
        md.append("")
    if "tts" in report:
        md += ["| TTS model | prompts | Whisper CER median | mean | by language | errors |", "|---|---|---|---|---|---|"]
        for m, r in report["tts"].items():
            langs = ", ".join("%s %s" % kv for kv in r["cer_by_language"].items())
            md.append(f"| {m} | {r['n']} | {r['cer_median']} | {r['cer_mean']} | {langs} | {r['errors']} |")
        md.append("")
    if "llm" in report:
        r = report["llm"]
        md += [f"LLM {r['model']}: {r['passed']}/{r['total']} checks passed.\n", "| check | result | detail |", "|---|---|---|"]
        md += [f"| {c['check']} | {'pass' if c['ok'] else 'FAIL'} | {c['detail'][:100].replace('|', '/')} |" for c in r["checks"]] + [""]
    open(os.path.join(a.out, "eval.md"), "w").write("\n".join(md))
    print("\n".join(md))
    bad = any(r["errors"] for r in report.get("asr", {}).values()) or any(r["errors"] for r in report.get("tts", {}).values()) \
        or ("llm" in report and report["llm"]["passed"] < report["llm"]["total"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
