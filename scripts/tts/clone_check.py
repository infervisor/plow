#!/usr/bin/env python3
"""Voice cloning through `POST /v1/audio/speech` (`ref_audio`, + `ref_text` for models that take a
transcript; without --ref-text the model must refuse one): whole and streamed
replies for each prompt text, written as OUT/clone_{whole,stream}_NN.wav with OUT/texts.json for
scripts/tts/asr_check.py, plus the request checks a cloning model must answer with 400.

  python scripts/tts/clone_check.py --url URL --model ID --ref REF.wav [--ref-text "..."] --out OUT [--voice S0] [--conc 8]
"""
import argparse
import base64
import json
import os
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from chatterbox_ref import PROMPTS as TEXTS


def post(url, body, stream):
    req = urllib.request.Request(f"{url}/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttfa, chunks = None, []
    try:
        r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        raise
    with r:
        while True:
            b = r.read1(65536) if stream else r.read()
            if not b:
                break
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(b)
            if not stream:
                break
    return ttfa, time.perf_counter() - t0, b"".join(chunks)


def status(url, body):
    try:
        post(url, body, False)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def wav_bytes(pcm, sr):
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm)) + pcm)


def wav(path, pcm, sr):
    with open(path, "wb") as f:
        f.write(wav_bytes(pcm, sr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text")
    ap.add_argument("--voice", default="S0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--sr", type=int, default=24000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref = base64.b64encode(open(a.ref, "rb").read()).decode()
    base = dict(model=a.model, voice=a.voice, ref_audio=ref, response_format="pcm")
    if a.ref_text:
        base["ref_text"] = a.ref_text
    texts, rows = {}, []

    def run(i):
        text = TEXTS[i % len(TEXTS)]
        stream = i % 2 == 1
        ttfa, total, pcm = post(a.url, dict(base, input=text, stream=stream, seed=i), stream)
        name = f"clone_{'stream' if stream else 'whole'}_{i:02d}"
        wav(f"{a.out}/{name}.wav", pcm, a.sr)
        return name, text, stream, ttfa, total, len(pcm) / 2 / a.sr

    with ThreadPoolExecutor(a.conc) as ex:
        for name, text, stream, ttfa, total, audio in ex.map(run, range(a.n)):
            texts[name + ".wav"] = text
            rows.append(dict(name=name, stream=stream, ttfa_ms=1000 * ttfa, latency_s=total, audio_s=audio))
    json.dump(texts, open(f"{a.out}/texts.json", "w"), indent=1)
    checks = {
        ("ref_audio without ref_text" if a.ref_text else "ref_text on a ref_audio-only model"): status(
            a.url, dict(model=a.model, voice=a.voice, input="Hi there.", ref_audio=ref,
                        **({} if a.ref_text else {"ref_text": "Hi there."}))),
        "ref_audio not base64": status(a.url, dict(base, input="Hi there.", ref_audio="@@not-audio@@")),
        "ref_audio too long": status(a.url, dict(base, input="Hi there.", ref_audio=base64.b64encode(
            wav_bytes(b"\0\0" * (a.sr * 30), a.sr)).decode())),
    }
    bad = {k: v for k, v in checks.items() if v != 400}
    stream_ttfa = [r["ttfa_ms"] for r in rows if r["stream"]]
    print(json.dumps(dict(
        n=len(rows), conc=a.conc, failed_checks=bad,
        med_stream_ttfa_ms=statistics.median(stream_ttfa) if stream_ttfa else None,
        med_rtf=statistics.median(r["latency_s"] / max(r["audio_s"], 1e-9) for r in rows),
        audio_s=sum(r["audio_s"] for r in rows), empty=sum(r["audio_s"] == 0 for r in rows))))
    sys.exit(1 if bad or any(r["audio_s"] == 0 for r in rows) else 0)


if __name__ == "__main__":
    main()
