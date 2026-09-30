#!/usr/bin/env python3
"""Model-switch latency on one co-serving plowrt: sequential requests, first each model alone
(solo), then round-robin across the models (alt). The alt - solo median is the switch cost.

switch_bench.py --url U --asr qwen3-asr --clip a.wav --tts veena:kavya --tts chatterbox:default
"""
import argparse, json, statistics, time

import requests


def asr(s, url, model, clip):
    with open(clip, "rb") as f:
        r = s.post(url + "/v1/audio/transcriptions", data={"model": model},
                   files={"file": ("a.wav", f.read(), "audio/wav")}, timeout=300)
    r.raise_for_status()


def tts(s, url, model, voice, text):
    r = s.post(url + "/v1/audio/speech", json=dict(model=model, input=text, voice=voice,
                                                  response_format="pcm", seed=7), timeout=300)
    r.raise_for_status()
    if not r.content:
        raise RuntimeError(f"{model}: empty audio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--asr")
    ap.add_argument("--clip")
    ap.add_argument("--tts", action="append", default=[], help="MODEL:VOICE")
    ap.add_argument("--text", default="Hello, how are you doing today?")
    ap.add_argument("--rounds", type=int, default=12)
    a = ap.parse_args()
    s = requests.Session()
    calls = []
    if a.asr:
        calls.append((a.asr, lambda: asr(s, a.url, a.asr, a.clip)))
    for spec in a.tts:
        model, voice = spec.split(":", 1)
        calls.append((model, lambda m=model, v=voice: tts(s, a.url, m, v, a.text)))
    for _, f in calls:
        f()
    lat = {m: {"solo": [], "alt": []} for m, _ in calls}
    for m, f in calls:
        for _ in range(a.rounds):
            t = time.perf_counter(); f(); lat[m]["solo"].append(time.perf_counter() - t)
    for _ in range(a.rounds):
        for m, f in calls:
            t = time.perf_counter(); f(); lat[m]["alt"].append(time.perf_counter() - t)
    for m, d in lat.items():
        solo, alt = statistics.median(d["solo"]) * 1e3, statistics.median(d["alt"]) * 1e3
        print(json.dumps(dict(model=m, solo_p50_ms=round(solo, 1), alt_p50_ms=round(alt, 1),
                              switch_ms=round(alt - solo, 1), alt_max_ms=round(max(d["alt"]) * 1e3, 1))))


if __name__ == "__main__":
    main()
