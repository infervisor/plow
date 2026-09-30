#!/usr/bin/env python3
"""Concurrent /v1/audio/transcriptions client: WER, latency p50/p90 and RTFx per concurrency.

served_bench.py --url http://127.0.0.1:8000 --model qwen3-asr --manifest manifest.json --conc 1,4,16
The manifest is a JSON list of {path, text, dur}. Every level sends each clip once (N workers
pull from a shared queue); RTFx = total audio seconds / wall seconds.
"""
import argparse, json, os, queue, sys, threading, time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_bench import wer  # noqa: E402


def post(session, url, model, path):
    with open(path, "rb") as f:
        data = f.read()
    t0 = time.perf_counter()
    r = session.post(url + "/v1/audio/transcriptions", data={"model": model},
                     files={"file": (os.path.basename(path), data, "audio/wav")}, timeout=300)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return dt, None, f"{r.status_code} {r.text[:200]}"
    return dt, r.json()["text"], None


def level(args, clips, n):
    q = queue.Queue()
    for i, c in enumerate(clips):
        q.put((i, c))
    out = [None] * len(clips)

    def worker():
        s = requests.Session()
        while True:
            try:
                i, c = q.get_nowait()
            except queue.Empty:
                return
            out[i] = post(s, args.url, args.model, c["path"])

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.perf_counter() - t0
    lat = sorted(o[0] for o in out)
    p = lambda q: 1e3 * lat[min(len(lat) - 1, int(q * len(lat)))]
    errs = [o[2] for o in out if o[2]]
    audio = sum(c["dur"] for c in clips)
    return dict(conc=n, clips=len(clips), errors=len(errs), first_error=errs[0] if errs else None,
                wer=round(wer([c["text"] for c in clips], [o[1] or "" for o in out]), 5),
                lat_p50_ms=round(p(.5), 1), lat_p90_ms=round(p(.9), 1),
                wall_s=round(wall, 3), rtfx=round(audio / wall, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--conc", default="1,4,16")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    clips = json.load(open(args.manifest))
    s = requests.Session()
    for c in clips[:4]:
        post(s, args.url, args.model, c["path"])
    failed = False
    for n in map(int, args.conc.split(",")):
        r = level(args, clips, n)
        r["tag"] = args.tag
        print(json.dumps(r), flush=True)
        failed |= r["errors"] > 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
