#!/usr/bin/env python3
"""Concurrent /v1/audio/transcriptions client: WER, latency p50/p90 and RTFx per concurrency.

served_bench.py --url http://127.0.0.1:8000 --model qwen3-asr --manifest manifest.json --conc 1,4,16
The manifest is a JSON list of {path, text, dur}. Every level sends each clip once (N workers
pull from a shared queue); RTFx = total audio seconds / wall seconds.

--rate R1,R2,...: open loop instead. Requests arrive as a Poisson process at R per second for
--duration s (clips drawn with replacement, --seed); those started in [--warmup, --duration -
--cooldown] are measured. Goodput = measured requests per second that succeed within --slo-ms.
"""
import argparse, json, os, queue, random, sys, threading, time

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


def open_loop(args, clips, rate):
    rng = random.Random(args.seed)
    t, plan = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= args.duration:
            break
        plan.append((t, rng.choice(clips)))
    out = [None] * len(plan)

    def one(i, c):
        out[i] = post(requests.Session(), args.url, args.model, c["path"])

    t0 = time.perf_counter()
    ts = []
    for i, (at, c) in enumerate(plan):
        time.sleep(max(0.0, t0 + at - time.perf_counter()))
        th = threading.Thread(target=one, args=(i, c))
        th.start()
        ts.append(th)
    [th.join() for th in ts]
    lo, hi = args.warmup, args.duration - args.cooldown
    m = [(c, o) for (at, c), o in zip(plan, out) if lo <= at < hi]
    span = hi - lo
    lat = sorted(o[0] for _, o in m)
    p = lambda q: 1e3 * lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None
    ok = [(c, o) for c, o in m if o[2] is None]
    good = [1 for _, o in ok if 1e3 * o[0] <= args.slo_ms]
    errs = [o[2] for _, o in m if o[2]]
    return dict(rate=rate, measured=len(m), errors=len(errs), first_error=errs[0] if errs else None,
                slo_ms=args.slo_ms, goodput_rps=round(len(good) / span, 3),
                slo_attainment=round(len(good) / max(1, len(m)), 4),
                wer=round(wer([c["text"] for c, _ in ok], [o[1] for _, o in ok]), 5) if ok else None,
                lat_p50_ms=p(.5) and round(p(.5), 1), lat_p90_ms=p(.9) and round(p(.9), 1),
                lat_p99_ms=p(.99) and round(p(.99), 1),
                audio_s_per_s=round(sum(c["dur"] for c, _ in ok) / span, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--conc", default="1,4,16")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rate", help="open loop: Poisson arrival rates per second, comma-separated")
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--warmup", type=float, default=20)
    ap.add_argument("--cooldown", type=float, default=10)
    ap.add_argument("--slo-ms", type=float, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    clips = json.load(open(args.manifest))
    s = requests.Session()
    for c in clips[:4]:
        post(s, args.url, args.model, c["path"])
    failed = False
    if args.rate:
        for rate in map(float, args.rate.split(",")):
            r = open_loop(args, clips, rate)
            r["tag"] = args.tag
            print(json.dumps(r), flush=True)
            failed |= r["errors"] > 0
        sys.exit(1 if failed else 0)
    for n in map(int, args.conc.split(",")):
        r = level(args, clips, n)
        r["tag"] = args.tag
        print(json.dumps(r), flush=True)
        failed |= r["errors"] > 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
