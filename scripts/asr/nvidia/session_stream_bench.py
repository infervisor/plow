#!/usr/bin/env python3
"""Streamed transcription over `X-Session-Id` appends: each clip goes up as `--chunk-s` WAV chunks
(`append=true`, a partial transcript per chunk) and the last chunk with `final=true`; WER over the
finals. `--mode whole` instead re-sends everything so far as a one-shot request per chunk: what a
client without sessions does, every partial re-transcribing the whole buffer.

session_stream_bench.py --url http://127.0.0.1:PORT --model qwen3-asr --manifest m.json [--long 29.5]

Engine time is the serving mux's tick time (`plowrt_tick_duration_seconds_sum`, /metrics) over
the run; the encoder's shows in the server log (`asr: partial encode` / `ASR completed`).
"""
import argparse, io, json, os, re, sys, time, uuid, wave

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_bench import wer  # noqa: E402

SR = 16000


def read_pcm(path):
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


def wav_bytes(pcm):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return b.getvalue()


def engine_s(s, url):
    text = s.get(url + "/metrics", timeout=30).text
    return sum(float(m) for m in re.findall(r'^plowrt_tick_duration_seconds_sum\{[^}]*\} ([0-9.e+-]+)$', text, re.M))


def post(s, url, model, pcm, fields, sid):
    headers = {"X-Session-Id": sid} if sid else {}
    t0 = time.perf_counter()
    r = s.post(url + "/v1/audio/transcriptions", data=dict(model=model, **fields), headers=headers,
               files={"file": ("a.wav", wav_bytes(pcm), "audio/wav")}, timeout=300)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    return dt, r.json()["text"]


def stream_clip(s, args, pcm):
    step = int(args.chunk_s * SR) * 2
    ends = list(range(step, len(pcm), step)) + [len(pcm)]
    sid = f"bench-{uuid.uuid4()}"
    partial, text, start = [], "", 0
    for k, end in enumerate(ends):
        last = k == len(ends) - 1
        if args.mode == "session":
            dt, text = post(s, args.url, args.model, pcm[start:end], {"final" if last else "append": "true"}, sid)
        else:
            dt, text = post(s, args.url, args.model, pcm[:end], {}, None)
        (partial if not last else []).append(dt)
        if last:
            final = dt
        start = end
    return partial, final, text


def pct(xs, q):
    xs = sorted(xs)
    return round(1e3 * xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", choices=["session", "whole"], default="session")
    ap.add_argument("--chunk-s", type=float, default=1.0)
    ap.add_argument("--long", type=float, default=0, help="one clip of this many seconds (clips concatenated)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--hyps", default=None, help="write the final transcripts here (JSON list)")
    args = ap.parse_args()
    clips = json.load(open(args.manifest))
    s = requests.Session()
    if args.long:
        pcm, text = b"", []
        for c in clips:
            p = read_pcm(c["path"])
            if len(pcm) + len(p) > args.long * SR * 2:
                continue
            pcm += p
            text.append(c["text"])
        work = [(pcm, " ".join(text))]
    else:
        work = [(read_pcm(c["path"]), c["text"]) for c in clips]
    stream_clip(s, args, work[0][0])  # warm
    e0, t0 = engine_s(s, args.url), time.perf_counter()
    partials, finals, hyps, per_clip = [], [], [], []
    for pcm, _ in work:
        c0 = time.perf_counter()
        p, f, h = stream_clip(s, args, pcm)
        per_clip.append(time.perf_counter() - c0)
        partials += p
        finals.append(f)
        hyps.append(h)
        if args.long:
            print(json.dumps(dict(partial_ms=[round(1e3 * x, 1) for x in p], final_ms=round(1e3 * f, 1))), flush=True)
    wall = time.perf_counter() - t0
    engine = engine_s(s, args.url) - e0
    out = dict(tag=args.tag, mode=args.mode, clips=len(work), audio_s=round(sum(len(p) for p, _ in work) / 2 / SR, 2),
               wer=round(wer([t for _, t in work], hyps), 5), partials=len(partials),
               partial_p50_ms=pct(partials, .5), partial_p90_ms=pct(partials, .9), partial_max_ms=pct(partials, 1.0),
               final_p50_ms=pct(finals, .5), final_p90_ms=pct(finals, .9),
               wall_s=round(wall, 3), engine_ms=round(1e3 * engine, 1))
    print(json.dumps(out), flush=True)
    if args.hyps:
        json.dump(hyps, open(args.hyps, "w"), indent=0)


if __name__ == "__main__":
    main()
