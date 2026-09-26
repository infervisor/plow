#!/usr/bin/env python3
"""Client for any OpenAI-shaped `POST /v1/audio/speech` server (plowrt serve, or the vLLM
baseline in scripts/tts/vllm_speech_server.py): same prompts, same client, two base URLs.

  python scripts/tts/tts_bench.py --url http://127.0.0.1:PORT --model M --conc 8 --n 32 \
      --stream --out DIR [--wav]

Per request: ttfa_s (first audio byte, streaming) or latency_s, audio_s, rtf. Summary: medians,
p90 TTFA, aggregate audio seconds per wall second. Writes DIR/<tag>.json and optional wavs
plus texts.json for scripts/tts/asr_check.py.
"""
import argparse, concurrent.futures as cf, json, os, statistics, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from veena_ref import PROMPTS

SR = 24000


def one(url, model, voice, text, stream, seed, extra):
    body = dict(model=model, input=text, voice=voice, response_format="pcm", stream=stream, seed=seed, **extra)
    req = urllib.request.Request(f"{url}/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttfa, chunks = None, []
    with urllib.request.urlopen(req, timeout=900) as r:
        while True:
            b = r.read1(65536) if stream else r.read()
            if not b:
                break
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(b)
            if not stream:
                break
    total = time.perf_counter() - t0
    pcm = b"".join(chunks)
    return dict(ttfa_s=ttfa, latency_s=total, audio_s=len(pcm) / 2 / SR, pcm=pcm)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wav", action="store_true")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"{'stream' if args.stream else 'full'}_c{args.conc}"
    extra = dict(temperature=0.0) if args.greedy else {}
    jobs = [(PROMPTS[i % len(PROMPTS)], i) for i in range(args.n)]
    for (spk, text), i in jobs[: args.warmup]:
        one(args.url, args.model, spk, text, args.stream, 1000 + i, extra)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(args.conc) as ex:
        res = list(ex.map(lambda j: one(args.url, args.model, j[0][0], j[0][1], args.stream, j[1] + 1, extra), jobs))
    wall = time.perf_counter() - t0
    texts = {}
    rows = []
    for ((spk, text), i), r in zip(jobs, res):
        if args.wav and r["audio_s"] > 0:
            import numpy as np, soundfile as sf
            name = f"{tag}_{i:02d}_{spk}.wav"
            sf.write(f"{args.out}/{name}", np.frombuffer(r["pcm"], dtype="<i2"), SR)
            texts[name] = text
        rows.append(dict(i=i, speaker=spk, ttfa_s=r["ttfa_s"], latency_s=r["latency_s"], audio_s=r["audio_s"],
                         rtf=r["latency_s"] / r["audio_s"] if r["audio_s"] else None))
    ok = [r for r in rows if r["audio_s"] > 0]
    summ = dict(tag=tag, url=args.url, conc=args.conc, n=len(rows), failed=len(rows) - len(ok), stream=args.stream,
                audio_s=sum(r["audio_s"] for r in ok), wall_s=wall,
                audio_s_per_s=sum(r["audio_s"] for r in ok) / wall,
                med_latency_s=statistics.median(r["latency_s"] for r in ok),
                med_rtf=statistics.median(r["rtf"] for r in ok))
    if args.stream:
        summ.update(med_ttfa_ms=1e3 * statistics.median(r["ttfa_s"] for r in ok),
                    p90_ttfa_ms=1e3 * pct([r["ttfa_s"] for r in ok], 0.9))
    json.dump(dict(summary=summ, rows=rows), open(f"{args.out}/{tag}.json", "w"), indent=1)
    if texts:
        old = json.load(open(f"{args.out}/texts.json")) if os.path.exists(f"{args.out}/texts.json") else {}
        old.update(texts)
        json.dump(old, open(f"{args.out}/texts.json", "w"), ensure_ascii=False, indent=0)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
