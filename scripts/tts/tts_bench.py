#!/usr/bin/env python3
"""Client for any OpenAI-shaped `POST /v1/audio/speech` server (plowrt serve, or the vLLM
baseline in scripts/tts/vllm_speech_server.py): same prompts, same client, two base URLs.

  python scripts/tts/tts_bench.py --url http://127.0.0.1:PORT --model M --conc 8 --n 32 \
      --stream --out DIR [--wav]

Per request: ttfa_s (first audio byte, streaming) or latency_s, audio_s, rtf, and for streams the
playback underruns after the first chunk (gaps, gap_s). Summary: medians, p90 TTFA, gap totals,
aggregate audio seconds per wall second. Writes DIR/<tag>.json and optional wavs
plus texts.json for scripts/tts/asr_check.py.
"""
import argparse, concurrent.futures as cf, http.client, json, os, statistics, sys, threading, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from veena_ref import PROMPTS as VEENA_PROMPTS
from chatterbox_ref import PROMPTS as CBX_TEXTS
from mtl_prompts import PROMPTS as MTL_PROMPTS

# Multilingual prompt set: each text carries its request `language`.
LANG_OF = {t: l for l, t in MTL_PROMPTS}

SR = 24000


SESSION = None
PROMPT_SET = None
# A chunk this late past the end of the audio already received is an underrun (clock jitter).
GAP_TOLERANCE_S = 0.001
# A gap longer than this is a stall (the playback buffer stayed dry for more than 100 ms).
STALL_S = 0.100


def underruns(arrivals, sr):
    """Playback underruns of a stream: `arrivals` = [(seconds, pcm16 bytes)] in order. Playback
    starts at the first chunk; a chunk arriving after the audio received so far has played out is
    a gap, and playback resumes at its arrival. Returns (gaps, total gap seconds, stalls): stalls
    are the gaps longer than STALL_S."""
    gaps, gap_s, stalls, end = 0, 0.0, 0, None
    for t, nbytes in arrivals:
        dur = nbytes / 2 / sr
        if end is None:
            end = t + dur
        elif t > end + GAP_TOLERANCE_S:
            stalls += t - end > STALL_S
            gaps, gap_s, end = gaps + 1, gap_s + (t - end), t + dur
        else:
            end += dur
    return gaps, gap_s, stalls


def one(url, model, voice, text, stream, seed, extra):
    body = dict(model=model, input=text, voice=voice, response_format="pcm", stream=stream, seed=seed, **extra)
    if PROMPT_SET == "chatterbox-mtl":
        body["language"] = LANG_OF[text]
    headers = {"Content-Type": "application/json"}
    if SESSION:
        # One X-Session-Id per client thread: its requests run one after another in the session.
        headers["X-Session-Id"] = f"{SESSION}-{threading.get_ident()}"
    req = urllib.request.Request(f"{url}/v1/audio/speech", data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    ttfa, chunks, arrivals = None, [], []
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            while True:
                b = r.read1(65536) if stream else r.read()
                if not b:
                    break
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                chunks.append(b)
                arrivals.append((time.perf_counter(), len(b)))
                if not stream:
                    break
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        # Overload (429 / queue TTL) or a stream cut mid-body: a failed request, not a failed run.
        print(f"request failed: {e}", file=sys.stderr)
        chunks = []
    total = time.perf_counter() - t0
    pcm = b"".join(chunks)
    gaps, gap_s, stalls = underruns(arrivals if chunks else [], SR)
    return dict(ttfa_s=ttfa, latency_s=total, audio_s=len(pcm) / 2 / SR, pcm=pcm, gaps=gaps, gap_s=gap_s, stalls=stalls)


def one_ws(url, model, voice, text, seed, extra):
    from websockets.sync.client import connect
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech"
    body = dict(model=model, input=text, voice=voice, response_format="pcm", stream=True, seed=seed, **extra)
    if PROMPT_SET == "chatterbox-mtl":
        body["language"] = LANG_OF[text]
    headers = {}
    if SESSION:
        headers["X-Session-Id"] = f"{SESSION}-{threading.get_ident()}"
    t0 = time.perf_counter()
    ttfa, chunks, arrivals = None, [], []
    try:
        with connect(ws_url, additional_headers=headers, open_timeout=900, close_timeout=30, ping_interval=None, ping_timeout=None) as ws:
            ws.send(json.dumps(body))
            for message in ws:
                if isinstance(message, bytes):
                    if ttfa is None:
                        ttfa = time.perf_counter() - t0
                    chunks.append(message)
                    arrivals.append((time.perf_counter(), len(message)))
                elif isinstance(message, str):
                    data = json.loads(message)
                    if "error" in data:
                        print(f"ws error: {data['error']}", file=sys.stderr)
                        break
                    if data.get("done"):
                        break
    except Exception as e:
        print(f"ws request failed: {e}", file=sys.stderr)
        chunks = []
    total = time.perf_counter() - t0
    pcm = b"".join(chunks)
    gaps, gap_s, stalls = underruns(arrivals if chunks else [], SR)
    return dict(ttfa_s=ttfa, latency_s=total, audio_s=len(pcm) / 2 / SR, pcm=pcm, gaps=gaps, gap_s=gap_s, stalls=stalls)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def card_sample_rate(url, model):
    """The model card's `x_plow_speech.sample_rate` (None when the server publishes none)."""
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
            cards = json.load(r)["data"]
    except Exception:
        return None
    for c in cards:
        if c.get("id") == model:
            return (c.get("x_plow_speech") or {}).get("sample_rate")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--ws", action="store_true", help="use WebSocket streaming")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wav", action="store_true")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--prompt-set", choices=["veena", "chatterbox", "chatterbox-mtl"], default="veena")
    ap.add_argument("--voice", default=None, help="override every prompt's voice")
    ap.add_argument("--ref-audio", default=None, help="clone every request's voice from this wav (a comma list: cycled)")
    ap.add_argument("--ref-text", default=None, help="transcripts of --ref-audio ('|' list, same order; Fish / Breeze)")
    ap.add_argument("--clone-list", default=None,
                    help="Seed-TTS-eval meta.lst (id|ref_text|ref_wav|text, wav paths relative to it): one request per line, "
                         "each cloned from its own reference; wavs are named <id>.wav (--n caps the lines)")
    ap.add_argument("--seed-offset", type=int, default=0, help="added to every request's seed (repeat runs)")
    ap.add_argument("--session", default=None, help="X-Session-Id prefix (one session per client thread)")
    args = ap.parse_args()
    global SESSION, PROMPT_SET, SR
    SR = card_sample_rate(args.url, args.model) or SR
    SESSION, PROMPT_SET = args.session, args.prompt_set
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"{'ws' if args.ws else ('stream' if args.stream else 'full')}_c{args.conc}"
    extra = dict(temperature=0.0) if args.greedy else {}
    if args.repetition_penalty is not None:
        extra["repetition_penalty"] = args.repetition_penalty
    refs = []
    if args.ref_audio:
        import base64
        refs = [{"ref_audio": base64.b64encode(open(f, "rb").read()).decode()} for f in args.ref_audio.split(",")]
        if args.ref_text:
            for r, t in zip(refs, args.ref_text.split("|"), strict=True):
                r["ref_text"] = t
    per = lambda i: dict(extra, **refs[i % len(refs)]) if refs else extra
    names = None
    if args.clone_list:
        import base64
        base = os.path.dirname(os.path.abspath(args.clone_list))
        lines = [l.rstrip("\n").split("|") for l in open(args.clone_list) if l.strip()][: args.n]
        names = [f"{i}.wav" for i, *_ in lines]
        clone = [dict(ref_audio=base64.b64encode(open(os.path.join(base, w), "rb").read()).decode(), ref_text=rt)
                 for _, rt, w, _ in lines]
        per = lambda i: dict(extra, **clone[i])
        args.n = len(lines)
    prompts = VEENA_PROMPTS if args.prompt_set == "veena" else [("default", t) for t in CBX_TEXTS]
    if args.prompt_set == "chatterbox-mtl":
        prompts = [("default", t) for _, t in MTL_PROMPTS]
    if names is not None:
        prompts = [("default", l[3]) for l in lines]
    if args.voice:
        prompts = [(args.voice, t) for _, t in prompts]
    jobs = [(prompts[i % len(prompts)], i) for i in range(args.n)]
    call_one = (lambda u, m, spk, txt, st, sd, ex: one_ws(u, m, spk, txt, sd, ex)) if args.ws else (lambda u, m, spk, txt, st, sd, ex: one(u, m, spk, txt, st, sd, ex))
    for (spk, text), i in jobs[: args.warmup]:
        call_one(args.url, args.model, spk, text, args.stream or args.ws, 1000 + i, per(i))
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(args.conc) as ex:
        res = list(ex.map(lambda j: call_one(args.url, args.model, j[0][0], j[0][1], args.stream or args.ws, j[1] + 1 + args.seed_offset, per(j[1])), jobs))
    wall = time.perf_counter() - t0
    texts = {}
    rows = []
    for ((spk, text), i), r in zip(jobs, res):
        if args.wav and r["audio_s"] > 0:
            import numpy as np, soundfile as sf
            name = names[i] if names is not None else f"{tag}_{i:02d}_{spk}.wav"
            sf.write(f"{args.out}/{name}", np.frombuffer(r["pcm"], dtype="<i2"), SR)
            texts[name] = dict(text=text, language=LANG_OF[text]) if args.prompt_set == "chatterbox-mtl" else text
        rows.append(dict(i=i, speaker=spk, ttfa_s=r["ttfa_s"], latency_s=r["latency_s"], audio_s=r["audio_s"],
                         gaps=r["gaps"], gap_s=r["gap_s"], stalls=r.get("stalls", 0),
                         rtf=r["latency_s"] / r["audio_s"] if r["audio_s"] else None))
    ok = [r for r in rows if r["audio_s"] > 0]
    if not ok:
        summ = dict(tag=tag, url=args.url, conc=args.conc, n=len(rows), failed=len(rows), error="all requests failed")
        print(json.dumps(summ), file=sys.stderr)
        sys.exit(1)
    summ = dict(tag=tag, url=args.url, conc=args.conc, n=len(rows), failed=len(rows) - len(ok), stream=args.stream,
                audio_s=sum(r["audio_s"] for r in ok), wall_s=wall,
                audio_s_per_s=sum(r["audio_s"] for r in ok) / wall,
                med_latency_s=statistics.median(r["latency_s"] for r in ok),
                med_rtf=statistics.median(r["rtf"] for r in ok))
    if args.stream or args.ws:
        summ.update(med_ttfa_ms=1e3 * statistics.median(r["ttfa_s"] for r in ok),
                    p90_ttfa_ms=1e3 * pct([r["ttfa_s"] for r in ok], 0.9),
                    p99_ttfa_ms=1e3 * pct([r["ttfa_s"] for r in ok], 0.99),
                    gaps=sum(r["gaps"] for r in ok), gap_streams=sum(r["gaps"] > 0 for r in ok),
                    gap_s=sum(r["gap_s"] for r in ok), stalls=sum(r["stalls"] for r in ok),
                    p90_rtf=pct([r["rtf"] for r in ok], 0.9))
        if len(ok) > args.conc:
            steady_ok = [r for r in rows[args.conc:] if r["audio_s"] > 0]
            if steady_ok:
                summ.update(
                    steady_med_ttfa_ms=1e3 * statistics.median(r["ttfa_s"] for r in steady_ok),
                    steady_p90_ttfa_ms=1e3 * pct([r["ttfa_s"] for r in steady_ok], 0.9),
                    steady_p99_ttfa_ms=1e3 * pct([r["ttfa_s"] for r in steady_ok], 0.99),
                )
    json.dump(dict(summary=summ, rows=rows), open(f"{args.out}/{tag}.json", "w"), indent=1)
    if texts:
        old = json.load(open(f"{args.out}/texts.json")) if os.path.exists(f"{args.out}/texts.json") else {}
        old.update(texts)
        json.dump(old, open(f"{args.out}/texts.json", "w"), ensure_ascii=False, indent=0)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
