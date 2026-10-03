#!/usr/bin/env python3
"""Voice-agent call simulator against one `plowrt serve`: N concurrent calls, each an
X-Session-Id session that loops turns of
  user speech  -> streaming ASR (1 s WAV chunks at real-time pace, `append` + `offset`, then `final`)
  agent turn   -> streaming chat completion with the call's history (optional: --llm-model)
  agent speech -> streaming TTS of the reply, played back on a simulated real-time clock.

  call_sim.py --url http://127.0.0.1:8080 --calls 200 --turns 3 --asr-model qwen3-asr \
      --llm-model gemma-4-e4b --tts-model chatterbox-mtl --voice default \
      --manifest audio/manifest.json --out results/calls200.json

SLOs (per turn, p95 over all turns): ASR final latency (last chunk sent -> final transcript),
LLM TTFT, TTS time to first audio, and playback underrun (audio not there when the player needs
it). Without --llm-model the agent replies with --reply text; without --asr-model the user speaks
for --user-s seconds without transcription (with neither: TTS-only load).

Each turn sends `X-Turn-Id: <call>.<turn>` and a W3C `traceparent` (one trace per turn), and records
the server's `Server-Timing` per stage (`srv_asr` / `srv_llm` / `srv_tts`: queue, wait-turn,
device, first, total, slack ms; the LLM's from the stream's final `: server-timing` comment).
"""
import argparse, asyncio, io, json, random, statistics, time, uuid

import aiohttp
import numpy as np
import soundfile as sf

ASR_SR = 16000
TTS_SR = 24000


def wav_bytes(samples):
    buf = io.BytesIO()
    sf.write(buf, samples, ASR_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else xs[min(len(xs) - 1, int(q * len(xs)))]


def server_timing(value):
    """`queue;dur=1.2, device;dur=3` -> {"queue": 1.2, "device": 3.0} (ms)."""
    out = {}
    for metric in (value or "").split(","):
        name, _, params = metric.strip().partition(";")
        for p in params.split(";"):
            k, _, v = p.strip().partition("=")
            if name and k == "dur":
                try:
                    out[name] = float(v)
                except ValueError:
                    pass
    return out


def turn_headers(sid, turn):
    """One turn's routing headers; a fresh span per request."""
    return {"X-Session-Id": sid, "X-Request-Id": uuid.uuid4().hex, "X-Turn-Id": turn["id"],
            "traceparent": f"00-{turn['trace']}-{uuid.uuid4().hex[:16]}-01"}


async def asr_turn(s, a, sid, clip, turn):
    """Real-time paced chunks; returns (transcript, final latency s, partial latencies)."""
    chunk = int(a.chunk_s * ASR_SR)
    sent, partial_lat, t_start = 0, [], time.perf_counter()
    for i in range(0, len(clip), chunk):
        piece = clip[i:i + chunk]
        last = i + chunk >= len(clip)
        form = aiohttp.FormData()
        form.add_field("model", a.asr_model)
        form.add_field("offset", str(sent))
        form.add_field("final" if last else "append", "true")
        form.add_field("file", wav_bytes(piece), filename="c.wav", content_type="audio/wav")
        # Real time: this chunk's audio ends at t_start + (i + len) / SR.
        due = t_start + (i + len(piece)) / ASR_SR
        await asyncio.sleep(max(0.0, due - time.perf_counter()))
        t0 = time.perf_counter()
        headers = turn_headers(sid, turn) if last else {"X-Session-Id": sid, "X-Request-Id": uuid.uuid4().hex}
        async with s.post(a.url + "/v1/audio/transcriptions", data=form, headers=headers) as r:
            body = await r.text()
            if r.status != 200:
                raise RuntimeError(f"asr {r.status}: {body[:200]}")
            timing = server_timing(r.headers.get("Server-Timing"))
        dt = time.perf_counter() - t0
        sent += len(piece)
        if last:
            return json.loads(body)["text"], dt, partial_lat, timing
        partial_lat.append(dt)


async def llm_turn(s, a, sid, history, turn):
    body = dict(model=a.llm_model, messages=history, max_tokens=a.max_tokens, temperature=0, stream=True)
    if a.logprobs:
        body.update(logprobs=True, top_logprobs=a.logprobs)
    t0, ttft, text = time.perf_counter(), None, []
    async with s.post(a.url + "/v1/chat/completions", json=body, headers=turn_headers(sid, turn)) as r:
        if r.status != 200:
            raise RuntimeError(f"llm {r.status}: {(await r.text())[:200]}")
        timing = server_timing(r.headers.get("Server-Timing"))
        async for line in r.content:
            line = line.decode().strip()
            if line.startswith(": server-timing "):
                timing = server_timing(line[len(": server-timing "):])
                continue
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
            if delta:
                ttft = ttft or time.perf_counter() - t0
                text.append(delta)
    return "".join(text).strip(), ttft, time.perf_counter() - t0, timing


async def tts_turn(s, a, sid, text, turn):
    """Streams speech; returns (ttfa s, audio s, underrun s) on a real-time playback clock."""
    body = dict(model=a.tts_model, input=text, voice=a.voice, response_format="pcm", stream=True)
    if a.language:
        body["language"] = a.language
    t0, first, got, underrun = time.perf_counter(), None, 0, 0.0
    async with s.post(a.url + "/v1/audio/speech", json=body, headers=turn_headers(sid, turn)) as r:
        if r.status != 200:
            raise RuntimeError(f"tts {r.status}: {(await r.text())[:200]}")
        timing = server_timing(r.headers.get("Server-Timing"))
        async for b in r.content.iter_any():
            now = time.perf_counter()
            if first is None:
                first = now
            else:
                # The player has consumed (now - first) s; audio received so far is got/SR s.
                behind = (now - first) - got / 2 / TTS_SR
                underrun = max(underrun, behind)
            got += len(b)
    return (first - t0) if first else None, got / 2 / TTS_SR, underrun, timing


async def call(s, a, idx, clips, rec):
    sid = f"call-{idx}-{uuid.uuid4().hex[:8]}"
    await asyncio.sleep(random.Random(a.seed * 1_000_003 + idx).uniform(0, a.ramp_s))
    history = [{"role": "system", "content": a.system}]
    rng = random.Random(a.seed * 1_000_003 + 1000 + idx)
    for turn in range(a.turns):
        row = dict(call=idx, turn=turn)
        t = {"id": f"{idx}.{turn}", "trace": uuid.uuid4().hex}
        try:
            if a.asr_model:
                clip = clips[rng.randrange(len(clips))]
                text, row["asr_final_s"], partials, row["srv_asr"] = await asr_turn(s, a, sid, clip, t)
                row["asr_partial_p50_s"] = statistics.median(partials) if partials else None
            else:
                await asyncio.sleep(a.user_s)
                text = "Hello."
            history.append({"role": "user", "content": text or "Hello."})
            if a.llm_model:
                reply, row["llm_ttft_s"], row["llm_total_s"], row["srv_llm"] = await llm_turn(s, a, sid, history, t)
            else:
                reply = a.reply
            history.append({"role": "assistant", "content": reply})
            (row["tts_ttfa_s"], row["tts_audio_s"], row["tts_underrun_s"],
             row["srv_tts"]) = await tts_turn(s, a, sid, reply or "Okay.", t)
            await asyncio.sleep(row["tts_audio_s"] + a.think_s)  # the user listens, then answers
        except Exception as e:  # noqa: BLE001 — every failure is a data point
            row["error"] = str(e)[:300]
        rec.append(row)


async def main_async(a):
    m = json.load(open(a.manifest)) if a.asr_model else []
    clips = []
    for c in m[: a.clips]:
        x, sr = sf.read(c["path"], dtype="float32")
        assert sr == ASR_SR, c["path"]
        clips.append(x)
    rec = []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=300)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        await asyncio.gather(*(call(s, a, i, clips, rec) for i in range(a.calls)))
    wall = time.perf_counter() - t0
    ok = [r for r in rec if "error" not in r]
    summ = dict(calls=a.calls, turns=len(rec), errors=len(rec) - len(ok), wall_s=round(wall, 1),
                first_error=next((r["error"] for r in rec if "error" in r), None))
    for k in ("asr_final_s", "asr_partial_p50_s", "llm_ttft_s", "llm_total_s", "tts_ttfa_s", "tts_underrun_s"):
        v = [r.get(k) for r in ok]
        if any(x is not None for x in v):
            summ[k.replace("_s", "") + "_p50_ms"] = round(1e3 * pct(v, 0.5), 1)
            summ[k.replace("_s", "") + "_p95_ms"] = round(1e3 * pct(v, 0.95), 1)
    under = [r["tts_underrun_s"] for r in ok if r.get("tts_underrun_s") is not None]
    summ["turns_with_underrun_gt_100ms"] = sum(u > 0.1 for u in under)
    # Server-side breakdown (Server-Timing), ms: srv_<stage>_<metric>_p50/_p95.
    for stage in ("asr", "llm", "tts"):
        for metric in ("queue", "wait-turn", "device", "first"):
            v = [r.get(f"srv_{stage}", {}).get(metric) for r in ok]
            if any(x is not None for x in v):
                key = f"srv_{stage}_{metric.replace('-', '_')}"
                summ[key + "_p50"], summ[key + "_p95"] = round(pct(v, 0.5), 1), round(pct(v, 0.95), 1)
    # Where the LLM turns over the TTFT SLO spent it on the server (medians).
    slow = [r.get("srv_llm", {}) for r in ok if (r.get("llm_ttft_s") or 0) * 1e3 > a.slo_ttft_ms]
    if slow:
        summ["srv_llm_slow"] = dict(turns=len(slow), **{m.replace("-", "_") + "_p50": pct([x.get(m) for x in slow], 0.5)
                                                      for m in ("queue", "wait-turn", "device", "first")})
    slo = dict(asr_final_p95_ms=a.slo_asr_ms, llm_ttft_p95_ms=a.slo_ttft_ms, tts_ttfa_p95_ms=a.slo_ttfa_ms)
    summ["slo_pass"] = (summ["errors"] == 0
                        and all(summ.get(k) is None or summ[k] <= v for k, v in slo.items())
                        and summ["turns_with_underrun_gt_100ms"] <= a.slo_underrun_frac * max(len(under), 1))
    print(json.dumps(summ), flush=True)
    if a.out:
        json.dump(dict(args=vars(a), summary=summ, turns=rec), open(a.out, "w"), indent=1)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--ramp-s", type=float, default=10.0, help="stagger call starts over this window")
    ap.add_argument("--seed", type=int, default=0, help="clip choice and start offsets; vary per run so "
                    "repeated runs on one server do not replay the same turns into the prefix cache")
    ap.add_argument("--think-s", type=float, default=1.0)
    ap.add_argument("--chunk-s", type=float, default=1.0)
    ap.add_argument("--asr-model")
    ap.add_argument("--user-s", type=float, default=3.0, help="user speech per turn without --asr-model")
    ap.add_argument("--llm-model")
    ap.add_argument("--tts-model", required=True)
    ap.add_argument("--voice", default="default")
    ap.add_argument("--language")
    ap.add_argument("--manifest", help="ASR clips (with --asr-model)")
    ap.add_argument("--clips", type=int, default=73)
    ap.add_argument("--system", default="You are a concise, friendly phone assistant. Answer in one or two short sentences.")
    ap.add_argument("--reply", default="Sure, I can help with that. Could you tell me a little more about what you need?")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--logprobs", type=int, default=0, help="request top_logprobs=N on LLM turns")
    ap.add_argument("--slo-asr-ms", type=float, default=500)
    ap.add_argument("--slo-ttft-ms", type=float, default=800)
    ap.add_argument("--slo-ttfa-ms", type=float, default=800)
    ap.add_argument("--slo-underrun-frac", type=float, default=0.01)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = asyncio.run(main_async(a))
    raise SystemExit(0 if s["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
