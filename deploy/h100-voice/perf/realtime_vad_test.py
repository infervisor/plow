#!/usr/bin/env python3
"""OpenAI Realtime transcription smoke for `server_vad` turn detection against `plowrt serve`.

  python perf/realtime_vad_test.py --model qwen3-asr --out results/realtime_vad.json \
      [--manifest data/librispeech-dummy/manifest.json] [--clips 10] [--silence-ms 500] [--threshold 0.5]

Kit copy of the repo harness scripts/asr/realtime_vad_smoke.py: --url defaults to $PLOW_URL (ws://),
PLOW_API_KEY is sent as a bearer token, manifest paths resolve relative to the manifest, and the
exit status is 1 unless the pauses scenario gives one turn per clip and the continuous WER is
within --max-wer.

Scenarios, each on its own session (pcm16 24 kHz, 100 ms appends, sent unpaced):
  pauses      three LibriSpeech clips joined by 2 s of silence and 1.5 s of -50 dBFS noise:
              one turn per clip; speech_started/stopped are compared with the clip bounds
  continuous  each of the first --clips manifest clips alone (then 1.5 s of silence): one turn,
              its transcript scored against the reference text
Writes every event and a summary; prints `REALTIME_VAD ...` lines.
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time
import wave

import numpy as np
import websockets

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH = {"Authorization": f"Bearer {os.environ['PLOW_API_KEY']}"} if os.environ.get("PLOW_API_KEY") else {}


def wer(refs, hyps):
    import re
    import jiwer

    def norm(s):
        s = s.lower().replace("\u2019", "'")
        return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s).split())
    return jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps])


def resample_16k_to_24k(x):
    return np.interp(np.arange(len(x) * 3 // 2) * 2 / 3, np.arange(len(x)), x).astype(np.float32)

RATE = 16000


def read_wav(path):
    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return x


async def session(url, model, audio16, args):
    """Stream `audio16` through one Realtime session; returns its server events (with arrival time)."""
    events = []
    pcm = np.clip(resample_16k_to_24k(audio16), -1, 1)
    pcm = (pcm * 32767).astype("<i2").tobytes()
    async with websockets.connect(f"{url}/v1/realtime?intent=transcription&model={model}", subprotocols=["realtime"],
                                  additional_headers=AUTH, max_size=1 << 24) as ws:
        created = json.loads(await ws.recv())
        assert created["type"] == "transcription_session.created", created
        await ws.send(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_transcription": {"model": model, "language": "en"},
            "turn_detection": {"type": "server_vad", "threshold": args.threshold,
                               "prefix_padding_ms": args.prefix_ms, "silence_duration_ms": args.silence_ms}}}))
        updated = json.loads(await ws.recv())
        assert updated["type"] == "transcription_session.updated", updated
        t0 = time.perf_counter()
        step = 4800  # 100 ms of 24 kHz s16
        for i in range(0, len(pcm), step):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[i:i + step]).decode()}))
        committed, completed = 0, 0
        while True:
            try:
                e = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout))
            except asyncio.TimeoutError:
                break
            e["t_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            events.append(e)
            committed += e["type"] == "input_audio_buffer.committed"
            completed += e["type"] in ("conversation.item.input_audio_transcription.completed",
                                       "conversation.item.input_audio_transcription.failed")
            if e["type"] == "error":
                break
            if committed and completed == committed and e["type"].endswith(("completed", "failed")):
                # Every committed turn answered; stop once no further event arrives promptly.
                try:
                    e = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                    e["t_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
                    events.append(e)
                    committed += e["type"] == "input_audio_buffer.committed"
                except asyncio.TimeoutError:
                    break
    return events


def turns(events):
    out = {}
    for e in events:
        item = e.get("item_id")
        if not item:
            continue
        t = out.setdefault(item, {"item": item})
        if e["type"] == "input_audio_buffer.speech_started":
            t["start_ms"] = e["audio_start_ms"]
        elif e["type"] == "input_audio_buffer.speech_stopped":
            t["end_ms"] = e["audio_end_ms"]
        elif e["type"] == "conversation.item.input_audio_transcription.completed":
            t["text"] = e["transcript"]
        elif e["type"] == "conversation.item.input_audio_transcription.failed":
            t["error"] = e["error"]
    return [t for t in out.values() if "start_ms" in t or "text" in t]


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("PLOW_URL", "http://127.0.0.1:8000").replace("http", "ws", 1))
    p.add_argument("--model", default="qwen3-asr")
    p.add_argument("--manifest", default=os.path.join(KIT, "data", "librispeech-dummy", "manifest.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--clips", type=int, default=10)
    p.add_argument("--silence-ms", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--prefix-ms", type=int, default=300)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-wer", type=float, default=0.10, help="pass bound on the continuous scenario's WER")
    args = p.parse_args()
    man = json.load(open(args.manifest))
    for m in man:
        m["path"] = os.path.join(os.path.dirname(os.path.abspath(args.manifest)), m["path"])
    rng = np.random.default_rng(7)
    result = {"args": vars(args)}

    # pauses: clip bounds are known, so turn boundaries can be checked.
    picks = [man[0], man[1], man[3]]
    parts, bounds, at = [np.zeros(RATE // 2, np.float32)], [], RATE // 2
    gaps = [np.zeros(2 * RATE, np.float32), (rng.standard_normal(3 * RATE // 2) * 10 ** (-50 / 20)).astype(np.float32),
            np.zeros(2 * RATE, np.float32)]
    for clip, gap in zip(picks, gaps):
        x = read_wav(clip["path"])
        bounds.append((at * 1000 // RATE, (at + len(x)) * 1000 // RATE))
        parts += [x, gap]
        at += len(x) + len(gap)
    ev = await session(args.url, args.model, np.concatenate(parts), args)
    tt = turns(ev)
    rows = []
    for i, t in enumerate(tt):
        b = bounds[i] if i < len(bounds) else (None, None)
        rows.append(dict(t, clip_start_ms=b[0], clip_end_ms=b[1],
                         wer=wer([picks[i]["text"]], [t.get("text", "")]) if i < len(picks) else None))
    result["pauses"] = {"events": ev, "turns": rows, "expected_turns": len(picks)}
    print(f"REALTIME_VAD pauses turns={len(tt)} expected={len(picks)}")
    for r in rows:
        print(f"  start {r.get('start_ms')} (clip {r['clip_start_ms']})  stop {r.get('end_ms')} (clip end {r['clip_end_ms']})"
              f"  wer {r['wer'] if r['wer'] is None else round(r['wer'], 4)}  {r.get('text', '')[:70]!r}")

    # continuous: one clip per session.
    cont = []
    for clip in man[:args.clips]:
        x = np.concatenate([read_wav(clip["path"]), np.zeros(3 * RATE // 2, np.float32)])
        ev = await session(args.url, args.model, x, args)
        tt = turns(ev)
        text = " ".join(t.get("text", "") for t in tt).strip()
        cont.append(dict(path=clip["path"], turns=len(tt), text=text, ref=clip["text"], wer=wer([clip["text"]], [text]),
                         bounds=[(t.get("start_ms"), t.get("end_ms")) for t in tt]))
    result["continuous"] = cont
    result["summary"] = dict(pauses_turns=len(result["pauses"]["turns"]), continuous_single_turn=sum(c["turns"] == 1 for c in cont),
                             continuous_clips=len(cont), continuous_wer=wer([c["ref"] for c in cont], [c["text"] for c in cont]))
    print(f"REALTIME_VAD continuous clips={len(cont)} single_turn={result['summary']['continuous_single_turn']} "
          f"wer={result['summary']['continuous_wer']:.4f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    s = result["summary"]
    # A clip may legitimately split at an in-sentence pause longer than --silence-ms; the check is
    # the turn count on the known pauses and the transcript quality across all turns.
    ok = s["pauses_turns"] == len(picks) and s["continuous_wer"] <= args.max_wer
    print(f"REALTIME_VAD {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
