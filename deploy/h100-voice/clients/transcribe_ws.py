#!/usr/bin/env python3
"""Native streaming ASR over WebSocket: GET /v1/audio/transcriptions/stream (protocol v1).

  python clients/transcribe_ws.py clients/samples/sample_en.wav [--model qwen3-asr] [--partials] [--deltas]
                                  [--continuous] [--realtime]

Sends the WAV as 16-bit PCM in credit-sized binary frames (u64 little-endian sequence number +
PCM), then {"type":"finish"}; prints partial / delta / final events. --realtime paces the audio at
1x (as a microphone would); --continuous uses the endpointed multi-segment mode.
Needs: pip install websockets
"""
import argparse
import asyncio
import json
import os
import struct
import sys
import time

import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import auth_headers, read_wav_pcm16, ws_url  # noqa: E402


async def run(a):
    pcm, sr = read_wav_pcm16(a.wav)
    start = {"type": "start", "version": 1, "model": a.model, "sample_rate": sr, "format": "pcm_s16le",
             "partials": a.partials, "deltas": a.deltas}
    if a.language:
        start["language"] = a.language
    if a.continuous:
        start["mode"] = "continuous"
    t0 = time.perf_counter()
    async with websockets.connect(ws_url("/v1/audio/transcriptions/stream"), additional_headers=auth_headers(),
                                  max_size=None) as ws:
        await ws.send(json.dumps(start))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"start refused: {ready}")
        print(f"ready: session {ready['session_id']}, credit {ready['credit_samples']} samples, "
              f"max chunk {ready['max_chunk_bytes']} B", file=sys.stderr)
        credit = ready["credit_samples"]
        max_bytes = ready["max_chunk_bytes"]
        state = {"credit": credit, "t_last": None}
        got_credit = asyncio.Event()

        async def sender():
            seq, off = 0, 0
            t_start = time.perf_counter()
            while off < len(pcm):
                while state["credit"] <= 0:
                    got_credit.clear()
                    await got_credit.wait()
                n = min(len(pcm) - off, max_bytes, state["credit"] * 2, sr // 5 * 2)  # <= 200 ms per frame
                if a.realtime:
                    await asyncio.sleep(max(0.0, t_start + (off + n) / 2 / sr - time.perf_counter()))
                await ws.send(struct.pack("<Q", seq) + pcm[off:off + n])
                state["credit"] -= n // 2
                seq += 1
                off += n
            state["t_last"] = time.perf_counter()
            await ws.send(json.dumps({"type": "finish"}))

        send_task = asyncio.create_task(sender())
        async for msg in ws:
            ev = json.loads(msg)
            kind = ev.get("type")
            if kind == "credit":
                state["credit"] += ev["credit_samples"]
                got_credit.set()
            elif kind == "partial":
                print(f"partial[{ev.get('segment', 0)}.{ev['revision']}]: {ev['text']}")
            elif kind == "delta":
                print(f"delta: {ev['text']!r}")
            elif kind == "final":
                lat = time.perf_counter() - (state["t_last"] or t0)
                seg = f" segment {ev['segment']} [{ev['start_ms']}-{ev['end_ms']} ms]" if "segment" in ev else ""
                print(f"final{seg}: {ev['text']}  (language {ev.get('language')}, {1000 * lat:.0f} ms after last audio)")
                if not a.continuous:
                    break
            elif kind == "done":
                print(f"done: {ev['segments']} segments")
                break
            elif kind == "error":
                print(f"error: {ev}", file=sys.stderr)
                if ev.get("terminal"):
                    raise SystemExit(1)
        await send_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="qwen3-asr")
    ap.add_argument("--language", default=None)
    ap.add_argument("--partials", action="store_true", help="revisable partial transcripts while audio arrives")
    ap.add_argument("--deltas", action="store_true", help="stream the final transcript as delta events")
    ap.add_argument("--continuous", action="store_true", help="endpointed multi-segment mode (unbounded audio)")
    ap.add_argument("--realtime", action="store_true", help="send audio at 1x real time")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
