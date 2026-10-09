#!/usr/bin/env python3
"""OpenAI Realtime transcription session: GET /v1/realtime?intent=transcription.

  python clients/realtime_transcribe.py clients/samples/sample_en.wav [--model qwen3-asr] [--manual]

Server VAD (default): audio is appended at 1x real time, followed by 1.5 s of silence so the
endpointer closes the turn; the server sends speech_started / speech_stopped / committed, then
transcription delta and completed events per turn. --manual turns VAD off and commits once.
Audio is pcm16 at 24 kHz (the OpenAI Realtime format); the server resamples to 16 kHz.
Needs: pip install websockets
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time

import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import API_KEY, read_wav_pcm16, resample_pcm16, ws_url  # noqa: E402

SR = 24000


async def run(a):
    pcm, sr = read_wav_pcm16(a.wav)
    pcm = resample_pcm16(pcm, sr, SR)
    url = ws_url(f"/v1/realtime?intent=transcription&model={a.model}")
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    turn = None if a.manual else {"type": "server_vad", "silence_duration_ms": a.silence_ms}
    async with websockets.connect(url, additional_headers=headers, subprotocols=["realtime"], max_size=None) as ws:
        created = json.loads(await ws.recv())
        print(f"{created['type']}", file=sys.stderr)
        transcription = {"model": a.model}
        if a.language:
            transcription["language"] = a.language
        await ws.send(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": "pcm16", "input_audio_transcription": transcription, "turn_detection": turn}}))
        state = {"t_end": None, "t_speech": None}

        async def sender():
            step = SR // 10 * 2  # 100 ms per append event
            t_start = time.perf_counter()
            audio = pcm + (b"\x00\x00" * int(1.5 * SR) if turn else b"")
            for off in range(0, len(audio), step):
                await asyncio.sleep(max(0.0, t_start + off / 2 / SR - time.perf_counter()))
                if off >= len(pcm) and state["t_speech"] is None:
                    state["t_speech"] = time.perf_counter()
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(audio[off:off + step]).decode()}))
            state["t_end"] = time.perf_counter()
            if not turn:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        task = asyncio.create_task(sender())
        async for msg in ws:
            ev = json.loads(msg)
            kind = ev["type"]
            if kind.endswith(".delta"):
                print(f"delta: {ev['delta']!r}")
            elif kind.endswith("input_audio_transcription.completed"):
                ref = state["t_speech"] or state["t_end"] or time.perf_counter()
                print(f"completed: {ev['transcript']}  ({1000 * (time.perf_counter() - ref):.0f} ms after the end of speech)")
                break
            elif kind == "error" or kind.endswith(".failed"):
                raise SystemExit(f"{kind}: {ev}")
            else:
                print(kind + "".join(f" {k}={ev[k]}" for k in ("audio_start_ms", "audio_end_ms", "item_id") if k in ev))
        await task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="qwen3-asr")
    ap.add_argument("--language", default=None)
    ap.add_argument("--manual", action="store_true", help="turn_detection null + one input_audio_buffer.commit")
    ap.add_argument("--silence-ms", type=int, default=500, help="server_vad silence_duration_ms (200..2000)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
