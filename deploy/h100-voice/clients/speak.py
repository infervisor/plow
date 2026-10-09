#!/usr/bin/env python3
"""POST /v1/audio/speech: text to a WAV file, streamed (default) or in one response.

  python clients/speak.py "Hello from the voice stack." out.wav [--model chatterbox-mtl]
         [--voice default] [--language en] [--no-stream]

Streaming asks for raw PCM (response_format=pcm: s16le mono, 24 kHz) and writes it to the WAV as
it arrives, printing time to first audio; --no-stream asks for a complete WAV.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WavWriter, request, wav_payload  # noqa: E402

SR = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--model", default="chatterbox-mtl")
    ap.add_argument("--voice", default="default", help="chatterbox*: default; veena: e.g. kavya, agastya")
    ap.add_argument("--language", default=None, help="chatterbox-mtl only (ISO 639-1, default en)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-stream", action="store_true")
    a = ap.parse_args()
    body = {"model": a.model, "input": a.text, "voice": a.voice,
            "response_format": "wav" if a.no_stream else "pcm", "stream": not a.no_stream}
    if a.language:
        body["language"] = a.language
    if a.seed is not None:
        body["seed"] = a.seed
    t0 = time.perf_counter()
    resp = request("POST", "/v1/audio/speech", body)
    if a.no_stream:
        pcm, sr = wav_payload(resp.read())
        w = WavWriter(a.out, sr)
        w.write(pcm)
        dur = w.close()
        dt = time.perf_counter() - t0
        print(f"{a.out}: {dur:.2f} s of audio in {dt:.2f} s (RTF {dt / max(dur, 1e-9):.3f})", file=sys.stderr)
        return
    w = WavWriter(a.out, SR)
    first = None
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        if not chunk:
            break
        first = first or time.perf_counter() - t0
        w.write(chunk)
    dur = w.close()
    dt = time.perf_counter() - t0
    print(f"{a.out}: {dur:.2f} s of audio; first audio {1000 * (first or dt):.0f} ms, total {dt:.2f} s "
          f"(RTF {dt / max(dur, 1e-9):.3f})", file=sys.stderr)


if __name__ == "__main__":
    main()
