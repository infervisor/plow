#!/usr/bin/env python3
"""POST /v1/audio/transcriptions: one file, JSON reply or (--stream) server-sent events.

  python clients/transcribe.py clients/samples/sample_en.wav [--model qwen3-asr] [--language en] [--stream]

WAV input: mono/stereo, 8/16/24/32-bit integer or 32-bit float, 8-48 kHz (resampled server-side),
0.5 to 30 s, at most 4 MiB.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import multipart, request, sse_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="qwen3-asr")
    ap.add_argument("--language", default=None, help="e.g. en; omit to auto-detect")
    ap.add_argument("--stream", action="store_true", help="SSE: transcript.text.delta events, then .done")
    a = ap.parse_args()
    fields = {"model": a.model}
    if a.language:
        fields["language"] = a.language
    if a.stream:
        fields["stream"] = "true"
    body, ctype = multipart(fields, {"file": (os.path.basename(a.wav), open(a.wav, "rb").read(), "audio/wav")})
    t0 = time.perf_counter()
    resp = request("POST", "/v1/audio/transcriptions", body, {"Content-Type": ctype})
    if not a.stream:
        reply = json.loads(resp.read())
        print(json.dumps(reply, ensure_ascii=False))
        print(f"latency {1000 * (time.perf_counter() - t0):.0f} ms", file=sys.stderr)
        return
    first = None
    for ev in sse_events(resp):
        if ev.get("type") == "transcript.text.delta":
            first = first or time.perf_counter() - t0
            print(ev["delta"], end="", flush=True)
        elif ev.get("type") == "transcript.text.done":
            print()
            print(json.dumps(ev, ensure_ascii=False))
        elif "error" in ev:
            raise SystemExit(f"error: {ev['error']}")
    total = time.perf_counter() - t0
    print(f"first delta {1000 * (first or total):.0f} ms, total {1000 * total:.0f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
