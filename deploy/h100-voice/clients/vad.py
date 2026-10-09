#!/usr/bin/env python3
"""POST /v1/audio/vad: Silero VAD speech segments of a WAV file (runs on the server CPU).

  python clients/vad.py clients/samples/sample_en.wav [--threshold 0.5] [--min-silence-ms 100]

WAV at 8-48 kHz, up to 10 minutes. Prints {duration, speech_duration, segments: [{start, end}]}
in seconds. Defaults follow Silero's get_speech_timestamps.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import multipart, request  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-speech-ms", type=int, default=None, help="min_speech_duration_ms")
    ap.add_argument("--min-silence-ms", type=int, default=None, help="min_silence_duration_ms")
    ap.add_argument("--speech-pad-ms", type=int, default=None)
    ap.add_argument("--max-speech-s", type=float, default=None, help="max_speech_duration_s")
    a = ap.parse_args()
    fields = {k: str(v) for k, v in {"threshold": a.threshold, "min_speech_duration_ms": a.min_speech_ms,
                                     "min_silence_duration_ms": a.min_silence_ms, "speech_pad_ms": a.speech_pad_ms,
                                     "max_speech_duration_s": a.max_speech_s}.items() if v is not None}
    body, ctype = multipart(fields, {"file": (os.path.basename(a.wav), open(a.wav, "rb").read(), "audio/wav")})
    print(json.dumps(json.loads(request("POST", "/v1/audio/vad", body, {"Content-Type": ctype}).read()), indent=1))


if __name__ == "__main__":
    main()
