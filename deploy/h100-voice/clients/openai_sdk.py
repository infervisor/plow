#!/usr/bin/env python3
"""The same endpoints through the official `openai` Python SDK (pip install openai).

  python clients/openai_sdk.py clients/samples/sample_en.wav

Shows: models.list, audio.transcriptions.create (plain and stream=True), chat.completions.create
(streamed), audio.speech.with_streaming_response.create to a WAV file.
"""
import os
import sys
import time

from openai import OpenAI

URL = os.environ.get("PLOW_URL", "http://127.0.0.1:8000").rstrip("/")
client = OpenAI(base_url=URL + "/v1", api_key=os.environ.get("PLOW_API_KEY") or "unused")

wav = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "samples", "sample_en.wav")

print("models:", [m.id for m in client.models.list().data])

with open(wav, "rb") as f:
    t = client.audio.transcriptions.create(model="qwen3-asr", file=f, language="en")
print("transcription:", t.text)

with open(wav, "rb") as f:
    stream = client.audio.transcriptions.create(model="qwen3-asr", file=f, stream=True)
    for event in stream:
        if event.type == "transcript.text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "transcript.text.done":
            print("\nstreamed transcription done:", event.text)

t0 = time.perf_counter()
first = None
stream = client.chat.completions.create(
    model="gemma-4-e4b", temperature=0, max_tokens=64, stream=True,
    messages=[{"role": "user", "content": "In one sentence, why is the sky blue?"}])
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        first = first or time.perf_counter() - t0
        print(chunk.choices[0].delta.content, end="", flush=True)
print(f"\nchat TTFT {1000 * first:.0f} ms")

with client.audio.speech.with_streaming_response.create(
        model="chatterbox-mtl", voice="default", input="Hello! This audio came through the OpenAI SDK.",
        response_format="wav") as resp:
    resp.stream_to_file("openai_sdk_speech.wav")
print("wrote openai_sdk_speech.wav")
