#!/usr/bin/env python3
"""Voice-agent round trip with per-stage timing: audio in -> ASR -> LLM -> TTS audio out.

  python clients/voice_agent.py clients/samples/sample_en.wav reply.wav [--asr qwen3-asr] [--llm gemma-4-e4b]
         [--tts chatterbox-mtl] [--voice default] [--language en] [--http-asr] [--turns 1]

1. ASR: the WAV streams over the native WebSocket at 1x real time (as from a microphone); the
   final transcript arrives after the last audio (--http-asr: one multipart upload instead).
2. LLM: streamed chat completion; the reply is cut at sentence ends as tokens arrive.
3. TTS: each sentence is synthesized as soon as it is complete (streamed PCM), appended to the
   output WAV in order.
All three requests carry one X-Session-Id, so a multi-turn call (--turns N, the same audio as each
user turn) reuses the session's server-side state. Timings are relative to the end of the user's
audio, the moment a real agent starts waiting.
Needs: pip install websockets
"""
import argparse
import asyncio
import json
import os
import queue
import re
import struct
import sys
import threading
import time
import uuid

import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WavWriter, auth_headers, multipart, read_wav_pcm16, request, sse_events, ws_url  # noqa: E402

TTS_SR = 24000
SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


async def asr_ws(a, pcm, sr, sid):
    start = {"type": "start", "version": 1, "model": a.asr, "sample_rate": sr, "format": "pcm_s16le"}
    if a.language:
        start["language"] = a.language
    async with websockets.connect(ws_url("/v1/audio/transcriptions/stream"),
                                  additional_headers=auth_headers({"X-Session-Id": sid}), max_size=None) as ws:
        await ws.send(json.dumps(start))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"ASR start refused: {ready}")
        credit, step = ready["credit_samples"], sr // 10 * 2
        t_start, seq, off = time.perf_counter(), 0, 0
        while off < len(pcm):
            while credit <= 0:
                ev = json.loads(await ws.recv())
                if ev["type"] == "credit":
                    credit += ev["credit_samples"]
                elif ev["type"] == "error":
                    raise SystemExit(f"ASR error: {ev}")
            n = min(step, len(pcm) - off, credit * 2)
            await asyncio.sleep(max(0.0, t_start + (off + n) / 2 / sr - time.perf_counter()))
            await ws.send(struct.pack("<Q", seq) + pcm[off:off + n])
            credit -= n // 2
            seq += 1
            off += n
        t_end = time.perf_counter()
        await ws.send(json.dumps({"type": "finish"}))
        async for msg in ws:
            ev = json.loads(msg)
            if ev["type"] == "final":
                return ev["text"], t_end, time.perf_counter()
            if ev["type"] == "error":
                raise SystemExit(f"ASR error: {ev}")
    raise SystemExit("ASR: connection closed without a final")


def asr_http(a, wav_path, sid):
    fields = {"model": a.asr}
    if a.language:
        fields["language"] = a.language
    body, ctype = multipart(fields, {"file": ("user.wav", open(wav_path, "rb").read(), "audio/wav")})
    t_end = time.perf_counter()  # the whole utterance is available now
    reply = json.loads(request("POST", "/v1/audio/transcriptions", body, {"Content-Type": ctype, "X-Session-Id": sid}).read())
    return reply["text"], t_end, time.perf_counter()


def llm_stream(a, history, sid, sentences, marks):
    body = {"model": a.llm, "messages": history, "max_tokens": a.max_tokens, "temperature": 0, "stream": True}
    resp = request("POST", "/v1/chat/completions", body, {"X-Session-Id": sid})
    buf, text = "", []
    for ev in sse_events(resp):
        if "error" in ev:
            raise SystemExit(f"LLM error: {ev['error']}")
        for ch in ev.get("choices", []):
            delta = ch.get("delta", {}).get("content")
            if not delta:
                continue
            marks.setdefault("llm_first_token", time.perf_counter())
            text.append(delta)
            buf += delta
            parts = SENTENCE_END.split(buf)
            for s in parts[:-1]:
                if s.strip():
                    marks.setdefault("first_sentence", time.perf_counter())
                    sentences.put(s.strip())
            buf = parts[-1]
    if buf.strip():
        marks.setdefault("first_sentence", time.perf_counter())
        sentences.put(buf.strip())
    marks["llm_done"] = time.perf_counter()
    sentences.put(None)
    return "".join(text).strip()


def tts_worker(a, sid, sentences, writer, marks):
    while True:
        s = sentences.get()
        if s is None:
            break
        body = {"model": a.tts, "input": s, "voice": a.voice, "response_format": "pcm", "stream": True}
        if a.tts_language:
            body["language"] = a.tts_language
        resp = request("POST", "/v1/audio/speech", body, {"X-Session-Id": sid})
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            marks.setdefault("first_audio", time.perf_counter())
            writer.write(chunk)
    marks["tts_done"] = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--asr", default="qwen3-asr")
    ap.add_argument("--llm", default="gemma-4-e4b")
    ap.add_argument("--tts", default="chatterbox-mtl")
    ap.add_argument("--voice", default="default")
    ap.add_argument("--language", default=None, help="ASR language (omit to auto-detect)")
    ap.add_argument("--tts-language", default=None, help="chatterbox-mtl language, e.g. en")
    ap.add_argument("--system", default="You are a concise, friendly voice assistant. Answer in one or two short sentences.")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--http-asr", action="store_true")
    ap.add_argument("--turns", type=int, default=1)
    a = ap.parse_args()
    pcm, sr = read_wav_pcm16(a.wav)
    sid = uuid.uuid4().hex
    history = [{"role": "system", "content": a.system}]
    writer = WavWriter(a.out, TTS_SR)
    report = []
    for turn in range(a.turns):
        if a.http_asr:
            text, t_end, t_final = asr_http(a, a.wav, sid)
        else:
            text, t_end, t_final = asyncio.run(asr_ws(a, pcm, sr, sid))
        history.append({"role": "user", "content": text})
        marks = {}
        sentences = queue.Queue()
        tts = threading.Thread(target=tts_worker, args=(a, sid, sentences, writer, marks))
        tts.start()
        reply = llm_stream(a, history, sid, sentences, marks)
        tts.join()
        history.append({"role": "assistant", "content": reply})
        ms = lambda t: round(1000 * (t - t_end)) if t else None  # noqa: E731
        row = {"turn": turn, "user": text, "agent": reply, "asr_final_ms": ms(t_final),
               "llm_ttft_ms": round(1000 * (marks.get("llm_first_token", t_final) - t_final)),
               "first_sentence_ms": ms(marks.get("first_sentence")), "first_audio_ms": ms(marks.get("first_audio")),
               "llm_done_ms": ms(marks.get("llm_done")), "tts_done_ms": ms(marks.get("tts_done"))}
        report.append(row)
        print(f"turn {turn}: user said: {text!r}\n        agent said: {reply!r}")
        print(f"        after end of user audio: ASR final {row['asr_final_ms']} ms | LLM first token "
              f"+{row['llm_ttft_ms']} ms | first sentence {row['first_sentence_ms']} ms | "
              f"FIRST AGENT AUDIO {row['first_audio_ms']} ms | audio done {row['tts_done_ms']} ms")
    dur = writer.close()
    print(f"{a.out}: {dur:.2f} s of agent audio")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
