#!/usr/bin/env python3
"""Endpoint smoke against a running `plowrt serve`: what the server advertises, a few requests each.

  smoke_client.py <url> <outdir> [--asr-manifest M] [--vad-wav W] [--tts MODEL=VOICE[:LANG] ...]
                  [--models a,b]

/health, /metrics and /v1/models first. Then per model card, by its `x_plow_endpoints`:
  chat/completions      two greedy chats with a checked answer, one streamed chat, raw completions
  audio/transcriptions  the first 3 manifest clips ([{path, text}], paths relative to the manifest):
                        HTTP (80% word overlap with the reference), SSE (= HTTP text), and, when
                        `audio/transcriptions/stream` is listed, the native WebSocket (= HTTP text)
                        and OpenAI Realtime with manual commit. Needs --asr-manifest (or
                        $ASR_MANIFEST); skipped without one. WebSocket legs need `websocket-client`
                        and numpy; skipped when missing.
  audio/speech          one PCM request and one streamed request (24 kHz): duration 1-20 s, peak
                        > 500. Voice `default`, no language, unless --tts names them. WAVs and
                        texts.json are written for scripts/tts/asr_check.py.
/v1/audio/vad with --vad-wav (else the first manifest clip): 404 = no VAD packet loaded (recorded,
not a failure). Writes <outdir>/smoke.json; exit 0 only if every request passed.
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tts"))
from mtl_prompts import _BY_LANG as TTS_TEXT  # noqa: E402

JSON = {"Content-Type": "application/json"}
BND = "plowsmokeboundary"


def post(url, path, body, headers=JSON):
    t0 = time.perf_counter()
    req = urllib.request.Request(url + path, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read(), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.perf_counter() - t0


def sse(url, path, body, headers=JSON):
    req = urllib.request.Request(url + path, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                yield json.loads(payload)


def multipart(fields, filename, data):
    head = "".join(f"--{BND}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n" for k, v in fields.items())
    head += (f"--{BND}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
             "Content-Type: audio/wav\r\n\r\n")
    return head.encode() + data + f"\r\n--{BND}--\r\n".encode(), {"Content-Type": f"multipart/form-data; boundary={BND}"}


def words(s):
    return "".join(c for c in s.upper() if c.isalnum() or c == " ").split()


def overlap_ok(ref, hyp):
    return len(set(ref) & set(hyp)) >= 0.8 * len(set(ref))


def smoke_llm(url, model, rec):
    for prompt, want in (("What is the capital of France? Answer in one short sentence.", "paris"),
                         ("Write the numbers from one to five in words, separated by commas.", "three")):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 64, "temperature": 0}).encode()
        st, data, dt = post(url, "/v1/chat/completions", body)
        j = json.loads(data) if st == 200 else {}
        text = (j.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        rec(endpoint="chat/completions", prompt=prompt, status=st, latency_s=round(dt, 3), text=text,
            usage=j.get("usage"), ok=st == 200 and want in text.lower())
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Name the largest planet in the solar system in one sentence."}],
                       "max_tokens": 48, "temperature": 0, "stream": True}).encode()
    t0, first, parts = time.perf_counter(), None, []
    for ev in sse(url, "/v1/chat/completions", body):
        d = (ev.get("choices") or [{}])[0].get("delta", {}).get("content")
        if d:
            first = first or time.perf_counter() - t0
            parts.append(d)
    text = "".join(parts)
    rec(endpoint="chat/completions stream=true", ttft_s=round(first or -1, 3), latency_s=round(time.perf_counter() - t0, 3),
        text=text, ok="jupiter" in text.lower())
    for prompt in ("The quick brown fox", "<bos>The quick brown fox"):
        body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 16, "temperature": 0}).encode()
        st, data, dt = post(url, "/v1/completions", body)
        text = (json.loads(data).get("choices") or [{}])[0].get("text", "") if st == 200 else ""
        rec(endpoint="completions", prompt=prompt, status=st, latency_s=round(dt, 3), text=text,
            ok=st == 200 and len(text.strip()) > 0, note="raw completion; content informational")


def smoke_asr_ws(url, model, pcm, http_text, ref, rec):
    try:
        import numpy as np
        import websocket
    except ImportError as e:
        rec(endpoint="ws", skipped=f"{e!r}", ok=True)
        return
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
    try:
        t0 = time.perf_counter()
        ws = websocket.create_connection(ws_url + "/v1/audio/transcriptions/stream", timeout=120)
        ws.send(json.dumps({"type": "start", "version": 1, "model": model, "sample_rate": 16000,
                            "format": "pcm_s16le", "deltas": True}))
        credit = int(json.loads(ws.recv()).get("credit_samples") or 16000)
        seq, off, final = 0, 0, None
        while off < len(pcm):
            n = min(32000, credit * 2, len(pcm) - off)
            if n <= 0:
                ev = json.loads(ws.recv())
                if ev.get("type") == "credit":
                    credit += int(ev.get("samples") or ev.get("credit_samples") or 16000)
                continue
            ws.send_binary(struct.pack("<Q", seq) + pcm[off:off + n])
            seq, off, credit = seq + 1, off + n, credit - n // 2
        ws.send(json.dumps({"type": "finish"}))
        while final is None:
            ev = json.loads(ws.recv())
            if ev.get("type") == "final":
                final = ev.get("text", "")
            elif ev.get("type") == "error":
                final = "ERROR " + json.dumps(ev)
        ws.close()
        rec(endpoint="ws /v1/audio/transcriptions/stream", latency_s=round(time.perf_counter() - t0, 3), text=final,
            ok=words(final) == words(http_text))
    except Exception as e:
        rec(endpoint="ws /v1/audio/transcriptions/stream", error=repr(e), ok=False)
    try:
        t0 = time.perf_counter()
        ws = websocket.create_connection(ws_url + f"/v1/realtime?intent=transcription&model={model}", timeout=120)
        json.loads(ws.recv())
        ws.send(json.dumps({"type": "transcription_session.update", "session": {"input_audio_format": "pcm16",
                "input_audio_transcription": {"model": model}, "turn_detection": None}}))
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        p24 = np.clip(np.interp(np.arange(int(len(x) * 1.5)) / 24000.0, np.arange(len(x)) / 16000.0, x),
                      -32768, 32767).astype(np.int16).tobytes()
        for i in range(0, len(p24), 48000):
            ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(p24[i:i + 48000]).decode()}))
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        text = None
        while text is None:
            ev = json.loads(ws.recv())
            if ev.get("type") == "conversation.item.input_audio_transcription.completed":
                text = ev.get("transcript", "")
            elif ev.get("type") in ("error", "conversation.item.input_audio_transcription.failed"):
                text = "ERROR " + json.dumps(ev)
        ws.close()
        rec(endpoint="ws /v1/realtime", latency_s=round(time.perf_counter() - t0, 3), text=text,
            equals_http=words(text) == words(http_text), ok=overlap_ok(ref, words(text)))
    except Exception as e:
        rec(endpoint="ws /v1/realtime", error=repr(e), ok=False)


def smoke_asr(url, model, clips, ws_listed, rec):
    for path, ref_text in clips:
        wav = open(path, "rb").read()
        body, h = multipart({"model": model}, os.path.basename(path), wav)
        st, data, dt = post(url, "/v1/audio/transcriptions", body, h)
        text = json.loads(data).get("text", "") if st == 200 else ""
        ref, hyp = words(ref_text), words(text)
        rec(endpoint="audio/transcriptions", clip=path, status=st, latency_s=round(dt, 3), text=text, ref=ref_text,
            ok=st == 200 and overlap_ok(ref, hyp))
        sbody, h = multipart({"model": model, "stream": "true"}, os.path.basename(path), wav)
        t0, first, done = time.perf_counter(), None, None
        for ev in sse(url, "/v1/audio/transcriptions", sbody, h):
            if ev.get("type") == "transcript.text.delta":
                first = first or time.perf_counter() - t0
            elif ev.get("type") == "transcript.text.done":
                done = ev.get("text", "")
        rec(endpoint="audio/transcriptions stream=true", clip=path, first_delta_s=round(first or -1, 3),
            latency_s=round(time.perf_counter() - t0, 3), text=done, ok=done is not None and words(done) == hyp)
        if ws_listed:
            with wave.open(path) as w:
                pcm = w.readframes(w.getnframes())
            smoke_asr_ws(url, model, pcm, text, ref, rec)


def smoke_tts(url, model, voice, lang, out, texts, rec):
    text = TTS_TEXT.get(lang or "en", TTS_TEXT["en"])[0]
    b = dict(model=model, input=text, voice=voice, response_format="pcm", stream=False, seed=1)
    if lang:
        b["language"] = lang
    tag = model.replace("/", "_")
    for stream in (False, True):
        b["stream"] = stream
        t0, first, chunks, st = time.perf_counter(), None, [], 0
        req = urllib.request.Request(url + "/v1/audio/speech", data=json.dumps(b).encode(), headers=JSON)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                st = r.status
                while c := r.read1(65536):
                    first = first or time.perf_counter() - t0
                    chunks.append(c)
        except urllib.error.HTTPError as e:
            st, chunks = e.code, []
        pcm, dt = b"".join(chunks), time.perf_counter() - t0
        name = f"smoke_{tag}{'_s' if stream else ''}.wav"
        with wave.open(os.path.join(out, name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)
        dur = len(pcm) / 2 / 24000
        peak = max((abs(v) for v in struct.unpack(f"<{len(pcm) // 2}h", pcm[: len(pcm) // 2 * 2])), default=0)
        texts[name] = {"text": text, "language": lang} if lang else text
        rec(endpoint="audio/speech" + (" stream=true" if stream else ""), voice=voice, language=lang, status=st,
            ttfa_s=round(first or -1, 3), latency_s=round(dt, 3), audio_s=round(dur, 3), peak=peak, wav=name,
            ok=st == 200 and 1.0 < dur < 20.0 and (stream or peak > 500))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--asr-manifest", default=os.environ.get("ASR_MANIFEST"))
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--vad-wav", default=os.environ.get("VAD_WAV"))
    ap.add_argument("--tts", action="append", default=[], metavar="MODEL=VOICE[:LANG]")
    ap.add_argument("--models", default=None, help="comma-separated subset of the served models")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    os.makedirs(a.out, exist_ok=True)
    res = {"url": url, "requests": []}

    def rec(**row):
        row.setdefault("model", rec.model)
        res["requests"].append(row)
        print(("ok   " if row["ok"] else "FAIL ") + json.dumps(row, ensure_ascii=False)[:300], flush=True)
    rec.model = None

    cards = []
    for path in ("/health", "/metrics", "/v1/models"):
        try:
            with urllib.request.urlopen(url + path, timeout=30) as r:
                body = r.read()
                rec(endpoint=path, status=r.status, bytes=len(body), ok=r.status == 200 and len(body) > 0)
                if path == "/v1/models":
                    cards = json.loads(body)["data"]
        except Exception as e:
            rec(endpoint=path, error=repr(e), ok=False)
    clips = []
    if a.asr_manifest:
        base = os.path.dirname(os.path.abspath(a.asr_manifest))
        for row in json.load(open(a.asr_manifest))[: a.clips]:
            clips.append((os.path.join(base, row.get("path") or row["audio"]), row["text"]))
    tts = {}
    for spec in a.tts:
        m, v = spec.split("=", 1)
        voice, _, lang = v.partition(":")
        tts[m] = (voice, lang or None)
    pick = set(a.models.split(",")) if a.models else None
    texts = {}
    for card in cards:
        m, eps = card["id"], card.get("x_plow_endpoints", [])
        if card.get("parent") or (pick and m not in pick):
            continue
        rec.model = m
        res.setdefault("models", {})[m] = eps
        try:
            if "chat/completions" in eps:
                smoke_llm(url, m, rec)
            if "audio/transcriptions" in eps:
                if clips:
                    smoke_asr(url, m, clips, "audio/transcriptions/stream" in eps, rec)
                else:
                    rec(endpoint="audio/transcriptions", skipped="no --asr-manifest", ok=True)
            if "audio/speech" in eps:
                smoke_tts(url, m, *tts.get(m, ("default", None)), a.out, texts, rec)
        except Exception as e:
            rec(endpoint="(model)", error=repr(e), ok=False)
    rec.model = None
    vad = a.vad_wav or (clips[0][0] if clips else None)
    if vad:
        body, h = multipart({}, os.path.basename(vad), open(vad, "rb").read())
        st, data, dt = post(url, "/v1/audio/vad", body, h)
        if st == 404:
            rec(endpoint="audio/vad", status=st, skipped="no VAD packet loaded", ok=True)
        else:
            segs = json.loads(data).get("segments") if st == 200 else None
            rec(endpoint="audio/vad", status=st, latency_s=round(dt, 3), segments=segs, ok=bool(segs))
    if texts:
        json.dump(texts, open(os.path.join(a.out, "texts.json"), "w"), ensure_ascii=False)
    bad = [r for r in res["requests"] if not r["ok"]]
    res["ok"] = not bad and bool(cards)
    json.dump(res, open(os.path.join(a.out, "smoke.json"), "w"), indent=1, ensure_ascii=False)
    print(f"SMOKE requests={len(res['requests'])} failed={len(bad)} ok={res['ok']}")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
