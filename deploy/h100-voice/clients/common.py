"""Shared helpers for the plow-voice client examples (standard library only).

Server and key come from the environment:
  PLOW_URL      base URL (default http://127.0.0.1:8000)
  PLOW_API_KEY  API key, sent as `Authorization: Bearer <key>` (omit when the server has none)
"""
import json
import os
import struct
import urllib.error
import urllib.request
import uuid
import wave

URL = os.environ.get("PLOW_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("PLOW_API_KEY", "")


def ws_url(path):
    return URL.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + path


def auth_headers(extra=None):
    h = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    h.update(extra or {})
    return h


def request(method, path, body=None, headers=None, timeout=600):
    """Returns the open response (caller reads/streams it). Raises SystemExit with the API error."""
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    h = {"Content-Type": "application/json"} if isinstance(body, (dict, list)) else {}
    h.update(headers or {})
    req = urllib.request.Request(URL + path, data=data, method=method, headers=auth_headers(h))
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} {path}: {e.read().decode(errors='replace')[:500]}")


def multipart(fields, files):
    """fields: {name: str}; files: {name: (filename, bytes, content_type)} -> (body, content_type)."""
    boundary = uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, (fn, data, ct) in files.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\n'
                   f"Content-Type: {ct}\r\n\r\n".encode() + data + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def sse_events(resp):
    """Yields each `data:` payload of a server-sent-events response (stops at [DONE])."""
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def read_wav_pcm16(path):
    """(mono s16le bytes, sample_rate) of a 16-bit PCM WAV; stereo is averaged."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: 16-bit PCM WAV expected")
        sr, ch, raw = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    if ch == 1:
        return raw, sr
    n = len(raw) // (2 * ch)
    s = struct.unpack(f"<{n * ch}h", raw)
    mono = [sum(s[i * ch:(i + 1) * ch]) // ch for i in range(n)]
    return struct.pack(f"<{n}h", *mono), sr


def resample_pcm16(pcm, src, dst):
    """Linear-interpolation resample of mono s16le (enough for examples; the server resamples
    8-48 kHz input itself for the HTTP and native WebSocket APIs)."""
    if src == dst:
        return pcm
    n = len(pcm) // 2
    s = struct.unpack(f"<{n}h", pcm)
    m = int(n * dst / src)
    out = []
    for i in range(m):
        x = i * src / dst
        j = int(x)
        f = x - j
        a = s[min(j, n - 1)]
        b = s[min(j + 1, n - 1)]
        out.append(int(round(a + (b - a) * f)))
    return struct.pack(f"<{m}h", *out)


class WavWriter:
    """Writes s16le mono PCM to a WAV file as it streams in; the header is patched on close."""

    def __init__(self, path, sample_rate):
        self.f = open(path, "wb")
        self.sr = sample_rate
        self.n = 0
        self.f.write(self._header(0))

    def _header(self, nbytes):
        return (b"RIFF" + struct.pack("<I", 36 + nbytes) + b"WAVEfmt " +
                struct.pack("<IHHIIHH", 16, 1, 1, self.sr, self.sr * 2, 2, 16) + b"data" + struct.pack("<I", nbytes))

    def write(self, pcm):
        self.f.write(pcm)
        self.n += len(pcm)

    def close(self):
        self.f.seek(0)
        self.f.write(self._header(self.n))
        self.f.close()
        return self.n / 2 / self.sr


def wav_payload(data):
    """(pcm s16le, sample_rate) from WAV bytes (a non-streamed /v1/audio/speech response)."""
    import io
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()
