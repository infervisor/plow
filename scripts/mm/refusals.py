#!/usr/bin/env python3
"""Media error paths of a plowrt server: each case is a request and the status it must get.

  refusals.py <base> <model> [--image png] [--audio wav] [--compressed clip.mp3 ...]

Image cases run when the model card lists `image`, audio cases when it lists `audio`; a model
without a modality must refuse its parts with 400. Compressed clips (mp3/flac/ogg) must be
accepted and transcribed to non-empty text. Prints one JSON line per case, then
`refusals pass|fail`; exits 1 on a failure.
"""
import argparse, base64, io, json, sys, urllib.error, urllib.request, wave


def post(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read().decode(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), dict(e.headers)


def long_wav(path, seconds):
    """`path` repeated to at least `seconds` (16-bit PCM)."""
    w = wave.open(path)
    frames, rate, params = w.readframes(w.getnframes()), w.getframerate(), w.getparams()
    reps = int(seconds * rate / max(1, w.getnframes())) + 1
    out = io.BytesIO()
    o = wave.open(out, "wb")
    o.setparams(params)
    o.writeframes(frames * reps)
    o.close()
    return out.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--image")
    ap.add_argument("--audio")
    ap.add_argument("--compressed", nargs="*", default=[])
    args = ap.parse_args()
    cards = json.load(urllib.request.urlopen(args.base + "/v1/models"))["data"]
    modalities = next(c for c in cards if c["id"] == args.model).get("x_plow_modalities", [])
    text = {"type": "text", "text": "hi"}
    b64 = lambda data: base64.b64encode(data).decode()
    cases = {
        "http_url": ([{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}, text], 400),
        "bad_base64": ([{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}, text], 400),
    }
    if args.image:
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(open(args.image, "rb").read())}}
        if "image" in modalities:
            cases["too_many_images"] = ([img] * 9 + [text], 400)
            cases["stream_image"] = ([img, text], 200)
        else:
            cases["unsupported_image"] = ([img, text], 400)
    if args.audio:
        clip = lambda data, fmt="wav": {"type": "input_audio", "input_audio": {"data": b64(data), "format": fmt}}
        wav = open(args.audio, "rb").read()
        if "audio" in modalities:
            cases["opus_format"] = ([clip(wav, "opus"), text], 400)
            cases["too_many_audio"] = ([clip(wav)] * 5 + [text], 400)
            cases["audio_too_long"] = ([clip(long_wav(args.audio, 31)), text], 400)
            cases["not_audio"] = ([clip(b"definitely not audio", "mp3"), text], 400)
            if "image" not in modalities:
                cases["stream_audio"] = ([clip(wav), text], 200)
            for path in args.compressed:
                fmt = path.rsplit(".", 1)[-1].lower()
                cases[f"accept_{fmt}"] = ([clip(open(path, "rb").read(), fmt), {"type": "text", "text": "Transcribe this audio."}], 200)
        else:
            cases["unsupported_audio"] = ([clip(wav), text], 400)
    ok = True
    for name, (content, want) in cases.items():
        body = {"model": args.model, "messages": [{"role": "user", "content": content}], "max_tokens": 16, "temperature": 0}
        if name.startswith("stream_"):
            body.update(stream=True, logprobs=True, top_logprobs=2)
        status, got, headers = post(args.base, body)
        passed = status == want
        if name.startswith("stream_") and passed:
            chunks = [l for l in got.splitlines() if l.startswith("data: {")]
            lp = sum(1 for l in chunks if (json.loads(l[6:]).get("choices") or [{}])[0].get("logprobs"))
            got = f"{len(chunks)} chunks, {lp} with logprobs"
            passed = len(chunks) > 1 and lp > 0
        elif name.startswith("accept_") and passed:
            got = json.loads(got)["choices"][0]["message"]["content"] or ""
            passed = bool(got.strip())
        if status == 503:
            got += f" (Retry-After: {headers.get('Retry-After')})"
        ok &= passed
        print(json.dumps({"case": name, "pass": passed, "status": status, "want": want, "body": str(got)[:160]}), flush=True)
    print("refusals", "pass" if ok else "fail")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
