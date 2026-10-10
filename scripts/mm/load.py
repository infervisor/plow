#!/usr/bin/env python3
"""Served multimodal load test: concurrent text, image, audio and image+audio chat requests.

  load.py <ref.json> --base URL --model M --out DIR [--concs 1,8,32] [--max-tokens 64]

`ref.json` comes from `scripts/mm/hf_ref.py all`; its items are the shared media (same bytes in
many requests, so the same bit-31 ids and prefix-cache hits). Unique media are minted per request
(a fresh PNG, a wav with perturbed low bits), so their ids never repeat.

  1. baseline  every shared request alone, greedy (temperature 0, top-2 logprobs)
  2. load      per concurrency: a round-robin mix, streaming and non-streaming alternated. Each
               shared answer must equal its baseline, or fork from it only at a near tie (top-2
               margin <= --tie-margin: cold vs cached prefill rounds differently). Unique requests
               must succeed with the expected prompt length.
  3. cancel    streaming media requests dropped after the first chunk and right after sending
  4. exhaust   a burst of unique image+audio requests larger than the slab: 503 + Retry-After
  5. leaks     /metrics plowrt_mm_slab_rows_reserved and _staged return to 0

--text-only runs only the text requests of the mix (steps 1-2, DIR/load-text.json): the baseline
that media TTFT and throughput are read against.

Writes DIR/load.json (and prints a summary); exits 1 when a check fails.
"""
import argparse, base64, http.client, io, json, random, re, struct, sys, time, urllib.parse, wave, zlib
from concurrent.futures import ThreadPoolExecutor

TEXTS = [
    "Write a short paragraph about the history of the Roman aqueducts.",
    "Explain in three sentences how a refrigerator works.",
    "List four facts about the planet Mars.",
    "What are the main differences between TCP and UDP? Answer briefly.",
]


def png(w, h, seed):
    """A unique RGB PNG: a gradient plus seeded noise (stdlib only)."""
    rng = random.Random(seed)
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            rows += bytes(((x * 255 // w + rng.randrange(32)) & 255, (y * 255 // h) & 255, rng.randrange(256)))
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 1)) + chunk(b"IEND", b""))


def perturbed_wav(path, seed):
    """`path` (16-bit PCM) with seeded low-bit noise: same length and tokens, new content hash."""
    w = wave.open(path)
    params, frames = w.getparams(), bytearray(w.readframes(w.getnframes()))
    rng = random.Random(seed)
    for i in range(0, len(frames), 2):
        frames[i] ^= rng.randrange(2)
    out = io.BytesIO()
    o = wave.open(out, "wb")
    o.setparams(params)
    o.writeframes(bytes(frames))
    o.close()
    return out.getvalue()


def b64(data):
    return base64.b64encode(data).decode()


class Client:
    def __init__(self, base, model):
        u = urllib.parse.urlparse(base)
        self.host, self.port, self.model = u.hostname, u.port or 80, model

    def conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=900)

    def metrics(self):
        c = self.conn()
        c.request("GET", "/metrics")
        text = c.getresponse().read().decode()
        c.close()
        vals = {}
        for line in text.splitlines():
            m = re.match(r"(plowrt_mm_\w+)\{([^}]*)\} (\S+)", line)
            if m:
                kind = re.search(r'kind="(\w+)"', m[2])
                key = m[1] + (":" + kind[1] if kind else "")
                vals[key] = vals.get(key, 0.0) + float(m[3])
        return vals

    def chat(self, messages, max_tokens, stream, cancel=None):
        """{status, text, tokens[(tok, top)], ttft, latency, prompt_tokens, completion_tokens, retry_after}."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "logprobs": True, "top_logprobs": 2}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        t0 = time.perf_counter()
        c = self.conn()
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        if cancel == "sent":
            c.close()
            return {"status": 0, "cancelled": "sent"}
        r = c.getresponse()
        res = {"status": r.status, "retry_after": r.getheader("Retry-After"), "ttft": None, "tokens": [], "text": ""}
        if r.status != 200:
            res["body"] = r.read().decode()[:200]
            res["latency"] = time.perf_counter() - t0
            c.close()
            return res
        if not stream:
            o = json.loads(r.read())
            ch = o["choices"][0]
            res["text"] = ch["message"]["content"] or ""
            res["tokens"] = [(e["token"], [(t["token"], t["logprob"]) for t in e["top_logprobs"]])
                             for e in (ch.get("logprobs") or {}).get("content") or []]
            res.update(prompt_tokens=o["usage"]["prompt_tokens"], completion_tokens=o["usage"]["completion_tokens"])
        else:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: {"):
                    continue
                o = json.loads(line[6:])
                if o.get("usage"):
                    res.update(prompt_tokens=o["usage"]["prompt_tokens"], completion_tokens=o["usage"]["completion_tokens"])
                for ch in o.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta and res["ttft"] is None:
                        res["ttft"] = time.perf_counter() - t0
                    res["text"] += delta or ""
                    res["tokens"] += [(e["token"], [(t["token"], t["logprob"]) for t in e["top_logprobs"]])
                                      for e in (ch.get("logprobs") or {}).get("content") or []]
                if cancel == "first" and res["ttft"] is not None:
                    c.close()
                    res["cancelled"] = "first"
                    return res
        res["latency"] = time.perf_counter() - t0
        c.close()
        return res


def margin(top):
    return top[0][1] - top[1][1] if len(top) > 1 else float("inf")


def agrees(base, got, tie):
    """(same tokens, or the first fork is a near tie in either run)."""
    for x, y in zip(base, got):
        if x[0] != y[0]:
            return min(margin(x[1]), margin(y[1])) <= tie
    return len(base) == len(got)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concs", default="1,8,32")
    ap.add_argument("--requests", default="45,72,144", help="requests per concurrency (c1 streams all, for TTFT)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tie-margin", type=float, default=0.25)
    ap.add_argument("--burst", type=int, default=40, help="exhaustion burst size (unique image+audio requests)")
    ap.add_argument("--text-only", action="store_true", help="the text requests of the mix only (a baseline for the cells)")
    args = ap.parse_args()
    ref = json.load(open(args.ref))
    cl = Client(args.base, args.model)
    items = ref["items"]
    image = next((i for i, it in enumerate(items) if it["kind"] == "image"), None)
    audio = next((i for i, it in enumerate(items) if it["kind"] == "audio"), None)
    has_image = image is not None
    def part(i):
        """Item i's content part as a reference case sent it (same bytes, same ids)."""
        for c in ref["cases"]:
            for k, p in zip(c["media"], [p for p in c["messages"][0]["content"] if p["type"] != "text"]):
                if k == i:
                    return p
        raise SystemExit(f"no reference case sends item {i}")

    user = lambda parts, text: [{"role": "user", "content": parts + [{"type": "text", "text": text}]}]
    shared = {f"text{i}": [{"role": "user", "content": t}] for i, t in enumerate(TEXTS)}
    if has_image:
        shared["image"] = user([part(image)], "Describe this image in one sentence.")
    if audio is not None:
        shared["audio"] = user([part(audio)], "Transcribe this audio.")
    if has_image and audio is not None:
        shared["image_audio"] = user([part(image), part(audio)], "Describe the image, then transcribe the audio.")
    seed = iter(range(10**9))

    def unique(kind):
        parts = []
        if kind in ("uimage", "uimage_audio"):
            parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(png(320, 240, next(seed)))}})
        if kind in ("uaudio", "uimage_audio"):
            parts.append({"type": "input_audio", "input_audio": {"data": b64(perturbed_wav(items[audio]["path"], next(seed))),
                                                                 "format": "wav"}})
        return user(parts, "Describe the image." if kind == "uimage" else "Transcribe this audio." if kind == "uaudio"
                    else "Describe the image, then transcribe the audio.")

    mix = ["text0", "text1", "text2", "text3"] + [k for k in ("image", "audio", "image_audio") if k in shared]
    mix += [k for k, ok in (("uimage", has_image), ("uaudio", audio is not None)) if ok]
    if args.text_only:
        mix = [k for k in mix if k.startswith("text")]
    report = {"model": args.model, "mix": mix, "checks": {}}
    fail = []

    print("== baseline", flush=True)
    base = {k: cl.chat(m, args.max_tokens, stream=False) for k, m in shared.items()}
    for k, r in base.items():
        if r["status"] != 200:
            fail.append(f"baseline {k}: {r['status']} {r.get('body')}")
    expect_prompt = {}
    for k in ("uimage", "uaudio"):
        if k in mix:
            r = cl.chat(unique(k), args.max_tokens, stream=False)
            expect_prompt[k] = r.get("prompt_tokens")
    report["baseline"] = {k: {"text": r["text"][:120], "prompt_tokens": r.get("prompt_tokens")} for k, r in base.items()}

    report["cells"] = []
    for conc, n in zip(map(int, args.concs.split(",")), map(int, args.requests.split(","))):
        before = cl.metrics()
        specs = [(mix[i % len(mix)], conc == 1 or i % 2 == 0) for i in range(n)]
        msgs = [shared[k] if k in shared else unique(k) for k, _ in specs]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            results = list(ex.map(lambda km: cl.chat(km[1], args.max_tokens, stream=km[0][1]), zip(specs, msgs)))
        wall = time.perf_counter() - t0
        after = cl.metrics()
        cell = {"conc": conc, "requests": n, "wall_s": round(wall, 3), "errors": 0, "mismatch": [], "ttft_ms": {}, "latency_ms": {}}
        out_tok = 0
        cats = {}
        for (k, stream), r in zip(specs, results):
            cat = "text" if k.startswith("text") else k
            if r["status"] != 200:
                cell["errors"] += 1
                fail.append(f"c{conc} {k}: {r['status']} {r.get('body')}")
                continue
            out_tok += r.get("completion_tokens") or 0
            cats.setdefault(cat, {"ttft": [], "lat": []})
            if stream and r["ttft"] is not None:
                cats[cat]["ttft"].append(r["ttft"] * 1e3)
            cats[cat]["lat"].append(r["latency"] * 1e3)
            if k in base and not agrees(base[k]["tokens"], r["tokens"], args.tie_margin):
                cell["mismatch"].append(k)
            if k in expect_prompt and r.get("prompt_tokens") != expect_prompt[k]:
                cell["mismatch"].append(f"{k} prompt_tokens {r.get('prompt_tokens')} != {expect_prompt[k]}")
            if k in base and r["text"] != base[k]["text"]:
                cell.setdefault("tie_forks", []).append(k)
        if cell["mismatch"]:
            fail.append(f"c{conc} mismatches: {cell['mismatch']}")
        cell["req_s"] = round(n / wall, 2)
        cell["out_tok_s"] = round(out_tok / wall, 1)
        for cat, v in sorted(cats.items()):
            cell["ttft_ms"][cat] = {"p50": pct(v["ttft"], 0.5), "p90": pct(v["ttft"], 0.9), "n": len(v["ttft"])}
            cell["latency_ms"][cat] = {"p50": pct(v["lat"], 0.5), "p90": pct(v["lat"], 0.9), "n": len(v["lat"])}
        enc = {}
        for kind in ("image", "audio"):
            calls = after.get(f"plowrt_mm_encode_total:{kind}", 0) - before.get(f"plowrt_mm_encode_total:{kind}", 0)
            secs = after.get(f"plowrt_mm_encode_seconds_total:{kind}", 0) - before.get(f"plowrt_mm_encode_seconds_total:{kind}", 0)
            if calls:
                enc[kind] = {"calls": int(calls), "mean_ms": round(secs / calls * 1e3, 2)}
        cell["encoder"] = enc
        report["cells"].append(cell)
        print(json.dumps(cell), flush=True)
    if args.text_only:
        report["fail"] = fail
        json.dump(report, open(f"{args.out}/load-text.json", "w"), indent=1)
        print("load", "pass" if not fail else "fail: " + "; ".join(fail[:5]))
        return 0 if not fail else 1

    def drained(timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            m = cl.metrics()
            if m.get("plowrt_mm_slab_rows_reserved", 0) == 0 and m.get("plowrt_mm_slab_rows_staged", 0) == 0:
                return True, m
            time.sleep(0.5)
        return False, cl.metrics()

    mkind = "uimage_audio" if has_image and audio is not None else "uimage" if has_image else "uaudio"
    print("== cancel", flush=True)
    with ThreadPoolExecutor(8) as ex:
        cancelled = list(ex.map(lambda i: cl.chat(unique(mkind), 256, stream=True, cancel="first" if i % 2 else "sent"), range(8)))
    ok, m = drained()
    report["checks"]["cancel"] = {"drained": ok, "cancelled": [r.get("cancelled") for r in cancelled],
                                  "reserved": m.get("plowrt_mm_slab_rows_reserved"), "staged": m.get("plowrt_mm_slab_rows_staged")}
    if not ok:
        fail.append(f"cancel: slab not drained {report['checks']['cancel']}")

    print("== exhaust", flush=True)
    full_before = cl.metrics().get("plowrt_mm_slab_full_total", 0)
    with ThreadPoolExecutor(args.burst) as ex:
        burst = list(ex.map(lambda i: cl.chat(unique(mkind), 256, stream=False), range(args.burst)))
    codes = [r["status"] for r in burst]
    refused = [r for r in burst if r["status"] == 503]
    ok_drain, m = drained()
    report["checks"]["exhaust"] = {
        "ok": codes.count(200), "refused_503": len(refused), "other": [c for c in codes if c not in (200, 503)],
        "retry_after": sorted({r.get("retry_after") for r in refused}), "slab_rows": m.get("plowrt_mm_slab_rows"),
        "slab_full_total_delta": m.get("plowrt_mm_slab_full_total", 0) - full_before, "drained": ok_drain,
        "body": refused[0].get("body") if refused else None}
    if any(c not in (200, 503) for c in codes) or any(r.get("retry_after") is None for r in refused) or not ok_drain:
        fail.append(f"exhaust: {report['checks']['exhaust']}")
    report["checks"]["leaks"] = {"reserved": m.get("plowrt_mm_slab_rows_reserved"), "staged": m.get("plowrt_mm_slab_rows_staged")}
    report["fail"] = fail
    json.dump(report, open(f"{args.out}/load.json", "w"), indent=1)
    print(json.dumps(report["checks"]))
    print("load", "pass" if not fail else "fail: " + "; ".join(fail[:5]))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
