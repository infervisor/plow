#!/usr/bin/env python3
"""Realistic-workload latency/throughput benchmark for OpenAI-compatible servers.

Streams `POST /v1/chat/completions` (stream=true, temperature=0) with prompts
from corpus.py, records per-request TTFT / TPOT / latency, and aggregates per
(workload, concurrency). Stdlib only; `tokenizers` is optional for client-side
prompt token counts.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus  # noqa: E402

TEXT_HEAD = 80
WARMUP_INDEX_BASE = 10_000_000


# --------------------------------------------------------------------------
# Tokenizer (optional)
# --------------------------------------------------------------------------
def load_tokenizer(path):
    if not path:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        print("note: `tokenizers` not importable (%s); prompt tokens fall back to server usage" % e, file=sys.stderr)
        return None
    f = path if os.path.isfile(path) else os.path.join(path, "tokenizer.json")
    tok = Tokenizer.from_file(f)
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


# --------------------------------------------------------------------------
# Minimal streaming HTTP/1.1 client on asyncio streams (no aiohttp)
# --------------------------------------------------------------------------
class HttpError(Exception):
    pass


async def _read_headers(reader):
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ", 2)[1])
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return status, headers


async def _body_chunks(reader, headers):
    """Yield raw body bytes as they arrive (dechunked if chunked)."""
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            line = await reader.readline()
            size = int(line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                # trailers until blank line
                while (await reader.readline()).strip():
                    pass
                return
            data = await reader.readexactly(size)
            await reader.readexactly(2)  # CRLF
            yield data
    elif "content-length" in headers:
        remaining = int(headers["content-length"])
        while remaining > 0:
            data = await reader.read(min(65536, remaining))
            if not data:
                return
            remaining -= len(data)
            yield data
    else:
        while True:
            data = await reader.read(65536)
            if not data:
                return
            yield data


def _delta_is_token(delta):
    # vLLM's first frame is role-only with content "" (not a token). plowrt's
    # role rides the first real token. Empty-content frames without a role are
    # plowrt partial-UTF-8 tokens and count.
    if "content" not in delta or delta["content"] is None:
        return False
    return delta["content"] != "" or "role" not in delta


async def stream_chat(url, body, timeout):
    """One streaming chat completion. Returns a partial record dict."""
    u = urllib.parse.urlsplit(url)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    payload = json.dumps(body).encode()
    req = (
        "POST %s HTTP/1.1\r\nHost: %s:%d\r\nContent-Type: application/json\r\n"
        "Accept: text/event-stream\r\nContent-Length: %d\r\nConnection: close\r\n\r\n"
        % (u.path, host, port, len(payload))
    ).encode() + payload

    rec = {
        "status": None, "error": None, "ttft_s": None, "tpot_s": None, "latency_s": None,
        "output_tokens": None, "content_chunks": 0, "finish_reason": None,
        "usage": None, "text_head": "",
    }
    t0 = time.perf_counter()
    reader = writer = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=(u.scheme == "https")), timeout)
        writer.write(req)
        await writer.drain()

        async def run():
            status, headers = await _read_headers(reader)
            rec["status"] = status
            if status != 200:
                buf = b""
                async for data in _body_chunks(reader, headers):
                    buf += data
                    if len(buf) > 4096:
                        break
                raise HttpError("HTTP %d: %s" % (status, buf[:400].decode("utf-8", "replace")))
            buf = b""
            times = []
            text = []
            done = False
            async for data in _body_chunks(reader, headers):
                t_arr = time.perf_counter()
                buf += data
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line, buf = buf[:nl].rstrip(b"\r"), buf[nl + 1:]
                    if not line.startswith(b"data:"):
                        continue
                    p = line[5:].strip()
                    if p == b"[DONE]":
                        done = True
                        break
                    try:
                        ev = json.loads(p)
                    except ValueError:
                        continue
                    if ev.get("usage"):
                        rec["usage"] = ev["usage"]
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        if _delta_is_token(d):
                            times.append(t_arr)
                            if d["content"]:
                                text.append(d["content"])
                        if ch.get("finish_reason"):
                            rec["finish_reason"] = ch["finish_reason"]
                if done:
                    break
            t_end = time.perf_counter()
            rec["latency_s"] = t_end - t0
            rec["content_chunks"] = len(times)
            full = "".join(text)
            rec["text_head"] = full[:TEXT_HEAD]
            rec["text_chars"] = len(full)
            n_out = None
            if rec["usage"] and rec["usage"].get("completion_tokens") is not None:
                n_out = rec["usage"]["completion_tokens"]
            if n_out is None:
                n_out = len(times)
            rec["output_tokens"] = n_out
            if times:
                rec["ttft_s"] = times[0] - t0
                if n_out > 1 and len(times) > 1:
                    rec["tpot_s"] = (times[-1] - times[0]) / (n_out - 1)
            elif not rec["error"]:
                rec["error"] = "no tokens streamed"

        await asyncio.wait_for(run(), timeout)
    except HttpError as e:
        rec["error"] = str(e)
    except asyncio.TimeoutError:
        rec["error"] = "timeout after %.0fs" % timeout
    except (OSError, asyncio.IncompleteReadError, ValueError) as e:
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        if rec["latency_s"] is None:
            rec["latency_s"] = time.perf_counter() - t0
        if writer is not None:
            writer.close()
    return rec


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def build_body(args, messages):
    body = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        body["ignore_eos"] = True
    if args.extra_body:
        body.update(json.loads(args.extra_body))
    return body


async def do_request(args, url, tok_count, workload, index, sem=None):
    req = corpus.make_request(workload, args.seed, index)
    body = build_body(args, req["messages"])
    content = "\n".join(m["content"] for m in req["messages"])
    t_start = time.time()
    rec = await stream_chat(url, body, args.timeout)
    rec.update({
        "index": index, "kind": req["kind"], "t_start": t_start,
        "prompt_tokens_client": tok_count(content) if tok_count else None,
        "prompt_tokens_server": (rec["usage"] or {}).get("prompt_tokens"),
        "prompt_chars": len(content),
    })
    rec["prompt_tokens"] = rec["prompt_tokens_client"] if tok_count else rec["prompt_tokens_server"]
    return rec


async def run_closed_loop(args, url, tok_count, workload, n, concurrency, index_base=0, progress=True):
    indices = list(range(index_base, index_base + n))
    results = []
    lock_i = [0]

    async def worker():
        while True:
            i = lock_i[0]
            if i >= len(indices):
                return
            lock_i[0] = i + 1
            rec = await do_request(args, url, tok_count, workload, indices[i])
            results.append(rec)
            if progress and args.verbose:
                print_rec(rec)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(min(concurrency, n))])
    return results, time.perf_counter() - t0


async def run_open_loop(args, url, tok_count, workload, n, rps, index_base=0):
    rng = random.Random(args.seed * 7919 + 1)
    results = []
    tasks = []

    async def one(i):
        rec = await do_request(args, url, tok_count, workload, i)
        results.append(rec)
        if args.verbose:
            print_rec(rec)

    t0 = time.perf_counter()
    next_t = t0
    for i in range(index_base, index_base + n):
        delay = next_t - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.ensure_future(one(i)))
        next_t += rng.expovariate(rps)
    await asyncio.gather(*tasks)
    return results, time.perf_counter() - t0


def print_rec(rec):
    err = (" ERR " + rec["error"]) if rec["error"] else ""
    print("  #%d %-10s in=%s out=%s ttft=%s tpot=%s lat=%.2fs fin=%s%s | %r" % (
        rec["index"], rec["kind"], rec["prompt_tokens"], rec["output_tokens"],
        fmt_ms(rec["ttft_s"]), fmt_ms(rec["tpot_s"]), rec["latency_s"], rec["finish_reason"], err,
        rec["text_head"]), file=sys.stderr)


# --------------------------------------------------------------------------
# Aggregation / reporting
# --------------------------------------------------------------------------
def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "p50": None, "p90": None, "n": 0}
    return {"mean": statistics.fmean(xs), "p50": pct(xs, 50), "p90": pct(xs, 90), "n": len(xs)}


def aggregate(recs, wall_s):
    ok = [r for r in recs if not r["error"]]
    out_tokens = sum(r["output_tokens"] or 0 for r in ok)
    return {
        "n": len(recs),
        "ok": len(ok),
        "errors": len(recs) - len(ok),
        "wall_s": wall_s,
        "ttft_s": stats([r["ttft_s"] for r in ok]),
        "tpot_s": stats([r["tpot_s"] for r in ok]),
        "latency_s": stats([r["latency_s"] for r in ok]),
        "prompt_tokens": stats([r["prompt_tokens"] for r in ok]),
        "output_tokens": stats([r["output_tokens"] for r in ok]),
        "output_tokens_total": out_tokens,
        "output_tok_per_s": out_tokens / wall_s if wall_s else None,
        "req_per_s": len(ok) / wall_s if wall_s else None,
        "finish_reasons": count_by(ok, "finish_reason"),
    }


def count_by(recs, key):
    d = {}
    for r in recs:
        d[str(r.get(key))] = d.get(str(r.get(key)), 0) + 1
    return d


def fmt_ms(x):
    return "-" if x is None else "%.0f" % (x * 1000)


def fmt(x, nd=1):
    return "-" if x is None else ("%%.%df" % nd) % x


def md_table(runs):
    hdr = ("| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms "
           "| latency p50/p90 s | out tok/s | req/s |")
    sep = "|" + "---|" * 11
    rows = [hdr, sep]
    for r in runs:
        a = r["agg"]
        rows.append("| %s | %s | %d | %d | %s | %s | %s/%s/%s | %s/%s/%s | %s/%s | %s | %s |" % (
            r["workload"], r["concurrency"], a["n"], a["errors"],
            fmt(a["prompt_tokens"]["mean"], 0), fmt(a["output_tokens"]["mean"], 1),
            fmt_ms(a["ttft_s"]["mean"]), fmt_ms(a["ttft_s"]["p50"]), fmt_ms(a["ttft_s"]["p90"]),
            fmt_ms(a["tpot_s"]["mean"]), fmt_ms(a["tpot_s"]["p50"]), fmt_ms(a["tpot_s"]["p90"]),
            fmt(a["latency_s"]["p50"], 2), fmt(a["latency_s"]["p90"], 2),
            fmt(a["output_tok_per_s"], 1), fmt(a["req_per_s"], 2)))
    return "\n".join(rows)


def md_report(result):
    m = result["meta"]
    out = ["# bench-api results", "",
           "- server: `%s`  model: `%s`" % (m["base_url"], m["model"]),
           "- time: %s  seed: %d  max_tokens: %d  warmup: %d" % (m["timestamp"], m["seed"], m["max_tokens"], m["warmup"]),
           "- prompt tokens counted by: %s" % m["prompt_token_source"],
           "- server models: %s" % ", ".join(m.get("server_models") or []) or "-",
           "", md_table(result["runs"]), ""]
    for r in result["runs"]:
        if r["workload"] == "mixed" and r.get("by_kind"):
            out.append("### mixed @ %s by kind" % r["concurrency"])
            out.append("")
            out.append("| kind | n | err | in tok | TTFT p50 ms | TPOT p50 ms | latency p50 s |")
            out.append("|---|---|---|---|---|---|---|")
            for k, a in sorted(r["by_kind"].items()):
                out.append("| %s | %d | %d | %s | %s | %s | %s |" % (
                    k, a["n"], a["errors"], fmt(a["prompt_tokens"]["mean"], 0),
                    fmt_ms(a["ttft_s"]["p50"]), fmt_ms(a["tpot_s"]["p50"]), fmt(a["latency_s"]["p50"], 2)))
            out.append("")
    out.append("## Samples")
    out.append("")
    for r in result["runs"]:
        out.append("**%s @ %s**" % (r["workload"], r["concurrency"]))
        out.append("")
        for rec in r["requests"][: m["samples"]]:
            out.append("- #%d %s in=%s out=%s fin=%s%s: `%s`" % (
                rec["index"], rec["kind"], rec["prompt_tokens"], rec["output_tokens"], rec["finish_reason"],
                (" ERR " + rec["error"]) if rec["error"] else "",
                rec["text_head"].replace("`", "'").replace("\n", " ")))
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
def fetch_models(base):
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except Exception as e:  # noqa: BLE001
        print("warning: GET /v1/models failed: %s" % e, file=sys.stderr)
        return None


def parse_arrival(s):
    if not s or s == "closed":
        return None
    kind, _, val = s.partition(":")
    if kind != "poisson" or not val:
        raise SystemExit("--arrival must be `closed` or `poisson:RPS`")
    return float(val)


def dry_run(args, tok_count):
    for w in args.workload:
        rows = []
        for i in range(args.requests):
            req = corpus.make_request(w, args.seed, i)
            content = "\n".join(m["content"] for m in req["messages"])
            rows.append((tok_count(content) if tok_count else corpus.est_tokens(content), req["kind"]))
        xs = sorted(t for t, _ in rows)
        print("%-11s n=%d prompt tokens min/p50/p90/max = %d/%d/%d/%d (%s)" % (
            w, len(xs), xs[0], pct(xs, 50), pct(xs, 90), xs[-1], "tokenizer" if tok_count else "estimate"))
        if args.verbose:
            for i, (t, k) in enumerate(rows):
                req = corpus.make_request(w, args.seed, i)
                print("--- #%d %s (%d tok)\n%s\n" % (i, k, t, req["messages"][-1]["content"][:600]))


async def amain(args):
    base = args.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = base + "/v1/chat/completions"
    tok_count = load_tokenizer(args.tokenizer)
    if args.dry_run:
        dry_run(args, tok_count)
        return 0
    models = fetch_models(base)
    if models is not None and args.model not in models:
        print("warning: model %r not in server list %s" % (args.model, models), file=sys.stderr)

    rps = parse_arrival(args.arrival)
    concs = [int(c) for c in args.concurrency.split(",")] if rps is None else [None]
    result = {
        "meta": {
            "base_url": base, "model": args.model, "server_models": models,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed, "max_tokens": args.max_tokens, "requests": args.requests,
            "warmup": args.warmup, "arrival": args.arrival, "ignore_eos": args.ignore_eos,
            "prompt_token_source": "tokenizer" if tok_count else "server usage",
            "samples": args.samples, "argv": sys.argv[1:], "fresh_prompts": args.fresh_prompts,
        },
        "runs": [],
    }
    for w in args.workload:
        for c in concs:
            label = c if rps is None else "poisson:%g" % rps
            print("== %s @ %s (%d requests)" % (w, label, args.requests), file=sys.stderr)
            if args.warmup:
                await run_closed_loop(args, url, tok_count, w, args.warmup, min(c or 4, args.warmup),
                                      index_base=WARMUP_INDEX_BASE, progress=False)
            # --fresh-prompts: every (workload, concurrency) cell gets prompts no earlier cell
            # used, so servers with a prefix/prompt cache (llama-server slots, vLLM APC) prefill
            # for real instead of replaying the c=1 cell's KV.
            base = concs.index(c) * args.requests if (args.fresh_prompts and rps is None) else 0
            if rps is None:
                recs, wall = await run_closed_loop(args, url, tok_count, w, args.requests, c, index_base=base)
            else:
                recs, wall = await run_open_loop(args, url, tok_count, w, args.requests, rps)
            recs.sort(key=lambda r: r["index"])
            agg = aggregate(recs, wall)
            by_kind = {}
            for k in sorted({r["kind"] for r in recs}):
                by_kind[k] = aggregate([r for r in recs if r["kind"] == k], wall)
            run = {"workload": w, "concurrency": label, "agg": agg, "by_kind": by_kind, "requests": recs}
            result["runs"].append(run)
            print("   ttft p50 %s ms | tpot p50 %s ms | %s out tok/s | %s req/s | %d err" % (
                fmt_ms(agg["ttft_s"]["p50"]), fmt_ms(agg["tpot_s"]["p50"]), fmt(agg["output_tok_per_s"]),
                fmt(agg["req_per_s"], 2), agg["errors"]), file=sys.stderr)
            for rec in recs[: args.samples]:
                print("   #%d %s: %r%s" % (rec["index"], rec["kind"], rec["text_head"],
                                           (" ERR " + rec["error"]) if rec["error"] else ""), file=sys.stderr)
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(result, f, indent=1)
            if args.md:
                with open(args.md, "w") as f:
                    f.write(md_report(result))
    print(md_table(result["runs"]))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:8093")
    ap.add_argument("--model", required=True, help="model id as listed by /v1/models")
    ap.add_argument("--workload", required=True,
                    help="comma-separated subset of: " + ",".join(corpus.WORKLOADS))
    ap.add_argument("--concurrency", default="1", help="comma-separated closed-loop worker counts, e.g. 1,2,4,8")
    ap.add_argument("--arrival", default="closed", help="`closed` (default) or `poisson:RPS` open-loop mode")
    ap.add_argument("--requests", type=int, default=16, help="requests per (workload, concurrency) cell")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tokenizer", help="dir containing tokenizer.json (or the file); needs `tokenizers`")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fresh-prompts", action="store_true",
                    help="distinct prompts per concurrency cell (defeats server prefix caches)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed requests before each cell")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-request timeout (s)")
    ap.add_argument("--ignore-eos", action="store_true", help="send ignore_eos=true (vLLM/plowrt extension)")
    ap.add_argument("--extra-body", help="JSON merged into every request body")
    ap.add_argument("--samples", type=int, default=2, help="sample outputs to print/report per cell")
    ap.add_argument("--out", help="results JSON path")
    ap.add_argument("--md", help="markdown report path")
    ap.add_argument("--dry-run", action="store_true", help="only print prompt-length stats (no server needed)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every request as it completes")
    args = ap.parse_args()
    args.workload = [w.strip() for w in args.workload.split(",")]
    bad = [w for w in args.workload if w not in corpus.WORKLOADS]
    if bad:
        ap.error("unknown workload(s) %s" % bad)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
