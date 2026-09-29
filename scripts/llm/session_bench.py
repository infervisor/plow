#!/usr/bin/env python3
"""Voice-agent multi-turn chat load: N concurrent calls, each a session of T turns.

session_bench.py --port P --model M --tokenizer HF_DIR [--calls 64] [--turns 6] [--max-tokens 48]
                 [--think 0.5] [--no-session] [--out r.json]

Every call shares one ~350-token system prompt; turn t sends the whole conversation so far (the
server's previous answers verbatim) plus a new ~40-60 token user utterance, streamed, with
`X-Session-Id: <call>` (omitted with --no-session). Between turns the caller "speaks" for --think
seconds. ignore_eos + fixed max_tokens keep the work identical across servers. Reports TTFT
p50/p90 for the first turn and for later turns, TPOT, total output tok/s and the prompt tokens the
server reported as cached (usage.prompt_tokens_details.cached_tokens or X-Session-Cached-Tokens).
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

SYSTEM = ("You are Ava, the phone receptionist of Bright Smile Dental. You speak in short, friendly, "
          "natural sentences because your words are read aloud by a speech synthesizer. Never use lists, "
          "markdown or emojis. The clinic is open Monday to Friday from 8am to 6pm and Saturday from 9am "
          "to 1pm. Services: cleanings, fillings, crowns, whitening, emergency visits and pediatric care. "
          "New patients need a photo ID and their insurance card. Cancellations require 24 hours notice. "
          "If a caller reports severe pain, swelling or bleeding, offer the earliest emergency slot. "
          "Always confirm the caller's name, date of birth and phone number before booking, and repeat "
          "the appointment time back to them. If you do not know something, say you will have the office "
          "manager call back. ") * 2
UTTER = [
    "Hi, I'm calling because I have a pretty bad toothache on the lower left side since yesterday",
    "My name is Jordan Miller, date of birth March fourth nineteen eighty nine",
    "Do you have anything open tomorrow morning, ideally before ten because I work at eleven",
    "Is that covered if I have Delta Dental through my employer, I am not sure about the plan",
    "Okay and how long does a cleaning usually take, and do I need to bring anything",
    "Can you also book my daughter, she is seven, for a checkup the same day if possible",
    "Actually wait, could we move mine to Thursday afternoon instead, something came up",
    "Great, and what is the address again, I have not been to the new location yet",
]


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else float("nan")


def turn(port, model, messages, session, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json"}
    if session:
        h["X-Session-Id"] = session
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), h)
    t0 = time.perf_counter()
    ttft, stamps, text, usage = None, [], [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        hdr_cached = int(r.headers.get("X-Session-Cached-Tokens") or 0)
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece is not None:
                    now = time.perf_counter()
                    ttft = ttft if ttft is not None else now - t0
                    stamps.append(now)
                    text.append(piece)
    tpot = (stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else float("nan")
    cached = max(((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0, hdr_cached)
    return {"ttft": ttft, "tpot": tpot, "text": "".join(text), "out": len(stamps),
            "prompt": (usage or {}).get("prompt_tokens", 0), "cached": cached or 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--calls", type=int, default=64)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--think", type=float, default=0.5)
    ap.add_argument("--no-session", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results, lock = [], threading.Lock()

    def call(k):
        rng = random.Random(k)
        time.sleep(rng.random() * 2.0)
        msgs = [{"role": "system", "content": SYSTEM}]
        sess = None if a.no_session else f"call-{k}-{int(time.time())}"
        for t in range(a.turns):
            msgs.append({"role": "user", "content": UTTER[(k + t) % len(UTTER)] + f" (call {k}, turn {t})"})
            r = turn(a.port, a.model, msgs, sess, a.max_tokens)
            r.update(call=k, turn=t)
            with lock:
                results.append(r)
            msgs.append({"role": "assistant", "content": r["text"]})
            time.sleep(a.think * (0.5 + rng.random()))

    t0 = time.perf_counter()
    th = [threading.Thread(target=call, args=(k,)) for k in range(a.calls)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    wall = time.perf_counter() - t0
    first = [r["ttft"] * 1e3 for r in results if r["turn"] == 0]
    later = [r["ttft"] * 1e3 for r in results if r["turn"] > 0]
    tpot = [r["tpot"] * 1e3 for r in results if r["tpot"] == r["tpot"]]
    later_prompt = sum(r["prompt"] for r in results if r["turn"] > 0)
    later_cached = sum(r["cached"] for r in results if r["turn"] > 0)
    summary = {
        "calls": a.calls, "turns": a.turns, "session": not a.no_session, "wall_s": wall,
        "ttft_first_p50": pct(first, 50), "ttft_first_p90": pct(first, 90),
        "ttft_later_p50": pct(later, 50), "ttft_later_p90": pct(later, 90), "ttft_later_p99": pct(later, 99),
        "tpot_p50": pct(tpot, 50), "tpot_p90": pct(tpot, 90),
        "out_tok_s": sum(r["out"] for r in results) / wall,
        "mean_prompt_later": later_prompt / max(1, len(later)),
        "cached_frac_later": later_cached / max(1, later_prompt),
    }
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items()}))
    if a.out:
        json.dump({"summary": summary, "turns": results}, open(a.out, "w"))


if __name__ == "__main__":
    main()
