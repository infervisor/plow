#!/usr/bin/env python3
"""Thinking-control stress report: leaks, cut-offs and first-answer latency per arm.

The comparison the thinking-control plan's S0-S4 ladder and the bridge-wording A/B (7.2) score:
the same conversations against one or more OpenAI-compatible endpoints (plowrt, the vLLM
reference) and request variants, every reply checked. Columns: strict leak flags (markers, a
`thought` label, planning-shaped lines), judge leaks (optional judge model), `finish=length` %,
longest reply, p50/p90 time to the first answer text (streamed), forced % (plow's
`x_plow_thinking`) and answer-reasoning % (replies opening with Wait/Alternatively/Hmm/Actually).
Every leak and every cut-off reply is printed in the report.

  thinking_stress.py --endpoint plow=http://127.0.0.1:8005 --endpoint vllm=http://node03:8005 \\
      --model gemma26b-thinking --prompts convs.jsonl --arm b160='{"thinking_token_budget":160}' \\
      --concurrency 8 --repeats 20 --out <scratch dir>

`--prompts`: JSONL, one `{"messages": [...]}` (plus any request fields) per line. Raw results go to
`<out>/results.jsonl`, the table to `<out>/report.md`; keep both in campaign scratch.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import statistics
import time
import urllib.request

DEFAULT_PROMPTS = [
    [{"role": "user", "content": "हेलो, कौन बोल रहा है?"}],
    [{"role": "user", "content": "আমি রাহুল বলছি, আপনি কে?"}],
    [{"role": "user", "content": "வணக்கம், யார் பேசுகிறீர்கள்?"}],
    [{"role": "user", "content": "Hello, who is this?"}],
]

MARKERS = ("<|channel", "<channel|>", "<think>", "</think>", "<|channel|>")
OPENERS = (
    "wait", "hmm", "alternatively", "self-correction", "the user", "the caller", "the prompt",
    "the instruction", "the system", "the most logical", "the best response", "the best reply",
    "let me", "let's", "i need to", "i should", "my response", "my reply",
)
ANSWER_REASONING = ("wait", "alternatively", "hmm", "actually")


def planning_line(line):
    """The reply guard's line shapes (`serve::reply_guard::planning`, complete lines)."""
    t = line.strip()
    if not t:
        return False
    if t[0] in "*([#":
        return True
    low = t.lower()
    for o in OPENERS:
        if low.startswith(o) and (len(low) == len(o) or not (low[len(o)].isalnum() or low[len(o)] in "-'")):
            return True
    m = re.match(r"([A-Za-z ]{1,32})([:.])", t)
    if m:
        words = m.group(1).split()
        if m.group(2) == ":" and len(words) <= 4:
            return True
        if m.group(2) == "." and len(words) <= 2 and (words[-1].lower() == "detected" or " ".join(words).lower() == "none"):
            return True
    return False


def strict_flags(content):
    flags = [m for m in MARKERS if m in content]
    if re.match(r"\s*thought\b", content):
        flags.append("thought-label")
    flags += [f"planning:{ln.strip()[:40]}" for ln in content.splitlines() if planning_line(ln)]
    return flags


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def one(endpoint, url, arm, overrides, model, conv, stream, timeout, rep, idx):
    body = {"model": model, **({"messages": conv} if isinstance(conv, list) else conv), **overrides}
    body["stream"] = stream
    t0 = time.monotonic()
    rec = {"endpoint": endpoint, "arm": arm, "rep": rep, "prompt": idx, "error": None}
    try:
        with post(url, body, timeout) as resp:
            if stream:
                content, reasoning, finish, thinking, first = "", "", None, None, None
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    f = json.loads(line[6:])
                    if "error" in f:
                        raise RuntimeError(f["error"])
                    for c in f.get("choices", []):
                        d = c.get("delta", {})
                        piece = d.get("content") or ""
                        if piece.strip() and first is None:
                            first = time.monotonic() - t0
                        content += piece
                        reasoning += d.get("reasoning") or d.get("reasoning_content") or ""
                        finish = c.get("finish_reason") or finish
                        thinking = c.get("x_plow_thinking") or thinking
            else:
                v = json.load(resp)
                c = v["choices"][0]
                content = c["message"].get("content") or ""
                reasoning = c["message"].get("reasoning") or c["message"].get("reasoning_content") or ""
                finish, thinking, first = c.get("finish_reason"), c.get("x_plow_thinking"), None
    except Exception as e:  # recorded per request; the report counts them
        rec["error"] = str(e)
        return rec
    rec.update(content=content, reasoning=reasoning, finish=finish, thinking=thinking,
               first_answer_s=first, total_s=time.monotonic() - t0, flags=strict_flags(content))
    return rec


JUDGE = ("You check replies of a phone voice agent. Answer LEAK if the reply contains the agent's "
         "internal reasoning, planning, notes about the prompt or the caller, or anything not meant "
         "to be spoken to the caller; otherwise answer OK. Reply:\n\n{reply}\n\nAnswer LEAK or OK.")


def judge(url, model, content, timeout):
    body = {"model": model, "temperature": 0, "max_tokens": 4,
            "messages": [{"role": "user", "content": JUDGE.format(reply=content)}]}
    with post(url, body, timeout) as resp:
        v = json.load(resp)
    return "LEAK" in (v["choices"][0]["message"].get("content") or "").upper()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def report(rows):
    out = ["| endpoint | arm | n | errors | strict leaks | judge leaks | finish=length % | longest reply |"
           " p50 first answer s | p90 first answer s | forced % | answer-reasoning % |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for key in sorted({(r["endpoint"], r["arm"]) for r in rows}):
        rs = [r for r in rows if (r["endpoint"], r["arm"]) == key]
        ok = [r for r in rs if not r["error"]]
        n = len(ok) or 1
        firsts = [r["first_answer_s"] for r in ok if r["first_answer_s"] is not None]
        leaks = [r for r in ok if r["flags"]]
        jl = [r for r in ok if r.get("judge_leak")]
        length = [r for r in ok if r["finish"] == "length"]
        forced = [r for r in ok if (r.get("thinking") or {}).get("forced")]
        ar = [r for r in ok if (re.findall(r"\w+", r["content"]) or [""])[0].lower() in ANSWER_REASONING]
        longest = max((len(r["content"]) for r in ok), default=0)
        fmt = lambda x: "-" if x is None else f"{x:.2f}"
        judged = any("judge_leak" in r for r in ok)
        out.append(f"| {key[0]} | {key[1]} | {len(ok)} | {len(rs) - len(ok)} | {len(leaks)} | "
                   f"{len(jl) if judged else '-'} | {100 * len(length) / n:.2f} | {longest} | "
                   f"{fmt(pct(firsts, 0.5))} | {fmt(pct(firsts, 0.9))} | {100 * len(forced) / n:.1f} | "
                   f"{100 * len(ar) / n:.1f} |")
        for r in leaks + jl + length:
            detail.append(f"- {key[0]}/{key[1]} prompt {r['prompt']} rep {r['rep']} "
                          f"finish={r['finish']} flags={r['flags']} judge={r.get('judge_leak')}: "
                          f"{json.dumps(r['content'], ensure_ascii=False)}")
    return "\n".join(out) + "\n\n## Leaks and cut-off replies\n\n" + ("\n".join(dict.fromkeys(detail)) or "none") + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", action="append", required=True, help="name=base-url")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", help="JSONL conversations (default: four greetings)")
    ap.add_argument("--arm", action="append", default=[], help="name=JSON request overrides")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--judge-url")
    ap.add_argument("--judge-model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    convs = DEFAULT_PROMPTS
    if a.prompts:
        convs = [json.loads(l) for l in open(a.prompts) if l.strip()]
    arms = [(n, json.loads(j)) for n, j in (x.split("=", 1) for x in a.arm)] or [("default", {})]
    endpoints = [x.split("=", 1) for x in a.endpoint]
    jobs = [(e, u, n, o, a.model, c, not a.no_stream, a.timeout, rep, i)
            for e, u in endpoints for n, o in arms for rep in range(a.repeats) for i, c in enumerate(convs)]
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(lambda j: one(*j), jobs))
    if a.judge_url:
        for r in rows:
            if not r["error"] and r["content"].strip():
                r["judge_leak"] = judge(a.judge_url, a.judge_model or a.model, r["content"], a.timeout)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    text = report(rows)
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)
    return 1 if any(r["error"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
