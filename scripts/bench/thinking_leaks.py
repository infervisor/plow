#!/usr/bin/env python3
"""Thinking/answer separation and leak audit for an OpenAI-compatible reasoning server.

Per prompt and budget arm, greedy, it checks:
  markers      no reasoning markers (<think>, </think>, <|channel>, <channel|>) in `content`
               or in the reasoning text
  separation   the split equals the raw generation: reasoning = the text inside the think block(s),
               content = the text outside them (raw from `--raw format`, i.e. llama.cpp's/plow's
               `reasoning_format: "none"`, or `--raw URL`, the same model served without a parser)
  overlap      no 60+ character run of the trace repeated in the answer; no planning-shaped answer
               line (the reply guard's shapes)
  stream       the streamed reasoning and content equal the buffered ones
  answer       a turn that finished `stop` with thinking on has a non-empty answer
  budgets      completion_tokens <= max_tokens; with a thinking budget the trace stays within it
               (plus the open marker, the newline grace and the forced end); with max_reply_tokens
               the answer stays within it
and records how the server treats a thinking budget that leaves the reply no room.

  thinking_leaks.py --url http://127.0.0.1:8000 --name plow --model qwen3-0.6b --raw format --out <scratch>
  thinking_leaks.py --url http://127.0.0.1:8001 --name vllm --model qwen3-0.6b --raw http://127.0.0.1:8002 ...
  thinking_leaks.py --url <parser-less server> --name raw --model M --record-raw raw.jsonl --out <scratch>, then
  --raw raw.jsonl against the parsing server (one GPU fits one of them at a time)

Writes <out>/<name>.jsonl and <out>/<name>.md; exit 1 when any check fails.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

# A line of the model deliberating with itself: the trace continuing inside the answer. Markdown,
# labels and lists are answers, not leaks (the reply guard's voice rules are stricter on purpose).
DELIBERATION = re.compile(
    r"^\s*(wait\b|hmm+\b|alternatively\b|on second thought|let me (think|check|see|verify|double|re-?check|make sure|recall|outline)"
    r"|let's (think|check|see|verify|double|test)|i (need|should|think|must|will|have) |okay,? so\b|so,? the user\b"
    r"|the user (is|was|wants|asked|might|wrote|says|mentioned)|maybe (i|they|the user)\b|actually,? (wait|let me|no)\b"
    r"|double-check)", re.I)

MARKERS = ("<think>", "</think>", "<|channel>", "<channel|>", "<|channel|>")
OPEN, CLOSE = "<think>", "</think>"
PROMPTS = [
    "What is 17 * 23?",
    "Is 221 a prime number? Answer yes or no with one reason.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Write an HTML snippet with a <div> and a <span> inside it.",
    "Explain in one sentence what the <think> tag is used for in reasoning models.",
    "Repeat exactly: the answer ends with </think> and nothing else.",
    "If I write <think>hello</think>, what text is between the tags?",
    "Translate 'good morning, how are you?' into Hindi.",
    "मेरा पार्सल कब आएगा? एक वाक्य में जवाब दो।",
    "আপনি কে বলছেন? এক লাইনে উত্তর দিন।",
    "Hello, who is this?",
    "Give me three tips for sleeping better, as a short list.",
    "What is the capital of Australia? One word.",
    "Sort these numbers: 9, 2, 7, 4, 1.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Think step by step: how many vowels are in the word 'onomatopoeia'?",
    "Wait, are you sure? Re-check: what is 12 squared?",
    "Respond only with JSON: {\"ok\": true} and nothing else.",
    "Count from 1 to 5, one number per line.",
]
ARMS = [
    ("nobudget", {"max_tokens": 2048}),
    # Both budget names: vLLM reads thinking_token_budget, llama.cpp reasoning_budget_tokens.
    ("b32.m256", {"thinking_token_budget": 32, "reasoning_budget_tokens": 32, "max_tokens": 256}),
    ("b128.m512", {"thinking_token_budget": 128, "reasoning_budget_tokens": 128, "max_tokens": 512}),
    ("b32.m256.r48", {"thinking_token_budget": 32, "reasoning_budget_tokens": 32, "max_tokens": 256,
                      "max_reply_tokens": 48}),
    ("off.m256", {"max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}),
]
RAW = {}
# Trace tokens allowed beyond the budget: open marker + label/newline, the newline grace, the
# forced end (a bare close marker unless the server's reasoning_end is longer).
SLACK = 2 + 8 + 40


def post(url, body, timeout=600):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)
    if not body.get("stream"):
        return r.status, json.load(r)
    frames = [json.loads(l[6:]) for l in (x.decode().strip() for x in r) if l.startswith("data: {")]
    return r.status, frames


def trace_of(m):
    return m.get("reasoning") or m.get("reasoning_content") or ""


def split_raw(raw):
    """(reasoning, content) of a raw generation: the text inside every think block, and outside."""
    inside, outside, rest = [], [], raw
    while True:
        o = rest.find(OPEN)
        c = rest.find(CLOSE)
        if o < 0 and c < 0:
            outside.append(rest)
            break
        if o >= 0 and (c < 0 or o < c):
            outside.append(rest[:o])
            rest = rest[o + len(OPEN):]
            c = rest.find(CLOSE)
            if c < 0:  # cut inside the block
                inside.append(rest)
                break
            inside.append(rest[:c])
            rest = rest[c + len(CLOSE):]
        else:  # a close with no open: the prompt opened the block
            inside.append(rest[:c])
            rest = rest[c + len(CLOSE):]
    return "\n".join(x.strip() for x in inside if x.strip()), "".join(outside).strip()


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def common_run(a, b, n=60):
    a, b = norm(a), norm(b)
    return any(a[i:i + n] in b for i in range(0, max(0, len(a) - n + 1), 10)) if len(a) >= n else False


def audit(a):
    rows, fails = [], 0
    for arm, extra in ARMS:
        for i, p in [(i, p) for _ in range(a.seeds) for i, p in enumerate(PROMPTS)]:
            seed = len([r for r in rows if r["arm"] == arm and r["prompt"] == i])
            base = {"model": a.model, "messages": [{"role": "user", "content": p}], "temperature": a.temperature,
                    **({"seed": seed} if a.temperature > 0 else {}), **extra}
            code, v = post(a.url, base)
            r = {"arm": arm, "prompt": i, "seed": seed, "status": code, "checks": {}}
            if code != 200:
                r["error"] = v
                fails += 1
                rows.append(r)
                continue
            c = v["choices"][0]
            m = c["message"]
            content, trace = m.get("content") or "", trace_of(m)
            usage = v.get("usage", {})
            thinking_on = extra.get("chat_template_kwargs", {}).get("enable_thinking", True)
            ck = r["checks"]
            ck["markers_in_content"] = [x for x in MARKERS if x in content] or None
            ck["markers_in_reasoning"] = [x for x in MARKERS if x in trace] or None
            ck["deliberation_in_answer"] = [l.strip()[:60] for l in content.splitlines() if DELIBERATION.match(l)] or None
            r["trace_overlaps_answer"] = common_run(trace, content)
            if thinking_on and c["finish_reason"] == "stop" and not content.strip() and not m.get("tool_calls"):
                ck["empty_answer_on_stop"] = True
            if usage.get("completion_tokens", 0) > extra["max_tokens"]:
                ck["over_max_tokens"] = usage["completion_tokens"]
            rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if "thinking_token_budget" in extra and rt is not None and rt > extra["thinking_token_budget"] + SLACK:
                ck["trace_over_budget"] = rt
            if "max_reply_tokens" in extra and rt is not None:
                if usage["completion_tokens"] - rt > extra["max_reply_tokens"] + 1:
                    ck["answer_over_reply_cap"] = usage["completion_tokens"] - rt
            code_s, frames = post(a.url, {**base, "stream": True})
            if code_s != 200:
                ck["stream_error"] = {"status": code_s, "error": frames}
                frames = []
            elif any("error" in f for f in frames):
                ck["stream_error"] = [f["error"] for f in frames if "error" in f]
            s_content = "".join(ch["delta"].get("content") or "" for f in frames for ch in f.get("choices", []))
            s_trace = "".join(ch["delta"].get("reasoning") or ch["delta"].get("reasoning_content") or ""
                              for f in frames for ch in f.get("choices", []))
            if (norm(s_content), norm(s_trace)) != (norm(content), norm(trace)):
                ck["stream_differs"] = {"stream": s_content[:80], "buffered": content[:80]}
            if a.raw and thinking_on:
                if a.raw == "format":
                    _, rv = post(a.url, {**base, "reasoning_format": "none"})
                    raw = rv["choices"][0]["message"].get("content") or ""
                elif a.raw.startswith("http"):
                    _, rv = post(a.raw, base)
                    raw = rv["choices"][0]["message"].get("content") or ""
                else:
                    raw = RAW.get((arm, i), "")
                er, ec = split_raw(raw)
                if (norm(er), norm(ec)) != (norm(trace), norm(content)):
                    ck["split_differs_from_raw"] = {"raw": raw[-160:], "reasoning": trace[-80:], "content": content[:80]}
            r.update(finish=c["finish_reason"], reasoning_tokens=rt, completion_tokens=usage.get("completion_tokens"),
                     content=content, reasoning=trace, thinking=c.get("x_plow_thinking"))
            r["checks"] = {k: x for k, x in ck.items() if x}
            fails += bool(r["checks"])
            rows.append(r)
    # A thinking budget with no room left for the reply.
    code, v = post(a.url, {"model": a.model, "messages": [{"role": "user", "content": PROMPTS[0]}],
                           "temperature": 0, "thinking_token_budget": 300, "max_tokens": 128})
    no_room = {"status": code, "finish": v["choices"][0]["finish_reason"] if code == 200 else None,
               "content": (v["choices"][0]["message"].get("content") if code == 200 else None),
               "error": v.get("error", {}).get("message") if code != 200 else None}
    if not rows:
        fails += 1
    return rows, fails, no_room


def report(name, rows, fails, no_room):
    out = [f"# {name}: thinking/answer separation audit", "",
           "| arm | n | errors | failed | markers in answer | markers in trace | split != raw |"
           " deliberation in answer | stream != buffered | empty answer | over budget/cap | finish length |"
           " trace overlaps answer (info) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, _ in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        ok = [r for r in rs if r["status"] == 200]
        cnt = lambda k: sum(k in r["checks"] for r in ok)
        out.append(f"| {arm} | {len(rs)} | {len(rs) - len(ok)} | {sum(bool(r['checks']) for r in ok)} | "
                   f"{cnt('markers_in_content')} | {cnt('markers_in_reasoning')} | {cnt('split_differs_from_raw')} | "
                   f"{cnt('deliberation_in_answer')} | {cnt('stream_differs')} | {cnt('empty_answer_on_stop')} | "
                   f"{cnt('trace_over_budget') + cnt('answer_over_reply_cap') + cnt('over_max_tokens')} | "
                   f"{sum(r.get('finish') == 'length' for r in ok)} | {sum(bool(r.get('trace_overlaps_answer')) for r in ok)} |")
    out += ["", f"Thinking budget 300 with max_tokens 128: {json.dumps(no_room, ensure_ascii=False)}", "",
            "## Failures", ""]
    for r in rows:
        if r.get("checks") or r["status"] != 200:
            out.append(f"- {r['arm']} prompt {r['prompt']}: {json.dumps(r.get('checks') or r.get('error'), ensure_ascii=False)[:600]}")
    if fails == 0:
        out.append("none")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", help="'format' (reasoning_format none), the URL of the model served without a "
                    "parser, or a --record-raw file")
    ap.add_argument("--record-raw", metavar="FILE",
                    help="only record each arm/prompt's raw content (a parser-less server) to FILE")
    ap.add_argument("--prompts", type=int, help="first N prompts only")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampled audit: each prompt's calls share a seed, so stream and raw stay comparable")
    ap.add_argument("--seeds", type=int, default=1, help="runs per prompt (seeds 0..N-1) when sampling")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global PROMPTS
    PROMPTS = PROMPTS[:a.prompts] if a.prompts else PROMPTS
    if a.record_raw:
        with open(a.record_raw, "w") as f:
            for arm, extra in ARMS:
                for i, p in enumerate(PROMPTS):
                    _, v = post(a.url, {"model": a.model, "messages": [{"role": "user", "content": p}],
                                        "temperature": 0, **extra})
                    f.write(json.dumps({"arm": arm, "prompt": i,
                                        "content": v["choices"][0]["message"].get("content") or ""}) + "\n")
        return 0
    if a.raw and not a.raw.startswith("http") and a.raw != "format":
        RAW.update({(r["arm"], r["prompt"]): r["content"] for r in map(json.loads, open(a.raw))})
    rows, fails, no_room = audit(a)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, a.name + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    text = report(a.name, rows, fails, no_room)
    with open(os.path.join(a.out, a.name + ".md"), "w") as f:
        f.write(text)
    print(text)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
