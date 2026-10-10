#!/usr/bin/env python3
"""Live tool-calling check of a serving plowrt (or any OpenAI-compatible server) with the OpenAI
Python SDK: a tool loop, streamed calls with incremental argument deltas, parallel calls,
`tool_choice` required / named, `strict`, logprobs on tool deltas, and reasoning separation.

    python3 scripts/llm/toolcall_live_check.py http://127.0.0.1:PORT [MODEL] --out result.json

Exit code = number of failed checks. Parallel calls are reported, not required: whether a model
emits two calls for a two-part question is its choice.
"""
import argparse
import json
import sys
import time

from openai import OpenAI

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "Get the current local time in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]

STRICT = [dict(TOOLS[0], function=dict(TOOLS[0]["function"], strict=True,
               parameters=dict(TOOLS[0]["function"]["parameters"], additionalProperties=False,
                               required=["city", "unit"])))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    c = OpenAI(base_url=a.base.rstrip("/") + "/v1", api_key="x", max_retries=0, timeout=300)
    model = a.model or c.models.list().data[0].id
    results = []

    def check(name, ok, **info):
        results.append({"check": name, "ok": bool(ok), **info})
        print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(info, ensure_ascii=False)[:400], flush=True)

    def chat(**kw):
        kw.setdefault("max_tokens", a.max_tokens)
        kw.setdefault("temperature", 0)
        return c.chat.completions.create(model=model, **kw)

    def stream(**kw):
        t0 = time.perf_counter()
        frames, content, reasoning, calls, finish, lp = 0, "", "", {}, None, 0
        arg_deltas = 0
        first_head = None
        for ch in chat(stream=True, **kw):
            if not ch.choices:
                continue
            frames += 1
            ch0 = ch.choices[0]
            d = ch0.delta
            content += d.content or ""
            reasoning += getattr(d, "reasoning_content", None) or ""
            if ch0.logprobs and ch0.logprobs.content and d.tool_calls:
                lp += 1
            for t in d.tool_calls or []:
                e = calls.setdefault(t.index, {"id": None, "name": None, "arguments": ""})
                if t.id:
                    e["id"] = t.id
                    if first_head is None:
                        first_head = bool(t.function and t.function.name)
                if t.function and t.function.name:
                    e["name"] = t.function.name
                if t.function and t.function.arguments:
                    e["arguments"] += t.function.arguments
                    arg_deltas += 1
            finish = ch0.finish_reason or finish
        return dict(frames=frames, content=content, reasoning=reasoning, calls=[calls[i] for i in sorted(calls)],
                    finish=finish, arg_deltas=arg_deltas, first_head=first_head, lp_frames=lp,
                    secs=round(time.perf_counter() - t0, 3))

    def args_ok(s, need=("city",)):
        try:
            v = json.loads(s)
        except Exception:
            return False
        return isinstance(v, dict) and all(k in v for k in need)

    q = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]

    # 1. tool loop, buffered
    r = chat(messages=q, tools=TOOLS)
    m = r.choices[0].message
    tc = m.tool_calls or []
    check("loop: call", tc and tc[0].function.name == "get_weather" and args_ok(tc[0].function.arguments)
          and r.choices[0].finish_reason == "tool_calls",
          content=m.content, calls=[(t.id, t.function.name, t.function.arguments) for t in tc],
          finish=r.choices[0].finish_reason)
    if tc:
        msgs = q + [{"role": "assistant", "content": m.content, "tool_calls": [t.model_dump() for t in tc]}]
        msgs += [{"role": "tool", "tool_call_id": t.id, "content": json.dumps({"temp_c": 18, "sky": "clear"})} for t in tc]
        r2 = chat(messages=msgs, tools=TOOLS)
        m2 = r2.choices[0].message
        text = m2.content or ""
        check("loop: answer after tool result", text and "18" in text and not text.lstrip().startswith("thought")
              and "<|" not in text and not m2.tool_calls,
              content=text, reasoning=getattr(m2, "reasoning_content", None), finish=r2.choices[0].finish_reason)
        s2 = stream(messages=msgs, tools=TOOLS)
        check("loop: streamed answer after tool result", s2["content"] and not s2["content"].lstrip().startswith("thought")
              and "<|" not in s2["content"], content=s2["content"], reasoning=s2["reasoning"], finish=s2["finish"])

    # 2. streamed call with incremental argument deltas
    s = stream(messages=q, tools=TOOLS)
    ok = (s["calls"] and s["calls"][0]["name"] == "get_weather" and s["calls"][0]["id"] and args_ok(s["calls"][0]["arguments"])
          and s["finish"] == "tool_calls" and s["first_head"])
    check("stream: call", ok, calls=s["calls"], finish=s["finish"], first_delta_has_name=s["first_head"])
    check("stream: arguments arrive in several deltas", s["arg_deltas"] > 1, arg_deltas=s["arg_deltas"], frames=s["frames"])

    # 3. parallel calls (reported)
    pq = [{"role": "user", "content": "I need two things: the weather in Paris and the local time in Tokyo. Call both tools."}]
    r = chat(messages=pq, tools=TOOLS)
    tc = r.choices[0].message.tool_calls or []
    results.append({"check": "parallel: buffered (info)", "ok": True, "calls": [(t.function.name, t.function.arguments) for t in tc]})
    print("INFO parallel buffered", [(t.function.name, t.function.arguments) for t in tc])
    s = stream(messages=pq, tools=TOOLS)
    check("parallel: streamed indices dense, ids unique", len({x["id"] for x in s["calls"]}) == len(s["calls"])
          and all(args_ok(x["arguments"]) for x in s["calls"]), calls=s["calls"])
    check("parallel: buffered and streamed agree on names", [t.function.name for t in tc] == [x["name"] for x in s["calls"]],
          buffered=[t.function.name for t in tc], streamed=[x["name"] for x in s["calls"]])
    r = chat(messages=pq, tools=TOOLS, parallel_tool_calls=False)
    check("parallel_tool_calls=false keeps one", len(r.choices[0].message.tool_calls or []) <= 1,
          calls=len(r.choices[0].message.tool_calls or []))

    # 4. tool_choice required / named / none
    hq = [{"role": "user", "content": "Hello! How are you today?"}]
    r = chat(messages=hq, tools=TOOLS, tool_choice="required")
    tc = r.choices[0].message.tool_calls or []
    check("required: a call even for small talk", tc and tc[0].function.name in ("get_weather", "get_time")
          and args_ok(tc[0].function.arguments, ()), calls=[(t.function.name, t.function.arguments) for t in tc],
          finish=r.choices[0].finish_reason)
    named = {"type": "function", "function": {"name": "get_time"}}
    r = chat(messages=q, tools=TOOLS, tool_choice=named)
    tc = r.choices[0].message.tool_calls or []
    check("named: exactly the named function", len(tc) == 1 and tc[0].function.name == "get_time"
          and args_ok(tc[0].function.arguments), calls=[(t.function.name, t.function.arguments) for t in tc])
    s = stream(messages=q, tools=TOOLS, tool_choice=named)
    check("named: streamed", len(s["calls"]) == 1 and s["calls"][0]["name"] == "get_time" and args_ok(s["calls"][0]["arguments"]),
          calls=s["calls"], arg_deltas=s["arg_deltas"])
    r = chat(messages=q, tools=TOOLS, tool_choice="none")
    check("none: text, no calls", not r.choices[0].message.tool_calls and r.choices[0].message.content,
          content=r.choices[0].message.content)

    # 5. strict
    try:
        r = chat(messages=q, tools=STRICT)
        tc = r.choices[0].message.tool_calls or []
        check("strict: valid arguments or none", all(args_ok(t.function.arguments, ("city", "unit")) for t in tc),
              calls=[(t.function.name, t.function.arguments) for t in tc])
    except Exception as e:  # a model that violates the schema gets invalid_tool_call
        check("strict: violation reported as invalid_tool_call", "invalid_tool_call" in str(e), error=str(e)[:300])

    # 6. logprobs on tool deltas
    s = stream(messages=q, tools=TOOLS, logprobs=True)
    check("logprobs ride tool-call deltas", s["calls"] and s["lp_frames"] > 0, lp_frames=s["lp_frames"])

    # 7. errors stay 400 for what cannot be served
    try:
        chat(messages=q, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}})
        check("named unknown function is a 400", False)
    except Exception as e:
        check("named unknown function is a 400", "400" in str(e) or "tool_choice" in str(e), error=str(e)[:200])

    fails = sum(not r["ok"] for r in results)
    print(f"{len(results) - fails}/{len(results)} passed")
    if a.out:
        json.dump({"model": model, "base": a.base, "results": results}, open(a.out, "w"), indent=1, ensure_ascii=False)
    return fails


if __name__ == "__main__":
    sys.exit(main())
