#!/usr/bin/env python3
"""Live check of the OpenAI surface fixes against a running plowrt.

Every case here corresponds to a defect that was found by auditing the server
against the OpenAI API, so a regression in any of them is a client-visible bug.

    python3 scripts/glm53_api_check.py http://127.0.0.1:18981 glm-5.3
"""
import json, sys, urllib.error, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18981"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm-5.3"
fails = []


def call(path, body=None, method=None, raw=None):
    url = BASE + path
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
            try:
                return r.status, json.loads(body or b"null")
            except json.JSONDecodeError:
                # /health is plain text by design.
                return r.status, {"_text": body.decode(errors="replace")}
    except urllib.error.HTTPError as e:
        raw_body = e.read()
        try:
            return e.code, json.loads(raw_body)
        except Exception:
            return e.code, {"_raw": raw_body.decode(errors="replace")}


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  -- ' + str(detail)[:300]}")
    if not ok:
        fails.append(name)


s, b = call("/health")
check("/health responds", s == 200, s)

s, b = call("/v1/models")
check("model card has created", s == 200 and isinstance(b["data"][0].get("created"), int), b)

s, b = call("/v1/chat/completions", {"model": MODEL, "max_tokens": 200, "temperature": 0,
    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}]})
check("chat 200", s == 200, b)
if s == 200:
    ch = b["choices"][0]
    check("chat has created", isinstance(b.get("created"), int), b.get("created"))
    check("finish_reason is an OpenAI value",
          ch["finish_reason"] in ("stop", "length", "tool_calls", "content_filter"), ch)
    check("reasoning split out of content",
          "</think>" not in (ch["message"].get("content") or ""), ch["message"])
    check("usage present", b["usage"]["completion_tokens"] > 0, b.get("usage"))
    print("      content:", json.dumps((ch["message"].get("content") or "")[:120]))
    print("      reasoning_content present:", ch["message"].get("reasoning_content") is not None)

s, b = call("/v1/chat/completions", {"model": MODEL, "max_tokens": 60, "temperature": 0,
    "stop": ["4"], "messages": [{"role": "user", "content": "Count: 1 2 3 4 5 6"}]})
check("stop string accepted and applied",
      s == 200 and "4" not in (b["choices"][0]["message"].get("content") or ""), b)

s, b = call("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
    "tools": [{"type": "function", "function": {"name": "f"}}]})
check("tools refused with an envelope",
      s == 400 and b.get("error", {}).get("code") == "unsupported_parameter", (s, b))

s, b = call("/v1/chat/completions", {"model": MODEL, "n": 3,
    "messages": [{"role": "user", "content": "hi"}]})
check("n>1 refused", s == 400 and "error" in b, (s, b))

s, b = call("/v1/chat/completions", {"model": "not-a-model",
    "messages": [{"role": "user", "content": "hi"}]})
check("unknown model 404 with code",
      s == 404 and b.get("error", {}).get("code") == "model_not_found", (s, b))

s, b = call("/v1/chat/completions", raw=b"{not json")
check("malformed JSON returns a JSON envelope",
      s in (400, 422) and isinstance(b.get("error"), dict), (s, b))

s, b = call("/v1/chat/completions", {"model": MODEL, "max_tokens": 8,
    "messages": [{"role": "assistant", "content": None},
                 {"role": "user", "content": "hi"}]})
check("null assistant content accepted", s == 200, (s, b))

s, b = call("/v1/completions", {"model": MODEL, "prompt": [1, 2, 3], "max_tokens": 4})
check("token-id prompt accepted", s == 200 and isinstance(b.get("created"), int), (s, b))

s, b = call("/v1/completions", {"model": MODEL, "prompt": ["a", "b"], "max_tokens": 4})
check("batched prompt refused", s == 400 and "error" in b, (s, b))

s, b = call("/v1/chat/completions", {"model": MODEL, "max_tokens": 4,
    "messages": [{"role": "user", "content": "x " * 60000}]})
check("context overflow is 400, not a retryable 429",
      s == 400 and b.get("error", {}).get("code") == "context_length_exceeded", (s, b))

s, b = call("/tokenize", {"model": MODEL, "prompt": "hello"})
check("tokenize reports count", s == 200 and b.get("count") == len(b.get("tokens", [])), b)

print()
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
