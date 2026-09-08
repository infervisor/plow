#!/usr/bin/env python3
"""Diff plowrt's chat-template render against the one `transformers` produces.

plowrt renders the checkpoint's own `chat_template.jinja` with minijinja, which
is not Jinja2-on-Python: HF templates call real `str` and `dict` methods that
minijinja has no notion of, and a missing one fails the whole render — which the
server answers as a 400 on every chat request for that family. This walks every
checkpoint under a models root, renders six conversation shapes through both
engines, and reports any divergence.

The reference env is configured the way `transformers.apply_chat_template`
configures it, so only jinja2 is needed — not torch.

    cargo build -p plowrt --example tmpl_probe
    python3 scripts/chat_template_check.py /workspace/models
"""
import datetime, json, os, subprocess, sys

from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.exceptions import TemplateError

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models"
PROBE = sys.argv[2] if len(sys.argv) > 2 else "target/debug/examples/tmpl_probe"

CASES = {
    "user_only": [{"role": "user", "content": "Hi"}],
    "system_user": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ],
    "multi_turn": [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ],
    "developer": [
        {"role": "developer", "content": "be terse"},
        {"role": "user", "content": "Hi"},
    ],
    "tool_result": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None},
        {"role": "tool", "tool_call_id": "c1", "content": "18C"},
    ],
    "null_content": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "?"},
    ],
}


def reference_env():
    def raise_exception(message):
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, **{"ensure_ascii": False, **kw})
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env


def find_template(d):
    """The same two locations, in the same order, that `ChatTemplate::load` reads."""
    for base in (d, os.path.join(d, "checkpoint")):
        path = os.path.join(base, "chat_template.jinja")
        if os.path.isfile(path):
            text = open(path).read()
            if text.strip():
                return text, path
        cfg_path = os.path.join(base, "tokenizer_config.json")
        if os.path.isfile(cfg_path):
            try:
                cfg = json.load(open(cfg_path))
            except ValueError:
                continue
            text = cfg.get("chat_template")
            if isinstance(text, str) and text.strip():
                return text, cfg_path + "#chat_template"
    return None, None


def specials(d):
    for base in (d, os.path.join(d, "checkpoint")):
        cfg_path = os.path.join(base, "tokenizer_config.json")
        if os.path.isfile(cfg_path):
            try:
                cfg = json.load(open(cfg_path))
            except ValueError:
                continue

            def tok(key):
                v = cfg.get(key)
                if isinstance(v, str):
                    return v
                if isinstance(v, dict):
                    return v.get("content")
                return None

            return tok("bos_token"), tok("eos_token")
    return None, None


def reference(d, messages):
    text, _ = find_template(d)
    bos, eos = specials(d)
    try:
        return reference_env().from_string(text).render(
            messages=messages, tools=None, add_generation_prompt=True,
            bos_token=bos, eos_token=eos,
        ), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def plow(d, messages):
    r = subprocess.run([PROBE, d, json.dumps(messages)], capture_output=True, text=True)
    if r.returncode != 0:
        return None, "probe crashed: " + r.stderr.strip()[-300:]
    out = json.loads(r.stdout)
    return out["out"], out["err"]


def project(messages):
    """What `serve::chat` actually hands the template: role + flattened text."""
    def text(m):
        c = m.get("content")
        if c is None:
            return ""
        if isinstance(c, str):
            return c
        return "".join(p.get("text", "") for p in c if p.get("type") == "text")

    return [{"role": m["role"], "content": text(m)} for m in messages]


fails = []
for name in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d) or find_template(d)[0] is None:
        continue
    for case, messages in CASES.items():
        messages = project(messages)
        want, want_err = reference(d, messages)
        got, got_err = plow(d, messages)
        if want_err and got_err:
            continue  # both engines refuse the shape, which is the template's call
        if want_err or got_err or want != got:
            fails.append((name, case, want, want_err, got, got_err))
            print(f"FAIL {name} {case}\n  transformers: {want_err or want!r}\n  plowrt      : {got_err or got!r}")
        else:
            print(f"ok   {name} {case}")

print(f"\n{len(fails)} divergence(s)")
sys.exit(1 if fails else 0)
