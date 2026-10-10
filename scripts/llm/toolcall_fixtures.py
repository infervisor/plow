#!/usr/bin/env python3
"""Generate the chat-template tool-calling parity fixtures for plowrt.

For each family: copy the checkpoint's own chat template, render a fixed set of OpenAI-shaped
tool conversations with transformers' `render_jinja_template` (the code path
`apply_chat_template(messages, tools=...)` runs), and record the text and, when the tokenizer is
available, the token ids. `crates/plowrt/src/serve/tools/parity_tests.rs` renders the same
requests through plowrt's request mapping and template engine and compares.

Only tokenizer/template files are downloaded (no weights):

    python3 scripts/llm/toolcall_fixtures.py --out crates/plowrt/tests/fixtures/toolcall \
        [--cache DIR] [--only qwen3 ...]
"""
import argparse
import copy
import hashlib
import json
import os
import sys

VLLM_TPL = os.environ.get("VLLM_TOOL_TEMPLATES", "/opt/dlami/nvme/lava-tts/toolperf/scratch/vllm_tpl/")

FAMILIES = {
    # name: (hub repo, local dir or None[, template file override[, named template]])
    "gemma4-e4b": ("google/gemma-4-E4B-it", None),
    "gemma4-12b": ("google/gemma-4-12b-it", "/opt/dlami/nvme/hf-cache/hub/gemma-4-12b-it-fp8"),
    "gemma4-31b": ("google/gemma-4-31B-it",
                   "/opt/dlami/nvme/lava-tts/hf/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"),
    "qwen3": ("Qwen/Qwen3-8B", None),
    "qwen2.5": ("Qwen/Qwen2.5-7B-Instruct", None),
    "qwen3.5": ("Qwen/Qwen3.5-27B", None),
    "qwen3-coder": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", None),
    "llama3.1": ("unsloth/Meta-Llama-3.1-8B-Instruct", None),
    "llama3.2": ("unsloth/Llama-3.2-3B-Instruct", None),
    "mistral-v0.3": ("mistralai/Mistral-7B-Instruct-v0.3", None),
    "glm4.5": ("zai-org/GLM-4.5", None),
    "glm5.3": ("zai-org/GLM-5.3", None),
    "kimi-k2": ("moonshotai/Kimi-K2-Instruct", None),
    "gpt-oss": ("openai/gpt-oss-20b", None),
    "deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", None),
    "deepseek-r1": ("deepseek-ai/DeepSeek-R1-0528", None),
    "mixtral": ("mistralai/Mixtral-8x7B-Instruct-v0.1", None),
    "llama3.3": ("unsloth/Llama-3.3-70B-Instruct", None),
    "glm5": ("zai-org/GLM-5", None),
    # The checkpoint's named `tool_use` template, which transformers picks when `tools` are passed.
    "hermes3": ("NousResearch/Hermes-3-Llama-3.1-8B", None, None, "tool_use"),
    # DeepSeek's own templates render no `tools`; vLLM ships tool templates for them
    # (examples/tool_chat_template_deepseek*.jinja, v0.11.0), rendered with each checkpoint's tokens.
    "deepseek-v3-tools": ("deepseek-ai/DeepSeek-V3-0324", None, VLLM_TPL + "deepseekv3.jinja"),
    "deepseek-r1-tools": ("deepseek-ai/DeepSeek-R1-0528", None, VLLM_TPL + "deepseekr1.jinja"),
    "deepseek-v3.1-tools": ("deepseek-ai/DeepSeek-V3.1", None, VLLM_TPL + "deepseekv31.jinja"),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
        }, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_flights",
        "description": "Search flights between two airports.",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string", "description": "Origin airport"},
            "destination": {"type": "string", "description": "Destination airport"},
            "passengers": {"type": "integer", "description": "Number of passengers"},
            "filters": {"type": "object", "description": "Optional filters", "properties": {
                "max_price": {"type": "number", "description": "Price cap"},
                "airlines": {"type": "array", "items": {"type": "string"}, "description": "Allowed airlines"},
            }},
        }, "required": ["origin", "destination"]}}},
]

W_ARGS = {"city": "Paris", "unit": "celsius"}
F_ARGS = {"origin": "O'Hare \"Intl\"", "destination": "Zürich", "passengers": 2,
          "filters": {"max_price": 450.5, "airlines": ["LX", "UA"]}}


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def cases():
    sys_msg = {"role": "system", "content": "You are a helpful travel assistant."}
    user = {"role": "user", "content": "What's the weather in Paris?"}
    one = {"role": "assistant", "content": None, "tool_calls": [call("a1B2c3D4e", "get_weather", W_ARGS)]}
    one_res = {"role": "tool", "tool_call_id": "a1B2c3D4e", "content": "{\"temp\": 18, \"sky\": \"clear\"}"}
    two_user = {"role": "user", "content": "Weather in Paris, and flights from O'Hare to Zürich for 2?"}
    two = {"role": "assistant", "content": None, "tool_calls": [
        call("a1B2c3D4e", "get_weather", W_ARGS), call("f5G6h7I8j", "search_flights", F_ARGS)]}
    two_res = [one_res, {"role": "tool", "tool_call_id": "f5G6h7I8j", "content": "[{\"flight\": \"LX9\", \"price\": 420}]"}]
    final = {"role": "assistant", "content": "It is 18°C and clear in Paris."}
    return [
        ("tools_first_turn", {"messages": [sys_msg, user], "tools": TOOLS}),
        ("tools_no_system", {"messages": [user], "tools": TOOLS}),
        ("tool_loop", {"messages": [user, one, one_res], "tools": TOOLS}),
        ("parallel_calls", {"messages": [two_user, two] + two_res, "tools": TOOLS}),
        ("multi_turn", {"messages": [sys_msg, user, one, one_res, final,
                                     {"role": "user", "content": "And tomorrow?"}], "tools": TOOLS}),
        ("history_without_tools", {"messages": [user, one, one_res]}),
        ("plain_chat", {"messages": [sys_msg, user, {"role": "assistant", "content": "Sunny."},
                                     {"role": "user", "content": "Thanks!"}]}),
    ]


def hf_messages(msgs, args_as_objects=True):
    """plowrt's request mapping (crates/plowrt/src/serve/tools/request.rs): tool-call `arguments`
    strings become objects, as vLLM does; an assistant tool-call turn's null content becomes ""."""
    out = copy.deepcopy(msgs)
    for m in out:
        if m.get("tool_calls") and m.get("content") is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:
            if args_as_objects and isinstance(tc["function"].get("arguments"), str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    return out


def render(render_jinja_template, tpl, req, bos, eos):
    """Arguments as objects first; a template that concatenates them as strings (DeepSeek's)
    fails on an object and is rendered with the JSON strings instead."""
    kw = dict(chat_template=tpl, add_generation_prompt=True, bos_token=bos, eos_token=eos)
    try:
        return render_jinja_template([hf_messages(req["messages"])], tools=req.get("tools"), **kw)[0][0]
    except Exception:
        if not any(m.get("tool_calls") for m in req["messages"]):
            raise
        return render_jinja_template([hf_messages(req["messages"], False)], tools=req.get("tools"), **kw)[0][0]


def load_template(d, tpl_file=None, variant=None):
    from transformers.utils.chat_template_utils import render_jinja_template  # noqa: F401
    cfg = json.load(open(os.path.join(d, "tokenizer_config.json")))
    if os.path.exists(os.path.join(d, "chat_template.jinja")):
        tpl = open(os.path.join(d, "chat_template.jinja")).read()
    else:
        tpl = cfg.get("chat_template")
        if isinstance(tpl, list):
            want = variant or "default"
            tpl = next((e["template"] for e in tpl if e["name"] == want), tpl[0]["template"])
    if tpl_file:
        tpl = open(tpl_file).read()
    tok = lambda k: (cfg.get(k)["content"] if isinstance(cfg.get(k), dict) else cfg.get(k))
    return tpl, tok("bos_token"), tok("eos_token")


def fetch(repo, cache):
    from huggingface_hub import hf_hub_download, list_repo_files
    d = os.path.join(cache, repo.replace("/", "__"))
    files = list_repo_files(repo)
    for f in ["tokenizer_config.json", "chat_template.jinja", "tokenizer.json"]:
        if f in files and not os.path.exists(os.path.join(d, f)):
            hf_hub_download(repo, f, local_dir=d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/opt/dlami/nvme/lava-tts/toolcalls/tpl")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    from transformers.utils.chat_template_utils import render_jinja_template
    for fam, (repo, local, *extra) in FAMILIES.items():
        if a.only and fam not in a.only:
            continue
        d = local or fetch(repo, a.cache)
        tpl, bos, eos = load_template(d, *extra)
        tokjs = os.path.join(d, "tokenizer.json")
        tokz = None
        if os.path.exists(tokjs):
            from tokenizers import Tokenizer
            tokz = Tokenizer.from_file(tokjs)
        out = []
        for name, req in cases():
            # transformers renders a request without tools with the default template, not the named one.
            if len(extra) > 1 and "tools" not in req:
                continue
            rec = {"name": name, "request": req}
            try:
                text = render(render_jinja_template, tpl, req, bos, eos)
                rec["expected"] = text
                if tokz is not None:
                    rec["ids"] = tokz.encode(text, add_special_tokens=False).ids
            except Exception as e:  # the template refuses this conversation
                rec["error"] = f"{type(e).__name__}: {e}"
            out.append(rec)
        fd = os.path.join(a.out, fam)
        os.makedirs(fd, exist_ok=True)
        open(os.path.join(fd, "chat_template.jinja"), "w").write(tpl)
        json.dump({"bos_token": bos, "eos_token": eos}, open(os.path.join(fd, "tokenizer_config.json"), "w"))
        json.dump({"repo": repo, "template_sha256": hashlib.sha256(tpl.encode()).hexdigest(),
                   "tokenizer": tokjs if tokz else None, "cases": out},
                  open(os.path.join(fd, "cases.json"), "w"), ensure_ascii=False, indent=1)
        print(fam, [(c["name"], "err" if "error" in c else len(c["expected"])) for c in out])


if __name__ == "__main__":
    sys.exit(main())
