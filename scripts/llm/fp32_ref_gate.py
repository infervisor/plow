#!/usr/bin/env python3
"""Numerics gate against an independent FP32 reference, with vLLM scored as a peer.

Bit-exact agreement with vLLM is not a quality bar: both stacks round differently from the
checkpoint's exact math, so near-tie flips fail any exact gate. Instead both are scored against
FP32 (FP8 checkpoint dequantized to FP32 weights, FP32 activations, TF32 off) on one fixed prompt
set, and the candidate passes if it is within the peer's distance to FP32 plus a tolerance.

  prompts   --hf DIR --corpus NAME=FILE ... --out prompts.json     fixed token-id prompt set (CPU)
  reference --hf DIR --prompts prompts.json --out ref.json         FP32 greedy continuation + top-20
            [--device cpu]                                         (default cuda)
  capture   --url URL --ref ref.json --arm NAME --out cap.json     a served stack, teacher-forced
  score     --ref ref.json CAP...                                  metrics per capture
  gate      --ref ref.json --cand CAP --peer CAP [thresholds]      verdict (exit 1 on fail)

Capture is teacher-forced on the FP32 continuation through the public completions API: each
request is greedy with top-20 logprobs from prompt + reference[:k]; every position up to and
including the first token that differs from the reference has the exact FP32 history, so the
next request restarts one past it. Requests per case = 1 + flips. The first request's free-run
output is the stack's own greedy continuation (agreement length, needle answer).
"""
import argparse
import hashlib
import json
import math
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOP = 20
EOS = (1, 50, 106)
DEFAULTS = dict(tie_margin=1.0, kl_ratio_max=1.25, kl_slack_max=0.002, top1_drop_max=0.01,
                cont_drop_max=0.05, needle_drop_max=0.0, needle_min=0.9)


def ids_sha(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- scoring (pure; unit tested)
def kl_top(ref_top, arm_top):
    """KL(ref || arm) over ref's top-k plus one remainder bucket. A ref token missing from the
    arm's top-k takes the arm's k-th logprob (an upper bound on its true value)."""
    if not arm_top:
        return float("inf")
    arm = dict(arm_top)
    floor = min(arm.values())
    kl, ref_rest, arm_rest = 0.0, 1.0, 1.0
    for t, lr in ref_top:
        la = arm.get(t, floor)
        kl += math.exp(lr) * (lr - la)
        ref_rest -= math.exp(lr)
        arm_rest -= math.exp(la)
    if ref_rest > 1e-9:
        kl += ref_rest * (math.log(ref_rest) - math.log(max(arm_rest, 1e-6)))
    return max(kl, 0.0)


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def agree_len(free, cont):
    n = 0
    for a, b in zip(free, cont):
        if a != b:
            break
        n += 1
    return n


def score(ref, cap, tie_margin=DEFAULTS["tie_margin"]):
    """Metrics of one capture against the FP32 reference, overall and per prompt kind."""
    if cap.get("ref_sha256") != ref["sha256"]:
        raise ValueError(f"capture {cap.get('arm')} was taken against another reference")
    groups = {}
    missing = 0
    for case in ref["cases"]:
        c = cap["cases"].get(case["id"])
        g = groups.setdefault(case["kind"], dict(kl=[], dec=0, dec_ok=0, cont=[], full=0, n=0, needle=[]))
        g["n"] += 1
        if c is None or c.get("error"):
            missing += len(case["pos"])
            g["cont"].append(0.0)
            if "expect" in case:
                g["needle"].append(False)
            continue
        for k, p in enumerate(case["pos"]):
            top = c["pos"].get(str(k))
            if top is None:
                missing += 1
                continue
            g["kl"].append(kl_top(p["top"], top))
            if p["margin"] > tie_margin:
                g["dec"] += 1
                g["dec_ok"] += top[0][0] == p["top"][0][0]
        n = agree_len(c["free"], case["cont"])
        g["cont"].append(n / len(case["cont"]))
        g["full"] += n == len(case["cont"])
        if "expect" in case:
            g["needle"].append(case["expect"] in c["free_text"])

    def summarize(gs):
        kl = [x for g in gs for x in g["kl"]]
        dec, dec_ok = sum(g["dec"] for g in gs), sum(g["dec_ok"] for g in gs)
        cont = [x for g in gs for x in g["cont"]]
        needle = [x for g in gs for x in g["needle"]]
        out = dict(positions=len(kl), kl_mean=sum(kl) / len(kl) if kl else float("nan"),
                   kl_p99=percentile(kl, 0.99), kl_max=max(kl, default=float("nan")),
                   decisive=dec, top1_decisive=dec_ok / dec if dec else float("nan"),
                   cont_frac=sum(cont) / len(cont) if cont else float("nan"),
                   cont_full=sum(g["full"] for g in gs), cases=sum(g["n"] for g in gs))
        if needle:
            out["needle_acc"] = sum(needle) / len(needle)
            out["needle_n"] = len(needle)
        return out

    res = summarize(list(groups.values()))
    res["missing_positions"] = missing
    res["by_kind"] = {k: summarize([g]) for k, g in sorted(groups.items())}
    return res


def verdict(cand, peer, th=None):
    """Pass iff the candidate is within the peer's distance to FP32 (plus tolerance) on every metric."""
    t = dict(DEFAULTS, **(th or {}))
    why = []
    if cand["missing_positions"]:
        why.append(f"candidate missing {cand['missing_positions']} scored positions")
    if peer["missing_positions"]:
        why.append(f"peer missing {peer['missing_positions']} scored positions")
    for m in ("kl_mean", "kl_p99"):
        lim = peer[m] * t["kl_ratio_max"] + t["kl_slack_max"]
        if not cand[m] <= lim:
            why.append(f"{m} {cand[m]:.4g} > peer {peer[m]:.4g} x {t['kl_ratio_max']} + {t['kl_slack_max']}")
    if not cand["top1_decisive"] >= peer["top1_decisive"] - t["top1_drop_max"]:
        why.append(f"top1_decisive {cand['top1_decisive']:.4f} < peer {peer['top1_decisive']:.4f} - {t['top1_drop_max']}")
    if not cand["cont_frac"] >= peer["cont_frac"] - t["cont_drop_max"]:
        why.append(f"cont_frac {cand['cont_frac']:.4f} < peer {peer['cont_frac']:.4f} - {t['cont_drop_max']}")
    if "needle_acc" in peer:
        c = cand.get("needle_acc", 0.0)
        if c < peer["needle_acc"] - t["needle_drop_max"]:
            why.append(f"needle_acc {c:.4f} < peer {peer['needle_acc']:.4f} - {t['needle_drop_max']}")
        if c < t["needle_min"]:
            why.append(f"needle_acc {c:.4f} < needle_min {t['needle_min']}")
    return why


# ---------------------------------------------------------------- prompt set (CPU)
SYSTEM = ("You are an autonomous engineering agent working in a Rust inference-server repository. "
          "Use tools when you need information, keep answers precise, and cite file names.")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Repository-relative path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Full-text search over the repository documentation.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search terms"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                       "required": ["city"]}}},
]


def tool_turn(i, name, args, result):
    cid = f"call_{i}"
    return [{"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]},
            {"role": "tool", "tool_call_id": cid, "name": name, "content": result}]


def chats(text_of):
    """Agentic multi-turn samples. text_of(name, tokens, offset) slices a corpus by token count."""
    return {
        "chat-weather": ([{"role": "user", "content": "Is it a good day for a bike ride in Lisbon?"},
                          *tool_turn(0, "get_weather", {"city": "Lisbon"},
                                     '{"city": "Lisbon", "temp_c": 19, "wind_kph": 31, "rain_mm": 0.0, "sky": "clear"}')],
                         TOOLS),
        "chat-support": ([{"role": "system", "content": "You are a concise support agent for a cloud GPU provider."},
                          {"role": "user", "content": "My training job died with CUDA out of memory after 3 hours."},
                          {"role": "assistant", "content": "Sorry about that. Which GPU type and batch size were you using, and did memory grow over time?"},
                          {"role": "user", "content": "H100 80GB, batch 32, and yes, nvidia-smi showed usage creeping up every epoch."},
                          {"role": "assistant", "content": "Growth across epochs usually means tensors are kept alive between steps, for example by accumulating losses without detaching them."},
                          {"role": "user", "content": "I do append loss to a list for logging. How should I fix it, in two steps?"}],
                         None),
        "chat-json": ([{"role": "system", "content": "Reply only with a JSON object with keys \"intent\", \"entities\" and \"priority\"."},
                       {"role": "user", "content": "Please move my Thursday 3pm dentist appointment to Friday morning, it's urgent."}],
                      None),
        "chat-code": ([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": "Why does the completions endpoint reject `echo`? Read the handler and explain."},
                       *tool_turn(0, "read_file", {"path": "crates/plowrt/src/serve/completion.rs"}, text_of("code", 2000, 0))],
                      TOOLS),
        "chat-research-4k": ([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": "Summarize how campaigns are supposed to measure performance against vLLM."},
                              *tool_turn(0, "search", {"query": "campaign playbook vLLM comparison"}, text_of("docs", 1800, 0)),
                              *tool_turn(1, "search", {"query": "measurement hazards prefix cache"}, text_of("docs", 1800, 40000))],
                             TOOLS),
        "chat-multi-8k": ([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": "I need a review of the serving code. Start with the request handler."},
                           *tool_turn(0, "read_file", {"path": "crates/plowrt/src/serve/openai.rs"}, text_of("code", 3000, 6000)),
                           {"role": "assistant", "content": "The handler parses OpenAI-compatible requests and validates sampling fields before scheduling. I will look at the docs next."},
                           {"role": "user", "content": "Good. Now check what the docs say about logprobs and then list three risks."},
                           *tool_turn(1, "search", {"query": "logprobs raw_logits"}, text_of("docs", 4000, 80000))],
                          TOOLS),
        "chat-book-12k": ([{"role": "user", "content": "Here is part of a travel journal.\n\n" + text_of("beagle", 12000, 120000)
                            + "\n\nWhich places does the author describe, and what struck him most about the people he met? Answer in a short paragraph."}],
                          None),
        "chat-long-15k": ([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": "Read the novel excerpt with the tool and tell me how Elizabeth's opinion of Darcy changes."},
                           *tool_turn(0, "read_file", {"path": "corpus/pride_and_prejudice.txt"}, text_of("pride", 15000, 60000))],
                          TOOLS),
    }


def cmd_prompts(a):
    from transformers import AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import glm53_needle_probe as needle
    tok = AutoTokenizer.from_pretrained(a.hf, local_files_only=True)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    corpora, sources = {}, {}
    for spec in a.corpus:
        name, path = spec.split("=", 1)
        corpora[name] = enc(Path(path).read_text(errors="replace"))
        sources[name] = dict(path=str(Path(path).resolve()), sha256=file_sha(path), tokens=len(corpora[name]))

    def text_of(name, n, offset):
        ids = corpora[name]
        if offset + n > len(ids):
            raise ValueError(f"corpus {name} has {len(ids)} tokens, need {offset + n}")
        return tok.decode(ids[offset:offset + n])

    cases = []
    lens = [int(x) for x in a.natural_lens.split(",")]
    for name in a.natural:
        for i, n in enumerate(lens):
            off = (i + 1) * 20011 % max(len(corpora[name]) - n, 1)  # distinct windows: no shared prefixes
            ids = [tok.bos_token_id] + corpora[name][off:off + n - 1]
            cases.append(dict(id=f"nat-{name}-{n}", kind="natural", prompt_ids=ids, max_new=a.natural_new))
    for cid, (msgs, tools) in chats(text_of).items():
        text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        cases.append(dict(id=cid, kind="chat", prompt_ids=enc(text), max_new=a.chat_new))
    prefix, ending = needle.chat_frame(a.hf)
    for n in [int(x) for x in a.needle_lens.split(",")]:
        for d in [float(x) for x in a.needle_depths.split(",")]:
            for nid, sentence, question, expect in needle.NEEDLES:
                q = question.split("\nA:")[0].removeprefix("Q: ") + "\nAnswer with only the requested value."
                ids = needle.exact_prompt(enc, n, d, sentence, q, prefix, ending)
                cases.append(dict(id=f"needle-{nid}-{n}-{d}", kind="needle", prompt_ids=ids, max_new=a.needle_new,
                                  expect=expect))
    if a.drop_over_ctx:
        over = [c for c in cases if len(c["prompt_ids"]) + c["max_new"] > a.max_ctx]
        for c in over:
            print(f"dropped {c['id']}: {len(c['prompt_ids'])} + {c['max_new']} exceeds --max-ctx {a.max_ctx}")
        cases = [c for c in cases if c not in over]
    for c in cases:
        if len(c["prompt_ids"]) + c["max_new"] > a.max_ctx:
            raise SystemExit(f"{c['id']}: {len(c['prompt_ids'])} + {c['max_new']} exceeds --max-ctx {a.max_ctx}")
        c["prompt_sha256_u32le"] = ids_sha(c["prompt_ids"])
    out = dict(schema="plow.fp32ref.prompts.v1", tokenizer=str(Path(a.hf).resolve()), sources=sources, cases=cases)
    Path(a.out).write_text(json.dumps(out))
    for c in cases:
        print(f"{c['id']:<28} {c['kind']:<8} {len(c['prompt_ids']):>6} +{c['max_new']}")
    print(f"{len(cases)} cases -> {a.out} sha256 {file_sha(a.out)}")


# ---------------------------------------------------------------- FP32 reference (GPU)
def fp32_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **_):
    """Exact FP32 attention in query chunks (head_dim 512 rules out the fused kernels at 16K).
    attention_mask is the sdpa-style mask: None = causal, bool = keep, float = additive."""
    import torch
    b, h, q, _d = query.shape
    kl = key.shape[2]
    rep = h // key.shape[1]
    key = key.repeat_interleave(rep, 1)
    value = value.repeat_interleave(rep, 1)
    out = torch.empty(b, h, q, value.shape[-1], dtype=torch.float32, device=query.device)
    kpos = torch.arange(kl, device=query.device)
    for s in range(0, q, 1024):
        e = min(q, s + 1024)
        sc = torch.matmul(query[:, :, s:e], key.transpose(2, 3)) * scaling
        if attention_mask is None:
            qpos = torch.arange(s, e, device=query.device) + (kl - q)
            sc.masked_fill_(kpos[None, :] > qpos[:, None], float("-inf"))
        elif attention_mask.dtype == torch.bool:
            sc.masked_fill_(~attention_mask[:, :, s:e], float("-inf"))
        else:
            sc += attention_mask[:, :, s:e]
        out[:, :, s:e] = torch.matmul(torch.softmax(sc, -1), value)
    return out.transpose(1, 2).contiguous(), None


def load_fp32(hf, device="cuda"):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register("sdpa", fp32_attention)
    cfg = AutoConfig.from_pretrained(hf).text_config
    if cfg.model_type == "gemma4_text":  # hub Gemma-4 (E2B/E4B per-layer inputs, 26B MoE, 31B)
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM as Gemma4UnifiedForCausalLM
    else:
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForCausalLM
    cfg._attn_implementation = "sdpa"
    cfg.dtype = torch.float32
    torch.set_default_dtype(torch.float32)
    with torch.device(device):
        model = Gemma4UnifiedForCausalLM(cfg).eval()
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    loaded, unused = set(), []
    files = sorted(Path(hf).glob("*.safetensors"))
    for path in files:
        with safe_open(str(path), "pt", device=device) as f:
            keys = set(f.keys())
            for k in sorted(keys):
                if not k.startswith("model.language_model.") or k.endswith("_scale"):
                    continue
                name = "model." + k.removeprefix("model.language_model.")
                if name not in params:  # e.g. k/v of KV-shared layers: the model never reads them
                    unused.append(k)
                    continue
                w = f.get_tensor(k).to(torch.float32)
                if k + "_scale" in keys:  # FP8 per-output-channel: dequantize exactly in FP32
                    w = w * f.get_tensor(k + "_scale").to(torch.float32)
                params[name].data.copy_(w.reshape(params[name].shape))
                loaded.add(name)
    model.tie_weights()
    loaded.add("lm_head.weight")
    unset = [n for n, _ in model.named_parameters() if n not in loaded]
    if unset:
        raise SystemExit(f"FP32 parameters not loaded from the checkpoint: {unset[:8]}")
    if unused:
        print(f"{len(unused)} checkpoint tensors have no FP32 model slot (unused by the model): {unused[:4]} ...")
    return model, [dict(file=str(p), sha256=file_sha(p)) for p in files]


def cmd_reference(a):
    import torch
    import transformers
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf, local_files_only=True)
    prompts = json.loads(Path(a.prompts).read_text())
    model, weights = load_fp32(a.hf, dev)
    cases = []
    t0 = time.time()
    with torch.no_grad():
        for c in prompts["cases"]:
            if a.only and c["id"] not in a.only.split(","):
                continue
            ids = c["prompt_ids"]
            x = torch.tensor([ids], device=dev)
            out = model(input_ids=x, use_cache=True, logits_to_keep=1)
            cont = []
            while True:
                t = int(out.logits[0, -1].argmax())
                cont.append(t)
                if t in EOS or len(cont) == c["max_new"]:
                    break
                out = model(input_ids=torch.tensor([[t]], device=dev), past_key_values=out.past_key_values, use_cache=True)
            del out
            # Teacher-forced pass over the full sequence: the scored distributions.
            logits = model(input_ids=torch.tensor([ids + cont[:-1]], device=dev), logits_to_keep=len(cont)).logits[0]
            lsm = torch.log_softmax(logits.float(), -1)
            v, i = lsm.topk(TOP)
            pos = [dict(top=[[int(t), float(l)] for t, l in zip(i[k].tolist(), v[k].tolist())],
                        margin=float(v[k, 0] - v[k, 1])) for k in range(len(cont))]
            mism = sum(p["top"][0][0] != t for p, t in zip(pos, cont))
            del logits, lsm
            if dev == "cuda":
                torch.cuda.empty_cache()
            row = dict(c, cont=cont, cont_text=tok.decode(cont), pos=pos, tf_argmax_mismatch=mism)
            if "expect" in c:
                row["ref_correct"] = c["expect"] in row["cont_text"]
            cases.append(row)
            print(f"{c['id']:<28} prompt {len(ids):>6} cont {len(cont):>3} min margin "
                  f"{min(p['margin'] for p in pos):.3f} tf-mismatch {mism} {time.time() - t0:.0f}s", flush=True)
    meta = dict(schema="plow.fp32ref.v1", prompts=str(Path(a.prompts).resolve()), prompts_sha256=file_sha(a.prompts),
                weights=weights, dtype="float32", tf32=False, attention="fp32 chunked math",
                dequant="fp8_e4m3 weight x per-channel scale in fp32", torch=torch.__version__,
                transformers=transformers.__version__, device=torch.cuda.get_device_name() if dev == "cuda" else f"cpu ({torch.get_num_threads()} threads)",
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), top=TOP)
    Path(a.out).write_text(json.dumps(dict(meta=meta, cases=cases)))
    print(f"reference: {len(cases)} cases -> {a.out} sha256 {file_sha(a.out)}")


def load_ref(path):
    ref = json.loads(Path(path).read_text())
    ref["sha256"] = file_sha(path)
    return ref


# ---------------------------------------------------------------- capture (served stack)
def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def parse_choice(choice):
    lp = choice["logprobs"]
    toks = [int(t.split(":")[1]) for t in lp["tokens"]]
    tops = [sorted(((int(k.split(":")[1]), float(v)) for k, v in d.items()), key=lambda x: -x[1])[:TOP]
            for d in lp["top_logprobs"]]
    return toks, [[list(x) for x in t] for t in tops]


def teacher_force(complete, prompt, cont):
    """Positions of `cont` scored on the exact reference history. complete(ids, n) -> (tokens,
    tops, text). Returns (free-run tokens, free text, {k: top}, requests, empty replies)."""
    pos, k, free, text, requests, empty = {}, 0, None, "", 0, 0
    while k < len(cont):
        toks, tops, txt = complete(prompt + cont[:k], len(cont) - k)
        requests += 1
        if not toks:
            # A reply with no tokens (plowrt drops a stop token without ignore_eos): retry, count it.
            empty += 1
            if empty > 3:
                raise RuntimeError(f"no tokens returned at position {k}")
            continue
        if free is None:
            free, text = toks, txt
        step = 0
        for t, top in zip(toks, tops):
            pos[k + step] = top
            step += 1
            if t != cont[k + step - 1]:
                break
        k += step
    return free, text, pos, requests, empty


def cmd_capture(a):
    ref = load_ref(a.ref)
    url = a.url.rstrip("/")
    model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]

    def complete(ids, n):
        r = post(url + "/v1/completions", {"model": model, "prompt": ids, "max_tokens": n, "temperature": 0,
                                           "logprobs": TOP, "return_tokens_as_token_ids": True,
                                           # plowrt omits a stop token from logprobs; the EOS position must be scored.
                                           "ignore_eos": True})
        c = r["choices"][0]
        if r["usage"]["prompt_tokens"] != len(ids):
            raise RuntimeError(f"server re-tokenized the prompt: {r['usage']['prompt_tokens']} != {len(ids)}")
        toks, tops = parse_choice(c)
        return toks, tops, c["text"]

    def one(case):
        t = time.time()
        try:
            free, text, pos, n, empty = teacher_force(complete, case["prompt_ids"], case["cont"])
            row = dict(free=free, free_text=text, pos={str(k): v for k, v in pos.items()}, requests=n, empty_replies=empty)
        except Exception as e:  # noqa: BLE001 - a failed case is scored as missing, not a crash
            row = dict(error=repr(e))
        print(f"  {a.arm} {case['id']:<28} {row.get('requests', 'ERR')} req {time.time() - t:.1f}s "
              f"{row.get('error', '')}", flush=True)
        return case["id"], row

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        rows = dict(pool.map(one, ref["cases"]))
    cap = dict(schema="plow.fp32ref.capture.v1", arm=a.arm, url=url, model=model, ref=str(Path(a.ref).resolve()),
               ref_sha256=ref["sha256"], concurrency=a.concurrency, elapsed_s=time.time() - t0,
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), cases=rows)
    Path(a.out).write_text(json.dumps(cap))
    print(f"capture {a.arm}: {len(rows)} cases, {sum(r.get('requests', 0) for r in rows.values())} requests, "
          f"{sum('error' in r for r in rows.values())} errors -> {a.out}")


# ---------------------------------------------------------------- report
def fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


COLS = ("positions", "kl_mean", "kl_p99", "kl_max", "decisive", "top1_decisive", "cont_frac", "cont_full",
        "cases", "needle_acc")


def table(scores):
    lines = ["| arm | kind | " + " | ".join(COLS) + " |", "|---|---|" + "---:|" * len(COLS)]
    for arm, s in scores.items():
        for kind, m in [("all", s), *s["by_kind"].items()]:
            lines.append(f"| {arm} | {kind} | " + " | ".join(fmt(m.get(c, "")) for c in COLS) + " |")
    return "\n".join(lines)


def thresholds(a):
    return {k: getattr(a, k) for k in DEFAULTS if getattr(a, k, None) is not None}


def cmd_score(a):
    ref = load_ref(a.ref)
    scores = {}
    for path in a.captures:
        cap = json.loads(Path(path).read_text())
        scores[cap["arm"]] = score(ref, cap, a.tie_margin if a.tie_margin is not None else DEFAULTS["tie_margin"])
    print(table(scores))


def cmd_gate(a):
    ref = load_ref(a.ref)
    th = dict(DEFAULTS, **thresholds(a))
    cand, peer = (json.loads(Path(p).read_text()) for p in (a.cand, a.peer))
    sc, sp = score(ref, cand, th["tie_margin"]), score(ref, peer, th["tie_margin"])
    why = verdict(sc, sp, th)
    rec = {"pass": not why, "why": why, "thresholds": th, "ref_sha256": ref["sha256"],
           "candidate": dict(arm=cand["arm"], **sc), "peer": dict(arm=peer["arm"], **sp)}
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(table({cand["arm"]: sc, peer["arm"]: sp}))
    print(f"\nverdict: {'PASS' if not why else 'FAIL'}" + "".join(f"\n  {w}" for w in why))
    return 0 if not why else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    pr = sp.add_parser("prompts")
    pr.add_argument("--hf", required=True)
    pr.add_argument("--corpus", nargs="+", required=True, metavar="NAME=FILE",
                    help="names used: pride, beagle, docs, code")
    pr.add_argument("--natural", nargs="+", default=["pride", "beagle", "docs", "code"])
    pr.add_argument("--natural-lens", default="128,1024,4096,8192,15872")
    pr.add_argument("--natural-new", type=int, default=32)
    pr.add_argument("--chat-new", type=int, default=64)
    pr.add_argument("--needle-lens", default="4096,15872")
    pr.add_argument("--needle-depths", default="0.1,0.5,0.9")
    pr.add_argument("--needle-new", type=int, default=16)
    pr.add_argument("--max-ctx", type=int, default=16384)
    pr.add_argument("--drop-over-ctx", action="store_true",
                    help="drop (and list) cases longer than --max-ctx instead of failing")
    pr.add_argument("--out", required=True)
    pr.set_defaults(f=cmd_prompts)
    rf = sp.add_parser("reference")
    rf.add_argument("--hf", required=True)
    rf.add_argument("--prompts", required=True)
    rf.add_argument("--only", help="comma-separated case ids (debug)")
    rf.add_argument("--device", default="cuda", help="torch device for the FP32 model (cuda or cpu)")
    rf.add_argument("--out", required=True)
    rf.set_defaults(f=cmd_reference)
    cp = sp.add_parser("capture")
    cp.add_argument("--url", required=True)
    cp.add_argument("--ref", required=True)
    cp.add_argument("--arm", required=True)
    cp.add_argument("--concurrency", type=int, default=16)
    cp.add_argument("--out", required=True)
    cp.set_defaults(f=cmd_capture)
    sc = sp.add_parser("score")
    sc.add_argument("--ref", required=True)
    sc.add_argument("--tie-margin", type=float)
    sc.add_argument("captures", nargs="+")
    sc.set_defaults(f=cmd_score)
    ga = sp.add_parser("gate")
    ga.add_argument("--ref", required=True)
    ga.add_argument("--cand", required=True)
    ga.add_argument("--peer", required=True)
    ga.add_argument("--out")
    for k in DEFAULTS:
        ga.add_argument("--" + k.replace("_", "-"), type=float, dest=k)
    ga.set_defaults(f=cmd_gate)
    a = p.parse_args()
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
