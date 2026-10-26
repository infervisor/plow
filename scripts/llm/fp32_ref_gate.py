#!/usr/bin/env python3
"""Numerics gate against an independent FP32 reference, with vLLM scored as a peer.

Bit-exact agreement with vLLM is not a quality bar: both stacks round differently from the
checkpoint's exact math, so near-tie flips fail any exact gate. Instead both are scored against
FP32 (FP8 checkpoint dequantized to FP32 weights, FP32 activations, TF32 off) on one fixed prompt
set, and the candidate passes if it is within the peer's distance to FP32 plus a tolerance.

  prompts   --hf DIR --corpus NAME=FILE ... --out prompts.json     fixed token-id prompt set (CPU)
  reference --hf DIR --prompts prompts.json --out ref.json         FP32 greedy continuation + top-20
            [--device cpu|offload]                                 (default cuda; offload keeps
                                                                   what fits on the GPU and streams
                                                                   the other decoder layers from
                                                                   pinned host memory, e.g. 31B)
  capture   --url URL --ref ref.json --arm NAME --out cap.json     a served stack, teacher-forced
            [--warm-turn]                                          (each case after its previous-turn prefix)
  score     --ref ref.json CAP...                                  metrics per capture
  gate      --ref ref.json --cand CAP... --peer CAP [thresholds]   verdict (exit 1 on fail); several CAPs pool

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
# Stop ids of a reference whose meta carries none (Gemma's, before `reference` read them from the
# checkpoint). A reference's continuation ends at its first stop token, and scoring stops there too:
# a served stack never produces the tokens after it, and the post-stop distribution is ill-conditioned
# (Qwen3.8 kept emitting <|im_start|><|im_end|> for 36 tokens past <|im_end|>, where FP32 was decisive
# with 1.6 nats and bf16 stacks disagreed by up to 1.2 nats of KL).
EOS = (1, 50, 106)
DEFAULTS = dict(tie_margin=1.0, kl_ratio_max=1.25, kl_slack_max=0.002, top1_drop_max=0.01,
                cont_drop_max=0.05, cont_decisive_drop_max=0.0, needle_drop_max=0.0, needle_min=0.9,
                kl_trim_frac=0.01, kl_over_at=0.1, kl_over_slack=0.005)
# Gated KL statistics. kl_p99 is reported, not gated: on 1414 positions it is the 14th-worst position,
# and one co-batching flip at serving concurrency moves it by 60% while the trimmed mean and p95 hold.
KL_GATED = ("kl_mean", "kl_trim_mean", "kl_p95")


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


def model_eos(hf):
    """The checkpoint's stop ids: generation_config.json's eos_token_id (an id or a list), else config.json's."""
    for name in ("generation_config.json", "config.json"):
        p = Path(hf) / name
        if p.is_file():
            e = json.loads(p.read_text()).get("eos_token_id")
            if e is not None:
                return sorted({int(x) for x in (e if isinstance(e, list) else [e])})
    raise SystemExit(f"{hf}: no eos_token_id in generation_config.json or config.json")


def scored_len(cont, eos):
    """Positions scored of a reference continuation: through its first stop token, else all of them."""
    for i, t in enumerate(cont):
        if t in eos:
            return i + 1
    return len(cont)


def score(ref, cap, tie_margin=DEFAULTS["tie_margin"], kl_trim_frac=DEFAULTS["kl_trim_frac"],
          kl_over_at=DEFAULTS["kl_over_at"], eos=None):
    """Metrics of one capture (or the pooled union of a list of captures of one arm) against the FP32
    reference, overall and per prompt kind.

    The scored positions are the reference's teacher-forced positions through its first stop token
    (`eos`, else the reference's meta, else EOS), the same set for every arm; a position the capture
    lacks counts as missing (and fails the gate), never as a smaller sample. A pooled list scores
    every case once per capture, so its fractions move in finer steps than one capture's.
    """
    caps = cap if isinstance(cap, list) else [cap]
    if not caps:
        raise ValueError("no capture to score")
    for c in caps:
        if c.get("ref_sha256") != ref["sha256"]:
            raise ValueError(f"capture {c.get('arm')} was taken against another reference")
    eos = set(eos if eos is not None else ref.get("meta", {}).get("eos", EOS))
    groups = {}
    missing = 0
    for case, c in ((case, cp["cases"].get(case["id"])) for cp in caps for case in ref["cases"]):
        g = groups.setdefault(case["kind"], dict(kl=[], dec=0, dec_ok=0, cont=[], cont_dec=[], full=0, n=0, needle=[]))
        g["n"] += 1
        L = scored_len(case["cont"], eos)
        cont, pos = case["cont"][:L], case["pos"][:L]
        if c is None or c.get("error"):
            missing += L
            g["cont"].append(0.0)
            g["cont_dec"].append(False)
            if "expect" in case:
                g["needle"].append(False)
            continue
        for k, p in enumerate(pos):
            top = c["pos"].get(str(k))
            if top is None:
                missing += 1
                continue
            g["kl"].append(kl_top(p["top"], top))
            if p["margin"] > tie_margin:
                g["dec"] += 1
                g["dec_ok"] += top[0][0] == p["top"][0][0]
        n = agree_len(c["free"], cont)
        g["cont"].append(n / L)
        g["full"] += n == L
        # A free-run divergence at a near tie (FP32 top-1/top-2 margin <= tie_margin) is rounding
        # luck, and greedy decoding makes everything after it incomparable; a divergence where FP32
        # was decisive is a real disagreement at an identical history.
        g["cont_dec"].append(n >= L or pos[n]["margin"] <= tie_margin)
        if "expect" in case:
            g["needle"].append(case["expect"] in c["free_text"])

    def summarize(gs):
        kl = [x for g in gs for x in g["kl"]]
        dec, dec_ok = sum(g["dec"] for g in gs), sum(g["dec_ok"] for g in gs)
        cont = [x for g in gs for x in g["cont"]]
        cont_dec = [x for g in gs for x in g["cont_dec"]]
        needle = [x for g in gs for x in g["needle"]]
        srt = sorted(kl)
        # At least one position trimmed when a fraction is asked for, none at 0, and never the
        # last one (a one-position reference, e.g. an immediate stop token, still scores).
        cut = 0 if kl_trim_frac <= 0 else max(1, round(kl_trim_frac * len(srt)))
        trim = srt[: len(srt) - min(cut, len(srt) - 1)] if srt else []
        over = sum(x > kl_over_at for x in kl)
        out = dict(positions=len(kl), kl_mean=sum(kl) / len(kl) if kl else float("nan"),
                   kl_trim_mean=sum(trim) / len(trim) if trim else float("nan"),
                   kl_p95=percentile(kl, 0.95), kl_p99=percentile(kl, 0.99), kl_max=max(kl, default=float("nan")),
                   kl_over=over, kl_over_frac=over / len(kl) if kl else float("nan"),
                   decisive=dec, top1_decisive=dec_ok / dec if dec else float("nan"),
                   cont_frac=sum(cont) / len(cont) if cont else float("nan"),
                   cont_decisive_frac=sum(cont_dec) / len(cont_dec) if cont_dec else float("nan"),
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
    for m in KL_GATED:
        lim = peer[m] * t["kl_ratio_max"] + t["kl_slack_max"]
        if not cand[m] <= lim:
            why.append(f"{m} {cand[m]:.4g} > peer {peer[m]:.4g} x {t['kl_ratio_max']} + {t['kl_slack_max']}")
    # Outlier count at a fixed KL: a handful of large disagreements that the mean and the quantiles
    # absorb still has to stay within the peer's count plus an absolute allowance.
    lim = peer["kl_over_frac"] + t["kl_over_slack"]
    if not cand["kl_over_frac"] <= lim:
        why.append(f"kl_over_frac {cand['kl_over_frac']:.4g} ({cand.get('kl_over', '?')} positions > {t['kl_over_at']}) "
                   f"> peer {peer['kl_over_frac']:.4g} + {t['kl_over_slack']}")
    if not cand["top1_decisive"] >= peer["top1_decisive"] - t["top1_drop_max"]:
        why.append(f"top1_decisive {cand['top1_decisive']:.4f} < peer {peer['top1_decisive']:.4f} - {t['top1_drop_max']}")
    if not cand["cont_frac"] >= peer["cont_frac"] - t["cont_drop_max"]:
        why.append(f"cont_frac {cand['cont_frac']:.4f} < peer {peer['cont_frac']:.4f} - {t['cont_drop_max']}")
    if not cand["cont_decisive_frac"] >= peer["cont_decisive_frac"] - t["cont_decisive_drop_max"]:
        why.append(f"cont_decisive_frac {cand['cont_decisive_frac']:.4f} < peer {peer['cont_decisive_frac']:.4f} "
                   f"- {t['cont_decisive_drop_max']} (free-run diverged where FP32 was decisive)")
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
    bos = [] if tok.bos_token_id is None else [tok.bos_token_id]  # GLM-5.3-Flash has no BOS
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
            bos = [] if tok.bos_token_id is None else [tok.bos_token_id]  # Qwen has no BOS
            ids = bos + corpora[name][off:off + n - len(bos)]
            cases.append(dict(id=f"nat-{name}-{n}", kind="natural", prompt_ids=ids, max_new=a.natural_new))
    for cid, (msgs, tools) in chats(text_of).items():
        text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        cases.append(dict(id=cid, kind="chat", prompt_ids=enc(text + a.gen_suffix), max_new=a.chat_new))
    prefix, ending = needle.chat_frame(a.hf)
    ending += a.gen_suffix
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
    out = dict(schema="plow.fp32ref.prompts.v1", tokenizer=str(Path(a.hf).resolve()), sources=sources,
               gen_suffix=a.gen_suffix, cases=cases)
    Path(a.out).write_text(json.dumps(out))
    for c in cases:
        print(f"{c['id']:<28} {c['kind']:<8} {len(c['prompt_ids']):>6} +{c['max_new']}")
    print(f"{len(cases)} cases -> {a.out} sha256 {file_sha(a.out)}")


# ---------------------------------------------------------------- FP32 reference (GPU)
# Keys past this take the long-context path of fp32_attention; SCORE_BYTES bounds one score chunk.
LONG_KV = 32768
SCORE_BYTES = 4 << 30


def fp32_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, **_):
    """Exact FP32 attention in query chunks (head_dim 512 rules out the fused kernels at 16K).
    attention_mask is the sdpa-style mask: None = causal (plus `sliding_window`; cmd_reference
    passes None masks for long prompts), bool = keep, float = additive. Long keys take narrower
    query chunks and a broadcast GQA product, so 262144 keys fit beside the weights."""
    import torch
    b, h, q, _d = query.shape
    kl = key.shape[2]
    kvh = key.shape[1]
    rep = h // kvh
    long_kv = kl > LONG_KV
    if not long_kv:
        key = key.repeat_interleave(rep, 1)
        value = value.repeat_interleave(rep, 1)
    out = torch.empty(b, h, q, value.shape[-1], dtype=torch.float32, device=query.device)
    kpos = torch.arange(kl, device=query.device)
    step = 1024 if not long_kv else max(16, min(1024, SCORE_BYTES // (h * kl * 4)))
    for s in range(0, q, step):
        e = min(q, s + step)
        if long_kv:
            qg = query[:, :, s:e].reshape(b, kvh, rep, e - s, _d)
            sc = torch.matmul(qg, key[:, :, None].transpose(3, 4)).reshape(b, h, e - s, kl) * scaling
        else:
            sc = torch.matmul(query[:, :, s:e], key.transpose(2, 3)) * scaling
        if attention_mask is None:
            qpos = torch.arange(s, e, device=query.device) + (kl - q)
            sc.masked_fill_(kpos[None, :] > qpos[:, None], float("-inf"))
            if sliding_window:
                sc.masked_fill_(kpos[None, :] <= qpos[:, None] - sliding_window, float("-inf"))
        elif attention_mask.dtype == torch.bool:
            sc.masked_fill_(~attention_mask[:, :, s:e], float("-inf"))
        else:
            sc += attention_mask[:, :, s:e]
        p = torch.softmax(sc, -1)
        del sc
        if long_kv:
            out[:, :, s:e] = torch.matmul(p.reshape(b, kvh, rep, e - s, kl), value[:, :, None]).reshape(b, h, e - s, -1)
        else:
            out[:, :, s:e] = torch.matmul(p, value)
    return out.transpose(1, 2).contiguous(), None


def dequant(w, scale):
    """Exact FP32 weight from checkpoint codes: per-output-channel scale ([N, 1], Gemma-4) or a
    [128, 128]-block scale grid (`_scale_inv`, GLM-5.3-Flash) expanded over the weight."""
    import torch
    w = w.to(torch.float32)
    if scale is None:
        return w
    scale = scale.to(torch.float32)
    if scale.shape[-2:] != (w.shape[-2], 1):
        scale = scale.repeat_interleave(128, -2).repeat_interleave(128, -1)[..., :w.shape[-2], :w.shape[-1]]
    return w * scale


def compact_experts_class():
    """Gemma-4 MoE experts kept in checkpoint storage (BF16, or FP8 + per-channel scale) and
    dequantized per expert to FP32 inside forward: the FP32 26B-A4B (103 GB) does not fit one GPU,
    and bf16 -> fp32 and fp8 x scale in fp32 are the exact weights a full-FP32 copy would hold."""
    import torch
    from torch import nn
    from transformers.activations import ACT2FN

    class CompactExperts(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.num_experts = config.num_experts
            self.act_fn = ACT2FN[config.hidden_activation]
            self.w, self.s = {}, {}
            # Empty stand-ins for the dense parameters transformers' init touches.
            self.register_buffer("gate_up_proj", torch.empty(0), persistent=False)
            self.register_buffer("down_proj", torch.empty(0), persistent=False)

        def load(self, which, w, scale):
            self.w[which] = w
            self.s[which] = None if scale is None else scale.to(torch.float32).reshape(w.shape[0], w.shape[1], 1)

        def weight(self, which, e):
            w = self.w[which][e].to(torch.float32)
            return w if self.s[which] is None else w * self.s[which][e]

        def forward(self, hidden_states, top_k_index, top_k_weights):
            out = torch.zeros_like(hidden_states)
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist():
                pos, tok = torch.where(mask[e])
                gate, up = nn.functional.linear(hidden_states[tok], self.weight("gate_up_proj", e)).chunk(2, dim=-1)
                h = nn.functional.linear(self.act_fn(gate) * up, self.weight("down_proj", e))
                out.index_add_(0, tok, h * top_k_weights[tok, pos, None])
            return out

    return CompactExperts


# GLM-5.3-Flash checkpoint names -> transformers' glm5_next modules (its from_pretrained renames).
GLM_RENAMES = (("self_attn.f_a_proj.", "self_attn.forget_gate.f_a_proj."),
               ("self_attn.f_b_proj.", "self_attn.forget_gate.f_b_proj."),
               ("self_attn.dt_bias", "self_attn.forget_gate.dt_bias"),
               ("self_attn.A_log", "self_attn.forget_gate.A_log"),
               ("hc_attn_", "attn_hc."), ("hc_ffn_", "ffn_hc."))


def glm_slot(name, w, scale, parts):
    """Checkpoint tensor -> glm5_next model slot: HF's renames, q/k/v conv1d and per-expert
    gate/up concatenated along dim 0 (exact: 2048 is a multiple of the 128 scale block). Returns
    None until a concatenation's other parts have arrived; `parts` holds the pending ones."""
    import torch
    for src, dst in GLM_RENAMES:
        name = name.replace(src, dst)
    parent, _, leaf = name.rpartition(".")
    stem, _, part = parent.rpartition(".")
    expert = leaf == "weight" and stem.rpartition(".")[0].endswith(".experts")
    if leaf == "weight" and (part.endswith("_conv1d") or expert and part != "down_proj"):
        order, target = ((("q_conv1d", "k_conv1d", "v_conv1d"), "conv1d.weight") if part.endswith("_conv1d")
                         else (("gate_proj", "up_proj"), "gate_up_proj"))
        group = parts.setdefault(stem, {})
        group[part] = (w, scale)
        if any(p not in group for p in order):
            return None
        del parts[stem]
        w = torch.cat([group[p][0] for p in order], 0)
        scale = None if scale is None else torch.cat([group[p][1] for p in order], 0)
        return f"{stem}.{target}", w, scale
    return (parent if expert else name), w, scale


def glm5_experts_class():
    """GLM-5.3-Flash routed experts (288 per layer, 300 GB of block-FP8 codes: no FP32 copy fits
    anywhere) kept per layer as one checkpoint-precision tensor, pinned in host memory under
    offload, and dequantized per hit expert on the compute device inside HF's own gate math."""
    import torch
    from torch import nn
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts

    class CompactExperts(Glm5NextTextExperts):
        def __init__(self, config):
            nn.Module.__init__(self)
            self.num_experts = config.num_local_experts
            self.swiglu_limit = config.swiglu_limit
            self.w, self.s = {}, {}
            self.register_buffer("gate_up_proj", torch.empty(0), persistent=False)
            self.register_buffer("down_proj", torch.empty(0), persistent=False)

        def load(self, which, e, w, scale, pin):
            if which not in self.w:
                self.w[which] = torch.empty((self.num_experts, *w.shape), dtype=w.dtype, device=w.device, pin_memory=pin)
                self.s[which] = None if scale is None else torch.empty(
                    (self.num_experts, *scale.shape), dtype=torch.float32, device=w.device, pin_memory=pin)
            self.w[which][e].copy_(w)
            if scale is not None:
                self.s[which][e].copy_(scale)

        def weight(self, which, e, device):
            s = self.s[which]
            return dequant(self.w[which][e].to(device, non_blocking=True), None if s is None else s[e].to(device))

        def forward(self, hidden_states, top_k_index, top_k_weights):
            dev = hidden_states.device
            final = torch.zeros_like(hidden_states)
            mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist():
                top_k_pos, token_idx = torch.where(mask[e])
                current = self._apply_gate(nn.functional.linear(hidden_states[token_idx], self.weight("gate_up_proj", e, dev)))
                current = nn.functional.linear(current, self.weight("down_proj", e, dev)) * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, current)
            return final

    return CompactExperts


def glm5_text_lm_class():
    """Text-only causal LM over Glm5NextTextModel: the hub Glm5NextForConditionalGeneration also
    instantiates the vision tower, which the reference never runs."""
    from torch import nn
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast
    from transformers.models.glm5_next import modeling_glm5_next as m

    class Glm5NextTextForCausalLM(m.Glm5NextPreTrainedModel):
        config: m.Glm5NextTextConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = m.Glm5NextTextModel(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=None, logits_to_keep=0, **kwargs):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                             use_cache=use_cache, **kwargs)
            logits = self.lm_head(out.last_hidden_state[:, slice(-logits_to_keep, None)])
            return MoeCausalLMOutputWithPast(logits=logits, past_key_values=out.past_key_values)

    return Glm5NextTextForCausalLM


def scaled_tensors(files, device="cuda"):
    """Yield (key, tensor, scale or None) for each language-model tensor. Scales are resolved
    checkpoint-wide (a shard boundary may split a tensor from its scale); an FP8 tensor with no
    scale anywhere is refused rather than loaded as raw e4m3/e5m2 codes."""
    import contextlib
    import torch
    from safetensors import safe_open
    with contextlib.ExitStack() as stack:
        shards = [stack.enter_context(safe_open(str(p), "pt", device=device)) for p in files]
        where = {k: f for f in shards for k in f.keys()}
        for f in shards:
            for k in sorted(f.keys()):
                # The text tower, plus an untied output head (Qwen3.5: top-level lm_head.weight).
                if not (k.startswith("model.language_model.") or k == "lm_head.weight") or k.endswith(("weight_scale", "weight_scale_inv")):
                    continue
                # Experts: <...>.experts.gate_up_proj + ".weight_scale"; dense: <...>.weight + "_scale";
                # block FP8: <...>.weight + "_scale_inv".
                sk = k + (".weight_scale" if k.rpartition(".")[0].endswith(".experts") else "_scale")
                if sk not in where and k + "_scale_inv" in where:
                    sk = k + "_scale_inv"
                w = f.get_tensor(k)
                scale = where[sk].get_tensor(sk) if sk in where else None
                if scale is None and w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    raise SystemExit(f"FP8 tensor {k} has no scale {sk} in any checkpoint shard")
                yield k, w, scale


def offload_layers(model, compact, headroom_gib):
    """Decoder layers kept in checkpoint precision (`compact`: name -> (weight, scale or None),
    on the GPU while it fits beside `headroom_gib` of activations, else in pinned host memory) and
    materialized in FP32 on the GPU per forward with the same `w.float() * scale.float()` as the
    resident load, so the reference is bit-identical to an all-FP32 model. Everything else lives
    on the GPU in FP32."""
    import torch
    layers = model.model.layers
    for name, child in model.model.named_children():
        if name != "layers":
            child.to("cuda")
    model.lm_head.to("cuda")
    named = []
    for i, layer in enumerate(layers):
        ps = [(p, tuple(p.shape), compact.pop(f"model.layers.{i}.{n}")) for n, p in layer.named_parameters()]
        for p, *_ in ps:
            p.data = torch.empty(0)
        layer.to("cuda")  # buffers; the parameters are placeholders
        named.append(ps)
    per = max(sum(w.numel() * w.element_size() for _, _, (w, _) in ps) for ps in named)
    free = torch.cuda.mem_get_info()[0] - int(headroom_gib * 2**30)
    on_gpu = max(0, min(len(layers), free // per))
    store = []
    for i, ps in enumerate(named):
        dev = "cuda" if i < on_gpu else None
        row = []
        for p, shape, (w, scale) in ps:
            w = w.to("cuda") if dev else w.contiguous().pin_memory()
            scale = None if scale is None else scale.to("cuda", torch.float32)
            row.append((p, shape, w, scale))
        store.append(row)
    torch.cuda.empty_cache()

    def pre(j):
        def hook(_m, _a):
            for p, shape, w, scale in store[j]:
                p.data = dequant(w.to("cuda", non_blocking=True), scale).reshape(shape)
        return hook

    def post(j):
        def hook(_m, _a, _o):
            for p, *_ in store[j]:
                p.data = torch.empty(0, device="cuda")
        return hook

    for j, layer in enumerate(layers):
        layer.register_forward_pre_hook(pre(j))
        layer.register_forward_hook(post(j))
    print(f"offload: {on_gpu}/{len(layers)} decoder layers in checkpoint precision on the GPU, "
          f"{len(layers) - on_gpu} in pinned host memory ({per / 2**30:.2f} GiB each)", flush=True)


def load_fp32(hf, device="cuda", compact=None, headroom_gib=40.0):
    import torch
    if device == "offload":
        compact = {}
        model, weights = load_fp32(hf, "cpu", compact)
        offload_layers(model, compact, headroom_gib)
        return model, weights
    from transformers import AutoConfig
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register("sdpa", fp32_attention)
    cfg = AutoConfig.from_pretrained(hf).text_config
    moe = bool(getattr(cfg, "enable_moe_block", False))
    glm = cfg.model_type == "glm5_next_text"
    if glm:  # GLM-5.3-Flash: 34 KDA + 11 sparse-MLA layers, 288 block-FP8 experts per MoE layer
        # Import before the modeling module binds its kernels: HF's pure-torch KDA / conv1d paths.
        sys.modules["fla"] = sys.modules["causal_conv1d"] = None
        from transformers.models.glm5_next import modeling_glm5_next
        modeling_glm5_next.Glm5NextTextExperts = glm5_experts_class()
        model_class = glm5_text_lm_class()
    elif moe or cfg.model_type == "gemma4_text":  # hub Gemma-4 (E2B/E4B per-layer inputs, 26B MoE, 31B)
        from transformers.models.gemma4 import modeling_gemma4
        if moe:
            modeling_gemma4.Gemma4TextExperts = compact_experts_class()
        model_class = modeling_gemma4.Gemma4ForCausalLM
    elif cfg.model_type in ("qwen3_5", "qwen3_5_text"):  # Qwen3.5/3.8 hybrid GDN + GQA; text tower of the VL wrapper
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as model_class
    else:
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForCausalLM as model_class
    cfg._attn_implementation = "sdpa"
    cfg.dtype = torch.float32
    torch.set_default_dtype(torch.float32)
    with torch.device(device):
        model = model_class(cfg).eval()
    params = dict(model.named_parameters())
    weights_of = set(params)
    params.update(dict(model.named_buffers()))
    loaded, unused = set(), []
    experts = 0
    parts = {}
    files = sorted(Path(hf).glob("*.safetensors"))
    for k, w, scale in scaled_tensors(files, device):
        name = "model." + k.removeprefix("model.language_model.") if k.startswith("model.language_model.") else k
        if glm:
            slot = glm_slot(name, w, scale, parts)
            if slot is None:
                continue
            name, w, scale = slot
        parent, _, leaf = name.rpartition(".")
        if moe and leaf in ("gate_up_proj", "down_proj") and parent.endswith(".experts"):
            model.get_submodule(parent).load(leaf, w, scale)
            experts += 1
            continue
        if glm and leaf in ("gate_up_proj", "down_proj") and parent.rpartition(".")[0].endswith(".experts"):
            owner, _, e = parent.rpartition(".")
            if owner + ".gate_up_proj" not in params:  # the MTP layer's experts
                unused.append(k)
                continue
            model.get_submodule(owner).load(leaf, int(e), w, scale, pin=compact is not None)
            experts += 1
            continue
        if name not in params:  # e.g. k/v of KV-shared layers, the GLM MTP layer: the model never reads them
            unused.append(k)
            continue
        if compact is not None:  # offload: keep checkpoint precision, materialize per forward
            if name.startswith("model.layers.") and name in weights_of:
                compact[name] = (w, scale)
            else:
                params[name].data = dequant(w, scale).reshape(params[name].shape)
            loaded.add(name)
            continue
        params[name].data.copy_(dequant(w, scale).reshape(params[name].shape))
        loaded.add(name)
    model.tie_weights()
    if getattr(cfg, "tie_word_embeddings", True):
        loaded.add("lm_head.weight")
    unset = [n for n, _ in model.named_parameters() if n not in loaded]
    if unset:
        raise SystemExit(f"FP32 parameters not loaded from the checkpoint: {unset[:8]}")
    if parts:
        raise SystemExit(f"incomplete concatenated tensors: {sorted(parts)[:4]}")
    if unused:
        print(f"{len(unused)} checkpoint tensors have no FP32 model slot (unused by the model): {unused[:4]} ...")
    if moe and experts != 2 * cfg.num_hidden_layers:
        raise SystemExit(f"expert tensors loaded: {experts}, expected {2 * cfg.num_hidden_layers}")
    if glm and experts != (want := 2 * cfg.num_local_experts * cfg.mlp_layer_types.count("sparse")):
        raise SystemExit(f"expert tensors loaded: {experts}, expected {want}")
    return model, [dict(file=str(p), sha256=file_sha(p)) for p in files]


def cmd_reference(a):
    import torch
    import transformers
    dev = "cuda" if a.device == "offload" else a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf, local_files_only=True)
    eos = model_eos(a.hf)
    prompts = json.loads(Path(a.prompts).read_text())
    t_load = time.time()
    model, weights = load_fp32(a.hf, a.device, headroom_gib=a.offload_headroom_gib)
    print(f"FP32 model loaded in {time.time() - t_load:.0f}s", flush=True)
    # HF's sliding cache keeps its last-window slice as a view of the whole prefill K/V, so every
    # sliding layer pins prompt-length K/V (31B at 32K: 54 GB in FP32). Copy the window out.
    from transformers import cache_utils
    window_update = cache_utils.DynamicSlidingWindowLayer.update

    def update_owning_window(self, *args, **kwargs):
        out = window_update(self, *args, **kwargs)
        if getattr(self.keys, "_base", None) is not None:
            self.keys, self.values = self.keys.clone(), self.values.clone()
        return out
    cache_utils.DynamicSlidingWindowLayer.update = update_owning_window
    # Long prompts compute their masks inside fp32_attention (causal + sliding window): HF would
    # materialize a q x kv mask per layer type, 64 GiB at 262144 tokens.
    no_masks = {t: None for t in set(getattr(model.config, "layer_types", None) or ["full_attention"])}
    eos_cfg = model.config.eos_token_id
    eos = set(eos_cfg if isinstance(eos_cfg, list) else [eos_cfg] if eos_cfg is not None else []) | {tok.eos_token_id} - {None}
    if model.config.model_type.startswith("gemma"):  # EOS holds Gemma-4 ids; they are ordinary tokens elsewhere ('"', 'S' in GLM)
        eos |= set(EOS)
    cases = []
    t0 = time.time()
    with torch.no_grad():
        for c in prompts["cases"]:
            if a.only and c["id"] not in a.only.split(","):
                continue
            ids = c["prompt_ids"]
            x = torch.tensor([ids], device=dev)
            masks = no_masks if len(ids) > LONG_KV else None
            t_case = time.time()
            out = model(input_ids=x, attention_mask=masks, use_cache=True, logits_to_keep=1)
            cont = []
            while True:
                t = int(out.logits[0, -1].argmax())
                if not cont:
                    t_prefill, t_decode = time.time() - t_case, time.time()
                cont.append(t)
                if t in eos or len(cont) == c["max_new"]:
                    break
                out = model(input_ids=torch.tensor([[t]], device=dev), past_key_values=out.past_key_values,
                            attention_mask=masks, use_cache=True)
            t_decode = (time.time() - t_decode) / max(len(cont) - 1, 1)
            del out
            # Teacher-forced pass over the full sequence: the scored distributions.
            logits = model(input_ids=torch.tensor([ids + cont[:-1]], device=dev), attention_mask=masks,
                           logits_to_keep=len(cont)).logits[0]
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
                  f"{min(p['margin'] for p in pos):.3f} tf-mismatch {mism} prefill {t_prefill:.1f}s "
                  f"decode {t_decode:.2f}s/tok {time.time() - t0:.0f}s", flush=True)
    meta = dict(schema="plow.fp32ref.v1", prompts=str(Path(a.prompts).resolve()), prompts_sha256=file_sha(a.prompts),
                weights=weights, dtype="float32", tf32=False, attention="fp32 chunked math",
                dequant="fp8_e4m3 weight x per-channel or [128,128]-block scale in fp32", torch=torch.__version__,
                transformers=transformers.__version__, device=(torch.cuda.get_device_name() + (" + host offload" if a.device == "offload" else "")) if dev == "cuda" else f"cpu ({torch.get_num_threads()} threads)",
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), top=TOP, eos=eos)
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


def warm_prefix(ids, turn=105):
    """The prompt up to its last turn start (a previous turn's history), else its first 7/8: sent
    before a case under `--warm-turn`, so the case attaches from the prefix cache like a next turn."""
    starts = [i for i, x in enumerate(ids) if x == turn and i > 1]
    return ids[:starts[-1] if starts else max(1, len(ids) * 7 // 8)]


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
            if a.warm_turn:
                complete(warm_prefix(case["prompt_ids"]), 1)
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


COLS = ("positions", "kl_mean", "kl_trim_mean", "kl_p95", "kl_p99", "kl_max", "kl_over", "decisive",
        "top1_decisive", "cont_frac", "cont_decisive_frac", "cont_full", "cases", "needle_acc")


def table(scores):
    lines = ["| arm | kind | " + " | ".join(COLS) + " |", "|---|---|" + "---:|" * len(COLS)]
    for arm, s in scores.items():
        for kind, m in [("all", s), *s["by_kind"].items()]:
            lines.append(f"| {arm} | {kind} | " + " | ".join(fmt(m.get(c, "")) for c in COLS) + " |")
    return "\n".join(lines)


def thresholds(a):
    th = {k: getattr(a, k) for k in DEFAULTS if getattr(a, k, None) is not None}
    if getattr(a, "eos", None):
        th["eos"] = [int(x) for x in a.eos.split(",")]
    return th


def score_with(ref, cap, th):
    """score() under a threshold dict (tie margin, trim fraction, outlier level, stop ids)."""
    return score(ref, cap, th["tie_margin"], th["kl_trim_frac"], th["kl_over_at"], th.get("eos"))


def gate_scores(ref, cands, peer, th):
    """(candidate, peer) metrics for a gate on one or more pooled candidate captures. The single peer
    capture is replicated to the candidate's capture count: the trimmed mean and the percentiles
    depend on the sample size, so an unreplicated peer would fail an identical pooled distribution."""
    cands = cands if isinstance(cands, list) else [cands]
    return score_with(ref, cands, th), score_with(ref, [peer] * len(cands), th)


def cmd_score(a):
    ref = load_ref(a.ref)
    th = dict(DEFAULTS, **thresholds(a))
    scores = {}
    for path in a.captures:
        cap = json.loads(Path(path).read_text())
        scores[cap["arm"]] = score_with(ref, cap, th)
    print(table(scores))


def cmd_gate(a):
    ref = load_ref(a.ref)
    th = dict(DEFAULTS, **thresholds(a))
    cands = [json.loads(Path(p).read_text()) for p in a.cand]
    cand, peer = cands[0], json.loads(Path(a.peer).read_text())
    sc, sp = gate_scores(ref, cands, peer, th)
    why = verdict(sc, sp, th)
    rec = {"pass": not why, "why": why, "thresholds": th, "ref_sha256": ref["sha256"],
           "candidate": dict(arm=cand["arm"], captures=len(cands), **sc), "peer": dict(arm=peer["arm"], **sp)}
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
    pr.add_argument("--gen-suffix", default="",
                    help="appended after the generation prompt of chat and needle cases, e.g. '</think>' for "
                         "GLM-5.3-Flash, whose template always opens a reasoning block (needles must answer "
                         "within --needle-new tokens)")
    pr.add_argument("--max-ctx", type=int, default=16384)
    pr.add_argument("--drop-over-ctx", action="store_true",
                    help="drop (and list) cases longer than --max-ctx instead of failing")
    pr.add_argument("--out", required=True)
    pr.set_defaults(f=cmd_prompts)
    rf = sp.add_parser("reference")
    rf.add_argument("--hf", required=True)
    rf.add_argument("--prompts", required=True)
    rf.add_argument("--only", help="comma-separated case ids (debug)")
    rf.add_argument("--device", default="cuda", help="torch device for the FP32 model (cuda, cpu, or offload: GPU + pinned host layers)")
    rf.add_argument("--offload-headroom-gib", type=float, default=40.0,
                    help="offload: GPU memory left for activations (16K prompts fit in 40; 32K needs ~56)")
    rf.add_argument("--out", required=True)
    rf.set_defaults(f=cmd_reference)
    cp = sp.add_parser("capture")
    cp.add_argument("--url", required=True)
    cp.add_argument("--ref", required=True)
    cp.add_argument("--arm", required=True)
    cp.add_argument("--concurrency", type=int, default=16)
    cp.add_argument("--out", required=True)
    cp.add_argument("--warm-turn", action="store_true",
                    help="first send each case's prompt up to its last turn start (multi-turn prefix-cache form)")
    cp.set_defaults(f=cmd_capture)
    sc = sp.add_parser("score")
    sc.add_argument("--ref", required=True)
    for k in ("tie_margin", "kl_trim_frac", "kl_over_at"):
        sc.add_argument("--" + k.replace("_", "-"), type=float, dest=k)
    sc.add_argument("--eos", help="comma-separated stop ids for a reference whose meta carries none")
    sc.add_argument("captures", nargs="+")
    sc.set_defaults(f=cmd_score)
    ga = sp.add_parser("gate")
    ga.add_argument("--ref", required=True)
    ga.add_argument("--cand", required=True, nargs="+", help="one capture, or several of one arm scored pooled")
    ga.add_argument("--peer", required=True)
    ga.add_argument("--out")
    for k in DEFAULTS:
        ga.add_argument("--" + k.replace("_", "-"), type=float, dest=k)
    ga.add_argument("--eos", help="comma-separated stop ids for a reference whose meta carries none")
    ga.set_defaults(f=cmd_gate)
    a = p.parse_args()
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
