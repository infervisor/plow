#!/usr/bin/env python3
"""Served logprobs vs HF transformers (bf16) for a causal LM checkpoint (Gemma-4 E4B).

  plow <port> <hf_dir> <out.json>   query a running `plowrt serve` (chat + completions, greedy,
                                    logprobs top-20) and record tokens / logprobs / prompt ids
  hf   <hf_dir> <plow.json> <out.json>  HF bf16 on the GPU: chat-template ids, teacher-forced
                                    log-softmax at every position of plow's sequences, and HF's
                                    own greedy continuation
  report <plow.json> <hf.json>      the gate table

Metrics per case, over the generated positions (teacher-forced on plow's own tokens):
top-1 agreement, top-5 overlap, |dlogprob| of the chosen token, KL(HF||plow) over HF's top-20 with a
shared remainder bucket, rel-L2 of the top-20 logprob vector (HF ids), greedy prefix length where
plow and HF generate identical tokens, and chat-template token equality.
"""
import json
import math
import sys
import urllib.request

N_GEN = 64
TOP = 20
SYSTEM = "You are a friendly phone assistant for a dental clinic. Keep answers short and speakable."
LONG = " ".join(
    f"Record {i}: patient asked about appointment slot {i % 7} on day {i % 5}, insurance plan {i % 3}, "
    f"and whether the clinic is open on public holidays." for i in range(60))
CHATS = [
    [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Hi, I'd like to book a cleaning next Tuesday afternoon."}],
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": "Do you take walk-ins?"},
     {"role": "assistant", "content": "We do, but booked patients go first. Would you like me to book you a slot?"},
     {"role": "user", "content": "Yes please, tomorrow morning if possible."}],
    [{"role": "user", "content": "Explain in three short sentences why the sky is blue."}],
    [{"role": "user", "content": "Summarize these call notes in two sentences:\n" + LONG}],
    [{"role": "user", "content": "Traduce al inglés: 'Necesito cambiar mi cita del jueves.'"}],
]
COMPLETIONS = [
    "The quick brown fox",
    "def fibonacci(n):\n",
    "Paris is the capital of",
]


def post(port, path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))


def cmd_plow(port, hf_dir, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]
    cases = []
    for i, msgs in enumerate(CHATS):
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        r = post(port, "/v1/chat/completions", {
            "model": model, "messages": msgs, "max_tokens": N_GEN, "temperature": 0,
            "logprobs": True, "top_logprobs": TOP, "return_tokens_as_token_ids": True})
        c = r["choices"][0]
        steps = [{"id": int(e["token"].split(":")[1]), "lp": e["logprob"],
                  "top": [[int(t["token"].split(":")[1]), t["logprob"]] for t in e["top_logprobs"]]}
                 for e in c["logprobs"]["content"]]
        cases.append({"name": f"chat{i}", "prompt_ids": ids, "plow_prompt_tokens": r["usage"]["prompt_tokens"],
                      "text": c["message"]["content"], "steps": steps, "finish": c["finish_reason"]})
    for i, p in enumerate(COMPLETIONS):
        ids = [tok.bos_token_id] + tok(p, add_special_tokens=False)["input_ids"]
        r = post(port, "/v1/completions", {
            "model": model, "prompt": ids, "max_tokens": N_GEN, "temperature": 0, "logprobs": TOP,
            "return_tokens_as_token_ids": True})
        c = r["choices"][0]
        lp = c["logprobs"]
        steps = [{"id": int(t.split(":")[1]), "lp": l,
                  "top": sorted([[int(k.split(":")[1]), v] for k, v in d.items()], key=lambda x: -x[1])}
                 for t, l, d in zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"])]
        cases.append({"name": f"cmpl{i}", "prompt_ids": ids, "plow_prompt_tokens": r["usage"]["prompt_tokens"],
                      "text": c["text"], "steps": steps, "finish": c["finish_reason"]})
    # raw_logits mode: the same first position, values are logits.
    r = post(port, "/v1/completions", {"model": model, "prompt": [tok.bos_token_id] + tok(COMPLETIONS[2], add_special_tokens=False)["input_ids"],
                                       "max_tokens": 1, "temperature": 0, "logprobs": 5,
                                       "logprobs_mode": "raw_logits", "return_tokens_as_token_ids": True})
    raw = r["choices"][0]["logprobs"]
    json.dump({"model": model, "cases": cases,
               "raw_logits": {"prompt_ids": [tok.bos_token_id] + tok(COMPLETIONS[2], add_special_tokens=False)["input_ids"], "tokens": raw["tokens"],
                              "values": raw["token_logprobs"], "top": raw["top_logprobs"]}},
              open(out, "w"))
    print(f"plow: {len(cases)} cases -> {out}")


def load_hf(hf_dir):
    import torch
    import transformers
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            m = getattr(transformers, cls).from_pretrained(hf_dir, dtype=torch.bfloat16).to("cuda")
            return m.eval()
        except Exception as e:  # noqa: BLE001 - try the next head
            print(f"{cls}: {e}", file=sys.stderr)
    raise SystemExit("no HF head loads this checkpoint")


def cmd_hf(hf_dir, plow_json, out):
    import torch
    p = json.load(open(plow_json))
    model = load_hf(hf_dir)
    res = []
    with torch.no_grad():
        for c in p["cases"]:
            prompt = c["prompt_ids"]
            gen = [s["id"] for s in c["steps"]]
            x = torch.tensor([prompt + gen], device="cuda")
            logits = model(input_ids=x).logits[0].float()
            lsm = torch.log_softmax(logits, -1)
            pos = []
            for k, tok in enumerate(gen):
                row = lsm[len(prompt) - 1 + k]
                v, i = row.topk(TOP)
                pos.append({"lp": row[tok].item(), "top": [[int(a), float(b)] for a, b in zip(i.tolist(), v.tolist())]})
            g = model.generate(torch.tensor([prompt], device="cuda"), max_new_tokens=N_GEN, do_sample=False)
            res.append({"name": c["name"], "pos": pos, "greedy": g[0, len(prompt):].tolist()})
        rl = p["raw_logits"]
        logits = model(input_ids=torch.tensor([rl["prompt_ids"]], device="cuda")).logits[0, -1].float()
        v, i = logits.topk(5)
        raw = [[int(a), float(b)] for a, b in zip(i.tolist(), v.tolist())]
    json.dump({"cases": res, "raw_logits_top5": raw}, open(out, "w"))
    print(f"hf: {len(res)} cases -> {out}")


def cmd_report(plow_json, hf_json):
    p, h = json.load(open(plow_json)), json.load(open(hf_json))
    hf = {c["name"]: c for c in h["cases"]}
    print("| case | prompt ids (HF/plow) | gen | top1 agree | top5 overlap | mean/max abs dlp | KL(HF||plow) mean/max | rel-L2 top20 | greedy identical |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot = {"t1": 0, "n": 0, "o5": 0.0, "kl": []}
    for c in p["cases"]:
        r = hf[c["name"]]
        n = len(c["steps"])
        t1 = o5 = 0
        dl, kls, rl2 = [], [], []
        for s, q in zip(c["steps"], r["pos"]):
            pt = {t: v for t, v in s["top"]}
            ht = {t: v for t, v in q["top"]}
            t1 += (s["top"][0][0] == q["top"][0][0]) if s["top"] else 0
            o5 += len({t for t, _ in s["top"][:5]} & {t for t, _ in q["top"][:5]}) / 5
            dl.append(abs(s["lp"] - q["lp"]))
            # KL over HF's top-20 plus one remainder bucket; a plow miss takes plow's 20th value.
            floor = min(pt.values()) if pt else -30.0
            kl, ph_rest, pp_rest = 0.0, 1.0, 1.0
            for t, v in ht.items():
                pv = pt.get(t, floor)
                kl += math.exp(v) * (v - pv)
                ph_rest -= math.exp(v)
                pp_rest -= math.exp(pv)
            if ph_rest > 1e-9:
                kl += ph_rest * (math.log(ph_rest) - math.log(max(pp_rest, 1e-9)))
            kls.append(kl)
            hv = [v for _, v in q["top"]]
            pv = [pt.get(t, floor) for t, _ in q["top"]]
            rl2.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(hv, pv)) / max(sum(a * a for a in hv), 1e-12)))
        g = [s["id"] for s in c["steps"]]
        same = next((k for k, (a, b) in enumerate(zip(g, r["greedy"])) if a != b), min(len(g), len(r["greedy"])))
        tot["t1"] += t1; tot["n"] += n; tot["o5"] += o5; tot["kl"] += kls
        print(f"| {c['name']} | {len(c['prompt_ids'])}/{c['plow_prompt_tokens']} | {n} | {t1}/{n} | {o5 / max(n, 1):.3f} | "
              f"{sum(dl) / max(n, 1):.4f}/{max(dl, default=0):.4f} | {sum(kls) / max(n, 1):.2e}/{max(kls, default=0):.2e} | "
              f"{sum(rl2) / max(n, 1):.4f} | {same}/{len(g)} |")
    print(f"\nall: top1 {tot['t1']}/{tot['n']} ({tot['t1'] / tot['n']:.4f}), top5 overlap {tot['o5'] / tot['n']:.4f}, "
          f"KL mean {sum(tot['kl']) / len(tot['kl']):.2e} max {max(tot['kl']):.2e}")
    rl = p["raw_logits"]
    print(f"raw_logits mode: plow first token {rl['tokens'][0]} value {rl['values'][0]:.4f}; "
          f"plow top5 {rl['top'][0]}; HF top5 logits {h['raw_logits_top5']}")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    {"plow": cmd_plow, "hf": cmd_hf, "report": cmd_report}[cmd](*a)
