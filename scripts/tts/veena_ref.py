#!/usr/bin/env python3
"""Veena (maya-research/Veena) reference TTS: HF transformers and vLLM engines.

Serves as (a) the numerics oracle for the plow bring-up (greedy token ids) and
(b) the baseline for the plow-vs-existing-method performance comparison.

Run under a GPU lease, one GPU:
  perf-data/tools/gpulease -n 1 veena-ref /root/tts-work/venv-ref/bin/python \
      scripts/tts/veena_ref.py --engine hf --conc 1 --out $RESULTS/veena_hf_c1

Metrics per request: ttft_s (request -> first generated token), ttfa_s (first
audio = first 7*FIRST_FRAMES codes generated + SNAC decode; measured on the
non-streaming path as ttft + FIRST_FRAMES/decode_rate is NOT assumed, it is
only reported for engines that stream), lm_s, snac_s, total_s, audio_s, rtf.
"""
import argparse, json, os, statistics, sys, time

SOH, EOH, SOA, EOA, SOS, EOS_SPEECH = 128259, 128260, 128261, 128262, 128257, 128258
AUDIO_BASE = 128266
N_AUDIO = 7 * 4096
SR = 24000

PROMPTS = [
    ("kavya", "आज मैंने एक नई तकनीक के बारे में सीखा जो कृत्रिम बुद्धिमत्ता का उपयोग करके मानव जैसी आवाज़ उत्पन्न कर सकती है।"),
    ("agastya", "Today I learned about a new technology that uses artificial intelligence to generate human-like voices."),
    ("maitri", "मैं तो पूरा presentation prepare कर चुका हूं! कल रात को ही मैंने पूरा code base चेक किया।"),
    ("vinaya", "The quick brown fox jumps over the lazy dog, and then it takes a long nap in the warm afternoon sun."),
    ("kavya", "Hello, how are you doing today?"),
    ("agastya", "नमस्ते, आप कैसे हैं? मुझे आशा है कि आपका दिन अच्छा गुजर रहा है।"),
    ("maitri", "Please confirm your appointment for tomorrow at three thirty in the afternoon."),
    ("vinaya", "भारत एक विशाल और विविधताओं से भरा हुआ देश है जहाँ अनेक भाषाएँ बोली जाती हैं।"),
]


def build_prompt_ids(tok, speaker, text):
    ids = tok.encode(f"<spk_{speaker}> {text}", add_special_tokens=False)
    return [SOH, *ids, EOH, SOA, SOS]


def max_new(text):
    return min(int(len(text) * 1.3) * 7 + 21, 700)


def snac_decode(snac, codes, device):
    import torch
    codes = [c for c in codes if AUDIO_BASE <= c < AUDIO_BASE + N_AUDIO]
    codes = codes[: len(codes) // 7 * 7]
    if not codes:
        return None
    l0, l1, l2 = [], [], []
    for i in range(0, len(codes), 7):
        f = [codes[i + k] - (AUDIO_BASE + k * 4096) for k in range(7)]
        l0.append(f[0]); l1 += [f[1], f[4]]; l2 += [f[2], f[3], f[5], f[6]]
    t = [torch.tensor(x, dtype=torch.int32, device=device).unsqueeze(0) for x in (l0, l1, l2)]
    with torch.no_grad():
        a = snac.decode(t)
    return a.squeeze().clamp(-1, 1).float().cpu().numpy()


def sync():
    import torch
    torch.cuda.synchronize()


def run_hf(args, reqs, tok):
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.generation.streamers import BaseStreamer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()

    class First(BaseStreamer):
        def __init__(self): self.t = None; self.n = 0
        def put(self, v):
            self.n += 1
            if self.n == 2 and self.t is None:  # n==1 is the prompt echo
                self.t = time.perf_counter()
        def end(self): pass

    results = []
    C = args.conc
    for b in range(0, len(reqs), C):
        batch = reqs[b:b + C]
        maxlen = max(len(r["ids"]) for r in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.tensor([[pad] * (maxlen - len(r["ids"])) + r["ids"] for r in batch]).cuda()
        att = torch.tensor([[0] * (maxlen - len(r["ids"])) + [1] * len(r["ids"]) for r in batch]).cuda()
        st = First()
        sync(); t0 = time.perf_counter()
        gen_kw = dict(do_sample=False) if args.greedy else dict(do_sample=True, temperature=0.4, top_p=0.9, repetition_penalty=1.05)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=att, max_new_tokens=max(max_new(r["text"]) for r in batch),
                                 pad_token_id=pad, eos_token_id=[EOS_SPEECH, EOA], streamer=st if C == 1 else None, **gen_kw)
        sync(); t1 = time.perf_counter()
        for r, row in zip(batch, out):
            gen = row[maxlen:].tolist()
            for k, x in enumerate(gen):
                if x in (EOS_SPEECH, EOA):
                    gen = gen[:k]; break
            results.append(dict(r, gen=gen, t_start=t0, t_end=t1, ttft=(st.t - t0) if st.t else None))
    return results


def run_vllm(args, reqs, tok):
    from vllm import LLM, SamplingParams
    llm = LLM(args.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=args.gpu_mem,
              enable_prefix_caching=False, max_num_seqs=max(args.conc, 8))
    kw = dict(temperature=0.0) if args.greedy else dict(temperature=0.4, top_p=0.9, repetition_penalty=1.05)
    results = []
    C = args.conc
    warm = SamplingParams(max_tokens=8, **kw)
    llm.generate([{"prompt_token_ids": reqs[0]["ids"]}], warm, use_tqdm=False)
    for b in range(0, len(reqs), C):
        batch = reqs[b:b + C]
        sps = [SamplingParams(max_tokens=max_new(r["text"]), stop_token_ids=[EOS_SPEECH, EOA], **kw) for r in batch]
        t0 = time.perf_counter()
        outs = llm.generate([{"prompt_token_ids": r["ids"]} for r in batch], sps, use_tqdm=False)
        t1 = time.perf_counter()
        for r, o in zip(batch, outs):
            m = o.metrics
            ttft = None
            if m is not None and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
                ttft = m.first_token_time - m.arrival_time
            gen = [x for x in o.outputs[0].token_ids if x not in (EOS_SPEECH, EOA)]
            results.append(dict(r, gen=gen, t_start=t0, t_end=t1, ttft=ttft))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "vllm"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--n", type=int, default=16, help="total requests (cycled over PROMPTS)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    if args.model is None:
        from huggingface_hub import snapshot_download
        args.model = snapshot_download("maya-research/Veena")
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    reqs = []
    for i in range(args.n):
        spk, text = PROMPTS[i % len(PROMPTS)]
        reqs.append(dict(i=i, speaker=spk, text=text, ids=build_prompt_ids(tok, spk, text)))

    import torch, soundfile as sf
    from snac import SNAC
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

    # warm-up pass (excluded): first call pays CUDA context / kernel selection
    res = (run_hf if args.engine == "hf" else run_vllm)(args, reqs, tok)

    rows = []
    for r in res:
        sync(); t = time.perf_counter()
        a = snac_decode(snac, r["gen"], "cuda")
        sync(); snac_s = time.perf_counter() - t
        dur = 0 if a is None else len(a) / SR
        lm_s = r["t_end"] - r["t_start"]
        if a is not None and not args.no_audio:
            sf.write(f"{args.out}/{args.engine}_c{args.conc}_{r['i']:02d}_{r['speaker']}.wav", a, SR)
        rows.append(dict(i=r["i"], speaker=r["speaker"], ntok=len(r["gen"]), audio_s=dur, lm_s=lm_s,
                         snac_s=snac_s, ttft_s=r["ttft"], rtf=(lm_s + snac_s) / dur if dur else None,
                         gen_head=r["gen"][:14]))
    tot_audio = sum(x["audio_s"] for x in rows)
    wall = max(r["t_end"] for r in res) - min(r["t_start"] for r in res)
    summ = dict(engine=args.engine, conc=args.conc, n=len(rows), greedy=args.greedy,
                audio_s=tot_audio, lm_wall_s=wall,
                throughput_audio_s_per_s=tot_audio / wall if wall else None,
                med_rtf=statistics.median(x["rtf"] for x in rows if x["rtf"]),
                med_lm_s=statistics.median(x["lm_s"] for x in rows),
                med_ttft_s=statistics.median(x["ttft_s"] for x in rows if x["ttft_s"]) if any(x["ttft_s"] for x in rows) else None)
    json.dump(dict(summary=summ, rows=rows, tokens={r["i"]: r["gen"] for r in res}), open(f"{args.out}/{args.engine}_c{args.conc}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
