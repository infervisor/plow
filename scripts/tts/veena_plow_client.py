#!/usr/bin/env python3
"""Veena through a running `plowrt serve` (/v1/completions, token-id prompts).

Greedy gate: compares generated audio-code ids with a reference JSON written by
veena_ref.py (`tokens` map), reporting exact-match length per prompt, and writes
wavs (SNAC decode on the local GPU, optional) for the ASR gate.

  python scripts/tts/veena_plow_client.py --port P --ref $RESULTS/veena/vllm_c1.json --out DIR
"""
import argparse, concurrent.futures as cf, json, os, statistics, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from veena_ref import PROMPTS, EOA, EOS_SPEECH, build_prompt_ids, max_new, snac_decode, SR


def complete(port, model, ids, max_tokens, greedy):
    body = dict(model=model, prompt=ids, max_tokens=max_tokens, stop_token_ids=[EOS_SPEECH, EOA],
                return_token_ids=True)
    body.update(dict(temperature=0.0) if greedy else dict(temperature=0.4, top_p=0.9))
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r, time.perf_counter() - t0


def token_ids(resp):
    """Generated ids of a /v1/completions response with `return_token_ids` (plowrt: top-level)."""
    return resp["token_ids"]["completion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--wav", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    tok = AutoTokenizer.from_pretrained(snapshot_download("maya-research/Veena"))
    model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models"))["data"][0]["id"]
    reqs = []
    for i in range(args.n):
        spk, text = PROMPTS[i % len(PROMPTS)]
        reqs.append(dict(i=i, speaker=spk, text=text, ids=build_prompt_ids(tok, spk, text)))
    warm, _ = complete(args.port, model, reqs[0]["ids"], 16, True)
    json.dump(warm, open(f"{args.out}/sample_response.json", "w"), indent=1)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(args.conc) as ex:
        outs = list(ex.map(lambda r: complete(args.port, model, r["ids"], max_new(r["text"]), not args.sample), reqs))
    wall = time.perf_counter() - t0
    ref = json.load(open(args.ref))["tokens"] if args.ref else {}
    rows = []
    snac = None
    if args.wav:
        import torch
        from snac import SNAC
        snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    for r, (resp, lat) in zip(reqs, outs):
        ch = resp["choices"][0]
        gen = token_ids(resp)
        gen = [x for x in gen if x not in (EOA, EOS_SPEECH)]
        rt = ref.get(str(r["i"]))
        agree = None
        if rt is not None:
            agree = next((k for k, (a, b) in enumerate(zip(gen, rt)) if a != b), min(len(gen), len(rt)))
        audio_s = (len([x for x in gen if x >= 128266]) // 7) * 2048 / SR
        rows.append(dict(i=r["i"], ntok=len(gen), ref_ntok=len(rt) if rt else None, agree_prefix=agree,
                         latency_s=lat, audio_s=audio_s, finish=ch.get("finish_reason")))
        if snac is not None:
            import soundfile as sf
            a = snac_decode(snac, gen, "cuda")
            if a is not None:
                sf.write(f"{args.out}/plow_c{args.conc}_{r['i']:02d}_{r['speaker']}.wav", a, SR)
        print(json.dumps(rows[-1]))
    tot_audio = sum(x["audio_s"] for x in rows)
    summ = dict(engine="plow", conc=args.conc, n=len(rows), audio_s=tot_audio, wall_s=wall,
                throughput_audio_s_per_s=tot_audio / wall,
                med_latency_s=statistics.median(x["latency_s"] for x in rows),
                med_rtf=statistics.median(x["latency_s"] / x["audio_s"] for x in rows if x["audio_s"]))
    json.dump(dict(summary=summ, rows=rows, tokens={r["i"]: token_ids(o[0]) for r, o in zip(reqs, outs)}),
              open(f"{args.out}/plow_c{args.conc}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
