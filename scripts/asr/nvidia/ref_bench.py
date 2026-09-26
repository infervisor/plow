"""Reference ASR baselines on NVIDIA: Qwen3-ASR via vLLM, Nemotron 3.5 via transformers.

Writes one JSON with per-clip transcripts, latencies, WER and RTF. `--profile DIR` records
a CUPTI kernel trace of a short warm window and prints the top kernels by device time.
Run under gpulease.
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def wer(refs, hyps):
    import jiwer

    return jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps])


def load(manifest, limit):
    man = json.loads(Path(manifest).read_text())[: limit or None]
    for m in man:
        wav, sr = sf.read(m["path"], dtype="float32")
        assert sr == 16000
        m["wav"] = wav
    return man


def summarize(name, man, hyps, lat_s, total_s, extra=None):
    audio = sum(m["dur"] for m in man)
    out = {
        "system": name,
        "clips": len(man),
        "audio_s": round(audio, 3),
        "wer": round(wer([m["text"] for m in man], hyps), 5),
        "seq_total_s": round(sum(lat_s), 4),
        "seq_rtfx": round(audio / sum(lat_s), 2),
        "lat_p50_ms": round(1e3 * statistics.median(lat_s), 2),
        "lat_p90_ms": round(1e3 * float(np.percentile(lat_s, 90)), 2),
        "batch_total_s": round(total_s, 4) if total_s else None,
        "batch_rtfx": round(audio / total_s, 2) if total_s else None,
        "transcripts": [{"path": m["path"], "ref": m["text"], "hyp": h} for m, h in zip(man, hyps)],
    }
    out.update(extra or {})
    return out


def top_kernels(prof, n=40):
    rows = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            r = rows.setdefault(e.name, [0, 0.0])
            r[0] += 1
            r[1] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    tot = sum(v[1] for v in rows.values()) or 1
    ranked = sorted(rows.items(), key=lambda kv: -kv[1][1])[:n]
    return tot, [{"kernel": k[:160], "calls": c, "us": round(t, 1), "pct": round(100 * t / tot, 2)} for k, (c, t) in ranked]


def qwen(args, man):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=args.eager,
        max_num_seqs=max(64, len(man)),
        profiler_config={"profiler": "torch", "torch_profiler_dir": args.profile} if args.profile else None,
    )
    prompt = "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
    if args.language:
        prompt += f"language {args.language}<asr_text>"
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    req = lambda m: {"prompt": prompt, "multi_modal_data": {"audio": (m["wav"], 16000)}}

    def parse(t):
        return t.split("<asr_text>", 1)[1] if "<asr_text>" in t else t

    for m in man[:3]:
        llm.generate([req(m)], sp, use_tqdm=False)
    lat, hyps = [], []
    for m in man:
        t0 = time.perf_counter()
        o = llm.generate([req(m)], sp, use_tqdm=False)[0]
        lat.append(time.perf_counter() - t0)
        hyps.append(parse(o.outputs[0].text))
    t0 = time.perf_counter()
    llm.generate([req(m) for m in man], sp, use_tqdm=False)
    total = time.perf_counter() - t0
    if args.profile:
        llm.start_profile()
        llm.generate([req(man[0])], sp, use_tqdm=False)
        llm.stop_profile()
    return summarize("vllm-qwen3-asr", man, hyps, lat, total, {"eager": args.eager})


def nemotron(args, man):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    try:
        from transformers import AutoModelForRNNT as Auto
    except ImportError:
        Auto = AutoModelForSpeechSeq2Seq
    proc = AutoProcessor.from_pretrained(args.model)
    model = Auto.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).cuda().eval()
    lang = args.language or "en-US"
    if args.lookahead is not None:
        proc.set_num_lookahead_tokens(args.lookahead)

    def offline(wavs):
        x = proc([w for w in wavs], sampling_rate=16000, language=lang, return_tensors="pt").to("cuda")
        x = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in x.items() if hasattr(v, "to")}
        if args.lookahead is not None:
            x["num_lookahead_tokens"] = args.lookahead
        out = model.generate(**x)
        return proc.batch_decode(out.sequences if hasattr(out, "sequences") else out, skip_special_tokens=True)

    def stream(wav):
        n0, n1 = proc.num_samples_first_audio_chunk, proc.num_samples_per_audio_chunk
        hop = proc.feature_extractor.hop_length
        f0, f1 = proc.num_mel_frames_first_audio_chunk, proc.num_mel_frames_per_audio_chunk
        stride = f1 * hop
        prompt = proc([wav[:n0]], sampling_rate=16000, language=lang, return_tensors="pt")["prompt_ids"].cuda()

        def chunks():
            yield proc([wav[:n0]], sampling_rate=16000, language=lang, is_streaming=True,
                       is_first_audio_chunk=True, return_tensors="pt")["input_features"][:, :f0]
            # Subsequent chunk k covers mel frames f0 + (k-1)*f1 ... ; its first sample is (f0 + (k-1)*f1)*hop - win/2.
            start = (f0 * hop) - proc.feature_extractor.win_length // 2
            while start < len(wav):
                seg = wav[start : start + n1]
                if len(seg) < n1:
                    seg = np.pad(seg, (0, n1 - len(seg)))
                yield proc([seg], sampling_rate=16000, language=lang, is_streaming=True,
                           is_first_audio_chunk=False, return_tensors="pt")["input_features"][:, :f1]
                start += stride

        out = model.generate(input_features=chunks(), prompt_ids=prompt, num_lookahead_tokens=proc.default_num_lookahead_tokens)
        return proc.batch_decode(out.sequences, skip_special_tokens=True)[0]

    run1 = (lambda w: stream(w)) if args.stream else (lambda w: offline([w])[0])
    with torch.inference_mode():
        for m in man[:3]:
            run1(m["wav"])
        lat, hyps = [], []
        for m in man:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hyps.append(run1(m["wav"]))
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
        total = None
        if not args.stream:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(0, len(man), args.batch):
                offline([m["wav"] for m in man[i : i + args.batch]])
            torch.cuda.synchronize()
            total = time.perf_counter() - t0
        extra = {"dtype": args.dtype, "stream": args.stream, "lookahead": proc.default_num_lookahead_tokens,
                 "streaming_latency_ms": proc.streaming_latency_ms, "batch": args.batch}
        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run1(man[0]["wav"])
                torch.cuda.synchronize()
            tot, rows = top_kernels(prof)
            extra["profile"] = {"clip": man[0]["path"], "gpu_us": round(tot, 1), "kernels": rows}
    return summarize("hf-nemotron-3.5-asr", man, hyps, lat, total, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["qwen", "nemotron"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="/root/asr-work/audio/manifest.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--language")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--lookahead", type=int)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    man = load(args.manifest, args.limit)
    res = (qwen if args.family == "qwen" else nemotron)(args, man)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("transcripts", "profile")}))
    for r in res.get("profile", {}).get("kernels", [])[:25]:
        print(f'{r["pct"]:6.2f}% {r["us"]:10.1f}us {r["calls"]:5d}  {r["kernel"][:110]}')


if __name__ == "__main__":
    main()
