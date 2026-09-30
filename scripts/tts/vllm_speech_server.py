#!/usr/bin/env python3
"""The existing-method baseline for Veena: vLLM (async engine, CUDA graphs) + PyTorch SNAC behind
the same OpenAI `POST /v1/audio/speech` contract plowrt serves (Orpheus-FastAPI style).

Same prompt framing, stop ids, token budget and sampling defaults as plowrt's speech pipeline;
streaming uses the same window/lookahead plan (6 frames, 2 lookahead). Same client:
scripts/tts/tts_bench.py.

  gpulease -n 1 vllm-speech python scripts/tts/vllm_speech_server.py --port 8700
"""
import argparse, asyncio, os, struct, sys, time, uuid

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

sys.path.insert(0, os.path.dirname(__file__))
from veena_ref import AUDIO_BASE, EOA, EOS_SPEECH, SOA, SOH, EOH, SOS, max_new

SR = 24000
WINDOW, LOOKAHEAD = 6, 2


def wav_header(data_bytes):
    return (b"RIFF" + struct.pack("<I", min(data_bytes + 36, 0xFFFFFFFF)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) + b"data" + struct.pack("<I", data_bytes))


def pcm16(x):
    return (np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--model", default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--max-seqs", type=int, default=64)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from snac import SNAC
    model = args.model or snapshot_download("maya-research/Veena")
    tok = AutoTokenizer.from_pretrained(model)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=False, max_num_seqs=args.max_seqs))
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    snac_lock = asyncio.Lock()

    def decode_frames(codes, frames):
        l0, l1, l2 = [], [], []
        for f in range(frames):
            c = codes[f * 7:(f + 1) * 7]
            l0.append(c[0]); l1 += [c[1], c[4]]; l2 += [c[2], c[3], c[5], c[6]]
        t = [torch.tensor(x, dtype=torch.int32, device="cuda").unsqueeze(0) for x in (l0, l1, l2)]
        with torch.inference_mode():
            return snac.decode(t).squeeze().float().cpu().numpy()

    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "veena"}]}

    @app.post("/v1/audio/speech")
    async def speech(req: Request):
        body = await req.json()
        text, voice = body["input"], body.get("voice", "kavya")
        stream, wav = bool(body.get("stream")), body.get("response_format", "wav") == "wav"
        ids = [SOH, *tok.encode(f"<spk_{voice}> {text}", add_special_tokens=False), EOH, SOA, SOS]
        sp = SamplingParams(temperature=body.get("temperature", 0.4), top_p=body.get("top_p", 0.9),
                            max_tokens=body.get("max_tokens") or max_new(text), stop_token_ids=[EOS_SPEECH, EOA],
                            seed=body.get("seed"))
        gen = engine.generate({"prompt_token_ids": ids}, sp, request_id=uuid.uuid4().hex)

        async def codes_stream():
            codes, seen = [], 0
            async for out in gen:
                new = out.outputs[0].token_ids[seen:]
                seen = len(out.outputs[0].token_ids)
                for t in new:
                    lo = AUDIO_BASE + (len(codes) % 7) * 4096
                    if lo <= t < lo + 4096:
                        codes.append(t - lo)
                yield codes, out.finished

        if not stream:
            codes = []
            async for c, _ in codes_stream():
                codes = c
            frames = len(codes) // 7
            if frames == 0:
                return JSONResponse({"error": "no audio"}, status_code=500)
            async with snac_lock:
                pcm = await asyncio.to_thread(decode_frames, codes, frames)
            data = pcm16(pcm)
            return Response((wav_header(len(data)) if wav else b"") + data, media_type="audio/wav" if wav else "audio/pcm")

        async def body_iter():
            if wav:
                yield wav_header(0xFFFFFFFF - 36)
            emitted = 0
            codes, done = [], False
            async for codes, done in codes_stream():
                n = len(codes) // 7
                upto = n if done else max(0, n - LOOKAHEAD)
                if upto > emitted:
                    e = min(n, upto + LOOKAHEAD)
                    s = min(max(0, e - WINDOW), emitted)
                    async with snac_lock:
                        pcm = await asyncio.to_thread(decode_frames, codes[s * 7:e * 7], e - s)
                    yield pcm16(pcm[(emitted - s) * 2048:(upto - s) * 2048])
                    emitted = upto
            n = len(codes) // 7
            if n > emitted:
                s = min(max(0, n - WINDOW), emitted)
                async with snac_lock:
                    pcm = await asyncio.to_thread(decode_frames, codes[s * 7:n * 7], n - s)
                yield pcm16(pcm[(emitted - s) * 2048:])

        return StreamingResponse(body_iter(), media_type="audio/wav" if wav else "audio/pcm")

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
