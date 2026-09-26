#!/usr/bin/env python3
"""Numerics / intelligibility / latency harness for libplow_s3gen.so (native Chatterbox S3Gen).

  perf-data/tools/gpulease -n 1 s3gen-check env HF_HOME=/root/tts-work/hf \\
      /root/tts-work/venv-ref/bin/python scripts/tts/s3gen_check.py \\
      --lib /root/tts-work/s3gen/libplow_s3gen.so --weights /root/tts-work/s3gen/s3gen.bin \\
      --voice /root/tts-work/s3gen/voice_default.bin --out /root/tts-work/results/s3gen

(a) numerics vs PyTorch S3Token2Wav fp32 (TF32 off), identical randomness: the library's CFM noise
    z is injected into the reference (torch.randn_like in CausalConditionalCFM.forward) and its
    SineGen phases / noise into HiFT (SineGen.forward replaced, phase integral in fp64 as in the
    library). GATE: mel rel-L2 < 1e-3 for every case (random tokens at 1.7 / 4 / 8 s of audio and
    the stock T3 tokens of the 8 prompts). Reported: wav rel-L2 with the native mel fed to the
    reference HiFT ("hift") and end to end ("e2e").
(b) intelligibility: stock ChatterboxTTS.generate() on the 8 prompts of chatterbox_ref.py (tokens
    captured at s3gen.inference); the same tokens through the library; Whisper CER via asr_check.py
    on both. GATE: native median CER <= 0.05 (and <= stock + 0.02).
(c) latency: median wall ms per utterance, plow_s3gen_synthesize_host (host tokens in, host PCM
    out, blocking) vs stock s3gen.inference (fp32, torch defaults), 1.7 / 4 / 8 s of audio.
"""
import argparse, ctypes, json, os, statistics, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chatterbox_ref import PROMPTS  # noqa: E402

LAT_TOKENS = (43, 100, 200)  # 1.72 / 4.0 / 8.0 s of audio with the builtin voice


class Native:
    def __init__(self, lib, weights, voice, max_batch=2, max_tokens=600, debug=False):
        os.environ["PLOW_S3GEN_DEBUG"] = "1" if debug else "0"
        self.lib = L = ctypes.CDLL(lib)
        L.plow_s3gen_create.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_void_p)]
        L.plow_s3gen_add_voice.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        L.plow_s3gen_synthesize_host.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong,
                                                 ctypes.POINTER(ctypes.c_int)]
        L.plow_s3gen_synthesize_batch_host.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        L.plow_s3gen_debug_read.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p,
                                            ctypes.c_long]
        L.plow_s3gen_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong)]
        L.plow_s3gen_destroy.argtypes = [ctypes.c_void_p]
        self.h = ctypes.c_void_p()
        rc = L.plow_s3gen_create(0, weights.encode(), max_batch, max_tokens, ctypes.byref(self.h))
        assert rc == 0, f"create rc={rc}"
        vid = ctypes.c_int()
        rc = L.plow_s3gen_add_voice(self.h, voice.encode(), ctypes.byref(vid))
        assert rc == 0, f"add_voice rc={rc}"
        self.vid = vid.value

    def synth(self, toks, seed=1):
        toks = np.ascontiguousarray(toks, dtype=np.int32)
        out = np.empty(960 * len(toks) + 960, np.float32)
        ns = ctypes.c_int()
        rc = self.lib.plow_s3gen_synthesize_host(self.h, self.vid, toks.ctypes.data, len(toks), out.ctypes.data,
                                                 out.size, seed, ctypes.byref(ns))
        assert rc == 0, f"synthesize rc={rc}"
        return out[: ns.value]

    def synth_batch(self, tok_list, seeds):
        B = len(tok_list)
        cat = np.ascontiguousarray(np.concatenate(tok_list), dtype=np.int32)
        n = np.array([len(t) for t in tok_list], np.int32)
        vids = np.full(B, self.vid, np.int32)
        sd = np.array(seeds, np.uint64)
        maxs = 960 * int(n.max()) + 960
        out = np.empty((B, maxs), np.float32)
        ns = np.zeros(B, np.int32)
        rc = self.lib.plow_s3gen_synthesize_batch_host(self.h, B, vids.ctypes.data, cat.ctypes.data, n.ctypes.data,
                                                       out.ctypes.data, maxs, sd.ctypes.data, ns.ctypes.data)
        assert rc == 0, f"synthesize_batch rc={rc}"
        return [out[b, : ns[b]] for b in range(B)]

    def read(self, name, n, b=0):
        a = np.zeros(n, np.float32)
        rc = self.lib.plow_s3gen_debug_read(self.h, name.encode(), b, a.ctypes.data, n)
        assert rc == 0, (name, rc)
        return a

    def stats(self):
        st = (ctypes.c_longlong * 6)()
        self.lib.plow_s3gen_stats(self.h, st)
        return list(st)

    def close(self):
        self.lib.plow_s3gen_destroy(self.h)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def numerics(nat, s3, gen, cases):
    import torch
    from chatterbox.models.s3gen import hifigan
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    P, Pf = gen["prompt_token"].shape[1], gen["prompt_feat"].shape[1]
    orig_randn, orig_sine = torch.randn_like, hifigan.SineGen.forward
    ok = True
    print("== (a) numerics vs PyTorch fp32 (TF32 off), identical noise ==")
    print(f"{'case':>12} {'ntok':>5} | {'mel relL2':>10} {'mel maxabs':>10} | {'wav relL2 hift':>14} {'wav relL2 e2e':>13}")
    for name, toks, seed in cases:
        n = len(toks)
        wav = nat.synth(toks, seed)
        st = nat.stats()
        t1cap, gcap = st[4], st[5]
        T1, G = 2 * (P + n), 2 * (P + n) - Pf
        L = len(wav)
        z = nat.read("noise", T1 * 80).reshape(T1, 80)
        mel = nat.read("mel", G * 80).reshape(G, 80)
        sn = nat.read("sine_noise", 9 * 480 * gcap).reshape(9, 480 * gcap)[:, :L]
        ph = nat.read("sine_phase", 9)

        def fwd(self, f0):
            B, _, Ls = f0.shape
            F_mat = torch.zeros((B, 9, Ls), device=f0.device)
            for i in range(9):
                F_mat[:, i:i + 1, :] = f0 * (i + 1) / self.sampling_rate
            theta = (2 * np.pi * (torch.cumsum(F_mat.double(), dim=-1) % 1)).float()
            sine = self.sine_amp * torch.sin(theta + torch.from_numpy(ph).view(1, 9, 1).to(f0.device))
            uv = (f0 > self.voiced_threshold).float()
            amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = amp * torch.from_numpy(sn[:, :Ls].copy())[None].to(f0.device)
            return sine * uv + noise, uv, noise

        try:
            torch.randn_like = lambda x, *a, **k: torch.from_numpy(z.T.copy())[None].to(x.device, x.dtype)
            with torch.inference_mode():
                mel_ref = s3.flow_inference(torch.from_numpy(np.asarray(toks)).long()[None].cuda(), ref_dict=gen,
                                            finalize=True)
        finally:
            torch.randn_like = orig_randn
        mel_ref = mel_ref[0].float().cpu().numpy().T
        wr = {}
        try:
            hifigan.SineGen.forward = fwd
            for tag, m in (("hift", mel), ("e2e", mel_ref)):
                with torch.inference_mode():
                    w, _ = s3.hift_inference(torch.from_numpy(m.T.copy())[None].cuda(), None)
                    w[:, :len(s3.trim_fade)] *= s3.trim_fade
                wr[tag] = w[0].float().cpu().numpy()[:L]
        finally:
            hifigan.SineGen.forward = orig_sine
        r = rel(mel, mel_ref)
        ok &= r < 1e-3 and mel.shape == mel_ref.shape
        print(f"{name:>12} {n:>5} | {r:10.3e} {np.abs(mel - mel_ref).max():10.3e} | {rel(wav, wr['hift']):14.3e} "
              f"{rel(wav, wr['e2e']):13.3e}", flush=True)
    print("NUMERICS GATE (mel rel-L2 < 1e-3):", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-cer", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import torch, soundfile as sf, perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts import ChatterboxTTS

    tts = ChatterboxTTS.from_pretrained(device="cuda")
    s3, gen = tts.s3gen, tts.conds.gen
    ok = True

    # ---- stock T3 tokens (and stock wavs) for the 8 prompts ----
    tok_path = os.path.join(args.out, "t3_tokens.json")
    captured = {}
    s3_inf = s3.inference

    def cap(*a, **k):
        captured["tok"] = k.get("speech_tokens", a[0] if a else None).detach().cpu().numpy().reshape(-1)
        return s3_inf(*a, **k)

    if not args.skip_cer or not os.path.exists(tok_path):
        s3.inference = cap
        toks = []
        for i, text in enumerate(PROMPTS):
            torch.manual_seed(i)
            wav = tts.generate(text)
            toks.append(captured["tok"].astype(np.int32).tolist())
            sf.write(os.path.join(args.out, f"stock_{i:02d}.wav"), wav.squeeze(0).numpy(), tts.sr)
        s3.inference = s3_inf
        json.dump(toks, open(tok_path, "w"))
    toks = json.load(open(tok_path))

    # ---- (a) numerics ----
    nat = Native(args.lib, args.weights, args.voice, debug=True)
    g = np.random.default_rng(0)
    cases = [(f"rand{n}", g.integers(0, 6561, n).tolist(), 11 + n) for n in LAT_TOKENS]
    cases += [(f"t3_{i:02d}", t, 100 + i) for i, t in enumerate(toks)]
    ok &= numerics(nat, s3, gen, cases)
    # determinism + batch independence
    a1 = nat.synth(toks[0], 5)
    a2 = nat.synth(toks[0], 5)
    b = nat.synth_batch([toks[0], toks[2]], [5, 6])
    c2 = nat.synth(toks[2], 6)
    det = np.array_equal(a1, a2)
    print(f"deterministic for a given seed (bit-exact repeat): {det}; batched (B=2, mixed lengths) vs single: "
          f"wav rel-L2 {rel(b[0], a1):.2e} / {rel(b[1], c2):.2e}, lengths equal {len(b[0]) == len(a1) and len(b[1]) == len(c2)}")
    ok &= det and len(b[0]) == len(a1) and rel(b[0], a1) < 1e-2 and rel(b[1], c2) < 1e-2
    nat.close()

    nat = Native(args.lib, args.weights, args.voice, debug=False)
    # ---- (b) intelligibility ----
    if not args.skip_cer:
        texts = {}
        for i, t in enumerate(toks):
            w = nat.synth(t, seed=1 + i)
            sf.write(os.path.join(args.out, f"native_{i:02d}.wav"), w, 24000)
            texts[f"native_{i:02d}.wav"] = PROMPTS[i]
            texts[f"stock_{i:02d}.wav"] = PROMPTS[i]
        json.dump(texts, open(os.path.join(args.out, "texts.json"), "w"), indent=0)
        med = {}
        for tag in ("stock", "native"):
            wavs = [os.path.join(args.out, f"{tag}_{i:02d}.wav") for i in range(len(toks))]
            r = subprocess.run([sys.executable, os.path.join(HERE, "asr_check.py"), *wavs, "--texts",
                                os.path.join(args.out, "texts.json"), "--max-cer", "1.0"],
                               capture_output=True, text=True)
            print(f"== (b) ASR {tag} ==\n" + r.stdout.strip())
            med[tag] = float([l for l in r.stdout.splitlines() if l.startswith("MEDIAN_CER=")][0].split("=")[1].split()[0])
        cer_ok = med["native"] <= 0.05 and med["native"] <= med["stock"] + 0.02
        print(f"CER GATE: native median {med['native']:.3f} vs stock {med['stock']:.3f} ->",
              "PASS" if cer_ok else "FAIL")
        ok &= cer_ok

    # ---- (c) latency ----
    if not args.skip_latency:
        print("\n== (c) latency, ms per utterance (median; native: 30 runs, stock: 10) ==")
        print(f"{'ntok':>5} {'audio s':>8} | {'native':>8} {'first call':>10} | {'stock':>8} | {'speedup':>7} | "
              f"{'kernels/utt':>11} {'graph launches':>14}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True  # torch defaults
        rows = []
        for n in LAT_TOKENS:
            t = g.integers(0, 6561, n).tolist()
            t0 = time.perf_counter()
            nat.synth(t, 1)
            first = (time.perf_counter() - t0) * 1e3
            for _ in range(5):
                nat.synth(t, 1)
            ts = []
            for _ in range(30):
                t0 = time.perf_counter()
                nat.synth(t, 1)
                ts.append((time.perf_counter() - t0) * 1e3)
            st = nat.stats()
            tt = torch.tensor(t).long().cuda()
            for _ in range(2):
                s3.inference(tt, ref_dict=gen)
            ss = []
            for _ in range(10):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                s3.inference(tt, ref_dict=gen)
                torch.cuda.synchronize()
                ss.append((time.perf_counter() - t0) * 1e3)
            nm, sm = statistics.median(ts), statistics.median(ss)
            rows.append(dict(ntok=n, audio_s=n * 0.04, native_ms=nm, stock_ms=sm, speedup=sm / nm, kernels=st[0]))
            print(f"{n:>5} {n * 0.04:8.2f} | {nm:8.2f} {first:10.1f} | {sm:8.1f} | {sm / nm:6.1f}x | {st[0]:>11} "
                  f"{st[1]:>14}", flush=True)
        json.dump(rows, open(os.path.join(args.out, "latency.json"), "w"), indent=1)
    nat.close()
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
