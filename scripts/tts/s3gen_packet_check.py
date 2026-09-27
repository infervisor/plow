#!/usr/bin/env python3
"""Numerics / intelligibility / timing harness for the S3Gen packet (s3gen.pkt, plowc
PLOW_TTS_VOCODER_DIR) run by plowrt's generic `packet_run` example.

  $GL -n 1 s3gen-pkt env PYTHONPATH= HF_HOME=... /opt/dlami/nvme/lava-tts/venv-cbx/bin/python \\
      scripts/tts/s3gen_packet_check.py --packet ASSETS/s3gen.pkt --runner TARGET/release/examples/packet_run \\
      --out RESULTS_DIR

(a) numerics vs PyTorch S3Token2Wav fp32 (TF32 off) with identical randomness: the packet's
    counter-based streams (CFM noise z = stream 1, SineGen initial phases = stream 2, SineGen noise
    = stream 3) are regenerated here and injected into the reference (torch.randn_like; SineGen with
    its phase integral in fp64). Reported per case: mel rel-L2 (gate < 1e-3), wav rel-L2 against the
    reference HiFT fed the packet's mel ("hift") and end to end ("e2e").
(b) batching: the 8 T3 cases at B=8 in one launch vs one at a time.
(c) intelligibility: Whisper CER (asr_check.py) of the packet's wavs for the stock T3 tokens of the
    8 prompts (captured from stock ChatterboxTTS.generate), next to the stock wavs.
(d) GPU time per program (encoder / CFM step / vocoder) at B=1, 100 tokens and at B=8.
"""
import argparse, json, os, re, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chatterbox_ref import PROMPTS  # noqa: E402

PIPE = "vocoder.synth"
M64 = (1 << 64) - 1


def mix64(z):
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(M64)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def rhash(seed, stream, a, b):
    a, b = np.asarray(a, np.uint64), np.asarray(b, np.uint64)
    key = (np.uint64(stream) << np.uint64(56)) ^ (a << np.uint64(32)) ^ b
    return mix64(np.uint64(seed) ^ mix64(key))


def rnormal(seed, stream, a, b):
    h1 = rhash(seed, stream, a, b)
    h2 = mix64(h1)
    u1 = ((h1 >> np.uint64(40)) + np.uint64(1)).astype(np.float64) / 16777216.0
    u2 = (h2 >> np.uint64(40)).astype(np.float64) / 16777216.0
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).astype(np.float32)


def runiform(seed, stream, a, b):
    return ((rhash(seed, stream, a, b) >> np.uint64(40)).astype(np.float64) / 16777216.0).astype(np.float32)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


class Packet:
    def __init__(self, packet, runner):
        self.packet, self.runner = packet, runner
        meta = self._meta()
        self.params = meta["parameters"]
        self.caps = sorted({tuple(int(x) for x in re.match(r"synth\.b(\d+)\.t(\d+)\.", r).groups()) for r in meta["programs"]})
        self.spt = self.params["codec.frame_samples"]

    def _meta(self):
        # The pipeline metadata is JSON inside the packet; find it by its version marker.
        raw = open(self.packet, "rb").read()
        i = raw.find(b'{"version"')
        depth, j = 0, i
        while True:
            c = raw[j:j + 1]
            depth += c == b"{"
            depth -= c == b"}"
            j += 1
            if depth == 0:
                break
        meta = json.loads(raw[i:j])
        return [p for p in meta["pipelines"] if p["name"] == PIPE][0]

    def cap(self, B, n):
        return min((c for c in self.caps if c[0] >= B and c[1] >= n), key=lambda c: (c[0] * c[1], c[1]))

    def run(self, toks, seeds, outs, iters=1, cap=None, stages=None):
        B = len(toks)
        cb, cn = cap or self.cap(B, max(len(t) for t in toks))
        with tempfile.TemporaryDirectory() as d:
            tk = np.zeros((cb, cn), np.uint32)
            for i, t in enumerate(toks):
                tk[i, :len(t)] = t
            files = {"codes": tk, "voice": np.zeros(cb, np.uint32), "seed": np.array(list(seeds) + [0] * (cb - B), np.uint64),
                     "lengths.0": np.array([len(t) for t in toks] + [0] * (cb - B), np.uint32)}
            cmd = [self.runner, self.packet, PIPE, f"synth.b{cb}.t{cn}", "--iters", str(iters)]
            if stages:
                cmd += ["--stages", str(stages)]
            for name, a in files.items():
                p = os.path.join(d, name + ".bin")
                a.tofile(p)
                cmd += ["--in", f"{name}={p}"]
            for name in outs:
                cmd += ["--out", f"{name}={os.path.join(d, name + '.out')}"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(r.stderr[-4000:])
            res = {n: np.fromfile(os.path.join(d, n + ".out"), np.float32) for n in outs}
            times = [float(l.split()[-1]) for l in r.stdout.splitlines() if l.startswith("stage ")]
        return res, times, (cb, cn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-cer", action="store_true")
    ap.add_argument("--rand-tokens", default="43,100,200", help="token counts of the random-token cases")
    ap.add_argument("--asr-python", default=sys.executable, help="python with transformers + jiwer for asr_check.py")
    ap.add_argument("--s3gen-weights", default=None, help="snapshot file replacing the stock S3Gen weights "
                    "(Multilingual V3: s3gen_v3.safetensors); the packet must be exported from the same file")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import torch, soundfile as sf, perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.models.s3gen import hifigan

    tts = ChatterboxTTS.from_pretrained(device="cuda")
    if args.s3gen_weights:
        import glob
        from safetensors.torch import load_file
        snap = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                                      "hub/models--ResembleAI--chatterbox/snapshots/*/"))[0]
        tts.s3gen.load_state_dict(load_file(os.path.join(snap, args.s3gen_weights)), strict=False)
        tts.s3gen.to("cuda").eval()
    s3, gen = tts.s3gen, tts.conds.gen
    pk = Packet(args.packet, args.runner)
    P, Pf = gen["prompt_token"].shape[1], gen["prompt_feat"].shape[1]
    assert Pf == 2 * P

    tok_path = os.path.join(args.out, "t3_tokens.json")
    if not os.path.exists(tok_path):
        captured, s3_inf = {}, s3.inference

        def cap(*a, **k):
            captured["tok"] = k.get("speech_tokens", a[0] if a else None).detach().cpu().numpy().reshape(-1)
            return s3_inf(*a, **k)

        s3.inference = cap
        toks = []
        for i, text in enumerate(PROMPTS):
            torch.manual_seed(i)
            wav = tts.generate(text)
            toks.append([int(x) for x in captured["tok"] if x < 6561])
            sf.write(os.path.join(args.out, f"stock_{i:02d}.wav"), wav.squeeze(0).numpy(), tts.sr)
        s3.inference = s3_inf
        json.dump(toks, open(tok_path, "w"))
    toks = json.load(open(tok_path))

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    orig_randn, orig_sine = torch.randn_like, hifigan.SineGen.forward
    g = np.random.default_rng(0)
    cases = [(f"rand{n}", g.integers(0, 6561, n).tolist(), 11 + n) for n in map(int, args.rand_tokens.split(","))]
    cases += [(f"t3_{i:02d}", t, 100 + i) for i, t in enumerate(toks)]
    print("== (a) numerics vs PyTorch fp32 (TF32 off), identical noise ==")
    print(f"{'case':>8} {'ntok':>5} {'cap':>9} | {'mel relL2':>10} {'mel maxabs':>10} | {'wav hift':>9} {'wav e2e':>9}")
    ok, single = True, {}
    for name, t, seed in cases:
        n = len(t)
        res, _, capx = pk.run([t], [seed], ["act.s3gen.mel", "pcm"])
        G, L = 2 * n, pk.spt * n
        mel = res["act.s3gen.mel"][: G * 80].reshape(G, 80)
        wav = res["pcm"][:L]
        single[name] = wav
        if name.startswith("t3_"):
            sf.write(os.path.join(args.out, f"packet_{name[3:]}.wav"), wav, 24000)
        T1 = 2 * (P + n)
        tt, cc = np.meshgrid(np.arange(T1), np.arange(80), indexing="ij")
        z = rnormal(seed, 1, tt, cc)  # [T1][80]
        ph = np.array([0.0] + [(runiform(seed, 2, i, 0) - np.float32(0.5)) * np.float32(2 * np.pi) for i in range(1, 9)],
                      np.float32)
        ii, nn = np.meshgrid(np.arange(9), np.arange(L), indexing="ij")
        sn = rnormal(seed, 3, ii, nn)  # [9][L]

        def fwd(self, f0):
            B, _, Ls = f0.shape
            cyc = torch.cumsum(f0.double(), dim=-1)
            theta = torch.cat([(2 * np.pi * ((cyc * (i + 1) / self.sampling_rate) % 1)) for i in range(9)], 1).float()
            sine = self.sine_amp * torch.sin(theta + torch.from_numpy(ph).view(1, 9, 1).to(f0.device))
            uv = (f0 > self.voiced_threshold).float()
            amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = amp * torch.from_numpy(sn[:, :Ls].copy())[None].to(f0.device)
            return sine * uv + noise, uv, noise

        try:
            torch.randn_like = lambda x, *a, **k: torch.from_numpy(z.T.copy())[None].to(x.device, x.dtype)
            with torch.inference_mode():
                mel_ref = s3.flow_inference(torch.tensor(t).long()[None].cuda(), ref_dict=gen, finalize=True)
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
        print(f"{name:>8} {n:>5} {str(capx):>9} | {r:10.3e} {np.abs(mel - mel_ref).max():10.3e} | "
              f"{rel(wav, wr['hift']):9.3e} {rel(wav, wr['e2e']):9.3e}", flush=True)
    print("NUMERICS GATE (mel rel-L2 < 1e-3):", "PASS" if ok else "FAIL")

    print("\n== (b) batch B=8 (the T3 cases, one launch) vs single ==")
    t3 = [c for c in cases if c[0].startswith("t3_")]
    res, times8, cap8 = pk.run([c[1] for c in t3], [c[2] for c in t3], ["pcm"], iters=5)
    per = cap8[1] * pk.spt
    worst = max(rel(res["pcm"][i * per: i * per + len(single[c[0]])], single[c[0]]) for i, c in enumerate(t3))
    print(f"capacity {cap8}: worst wav rel-L2 batched vs single {worst:.2e}")
    ok &= worst < 1e-3

    print("\n== (d) GPU time per program (us, median of 5) ==")
    _, times1, cap1 = pk.run([g.integers(0, 6561, 100).tolist()], [1], [], iters=5)
    for tag, cp, tm in (("B=1 100 tok", cap1, times1), ("B=8 T3 cases", cap8, times8)):
        cfm = tm[1:-1]
        print(f"{tag} cap {cp}: encoder {tm[0]:.0f}  cfm step {np.median(cfm):.0f} (x{len(cfm)})  vocoder {tm[-1]:.0f}"
              f"  total {sum(tm) / 1e3:.1f} ms")
    json.dump(dict(b1=dict(cap=cap1, us=times1), b8=dict(cap=cap8, us=times8)), open(os.path.join(args.out, "timing.json"), "w"))

    if not args.skip_cer:
        texts = {}
        for i in range(len(toks)):
            texts[f"packet_{i:02d}.wav"] = PROMPTS[i]
            texts[f"stock_{i:02d}.wav"] = PROMPTS[i]
        json.dump(texts, open(os.path.join(args.out, "texts.json"), "w"), indent=0)
        for tag in ("stock", "packet"):
            wavs = [os.path.join(args.out, f"{tag}_{i:02d}.wav") for i in range(len(toks))]
            r = subprocess.run([args.asr_python, os.path.join(HERE, "asr_check.py"), *wavs, "--texts",
                                os.path.join(args.out, "texts.json"), "--max-cer", "1.0"], capture_output=True, text=True)
            print(f"== (c) ASR {tag} ==\n" + r.stdout.strip() + r.stderr[-2000:] * (r.returncode != 0))
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
