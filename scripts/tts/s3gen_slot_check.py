#!/usr/bin/env python3
"""Voice-slot (`vsynth`) harness for an s3gen.pkt exported with --clone-slots / --prompt-max, run by
plowrt's `packet_run` example.

  env PYTHONPATH= venv-cbx/bin/python scripts/tts/s3gen_slot_check.py --packet ASSETS/s3gen.pkt \\
      --runner TARGET/release/examples/packet_run --golden CBX_GOLDEN_DIR --out RESULTS_DIR

Voices: the built-in voice, a cloned voice's conditionals (cbx_golden.py's s3gen_prompt_token,
s3gen_prompt_feat truncated to 2P rows, s3gen_embedding) and the same clone cut to 60 prompt tokens.
(a) per voice and case: mel vs PyTorch flow_inference with that voice's ref_dict and the packet's
    CFM noise (gate rel-L2 < 1e-3).
(b) slot isolation: the voices in one B=4 launch (different prompt lengths) vs each alone.
(c) the built-in voice on vsynth vs its synth capacity (same tokens, seed).
(d) GPU time of a vsynth vs synth launch.
(e) --clone-wav: stock ChatterboxTTS T3 tokens in the clip's voice rendered on vsynth with the
    clip's conditionals (and with the built-in voice's); Whisper CER and speaker similarity.
"""
import argparse, json, os, re, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s3gen_packet_check import PIPE, Packet, rel, rnormal  # noqa: E402

MEL = 80


class SlotPacket(Packet):
    def __init__(self, packet, runner):
        super().__init__(packet, runner)
        meta = self._meta()
        self.vcaps = sorted({tuple(int(x) for x in re.match(r"vsynth\.b(\d+)\.t(\d+)\.", r).groups())
                             for r in meta["programs"] if r.startswith("vsynth.")})
        self.slots, self.pmax = self.params["voice.slots"], self.params["prompt.max"]

    def vrun(self, toks, seeds, voices, outs, iters=1):
        """`voices[i]` = (prompt_token, prompt_feat [2P][80], spks [80]); item i reads slot i."""
        B = len(toks)
        cb, cn = min((c for c in self.vcaps if c[0] >= B and c[1] >= max(map(len, toks))),
                     key=lambda c: (c[0] * c[1], c[1]))
        S, pm = self.slots, self.pmax
        tk = np.zeros((cb, cn), np.uint32)
        st, sf_, ss, sl = (np.zeros((S, pm), np.uint32), np.zeros((S, 2 * pm, MEL), np.float32),
                           np.zeros((S, MEL), np.float32), np.zeros(S, np.uint32))
        for i, (t, (pt, pf, sp)) in enumerate(zip(toks, voices)):
            tk[i, :len(t)] = t
            st[i, :len(pt)], sf_[i, :len(pf)], ss[i], sl[i] = pt, pf, sp, len(pt)
        files = {"codes": tk, "voice": np.arange(cb, dtype=np.uint32) % S,
                 "seed": np.array(list(seeds) + [0] * (cb - B), np.uint64),
                 "lengths.0": np.array([len(t) for t in toks] + [0] * (cb - B), np.uint32),
                 "slot.prompt_token": st, "slot.prompt_feat": sf_, "slot.spks": ss, "slot.prompt_len": sl}
        with tempfile.TemporaryDirectory() as d:
            cmd = [self.runner, self.packet, PIPE, f"vsynth.b{cb}.t{cn}", "--iters", str(iters)]
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
    ap.add_argument("--golden", required=True, help="cbx_golden.py output (s3gen_* arrays)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="43,100,240")
    ap.add_argument("--clone-wav", help="(e): reference clip of the cloned voice")
    ap.add_argument("--clone-n", type=int, default=6)
    ap.add_argument("--asr-python", default=sys.executable, help="python with transformers + jiwer for asr_check.py")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import torch, soundfile as sf, perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts import ChatterboxTTS

    tts = ChatterboxTTS.from_pretrained(device="cuda")
    s3, gen = tts.s3gen, tts.conds.gen
    pk = SlotPacket(args.packet, args.runner)
    aff = s3.flow.spk_embed_affine_layer
    sw, sb = aff.weight.detach().double().cpu(), aff.bias.detach().double().cpu()

    def voice(tok, feat, emb):
        tok = np.asarray(tok).reshape(-1).astype(np.int64)
        feat = np.asarray(feat, np.float32).reshape(-1, MEL)[: 2 * len(tok)]
        x = torch.from_numpy(np.asarray(emb, np.float64).reshape(-1))
        spks = (sw @ (x / x.norm().clamp_min(1e-12)) + sb).float().numpy()
        ref = dict(prompt_token=torch.from_numpy(tok)[None].cuda(),
                   prompt_token_len=torch.tensor([len(tok)]).cuda(),
                   prompt_feat=torch.from_numpy(feat.copy())[None].cuda(),
                   prompt_feat_len=None,
                   embedding=torch.from_numpy(np.asarray(emb, np.float32).reshape(1, -1)).cuda())
        return (tok.astype(np.uint32), feat, spks), ref

    g = lambda k: np.load(os.path.join(args.golden, f"s3gen_{k}.npy"))
    voices = {
        "builtin": voice(gen["prompt_token"].detach().cpu().numpy(), gen["prompt_feat"].detach().cpu().numpy(),
                         gen["embedding"].detach().cpu().numpy()),
        "clone": voice(g("prompt_token"), g("prompt_feat"), g("embedding")),
        "clone60": voice(g("prompt_token")[:, :60], g("prompt_feat"), g("embedding")),
    }
    print(f"slots {pk.slots} prompt_max {pk.pmax} vsynth capacities {pk.vcaps}")
    print("prompt tokens:", {k: len(v[0][0]) for k, v in voices.items()})

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    orig_randn = torch.randn_like
    rng = np.random.default_rng(0)
    cases = [(f"rand{n}", rng.integers(0, 6561, n).tolist(), 11 + n) for n in map(int, args.tokens.split(","))]

    def reference(t, seed, ref):
        n, P = len(t), int(ref["prompt_token_len"][0])
        tt, cc = np.meshgrid(np.arange(2 * (P + n)), np.arange(MEL), indexing="ij")
        z = rnormal(seed, 1, tt, cc)
        try:
            torch.randn_like = lambda x, *a, **k: torch.from_numpy(z.T.copy())[None].to(x.device, x.dtype)
            with torch.inference_mode():
                m = s3.flow_inference(torch.tensor(t).long()[None].cuda(), ref_dict=ref, finalize=True)
        finally:
            torch.randn_like = orig_randn
        return m[0].float().detach().cpu().numpy().T

    print("\n== (a) vsynth mel vs PyTorch, identical noise ==")
    ok, single = True, {}
    for vname, (vin, ref) in voices.items():
        for name, t, seed in cases:
            n = len(t)
            res, _, cap = pk.vrun([t], [seed], [vin], ["act.s3gen.mel", "pcm"])
            mel = res["act.s3gen.mel"][: 2 * n * MEL].reshape(2 * n, MEL)
            single[vname, name] = res["pcm"][: n * pk.spt]
            mr = reference(t, seed, ref)
            r = rel(mel, mr) if mel.shape == mr.shape else float("inf")
            ok &= r < 1e-3
            print(f"{vname:>8} {name:>7} cap {str(cap):>9} mel relL2 {r:.3e} maxabs {np.abs(mel - mr).max():.3e}", flush=True)
            if name == cases[-1][0]:
                sf.write(os.path.join(args.out, f"{vname}_{name}.wav"), single[vname, name], 24000)
    print("NUMERICS GATE (mel rel-L2 < 1e-3):", "PASS" if ok else "FAIL")

    print("\n== (b) slot isolation: one launch, mixed prompt lengths ==")
    name, t, seed = cases[0]
    order = ["clone", "builtin", "clone60", "clone"]
    res, tb, capb = pk.vrun([t] * 4, [seed] * 4, [voices[v][0] for v in order], ["pcm"])
    per = capb[1] * pk.spt
    worst = max(rel(res["pcm"][i * per: i * per + len(t) * pk.spt], single[v, name]) for i, v in enumerate(order))
    print(f"capacity {capb}: worst wav rel-L2 batched vs single {worst:.2e}")
    ok &= worst < 1e-3

    print("\n== (c) built-in voice: vsynth vs synth ==")
    rs, ts, caps = pk.run([t], [seed], ["pcm"])
    r = rel(rs["pcm"][: len(t) * pk.spt], single["builtin", name])
    print(f"synth {caps} vs vsynth: wav rel-L2 {r:.2e}")
    ok &= r < 1e-3

    print("\n== (d) GPU time, 100 tokens B=1 (us, median of 5) ==")
    t100 = rng.integers(0, 6561, 100).tolist()
    _, tv, cv = pk.vrun([t100], [1], [voices["clone"][0]], [], iters=5)
    _, tsy, csy = pk.run([t100], [1], [], iters=5)
    for tag, cp, tm in (("vsynth", cv, tv), ("synth", csy, tsy)):
        print(f"{tag} {cp}: encoder {tm[0]:.0f} cfm part {np.median(tm[1:-1]):.0f} vocoder {tm[-1]:.0f} total {sum(tm) / 1e3:.1f} ms")
    if args.clone_wav:
        print("\n== (e) cloned voice end to end: stock T3 tokens in that voice, vsynth audio ==")
        from chatterbox_ref import PROMPTS
        tts.prepare_conditionals(args.clone_wav, exaggeration=0.5)
        cg = tts.conds.gen
        cv, _ = voice(cg["prompt_token"].detach().cpu().numpy(), cg["prompt_feat"].detach().cpu().numpy(), cg["embedding"].detach().cpu().numpy())
        captured, s3_inf = {}, s3.inference

        def cap(*a, **k):
            captured["tok"] = k.get("speech_tokens", a[0] if a else None).detach().detach().cpu().numpy().reshape(-1)
            return s3_inf(*a, **k)

        s3.inference, texts, groups = cap, {}, {"stock": [], "packet": [], "default": []}
        for i, text in enumerate(PROMPTS[: args.clone_n]):
            torch.manual_seed(i)
            w = tts.generate(text)
            tok = [int(x) for x in captured["tok"] if x < 6561]
            for tag, v in (("packet", cv), ("default", voices["builtin"][0])):
                res, _, _ = pk.vrun([tok], [100 + i], [v], ["pcm"])
                sf.write(os.path.join(args.out, f"{tag}_{i:02d}.wav"), res["pcm"][: len(tok) * pk.spt], 24000)
            sf.write(os.path.join(args.out, f"stock_{i:02d}.wav"), w.squeeze(0).numpy(), tts.sr)
            for tag in groups:
                groups[tag].append(os.path.join(args.out, f"{tag}_{i:02d}.wav"))
                texts[f"{tag}_{i:02d}.wav"] = text
        s3.inference = s3_inf
        tp = os.path.join(args.out, "texts.json")
        json.dump(texts, open(tp, "w"), indent=0)
        r = subprocess.run([args.asr_python, os.path.join(HERE, "asr_check.py"), *groups["packet"], *groups["stock"],
                            "--texts", tp, "--max-cer", "1.0"], capture_output=True, text=True)
        print(r.stdout.strip() + r.stderr[-2000:] * (r.returncode != 0))
        r = subprocess.run([sys.executable, os.path.join(HERE, "speaker_sim.py"), args.clone_wav,
                            *(f"{k}={','.join(v)}" for k, v in groups.items())], capture_output=True, text=True)
        print(r.stdout.strip() + r.stderr[-2000:] * (r.returncode != 0))
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
