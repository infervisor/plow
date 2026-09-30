#!/usr/bin/env python3
"""Numerics of s3gen.pkt's cached-prompt capacities (`prefill.v*` + `csynth.b*.t*`) against a
PyTorch implementation of the same computation, with identical randomness (the packet's counter
streams regenerated here, as in s3gen_packet_check.py):

  prefill: the voice prompt alone through the encoder and the 10 guided Euler steps (noise from
           seed 0x5eed); the attention K/V of prompt mel rows [0, 2P - 8) are kept per step/block.
  window:  encoder over [prompt | tokens]; the CFM over [prompt mel rows 2P-8..2P | tokens], its
           attention keys = [cached prompt K/V | own K/V]; HiFT on the tokens' mel with the NSF
           phase shifted to `phase` at sample `seam` (and read back at `next_seam`).

Also reported (not gated): the cached render vs the reference whole render (prompt attending to
the tokens), next to two reference renders with different noise.

  $GL -n 1 s3gen-cached env PYTHONPATH= HF_HOME=... <v3 venv python> scripts/tts/s3gen_cached_check.py \\
      --packet ASSETS/s3gen.pkt --runner TARGET/release/examples/packet_run [--mtl]
"""
import argparse, json, math, os, re, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s3gen_packet_check import PIPE, Packet, rel, rnormal  # noqa: E402

PREFILL_SEED = 0x5EED
TAIL = 8
NH = 9


class Proc:
    """Attention processor: capture the first `keep` K/V rows, or prepend the captured ones."""
    mode, keep, store, idx = None, 0, {}, 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        import torch
        import torch.nn.functional as F
        q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        i = Proc.idx
        Proc.idx += 1
        mask = attention_mask
        if Proc.mode == "capture":
            Proc.store[i] = (k[:, :Proc.keep].clone(), v[:, :Proc.keep].clone())
        elif Proc.mode == "use":
            ck, cv = Proc.store[i]
            k, v, mask = torch.cat([ck, k], 1), torch.cat([cv, v], 1), None
        B, Tq, _ = q.shape
        h = attn.heads
        q = q.view(B, Tq, h, -1).transpose(1, 2)
        k = k.view(B, k.size(1), h, -1).transpose(1, 2)
        v = v.view(B, v.size(1), h, -1).transpose(1, 2)
        if mask is not None and mask.dim() == 3:
            mask = mask.view(B, 1, mask.size(-2), mask.size(-1))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, Tq, -1)
        return attn.to_out[1](attn.to_out[0](o))


def noise(seed, rows):
    tt, cc = np.meshgrid(np.arange(rows), np.arange(80), indexing="ij")
    return rnormal(seed, 1, tt, cc)  # [rows][80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--mtl", action="store_true", help="Multilingual V3 weights (s3gen_v3.safetensors)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from chatterbox.models.s3gen import hifigan
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mtl:
        from t3_mtl_ref import load_mtl
        m = load_mtl()
    else:
        import perth
        if getattr(perth, "PerthImplicitWatermarker", None) is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker
        from chatterbox.tts import ChatterboxTTS
        m = ChatterboxTTS.from_pretrained(device="cuda")
    s3 = m.s3gen
    ref = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in m.conds.gen.items()}
    flow = s3.flow
    P = ref["prompt_token"].shape[1]
    pk = Packet(args.packet, args.runner)
    meta = pk._meta()
    caps = sorted({tuple(int(x) for x in re.match(r"csynth\.b(\d+)\.t(\d+)\.", r).groups()) for r in meta["programs"] if r.startswith("csynth.")})
    if not caps:
        raise SystemExit("packet has no cached capacities")
    spt = pk.spt
    emb = flow.spk_embed_affine_layer(F.normalize(ref["embedding"].float(), dim=1))
    t_span = 1 - torch.cos(torch.linspace(0, 1, 11, device="cuda") * 0.5 * math.pi)

    def encode(token):
        tl = torch.tensor([token.size(1)], device="cuda")
        h, _ = flow.encoder(flow.input_embedding(token.long()), tl)
        return flow.encoder_proj(h)

    procs = [mod for mod in flow.decoder.estimator.modules() if mod.__class__.__name__ == "Attention"]
    saved = [p.processor for p in procs]
    for p in procs:
        p.set_processor(Proc())
    with torch.inference_mode():
        h = encode(ref["prompt_token"])
        zp = torch.from_numpy(noise(PREFILL_SEED, 2 * P).T.copy())[None].cuda()
        Proc.mode, Proc.keep, Proc.idx, Proc.store = "capture", 2 * P - TAIL, 0, {}
        flow.decoder.solve_euler(zp, t_span=t_span, mu=h.transpose(1, 2).contiguous(), mask=torch.ones(1, 1, 2 * P, device="cuda"),
                                 spks=emb, cond=ref["prompt_feat"].transpose(1, 2).contiguous())
        Proc.mode = None

    def reference_mel(tok, seed):
        n = len(tok)
        T = TAIL + 2 * n
        with torch.inference_mode():
            h = encode(torch.cat([ref["prompt_token"], torch.tensor(tok, device="cuda")[None]], 1))[:, 2 * P - TAIL:]
            conds = torch.zeros(1, 80, T, device="cuda")
            conds[:, :, :TAIL] = ref["prompt_feat"][:, 2 * P - TAIL:].transpose(1, 2)
            z = torch.from_numpy(noise(seed, T).T.copy())[None].cuda()
            Proc.mode, Proc.idx = "use", 0
            feat = flow.decoder.solve_euler(z, t_span=t_span, mu=h.transpose(1, 2).contiguous(), mask=torch.ones(1, 1, T, device="cuda"),
                                            spks=emb, cond=conds)
            Proc.mode = None
        return feat[0, :, TAIL:].float().cpu().numpy().T  # [2n][80]

    def reference_wav(mel, seed, phase, seam, next_seam):
        L = mel.shape[0] * spt // 2
        ii, nn = np.meshgrid(np.arange(NH), np.arange(L), indexing="ij")
        from s3gen_packet_check import rnormal as rn
        sn = rn(seed, 3, ii, nn)
        out = {}

        def fwd(self, f0):
            cyc = torch.cumsum(f0.double(), dim=-1)
            theta = torch.cat([(2 * np.pi * ((cyc * (i + 1) / self.sampling_rate) % 1)) for i in range(NH)], 1).float()
            theta = theta - theta[:, :, seam:seam + 1] + torch.from_numpy(phase).view(1, NH, 1).to(f0.device)
            out["phase"] = theta[0, :, next_seam].cpu().numpy()
            sine = self.sine_amp * torch.sin(theta)
            uv = (f0 > self.voiced_threshold).float()
            amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            nz = amp * torch.from_numpy(sn[:, :f0.shape[-1]].copy())[None].to(f0.device)
            return sine * uv + nz, uv, nz

        orig = hifigan.SineGen.forward
        try:
            hifigan.SineGen.forward = fwd
            with torch.inference_mode():
                w, _ = s3.hift_inference(torch.from_numpy(mel.T.copy())[None].cuda(), None)
                w[:, :len(s3.trim_fade)] *= s3.trim_fade
        finally:
            hifigan.SineGen.forward = orig
        return w[0].float().cpu().numpy()[:L], out["phase"]

    g = np.random.default_rng(1)
    cases = [(n, 7 + n) for n in (12, 28, 40, 60)]
    ok = True
    print(f"{'ntok':>5} {'cap':>9} | {'mel relL2':>10} | {'wav relL2':>10} {'phase err':>10}")
    batch = []
    for n, seed in cases:
        tok = g.integers(0, 6561, n).tolist()
        phase = np.concatenate([[0.0], g.uniform(-np.pi, np.pi, NH - 1)]).astype(np.float32)
        seam, next_seam = int(g.integers(0, n // 2)) * spt, (n - 3) * spt
        batch.append((tok, seed, phase, seam, next_seam))
    B = len(batch)
    cb, cn = min((c for c in caps if c[0] >= B and c[1] >= max(len(b[0]) for b in batch)), key=lambda c: (c[0] * c[1], c[1]))
    with tempfile.TemporaryDirectory() as d:
        def put(name, a):
            p = os.path.join(d, name + ".bin")
            a.tofile(p)
            return p
        tk = np.zeros((cb, cn), np.uint32)
        ph = np.zeros((cb, NH), np.float32)
        for i, (tok, _, phase, _, _) in enumerate(batch):
            tk[i, :len(tok)] = tok
            ph[i] = phase
        files = {"codes": tk, "voice": np.zeros(cb, np.uint32), "seed": np.array([b[1] for b in batch] + [0] * (cb - B), np.uint64),
                 "lengths.0": np.array([len(b[0]) for b in batch] + [0] * (cb - B), np.uint32), "phase": ph,
                 "seam": np.array([b[3] for b in batch] + [0] * (cb - B), np.uint32),
                 "next_seam": np.array([b[4] for b in batch] + [0] * (cb - B), np.uint32)}
        cmd = [args.runner, args.packet, PIPE, f"csynth.b{cb}.t{cn}", "--before", "prefill.v0",
               "--before-in", f"voice={put('pv', np.zeros(1, np.uint32))}", "--before-in", f"lengths.0={put('pl', np.zeros(1, np.uint32))}",
               "--before-in", f"seed={put('ps', np.array([PREFILL_SEED], np.uint64))}"]
        for name, a in files.items():
            cmd += ["--in", f"{name}={put(name, a)}"]
        outs = ["act.s3gen.mel", "pcm", "phase_out"]
        for name in outs:
            cmd += ["--out", f"{name}={os.path.join(d, name + '.out')}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr[-4000:])
        res = {k: np.fromfile(os.path.join(d, k + ".out"), np.float32) for k in outs}
        print(r.stdout.strip().splitlines()[-1])
    for i, (tok, seed, phase, seam, next_seam) in enumerate(batch):
        n = len(tok)
        mel = res["act.s3gen.mel"][i * 2 * cn * 80:][: 2 * n * 80].reshape(2 * n, 80)
        wav = res["pcm"][i * cn * spt:][: n * spt]
        pout = res["phase_out"][i * NH:(i + 1) * NH]
        mref = reference_mel(tok, seed)
        wref, pref = reference_wav(mel, seed, phase, seam // 1, next_seam)
        dphi = float(np.abs(np.angle(np.exp(1j * (pout - pref)))).max())
        rm, rw = rel(mel, mref), rel(wav, wref)
        # The NSF phase integrates f0 over the window in f32 (as the full path does): its error
        # grows with the sample index, so the wav and phase bounds are loose.
        ok &= rm < 1e-3 and rw < 5e-2 and dphi < 0.2
        print(f"{n:>5} {str((cb, cn)):>9} | {rm:10.3e} | {rw:10.3e} {dphi:10.3e}", flush=True)
    for p, s in zip(procs, saved):
        p.set_processor(s)
    print("CACHED NUMERICS GATE (mel rel-L2 < 1e-3, wav < 5e-2, phase < 0.2 rad):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
