#!/usr/bin/env python3
"""Stage gates for voice_cond.pkt (crates/devgen/src/voice_cond.rs) against cbx_golden.py /
cbx_golden_stages.py arrays, each stage run alone by plowrt's `packet_run` with golden inputs
injected into its input tensors.

  python scripts/tts/voice_cond_check.py --packet ASSETS/voice_cond.pkt \\
      --runner TARGET/release/examples/packet_run --golden GOLDEN_DIR [--cap 10] [--stages front,tok,camp]
"""
import argparse, os, subprocess, sys, tempfile

import numpy as np

PIPE = "voice.cond"


def rel(a, b):
    a, b = np.asarray(a, np.float64).reshape(-1), np.asarray(b, np.float64).reshape(-1)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def run(args, role, ins, outs):
    with tempfile.TemporaryDirectory() as d:
        cmd = [args.runner, args.packet, PIPE, role]
        for name, a in ins.items():
            p = os.path.join(d, name.replace("/", "_") + ".bin")
            np.ascontiguousarray(a).tofile(p)
            cmd += ["--in", f"{name}={p}"]
        for name, _ in outs.items():
            cmd += ["--out", f"{name}={os.path.join(d, name + '.out')}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-4000:])
        res = {n: np.fromfile(os.path.join(d, n + ".out"), dt) for n, dt in outs.items()}
        us = sum(float(l.split()[-1]) for l in r.stdout.splitlines() if l.startswith("stage "))
    return res, us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--stages", default="front,tok,camp")
    a = ap.parse_args()
    G = lambda k: np.load(os.path.join(a.golden, k + ".npy"))
    u32 = lambda v: np.array([v], np.uint32)
    stages, ok = a.stages.split(","), True

    def gate(name, got, ref, tol):
        nonlocal ok
        r = rel(got, ref)
        ok &= r < tol
        print(f"  {name:<28} rel {r:.2e}  ({'ok' if r < tol else 'FAIL'} < {tol:g})")

    w24 = G("wav24_full").astype(np.float32)
    if "front" in stages:
        res, us = run(a, f"stage.front.c{a.cap}", {"pcm": w24, "samples": u32(len(w24))},
                      {"act.vc.w16": np.float32, "act.vc.w16s": np.float32, "act.vc.mel_t3": np.float32,
                       "act.vc.mel_gen": np.float32, "out.vc.prompt_mel": np.float32, "act.vc.fbank": np.float32,
                       "act.vc.tm3": np.uint32, "act.vc.tmg": np.uint32, "act.vc.tmat": np.uint32,
                       "act.vc.tfb": np.uint32})
        print(f"front ({us / 1e3:.2f} ms): lengths tm3 {res['act.vc.tm3'][0]} tmg {res['act.vc.tmg'][0]} "
              f"tmat {res['act.vc.tmat'][0]} tfb {res['act.vc.tfb'][0]}")
        w16 = G("s3gen_wav16_torchaudio")
        gate("G1 torchaudio 16k", res["act.vc.w16"][: len(w16)], w16, 1e-5)
        w16s = G("wav16_librosa")
        gate("   soxr-like 16k (approx)", res["act.vc.w16s"][: len(w16s)], w16s, 5e-3)
        # The T3 path's 16 kHz stream approximates librosa's soxr (FIR rel 1.5e-3), so its mel only
        # tracks the golden; its gate is the T3 token agreement end to end.
        for k, g, tol in (("act.vc.mel_t3", "t3_tok_logmel", 1e-2), ("act.vc.mel_gen", "s3gen_tok_logmel", 1e-5)):
            ref = G(g).T
            gate(f"G2 {g}", res[k][: ref.size].reshape(ref.shape), ref, tol)
        pf = G("s3gen_prompt_feat")[0]
        gate("G3 prompt mel", res["out.vc.prompt_mel"][: pf.size], pf, 1e-5)
        fb = G("campplus_fbank")
        gate("G4 Kaldi fbank", res["act.vc.fbank"][: fb.size], fb, 1e-5)
    if "tok" in stages:
        m3, mg = G("t3_tok_logmel").T, G("s3gen_tok_logmel").T
        res, us = run(a, f"stage.tok.c{a.cap}",
                      {"act.vc.mel_t3": m3, "act.vc.tm3": u32(len(m3)), "act.vc.mel_gen": mg,
                       "act.vc.tmg": u32(len(mg)), "act.vc.tmat": u32(G("s3gen_prompt_feat").shape[1])},
                      {"out.vc.t3_tokens": np.uint32, "out.vc.t3_count": np.uint32,
                       "out.vc.gen_tokens": np.uint32, "out.vc.prompt_len": np.uint32})
        t3 = G("t3_cond_prompt_speech_tokens").reshape(-1)
        n3 = int(res["out.vc.t3_count"][0])
        eq3 = int((res["out.vc.t3_tokens"][: len(t3)] == t3).sum())
        gt = G("s3gen_prompt_token").reshape(-1)
        P = int(res["out.vc.prompt_len"][0])
        eqg = int((res["out.vc.gen_tokens"][: len(gt)] == gt).sum())
        ok &= n3 == len(t3) and eq3 == len(t3) and P == len(gt) and eqg == len(gt)
        print(f"tok ({us / 1e3:.2f} ms): G6 T3 tokens {eq3}/{len(t3)} (count {n3})  s3gen tokens {eqg}/{len(gt)} (P {P})")
    if "camp" in stages:
        fcm = G("camp_fcm")[0].T
        res, us = run(a, f"stage.camp.c{a.cap}", {"act.vc.fcm": fcm, "act.vc.tfb": u32(len(fcm))},
                      {"out.vc.spks": np.float32})
        print(f"camp ({us / 1e3:.2f} ms):")
        gate("G8 spks80 (from golden FCM)", res["out.vc.spks"], G("s3gen_spks80").reshape(-1), 1e-4)
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
