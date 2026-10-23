#!/usr/bin/env python3
"""Cloned-voice cached windows (`cprefill` + `cvsynth`) of an s3gen.pkt exported with --clone-slots,
--prompt-max and --cache-slots, run by plowrt's `packet_run` example (whole renders: seam past the
window).

(a) the built-in voice written to a voice slot: cprefill + cvsynth vs its own prefill.v0 + csynth
    (same approximation, same noise; gate mel rel-L2 < 2e-5, wav < 1e-2).
(b) cloned voices over a prompt-length sweep: cvsynth vs the Full slot render vsynth (the cached
    approximation, reported next to the built-in voice's csynth vs synth gap; gate finite and
    mel within 4x the built-in gap + 1e-3).
(c) batch mix: one B=4 cvsynth launch over three cache slots vs each item alone (rel-L2 < 1e-5).

  python scripts/tts/s3gen_cslot_check.py --packet ASSETS/s3gen.pkt --runner .../packet_run \\
      --voice-dir PREP/s3gen --golden CLONE_GOLDEN_DIR [--tokens 40,100]
"""
import argparse, json, os, re, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s3gen_packet_check import PIPE, Packet, rel, runiform  # noqa: E402

MEL = 80
PREFILL_SEED = 0x5EED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--voice-dir", required=True, help="the s3gen export (builtin voice tables)")
    ap.add_argument("--golden", required=True, help="cbx_golden.py dir of a long clip (P = 250)")
    ap.add_argument("--tokens", default="40,100")
    a = ap.parse_args()
    pk = Packet(a.packet, a.runner)
    meta = pk._meta()
    caps = lambda kind: sorted({tuple(int(x) for x in re.match(kind + r"\.b(\d+)\.t(\d+)\.", r).groups())
                                for r in meta["programs"] if r.startswith(kind + ".")})
    S, pm, C = pk.params["voice.slots"], pk.params["prompt.max"], pk.params["voice.cache_slots"]
    nh = pk.params["vocoder.harmonics"]
    spt = pk.spt
    print(f"slots {S} cache slots {C} prompt_max {pm}; cvsynth {caps('cvsynth')}")

    from safetensors.numpy import load_file
    w = load_file(os.path.join(a.voice_dir, "model.safetensors"))
    bt = w["voice.prompt_token"].reshape(-1).astype(np.uint32)
    builtin = (bt, w["voice.prompt_feat"].reshape(-1, MEL)[: 2 * len(bt)], w["voice.spks"].reshape(-1))
    G = lambda k: np.load(os.path.join(a.golden, k + ".npy"))
    ct, cf, cs = G("s3gen_prompt_token").reshape(-1).astype(np.uint32), G("s3gen_prompt_feat")[0], G("s3gen_spks80").reshape(-1)
    clone = lambda P: (ct[:P], cf[: 2 * P], cs)

    def run(role, ins, outs, befores=()):
        with tempfile.TemporaryDirectory() as d:
            put = lambda k, x: (np.ascontiguousarray(x).tofile(p := os.path.join(d, k + ".bin")), p)[1]
            cmd = [a.runner, a.packet, PIPE, role]
            for k, (brole, bins) in enumerate(befores):
                cmd += ["--before", brole]
                for n, x in bins.items():
                    cmd += ["--before-in", f"{n}={put(f'b{k}_{n}', x)}"]
            for n, x in ins.items():
                cmd += ["--in", f"{n}={put(n, x)}"]
            for n in outs:
                cmd += ["--out", f"{n}={os.path.join(d, n + '.out')}"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(r.stderr[-3000:])
            return {n: np.fromfile(os.path.join(d, n + ".out"), np.float32) for n in outs}

    def common(toks, seeds, cb, cn):
        tk = np.zeros((cb, cn), np.uint32)
        ph = np.zeros((cb, nh), np.float32)
        for i, (t, sd) in enumerate(zip(toks, seeds)):
            tk[i, :len(t)] = t
            ph[i, 1:] = [(runiform(sd, 2, h, 0) - np.float32(0.5)) * np.float32(2 * np.pi) for h in range(1, nh)]
        return {"codes": tk, "phase": ph, "seam": np.full(cb, 0xFFFFFFFF, np.uint32),
                "next_seam": np.zeros(cb, np.uint32),
                "seed": np.array(list(seeds) + [0] * (cb - len(toks)), np.uint64),
                "lengths.0": np.array([len(t) for t in toks] + [0] * (cb - len(toks)), np.uint32)}

    def tables(voices):
        st, sf, ss, sl = (np.zeros((S, pm), np.uint32), np.zeros((S, 2 * pm, MEL), np.float32),
                          np.zeros((S, MEL), np.float32), np.zeros(S, np.uint32))
        for i, (pt, pf, sp) in enumerate(voices):
            st[i, :len(pt)], sf[i, :len(pf)], ss[i], sl[i] = pt, pf, sp, len(pt)
        return {"slot.prompt_token": st, "slot.prompt_feat": sf, "slot.spks": ss, "slot.prompt_len": sl}

    def pick(kind, b, n):
        return min((c for c in caps(kind) if c[0] >= b and c[1] >= n), key=lambda c: (c[0] * c[1], c[1]))

    def cvsynth(toks, seeds, voices, cslots, outs):
        """Item i reads voice slot i (voices[i]) and cache slot cslots[i]; each distinct cache slot is
        prefilled from the first item naming it."""
        cb, cn = pick("cvsynth", len(toks), max(map(len, toks)))
        tab = tables(voices)
        befores, seen = [], set()
        for i, c in enumerate(cslots):
            if c in seen:
                continue
            seen.add(c)
            bins = {"voice": np.array([i], np.uint32), "cslot": np.array([c], np.uint32),
                    "lengths.0": np.zeros(1, np.uint32), "seed": np.array([PREFILL_SEED], np.uint64)}
            if not befores:
                bins.update(tab)
            befores.append(("cprefill", bins))
        ins = common(toks, seeds, cb, cn)
        ins.update(tab)
        ins["voice"] = np.arange(cb, dtype=np.uint32) % S
        ins["cslot"] = np.array(list(cslots) + [cslots[0]] * (cb - len(toks)), np.uint32)
        return run(f"cvsynth.b{cb}.t{cn}", ins, outs, befores)

    def vsynth(t, seed, voice):
        cb, cn = pick("vsynth", 1, len(t))
        ins = common([t], [seed], cb, cn)
        ins.update(tables([voice]))
        ins["voice"] = np.zeros(cb, np.uint32)
        return run(f"vsynth.b{cb}.t{cn}", ins, ["act.s3gen.mel"])

    def builtin_run(kind, t, seed):
        cb, cn = pick(kind, 1, len(t))
        ins = common([t], [seed], cb, cn)
        ins["voice"] = np.zeros(cb, np.uint32)
        befores = [("prefill.v0", {"voice": np.zeros(1, np.uint32), "lengths.0": np.zeros(1, np.uint32),
                                   "seed": np.array([PREFILL_SEED], np.uint64)})] if kind == "csynth" else []
        return run(f"{kind}.b{cb}.t{cn}", ins, ["act.s3gen.mel", "pcm"], befores)

    ok, rng = True, np.random.default_rng(0)
    cases = [rng.integers(0, 6561, n).astype(np.uint32) for n in map(int, a.tokens.split(","))]
    gm = lambda r, t: r["act.s3gen.mel"].reshape(-1, MEL)[: 2 * len(t)] if r["act.s3gen.mel"].size >= 2 * len(t) * MEL else None

    print("\n== (a) built-in voice in a slot: cprefill + cvsynth vs prefill.v0 + csynth ==")
    gaps = []
    for t in cases:
        seed = 11 + len(t)
        ref = builtin_run("csynth", t, seed)
        full = builtin_run("synth", t, seed)
        got = cvsynth([t], [seed], [builtin], [C - 1], ["act.s3gen.mel", "pcm"])
        rm = rel(got["act.s3gen.mel"][: 2 * len(t) * MEL], ref["act.s3gen.mel"][: 2 * len(t) * MEL])
        rw = rel(got["pcm"][: len(t) * spt], ref["pcm"][: len(t) * spt])
        gap = rel(ref["act.s3gen.mel"][: 2 * len(t) * MEL], full["act.s3gen.mel"][: 2 * len(t) * MEL])
        gaps.append(gap)
        good = rm < 2e-5 and rw < 1e-2
        ok &= good
        print(f"n={len(t):4d}  mel {rm:.2e}  wav {rw:.2e}  (builtin csynth vs synth {gap:.2e})  {'ok' if good else 'FAIL'}")
    bound = 4 * max(gaps) + 1e-3

    print(f"\n== (b) cloned voices: cvsynth vs vsynth over P (gate finite, mel < {bound:.2e}) ==")
    for P in [p for p in (4, 5, 8, 9, 12, 13, 25, 110, 157, 158, 249, 250) if p <= min(pm, len(ct))]:
        t = cases[-1]
        got = cvsynth([t], [7], [clone(P)], [0], ["act.s3gen.mel"])["act.s3gen.mel"][: 2 * len(t) * MEL]
        full = vsynth(t, 7, clone(P))["act.s3gen.mel"][: 2 * len(t) * MEL]
        r = rel(got, full)
        good = bool(np.isfinite(got).all()) and r < bound
        ok &= good
        print(f"P={P:4d}  cached vs full mel {r:.2e}  {'ok' if good else 'FAIL'}")

    print("\n== (c) batch mix: B=4 over cache slots {0, 1, 2, 1} vs alone ==")
    t = cases[0]
    voices = [clone(250), builtin, clone(25), clone(250)]
    cmap = [0, 1, 2 % C, 0]
    mix = cvsynth([t] * 4, [3, 4, 5, 6], voices, cmap, ["pcm"])["pcm"]
    cb, cn = pick("cvsynth", 4, len(t))
    worst = 0.0
    for i, (v, c) in enumerate(zip(voices, cmap)):
        alone = cvsynth([t], [3 + i], [v], [c], ["pcm"])["pcm"][: len(t) * spt]
        worst = max(worst, rel(mix[i * cn * spt:][: len(t) * spt], alone))
    good = worst < 1e-5
    ok &= good
    print(f"worst item rel-L2 {worst:.2e}  {'ok' if good else 'FAIL'}")
    print("CACHED SLOT GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
