"""p2_gate.py <stage.json>...: P2 numerical gate for l2r_layer output against its ref_layer.py dump.

Per boundary: the stage's relative RMS error vs the FP32 reference must be <= 1.5 x the BF16 reference's
(ref_layer.py BF16 mode: same rounding points, torch accumulation order); output cosine >= 0.99999 on the first
and the last step. With L2R_BATCH > 1 every row is gated against its own dump. Thresholds are fixed; do not relax
them to pass."""
import json, sys

RATIO, COS = 1.5, 0.99999


def gate(err, ref_dir):
    meta = json.load(open(ref_dir + "/meta.json"))
    bar = meta["bf16_ref_err"]
    rows, ok = [], True
    for k, e in err.items():
        if e is None:
            continue
        ref = bar["out" if k == "out_last" else k]["rel_rms"]
        if meta.get("pairs") == 0 and k.startswith("out"):
            # an expert group no token of this row routes to: the partial is exactly zero, so it must be reproduced exactly
            good = e[2] == 0.0
            ok &= good
            rows.append(f"{k}:max_abs {e[2]:.2e} (no pairs){'' if good else '!'}")
            continue
        ratio = e[0] / ref if ref else float("inf")
        good = ratio <= RATIO and (not k.startswith("out") or e[1] >= COS)
        ok &= good
        rows.append(f"{k}:{e[0]:.2e}/{ref:.2e}={ratio:.2f}{'' if good else '!'}")
    return ok, rows



if __name__ == "__main__":
    ok_all = True
    for f in sys.argv[1:]:
        s = json.loads(open(f).read().strip().splitlines()[-1])
        ok, rows = gate(s["err"], s["ref"])
        bad_rows = []
        for i, r in enumerate(s.get("rows", [])):
            rok, rrows = gate(r["err"], r["dir"])
            if not rok:
                bad_rows.append(f"row {i} {r['dir'].split('/')[-1]}: " + " ".join(x for x in rrows if x.endswith("!")))
            ok &= rok
        ok_all &= ok
        st = s["step_us"]
        print(f"{'PASS' if ok else 'FAIL'} {s['ref'].split('/')[-1]} b{s.get('batch', 1)} {s['gemv']} step p50 {st['p50']:.2f} "
              f"p99 {st['p99']:.2f} us out cos {s['err']['out'][1]:.8f}")
        print("   ", " ".join(rows))
        for b in bad_rows:
            print("    FAIL", b)
    sys.exit(0 if ok_all else 1)
