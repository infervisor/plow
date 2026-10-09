"""p2_gate.py <stage.json>...: P2 numerical gate for l2r_layer output against its ref_layer.py dump.

Per boundary: the stage's relative RMS error vs the FP32 reference must be <= 1.5 x the BF16 reference's
(ref_layer.py BF16 mode: same rounding points, torch accumulation order); output cosine >= 0.99999 on the first
and the last step. Thresholds are fixed; do not relax them to pass."""
import json, sys

RATIO, COS = 1.5, 0.99999
ok_all = True
for f in sys.argv[1:]:
    s = json.loads(open(f).read().strip().splitlines()[-1])
    meta = json.load(open(s["ref"] + "/meta.json"))
    bar = meta["bf16_ref_err"]
    rows, ok = [], True
    for k, e in s["err"].items():
        if e is None:
            continue
        ref = bar["out" if k == "out_last" else k]["rel_rms"]
        ratio = e[0] / ref if ref else float("inf")
        good = ratio <= RATIO and (not k.startswith("out") or e[1] >= COS)
        ok &= good
        rows.append(f"{k}:{e[0]:.2e}/{ref:.2e}={ratio:.2f}{'' if good else '!'}")
    ok_all &= ok
    st = s["step_us"]
    print(f"{'PASS' if ok else 'FAIL'} {s['ref'].split('/')[-1]} {s['gemv']} step p50 {st['p50']:.2f} p99 {st['p99']:.2f} us "
          f"out cos {s['err']['out'][1]:.8f}")
    print("   ", " ".join(rows))
sys.exit(0 if ok_all else 1)
