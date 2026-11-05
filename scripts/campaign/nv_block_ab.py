#!/usr/bin/env python3
"""Frozen four-arm NVIDIA single-block A/B (the NVIDIA counterpart of `campaign.py block-ab`).

    campaign.py nv-block-ab --control <block assets> --treatment <block assets> \
        --verb bench|prefill-bench --args "--batch 1,8 --ctx 4096" --ceilings c.json --out <dir> --note "..."

`prepare` freezes both block asset dirs, the block_run binary and the plowrt used for the floor,
and writes `run.sh`: ctl / treat / ctl2 / treat2 `block_run` runs with identical arguments in one
lease, then the scorer. `score` (also `nv_block_ab.py score <dir>`, CPU) refuses arms whose packet,
binary or device do not pair up, and per point prints the HARD RULE table (block time = treatment
mean, roofline floor of the treatment packet, % of floor, delta vs current = control mean) with an
A/B verdict: PASS (faster) / REGRESSION only when |delta| exceeds max(control drift, treatment
spread) and the treatment spread is at most 3x the control drift (agent-tools.md §7); otherwise
NOISE, or UNCONVICTABLE when the spread rule fails.
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block_roofline  # noqa: E402

ARMS = ("ctl", "treat", "ctl2", "treat2")
PAIRS = {"ctl": "control", "ctl2": "control", "treat": "treatment", "treat2": "treatment"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(control, treatment, block_run, plowrt, out, verb, args, env, note):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, src in (("control", control), ("treatment", treatment)):
        src = Path(src).resolve()
        if not (src / "model.pkt").is_file() or not (src / "block.json").is_file():
            raise SystemExit(f"{src}: not a block asset dir (model.pkt + block.json)")
        shutil.copytree(src, out / name, symlinks=True)
    for name, src in (("block_run", block_run), ("plowrt", plowrt)):
        shutil.copy2(src, out / name)
        (out / name).chmod(0o755)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items())]
    for arm in ARMS:
        cmd = [str(out / "block_run"), str(out / PAIRS[arm]), verb, *shlex.split(args),
               "--out-dir", str(out / arm)]
        lines.append(f"{shlex.join(cmd)} > {shlex.quote(str(out / (arm + '.log')))} 2>&1")
    run = out / "run.sh"
    run.write_text("\n".join(lines) + "\n")
    run.chmod(0o755)
    record = {"scope": "single-block four-arm A/B; not serving qualification", "note": note,
              "verb": verb, "args": args, "env": env,
              "control_packet_sha256": sha(out / "control/model.pkt"),
              "treatment_packet_sha256": sha(out / "treatment/model.pkt"),
              "block_run_sha256": sha(out / "block_run"), "plowrt_sha256": sha(out / "plowrt")}
    (out / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    return out


def verdict(c1, c2, t1, t2):
    drift, spread = abs(c1 - c2), abs(t1 - t2)
    control, treatment = (c1 + c2) / 2, (t1 + t2) / 2
    delta = treatment - control
    if spread > 3 * drift and spread > 0:
        return "UNCONVICTABLE", delta, drift, spread
    if abs(delta) <= max(drift, spread):
        return "NOISE", delta, drift, spread
    return ("PASS" if delta < 0 else "REGRESSION"), delta, drift, spread


def load_arm(out, arm):
    with open(out / arm / "sweep.json") as f:
        doc = json.load(f)
    prov = doc.get("provenance") or {}
    return block_roofline.measured_points(doc), prov


def score(out, ceilings=None, router_table=None, gpu="h100"):
    out = Path(out).resolve()
    arms = {arm: load_arm(out, arm) for arm in ARMS}
    for arm, (_, prov) in arms.items():
        want = sha(out / PAIRS[arm] / "model.pkt")
        if prov.get("packet_sha256") != want:
            raise ValueError(f"{arm}: measured packet {prov.get('packet_sha256')} is not the frozen {PAIRS[arm]}")
        if prov.get("binary_sha256") != sha(out / "block_run"):
            raise ValueError(f"{arm}: block_run binary differs from the frozen one")
    if len({prov.get("device") for _, prov in arms.values()}) != 1:
        raise ValueError("arms ran on different devices")
    points = set(arms["ctl"][0])
    if not points or any(set(m) != points for m, _ in arms.values()):
        raise ValueError("arms measured different point sets")
    text, desc = block_roofline.disassemble(out / "treatment", str(out / "plowrt"))
    progs = block_roofline.op_roof.parse(text)
    ns = argparse.Namespace(ceilings=ceilings, bw=None, tflops=None, fp8_tflops=None)
    bw, tf, f8, source = block_roofline.op_roof.ceilings(ns, block_roofline.lookup_gpu(gpu))
    router = block_roofline.op_roof.load_router(router_table, progs)
    m = {arm: arms[arm][0] for arm in ARMS}
    treat = {k: (m["treat"][k] + m["treat2"][k]) / 2 for k in points}
    control = {k: (m["ctl"][k] + m["ctl2"][k]) / 2 for k in points}
    rows = block_roofline.build_rows(progs, desc, treat, control, {}, (bw, tf, f8), 2.0, 2.0, router)
    for r in rows:
        k = tuple(r["key"])
        r["verdict"], r["delta_ms"], r["drift_ms"], r["spread_ms"] = verdict(
            m["ctl"][k], m["ctl2"][k], m["treat"][k], m["treat2"][k])
    lines = [f"Ceilings: {bw:.0f} GB/s, {tf:.0f} BF16 / {f8:.0f} FP8 TFLOP/s ({source})", "",
             block_roofline.table(rows), "", "| point | verdict | control drift | treatment spread |",
             "|---|---|---:|---:|"]
    lines += [f"| {r['point']} | {r['verdict']} | {r['drift_ms']:.4f} ms | {r['spread_ms']:.4f} ms |" for r in rows]
    report = "\n".join(lines) + "\n"
    (out / "comparison.md").write_text(report)
    (out / "comparison.json").write_text(json.dumps({"ceilings_source": source, "rows": rows,
                                                     "arms": {a: arms[a][1] for a in ARMS}}, indent=2) + "\n")
    return rows, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("out")
    s.add_argument("--ceilings")
    s.add_argument("--router-table")
    s.add_argument("--gpu", default="h100")
    a = ap.parse_args()
    rows, report = score(a.out, a.ceilings, a.router_table, a.gpu)
    print(report)
    sys.exit(0 if all(r["verdict"] in ("PASS", "NOISE") for r in rows) else 1)


if __name__ == "__main__":
    main()
