#!/usr/bin/env python3
"""Screen plain W8A16 decode projections from an audited packet inventory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cubin", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(args.audit.read_text())
    cases = {}
    for program in audit["programs"]:
        if program["phase"] != "decode":
            continue
        for inst in program["instructions"]:
            if inst["op"] != "GemvFp8":
                continue
            key = (*inst["i"][:3], inst["blocks"])
            cases.setdefault(key, []).append([program["index"], inst["pc"]])
    if not cases:
        raise RuntimeError("No decode GemvFp8 instructions")
    report = {
        "scope": "Isolated plain GemvFp8 only; no fused GLU, attention, head, interpreter or serving qualification",
        "audit_sha256": sha(args.audit), "packet_sha256": audit["packet_sha256"],
        "probe_sha256": sha(args.probe), "cases": [], "complete": False,
    }
    env = dict(os.environ, PATH="/usr/local/cuda/bin:/usr/bin:/bin",
               LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu")
    env.pop("PLOW_PROBE_CUBIN", None)
    if args.cubin:
        env["PLOW_PROBE_CUBIN"] = str(args.cubin.resolve())
        report["cubin_sha256"] = sha(args.cubin)
        report["scope"] = "Plain GemvFp8 body and loaded interpreter; split16=dependency edges, split32=no edges; no fused GLU, attention, head or serving qualification"
    for key, pcs in sorted(cases.items()):
        log = args.output / ("-".join(map(str, key)) + ".log")
        with log.open("w") as stream:
            result = subprocess.run([str(args.probe.resolve()), *map(str, key)],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        variants = []
        for line in log.read_text().splitlines():
            if line.startswith("M=") and line.endswith(" PASS"):
                fields = dict(word.split("=", 1) for word in line.split()[:-1])
                if tuple(int(fields[x]) for x in ("M", "N", "K")) == key[:3]:
                    variants.append({k: float(v) if k in ("median_us", "min_us", "max_us", "relL2", "max_abs") else int(v)
                                     for k, v in fields.items()})
        passed = result.returncode == 0 and len(variants) == (10 if args.cubin else 8)
        entry = {"shape": key[:3], "packet_blocks": key[3], "pcs": pcs,
                 "variants": variants, "log": log.name, "log_sha256": sha(log), "passed": passed}
        if passed:
            control = variants[7]
            best = min(variants[:8], key=lambda v: v["median_us"])
            entry.update(packet_control_us=control["median_us"], best=best,
                         speedup=control["median_us"] / best["median_us"])
        report["cases"].append(entry)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(key, "PASS" if passed else "FAIL", flush=True)
        if not passed:
            raise RuntimeError(f"Probe failed: {log}")
    report["complete"] = True
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
