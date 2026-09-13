#!/usr/bin/env python3
"""Checkpoint P certificates for knob default flips (plans/lean-knob-verification.md §5.3).

    scripts/perf_cert.py make --knob ID --request REQ.json [--ledger PATH] [--out FILE]
    scripts/perf_cert.py verify CERT.json...
    scripts/perf_cert.py stamp BUILD_JSON CERT.json...

`make` fills the request's `ledger` from the ledger (every entry its touched and serving
treatments name, with their controls), runs `plow_verify P`, and writes
perf-certs/<ID>.json = {"schema", "knob", "request", "certificate"} only when P accepts. The
request is the checkpoint P payload (crates/lean_verify/src/checkpoints/perf.rs): `touched`
[{rung, treat, neutral_evidence?}], `untouched` [{rung, base, variant}] from checkpoint S,
`tier4`, `serving`, `numeric`, `facts`. Committed certificates carry their ledger entries, so
`verify` (and the CI gate, scripts/perf_gate_ci.sh) re-runs P without the ledger.

`stamp` re-verifies each certificate and records it in `build.json` as `rungs[].perf_cert`,
right after `knobs`, where plowrt's load check reads it. Nothing else in the manifest changes.

Verifier: $PLOW_VERIFY_BIN, else lean-plow/.lake/build/bin/plow_verify, else plow_verify on PATH.
Exit: 0 accepted, 1 rejected or insufficient_evidence, 2 usage or verifier missing.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.environ.get("LEDGER_PATH", "/workspace/plow-ledger/ledger.jsonl")


def verifier():
    for p in (os.environ.get("PLOW_VERIFY_BIN"), os.path.join(ROOT, "lean-plow/.lake/build/bin/plow_verify")):
        if p and os.path.isfile(p):
            return p
    return shutil.which("plow_verify")


def run_p(request):
    bin_ = verifier()
    if not bin_:
        sys.exit("perf_cert: plow_verify not found (set PLOW_VERIFY_BIN or `lake build` lean-plow)")
    out = subprocess.run([bin_], input=json.dumps({"checkpoint": "P", "payload": request}),
                         capture_output=True, text=True, check=False)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        sys.exit(f"perf_cert: plow_verify gave no certificate: {out.stderr.strip()[:400]}")


def fill_ledger(request, path):
    known = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                known[e["id"]] = e
    want = [t["treat"] for t in request.get("touched", [])] + list(request.get("serving", []))
    picked = {}
    for eid in want:
        e = known.get(eid)
        if e is None:
            sys.exit(f"perf_cert: {eid} is not in {path}")
        picked[eid] = e
        for ref in ("control_of", "repeat_control_of"):
            if ref in e and e[ref] in known:
                picked[e[ref]] = known[e[ref]]
    return list(picked.values())


def verdict(cert):
    return cert.get("notes") if cert.get("ok") else cert.get("reason")


def make(args):
    opts = dict(zip(args[::2], args[1::2]))
    if "--knob" not in opts or "--request" not in opts:
        print(__doc__, file=sys.stderr)
        return 2
    knob = opts["--knob"]
    request = json.load(open(opts["--request"]))
    if "ledger" not in request:
        request["ledger"] = fill_ledger(request, opts.get("--ledger", LEDGER))
    cert = run_p(request)
    print(verdict(cert))
    if not cert.get("ok"):
        return 1
    out = opts.get("--out", os.path.join(ROOT, "perf-certs", f"{knob}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"schema": 1, "knob": knob, "request": request, "certificate": cert}, f, indent=1)
        f.write("\n")
    print(f"wrote {out}")
    return 0


def check(path):
    doc = json.load(open(path))
    cert = run_p(doc["request"])
    print(f"{path}: {verdict(cert)}")
    return doc, cert.get("ok", False)


def verify(paths):
    return 0 if all([check(p)[1] for p in paths]) else 1


def head(text):
    """End of the `knobs` value and, if the next top-level key is `rungs`, the end of its value."""
    dec = json.JSONDecoder()
    i = text.index("{") + 1
    knobs_end = None
    while True:
        while text[i] in " \t\r\n,":
            i += 1
        if text[i] == "}":
            return knobs_end, None
        k, i = dec.raw_decode(text, i)
        while text[i] in " \t\r\n:":
            i += 1
        _, end = dec.raw_decode(text, i)
        if knobs_end is not None:
            return knobs_end, end if k == "rungs" else None
        if k == "knobs":
            knobs_end = end
        i = end


def stamp(build, paths):
    rungs = {}
    for p in paths:
        doc, ok = check(p)
        if not ok:
            return 1
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        req = doc["request"]
        rows = [(t["rung"], "measured") for t in req.get("touched", [])]
        rows += [(u["rung"], "carry_over") for u in req.get("untouched", [])]
        for rung, basis in rows:
            rungs.setdefault(rung, []).append(
                {"checkpoint": "P", "knob": doc["knob"], "basis": basis, "cert_sha256": sha})
    block = [{"rung": r, "perf_cert": c} for r, c in sorted(rungs.items())]
    text = open(build).read()
    knobs_end, rungs_end = head(text)
    if knobs_end is None:
        sys.exit(f"perf_cert: {build} has no knobs block; emit with checkpoint K first")
    body = json.dumps(block, indent=2).replace("\n", "\n  ")
    text = text[:knobs_end] + f',\n  "rungs": {body}' + text[rungs_end or knobs_end:]
    tmp = build + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, build)
    print(f"stamped {len(block)} rungs into {build}")
    return 0


def main(argv):
    cmd, args = (argv[1], argv[2:]) if len(argv) > 1 else (None, [])
    if cmd == "make":
        return make(args)
    if cmd == "verify" and args:
        return verify(args)
    if cmd == "stamp" and len(args) >= 2:
        return stamp(args[0], args[1:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
