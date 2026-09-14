#!/usr/bin/env bash
# Checkpoint P gate on default flips (plans/lean-knob-verification.md §5.3).
#
#   scripts/perf_gate_ci.sh <base-ref>
#
# A knob is flipped when, against <base-ref>, its registry line (crates/devgen/src/knob_spec.rs,
# crates/plowrt/src/knob_spec.rs) changes its default or status and either side is a production
# default or Qualified, or when the body of a `Default::Production` const it uses changes. Each
# flipped knob needs perf-certs/<knob id>.json, written by `scripts/perf_cert.py make`, and
# `scripts/perf_cert.py verify` must accept it: a rejected or insufficient_evidence certificate
# fails the run. A missing certificate fails the run on any runner.
#
# Environment:
#   PLOW_VERIFY_BIN   a plow_verify that implements checkpoint P. Unset (and no lean-plow build),
#                     the run checks the certificates exist, emits a GitHub warning naming the
#                     unverified knobs, and exits 0.
set -euo pipefail
base=${1:?usage: perf_gate_ci.sh <base-ref>}

flipped=$(python3 - "$base" <<'PY'
import re, subprocess, sys

base = sys.argv[1]
paths = ["crates/devgen/src/knob_spec.rs", "crates/plowrt/src/knob_spec.rs"]


def args(text, i):
    out, depth, start = [], 0, i
    while True:
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                out.append(text[start:i].strip())
                return out
            depth -= 1
        elif c == '"':
            i = text.index('"', i + 1)
        elif c == "," and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
        i += 1


def table(text):
    consts = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(
        r"^(?:pub )?const (\w+): (?:Default|Status) = (\w+::\w+)(.*?;)$", text, re.M | re.S)}
    knobs = {}
    for m in re.finditer(r"KnobSpec::new\(", text):
        a = args(text, m.end())
        if len(a) >= 6 and a[0].startswith('"'):
            knobs[a[0].strip('"')] = (a[4], a[5])
    return consts, knobs


def kind(tok, consts):
    head = consts[tok][0] if tok in consts else tok
    return head.split("{")[0].split("(")[0].strip()


def gated(default, status, consts):
    return kind(default, consts) == "Default::Production" or kind(status, consts) == "Status::Qualified"


hit = set()
for path in paths:
    try:
        old = subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        old = ""
    new = open(path).read()
    oc, ok = table(old)
    nc, nk = table(new)
    changed_prod = {n for n, (h, body) in nc.items() if h == "Default::Production" and oc.get(n) != (h, body)}
    for id_, (d, s) in nk.items():
        prev = ok.get(id_)
        if prev != (d, s) and (gated(d, s, nc) or (prev and gated(*prev, oc))):
            hit.add(id_)
        elif d in changed_prod:
            hit.add(id_)
print("\n".join(sorted(hit)))
PY
)

if [ -z "$flipped" ]; then
    echo "perf-gate: no production default or Qualified status changed since $base"
    exit 0
fi
certs=()
for knob in $flipped; do
    cert="perf-certs/$knob.json"
    [ -f "$cert" ] || { echo "::error title=perf certificate missing::$knob changes a production default or Qualified status without $cert (scripts/perf_cert.py make)"; exit 1; }
    certs+=("$cert")
done
if [ -z "${PLOW_VERIFY_BIN:-}" ] && [ ! -x lean-plow/.lake/build/bin/plow_verify ]; then
    echo "::warning title=perf certificates not verified::checkpoint P did not re-run for $(paste -sd, <<< "$flipped"): PLOW_VERIFY_BIN not set on this runner"
    exit 0
fi
python3 scripts/perf_cert.py verify "${certs[@]}"
