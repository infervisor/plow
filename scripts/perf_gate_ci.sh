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


envs = {}


def table(text):
    consts = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(
        r"^(?:pub )?const (\w+): (?:Default|Status) = (\w+::\w+)(.*?;)$", text, re.M | re.S)}
    knobs = {}
    for m in re.finditer(r"KnobSpec::new\(", text):
        a = args(text, m.end())
        if len(a) >= 6 and a[0].startswith('"'):
            knobs[a[0].strip('"')] = (a[4], a[5])
            envs[a[0].strip('"')] = a[1].removeprefix('Some("').removesuffix('")')
    return consts, knobs


def kind(tok, consts):
    head = consts[tok][0] if tok in consts else tok
    return head.split("{")[0].split("(")[0].strip()


def gated(default, status, consts):
    return kind(default, consts) == "Default::Production" or kind(status, consts) == "Status::Qualified"


def resolved(tok, consts):
    return "".join(consts[tok]) if tok in consts else tok


OFF = re.compile(r'\bFALSE\b|Val::Bool\(false\)|Val::Unset|Val::Nat\(0\)|Val::Str\("(0|off|)"\)')


def off(default, consts):
    text = resolved(default, consts)
    return text.startswith("Default::Static") and bool(OFF.search(text))


hit = {}
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
        was = f"{prev[0]} / {kind(prev[1], oc)}" if prev else "new"
        now = f"{d} / {kind(s, nc)}"
        row = f"{envs.get(id_, '')}\t{was} -> {now}"
        if prev != (d, s) and (gated(d, s, nc) or (prev and gated(*prev, oc))):
            hit[id_] = "flip\t" + row
        elif d in changed_prod:
            hit[id_] = "flip\t" + row
        elif (not prev or resolved(prev[0], oc) != resolved(d, nc)) and not off(d, nc):
            # A non-off default outside Production/Qualified skips checkpoint P: Candidate says
            # so on purpose (reported), OptIn/Diagnostic misdescribes the default (refused).
            cls = "candidate" if kind(s, nc) == "Status::Candidate" else "unqualified"
            hit[id_] = cls + "\t" + row
print("\n".join(f"{k}\t{v}" for k, v in sorted(hit.items())))
PY
)

if [ -z "$flipped" ]; then
    echo "perf-gate: no production default or Qualified status changed since $base"
    exit 0
fi
certs=()
fail=0
knobs=()
while IFS=$'\t' read -r knob class env change; do
    knobs+=("$knob")
    if [ "$class" = candidate ]; then
        echo "::warning title=default-on candidate::$knob ($env): $change. Default-on without a certificate; promoting it to Status::Qualified needs perf-certs/$knob.json (scripts/perf_cert.py make)"
        continue
    fi
    if [ "$class" = unqualified ]; then
        echo "::error title=default-on knob not qualified::$knob ($env): $change. The default is not off but the status claims an opt-in or diagnostic knob, so checkpoint P never sees it: register it Status::Candidate (default-on, evidence pending) or Status::Qualified (evidence + perf-certs/$knob.json, scripts/perf_cert.py make), or default it off"
        fail=1
        continue
    fi
    cert="perf-certs/$knob.json"
    if [ ! -f "$cert" ]; then
        echo "::error title=perf certificate missing::$knob ($env): $change. Needs $cert: a same-binary A/B of the knob's default vs its rollback recorded in the ledger, then scripts/perf_cert.py make --knob $knob --request <request.json>"
        fail=1
        continue
    fi
    certs+=("$cert")
done <<< "$flipped"
[ "$fail" = 0 ] || exit 1
if [ ${#certs[@]} -eq 0 ]; then
    echo "perf-gate: no certificate to verify since $base"
    exit 0
fi
flipped=$(printf '%s\n' "${knobs[@]}")
if [ -z "${PLOW_VERIFY_BIN:-}" ] && [ ! -x lean-plow/.lake/build/bin/plow_verify ]; then
    echo "::warning title=perf certificates not verified::checkpoint P did not re-run for $(paste -sd, <<< "$flipped"): PLOW_VERIFY_BIN not set on this runner"
    exit 0
fi
python3 scripts/perf_cert.py verify "${certs[@]}"
