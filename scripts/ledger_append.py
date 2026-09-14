#!/usr/bin/env python3
"""Append measurements to the performance ledger (plans/lean-knob-verification.md §5.1).

The ledger is append-only JSONL, one entry per arm per rung per metric; checkpoint P reads it
(`plowrt perf-cert`, `lean_verify::checkpoints::perf`). Schema:
docs/schemas/perf-ledger-entry-v1.schema.json.

    scripts/ledger_append.py [--ledger PATH] ENTRIES...   # files holding an entry, an array, or JSONL
    ... | scripts/ledger_append.py -                      # the same on stdin
    scripts/ledger_append.py --self-test

Default ledger: $LEDGER_PATH, else /workspace/plow-ledger/ledger.jsonl.

What the writer enforces, so checkpoint P can compute a floor instead of guessing one:
* `samples` (preferred) or `stats` {n, median, mad}; with samples, stats are computed here and
  must agree with any stats given;
* ids are unique;
* a treatment names `control_of` and `repeat_control_of`, both already in the ledger or earlier
  in the same batch, from the same job, hardware, rung and metric.
A stats-only record without `mad`, or with fewer than three samples, is accepted but makes the
rung `insufficient_evidence` in checkpoint P.

Calling it from a harness, one line per run (ctrl, treat, ctrl2, in that order):

    # proto / attrib-trace: per-chunk samples in ms for one rung
    python3 scripts/ledger_append.py - <<EOF
    {"id": "$JOB:P8192-S:ctrl", "job": "$JOB", "harness": "proto", ...,
     "metric": "chunk_ms", "better": "lower", "samples": [664.7, 667.2, 666.8]}
    EOF
    # serve-ab / bench: per-request TTFT from a vLLM bench JSON (`ttfts`, seconds) as samples,
    # output tok/s as one sample per repeated run.

Python harnesses may import `append(entries, path)` and `stats(samples)` instead.
"""
import datetime
import fcntl
import json
import os
import statistics
import sys
import tempfile

DEFAULT = "/workspace/plow-ledger/ledger.jsonl"
HARNESSES = {"proto", "attrib", "bench", "serve-ab"}
REQUIRED = ["id", "job", "harness", "hardware", "rung", "recipe_digest", "knob_delta", "metric",
            "better"]


def stats(samples):
    xs = sorted(samples)
    med = statistics.median(xs)
    return {
        "n": len(xs),
        "median": med,
        "mad": statistics.median(abs(x - med) for x in xs),
        "p95": xs[max(0, round(0.95 * len(xs)) - 1)],
    }


def check(e, known):
    missing = [k for k in REQUIRED if k not in e]
    if missing:
        raise ValueError(f"{e.get('id', '?')}: missing {missing}")
    eid = e["id"]
    if eid in known:
        raise ValueError(f"{eid}: duplicate id")
    if e["harness"] not in HARNESSES:
        raise ValueError(f"{eid}: harness {e['harness']!r} not in {sorted(HARNESSES)}")
    if e["better"] not in ("lower", "higher"):
        raise ValueError(f"{eid}: better must be lower or higher")
    if not {"box", "rocm", "driver", "firmware"} <= set(e["hardware"]):
        raise ValueError(f"{eid}: hardware needs box, rocm, driver, firmware")
    if not {"digest", "role", "rows", "prior", "topology"} <= set(e["rung"]):
        raise ValueError(f"{eid}: rung needs digest, role, rows, prior, topology")
    samples = e.get("samples") or []
    if samples:
        computed = stats(samples)
        given = e.get("stats")
        if given and (given.get("n") != computed["n"] or abs(given["median"] - computed["median"]) > 1e-9):
            raise ValueError(f"{eid}: stats disagree with samples")
        e["stats"] = {**computed, **{k: v for k, v in (given or {}).items() if k not in computed}}
    elif not e.get("stats") or "median" not in e["stats"] or "n" not in e["stats"]:
        raise ValueError(f"{eid}: needs samples or stats with n and median")
    for ref in ("control_of", "repeat_control_of"):
        if ref not in e:
            continue
        c = known.get(e[ref])
        if c is None:
            raise ValueError(f"{eid}: {ref} {e[ref]!r} is not in the ledger")
        for k in ("job", "hardware", "metric"):
            if c[k] != e[k]:
                raise ValueError(f"{eid}: {ref} {c['id']} has a different {k}")
        if c["rung"]["digest"] != e["rung"]["digest"]:
            raise ValueError(f"{eid}: {ref} {c['id']} measured a different rung")
    if ("control_of" in e) != ("repeat_control_of" in e):
        raise ValueError(f"{eid}: a treatment names both control_of and repeat_control_of")
    e.setdefault("date", datetime.date.today().isoformat())
    return e


def read(path):
    known = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    known[e["id"]] = e
    return known


def append(entries, path=None):
    path = path or os.environ.get("LEDGER_PATH", DEFAULT)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        known = read(path)
        lines = []
        for e in entries:
            e = check(dict(e), known)
            known[e["id"]] = e
            lines.append(json.dumps(e, sort_keys=True))
        f.write("".join(line + "\n" for line in lines))
        f.flush()
        os.fsync(f.fileno())
    return len(lines)


def load(text):
    text = text.strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    if text[0] == "{" and "\n{" not in text:
        return [json.loads(text)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def self_test():
    base = {"job": "j", "harness": "proto", "recipe_digest": "r", "knob_delta": {},
            "hardware": {"box": "8xMI300X", "rocm": "7.14", "driver": None, "firmware": None},
            "rung": {"digest": "d", "role": "prefill", "rows": 8192, "prior": 0, "topology": "ordinary"},
            "metric": "chunk_ms", "better": "lower"}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.jsonl")
        ctrl = {**base, "id": "c", "samples": [10.0, 11.0, 12.0]}
        ctrl2 = {**base, "id": "c2", "samples": [10.5, 11.0, 11.5]}
        treat = {**base, "id": "t", "samples": [8.0, 8.5, 9.0], "control_of": "c", "repeat_control_of": "c2"}
        assert append([ctrl, ctrl2, treat], path) == 3
        got = read(path)
        assert got["c"]["stats"] == {"n": 3, "median": 11.0, "mad": 1.0, "p95": 12.0}, got["c"]["stats"]
        for bad, why in [
            ({**ctrl}, "duplicate"),
            ({**base, "id": "x", "samples": [1.0], "control_of": "nope", "repeat_control_of": "c2"}, "not in"),
            ({**base, "id": "y"}, "needs samples"),
            ({**base, "id": "z", "samples": [1.0], "stats": {"n": 1, "median": 2.0}}, "disagree"),
            ({**base, "id": "w", "job": "other", "samples": [1.0], "control_of": "c", "repeat_control_of": "c2"},
             "different job"),
        ]:
            try:
                append([bad], path)
            except ValueError as e:
                assert why in str(e), (why, e)
            else:
                raise AssertionError(f"accepted a bad entry ({why})")
        stats_only = {**base, "id": "s", "stats": {"n": 8, "median": 51.85}}
        assert append([stats_only], path) == 1
        assert load(json.dumps([ctrl, ctrl2])) == [ctrl, ctrl2]
    print("ledger_append self-test ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        return self_test()
    args = argv[1:]
    path = None
    if args[:1] == ["--ledger"]:
        path, args = args[1], args[2:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    entries = []
    for a in args:
        entries += load(sys.stdin.read() if a == "-" else open(a).read())
    n = append(entries, path)
    print(f"appended {n} entr{'y' if n == 1 else 'ies'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
