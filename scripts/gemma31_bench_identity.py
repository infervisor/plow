#!/usr/bin/env python3
"""Character-identity gate + median table for two `bench_packed_serve.py` logs.

    python3 scripts/gemma31_bench_identity.py <ctl.json>... -- <cand.json>...

Completions are keyed by (input, output, concurrency, request index) — the prompt for a
request is a pure function of its index (`bench_packed_serve.py: request(length, output, seed)`),
so the same key is the same prompt in both arms. Every repeat of a greedy run must also produce
the same text, which is checked within each arm before the arms are compared.
"""
import json, statistics, sys, collections

ctl_files, cand_files, cur = [], [], None
for a in sys.argv[1:]:
    if a == "--":
        cur = cand_files
        continue
    (cand_files if cur is not None else ctl_files).append(a)


def load(paths):
    texts = collections.defaultdict(set)
    metric = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            key = (r["input"], r["output"], r["concurrency"])
            for i, q in enumerate(r["requests"]):
                texts[key + (i,)].add(q["text"])
            metric[key]["tok_s"].append(r["output_tok_s"])
            metric[key]["ttft"].append(r["ttft_ms"])
            if r["tpot_ms"]:
                metric[key]["tpot"].append(r["tpot_ms"]["p50"])
    return texts, metric


ct, cm = load(ctl_files)
dt, dm = load(cand_files)

# The verdict is on the SET of completions a key produced, not on one representative. A
# repeated-token prompt at concurrency 4 can sit on a near-tie and flip between two
# continuations WITHIN an arm; picking a representative would then report a spurious
# difference (or hide a real one). Equal sets = the arms are indistinguishable, and the
# instability is reported separately as the property of the schedule that it is.
diff = [k for k in sorted(ct) if k in dt and ct[k] != dt[k]]
unstable = [k for k in sorted(set(ct) & set(dt)) if len(ct[k]) > 1 or len(dt[k]) > 1]
missing = sorted(set(ct) ^ set(dt))

print(f"completions compared: {len(set(ct) & set(dt))}")
print(f"missing on one side: {len(missing)}")
print(f"CHARACTER-IDENTICAL: {'YES' if not diff and not missing else 'NO'}"
      f"  ({len(diff)} keys differ)")
print(f"unstable within an arm (same key, repeated greedy runs): {len(unstable)}")
for k in unstable[:5]:
    print(f"  {k}: {len(ct[k])} variant(s) control, {len(dt[k])} candidate"
          f"{' — SAME SET' if ct[k] == dt[k] else ''}")
for k in diff[:5]:
    a, b = sorted(ct[k])[0], sorted(dt[k])[0]
    i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(f"  {k}: first divergence at char {i}: {a[max(0,i-30):i+30]!r} vs {b[max(0,i-30):i+30]!r}")

h = f"{'in/conc':>10}{'ctl tok/s':>11}{'cand tok/s':>12}{'d%':>8}{'ctl TPOT':>10}{'cand TPOT':>11}{'d%':>8}"
print()
print(h)
print("-" * len(h))
for k in sorted(set(cm) & set(dm)):
    a, b = cm[k], dm[k]
    ts, tc = statistics.median(a["tok_s"]), statistics.median(b["tok_s"])
    ps = statistics.median(a["tpot"]) if a["tpot"] else float("nan")
    pc = statistics.median(b["tpot"]) if b["tpot"] else float("nan")
    print(f"{k[0]}/{k[2]:<8}{ts:>11.2f}{tc:>12.2f}{100*(tc-ts)/ts:>7.1f}%"
          f"{ps:>10.2f}{pc:>11.2f}{100*(pc-ps)/ps:>7.1f}%")
