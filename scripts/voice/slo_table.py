#!/usr/bin/env python3
"""slo_table.py calls*.json: call_sim results as one markdown row per run (call count), p50/p95 ms
per stage, underrun turns, errors, wall, and the SLOs each run misses (thresholds from its args)."""
import json
import sys

COLS = (("asr_final", "ASR final", "slo_asr_ms"), ("asr_partial_p50", "ASR partial", None),
        ("llm_ttft", "LLM TTFT", "slo_ttft_ms"), ("tts_ttfa", "TTS TTFA", "slo_ttfa_ms"))


def row(path):
    d = json.load(open(path))
    s, a = d["summary"], d["args"]
    cells, miss = [], []
    for key, name, slo in COLS:
        p50, p95 = s.get(f"{key}_p50_ms"), s.get(f"{key}_p95_ms")
        cells.append("-" if p50 is None else f"{p50:.0f}/{p95:.0f}")
        if slo and p95 is not None and p95 > a[slo]:
            miss.append(f"{name} p95")
    turns = s["turns"] - s["errors"]
    if s["turns_with_underrun_gt_100ms"] > a["slo_underrun_frac"] * max(turns, 1):
        miss.append("underrun")
    if s["errors"]:
        miss.append("errors")
    verdict = "**pass**" if s["slo_pass"] else ", ".join(miss) or "fail"
    return (s["calls"], f"| {s['calls']} | " + " | ".join(cells) +
            f" | {s['turns_with_underrun_gt_100ms']}/{turns} | {s['errors']} | {s['wall_s']:.0f} s | {verdict} |")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 else 2)
    print("| calls | " + " | ".join(n for _, n, _ in COLS) + " | underrun | errors | wall | SLOs |")
    print("|---|" + "---|" * (len(COLS) + 4))
    for _, line in sorted(row(p) for p in sys.argv[1:]):
        print(line)
    print("\ncells p50/p95 ms; underrun = turns with > 100 ms playback underrun / turns without error")


if __name__ == "__main__":
    main()
