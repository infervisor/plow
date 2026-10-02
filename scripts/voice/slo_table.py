#!/usr/bin/env python3
"""slo_table.py calls*.json: call_sim results as one markdown row per run (call count), p50/p95 ms
per stage, underrun turns, errors, wall, and the SLOs each run misses (thresholds from its args).
Runs that recorded the server's Server-Timing get a second table: server queue / wait-turn / device
p50/p95 ms per stage, and where the LLM turns over the TTFT SLO spent their time (medians)."""
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


SRV = ("queue", "wait_turn", "device")


def srv_row(path):
    s = json.load(open(path))["summary"]
    if not any(k.startswith("srv_") for k in s):
        return None
    cells = []
    for stage in ("asr", "llm", "tts"):
        for m in SRV:
            p50, p95 = s.get(f"srv_{stage}_{m}_p50"), s.get(f"srv_{stage}_{m}_p95")
            cells.append("-" if p50 is None else f"{p50:.0f}/{p95:.0f}")
    slow = s.get("srv_llm_slow")
    tail = "-" if not slow else (f"{slow['turns']}: " + " / ".join(
        "-" if slow.get(f"{m}_p50") is None else f"{slow[f'{m}_p50']:.0f}" for m in SRV + ("first",)))
    return s["calls"], f"| {s['calls']} | " + " | ".join(cells) + f" | {tail} |"


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 else 2)
    print("| calls | " + " | ".join(n for _, n, _ in COLS) + " | underrun | errors | wall | SLOs |")
    print("|---|" + "---|" * (len(COLS) + 4))
    for _, line in sorted(row(p) for p in sys.argv[1:]):
        print(line)
    print("\ncells p50/p95 ms; underrun = turns with > 100 ms playback underrun / turns without error")
    srv = sorted(r for r in (srv_row(p) for p in sys.argv[1:]) if r)
    if srv:
        names = [f"{st} {m.replace('_', '-')}" for st in ("ASR", "LLM", "TTS") for m in SRV]
        print("\n| calls | " + " | ".join(names) + " | LLM over SLO: n: queue / wait-turn / device / first |")
        print("|---|" + "---|" * (len(names) + 1))
        for _, line in srv:
            print(line)
        print("\nserver Server-Timing p50/p95 ms; wait-turn = admission to first output outside the model's own"
              " ticks; LLM over SLO = medians over turns whose client TTFT missed the SLO")


if __name__ == "__main__":
    main()
