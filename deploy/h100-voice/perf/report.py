#!/usr/bin/env python3
"""report.py <run dir>: markdown summary of a perf/run_perf.py run (also written to <dir>/report.md)."""
import glob
import json
import os
import sys


def jsonl(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def f(v, nd=0):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    d = sys.argv[1]
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    out = [f"# plow-voice performance run\n\nServer {meta.get('url', '?')}, started {meta.get('started', '?')}.\n"]
    asr = sorted(glob.glob(os.path.join(d, "asr-*.jsonl")))
    if asr:
        out += ["## ASR (73 LibriSpeech clips per level, closed loop)\n",
                "| model | conc | WER | latency p50 / p90 ms | RTFx | errors |", "|---|---|---|---|---|---|"]
        for p in asr:
            m = os.path.basename(p)[4:-6]
            for r in jsonl(p):
                out.append(f"| {m} | {r['conc']} | {100 * r['wer']:.3f}% | {f(r['lat_p50_ms'])} / {f(r['lat_p90_ms'])} | "
                           f"{f(r['rtfx'], 1)} | {r['errors']} |")
        out.append("")
    tts = sorted(glob.glob(os.path.join(d, "tts-*", "c*.json")))
    if tts:
        out += ["## TTS (streamed, 24 kHz PCM)\n",
                "| model | conc | requests | TTFA p50 / p90 ms | RTF p50 | audio s/s | failed |", "|---|---|---|---|---|---|---|"]
        rows = []
        for p in tts:
            s = json.load(open(p))["summary"]
            m = os.path.basename(os.path.dirname(p))[4:]
            rows.append((m, s["conc"], f"| {m} | {s['conc']} | {s['n']} | {f(s.get('med_ttfa_ms'))} / {f(s.get('p90_ttfa_ms'))} | "
                                       f"{f(s['med_rtf'], 3)} | {f(s['audio_s_per_s'], 1)} | {s['failed']} |"))
        out += [r for _, _, r in sorted(rows)]
        out.append("")
    llm = sorted(glob.glob(os.path.join(d, "llm-*.jsonl")))
    if llm:
        out += ["## LLM (random prompts, ISL / OSL as listed, streamed)\n",
                "| model | conc | requests | ISL/OSL | TTFT p50 / p99 ms | TPOT p50 / p99 ms | output tok/s | total tok/s | errors |",
                "|---|---|---|---|---|---|---|---|---|"]
        for p in llm:
            for r in jsonl(p):
                out.append(f"| {r['model']} | {r['conc']} | {r['requests']} | {r['isl']}/{r['osl']} | "
                           f"{f(r['ttft_p50_ms'], 1)} / {f(r['ttft_p99_ms'], 1)} | {f(r['tpot_p50_ms'], 2)} / {f(r['tpot_p99_ms'], 2)} | "
                           f"{f(r['output_tok_s'])} | {f(r['total_tok_s'])} | {r['errors']} |")
        out.append("")
    voice = sorted(glob.glob(os.path.join(d, "voice-calls*.json")), key=lambda p: json.load(open(p))["summary"]["calls"])
    if voice:
        out += ["## Voice agent (concurrent calls x turns; p50 / p95 ms per turn)\n",
                "| calls | turns | ASR final | LLM TTFT | LLM reply done | TTS first audio | E2E first audio | underrun turns | errors | SLO |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for p in voice:
            s = json.load(open(p))["summary"]
            g = lambda k: f"{f(s.get(k + '_p50_ms'))} / {f(s.get(k + '_p95_ms'))}"  # noqa: E731
            out.append(f"| {s['calls']} | {s['turns']} | {g('asr_final')} | {g('llm_ttft')} | {g('llm_total')} | {g('tts_ttfa')} | "
                       f"{g('e2e_first_audio')} | {s['turns_with_underrun_gt_100ms']} | {s['errors']} | "
                       f"{'pass' if s['slo_pass'] else 'miss'} |")
        out.append("\nE2E first audio = end of the user's speech to the first agent audio (ASR final + whole LLM "
                   "reply + TTS TTFA). SLO (per run, p95): ASR final <= 500 ms, LLM TTFT <= 800 ms, TTS first "
                   "audio <= 800 ms, <= 1% of turns with > 100 ms playback underrun, no errors.\n")
    text = "\n".join(out)
    open(os.path.join(d, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
