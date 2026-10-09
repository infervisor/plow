#!/usr/bin/env python3
"""Performance sweep against a running plow-voice server: every served model alone at rising
concurrency, then the mixed voice-agent workload, with a markdown report.

  python perf/run_perf.py --out results/perf-$(date +%Y%m%d-%H%M) [--quick] [--calls 16,32,64]

Stages (each writes JSON into --out; report.md summarizes all of them):
  ASR   perf/asr_bench.py, closed loop over the 73-clip LibriSpeech set: WER, latency p50/p90, RTFx
  TTS   perf/tts_bench.py, streamed: time to first audio p50/p90, RTF, audio seconds per second
  LLM   perf/llm_bench.py, ISL 1000 / OSL 128: TTFT p50/p99, TPOT p50/p99, tokens per second
  Voice perf/voice_agent_load.py: N concurrent calls x 3 turns (streaming ASR at 1x -> streamed
        chat -> streamed TTS played on a real-time clock), p50/p95 per stage, playback underruns
Models come from GET /v1/models (by their advertised endpoints); --models restricts them.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
MANIFEST = os.path.join(KIT, "data", "librispeech-dummy", "manifest.json")
URL = os.environ.get("PLOW_URL", "http://127.0.0.1:8000").rstrip("/")
TTS_SET = {"veena": ("veena", None), "chatterbox": ("chatterbox", "default"), "chatterbox-mtl": ("chatterbox-mtl", "default")}


def get(path):
    h = {"Authorization": f"Bearer {os.environ['PLOW_API_KEY']}"} if os.environ.get("PLOW_API_KEY") else {}
    with urllib.request.urlopen(urllib.request.Request(URL + path, headers=h), timeout=30) as r:
        return json.loads(r.read())


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  rc={rc} ({time.time() - t0:.0f} s) -> {log}", flush=True)
    return rc


def jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None, help="comma-separated subset of served models")
    ap.add_argument("--quick", action="store_true", help="fewer levels and requests (~5 min)")
    ap.add_argument("--asr-conc", default=None)
    ap.add_argument("--tts-conc", default=None)
    ap.add_argument("--llm-conc", default=None)
    ap.add_argument("--calls", default=None, help="concurrent voice calls per mixed run, e.g. 16,32,64")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--skip", default="", help="comma-separated stages to skip: asr,tts,llm,voice")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    py = sys.executable
    cards = get("/v1/models")["data"]
    want = set(a.models.split(",")) if a.models else None
    models = [c for c in cards if want is None or c["id"] in want]
    asr = [c["id"] for c in models if "audio/transcriptions" in c.get("x_plow_endpoints", [])]
    tts = [c["id"] for c in models if "audio/speech" in c.get("x_plow_endpoints", []) or c["id"] in TTS_SET]
    llm = [c["id"] for c in models if c["id"] not in asr and c["id"] not in tts]
    skip = set(filter(None, a.skip.split(",")))
    asr_conc = a.asr_conc or ("1,16" if a.quick else "1,16,64")
    tts_conc = a.tts_conc or ("1,8" if a.quick else "1,8,32")
    llm_conc = a.llm_conc or ("1,16" if a.quick else "1,16,64")
    calls = a.calls or ("16" if a.quick else "16,32,64")
    meta = dict(url=URL, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), asr=asr, tts=tts, llm=llm,
                asr_conc=asr_conc, tts_conc=tts_conc, llm_conc=llm_conc, calls=calls, turns=a.turns)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"ASR {asr}  TTS {tts}  LLM {llm}")
    if "asr" not in skip:
        for m in asr:
            run([py, f"{HERE}/asr_bench.py", "--model", m, "--manifest", MANIFEST, "--conc", asr_conc],
                os.path.join(a.out, f"asr-{m}.jsonl"))
    if "tts" not in skip:
        for m in tts:
            pset, voice = TTS_SET.get(m, ("chatterbox", "default"))
            for c in map(int, tts_conc.split(",")):
                n = max(8 if a.quick else 16, 2 * c)
                cmd = [py, f"{HERE}/tts_bench.py", "--model", m, "--conc", str(c), "--n", str(n), "--stream",
                       "--prompt-set", pset, "--out", os.path.join(a.out, f"tts-{m}"), "--tag", f"c{c}"]
                if voice:
                    cmd += ["--voice", voice]
                run(cmd, os.path.join(a.out, f"tts-{m}-c{c}.log"))
    if "llm" not in skip:
        for m in llm:
            run([py, f"{HERE}/llm_bench.py", "--model", m, "--conc", llm_conc, "--n", "0"] + (["--osl", "64"] if a.quick else []),
                os.path.join(a.out, f"llm-{m}.jsonl"))
    if "voice" not in skip and asr and tts and llm:
        for n in calls.split(","):
            run([py, f"{HERE}/voice_agent_load.py", "--calls", n, "--turns", str(a.turns), "--asr-model", asr[0],
                 "--llm-model", llm[0], "--tts-model", tts[0], "--voice", TTS_SET.get(tts[0], (0, "default"))[1] or "kavya",
                 "--manifest", MANIFEST, "--seed", n, "--out", os.path.join(a.out, f"voice-calls{n}.json")],
                os.path.join(a.out, f"voice-calls{n}.log"))
    subprocess.run([py, f"{HERE}/report.py", a.out])


if __name__ == "__main__":
    main()
