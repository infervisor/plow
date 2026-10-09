# Performance on one H100 80GB

All numbers: one H100 80GB HBM3 (SXM5, driver 595.91.07), AMD EPYC 7R13 host, this kit's
`plowrt` (commit `4164dc45`), measured with this kit's `perf/` scripts against a server started
by `deploy/plow-voice.sh run`. Every run reported 0 model evictions or switches, 0 out-of-memory
events and 0 failed requests unless a row says otherwise. Single passes; expect a few percent of
run-to-run spread.

## Reproduce

```bash
.venv/bin/python perf/run_perf.py --out results/perf-$(date +%Y%m%d-%H%M)          # ~15 min, full sweep
.venv/bin/python perf/run_perf.py --out results/perf-quick --quick                   # ~5 min
```

It discovers the served models from `/v1/models`, runs each alone at rising concurrency, then the
mixed voice-agent load, and writes `report.md` plus raw JSON. Individual tools:

| script | measures |
|---|---|
| `perf/asr_bench.py` | closed loop over the 73 LibriSpeech clips per concurrency level (each clip once): WER, latency p50/p90, RTFx (audio seconds per wall second); `--rate R` for Poisson open loop with an SLO |
| `perf/tts_bench.py` | streamed speech at concurrency C: time to first audio (TTFA) p50/p90, RTF, audio seconds per wall second |
| `perf/llm_bench.py` | random-token prompts of ISL tokens, OSL generated tokens (ignore_eos), streamed: TTFT, TPOT, tokens/s |
| `perf/voice_agent_load.py` | N concurrent calls x 3 turns (below) |
| `perf/realtime_vad_test.py` | Realtime `server_vad` turn detection: turn boundaries on known pauses, transcript WER |

**Voice-agent load** (`voice_agent_load.py`): each call is an `X-Session-Id` session; per turn the
user speaks a LibriSpeech clip streamed to `/v1/audio/transcriptions` in 1 s chunks at real-time
pace, the transcript goes to a streamed chat completion with the call's history (64 max tokens),
the whole reply is synthesized as streamed PCM and "played" on a real-time clock (an underrun =
audio not there when the player needs it), then the user thinks 1 s and speaks again. Calls start
staggered over 10 s. Per turn: ASR final latency (last audio chunk sent -> final transcript), LLM
TTFT, LLM reply complete, TTS time to first audio, and E2E first audio = end of user speech -> first
agent audio (ASR final + full LLM reply + TTS TTFA; `clients/voice_agent.py` instead streams
sentences to TTS as they are generated and gets first audio in ~300-350 ms at low load). SLO per
run (p95): ASR final <= 500 ms, LLM TTFT <= 800 ms, TTS TTFA <= 800 ms, <= 1% of turns with a
> 100 ms underrun, no errors.

## voice-core (default profile): qwen3-asr + chatterbox-mtl + gemma-4-e4b + Silero VAD

Startup 70 s (warm page cache); 63.5 GiB after startup; peak 74.4 GiB during the sweep.

Voice agent (p50 / p95 ms per turn):

| calls | turns | ASR final | LLM TTFT | LLM reply done | TTS first audio | E2E first audio | underrun turns | SLO |
|---|---|---|---|---|---|---|---|---|
| 16 | 48 | 83 / 149 | 21 / 220 | 195 / 665 | 172 / 513 | 465 / 1285 | 0 | pass |
| 32 | 96 | 112 / 319 | 63 / 353 | 459 / 1374 | 333 / 727 | 911 / 1926 | 0 | pass |
| 64 | 192 | 155 / 573 | 76 / 385 | 899 / 2174 | 474 / 848 | 1614 / 3050 | 0 | ASR final p95 and TTS TTFA p95 over |

**Capacity: 32 concurrent voice calls within every SLO**; at 64 calls nothing fails or underruns,
but the p95 ASR final (573 ms) and TTS first audio (848 ms) pass their 500 / 800 ms bounds.

Each model alone in this profile (the others idle) vs the same model alone at its full packet
context (`single-<model>` profile, same client, same plowrt):

| model | load | voice-core | single model | |
|---|---|---|---|---|
| qwen3-asr | c1 latency p50 / RTFx | 66 ms / 86 | 66 ms / 86 | same |
| qwen3-asr | c16 RTFx (p50) | 500 (179 ms) | 501 (177 ms) | same |
| qwen3-asr | c64 RTFx (p50) | 538 (432 ms) | 538 (437 ms) | same |
| qwen3-asr | WER, every level | 3.913% | 3.913% | release gate 3.913% |
| chatterbox-mtl | c1 TTFA p50 / RTF | 134 ms / 0.122 | 134 ms / 0.122 | same |
| chatterbox-mtl | c8 audio s/s (TTFA p50) | 27.0 (302 ms) | 30.3 (282 ms) | -11% |
| chatterbox-mtl | c32 audio s/s (TTFA p50) | 46.1 (524 ms) | 50.3 (970 ms) | -8% throughput, lower TTFA |
| gemma-4-e4b | c1 TTFT p50 / TPOT p50 | 22.6 / 5.71 ms | 19.7 / 5.71 ms | +3 ms TTFT |
| gemma-4-e4b | c16 out tok/s (TPOT p50) | 1795 (8.57 ms) | 1852 (8.05 ms) | -3% |
| gemma-4-e4b | c64 out tok/s (TPOT p50) | 3169 (13.2 ms) | 4128 (14.6 ms) | -23%: per-request sliding caches (DEPLOY.md section 4) |

Against the release single-model baseline (BASELINE.md, plowrt 09e53e76, same harness for ASR and
TTS; vllm bench serve for E4B):

* qwen3-asr c16 RTFx 420 -> 500: the VMM prefix-cache cost of that release is fixed in this plowrt.
  c1 p50 55.6 -> 66 ms: the Silero no-speech gate runs the VAD over each upload before the model
  (+11 ms for a typical 5-10 s clip; served without the VAD packet: 54.2 ms, c16 RTFx 545).
* chatterbox-mtl c1 TTFA 136 -> 134 ms, c8 32.2 (n=64) -> 27-30 aps (n=16 here).
* gemma-4-e4b c1 TTFT 21.1 ms / TPOT 5.91 ms, c64 4062 output tok/s: single-model numbers match
  (19.7 / 5.71 ms, 4128 tok/s with this client).

## voice-veena: qwen3-asr + veena + gemma-4-e4b + Silero VAD

Startup 34 s; 67.1 GiB after startup; **peak 76.5 GiB**, the least headroom of the profiles.

| calls | turns | ASR final | LLM TTFT | LLM reply done | TTS first audio | E2E first audio | underrun turns | SLO |
|---|---|---|---|---|---|---|---|---|
| 16 | 48 | 96 / 173 | 24 / 128 | 271 / 497 | 104 / 185 | 491 / 732 | 0 | pass |
| 32 | 96 | 123 / 179 | 109 / 295 | 449 / 1332 | 129 / 274 | 717 / 1582 | 1 | underrun 1/96 |
| 64 | 192 | 126 / 259 | 160 / 455 | 827 / 2651 | 202 / 407 | 1197 / 2922 | 15 | underruns |

Capacity: 16 calls within every SLO, about 30 in practice. Veena's first audio is fast (81 ms at
c1, 177 ms p50 at c32), but its stream rate falls with concurrency (RTF 0.30 at c1, 0.48 at c32),
which is what underruns at 32-64 calls. Per model in this profile: veena c1 TTFA 81 ms /
3.4 aps, c8 19.8 aps, c32 52.9 aps (TTFA p90 401 ms); qwen3-asr as in voice-core; gemma-4-e4b c64
2801 tok/s with TTFT p50 1.4 s (its KV budget is smaller beside Veena: long-prompt bursts queue).

## Notes

* All models share the GPU by turns (`--co-sched deadline`, the multi-model default): a request
  owed its first token or audio runs first, then decode throughput, then ASR partials. One model's
  heavy load therefore adds latency to the others, which the voice-agent tables include.
* The Silero VAD runs on the host CPU: about 70 us per stream per 32 ms frame on one core (256
  live streams: 17.6 ms per 32 ms tick on one thread, 3.6 ms on 8). Give the server at least 8
  free CPU cores; plowrt also uses host threads for audio feature extraction and tokenization.
* Startup: 70 s for voice-core with the checkpoints in the page cache; a cold start reads ~22 GB.
