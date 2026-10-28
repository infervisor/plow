# H100 speech + Gemma 4 E4B: reproducing assets and numbers

Five recipes under `recipes/infervisor/*/sm90a-h100-tp1.toml` carry everything that decides a
number: pinned checkpoint revision, emit knobs (`[emit.env]`), serve settings (`[serve].env`) and
accuracy gates (`[gates]`). Build and gate from a clean checkout; nothing else is needed.

| Recipe | Checkpoint (revision) | Serve env |
|---|---|---|
| `veena` | maya-research/Veena `8b770f9e` + hubertsiuzdak/snac_24khz `d73ad176` | per-rung Lt algos |
| `qwen3-asr` | Qwen/Qwen3-ASR-1.7B `7278e1e7` | defaults |
| `chatterbox` | ResembleAI/chatterbox `5bb1f6ee` (`s3gen.safetensors`) | defaults |
| `chatterbox-mtl` | ResembleAI/chatterbox `5bb1f6ee` (`t3_mtl23ls_v3`, `s3gen_v3`) | defaults |
| `gemma-4-e4b` | google/gemma-4-E4B-it `ee0ef602` | per-rung Lt algos |

Scheduling is the runtime's default `auto` objective (`docs/flags-reference.md`, "Serving
objective"): adaptive multistep and the lookahead-1 decode pipeline are derived, not set. A
runtime older than the objective (the `plow_git` pin is the compiler commit that emitted the
assets) needs `PLOW_MULTISTEP_ADAPTIVE=true PLOW_DECODE_PIPELINE=true` for Veena and E4B.

## Prerequisites (once per box)

1. Toolchain: `nix develop` (or the equivalent CUDA 12.9 + Rust env; set `PLOW_CAMPAIGN_NO_NIX=1`).
2. `plow_verify`: `cd lean-plow && nix develop -c lake build`. Emit refuses to run without it.
3. Checkpoints at the pinned revisions in `$HF_HOME/hub`:
   `huggingface-cli download <repo> --revision <sha>` for each row above. `{hf:org/name@rev}`
   resolves exactly that snapshot and stops with the download command if it is missing.
4. Python environments (exported):
   - `PYREF`: torch, transformers, snac, openai-whisper (Veena prep, gates, bench client).
   - `CBX_PY`: a venv with `chatterbox-tts` 0.1.7 (Chatterbox prep and gates); for the MTL recipe,
     a venv with the upstream git package, whose `mtl_tts` loads `t3_mtl23ls_v3`.
   - `ASR_MANIFEST`: `python3 scripts/asr/nvidia/get_audio.py <dir>` writes the audio + manifest.
5. For the S3Gen gate: `cargo build --release -p plowrt --features cuda --example packet_run`.

## Build and gate

```
python3 scripts/campaign/campaign.py build recipes/infervisor/<m>/sm90a-h100-tp1.toml --out <dir>
python3 scripts/campaign/campaign.py gate  recipes/infervisor/<m>/sm90a-h100-tp1.toml \
    --assets <dir>/assets --out <dir>/gate
```

`gate` serves with the recipe's `[serve].env` inside one GPU lease. Expected results (clean
rebuild of all five, H100 SXM5):

| Recipe | Gate | Result |
|---|---|---|
| qwen3-asr | WER (LibriSpeech-clean subset) | 3.913% |
| gemma-4-e4b | logit parity vs HF | top1 0.985, KL 8e-4 (gate top1 ≥ 0.98; varies 0.980-0.985 run to run) |
| veena | Whisper CER median, n=80 | 0.005 |
| chatterbox | CER median; S3Gen mel rel-L2 | 0; 1.2e-5 |
| chatterbox-mtl | CER median (worst language fr 0.148); S3Gen mel rel-L2 | 0; 1.2e-5 |

## Performance check

Serve with the recipe's env (`campaign.py serve <recipe> --assets <dir>/assets`, or export the
`[serve].env` keys and run `plowrt serve --assets <dir>/assets`), inside a lease:

- TTS: `perf-data/tools/gpulease -n 1 tts scripts/tts/plow_speech_probe.sh <dir>/assets <res>
  "--conc 64 --stream --n 128"` (Chatterbox: add `--prompt-set chatterbox[-mtl] --voice default`).
- ASR: `scripts/asr/nvidia/served_bench.py --url ... --model qwen3-asr --manifest $ASR_MANIFEST
  --conc 1` (and `--conc 64,128`).
- E4B / Veena tokens: `scripts/bench/llm_grid.sh` (ISL 128 / OSL 512, greedy, per-cell seeds).

Recorded and reproduced numbers, with the vLLM baselines and evidence paths, are in the results
summaries: [TTS](../bringup/results/tts-h100/summary.md),
[Qwen3-ASR](../bringup/results/qwen3-asr-h100/summary.md) (RTFx c64/c128 use `manifest_x4.json`),
[Gemma 4 E4B](../bringup/results/gemma4-h100/summary.md#gemma-4-e4b-voice-agent-llm).

Single runs vary about ±1.5% (c200 MTL steady aps ±1.5% over 6 runs). E4B and Veena served greedy
text can differ between runs at c ≥ 64 (rung assignment follows arrival order); c1 is identical.
`PLOW_LT_RUNG_ALGOS` picks cuBLASLt algorithms by timing at load, so B=48/96 decode digests can
differ between server restarts; drop it from `[serve].env` for restart-stable digests (~1% at B=64).

## Voice agent (ASR + Gemma + Chatterbox MTL on one GPU)

`scripts/voice/serve_voice_agent.sh` co-serves the three built asset dirs with `--co-sched
deadline`; `scripts/voice/call_sim.py` + `slo_table.py` report per-call SLOs.

### Gemma 4 26B FP8 + Qwen3-ASR + Chatterbox-MTL (50 Concurrent Calls Qualified)

To serve **50 concurrent calls** on a single NVIDIA H100 80GB SXM5 co-serving
`Qwen3-ASR (1.7B)` + `Gemma-4-26B-A4B-it FP8` + `Chatterbox-MTL (T3 + S3Gen v3)`:

1. **Continuous Cross-Model Batching**: the unified token batch (always on) unifies execution batches across models.
2. **Vocoder Stream Batch Capping**: `PLOW_TTS_TURN_BATCH=16` caps vocoder launches to 16 parallel streams, bounding render hold time to $\le 440\text{ ms}$ and eliminating tail latency spikes on LLM TTFT and ASR final.
3. **Speculative Decoding**: `PLOW_SPEC=1` with Gemma-4 assistant MTP drafter (`k_max=7`, `rows=16`) reduces LLM decode turns by ~40%.
4. **Deadline Co-Scheduling**: `--co-sched deadline` schedules `Urgency::Final` turns with immediate priority over background batches.
5. **Zero-Latency Yielding**: `PLOW_MULTISTEP=0` ensures decode batches do not block latency-critical turn transitions.

#### Hardware Results (1x H100 80GB SXM5, 3 turns/call)

| calls | ASR final | ASR partial | LLM TTFT | TTS TTFA | E2E first audio | underrun | errors | wall | SLOs |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 104/268 | 105/655 | 47/261 | 340/639 | 915/1626 | 0/90 | 0 | 78 s | **pass** |
| 50 | 121/479 | 76/424 | 59/420 | 363/779 | 1194/2460 | 0/150 | 0 | 86 s | **pass** |

*Cells p50/p95 ms; underrun = turns with > 100 ms playback underrun / turns without error.*

Server-Timing breakdown (p50/p95 ms):

| calls | ASR queue | ASR wait-turn | ASR device | LLM queue | LLM wait-turn | LLM device | TTS queue | TTS wait-turn | TTS device | LLM over SLO |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 32/96 | 69/227 | 0/0 | 0/27 | 10/236 | 22/38 | 1/334 | 220/384 | 43/87 | - |
| 50 | 42/113 | 73/419 | 0/0 | 0/28 | 29/396 | 23/43 | 1/311 | 278/549 | 52/111 | - |

Reproduction: `scripts/voice/bench_gemma26b_voice50.sh`.
