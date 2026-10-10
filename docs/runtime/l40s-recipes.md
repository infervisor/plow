# L40S (Ada, sm_89): ASR, TTS and Gemma 4 E4B

The L40S (AD102: 142 SMs, 48 GB GDDR6, 864 GB/s datasheet, 841 GB/s measured read with
`scripts/tts/hbm_bw.py`) runs the same sm_89 objects as the L4 (`docs/runtime/asr.md`, "L4"):
the sm_120 warp32 interpreter built for Ada, mma.sync and cp.async, no wgmma/TMA, 99 KiB of
shared memory per block. Only the packets differ (142 SMs, `--gpu l40s`).

| model | recipe (`recipes/infervisor/`) | gate | L40S result |
|---|---|---|---|
| Qwen3-ASR 1.7B | `qwen3-asr/sm89-l40s-tp1.toml` | WER, 73 clips | 3.826% |
| Qwen3-ASR 0.6B | `qwen3-asr-0.6b/sm89-l40s-tp1.toml` | WER | 4.261% |
| Nemotron 3.5 (Q8_0 RNNT) | `nemotron-3.5-asr/sm89-l40s-tp1.toml` | WER | 5.13% |
| Orpheus 3B TTS | `orpheus/sm89-l40s-tp1.toml` | Whisper CER median | 0.000 (n=80) |
| Veena TTS | `veena/sm89-l40s-tp1.toml` | Whisper CER median | 0.005-0.016 (n=80, sampled) |
| Chatterbox TTS | `chatterbox/sm89-l40s-tp1.toml` | CER / S3Gen mel rel-L2 | 0.000 (n=32) / 1.4e-5 |
| Chatterbox MTL (23 languages) | `chatterbox-mtl/sm89-l40s-tp1.toml` | CER median / worst language / S3Gen rel-L2 | 0.000 / fr 0.148 / 1.4e-5 |
| Gemma 4 E4B | `gemma-4-e4b/sm89-l40s-tp1.toml` | logit parity vs HF bf16 | top1 0.9867, KL mean 7.8e-4 |

ASR WERs equal the L4 and H100 numbers; the decode change below leaves every gate unchanged.

## Build and serve

Inside `nix develop` (on a box without flakes enabled in `nix.conf`, export
`NIX_CONFIG="experimental-features = nix-command flakes"` so `campaign.py`'s own `nix develop`
works too). `plowrt` loads cuBLASLt from `$CUDA_PATH/lib` when no system toolkit is on the loader
path, so serve from the dev shell (Qwen, Orpheus, Veena and Chatterbox route prefill through it).

```sh
cargo build --release -p plowc && cargo build --release -p plowrt --features cuda,gguf
python3 scripts/campaign/campaign.py build recipes/infervisor/qwen3-asr/sm89-l40s-tp1.toml --out <dir>
python3 scripts/campaign/campaign.py gate recipes/infervisor/qwen3-asr/sm89-l40s-tp1.toml \
  --assets <dir>/assets --out <dir>/gate     # PYREF, ASR_MANIFEST as in the recipe
# Orpheus: --hf-dir <canopylabs or unsloth orpheus-3b-0.1-ft snapshot>
# Chatterbox: CBX_PY=<python with chatterbox-tts 0.1.7> for the T3/S3Gen prep and the S3Gen gate
# Chatterbox MTL: CBX_PY=<python with upstream chatterbox (git), whose mtl_tts loads t3_mtl23ls_v3>

# Nemotron 3.5: packet for 142 SMs + the specialized speech object (`gguf` feature in plowrt)
scripts/asr/nvidia/nemotron_l4_build.sh nemotron-3.5-asr-streaming-0.6b.q8_0.gguf <dir> 142
```

The runtime CMake takes nvcc from `$PLOW_NVCC` (the dev shell's toolkit) when it is set.
The three L40S ASR builds serve with the same `plowrt serve` command line as the L4 ones
([asr.md](asr.md)).

The recipes keep the L4 contracts at 48 GB sizes: decode ladders to 32 rows and the default
192-chunk packed encoder buckets. One `plowrt serve --assets <1.7B> --assets <0.6B>
--asr-packet nemotron-3.5-asr=...` hosts all three ASR models in 31.5 GiB (peak 31.5 GiB).

## Decode against the roofline

`scripts/bench/step_grid.sh` + `scripts/bench/op_roof.py --gpu l40s` (841 GB/s), ctx 1024, ms per
step:

| model | B=1 | B=16 | B=32 | B=64 | B=128 | % of roofline B=1 / 16 / 32 / 64 / 128 | tok/s at 128 |
|---|---|---|---|---|---|---|---|
| Qwen3-ASR 1.7B | 5.02 | 7.30 | 9.67 | 14.38 | 23.82 | 84 / 88 / 90 / 93 / 95 | 5373 |
| Qwen3-ASR 0.6B | 2.33 | 4.44 | 6.80 | 11.51 | 20.87 | 67 / 83 / 88 / 92 / 95 | 6134 |
| Orpheus / Veena (Llama 3B) | 8.96 | 11.38 | 13.66 | 18.48 | 28.14 | 89 / 89 / 91 / 93 / 94 | 4549 |
| Chatterbox / MTL T3 | 2.20 | 4.55 | 7.04 | 12.36 | 22.44 | 62 / 80 / 86 / 89 / 93 | 5705 |
| Gemma 4 E4B | 13.61 | 15.27 | 15.67 | 18.61 | 24.87 | 82 / 82 / 89 / 90 / 91 | 5148 |

The decode ladders run to the 128-row clamp (slots = the widest rung; `auto` picks the rung per
tick). Rungs past 32 would take ceil(B/32) passes of the mma walk (Orpheus B=128 at 69% of the
roofline: projections 43-52%, lm_head 24%), so their projections and lm_head run on cuBLASLt,
which reads the weights once (`PLOW_EMIT_DECODE_CUBLASLT`, now emitted for sm_89 and for
Qwen3/Qwen3-ASR). From 32 rows on Gemma, Orpheus and Veena (B=32 also gains); from 64 on Qwen
and Chatterbox, whose B=32 loses on the library. B=64/128 before -> after, ms: Orpheus
22.07/38.75 -> 18.48/28.14, Gemma 25.06/42.82 -> 18.61/24.87, Qwen 1.7B 16.44/29.93 ->
14.38/23.82, Chatterbox 12.86/24.16 -> 12.36/22.44. Floors are `op_roof.py` per-program floors;
at 128 rows attention (KV) is the largest term, at 98-99% of bandwidth.


Batched decode GEMVs (B >= 2) walk the weights on the tensor cores (`op_gemv_mma.cuh`,
`PLOW_NV_GEMV_MMA`), as on H100. The CUDA-core dot8 walk they replace is issue-bound above one
row: Orpheus B=16 ran at 51% of the roofline (GEMVs 37-57% of bandwidth), now 86%.

| model | B=8 | B=16 | B=32 |
|---|---|---|---|
| Orpheus | 12.22 -> 10.31 | 20.01 -> 11.82 | 20.94 -> 14.30 |
| Gemma 4 E4B | 17.07 -> 14.46 | 27.92 -> 15.28 | 28.70 -> 16.57 |
| Qwen3-ASR 1.7B | 7.81 -> 6.34 | 10.93 -> 7.57 | 14.55 -> 10.02 |

The speech object keeps the dot8 walk: the walk's static reduction smem on top of the 96 KiB
speech arena passes the 99 KiB block limit (the ASR front end fails to load). Measured and not
taken on Ada: the paired walk (`PLOW_NV_GEMV_MMA_PAIR`, +1-3%), the walk at B=1
(`PLOW_NV_GEMV_MMA_B1`, B=1 +1.7%), `PLOW_FUSE_KV_HNR` and GLU fusion (no change). The
`PLOW_NV_DENSE_TUNE` / gemv_k8 arms and decode cuBLASLt are emitted for sm_90a only;
`PLOW_NV_PTXSYNC=3` faults (illegal instruction) and gate sleep 1/16 vs 64 ns is within noise.
The gemv_k8 K-split walk (mma.sync only) builds for Ada with its Hopper guards widened but loses
at B=1: Qwen 1.7B 5.03 -> 5.78 ms, Gemma E4B 13.60 -> 13.77 (k8 to 32 rows: B=16 15.29 -> 14.84,
B=1 13.83), so it stays Hopper-only.

The hd128 recipes (Qwen, Orpheus, Veena) set `PLOW_NV_FA_FOLD`: the last flash split merges, so
each layer loses its FlashMerge level (decode sync costs ~1.1 us per dependency level on Ada).
The streamed Hopper flash kernel is not built for sm_89; the fold runs on the interpreter's item.
B=16 ctx 1024: Qwen 0.6B / 1.7B / Orpheus 4.57/7.57/11.81 -> 4.40/7.32/11.37 ms, gates unchanged
(Chatterbox T3 is hd64, Gemma E4B already folds). At B=1 the large GEMVs run at 89-99% of
bandwidth; the rest is the interpreter skeleton (0.28-0.56 ms) and per-op latency of the small
norm/rope/per-layer-input ops, which is why the ~1.2 GB models (Qwen3-ASR 0.6B, Chatterbox T3)
sit at 62-66% at B=1.

## Serving

73-clip LibriSpeech dummy set (`scripts/asr/nvidia/served_bench.py`), each model alone:

| model | conc | WER | p50 | p90 | RTFx |
|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | 1 / 16 | 3.826% | 125 / 307 ms | 225 / 481 ms | 46.0 / 283.7 |
| Qwen3-ASR-0.6B | 1 / 16 | 4.261% | 66 / 155 ms | 119 / 249 ms | 86.5 / 593.5 |
| Nemotron 3.5 (Q8_0) | 1 / 4 | 5.13% | 60 / 205 ms | 79 / 247 ms | 103.5 / 123.2 |

All three in one serve at once (Qwen c16 + c16, Nemotron c4): WER unchanged; RTFx 111.9 / 114.8
/ 75.5.

Streaming on that serve, all three models, the four longest clips under 20 s: SSE deltas join to
the HTTP transcript (first delta 57-91 ms), the native WebSocket at 16 kHz (partials + deltas) and
48 kHz equals it, OpenAI Realtime (manual commit) equals it, and continuous mode over three clips
with 1 s gaps yields three ordered segments within 0-3.5% WER of the whole-clip transcripts.

TTS (`scripts/tts/tts_bench.py`, streaming):

| model | conc | TTFA median | RTF median | audio s per s |
|---|---|---|---|---|
| Orpheus | 1 / 32 | 141 / 299 ms | 0.75 / 1.21 | 1.34 / 20.3 |
| Veena | 1 / 32 | 141 / 309 ms | 0.75 / 1.22 | 1.34 / 21.2 |
| Chatterbox | 1 / 8 | 173 / 293 ms | 0.16 / 0.31 | 6.5 / 24.7 |
| Chatterbox MTL | 1 / 8 | 175 / 420 ms | 0.17 / 0.33 | 5.9 / 23.9 |

A SNAC codec-LM stream needs ~83 decode tokens per audio second, so BF16 Orpheus/Veena stay real
time per stream up to 16 concurrent streams (11.8 ms steps); at 32 even the 12.5 ms roofline step
is slower than real time.

## Production traffic (one model per L40S, `--objective auto`)

Each model alone on one GPU from its canonical recipe (ladder to 128, `PLOW_VMM_PREFIX=1`),
`plowrt serve --objective auto` with no other serving limits; offered load swept until the SLO
breaks. Goodput counts requests that meet the SLO.

| model | load generator | SLO | holds SLO up to | at that load |
|---|---|---|---|---|
| Gemma 4 E4B | `agentic_turns.py --open-loop` (multi-turn sessions, 512-token system, 300-token turns, 256 out) | TTFT <= 1 s, TPOT <= 50 ms | 32 concurrent sessions, 100% | goodput 3.83 req/s, TTFT p50/p99 195/749 ms, TPOT p99 32 ms, 7.2k tok/s |
| Qwen3-ASR 1.7B | `served_bench.py --rate` (Poisson, 73-clip set) | final <= 1 s | 40 req/s, 99.4% | p50/p99 285/850 ms, 262 audio s/s, WER 3.8% |
| Qwen3-ASR 0.6B | same | final <= 1 s | 80 req/s, 99.4% | p50/p99 219/889 ms, 538 audio s/s, WER 4.2% |
| Orpheus | `call_sim.py` TTS only, 3 turns per call | TTFA p95 <= 800 ms, underrun <= 1% | 48 calls | TTFA p50/p95 482/545 ms, no underrun |
| Veena | same | same | 48 calls | TTFA p50/p95 402/507 ms, no underrun |
| Chatterbox | same | same | 64 calls | TTFA p50/p95 450/770 ms, no underrun |
| Chatterbox MTL | same | same | 32 calls (64: TTFA p95 825 ms) | TTFA p50/p95 350/713 ms, no underrun |

Gemma before this campaign's prefill and cache work (interpreter prefill GEMMs, no prefix
cache): 82% at 8 sessions, 57% at 16 (goodput 1.12 req/s), 0% from 32 (TTFT p50 13.9 s). Each
agentic turn re-prefilled its whole history at 3.5k tok/s; with cuBLASLt prefill (11.3k tok/s)
and ~50% of prompt tokens cached, 32 sessions hold 100%.

On CUDA `auto` steers by live decode width, queue depth and KV share; `--ttft-slo-ms` /
`--tbt-slo-ms` only feed the goodput counters (`plowrt_slo_*`). Past the knee `auto` keeps TPOT
under the 50 ms target and the queue absorbs the excess (Gemma at 48 sessions: TPOT p99 50 ms,
TTFT p50 5.4 s), so the admission rate, not the SLO, bounds TTFT there.

## Production soak (one model per L40S)

Each model alone on one GPU, `plowrt serve --objective auto` with recipe defaults, ~25 min:
warm 0.5x knee, sustained 0.9x (10 min), overload 2x (3 min), recovery 0.5x (5 min), 3x bursts
of 15 s per minute. Mixed shapes: ASR 1-30 s clips at 8-48 kHz over HTTP, SSE, WebSocket and
OpenAI Realtime; multi-turn and long-prompt chat, streamed and not; short and long TTS texts,
streamed and whole, mixed languages on MTL; 2% client disconnects, 1% malformed requests.

| model | 2xx | shed (429 / in-band rate limit) | 4xx (malformed) | 5xx | accuracy after |
|---|---|---|---|---|---|
| Qwen3-ASR 1.7B | 32958 | 176 / 29 | 325 | 0 | WER 0.03826 |
| Qwen3-ASR 0.6B | 61324 | 4358 / 717 | 671 | 0 | WER 0.04261 |
| Nemotron 3.5 | 10486 | 3866 / 647 | 129 | 0 | WER 0.0513 |
| Gemma 4 E4B | 5461 | queues | 15 | 0 | greedy 10-11/11 equal to a fresh server |
| Orpheus | 2276 | queues | 21 | 0 | CER 0.000 |
| Veena | 3364 | queues | 24 | 0 | CER 0.000 |
| Chatterbox | 4191 | queues | 43 | 0 | CER 0.000 |
| Chatterbox MTL | 2108 | queues | 23 | 0 | CER 0.000 |

`/health` answered 200 on every 1 s sample; no panic, CUDA error or restart; latency back to warm
levels within 10 s of recovery; GPU memory and RSS step up at peak concurrency and hold flat
through recovery. Gemma and TTS queue rather than shed, so 3x bursts stretch their tails (Gemma
TTFT p99 10.5 s, Chatterbox TTFA p99 18.7 s). Orpheus and Veena (v7 packets) capped a request at
1200 / 700 generated tokens (~14.6 / 8.5 s of audio) and cut longer input short with a 200; since
then long input is segmented and codec-LM streams are admitted against real time
([tts.md](tts.md#long-inputs-and-real-time-admission-codec-lm-veena-orpheus)): Veena soak at knee 48
0 5xx, underrun >100 ms 0% sustained / 0.4% overload / 2% burst (rest shed with 429 +
Retry-After), CER 0.005; Orpheus holds 0% sustained only at knee 32 (full-length audio is 37% longer
than the clipped v7 audio the old knee of 48 was measured with), and still underruns 10-22% of
admitted streams under 2-3x overload.
