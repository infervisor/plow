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
`scripts/asr/nvidia/l4_asr_deploy.sh` installs the three L40S ASR builds the same way as the L4
ones (same objects, same layout).

The recipes keep the L4 contracts at 48 GB sizes: decode ladders to 32 rows and the default
192-chunk packed encoder buckets. One `plowrt serve --assets <1.7B> --assets <0.6B>
--asr-packet nemotron-3.5-asr=...` hosts all three ASR models in 31.5 GiB (peak 31.5 GiB).

## Decode against the roofline

`scripts/bench/step_grid.sh` + `scripts/bench/op_roof.py --gpu l40s` (841 GB/s), ctx 1024, ms per
step:

| model | B=1 | B=8 | B=16 | B=32 | % of roofline B=1 / 16 / 32 |
|---|---|---|---|---|---|
| Qwen3-ASR 1.7B | 5.04 | 6.17 | 7.32 | 9.68 | 84 / 88 / 90 |
| Qwen3-ASR 0.6B | 2.34 | 3.27 | 4.41 | 6.81 | 67 / 85 / 89 |
| Orpheus / Veena (Llama 3B) | 8.96 | 10.19 | 11.37 | 13.81 | 89 / 90 / 91 |
| Chatterbox / MTL T3 | 2.20 | 3.28 | 4.54 | 7.04 | 62 / 81 / 87 |
| Gemma 4 E4B | 13.61 | 14.45 | 15.26 | 16.55 | 82 / 82 / 84 |

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
