# TTS on NVIDIA (sm_90a)

Veena (`maya-research/Veena`) and Chatterbox (`ResembleAI/chatterbox`, English)
compile to plow packets and serve as OpenAI `POST /v1/audio/speech` from
`plowrt serve`. Design: [24 — TTS pipelines](../arch/24-tts-pipelines.md).
Every stage is a plowc packet; plowrt ships no model-specific code or native library.

## Recipes

The checked-in recipes are the source of truth for the knobs and rungs (the commands below
are the same steps by hand):

| model | recipe |
|---|---|
| Veena | `recipes/infervisor/veena/sm90a-h100-tp1.toml` |
| Chatterbox | `recipes/infervisor/chatterbox/sm90a-h100-tp1.toml` |
| Qwen3-ASR | `recipes/infervisor/qwen3-asr/sm90a-h100-tp1.toml` |

```sh
# prep steps (exports) run first; build-record.json pins commit, prep and every hash
python3 scripts/campaign/campaign.py build recipes/infervisor/veena/sm90a-h100-tp1.toml --out $OUT
```

Hosts without nix: `PLOW_CAMPAIGN_NO_NIX=1`, `CARGO_TARGET_DIR` holding a release `plowc`,
`PYREF` (torch + snac) and, for Chatterbox, `CBX_PY` (chatterbox-tts 0.1.7).

Reproduced 2026-09-27 at 7c0a7f7e into fresh directories (H100): every packet
(`model.pkt`, `encoder.pkt`, `codec.pkt`, `s3gen.pkt`) byte-identical to the working assets,
cubins SASS-identical; Qwen3-ASR WER 3.913% (p50 60.1 ms), SNAC parity rel-L2 ≤ 5.3e-6,
Chatterbox CER 0.000, Veena CER 0.026 (the code-mixed prompt is translated by the Whisper judge).

## Veena

```sh
# SNAC export (torch), then LM packet + codec.pkt + paired sm_90a objects
python scripts/tts/snac_export.py $SNAC
PLOW_TTS_CODEC_DIR=$SNAC PLOW_TTS_PROFILE=veena PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 \
  PLOW_EMIT_PREFILL_CUBLASLT=1 PLOW_NO_GLU_FUSE=1 PLOW_FUSE_KV_HNR=1 \
  plowc --hf-dir $VEENA --gpu h100 --arch sm_90a --max-ctx 2048 \
        --emit devblob+cubin --served-name veena --out $ASSETS

plowrt serve --assets $ASSETS --port 8080
curl -s localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"<id from /v1/models>","input":"Hello there","voice":"kavya"}' > out.wav
```

`PLOW_EMIT_PREFILL_CUBLASLT` + `PLOW_NO_GLU_FUSE` route the prefill projections
(including unfused gate/up) to cuBLASLt: TTFT for a 60-token prompt 56.7 → 5.8 ms.
The runtime `dlopen`s `libcublasLt.so`, so the CUDA library directory must be
on `LD_LIBRARY_PATH`.

Voices are the checkpoint's speaker tags (`kavya`, `agastya`, `maitri`, `vinaya`).
Defaults: temperature 0.4, top_p 0.9, no repetition penalty (device sampling; a
penalty switches that request to host sampling).

## Chatterbox

```sh
python scripts/tts/chatterbox_prep.py $T3_HF          # Llama-shaped T3 checkpoint + voice rows
python scripts/tts/s3gen_export.py $S3GEN             # S3Gen weights + voices for the packet
PLOW_TTS_VOCODER_DIR=$S3GEN PLOW_EMIT_PREFILL_CUBLASLT=1 PLOW_NO_GLU_FUSE=1 \
  plowc --hf-dir $T3_HF --gpu h100 --arch sm_90a --max-ctx 2048 --emit devblob+cubin \
        --served-name chatterbox --out $CBX

plowrt serve --assets $CBX --port 8080
```

The T3 asset runs on the shared text mux as guided jobs (two slots per request for
guidance); `s3gen.pkt` renders audio on a render thread.

## Concurrency

Speech requests take the LLM path on each model's mux: packed prefill (several requests' prompt
rows in one launch; the host overlay rows, CFG pairs' conditional and unconditional rows and
the position bases staged per launch row) and, in the same tick, the decode launch. All three
packets carry the packed-prefill contract (hd64/128 attention, `EmbedOverlayBf16` /
`EmbedPosBf16`). The unified token batch (decode rows inside the prefill launch) stays off for
them: it is qualified on hd256/512 only, and overlay rows would need its row order.

Packed prefill of 8 x 60-row prompts, one launch vs serial (H100): Veena 10.0 vs 50.0 ms,
Chatterbox T3 7.1 vs 29.4 ms, Qwen3-ASR 7.5 vs 38.9 ms. A CFG pair's two members are
separate packed requests; the request's first token is drawn once both rows are in.

Admission: requests past the slots queue (4 engine batches of ingress; the ASR front bounds
its own requests in flight at 256 and waits for mux room instead of answering 429). A CFG
request takes two slots, so the 128-slot rung serves 64 Chatterbox requests; the rest queue
(`DECODE_RUNG_MAX` = 128 is the packet format's decode/prefill boundary).

## Model names

`plowc --served-name NAME` writes `served_name` into `weights.json`; `plowrt serve` registers
the bundle under it (default: the HF repo id for a hub-cache `--hf-dir`, else the network
slug). The recipes set `veena`, `chatterbox` and `qwen3-asr`.

## Co-serving

One `plowrt serve --assets $ASR --assets $VEENA --assets $CBX` serves all three from one GPU:
each model has its own mux and KV, and the device changes hands at launch boundaries
(`--co-sched free|rr`). `scripts/tts/plow_coserve_probe.sh` runs each model alone, the
switch latency (`switch_bench.py`) and all three under concurrent load in one lease.

H100, 2026-09-27, reproduced assets, one process:

| | solo (same process) | mixed, `--co-sched free` | mixed, `--co-sched rr` |
|---|---|---|---|
| Qwen3-ASR c4 (RTFx, p50) | 168.6, 134 ms | 21.6, 1323 ms | 25.3, 738 ms |
| Veena stream c8 (audio s/s) | 14.76 | 9.46 | 11.73 |
| Chatterbox full c4 (audio s/s) | 12.84 | 2.82 | 4.86 |

Gates hold co-served (WER 3.913%, CER 0.026 / 0.000); switching between models costs < 0.5 ms
(back-to-back requests alternating across the three vs one model). Under mixed load the GPU is
saturated: persistent cooperative kernels take the whole device, so models alternate, and the
solo-normalized shares sum to ~1.0 (`free`) and ~1.3 (`rr`). Use `rr` when co-serving speech.
Tried and reverted (no gain): single-step ticks while a co-tenant waits, and holding the turn for
a whole encoder/codec/vocoder sequence (worse: a 166 ms S3Gen render then blocks everyone).

## Validation tools (`scripts/tts/`)

| tool | use |
|---|---|
| `veena_ref.py`, `chatterbox_ref.py`, `t3_ref.py` | HF / vLLM / fp32 references |
| `asr_check.py` | Whisper round-trip CER gate |
| `tts_bench.py` | TTFA / RTF / audio s/s client for any `/v1/audio/speech` server |
| `plow_speech_probe.sh`, `vllm_speech_probe.sh` | plowrt vs vLLM+SNAC speech servers, same client, one lease each |
| `snac_export.py`, `s3gen_export.py` | exports the codec lowerings read (`PLOW_TTS_CODEC_DIR`, `PLOW_TTS_VOCODER_DIR`) |
| `s3gen_packet_check.py`, `crates/plowrt/examples/codec_check.rs` | codec packet numerics vs torch / the reference decoders |
| `crates/plowrt/examples/packet_bench.rs` | per-program GPU time; with `PLOW_DEBUG_MAX_INST` per-op costs |
| `snac_check.py`, `sample_kernel_bench.py` | reference SNAC library and sampler numerics/latency |
| `crates/plowrt/examples/t3_check.rs` | T3 logits gate vs `t3_ref.py` |

Every GPU run goes through `perf-data/tools/gpulease`.
