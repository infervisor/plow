# 24 — TTS packet pipelines

Plow represents text-to-speech the way it represents speech recognition
([20 — ASR pipelines](20-asr-pipelines.md)): the compiled asset declares its
driver, programs, tensors and a numeric contract in `packet_pipeline.json`; the
runtime binds that declaration and never names a checkpoint. Serving is the
OpenAI `POST /v1/audio/speech` route on `plowrt serve`.

## Drivers

| Driver | Model family | Stages | Engine |
|---|---|---|---|
| `tts.codec_lm.v1` | Veena / Orpheus: a causal LM whose vocabulary carries audio-codec codes | LM (packet) → SNAC-24k codec | the text engine's continuous-batching mux |
| `tts.t3_cfg.v1` | Chatterbox: T3 speech-token LM with classifier-free guidance | T3 (packet) → S3Gen (encoder, flow matching, vocoder) | its own `GpuEngine`, owned by a speech worker |

Both are emitted by devgen: `tts.codec_lm.v1` with `plowc --tts-profile veena`
(`crates/devgen/src/tts.rs`, family constants in `SpeechProfile`),
`tts.t3_cfg.v1` from the `chatterbox_t3` block the prep script writes into
`config.json` (`scripts/tts/chatterbox_prep.py`).

## Contracts

`tts.codec_lm.v1` carries the causal roles (`prefill.<rows>`, `decode.<rows>`,
`tokens`, `positions`, `kv_lengths`) and, as u64 parameters:

- `prompt.format` (1 = `<spk_{voice}> {input}`), `prompt.prefix.*`, `prompt.suffix.*`, `stop.*`;
- `codec.kind` (1 = SNAC-24k, 7 codes per 85.3 ms frame in Orpheus order),
  `codec.frame_codes`, `codec.codebook`, `codec.frame_samples`, `audio.token_base`,
  `audio.sample_rate`;
- `tokens.per_char_frames_f32`, `tokens.max_new_cap`, `sampling.temperature_f32`,
  `sampling.top_p_f32` (floats as `f32::to_bits`, the pipeline-schema convention).

A voice is valid when its speaker tag is one vocabulary token; the runtime needs
no voice list.

`tts.t3_cfg.v1` adds the embedding handoff roles `overlay` / `overlay_index`
(every prefill row is a host embedding: voice conditioning rows, text
embeddings with learned positions, two BOS rows) and `pos_base` (the per-slot
speech start for the decode embedding), plus `t3.*` parameters: text and speech
control ids, `max_speech_tokens`, `cfg_weight_f32`, `temperature_f32`,
`min_p_f32`, `top_p_f32`, `repetition_penalty_f32`, `s3_valid_below`.

## Device operations

| Op | What | Backends |
|---|---|---|
| `EmbedOverlayBf16` (179) | row = `table[tok]` or a BF16-rounded host overlay row | CPU golden, Metal, CUDA |
| `EmbedPosBf16` (184) | `table[tok] + pos_table[pos - base]` per row | CPU golden, CUDA |

`EmbedOverlayBf16` is the same generic encoder-to-decoder handoff Qwen3-ASR
uses. `EmbedPosBf16` is T3's decode embedding (`speech_emb` + learned
`speech_pos_emb`).

## Codec stages

Codec stages are native CUDA shared objects shipped in the asset's `codec/`
directory, built by the same `plowc --emit devblob+cubin` pass as the
interpreter objects (CMake `PLOW_TTS_SNAC`):

- `codec/libplow_snac.so` + `codec/snac24k.bin` (`runtime/nvidia/snac`): SNAC-24k decode,
  fp32-accurate (three-pass TF32 GEMMs), one CUDA graph per `(batch, frames)` shape.
- `codec/libplow_s3gen.so` + `codec/s3gen.bin` + `codec/voices/*.bin` (`runtime/nvidia/s3gen`).

The codec worker (`crates/plowrt/src/tts/codec.rs`) batches decodes of equal
frame count across concurrent requests into one call, so streams share codec
launches the way they share LM decode steps.

## Serving

`/v1/audio/speech` accepts `model`, `input`, `voice`, `response_format`
(`wav` | `pcm`, 24 kHz mono s16), `stream`, and sampling overrides
(`temperature`, `top_p`, `repetition_penalty`, `seed`, `max_tokens`).

- `tts.codec_lm.v1`: the request is a token-id `Job` on the model's mux. With
  `stream: true` each completed frame decodes a window of 6 frames and emits the
  frames that have 2 frames of right context (`stream_step`); the final flush
  emits the rest. The LM stream is drained on its own task so a slow codec never
  backs up the mux.
- `tts.t3_cfg.v1`: `serve` hands such assets to a Chatterbox worker instead of
  the text registry. The T3 thread batches requests continuously over slot
  pairs (conditional / unconditional), combines their logits on the host
  (`cond + w (cond - uncond)`, repetition penalty, temperature, min_p) and feeds
  the drawn token to both slots; finished token sequences go to the S3Gen
  thread.

## Adding another TTS family

1. Reuse `tts.codec_lm.v1` (a new `SpeechProfile` + codec kind) or `tts.t3_cfg.v1`, or
   introduce a versioned driver when the host state machine differs.
2. Declare roles and the numeric contract in devgen; the runtime binds them.
3. Gate the LM against an fp32 reference on last-prefill logits (rel-L2, top-1), and
   the codec stage against its PyTorch module (rel-L2).
4. Gate the whole pipeline on Whisper round-trip CER (`scripts/tts/asr_check.py`) and
   measure TTFA / RTF / audio seconds per second against the existing serving stack with
   the same client (`scripts/tts/tts_bench.py`).
