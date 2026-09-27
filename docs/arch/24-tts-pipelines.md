# 24 — TTS packet pipelines

Plow represents text-to-speech the way it represents speech recognition
([20 — ASR pipelines](20-asr-pipelines.md)): the compiled asset declares its
driver, programs, tensors and a numeric contract in `packet_pipeline.json`; the
runtime binds that declaration and never names a checkpoint. Serving is the
OpenAI `POST /v1/audio/speech` route on `plowrt serve`.

## Drivers

Every stage is a plowc-emitted packet on the generic interpreter; plowrt binds drivers by name
and never names a model. Host work (tokenization, text rules, sampling, prompt-row assembly,
stream scheduling) runs on CPU threads from packet metadata.

| Driver | Packet | What the host does |
|---|---|---|
| `tts.codec_lm.v1` | `model.pkt` (causal LM whose vocabulary carries codec codes; Veena) | prompt template + prefix/suffix ids, stop ids, code demux; the LM runs on the text engine's mux |
| `tts.guided_lm.v1` | `model.pkt` (causal LM with classifier-free guidance; Chatterbox T3) | text rules + tokenizer, prefill rows from `in.prompt.*` tensors, CFG slot pairs, sampling chain from `lm.*` parameters |
| `codec.v1` | `codec.pkt` (SNAC), `s3gen.pkt` (S3Gen) | pick the (batch, units) capacity, write codes/tokens, per-item lengths, seeds and voice index, run one program sequence, read PCM |

Emission: `PLOW_TTS_PROFILE=veena` (+ `PLOW_TTS_CODEC_DIR`, the export of
`scripts/tts/snac_export.py`) or a `chatterbox_t3` config block (+ `PLOW_TTS_VOCODER_DIR`, the
export of `scripts/tts/s3gen_export.py`). Both codec packets run on the speech interpreter
object (`interp_sm90a_speech.cubin`), built by `--emit devblob+cubin` when a codec packet is
present.

## Codec packets

- `codec.pkt` (`crates/devgen/src/codec.rs`): SNAC-24k from generic ops — code demux (CopyCols),
  projected codebooks (GatherRows), conv1d / conv-transpose1d with snake, RandF32 noise, tanh.
  One program per `(batch, frames)` capacity; per-item valid lengths and seeds make an item decode
  exactly as it would alone. Parity with the reference decoder: rel-L2 ~4e-6.
- `s3gen.pkt` (`crates/devgen/src/s3gen.rs`): conformer encoder (relative-position attention as
  AttentionF32 + bias), ten CFG Euler steps with baked `t`/`dt`, HiFT vocoder (fp64 SineGen phase
  via CumSumF64, iSTFT as a conv-transpose). One program sequence per `(batch, tokens)` capacity;
  voices are packet tensors selected by index. Mel rel-L2 vs torch fp32 ~1e-6.

The codec driver (`crates/plowrt/src/tts/codec.rs`) batches pending decodes into one launch and
captures every capacity's CUDA graph at load.

## Serving

`/v1/audio/speech` accepts `model`, `input`, `voice`, `response_format`
(`wav` | `pcm`, 24 kHz mono s16), `stream`, and sampling overrides
(`temperature`, `top_p`, `repetition_penalty`, `seed`, `max_tokens`).

- `tts.codec_lm.v1`: the request is a token-id `Job` on the model's mux. With `stream: true`
  each completed frame decodes a window (`stream.window_frames`) and emits the frames that have
  `stream.lookahead_frames` of right context; the final flush emits the rest.
- `tts.guided_lm.v1`: `serve` hands such assets to a guided speech worker (its own engine; two
  slots per request). Tokens stream to the render thread as they are committed; streams
  re-render their prefix every `stream.chunk_tokens` and emit all but `stream.hold_tokens`,
  crossfading `stream.fade_samples`; the LM yields while a first chunk renders.

## Adding another TTS family

1. Reuse a driver (new profile / parameters / strings) or introduce a versioned driver when the
   host state machine differs; declare roles and the contract in devgen.
2. Lower every compute stage to a packet from generic ops (add a generic op with a CPU golden
   and a CUDA arm when one is missing — never a model-specific op or a native library).
3. Gate the LM on last-prefill logits vs fp32 (rel-L2, top-1), each codec packet vs its PyTorch
   module (rel-L2), and the pipeline on Whisper CER (`scripts/tts/asr_check.py`); measure with
   `scripts/tts/tts_bench.py` against the reference stack using the same client.
