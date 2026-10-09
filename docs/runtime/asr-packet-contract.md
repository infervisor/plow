# ASR / VAD packet contract

plowrt serves speech recognition and voice activity as generic executors. Everything that is
model knowledge (graphs, frontend, decode parameters, prompt layout, vocabulary, language, segment
policy) is packet data written by the emitters (devgen, the `asr_*_compile` tools,
`asr_packet_upgrade`). Adding a model of an existing family needs no plowrt code. The schema
lives in `crates/plow-asset/src/speech_contract.rs` (shared by emitters and runtime) and
`crates/plow-asset/src/packet_pipeline.rs`.

## Versioning and refusal

Each speech pipeline declares the parameter `contract`. plowrt implements:

| driver | contract | constant |
|---|---|---|
| `vad.frame.v1` | 1 | `VAD_CONTRACT` |
| `rnnt.greedy.v1` | 1 | `ASR_CONTRACT` |
| `causal.v1` with an audio overlay (audio LM decoder) | 1 | `ASR_CONTRACT` |

Fail closed:

| condition | result |
|---|---|
| contract newer than plowrt's | load error naming both versions |
| contract missing or older (a contract-0 packet, including every `vad.silero.v1`) | load error: re-emit, or run `asr_packet_upgrade` |
| a required parameter, string, tensor role, program role or `asr_vocabulary.json` section missing | load error naming it |
| a VAD op form the host interpreter does not run | load error (`asr::vad::validate`) |
| a device packet op its speech object lacks | module refused ("lacks speech ops ... rebuild") |
| VAD `executor` other than host, a VAD rate other than 16 kHz, an audio-LM `audio.sample_rate`/`audio.max_seconds` other than the serving frontend's | load error |
| policy default outside its declared bound | load error |

`--asr-packet`, `--asr-vad-packet` and `--assets` all go through these checks at startup; a
configured packet that fails them fails startup. The bundle-level `build.json`
`runtime_requires.plowrt_contract` check (`docs/serving-deploy-runtime.md`) stays the gate for
whole-runtime changes.

## VAD: `vad.frame.v1`

A frame classifier run per stream by plowrt's host op interpreter (`asr::vad`). It is not
Silero-specific: any model lowered to these ops and roles runs.

| kind | name | meaning |
|---|---|---|
| program | `step.0` .. `step.{state_banks-1}` | one program per state bank; frame `n` runs bank `n mod state_banks` |
| tensor | `input` | `context_samples + frame_samples` FP32: the previous frame's last `context_samples`, then the new frame |
| tensor | `probability` | one FP32 speech probability |
| tensor | `state.N` | recurrent state (initialised tensors the programs write); per-stream arena |
| param | `contract` = 1, `executor` = 1 (host; 2 = device, not implemented), `state_banks` | |
| param | `sample_rate` (must be 16000), `frame_samples`, `context_samples` | geometry |
| param | `policy.threshold_f32`, `policy.release_offset_f32`, `policy.release_floor_f32` | speech from `threshold`; silence below `max(threshold - release_offset, release_floor)` |
| param | `policy.min_speech_ms`, `policy.min_silence_ms`, `policy.speech_pad_ms`, `policy.max_speech_ms` (0 = none), `policy.min_silence_at_max_speech_ms` | `/v1/audio/vad` defaults |
| param | `gate.min_speech_ms`, `gate.min_silence_ms`, `gate.speech_pad_ms`, `gate.min_total_speech_ms` | the no-speech upload gate |
| param | `bounds.max_duration_ms` | largest duration a request may set |

Ops the host interpreter runs: `Conv1dF32` (groups, stride, dilation, zero/reflect pad, fused
ReLU), `DenseGemmF32`, `CopyColsF32`, `BinaryF32`, `UnaryF32` (identity, tanh, exp, abs,
sigmoid, ReLU, sqrt), `ScaledAddF32`, `LstmCellF32`. Requests override `threshold` (0..=1) and the
durations (`<= bounds.max_duration_ms`) on `/v1/audio/vad`; Realtime `server_vad.threshold` sets
the turn detector's threshold, the release offset stays the packet's.

Why the host: a step is ~70 us of FP32 work on one core, every 32 ms per stream; a device launch
per stream-frame would cost more than the step. `executor` keeps the choice in the packet.

## RNNT / TDT: `rnnt.greedy.v1`

Greedy transducer over packet programs (`asr::rnnt`). Parameters existing before contract 1:
`blank_id`, `max_symbols_per_frame`, `frames`, `joint_batch_max`, `tdt.durations` +
`tdt.duration.N` (TDT), `frame_transform.*`, `input_frames`, the log-mel frontend
(`audio.frontend.*`, `audio.sample_rate`, `audio.min_samples`, `audio.max_samples`, tensor
`audio.frontend.filterbank`) and the cache-aware stream (`stream.*` programs, tensors, geometry).
Contract 1 adds the output, so the runtime needs no tokenizer file:

| kind | name | meaning |
|---|---|---|
| section | `asr_vocabulary.json` | `{"pieces": [...]}`, token id -> piece |
| string | `output.detokenizer` | `sentencepiece` (the one detokenizer) |
| string | `output.word_boundary` | piece prefix that starts a word (`▁`) |
| string | `output.no_space_before` | newline list of pieces glued to the previous word |
| param | `output.skip_bracketed` | 1: pieces `<...>` are not text |
| string | `language` | the one language a prompted packet transcribes; absent = the model detects it |
| string | `language.aliases` | `requested=language` lines accepted for `language` |

## Audio LM: `causal.v1` + audio overlay

The decoder packet (`model.pkt`, `overlay_rows > 0`) plus its encoder sidecar
(`encoder.packet` string -> `forward.v1` pipeline `audio.encode`: programs `forward.*`/`packed.*`,
the log-mel frontend parameters and filterbank, `input.chunk_frames`, `encoder.frame_stride`,
`attention.window_rows`). Prompt and output: `prompt.messages` (chat-template messages with
`{context}`), `audio.marker`, `audio.token_id`, `prompt.language_suffix` (`{language}`),
`prompt.context_forbidden`, `output.text_marker`, `output.language_prefix`,
`output.language_none`, `languages`, `language.aliases`, `stop.count`/`stop.N`,
`output.max_tokens`. Contract 1 adds the host policy (`AudioLmPolicy`):

| param | meaning |
|---|---|
| `finalize.padding_samples`, `finalize.padding_amplitude_f32` | low-level noise after a stream's last piece so the model hears it end |
| `output.reserve_per_row`, `output.reserve_extra` | context positions kept for the transcript: `per_row * audio rows + extra` |

## Capabilities

What the API exposes follows from the packet: languages (`languages` / `language`), streaming
(a cache-aware `stream.*` program set enables incremental RNNT sessions; otherwise partials re-run
the offline path), token streaming (RNNT emits a delta per token). Timestamps are not part of
contract 1.

## Generic runtime policy (stays in plowrt)

API limits and session policy are the server's, not a model's: uploads 0.5-30 s (an audio-LM
packet declaring other limits is refused), 8-48 kHz input resampled to 16 kHz, the energy
endpointer used without a VAD packet (20 ms frames, 12 dB over the noise floor), turn timing
(100 ms onset, 200 ms context, overlong cuts at the quietest frame), Realtime defaults
(OpenAI's `server_vad` 0.5 / 300 / 500 ms), partial-transcript cadence and local agreement.

## Upgrading contract-0 packets

`asr_packet_upgrade` rewrites only the metadata sections (`packet::devbuild::
replace_metadata_sections`): programs, tensors, weights and every other section stay byte for
byte, so numerics and object pairing are unchanged. The values it writes are the ones the
emitters now write.

```sh
cargo build --release -p plowrt --features gguf --example asr_packet_upgrade
asr_packet_upgrade vad      silero_vad.pkt               silero_vad.v1.pkt
asr_packet_upgrade rnnt     nemotron.pkt MODEL.gguf      nemotron.v1.pkt     # also parakeet.pkt
asr_packet_upgrade audio-lm qwen3-asr/model.pkt          model.v1.pkt        # 1.7B and 0.6B
```

`encoder.pkt` sidecars need no change. A kit carrying upgraded packets needs new sha256 entries
(`PAIRING.txt`, `SHA256SUMS`, `MANIFEST.json`).

## Audit (2026-10-09): model logic that was in plowrt

Class (a) = generic runtime mechanism, kept; (b) = model knowledge, moved to the packet.

| where (before) | what | class | now |
|---|---|---|---|
| `asr/vad.rs` driver `vad.silero.v1`, fixed two banks | Silero-named driver | b | `vad.frame.v1`, `state_banks` |
| `asr/vad.rs` `SegmentOptions::DEFAULT` (0.5 / 250 / 100 / 30 / inf / 98 ms) | Silero `get_speech_timestamps` defaults | b | `policy.*` |
| `asr/vad.rs` `threshold - 0.15`, floor 0.01 | Silero hysteresis | b | `policy.release_*` |
| `asr/endpoint.rs:157` same hysteresis in turn detection | Silero hysteresis | b | `policy.release_*` |
| `asr/serving.rs:1040-1043` gate 150 / 150 / 0 ms, 250 ms total | no-speech gate tuned to Silero | b | `gate.*` |
| `asr/serving.rs:1728` continuous-mode threshold 0.5 | Silero default | b | `policy.threshold_f32` |
| `asr/vad.rs` op interpreter, arena, AVX2/FMA, segment algorithm | generic | a | kept |
| `asr/packet.rs` vocabulary from GGUF `asr.tokenizer.vocab` | NeMo GGUF layout | b | `asr_vocabulary.json` |
| `asr/packet.rs` language from GGUF `asr.rnnt.prompt_dictionary` + `prompt_index` | NeMo prompt dictionary | b | `language` |
| `asr/packet.rs` `en-US` accepts `en`/`english` | Nemotron language alias | b | `language.aliases` |
| `asr/rnnt.rs` `detokenize_sentencepiece` (`▁`, `<...>`, `. ? ! । ॥`) | tokenizer rules | b | `output.*` |
| `asr/rnnt.rs` greedy RNNT / TDT loop, frame transform, stream | generic, packet-parameterised | a | kept |
| `asr/audio_lm.rs:558-563` 1 s padding at 100/32768 | Qwen-tuned finalisation | b | `finalize.*` |
| `asr/audio_lm.rs:609-613` reserve `rows + 64` | Qwen output-rate heuristic | b | `output.reserve_*` |
| `asr/audio_lm.rs` prompt render, parse, context fit, language aliases | packet strings already | a | kept |
| `asr/serving/shared.rs:70,86` `frames * 10` ms | 10 ms mel hop | b | packet hop |
| `asr/serving/shared.rs:431` `STABLE_MARGIN_FRAMES = 4` | Whisper STFT reach | b | from packet `fft`/`hop` |
| `asr/frontend.rs` `QwenFrontend` (Whisper 128 mel, 400/160, log10, max-8) | Qwen frontend | b | test oracle + Metal only; serving uses the packet frontend |
| `asr/frontend.rs` `PacketLogMelFrontend`, resampler, WAV | generic | a | kept |
| `asr/audio_lm/cuda.rs` | packet encoder + overlay decoder | a | kept |
| `asr/serving*.rs` 30 s / 0.5 s, 16 kHz, Realtime defaults, partial cadence | API / session policy | a | kept |
| `asr/nemotron/*` (918 lines) | NeMo GGUF -> plan builders, NeMo C FFI | b | not on any serving path; used by the compile/Metal-reference examples. Open: move beside the examples |
| `asr/conformer.rs`, `asr/subsampling.rs` | CPU reference graphs | b / a | reference only. Open |
| `exec/apple/asr.rs` (1916 lines), `asr/audio_lm/metal.rs` | hand-written Qwen audio encoder on Metal | b | Metal only. Open: run `encoder.pkt` as CUDA does |
