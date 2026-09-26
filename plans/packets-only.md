# Packets-only speech: plowc emits every stage, plowrt only runs packets

User rule (2026-09-26): all model support goes via plowc packets; plowrt just runs packets (CPU
threads allowed for host tasks); no model-specific ops or segments in plowrt.

## Target shape

- Compute: every stage is a `forward.v1` / `causal.v1` packet program on the generic interpreter
  (`CudaPacketRuntime` for forward packets, `GpuEngine` for causal LMs). New ops are generic
  (named after the math, not the model) with CPU golden + CUDA arm + gate test.
- Orchestration: a packet declares a pipeline graph; plowrt drivers are generic over parameters.
- Host tasks on CPU threads, parameterised by packet metadata: tokenizer (tokenizer.json), text
  rewrite rules, mel/STFT frontend constants, sampling chain, stream windowing.

## Stages to move

| stage | today | target |
|---|---|---|
| SNAC-24k decode | `codec/libplow_snac.so` | `codec.pkt` (forward.v1, frame buckets) |
| S3Gen (encoder + CFM x10 + HiFT) | `codec/libplow_s3gen.so` | `s3gen.pkt`: encoder program, one CFM step program run N times by a generic `ode.euler.v1` loop, vocoder program |
| T3 controller (CFG pairs, voice rows, punc_norm) | `tts/t3.rs`, `tts/chatterbox.rs` | generic guided-LM driver; voice rows + prefill layout as packet tensors/params |
| Veena contract | `tts/mod.rs` (generic-ish) | keep, but codec via packet |
| Qwen3-ASR | `asr/qwen*.rs` (prompt, audio rows) | generic encoder->overlay->causal driver |

## New generic ops (SNAC first)

- GatherRowsF32: out[t] = table[idx[(t / rep) * stride + offset]] (codebook lookup + nearest upsample).
- Conv1dF32: channels-last [T][C], groups (1 or C), kernel, stride, dilation, pad; flags: pre-snake
  (alpha tensor), post tanh, residual add.
- ConvTranspose1dF32: kernel, stride, pad; pre-snake flag.
- NoiseMulAddF32: x += N(0,1)[seed, block, t] * y (counter-based, same hash as snac.cu).

(Inventory of the rest pending.)
