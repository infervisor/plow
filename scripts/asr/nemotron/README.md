# Nemotron 3.5 ASR support

`reference.py` runs NVIDIA's official NeMo-Speech.cpp on Apple Metal and records
the exact command, asset hashes, raw JSON, diagnostics and process duration.
This script is the external correctness control. The native packet path and
Plow HTTP/WebSocket serving are described below.

The requested [checkpoint](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
uses cache-aware FastConformer + an RNNT recurrent predictor/joint network.
It requires a different decoder and streaming state from Qwen ASR.

## Pinned assets

- Model revision: `ea30d66debe3740a08b573244286791d423d6b3e`.
- [Q8_0 GGUF](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf):
  742090464 bytes; SHA256 `3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1`.
- [Official macOS ARM64 Metal release](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.1.0/nemo-speech-0.1.0-macos-aarch64-metal.tar.gz):
  SHA256 `f1dff4f9dd9c96214f8cb78b982812459132df8a4ad1a42409fd94de4a366244`.
- Model license: OpenMDW 1.1; runtime: Apache 2.0. Preserve the release's license files.

Download these files, verify their hashes before extraction/use, and extract the
runtime archive to a local directory. No model conversion or Python ML package
installation is required. The runner uses Python 3.11+ standard library only and
checks the pinned GGUF hash on every invocation. It records the executable hash;
verify the entire downloaded release archive separately.

Current workspace assets are under `plans/asr-eval-assets/nemotron-3.5/`.

## Native packet path

The compiler emits a self-describing `rnnt.greedy.v1` packet containing the
frontend contract, shared weights, duration-capacity encoder programs, predictor
state and joint rungs. The controller and packet contract are backend-neutral;
Metal is the currently verified optimized executor.

```sh
nix develop -c cargo run --release -p plowrt --features metal,gguf \
  --example asr_nemotron_pipeline_compile -- \
  plans/asr-eval-assets/nemotron-3.5/model.gguf \
  200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000 \
  /tmp/nemotron.pkt 16 16

nix develop -c cargo run --release -p plowrt --features metal,gguf \
  --bin plowrt -- \
  asr --packet /tmp/nemotron.pkt \
  --tokenizer plans/asr-eval-assets/nemotron-3.5/model.gguf \
  --audio AUDIO.wav --language en-US --backend metal

# Omit --audio to serve HTTP and the WebSocket replay protocol.
nix develop -c cargo run --release -p plowrt --features metal,gguf \
  --bin plowrt -- \
  asr --packet /tmp/nemotron.pkt \
  --tokenizer plans/asr-eval-assets/nemotron-3.5/model.gguf \
  --backend metal --port 8080 --websocket
```

The optional final compiler argument overrides the decoder-visible trailing
encoder frames. Nemotron 3.5 defaults to two right-context windows: six frames
for its declared three-frame offline lookahead. This was selected by a 2..8
frame sweep, not by changing the portable runtime.

On the 897-file Coval `stt-v3` corpus, the 200-frame ladder completed without
errors on M4 Pro at 64.40x realtime. Whisper normalization revision 2 with
`whisper-normalizer` 0.1.12 and `jiwer` 4.0.0 produced 6.1585% mean item WER and
6.2491% corpus WER. NVIDIA's official Metal runtime on the same files measured
62.64x realtime, 6.1862% mean item WER and 6.2156% corpus WER. Plow therefore
improves mean item WER and throughput; the official runtime remains ahead by
0.0335 percentage points on corpus WER. Local evidence is recorded in
`plans/asr-coval-stt-v3-nemotron-tight-trailing6-score.json`.

## In-process Plow adapter

Build the optional adapter and run the model inside the Plow process:

```sh
nix develop -c cargo build -p plowrt --release --features nemo-asr \
  --example asr_nemotron
nix develop -c target/release/examples/asr_nemotron \
  plans/asr-eval-assets/nemotron-3.5/nemo-speech/lib/libnemo_speech_asr_c.1.dylib \
  plans/asr-eval-assets/nemotron-3.5/model.gguf \
  plans/asr-eval-assets/dev-clean-100/652-129742-0016.wav 0 5
```

Use `cpu` instead of `0` for CPU execution. The Rust adapter implements Plow's
backend-neutral `Transcriber` interface and dynamically loads NVIDIA's stable C ABI.
It validates GGUF structure plus `general.architecture=asr` and
`asr.head_type=rnnt` before loading that library.
The runtime library provides its own kernels; this is in-process Plow integration,
not native packet execution. The same adapter can load a compatible Linux CUDA or
CPU library without Metal types in shared code.

The native bring-up keeps the published Q8_0 blocks in GGUF and exposes a
backend-neutral scalar matvec oracle. Inspect the Q8 inventory or verify one
matrix with:

```sh
nix develop -c cargo run --release -p plowrt --features gguf \
  --example asr_nemotron_q8 -- \
  plans/asr-eval-assets/nemotron-3.5/model.gguf --list
nix develop -c cargo run --release -p plowrt --features metal,gguf \
  --example asr_nemotron_q8_metal -- \
  plans/asr-eval-assets/nemotron-3.5/model.gguf \
  decoder.prediction.dec_rnn.lstm.ih_l0.weight 50 2 4
nix develop -c cargo run --release -p plowrt --features metal,gguf \
  --example asr_nemotron_q8_gemm_metal -- \
  plans/asr-eval-assets/nemotron-3.5/model.gguf \
  encoder.layers.0.feed_forward1.linear1.weight 128 50 64
nix develop -c cargo run --release -p plowrt --features gguf \
  --example asr_nemotron_frontend -- \
  plans/asr-eval-assets/nemotron-3.5/model.gguf AUDIO.wav /tmp/nemotron-mel.f32
nix develop -c cargo run --release -p plowrt --features metal,gguf \
  --example asr_nemotron_subsampling_metal -- \
  plans/asr-eval-assets/nemotron-3.5/model.gguf AUDIO.wav 50
nix develop -c cargo run --release -p plowrt --features metal,gguf \
  --example asr_nemotron_projection_metal -- \
  plans/asr-eval-assets/nemotron-3.5/model.gguf AUDIO.wav 50
```

The following Metal component measurements are lower-level gates for the native
packet executor.
Its launch accepts 1, 2, 4 or 8 SIMD groups per threadgroup and one or four
output rows per SIMD group for GEMV. GEMM accepts 32- or 64-frame tiles; the
M4-informed selector uses 64 at M >= 192 or N >= 2048 and exposes the choice
for per-device tuning. Neither choice changes the GGUF asset.

The M4 Pro GEMM gate matched the scalar f64 oracle on real FastConformer Q8_0
weights, including M=33/N=3 tails in unit coverage. At M=128, the 64-frame tile
ran the `[1024,4096]` first feed-forward projection in 387.7 microseconds median
(2.77 effective TOPS); the 32-frame tile ran the `[4096,1024]` second projection
in 488.0 microseconds (2.20 effective TOPS) and `[1024,1024]` attention in 121.9
microseconds (2.20 effective TOPS). These isolated dispatch timings exclude
model orchestration and are selection data, not end-to-end ASR performance.

The backend-neutral CPU frontend reads its geometry and embedded F32 mel basis
from GGUF. On `121-127105-0003.wav`, all 98,944 output values were compared with
an independent Torch STFT + librosa Slaney reference: max absolute error was
`3.36e-4`, mean absolute error `1.46e-6`, shape `773x128`, and the invalid final
centered frame was zero. Streaming carry handling and a device frontend remain.

The causal 8x subsampling gate consumes the canonical F16 convolution weights.
Its portable scalar path matches an independent PyTorch recomputation over all
426,496 outputs with `1.10e-3` max and `2.26e-6` mean absolute error. On M4 Pro,
the five-kernel Metal path produces `[98,17,256]` from `[773,128]` in 1.285 ms
median over 50 runs, with `4.02e-5` max and `9.75e-7` mean error against the
scalar oracle. It uses one command buffer and two ping-pong intermediate buffers.
The operation sequence, tensor views, shapes, padding and activations live in the
backend-neutral `asr::subsampling` plan. Nemotron only binds its GGUF names and
metadata; Metal only uploads buffers and dispatches supported plan operations.
CPU uses the same plan, and future CUDA/HSA adapters do not need family-specific
model code.
The Q8 pre-encoder projection is also implemented through the backend-neutral
`Q8LinearPlan`, which owns the canonical Q8_0 matrix and F32 bias contract. The
GGUF loader binds `encoder.pre_encode.out`; the Metal adapter consumes only the
generic linear plan. On the same real input it maps `[98,4352]` to `[98,1024]`
and matches the scalar f64 oracle within `1.04e-3` max and `4.06e-5` mean absolute
error. Warm isolated Metal medians varied from 0.504 to 0.779 ms across three
separate 20/50-repeat processes; this process-to-process range is recorded rather
than used to retune the portable selector. FastConformer blocks remain separate gates.

FastConformer operation order, layer normalization, Macaron feed-forwards,
relative-position attention, chunk-limited masking, causal GLU convolution and
residuals now have a backend-neutral plan plus scalar CPU oracle. Nemotron code
only binds GGUF metadata and tensor names. A five-frame layer-0 run using the real
Q8_0/F16/F32 tensors matched `block_reference.py`, an independent NumPy/GGUF
implementation, within `1.53e-5` max and `1.24e-6` mean absolute error. The five
frames cross the four-frame attention chunk boundary.

Run the portable layer gate and independent comparison with an environment that
contains `numpy` and `gguf`:

```sh
nix develop -c cargo run --release -p plowrt --features gguf \
  --example asr_nemotron_conformer -- \
  plans/asr-eval-assets/nemotron-3.5/model.gguf 0 5 /tmp/nemotron-block.f32
python scripts/asr/nemotron/block_reference.py \
  plans/asr-eval-assets/nemotron-3.5/model.gguf 0 5 /tmp/nemotron-block.f32
```

The same plans bind all 24 layers. The scalar full encoder completed real audio
with output shape `[201,1024]` and finite values. Its 79.10-second encoder time is
a correctness oracle measurement, not an optimized CPU result. Metal, CUDA/HSA
and optimized CPU executors must consume these plans; accelerator policy remains
outside the Nemotron family module.

The Apple adapter now consumes that generic encoder plan without importing GGUF
or Nemotron types. It copies all 24 layers once, shares pipelines and scratch
buffers, retains intermediate tensors on the GPU, and submits the complete encoder
as one command buffer. The chunk-limited attention kernel fuses relative scores,
softmax and value reduction in one threadgroup using the model's bounded 60-frame
window; it allocates no `[heads,T,T]` score tensor. On the same 15.99-second clip,
its `[201,1024]` result matched the full scalar encoder within `2.00e-6` max and
`1.95e-8` mean absolute error. Three warm M4 Pro executions measured 117.80–117.92
ms GPU time, down 19.2% from the materialized-attention 146.02 ms median. This
excludes the frontend, subsampling, pre-encoder projection and RNNT head, so it is
an encoder component result rather than end-to-end latency.

```sh
nix develop -c cargo run --release -p plowrt --features metal,gguf \
  --example asr_nemotron_encoder_metal -- \
  plans/asr-eval-assets/nemotron-3.5/model.gguf AUDIO.wav 5
```

On M4 Pro, five retained-model repetitions of the 15.99-second reference clip were
identical to the official offline transcript. Warm Metal calls took 0.2554–0.2561
seconds (about 62.5× realtime); warm CPU calls took 0.5816–0.5874 seconds (about
27.4× realtime). Startup/model load and first-call warmup are excluded from those
ranges. These are repeated calls on one clip, not corpus throughput or a SOTA claim.
Evidence: `plans/nemotron-plow-adapter-2026-09-11/`.

```sh
nix develop -c sh -c 'plans/asr-eval-assets/venv/bin/python scripts/asr/nemotron/reference.py \
  --runtime plans/asr-eval-assets/nemotron-3.5/nemo-speech/bin/nemo-speech \
  --model plans/asr-eval-assets/nemotron-3.5/model.gguf \
  --audio plans/asr-eval-assets/dev-clean-100/652-129742-0016.wav \
  --output plans/nemotron-reference-new \
  --language en-US --right-context 3 --stream'
```

Omit `--stream` for offline transcription. Output directories must be new to
preserve existing evidence. Inherited `NEMO_SPEECH_*` overrides are removed.
The command explicitly selects Metal, language, lookahead and disables dynamic
batching. Lookahead 3 corresponds to a 320 ms acoustic chunk; the file CLI feeds
160 ms input pieces into the cached recognizer without realtime pacing. Process
time includes model load and warmup. It is not steady-state inference time,
TTFT, or a paced serving-latency measurement.

## Verified on M4 Pro, 2026-09-11

Offline and streaming execution succeeded on the 15.99-second LibriSpeech
`652-129742-0016` clip. Each produced 0 errors / 26 reference words with the
repository's `scripts/asr/score.py` normalization; punctuation differs. A verbose
streaming repeat produced the same text and logs identify the Apple M4 Pro Metal
backend and RNNT model. This single clip does not establish corpus WER.

Evidence: `plans/nemotron-asr-2026-09-11/{offline,stream,stream-verbose}/`,
`quality.json`, pinned Hugging Face/release metadata and checkpoint configs.

JSON word timestamps come from the reference runtime, not a forced aligner.
They are not validated here: an offline end timestamp extends past the recording
and some streaming word intervals overlap. Confidence values are uncalibrated.
Native Plow integration must separately validate frame timing and end-of-stream
flush behavior before advertising timestamp or alignment capability.

## Shared-model throughput smoke test

The official runtime's `bench asr` supports concurrent requests against one
resident model. On this M4 Pro, an initial 16-clip dev-clean subset (96.08 seconds),
two repetitions per level, gave:

| Concurrent requests | Offline audio seconds / wall second | Streaming audio seconds / wall second |
|---:|---:|---:|
| 1 | 56.75 | 6.52 |
| 2 | 57.66 | 10.31 |
| 4 | 60.44 | 14.30 |
| 8 | 62.97 | 17.36 |

Each mode completed 128 transcriptions with zero raw transcript differences
against its own first result for each input. This measures repeat/concurrency
consistency, not corpus WER. Peak process RSS across each complete sweep was
1.85 GB offline and 1.10 GB streaming; this is not per-session GPU memory.

Explicit settings: Metal, `en-US`, lookahead 3, maximum batch 8, queue depth 32,
5 ms queue wait, 20 ms ingress cohort wait, 16 state slots, no offline bucketing.
Batching remained enabled at concurrency 1 because the entire sweep shares one
recognizer. WAV loading, engine warmup and one input warmup were excluded from
the reported wall intervals; worker launch, scheduling and recognition were
included. The engine warmed batch shape 8; additional shapes can compile during
measurement. Levels ran once in ascending order. No confidence interval,
maximum supported session count, paced-stream latency, or cloud/M5 performance
claim follows from this short test.

Reproduce with the pinned runtime, replacing `INPUT_DIRECTORY` with WAV files:

```sh
nix develop -c plans/asr-eval-assets/nemotron-3.5/nemo-speech/bin/nemo-speech \
  bench asr INPUT_DIRECTORY \
  --model plans/asr-eval-assets/nemotron-3.5/model.gguf \
  --mode stream --concurrency 1,2,4,8 --repetitions 2 --warmup 1 \
  --device metal --language en-US --json --verbose \
  --asr.streaming.rnnt_right_context 3 \
  --asr.batching.max_batch_size 8 --asr.batching.max_queue_depth 32 \
  --asr.batching.max_queue_delay_us 5000 \
  --asr.batching.ingress_cohort_delay_us 20000 \
  --asr.batching.state_arena_slots 16 --asr.batching.offline_bucket_ms 0
```

Use `--mode offline` for the other sweep. Clear inherited `NEMO_SPEECH_*`
overrides. Evidence and the exact input manifest are archived under
`plans/nemotron-throughput-2026-09-11/`; its runner records both commands and
uses `/usr/bin/time -l` for process memory measurements.
