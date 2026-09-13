# ASR on Apple Silicon

The initial ASR backend runs [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
with a Rust audio frontend, Metal audio encoder/projector, and the compiled Metal
Qwen3 decoder. It requires macOS with Apple Silicon and the `metal` Cargo feature. Metal shaders
are compiled on the destination GPU; pipeline SIMD, thread and memory limits are
checked. Unknown GPUs or unmatched compiled executor counts use instruction-ordered
dispatch to avoid assuming persistent-grid residency. Actual cross-device validation
remains pending; M4 Pro tuning does not establish M3/M5 performance.

Execution currently uses the CPU frontend and GPU stages directly, without plow's
heterogeneous scheduler or ANE. Encoder linears and convolutions use validated tiled SIMD-group
kernels. Encoder attention distributes independent keys and output channels across
a SIMD group while preserving serial reduction order; scalar attention remains a
fallback when the optional pipeline is unavailable. Normalization remains baseline.
Use release Rust builds for performance measurements. No forced aligner is loaded.

## Compile and transcribe

Download the complete checkpoint into `models/Qwen3-ASR-1.7B`. The reference revision
is `7278e1e70fe206f11671096ffdd38061171dd6e5`. Keep all processor/tokenizer JSON files,
`vocab.json`, `merges.txt`, the safetensors index and both weight shards.

```sh
nix develop -c cargo run -p plowc -- \
  --hf-dir models/Qwen3-ASR-1.7B --gpu m4pro --max-ctx 2048 \
  --out plow-out/qwen3-asr-1.7b

nix develop -c cargo run -p plowrt --features metal,dist -- asr \
  --packet plow-out/qwen3-asr-1.7b/model.pkt \
  --tokenizer models/Qwen3-ASR-1.7B --audio speech.wav
```

Input must be 16 kHz mono or stereo WAV, 0.5–30 seconds. Integer PCM (8/16/24/32 bit)
and 32-bit float WAV are accepted. Stereo is averaged to mono. Output is JSON with
`text` and, when detected, `language`.

`--language en` or `--language English` forces a supported language. Omission enables
language detection. `--prompt` supplies context, limited to 256 tokens. Decoding is
greedy with a 1024-token output limit; the compiled context must fit prompt plus output.
Timestamps, translation, resampling, compressed media, batching and other model families
are not implemented.

## Weight resources inside an asset bundle

The native loader reads every `*.safetensors` shard in the supplied checkpoint
directory. Quantized tensors can therefore be packaged alongside the original
shards and processor/tokenizer files. Give the additional shards distinct filenames
and check that tensor names do not overlap. MXFP4 tensors use the `mxfp4/` prefix;
the original embedding lookup and audio encoder weights are still required.

With these resources under `<assets>/checkpoint`, pass that directory as
`--checkpoint`; no `PLOW_MXFP4_DIR` override is needed. Clear weight-directory
overrides when validating the package, since an explicit twin overrides matching
checkpoint tensors. Compiler-created checkpoint symlinks must be materialized for
an independently movable bundle. Local hard links avoid duplicating immutable
weight data on the same filesystem; copy the files when exporting elsewhere.

The compiler can combine an MXFP4 decoder body with an FP8 tied head by passing
`--mxfp4 --mx4-head 0 --fp8-head`. Generate the FP8 resource with
`quantize_fp8 ... thinker.model. --head-only`; the resulting shard contains only
`fp8/thinker.model.embed_tokens.weight` and its per-row scale. Package that shard,
the MXFP4 shard and the original checkpoint together. This precision split is
backend-neutral packet policy; each device backend must implement the referenced
MXFP4 and FP8 opcodes.

An M4 Pro smoke test with colocated original/MXFP4 shards, no resource symlinks,
and no weight-directory overrides matched the reference transcript and language
from a different working directory. This validates resource loading, not generic
Metal serving: the current ASR command takes `--packet` and `--tokenizer` (`--blob` and `--checkpoint` remain aliases) and
constructs the dedicated Qwen audio encoder.

## Local serving

```sh
nix develop -c cargo run -p plowrt --features metal,dist -- asr \
  --packet plow-out/qwen3-asr-1.7b/model.pkt \
  --tokenizer models/Qwen3-ASR-1.7B --port 8080 --websocket

curl --fail http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=decode -F file=@speech.wav -F language=en
```

The server binds loopback and admits a bounded queue of HTTP requests or WebSocket
sessions. Its model-owning worker forms cohorts up to the packet's batch capacity;
a full queue returns 429. Omit `--websocket` to disable the streaming route. The
request model is the packet pipeline name (`decode` for this Qwen asset and
`transcribe` for Nemotron). This uses the `asr` command; the generic text `serve`
registry is unchanged.

Multipart fields: required `file` and `model`; optional `language`, `prompt`,
`response_format=json|text`, `temperature=0`. Unknown/duplicate fields and unsupported
generation options fail. The body limit is 4 MiB. Malformed audio returns 400,
unsupported sample rate/format 415, excessive duration 413, unknown model 404, and
engine failures 500. `GET /health` reports readiness after model loading.
Uploads have a 30-second deadline. Successful inference logs frontend, encoder,
prefill, decode and total time, token count, audio duration and real-time factor.
CLI transcript JSON is written to stdout; diagnostics go to stderr.

## WebSocket protocol v1

Connect to `ws://127.0.0.1:8080/v1/audio/transcriptions/stream` and send:

```json
{"type":"start","version":1,"model":"decode","sample_rate":16000,"format":"pcm_s16le"}
```

Optional start fields are `language` and `prompt`. The server's `ready` event contains
a session ID, audio limits, `credit_samples`, and `partial_mode=final_only`.

- Each binary message: little-endian `u64` sequence number, starting at zero, then
  mono little-endian signed 16-bit PCM. Maximum PCM payload: 32,000 bytes.
- Credit counts PCM samples, excluding the eight-byte sequence header. Send no more
  than the outstanding grant. `credit` events add further credit after processing.
- Send `{"type":"finish"}` once after the last samples.
  One `final` event contains the complete transcript. Its stable prefix covers all
  UTF-8 bytes of the final text.
- `{"type":"cancel"}` or disconnection cancels the session. Terminal `error` events
  end unsuccessful sessions. One utterance per connection, maximum 30 seconds.

The server accumulates streamed PCM and decodes once at finalization. Partial
transcripts require reusable encoder and decoder state; replaying the full model on
each chunk is intentionally excluded. Credit keeps memory bounded. Idle input and
blocked output have 30-second deadlines.

An adapter may reserve part of the advertised audio limit for finalization input.
Qwen currently appends one second of deterministic low-level audio to close clipped
endpoints; other adapters default to no final padding.

Run the example client against the local server:

```sh
nix develop -c cargo run -p plowrt --example asr_stream -- \
  ws://127.0.0.1:8080/v1/audio/transcriptions/stream \
  decode speech.wav
```

## Reference checks

`scripts/asr/reference.py` exports fixtures using qwen-asr 0.0.6,
Transformers 4.57.6 and Torch 2.14.0. Python dependencies are only for validation.
Use a separate Python environment under `nix develop`; record its executable as
`REFERENCE_PYTHON` in the commands below.

```sh
nix develop -c "$REFERENCE_PYTHON" scripts/asr/reference.py \
  --checkpoint models/Qwen3-ASR-1.7B --audio speech.wav \
  --out /tmp/asr-reference --stage model --packed-windows

nix develop -c cargo run -p plowrt --features metal --example asr_check -- \
  models/Qwen3-ASR-1.7B speech.wav /tmp/asr-reference encoder-reference-input

nix develop -c cargo run -p plowrt --features metal --example asr_check -- \
  models/Qwen3-ASR-1.7B speech.wav /tmp/asr-reference decoder \
  plow-out/qwen3-asr-1.7b/model.pkt

nix develop -c cargo run -p plowrt --features metal --example asr_check -- \
  models/Qwen3-ASR-1.7B speech.wav /tmp/asr-reference model \
  plow-out/qwen3-asr-1.7b/model.pkt
```

The pinned upstream eager encoder omits its packed-window attention mask. For clips
over eight seconds, `--packed-windows` explicitly installs the upstream mask to match
the native encoder's window semantics. Fixtures record this choice. Short clips also
work with unmodified eager attention.

The frontend gate is relative L2 ≤1e-5. The encoder gate is ≤0.02 with identical mel
inputs, isolating frontend BF16 rounding effects. `encoder` instead measures the combined
native frontend and encoder. `decoder` checks first logits (relative L2 ≤0.02) and every greedy token using
reference audio embeddings; `model` checks the complete native transcript and language.

The decoder numerical gate currently **fails**: first-logit relative L2 is 0.02706
for short English, 0.03849 for full English and 0.04954 for Chinese. All reference
greedy token IDs match on these three clips. This is an experimental backend with
an unresolved numerical parity issue, not a claim of reference-equivalent math.
An intermediate-rounding experiment did not consistently improve the error and
was removed; the adapter uses the existing Metal decoder kernels. The diagnostic
threshold remains unchanged. A layer trace exactly attributes the first RMSNorm
difference to native FP32 intermediate fusion vs reference BF16 intermediate rounding.
It does not account for the complete final-logit error. Do not treat matching smoke transcripts as a corpus
accuracy result.

Initial scalar component checks on M4 Pro / 24 GiB unified memory (historical
timings; see tiled measurements below):

| input | frontend relative L2 | encoder relative L2, shared mel input | encoder time |
|---|---:|---:|---:|
| English, 5 seconds | 1.17e-6 | 0.01199 | 0.35 s |
| English, 15.05 seconds | 1.08e-6 | 0.01370 | 0.90 s |
| Chinese, 4.20 seconds | 1.06e-6 | 0.01107 | 0.36 s |
| Silence, 0.5 seconds | 0 | 0.01263 | 0.19 s |
| 440 Hz sine, 30 seconds | 6.3e-7 | 0.01023 | 1.70 s |

These timings exclude model load and text decoding. Small frontend errors cross BF16
rounding boundaries: combined frontend/encoder relative L2 on the 15-second English
clip was 0.02219. Transcript checks are a separate end-to-end gate.

End-to-end validation on the same machine used debug Rust binaries and compiled
Metal shaders. The CLI and HTTP JSON/text responses matched official English and
Chinese transcripts, including forced Chinese. Warm HTTP took 2.94 s for 15.05 s
of English and 0.80 s for 4.20 s of Chinese. The English stages were frontend
0.832 s, encoder 0.890 s, prefill 0.251 s and decode 0.948 s. These are individual
observations, not median/p95 benchmarks; use optimized host builds before tuning.

The real WebSocket client sent a five-second English clip, received four partial
revisions and one final matching the CLI transcript. Early partial text and language
can both change. Serving checks also rejected unsupported sample rate, unknown
model/language and nonzero temperature with their documented statuses.

Verification includes nine ASR unit/Metal/protocol tests, six tokenizer/template
tests, two nested-config tests and 17 existing compiler golden tests. The runtime
library suite passed 327 tests (one ignored). A real Llama checkpoint regression
confirmed embedding injection preserves ordinary prefill/decode behavior and resets
correctly between requests. Decoder packet ordering and LDS checks passed; these
certificates do not cover the separate audio encoder or floating-point arithmetic.

## Extension boundary

The latest controlled comparison on the 100-clip dev-clean subset used alternating
ten-clip batches, the same local BF16 checkpoint, greedy automatic-language decoding,
and one excluded warmup per process. Native defaults before SIMD encoder attention
took **76.90 s** versus
**59.45 s** for MLX 0.32.2 / mlx-qwen3-asr 0.4.0: native took 29.4% longer
(paired batch-bootstrap time-ratio 95% interval [1.2807, 1.3069]). Native produced
47 word errors versus MLX's 48 of 2,062 words, with one differing raw transcript and
zero failures. Loading and WAV decoding are excluded. This is a development subset,
not a held-out or streaming benchmark. The optimization comparisons below are
historical paired experiments, not timings comparable across runs.

The tiled encoder linear kernel can be checked with `asr_check` modes
`encoder-tiled`, `encoder-compare` (alternating scalar/tiled with exact output comparison)
and `model-tiled`. `encoder-profile` isolates each dispatch in its own command buffer
and prints GPU timestamps; this changes scheduling and is for bottleneck diagnosis,
not end-to-end latency measurement. These modes use reference mel except `model-tiled`,
which runs the complete native frontend. The regular adapter now uses tiled linears.
The scalar implementation remains available for paired validation in `asr_dataset`.
Rust examples are grouped in `crates/plowrt/examples/asr/`. Layer trace tools and
convolution comparisons are documented in [the tooling README](../../scripts/asr/README.md).

On a deterministic, speaker-balanced 100-utterance LibriSpeech dev-clean subset
(760.58 seconds, 2,062 reference words), scalar and tiled produced identical transcripts:
47 word errors, normalized WER 2.279%, and no failures. Tiled execution took 87.84 s
versus scalar's 114.32 s, a 23.2% time reduction. Local BF16 MLX took 70.03 s with
48 word errors (2.328% WER). This small development subset does not establish better
accuracy or state-of-the-art speed. Normalization and reproducible commands are in
[the tooling README](../../scripts/asr/README.md); file/WAV loading is excluded from
these engine timings. The existing decoder numerical diagnostic remains open.
The pinned PyTorch BF16 reference with explicit packed encoder windows matched
all 100 native normalized transcripts and the same WER; one raw transcript differed
in punctuation/case. This does not establish logit equivalence.

The subsequent paired convolution comparison used tiled linears in both variants:
87.87 s with scalar convolution vs 86.63 s with tiled convolution (1.41% reduction,
RTF 0.11390). All 100 transcripts still match, with 2.279% WER and no failures.
The next paired run compared direct tiled convolution with bounded packed panels:
88.97 s vs 86.00 s (3.34% reduction, RTF 0.11308). All 100 raw transcripts match,
with the same 47 word errors and no failures. The paired speaker-bootstrap 95%
time-ratio interval is [0.96306, 0.97028]. Packed panels are now the adapter default
for convolutions two and three; convolution one remains direct tiled. Reusing panel
buffers bounds extra declared scratch at 19.4 MB across both stages, independent
of clip length within the supported range. Native remains slower than the earlier
70.03 s local MLX BF16 run; timings from separate runs are not a paired comparison.
The numerical diagnostic and broader quality and performance evaluation remain open.

The adapter selects a 32×64 encoder matrix tile when its Metal pipeline supports
256-thread groups and the required 20 KiB threadgroup storage. It falls back to
16×32 (128 threads, 8 KiB) or 8×8 when the larger pipelines are unavailable.
An optional direct epilogue reduces the 32×64 tile's scratch to 12 KiB. Before
enabling it, the adapter runs exact row/column mapping and guard probes on the
destination GPU/compiler. Probe failure retains the shared-result path. See the
[tooling results](../../scripts/asr/README.md) for the paired corpus comparison.
`encoder-wide-compare` checks exact projected-output parity for 8×8 vs 16×32, and
`asr_dataset --compare-wide` alternates the two tiles with packed convolution fixed.
On the same 100-clip M4 Pro development subset, the paired run took 86.05 s with
the narrow tile vs 84.74 s with the wider tile (1.52% less time). All transcripts
matched; normalized WER stayed 2.279%, with zero failures. This does not establish
state-of-the-art performance or performance on another Apple GPU. The half-second
silence encoder fixture was slightly slower (about 38 vs 40 ms).

The subsequent 16×32 vs 32×64 comparison, with QKV dot4 fixed in both paths,
took 80.06 vs 78.51 s (1.94% less time). All transcripts matched, with 2.279% WER
and no failures. The paired speaker-bootstrap 95% time-ratio interval was
[0.97763, 0.98357]. Half-second silence remained a tradeoff (about 39 vs 42 ms
for the encoder). `encoder-large-compare` and `asr_dataset --compare-large`
reproduce these tile comparisons. `encoder-profile-wide` isolates dispatch timings
for the 16×32/panel baseline; it is not an end-to-end timing mode.

Qwen ASR defaults to a decoder QKV kernel that shares each activation load across
four output rows and retains the existing normalization path. Set
`PLOW_METAL_QKV_DOT4=0` to use the prior kernel; `1` enables it for other Metal
adapters, whose defaults are unchanged. It passed 100 guarded kernel comparisons,
three reference transcript fixtures and exact decoder-logit comparison. With the
encoder configuration fixed, a 100-clip comparison in alternating ten-clip batches
took 82.93 vs 80.09 s (3.42% less time), with identical transcripts, 2.279% WER
and no failures. Loading and first-clip warmup are excluded. The paired batch
95% time-ratio interval is [0.95758, 0.97447]. This remains a development-subset
result on M4 Pro. `asr_decoder_trace --profile-decode` saves `decode-logits.bf16`
for comparison between shader variants; use the environment switch explicitly
because this diagnostic loads the generic Metal engine.

ASR also uses one query head per decode work item. For Qwen's 16 query heads,
this exposes 16 work items instead of eight serial two-head groups, with each
head's arithmetic unchanged. Set `PLOW_METAL_DECODE_HEADS=0` for the grouped
reference path; `1` enables single-head work for a generic Metal engine.
On the same development corpus, an alternating ten-clip-batch comparison took
84.20 vs 80.90 s, with identical transcripts and 2.279% WER. The paired batch
95% time-ratio interval was [0.94229, 0.97667]. Absolute timings drifted upward
relative to the previous run; use this paired comparison for the promotion decision.
GPU tests cover exact partial outputs, GQA, split/batch and ring indexing. The
inspected compiler dependency maps use per-head ownership; arbitrary custom packet
dependency maps require separate validation.

`asr::Transcriber` accepts waveform samples and returns `Transcript`. Shared HTTP and
WebSocket code does not require an autoregressive decoder, audio placeholders, or mel
features. A future Whisper adapter can retain encoder output and cross-attention state;
CTC can collapse frame logits; a transducer can advance its predictor and audio states.
Each needs its own frontend, weights, executor and quality checks.

The initial Qwen encoder uses dedicated Metal dispatches with BF16-rounded values in
f32 buffers. It completes before shared-memory embedding splice and decoder prefill.
Packet-integrated audio ops, ASR model-registry integration, memory reuse and kernel
tuning remain follow-ups. There is no CPU fallback for model execution.

An opt-in GPU handoff now retains the completed encoder output and splices BF16
audio/text embeddings directly into decoder input. GPU-only execution now encodes
splice into decoder prefill's command buffer; encoder completion still waits.
This does not yet implement asynchronous pipeline scheduling. The latest paired
100-clip comparison produced identical transcripts, but no demonstrated speedup:
85.72 s host vs 85.84 s device. The adapter therefore retains the CPU handoff by
default. Reproduction modes are in the tooling README.

Qwen orchestration is now backend-neutral. Frontend and prompt preparation, language
handling, cancellation, cohort state, occupied-row decode selection, termination and
telemetry call a small execution contract. The Metal adapter owns Metal buffers,
encoder dispatch, embedding splice, packet rung selection and tuning controls. The
shared Qwen module compiles with the CPU, CUDA, HSA and Metal feature sets and contains
no Apple API types. CPU/CUDA/HSA Qwen execution still requires encoder, splice and
packet adapters plus hardware validation; feature compilation alone is not support.

Packet RNNT execution also exposes opt-in phase profiling through `PacketRnnt`.
The counters report encoder runtime, predictor-plus-joint runtime, joint-only runtime,
host/device transfer time, and submission counts using the active packet backend.
Profiling is disabled by default. On Metal, packet programs made entirely from the
generic RNNT primitive set can use compact 256-thread GEMV and pointwise dispatches;
unsupported shapes and devices continue through the interpreter. This specialization
does not alter the packet, compiler, or RNNT controller contract used by CPU and future
CUDA/HSA implementations.

Packet pipelines may set the backend-neutral `ordered_dispatch` parameter when their
dependency graph must make progress without concurrent workgroups. Metal honors the hint
with compiler-order instruction dispatch and automatically selects dedicated MXFP4 kernels
where their packet geometry is supported. CPU execution is already ordered; CUDA and HSA
can choose their own implementation of the same contract. This avoids relying on a model
flag or an operator environment variable.

The final Qwen packet reliability run completed all 100 `dev-clean` recordings without a
device fault after the default cooperative path had stalled at recording 52 and reproduced
a residual wait timeout in isolation. The packet-selected ordered path took 52.590 seconds
for 760.58 seconds of audio (14.46× realtime), with 2.5218% normalized WER and 0.9082% CER.
Its transcripts exactly matched the earlier explicitly ordered control. A B4 1/2/4 decode
ladder completed the same corpus in 50.964 seconds (14.92× realtime); three transcripts
differed from B1 and its WER was 2.4733%, so batching retains a separate numerical result.
These figures are M4 Pro development-subset measurements, not a state-of-the-art claim.

The current packet checkpoint uses the MXFP4 B4 1/2/4 ladder for Qwen and the packetized
Conformer/RNNT pipeline for Nemotron. Qwen took 50.964 s for 760.58 s of audio (14.92×
realtime) with 51/2,062 word errors (2.473% WER). Nemotron took 12.747 s (59.67× realtime)
with 66/2,062 errors (3.201% WER). Qwen cohorts contain up to four recordings; Nemotron was
sequential. The earlier 40.156 s Qwen legacy-executor result and 12.686 s NeMo-Speech
result remain local controls. These development-subset results share the same manifest and
scorer but are not a published-benchmark or state-of-the-art claim.
