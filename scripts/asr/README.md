# ASR validation tools

Keep ASR reference, dataset preparation, scoring and benchmarking tools here.

The CPU frontend skips zero-only edges of each mel filter while preserving
nonzero accumulation order. Non-finite FFT power retains the dense calculation.
Full mel outputs match the dense implementation in boundary/amplitude tests.
On M4 Pro with the FP8 split packet, dev-clean-100 took 56.520→54.926 s
(2.82% less time), with all ten alternating process batches improving, identical
transcripts, 2.328% normalized WER and zero failures. Logged frontend time fell
1.279→0.191 s; other stages also varied, so not all total savings are attributable
to this change. The paired batch dense/sparse time-ratio interval was
[1.01953, 1.04338]. This portable Rust optimization is enabled by default.

The local experimental FP8 asset is `plow-out/qwen3-asr-1.7b-fp8/model.pkt`, with
weight twins in `plans/asr-eval-assets/qwen3-fp8`. It uses per-output-channel E4M3
decoder weights and decode head, BF16 activations/KV and the existing audio encoder.
The embedding lookup retains the original BF16 table. These generated files are
gitignored; retain the original checkpoint alongside them.

```sh
nix develop -c env PLOW_FP8_DIR=plans/asr-eval-assets/qwen3-fp8 \
  target/release/plowrt asr --packet plow-out/qwen3-asr-1.7b-fp8/model.pkt \
  --tokenizer models/Qwen3-ASR-1.7B --audio /path/to/audio.wav
```

To recreate it, build `plowc` and the `quantize_fp8` example, then run:

```sh
nix develop -c target/release/examples/quantize_fp8 \
  models/Qwen3-ASR-1.7B plans/asr-eval-assets/qwen3-fp8 thinker.model.
nix develop -c target/debug/plowc --hf-dir models/Qwen3-ASR-1.7B \
  --gpu m4pro --max-ctx 2048 --fp8 --w8a16 --fp8-head \
  --out plow-out/qwen3-asr-1.7b-fp8
```

The Rust quantizer already includes the tied head; no extra head shard is needed.
Pass `--head-only` after the optional prefix when an FP8 head is paired with a
different decoder-body encoding. This writes only the tied weight and scale instead
of every decoder projection. For Qwen3-ASR-1.7B the output is 298 MiB versus 1.7 GiB;
both head tensors match the full FP8 twin byte-for-byte.

An experimental mixed asset uses an MXFP4 decoder body and FP8 tied head:

```sh
nix develop -c target/release/examples/quantize_fp8 \
  models/Qwen3-ASR-1.7B plans/asr-eval-assets/qwen3-fp8-head \
  thinker.model. --head-only
nix develop -c target/debug/plowc --hf-dir models/Qwen3-ASR-1.7B \
  --gpu m4pro --max-ctx 2048 --mxfp4 --mx4-prefill 1 \
  --mx4-head 0 --fp8-head --emit-decode-batch 4 \
  --emit-decode-batch-ladder 1,2,4 \
  --out plow-out/qwen3-asr-1.7b-mx4-fp8-head
```

On the held-out 100-clip test-clean sample, matched B1/B2/B4 assets and alternating
ten-clip processes gave 43/2,151 errors (1.999% WER) for the FP8 head versus
44/2,151 (2.046%) for the MXFP4 head. The FP8 head took 7.68% more inference time,
with a paired batch-bootstrap ratio interval of [1.0721, 1.0816], and lost every
batch. The one-word quality difference is not statistically established: its paired
speaker-bootstrap interval crosses zero. Keep the MXFP4 head for throughput and use
the FP8 head as a measured quality/latency tradeoff. Isolated four-step replay matched
normal execution exactly with either the full or head-only FP8 shard.

Aligned W8A16 prefill weight loads decode E4M3 directly to BF16 bits. The
`runtime/apple/probe/fp8castcheck.m` probe checks all 256 codes in each vector
position, including signed zero and NaNs, plus 60 exact matrix/GLU output and
guard cases. Build with a macOS 15+ SDK, ARC, Foundation/Metal and
`-I runtime/common`; pass `runtime/apple/interp.metal` to the executable.
Two dev-clean-100 runs with reversed clip/backend order reduced prefill time
by 2.29% and 1.74%, improving all ten batches in each run. All transcripts
matched, with 2.328% normalized WER and zero failures. Total time fell
55.567→55.294 s and 54.934→54.868 s; the repeat's total-time confidence interval
includes no gain. This supports a prefill improvement, not a reliable overall
ASR speedup. Four decode-step logits also matched the previous implementation
byte-for-byte. Other Apple GPUs remain unmeasured.

On M4 Pro, alternating ten-clip process batches over dev-clean-100 took 75.327 s
BF16 vs 56.834 s FP8 (24.55% less time). All ten batches improved; paired batch
bootstrap FP8/BF16 ratio 95% interval [0.7444, 0.7672]. Normalized WER changed
from 2.279% to 2.328% (47 vs 48 errors over 2,062 words); seven raw transcripts
changed and neither backend failed. Decode improved 52.536→33.243 s while
prefill regressed 12.382→13.446 s. This is a lossy precision tradeoff on a
development subset, not a same-precision comparison with MLX or a held-out
quality result. Other Apple GPUs remain unmeasured; BF16 stays the default asset.

A subsequent held-out test-clean sample (100 clips, 40 speakers, 802.985 seconds,
2,151 reference words) used the unchanged FP8 recipe and rotating three-backend
ten-clip batches, with one excluded warmup per process:

| Backend | Total inference | Normalized WER | Failures |
|---|---:|---:|---:|
| Native BF16 | 77.653 s | 1.767% (38 errors) | 0 |
| Native FP8 | 59.046 s | 1.674% (36 errors) | 0 |
| MLX BF16 | 62.199 s | 1.674% (36 errors) | 0 |

Native FP8 took 5.07% less time than MLX BF16 (9/10 batches; paired batch ratio
95% interval [0.9371, 0.9638]). This is a cross-precision result; native BF16
remained 24.85% slower than MLX BF16. The comparator was MLX 0.32.2 with
[mlx-qwen3-asr 0.4.0](https://pypi.org/project/mlx-qwen3-asr/0.4.0/), using the same
checkpoint, automatic language and 1,024-token limit. These English subset
results do not establish broad quality equivalence or superiority over quantized
MLX. Multilingual and streaming benchmarks remain open.

A subsequent comparison against MLX Q8 group-size 64 on the same held-out
100 clips took 58.021 s native FP8 vs 41.098 s MLX Q8. Both produced 36 word
errors (1.674% normalized WER), with zero failures and six raw transcript
differences. MLX won all ten alternating process batches; the native/MLX time
ratio was 1.4118, with paired batch-bootstrap 95% interval [1.4047, 1.4206].
Native FP8 therefore remains 41.18% slower than this quantized comparator.
MLX quantized 345 linear/embedding modules, including audio encoder linears;
native FP8 retains the existing encoder and BF16 embedding lookup. These are
different quantization recipes, not equivalent formats. `mlx_dataset.py` records
the checkpoint's quantization settings and labels this backend `mlx-q8-g64`.
This subset establishes neither broad quality equivalence nor state-of-the-art
native performance.

`plowc --fuse-qkv-fp8` opts Apple packets into one FP8 Q/K/V projection
instruction per decoder layer. It uses the same weight twins and dot-product
math as the split packet; the default stays split. Compile to a separate output
directory and use a runtime built with the Metal and CPU implementations of
`GemvQkvFp8`:

```sh
nix develop -c target/debug/plowc --hf-dir models/Qwen3-ASR-1.7B \
  --gpu m4pro --max-ctx 2048 --fp8 --w8a16 --fp8-head --fuse-qkv-fp8 \
  --out plow-out/qwen3-asr-1.7b-fp8-qkv
```

`runtime/apple/probe/qkvfp8check.m` compares split/fused projections on Metal
and in the CPU golden implementation. Its 30 cases include uneven partitions,
K tails, multiple rows, empty V and output guards. Build with a macOS 15+ SDK,
ARC, Foundation/Metal, `-I runtime/common -I runtime/cpu/dev`, and link
`runtime/cpu/dev/golden/fp8.c` compiled as C. Four full-model decode steps also
matched byte-for-byte between split and fused packets and passed replay checks.
On dev-clean-100, however, fused took 56.317 s vs 55.961 s split (0.636% slower).
All 100 transcripts matched, with 2.328% normalized WER and zero failures.
Split won all ten alternating process batches; the paired batch-bootstrap
split/fused ratio interval was [0.99259, 0.99488]. Decode regressed
32.714→33.033 s. Isolated dispatch improvements did not survive normal execution;
fusion remains an experiment, not the recommended ASR packet.

`PLOW_METAL_GLU_PAIR=1` enables an experimental decoder GLU kernel that reuses
input loads across gate/up dot products. The default remains the original kernel.
On M4 Pro, a 100-clip comparison in alternating ten-clip process batches took
76.744 s baseline vs 75.829 s candidate (1.19% less time), with all transcripts
unchanged, 2.279% WER and no failures. The paired batch-bootstrap candidate/baseline
95% interval [0.9740, 1.0003] includes no gain; this does not justify promotion.
The reversed batch/variant-order repeat took 74.550 s baseline vs 74.956 s
candidate (0.55% slower), again with identical transcripts. The gain did not
reproduce; keep the default kernel.
The standalone `runtime/apple/probe/glucheck.m` checks 400 cases including K tails,
partitions, biases, activations and output guards. Build it with a macOS 15+ SDK,
ARC, Foundation/Metal frameworks and `-I runtime/common`; pass
`runtime/apple/interp.metal` to the executable. Real-model decode replay and
candidate/baseline logits also matched exactly on the saved reference fixture.

`asr_dataset CHECKPOINT BLOB MANIFEST --single` runs each included clip once with
the adapter's current defaults, after one excluded first-clip warmup. It reports
backend `native`; record environment switches alongside results when comparing
process-level variants such as `PLOW_METAL_QKV_DOT4=0` vs `1`. Model loading is
outside the reported transcription time. Existing comparison flags retain their
explicit encoder configurations.

`asr_check ... encoder-large-compare 4` compares 16×32 and 32×64 encoder tiles
with exact projected-output checks. `asr_dataset ... --compare-large` alternates
these tiles per clip with packed convolution and the default QKV decoder fixed.
`asr_check ... encoder-profile-wide 3` profiles the 16×32/panel baseline using
isolated dispatches; its wall time includes profiling waits.
`encoder-profile-current` instead profiles the promoted encoder configuration,
including supported SIMD attention and validated direct epilogues. Startup probe
dispatches precede the first encoder iteration; exclude them and cold timings from
steady-state attribution. Isolated dispatch timing changes scheduling and cache
behavior, so use corpus comparisons for end-to-end claims.

`asr_check ... encoder-attention-compare 4` checks exact encoder output parity
between scalar and SIMD attention. `asr_dataset ... --compare-attention` alternates
these paths per clip with 32×64 FP32 linears, packed convolution and current
decoder defaults fixed. Each path warms up before timing. SIMD attention retains
serial dot-product and weighted-sum order; it distributes independent keys and
output dimensions across lanes.
The M4 Pro dev-clean-100 paired comparison took 76.15 s with scalar attention
versus 74.49 s with SIMD attention (2.18% less time). All 100 raw transcripts
matched, with 2.279% WER and zero failures. The paired speaker-bootstrap
scalar/SIMD time-ratio 95% interval was [1.0180, 1.0261]. The ASR adapter now
selects SIMD attention when supported, with scalar fallback. Other Apple GPUs
remain unmeasured; this result does not establish state-of-the-art performance.

`asr_check ... encoder-direct-compare 4` checks the direct matrix epilogue against
the shared-result epilogue. `asr_dataset ... --compare-direct` alternates them with
32×64 FP32 matrices, packed convolution and SIMD attention fixed. The M4 Pro
dev-clean-100 run took 74.73 versus 73.84 s (1.19% reduction; paired speaker
bootstrap candidate/baseline ratio 95% interval [0.9848, 0.9915]). All transcripts
matched, with 2.279% WER and zero failures. The direct epilogue reduces declared
matrix threadgroup scratch from 20 to 12 KiB.

The adapter enables the direct epilogue only after two startup GPU probes verify
row/column fragment mapping, exact expected outputs, tails and guards on that
device/compiler. An unavailable or failed probe retains the shared-result path.
Comparison flags disable default direct selection before setting their variants.

The experimental 64×64 matrix tile uses 16 KiB scratch and requires its own
65×65×67 startup coordinate probes. `encoder-tile64-compare` checks unconditional
use against 32×64; `encoder-tile64-selective` checks a policy using 64×64 for
1024-wide linears with 193–256 or 385–448 rows, and 800×480 convolution panels. Other shapes
retain 32×64. Both modes compare exact projected outputs.
`encoder-profile-tile64` provides isolated dispatch attribution;
`asr_dataset ... --compare-tile64` runs the paired selective-policy corpus test.
Unconditional 64×64 regressed short fixtures. The original minimum-row policy
regressed intermediate corpus lengths. The narrower policy follows boundary
sweeps. Two M4 Pro dev-clean-100 runs (forward/reversed clip order) reduced
encoder time by 2.63%/2.85% and total time by 0.37%/0.29%. All transcripts matched,
with 2.279% normalized WER and zero failures. Paired speaker-bootstrap baseline/
candidate time-ratio 95% intervals were [1.00044, 1.00708] and [1.00119, 1.00491].
The default remains validated 32×64 direct epilogues. Other Apple GPUs are unmeasured.

`asr_matrix_sweep 12` measures 32×64 vs 64×64 over 108 matrix shapes;
repeat with `asr_matrix_sweep 12 --reverse` to reverse the shape order.
Build with `cargo build --release -p plowrt --features metal,dist --example asr_matrix_sweep`.
It emits JSONL GPU timings and pipeline limits, alternates tile order, excludes
two warmups per shape, and checks exact outputs plus trailing guards each pair.
Warmups exercise an extra group; timed grids match production dispatch sizes.
The buffers use deterministic BF16-exact inputs and FP32 storage, with bias enabled.
These isolated, reused-buffer timings do not establish full-model speedups or
actual pipeline occupancy. On M4 Pro, both sweep orders favored 64 rows at
M=193/255/256,N=1024, but regressed at M=257/319/320. Larger shapes contradict
a universal 64-group wave rule. Keep production selection unchanged until an
end-to-end comparison validates a device-specific candidate.

`asr_check ... encoder-bf16-compare 4` checks exact encoder output parity between
FP32 and compact BF16 linear-weight storage. `asr_dataset ... --compare-bf16-weights`
alternates these paths with 32×64 linears, packed convolution and current decoder
defaults fixed. Both paths warm up before measurement. Compact weights reconstruct
the same FP32 operands; accumulation is unchanged. This experimental mode retains
both weight copies and is not the production default.
On the M4 Pro dev-clean-100 comparison, total inference time was 76.75 s with
FP32 weights versus 75.84 s with BF16 weights (1.19% reduction; paired speaker
bootstrap ratio 95% interval [0.9812, 0.9948]). All transcripts matched, with
2.279% normalized WER and zero failures. Candidate p50/p95 were slightly worse;
this supports an aggregate improvement on this subset, not a tail-latency claim.

For decode-head grouping comparisons, run `asr_dataset ... --single` in separate
processes with `PLOW_METAL_DECODE_HEADS=0` and `1`, recording variant order and
excluding each process's warmup. For `asr_decoder_trace ... --profile-decode`, also
set `PLOW_METAL_QKV_DOT4=1` to match the ASR decoder tuning: the trace tool loads
the generic Metal engine. Compare the saved `decode-logits.bf16` bytes as well as
timings. The differential Metal probe is `runtime/apple/probe/decode_heads.m`.
Rust validation examples live in `crates/plowrt/examples/asr/`; Cargo example names
include `asr_check`, `asr_dataset`, `asr_batch_dataset`, `asr_nemotron_dataset`,
`asr_stream` and `asr_decoder_trace`.
Runtime code lives in `crates/plowrt/src/asr/`; Metal execution is in
`crates/plowrt/src/exec/apple/asr.rs` and `runtime/apple/asr.metal`.
See [runtime instructions](../../docs/runtime/asr.md) for model compilation and serving.

| Tool | Purpose | Python dependencies |
|---|---|---|
| `reference.py` | Export frontend, encoder, decoder and transcript reference fixtures | qwen-asr, Transformers, Torch, NumPy, soundfile |
| `reference_dataset.py` | Run the pinned CPU reference with explicit packed encoder windows | same reference environment |
| `librispeech_manifest.py` | Verify an official archive and prepare a deterministic speaker-balanced WAV manifest | soundfile |
| `mlx_dataset.py` | Run the same manifest through a local BF16 MLX checkpoint | mlx-qwen3-asr |
| `mlx_decoder_check.py` | Measure cross-backend decoder drift with reference embeddings | mlx-qwen3-asr, NumPy |
| `decoder_reference.py` | Export decoder layer tensors from reference audio embeddings | reference environment |
| `compare_decoder_trace.py` | Compare native instruction traces and reference tensors; optional RMSNorm rounding attribution | NumPy; Torch/safetensors for attribution |
| `roofline.py` | Ideal tensor traffic, achieved matrix rates and explicit calibrated bandwidth/compute floors from GPU profiles | standard library |
| `score.py` | Corpus WER/CER, latency summaries and paired speaker bootstrap | jiwer |
| `test_score.py` | Scoring and normalization regression checks | jiwer |

`asr_packet_dataset` is the model-independent native packet corpus runner. It
selects the transcriber from packet driver metadata, warms one request, retains
one loaded engine, and emits the same JSONL schema consumed by `score.py`:

```sh
nix develop -c sh -c 'exec "$@" > /path/packet.jsonl' sh \
  cargo run --release -p plowrt --features metal,gguf,hf-tokenizer \
  --example asr_packet_dataset -- \
  /path/model.pkt /path/tokenizer-or-gguf /path/manifest.jsonl metal
```

Pass `English` (or `en`) for Qwen and `en-US` for the current Nemotron asset when
an explicit language is needed; the two model families use different language labels.

`coval_stream.py` preserves Coval's streaming measurement contract: it verifies
the canonical WAV hashes, trims to the manifest speech-end offset, sends PCM16 in
100 ms chunks on absolute realtime deadlines, excludes the handshake, retains
failures, and reports first transcript, forced-final, TTFS, RTF, and revision-2
Whisper-normalized WER. Use the packet pipeline name reported by the server (`decode`
for the current Qwen packet):

```sh
plans/asr-eval-assets/venv/bin/pip install \
  websockets==15.0.1 whisper-normalizer==0.1.12 jiwer==4.0.0
plans/asr-eval-assets/venv/bin/python scripts/asr/coval_stream.py \
  ws://127.0.0.1:8080/v1/audio/transcriptions/stream decode \
  /path/coval/runner/src/coval_bench/datasets/manifests/stt-v3.json \
  plans/asr-eval-assets/coval-stt-v3/audio \
  --concurrency 8 --output /tmp/qwen-stream.jsonl \
  --summary /tmp/qwen-stream-summary.json
```

Use separate reference and MLX virtual environments. Record resolved package versions;
these are validation dependencies, not native runtime dependencies. Run commands through
`nix develop -c`.

```sh
nix develop -c "$REFERENCE_PYTHON" scripts/asr/librispeech_manifest.py \
  /path/dev-clean.tar.gz --checksums /path/md5sum.txt \
  --split dev-clean --count 100 --out /path/dev-clean-100

nix develop -c sh -c 'exec "$@" > /path/native.jsonl' sh \
  cargo run --release -p plowrt --features metal --example asr_batch_dataset -- \
  models/Qwen3-ASR-1.7B plow-out/qwen3-asr-1.7b-b4/model.pkt \
  /path/dev-clean-100/manifest.jsonl

nix develop -c sh -c 'exec "$@" > /path/nemotron.jsonl' sh \
  cargo run --release -p plowrt --features nemo-asr \
  --example asr_nemotron_dataset -- \
  /path/libnemo_speech.dylib /path/nemotron-model \
  /path/dev-clean-100/manifest.jsonl 0 en-US

nix develop -c sh -c 'exec "$@" > /path/mlx.jsonl' sh \
  "$MLX_PYTHON" scripts/asr/mlx_dataset.py \
  models/Qwen3-ASR-1.7B /path/dev-clean-100/manifest.jsonl

nix develop -c "$REFERENCE_PYTHON" scripts/asr/score.py \
  /path/dev-clean-100/manifest.jsonl /path/native.jsonl /path/nemotron.jsonl \
  /path/mlx.jsonl \
  --out /path/report.json
```

Redirect JSONL inside the development shell as shown; redirecting `nix develop`
itself also captures shell startup banners and produces invalid JSONL.

`asr_batch_dataset` reports exact cohort time and assigns an equal share to each row so
aggregate throughput remains exact; row latency percentiles are cohort averages. The
older `asr_dataset` example alternates scalar/tiled order per utterance using one loaded
model, after warming both variants. Timers exclude file loading and WAV decoding.
The MLX runner uses preloaded waveforms, explicit BF16, automatic language and the same
1024-token output cap. Run backends serially to avoid shared-GPU interference.
Retain failed and excluded rows; never score only successful utterances.
The CPU reference runner is for correctness comparison; its latency is not a competing
Apple GPU performance baseline. It records the explicit upstream packed-window mask patch.

Scoring reports raw WER and normalized WER/CER. Normalization uses Unicode NFKC,
case folding, curly-apostrophe mapping, punctuation splitting (preserving apostrophes)
and whitespace collapse. It does not expand numbers. CER excludes whitespace.
Pairwise timing ratios are left total inference time / right total inference time,
with paired speaker-bootstrap intervals. They describe workload sampling uncertainty,
not repeated-run thermal or system variance.
Every backend must cover all eligible manifest IDs exactly once. Errors remain empty
hypotheses and count as deletions. These normalization rules must be named when comparing
with published scores. A development subset is not a held-out benchmark result.

Decoder diagnosis uses existing model reference fixtures from `reference.py`:

```sh
nix develop -c "$REFERENCE_PYTHON" scripts/asr/decoder_reference.py \
  models/Qwen3-ASR-1.7B /path/short-model --out /path/decoder-reference
nix develop -c cargo run -p plowrt --features metal --example asr_decoder_trace -- \
  models/Qwen3-ASR-1.7B plow-out/qwen3-asr-1.7b/model.pkt \
  /path/short-model /path/decoder-native
nix develop -c "$REFERENCE_PYTHON" scripts/asr/compare_decoder_trace.py \
  /path/decoder-native /path/decoder-reference --checkpoint models/Qwen3-ASR-1.7B
```

The trace requires one prefill bucket large enough for the prompt and verifies replay
logits against normal execution. It is a diagnostic, not a latency benchmark.

To compare convolution kernels, pass `--compare-conv` to `asr_dataset`. Both variants
use tiled linears; `native-tiled` retains scalar convolution, `native-tiled-conv`
uses tiled convolution. The original invocation still compares scalar/tiled linears
with scalar convolution in both variants. `asr_check ... encoder-conv-compare 4`
alternates convolution implementations on reference mel and requires exactly matching
projected embeddings. Treat the first pair as warmup when reporting warm latency.

`encoder-profile-tiled` profiles the production direct tiled encoder. The
`asr_decoder_trace ... --profile-decode` variant records four isolated decode-step
profiles and requires exact normal/replay logits for each. Neither is a serving
latency benchmark. `roofline.py PROFILE --kind encoder|decoder --out REPORT`
reports matrix FLOPs and ideal tensor traffic. Supply measured per-device ceilings
with `--bandwidth-gbps`, `--compute-tflops` and `--ceiling-source`; it never infers
hardware rates from chip names. Ideal traffic is not measured DRAM traffic.

`encoder-packed` and `encoder-packed-compare` evaluate convolution
patch packing with buffers reused across audio chunks. `asr_dataset --compare-packed-conv`
compares direct tiled vs packed convolutions with tiled linears in both variants.
The adapter defaults to packed panels for convolutions two and three; convolution
one remains direct tiled. Extra declared scratch is at most 19.4 MB across both
stages, independent of clip length within the supported range. This is an allocation
bound, not a measured process memory reduction.

`asr_dataset ... --compare-handoff` alternates CPU embedding construction and GPU
embedding splice with the same packed encoder. `asr_check ... model-device BLOB`
checks the GPU handoff against a waveform-to-transcript fixture. The device path
retains the completed encoder buffer and writes BF16 rows directly into decoder
input. GPU-only prefill encodes splice in the decoder command buffer; CPU/ANE
offload retains completed staging. The encoder completion wait remains. The latest
100-clip corpus produced identical transcripts, but its paired timing was neutral
(85.72 s host vs 85.84 s device), so it stays opt-in.


The encoder reuses eight request-local layer buffers across its 24 layers.
This replaces 264 intermediate allocations while keeping returned audio separately
owned. On M4 Pro, a paired 100-clip dev-clean run reduced encoder time from
8.271 to 7.771 seconds and total inference from 55.279 to 54.671 seconds.
All transcripts matched (48/2062 word errors); nine of ten batches improved.
Direct encoder outputs matched the original allocation path at six feature
lengths spanning 50–3000 frames, including retained outputs across repeated calls.
Declared layer-intermediate storage at 206 rows falls from 270.375 to 8.852 MiB;
this excludes other buffers and is not a measured process-memory reduction.

Optional MXFP4 decoder weights trade accuracy for lower latency. On the same
100-clip dev-clean sample, FP8 took 54.379 s vs MXFP4 50.051 s (7.96% less time,
all ten paired batches faster). Word errors increased from 48 to 51 over 2062
words (2.328% → 2.473% WER); 28 raw transcripts differed. This does not establish
quality equivalence or state-of-the-art performance. The encoder, embedding
lookup, activations and KV remain BF16. Held-out comparison is still required.

```sh
nix develop -c sh -c 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 plans/asr-eval-assets/venv/bin/python perf-data/tools/quantize_mxfp4.py models/Qwen3-ASR-1.7B plans/asr-eval-assets/qwen3-mxfp4 thinker.model. --extra thinker.model.embed_tokens.weight'
nix develop -c target/debug/plowc --hf-dir models/Qwen3-ASR-1.7B --gpu m4pro --max-ctx 2048 --mxfp4 --mx4-prefill 1 --out plow-out/qwen3-asr-1.7b-mxfp4
nix develop -c sh -c 'PLOW_MXFP4_DIR=plans/asr-eval-assets/qwen3-mxfp4 target/release/plowrt asr --packet plow-out/qwen3-asr-1.7b-mxfp4/model.pkt --tokenizer models/Qwen3-ASR-1.7B --audio AUDIO.wav'
```

The explicit `--extra` includes Qwen ASR's tied decoder head: the generic Python
quantizer does not inspect `thinker_config` for the tied-embedding setting.

Held-out comparison (100 test-clean clips, 2151 words, alternating process order,
excluded warm-up; same source checkpoint):

| Backend | Inference time | Word errors | WER |
| --- | ---: | ---: | ---: |
| Native MXFP4 | 52.128 s | 44 | 2.046% |
| MLX Q8, group 64 | 41.655 s | 36 | 1.674% |
| MLX Q4, group 64 | 31.446 s | 42 | 1.953% |

All 300 transcriptions completed. Native MXFP4 was slower in every batch:
25.14% slower than MLX Q8 and 65.77% slower than MLX Q4. MLX quantizes 345
modules, including encoder linears and embeddings; native MXFP4 quantizes 197
decoder/head projections. These are different recipes, not identical formats.
MXFP4 remains optional; this result rules out a performance-leadership claim.

Ordered MXFP4 dispatch uses 64-thread kernels for aligned MXFP4 GEMV/GLU and a
dedicated 1024-thread entrypoint for supported prefill GEMMs. Current causal ASR
packets request it through the backend-neutral `ordered_dispatch` pipeline parameter,
and Metal loads eligible MXFP4 kernels automatically. Older packets can still select
the same diagnostic path with `PLOW_METAL_SERIAL=1 PLOW_METAL_MX4_DEDICATED=1`.
Unsupported kernel shapes fall back to generic ordered execution.
On 100 dev-clean clips, persistent execution took 50.072 s, generic ordered
58.199 s, and specialized ordered 49.070 s. All transcripts matched, with zero
failures; the specialized path won all ten paired batches against both controls.
Four full-model decode-logit comparisons and 65 aligned kernel output/guard
checks passed. This remains opt-in and has only been measured on M4 Pro.
Use it for full-model execution: serial mode is not a partial-segment or
heterogeneous execution path. The kernel probe is
`runtime/apple/probe/mx4dispatchcheck.m` (pass `runtime/apple/interp.metal`).

The prefill entrypoint preserves the existing GEMM arithmetic and tile selection.
Two paired runs over 45 development clips, with reversed process order, measured
0.60% and 0.61% lower prefill time, and 0.44% and 0.15% lower total time.
All 180 transcriptions matched across variants/repeats (22 errors / 912 words per
45-clip run), and full prefill logits matched the previous entrypoint. These are
small M4 Pro improvements, not a performance-leadership result. The kernel probe
`runtime/apple/probe/mx4prefillcheck.m` takes `runtime/apple/interp.metal` and checks
140 opcode/shape/partition cases, including row offsets, tails and extra groups.

Affine Q4 is an additional experimental decoder format, distinct from MXFP4:
unsigned nibbles in U32 words, with BF16 scale and additive bias per 64 weights.
The local lossless import of the MLX Q4 decoder is
`plans/asr-eval-assets/qwen3-affine-q4`; its `conversion.json` records source,
output and per-tensor hashes. Encoder and embedding lookup remain original BF16.

```sh
nix develop -c target/debug/plowc --hf-dir models/Qwen3-ASR-1.7B --gpu m4pro --max-ctx 2048 --affine-q4 --out plow-out/qwen3-asr-1.7b-affine-q4
nix develop -c sh -c 'PLOW_AFFINE_Q4_DIR=plans/asr-eval-assets/qwen3-affine-q4 target/release/plowrt asr --packet plow-out/qwen3-asr-1.7b-affine-q4/model.pkt --tokenizer models/Qwen3-ASR-1.7B --audio AUDIO.wav'
```

Both phases use Q4 projections; the last-row output head uses Q4 GEMV. Gate/up
projections round to BF16 separately before GLU. Loading validates all weight,
scale and bias dimensions, dtypes and byte counts. Initial emission supports
dense Qwen3/Llama on Metal or the CPU tier, TP1; heterogeneous and packed
prefill routes are rejected.

On 100 dev-clean clips (2,062 words), ten rotating process batches with excluded
warmup produced:

| Backend | Inference time | Word errors | Normalized WER |
|---|---:|---:|---:|
| Native affine Q4 | 50.982 s | 51 | 2.473% |
| Native MXFP4, dedicated ordered dispatch | 49.471 s | 51 | 2.473% |
| MLX Q4 group64 | 30.045 s | 47 | 2.279% |

All 300 transcriptions completed. Affine Q4 lost every batch against both
controls: time ratios 1.0305 vs MXFP4 (95% batch-bootstrap interval
[1.0243, 1.0377]) and 1.6969 vs MLX Q4 ([1.6883, 1.7046]). MLX also quantizes
encoder linears and embedding lookup, so these are different recipes.
Four decode replays matched normal logits exactly; the reference audio produced
the expected transcript. This establishes working native Q4 execution, not a
performance lead or a default change. Evidence and the import prototype are in
`plans/asr-affine-q4-2026-09-11/`.


A subsequent M4 Pro experiment tested 64-thread affine Q4 GEMV in ordered
execution. Across the same 100 development clips, persistent affine Q4 took
51.058 s, the candidate 59.818 s, and dedicated ordered MXFP4 49.882 s.
All 300 transcriptions completed; the two affine modes produced identical
transcripts (51 word errors, 2.473% WER). The candidate was 17.16% slower
overall and lost every paired batch, despite faster isolated GEMV kernels.
Prefill increased from 15.099 s to 24.416 s; decode decreased from 27.911 s
to 27.323 s. The runtime change was rejected. The existing persistent affine
path remains available; no affine dedicated-dispatch flag is shipped.
Local source, probes, hashes and results are archived in
`plans/asr-affine-q4-dispatch-2026-09-11/`.


A follow-up restricted the candidate to complete GPU-only single-row decode,
preserving persistent prefill. On another 100-clip paired run, persistent
affine Q4 took 50.906 s, decode-only specialization 50.957 s, and MXFP4
49.391 s. All 300 transcriptions succeeded; both affine modes produced
identical transcripts. Persistent/candidate time ratio was 0.9990
(95% paired batch-bootstrap interval [0.9743, 1.0139]), so no reliable
end-to-end gain was established. Four decode-logit steps also matched exactly.
The follow-up was rejected; evidence is local in
`plans/asr-affine-q4-decode-dispatch-2026-09-11/`.


Encoder normalization now stages a row across 32 threads, preserves the serial
mean/variance reduction, and distributes output writes for width 1,024 and at
most 256 rows. Other shapes and devices unable to build the optional kernel
use the original path. No weights or packet format change.

On M4 Pro, two alternating 100-clip dev-clean comparisons with native MXFP4
showed total time 50.186 → 49.381 s and 49.207 → 48.817 s (1.60% and 0.79%
less). Encoder time decreased 6.19% and 6.13%. The candidate won nine of ten
batches in each run; baseline/candidate batch-bootstrap 95% intervals were
[1.0015, 1.0326] and [1.0030, 1.0123]. All 400 transcriptions succeeded with
identical paired transcripts: 51/2,062 word errors (2.473% WER).
Isolated probes covered 276 exact output/guard cases across three runs;
the retained boundary probe is `runtime/apple/probe/asrnormcheck.m`, taking
`runtime/apple/asr.metal` as its argument. Evidence is local in
`plans/asr-norm-staging-2026-09-11/`. Other Apple GPU performance is unmeasured.


The encoder also has an optional implicit-convolution path: gather patches into
the matrix tile and write channel-major output directly, removing unfold and
unpack submissions. It requires the validated direct-epilogue mapping and a
compatible 256-thread pipeline; otherwise the existing path remains active.

Two 100-clip MXFP4 comparisons on M4 Pro preserved every paired transcript
(400 successful transcriptions; 51/2,062 word errors in every variant).
Encoder time fell 8.414 → 7.806 s and 7.266 → 6.870 s (7.22% and 5.46%).
Total time was 52.709 → 51.954 s and 48.750 → 48.578 s. The repeat's
baseline/candidate 95% batch-bootstrap interval [0.9994, 1.0065] includes no
total-time change; the encoder gain is more consistent than overall latency.
Seven exact convolution cases include odd shapes, multiple chunks, output guards
and an extra dispatched group. The probe is runtime/apple/probe/asrconvcheck.m,
taking runtime/apple/asr.metal. Evidence: plans/asr-implicit-conv-2026-09-11/.

For the second ASR family, see [Nemotron support](nemotron/README.md). Native
`rnnt.greedy.v1` packets now run frontend, FastConformer, predictor and joint
stages through Plow. NVIDIA's library remains the external Metal control; native
cached streaming and wide-request RNNT batching remain pending.


Metal exposes prefill_slot_embeddings for host BF16 prompt embeddings and
prefill_slot_embeddings_staged for device handoff into an individual decode slot.
The staging callback can encode a copy into the prefill command buffer; it must
not commit that buffer and must keep operands alive until completion. Errors
restore the KV table base, without rolling back completed KV writes.

The asr_slot_check example compares host and device staging against isolated
single-slot execution. Full reference-652 embeddings and a half-length prefix
exercise different lengths, four decode steps, swapped slot reuse and rejected
inputs/callback errors. Normal and 128-row chunked packets pass in persistent
and dedicated ordered modes: all 128 BF16 logit rows and sampled tokens match.
The prefix is a numerical fixture, not a second complete audio transcription.
The callback-error test fails before first-chunk submission.

The slot checker also accepts larger batch packets. It uses one distinct prefix
length per slot and checks forward/reversed assignments. B2 and B4 packets pass
in persistent and dedicated ordered modes: all 192 full BF16 logit rows and
sampled tokens match isolated execution. These remain numerical prefix fixtures;
the full-audio batch checker separately covers distinct recordings and idle slots.

This early slot fixture preceded the bounded ASR server mux and full corpus
gates. Continuous encoder/prefill refill remains pending. Evidence:
plans/asr-slot-device-staging-2026-09-11/.

`QwenAsr::transcribe_batch` accepts a fixed cohort up to the packet's slot count.
It runs each recording's encoder and prefill sequentially, then shares decoder
steps across slots. Results preserve input order; completed slots remain idle
until the cohort finishes. Language/context are shared by the cohort, and any
error or cancellation aborts it. This API does not change HTTP/WS admission.

Build `asr_batch_check` to compare full-audio batched transcripts against B1:

```sh
nix develop -c cargo build --release -p plowrt --features metal,dist --example asr_batch_check
# Set the same precision/dispatch environment for both packet variants.
target/release/examples/asr_batch_check CHECKPOINT B1_PACKET BATCH_PACKET MANIFEST OUT.jsonl
```

The checker reads capacity from the batch packet. Use more recordings than that
capacity, with a count that is not a multiple of it, and a new output path. The
checker tests host/device handoff, forward/reversed order, full/partial cohorts,
rejected admission and reuse after a later recording fails preparation. It
records whole-cohort times; these are not per-request latency measurements.

Nine-recording checks passed with B2 and B4 MXFP4 packets on M4 Pro: all 72 batch
outputs matched isolated B1. The B4 packet was compiled with
`--emit-decode-batch 4`. This establishes fixed-cohort correctness, not a throughput
recommendation: without four-row kernel specialization, B4 was slower than B2 in
this initial sample. HTTP/WS admission remains single-request.

Validated on M4 Pro with MXFP4: nine recordings in both dispatch modes, then
99 dev-clean recordings in ordered mode. All 396 expanded-gate batch outputs
match isolated B1 text/language; pooled WER is unchanged at 51/2,038 words
(2.502%). B1 also matches archived pre-refactor transcripts.

This cohort path is not a performance recommendation. Sequential diagnostic
runs took 47.92 s for B1, 48.53–49.11 s for host-handoff B2, and 58.43–64.64 s
for device-handoff B2. Default single-request host handoff remains unchanged.
Evidence: plans/asr-audio-batching-2026-09-11/.

The two-row MXFP4 op91 kernel now shares decoded weights across both inputs,
preserving the original reduction order. It selects only M=2, K divisible by 32,
and partitions aligned to four output columns; other cases use the existing
path. No weight repacking or device-specific asset format is required.

The four-row op91 specialization shares each weight across four inputs, retaining
the same K/partition guards and reduction order. It is compiled only for four-slot
assets containing a four-row MXFP4 GEMV or GLU; B1/B2 shaders exclude the helpers and branches.
Five paired nine-recording groups on M4 Pro reduced B4 host-handoff time from
25.5708 to 23.0427 seconds (9.9% less), with wins in all five groups. Device handoff
improved by 9.7%.
All 360 batched outputs matched B1 and the previous kernel: 22 errors / 912 words
per 45-clip pass. The B1 time-ratio interval included no change. This compares
optimized B4 with unoptimized B4; it does not establish superiority over B2 or
an external ASR runtime. GLU was unchanged in that comparison. The broader B2 control averaged
0.6% more host-handoff time; its ratio interval [0.9990, 1.0179] spans no change,
so a small timing regression is not ruled out. The probe
`runtime/apple/probe/mx4fourrowcheck.m` takes baseline and candidate Metal source
paths, plus optional `--dedicated`, to compare both implementations.

Four-row GLU (op92) also reuses weights when M=4 and K is divisible by 32,
preserving reduction order, activation and BF16 rounding. Against the retained
four-row op91 baseline, five alternating paired nine-recording groups reduced
B4 host-handoff time from 23.1014 to 20.5466 seconds (11.1% less) and device-handoff
time by 10.2%, with wins in all five groups. Host time-ratio 95% block-bootstrap
interval: [0.8863, 0.8939]; B1 control: [0.9966, 1.0037]. All 360 batch outputs
matched baseline and B1, with 22 errors / 912 words per 45-clip pass. The 192-row
full-logit checks, 72 additional audio outputs and 560 kernel/guard cases passed.
The probe enables the four-row compilation guard for both source versions and
covers all four activation modes with finite parameters. Exceptional activation
parameters and performance on other Apple GPUs remain untested. These results
compare fixed B4 cohorts, not live serving capacity or external runtimes.
Evidence: plans/asr-mx4-four-row-glu-2026-09-11/.

On M4 Pro, five paired nine-recording blocks (alternating binary order) reduced
host-handoff B2 time from 22.8740 to 21.2460 s: 7.1% less time, gains in 5/5 blocks.
The block-bootstrap throughput ratio 95% interval is 1.063–1.088. Single-request
control timing is unchanged within that test's uncertainty. All 360 batch
transcripts match the prior binary and B1; WER stays 22/912 words on 45 recordings.
Exact kernel/guard and full-logit checks also pass. This is an improvement over
the prior cohort implementation, not a state-of-the-art or live-stream capacity
result. M3/M5 performance remains unmeasured.
Evidence: plans/asr-mx4-batch-reuse-2026-09-11/.

Two-row MXFP4 GLU (op92) now also shares weights/scales across inputs when
M=2 and K is divisible by 32. Its reduction order, activation and BF16 rounding
remain unchanged; other shapes use the previous path.

Against the op91-reuse baseline, five alternating paired audio blocks reduced
host-handoff B2 time from 22.7551 to 20.9780 s (7.8% less time, 5/5 wins).
The block-bootstrap throughput ratio 95% interval is 1.062–1.111. All 360
batch transcripts match B1 and the prior binary; WER stays 22/912 words on
45 recordings. Single-request timing shows no reliable change.
195 kernel/guard checks and 64 full-logit comparisons pass, including generic
dispatch, tail fallbacks and activation variants. This is a measured B2 gain;
it does not establish state-of-the-art performance or M3/M5 capacity.
Evidence: plans/asr-mx4-glu-batch-reuse-2026-09-11/.

Set RUST_LOG=plowrt::asr::qwen=info when running asr_batch_check to report
cohort stage wall times and decode utilization. `launched_decode_rows` counts
the rows selected by the decode ladder; `active_decode_rows` counts rows producing
output tokens. Their difference includes finished and unused slots. For legacy
single-rung assets, launched rows equal slots times decode steps. These traces are diagnostic;
they do not measure pure GPU time or establish serving capacity.

The latest 99-recording cumulative check preserved all 792 batch transcripts
across old/new binaries (51/2,038 word errors). Its sequential timing had large
control/order drift, so it does not establish a clean cumulative speedup.
A separate nine-recording profile validated the counters and found 23.7% idle
decode rows; prefill/decode account for approximately 39%/44% of cohort time.
Evidence: plans/asr-batch-cumulative-2026-09-11/.

A B4 asset with decode rungs 1, 2 and 4 now narrows execution to the highest
occupied slot as trailing requests finish. Five paired nine-recording groups
reduced host-handoff time from 20.4651 to 19.1142 seconds (6.6% less) and device
handoff by 6.4%, with wins in all five groups. Idle rows fell from 47.6% to 29.8%.
All 360 batch outputs matched fixed B4 and B1; both passes retained 22 errors /
912 words. The B1 control interval included no change. The width-switch fixture
checks 4→1→2→4 execution, both slot orders and both dispatch modes. A persistent
interpreter timeout found during validation was fixed by compiling base and M4
pipelines separately; three further persistent repetitions passed. Occupied-slot
selection is shared scheduler code used by CPU/Metal slot execution. CUDA and AMD
use the same decode-rung contract, but this Qwen ASR path has not yet run there.
Evidence: plans/asr-decode-ladder-2026-09-11/.
