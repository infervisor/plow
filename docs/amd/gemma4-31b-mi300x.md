# Gemma 4 31B: packed prefill on MI300X

Qualified on 2026-09-07: one MI300X, TP1, BF16, batch ladder 1/2/4, prefill
rungs 128/512/1024, context 8192. Packing remains opt-in. It improves throughput
against the same chunked configuration on long concurrent prompts, but does not
beat whole-prefill serving or vLLM in this measurement.

## Build and run

Use `nix develop` for every command. Build the compiler/runtime with
`cargo build --release -p plowc -p plowrt --features plowrt/hsa` and build
`lean-plow` with `lake build` from its directory. Compile from the **complete
checkpoint**: configuration-only emission substitutes identity `layer_scalar`
values and must never be used for numerical validation or serving.

The tested checkpoint is `build-gemma31/checkpoint`. The qualified assets are
`build-gemma31/qualified-assets`; the earlier `build-gemma31/assets` directory
contains structural assets and is unsuitable for serving.

```bash
PLOW_VERIFY_BIN="$PWD/lean-plow/.lake/build/bin/plow_verify" \
PLOW_L2_PLACE=0 PLOW_AMD=1 PLOW_DECODE_BATCH=4 PLOW_DECODE_BATCH_LADDER=1,2,4 \
  target/release/plowc --hf-dir "$PWD/build-gemma31/checkpoint" \
  --gpu MI300X --arch gfx942 --num-gpus 1 --max-ctx 8192 \
  --out "$PWD/build-gemma31/qualified-assets"

PLOW_DECODE_BATCH=4 scripts/build_gfx942.sh "$PWD/build-gemma31/hsaco"

perf-data/tools/gpulease -n 1 gemma31-packed \
  env PLOW_HSACO="$PWD/build-gemma31/hsaco" \
  PLOW_PF_BATCH=1 PLOW_PF_CHUNK=512 PLOW_MULTISTEP=4 \
  target/release/plowrt serve --assets "$PWD/build-gemma31/qualified-assets" --port 8000
```

The asset directory must link `checkpoint` to the full checkpoint, `tokenizer.json`
to its tokenizer, and `hsaco` to the built objects. `/v1/models` reports the asset's
network name; the qualification bundle uses `gemma-4-31B-it`. A fresh emission
from a directory named `checkpoint` uses that basename instead.

Use `gpulease` for every GPU process, including tests. It must detect all eight
GPUs on this host. The current Nix SMI setup needs its Python environment first
in PATH (`/nix/store/3i4w73bg0ak00b1rx6p3n6ks1sjfpykc-python3-3.13.13-env/bin`).
The project compiler is the Nix ROCm 7.14.0 toolchain enforced by the build script.

## Runtime contract

- Initialized middle chunks can share a larger compiled prefill rung. Initial
  admission, final sampling and prefix snapshot boundaries remain isolated.
- Dense BF16 attention uses each request's KV slot and position, including ring
  wraparound and split attention partials. Parked rows do not write request state.
- Every routed kernel object must advertise
  `plow_packed_prefill_dense_consumers_1`. Unsupported operator families and legacy
  L2-domain packets fall back to isolated scheduling; explicit staging rejects them.
- Cursor frontiers commit after successful execution. Binding cleanup runs on
  success and failure. Multistep admission rejects inactive and duplicate slots
  and clamps the quantum to every active context's remaining capacity.
- Deferred decoding captures each token on device and reads once per quantum.
  Per-token drains and counter checks remain enabled.
- AMD and NVIDIA share context-budget calculation. Single-GPU AMD and TP reuse
  host cursor scheduling. AMD packed and mixed dense attention share span dispatch.
  Intel support here is the CPU backend; no Intel GPU backend was added.
- On AMD, packed prefill and multistep decode execute sequentially within a mux
  tick. The mixed prefill/decode kernel path has no AMD serving adapter and is
  not qualified as a single fused launch.

## Verification

482 combined CPU/CUDA/HSA library tests and eight API integration tests passed.
Three full-model GPU cases compare packed prefill and two four-token decode
quanta with isolated greedy execution: short padded packs, unequal positions
crossing ring boundaries, and a full 1024-row pack. Rejected members, duplicate
feeds, sparse slots and slot reuse are covered.

```bash
perf-data/tools/gpulease -n 1 gemma31-parity \
  env PLOW_GPU_TEST=1 PLOW_GPU_ASSETS="$PWD/build-gemma31/qualified-assets" \
  cargo test -p plowrt --features hsa --test hsa_multistep -- --test-threads=1

python3 scripts/verify_packed_serve.py \
  --control http://127.0.0.1:18911 --candidate http://127.0.0.1:18913 \
  --model gemma-4-31B-it --max-ctx 8192
```

The HTTP test passed for both 128- and 512-token chunks: concurrent ragged
requests, exact output limits 1/3/7/17, cancellation during prefill and decode,
recovery, and oversized-context rejection. Debug mux logs confirmed actual
packed launches. Both Plow configurations and vLLM answered `Paris` in the chat
smoke test. GPU execution was validated on MI300X only; CUDA/CPU validation was
through host tests. This is functional qualification, not a long-running soak.
An earlier separate VMM pool test failed on its mutable-environment assumption
with the immutable runtime configuration; that unchanged test is outside this
qualification, which uses the default non-VMM AMD KV path.

## Comparison

Each cell is **aggregate output tokens/s / median TTFT in milliseconds**.
64 output tokens, concurrency 1 or 4, two repetitions after warmup. Text fixtures
and usage checks are identical; prefix caching is disabled. These historical
runs used four short warmup requests and two repetitions. The current benchmark
defaults to a full warmup per case and five repetitions.
Whole prefill uses `PLOW_PF_NO_CHUNK=1`, packing off and multistep 1. Chunked
configurations use multistep 4. `Chunk 512` disables packing; `Packed 512` enables
it with the same chunk size. Results are indicative, not a fully matched
scheduler comparison: vLLM used its 16384-token batching budget while Plow used
a 2048-token prefill interleave budget. Its compiled native GELU also used the
erf approximation; Plow uses the checkpoint's tanh approximation. They do not
establish an apples-to-apples performance win or a load limit.

| Input / concurrency | Whole prefill | Chunk 512 | Packed 512 | Packed 128 | vLLM 0.28 |
|---|---:|---:|---:|---:|---:|
| 128 / 1 | 26.0 / 97 | 26.9 / 95 | 26.7 / 96 | 26.9 / 97 | 55.9 / 49 |
| 128 / 4 | 82.0 / 291 | 81.0 / 341 | 78.2 / 391 | 78.9 / 388 | 189.2 / 110 |
| 1024 / 1 | 24.5 / 264 | 24.7 / 324 | 24.1 / 327 | 20.9 / 765 | 50.1 / 137 |
| 1024 / 4 | 66.3 / 711 | 58.5 / 1029 | 60.3 / 1177 | 47.5 / 2444 | 138.3 / 514 |
| 4096 / 1 | 18.2 / 1148 | 17.4 / 1396 | 17.0 / 1413 | 11.6 / 3188 | 36.5 / 555 |
| 4096 / 4 | 34.3 / 2922 | 25.3 / 4423 | 31.7 / 4905 | 20.0 / 9753 | 72.7 / 2140 |

At 4096 input tokens and four requests, packing improves matched chunked
throughput by 25% (25.3 → 31.7 tokens/s), with median TTFT increasing from 4.42s
to 4.91s. Whole prefill reaches 34.3 tokens/s and vLLM reaches 72.7 tokens/s.
Use 512-token chunks when enabling this path; the tested 128-token setting adds
substantial overhead. Packing is not enabled by default.

vLLM is the official 0.28.0 ROCm wheel from
`https://wheels.vllm.ai/rocm/0.28.0/rocm723`, with PyTorch
`2.12.0+git6bbd260` (build HIP `7.2.53211`). It runs with system-built ROCm
`/opt/rocm/core-7.14/lib`; loading the Nix ROCm libraries into that wheel crashed
PyTorch during import. Its Python uses Nix glibc, and its host C-extension
compiler wrapper clears `LD_LIBRARY_PATH` before invoking system GCC. Plow's
HIP kernels were built with the Nix 7.14 compiler throughout.

[Raw measurements and provenance](gemma4-31b-mi300x-20260907.json).

For new comparisons, pass `--manifest FILE` to `scripts/bench_packed_serve.py`.
Record checkpoint and tokenizer hashes, weight/activation/KV precision, TP,
GPU identity, toolchain, maximum context and sequences, token budgets, prefix
caching and backend flags. Use the same input/output/concurrency matrix and
warmups. The runner verifies exact token counts and records prompt hashes,
request latency, TPOT and text delivery timestamps. Text chunk gaps are not
token ITL: decoding and transport can combine multiple tokens in one chunk.

Current Plow FP8 prefill and decode use different activation precisions;
decode retains BF16 activations. This hybrid must not be compared as equivalent
to vLLM W8A8. Tensorwise versus per-token scales and AITER's FNUZ activation
range also require an explicit matching profile before FP8 results qualify.

## Matched BF16 rerun

The rerun uses 128 output tokens, five repetitions and one full warmup per case.
Both engines use the same checkpoint/tokenizer, BF16 weights/activations/KV,
TP1, context 8192, maximum four sequences, disabled prefix caching and greedy
sampling. vLLM uses a 2048-token batching budget and its custom tanh-GELU
kernel; compilation and GPU graphs remain enabled. Plow uses a 2048-token
prefill interleave budget, packed 512-row chunks and multistep 4. Those budget
settings have different scheduler semantics, recorded in the result manifest.
Plow counts prefill rows only: 2048 prefill rows plus four decode rows can total
2052 tokens, whereas vLLM counts both phases toward its 2048-token limit.

Plow's existing decode tier mechanism selects a dedicated MM1 object for one
request and retains MM4 for batches. This improves solo throughput by 16–17%
over the prior MM4-only object in the 128/1024-input experiment. All compared
Plow trajectories matched through 128 output tokens, and the clean build passed
HTTP parity, cancellation, output limits and slot-reuse checks.

```bash
PLOW_OCC4=1 PLOW_DECODE_BATCH=1 PLOW_ROWS_ONLY='=interp_decode' \
  scripts/build_gfx942.sh "$PWD/build-gemma31/hsaco-occ4-b1-clean"

# Set before the packed-serving command above:
export PLOW_HSACO_LOWRUNG="$PWD/build-gemma31/hsaco-occ4-b1-clean:1"
```

Do not use OCC4 for the batched object: the build script rejects a known hang. The
tested `PLOW_DEC_SQUEEZE` batched alternative regressed and was not promoted.

Each cell is **output tokens/s / median TTFT ms / median TPOT ms**.

| Input / concurrency | Plow packed + MM1 tier | vLLM 0.28 tanh |
|---|---:|---:|
| 128 / 1 | 31.50 / 95.53 / 31.24 | 55.69 / 51.55 / 17.69 |
| 128 / 4 | 87.03 / 353.87 / 42.69 | 187.88 / 115.94 / 20.52 |
| 1024 / 1 | 29.33 / 328.53 / 31.77 | 51.06 / 139.09 / 18.64 |
| 1024 / 4 | 73.33 / 1182.51 / 44.84 | 153.32 / 451.41 / 22.68 |
| 4096 / 1 | 23.35 / 1419.53 / 31.99 | 41.79 / 572.49 / 19.61 |
| 4096 / 4 | 47.07 / 4933.37 / 45.99 | 98.11 / 1672.14 / 27.41 |

vLLM remains faster in every measured case. These short closed-loop batches
do not establish saturation throughput or tail SLOs. Clocks/power were not
pinned; runs used separate single-MI300X leases. The manifest records the
actual GPU visibility, object hashes, checkpoint hashes and runtime settings.

[Measurements and manifests](gemma4-31b-mi300x-matched-bf16-20260907.json).

The vLLM activation correction is:

```bash
--compilation-config '{"custom_ops":["none","+gelu_and_mul"]}'
```

The generated graph was checked for `_C.gelu_tanh_and_mul`; the startup warning
alone cannot determine which implementation runs. The FP8 weight exporter now
provides `--scale-mode vllm-channel`, with the reference per-channel scale floor
and separate provenance metadata. This exports weights only; it does not
enable W8A8 decode.

The complete Gemma checkpoint was exported with this mode. Every row of all
410 projection matrices was checked against vLLM's GPU `_fp8_channel_scale`
and `_fp8_quant_per_channel`: zero byte mismatches and zero scale mismatches.
The comparison masks FN negative zero, reinterprets FN bytes as FNUZ and
doubles scales. Actual exporter tests also passed zero/tiny/midpoint inputs,
existing-output refusal and nonfinite-source rejection. The export and its
provenance live in `build-gemma31/fp8-ptpc-export`.

## Optional decode placement

For dense Gemma emission, `PLOW_L2_PLACE=1 PLOW_L2_PLACE_PREFILL=0` places decode
queues across L2 domains while retaining the ordinary prefill programs. The
runtime identifies placement per program using its ordered segment count; an
unplaced prefill with an even number of segments must not be mistaken for a
placed program. The new option defaults to true, preserving existing emission.

The decode-only configuration passed the 13 asset parser tests, the emitter
wave-class placement regression, all three full-model packed/multistep HSA
parity cases, and HTTP parity, cancellation, context-limit and slot-reuse checks.
All three emitted prefill program bodies and queue appendices were byte-identical
to the unplaced assets. The full-model runs used the Nix ROCm 7.14 objects and
MM1 decode tier above. This option is separate from mixed-phase kernel fusion.
