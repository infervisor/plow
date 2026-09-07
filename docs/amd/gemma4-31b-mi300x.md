# Gemma 4 31B: packed prefill on MI300X

Qualified on 2026-09-07: one MI300X, TP1, BF16, context 8192. Runtime fusion
passed four-slot and eight-slot numerical and serving checks and remains opt-in.
The stock `vllm bench serve` baseline below favors vLLM in every throughput,
TTFT and TPOT cell. Batched terminal-prefill completion subsequently improves
Plow's concurrent throughput by about 9–10%; the overall vLLM gap remains open.

## Batched terminal-prefill completion

With runtime fusion enabled, the scheduler can initialize multiple cold cursors
before selecting packed prefixes. It reserves each prompt's final token for an
ordinary batched decode operation, which returns that request's first generated
token. Ready terminal rows can share that operation with live decode requests.
Positions and cursor-to-live transitions commit after successful execution.
The no-interleave/defer-decode policies park live decoders during completion.
This follows the existing CUDA packed-prefix/terminal-decode phase design and
requires no new kernel ABI or compiler row combinations.

Four cold 128-token requests pack 508 prefix rows into T512, then complete
together through native decode. Eight such requests pack 1016 rows into T1024.
The runtime flag remains opt-in; unsupported configurations retain ordinary
execution. Prefix-cache configurations retain their isolated snapshot boundaries.

A stock-client Plow before/after experiment uses the same BF16 checkpoint,
ordinary assets, WPE5 MM1 tier, kernels, PF512 and configured multistep 4. Each
cell has four requests, 128 output tokens, one full-corpus warmup and three
measured repetitions. Both clients run concurrently on separately leased GPUs;
clocks and power remain unpinned. All 24 runs pass actual token accounting.

| Input / concurrency | Output tokens/s, before → after | Median TTFT ms, before → after | Median TPOT ms, before → after |
|---|---:|---:|---:|
| 128 / 4 | 91.29 → 100.06 | 548.36 → 278.87 | 38.94 → 37.80 |
| 1024 / 4 | 73.08 → 80.29 | 1753.86 → 1195.38 | 40.43 → 40.21 |
| 4096 / 4 | 46.71 → 50.91 | 5565.46 → 4793.25 | 41.59 → 41.13 |

The final-token transformer uses native GEMV arithmetic, whereas the former
isolated prefill completion used MFMA. Numerical batching checks therefore replay
identical packed prefixes and physical decode widths, then compare separate
terminal completion with batched completion. All 15 checked full-vocabulary
first/next-token logit rows are bitwise equal. Independent HTTP isolated/concurrent
greedy parity, cancellation, output limits, slot reuse and context recovery pass.
The B8 GPU suite also covers eight cold requests and reordered slots 7/2/5;
long-context cases reach prompt lengths 8191 and 8175.

The synthetic benchmark's full completion texts match before/after in 12/12,
10/12 and 9/12 measured requests at input lengths 128/1024/4096. This experiment
does not establish universal text equality across different phase schedules or
production tail SLOs. The numerical check above controls the phase schedule
explicitly. Separate no-interleave/defer-decode HTTP suites complete 78 requests,
58 exact parity checks and 24 intentional disconnects with successful recovery;
logs confirm live decoding is parked during terminal completion.

[Measurements, phase-matched logits and policy qualification](gemma4-31b-mi300x-terminal-20260907.json).

## Runtime-fusion baseline before batched completion

These results use source `5feeb384`, the ordinary `decode-placed-assets`,
runtime fusion enabled, packed 512-token prefill chunks, multistep 4, decode
ladder 1/2/4, the WPE5 MM1 tier and decode L2 placement. Both servers use the
same full checkpoint and tokenizer, BF16 weights/activations/KV, TP1, context
8192, four maximum sequences, greedy sampling, ignored EOS and disabled prefix
caching. vLLM is `0.28.0+rocm723`, with compilation/graphs enabled and its
custom tanh-GELU operation verified in the generated graph.

The unmodified installed `vllm bench serve` client measured both HTTP servers:
four requests per run, 128 output tokens, one full-corpus warmup and five measured
repetitions per cell/backend. All 72 runs passed request and actual token-usage
accounting. Each value below is the median of five per-run metrics.

| Input / concurrency | Plow tokens/s / TTFT ms / TPOT ms | vLLM tokens/s / TTFT ms / TPOT ms |
|---|---:|---:|
| 128 / 1 | 35.58 / 94.68 / 27.57 | 56.48 / 48.35 / 17.47 |
| 128 / 4 | 91.16 / 552.55 / 38.99 | 195.91 / 114.38 / 19.67 |
| 1024 / 1 | 31.86 / 339.64 / 28.95 | 52.41 / 149.99 / 18.05 |
| 1024 / 4 | 72.99 / 1757.02 / 40.53 | 158.05 / 558.52 / 21.11 |
| 4096 / 1 | 25.28 / 1428.82 / 28.61 | 42.30 / 606.85 / 19.04 |
| 4096 / 4 | 46.93 / 5552.63 / 41.43 | 100.52 / 2332.10 / 21.74 |

Both token budgets are nonbinding for this corpus: Plow's prefill-only interleave
budget is disabled (`PLOW_PF_INTERLEAVE=0`), while vLLM's combined budget is
16384 tokens. The engines retain their own chunking and scheduling algorithms.
Paired clients run concurrently on separately leased MI300X GPUs; clocks and
power are not pinned, and the host also runs other GPU workloads. These short
closed-loop batches do not establish saturation throughput or tail SLOs.

Exact paired completion text matches in all 20 measured requests per cell except
1024/C1 (10/20) and 1024/C4 (16/20). Token counts remain identical. Cross-engine
text equality is therefore not claimed; fusion's numerical qualification uses
ordinary Plow with matched chunk boundaries separately. Stock ITL measures SSE
event spacing: Plow delivers tokens in multistep bursts and emits an empty final
choice, so its tiny median event interval is not per-token GPU decode latency.

[Current measurements, manifests and raw-result hashes](gemma4-31b-mi300x-bench-serve-20260907.json).
The full raw CLI results and fixed corpus are retained in
`build-gemma31/bench-serve/stock-results.zip`. Earlier tables below describe
different configurations and are historical comparisons.
[GPU and scheduler trace analysis](gemma4-31b-mi300x-trace-20260907.md) locates
the remaining projection/attention gap and serialized prefill completion.

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

- Initialized middle chunks can share a larger compiled prefill rung. Runtime
  fusion additionally initializes cold cursors and batches terminal prompt tokens
  through decode; ordinary fallback and prefix snapshots retain their boundaries.
- Dense BF16 attention uses each request's KV slot and position, including ring
  wraparound and split attention partials. Parked rows do not write request state.
- Ordinary packed-prefill kernel objects must advertise
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
- Runtime fusion executes prefill and one token per active decode request in a
  single launch. With fusion disabled, packed prefill and multistep decode run
  sequentially within a mux tick.

### Runtime fusion

Enable fusion on the same ordinary assets:

Set `PLOW_FUSION=1` in the serving command above, or add `--fusion` to
`plowrt serve`. `--fusion=false` disables it.

Fusion defaults off. The ordinary gfx942 build includes `interp_mixed_gq.elf`.
At model load, the runtime derives one immutable schedule per ordinary prefill
bucket, reusing the packet's attention splits, tensor bindings and numerical
parameters. Private input metadata and attention scratch keep the ordinary
execution path independent. No fusion compiler options or mixed asset metadata
are required. The former `--mixed-rows` and `--mixed-object` options were removed.

The scheduler packs requests automatically. With 500 prefill tokens and two
active decode requests, the 512-row schedule uses two leading decode rows and 500
prefill rows. The runtime supplies token IDs, positions,
decode-slot mappings and prefill spans. No request-side row flags are needed.
The same schedule accepts one, two or three decode rows with four physical slots.
The interpreter derives
active row counts from the spans, runs GEMV on decode rows in bands of up to
four, and GEMM on prefill rows. This keeps each phase's arithmetic consistent
when the pack changes, at the cost of reading projection weights for both phases.
Fused GLU bands also respect the object's activation scratch capacity. Attention uses
request-local KV positions; sampling covers only active decode rows. Inactive
work still completes the fixed dependency schedule. Unsupported batches use the
ordinary path. `PLOW_MULTISTEP` separately controls consecutive decode steps.

The runtime validates every synthesized schedule and the object's capabilities
before enabling fusion. Shared validation and staging serve AMD and NVIDIA;
runtime schedule synthesis and dynamic interpreter dispatch currently target AMD.

Mixed serving requires dense BF16, one GPU and a compatible
object. Context and continuation bounds are checked before admission; prefix
cache and unsupported configurations use ordinary execution. An unsupported
schedule or object produces a startup warning and leaves ordinary execution
available. The qualified
Gemma 4 31B profile uses capacities 64/128/256/512/1024/2048 and up to three decode requests
alongside prefill, with four physical request slots.
An eight-slot profile also passed capacities 128/512/1024 with one through seven
decode rows, including reordered slots and decode bands crossing row four.

Numerical prefill checks replay identical chunk boundaries and row buckets in
the ordinary path. Whole-prompt prefill can produce different BF16 logits even
without fusion; those differences are reported separately, with greedy tokens
still compared. Request isolation is checked bitwise in both directions.

## Verification

494 CPU/CUDA/HSA library tests and eight API tests passed together.
Runtime-only fusion passed full-model capacity and multispan GPU suites on
ordinary assets, without mixed metadata. A separate fusion-disabled process
verified rejection before launch or state mutation and exact ordinary recovery.
Every tested decode logit was bitwise equal to ordinary execution. Matched
prefill at capacities 256/512/1024/2048 and all multispan cases was bitwise equal;
capacities 64/128 passed relative L2 <=1% and maximum absolute logit error <=0.5.
The expanded profile exercised the actual 4096-row sliding KV ring. Both request
isolation directions were bitwise exact. The tracked attention primitive has ten
cases: corrected arithmetic passes and the old four-wave arithmetic fails.
Runtime-only mux and HTTP checks passed with observed fused launches, including
cancellation, output limits, slot reuse and context rejection/recovery.
An additional 305.7-second serving exercise and near-context checks completed
298 requests, including 284 exact comparisons with isolated greedy baselines,
and 96 deliberate disconnects without errors or output/usage mismatches.
Near-context cases reached exactly 8192 total tokens with output limits
1/17/64/128; recovery and concurrent requests passed. Logs recorded 261 additional
mixed launches during the exercise. Artifacts are in
`build-gemma31/runtime-http-qualification/`. This is a bounded exercise, not a
long-running production soak.

The normal CMake mixed GQ object SHA256 is
`b1deafe6490f3f57c18ba47a803785177a260884a7ba91d2f2dc952782bc86a5`.
It uses 256 threads, 64,544 bytes LDS and 468 VGPRs, with no VGPR spills;
the resource report lists 1,596 private bytes per thread. Numerical qualification
does not establish a performance win over vLLM or ordinary execution.

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

## Earlier prototype cost

This historical measurement used the retired compiler-emitted capacity prototype,
not the runtime-only schedule builder. Five paired repetitions, one warmup per
case, four concurrent requests and 128
output tokens compare ordinary Plow with dynamic fusion. Both servers use the
same runtime binary, checkpoint, precision, decode objects, 512-token prefill
chunks, multistep 4 and disabled prefix caching. Engine order alternates between
repetitions. All paired completion texts match.

| Input tokens | Output tokens/s, ordinary → fusion | TTFT ms, ordinary → fusion | TPOT ms, ordinary → fusion |
|---|---:|---:|---:|
| 1024 | 78.83 → 74.88 | 1163 → 1697 | 41.07 → 39.59 |
| 4096 | 49.40 → 47.71 | 4866 → 5435 | 42.32 → 40.80 |

Fusion reduces median TPOT by about 3.6%, but throughput falls 3–5% and TTFT
increases. It remains opt-in. The phase-specific projection paths preserve
arithmetic while reading weights for both phases; the universal interpreter also
has different resource and matrix-tile choices from ordinary execution. This
measurement does not isolate those costs. GPUs were separately leased and clocks
were not pinned. This is a Plow comparison, not a new vLLM result.

[Measurements and provenance](gemma4-31b-mi300x-dynamic-20260907.json).

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

For new comparisons, use `vllm bench serve` against both HTTP endpoints after
the runtime fusion qualification passes.
Record checkpoint and tokenizer hashes, weight/activation/KV precision, TP,
GPU identity, toolchain, maximum context and sequences, token budgets, prefix
caching and backend flags. Use the same input/output/concurrency matrix and
warmups. Save the benchmark JSON and exact command/environment alongside the
manifest. Verify server usage counts against the benchmark's requested lengths.
Text chunk gaps are not
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
