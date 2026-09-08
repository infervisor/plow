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

Measurements, phase-matched logits and policy qualification (raw artefact removed; see the tables above).

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

Current measurements, manifests and raw-result hashes (raw artefact removed; see the tables above).
The full raw CLI results and fixed corpus are retained in
`build-gemma31/bench-serve/stock-results.zip`. Earlier tables below describe
different configurations and are historical comparisons.
The GPU and scheduler trace analysis below locates
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

- Initialized middle chunks can share a larger compiled prefill rung. `--pf-batch`
  initializes cold cursors up front — but only when a pack could actually form, i.e.
  when two chunks fit one rung — so a burst of arrivals can co-pack on its first tick.
  Runtime fusion initializes them unconditionally and additionally batches terminal
  prompt tokens through decode. Ordinary fallback and prefix snapshots retain their
  boundaries.
- Co-packing needs at least two chunks to fit in one compiled prefill rung, so a
  `PLOW_PF_CHUNK` equal to the widest rung disables it silently. The engine logs
  `packed prefill routing ... capable_rungs=[...]` at load; a nonzero count there
  with no `AMD packed prefill advanced` in the log means the chunk, not the objects.
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

Measurements and provenance (raw artefact removed; see the tables above).

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

Raw measurements and provenance (raw artefact removed; see the tables above).

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

Measurements and manifests (raw artefact removed; see the tables above).

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

## Runtime fusion on/off, current schedule builder (2026-09-07)

The earlier fusion cost table above measured the retired compiler-emitted capacity
prototype. This one measures `PLOW_FUSION=1` against the same server binary,
assets and objects with fusion off, on separately leased single GPUs, three
repetitions per cell after one warmup, medians reported. The server logs
`runtime prefill/decode fusion enabled` in the fusion arm, so the mixed step is
live rather than silently refused.

| Input / concurrency | Ordinary tokens/s / TTFT ms / TPOT ms | Fusion tokens/s / TTFT ms / TPOT ms |
|---|---:|---:|
| 128 / 1 | 30.81 / 93.0 / 31.50 | 30.38 / 92.7 / 31.97 |
| 128 / 4 | 87.26 / 341.9 / 39.52 | 99.24 / 202.4 / 37.72 |
| 1024 / 1 | 27.16 / 323.6 / 32.26 | 26.84 / 325.7 / 32.66 |
| 1024 / 4 | 65.29 / 1153.4 / 42.31 | 68.27 / 1075.6 / 41.83 |
| 4096 / 1 | 18.52 / 1399.0 / 32.63 | 18.34 / 1409.7 / 33.00 |
| 4096 / 4 | 33.04 / 4865.6 / 44.05 | 33.81 / 4807.1 / 43.26 |

At concurrency 1 there is nothing to fuse and the three cells move by −0.4% to
−1.4% on throughput, which is within this harness's run-to-run spread. At
concurrency 4 fusion wins every cell: +13.7%, +4.6% and +2.3% on throughput at
inputs 128, 1024 and 4096, with TTFT down 40.8%, 6.7% and 1.2% and TPOT down
4.6%, 1.1% and 1.8%. The gain shrinks as the prompt grows because a longer
prefill occupies more of the step and leaves proportionally less decode to ride
along with it. The retired prototype's regression therefore does not describe
the current builder, and the terminal-prefill fix is what changed the TTFT sign.

Fusion remains unavailable outside single-GPU dense BF16: the engine gate is
`fusion && tp.is_none() && batch > 1`, and program synthesis additionally
requires ordinary `FlashDecode`/`FlashPrefill` attention, so every MLA, MoE and
FP8 blob — GLM-5.3 included — falls back to ordinary execution. On the TP path
that fallback is silent; only the synthesis-failure path logs a warning.

## Full context sweep, and the prefill chunk that pays for it (2026-09-07)

The comparison above stops at 4096 input tokens because the shipping blob was
emitted with `--max-ctx 8192`. Two new blobs at `--max-ctx 131072` extend it:
`build-gemma31/assets-ctx131072` at the emitter's default prefill chunk and
`build-gemma31/assets-ctx131072-chunk8192` at `PLOW_MAX_CHUNK=8192`. Both carry
`tile_source: measured` for the shapes the tile store covers. vLLM 0.28 serves
the same checkpoint at `--max-model-len 131072`, `--max-num-batched-tokens 8192`.
Same client, same corpus, three repetitions after one warmup, medians reported,
each engine on its own leased GPU.

Two corpus limits are structural rather than chosen: a prompt of exactly
`max_ctx` leaves no room for the 64 output tokens, so the longest cell is
122880; and concurrency 4 at 32768 needs the whole 131072-slot KV cache, so the
batched column stops at 16384.

Each cell is **output tokens/s / median TTFT ms / median TPOT ms**.

| Input / concurrency | Plow, default chunk | Plow, chunk 8192 | vLLM 0.28 |
|---|---:|---:|---:|
| 128 / 1 | 30.79 / 91.8 / 31.53 | 30.51 / 92.1 / 31.83 | 55.31 / 45.9 / 17.63 |
| 128 / 4 | 85.95 / 341.7 / 40.20 | 86.41 / 332.5 / 39.96 | 180.39 / 109.2 / 20.79 |
| 512 / 1 | 29.63 / 156.4 / 31.80 | 29.38 / 157.0 / 32.09 | 52.32 / 81.9 / 18.11 |
| 512 / 4 | 78.31 / 506.3 / 42.18 | 77.43 / 558.7 / 41.88 | 158.74 / 270.7 / 21.29 |
| 2048 / 1 | 23.81 / 670.8 / 32.02 | 25.70 / 456.5 / 32.27 | 44.39 / 265.7 / 18.66 |
| 2048 / 4 | 49.46 / 2345.6 / 43.25 | 57.46 / 1224.3 / 49.48 | 103.95 / 1050.8 / 22.40 |
| 8192 / 1 | 12.75 / 2974.0 / 32.44 | 16.85 / 1741.0 / 32.65 | 25.58 / 1218.9 / 20.36 |
| 8192 / 4 | 18.92 / 10553.0 / 45.42 | 22.33 / 7930.9 / 54.25 | 39.86 / 4254.3 / 33.92 |
| 16384 / 4 | — | 12.03 / 17380.6 / 60.13 | 19.30 / 8477.3 / 74.90 |
| 32768 / 1 | 3.45 / 16404.4 / 34.05 | 4.99 / 10673.5 / 34.09 | 6.87 / 7827.6 / 23.61 |
| 122880 / 1 | 0.34 / 183744.8 / 116.08 | 0.68 / 91742.1 / 39.48 | 0.89 / 69744.7 / 28.52 |

### The chunk is worth 1.5-2x of TTFT above 2048 tokens

`default_chunk` derives the prefill chunk from the model's attention window, so
Gemma-4's 1024-token window sets a 1024-token chunk and a prefill bucket ladder
of 128/512/1024. That is a deliberate memory trade documented at the emitter:
the sliding-layer KV ring is `window + chunk - 1` rounded up, so chunk 1024
rings 2048 rows where chunk 8192 rings 16384 — 0.625 against 5.0 GiB per
sequence. What the comment predicts but does not price is the other side:
a 122880-token prompt is then 120 chunk launches, each re-reading the KV
accumulated so far.

Priced here, on a 192 GiB card where the memory is available, the launches cost
far more than the ring: TTFT improves 1.47x at 2048, 1.71x at 8192, 1.54x at
32768 and 2.00x at 122880, and throughput follows (+8% to +100%). Below 2048
the two blobs are indistinguishable, as they must be — a 512-token prompt is
one chunk either way. The blob pays 90.00 GiB of KV cache against 46.25.

The emitter's default is not wrong for a memory-constrained deployment, and it
is not changed here. For long-prompt serving on this class of card, emit with
`PLOW_MAX_CHUNK=8192`. The narrower fix worth investigating is decoupling the
two: the ring only has to cover the chunk on **sliding** layers, while the
launch count is set globally, so a per-layer-kind chunk would take the TTFT win
without the ring cost.

### Standing against vLLM

Plow closes with context but still trails on every cell. At the chunk-8192
blob, plow's throughput is 0.48-0.55x of vLLM below 2048 tokens, 0.66x at 8192,
0.73x at 32768 and 0.76x at 122880; TTFT tracks it. The one metric plow already
wins is TPOT at 16384/4 (60.13 against 74.90 ms), where vLLM's batched decode
degrades faster than plow's. Since the decode trace attributes 74-83% of the
gap to projections rather than attention, the remaining distance is a
projection-kernel problem at short contexts and a prefill problem at long ones.
## FP8 on one MI300X: full context sweep, both engines (2026-09-07)

The FP8 paragraph above said Plow's FP8 prefill and decode use different
activation precisions and must not be compared as equivalent to vLLM W8A8. That
is still true, and this section measures it rather than deferring it.

### What each engine's FP8 actually is

Plow's blob, emitted with `PLOW_FP8=1 PLOW_W8A8=1`, reports:

```json
"precision": { "weight_enc": "fp8", "act_enc": "fp8", "kv_enc": "bf16", "expert_enc": "none" }
```

`act_enc` is derived from the presence of a `QuantFp8` op anywhere in the
stream, and the axis is not phase-aware. `QuantFp8` appears only in the prefill
programs. The decode program's `GemvFp8` / `GemvGluFp8` arms take **BF16**
activations and convert the weight on load, one byte per element. The honest
label is therefore **W8A8 prefill, W8A16 decode, BF16 KV, BF16 `lm_head`** — the
`precision` triple alone overstates it, and the numbers below should be read
against the honest label, not the triple.

vLLM's FP8 is W8A8 in both phases: per-output-channel weight scale, dynamic
per-token activation scale, BF16 KV, unquantized `lm_head`. Two arms produce it,
and both were measured:

* `--quantization fp8_per_channel` → `Fp8PtpcOnlineLinearMethod`, quantizing the
  BF16 checkpoint in process. **This is the reference arm** in the tables below.
* `--model build-gemma31/fp8-vllm-ckpt` → `quantization=compressed-tensors`,
  `Selected RowWiseTorchFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8`,
  30.41 GiB of weights.

Both arms select the same kernel and land within 1-2% of each other on every
metric of the three cells measured on both (128/1, 2048/1, 8192/1), and their
greedy completions agree on 75.4% of tokens — the same quantization reached two
ways. The checkpoint directory carries the `fp8-ptpc-export` bytes under the
HuggingFace tensor spelling; its 27.3 GiB data region is a byte-for-byte copy of
the export, nothing is requantized, and `scripts/gemma4_fp8_vllm_ckpt.py`
rebuilds it.

Plow serves the same export through `PLOW_FP8_DIR`, so both engines run the same
FP8 weights: 410 projections, `max(row amax/448, 1/(448*512))` per output
channel, verified row-by-row against vLLM's own `_fp8_quant_per_channel`.

### The sweep

Outputs 64, three repetitions after one full warmup, medians, separate
single-GPU leases, `scripts/bench_packed_serve.py` (which asserts the served
`prompt_tokens` against the requested length, so the corpus is exact). Inputs
reach 122880 rather than 131072 because 64 output tokens must fit inside
`max_ctx`; concurrency 4 stops at 16384 because four 32768-token prompts need the
whole KV cache of the 131072 blob. Each cell is **output tokens/s / median TTFT
ms / median TPOT ms**. Ratios are Plow / vLLM for throughput and vLLM / Plow for
the latencies, so below 1.00 always means Plow loses.

| Input / concurrency | Plow FP8 | vLLM 0.28 FP8 | tok/s | TTFT | TPOT |
|---|---:|---:|---:|---:|---:|
| 128 / 1 | 36.34 / 95.2 / 26.44 | 66.19 / 39.9 / 14.70 | 0.55x | 0.42x | 0.56x |
| 512 / 1 | 35.06 / 146.7 / 26.65 | 64.39 / 54.7 / 14.90 | 0.54x | 0.37x | 0.56x |
| 2048 / 1 | 29.36 / 488.2 / 26.85 | 55.20 / 187.4 / 15.43 | 0.53x | 0.38x | 0.57x |
| 8192 / 1 | 16.31 / 2204.8 / 27.27 | 31.37 / 941.4 / 17.44 | 0.52x | 0.43x | 0.64x |
| 32768 / 1 | 4.37 / 12832.6 / 29.00 | 7.97 / 6680.2 / 21.09 | 0.55x | 0.52x | 0.73x |
| 122880 / 1 | 0.60 / 103563.8 / 35.46 | 0.95 / 65893.8 / 26.17 | 0.64x | 0.64x | 0.74x |
| 128 / 4 | 90.01 / 325.2 / 38.26 | 247.10 / 77.3 / 15.21 | 0.36x | 0.24x | 0.40x |
| 512 / 4 | 84.51 / 430.6 / 39.55 | 220.63 / 168.5 / 15.73 | 0.38x | 0.39x | 0.40x |
| 2048 / 4 | 57.26 / 1656.6 / 42.95 | 145.62 / 707.3 / 16.67 | 0.39x | 0.43x | 0.39x |
| 8192 / 4 | 22.30 / 8457.4 / 46.24 | 51.41 / 3290.1 / 26.37 | 0.43x | 0.39x | 0.57x |
| 16384 / 4 | 11.04 / 19885.9 / 50.62 | 23.71 / 6873.0 / 61.42 | 0.47x | 0.35x | **1.21x** |

vLLM leads every cell but one. The single exception is TPOT at 16384/4, where
vLLM's per-token latency rises to 61.42 ms against Plow's 50.62 ms; vLLM still
wins that cell on throughput and TTFT, so this is one metric in one cell, not a
won workload.

### FP8 against each engine's own BF16

Same host, same day, same client and corpus. The BF16 columns are the matched
BF16 context sweep; the FP8 columns are the table above.

| Input / concurrency | Plow tok/s BF16 → FP8 | vLLM tok/s BF16 → FP8 | Plow TPOT | vLLM TPOT |
|---|---:|---:|---:|---:|
| 128 / 1 | 30.79 → 36.34 (+18%) | 55.31 → 66.19 (+20%) | 31.53 → 26.44 (−16%) | 17.63 → 14.70 (−17%) |
| 512 / 1 | 29.63 → 35.06 (+18%) | 52.32 → 64.39 (+23%) | 31.80 → 26.65 (−16%) | 18.11 → 14.90 (−18%) |
| 2048 / 1 | 23.81 → 29.36 (+23%) | 44.39 → 55.20 (+24%) | 32.02 → 26.85 (−16%) | 18.66 → 15.43 (−17%) |
| 8192 / 1 | 12.75 → 16.31 (+28%) | 25.58 → 31.37 (+23%) | 32.44 → 27.27 (−16%) | 20.36 → 17.44 (−14%) |
| 32768 / 1 | 3.45 → 4.37 (+27%) | 6.87 → 7.97 (+16%) | 34.05 → 29.00 (−15%) | 23.61 → 21.09 (−11%) |
| 128 / 4 | 85.95 → 90.01 (+5%) | 180.39 → 247.10 (+37%) | 40.20 → 38.26 (−5%) | 20.79 → 15.21 (−27%) |
| 512 / 4 | 78.31 → 84.51 (+8%) | 158.74 → 220.63 (+39%) | 42.18 → 39.55 (−6%) | 21.29 → 15.73 (−26%) |
| 2048 / 4 | 49.46 → 57.26 (+16%) | 103.95 → 145.62 (+40%) | 43.25 → 42.95 (−1%) | 22.40 → 16.67 (−26%) |
| 8192 / 4 | 18.92 → 22.30 (+18%) | 39.86 → 51.41 (+29%) | 45.42 → 46.24 (+2%) | 33.92 → 26.37 (−22%) |

**FP8 does not close the gap; at concurrency 4 it widens it.** At concurrency 1
the two engines gain almost identically and the throughput ratio is unchanged
(BF16 0.50-0.57x, FP8 0.52-0.55x). At concurrency 4 vLLM gains 29-40% while Plow
gains 5-18%, so the ratio falls from a BF16 0.475-0.493x to an FP8 0.36-0.43x.
Plow's batched decode barely responds to halving the weight stream, which is the
signature of a step whose cost is not the weight stream.

### Where the FP8 decode time goes

Decode weight bytes per step, summed from `plowrt disasm <assets> --program 1`:

| | weight bytes/step | decode ops/step |
|---|---:|---:|
| BF16 blob | 61.392 GB | 251 |
| FP8 blob | 32.105 GB | 351 |

FP8 moves 52.3% of the bytes and issues 40% more operations. The extra 100
operations are the QKV projections: the BF16 emit fuses q/k/v into one `GemvQkv`
on the 50 non-`k_eq_v` layers, and the FP8 emit has no fused twin turned on
(`PLOW_FUSE_QKV_FP8` defaults off, and `devgen/src/lib.rs` records that fusing
measured *slower* on the dynamic scheduler — a known trade, not an oversight).
`lm_head` stays BF16 in both engines and is 2.819 GB, 8.8% of the FP8 step.

Dividing those bytes by the measured TPOT gives the effective weight-stream rate:

| | BF16 | FP8 |
|---|---:|---:|
| Plow, concurrency 1 | 1947 GB/s | 1214 GB/s |
| vLLM, concurrency 1 | 3482 GB/s | 2184 GB/s |
| Plow, concurrency 4 | 1527 GB/s | 839 GB/s |
| vLLM, concurrency 4 | 2953 GB/s | 2111 GB/s |

At concurrency 1 both engines lose the same fraction of effective bandwidth when
the bytes halve (Plow −38%, vLLM −37%): the per-step costs that do not shrink
with the weights are proportionally larger in a shorter step, and that is not a
Plow-specific FP8 defect. At concurrency 4 they diverge — vLLM holds 2111 GB/s
while Plow falls to 839 GB/s. **Plow's batched FP8 decode is not
weight-bandwidth-bound at all**, so no further weight-encoding work can move it;
the cost is in the step, not in the stream. That is where the batched decode gap
now lives, and it is the same conclusion the BF16 trace reached from the other
side.

### Quality

FP8 changes the arithmetic, so every speed number above needs its quality number
beside it. Twelve prompts (three each at 128 / 512 / 2048 / 8192 tokens of
context) go through each server's own chat template at temperature 0 for 64
tokens; the two captures are re-tokenized and compared. Text is the only common
surface, because plowrt refuses `logprobs` rather than returning null.

| Pair | token agreement | identical completions | first divergence, median over lengths |
|---|---:|---:|---:|
| Plow FP8 vs Plow BF16 | 0.520 | 2/12 | token 2-29 |
| vLLM FP8 (online PTPC) vs vLLM BF16 | 0.529 | 3/12 | token 3-29 |
| vLLM FP8 (checkpoint) vs vLLM BF16 | 0.550 | 2/12 | token 3-29 |
| vLLM FP8 (checkpoint) vs vLLM FP8 (online) | 0.754 | 5/12 | token 22-40 |

**Plow's FP8 costs the same greedy agreement vLLM's FP8 costs** — 0.520 against
0.529 — so the speed comparison above is between two arms that have paid the
same quality price. Greedy trajectories separate early in both engines because a
64-token continuation only has to cross one near-tied logit to diverge for good;
the last row is the control that says how much of that is FP8 rather than the
route taken to it.

A first attempt at this measurement fed raw markdown to `/v1/completions` with
`add_special_tokens=false` — what the throughput harness sends. With no BOS and
no turn structure this instruction-tuned checkpoint collapses into single-token
repetition within a few tokens, and the agreement number then reports which
degenerate attractor each arm fell into. Those captures are retained but are not
reported as quality.

### Two defects found and fixed

**The assembled checkpoint's scale tensor was misnamed.** The twin export keys a
scale as `fp8/<module>.weight_scale`, which is already the HuggingFace spelling
once the prefix comes off; the assembler appended a second suffix and produced
`<module>.weight.weight_scale`. vLLM skipped the unrecognised name in silence,
left `weight_scale` at its uninitialised `torch.empty` contents, and served a
model that answered all twelve prompts with one repeated token — 0.000 token
agreement against its own BF16 — while logging a clean startup. Scales are now
written under the right name with vLLM's `ChannelQuantScaleParameter` shape
`(out_features, 1)`. The bytes never changed.

**The FP8 blob could not be served against the shipped objects at all.** With
`PLOW_L2_PLACE_PREFILL` at its default `true`, plowrt refuses
`interp_prefill_fp8_gq.elf`: `build_gfx942.sh` gates `-DPLOW_L2_PLACE_DISPATCH`
on `PLOW_L2HIER_PF`, which is off, so the prefill objects would read a placed
program's `seg` as a wave class. The BF16 blobs in this tree were emitted with
`PLOW_L2_PLACE_PREFILL=0` and never hit it. The FP8 emit now matches them, and
decode queues stay placed in both.

### The W8A8 tile campaign, and what it did not fix

The FP8 blob took 180 of its 1170 dense-GEMM tiles from measurement and the
analytical model for the other 990, while the BF16 blob of the same model is
fully measured. The cause was that `plowc tune gemm` could not measure W8A8 at
all: the host sweep refused with `unknown quant W8A8 (want None or Mxfp4)`.

`runtime/bench/gemm/gemm_tile_sweep.c` now has a W8A8 arm — e4m3 bytes on both
operands, an f32 per-row activation scale and an f32 per-channel weight scale,
with an f64 OCP-e4m3 oracle that applies both scale rows once, matching the
kernel epilogue — and `gemm_fp8_c2` (128x256) joins the golden wrappers so all
five rungs `tunedb::RUNGS` maps to `Gemm*Fp8` exist. The campaign published 125
qualified records for the 22 W8A8 shapes this model demands, every spot-check
passing, and store coverage went from 3/25 to 25/25 HIT.

Selection still does not use them. `build.json` reports `tile_measured: 180` and
`tile_source: mixed` after the campaign, and the re-emitted `model.pkt` is
**byte-identical** (sha256 `6c540a33528a9d72…`) to the one built before it, so
the refusal is downstream of the store lookup — in `kernelcaps::select_kernel`
or in the gfx942 dense-GEMM inventory's W8A8 candidate set, not in the store.
The records now exist to test that against; no FP8 prefill number in this
section benefits from them.

Separately, `devgen`'s `tuned_tile_selection` suite fails five tests on GLM-5.2's
own shapes on this branch. It fails them identically with the campaign reverted:
pre-existing, and not caused by this work.

Measurements, manifests, precision triples and raw-result hashes (raw artefact removed; see the tables above).
## Batch-width-matched decode objects (2026-09-07)

The projection deficit at concurrency 1 was not a kernel defect. `PLOW_GEMV_MM` is a
COMPILED ceiling and one instantiation serves every `M <= MM` by predicating each
activation row with `live = (m < M)` and then computing its dot product anyway. The
shipping decode object for this blob is `PLOW_GEMV_MM=4` — `plow_gemv_mm_cap_4` in
`interp_decode_gq.elf` — while the blob carries the T=1, T=2 and T=4 decode programs,
so every batch-1 token ran four dot chains per weight chunk and discarded three. On
gfx942, which has no `v_dot2c_f32_bf16`, each discarded row is 24 VALU operations per
16 bytes of weight, and the unroll is cut with it: `GV_UNROLL_M4` is 6 against
`GV_UNROLL`'s 11.

### The projection primitive, per shape and per bucket

`runtime/bench/amd/gemma31_gemv_decode_bench.{hip,cpp}` includes `gemv_rows` verbatim
from `op_gemm.h`, mirrors the interpreter's launch geometry (512 threads; the grid is
the packet's own `b`, from `plowrt disasm --program 4`), reproduces `d_gemv_t`'s
staged/unstaged decision and packet framing, walks a fresh weight slab out of a 3 GiB
arena so the number is a stream, and takes a hipEvent median of 41 with a palindromic
interleave and a byte-identical A/A control arm. Microseconds per packet:

| shape | N × K | nblk | T=4 packets | MM=1 M=1 | MM=2 M=2 | MM=4 M=1 | MM=4 M=2 | MM=4 M=4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| q_proj | 8192 × 5376 | 152 | 60 | 22.73 | 28.47 | 41.49 | 41.90 | 42.85 |
| k_proj, v_proj | 4096 × 5376 | 76 | 120 | 14.93 | 20.28 | 34.75 | 35.22 | 36.29 |
| o_proj sliding | 5376 × 8192 | 304 | 50 | 21.23 | 23.66 | 28.27 | 29.33 | 32.09 |
| o_proj full | 5376 × 16384 | 304 | 10 | 43.25 | 45.74 | 55.17 | 59.26 | 61.94 |
| gate, up | 21504 × 5376 | 152 | 120 | 57.12 | 73.15 | 102.13 | 108.01 | 104.64 |
| down | 5376 × 21504 | 304 | 60 | 55.63 | 59.69 | 67.82 | 72.72 | 77.53 |
| lm_head | 262144 × 5376 | 304 | 1 | 678.9 | 713.7 | 914.3 | 926.8 | 934.2 |
| **per-token total, ms** | | | | **15.52** | **18.86** | **25.86** | **27.05** | **27.29** |

The MM=1 body reaches 3875–4156 GB/s of weight stream on the six shapes whose grid
fills the machine, which is 73–78% of this part's 5.3 TB/s peak and the same order as
the 3.8–4.2 TB/s implied by vLLM's own decode projection time. k_proj and v_proj read
2950 GB/s measured ALONE, and that is a fill artifact rather than a body property:
each dispatches 76 of 304 workgroups, and in the program they run concurrently with
q_proj's 152. The MM=4 body reaches 1214–3017 GB/s. A batch-1 packet on
the MM=4 object therefore pays 1.67x for arithmetic it discards, and that single fact
accounts for the concurrency-1 projection gap this campaign set out to explain. These
are isolated body timings: no packet gates, model, sampling or serving costs. The A/A
control read 0.994–1.026 across two independent runs, so per-shape ratios below 3% are
inside this harness's noise.

MM=8 is measured (61.87 ms per token, 460–1639 GB/s) but is not a bucket this blob
emits; it is in the sweep so the M axis is complete, not because anything dispatches it.

### What landed

1. **A decode tier per rung.** `scripts/build_gfx942.sh` gains `PLOW_DECODE_TIER=<n>`
   (build the decode rows for a low-rung object that will only ever be handed packets
   of at most n rows) and `PLOW_DECODE_TIERS=1,2` (re-enter once per width and fill
   `$OUT/lowrung1`, `$OUT/lowrung2`). `scripts/glm53_serve_inner.sh` hands them to
   plowrt's existing `PLOW_HSACO_LOWRUNG` co-load automatically when the
   subdirectories exist, so the mechanism that was previously an opt-in experiment is
   now the default shape of a build. An explicit `PLOW_HSACO_LOWRUNG` still wins.
2. **The fused projections skip the general dispatch prologue.** `plow_exec` saves 112
   VGPR dwords per lane before it selects an opcode; op 10 `Gemv` already bypassed it
   in `interp.hip`, and ops 20 `GemvGlu` and 22 `GemvQkv` now do too. That matters
   exactly for the low rungs: `plowrt disasm --program 1` emits GemvQkv and GemvGlu
   where the T=4 program emits five plain Gemvs, so the narrow programs were still
   paying the prologue on most of their weight bytes. **MM=1 only** — inlining those
   two helpers a second time takes the MM=2 object from spill 6 to spill 117, and the
   wide object never sees either opcode once the tiers exist.
3. **The R-split on the wide bucket's staged shapes.** At MM=4 the unroll is capped by
   registers, not by the row: UN=8 and UN=11 measure 0.17–0.58x because they spill
   (119 `scratch_load_dword` in the UN=8 body). `gemv_rows_rs` buys the same in-flight
   loads with R accumulators instead of R×UN weight vectors. The sign flips exactly on
   `d_gemv_t`'s staged/unstaged line, which is why that is the whole predicate:

   | staged (M·K ≤ GM_LDS_HALVES) | vs base | unstaged (x re-read from global) | vs base |
   |---|---:|---|---:|
   | gate, up 21504×5376 | 1.052x | o_proj sliding 5376×8192 | 0.750x |
   | q_proj 8192×5376 | 1.051x | o_proj full 5376×16384 | 0.795x |
   | k/v_proj 4096×5376 | 1.029x | down 5376×21504 | 0.823x |
   | lm_head 262144×5376 | 1.108x | | |

   Weighted by the T=4 program's own packet counts that is −0.96 ms of the 27.29 ms
   per-token plain-GEMV projection total. `interp_decode_gq` stays at 256 VGPR /
   spill 2.

All three are arithmetic-preserving by construction. Compiling a narrower bucket,
batching a different number of loads and moving which wave takes which column each
leave every output column's chunk order, lane→k map, `dot8(w, x, 0.0f)` pair nesting,
per-chunk accumulation and `wave_sum` exactly as they were.

### Served, before and after

One MI300X per run through `perf-data/tools/gpulease`, the same Gemma-4 31B BF16
checkpoint, assets and `plowrt` binary, `scripts/bench_packed_serve.py` at inputs
128/1024/4096, 64 output tokens, concurrency 1 and 4, one warmup and three measured
repetitions, medians reported. Control is the same 45-object build with no tier
directories; candidate is that build plus `lowrung1`/`lowrung2` and the two kernel
changes above.

| Input / concurrency | Control tokens/s / TTFT ms / TPOT ms | Candidate tokens/s / TTFT ms / TPOT ms | Throughput | TPOT |
|---|---:|---:|---:|---:|
| 128 / 1 | 29.88 / 93.9 / 32.50 | 41.09 / 93.5 / 23.23 | +37.5% | −28.5% |
| 128 / 4 | 85.36 / 339.1 / 40.37 | 88.50 / 317.5 / 39.10 | +3.7% | −3.1% |
| 1024 / 1 | 27.19 / 266.5 / 33.11 | 36.02 / 265.8 / 23.98 | +32.5% | −27.6% |
| 1024 / 4 | 68.38 / 764.3 / 45.44 | 70.43 / 742.4 / 44.09 | +3.0% | −3.0% |
| 4096 / 1 | 19.55 / 1162.0 / 33.50 | 23.85 / 1154.7 / 24.26 | +22.0% | −27.6% |
| 4096 / 4 | 31.76 / 3345.5 / 67.55 | 32.45 / 3296.1 / 65.90 | +2.2% | −2.4% |

No cell regresses on any metric. Median TTFT moves −0.3% to −6.4%. The intermediate
arms separate the three changes: the tiers alone are +17.6 / +15.8 / +11.0% at
concurrency 1 and neutral at 4; the fused-projection direct dispatch adds a further
+17.7 / +15.3 / +10.3% at concurrency 1; the R-split adds +3.3 / +1.6 / +1.0% at
concurrency 4 while leaving concurrency 1 inside its own ±0.8% run-to-run band.

### Identity

Every arm is checked twice. All 45 measured completions are CHARACTER-IDENTICAL to the
control in each served comparison (135 of 135 across the three). Full-vocabulary logit
rows — `plowrt amd-bench --dump-logits`, contexts 128, 1024, 4096 and 8160, 24 decode
steps plus the prefill row at each, 100 rows per arm — are BITWISE identical to the
control both for the wide object carrying the R-split and for the tiered configuration,
and the ragged four-slot batched runs at 128 and 4096 produce identical sampled chains.
The primitive harness additionally compares every arm elementwise on device against the
shipping body across all seven shapes and four buckets: 0 mismatches.

### What was measured and did not pay

* **A wave-uniform M ladder inside `gemv_rows`** (an arm in the bench, not landed).
  Branching on the runtime row count so an MM=4 object stops computing dead rows
  recovers most of the tier's win without a second object: the MM=4 per-token total
  falls 25.86 → 17.62 ms at M=1 and 27.05 → 20.10 ms at M=2. It costs 1.9% at M=4,
  where the ladder is pure code growth, and it cannot reach the 15.52 ms the matched
  bucket gets because the unroll is still 6. With the tier built it is redundant, so it
  stays in the harness as the measured alternative rather than in the kernel.
* **K-tiled activation staging for the unstaged shapes.** `d_gemv_t` stages x only when
  `M·K ≤ GM_LDS_HALVES`, which fails for down (M≥2), o_proj full (M≥2) and o_proj
  sliding (M=4 — 32,768 halves against a 32,256 budget), so those read x from global
  once per output column and k-chunk: four extra 16-byte loads per weight load at M=4.
  Walking K in LDS-sized tiles with one accumulator per (column, row) removes that and
  is bit-exact (the tile boundary falls on a chunk boundary), and it is 0.79–0.89x at
  MM=4 — the per-tile barriers and the held accumulators cost more than the global
  re-reads. Its first version was also WRONG, and the bug is worth stating because the
  single-tile body invites it: past the end of a tile the weight row still holds real
  data, so the k overshoot has to be steered past `num_records` explicitly instead of
  being left to the row-length bounds check `gemv_rows` relies on.
* **Larger unrolls at MM=4.** UN=8 and UN=11 are 0.17–0.58x on every shape. They report
  `.vgpr_spill_count: 0` and spill anyway; the ISA is the authority (119
  `scratch_load_dword` / 108 `scratch_store_dword` in the UN=8 body).
* **R=4 at MM=4.** Best on gate/up (1.076x) and lm_head (1.151x), worst on q_proj
  (0.938x) and k/v_proj (0.887x); R=2 wins the instance-weighted total.

### An arithmetic-preserving MFMA skinny-GEMM is not constructible for this body

vLLM's `wvSplitK_hf_sml` accumulates with `__builtin_amdgcn_mfma_f32_4x4x4bf16_1k`.
plow's value for one output column is `wave_sum` over 64 lanes of a per-lane chain of
`dot8(w, x, 0.0f)` results, and `dot8` is four nested `plow_dot2_bf16` calls, each an
`fma(a0, b0, fma(a1, b1, acc))`. Every MFMA variant this part has reduces at least four
products in hardware order inside one instruction, so no arrangement of them reproduces
that nesting; the only bit-exact use of MFMA here would be to change plow's reference
order, which is a different arithmetic and needs full-model requalification rather than
an identity gate. That is consistent with the earlier combined gate/up MFMA candidate,
which reached 3.4891% full-model relative L2 and was rejected. It is also not where the
remaining batch-4 gap is: that candidate improved its primitive 166.7 → 132.3 µs,
−20.6%, against a batch-4 projection deficit of about 71%. Numerics are not what stands
between plow and vLLM at batch 4.

### Remaining distance

Against the vLLM 0.28 scorecard for the same checkpoint and corpus (55.63 / 49.65 /
36.02 tokens/s at concurrency 1 and 186.41 / 136.18 / 69.77 at concurrency 4), the
candidate is at 0.74 / 0.73 / 0.66 of vLLM at concurrency 1 and 0.47 / 0.52 / 0.47 at
concurrency 4; median TPOT is 1.26–1.32x vLLM's at concurrency 1 and 1.94–2.13x at
concurrency 4. The vLLM column is the earlier separately-leased scorecard rather than a
fresh paired run, so those ratios are directional; only the plow-vs-plow columns above
are a matched measurement.

The concurrency-1 projection is now within its own primitive's reach: 15.52 ms of
matched-bucket weight streaming per token against a 23.2 ms TPOT, with the recorded
attention deficit (2.35 vs 0.81 ms) and host overhead accounting for most of the rest.
The concurrency-4 projection is not: 27.29 ms per token against vLLM's roughly 16, and
the levers reachable inside one body are exhausted — the unroll spills, split-K and
N-splits are recorded above as regressions, more columns per wave buys 3.5%, and "more
in-flight bytes per CU" is falsified as the explanation by R=2 itself, which doubles
them for a 3–5% return. What is left is a body that does not spend 24 VALU operations
per 16 bytes of weight per activation row, and on gfx942 that means either
`v_dot2c_f32_bf16` (CDNA4 only) or an accumulation order that is not plow's.

## Consolidated standing, first full sweep (2026-09-07) — superseded

The intermediate table this section carried is superseded twice over by
"Definitive sweep, everything landed" below, which measures the same corpus with
the attention work, the GEMM k-tile and the refreshed tile store in. Removed rather
than kept beside it: three tables of the same eleven cells, differing only in which
changes had landed, is a record that has to be read in order to be read correctly.
The findings that section established and the final table does not restate are the
chunk analysis and the runtime-fusion measurement, both in their own sections above.

splitting on this part.
## Dense flash prefill: half-wave reductions and the interior-tile mask

Two bit-identical changes to `d_flash_prefill` (`runtime/amd/op_attention.h`), the dense
GQA/MHA flash prefill body shared by every dense model Plow ships: Gemma 4's sliding `hd=256`
and full `hd=512` layers, and Llama/Qwen `hd=128`. Both are on by default in
`scripts/build_gfx942.sh` and both are compiled into the four-wave flash object only;
`PLOW_FA_RED_DPP=0` and `PLOW_FA_FASTMASK=0` restore the previous bodies.

**DPP/swizzle half-wave reductions** (`PLOW_WAVE_RED_DPP`, `runtime/amd/amd_common.h`).
`__shfl_xor` has exactly one lowering on gfx9 — `ds_bpermute_b32`, an LDS crossbar operation —
so each of the five butterfly steps in `half_wave_max` and `half_wave_sum` is an LDS round trip
with its own `s_waitcnt lgkmcnt`. Disassembling the shipped four-wave object shows that one KV
tile of `d_flash_prefill<256>` costs 1651 instructions, of which 160 are `ds_bpermute`
(16 accumulator rows x 5 steps x {max, sum}) and 163 are `s_waitcnt`, against 64 MFMAs. At one
wave per SIMD no co-resident wave hides any of that latency. DPP is a VALU operand modifier and
covers the xor-1/2/4/8 steps: `quad_perm[1,0,3,2]` is lane^1, `quad_perm[2,3,0,1]` is lane^2,
and once a quad (respectively an octet) holds a single value, `ROW_HALF_MIRROR` acts as lane^4
and `ROW_MIRROR` as lane^8. Only the final xor-16 crosses a DPP row, and `ds_swizzle_b32` in
bit-mask mode does it in one LDS operation with no address register. `fmaxf` is associative,
commutative and exact, so the max runs ascending and needs one LDS operation; f32 addition is
not, so the sum keeps its original 16,8,4,2,1 tree and only its last two steps become DPP. 160
LDS permutes per tile become 64, and both reductions stay bit-identical.

**Interior-tile mask skip** (`FA_FASTMASK`). Every term of the causal/window/bounds mask is
monotone in the row and in the column, so evaluating it once at the two extremes of a 32x32
tile decides it for the whole tile and the mask collapses to the scale multiply. Every operand
is wave-uniform, so this is a scalar branch and not lane divergence, and the taken path is
exactly the all-valid path of the masked form. The causal triangle makes most tiles of a large
q-tile interior.

Static cost of one KV-tile loop body, from `llvm-objdump` of a standalone probe built with the
four-wave flash-object axes (`PLOW_WG_WAVES=4 FA_DC=256 FA_DBUF=1`):

| arm | instructions | LDS permutes | `s_waitcnt` | VALU | MFMA | VGPR | scratch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d_flash_prefill<256>` shipped | 1651 | 160 `ds_bpermute` | 163 | 791 | 64 | 419 | 0 |
| + DPP reductions | 1589 | 64 `ds_swizzle` | 91 | 887 | 64 | 407 | 0 |
| + interior-tile mask | 1584 | 64 `ds_swizzle` | 91 | 909 | 64 | 377 | 0 |
| `d_flash_prefill<512>` shipped | 1848 | 160 `ds_bpermute` | 160 | 846 | 64 | 512 | 47 B |
| + both | 1816 | 64 `ds_swizzle` | 91 | 981 | 64 | 512 | 10 B |

The `+ interior-tile mask` rows are static counts over both mask paths; the interior path
executes fewer. The interpreter object keeps its shape: `interp_flash` stays VGPR 512 /
AGPR 256 / LDS 58,368 with zero spill.

### Cold-prefill TTFT

`plowrt bench --prefill-sweep`, one MI300X, the `assets-glm53` blob, p50 of nine timed requests
after three warmups, best of two passes per lease. Object directories differ only in
`interp_flash*`; every other object is byte-identical across the arms.

| lease | arm | 128 tok | 1024 tok | 4096 tok |
|---|---|---:|---:|---:|
| A | control | 92.85 | 261.82 | 1146.85 |
| A | DPP reductions only | 92.66 (-0.2%) | 260.42 (-0.5%) | 1131.39 (-1.3%) |
| A | interior-tile mask only | 92.50 (-0.4%) | 259.88 (-0.7%) | 1129.62 (-1.5%) |
| A | both | 92.38 (-0.5%) | 257.36 (-1.7%) | 1110.99 (-3.1%) |
| B | control | 93.15 | 263.88 | 1155.89 |
| B | both | 92.58 (-0.6%) | 258.67 (-2.0%) | 1116.03 (-3.5%) |
| B | both + head-major work order | 92.58 (-0.6%) | 260.03 (-1.5%) | 1119.70 (-3.1%) |
| B | both + lazy rescale | 94.30 (+1.2%) | 264.73 (+0.3%) | 1151.07 (-0.4%) |
| C | control | 92.36 | 262.55 | 1150.97 |
| C | shipped candidate | 91.80 (-0.6%) | 257.26 (-2.0%) | 1110.79 (-3.5%) |

The two effects are close to additive and the result reproduces across three separate leases.
`plowrt bench` records an `output_checksum` per length; it is identical for every arm at every
length in all three leases.

### Serving

`scripts/bench_packed_serve.py` against `scripts/glm53_serve_inner.sh`, 128 output tokens, one
full warmup and three timed repetitions per cell, two interleaved control/candidate rounds on
one lease. Each cell is the median over the six timed repetitions.

| in / conc | tok/s ctrl -> cand | median TTFT ms ctrl -> cand | best TTFT ms ctrl -> cand | median TPOT ms ctrl -> cand |
|---|---:|---:|---:|---:|
| 128 / 1 | 30.41 -> 30.53 | 102.9 -> 101.7 | 102.6 -> 101.2 | 32.33 -> 32.21 |
| 128 / 4 | 94.46 -> 94.80 | 338.3 -> 335.2 | 336.4 -> 332.3 | 39.12 -> 39.00 |
| 1024 / 1 | 28.66 -> 28.90 | 266.5 -> 260.2 | 265.9 -> 258.6 | 33.07 -> 32.83 |
| 1024 / 4 | 82.18 -> 81.98 | 763.9 -> 811.6 | 758.2 -> 743.5 | 42.15 -> 41.89 |
| 4096 / 1 | 23.76 -> 24.05 | 1160.0 -> 1116.8 | 1148.3 -> 1112.5 | 33.29 -> 33.11 |
| 4096 / 4 | 48.42 -> 49.28 | 3330.4 -> 3224.4 | 3304.0 -> 3209.3 | 53.44 -> 52.85 |

TPOT improves or is unchanged in every cell. The 1024/4 median is not a usable comparison at
this sample size: that cell is bimodal in both engines — the six control TTFTs are
758, 761, 760, 767, 768, 835 and the six candidate TTFTs are 744, 744, 809, 817, 814, 815 — and
which mode a server process lands in is a packing/admission artifact that does not change within
a process. Read the `best TTFT` column and the cold-prefill sweep above for that cell.

### Identity

Greedy completions, `temperature 0`, concurrency 1, three distinct natural-language passages at
each of 128, 1024 and 4096 input tokens (prompts are trimmed by binary search on the server's
own `prompt_tokens`; the third 4096 passage lands on 3996), 64 output tokens each:
**9/9 character-identical** between the control and candidate objects, and 9/9 identical between
two separate control server processes. The 90 completion texts recorded by the serving benchmark
above (128 output tokens each) are also character-identical, as are the per-length
`output_checksum` values from all three cold-prefill leases.

The `runtime/tests/attention_gfx950_test.c` golden harness, rebuilt against both object sources,
reports the same 18 cases PASS with identical max and rms error values in every case — the
reductions and the mask skip are bit-identical by construction and measurably so.

### Nulls

- **Head-major work order** (`FA_HEAD_MAJOR`, default off). The shipped work order is
  split-fastest, then head, then q-tile, so the ~304 work items live at any instant span every
  head: for a Gemma sliding layer at chunk 1024 that is all 16 KV heads, a 16 MiB window against
  one XCD's 4 MiB L2. Making the head the slowest axis narrows the live window to about seven KV
  heads. Measured on top of the winning pair it is a small regression: 1024 tok
  258.67 -> 260.03 ms, 4096 tok 1116.03 -> 1119.70 ms. The KV re-stream is not what this kernel
  is waiting on.
- **Lazy rescale** (`FA_LAZY_RESCALE` / `PLOW_FA_LAZY`, default off). The wave vote is now free —
  `m_st[i]` is half-wave uniform, so `half_wave_max(p[i]) > m_st[i]` is true exactly when some
  lane has `p[i] > m_st[i]`, and 16 compares replace 16 32-lane reductions. It is still a
  regression: 128 tok 92.58 -> 94.30 ms, 1024 tok 258.67 -> 264.73 ms, 4096 tok
  1116.03 -> 1151.07 ms, and it costs the object 22 B of scratch. With 32 query rows per wave the
  probability that no row's running max moves over a 32-column tile is small, so the skip almost
  never fires and only its cost remains.
- **Query tile 64 -> 128 was already done.** The roofline note asks for it, but `FA_BQ` is 32
  query rows *per wave* and all `PLOW_WAVES` waves share one staged K/V tile, so the workgroup
  q-tile is already 128 rows in the four-wave flash object (256 in the eight-wave one). Doubling
  it again to 256 is not attractive at Plow's chunk sizes: it needs two of everything the softmax
  holds (the same register doubling that made `FA_BKV_D128=64` a 30% loss), it adds about 11%
  fully-masked work on the diagonal block, and it halves the number of work items — at chunk 1024
  the sliding layers go from 8 x 32 x 3 = 768 items to 384 for 304 CUs, where the largest single
  item then exceeds the mean per-workgroup load. It was not built. The head-major result above is
  the direct evidence against its premise: both levers pay off only if the per-query-block KV
  re-stream is the constraint, and it is not.

### `head_dim` 64 and the four-wave V-slab overrun

Separate from the performance work, in its own commit.

`exec_flash_prefill`, `exec_flash_decode` and both `FlashMerge` dispatches had arms for 128, 256
and 512 only, with no `else`. An `hd=64` packet — GPT-OSS's geometry — therefore fell through and
wrote nothing: attention silently skipped rather than refused. The arms are added for the bf16
rows. They are deliberately not added to the fp8-KV rows: no fp8-KV model has a 64-wide head, and
the extra instantiation makes the compiler outline `d_flash_mla_prefill_v2` (K3's hot prefill
body) out of the interpreter, which `scripts/asm_audit.py` flags on `interp_flash_fp8kv`. This
closes the dispatch hole only — `d_flash_merge` still has no attention-sink fold, so a GPT-OSS
blob is not numerically complete on AMD until that lands.

Instantiating `d_flash_prefill<64>` exposed a live bug in the `FA_DBUF` prefetch path. The V slab
is `DCH = min(FA_DC, DV)` wide, but the register-held double buffer walked it `FA_DC` wide, so
for any head narrower than the output chunk it read past the end of every V row and wrote past
the end of `Vsm` into `Psm`. That is `FA_DC=256` against `DCH=128` in the shipped four-wave flash
object — the object Llama/Qwen `hd=128` segments are dispatched to. Rebuilding the golden harness
with the four-wave axes reproduces it exactly: all six `hd=128` prefill cases fail with max error
0.23-0.73 against `|O|max` about 1, while `hd=256` and `hd=512` pass untouched. With the width
fixed all eighteen cases pass, and the `hd=256`/`hd=512` disassembly is byte-identical before and
after. The eight-wave prefill interpreter is built `FA_DBUF=0` and was never affected, which is
why the existing golden run — built with the default axes — never saw it.

Narrowing that walk changes `d_flash_prefill<128, true>`'s code size enough to flip an inlining
decision in `interp_flash_fp8kv`, which moved the `v_mfma_f32_16x16x16_bf16` assertion in
`scripts/asm_expect_gfx942.json` from the interpreter entry point to `d_flash_mla_prefill_v2`;
that file records that the assertion now follows the body and has to be re-homed if a later
compiler inlines it again.

## Consolidated standing after the attention work (2026-09-07)

Re-measured with the DPP/swizzle wave reductions, the interior-tile mask skip
and the `FA_DBUF` V-slab fix in the objects, everything else unchanged from the
table above. Same corpus, client and lease discipline.

| Input / concurrency | Plow, start | Plow, consolidated | vLLM 0.28 | tok/s | TTFT | TPOT |
|---|---:|---:|---:|---:|---:|---:|
| 128 / 1 | 30.79 / 91.8 / 31.53 | 41.44 / 92.4 / 23.04 | 55.31 / 45.9 / 17.63 | 0.75x | 0.50x | 0.77x |
| 128 / 4 | 85.95 / 341.7 / 40.20 | 88.38 / 315.9 / 39.22 | 180.39 / 109.2 / 20.79 | 0.49x | 0.35x | 0.53x |
| 512 / 1 | 29.63 / 156.4 / 31.80 | 39.43 / 155.3 / 23.30 | 52.32 / 81.9 / 18.11 | 0.75x | 0.53x | 0.78x |
| 512 / 4 | 78.31 / 506.3 / 42.18 | 80.74 / 473.4 / 41.05 | 158.74 / 270.7 / 21.29 | 0.51x | 0.57x | 0.52x |
| 2048 / 1 | 23.81 / 670.8 / 32.02 | 33.43 / 434.0 / 23.49 | 44.39 / 265.7 / 18.66 | 0.75x | 0.61x | 0.79x |
| 2048 / 4 | 49.46 / 2345.6 / 43.25 | 58.53 / 1220.2 / 48.33 | 103.95 / 1050.8 / 22.40 | 0.56x | 0.86x | 0.46x |
| 8192 / 1 | 12.75 / 2974.0 / 32.44 | 20.31 / 1644.1 / 23.95 | 25.58 / 1218.9 / 20.36 | 0.79x | 0.74x | 0.85x |
| 8192 / 4 | 18.92 / 10553.0 / 45.42 | 23.28 / 7563.7 / 52.65 | 39.86 / 4254.3 / 33.92 | 0.58x | 0.56x | 0.64x |
| 16384 / 4 | — | 12.69 / 16379.5 / 58.24 | 19.30 / 8477.3 / 74.90 | 0.66x | 0.52x | **1.29x** |
| 32768 / 1 | 3.45 / 16404.4 / 34.05 | 5.54 / 9938.5 / 25.52 | 6.87 / 7827.6 / 23.61 | 0.81x | 0.79x | 0.93x |
| 122880 / 1 | 0.34 / 183744.8 / 116.08 | 0.75 / 83682.6 / 31.24 | 0.89 / 69744.7 / 28.52 | 0.84x | 0.83x | 0.91x |

The attention work is worth 1.4-9.4% of TTFT on its own, rising with context —
0.0% at 128 tokens, 4.4% at 8192 and 9.4% at 122880 — which is the shape a
softmax-latency fix should have.

Every ratio improves with context and none reaches 1.0 except TPOT at 16384/4
and 32768/4. Solo throughput 0.50-0.56x → **0.75-0.84x**, solo TTFT 0.50-0.76x →
0.50-0.83x, solo TPOT 0.55-0.69x → 0.77-0.93x. Batched throughput remains
0.49-0.66x.

The two remaining gaps have different characters. **Short-prompt TTFT is flat**
across every change landed: 91.8 → 92.4 ms at 128 tokens while its compute
shrank, which points at a fixed per-request floor rather than at kernel speed.
**Batched decode is compute-bound where vLLM is memory-bound**: plow moves
weights at 839 GB/s against vLLM's 2111 and a measured 4112 GB/s ceiling,
because `gemv_rows` spends about 24 VALU ops per 16 B of weight per activation
row and gfx942 has no `v_dot2c_f32_bf16`, so batch 4 pays 4x the VALU for 1x the
bytes. Closing that needs MFMA, and no MFMA arrangement reproduces plow's
nested-`fma` reference — it would redefine the decode arithmetic.

## Where the prefill milliseconds are, and the tile sized for the wrong machine (2026-09-08)

TTFT is the widest metric in the consolidated standing above and it is worst at
the SHORT prompts — 0.50x of vLLM at 128 input tokens, closing to 0.76x at
122880. That shape (worst when the prompt is small, improving with length)
points at fixed per-prefill cost and at the small-M GEMM path rather than at
asymptotic kernel efficiency. This section measures the 92 ms, then spends some
of it. Everything below is BF16, TP1, one leased MI300X, the 131072-context blob
at `PLOW_MAX_CHUNK=8192`, objects `build-gemma31/hsaco-tiered`.

### There is no host-side floor: 99.4% of TTFT is one GPU drain

`PLOW_TTFT_LOG=1` partitions `[handler entry, first SSE frame]` — the interval
`vllm bench serve` stamps as TTFT — into host phases and the GPU wait. Medians
of the four requests per length in one `bench_packed_serve.py` run at
concurrency 1, milliseconds:

| prompt | TTFT | tokenize | submit→prefill | enqueue (121 AQL) | drain (GPU) | detok+send | HTTP/axum/SSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 92.04 | 0.300 | 0.053 | 0.039 | **91.437** | 0.014 | 0.059 |
| 512 | 154.87 | 0.816 | 0.051 | 0.041 | **153.761** | 0.015 | 0.056 |
| 2048 | 432.86 | 3.141 | 0.057 | 0.041 | **429.427** | 0.018 | 0.045 |
| 8192 | 1644.36 | 13.811 | 0.076 | 0.040 | **1630.209** | 0.019 | 0.056 |

Nothing is hiding on the host. At 128 tokens every host phase together is
0.47 ms — tokenisation, admission, the 121-segment enqueue, detokenising the
first token, and the whole axum/SSE path. `begin_slot`, `plan_chunks`,
`prefill_prepare` and `rearm_prog` are each under 1 us. So the "fixed tens of
ms floor" hypothesis is dead: **the 92 ms is 91.4 ms of GPU and 0.5 ms of
everything else**, and the 128-token TTFT is a pure prefill number — the first
token is sampled by the prefill program's own argmax, so no decode step is
included in it.

Tokenisation is the only host phase that scales, and at 8192 tokens it is 0.8%.

### The GPU split

Captured with plow's own packet timestamps (`PLOW_TRACE_RAW`, the 40-byte
`PlowTraceRec` per (workgroup, packet), `s_memrealtime` at 100 MHz) on a
single-GPU `amd-bench --prompt` run, reduced by walking packets in instruction
order against a monotone clock so overlapping independent packets are not
double counted (`start = max(prev_end, arrive)`, `gate = ready - start`,
`body = end - max(ready, start)`). Concurrent siblings — the q/k/v trio, the
gate/up pair — therefore appear as ONE chain entry, charged to whichever comes
first in instruction order.

**T=128, one chunk, 121 launches, 1016 packets, chain span 90.82 ms** (the
served drain for the same shape is 91.44 ms):

| | ms | % |
|---|---:|---:|
| **projection GEMM** | **69.48** | **76.5** |
| — down_proj, 60 | 26.79 | 29.5 |
| — gate+up pair, 60 | 22.06 | 24.3 |
| — o_proj, 60 | 12.02 | 13.2 |
| — q/k/v trio, 60 | 8.61 | 9.5 |
| flash prefill + merge, 120 | 8.02 | 8.8 |
| norm / rope / GLU, 481 | 11.36 | 12.5 |
| lm_head + softcap + argmax, 4 | 0.67 | 0.7 |
| unallocated gaps across 121 launches | 1.29 | 1.4 |

**T=2048, chain span 431.66 ms** (served drain 429.4 ms):

| | ms | % |
|---|---:|---:|
| **projection GEMM** | **324.24** | **75.1** |
| — gate/up, fused `GemmGlu` | 151.93 | 35.2 |
| — down_proj | 80.22 | 18.6 |
| — q/k/v trio | 56.65 | 13.1 |
| — o_proj | 35.44 | 8.2 |
| flash prefill + merge | 78.75 | 18.2 |
| norm / rope | 26.43 | 6.1 |
| lm_head + sampling | 0.71 | 0.2 |
| gaps | 1.53 | 0.4 |

Projections are three quarters of a prefill chunk at BOTH shapes. Attention
grows from 8.8% to 18.2% with the KV span; norms shrink from 12.5% to 6.1% as
their per-packet cost is amortised over more rows. Launch gaps are 1.4% of the
short envelope over 121 dispatches, about 10.6 us each — a real cost, but not
the one to spend a campaign on.

### Why the short GEMMs are slow: CU fill, not kernel efficiency

`blocks` (the emitted slice count) and the tile geometry together decide how
many of the 304 CUs do anything. At T=128, with `GemmSmall` = 64x128 and
`GemmMed` = 128x128:

| op | N | K | tile | tiles = ⌈M/BM⌉·⌈N/BN⌉ | slices | CUs busy | measured us/layer |
|---|---:|---:|---|---:|---:|---:|---:|
| q_proj | 8192 | 5376 | GemmSmall | 2·64 = 128 | 152 | 128 | 142 (trio) |
| k/v_proj | 4096 | 5376 | GemmSmall | 2·32 = 64 | 76 | 64 | — |
| o_proj | 5376 | 8192 | GemmSmall | 2·42 = 84 | 304 | **84** | 171 |
| gate, up | 21504 | 5376 | GemmMed | 1·168 = 168 | 152 | 152, **2 rounds** | 368 (pair) |
| down_proj | 5376 | 21504 | GemmSmall | 2·42 = 84 | 304 | **84** | 446 |

Dividing each op's per-tile operand bytes `(BM+BN)·K·2` by its measured time
gives 18.1-18.5 GB/s per CU for o_proj and down_proj and 14.5-15 GB/s when 256
or 304 CUs are live — i.e. **a flat per-CU streaming rate, so wall time is set
by how many CUs the shape lights up.** down_proj lights 84 of 304 and is 30% of
the envelope on its own; the aggregate it achieves is 1.55 TB/s against the
part's 5.3.

At 8 waves the wave grid is `WM=2, WN=4`, and `BM % (WM·32) == 0`,
`BN % (WN·32) == 0` pin the smallest legal tile at 64x128. o_proj and down_proj
are therefore already on the smallest tile the object can dispatch: 84 tiles is
the maximum this kernel can make of `M=128, N=5376`, and closing that gap needs
either a 4-wave GEMM path (BN=64 becomes legal) or split-K, which changes the
accumulation order.

### The compiled ceiling: a tile chosen against 304 CUs and then handed 76

gate/up above is not on that floor — it is paying two rounds. The cause is the
same shape as `PLOW_GEMV_MM`. `pick_tile` ranks candidates by
`rounds x per-tile cost` with `rounds = ceil(tiles / n_units)`, and every
projection passed the GLOBAL `n_cu = 304` as `n_units`. But q/k/v and gate/up
are emitted as CONCURRENT ops over DISJOINT CU sets (`split3`/`split2`, so the
siblings do not contend), so the op is then handed 76 or 152. **The tile is
selected for a machine width the shape never sees**, and the CU-fill term — the
whole reason the small rungs exist — is evaluated against the wrong number:

* T=128 gate and up: 168 `GemmMed` tiles each on 152 slices, two rounds of a
  128x128 tile. `GemmWide` (128x256) is 84 tiles: one round, 384 operand bytes
  per K against 2x256.
* T=512 k and v: 256 `GemmSmall` tiles on 76 slices — **four rounds**.
  `GemmWide` needs one. q_proj: 256 `GemmMed` tiles on 152, two rounds against
  one.

Passing the op's real budget is the fix, and it is arithmetic-preserving: BM/BN
only change which workgroup owns which output block, and each output element is
still accumulated over the full K in the same order by one workgroup.

It is SCOPED to `tiles(m, n) <= budget`, and that bound is measured rather than
cautious. Once `tiles > n_units` the budget is nearly a common divisor across
every candidate and cancels out of the ranking, so substituting it there
re-decides the shape on ceiling quantisation alone. Applied unconditionally it
flipped q/k/v at T=2048 from `GemmWide` to the 192x256 rung — which the byte
model prefers (3 rounds x 448 against 4 x 384) and the machine does not:

| prefill-sweep p50 TTFT, ms | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|
| baseline blob | 92.2 | 155.8 | 439.0 | 1674.1 |
| budget everywhere | 87.4 (−5.2%) | 148.5 (−4.7%) | 444.6 (**+1.3%**) | 1678.2 (+0.2%) |
| budget where `tiles <= budget` | 88.0 (−5.2%) | 149.2 (−4.9%) | 444.4 (+0.3%) | 1696.5 (+0.3%) |

Under the scoped rule the T=2048 and T=8192 programs are byte-identical to the
baseline blob, so their columns are run-to-run spread. T=128 moves gate/up
`GemmMed → GemmWide`; T=512 moves q/k/v to `GemmWide`.

### Two nulls in the GEMM schedule

Both are object-only A/Bs against a prefill object rebuilt from this tree, which
is itself within 0.4% of the shipped one at every length (92.8/156.8/443.0/1691.6
against 92.7/157.1/444.7/1693.5), so the rebuild is not a confound.

| prefill-sweep p50 TTFT, ms | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|
| rebuilt control | 92.7 | 157.1 | 444.7 | 1693.5 |
| `GM_PGR2=1` (two-deep global prefetch) | 107.0 (+15.4%) | 188.9 (+20.2%) | 534.4 (+20.2%) | 2015.4 (+19.0%) |
| `GM_SM_BK=128` (halve GemmSmall's k-tiles) — **NOT A NULL, see below** | 92.6 (−0.0%) | 157.3 (+0.1%) | 446.3 (+0.4%) | 1705.4 (+0.7%) |

The hypothesis both tested was that the flat 18 GB/s per CU is exposed global
latency: at `GM_DBUF=1` each global load has exactly ONE k-tile of MFMA to land
behind, and a 64x128 k-tile is only 0.25 us of MFMA against a >1 us HBM round
trip. Doubling the prefetch DEPTH costs 15-21% at every length — the same sign
and the same reason `op_gemm.h` already records at M=4096, so the regression is
not a short-prompt effect. Halving the NUMBER of exposures (BK 64 → 128, which
still fits the arena at 52,224 B) is a flat null. Prefetch depth is not what the
short-prompt mainloop is short of; all outputs stayed checksum-identical in both
arms.

**The `GM_SM_BK=128` row is void.** `GM_SM_BK` was a bare `#define` in
`op_gemm.h`, not `#ifndef`-guarded, so the `-D` that this arm passed was silently
overridden by the header and the arm measured the shipping object — which is
exactly why it reads as byte-identical rather than merely close. Guarded and
re-measured it is worth −15.2% of TTFT at 128 tokens. See "What hipBLASLt does
with these shapes" below.

### The per-packet protocol floor

`PLOW_TRACE_PHASE=1` on the prefill object splits each packet into its four
protocol phases in one traced run. Microseconds per packet, max over the
workgroups of that packet, T=128 (`FlashPrefill` is excluded: it runs on the
4-wave flash object, which was not rebuilt with the instrument):

| op | pkts | claim+gate | acquire | body | publish |
|---|---:|---:|---:|---:|---:|
| GemmMed | 120 | 34.7 | 3.1 | 368.8 | 8.6 |
| GemmSmall | 290 | 23.8 | 4.6 | 211.3 | 9.1 |
| FlashMerge | 60 | 8.5 | 1.5 | 61.0 | 9.4 |
| Glu | 60 | 141.6 | 3.1 | 16.8 | 7.4 |
| HeadNormRope | 180 | 52.1 | 2.1 | 15.9 | 8.1 |
| RmsNorm | 121 | 325.5 | 2.6 | 12.0 | 9.9 |
| NormResidual | 120 | 309.3 | 2.6 | 9.8 | 11.5 |
| ArgmaxFin | 1 | 32.4 | 0.8 | 9.0 | 2.0 |

`claim+gate` is dependency waiting and belongs to the producer, not to these
ops. What does not is the rest: **acquire plus publish is 11-14 us on every
packet regardless of what the packet does**, and the `body` column has a floor
of its own — `ArgmaxFin` reduces 64 partials on ONE workgroup and takes 9.0 us.
That floor is `plow_exec`'s prologue, the same 112-VGPR-dwords-per-lane scratch
save the decode campaign removed for ops 10/20/22 by calling the inlined helper
directly (raw counter dump not committed; the figures are in this section); the prefill
object is explicitly excluded from that path (`!PLOW_BUCKET_PREFILL` in
`interp.hip`). At roughly 12 serial packets per layer over 60 layers the
protocol plus prologue is 8-14 ms of the 90.8 ms envelope, about 10-15%.

Two ways to spend it: give the prefill object the direct-dispatch arm, or take
packets OFF the serial chain.

**The direct-dispatch arm is a NULL here, and the control bracket is what says
so.** A `plow_exec_light` — the four cheap opcodes (`RmsNorm`, `NormResidual`,
`NormResidualNorm`, `Glu`) behind their own real call, so the callee saves only
what it uses and the dispatch loop's allocation is untouched — builds clean
across all 52 gfx942 objects (both asm audits PASS, no LDS or register
cliff) and moves `interp_prefill`'s spill 126 → 45 (its `_gq`
twin 118 → 143). It then measures, against a control/candidate/control bracket
at 11 repetitions:

| prefill-sweep p50 TTFT, ms | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|
| control | 85.1 | 144.0 | 434.6 | 1686.4 |
| light arm | 83.6 (−1.7%) | 143.6 (−0.3%) | 436.2 (+0.4%) | 1688.2 (+0.1%) |
| control, repeated | 83.9 (−1.4%) | 143.9 (−0.1%) | 436.2 (+0.4%) | 1691.8 (+0.3%) |

The CONTROL moved by as much as the candidate at every one of the four lengths,
in the same direction. So the transform that recovered the prologue on decode
does not recover it here at this register budget, and the 9 us floor the phase
table prices is not reachable by moving these four opcodes off `plow_exec`.
All checksums matched in all three arms; the change is not carried.

Taking packets off the chain does pay —

### Prefill sandwich-norm fusion is worth 3% of TTFT at short prompts

`PLOW_PF_GFUSE=1` at emit fuses the end-of-layer `NormResidual` + `RmsNorm` pair
into one `NormResidualNorm` packet. It was qualified bitwise for ordinary
prefill in the norm-fusion campaign
(report (raw artefact removed; see the tables above)) but its stock serving
comparison was interrupted and never completed. It costs 120 instructions out of
the T=128 program's 1016 and, at 22 us of protocol-plus-body each, that is
exactly the packet-count lever the phase table names. Measured here on top of
the tile fix, prefill-sweep p50 TTFT:

| ms | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|
| baseline blob | 92.3 | 155.3 | 437.3 | 1681.8 |
| + budget-scoped tile | 87.0 (−5.7%) | 147.2 (−5.2%) | 438.4 (+0.2%) | 1689.5 (+0.5%) |
| + `PLOW_PF_GFUSE=1` | 84.2 (−8.8%) | 143.8 (−7.3%) | 434.2 (−0.7%) | 1684.5 (+0.2%) |

It stays an EMIT RECIPE FLAG rather than a new default: the loader refuses a
fused-norm blob against an object without the `PLOW_HAS_NORM_RESIDUAL_NORM`
marker, so flipping the default would break every existing object set. The
gfx942 objects in `build-gemma31/hsaco-tiered` carry it.

### Served result and qualification

Both blobs served through `plowrt serve` on one leased MI300X with the shipping
objects and `PLOW_PF_BATCH=1 PLOW_PF_CHUNK=8192 PLOW_MULTISTEP=4`, the same
client and corpus as the consolidated table, medians of three after one warmup,
concurrency 1, 64 output tokens. The control is a blob re-emitted from this
tree; its tile selection is IDENTICAL to the shipped blob's at every bucket
(the tuning store is stale against the live digest, so both resolve
analytically), which is what makes it a valid stand-in.

Each cell is **output tokens/s / median TTFT ms / median TPOT ms**.

| Input / conc 1 | control | + tile budget + `PLOW_PF_GFUSE` | vLLM 0.28 | TTFT vs vLLM |
|---|---:|---:|---:|---:|
| 128 | 41.22 / 92.0 / 23.18 | 41.18 / **85.8** / 23.30 | 55.31 / 45.9 / 17.63 | 0.50x → **0.53x** |
| 512 | 38.96 / 164.7 / 23.47 | 39.10 / **154.3** / 23.53 | 52.32 / 81.9 / 18.11 | 0.50x → **0.53x** |
| 2048 | 33.07 / 446.2 / 23.63 | 32.78 / 445.4 / 23.92 | 44.39 / 265.7 / 18.66 | 0.60x → 0.60x |
| 8192 | 20.02 / 1679.0 / 24.08 | 19.94 / 1686.3 / 24.17 | 25.58 / 1218.9 / 20.36 | 0.73x → 0.72x |
| 32768 | 5.46 / 10110.2 / 25.67 | 5.45 / 10123.4 / 25.77 | 6.87 / 7827.6 / 23.61 | 0.77x → 0.77x |

TTFT falls 6.7% at 128 and 6.3% at 512 and is flat within run-to-run spread
above 2048, which is what the mechanism predicts: only the T=128 and T=512
programs changed. Throughput and TPOT move by less than 1% in both directions —
64 output tokens amortise a 6 ms prefill saving to below this harness's noise,
so this is a TTFT change, not a throughput one. Clocks are unpinned and the two
arms ran on separately leased cards.

**Qualification.** Greedy completions, temperature 0, `ignore_eos`, 64 tokens,
three prompts at each of 128 / 2048 / 8192 input tokens, on two corpora — the
bench corpus (one repeated word) and deterministic pseudorandom token-id rows.
The second corpus is not decoration: the bench corpus's continuation is
degenerate, three distinct texts across its nine rows, where the id rows give
eight. All 18 completions are character-identical between control and candidate,
for the tile change alone AND for the tile change plus the norm fusion, and the
prefill-sweep `output_checksum` agrees at all four lengths in every A/B above,
including both rejected object variants. That is the expected result: the tile change moves
which workgroup owns which output block and the norm fusion preserves the BF16
residual boundary; neither reorders an accumulation.

### What the numbers say is left

Ranked by the T=128 envelope they own, and none of them is host work or launch
overhead:

1. **o_proj and down_proj: 38.8 ms, at 27.6% machine occupancy.** `M=128,
   N=5376` makes 84 tiles on the smallest tile the 8-wave GEMM can dispatch, and
   84 workgroups at 18.5 GB/s each is 1.55 TB/s on a 5.3 TB/s part. The two
   escapes are a 4-wave GEMM path — `WN=2` legalises `BN=64`, doubling the tile
   count at the same per-tile bytes — and split-K, which needs its reduction
   qualified because it reorders the accumulation. Everything reachable by
   re-picking among the compiled rungs is now taken.
2. **The `plow_exec` prologue, 8-14 ms.** The phase table above prices it: a
   9 us body floor on a packet that does nothing plus 8-11 us of publish, on
   roughly 12 serial packets per layer. It is REAL and it is NOT reachable the
   way decode reached it — the `plow_exec_light` bracket above is a null for the
   four cheap opcodes. What is untested is `HeadNormRope` (180 packets, the
   largest count, four head-dim template instantiations) and the GEMM opcodes
   themselves, which is where the register budget bites hardest.
3. **Attention, 8.8% at T=128 rising to 18.2% at T=2048.** Owned by the dense
   flash prefill kernel, not by the prefill program structure.

Two things that are NOT on the list, because they were measured and are not
there: host-side per-request cost (0.5 ms of a 92 ms TTFT) and the launch
schedule (1.3 ms of unallocated gap across 121 dispatches).
## MFMA batched decode: the arithmetic wall is real, removing it is not enough

**This documents a candidate that is NOT shipped.** `GV_MFMA4` defaults to 0 in
`runtime/amd/op_gemm.h` and `PLOW_GEMV_MFMA4` defaults to 0 in `scripts/build_gfx942.sh`. It
changes the model's arithmetic, so whether it ships is a model-owner decision and not a
performance one. Everything below is the evidence for that decision.

### The question

After the batch-width-matched decode tiers, concurrency 1 is fixed and concurrency 4 is not.
Standing on this box (medians of 3, one leased card each, same client; tokens/s / TTFT ms /
TPOT ms):

| in / conc | plow | vLLM 0.28 | plow / vLLM tok/s |
|---|---|---|---:|
| 128 / 4 | 88.71 / 317.0 / 39.02 | 180.39 / 109.2 / 20.79 | 0.49x |
| 512 / 4 | 80.80 / 478.6 / 40.93 | 158.74 / 270.7 / 21.29 | 0.51x |
| 2048 / 4 | 57.99 / 1254.0 / 48.45 | 103.95 / 1050.8 / 22.40 | 0.56x |
| 8192 / 4 | 22.49 / 7927.7 / 53.10 | 39.86 / 4254.3 / 33.92 | 0.56x |

The diagnosis is arithmetic, not memory. `gemv_rows` spends about 24 VALU ops per 16 bytes of
weight **per activation row** — gfx942 has no `v_dot2c_f32_bf16`, so `plow_dot2_bf16` emulates
it with shifts plus FMAs. The weight byte stream does not depend on the batch width and that
work does, so at batch 4 the arithmetic is 4x while the bytes are 1x. Per CU, per KB of weight,
at M=4 and 2.1 GHz:

| | CU-cycles per KB |
|---|---:|
| memory, 4112 GB/s over 304 CUs | 160 |
| VALU, 4 rows x 24 ops x 4 cycles | 96, before loads, LDS, addressing or the loop |
| MFMA, 4 rows x 2 `v_mfma_f32_4x4x4bf16_1k` x 8 cycles | 16 |

vLLM's decode GEMV (`wvSplitK_hf_sml_`, v0.28 `csrc/rocm/skinny_gemms.cu`) is on the matrix
core. So: how fast can an MFMA batched decode be here, what does it cost numerically, and does
it close the gap?

### The kernel

`gemv_rows_mfma4<MM, XLDS, UN, YT>` (op_gemm.h). The mapping is vLLM's, and the reason it is
that mapping and not "32 output rows in the MFMA's M" is BYTES PER INSTRUCTION.
`v_mfma_f32_4x4x4bf16_1k` is 16 *independent* 4x4x4 blocks: block `b = lane/4`, and inside a
block lane `l` supplies A row / B column `l%4`. Give both operands the SAME per-lane 4-half
k-window and the accumulator DIAGONAL holds that lane's dot4; the off-diagonal is junk that is
never read. Three quarters of the matrix is wasted, and it does not matter:

| instruction | weight halves retired per wave | cycles | bytes/cycle |
|---|---:|---:|---:|
| `v_mfma_f32_4x4x4bf16_1k` | 256 (2 per 16 B) | 8 | 64 |
| `v_mfma_f32_16x16x16bf16_1k` | 256 | 16 | 32 |
| `v_mfma_f32_32x32x8bf16_1k` | 256 | 32 | 16 |

The 4x4 arm is the only one that keeps the SHIPPED memory pattern — one wave owns one output
row and its 64 lanes read 1024 CONTIGUOUS bytes of that row through one buffer descriptor —
while still retiring 16 B of weight per lane per two instructions. It is also why the existing
M=1 32x32 arm (`gemv_rows_mfma`, `GV_MFMA`) loses by 1.6%: same bytes, four times the cycles,
and a dependent accumulator chain.

The workgroup's column interval `[gv_n0, gv_n1)` is unchanged, so PLOW_FINE's gemv->headnorm
dependency map is unaffected — only which wave takes which column moves, exactly as for
`gemv_rows_rs`. Activations are LDS-staged when `d_gemv_t`'s own `M*K <= GM_LDS_HALVES` test
passes and read from global otherwise; the unstaged shapes (`o_proj` at M=4 is 32768 halves
against a 32256 budget, `down` is 86016) carry a third of the weight bytes, so the arm is hooked
into both branches or most of the win is left behind.

The arm REFUSES `MM == 1`. At batch 1 the body is already memory-bound and the shipped VALU
arithmetic is bit-identical to every decode golden in the tree, so the rung-1 tier keeps it and
enabling the flag cannot perturb a batch-1 sequence at all. The served concurrency-1 rows below
are the negative control that says so.

### The primitive

`runtime/bench/amd/gemma31_mfma_decode_bench.{hip,cpp}`, built by
`scripts/mfma_decode_build.sh`, run by `scripts/mfma_decode_run.sh`. Same methodology as
`gemma31_gemv_decode_bench`: production bodies included verbatim, launch geometry mirroring the
persistent interpreter (blockDim 512, grid = the packet's own `b` from `plowrt disasm
--program 4`), `d_gemv_t`'s own staged/unstaged decision and packet framing, an in-kernel rep
loop walking a FRESH weight slab out of a 3 GiB arena, hipEvent median-of-41, palindromic
interleave, and an A/A control arm that is byte-identical device code under a second name.

**A/A control: 0.990-1.040 over 28 cells in two runs.** 27 of the 28 fall in 0.990-1.008; the
one outlier is `o_slide` at MM=4, which read 1.040 in one run and 0.990 in the other. Ratios
below about 4% on that one shape are not resolvable here; nothing reported below is that small.

Compiled bucket MM=4, runtime M=4 — the shipping decode bucket. `base` is the shipped
`gemv_rows` at `GV_UNROLL_M4=6`; the MFMA column is UN=6/YT=2, the best standalone of thirteen
tiles measured.

| shape | N x K | blk | base us | base GB/s | MFMA us | MFMA GB/s | speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| q_proj | 8192 x 5376 | 152 | 42.940 | 2051 | 23.763 | 3706 | 1.807x |
| kv_proj | 4096 x 5376 | 76 | 36.319 | 1213 | 16.714 | 2636 | 2.173x |
| o_slide | 5376 x 8192 | 304 | 32.062 | 2747 | 22.984 | 3832 | 1.395x |
| o_full | 5376 x 16384 | 304 | 61.748 | 2853 | 45.875 | 3840 | 1.346x |
| gate_up | 21504 x 5376 | 152 | 109.403 | 2113 | 59.947 | 3856 | 1.825x |
| down | 5376 x 21504 | 304 | 77.146 | 2997 | 57.443 | 4025 | 1.343x |
| lm_head | 262144 x 5376 | 304 | 919.333 | 3066 | 707.179 | 3986 | 1.300x |
| **per token** | | | **27.832 ms** | 2198 | **16.388 ms** | 3733 | **1.698x** |

The per-token row weights each shape by the T=4 program's own instance count (60 / 120 / 50 /
10 / 120 / 60 / 1) and covers 61.17 GB of weights.

The M sweep, per-token ms over the same instance counts:

| bucket | base | best standalone MFMA tile | speedup |
|---|---:|---|---:|
| MM=1, M=1 | 15.292 | 14.113 (UN=3, YT=4) | 1.084x |
| MM=2, M=2 | 19.007 | 15.039 (UN=11, YT=1) | 1.264x |
| MM=4, M=4 | 27.832 | 16.388 (UN=6, YT=2) | 1.698x |
| MM=8, M=8 | 61.599 | 41.844 (UN=3, YT=2) | 1.472x |

**The right way to read this is against the batch-1 floor.** MM=1's 15.292 ms is what streaming
these weights once costs in this harness, and it is memory-bound: the MM=1 base reaches
3982-4219 GB/s on every shape. The shipped VALU body at M=4 costs 1.82x that floor; the MFMA arm
costs **1.07x** it. Per shape the MFMA arm holds 87-96% of the batch-1 rate where the VALU body
holds 40-73%:

| shape | M=1 base GB/s | M=4 VALU (% of M=1) | M=4 MFMA (% of M=1) |
|---|---:|---:|---:|
| q_proj | 3982 | 2051 (52%) | 3706 (93%) |
| kv_proj | 3014 | 1213 (40%) | 2636 (87%) |
| o_slide | 4173 | 2747 (66%) | 3832 (92%) |
| o_full | 4087 | 2853 (70%) | 3840 (94%) |
| gate_up | 4119 | 2113 (51%) | 3856 (94%) |
| down | 4179 | 2997 (72%) | 4025 (96%) |
| lm_head | 4219 | 3066 (73%) | 3986 (94%) |

So the answer to "how fast can an MFMA batched decode be on this part" is: **essentially at the
memory roofline — batch 4 for within 7% of the price of batch 1's weight stream.** The one shape
that stays short, `kv_proj`, is grid-limited rather than body-limited: the emitter dispatches it
at `b=76`, a quarter of the 304 CUs.

MM=1's 1.084x is not a serving win and is not taken: it is entirely on the shapes whose slab is
small enough to see cache, while `lm_head` (a 2.8 GB slab, one rep) reads 0.990x.

### Resources

Standalone bench object, gfx942, ROCm 7.14. Spills are counted from the ISA (`scratch_load_*` /
`scratch_store_*`), not from `.vgpr_spill_count` — earlier candidates in this tree reported 0
there and spilled anyway. `scripts/mfma_decode_res.sh` prints both.

| kernel | VGPR | AGPR | SGPR | LDS | `scratch_load` | `scratch_store` | MFMA | VALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `k_m4_base` (VALU, UN=6) | 237 | 0 | 105 | 64512 | 0 | 0 | 0 | 1226 |
| `k_m4_mf_u3y1` | 70 | 0 | 93 | 64512 | 0 | 0 | 48 | 320 |
| `k_m4_mf_u2y2` | 86 | 0 | 99 | 64512 | 0 | 0 | 64 | 521 |
| `k_m4_mf_u6y1` | 90 | 0 | 97 | 64512 | 0 | 0 | 96 | 364 |
| `k_m4_mf_u4y2` | 146 | 0 | 97 | 64512 | 0 | 0 | 128 | 549 |
| `k_m4_mf_u2y4` (the default) | 146 | 0 | 106 | 64512 | 0 | 0 | 128 | 917 |
| `k_m4_mf_u3y3` | 150 | 0 | 103 | 64512 | 0 | 0 | 144 | 733 |
| `k_m4_mf_u4y3` | 178 | 0 | 103 | 64512 | 0 | 0 | 192 | 747 |
| `k_m4_mf_u6y2` (best standalone) | 194 | 0 | 97 | 64512 | 0 | 0 | 192 | 579 |
| `k_m4_mf_u8y2` | 242 | 0 | 97 | 64512 | 0 | 0 | 256 | 619 |
| `k_m2_base` (VALU, UN=11) | 253 | 0 | 98 | 64512 | 0 | 0 | 0 | 1221 |
| `k_m2_mf_u11y1` (best) | 104 | 0 | 94 | 64512 | 0 | 0 | 88 | 321 |
| `k_m8_base` (VALU, UN=3) | 135 | 0 | 106 | 64512 | 0 | 0 | 0 | 1246 |
| `k_m8_mf_u3y2` (best) | 114 | 0 | 106 | 64512 | 0 | 0 | 192 | 947 |

**No arm spills**, and the accumulators land in arch VGPRs, not AGPRs, on every tile. Every MFMA
tile at MM=4 except UN=8/YT=2 is *below* the VALU body it replaces.

In the megakernel the arms are indistinguishable by resource. `interp_decode_gq.elf`,
`plow_interp_dec_gfx942_gq`, all three at **256 VGPR / 0 AGPR / 108 SGPR / 64568 B LDS / 2
`scratch_load` / 2 `scratch_store`** (the pre-existing spill 2). Only the mix moves:

| object set | `v_mfma` | VALU | `buffer_load` | `ds_read` |
|---|---:|---:|---:|---:|
| control | 0 | 5824 | 36 | 91 |
| candidate UN=6 YT=2 | 192 | 4211 | 36 | 67 |
| candidate UN=2 YT=4 (default) | 128 | 4507 | 28 | 51 |

The `lowrung1` (MM=1) objects are byte-for-byte the control's.

With the flag OFF the decode megakernel is byte-for-byte the tree without this change at all:
`interp_decode_gq`'s disassembly built from the pre-change `op_gemm.h` and from the current one
at `GV_MFMA4=0` differ in nothing but the objdump filename header. The template is defined
unconditionally and instantiated only under the flag, so including the header never changes
behaviour and building without `PLOW_GEMV_MFMA4=1` never does either.

### The tile the megakernel wants is not the tile the primitive wants

`plowrt amd-bench --batched` on a ragged batch of four (prompts of 1024, 128, 4096 and 1024
tokens, so the positions are ragged too), 64 dispatches, no server and no scheduler. Two
palindromic repetitions, reproducible to 0.15%. VALU control: **36.67 ms** per dispatch.

| tile | VGPR | standalone ms/token at M=4 | megakernel batch-4 tpot |
|---|---:|---:|---:|
| UN=2 YT=4 | 146 | 16.859 | **30.32 ms** |
| UN=3 YT=3 | 150 | 16.970 | 30.42 ms |
| UN=4 YT=3 | 178 | 16.699 | 30.52 ms |
| UN=6 YT=2 | 194 | 16.388 (best standalone) | 32.41 ms |
| UN=2 YT=2 | 86 | 19.182 | 32.58 ms |
| UN=11 YT=1 | 128 | 17.860 | 35.22 ms |

A 9% standalone spread maps to a 16% spread in the model, **ordered by YT and not by standalone
time**. YT buys in-flight weight loads with COLUMNS and divides the LDS activation traffic per
weight byte by YT, because the activation fragment is loaded once per (chunk, row) and reused
across all YT columns. Depth (UN) buys the same in-flight loads with REGISTERS, and inside a
256-VGPR interpreter whose other arms are live there are none to spend — which is exactly why
the ranking inverts. YT=3 and YT=4 are a plateau (30.32 / 30.42 / 30.52), so the header default
is **UN=2 / YT=4**, and it is 1.21x the VALU control in the model against 1.65x standalone.

Any tile is arithmetically the same kernel: neither UN nor YT reassociates a row. Checked, not
assumed — the UN=6/YT=2 and UN=2/YT=4 objects produce **bit-identical logit rows on all 195
dumped rows** across three prompts at 128, 1024 and 4096 input tokens.

MM=2's tile is the standalone winner and is NOT measured in the megakernel: the `lowrung2`
object only ever sees 2-row packets, which a concurrency-4 workload barely emits. If it ever
matters, re-run the probe above against `lowrung2` before trusting UN=11/YT=1 there.

### Served

`scripts/mfma_serve_ab.sh` under one lease on one card: two interleaved rounds (control first,
then candidate first), a fresh server process per arm, `scripts/bench_packed_serve.py` at inputs
128/512/2048/8192, 64 output tokens, concurrency 1 and 4, one warmup and three timed repetitions
per cell per round, so each cell is the median of six. Control and candidate are built from the
SAME tree by `scripts/mfma_objsets.sh`, both with `PLOW_DECODE_TIERS=1,2`; the only difference
is `PLOW_GEMV_MFMA4`. Serving flags are the standing ones (`PLOW_PF_BATCH=1 PLOW_PF_CHUNK=8192
PLOW_MULTISTEP=4 PLOW_TP_NO_AUDIT=1`), assets `build-gemma31/assets-ctx131072-chunk8192`.

Candidate = the default UN=2/YT=4 tile:

| in / conc | tok/s ctl -> cand | TTFT ms ctl -> cand | TPOT ms ctl -> cand |
|---|---:|---:|---:|
| 128 / 1 | 41.07 -> 40.95 (-0.3%) | 92.2 -> 92.5 (+0.3%) | 23.27 -> 23.33 (+0.3%) |
| 128 / 4 | 87.85 -> 103.39 (**+17.7%**) | 315.5 -> 317.1 (+0.5%) | 39.48 -> 32.82 (**-16.9%**) |
| 512 / 1 | 38.47 -> 38.33 (-0.4%) | 159.1 -> 159.6 (+0.3%) | 23.88 -> 23.96 (+0.3%) |
| 512 / 4 | 80.09 -> 92.55 (**+15.6%**) | 502.5 -> 497.0 (-1.1%) | 41.39 -> 34.75 (**-16.0%**) |
| 2048 / 1 | 32.67 -> 32.45 (-0.7%) | 444.2 -> 447.8 (+0.8%) | 24.04 -> 24.19 (+0.6%) |
| 2048 / 4 | 57.86 -> 64.37 (**+11.2%**) | 1235.1 -> 1219.3 (-1.3%) | 48.85 -> 42.40 (**-13.2%**) |
| 8192 / 1 | 19.76 -> 19.57 (-1.0%) | 1694.1 -> 1718.2 (+1.4%) | 24.52 -> 24.64 (+0.5%) |
| 8192 / 4 | 22.80 -> 23.81 (**+4.4%**) | 7771.2 -> 7704.1 (-0.9%) | 53.26 -> 46.95 (**-11.8%**) |

Concurrency 1 is flat to within its own run-to-run band, which is the expected negative control:
the rung-1 tier is MM=1 and refuses the arm. The 8192/4 throughput moves least because that cell
is TTFT-dominated (7.7 s of prefill against 3.0 s of decode); its TPOT moves -11.8% like the
rest. The same sweep run earlier against the *standalone*-best UN=6/YT=2 tile gave +11.5 / +10.1
/ +7.9 / +3.3% throughput and -11.4 / -11.1 / -9.3 / -8.3% TPOT — the retune is worth about six
points of throughput on its own.

### Against vLLM

Same byte accounting throughout: 61.17 GB of decode-GEMV weights per step, and TPOT is the step.

| in / conc 4 | plow ctl | plow cand | vLLM 0.28 | cand / vLLM |
|---|---:|---:|---:|---:|
| 128 tok/s | 87.85 | 103.39 | 180.39 | 0.57x |
| 512 tok/s | 80.09 | 92.55 | 158.74 | 0.58x |
| 2048 tok/s | 57.86 | 64.37 | 103.95 | 0.62x |
| 8192 tok/s | 22.80 | 23.81 | 39.86 | 0.60x |
| 128 TPOT ms | 39.48 | 32.82 | 20.79 | 1.58x |
| 128 weight stream | 1549 GB/s | 1864 GB/s | 2942 GB/s | 0.63x |

**It does not close the gap.** Concurrency-4 throughput at 128 tokens goes from 0.49x to 0.57x
of vLLM 0.28, and TPOT from 1.90x to 1.58x. It removes the arithmetic wall completely and the
gap survives it.

### Where the rest of the gap is, and where it is not

The isolated GEMV family at the default tile drops 27.832 -> 16.859 ms per token, i.e. -10.97 ms;
the served step drops 39.48 -> 32.82, i.e. -6.66 ms, so 61% of the primitive's saving reaches the
token. The residual is bounded from below and it is not the GEMV:

* Even a **free** decode GEMV could not take the step below `32.82 - 16.859 = 15.96 ms`, which is
  77% of vLLM's entire 20.79 ms step — and vLLM must itself spend about 14.9 ms of that step just
  streaming 61.17 GB at the measured 4112 GB/s ceiling. So plow's non-GEMV per-step cost is on the
  order of 16 ms against vLLM's ~6 ms, and **that difference, not the projection arithmetic, is
  what remains**.
* The tile probe says further GEMV micro-optimisation is spent: YT=3 and YT=4 are already a
  plateau within 0.7% of each other, and the standalone-best tile is 7% *worse* in the model. A
  deeper unroll, a wider tile, split-K or a different MFMA shape cannot buy back the missing time.
* The concrete next lever is already priced in this tree and costs no arithmetic: the per-packet
  L2 gate. "Every workgroup in a packet issues `buffer_wbl2` + `buffer_inv` at the gate; those are
  PER-L2, so each XCD does the same writeback and the same invalidate once per participating
  workgroup and they SERIALISE. At b=304 that is ~30 us per packet before a single weight byte
  moves" (`scripts/build_gfx942.sh`, `PLOW_GATE_HIER`). The T=4 program issues 421 GEMV packets
  per token. `PLOW_GATE_HIER` measured -16.0% of a decode token on Gemma-4-12B, and it needs an
  L2-placed blob (`PLOW_L2_PLACE=1` at emit, `PLOW_L2_PLACE_DISPATCH=1` at run) built as a pair
  with the objects.

### The numerics price

`scripts/mfma_numerics_ab.sh` + `scripts/mfma_numerics_report.py`: `plowrt amd-bench
--dump-logits`, three natural-language prompts at each of 128, 1024 and 4096 input tokens
(`scripts/mfma_numerics_prompts.py`), 64 decode steps each, candidate against control, same card,
same lease. `--dump-logits` writes the whole `[4, vocab]` `act.logits` tensor; row 0 is the
sequence whose greedy ids amd-bench reports, and every number below is row 0.

`step0` is one decode step off a KV cache both arms wrote identically — the pure per-step kernel
difference, with no history in it. `med`/`max` are over all 64 steps, where the arms decode the
same tokens but write slightly different K/V rows, so they carry accumulated history. `h64` is
restricted to the reference row's top 64 logits, which is the part a sampler ever reaches.

| prompt | prefill | step0 rel L2 | step0 h64 | med | max | med h64 | max h64 | agreement | first div |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 128 / p0 | identical | 1.261e-02 | 4.584e-03 | 1.371e-02 | 7.433e-02 | 7.862e-03 | 6.117e-02 | 64/64 | none |
| 128 / p1 | identical | 1.629e-02 | 4.656e-03 | 1.277e-02 | 3.416e-02 | 1.247e-02 | 6.517e-02 | 64/64 | none |
| 128 / p2 | identical | 1.898e-02 | 8.860e-03 | 3.954e-02 | 9.740e-02 | 4.593e-02 | 6.054e-01 | 64/64 | none |
| 1024 / p0 | identical | 2.541e-03 | 8.004e-03 | 5.397e-03 | 3.983e-02 | 1.075e-02 | 7.525e-02 | 64/64 | none |
| 1024 / p1 | identical | 5.626e-03 | 2.004e-02 | 5.257e-03 | 1.279e-01 | 1.729e-02 | 3.903e-01 | 64/64 | none |
| 1024 / p2 | identical | 4.367e-03 | 1.669e-02 | 4.413e-03 | 3.667e-02 | 1.527e-02 | 1.327e-01 | 64/64 | none |
| 4096 / p0 | identical | 5.381e-03 | 9.644e-03 | 4.062e-03 | 1.196e-02 | 1.352e-02 | 4.532e-02 | 64/64 | none |
| 4096 / p1 | identical | 6.288e-03 | 8.350e-03 | 4.655e-03 | 6.388e-02 | 1.544e-02 | 1.543e-01 | 64/64 | none |
| 4096 / p2 | identical | 2.452e-03 | 8.509e-03 | 3.523e-03 | 8.095e-03 | 1.274e-02 | 3.837e-02 | 64/64 | none |

**576/576 greedy tokens agree, over nine prompts, with no first divergence anywhere.** The
prefill logit row is BIT-IDENTICAL in all nine: prefill runs the GEMM, and nothing here touches
it. The whole-row relative L2 of one decode step is **0.25% to 1.9%**; restricted to the top-64
head it is 0.46% to 2.0%. For scale, the earlier combined gate/up MFMA candidate that was
rejected measured 3.4891% full-model relative L2 for a 20.6% primitive gain; this is about half
that error for 1.65x standalone and 1.21x in the model.

The `max` column exceeds `med` on some prompts (up to 0.60 on the top-64 head at 128/p2, step 63)
and that is history, not a fault: once a step's logits differ at 1e-2, the K and V rows written
that step differ too, so a late step inside a degenerate repeat loop can show a much larger
residual while still choosing the same token by a wide margin. Agreement stays 100% through it.

The 90 served completions from the benchmark above (64 output tokens each, temperature 0, inputs
128/512/2048/8192, concurrency 1 and 4) are **character-identical between the two object sets in
every cell**, at concurrency 4 as well as 1.

### Batch-width independence

A prompt's output must not depend on how many other sequences share the step, and for this arm
that is a property of the code rather than a test result: `acc[m][y]` sees only row `m`'s
activations and column `n+y`'s weights, in increasing k order, through the same instruction
sequence for every `m`. Neither `MM` nor `UN` nor `YT` reassociates a row.

Checked at both levels anyway:

* **Primitive.** For every shape and every arm, row 0 computed with runtime `M=1` is BIT-IDENTICAL
  to row 0 of the same kernel run at the bucket's full width (MM=2, 4 and 8).
* **Model.** `amd-bench --batched` on a ragged batch of four (1024, 128, 4096 and 1024 prompt
  tokens, so the positions are ragged too): slot 0's chain is `[5690, 8112, 834, 3187, 157915]`,
  exactly the first five ids of the same prompt decoded alone, and the harness's own `same-prompt
  slots agree` gate passes. Identical for the control and the candidate, and the candidate's four
  slots match the control's token for token.

### The verdict

* **How fast can an MFMA batched decode be on this part?** At the memory roofline. At batch 4 it
  runs the whole projection family in 1.07x the time the same harness needs to stream those
  weights once at batch 1, holding 87-96% of the batch-1 byte rate per shape where the shipped
  VALU body holds 40-73%. There is nothing meaningful left in this primitive.
* **What does it cost numerically?** It redefines the decode reference arithmetic — bit identity
  is impossible by construction, because every MFMA on gfx942 reduces at least four products in
  hardware order inside one instruction and no arrangement reproduces `dot8`'s nested `fma` pairs.
  Measured: 0.25-1.9% relative L2 on a decode logit row, 576/576 greedy tokens identical over nine
  prompts at three context lengths, prefill bit-identical, all 90 served completions
  character-identical, and per-row values provably independent of the batch width.
* **Does it close the vLLM gap?** No. Concurrency-4 throughput goes from 0.49x to 0.57x of vLLM
  0.28 at 128 input tokens and TPOT from 1.90x to 1.58x. It removes the arithmetic wall
  completely and the gap survives it: even a free decode GEMV would leave the step at ~16 ms
  against vLLM's whole 20.79 ms.
* **Is it worth redefining plow's decode reference arithmetic?** That is the model owner's call.
  The honest framing is +4.4 to +17.7% concurrency-4 throughput and -11.8 to -16.9% TPOT, zero
  effect on concurrency 1, no measured token change anywhere, in exchange for a permanent loss of
  bit-exactness against every existing decode golden. The measurement's own recommendation is to
  take the free lever first: the per-packet L2 gate is bigger on this workload and costs no
  arithmetic at all, and after this arm the GEMV body is no longer what the step is waiting on.

## Consolidated sweep before the GEMM k-tile fix (2026-09-08) — superseded

Superseded by "Definitive sweep, everything landed" below; see the note on the
first superseded sweep for why the intermediate tables are not kept.


## What hipBLASLt does with these shapes, and the one thing plow had never actually built (2026-09-08)

The section above leaves the prefill-GEMM gap as "84 tiles over 304 CUs, and the
escapes are a 4-wave path or split-K". vLLM completes the equivalent prefill in
about 28 ms through hipBLASLt, and hipBLASLt's compiled assets and solution
indices are on this host, so the question of what it actually picks is decidable
rather than arguable. This section mines that answer key, prices plow's own
ladder against it on the same clock, and spends the one arithmetic-preserving
finding that fell out.

Everything below is BF16, TP1, one leased MI300X, `gemm_tile_sweep` for the plow
primitive and a hipBLASLt harness linked against `/opt/rocm-7.2.4/lib` for the
reference. Object set `build-gemma31/hsaco-tiered` with the prefill rows rebuilt
from this tree; blob `build-gemma31/assets-final-plain` (131072 ctx,
`PLOW_MAX_CHUNK=8192`) unchanged across every arm, so the A/Bs are object-only.

### The answer key

`hipblasLtMatmulAlgoGetHeuristic` top-1 for each shape, with the solution name
decoded through the `TensileLibrary_*_gfx942.dat` msgpack indices
(`/opt/rocm-7.2.4/lib/hipblaslt/library`, 309,219 solutions across the 555 gfx942
`.dat` indices) and the launch grid read back with `rocprofv3 --kernel-trace`. hipBLASLt
is column-major and plow's `out[M,N] = a[M,K] · w[N,K]ᵀ` maps to it with the
operands swapped, so **Tensile's M is plow's N**; the tiles below are printed in
plow's order, `BM(tokens) x BN(features) x BK`. GSU is the runtime split-K
factor, recovered as `workgroups / tiles` from the traced grid — the solution
name only records `GSU=-1`, "decided at dispatch".

`plow ms` is the fastest tile `pick_tile` can select, measured with
`gemm_tile_sweep` **after** the `GM_SM_BK=128` change below; `hbl ms` is a
hipEvent median of 41 launches after 5 warm-ups.

| op | N x K | M | plow tile | tiles | CUs | plow ms | hipBLASLt tile | waves | GSU | hbl ms | ratio |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| q_proj | 8192 x 5376 | 128 | 64x128x128 | 128 | 128 | 0.0699 | 64x64x128 | 4 | 1 | 0.0393 | 1.78x |
| k/v_proj | 4096 x 5376 | 128 | 64x128x128 | 64 | 64 | 0.0665 | 64x32x256 | 4 | 1 | 0.0268 | 2.49x |
| o_proj sliding | 5376 x 8192 | 128 | 64x128x128 | 84 | 84 | 0.0943 | 48x64x256 | 4 | 1 | 0.0391 | 2.41x |
| o_proj full | 5376 x 16384 | 128 | 64x128x128 | 84 | 84 | 0.1697 | 128x64x128 | 4 | **3** | 0.0630 | 2.69x |
| gate, up | 21504 x 5376 | 128 | 128x128x64 | 168 | 168 | 0.1249 | 64x160x128 | 4 | 1 | 0.0698 | 1.79x |
| down_proj | 5376 x 21504 | 128 | 64x128x128 | 84 | 84 | 0.2173 | 128x64x128 | 4 | **3** | 0.0756 | 2.87x |
| q_proj | 8192 x 5376 | 512 | 128x128x64 | 256 | 256 | 0.1365 | 128x128x64 | 4 | 1 | 0.0822 | 1.66x |
| k/v_proj | 4096 x 5376 | 512 | 64x128x128 | 256 | 256 | 0.0844 | 128x64x128 | 4 | 1 | 0.0474 | 1.78x |
| o_proj sliding | 5376 x 8192 | 512 | 128x128x64 | 168 | 168 | 0.1593 | 96x128x128 | 4 | 1 | 0.0802 | 1.99x |
| o_proj full | 5376 x 16384 | 512 | 128x128x64 | 168 | 168 | 0.3138 | 96x128x128 | 4 | 1 | 0.1537 | 2.04x |
| gate, up | 21504 x 5376 | 512 | 192x256x64 | 252 | 252 | 0.3222 | 128x320x64 | 4 | 1 | 0.1640 | 1.96x |
| down_proj | 5376 x 21504 | 512 | 128x128x64 | 168 | 168 | 0.4355 | 96x128x128 | 4 | 1 | 0.2037 | 2.14x |
| q_proj | 8192 x 5376 | 2048 | 128x256x64 | 512 | 304 | 0.3949 | 128x224x64 | 4 | 1 | 0.2521 | 1.57x |
| k/v_proj | 4096 x 5376 | 2048 | 128x256x64 | 256 | 256 | 0.2107 | 128x224x64 | 4 | 1 | 0.1270 | 1.66x |
| o_proj sliding | 5376 x 8192 | 2048 | 192x256x64 | 231 | 231 | 0.4314 | 128x288x64 | 4 | 1 | 0.2318 | 1.86x |
| o_proj full | 5376 x 16384 | 2048 | 192x256x64 | 231 | 231 | 0.8443 | 128x288x64 | 4 | 1 | 0.4334 | 1.95x |
| gate, up | 21504 x 5376 | 2048 | 128x256x64 | 1344 | 304 | 1.0992 | 256x192x64 | 4 | 1 | 0.5576 | 1.97x |
| down_proj | 5376 x 21504 | 2048 | 192x256x64 | 231 | 231 | 1.1197 | 128x288x64 | 4 | 1 | 0.7226 | 1.55x |
| q_proj | 8192 x 5376 | 8192 | 192x256x64 | 1376 | 304 | 1.4515 | 224x256x64 | 4 | 1 | 0.8846 | 1.64x |
| k/v_proj | 4096 x 5376 | 8192 | 128x256x64 | 1024 | 304 | 0.7774 | 224x256x64 | 4 | 1 | 0.4417 | 1.76x |
| o_proj sliding | 5376 x 8192 | 8192 | 192x256x64 | 903 | 304 | 1.3612 | 192x256x64 | 4 | 1 | 0.8382 | 1.62x |
| o_proj full | 5376 x 16384 | 8192 | 192x256x64 | 903 | 304 | 2.7569 | 192x256x64 | 4 | 1 | 1.7614 | 1.57x |
| gate, up | 21504 x 5376 | 8192 | 192x256x64 | 3612 | 304 | 3.5974 | 192x256x64 | 4 | 1 | 2.2261 | 1.62x |
| down_proj | 5376 x 21504 | 8192 | 192x256x64 | 903 | 304 | 3.6832 | 192x256x64 | 4 | 1 | 2.5944 | 1.42x |

`lm_head` (262144 x 5376), off the per-layer path and 0.7% of the envelope, for
completeness: hipBLASLt 0.874 / 2.143 / 7.564 / 31.334 ms at M = 128 / 512 /
2048 / 8192.

Four things the key says, none of which the occupancy argument predicted:

1. **hipBLASLt runs 4 waves per workgroup on every one of these 24 shapes.**
   `WG64_4_1`, `WG16_4_4`, `WG32_8_1` — all 256 threads. Not once does it pick an
   8-wave kernel, at any M, including M=8192 where the tile is the same 192x256
   plow ships. plow's prefill object is 8 waves at occupancy 2 and its own
   `op_gemm.h` note records 8 waves as +7% over 4 at 256x256 — measured at
   M=4096 on a shape that fills the machine. It is not the axis that separates
   the two libraries here.

2. **It sizes the tile so the tile COUNT lands near 304, and it uses
   non-power-of-2 tiles to do it.** 48-, 96-, 160-, 224-, 288- and 320-wide
   macro-tiles appear across the table, and the resulting workgroup counts are
   252, 256, 270, 272, 304 — never 84 and never 168. plow's wave-grid assert
   forces `BM % 64 == 0` and `BN % 128 == 0`, so on N=5376 its feature-tile count
   can only be 42, 21 or 14, times 1 or 2 in the token direction at M=128. It
   cannot land near 304 on these shapes at any legal tile.

3. **Split-K is its LAST resort, not its policy.** GSU > 1 appears exactly twice
   in 24 shapes — down_proj and o_proj-full at M=128 — which are precisely the
   two shapes the occupancy section flagged, and exactly the two where no tile
   shape can reach 250 workgroups (5376 features x 128 tokens gives 84 x 1 at
   the 64-wide tile it wants). Everywhere else it reshapes the tile instead. So
   the library agrees with the diagnosis *and* with the policy: reorder the
   accumulation only where geometry leaves nothing else.

4. **At small M it runs a DEEP k-tile.** `depthU` is 128 or 256 on every M=128
   and M=512 row and drops to 64 only at M >= 2048. plow ran BK=64 everywhere.

### The `-D` that was never applied

Point 4 is the one plow could act on inside its existing object, and the record
said it had been tried: the "Two nulls in the GEMM schedule" table above reports
`GM_SM_BK=128` at 92.6 ms against a 92.7 ms control — a flat null, all outputs
checksum-identical.

That measurement was of an unchanged object. `op_gemm.h` declared the per-rung
geometry as bare `#define`s:

```c
#define GM_SM_BM 64
#define GM_SM_BN 128
#define GM_SM_BK 64
```

while `build_gfx942.sh` documents exactly these as reachable through its `GM_AX`
raw-`-D` escape hatch ("the per-rung geometry and schedule knobs op_gemm.h
`#ifndef`-guards"). They were not guarded. A bare `#define` in a header the
command line has already defined is a **redefinition, and the header wins** —
last definition. The recipe compiles with `-w`, so the warning that says so is
suppressed. Reduced to two lines:

```
$ printf '#define GM_SM_BK 64\nint x[GM_SM_BK];\n' > t.c
$ gcc -c -w -DGM_SM_BK=128 t.c && llvm-readelf --symbols t.o | grep ' x$'
     2: 0000000000000000   256 OBJECT  GLOBAL DEFAULT     3 x        # 256 B = 64 ints
```

Every A/B ever run through `GM_AX` against `GM_SM_*` or `GM_MD_*` measured the
shipping object. The null is a build artefact, not a result.

### BK=128 on the 64x128 rung: 1.10x to 1.56x, everywhere

Guarded now, and defaulted to 128 on CDNA3 (CDNA4 runs this ladder
double-buffered in 160 KiB of LDS — a different regime, left alone). The stage
grows from 27,648 to 52,224 B, still inside the 64 KiB arena and inside the
prefill object's 64,512 B GEMM union, so no other rung moves and occupancy does
not change. `gemm_tile_sweep`, whole-GPU wall time in ms, 64x128 BK=64 → BK=128:

| shape | M=128 | M=512 | M=2048 | M=8192 |
|---|---|---|---|---|
| q_proj 8192x5376 | .0954 → .0699 | .1840 → .1435 | .6571 → .4855 | 2.528 → 1.868 |
| k/v_proj 4096x5376 | .0952 → .0665 | .0979 → .0844 | .3565 → .2717 | 1.306 → 0.958 |
| o_proj sliding 5376x8192 | .1182 → .0943 | .2419 → .2055 | .5894 → .5084 | 2.234 → 1.845 |
| o_proj full 5376x16384 | .2319 → .1697 | .4784 → .3962 | 1.233 → 0.996 | 5.381 → 3.982 |
| gate, up 21504x5376 | .1855 → .1417 | .4478 → .3461 | 1.803 → 1.389 | 7.074 → 5.445 |
| down_proj 5376x21504 | .3385 → .2173 | .6713 → .5210 | 1.992 → 1.467 | 7.189 → 5.413 |

1.10x to 1.56x, no shape and no M where it loses, and the sweep's 24-sample f64
oracle passes on every cell. **It is arithmetic-preserving.** BK only sets where
the k-tile barriers fall: the mainloop still walks k strictly ascending in
MFMA_K steps and each output element is still accumulated by one workgroup over
the full K in that order, so the f32 accumulator sees the same adds in the same
sequence.

It is also invisible to `pick_tile`. `tile_cost`'s compute term is
`k_iters · macs(bm·bn·bk)` = `(k/bk) · bm·bn·bk`, in which `bk` cancels, and the
LDS filter passes at 64 and at 128. No emit changes, no Rust rung table moves.

### Served

Object-only A/B on the shipped blob, `bench_packed_serve.py`, concurrency 1, one
warm-up and three measured repeats, `PLOW_PF_BATCH=1 PLOW_PF_CHUNK=8192
PLOW_MULTISTEP=4`. The control is rebuilt from **this** tree with
`GM_AX="-DGM_SM_BK=64"`, so the newly-working guard is the only thing separating
the arms.

| p50 TTFT, ms | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|
| control (BK=64) | 86.78 | 147.00 | 436.32 | 1658.42 |
| BK=128 | **73.56** | 146.91 | 435.92 | 1669.26 |
| | **−15.2%** | −0.1% | −0.1% | +0.7% |

The win is confined to T=128 because that is the only length at which
`pick_tile` selects `GemmSmall` at all; at 512 and above every projection
dispatches a wider rung, those three programs run byte-identical opcodes in both
arms, and their columns are run-to-run spread.

**Identity: 9/9 greedy completions character-identical**, three distinct
natural-language prompts x 64 output tokens at each of 128, 2048 and 8192 input
tokens (prompt lengths 159 / 2079 / 8223 by the server's own `prompt_tokens`),
temperature 0, concurrency 1. Token agreement 1.0000 at every length, no
divergence position at any prompt.

### The two structural escapes, priced against each other

Both were measured as primitives before either was built into the interpreter.
`test_kernels.hip` now carries bench-only experimental rungs and
`gemm_tile_sweep` takes a `PLOW_TILE_WAVES` launch override, so the 4-wave
ladder can be timed against the shipping one on the same clock.

**(a) A 4-wave path, which is the only way BN=64 becomes legal.** The wave-grid
assert is `BN % (WN·32) == 0`; at 8 waves WN=4 pins BN to a multiple of 128, at
4 waves WN=2 admits 64. Built a 4-wave `test_kernels.elf` and swept the same
shapes. Best 4-wave tile against best 8-wave tile, ms at M=128:

| shape | 8w best | best 4-wave tile | ms | gain |
|---|---:|---|---:|---:|
| q_proj | 0.0699 | 64x64x128 | 0.0605 | 1.16x |
| k/v_proj | 0.0665 | 64x64x128 | 0.0553 | 1.20x |
| o_proj sliding | 0.0943 | 64x64x128 | 0.0732 | 1.29x |
| o_proj full | 0.1697 | 64x64x128 | 0.1286 | 1.32x |
| down_proj | 0.2173 | 64x64x128 | 0.1812 | 1.20x |
| gate, up | 0.1249 | 128x128x64 | 0.1383 | **0.90x** |

64x64 doubles the tile count on the N=5376 shapes (84 → 168) and buys 1.16-1.32x
where the machine was starved. But the same 4-wave object is **uniformly worse
at every M >= 512** — the whole ladder loses 10-20% there (down_proj at M=8192:
3.683 ms at 8 waves against 4.401 at 4; gate/up at M=2048: 1.047 against 1.224),
and gate/up already fills the machine at M=128 so it loses there too. A 4-wave
GEMM cannot be the prefill object; it can only be a second object routed by
chunk length, since `PLOW_WAVES` is per-object (`d_gemm_t` static_asserts
`THREADS == PLOW_THREADS`) and the host launch constant follows it
(`WG_THREADS_4` vs `WG_THREADS_8`; dispatching a 4-wave object at 512 threads is
`INVALID_ISA`, not a slowdown). **Not landed**: 1.2-1.3x on one prompt length,
bought with a new object family, a `PrefillArm` route, a rung that is legal in
only one object, and a fused-GLU epilogue that cannot follow it (the epilogue
selects gate against up on the SN axis and static_asserts `SN == 2`, which at 4
waves means BN=128).

**(b) Split-K, priced by traffic-and-occupancy analogue.** A split-K=S run of
(M, N, K) launches `S · tiles(M,N)` workgroups, each streaming K/S of the weight,
with every weight byte still read `tm` times. `(M, N·S, K/S)` has the same
workgroup count, the same per-workgroup K depth and the same total weight bytes,
and no reduction — so it is an upper bound on the mainloop, and the gap to it is
what a reduce would have to fit inside. On the 64x128x128 rung at M=128:

| shape | S=1 | S=2 | S=4 | S=8 |
|---|---:|---:|---:|---:|
| down_proj 5376x21504 | 0.2359 | **0.1308** | 0.1495 | 0.1137 |
| o_proj full 5376x16384 | 0.1686 | **0.1049** | 0.1125 | — |
| o_proj sliding 5376x8192 | 0.0935 | **0.0616** | 0.0644 | — |
| q_proj 8192x5376 | 0.0821 | **0.0495** | — | — |

Split-K=2 is worth **1.52x to 1.80x** on exactly the shapes the 4-wave path buys
1.16-1.32x on, from inside the existing 8-wave object — it needs no new wave
count, no second object and no route. It is the escape the measurement supports,
and it is the one hipBLASLt itself reaches for on two of these same shapes.

It is **not landed and not built**, for two reasons. It reorders accumulation, so
by the standing policy on this branch it can only ship behind a build macro,
default OFF, with its numerics price reported — and a default-off candidate does
not move the served default, which is what this campaign was for. And the body
cannot express it yet: `d_gemm_t` takes `K` as both the reduction length and the
operand row stride, so a K-slice needs a `k0`/`kend` pair threaded through
`GM_FETCH`/`GM_DMA` before a partial can be computed at all. That change, an f32
partials tensor sized `S · M · N`, a reduce opcode and the emit plumbing are the
next piece of work, and the table above is what it is worth.

### Two nulls, with numbers

* **The 8-wave ladder is already picking correctly.** `gemm_tile_sweep` sweeps
  every compiled tile, including the four calibration-only ones (256x128,
  320x128, 192x128, 128x384) that have no opcode. Comparing the best *selectable*
  rung against the best *compiled* tile over all 24 shapes: identical at 20 of
  24, and the largest gap anywhere is **1.09x** (o_proj-full at M=2048, where
  320x128x64 beats 192x256x64 by 0.069 ms). Promoting the missing rungs to
  opcodes is not where the 2x lives.
* **A deeper k-tile on the wider rungs does not fit.** 128x128 at BK=128 is
  69,632 B against a 65,536 B arena, and 64x128 at BK=192 is 76,800 B. On an
  8-wave grid, 64x128x128 is the deepest legal small tile; the rung landed above
  is the ceiling of this axis, not a step on it.

### Where the distance actually is

After BK=128, plow's best selectable tile is **1.42x to 2.87x** off hipBLASLt on
the same shape and the same card, worst at M=128 (2.4-2.9x on o_proj and
down_proj) and narrowing to 1.4-1.8x at M=8192. The M=128 half of that is
occupancy — 84 workgroups against 252 — and split-K=2 closes about 1.6x of it.
The residue that survives at M=8192, where both libraries run the same 192x256
tile on all 304 CUs, is not occupancy and not tile choice: it is per-CU
throughput inside the mainloop, and hipBLASLt's schedule there differs in ways
this campaign priced but did not port — `PGR2` two-deep global prefetch,
`GRVWA8`/`GRVWB8` 16-byte global read widths, `LRVW8` LDS read width, and
`TLDS1` direct-to-LDS loads. `GM_PGR2=1` on its own is a 15-21% regression at
every length (recorded above), so that schedule is not separable one knob at a
time.

### Required follow-up, and one thing that was already red

`devgen --test tuned_tile_selection` has **five failing cases, and they were
failing at the branch tip before this work** — every one of the 2,073 gfx942
records is stale (digest `5b0e63ecbc2b0828` at 06cd7174, `963afaa3a1c58329`
here), so `pick_tile` has been falling back to the analytical model and
reporting tier `portable`. That is why the served A/B above is honest as it
stands: the shipped blob's tile choices come from the model, not from the store,
in both arms, and the `GM_SM_BK` change is invisible to the model. It also means
`scripts/rebench_tune_gemm_gfx942.sh` is owed a run against this implementation
before any of those cases can go green.

## Definitive sweep, everything landed (2026-09-08)

Adds the `GM_SM_BK` 64→128 k-tile — the change the `#ifndef` guard made
reachable for the first time — and the refreshed gfx942 tile store, on top of
the configuration in the table above. `+gfuse` is the same blob emitted with
`PLOW_PF_GFUSE=1`. Same corpus, client and lease discipline throughout; each
cell is **tokens/s / median TTFT ms / median TPOT ms**.

| Input / conc | Plow, start | Plow, final | Plow + gfuse | vLLM 0.28 | tok/s | TTFT | TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 / 1 | 30.79 / 91.8 / 31.53 | 41.86 / 74.1 / 23.09 | 41.77 / 72.0 / 23.17 | 55.31 / 45.9 / 17.63 | 0.76x | 0.64x | 0.76x |
| 128 / 4 | 85.95 / 341.7 / 40.20 | 90.13 / 297.0 / 39.01 | 90.35 / 267.0 / 38.96 | 180.39 / 109.2 / 20.79 | 0.50x | 0.41x | 0.53x |
| 512 / 1 | 29.63 / 156.4 / 31.80 | 39.49 / 147.2 / 23.39 | 39.44 / 144.5 / 23.47 | 52.32 / 81.9 / 18.11 | 0.75x | 0.57x | 0.77x |
| 512 / 4 | 78.31 / 506.3 / 42.18 | 81.04 / 481.5 / 41.14 | 80.14 / 494.6 / 41.07 | 158.74 / 270.7 / 21.29 | 0.51x | 0.56x | 0.52x |
| 2048 / 1 | 23.81 / 670.8 / 32.02 | 33.30 / 435.3 / 23.58 | 33.27 / 432.9 / 23.66 | 44.39 / 265.7 / 18.66 | 0.75x | 0.61x | 0.79x |
| 2048 / 4 | 49.46 / 2345.6 / 43.25 | 58.59 / 1213.2 / 48.74 | 58.17 / 1226.2 / 48.62 | 103.95 / 1050.8 / 22.40 | 0.56x | 0.87x | 0.46x |
| 8192 / 1 | 12.75 / 2974.0 / 32.44 | 20.11 / 1666.5 / 24.02 | 20.08 / 1666.7 / 24.12 | 25.58 / 1218.9 / 20.36 | 0.79x | 0.73x | 0.85x |
| 8192 / 4 | 18.92 / 10553.0 / 45.42 | 22.92 / 7756.4 / 52.89 | 23.05 / 7662.2 / 52.84 | 39.86 / 4254.3 / 33.92 | 0.58x | 0.56x | 0.64x |
| 16384 / 4 | — | 12.51 / 16656.3 / 58.64 | 12.59 / 16557.1 / 58.04 | 19.30 / 8477.3 / 74.90 | 0.65x | 0.51x | **1.29x** |
| 32768 / 1 | 3.45 / 16404.4 / 34.05 | 5.45 / 10122.5 / 25.74 | 5.46 / 10090.1 / 25.73 | 6.87 / 7827.6 / 23.61 | 0.80x | 0.78x | 0.92x |
| 122880 / 1 | 0.34 / 183744.8 / 116.08 | 0.74 / 85040.1 / 31.50 | 0.74 / 84933.5 / 31.52 | 0.89 / 69744.7 / 28.52 | 0.82x | 0.82x | 0.90x |

Ratios use whichever of the two Plow arms is better on that metric.

Over this branch: solo throughput **0.50-0.56x → 0.75-0.82x** of vLLM, solo TPOT
**0.55-0.69x → 0.76-0.92x**, solo TTFT **0.50-0.76x → 0.57-0.82x**. The single
largest TTFT move is at 128 tokens, 91.8 → 72.0 ms, of which the k-tile alone is
13.5 ms. Batched throughput stays 0.50-0.65x with the MFMA decode arm off.

**The goal — beat vLLM by a margin on every metric — is not met.** Plow leads on
TPOT at 16384/4 and 32768/4 and trails everywhere else.

## Runtime fusion re-measured on the final configuration (2026-09-08)

The earlier fusion table was taken before the prefill work landed. Re-running it
on the configuration this branch now ships — 8192 prefill chunk, budget-aware
tile pick, `GM_SM_BK=128`, batch-width-matched decode tiers — changes the
verdict. Same blob, same objects, one leased card, `PLOW_FUSION=1` against the
identical run with it off; the server logs `runtime prefill/decode fusion
enabled`, so the mixed step is live rather than silently refused.

| Input / conc | Fusion off | Fusion on | tokens/s on ÷ off |
|---|---:|---:|---:|
| 128 / 1 | 41.86 / 74.1 / 23.09 | 40.66 / 78.5 / 23.74 | 0.971x |
| 128 / 4 | 90.13 / 297.0 / 39.01 | **95.11 / 191.9 / 37.95** | **1.055x** |
| 512 / 1 | 39.49 / 147.2 / 23.39 | 38.19 / 152.6 / 24.17 | 0.967x |
| 512 / 4 | 81.04 / 481.5 / 41.14 | **83.03 / 509.7 / 40.31** | **1.025x** |
| 2048 / 1 | 33.30 / 435.3 / 23.58 | 31.73 / 450.5 / 24.86 | 0.953x |
| 2048 / 4 | 58.59 / 1213.2 / 48.74 | 54.38 / 1208.3 / 53.77 | 0.928x |
| 8192 / 1 | 20.11 / 1666.5 / 24.02 | 19.01 / 1703.8 / 26.39 | 0.945x |
| 8192 / 4 | 22.92 / 7756.4 / 52.89 | 18.88 / 9734.2 / 59.07 | **0.824x** |
| 16384 / 4 | 12.51 / 16656.3 / 58.64 | 11.05 / 18894.3 / 65.96 | 0.883x |

Fusion is now a win **only for short prompts at concurrency 4** — 128 tokens
gains 5.5% of throughput and 35.4% of TTFT, 512 gains 2.5% — and a loss
everywhere else, reaching −17.6% at 8192/4. The earlier +13.7% at 128/4 was
measured when prefill was slower; making the prefill faster shrank the work
fusion has to hide while leaving its cost in place.

The concurrency-1 rows are the more interesting signal: fusion should be inert
there, because the mixed step needs at least one decode and one prefill row to
have anything to fuse. They lose 2.9-5.5% of throughput and 0.7-2.4 ms of TPOT
consistently across four prompt lengths, which says **loading the mixed step
costs something even on the ticks that never use it** — the object is a
different `interp_mixed_gq.elf` build with its own tile and wave choices
(`GM_BM=64 GM_BN=128`, 4 waves), so a fused-capable server is running a
different decode body all the time.

**Therefore fusion should not be defaulted on as it stands.** The shape of the
result — a real win in one corner, a real loss in the rest, and a fixed cost on
every tick — is an argument for making the mixed step *selected per tick by the
scheduler* rather than *armed for the process*: engage it when the queue
actually holds a short prefill alongside live decodes, and run the ordinary
decode object otherwise. That is a scheduling change, not a kernel one.
## The decode residue, and the gate that was compiled in but inert (2026-09-08)

The MFMA section above bounds what is left after the projection GEMV: "even a
**free** decode GEMV could not take the step below 15.96 ms". That bound is
arithmetic on two separately-measured numbers, not an attribution. This section
measures the concurrency-4 decode step directly, spends the largest thing the
measurement finds, and records two packet-count nulls that the same instrument
explains.

Everything below is BF16, TP1, one leased MI300X, the 131072-context blob at
`PLOW_MAX_CHUNK=8192`, decode ladder 1/2/4, objects `build-gemma31/hsaco-tiered`
unless a row says otherwise. The control blob is re-emitted from this tree; it
reproduces the shipped blob's served numbers at every cell (128/4: 89.2 vs 90.1
tok/s, 8192/4: 22.8 vs 22.9), which is what makes it a valid A/B partner.

### The instrument: `--batched` never wrote a trace

`PLOW_TRACE_PHASE=1` splits each packet into claim+gate / acquire / body /
publish, and the prefill half of this document already uses it. The decode half
could not: `amd-bench --batched` is the fourth exit of `amd_bench` that returned
before `trace_dump_1`, so `PLOW_TRACE_RAW` produced **no file at all** for the
one shape a concurrency-4 question needs. That function's own doc comment was
written about exactly this failure mode on the other three exits. One call fixes
it; the trace buffer already holds the last (steady-state) launch.

Reducing it needs one more thing. The familiar per-op table is the MAX over a
packet's workgroups, summed over packets, and for decode it over-counts by 2.7x:

| op | pkts | wg | claim+gate | acquire | body | publish | us/pkt | ms raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemv | 411 | 178 | 48.6 | 0.2 | 103.8 | 3.8 | 156.4 | 64.27 |
| NormResidualNorm | 120 | 4 | 87.2 | 0.1 | 10.6 | 3.6 | 101.6 | 12.19 |
| FlashDecode | 60 | 304 | 27.4 | 0.2 | 77.5 | 5.8 | 110.9 | 6.65 |
| HeadNormRope | 180 | 16 | 13.2 | 0.1 | 17.8 | 4.7 | 35.8 | 6.45 |
| Glu | 60 | 21 | 86.2 | 0.1 | 14.2 | 5.7 | 106.2 | 6.37 |
| FlashMerge | 60 | 128 | 31.3 | 0.2 | 26.1 | 5.9 | 63.4 | 3.81 |
| **total** | **896** | | **41.83 ms** | 0.13 | **54.32 ms** | **3.91 ms** | | **100.20 ms** |

against a 36.93 ms step. Decode packets OVERLAP — q/k/v run concurrently on
disjoint CU sets, gate and up likewise, and every packet's gate wait runs under
its producer's body — so `scripts/gemma31_dec_phase.py` also walks the packets in
program order against a frontier, charges each one `end - max(frontier, first
arrival)`, and splits that contribution innermost-first across the closing
workgroup's publish, then body, then gate. Those columns sum to the step.

### Where the 36.93 ms goes

| op | pkts | gate ms | body ms | publish ms | total ms | share |
|---|---:|---:|---:|---:|---:|---:|
| Gemv | 411 | 0.247 | 25.338 | 0.707 | 26.291 | 71.2% |
| FlashDecode | 60 | 0.049 | 4.641 | 0.086 | 4.776 | 12.9% |
| FlashMerge | 60 | 0.056 | 1.558 | 0.139 | 1.753 | 4.7% |
| NormResidualNorm | 120 | 0.062 | 1.271 | 0.411 | 1.743 | 4.7% |
| HeadNormRope | 180 | 0.041 | 1.114 | 0.084 | 1.239 | 3.4% |
| Glu | 60 | 0.040 | 0.835 | 0.126 | 1.001 | 2.7% |
| Embed + RmsNorm + SoftCap + Argmax + ArgmaxFin | 5 | 0.005 | 0.109 | 0.012 | 0.126 | 0.3% |
| **total** | **896** | **0.499** | **34.866** | **1.564** | **36.930** | |

Three things this settles.

* **Launch/queue idle is 0.000 ms.** The whole step is one persistent kernel; the
  union of the packet spans covers the envelope exactly. There is no host-side or
  dispatch-side residue to find on the decode path, as there was none on prefill.
* **The per-packet protocol is 2.06 ms, 5.6% of the step** — not the 8-11 us x
  421 packets the raw table suggests. A packet's publish overlaps its successor's
  gate, so most of it is free. `publish` and `gate` are the only two columns any
  packet-protocol change can move, and together they are 2 ms.
* **The non-GEMV critical path is 10.6 ms, not 16.** The 15.96 ms bound above
  subtracts the *standalone* GEMV primitive (16.86 ms) from the served step;
  in-step the GEMV bodies are 25.3 ms of 36.9. The rest is attention (6.5 ms),
  norms and GLU (4.0 ms), protocol (2.1 ms) and the head (0.1 ms).

Keyed by weight tensor instead of by opcode (`gemma31_dec_phase_proj.py`), the
GEMV splits into the four SERIAL projection stages per layer — the concurrent
member of each pair clips to ~0 because its partner closes the packet:

| stage | pkts | wg | us/pkt | GB moved | effective |
|---|---:|---:|---:|---:|---:|
| gate\|up | 60 + 60 | 152 + 152 | 159.0 | 0.462 | 2.91 TB/s |
| down | 60 | 304 | 118.6 | 0.231 | 1.95 TB/s |
| q\|k\|v | 60 + 60 + 50 | 152 + 76 + 76 | 67.6 | 0.176 | 2.61 TB/s |
| o | 60 | 304 | 60.5 | 0.088 | 1.46 TB/s |

(placed arm, per layer; the ceiling measured on this part is 4112 GB/s.) The
remaining GEMV headroom is concentrated in the two large-K / small-N shapes,
`o_proj` (K=8192, N=5376) and `down_proj` (K=21504, N=5376), which run at 36% and
47% of the byte ceiling while the fat-N shapes reach 71%.

### The blob was never placed, so `PLOW_GATE_HIER` never ran

`scripts/build_gfx942.sh` has shipped `-DPLOW_GATE_HIER=1` on the decode
global-queue object by default, and `interp_decode_gq.elf` in `hsaco-tiered`
does advertise `plow_gate_hier_1`. `plowc` has defaulted `PLOW_L2_PLACE` on for
gfx942 for just as long. Neither reached the blob: **`build-gemma31/assets-final-plain`
starts `PLOWDEV\x09`, not `PLOWDEV\x0b`** — no program in it is L2-placed. The
hierarchy's runtime precondition is `prog.l2_domains != 0`, so the two-level
rendezvous compiled into the object was dead code on every number in this
document.

The cause is a default, not an oversight in the recipe. `PLOW_L2_PLACE_PREFILL`
also defaults on; the shipped PREFILL objects are built WITHOUT
`-DPLOW_L2_PLACE_DISPATCH` (gated behind `PLOW_L2HIER_PF`, off); and plowrt
correctly refuses a placed prefill program against them — the refusal this
document already records for the FP8 blob. The only way to emit a *servable*
blob was `PLOW_L2_PLACE=0`, which throws the decode half away with it.

So prefill placement is now **off by default on AMD**, matching the objects the
build script actually produces. `PLOW_L2_PLACE_PREFILL=1` still asks for it (pair
it with `PLOW_L2HIER_PF=1` objects); NVIDIA is untouched. A plain
`plowc --arch gfx942 --gpu MI300X` emit is now byte-identical to one made with
`PLOW_L2_PLACE=1 PLOW_L2_PLACE_PREFILL=0`, and carries `PLOWDEV\x0b`.

### Served result, and which half pays

Palindromic two-round A/B, same objects and client in both arms, three repeats
after one warmup per cell, 64 output tokens, medians of six:

| input / conc | ctl tok/s | placed tok/s | ctl TPOT | placed TPOT | TPOT |
|---|---:|---:|---:|---:|---:|
| 128 / 1 | 41.14 | 44.22 | 23.43 | 21.70 | **-7.4%** |
| 128 / 4 | 89.17 | 95.38 | 39.30 | 36.53 | **-7.0%** |
| 512 / 1 | 38.23 | 41.15 | 24.10 | 22.20 | -7.9% |
| 512 / 4 | 79.46 | 84.23 | 41.72 | 38.73 | -7.2% |
| 1024 / 1 | 35.76 | 38.52 | 24.30 | 22.27 | -8.4% |
| 1024 / 4 | 69.61 | 74.44 | 44.34 | 41.53 | -6.3% |
| 2048 / 1 | 32.17 | 34.34 | 24.40 | 22.44 | -8.0% |
| 2048 / 4 | 57.65 | 60.20 | 49.41 | 46.43 | -6.0% |
| 4096 / 1 | 26.88 | 28.37 | 24.65 | 22.65 | -8.1% |
| 4096 / 4 | 39.02 | 40.56 | 50.65 | 47.86 | -5.5% |
| 8192 / 1 | 19.49 | 20.24 | 24.82 | 22.99 | -7.4% |
| 8192 / 4 | 22.76 | 23.21 | 53.43 | 50.92 | -4.7% |

**IDENTITY.** 35 completion keys over two corpora (128/512/2048/8192 and
128/1024/4096, concurrency 1 and 4, 64 output tokens, four distinct prompts per
length at concurrency 4). Every key produces the same SET of completions in both
arms; 34 of the 35 are a single stable string, character-identical across arms
and repeats. The exception, `1024 tokens / concurrency 4 / request 3`, flips
between two continuations that diverge at character 4 (` a l l l...` vs
` a a a a...`) — **in both arms, between the same two strings**, across three
repeats. That is a pre-existing near-tie on a degenerate repeated-token prompt
under concurrency-4 scheduling, not a property of this change; it is recorded
here rather than filed away because it is the first time this corpus has been
checked for within-arm repeatability at all.

Which half of the pair pays is decidable now that a placed blob exists, and it
is not the half the -1.5% figure in `build_gfx942.sh` would suggest.
`PLOW_GATE_HIER=0` is a new A/B arm on the build script (objects otherwise
identical); `amd-bench --batched`, ctx 1024, 48 steps, two reps:

| blob | decode object | ms/token |
|---|---|---:|
| placed | `PLOW_GATE_HIER=1` | **33.94** |
| placed | `PLOW_GATE_HIER=0` | 36.55 |
| unplaced | `PLOW_GATE_HIER=1` (inert) | 36.90 |

**Placement alone is worth 0.9%; the two-level gate is worth 7.1%.** Placement's
job here is to make the hierarchy expressible, not to buy locality on its own.

And the saving lands in the op BODIES, not in the gate column — the phase table
goes `Gemv` body 25.338 -> 24.361, `FlashDecode` 4.641 -> 3.523, `FlashMerge`
1.558 -> 1.093, `Glu` 0.835 -> 0.559 ms, while the two protocol columns barely
move (gate 0.499 -> 0.761, publish 1.564 -> 1.451). That is the mechanism working
as designed rather than a contradiction: the ordinary gate has every
participating workgroup issue `buffer_inv`, a full L1+L2 invalidate, so a packet
with 304 workgroups throws its XCD's L2 away ~38 times over. Under the hierarchy
one workgroup per XCD does the L2 and the rest do L1 only. The cost of the
invalidate was never in the gate; it was in the next packet's body, re-fetching
what it had just discarded.

### NULL: fusing the T=4 decode GEMVs is a 9% LOSS

The T=4 decode program issues 411 separate `Gemv` packets and 60 `Glu`, where
the T=1 and T=2 rungs of the same blob issue 141 `Gemv` + 60 `GemvGlu` + 50
`GemvQkv`. The fusion audit recorded the reason as "emitter eligibility fails the
conservative shared-memory capacity bound", and the bound really is wrong:
`gemv_fused_input_fits` compares `M*hidden` against `hwspec`'s gfx942
`decode_gemm_tile`, which is the `PLOW_OCC4` / `PLOW_DEC_SQUEEZE` re-cut
(128x256x32, 15,360 halves), while a default decode object is built at
192x256x64 and its arena is 32,256 halves. At `hidden = 5376` that is the
difference between refusing T=3 and admitting T=6.

`PLOW_DEC_STAGE_HALVES` lets the emit state the object's real arena. It is
checked at both ends rather than asserted, because the fused bodies stage `M*K`
halves of `x` with no global-memory fallback: the object publishes its arena as
`plow_dec_stage_halves`, `check_dec_stage_capacity` compares every fused decode
GEMV's `M*K` against it at load, and the blob's `requires` refuses an object too
old to publish the number at all. (Verified: pairing the fused blob with
`hsaco-tiered` is refused with that message rather than served.)

Emitted at 32,256 the T=4 program drops **896 packets to 676, -24.6%**. It is
slower, reproducibly, and both halves lose independently — `amd-bench
--batched`, ctx 1024, 48 steps, two reps, all three blobs L2-placed and served by
the same object set:

| T=4 program | insts | ms/token |
|---|---:|---:|
| no fusion (411 Gemv + 60 Glu) | 896 | **34.45** |
| + fused gate\|up\|GLU (60 `GemvGlu`) | 776 | 35.98 (+4.5%) |
| + fused q\|k\|v (50 `GemvQkv`) | 676 | 37.60 (+9.2%) |

The critical-path table says why, and it is not the packet count. Per-projection,
placed -> fused:

| | placed | fused |
|---|---:|---:|
| `gate_proj` + `up_proj` + `Glu` | 9.541 + 0.045 + 0.924 = 10.510 ms | `GemvGlu` **11.508 ms** |
| `q_proj` + `k_proj` + `v_proj` | 4.053 + 0.005 + 0.004 = 4.062 ms | `GemvQkv` + 10 unfused **5.599 ms** |

The two deleted `Glu` and gate packets are worth 0.97 ms; the fused body costs
2.0 ms more than the pair it replaces. `gate_proj` at 152 workgroups moves half
the FF weights in 159.0 us; `GemvGlu` at 304 workgroups moves all of them in
191.8 us — identical bytes, identical CU count, **20.6% slower per byte**. In
the split form each workgroup streams ONE weight matrix over a wide column
range; in the fused form each streams TWO (or three, for QKV) over a range half
as wide, so the number of concurrent weight streams per CU doubles and each is
half as contiguous. This is the same verdict the fp8 QKV fusion recorded ("split3
12.080 vs fused 12.232") arrived at from the opposite direction, and it now has
a mechanism and a bf16 number. `PLOW_DEC_STAGE_HALVES` ships unset and
byte-identical; the fusion stays off at T=4 because it is slower, not because
the emitter cannot express it.

### What the residue is now

At 34.15 ms (traced) / 36.5 ms (served, 128/4) the step is:

| | ms | share |
|---|---:|---:|
| GEMV bodies | 24.36 | 71.3% |
| FlashDecode + FlashMerge | 4.62 | 13.5% |
| NormResidualNorm + HeadNormRope + Glu | 2.86 | 8.4% |
| per-packet protocol (gate + publish) | 2.21 | 6.5% |
| head (Embed, RmsNorm, SoftCap, Argmax, ArgmaxFin) | 0.10 | 0.3% |

The non-GEMV residue is **8.84 ms**, against the ~16 ms this campaign started
from. Ranked by what is left:

1. **GEMV bodies, 24.4 ms at 57% of the byte ceiling.** The MFMA arm removes the
   arithmetic wall; the shape-specific gap is `o_proj` and `down_proj` at 36-47%.
2. **Attention, 4.6 ms.** `FlashDecode` at 61.5 us/packet on 304 workgroups.
3. **The b=4 and b=21 packets, 2.9 ms.** `NormResidualNorm` runs 120 times on
   FOUR workgroups (12.9 us each, ~24 GB/s); `Glu` on 21. Both are latency-bound
   on their own workgroup count, and neither can widen without a cross-workgroup
   reduction that costs a packet.
4. **Protocol, 2.2 ms**, of which the two-level gate has already taken its share.

`PLOW_L2HIER_PF` — the same pair on the prefill objects — is NOT enabled by this
work and is not a free follow-up: Gemma's prefill program is split across the
prefill and flash objects, which is exactly the mixed-protocol shape
`build_gfx942.sh` records as hanging `amd-bench` for 680 s. It needs the flash
object in the pair and the hang test re-run.

The placed blob is published at `build-gemma31/assets-final-placed` rather than
written over `assets-final-plain`, which several concurrent A/Bs in this tree are
using as their control. It is what `plowc` now emits by default; re-emitting the
shipped bundle is the same command with no placement flags at all.

## Dense packed prefill: built, routable, and unreachable at the shipping chunk (2026-09-08)

Gemma 4 31B is dense/GQA, so co-packing several requests' prefill chunks into one
launch has always been the obvious way to spend a concurrent tick better. The
question asked here was what it would take to bring packed prefill to the dense
path. The answer is that it was already there, and that nothing about the feature
was stopping it.

### Everything the dense path needs was already built

- **Device.** `PLOW_PACKED_PREFILL_DENSE_CONSUMERS` gates the dense norm consumers
  (`runtime/amd/op_norm.h`) and a dense packed flash-prefill span path
  (`runtime/amd/interp.hip:1036`) that shares `PLOW_DENSE_SPAN` with the mixed step.
- **Objects.** `scripts/build_gfx942.sh` already passes
  `-DPLOW_PACKED_PREFILL_DENSE_CONSUMERS=1` in both `AX_PREFILL` (line 93) and
  `AX_FLASH` (line 210), so every prefill and flash object in
  `build-gemma31/hsaco-tiered` exports `plow_packed_prefill_dense_consumers_1`
  alongside the audited `plow_packed_prefill_abi_1`. There is no
  `interp_packed_dense` object and there does not need to be: unlike the MLA and
  KDA families, the dense consumers compile into the ordinary interpreter.
- **Host.** `check_packed_dense_program`, the `packed_dense` program field and the
  symbol check were all in place.
- **TP is not a requirement.** `advance_packed_prefill` dispatches through both
  `Ranks::One` and `Ranks::Tp`, and `amd_tp.rs:1517` implements packed prefill as
  an all-rank transaction. The `--amd-packed-prefill-route` doc comment claimed mux
  co-packing "requires ... a TP engine"; that was wrong, and that flag does not gate
  the dense path at all — it loads the MLA and KDA family objects only. Corrected.

`PLOW_PACKED_PREFILL_ROUTE` is therefore irrelevant to Gemma, and dense co-packing
needs only `--pf-batch` and two concurrent prefills.

### The load line, because a silent no-op has cost this branch twice

Nothing in the system said whether packing ran. A program that declines simply
returns `None` from `packable_prefill_span` and the mux falls back to isolated
prefill with no error anywhere. So the engine now names its verdict at load:

```
packed prefill routing dense_consumers=true packet_abi=true mla_norm=false
  mla_flash=false kda=false capable_rungs=[128, 512, 1024, 2048, 4096, 8192] refused="-"
```

Every rung on this blob is routable. The feature is not the constraint.

### What actually stops it is arithmetic, not capability

`sched::prefill::admit` packs **whole** spans into one compiled prefill rung. Two
requests can share a launch only if two chunks fit in one rung. The Gemma ladder
tops out at 8192 and the shipping recipe runs `PLOW_PF_CHUNK=8192`, so a single
chunk fills the widest rung, exactly one span is admitted, the mux's
`packed.len() >= 2` test fails, and packing can never run.

This is not a subtle effect. Over the whole 36-cell sweep:

| `PLOW_PF_CHUNK` | packed dispatches |
|---|---:|
| 8192 (shipping) | **0** |
| 2048 | 40 |
| 512 | 384 |

A `packed.len() >= 2` that never fires is indistinguishable from a feature that
does not help, which is why this is now pinned by
`sched::prefill::tests::a_chunk_filling_the_widest_rung_admits_one_span_and_half_of_it_admits_two`.

Note the direct consequence: **packing cannot help any prompt that fits in one
chunk.** At chunk 2048 only the 8192-token cells can pack at all.

The mechanism the win comes from is worth stating plainly, because it is the same
fact read the other way. `PLOW_PF_CHUNK` caps the chunk; it does not narrow the
compiled ladder. With `requested_max_chunk=2048` against
`buckets=[128, 512, 1024, 2048, 4096, 8192]`, four requests' 2048-row chunks run
together on the 8192 rung — one launch where there would have been four. Packing
pays exactly the launch-count arithmetic that `docs/arch/13-prefill-chunking.md`
identifies as the thing to minimise; it just needs the ladder to be wider than the
chunk, which the shipping recipe deliberately makes untrue.

### The second-order bug: packs capped at two members

Even with a workable chunk, packs came out at two or three members, never the four
the engine has slots for. Co-packing can only consider a slot that already has a prefill cursor, and
with fusion off the only thing that creates one is the isolated prefill path —
which the mux skips on any tick where a pack ran
(`amd_prefill_isolated_fallback`). A burst of N fresh requests therefore
bootstraps to a two-member pack and stops, because slots 3..N never get an
isolated tick in which to acquire a cursor.

`prepare_packed_prefill_slot` exists precisely to seed those cursors up front, but
it early-returned unless `mixed_step_rows(1, 1).is_some()` — i.e. unless runtime
fusion had synthesized a mixed-step program. That coupling is incidental: seeding
a cursor and peeling the terminal row are two different features and only the
second needs fusion. Splitting them lets packing reach its designed member count
with fusion off. The terminal split stays gated, because with fusion off
`terminal_prefill_ready` is false and a peeled 1-row chunk would only add an
isolated launch per request.

### Measured

One MI300X, TP1, BF16. Same blob (`build-gemma31/assets-final-plain`, 131072 ctx,
emit chunk 8192) and the same objects (`build-gemma31/hsaco-tiered`) in every arm —
no object was rebuilt, because none needed to be. Prefill L2 placement is off in
every arm regardless of `PLOW_L2_PLACE_DISPATCH=1` in `glm53_serve_inner.sh`:
`interp_prefill_gq.elf` carries no `plow_l2_place_dispatch_1`, so the pairing is
refused (see "Stop placing AMD prefill by default"). It is constant across arms and
does not affect any delta here. `bench_packed_serve.py`, 64
output tokens, 3 repeats after 1 warmup, medians. Each cell is
**tokens/s / median TTFT ms / median TPOT ms**.

This chunk study ran at `092e801d`, before prefill L2 placement was turned off by
default; a re-baselined table on the current tip follows it, and the two are not
directly comparable in absolute terms. The conclusions are drawn from the
re-baselined one; this table is what located the chunk that matters.

The "chunk 8192 (shipping)" column is the configuration of the "Definitive sweep,
everything landed" table above — `PLOW_PF_BATCH=1 PLOW_PF_CHUNK=8192 PLOW_MULTISTEP=4`
— and reproduces it (128/4: 90.13 there, 91.69 here; 8192/4: 22.92 there, 22.86 here).
Note this is NOT the `PLOW_PF_CHUNK=512` recipe printed under "Build and run"; that
one is older and is superseded below.

Arms differ only in `PLOW_PF_CHUNK` and `PLOW_PF_BATCH`; `PLOW_FUSION=0`
throughout this table. "+ packing" is `--pf-batch`; the bare chunk columns are the
same chunk with `--pf-batch` off.

| Input / conc | chunk 8192 (shipping) | chunk 2048 | chunk 2048 + packing | chunk 512 | chunk 512 + packing |
|---|---|---|---|---|---|
| 128 / 1 | 42.04 / 72.9 / 23.01 | 41.65 / 73.5 / 23.22 | 41.71 / 74.5 / 23.17 | 41.53 / 73.6 / 23.29 | 41.56 / 74.6 / 23.26 |
| 128 / 4 | 91.69 / 265.6 / 38.37 | 91.18 / 267.5 / 38.58 | 91.37 / 268.8 / 38.44 | 90.91 / 268.7 / 38.69 | 89.35 / 317.3 / 38.69 |
| 128 / 8 | 94.82 / 1568.4 / 39.92 | 93.59 / 1583.7 / 39.92 | 93.52 / 1585.2 / 40.09 | 93.07 / 1594.4 / 40.18 | 92.90 / 1594.9 / 40.19 |
| 512 / 1 | 39.02 / 149.0 / 23.66 | 38.93 / 149.4 / 23.72 | 38.75 / 150.3 / 23.82 | 38.74 / 150.7 / 23.83 | 38.76 / 150.6 / 23.81 |
| 512 / 4 | 82.03 / 476.3 / 40.63 | 80.50 / 499.8 / 40.79 | 81.45 / 481.2 / 40.89 | 81.41 / 454.8 / 40.92 | 81.16 / 486.2 / 41.03 |
| 512 / 8 | 84.29 / 1943.7 / 44.16 | 83.41 / 1936.6 / 43.98 | 83.96 / 1955.3 / 44.45 | 83.34 / 1938.3 / 44.04 | 83.65 / 1962.7 / 44.56 |
| 2048 / 1 | 32.88 / 441.5 / 23.89 | 32.72 / 445.2 / 23.98 | 32.72 / 442.7 / 24.02 | 29.95 / 625.5 / 23.99 | 29.83 / 628.6 / 24.07 |
| 2048 / 4 | 58.11 / 1276.8 / 48.32 | 58.61 / 1193.1 / 48.63 | 58.78 / 1187.7 / 48.50 | 44.56 / 1972.5 / 52.71 | 51.48 / 2213.5 / 42.04 |
| 2048 / 8 | 59.84 / 3274.1 / 58.51 | 59.54 / 3291.7 / 58.80 | 59.64 / 3284.6 / 58.72 | 47.18 / 4525.8 / 67.30 | 51.37 / 4712.8 / 45.67 |
| 8192 / 1 | 19.71 / 1704.1 / 24.47 | 17.91 / 2035.9 / 24.38 | 17.97 / 2022.7 / 24.39 | 14.77 / 2788.6 / 24.56 | 14.75 / 2797.4 / 24.49 |
| 8192 / 4 | 22.86 / 7745.1 / 53.12 | 22.35 / 5490.8 / 87.47 | 24.54 / 6982.1 / 52.96 | 15.12 / 8643.2 / 101.39 | 19.89 / 9952.9 / 44.35 |
| 8192 / 8 | 22.88 / 10547.6 / 86.53 | 22.94 / 10915.8 / 135.61 | 24.28 / 11754.7 / 64.56 | 15.70 / 16479.3 / 121.73 | 19.93 / 16361.6 / 47.86 |

Three things fall out.

1. **Concurrency 1 never moves with packing** (8192/1: 17.91 → 17.97 tokens/s), which
   is the control working: one request in flight has nothing to pack with.
2. **Packing only pays where it can fire.** At chunk 2048 that is the 8192-token
   cells alone; every shorter prompt is a single chunk. At chunk 512 the 2048-token
   cells join in.
3. **The chunk reduction costs more than packing returns, except at 8192 tokens.**

Isolating the packing delta at matched chunk — the only comparison in which packing
is the single variable:

| Cell | chunk 2048 → +packing | chunk 512 → +packing |
|---|---|---|
| 2048 / 4 | — (single chunk) | **+15.5%** tok/s, TPOT −20.2%, TTFT +12.2% |
| 2048 / 8 | — (single chunk) | **+8.9%** tok/s, TPOT −32.1%, TTFT +4.1% |
| 8192 / 4 | **+9.8%** tok/s, TPOT −39.4%, TTFT +27.2% | **+31.5%** tok/s, TPOT −56.3%, TTFT +15.2% |
| 8192 / 8 | **+5.8%** tok/s, TPOT −52.4%, TTFT +7.7% | **+26.9%** tok/s, TPOT −60.7%, TTFT −0.7% |

Packing is a real and large effect — up to +31.5% throughput and −60.7% TPOT — but
it buys throughput and inter-token latency with time-to-first-token, and it can only
be reached by shrinking a chunk whose size is itself worth 1.5-2x of TTFT above 2048
tokens ("The chunk is worth 1.5-2x of TTFT above 2048 tokens", above). Against the
shipping `PLOW_PF_CHUNK=8192` arm, only two cells in the whole sweep come out ahead:
8192/4 (22.86 → 24.54, **+7.3%**) and 8192/8 (22.88 → 24.28, **+6.1%**). Against
that, 8192/1 regresses 19.71 → 17.97 (**−8.8%**), purely from the smaller chunk.

Each arm is one server, started with `scripts/glm53_serve_inner.sh` inside
`nix develop` under a `gpulease -n 1`, then:

```bash
python3 scripts/bench_packed_serve.py --url http://127.0.0.1:$PORT \
  --out $ARM.jsonl --label $ARM --inputs 128 512 2048 8192 \
  --outputs 64 --concurrency 1 4 8 --repeats 3 --warmups 1
```

Whether packing actually ran is read back from the server log — `grep -c 'AMD packed
prefill advanced'` for the count and `grep -o 'spans=[0-9]*'` for the member sizes.
Do not infer it from the numbers.

### `--pf-batch` without packing is a pure loss

`--pf-batch` does two things: fair round-robin prefill admission, and co-packing.
At the shipping chunk only the first is reachable, and it is not free:

| Cell | chunk 8192, no `--pf-batch` | chunk 8192, `--pf-batch` |
|---|---|---|
| 8192 / 4 | 23.05 / 5172.8 / 87.03 | 22.86 / **7745.1** / 53.12 |
| 8192 / 8 | 23.64 / 10430.6 / 133.67 | 22.88 / 10547.6 / 86.53 |

Round-robin admission trades TTFT for TPOT at long prompts and gains no throughput.
So `--pf-batch` is worth enabling only at a chunk where packing can actually fire.

### Fusion vs packing vs neither

Packed prefill and the runtime mixed step are mutually exclusive per dispatch —
`amd_mixed_step.rs` rejects a mixed request while a packed binding is live — but not
per server: the fusion arm below ran 215 mixed launches AND 25 packed ones across
different ticks. What a deployment really picks is a chunk, and the chunk decides
which of the two it gets. Measured at the same shapes:

| Input / conc | neither | fusion (chunk 8192) | packing (chunk 2048) |
|---|---|---|---|
| 128 / 1 | 41.12 / 74.8 / 23.52 | 41.14 / 74.5 / 23.50 | 41.16 / 74.7 / 23.49 |
| 128 / 4 | 89.51 / 273.5 / 39.27 | 93.72 / 186.2 / 39.18 | 88.96 / 298.2 / 39.17 |
| 128 / 8 | 91.95 / 1613.7 / 40.65 | 94.23 / 1626.0 / 39.25 | 91.13 / 1661.2 / 41.02 |
| 512 / 1 | 38.25 / 150.9 / 24.16 | 38.23 / 151.4 / 24.17 | 38.29 / 151.1 / 24.13 |
| 512 / 4 | 80.55 / 458.3 / 41.39 | 83.46 / 387.9 / 41.35 | 80.50 / 484.6 / 41.40 |
| 512 / 8 | 82.43 / 1957.5 / 44.50 | 80.80 / 2118.8 / 43.82 | 82.08 / 2009.6 / 44.95 |
| 2048 / 1 | 32.32 / 446.5 / 24.34 | 32.23 / 446.7 / 24.42 | 32.30 / 447.1 / 24.36 |
| 2048 / 4 | 58.13 / 1194.9 / 49.13 | 53.64 / 1225.0 / 54.51 | 57.22 / 1291.9 / 49.15 |
| 2048 / 8 | 59.14 / 3305.6 / 59.20 | 50.96 / 3852.3 / 59.83 | 58.85 / 3350.2 / 59.54 |
| 8192 / 1 | 19.57 / 1705.1 / 24.82 | 19.55 / 1703.8 / 24.88 | 17.76 / 2040.0 / 24.80 |
| 8192 / 4 | 22.78 / 5219.6 / 88.21 | 18.71 / 9800.9 / 59.82 | 25.24 / 6649.2 / 53.59 |
| 8192 / 8 | 23.26 / 10582.6 / 135.78 | 18.03 / 13682.2 / 117.29 | 25.08 / 11787.9 / 62.12 |

They are near mirror images.

- **Fusion is a short-prompt win and a long-prompt loss.** 128/4: 89.51 → 93.72
  tokens/s (+4.7%) with TTFT 273.5 → 186.2 ms (−31.9%); 512/4: +3.6% with TTFT
  −15.4%. But 2048/8 falls 59.14 → 50.96 (−13.8%) and 8192/4 falls 22.78 → 18.71
  (−17.9%).
- **Packing is a long-prompt win and short-prompt neutral-to-slightly-negative.**
  Nothing below 8192 tokens gains; 8192/4 rises 22.78 → 25.24 (+10.8%) with TPOT
  88.21 → 53.59 (−39.2%).

So the two features do not compete for the same workload, and a deployment does not
have to choose in the abstract — it chooses by prompt length. Under ~2048 tokens,
fusion. At 8192 tokens and concurrency ≥ 4, packing. Neither helps at concurrency 1.

Turning both on is worse than either. Measured at `092e801d` (the chunk study's
base): at chunk 2048 with `PLOW_FUSION=1` the server ran 87 mixed launches and 53
packed ones, and the mixed launches displace packed ones
while carrying fusion's long-prompt penalty:

| Input / conc | packing only | packing + fusion |
|---|---|---|
| 2048 / 8 | 59.37 / 3322.9 / 59.01 | 49.65 / 4147.6 / 69.47 |
| 8192 / 4 | 25.42 / 6601.2 / 53.24 | 24.82 / 6480.1 / 59.02 |
| 8192 / 8 | 25.32 / 11669.8 / 61.59 | 22.84 / 12119.2 / 69.46 |

Pick one.

One detail worth recording: the fusion arm logged 25 packed dispatches alongside its
215 mixed launches. That is the terminal split at work — peeling the last row turns a
single-chunk 128-token prompt into a two-chunk plan, and two 127-row chunks fit a 512
rung. Fusion is currently the only way a short prompt becomes packable at all.

Fusion's own margin has shrunk as the branch improved. The 2026-09-07 measurement
("Runtime fusion on/off, current schedule builder") recorded +13.7% throughput and
−40.8% TTFT at 128/4; against the current tip, with the GEMM k-tile and the
batch-width-matched decode objects landed, the same cell reads **+4.7% and −31.9%**.
The baseline moved, not the feature.

### Cursor seeding, measured

With the seeding fix, packs reach their designed member count. Same chunk, same
binary otherwise; pack-size histogram over the 8192-token cells:

| pack size | 2 members | 3 | 4 |
|---|---:|---:|---:|
| before seeding | 32 | 8 | **0** |
| after seeding | 6 | 4 | **25** |
| after seeding, re-baselined | 8 | 3 | **25** |

Four physical slots, so four members is the ceiling. And it shows up in the served
numbers — this is the only configuration in the sweep that beats the `--pf-batch`
arm on throughput and TTFT at once:

| Input / conc | chunk 8192 (shipping) | chunk 2048 + packing | + cursor seeding |
|---|---|---|---|
| 8192 / 1 | 19.71 / 1704.1 / 24.47 | 17.97 / 2022.7 / 24.39 | 17.88 / 2023.6 / 24.68 |
| 8192 / 4 | 22.86 / 7745.1 / 53.12 | 24.54 / 6982.1 / 52.96 | **25.42 / 6601.2** / 53.24 |
| 8192 / 8 | 22.88 / 10547.6 / 86.53 | 24.28 / 11754.7 / 64.56 | **25.32** / 11669.8 / **61.59** |

Against the shipping arm that is **+11.2% throughput with TTFT −14.8%** at 8192/4,
and **+10.7% throughput with TPOT −28.8%** at 8192/8. Against the no-feature arm
(`--pf-batch` off, chunk 8192) it is +10.3% throughput and −38.8% TPOT at 8192/4,
but TTFT is 27.6% worse — round-robin admission, not packing, is what costs that.

8192/1 still sits 9.3% below the shipping arm. That is the chunk, not the feature:
one request cannot pack, so it pays the smaller chunk and receives nothing back.

Seeding is not free, though, and it had to be gated. The slot pays its state clear
and chunk planning in the arrival tick rather than spread across ticks, and where no
pack can form there is nothing to repay it. Measured at `PLOW_PF_CHUNK=8192`, where
packing is impossible, seeding unconditionally cost 1.3-2.8% of short-prompt
throughput:

| Input / conc | `--pf-batch`, no seeding | + unconditional seeding |
|---|---|---|
| 128 / 4 | 91.69 / 265.6 / 38.37 | 90.30 / 297.3 / 38.93 |
| 128 / 8 | 94.82 / 1568.4 / 39.92 | 92.20 / 1651.5 / 40.70 |
| 512 / 8 | 84.29 / 1943.7 / 44.16 | 82.34 / 1995.9 / 44.59 |
| 8192 / 4 | 22.86 / 7745.1 / 53.12 | 23.02 / 7683.3 / 53.16 |

So seeding is gated on the same arithmetic that decides whether packing can run at
all — two chunks must fit one rung. At chunk 8192 it is skipped and the cost is
gone; at chunk 2048 it runs and packs reach four members. The fusion path keeps
seeding unconditionally, because it needs the cursor for `finish_prefill_batch`
whether or not a pack forms.

### Re-baselined on the current tip

The chunk study above ran at `092e801d`. Re-run at `acc02f3c`, after prefill L2
placement was turned off by default, with all four arms on one binary. Absolute
throughput is 0.5-2.7% lower across the board on this base; the deltas between arms
are what matters and they hold.

| Input / conc | neither | chunk 8192 + --pf-batch | chunk 2048 + packing |
|---|---|---|---|
| 128 / 1 | 41.12 / 74.8 / 23.52 | 41.57 / 74.7 / 23.25 | 41.16 / 74.7 / 23.49 |
| 128 / 4 | 89.51 / 273.5 / 39.27 | 90.13 / 270.9 / 39.03 | 88.96 / 298.2 / 39.17 |
| 128 / 8 | 91.95 / 1613.7 / 40.65 | 92.32 / 1605.4 / 40.45 | 91.13 / 1661.2 / 41.02 |
| 512 / 1 | 38.25 / 150.9 / 24.16 | 38.53 / 152.0 / 23.94 | 38.29 / 151.1 / 24.13 |
| 512 / 4 | 80.55 / 458.3 / 41.39 | 80.71 / 482.1 / 41.33 | 80.50 / 484.6 / 41.40 |
| 512 / 8 | 82.43 / 1957.5 / 44.50 | 82.24 / 2005.5 / 44.85 | 82.08 / 2009.6 / 44.95 |
| 2048 / 1 | 32.32 / 446.5 / 24.34 | 32.46 / 443.9 / 24.26 | 32.30 / 447.1 / 24.36 |
| 2048 / 4 | 58.13 / 1194.9 / 49.13 | 57.64 / 1236.7 / 49.06 | 57.22 / 1291.9 / 49.15 |
| 2048 / 8 | 59.14 / 3305.6 / 59.20 | 59.17 / 3301.5 / 59.23 | 58.85 / 3350.2 / 59.54 |
| 8192 / 1 | 19.57 / 1705.1 / 24.82 | 19.59 / 1699.3 / 24.85 | 17.76 / 2040.0 / 24.80 |
| 8192 / 4 | 22.78 / 5219.6 / 88.21 | 22.72 / 7765.9 / 53.74 | 25.24 / 6649.2 / 53.59 |
| 8192 / 8 | 23.26 / 10582.6 / 135.78 | 22.70 / 10613.6 / 87.47 | 25.08 / 11787.9 / 62.12 |

Packing fired 36 times over the 8192-token cells, 25 of them at the full four
members. Reading the columns at 8192 tokens:

| | vs `--pf-batch` at chunk 8192 | vs neither |
|---|---|---|
| 8192 / 4 | **+11.1%** tok/s, TTFT **−14.4%**, TPOT −0.3% | +10.8% tok/s, TPOT **−39.2%**, TTFT +27.4% |
| 8192 / 8 | **+10.5%** tok/s, TPOT **−29.0%**, TTFT +11.1% | +7.8% tok/s, TPOT **−54.2%**, TTFT +11.4% |
| 8192 / 1 | −9.3% tok/s, TTFT +20.0% | −9.2% tok/s, TTFT +19.6% |

That is the whole result. Co-packing is worth about 11% of throughput and up to 54%
of TPOT at 8192-token prompts with four or more concurrent requests, and it costs
nothing anywhere: the concurrency-1 and sub-2048 rows above are the price of the
smaller chunk, which is the PRECONDITION for packing rather than an effect of it.
The matched-chunk comparison isolates that — 19.59 against 19.57 tokens/s at
8192/1, packing on against off.

The consequence for design: the cost is attached to `PLOW_PF_CHUNK` being a global
emit-time setting, not to co-packing. A scheduler that admitted two half-rung spans
when two requests are queued, instead of requiring every prompt to be chunked
smaller, would take the 8192/4 win without charging the solo case for it. That is
the same conclusion runtime fusion reached from the other direction: select per
tick, do not arm the process.

### Identity

Two corpora, because neither one covers both of the conditions that matter.

**Raw corpus — the only one that actually witnessed packing.** Three prompts at each
of 128 / 512 / 2048 / 8192 input tokens, 64 greedy tokens each, at concurrency 1 and
4; 24 completions per arm. The 8192-token concurrency-4 case logged 3 packed
dispatches, which is full coverage of it: four requests of four chunks each, the
first three chunks of all four packed together, the terminal chunk isolated.

| comparison | result |
|---|---|
| chunk 2048, packing off vs on | **24/24 character-identical** |
| chunk 8192 vs chunk 2048 + packing | **24/24 character-identical** |

Packing does not reorder arithmetic. The second row is a stronger claim than it
looks: the chunk plan differs between those two arms, and §"Acceptance class" above
notes a different plan is only *usually* text-identical, because the determinant is
which bucket runs last. Here it was identical anyway.

**Chat corpus — stronger text, but it does not witness packing.** These raw prompts
are a repeated word sent with `add_special_tokens=false`, and
`scripts/gemma4_greedy_quality.py` documents that this collapses the
instruction-tuned checkpoint into single-token repetition within a few tokens. A very
stable argmax can mask a small logit change. So the same off/on comparison was run
through that script's `chat` mode, which builds prompts from natural technical prose
and generates ordinary text:

```
len=  128 agreement=1.0000 identical_text=3/3
len=  512 agreement=1.0000 identical_text=3/3
len= 2048 agreement=1.0000 identical_text=3/3
len= 8192 agreement=1.0000 identical_text=3/3
overall token agreement 1.0000
```

**That run logged zero packed dispatches**, because `gemma4_greedy_quality.py` is
sequential and co-packing needs two prefills in flight. So it is an A/A with respect
to packing: it establishes that the chunk-2048 configuration is deterministic on
non-degenerate text, and nothing more.

Stated plainly: the only direct evidence that co-packing preserves output is the raw
corpus at concurrency 4. It is unambiguous as far as it goes — 24/24, byte for byte,
with packing demonstrably firing — but a natural-prose corpus driven at concurrency
≥ 4 with exact prompt lengths does not exist yet and would be the stronger test.

### Recommendation: do not flip the default

Dense packed prefill works, and where it fires it is worth a lot. It is still not a
default, for three reasons that the sweep makes concrete.

1. **It is unreachable at the chunk this model ships with.** Making it reachable
   means halving `PLOW_PF_CHUNK`, and that chunk is worth 1.5-2x of TTFT above 2048
   tokens. The trade only comes out positive at 8192-token prompts.
2. **Reaching it regresses the solo long-prompt case by 9.3% — the chunk, not the
   feature.** At a matched chunk the packing arm and the no-packing arm are within
   0.1% at 8192/1 (19.59 against 19.57 tokens/s), which is what must happen: one
   request can never satisfy `packed.len() >= 2`. The whole 9.3% is `PLOW_PF_CHUNK`
   2048 against 8192, and the chunk-2048 arm loses it with packing OFF (−9.2%) as
   readily as ON (−9.3%). So the bet on the table is not "solo throughput for
   concurrent throughput" — it is "give up the 8192 chunk for everyone in order to
   make packing reachable for the concurrent case".
3. **Nothing at or below 2048 input tokens gains**, because those prompts are a
   single chunk and a single chunk cannot co-pack — and they lose 0.7-1.3% to the
   smaller chunk and to seeding they cannot use.

`--pf-batch` therefore stays off by default, and `PLOW_PF_CHUNK` stays at the emit
chunk. What changes is that the configuration is now documented, observable and
reachable rather than silently inert:

```bash
# Long-prompt, concurrent serving (>= ~8192 input tokens, concurrency >= 4):
PLOW_PF_BATCH=1 PLOW_PF_CHUNK=2048 ...

# Short-prompt, concurrent serving (<= ~2048 input tokens):
PLOW_FUSION=1 ...

# Anything else, including concurrency 1: neither.
```

The shipping recipe under "Build and run" above used `PLOW_PF_CHUNK=512` alongside
`PLOW_PF_BATCH=1`. That pairing is worse than both alternatives on this branch: with
packing on, chunk 512 still runs 11-14% below the shipping arm at 2048 and 8192
tokens (58.11 → 51.48 and 22.86 → 19.89 tokens/s at concurrency 4), and 21.8% below
chunk 2048 + seeding at 8192/4. Read it as superseded by the table above.

No object needs rebuilding for any of this. The dense consumers have been in every
`interp_prefill*` and `interp_flash*` object all along.

### What remains

- **Packing is capped by the physical slot count, not by the rung.** The engine runs
  four KV slots, so a pack never exceeds four members even at concurrency 8, where
  the rung has room for four 2048-row spans and the queue has eight requests waiting.
  The 8192/8 cells gain less than 8192/4 for exactly this reason. Widening the slot
  count is a separate change with its own KV cost.
- **Short prompts stay unpackable without fusion.** A prompt that fits in one chunk
  has one chunk, and `packable_prefill_step` requires at least two. Applying the
  terminal split unconditionally would create a second chunk and make short prompts
  packable, but it also costs an isolated 1-row launch per request whenever
  `finish_prefill_batch` is unavailable. Whether the pack pays for that launch is
  unmeasured.
- **The TTFT cost is round-robin admission, not packing.** `--pf-batch` couples fair
  admission to co-packing, and at 8192/4 the fair-admission half alone moves TTFT
  from 5172.8 to 7745.1 ms with no throughput to show for it. Separating the two
  knobs would let a deployment take packing without the fairness trade.
- **The identity corpus that witnesses packing is the weak one.** The raw repeated-word
  corpus runs at concurrency 4 and saw packing fire; the natural-prose corpus is
  sequential and did not. A prose corpus with exact prompt lengths driven at
  concurrency >= 4 would settle it properly.
- **The seeding gate is per-engine, not per-request.** It compares the configured
  chunk against the widest rung, so at chunk 2048 it seeds every arrival — including
  128-token prompts that are one chunk and can never pack. That is most of the
  0.7-1.3% short-prompt cost still visible in the packing column. Gating on
  `prompt.len() > chunk` as well is a one-line change; it is not made here because it
  is unmeasured.
- **Only gfx942 / Gemma 4 31B was measured.** The dense consumers are in every
  gfx942 prefill and flash object, so any dense BF16 blob on this arch should behave
  the same way, but nothing else was run.


---

## Gemma 4 31B MI300X: why runtime fusion still loses to vLLM

The dominant deficit is projection/activation execution, followed by attention
and prefill scheduling. Fusion removes a launch boundary but preserves separate
decode and prefill projection calculations. Most tokens in this benchmark are
generated after prefill has finished, so mixed execution cannot improve them.

## After the terminal-prefill fix

The runtime now initializes cold packed cursors and completes terminal prompt
rows through batched decode. The
24-run before/after comparison (raw artefact removed; see the tables above)
holds GPU objects, BF16 precision, PF512 and multistep 4 fixed. At concurrency 4,
throughput improves 9–10%; median TTFT falls 49%, 32% and 14% for prompt lengths
128, 1024 and 4096. The prior terminal-token serialization was a real bottleneck,
but fixing it leaves steady decode projections and attention unchanged.

| Input tokens / concurrency 4 | New Plow tokens/s | Earlier vLLM tokens/s | New Plow TTFT, ms | Earlier vLLM TTFT, ms |
|---|---:|---:|---:|---:|
| 128 | 100.06 | 195.91 | 279 | 114 |
| 1024 | 80.29 | 158.05 | 1195 | 559 |
| 4096 | 50.91 | 100.52 | 4793 | 2332 |

This juxtaposes the new three-repetition Plow study with the earlier
five-repetition vLLM scorecard; it is not a fresh paired backend comparison.
Checkpoint, corpus, precision and output counts match. Clocks are unpinned.
Full output texts are not universally identical across phase schedules.

The updated server log accounts for all 512 output tokens in each of 12 runs.
Across nine measured runs, only 24 of 4572 decode advances occur inside mixed
prefill/decode launches (0.525%). Terminal batches produce another 24 decode
advances; the remaining 4524 use ordinary decode. At least 98.43% of decode
advances occur after the final request receives its first token. These are
sampled request rows, not GPU launch counts. This closed-loop workload gives
fusion little opportunity to affect steady throughput; staggered or saturated
arrivals require a separate measurement.

The completed 48-run multistep 1 vs 4 study identifies a separate delivery cost.
Multistep 4 returns tokens in bursts: p95 SSE-event gaps are about 114–116 ms at
concurrency 1 and 75–80 ms at concurrency 4. Multistep 1 reduces those to 30 ms and
39–42 ms, respectively, while observed throughput falls 1.55–3.18%. All 96 paired
completion texts match exactly, with 192 requests and 24576 output tokens
accounted for. Configured multistep 4 uses actual quantum 2 at occupancy 3–8.
These delivery intervals are not isolated GPU-token timings.

Post-terminal scheduler and multistep evidence (raw artefact removed; see the tables above)
includes window accounting and verifies the new server uses the same four
projection/prefill/mixed object hashes as the native trace below. The native
GPU timings remain the earlier captures; the scheduler evidence is new.

## Serving result

The stock `vllm bench serve` scorecard (raw artefact removed; see the tables above)
uses source `5feeb384`, one separately leased MI300X per backend, the same Gemma
4 31B checkpoint/tokenizer, BF16 weights/activations/KV, TP1, greedy sampling and
128 output tokens. Both token budgets are nonbinding; prefix caching is disabled.
Plow enables runtime fusion, packed prefill, multistep 4, decode ladder 1/2/4,
the WPE5 MM1 object and decode L2 placement. vLLM 0.28.0 uses compilation/graphs
and the verified tanh-GELU operation. These serving results exclude profiling.

For 1024 input tokens and concurrency 4, Plow produces 72.99 output tokens/s vs
158.05 for vLLM. Median TTFT is 1757 vs 559 ms; TPOT is 40.53 vs 21.11 ms.
The six-cell matrix favors vLLM throughout. Separate GPUs and unpinned clocks
limit the precision of small differences; this is not a saturation benchmark.

## Decode attribution

Plow's existing GPU packet timestamps were captured using the shipping objects
and exact benchmark assets. A private diagnostic host build supplies a separate
mixed trace buffer; it does not change the GPU kernels. Three captures per shape
cover D1/D2/D4 at contexts 128 and 4096, prefill 128/512/1024, and mixed capacities
512/1024 with one decode and three prefill requests.

vLLM's Torch profiler captures the same fixed corpus at 128/C1 and 4096/C4.
Its short trace contains 508 decode annotations; its long trace contains 126
four-request decode annotations, one three-request decode annotation, one prefill
annotation and one mixed annotation. A completeness audit finds all expected core
GPU records in 507 short iterations and 94 long four-request iterations. The
table uses only those complete iterations for vLLM. Exclusion is based on kernel
inventory, not speed; all benchmark repetitions remain in the serving scorecard.

| GPU work, ms | Plow D1, context 128 | vLLM C1, contexts 129–255 | Plow D4, context 4096 | vLLM D4, long-prompt run |
|---|---:|---:|---:|---:|
| Full step envelope | 27.47 | 17.81 | 39.25 | 21.99 |
| Projection + activation | 22.66 | 14.62 | about 28.90 | 16.05 |
| Attention + merge | 2.35 | 0.81 | 6.22 | 3.32 |
| Recorded Plow dependency gate | 0.79 | — | 1.05 | — |

Projection plus activation accounts for roughly 8 ms of the short decode gap
and 13 ms of the long batched gap, about 83% and 74% of the respective envelope
differences. Plow's fused `GemvGlu` already includes the
activation; vLLM's separate GELU kernel is included here for a fairer grouping.
The vLLM trace identifies its custom ROCm `wvSplitK_hf_sml` skinny-matrix kernel
as the dominant decode projection implementation. Prefill uses hipBLASLt kernels.
The [v0.28.0 ROCm kernel source](https://github.com/vllm-project/vllm/blob/v0.28.0/csrc/rocm/skinny_gemms.cu)
uses LDS-staged activations, a persistent output-row walk and BF16 4x4x4 MFMA
accumulation for this path. It is a vLLM custom operation, not a rocBLAS kernel.

Plow's D4 projection time grows to about 28 ms while vLLM's projection time is
about 15.26 ms before activation. Long-context attention adds another
approximately 2.9 ms disadvantage.
The dedicated MM1 object helps solo decode, but does not address the batched path.
Within Plow's B4 projection envelope, gate/up accounts for 10.95 ms and down
projection for 6.43 ms; the output head contributes only 0.96 ms. This makes
gate/up a higher-value first target than head-only tuning.

Plow's isolated host call exceeds the measured GPU envelope by only about
0.14–0.15 ms for decode. Recorded dependency gates consume about 3% of the
envelope. Eliminating those waits would not close the observed gap. These gates
do not account for every interpreter instruction, and packet bodies include
publication work, so this is not a proof that interpreter overhead is zero.
A separate HSA-only trace of the unmodified server confirms that completion/copy
waits cover 98.0% of the short C1 serving duration and 98.5% of the long C4
duration. These are blocking waits, not a measurement of pure arithmetic; other
HSA API time and copy-event sums overlap and must not be added as exclusive host
costs. They support prioritizing GPU execution over HTTP or host launch overhead.

## Attention work distribution

The long B4 attention gap has two substantial parts. Across 50 sliding layers,
Plow's attention core takes about 3.20 ms vs 2.30 ms in vLLM. Across ten full
attention layers, it takes 1.71 ms vs 0.65 ms. The separate merge passes add
1.32 ms vs 0.37 ms. These independently aggregated shape values need not sum
exactly to the total above.

Full attention with head dimension 512 is especially expensive: about 171
microseconds per layer in Plow vs 65 in vLLM. Plow's current grouped decode
path defaults to two of the eight query heads sharing a KV head per group; vLLM
handles that whole query-head group in one program. Plow uses vector arithmetic
for dot products and FP32 probability/value accumulation, while vLLM's Triton
path uses matrix dot operations and casts probabilities to BF16 before the
value product. Plow also uses ten fixed splits in this B4 program, while the
installed vLLM source selects sixteen segments. The vLLM trace does not record
those launch arguments; layer-shape attribution follows the model's ordered
60-layer execution, with complete call counts checked.

These are concrete differences to test, not proof of a bandwidth or occupancy
cause. Group reuse and merge work are useful targets; changing dot-product or
probability precision needs full-logit qualification. Copying the faster
arithmetic without that check would repeat the rejected projection experiment.

## What fusion actually saves

The benchmark server log accounts for 512 output tokens in every run: four
first tokens from prefill and 508 decode request advances. Concurrency-1 runs
contain no mixed launches. Each concurrency-4 run contains one mixed launch for
128-token prompts or two for 1024/4096-token prompts. Across the 15 measured C4
runs, only 25 of 7620 decode request advances occur inside a mixed launch
(0.328%). About 493–495 of 508 advances per run occur after the last prefill.
These are request advances, not GPU launch counts: multistep decode produces
several advances per launch.

The implementation in `runtime/amd/interp.hip`, `exec_gemm` and
`exec_mixed_gemm_glu`, first executes GEMV on the decode prefix, synchronizes,
then executes GEMM on the prefill suffix. Both paths access the projection
weights. This preserves each phase's qualified BF16 arithmetic, but does not
provide a single shared matrix operation over every packed row. Actual HBM
traffic saved or reread has not been measured with hardware counters.

The mixed T1024/D1/P3 diagnostic spends approximately 334 of 398 ms in
projection/activation packets, with only 4.4 ms in recorded gates. Ordinary
prefill T512 spends about 124 of 158 ms in projection packets. These shapes have
different row counts and attention spans; their totals are not a matched
fusion-on/off performance comparison. They locate the work that needs tuning.

## Earlier prefill completion serialized first-token delivery

This section describes source `5feeb384`, before the terminal-prefill fix above.

In the first measured 128/C4 run, requests arrive within 0.55 ms of one another,
but their TTFTs are 98.56, 476.75, 631.70 and 785.80 ms. The server log shows:

1. The first ordinary prefill completes, followed by a four-token decode quantum
   taking about 109 ms.
2. One mixed launch processes prefixes for the other three requests alongside
   one decode row; the next log event is about 190 ms later.
3. Their first tokens complete separately, with about 80 ms of ordinary prefill
   and about 75 ms of intervening decode between successive completions.

These event intervals are wall times, not isolated GPU durations. The earlier code
explains the pattern: fresh requests had no packable ordinary cursor;
`packable_prefill_step` excludes terminal chunks; `mixed_cursor_rows` excludes
the final prompt token. Partial mixed advancement retains the original ordinary
bucket. Consequently each remaining terminal token took its own ordinary
prefill completion, with decode quanta interleaved. Fusion did not
batch first-token sampling for those prefill requests. The new runtime resolves
this scheduling limitation without changing model compilation.

## Next changes justified by the evidence

1. Improve BF16 batched gate/up projections against vLLM's actual ROCm skinny
   implementation. The actual combined gate/up MFMA candidate improved its
   primitive from 166.7 to 132.3 microseconds, but full-model relative L2 reached
   3.4891%; it was rejected despite matching 128 greedy tokens. An
   arithmetic-preserving split regressed 34.6%. Primitive speed and token
   agreement alone do not qualify a replacement. Preserve numerical gates and
   measure work distribution and resource effects before naming a hardware cause.
2. With terminal completion fixed, prioritize phase-aware projection
   implementations. The completed 24-run stock comparison rejects increasing
   the runtime chunk from 512 to 1024: throughput falls 1.74%, 8.52% and 3.34%
   at prompt lengths 128, 1024 and 4096, with all 36 measured paired texts exact.
   Smaller default prefill tiles were also bitwise qualified but slower.
   A combined matrix operation over decode and
   prefill rows still needs numerical qualification because it changes the
   preserved decode arithmetic. Runtime row metadata already provides the
   dispatch decision; model compilation for every row combination is unnecessary.
3. Address batched long-context attention after projections; its measured gap is
   material. Launch-count or host-only changes have much less demonstrated upside.

The WPE4 MM1 candidate qualifies 107 logit rows bitwise. Its completed 48-run
stock comparison improves solo throughput 2.47%, 7.02% and 3.80%, with all 96
paired texts exact. It does not improve the unchanged wider decode path or
consistently improve TTFT. Separate GPUs and unpinned clocks limit attribution
of small differences. A mixed-tile
candidate has only a small serving signal. Removing unreachable FlashPrefill
code from the ordinary eight-wave prefill role reduced static executor scratch
operations 242→2, yet improved cold latency only 0.4–1.6% and regressed one
continuation case. Static spill counts therefore do not establish the dominant
cause, and the lean role needs safe loader routing before promotion.
None closes a roughly twofold batched gap. Matched FP8 full-model qualification
is separate; production W8A16 decode is not equivalent to vLLM W8A8.

## Follow-up probes rule out simple resource changes

Gate and up already execute concurrently on disjoint sets of 152 workers.
Native packet timestamps show both ready together and finishing about 200
microseconds later. A bounded probe using the shipping interpreter confirms
that increasing each projection to 304 logical slices helps an isolated GEMV
(124.75 to 116.29 microseconds), but slows the concurrent pair from 196.29 to
218.53 microseconds, an 11.3% regression. All outputs are bitwise identical.
Counting 152 slices per projection therefore does not establish device underfill.

The smaller-LDS WPE3/4/5 occupancy candidates complete all 107 full-model logit
rows bitwise, without reproducing a Gemma hang. Nevertheless, long B4 decode
regresses from 38.81 ms to 50.20, 97.69 and 157.44 ms. MM2 row-walk candidates
also regress at the primitive level, including the actual 152-slice gate shape.
Neither experiment justifies a production change. Resource counts alone have
not identified the physical bottleneck.

Follow-up measurements and artifact hashes (raw artefact removed; see the tables above)
preserve the completed chunk, lowrung, occupancy and slice comparisons. These
are Plow candidate/control studies, not new vLLM backend measurements.

## Interpreter register traffic

The subsequent ROCm 7.14 counter experiment identifies substantial register-save
traffic at the general `plow_exec` call. Its prologue saves 112 VGPR dwords per
lane before selecting the operation. Across 304 workgroups of 512 threads, those
stores represent 66.5 MiB. A NOP schedule retains about 69,331 KiB of external
writes; an empty queue writes 9.5 KiB. The counters exclude copies and clears,
and the ISA confirms the save/restore sequence. External traffic counters are
not specific to scratch addresses.

Calling the existing inlined GEMV helper directly reduces gate/up writes from
about 69,848 KiB to 370 KiB and down-projection writes from about 69,406 KiB to
69 KiB. Arithmetic and queue synchronization remain unchanged. The final
gfx942-scoped, pointer-scalarized candidate passes all 107 full-logit rows bitwise
at contexts 128, 4096 and 8176, including batch transitions and sparse mixed steps.
B4 median step times improve 7.0–7.8%. All timing samples remain in the report:
a 69.52 ms outlier makes the candidate's context-4096 mean slower than control.
These bounded model timings are from separate leases.

The completed C4 serving crossover runs both GPU assignments with the same
frozen server, checkpoint, assets, PF512, fusion, multistep and MM1 object.
Only the ordinary decode GQ object changes. Across 36 stock `vllm bench serve`
runs, all 144 requests and 18,432 output tokens are accounted for; all 48 measured
paired texts match. Throughput improves 7.53%, 7.56% and 4.30% at input lengths
128, 1024 and 4096. Median TPOT improves 7.38%, 7.84% and 7.59%.
Short median TTFT regresses from 243.08 to 274.42 ms; medium and long TTFT change
less than 1%. Per-request first-token times show varying groups, consistent with
variable packing/admission, but do not establish its cause. No run is excluded.
This is a Plow candidate/control improvement, not an all-metrics or vLLM win.

The separate MM1-only model study also passes 107 logit rows bitwise. A
same-GPU control/candidate/control bracket shows 17.54–18.05% lower solo step
medians, with 0.50–1.35% control drift. Its completed 36-run C1 serving crossover
accounts for 144 requests and 18,432 tokens, with all 48 measured paired texts
exact. Throughput improves 20.65%, 13.02% and 11.22%; median TPOT falls 17.44%,
13.00% and 14.77% at inputs 128, 1024 and 4096. Medium and long TTFT regress
3.31% and 2.46%. Combining MM1 and MM4 also passes all 107 logit rows bitwise.
The guarded source is integrated. The fresh six-cell comparison (raw artefact removed; see the tables above) completes 72 runs with 288 requests, 36,864 output tokens and zero failures. vLLM still leads throughput, TTFT and TPOT in every cell; 101/120 paired completion texts match. All experiment servers are stopped.
MM8 remains unqualified at full-model level; its head primitive regresses up to
12%, so it must not inherit the optimization without further evidence.

Counter, model and artifact-recovery evidence (raw artefact removed; see the tables above)
distinguishes the initial counter candidate from the final scoped candidate.
The result demonstrates avoidable interpreter traffic, but does not establish
bandwidth saturation or explain the entire gap to vLLM.

## Prefill norm fusion compatibility

The runtime now accepts ordinary packed prefill containing `NormResidualNorm`.
During mixed-program synthesis it expands this operation into the existing
`NormResidual` and `RmsNorm` consumers, preserving the BF16 residual boundary,
optional weights and scale. The resulting mixed instruction and dependency
tables match the unfused assets exactly at capacities 128, 512 and 1024.
No new mixed-row model variants or kernel arithmetic are introduced.

The loader derives the required norm capability from the prefill instructions
and checks an exported marker before loading the object. Legacy and inventory-
pruned objects lacking that marker are rejected for fused-norm prefill assets;
ordinary assets without the opcode retain compatibility. Existing fused-norm
prefill assets therefore require a rebuilt prefill object.

The qualification report (raw artefact removed; see the tables above) records
12 ordinary full-logit rows bitwise, all mixed numerical gates, cold packing
across three capacities, terminal/mux lifecycle tests, 27 HTTP requests and eight
deliberate disconnects. Mixed T128 prefill remains within the existing numerical
gate rather than universally bitwise. Bounded ordinary prefill timings improve
1.36–2.65%; this comparison also replaces the older shipping object with a
qualified rebuild. Stock serving with identical objects was interrupted at the user’s freeze request; completing that comparison remains pending.
`PLOW_PF_GFUSE` remains off by default, and runtime fusion remains opt-in.

## Compiler/runtime features actually used

The measured assets already use decode L2 placement and hierarchical decode
gates, including the MM1 object. Both objects advertise the corresponding
capability symbols. Ordinary prefill has 121 segments per bucket; runtime
routing alternates the general eight-wave object with the four-wave
FlashPrefill object. Prefill remains unplaced. Runtime mixed execution uses
its separate four-wave object and does not inherit decode-only hierarchy.

The actual instruction inventory contains 50 fused QKV and 60 fused gate/up/GLU
operations at decode widths one and two, plus 120 sandwich norm/residual/norm
operations. Width four retains separate projections and 60 GLU operations;
the fused QKV/GLU input exceeds the emitter's conservative LDS capacity bound.
Ordinary prefill fuses gate/up/GLU at 512 and 1024 rows, while the 128-row program
retains the separate operations. Fusion is therefore shape- and capability-
dependent, not universally enabled by one switch.

The existing prefill sandwich-norm fusion option is a remaining experiment.
Its `NormResidualNorm` opcode is currently rejected by mixed-program synthesis,
and the mixed kernel lacks the required logical-row handling for that helper.
It cannot simply be enabled in the current fused serving configuration.
Likewise, the shipping kernels are not a complete port of AITER/vLLM: the
matched vLLM baseline uses its custom skinny projection kernel and Triton
attention with AITER disabled. Actual operator counts and feature audit (raw artefact removed; see the tables above).

## Native numerical reference control

A separate BF16 control uses two fixed synthetic prompts of 16 and 128 tokens,
each followed by three teacher-forced decode steps. Unmodified vLLM retains
compilation, graphs and its native tanh-GELU custom operation. All eight
full-vocabulary reference rows repeat bitwise across two independent engines.

Shipping Plow agrees on all eight greedy tokens, but seven rows fail the
unchanged 1% relative-L2 / 0.5 maximum-absolute-error limits against that native
reference. Relative L2 ranges from 0.98% to 2.79%; maximum absolute error ranges
from 0.44 to 0.75. This demonstrates a numerical discrepancy predating the
private FP8 implementation. It does not explain all of the larger FP8 error,
and error norms from the two precision paths cannot be subtracted.

This native-reference control is distinct from the existing fusion tests,
which compare mixed execution with matched ordinary Plow phase arithmetic.
Neither reference nor threshold has been replaced. These synthetic cases are
diagnostics, not a broad model-quality or serving-performance evaluation.
Rows, settings and artifact hashes (raw artefact removed; see the tables above)
include the differing maximum-context capacities; every request fits both.

## Measurement limits and artifacts

Plow values are medians of three isolated captures; vLLM values are means over
the stated complete serving iterations. Context positions, timing instrumentation and
phase boundaries differ, so the table explains the deficit rather than replacing
the unprofiled scorecard. Independent medians need not sum exactly. Plow packet
envelopes use the existing 100 MHz timestamp conversion and closely match host
wall times. Ordinary prefill retains 1.3–1.6 ms of unallocated envelope gaps.

Torch profiling perturbs host launch timing: the short prefill capture has about
32 ms of GPU work but a 70 ms envelope, whereas unprofiled TTFT is 48 ms. Do not
use that profiled envelope as serving latency. One long-run GPU annotation was
duplicated across streams; it is merged by device and external ID before phase
counting. No overlapping iteration annotations remain. Full-iteration windows
include the output head and sampling, with small next-step preparation possibly
assigned to the preceding window.

ROCm kernel-trace interposition stalled Plow's asynchronous completion handling;
that failed capture is excluded. Native packet traces provide the GPU breakdown.
The earlier all-annotation long-run mean understated projection plus activation
as 15.07 ms and attention as 3.12 ms because some iterations lacked GPU records.
The corrected complete-iteration values are 16.05 ms and 3.32 ms. The audit checks
241 projection, 60 GELU, 60 attention, 60 reduction, 60 KV-write, 141 norm and two
sampling records per decode iteration. Thirty-two long D4 iterations and the
final D3 iteration are incomplete; one short iteration lacks a projection record.
The raw trace and old aggregates are retained, with corrected attribution under
`vllm_validated_decode` in the evidence JSON. No missing time is invented. The
complete subset may bias growing-context timing, so exact gap percentages remain
diagnostic estimates. The unprofiled serving comparison is unaffected.

Hardware counters now establish substantial interpreter register-save traffic.
They do not establish bandwidth saturation, cache misses or occupancy as the
dominant cause of the full-model gap.

Compact trace evidence and provenance (raw artefact removed; see the tables above).
Raw captures, reduction scripts and diagnostic sources are retained under
`build-gemma31/plow-profile/`, `build-gemma31/plow-profile-hsa/` and
`build-gemma31/vllm-profile/`.


---

## Gemma MI300X freeze — 2026-09-07

Branch: `perf/gemma31-mi300x-consolidated`.
Status: **FROZEN at the user's request. Experiments stopped.**
No new integration branch was created. Resume only on explicit user instruction.

## Applied and qualified

- `39cb856e`: BF16 gfx942 MM1/MM4 direct GEMV dispatch and fused-norm packed-prefill compatibility, with loader capability checks and runtime mixed-program expansion. Runtime fusion remains opt-in; PF_GFUSE remains off by default.
- Verification: 496 combined CPU/CUDA/HSA library tests, eight API tests, exact real-asset mixed instruction/dependency parity, GPU numerical/lifecycle/HTTP gates and default Nix build pass.
- Latest fetched main `49f8c089` is included. PR22 is open; this freeze does not merge it.
- Fresh stock vLLM0.28 ROCm comparison completed72 runs,288 requests and36,864 output tokens with zero failures. vLLM still leads throughput/TTFT/TPOT in all six cells;101/120 measured paired completion texts match. This is not a completed performance goal. See scorecard (raw artefact removed; see the tables above).

## Stopped and pending

| Experiment | State at freeze | Required before promotion |
|---|---|---|
| GFUSE off/on stock serving | Interrupted by user; fixed inputs and completed/partial outputs retained. Both servers and benchmark clients stopped. | Complete a new controlled crossover; do not present partial results as a complete comparison. |
| MM1 direct fused GLU/QKV dispatch |107 full-logit rows bitwise; additional model improvement1.45–2.72%. Patch saved, not applied. | Serving crossover and regression assessment. |
| MM4 direct FlashDecode dispatch |107 rows bitwise; model improvement2.0–2.8%, but projection primitives regress3–9%. Patch saved, not applied. | Serving comparison and resource tradeoff decision. Noinline alternative is rejected. |
| Native FP8 PTPC W8A8 | Private implementation fails full-model gates. Valid observers reproduce all eight production logit rows bitwise. Native fused norm-quant rounding mismatch identified; candidate correction is primitive-only. | Compare captured fused outputs, implement profile-specific correction, pass unchanged full-model gates, then qualify B4/mixed/serving. |
| Streaming host capture | Prior correctness/HTTP gates pass; serving reduces delivery gaps but regresses several throughput cells. Patch saved, not applied. | Controlled comparison with the qualified kernel configuration; investigate callback/HTTP cost. |
| Broader optimization studies | Expanded GFUSE capacities64/256/2048 and B8, attention grouping, combined tuning profiles, staggered arrivals and saturation studies remain pending. | Explicit resumption, defined matched workloads and numerical/performance gates. |

Saved patches and FP8 handoff: pending artifacts (raw artefact removed; see the tables above).
They are review material, not active production changes. Their base revisions differ;
inspect context before applying them on a later integration branch. Large model,
ELF, raw logit and profiler artifacts remain under `/app/plow/build-gemma31/`,
with paths and hashes retained in the reports. Existing artifacts were not deleted.

## Background cleanup

The fresh Plow/vLLM backend study and both direct-GEMV crossovers completed and
released their leases. The remaining GFUSE orchestrator and its benchmark/server
descendants were interrupted at the user's request. Root verified their PIDs were
gone, no GFUSE lease wrapper remained, and ports18940–18949 had no listeners.
FP8 and kernel agents report no owned GPU processes or leases. Unrelated machine
workloads were left untouched.

The GFUSE stop record (raw artefact removed; see the tables above) preserves44/72 completed runs,176 requests and22,528 output tokens, plus two interrupted run directories without completed accounting. No complete-crossover performance conclusion is drawn. GitHub branch checks are separate from stopped GPU experiments.

---

## L2 placement and the per-XCD gate, made default and made visible (2026-09-08)

The section above established that the feature existed and was inert. This one re-measures it
against a control emitted from the same tree, states what was made default and what deliberately
was not, and adds the two things that were missing from a feature nobody could tell was running:
a load-time claim that separates ARMED from FIRING, and a `build.json` field that moves when
placement does.

Everything below is BF16, TP1, one leased MI300X, 131072-context blobs at `PLOW_MAX_CHUNK=8192`,
decode ladder 1/2/4, 64 output tokens, `scripts/bench_packed_serve.py` with 1 warmup and 3 repeats
per cell, palindromic two-round A/B (medians of six per cell). Control and candidate blobs are
emitted by the same `plowc` from the same tree and differ ONLY in `PLOW_L2_PLACE`; `build.json` is
byte-identical between them apart from the new `l2_placement` key. The control reproduces the
shipped numbers at every cell it shares with them (128/1 41.80 vs 41.86 tok/s, 8192/4 22.90 vs
22.92, 32768/1 5.45 / 10115 ms / 25.72 ms vs 5.45 / 10122 / 25.74), which is what makes it a valid
A/B partner.

### Is the feature active in the shipping configuration?

**Not in anything published before 2026-09-08, and yes in what the tree emits today.** The four
switches and where each stands:

| half | where | state |
|---|---|---|
| decode object carries `-DPLOW_GATE_HIER` + `-DPLOW_L2_PLACE_DISPATCH` | `scripts/build_gfx942.sh` `PLOW_L2HIER` | default ON, and `build-gemma31/hsaco-tiered` carries both markers |
| blob's decode programs L2-placed | `plowc` `PLOW_L2_PLACE` | default ON for gfx942/gfx950 |
| blob's PREFILL programs L2-placed | `plowc` `PLOW_L2_PLACE_PREFILL` | default OFF on AMD, matching the objects the build script produces |
| runtime accepts a placed blob | `PLOW_L2_PLACE_DISPATCH` | **not consulted on AMD** — see below |

`build-gemma31/assets-final-plain`, the blob under every number in this document before that
date, starts `PLOWDEV\x09`. A default emit from this tree starts `PLOWDEV\x0b`.

### What it buys

Unplaced control vs decode-placed candidate, objects `build-gemma31/hsaco-tiered` unchanged
across both arms:

| input / conc | ctl tok/s | placed tok/s | tok/s | ctl TTFT | placed TTFT | ctl TPOT | placed TPOT | TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 / 1 | 41.80 | 44.92 | **+7.5%** | 73.4 | 74.8 | 23.14 | 21.43 | **-7.4%** |
| 128 / 4 | 90.57 | 97.57 | **+7.7%** | 268.3 | 263.2 | 38.76 | 35.85 | **-7.5%** |
| 512 / 1 | 38.92 | 41.92 | **+7.7%** | 149.7 | 150.1 | 23.72 | 21.84 | **-7.9%** |
| 512 / 4 | 81.15 | 86.93 | **+7.1%** | 480.5 | 469.2 | 40.98 | 38.05 | **-7.2%** |
| 2048 / 1 | 32.81 | 34.83 | **+6.2%** | 442.2 | 443.3 | 23.94 | 22.12 | **-7.6%** |
| 2048 / 4 | 57.98 | 61.35 | **+5.8%** | 1255.4 | 1201.5 | 48.58 | 45.81 | **-5.7%** |
| 8192 / 1 | 19.80 | 20.50 | **+3.5%** | 1689.1 | 1692.7 | 24.48 | 22.68 | **-7.3%** |
| 8192 / 4 | 22.90 | 23.47 | **+2.5%** | 7749.6 | 7628.0 | 52.97 | 50.47 | **-4.7%** |
| 32768 / 1 | 5.45 | 5.49 | +0.7% | 10115.2 | 10130.1 | 25.72 | 24.10 | **-6.3%** |

TPOT is down 4.7-7.9% at every one of the ten cells; TTFT is flat, which is the correct negative
control — only the decode programs are placed. Throughput follows TPOT and decays with prompt
length exactly as the decode share of the wall clock does. This reproduces the twelve-cell result
in the section above on independently emitted blobs.

### Does the output hold?

**Yes, on both corpora, at every cell, with no within-arm instability.**

* `bench_packed_serve.py` corpus, 128/512/2048/8192 x concurrency 1 and 4 x two rounds x three
  repeats: **20 completion keys, all character-identical between arms, 0 unstable within an arm.**
* `scripts/gemma31_prompt_identity.py` — new, because the bench derives its prompt from the
  request index and therefore carries exactly ONE prompt per length at concurrency 1, which
  cannot satisfy a ">= 3 prompts" gate. Four distinct prompts per length at 128 / 2048 / 8192,
  concurrency 1 AND 4, 64 greedy tokens, two repeats: **24 completion keys, all
  character-identical between arms, 0 unstable.**

### What was made default, and what was not

**Decode placement and the two-level gate: default, and already were.** `PLOW_L2HIER=1` in the
build script and `PLOW_L2_PLACE` in `plowc` both default on; the missing piece was prefill
placement's default, fixed in the section above. Nothing further needed flipping. This work
verified the pairing end to end rather than inheriting it.

**`PLOW_L2_PLACE_DISPATCH` stays `false`, deliberately.** It looks like the flag standing between
the shipping configuration and the win, and it is not: `AmdEngine::load` parses with
`l2_dispatch_ok = true` and then checks every code object for `plow_l2_place_dispatch_1`.
Inspecting the object is strictly stronger than an operator asserting it, and the assertion is
the ONLY guard on the CUDA path, where nothing reads the cubin. Flipping it would buy AMD nothing
and would remove a real check elsewhere. Verified rather than reasoned: serving
`assets-placed` with the variable UNSET logs

```
L2 hierarchical gate: FIRING — one L2 writeback+invalidate per XCD per packet
  armed=true firing=true l2_domains=8 rendezvous_entries=292732 decode_queue_entries=295749
```

`scripts/glm53_serve_inner.sh` still exports it, now with a comment saying it is not required on
AMD.

**Prefill placement stays opt-in.** Measured below; the win is real but confined to TTFT at long
prompts, it costs a rebuild of every prefill object, and one cell regresses reproducibly.

### ARMED is not FIRING, and the load path now says which

Three preconditions have to line up before a single `buffer_wbl2` is saved, and until now nothing
printed any of them. `interp.hip` takes the hierarchy only when `prog.hier_base != 0` (i.e. the
program is L2-placed), the stream entry's per-domain slice count `PLOW_SE_NPER` is above 1, and
the object was built with `PLOW_GATE_HIER`. The engine now evaluates that same predicate on the
host and says so once per rank. The three states, all captured from real serves:

| pairing | line |
|---|---|
| placed blob, `hsaco-tiered` | `FIRING — one L2 writeback+invalidate per XCD per packet` (`l2_domains=8`, 292732 of 295749 decode queue entries rendezvous) |
| unplaced blob, same objects | `ARMED BUT INERT — the decode object carries PLOW_GATE_HIER and this blob is NOT L2-placed, which is its runtime precondition. Re-emit with PLOW_L2_PLACE=1` |
| placed blob, objects without `-DPLOW_GATE_HIER` | `off: the blob is L2-placed but the decode object was built without -DPLOW_GATE_HIER` |

The second line is what every number in this document before today would have printed.

### The mismatch is refused by name, in both directions

A placed blob against an object without the dispatch axis was already a hard stop. It now names
BOTH halves and BOTH fixes, because the emit-side flag and the object-side flag are different
names living in different files and a reader who knows one cannot act on
"lacks `plow_l2_place_dispatch_1`". Serving the prefill-placed blob against `hsaco-tiered`:

```
L2 PLACEMENT PAIRING: this blob's prefill programs are L2-placed (PLOW_L2_PLACE; `build.json`
records it under `l2_placement`), but .../interp_prefill_gq.elf was built WITHOUT
-DPLOW_L2_PLACE_DISPATCH. A placed program's `seg` is an L2 domain, not a wave class, so this
object would run every packet on the wrong domain — plausible output, inverted locality, no
error. Fix EITHER half: rebuild the objects with scripts/build_gfx942.sh PLOW_L2HIER_PF=1, which
puts -DPLOW_L2_PLACE_DISPATCH on the prefill AND flash rows; build_gfx950.sh already passes it
under PLOW_L2_PLACE=1, or re-emit the blob with no PLOW_L2_PLACE_PREFILL=1, which is the AMD
default and leaves decode placement on.
```

The reverse direction — an unplaced blob against objects that DO carry the axis — is legal and
byte-identical, which is why it logs rather than refuses.

### `build.json` now records placement

A placed and an unplaced emit of the same model produced **byte-identical `build.json`**. That is
how a whole document's worth of numbers came to be taken with the feature off and nothing in the
artifact disagreeing. `crates/devgen/src/manifest.rs` now writes a top-level `l2_placement`:

```json
{"domains": 8, "decode": true, "prefill": false,
 "requires_object_define": "PLOW_L2_PLACE_DISPATCH"}
```

Outside `pairing_hash` on purpose, like `dispatch_audit`: placement does not change what
`plow_config.h` compiles, and stamping it would invalidate every existing packet/object pair.
The object side already had its half — `scripts/obj_baseline_gfx942.json` records
`PLOW_L2_PLACE_DISPATCH` and `PLOW_GATE_HIER` per row, and `check_build_matrix.py` already
refuses a `plow_gate_hier_1` object that lacks `plow_l2_place_dispatch_1`.

### `PLOW_L2HIER_PF=1` never compiled

The build script's prefill opt-in added `-DPLOW_L2_PLACE_DISPATCH=1 -DPLOW_GATE_HIER=1` to
`AX_PREFILL`, and `AX_PREFILL` carries `-DPLOW_BUCKET_DECODE=0`. The guard at the top of
`interp.hip` is

```c
#if PLOW_GATE_HIER && (!PLOW_BUCKET_DECODE || !PLOW_GLOBAL_QUEUE || !PLOW_L2_PLACE_DISPATCH)
#error "PLOW_GATE_HIER requires a decode global-queue object with L2-domain dispatch"
#endif
```

so the row does not build. Compiled by hand with exactly those axes: one error, no object. Had it
built, `plowrt`'s `check_gate_hier_object` would have refused it a second time at load. **The
two-level gate is decode-only by construction**, and the hierarchy half of `PLOW_L2HIER_PF` was
never a thing that could be measured. The flag now adds `-DPLOW_L2_PLACE_DISPATCH=1` alone, which
is the half a prefill-placed blob actually needs.

That also disposes of the 680-second `amd-bench` hang this file records as the reason not to widen
the define: that hang needed one placed program's segments split across objects with and without
`PLOW_GATE_HIER`, and no prefill or flash object can carry `PLOW_GATE_HIER` at all. With
placement alone on the prefill AND flash rows, Gemma's split-object prefill program ran to
completion at every prompt length, in normal wall time, on the first attempt.

### Prefill placement: measured, and left opt-in

`hsaco-pfplace` = `PLOW_DECODE_BATCH=4 PLOW_DECODE_TIERS=1,2 PLOW_L2HIER_PF=1`. One object set
serves BOTH arms (the axis is inert when `l2_domains == 0`), so this is a blob-only A/B:
decode-placed vs decode+prefill-placed.

| input / conc | TTFT dec | TTFT +pf | TTFT | tok/s dec | tok/s +pf | TPOT dec | TPOT +pf |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 / 1 | 74.1 | 73.1 | **-1.4%** | 45.24 | 45.05 | 21.28 | 21.38 |
| 128 / 4 | 272.9 | 259.2 | **-5.0%** | 97.74 | 97.58 | 35.82 | 35.91 |
| 512 / 1 | 149.8 | 148.8 | -0.7% | 42.10 | 42.02 | 21.74 | 21.81 |
| 512 / 4 | 467.9 | 497.1 | +6.2% | 86.82 | 86.09 | 38.00 | 38.07 |
| 2048 / 1 | 442.0 | 432.6 | **-2.1%** | 34.97 | 35.08 | 22.03 | 22.09 |
| 2048 / 4 | 1200.2 | 1238.2 | **+3.2%** | 61.40 | 61.10 | 45.80 | 45.63 |
| 8192 / 1 | 1691.2 | 1586.1 | **-6.2%** | 20.54 | 21.25 | 22.60 | 22.61 |
| 8192 / 4 | 7686.4 | 7467.9 | **-2.8%** | 23.41 | 23.83 | 50.34 | 50.26 |
| 32768 / 1 | 10162.7 | 9749.6 | **-4.1%** | 5.47 | 5.65 | 24.53 | 24.90 |

TPOT moves by at most 0.5% in either direction, which is the negative control this arm should
produce: the decode programs are identical in both blobs. **The 512/4 cell is not a result** — it
is bimodal in BOTH arms across rounds (527.7 / 467.6 control, 527.9 / 466.2 candidate), so the
paired medians land on different modes; it is a concurrency-4 scheduling bistability that predates
this change. The 2048/4 loss IS reproducible (1196/1201 vs 1237/1239).

The prefill-placed arm is **character-identical** to the decode-placed one on the same
multi-prompt corpus: 24 completion keys over 128 / 2048 / 8192 tokens x concurrency 1 and 4 x
four prompts x two repeats, no key differing, none unstable within an arm.

So prefill placement is a TTFT win that grows with prompt length, worth -6.2% at 8192 tokens
solo, against one reproducible +3.2% regression and no effect on decode. **Left opt-in**, for
reasons that are about blast radius rather than about the number: it changes every prefill object
in the tree, so every existing object directory becomes unusable with a default emit the moment
it lands, and the payoff is confined to one metric at long prompts. `PLOW_L2HIER_PF=1` plus
`PLOW_L2_PLACE_PREFILL=1` now works end to end and is the arm that prices it.

### gfx950

Nothing here is arch-specific and nothing here was run on gfx950. The load-time claim and the
pairing refusal read the blob and the object symbols, which are the same on both targets, and
`scripts/build_gfx950.sh` already gates the same two defines behind `PLOW_L2_PLACE` /
`PLOW_GATE_HIER`, both default on. One asymmetry worth knowing before anyone tries this there:
gfx950 puts `-DPLOW_L2_PLACE_DISPATCH` on its PREFILL and FLASH objects by default, so on that
target `PLOW_L2_PLACE_PREFILL=1` at emit needs no object rebuild at all — the gfx942 measurement
above is a caution to re-take, not a number to carry across.
