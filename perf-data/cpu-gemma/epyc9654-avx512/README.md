# Gemma-4-12B CPU end-to-end performance

Measured 2026-09-07 on this node with Plow runtime/compiler commit `d79c6d68`
(`cpu-production-avx512`). The benchmark probe additionally accepts concurrency
and prompt-padding options. Full 48-layer BF16 text generation, not a block
microbenchmark. Both 96-core and 192-core configurations passed the eight HTTP
correctness/serving checks before performance measurements.

## Hardware and execution

- Two AMD EPYC 9654 sockets: 192 physical cores, 384 logical CPUs, eight NUMA nodes.
- AVX-512 BF16/VNNI; CPU-only runtime, no CUDA/HSA/HIP libraries mapped.
- Separate bundles compiled with `--n-cu 96` and `--n-cu 192`; matching
  `--cpu-threads 96` and `--cpu-threads 192` at runtime.
- The 192-worker run used 192 distinct physical cores, 24 on each NUMA node.
  SMT siblings were not used. The 96-worker run used half the physical cores.
- `--cpu-numa auto`: worker affinity, interleaved large-tensor mappings,
  huge-page advice, and worker-local scratch first touch. Shared weights;
  no per-node replicas or NUMA tensor parallelism.
- Context 2048, prefill buckets 128/512, decode ladder 1/2/4, greedy generation.

Checkpoint: `google/gemma-4-12B-it` at revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. The 22.28 GiB safetensors file
passed SHA-256 verification against the Hugging Face LFS hash. The runtime
loaded 22.18 GiB of weights. [Checkpoint identity](checkpoint.json).

## Performance

Each row contains four requests, each capped at 128 output tokens: 512 output
tokens total. Short prompts contain 28–30 input tokens. Long prompts contain
1043–1045 input tokens. All responses reached the cap with `finish_reason=length`;
none used cached prompt tokens. The HTTP smoke checks warm the engine first.

| Physical cores | Concurrent requests | Prompt | Aggregate output tokens/s | Median first-content latency | Median time/output token after first |
| ---: | ---: | --- | ---: | ---: | ---: |
| 96 | 1 | Short | 3.02 | 0.826 s | 327.4 ms |
| 96 | 4 | Short | 10.99 | 2.561 s | 342.7 ms |
| 192 | 1 | Short | 3.33 | 1.583 s | 290.5 ms |
| 192 | 4 | Short | 11.34 | 4.407 s | 317.4 ms |
| 96 | 1 | Long | 2.39 | 11.536 s | 331.2 ms |

Aggregate output rate includes prefill and HTTP overhead: total output tokens
divided by the entire run's wall time. First-content latency is measured at the
client's first nonempty SSE content event. Per-request time/output token is
`(request duration - first-content latency) / (output tokens - 1)`; the table
reports its median. These are client-observed metrics, not isolated kernel times.

For the long prompts, median input-token count divided by first-content latency
was 90.5 input tokens/s. This is an effective end-to-end prefill rate, including
request/tokenization/first-token overhead, not a pure GEMM throughput measurement.

Doubling cores increased observed serial output rate by about 10.3% and batch
output rate by about 3.1%, while increasing first-content latency. These are
small samples on a shared node, with different physical page distributions
between loads. They do not isolate the effect of core count or establish an SLA.

Raw data: [summary](summary.json), [96-core serial](short-c1.json),
[96-core batch](short-c4.json), [192-core serial](short-c1-192.json),
[192-core batch](short-c4-192.json), [96-core long prompts](long-c1.json).

## Correctness and memory

Both configurations passed health/model discovery, single-request and streaming
answers, four concurrent capital-city questions, multi-chunk prefill, and
generation after disconnecting an active stream. The 932-token prompt correctly
returned `Paris` in 10.497 s at 96 cores and 11.337 s at 192 cores. These checks
cover basic text correctness and serving behavior, not broad model-quality parity.
[96-core HTTP results](http-e2e.json), [192-core HTTP results](http-e2e-192.json).

The 96-core engine loaded in 5.947 s. Its measured resident memory was 26.29 GiB;
the 192-core sample was 27.09 GiB. Peak RSS during loading was approximately
47.7 GiB, including resident checkpoint mappings and copied weights.

NUMA policy was accepted, but actual page placement remained uneven. At 96 cores,
interleaved mappings were overwhelmingly resident on nodes 3 and 2. At 192 cores,
pages spanned all eight nodes but node 3 still held nearly half. This host had
little free memory and substantial file cache; allocation fallback and huge-page
availability were not isolated. No global kernel/cache settings were changed.
NUMA is enabled, but balanced memory bandwidth is not established by this run.
[96-core snapshot](host.json), [192-core snapshot and affinity proof](host-192.json).

The compiler's Lean verifier was unavailable and it recorded an unverified
manifest. Runtime checks are not a formal ordering certificate.
[96-core compiler output](compile.txt), [192-core compiler output](compile-192.txt).

## Reproduce the all-physical-core run

From `/app/plow/.worktrees/cpu-production-avx512`:

```sh
nix develop --command target/release/plowc \
  --hf-dir /workspace/models/gemma-4-12B-it-cpu \
  --emit devblob --arch sm_120a --gpu rtx6000pro --n-cu 192 \
  --max-ctx 2048 --emit-max-chunk 512 --emit-decode-batch-ladder 1,2,4 \
  --out build-cpu-12b-192/assets --no-tuning
nix develop --command target/release/plowrt serve \
  --assets build-cpu-12b-192/assets \
  --rt-checkpoint /workspace/models/gemma-4-12B-it-cpu \
  --cpu-isa avx512 --cpu-threads 192 --cpu-numa auto --port 18680
```

The compiler uses existing shared device-blob architecture metadata; this does
not build or execute GPU kernels. In another terminal, run probes sequentially:

```sh
nix develop --command python3 perf-data/probes/cpu_http_e2e.py --output /tmp/12b-e2e.json
nix develop --command python3 perf-data/probes/cpu_http_decode.py \
  --concurrency 4 --output /tmp/12b-c4.json
nix develop --command python3 perf-data/probes/cpu_http_decode.py \
  --concurrency 1 --output /tmp/12b-c1.json
```

For the 96-core run, use `--n-cu 96`, `--cpu-threads 96`, and bundle
`build-cpu-12b/assets`. Run the serial and concurrent probes; the long-prompt
probe additionally uses `--concurrency 1 --padding-repeats 112`.

The 192-core server was left running on port 18680 after these checks.
