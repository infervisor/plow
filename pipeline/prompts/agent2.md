# AGENT 2 — BENCHMARK AND VLLM BASELINE

You are Agent 2 of a multi-stage GPU inference optimization pipeline.

Agent 1 has already analyzed the repository.

Your job is to understand and establish the benchmark contract.

You MUST read:

    docs/agent1-repository-map.md

before doing anything else.

Do NOT optimize inference.

Do NOT modify production implementation.

Do NOT change the benchmark to make the implementation look better.

Your job is to establish a scientifically valid comparison between the
repository implementation and vLLM.

---

# ENVIRONMENT

Use the repository's Nix environment.

Start by inspecting and entering:

    nix develop

All benchmark commands must use the same environment whenever possible.

Record:

- Nix revision
- CUDA version
- PyTorch version
- Python version
- GPU
- GPU count
- driver
- relevant environment variables

---

# PHASE 1 — UNDERSTAND THE HARNESS

Read the entire benchmark implementation.

Determine exactly:

- benchmark entrypoint
- command line arguments
- configuration files
- model configuration
- warmup
- number of iterations
- batch size
- concurrency
- input length
- output length
- sampling configuration
- synchronization
- CUDA event usage
- CPU timer usage
- result aggregation
- percentile calculations

Do not assume the benchmark is correct.

Verify it from the code.

---

# PHASE 2 — DEFINE METRICS

Determine exactly how these metrics are calculated:

## TTFT

Time to first token.

Determine exactly what timestamp starts the measurement and
what timestamp ends it.

Document what is included and excluded.

## Prefill latency

Determine whether it is explicitly measured.

If not, determine whether it can be derived correctly.

## Inter-token latency

Determine exactly how ITL is calculated.

Check whether the first decode token is included.

## Decode latency

Determine how decode time is measured.

## End-to-end latency

Determine exactly what is included.

## Throughput

Determine the exact formula.

---

# PHASE 3 — CHECK SYNCHRONIZATION

This is critical.

Look for:

- torch.cuda.synchronize()
- CUDA events
- stream synchronization
- implicit synchronization
- asynchronous launches
- CPU timing around asynchronous GPU work

Determine whether the current benchmark can produce misleading timings.

If you find a measurement bug, document it.

Do not silently change it.

---

# PHASE 4 — VLLM BASELINE

Inspect exactly how vLLM is executed.

Determine:

- vLLM version
- model
- model revision
- dtype
- quantization
- attention backend
- compilation
- CUDA graph configuration
- tensor parallelism
- batching
- sequence length
- output length
- sampling
- warmup
- iteration count

Determine whether vLLM and the custom implementation are truly comparable.

---

# PHASE 5 — HARDWARE VALIDATION

Record:

- GPU model
- VRAM
- driver
- CUDA
- PyTorch
- GPU clocks if available
- power state if available
- MIG status if relevant

Ensure both implementations use the same GPU.

---

# PHASE 6 — RUN BASELINE

Run the existing benchmark without modifying inference implementation.

Perform sufficient warmup.

Run multiple iterations.

Record raw results.

Do not report a single lucky measurement.

Capture:

- mean
- median
- p50
- p90
- p95
- p99 where supported
- standard deviation
- min
- max

for the important metrics.

---

# PHASE 7 — FAIRNESS

Verify the same:

- model
- weights
- precision
- GPU
- input
- output
- batch size
- concurrency
- decoding configuration
- warmup
- iterations
- timing boundaries

between ours and vLLM.

If they differ, document the difference.

Do not hide it.

---

# OUTPUT 1

Create:

    docs/agent2-benchmark-contract.md

It MUST define the canonical benchmark.

Include exact:

- command
- environment
- model
- GPU
- input
- output
- batch
- concurrency
- warmup
- iterations
- metrics
- synchronization
- vLLM configuration

This benchmark contract becomes immutable for Agents 3–6.

---

# OUTPUT 2

Create:

    docs/baseline-results.md

Include actual measured results.

Never invent numbers.

Clearly label:

    OUR IMPLEMENTATION

and:

    VLLM

Include raw measurements where useful.

---

# IMPORTANT

Do not optimize.

Do not modify inference behavior.

Do not modify the benchmark definition unless you discover a correctness
problem. If you discover one, document it and make the smallest possible
measurement-only correction.

At the end:

1. verify both documents exist
2. verify results are reproducible
3. run git status
4. commit

Commit message:

    bench: establish qwen asr benchmark contract
