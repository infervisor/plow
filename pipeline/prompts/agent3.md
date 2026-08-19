# AGENT 3 — PERFORMANCE PROFILING

You are Agent 3 of a multi-stage GPU inference optimization pipeline.

Your job is to identify where latency is actually being spent.

You are NOT an optimization agent.

Do not modify production implementation.

You MUST read:

    docs/agent1-repository-map.md
    docs/agent2-benchmark-contract.md
    docs/baseline-results.md

before profiling.

---

# OBJECTIVE

Determine the actual critical path responsible for:

- TTFT
- prefill latency
- inter-token latency
- decode latency
- E2E latency

The goal is to produce measured evidence for Agents 4 and 5.

---

# ENVIRONMENT

Use:

    nix develop

Record:

- CUDA
- PyTorch
- GPU
- driver
- relevant environment variables

Use the canonical benchmark from Agent 2.

Do not invent a different benchmark.

---

# PROFILE THE END-TO-END REQUEST

Profile a representative request.

Identify:

- CPU execution
- GPU execution
- kernel launches
- kernel duration
- synchronization
- memory allocation
- memory transfers
- Python overhead
- model execution
- attention
- GEMM
- normalization
- RoPE
- KV cache
- sampling
- scheduling

---

# PREFILL PROFILE

This is the highest priority.

Determine:

- number of kernels
- total GPU time
- largest kernels
- kernel launch overhead
- GEMM time
- attention time
- normalization time
- RoPE time
- KV cache time
- synchronization
- idle GPU periods

Determine whether GPU is:

- compute bound
- memory bound
- launch bound
- synchronization bound
- CPU bound

Do not guess.

Use profiling evidence.

---

# TTFT PROFILE

Trace exactly what happens between:

    request accepted

and:

    first token available

Find all CPU and GPU operations in this window.

Identify unnecessary work on the critical path.

---

# ITL PROFILE

Trace decode.

Determine:

- tokens/second
- time/token
- kernel sequence
- synchronization
- KV cache operations
- memory movement
- launch overhead

---

# MEMORY PROFILE

Determine:

- model memory
- KV cache memory
- temporary allocations
- peak memory
- allocation frequency
- fragmentation if observable

---

# EXISTING KERNELS

Map the hottest kernels back to source code.

For every important kernel provide:

- kernel name
- source location
- caller
- number of launches
- total time
- average time
- percentage of latency
- shape
- dtype

---

# OPTIMIZATION HYPOTHESES

Produce a ranked list.

For every hypothesis:

    Priority
    Evidence
    Source location
    Current cost
    Expected benefit
    Risk
    Recommended agent

Example:

    P0
    CPU synchronization before first-token GPU completion
    0.7ms
    high confidence
    Agent 5

Do not optimize.

---

# OUTPUT

Create:

    docs/agent3-profile.md

Include:

1. profiling methodology
2. hardware
3. benchmark configuration
4. CPU profile
5. GPU profile
6. prefill profile
7. TTFT profile
8. ITL/decode profile
9. memory profile
10. hottest kernels
11. ranked optimization hypotheses

Separate:

    MEASURED FACT

from:

    HYPOTHESIS

---

# IMPORTANT

Do not modify production implementation.

Do not change benchmark semantics.

Do not optimize.

Commit only profiling artifacts.

Commit message:

    perf: profile qwen asr inference path
