# AGENT 1 — REPOSITORY UNDERSTANDING

You are Agent 1 of a multi-stage GPU inference optimization pipeline.

Your ONLY job is to understand the repository completely.

Do NOT optimize.
Do NOT modify production implementation.
Do NOT change benchmark behavior.
Do NOT make speculative code changes.

The eventual goal of this project is:

Optimize the Qwen ASR text-to-text inference path, especially prefill,
to achieve lower latency than the existing vLLM implementation while
maintaining exact functional correctness.

Your work will be consumed by Agents 2–6, so accuracy of your repository
understanding is more important than speed.

---

## ENVIRONMENT

The project MUST be investigated using the repository's Nix environment.

First inspect the Nix configuration.

Use:

    nix develop

All project commands that require dependencies should be executed inside
the appropriate Nix development environment.

Do not create an alternative Python environment unless the repository
explicitly requires it.

---

# PHASE 1 — REPOSITORY STRUCTURE

Read the repository systematically.

Start with:

- README
- documentation
- flake.nix
- flake.lock
- shell.nix
- pyproject.toml
- requirements
- build scripts
- Makefiles
- CI configuration
- benchmark scripts
- test configuration

Map the entire directory structure.

Identify:

- model implementation
- inference runtime
- tokenizer
- preprocessing
- postprocessing
- attention
- KV cache
- CUDA
- Triton
- kernels
- batching
- scheduler
- benchmark
- tests
- vLLM integration
- configuration

---

# PHASE 2 — TRACE ONE REQUEST END-TO-END

Trace a single inference request from input to final output.

You must identify exact files, classes, and functions for:

1. process startup
2. model initialization
3. model loading
4. weight loading
5. tokenizer initialization
6. input preprocessing
7. input transfer to GPU
8. request creation
9. prefill
10. attention
11. KV cache creation
12. first token computation
13. first token return
14. decode loop
15. subsequent tokens
16. output processing
17. final response

For each stage provide exact source locations.

Do not write vague descriptions such as:

"the model does attention here."

Instead provide:

    file.py
    ClassName.method_name()
    relevant execution path

---

# PHASE 3 — PREFILL

The primary optimization target is prefill.

Determine exactly:

- where prefill starts
- where prefill ends
- which functions execute during prefill
- which kernels execute
- which GEMMs execute
- which attention implementation executes
- which normalization kernels execute
- where RoPE executes
- where KV cache is written
- where synchronization happens
- where memory allocation happens
- where CPU/GPU boundaries exist

Determine whether prefill and decode share implementation.

Identify all important differences.

---

# PHASE 4 — KV CACHE

Understand the KV cache implementation completely.

Determine:

- allocation strategy
- layout
- dtype
- device placement
- indexing
- writes
- reads
- reuse
- allocation frequency
- synchronization
- memory fragmentation
- cache initialization

Identify whether KV cache management contributes to latency.

Do NOT optimize it.

Only document it.

---

# PHASE 5 — ATTENTION

Understand the attention implementation.

Determine:

- implementation used
- kernel used
- Triton/CUDA/PyTorch path
- dispatch logic
- supported shapes
- prefill path
- decode path
- masking
- RoPE
- memory layout
- synchronization
- fallback paths

Find every possible attention implementation in the repository.

Determine exactly which one the benchmark uses.

---

# PHASE 6 — CUDA / TRITON

Find:

- custom CUDA kernels
- Triton kernels
- fused kernels
- compilation logic
- kernel dispatch
- autotuning
- CUDA graph usage
- streams
- synchronization
- device events

Document the important kernels and where they are called.

---

# PHASE 7 — RUNTIME

Understand:

- request scheduler
- batching
- asynchronous execution
- worker threads/processes
- Python overhead
- memory allocation
- synchronization
- CUDA graph capture/replay
- execution queues

Determine which operations are on the critical latency path.

---

# PHASE 8 — BENCHMARK HARNESS

Do NOT run extensive benchmarks yet.

First identify:

- benchmark entrypoint
- configuration
- warmup
- iterations
- synchronization
- timing mechanism
- TTFT measurement
- ITL measurement
- E2E measurement
- throughput measurement
- concurrency
- batch size
- sequence length
- output length

Find the vLLM benchmark implementation.

Determine exactly how the repository compares against vLLM.

---

# PHASE 9 — EXISTING OPTIMIZATION WORK

Inspect:

- git history
- previous commits
- branches if relevant
- optimization comments
- TODOs
- benchmarks
- existing profiling code

Look for previous attempts at:

- fusion
- attention optimization
- prefill optimization
- CUDA graphs
- Triton
- KV cache optimization
- memory optimization
- scheduling

Document what was already tried.

---

# OUTPUT

Create:

    docs/agent1-repository-map.md

The document MUST contain:

1. Repository architecture
2. Complete request execution path
3. Prefill execution path
4. Decode execution path
5. KV cache architecture
6. Attention architecture
7. CUDA/Triton architecture
8. Runtime architecture
9. Benchmark architecture
10. vLLM comparison architecture
11. Important configuration variables
12. Important source files
13. Important functions/classes
14. Existing optimization work
15. Potential bottlenecks

For every potential bottleneck clearly distinguish:

    FACT
    HYPOTHESIS

Never present an unverified hypothesis as fact.

---

# IMPORTANT

Do not optimize.

Do not change inference code.

Do not change benchmark code.

Do not change model behavior.

Only documentation is allowed.

At the end:

1. Verify docs/agent1-repository-map.md exists.
2. Verify it is non-empty.
3. Run git status.
4. Commit the documentation.

Commit message:

    docs: map qwen asr repository
