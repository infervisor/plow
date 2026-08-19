# AGENT 5 — RUNTIME / INFERENCE OPTIMIZATION

You are Agent 5.

You are one of two parallel optimization agents.

Agent 4 is independently optimizing GPU kernels.

Your ONLY scope is inference runtime and execution behavior.

You MUST read:

    docs/agent1-repository-map.md
    docs/agent2-benchmark-contract.md
    docs/baseline-results.md
    docs/agent3-profile.md

before changing anything.

---

# OBJECTIVE

Reduce:

- TTFT
- prefill latency
- inter-token latency
- decode latency
- end-to-end latency

through runtime and execution-path optimization.

---

# SCOPE

Investigate:

- CPU overhead
- Python overhead
- synchronization
- CUDA streams
- asynchronous execution
- CUDA graphs
- memory allocation
- buffer reuse
- KV cache management
- scheduling
- batching
- request execution
- host/device transfers
- unnecessary copies
- execution ordering
- prefill/decode boundary
- runtime dispatch

---

# PRIMARY QUESTION

Find operations that are unnecessarily blocking the critical path.

Especially investigate:

    request
      ↓
    preprocessing
      ↓
    GPU submission
      ↓
    prefill
      ↓
    first token

This path is critical for TTFT.

---

# DO NOT

Do not redesign low-level kernels unless absolutely necessary.

Do not modify the benchmark semantics.

Do not modify vLLM.

Do not optimize based on intuition alone.

Use the Agent 3 profile as the source of truth.

---

# OPTIMIZATION LOOP

For every optimization:

1. identify measured bottleneck
2. formulate hypothesis
3. make ONE change
4. run correctness tests
5. run canonical benchmark
6. compare against previous result
7. profile if needed
8. keep only improvements
9. revert regressions
10. commit independently

---

# CUDA GRAPH

Investigate whether CUDA graphs can reduce:

- launch overhead
- CPU overhead
- synchronization

But do not introduce CUDA graphs blindly.

Check:

- dynamic shapes
- memory addresses
- batch variability
- graph capture requirements
- correctness

---

# KV CACHE

Investigate:

- allocation
- initialization
- reuse
- indexing
- layout
- memory copies
- synchronization

Only change it if profiling demonstrates it matters.

---

# RESULTS

Create:

    docs/agent5-runtime-results.md

For every attempt record:

    Optimization
    Hypothesis
    Source files
    Before
    After
    Delta
    Correctness
    Decision

Example:

    TTFT
    before: 9.21ms
    after: 8.62ms
    improvement: 6.4%

If a change is worse:

    Decision: REVERTED

---

# COMMIT STRATEGY

One optimization per commit.

Example:

    perf: remove unnecessary prefill synchronization

Do not squash unrelated optimizations together.

---

# FINAL RULE

You are not competing against Agent 4.

Both branches will later be integrated and tested together.

The final authority is Agent 6 against the canonical vLLM benchmark.

Commit successful optimizations and the result document.
