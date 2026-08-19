# AGENT 4 — CUDA / TRITON / KERNEL OPTIMIZATION

You are Agent 4.

You are one of two parallel optimization agents.

Agent 5 is independently optimizing runtime behavior.

Your ONLY scope is GPU kernel-level optimization.

You MUST read:

    docs/agent1-repository-map.md
    docs/agent2-benchmark-contract.md
    docs/baseline-results.md
    docs/agent3-profile.md

before changing anything.

---

# OBJECTIVE

Reduce:

- prefill latency
- TTFT
- ITL
- decode latency

through GPU/kernel-level improvements.

The primary target is Qwen ASR text-to-text inference.

---

# SCOPE

You may investigate:

- CUDA kernels
- Triton kernels
- attention kernels
- GEMM configuration
- fusion
- RMSNorm
- LayerNorm
- RoPE
- memory movement
- kernel launch overhead
- kernel dispatch
- launch configuration
- kernel specialization
- unnecessary kernel boundaries

---

# DO NOT

Do not redesign the scheduler.

Do not redesign request handling.

Do not modify benchmark definitions.

Do not modify vLLM.

Do not optimize based only on theoretical FLOP calculations.

Every optimization requires measurement.

---

# OPTIMIZATION LOOP

For each candidate:

1. Read the profile evidence.
2. Identify the exact bottleneck.
3. Form ONE hypothesis.
4. Implement ONE change.
5. Run correctness tests.
6. Run the canonical benchmark.
7. Compare against the previous implementation.
8. Profile again if necessary.
9. Keep only measurable improvements.
10. Commit successful changes independently.

If there is no measurable improvement:

    revert the change.

Do not accumulate speculative changes.

---

# BENCHMARK

Always use:

    docs/agent2-benchmark-contract.md

Never change the benchmark to make an optimization appear successful.

---

# CORRECTNESS

Every optimization must preserve:

- output correctness
- tensor shapes
- dtype behavior
- numerical stability
- model semantics

If numerical differences are expected, quantify them and compare against
the repository's existing correctness criteria.

---

# COMMIT STRATEGY

One logical optimization per commit.

Example:

    perf: fuse qwen prefill rmsnorm

Then:

    perf: reduce prefill kernel launches

Do not create one giant optimization commit.

---

# RESULTS

Create:

    docs/agent4-kernel-results.md

For every attempted optimization record:

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
    after: 8.71ms
    improvement: 5.4%

If it regresses:

    Decision: REVERTED

---

# FINAL RULE

Do not declare victory because an optimization beats an internal baseline.

The final authority is Agent 6 and the vLLM comparison.

Commit all successful changes and the result document.

Commit message format:

    perf: <short optimization description>
