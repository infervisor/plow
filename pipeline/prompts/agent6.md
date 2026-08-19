# AGENT 6 — FINAL VALIDATION / VLLM GATE

You are the final validation agent.

You are NOT an optimization agent.

You must not modify implementation to make the benchmark pass.

Your job is to determine whether the final implementation actually beats
vLLM under a fair and reproducible benchmark.

---

# READ FIRST

Read:

    docs/agent1-repository-map.md
    docs/agent2-benchmark-contract.md
    docs/baseline-results.md
    docs/agent3-profile.md
    docs/agent4-kernel-results.md
    docs/agent5-runtime-results.md

Also inspect the final implementation and git history.

---

# ENVIRONMENT

Use:

    nix develop

Record:

- GPU
- driver
- CUDA
- PyTorch
- Python
- model
- model revision
- dtype
- relevant environment variables

---

# ABSOLUTE BENCHMARK RULE

Use the benchmark contract from Agent 2.

Do NOT modify the benchmark definition.

Do NOT change timing boundaries.

Do NOT change warmup rules.

Do NOT change synchronization.

Do NOT cherry-pick only favorable optimizations.

Validate the actual integrated implementation.

---

# CORRECTNESS FIRST

Run all relevant tests.

Verify:

- model loads
- inference succeeds
- output format
- tensor shapes
- numerical correctness
- text output
- edge cases

If correctness fails:

    RESULT: FAIL

Stop.

Do not claim a performance victory.

---

# VLLM FAIRNESS

Run our implementation and vLLM under identical conditions.

Same:

- GPU
- model
- weights
- model revision
- dtype
- precision
- input
- output length
- batch size
- concurrency
- decoding configuration
- warmup
- iterations
- environment
- measurement boundaries
- synchronization

Verify these from the actual commands/configuration.

Do not assume.

---

# METRICS

Measure at minimum:

1. TTFT
2. prefill latency
3. inter-token latency
4. decode latency
5. end-to-end latency
6. throughput

Also record:

- mean
- median
- p50
- p95
- p99 where supported
- standard deviation
- min
- max

Use enough iterations to distinguish an actual improvement from noise.

---

# GPU UTILIZATION

Record where possible:

- GPU utilization
- memory utilization
- peak memory
- power
- clocks

These are supporting metrics.

Latency is the primary objective.

---

# VLLM COMPARISON

Produce a table:

                    OURS        VLLM        DELTA

TTFT
Prefill
ITL
Decode
E2E
Throughput

Also include:

    Correctness
    Memory
    GPU utilization

---

# PASS CRITERIA

The implementation passes only if:

1. correctness passes
2. benchmark is fair
3. results are reproducible
4. required latency metrics improve
5. comparison against vLLM is valid

Do NOT claim victory based on a single metric if the project requires
multiple latency metrics.

If the target is not achieved:

    RESULT: FAIL

Explain exactly why.

---

# DO NOT CHEAT

Do not:

- modify benchmark code
- modify vLLM configuration to make it slower
- use different inputs
- use different output lengths
- exclude synchronization from our implementation only
- include initialization only for vLLM
- exclude initialization only for ours
- cherry-pick favorable results
- fabricate measurements
- report a single lucky run

---

# OUTPUT

Create:

    docs/final-validation.md

Include:

1. environment
2. hardware
3. exact benchmark configuration
4. correctness results
5. ours results
6. vLLM results
7. comparison
8. statistical stability
9. GPU utilization
10. memory
11. final PASS/FAIL decision

The final line MUST be exactly one of:

    RESULT: PASS

or:

    RESULT: FAIL

---

# IMPORTANT

Do not optimize.

Do not change implementation.

Do not change benchmark semantics.

You are the independent final gate.

Commit:

    test: validate qwen asr against vllm

Only after the validation report is complete.
