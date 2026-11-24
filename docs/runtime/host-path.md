# Runtime Host Path & Dispatch Loop

Crisp technical specification of the `plowrt` host dispatch cycle and scheduling pipeline.

---

## 1. Request Lifecycle & Pipeline Stages

```
Ingress (HTTP/gRPC) → Admission Scheduler → Dispatch Batcher → Hardware Queue → Token Commit → Streaming
```

1. **Ingress:** HTTP/OpenAI-compatible server receives request, decodes input text tokens, and queries the `SessionStore`.
2. **Admission Scheduler:** Evaluates Invariant 1 (monotonic burst draining) and verifies available Tier 2 dynamic KV blocks before admittance.
3. **Dispatch Batcher:**
   * Packs chunked prefill requests into uniform query lengths (`PLOW_PF_CHUNK`).
   * Binds ready decode slots to fixed hardware rungs (16, 32, 64, 128) matching pre-recorded CUDAGraphs.
4. **Hardware Submission:** Enqueues command buffers (AQL packets for HSA, stream executions for CUDA) directly to GPU hardware queues.
5. **Token Commit & Streaming:** Reads generated tokens from pinned output memory, updates KV cache pointers, and pushes SSE response chunks.

---

## 2. Stability Invariants on the Host Path

* **Invariant 1 (Monotonic Burst Draining):** $C_{\text{admit}}(t) = \max(C_{\text{nominal}}, N_{\text{active}}(t))$, contracting monotonically as active requests complete.
* **Invariant 2 (Nominal Steady-State Floor):** Sustained concurrency guaranteed at $C_{\text{nominal}}$ under saturated demand.
* **Invariant 3 (Dwell Time):** Minimum dwell time $T_{\text{dwell}} \ge 500\text{ ms} \lor 50\text{ steps}$ before rung reconfiguration.

---

## 3. Telemetry & Profiling

* Enable fine-grained step timing: `PLOW_TICK_LOG=1` or `PLOW_TRACE_RAW=1`.
* Audit scheduling gaps and waterfall breakdown: `scripts/bench/waterfall.py <plow-res> <peer-res>`.
