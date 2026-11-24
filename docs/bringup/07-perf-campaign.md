# Stage 7 — Performance Campaign

Full serving evaluation under production traffic distributions against reference baseline engines (e.g. vLLM).

* **Precondition:** Stage 6 passes. Canonical recipe validated and committed.
* **Output:** Strict 12-row final campaign comparison report rendered via `campaign.py report`.
* **Architecture Reference:** [`docs/bringup/agent-tools.md`](agent-tools.md) §5 and §6.

---

## 1. Symmetric Serving Protocol

To ensure valid scientific comparisons, Plow and reference engines execute under strictly symmetric parameters:
* **Identical Workloads:** Exact same prompt dataset, token lengths, arrival timestamps, and concurrency.
* **Precision Parity:** Matching FP8 / FP16 / BF16 formats; identical weight quantization scales.
* **Hardware Parity:** Identical GPU lease, power state, and clock settings within the same queue session.

---

## 2. Serving Grid Execution

Run symmetric benchmarks across traffic patterns:
```bash
# 1. Start Plowrt server
scripts/bench/gpuq.py submit plow-serve 8 \
  plowrt serve --recipe recipes/infervisor/<model>/<label>.toml --port 8100

# 2. Run benchmark sweep (closed-loop agentic & open-loop Poisson production traffic)
scripts/bench/llm_grid.sh plow <scratch>/plow_results --agentic --prod

# 3. Start reference server & run identical sweep
scripts/bench/gpuq.py submit ref-serve 8 \
  vllm serve <model_path> --port 8200
scripts/bench/llm_grid.sh vllm <scratch>/ref_results --agentic --prod
```

---

## 3. Numerical Parity Gate (`llm_fp32_ref`)

Validate generation quality against the FP32 ground truth reference:
```bash
python3 scripts/campaign/campaign.py gate-accuracy \
  --candidate <scratch>/plow_results \
  --reference <scratch>/ref_results \
  --packet-sha256 <sha256>
```

### Numerical Acceptance Thresholds
* **KL Divergence:** Trim-mean KL divergence $\le 1.10\times$ reference peer.
* **Top-1 Agreement:** Greedy continuation top-1 token match $\ge \text{peer} - 0.005$.
* **Decisive Continuations:** Discrepancy $\le 2\text{ cases}$ across validation corpus.
* **Needle Retrieval:** Exact match accuracy $\ge \text{peer}$ on multi-needle evaluation.

---

## 4. Diagnostic Waterfall Telemetry Audit

Audit execution timing breakdown using the waterfall analyzer:
```bash
python3 scripts/bench/waterfall.py <scratch>/plow_results <scratch>/ref_results
```
* **Prefix-Cache Integrity:** Assert reference prefix-cache hit rate is zero when evaluating cold starts ($>5\%$ on reference fails the test).
* **Wall Time Split:** Analyze breakdown across prefill, decode, host scheduling gap, and idle time.
* **Step Latency:** Confirm decode ms/step scales linearly with active batch size.

---

## 5. Strict 12-Row Final Comparison Report

Generate the mandatory 12-row performance report exclusively via `campaign.py report`:
```bash
python3 scripts/campaign/campaign.py report \
  --plow <scratch>/plow_results \
  --peer <scratch>/ref_results \
  --out docs/bringup/results/<campaign>/comparison.md
```

### Report Schema (12 Required Rows)
1. **TTFT p50 ($\mu$s)**
2. **TTFT p90 ($\mu$s)**
3. **TTFT p99 ($\mu$s)**
4. **TPOT p50 ($\mu$s)**
5. **TPOT p90 ($\mu$s)**
6. **TPOT p99 ($\mu$s)**
7. **End-to-End Latency p50 (ms)**
8. **End-to-End Latency p99 (ms)**
9. **Generation Throughput (tok/s)**
10. **Total Serving Throughput (tok/s)**
11. **Server Peak RSS (GB)**
12. **GPU Memory Footprint (GB)**

Any campaign report missing rows or altering this schema is rejected.

---

## 6. Gate Checklist

Before advancing to Stage 8 or closing campaign:
- [ ] Symmetric serving executed under identical GPU lease and traffic distribution.
- [ ] `llm_fp32_ref` accuracy gate passes all four statistical thresholds.
- [ ] Waterfall audit confirms clean scheduling gaps and zero rogue prefix-cache hits.
- [ ] Strict 12-row comparison table generated and committed to `docs/bringup/results/<campaign>/comparison.md`.
