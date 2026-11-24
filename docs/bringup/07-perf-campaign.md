# Stage 7 — Performance Campaign

Conduct final comparative performance evaluation against reference baseline (vLLM) under strict accuracy gating.

* **Precondition:** Stage 6 gated out; production recipe TOML created in `recipes/infervisor/<model>/<label>.toml`; [`target.md`](target.md) parameters defined.
* **Gate out:** `llm_fp32_ref` passes at all declared concurrencies (`c1`, `c8`, `c32`); strict 12-row comparison table rendered via `campaign.py report` and committed to `docs/bringup/results/<campaign>/comparison.{md,csv}`.
* **Scope:** Measurement and qualification only. Any further optimization loops return to Stage 5, Stage 6, or Stage 8.

---

## 1. Campaign Plan Definition

Document prior to execution:
1. **Model & Precision:** Target model ID, weight precision, and KV cache precision.
2. **Traffic Configurations:**
   * Single-turn static sweep (ISL/OSL grid).
   * Multi-turn agentic sweep (`--agentic`, prefix-cache growth).
   * Open-loop production Poisson mix (`--prod`, `PROD_RATES`).
3. **Concurrency Ladder:** Evaluation concurrency points (e.g. 1, 8, 32, 64, 128).
4. **Reference Baseline:** Peer engine version, launch flags, identical checkpoint, matched prefix-cache policy, and hardware environment.

---

## 2. Step-by-Step Execution

All GPU jobs must be submitted via `scripts/bench/gpuq.py submit <label> <ngpu> <cmd...>`.

### Step 1: Preflight Verification (CPU)
```bash
nix develop --command scripts/bench/plowbench-doctor.sh <assets> <objdir>
```
Confirms environment variables, binary versions, capability inventory, and hardware accessibility.

### Step 2: Quality & Accuracy Gating
Gate the exact emitted packet scheduled for serving:
```bash
# 1. Generate run script
python3 scripts/campaign/campaign.py gate <recipe> --assets <assets> --out <gate-dir> --only llm_fp32_ref --dry-run

# 2. Execute gate under GPU queue
python3 scripts/bench/gpuq.py submit gate <ngpu> bash <gate-dir>/run.sh

# 3. Score results
python3 scripts/campaign/campaign.py gate <recipe> --assets <assets> --out <gate-dir> --only llm_fp32_ref --score-only
```
Promotion requires passing `[gates.llm_fp32_ref]` at all configured concurrencies (`concurrency = [1, 8, 32]`). Serving cells cannot be reported without a passing gate for the exact packet hash.

### Step 3: Symmetric Serving Evaluation
Execute both arms with identical clients, prompts, seeds, and concurrency:
```bash
# Infervisor serving arm
python3 scripts/bench/gpuq.py submit grid-plow <ngpu> env ASSETS=<assets> HF=<hf> KV_DTYPE=<dtype> \
    SERVE_ARGS="<recipe-flags>" scripts/bench/llm_grid.sh plow <scratch>/res/plow [--agentic|--prod]

# vLLM reference arm (reuse matching baseline if already captured)
python3 scripts/bench/gpuq.py submit grid-vllm <ngpu> env HF=<hf> VLLM_ARGS="<matched-flags>" \
    scripts/bench/llm_grid.sh vllm <scratch>/res/vllm [--agentic|--prod]
```

### Step 4: Waterfall & Telemetry Audit
Audit execution consistency:
```bash
python3 scripts/bench/waterfall.py <scratch>/res/plow <scratch>/res/vllm
```
* Fails if vLLM reference prefix-cache hit rate exceeds $5\%$ on random prompts.
* Validates run-to-run spread and breaks down wall-clock execution time (prefill, decode, host gaps, idle).

### Step 5: Render Strict Final Report
Generate the authoritative comparison report:
```bash
python3 scripts/campaign/campaign.py report \
    --baseline <scratch>/res/vllm \
    --infervisor <scratch>/res/plow \
    --gate <gate-dir>/gates.json \
    --out <scratch>/report
```
* Renders the mandatory 12-row comparison table ([agent-tools.md](agent-tools.md)).
* Appends validated rows to `docs/bringup/results/<campaign>/comparison.csv`.
* Commits summary to `docs/bringup/results/<campaign>/comparison.md`.

---

## 3. Results Layout

* **Git Repository:** Only curated summaries live in `docs/bringup/results/<campaign>/comparison.{md,csv}`.
* **External Scratch:** Raw logs, client traces, server stdout/stderr, and profiler dumps reside in `$RESULTS` outside the repository.
* **Reference Baselines:** Stored in `perf-data/campaign/` for regression tracking.

---

## 4. Gate Checklist

- [ ] `llm_fp32_ref` passes across all target concurrencies with valid `packet_sha256`.
- [ ] Precision parity audit confirms Plow is not narrower than vLLM in any pipeline stage.
- [ ] Both serving arms ran with identical client seeds, prompts, and traffic models.
- [ ] Benchmark runs executed with $\ge 2$ repeats; run spread is documented ($<10\%$).
- [ ] Final report successfully rendered via `campaign.py report` with exit code 0.
- [ ] Results documented in `docs/bringup/results/<campaign>/` and indexed in `results/README.md`.
