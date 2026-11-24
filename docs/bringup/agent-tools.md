# Agent Tools & Bringup Harness

The tool surface, execution environment, and mandatory rules for model bringup, kernel tuning, and performance campaigns.

**Rule:** Look here before writing any probe or test script. Use existing tools in `scripts/bench`, `scripts/campaign`, or `perf-data`; extend in place if needed. Scratch outputs belong outside the repo.

---

## 1. Mandatory Rules

### HARD RULE: Perf work iterates on a single block against roofline floor

Screen every kernel, fusion, route, or knob on **one block** (one layer or smallest layer span containing affected layer kinds) at served shape, precision, rungs, and context against a **calibrated** roofline floor. Whole-network benchmarks, FP32 gates, and serving grids run only for candidates that win at block level.

Every campaign report, PR, and `comparison.md` **starts with the block-vs-roofline table**:

| Block (layer kind, rung, ctx) | Block Time | Roofline Floor | % of Floor | Delta vs Current |
|---|---|---|---|---|

* `% of Floor` = `floor / measured time`.
* `Delta vs Current` = candidate minus production block at identical point in the same lease.
* Roofline certification: `scripts/campaign/certify_kernel.py` enforces $\ge 80\%$ decode and $\ge 75\%$ prefill efficiency against calibrated roofline floor.

#### Block Harness Tooling

| Step | NVIDIA | AMD |
|---|---|---|
| Emit one block | `campaign.py build <recipe> --block L[..R] --out <dir>` | `plowc --emit devblob --block` |
| Benchmark | `block_run <asset> bench --batch B --ctx N`<br>`block_run <asset> prefill-bench --query-rows Q --prior P` | `campaign.py block-bench`<br>`PLOW_LAYERS=N` sweeps |
| Calibrate ceilings | `scripts/bench/calibrate_ceilings.py --out ceilings.json` | Measured probe in `perf-data/probes/` |
| Floor & report table | `scripts/campaign/block_roofline.py <assets> --ceilings ceilings.json --plow sweep.json --current <prod.json>` | `campaign.py block-roofline` |
| Certification gate | `scripts/campaign/certify_kernel.py --input sweep.json --ceilings ceilings.json` | `scripts/campaign/certify_kernel.py --input sweep.json --ceilings ceilings.json` |
| A/B comparison | `campaign.py nv-block-ab --control <base> --treatment <cand> --verb bench\|prefill-bench --args "..." --ceilings ceilings.json` | `campaign.py block-ab` |
| MoE routing table | `PLOW_NV_DUMP_KV=<dir> PLOW_NV_DUMP_ACT=<len>:<pc>` → `scripts/bench/router_table.py` | `rank0.act.tab.bin` capture |
| Reference layer | `scripts/block_layer_bench.py` (vLLM decoder layer) + `scripts/block_compare.py` | `runtime/tests/*_block_*_test.c` oracles |
| Generator search | `scripts/gen_kernels/gen_search.py submit\|report <spec>` (TileLang/Triton schedules vs current vs floor) | — |
| Block parity | `block_run packed-check` / `mixed-check` | `amd-block --dump`, C oracle harnesses |

* **Generator search before hand-tuning:** When an op sits far from roofline after initial tuning, run `gen_search.py` before further manual optimization. Record best generated reference in the report.
* **ISA reference:** Consult `docs/bringup/isa/<target>.md` before writing or modifying low-level kernels.

### HARD RULE: Optimization order

Order is strict for all models (LLM, speech encoders, depth/code-predictors, codecs, vocoders):

1. **Roofline table first:** Compute block-vs-roofline table per op using calibrated ceilings. Rank stages by `share of device time × distance from floor`.
2. **Op fusion:** Inspect disassembly. Fuse ops sharing row passes (residuals, norms, activations, bias, casts, epilogues) that preserve the numeric contract. Document un-fusable ops.
3. **Kernels to roofline:** Tune hot ops via [Stage 4](04-kernel-tuning.md). Target $\ge 80\%$ decode, $\ge 75\%$ prefill floor efficiency. Ops left $>2\times$ from floor require written justification.
4. **Scheduling & serving knobs:** Tune holds, batching windows, admission SLOs, decode ladders, or KV budgets only after steps 1–3 are closed.
5. **Peer comparison last:** Run serving grids against vLLM to confirm gains. Never use serving grids to select work.

Speech pipeline tools:
* Op timings: `packet_bench <pkt> <program> <role> N --each|--sweep|--seq` (under `gpuq.py`).
* Ops/shapes inspection: `plowrt disasm --format json <pkt>`.
* Roofline table: `scripts/campaign/speech_roofline.py DISASM.json PROGRAM SWEEP.jsonl`.
* Op oracles & parity: `runtime/tests/speech_f32_op_test.cu`, `asr_encoder_pack_check`.
* Demand audit: `scripts/campaign/bucket_demand.py`.

### HARD RULE: Precision parity with peer stage-by-stage

Comparisons against reference engines (vLLM, vLLM-Omni) are valid only when arithmetic matches in every stage:

1. **Audit peer precision:** Read from launch logs (`dtype=`), engine configs, and source flags (`allow_tf32`, internal casts). Never infer precision from model cards or raw checkpoints.
2. **Audit plowrt precision:** Inspect packet via `plowrt disasm --format json` (weight dtype, half input/output bits, split-bf16, bf16 taps, 3xTF32, KV dtype).
3. **Publish parity table:** Include in results doc: `Model | Stage | Peer (line/log) | plowrt | Status`.
4. **Never narrower than peer:** Any plowrt stage narrower than the peer marks the cell **NOT MATCHED**.
5. **Wider stages must be tracked:** Any stage wider than the peer requires a written, measured justification (e.g. PCM contract) or must be aligned to peer precision.

---

## 2. Working Rules & Engineering Standards

### Product & Runtime Invariants
* **Prefix cache:** Always enabled (`PLOW_PREFIX_CACHE=1`, `PLOW_VMM_PREFIX=1`). Disabling is only for attribution diagnostics.
* **Prefill rungs:** Packets must carry the 8192-row prefill rung. Launch width is managed by plowrt `auto` scheduling, never hardcoded per recipe.
* **Single binary:** One `plowrt` binary and standard library set serve all models. No per-model runtime builds.
* **Packet contract:** Model-specific logic lives in packet metadata, `serve.json`, and symbol tables (`docs/runtime/*-packet-contract.md`). `plowrt` remains generic.
* **Static runtime:** `plowrt` dlopens only vendor libraries (`libcuda`, `cuBLASLt`, `libhsa-runtime64`). Third-party kernels (FA4/FA3/FlashInfer) ship as packet cubins launched via generic host stubs (`cuLaunchKernelEx`).
* **Knob registration:** Every `PLOW_*` knob, env var, or `#if PLOW_*` define must be registered in `crates/devgen/src/knob_spec.rs` or `crates/plowrt/src/knob_spec.rs`. Verified via `cargo test -p devgen --lib knob` and `cargo test -p plowrt --features cuda,hsa --lib knob`.
* **Runtime stability invariants:** Implementations must preserve Invariant 1 (monotonic burst draining), Invariant 2 (nominal steady-state floor), Invariant 3 (dwell time & hysteresis), and Tiered Memory Governance Tiers 0–3 ([`06-runtime-opt.md`](06-runtime-opt.md)).

### Benchmarking & Leases
* **GPU Queue:** All GPU executions run through `scripts/bench/gpuq.py submit <label> <ngpu> <cmd...>` (FIFO, queue root `/tmp/plow-gpuq`). Never lease bare cards or run raw unqueued workloads.
* **Pass env explicitly:** The queue runner scrubs inherited `PLOW_*` variables. Prefix commands with required environment variables (`env PLOW_FOO=1 ...`).
* **Screen before promote:**
  1. Screen: 1 block roofline + `step_bench` at affected rungs + 1 concurrency gate (`c16`, or `c8` for Qwen) in a single lease.
  2. Promote: `c1`, `c8`, `c32` FP32 gates + full production grid (2 rates × 2–3 reps).
* **Reference arms:** Capture vLLM reference baselines once; reuse matching baselines across screening runs.

### Branches, Review & Verification
* **Integration branch:** All work branches from and integrates into `origin/prod-next` via the coordinator. Never push directly to `main` or `prod-next`.
* **Kernel review:** New/modified kernels require:
  1. Pre-coding design consult (shapes, tiling, registers/smem, sync, roofline).
  2. Post-implementation diff review (bounds, tail rows, KV indexing past `kv_len`, races, alignment, FP8 accumulation).
  3. Blocking findings must be resolved before rung qualification.
* **Lean proofs:** Verified contracts require `scripts/lean_correctness_ci.sh` to pass. Verifier hash updates in `lean-plow/approved-verifiers.json` require explicit approval.

---

## 3. Environment & Preflight

### Environment Contract
All builds, emissions, benchmarks, and gates run inside `nix develop`:
```bash
nix develop --command <cmd>
```
`ROCM_PATH` must be present in the shell; absence indicates execution outside Nix.
*(Exception: `PLOW_CAMPAIGN_NO_NIX=1` on non-nix bare metal).*

External (non-Nix) dependencies:
* `PB_VLLM`: vLLM client launcher/venv path (`/app/plow/build-gemma31/vllm-python` on lab host, or resolved via `PATH`).
* `PB_VLLM_ROCM_LIB`: Lab ROCm runtime linked by vLLM (`/opt/rocm/core-7.14/lib`).
* `gpulease`: GPU leasing binary (`perf-data/tools/gpulease`).
* `PLOW_CKPT`, `GLM_RAW`: Checkpoint paths.

### Preflight Doctor
Run CPU-only preflight before requesting any GPU lease:
```bash
nix develop --command scripts/bench/plowbench-doctor.sh [assets-dir] [object-dir] [plowrt] [arch] [serve|block]
```
Checks: Nix environment, dangerous inherited `PLOW_*` env vars, binary dependencies, cuBLASLt version compatibility, packet hashes, GPU queue health, scratch disk capacity, and capability inventory.

### Capability Inventory
Audit declared routes statically:
```bash
plowrt inventory <assets> [--format json]
```
Ensures all required capabilities (prefill buckets, decode ladder, prefix cache, live KV, packed prefill, compact terminal, token batch) are `ON`. Any `OFF` capability fails unless listed in `[capabilities]` of the recipe or passed via `PB_CAPABILITY_WAIVERS`.

---

## 4. Primary Tool Surface

### Unified Campaign Driver (`campaign.py`)
```bash
# Build packet from recipe
python3 scripts/campaign/campaign.py build recipes/<model>/<cell>.toml --out <out-dir>

# Accuracy gates
python3 scripts/campaign/campaign.py gate <recipe> --assets <assets> --out <gate-dir> [--only <gate>] [--score-only]

# Serving benchmark
python3 scripts/campaign/campaign.py serve-bench <recipe> --assets <assets> --out <bench-dir>

# Block workflows
python3 scripts/campaign/campaign.py block-roofline <assets> --ceilings ceilings.json --plow sweep.json
python3 scripts/campaign/campaign.py block-ab --control <base> --treatment <cand> --verb bench --args "..."

# Render strict report
python3 scripts/campaign/campaign.py report --baseline <res>/vllm --infervisor <res>/plow --gate <gate>/gates.json --out <dir>
```

### End-to-End Serving Grids (`llm_grid.sh`)
```bash
# Standard serving sweep (greedy + sampled, 2 repeats)
env KV_DTYPE=<dtype> scripts/bench/llm_grid.sh plow <res>/plow [--agentic] [--prod]
env scripts/bench/llm_grid.sh vllm <res>/vllm [--agentic] [--prod]

# Options:
# --agentic: multi-turn prompt growth with prefix caching (c32/64/128 x 10 turns to ~16K)
# --prod: open-loop Poisson arrival traffic (rates set via PROD_RATES='0.5 1.0')
```

### Diagnostic & Audit Tools
* `scripts/bench/waterfall.py <plow-res> <vllm-res>`: Audit spread, prefix-cache hit rates ($>5\%$ on reference fails), wall time split (prefill, decode, host gap, idle), and decode ms/step.
* `scripts/bench/host_res.py report <plow-res> <vllm-res> --cell <tag>`: Compare server RSS, pinned memory, host tier footprint, CPU utilization, and startup times.
* `scripts/campaign/bucket_demand.py <metrics-url|file>`: Analyzes `plowrt_bucket_*` telemetry to identify unserved prefill/decode capacities and compute optimal bucket additions.
* `scripts/bench/step_boundary.sh <assets> <out>`: Tests KV boundary edge cases (ctx 4094, 4095, 4096, 8191, 8192) under `CUDA_LAUNCH_BLOCKING=1` to catch out-of-bounds KV reads.
* `scripts/bench/plowbench.sh`: Shared shell helper library. Provides `pb_free_port`, `pb_serve_start`, `pb_serve_stop`, `pb_bench`, `pb_cell`, and `pb_result`. Source this script in any custom automation.

---

## 5. Quality & Numerics Gates

Cross-stack accuracy is gated against independent FP32 references, never against peer bit-identity.

### LLM FP32 Reference Gate (`llm_fp32_ref`)
De-quantized weights, FP32 activations, TF32 disabled, exact chunked attention.
Scores Plow and vLLM against the identical FP32 token-id reference on fixed corpora (natural text, chat agentic, long-context needles).

```bash
# 1. Generate reference (done once per model)
python3 scripts/llm/fp32_ref_gate.py prompts --hf $HF --corpus ... --out prompts.json
scripts/bench/gpuq.py submit fp32-ref 1 python3 scripts/llm/fp32_ref_gate.py reference --hf $HF --prompts prompts.json --out ref.json

# 2. Gate candidate packet
campaign.py gate <recipe> --assets <assets> --out <gate-dir> --only llm_fp32_ref
```

#### Acceptance Thresholds (vs FP32 Reference)
| Metric | Definition | Pass Threshold (Plow vs Peer) |
|---|---|---|
| `kl_mean`, `kl_trim_mean`, `kl_p95` | KL divergence per position (top-20 + remainder); trimmed drops worst 1% | $\le \text{Peer} \times 1.25 + 0.002$ |
| `kl_over_frac` | Fraction of positions with $\text{KL} > 0.1$ nat | $\le \text{Peer} + 0.005$ |
| `top1_decisive` | Top-1 match where FP32 top-1/top-2 margin $> 1.0$ nat | $\ge \text{Peer} - 0.01$ |
| `cont_frac` | Greedy continuation tokens matching FP32 before divergence | $\ge \text{Peer} - 0.05$ |
| `cont_decisive_frac` | Cases where greedy continuation never diverges on decisive tokens | $\ge \text{Peer} - 0.00$ |
| `needle_acc` | Needle answer retrieved in greedy continuation | $\ge \text{Peer}$ and $\ge 0.90$ |

* Gated across concurrencies (`concurrency = [1, 8, 32]`). Every concurrency must pass.
* `kl_p99` and `kl_max` are reported for visibility but not hard-gated due to single-token co-batching sensitivity.

#### Gate Relaxation Rule
A candidate may exceed default `llm_fp32_ref` tolerances only if:
1. Perf gain is $\ge 5\%$ on affected step time or $\ge 3\%$ on production-mix goodput/latency.
2. Accuracy remains at vLLM level in $\ge 2$ of 3 runs: `kl_trim_mean`/`kl_p95` $\le 1.10\times$ peer, `top1_decisive` $\ge \text{peer} - 0.005$, `cont_decisive` $\ge \text{peer} - 2\text{ cases}$, `needle_acc` $\ge \text{peer}$.
3. Gains and per-run evidence are recorded in the recipe header.

---

## 6. Strict Final Performance Reporting

All final comparisons (PRs, summaries, `comparison.md`) must be generated via `campaign.py report`. Hand-written tables are rejected.

```bash
campaign.py report --baseline <res>/vllm --infervisor <res>/plow --gate <gate>/gates.json --out <dir>
```

Renders the mandatory 12-row table per cell:

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | `<model-id>` | Same as baseline |
| Precision / quantization | `<dtype>` | Same as baseline |
| Input / output length | `<isl> / <osl>` | Same as baseline |
| Traffic / concurrency | `<concurrency>` | Same as baseline |
| GPU type & count | `<gpu> x<n>` | Same as baseline |
| Serving stack | `<peer-version-and-flags>` | Infervisor (plowrt `<sha>`, pkt `<sha>`) |
| Output quality / correctness | `<gate-status>` | Equivalent |
| Peak GPU memory | `<gib>` | `<gib>` |
| Total throughput | `<tok/s>` | `<tok/s> (<ratio>x)` |
| Throughput / GPU | `<tok/s>` | `<tok/s> (<ratio>x)` |
| TTFT P99 | `<ms>` | `<ms> (<ratio>x)` |
| TPOT P99 | `<ms>` | `<ms> (<ratio>x)` |

### Report Rules
* All 12 rows are mandatory. Missing cells or unverified gates fail report generation.
* "Same as baseline" is allowed only when values are literally identical; otherwise marked **NOT MATCHED**.
* Output quality must link passing `llm_fp32_ref` for the exact served packet hash; otherwise marked **NOT EQUIVALENT**.
* Numbers are means over $\ge 2$ identical repeats. Spread $((max - min) / mean) > 10\%$ flags the metric.
* Exit codes: `0` = valid comparison; `1` = not matched or not equivalent; `2` = rejected input.

---

## 7. Measurement Hazards & A/B Protocol

### Hazardous Environment Variables
These flags alter execution paths and invalidate comparative measurements:
* `PLOW_PREFILL_SEG_TIMING=1`: Forces serial synchronous segment drains (~$3.7\times$ inflation). Use for attribution only, never for latency numbers.
* `PLOW_TUNEDB`: Loads measured GEMM tiles; slower than default schedules. Must not be used in shipping recipes.
* `PLOW_DECODE_MIN_RUNG`: Forces minimum decode launch width.
* `PLOW_OBJECTIVE`: Overrides `auto` scheduling. Must remain unset in production recipes.
* `PLOW_HSACO_LOWRUNG`: Overrides code object tiers for narrow rungs.

### Four-Arm A/B Scoring Protocol
Candidate evaluations use interleaved four-arm runs: `control1 → treat1 → control2 → treat2`.
* Control drift: $D_{ctrl} = |ctrl_1 - ctrl_2|$
* Treatment spread: $S_{treat} = |treat_1 - treat_2|$
* Validity condition: If $S_{treat} > 3 \times D_{ctrl}$, the run is invalidated by noise and must be rerun.

### System & Memory Hazards
* **Target directory collisions:** Do not use shared `CARGO_TARGET_DIR` across different builds. Rebuilding without features can silently replace the binary with the CPU interpreter.
* **Host RAM exhaustion:** Full model emits take 15–38 GB RAM per compiler process (`ptxas`/`cicc`). Never run emits concurrently with active serving benchmarks.
* **KV live bounds:** With `PLOW_VMM_LIVE`, only rows `[0, kv_len)` are mapped. Kernels reading past `kv_len` trigger unrecoverable `Xid 31` device faults. Run `step_boundary.sh` to certify KV boundaries.
