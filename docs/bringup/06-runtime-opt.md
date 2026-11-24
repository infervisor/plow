# Stage 6 — Runtime Optimization

Serve the complete model on `plowrt` to achieve TTFT and TPOT targets at declared concurrency while fitting memory.

* **Precondition:** Stage 5 gated out (block latency certified against roofline). Full-model packet emitted. Hardware parameters from [`target.md`](target.md) defined.
* **Gate out:** Target concurrency served with TTFT/TPOT within budget, memory fits without spurious shedding, all numerics-affecting levers verified, and parameters recorded in the model's `recipes/infervisor/<model>/<label>.toml`.

---

## 1. Runtime Execution Architecture

```
HTTP / UDS  →  Continuous Batching Mux (serve/mux.rs)  →  Device Engine (serve::engine::ServeEngine)
                Slot table · Admission · Hold              Persistent / Segmented Interpreter
```

### Engine Backends
* `Cuda` (`exec::gpu::GpuEngine`): Slotted sequence table, chunked prefill, VMM paging, prefix cache, on-device sampling.
* `CudaGroup` (`serve/cuda_group.rs`): Tensor-parallel CUDA execution across local process rank groups.
* `Amd` (`exec::amd::AmdServe`): Fixed-capacity sequence slots, compiled decode ladder, per-slot prefix snapshots, optional TP.
* `Cpu` (`serve/cpu_serve.rs`): Reference interpreter with pinned worker threads.

### Performance Authority
`plowrt bench` is the authoritative performance test. It executes the exact serving mux and engine while excluding model loading and warmup from measured intervals:
```bash
plowrt bench --assets <assets> --concurrency <C> --requests <N> --output-len <L>
```

---

## 2. Tuning Levers

Tune levers sequentially. Re-measure performance after each adjustment.

### 1. Concurrency & Decode Ladder
* Decode batch width and ladder are compiled into the packet (`PLOW_DECODE_BATCH`, `PLOW_DECODE_BATCH_LADDER`).
* `PLOW_OBJECTIVE` defaults to `auto`, switching dynamically between latency and throughput rules. Recipes must retain `auto`.
* Decode is bandwidth-bound against `$BW_BOUND`. Size decode width to meet TPOT targets rather than maximizing peak theoretical throughput.

### 2. KV Cache & Memory Budget
Compute memory fit before deployment:
```
ring            = min(ctx, next_pow2(window + chunk - 1))
sliding_per_seq = ring × n_slide × kvh_slide × hd_slide × 2 × sizeof(dtype)
full_per_seq    = ctx  × n_full  × kvh_full  × hd_full  × 2 × sizeof(dtype)
total_vram      = weights + B × (sliding_per_seq + full_per_seq) + activations
```
* `PLOW_MAX_CHUNK` (emit-time): Sizing lever for KV ring buffers. Smaller chunk sizes reduce memory footprint.
* `--kv-pool-mib` / `PLOW_KV_POOL_MIB`: Retains freed KV blocks in reserve pool.
* KV OOM manifests as admission shedding (`admit_into`), not a crash. Shrink chunk size or reduce concurrency; do not expand slot limits beyond physical VRAM.

### 3. Prefix Caching (Mandatory)
* Master flag: `--prefix-cache` / `PLOW_PREFIX_CACHE` (default enabled). Never disable in production recipes.
* VMM sharing: `--vmm-prefix` / `PLOW_VMM_PREFIX` maps shared full-attention KV across concurrent requests.
* Block granularity: `--vmm-block-mib` tunes deduplication granularity for long contexts.

### 4. Prefill Chunking & Interleaving
* `--pf-interleave` / `PLOW_PF_INTERLEAVE`: Limits prefill rows executed per tick to preserve decode TPOT under load.
* `--pf-chunk` / `PLOW_PF_CHUNK`: Caps per-request chunk size to control tail latency.
* `--amd-ragged-chunk` / `PLOW_RAGGED_CHUNK`: Executes final prefill chunks at exact row count (default enabled).
* *Caution:* Co-packed cross-request prefill affects attention split boundaries and requires numerical re-verification via `llm_fp32_ref`.

### 5. Admission & Request Scheduling
* `--slo-ms`: Sets baseline latency SLO. Controller sheds requests when estimated service time exceeds SLO.
* `--max-hold-ms`: Batch-formation hold time on idle engines.
* Metrics: Distinguish controller drops (`plowrt_admit_shed_total`) from true queue capacity exhaustion (`plowrt_rejected_total`).

### 6. Multi-GPU Modes
* **Tensor Parallelism (TP):** Emit-time configuration (`plowc --parallel tp`). Multi-rank execution verified via `plowrt devices --tp $NGPU`.
* **Data Parallelism (DP):** Serve-time configuration (`plowrt serve --dp N`). Hosts N independent replicas with per-replica prefix caches.
* **Prefill-Decode Disaggregation (PD):** Splits ranks into dedicated prefill and decode pools (`--pd model=0/1`).

---

## 3. Step-by-Step Execution

### Step 1: CPU Schedule Sanity
```bash
plowrt simulate --assets <assets> --all-buckets --chrome sim.json
```
Verify zero deadlocks, dependency satisfaction, and correct makespan.

### Step 2: Establish Engine Floor
```bash
# Concurrency 1 baseline
plowrt bench --assets <assets> --prompt-ids 1,2,3,4 --concurrency 1 --requests 8 --output-len 64

# Target concurrency
plowrt bench --assets <assets> --prompt-ids 1,2,3,4 --concurrency <target> --requests 32 --output-len 64
```
Extract TTFT, TPOT, ITL, and rung dispatch distributions.

### Step 3: Cold Prefill Sweep
Measure prefill scaling without prefix-cache reuse:
```bash
env PLOW_PREFIX_CACHE=0 plowrt bench --assets <assets> \
  --prefill-sweep --prefill-lengths 512,1024,2048,4096,8192 --prefill-reps 3
```

### Step 4: Full Serving Evaluation
Launch server and drive target traffic profile:
```bash
python3 scripts/campaign/campaign.py serve <recipe> --assets <assets>
```
Scrape `/metrics` to audit:
* `rate(plowrt_batch_size_sum[5m]) / rate(plowrt_batch_count_total[5m])`: Effective batch size.
* `plowrt_admit_shed_total`: Request sheds under load.

### Step 5: Optimization & Recording
Tune levers one variable at a time in order:
1. Decode ladder width (guided by TPOT budget).
2. Prefill interleaving / chunking (to control TTFT tails).
3. VMM prefix block size.
Record all qualified parameters in the production recipe TOML (`recipes/infervisor/<model>/<label>.toml`).

---

## 4. Attribution Ladder

When latency targets are missed, diagnose bottlenecks hierarchically:

| Tier | Diagnostic Question | Tool / Instrument | Vendor |
|---|---|---|---|
| 1 | Client vs. Server overhead? | `PLOW_TTFT_LOG=1` | Both |
| 2 | Which execution phase? | `PLOW_PF_SEG_TIME=1` | NVIDIA |
| 3 | Which operator in block? | `PLOW_PF_TRACE_LOG=1` (with `-DPLOW_NV_TRACE=1`) | NVIDIA |
| 4 | Host bottleneck vs. Device kernel? | `PLOW_STEP_TIME=1` (`gap_us` vs `dev_interp_ms`) | NVIDIA |
| 5 | Kernel-level stall mechanism? | `ncu` / `rocprof` profile under queue | Both |
| 6 | Hardware thermal/power throttling? | GPU clock, power, and throttle telemetry | Both |

---

## 5. Gate Checklist

Before advancing to Stage 7:
- [ ] TTFT within budget at concurrency 1 and controlled at target concurrency.
- [ ] TPOT within budget at target concurrency.
- [ ] Memory verified without OOM shedding (`PLOW_LOAD_PROFILE=1`).
- [ ] `plowrt_admit_shed_total` is zero under target load.
- [ ] All numerics-affecting levers (co-packed prefill, ragged tail, prefix sharing) pass `llm_fp32_ref`.
- [ ] Multi-GPU TP ranks produce bit-identical token streams across all sequences.
- [ ] Final serving parameters committed to canonical recipe TOML in `recipes/infervisor/`.
