# Stage 5 — Single-Block Sweep

Screen every kernel, fusion, route, or knob on an isolated layer block against calibrated roofline floors before whole-model benchmarking.

* **Hard Rule:** Every candidate must be screened on **one block** at served shape, precision, and rungs ([agent-tools.md](agent-tools.md)).
* **Precondition:** Stages 1–4 complete; full-model `model.pkt` emitted; [`target.md`](target.md) filled.
* **Gate out:** Block matches host oracle numerically **and** qualifies against `certify_kernel.py` ($\ge 80\%$ decode, $\ge 75\%$ prefill efficiency).

---

## 1. Measurement Tiers

| Tier | Harness | Purpose |
|---|---|---|
| **1: Prune** | Standalone kernel probes (`runtime/tests/`, `runtime/bench/`) | Numerical validation; discard non-viable candidates. |
| **2: Sweep** | `block_run bench` / `campaign.py block-bench` | Primary evaluation tier: real megakernel, isolated layer block. |
| **3: Confirm** | `examples/step_bench` | Whole-packet execution with real layer transitions; top 2–3 finalists. |
| **4: Accept** | `plowrt serve` grid ([Stage 7](07-perf-campaign.md)) | Serving-level admission, batching, and queue dynamics. |

*Tier 2 Scope:* Evaluates compute and memory bottlenecks. Cannot score prefill dispatch classing, bucket policies, or host runtime gaps.

---

## 2. Emission & Harness Tooling

### Emitting an Isolated Block
```bash
# General block emission
plowc --emit devblob --block <L> --arch $ISA --gpu $GPU --n-cu $NCU \
  --max-ctx $MAXCTX --num-gpus $NGPU --hf-dir <ckpt> --out <assets-dir>

# Recipe-exact block emission (NVIDIA)
python3 scripts/campaign/campaign.py build recipes/infervisor/<model>/<label>.toml \
  --block 4..6 --out <scratch>/block-build
```

### Execution Harnesses

| Platform | Numerical Oracle | Latency & Roofline Benchmark |
|---|---|---|
| **NVIDIA** | C oracles (`runtime/tests/*_block_*_test.c`) | `block_run bench` / `prefill-bench`<br>`campaign.py nv-block-ab`<br>`scripts/campaign/block_roofline.py` |
| **AMD** | C oracles / `plowrt amd-block --inspect` | `campaign.py block-bench`<br>`campaign.py block-ab`<br>`campaign.py block-roofline` |

---

## 3. Step-by-Step Execution

### Step 1: CPU Simulation Sanity
Confirm packet execution dependencies without hardware:
```bash
./scripts/block_sim.sh <hf-dir> decode:1:128
```
Validates makespan is finite, dependency DAG resolves, and memory bounds are respected.

### Step 2: Numerical Verification (Hard Gate)
Validate block outputs against host reference oracle before taking any timing measurements:
* **NVIDIA:** Run family C oracle harness (`runtime/tests/net_gemma_block_test.c`, etc.).
* **AMD:** Run `runtime/tests/<family>_real_oracle.py` under `gpuq.py submit`.
* **Precision delta checks:** Use `plowrt amd-block --inspect act.x,act.hn --dump <out.bin>` to ensure no unwritten tensors or NaN/Inf values exist.

### Step 3: Latency Benchmarking & Roofline Certification
Execute under `scripts/bench/gpuq.py submit`:

```bash
# Decode benchmark across batch and context
./target/release/examples/block_run <assets> bench \
  --batch 1,4,16 --ctx 128,1024,4096 --iters 100 --warmup 20

# Prefill benchmark across query rows and prefix lengths
./target/release/examples/block_run <assets> prefill-bench \
  --query-rows 512,2048 --prior 0,4096,16384

# Calibrated roofline report
python3 scripts/campaign/block_roofline.py <assets> \
  --ceilings ceilings.json --plow sweep.json --current <prod_sweep.json>

# Automated roofline certification gate
python3 scripts/campaign/certify_kernel.py --input sweep.json
```

Four-Arm A/B Comparison:
```bash
python3 scripts/campaign/campaign.py nv-block-ab \
  --control <prod-block> --treatment <cand-block> \
  --args "--batch 1,8 --ctx 4096 --iters 200" --ceilings ceilings.json --out <scratch>/ab
```
Enforces four interleaved runs (`ctrl1 → treat1 → ctrl2 → treat2`) in a single lease. Invalidates runs where treatment spread exceeds $3\times$ control drift.

---

## 4. Native Kernel Inner-Loop Contract

All kernels evaluated in the block harness must comply with:
* **Control flow:** Inner-loop branches must be warp/wavefront uniform.
* **Memory alignment:** Global transactions must be 16-byte coalesced. Gather patterns require documented justification.
* **Shared memory / LDS:** Swizzled or padded layouts to prevent bank conflicts (`ncu` conflict counter $= 0$).
* **Register & scratch spills:** Zero local memory / scratch spills in hot inner loops.
* **Synchronization:** Explicit producer-consumer synchronization. No divergent lane barriers.
* **Precision & aliasing:** Pointers marked `__restrict__`. Reductions performed in registers or shared memory before global atomics.

---

## 5. Gate Checklist

Before advancing to Stage 6:
- [ ] Block output matches reference oracle within precision tolerance across all substeps.
- [ ] Zero unwritten outputs, NaNs, or Infinities detected (`--inspect`).
- [ ] `certify_kernel.py` passes: $\ge 80\%$ decode efficiency (memory-bound) and $\ge 75\%$ prefill efficiency (compute-bound) against calibrated roofline.
- [ ] Mandatory 5-column block-vs-roofline table completed and included in report header.
- [ ] All benchmark runs executed uncontended under `gpuq.py` (exit code 0).
- [ ] Emission configuration matches target hardware (`$GPU`/`$ISA`).

---

## 6. Common Pitfalls

* **Whole-block vs. per-op scaling:** Isolated blocks contain un-fused norm/residual prologue kernels that disappear in whole-model execution. Rely on per-op GEMV and attention timings rather than raw whole-block latency.
* **Eager reference distortion:** Comparing Plow against un-captured eager PyTorch inflates speedup ratios. Always graph-capture reference comparisons.
* **MoE routing sensitivity:** MoE blocks depend on token routing distributions. Use representative router tables (`scripts/bench/router_table.py`) rather than uniform random routing.
