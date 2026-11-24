# Stage 4 — Kernel Tuning to the Roofline

Optimizes hot kernel families (prefill GEMM, decode GEMV, attention, MoE) on `$GPU` to a calibrated fraction of the hardware roofline.

* **Precondition:** Stage 3 passes. Hardware parameters filled in [`target.md`](target.md) (`$GPU`, `$ISA`, `$BW_BOUND`, `$COMPUTE_CEIL`, `$NCU`).
* **Hardware Guide:** Consult [`isa/README.md`](isa/README.md) for target instruction constraints before editing kernel code.
* **Gate out:** All hot shapes qualified against oracles and evaluated against calibrated roofline floor.

---

## 1. Principles & Promotion Order

* **Compiler reads, tuning writes:** `plowc` consumes qualified tuning records from `tuning/`; only an explicit tuning run publishes them.
* **Probed capability:** Use `plowc tune inventory` to extract actual tunable macros from compiled objects. Never rely on unverified `#define` assumptions.
* **Correctness precedes latency:** Measurements without passing oracle verification are marked `provisional` and rejected by compiler selection.

### Promotion Ladder
1. Derive exact shape, precision, layout, and output contract from emitted packet.
2. Beat vendor library (cuBLASLt, hipBLASLt, AITER) in same-session A/B on identical inputs with oracle checks.
3. Fused kernel must outperform the exact sequence of unfused operations it replaces.
4. Validate on isolated block ([Stage 5](05-single-block-sweep.md)) before full-model execution.

---

## 2. Tuning Workflow

### Step 0: Calibrate Hardware Ceilings
Do not rely on datasheet numbers. Measure empirical bounds on `$GPU`:
* **NVIDIA:**
  ```bash
  python3 scripts/bench/gpuq.py submit ceil 1 python3 scripts/bench/calibrate_ceilings.py --out <scratch>/ceilings.json
  ```
* **AMD:** Run `runtime/tests/decode_bw_probe.hip` (median of 3 uncontended runs). Land measured value in `crates/hwspec/src/registry.rs`.

### Step 1: Inventory Tunable Kernels
```bash
plowc --list-gpus                                         # Verify $GPU is recognized
plowc tune inventory --gpu "$GPU" --root .                # Enumerate kernels, macros, and aliases
plowc tune status --gpu "$GPU" --db tuning                # Check current tuning database
```

Audit built code objects for resource usage and register pressure:
* NVIDIA: `cuobjdump -symbols <obj>.cubin` and `cuobjdump -res-usage <obj>.cubin`
* AMD: `scripts/asm_audit.py [--expect scripts/asm_expect_$ISA.json] <obj>.elf`

### Step 2: Extract Production Shapes
Extract actual shapes demanded by the model compiler:
```bash
plowc --hf-dir <ckpt> --arch $ISA --max-ctx $MAXCTX --n-cu $NCU --num-gpus $NGPU tune shapes --gpu "$GPU"
PLOW_TUNE_DUMP=1 plowc --hf-dir <ckpt> --emit devblob ...   # Extract GEMV shapes for decode
```

### Step 3: Execute Tuning Sweeps
All GPU runs go through `gpuq.py submit`. Contended runs (exit code 76) must be discarded.

| Sweep | Architecture | Tool / Harness | Consumer |
|---|---|---|---|
| Prefill GEMM tiles | AMD | `plowc tune gemm --gpu "$GPU" --obj <objdir> --samples <out.jsonl>` | `devgen::pick_tile` |
| Decode GEMV rungs | AMD | `TUNE_GPU=$GPU scripts/gemv_campaign_lease.sh <obj> <out.jsonl> <tag>` | `tunedb-gemv` |
| Decode knobs | NVIDIA | `scripts/tune_decode_sweep.sh --block <L> --block-run <block_run>` | `tunedb-decode` |
| Attention split count | Both | Truncated-model sweep across shortest and longest context | Program metadata |

### Step 4: Roofline Attainment Analysis
* **Bandwidth-bound (Decode GEMV / Attention):** Compute bytes moved divided by execution time vs. `$BW_BOUND`.
* **Compute-bound (Prefill GEMM):** Compute delivered TFLOPS vs. `$COMPUTE_CEIL`.
* Check tile count and SM/CU occupancy. A candidate that improves latency while dropping occupancy may fail when integrated into the full engine.

### Step 5: Publish & Ingest Records
* **AMD Prefill GEMM:** Ingest records into tuning database:
  ```bash
  plowc tune ingest --samples <gemm.jsonl>
  ```
* **AMD Decode GEMV:** Ingest via `tunedb-gemv`:
  ```bash
  tunedb-gemv ingest --db tuning --gpu "$GPU" --samples <gemv.jsonl>
  ```
* **NVIDIA Decode Knobs:** Generate define string for cubin build:
  ```bash
  PLOW_EXTRA_DEFINES="$(tunedb-decode best --db tuning --hardware $VENDOR/$ISA/<sku> --print defines)"
  ```
* **Verification:** Verify selection selects the measured record:
  ```bash
  plowc tune select --gpu "$GPU" --shape M,N,K
  ```

---

## 3. Gate Checklist

Before advancing to Stage 5:
- [ ] Every hot shape identified in `tune shapes` is `HIT` or has documented justification.
- [ ] All winning records passed oracle verification and meet statistical significance (`Stats::beats`).
- [ ] `plowc tune select` returns `measured` rationale for published shapes on `$GPU`.
- [ ] Hardware roofline percentages are computed against calibrated empirical ceilings.
- [ ] All benchmark jobs completed uncontended (exit code 0; exit code 76 discarded).
- [ ] Changed objects pass resource audit: no register spill regressions, correct instruction mnemonics.

---

## 4. Common Pitfalls

* **Interpreter divergence:** Standalone micro-benchmarks have different register allocation than the monolithic interpreter. Always validate candidates inside interpreter harnesses.
* **Silent alias misses:** Dense GEMM opcodes alias on NVIDIA; sweeping tiles requires rebuilding the object with modified `#define` strings rather than passing runtime flags.
* **Calibrated vs. datasheet ceilings:** Quoting efficiency against datasheet ceilings invalidates roofline claims. Calibrate ceilings on the physical GPU.
