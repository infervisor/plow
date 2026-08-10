# Agent — Stage 4: Kernel Tuning to the Roofline

You are executing **Stage 4** of the model-bringup playbook. Your job: identify
the target model's hot kernels, **measure each family against its roofline
ceiling**, sweep the knobs that are genuinely knobs, **register the winners
where the compiler reads them**, and decide whether the kernels are ready for
Stage 5 (single-block sweep). Read `docs/bringup/04-kernel-tuning.md` first —
it has the full methodology, commands, and pitfalls; its backbone is
`docs/arch/11-tuning-coverage.md` (what is tunable) and
`docs/arch/12-using-the-tuner.md` (how to measure acceptably). This prompt is
the executable checklist.

## Preconditions (from Stages 1–3)

* Stage 3 passed: every compiled bucket verifies, so the schedule you time is
  the schedule that ships.
* You know the target GPU (`--gpu`, `--n-cu`, ISA) and have its vendor
  toolchain — `hipcc` or `nvcc` is required even for GPU-free probing, because
  capability is derived from a built object, never declared.
* You have the GPU itself and `perf-data/harness/gpulease` for measurement.
  Use `nix develop` for all cargo builds; `tuner` is a default `plowc` feature.

If any is missing, **stop and report** — a tuning campaign without the
toolchain or the lease produces numbers that cannot be trusted or filed.

## Inventory first — it determines the whole campaign

Before any GPU time, establish per family whether there is anything to tune:

```bash
plowc tune inventory --gpu "<name>" --root .     # executable kernels + ALIASES
plowc tune status   --gpu "<name>" --db tuning   # what the store already holds
```

* The **aliases** block is the most important output: opcodes listed together
  reach one body, and ranking within a group measures dispatch noise. On
  NVIDIA the dense-GEMM opcodes alias and the tile is object-wide — the tuning
  axis is *which object is built* (a `-D` grid), not which opcode is emitted.
  On AMD the same opcodes are distinct and genuinely rankable.
* Check every macro you plan to sweep with `kernelcaps::classify_macro` —
  `Overridable` only. A `-D` on a hard `#define` is a redefinition; the numbers
  describe a tile that was never built.
* Check the family has a **correctness oracle** (`11-tuning-coverage.md`,
  oracle table). No oracle → nothing you measure can become a `qualified`
  record. On this tree the time-and-validate harnesses are AMD
  (`interp_gemm_bench.c`, `qwen_interp_bench.c`, `dsa_gather_bench.c`,
  `tp_allreduce_bench.c`); NVIDIA timing and oracles live in separate files.

Record the verdict per family (GEMM / GEMV / attention / MoE / collectives)
before proceeding: tunable now, blocked-on-oracle, blocked-on-knob, or alias
no-op.

## Procedure

### 1. Establish the ceilings

* Analytic: the cost model's tile candidates come from `SmSpec.shared_mem`
  minus `costmodel::kernel_reservation_bytes(arch)`, double-buffered; DMA cost
  from datasheet bandwidth (ranking only). For **reported** roofline %, use
  measured denominators — `MemorySpec::bandwidth_for_bound()` for bandwidth,
  the measured sustained MFMA ceiling for compute (gfx950 1660 TF/s, gfx942
  937 TF/s — not the datasheet numbers).
* Machine: single-CU roofline of the production op bodies:
  ```bash
  cd runtime && cmake -B build -DPLOW_ROCM=ON -DPLOW_UBENCH=ON -DPLOW_HIP_ARCH=<arch>
  cmake --build build --target ubench
  cd ubench && ./run_ubench_cu.sh --timing-only        # --decode for M=1 shapes
  ```
  Evidence only (no oracle) — use it to classify each kernel compute- vs
  bandwidth-bound and to size the gap.

### 2. Derive the hot shapes from demand — never hand-author

```bash
plowc --hf-dir <ckpt> --max-ctx <c> --n-cu <N> --num-gpus <g> tune shapes --gpu <name>
PLOW_TUNE_DUMP=1 plowc --emit devblob ...     # TUNEDUMP_GEMV census for the decode side
```

The compile flags go **before** the word `tune` — a shape list is only correct
for one configuration. Hand-authored lists are how GLM-5.2 prefill ended up
100% unmeasured.

### 3. Sweep, under the lease

Every timed run: `perf-data/harness/gpulease <label> <cmd>`. Exit 76 =
contended = **failed measurement** (`tunedb::measurement_is_trustworthy`
accepts only 0). Wrap the run, not the build. Repeat until clean.

* **Prefill GEMM (AMD)** — one command measures, ingests, verifies:
  ```bash
  scripts/rebench_build_objs.sh <objdir>       # build test_kernels.elf objects (outside the lease)
  nix develop --command ./target/release/plowc \
      --hf-dir <ckpt> --max-ctx <c> --n-cu <N> --num-gpus <g> \
      tune gemm --gpu <name> --root . --obj <objdir> --samples <out.jsonl> --lease
  ```
  Manual alternative: `gemm_tile_sweep <M> <N> <K> [label] [quant]` with
  `PLOW_GEMM_JSONL=<path>`, then `plowc tune ingest --samples <path>`.
* **Decode GEMV (AMD):** `gemv_row_sweep <N> <K>` with `PLOW_GEMV_JSONL`
  (shape list via `scripts/rebench_tune_gemv.sh` from the census), then
  `tunedb-gemv ingest --db tuning --gpu <SKU> --samples <jsonl>`. `--gpu` is
  required — it decides the cell.
* **Decode knobs (NVIDIA object grid):** `scripts/tune_decode_sweep.sh`
  (sweeps `PLOW_NV_FORCE_MINBLK`, `GV_UNROLL*`, `GV_MM_MAX`, MoE knobs jointly,
  scored by end-to-end step TPOT), then `tunedb-decode ingest`.

Respect the couplings: `GV_MM_MAX` moves the register ceiling of every arm in
the object; MoE grouped GEMM shares the dense `PGM_*` tile; `PGM_BM` is
packet-layout-visible. These cannot be swept independently.

### 4. Interpret

* Compute the roofline % per family against the **binding** side: bytes/time
  vs `bandwidth_for_bound()` for decode GEMV/attention; TF/s vs measured
  sustained peak for prefill GEMM. Name which denominator you used.
* For small-M GEMM, read TILE COUNT / CU FILL — fill, not tile efficiency, is
  usually the limit at M=128. A tile that wins 3% while dropping fill has not
  won.
* A winner must beat under `Stats::beats` (median advantage exceeds jitter in
  both candidates). A 1 ns edge under 30 ns of spread is not a result.
* An occupancy delta is only real with a matched-grid control — the measured
  occ-2 "win" on decode flash was wave quantization
  (`perf-data/px16-decode-occupancy.md`).

### 5. Register winners and verify consumption

* AMD GEMM/GEMV winners → `qualified` records (oracle passed,
  ≥ `Stats::MIN_SAMPLES`, decisive). NVIDIA decode winner → build defines:
  ```bash
  PLOW_EXTRA_DEFINES="$(tunedb-decode best --db tuning --hardware <cell> --print defines)" \
    scripts/build_sm120_cubin.sh out/interp_sm120.cubin
  ```
* Then prove the compiler sees them:
  ```bash
  plowc tune status --gpu <name> --db tuning
  plowc tune select --gpu <name> --shape M,N,K    # expect rationale: measured, tier: sku-calibrated
  ```
  A miss is byte-identical to "never measured" — if a just-measured shape still
  reports `analytical cold start` / `portable`, the campaign published into the
  wrong cell or under a stale digest. This check is the difference between a
  working campaign and a green-gated no-op.

## Gate (into Stage 5)

Pass when **all** hold:

1. Every demanded hot-kernel shape is HIT in the store, or its MISS is
   explicitly explained (no oracle / no knob / alias group).
2. Winners are `qualified` records or recorded `-D` define strings; nothing
   selectable came from an oracle-less harness.
3. `plowc tune select` reports `measured` / `sku-calibrated` on covered shapes.
4. Per-family roofline % is written down with measured denominators and clean
   (`gpulease` rc=0) provenance.
5. Untunable families are documented as such — Stage 5 will re-check that
   "tuned" was real before trusting block latency.

## Pitfalls to actively guard against

* Ranking inside an alias group = dispatch noise; the "winner" will not
  reproduce. Body-level aliases (AMD MoE group ops wrap the expert ops) do not
  appear in the aliases block — known blind spot.
* Sweeping a hard `#define` measures a tile that was never built. Classify
  first. NVIDIA attention tiles are template literals, not macros — that sweep
  is a code edit, not a parameter search.
* Standalone kernels do not transfer: same tile family measured 212 vs 770
  TF/s standalone vs inlined in the interpreter. Measure through the
  interpreter (`interp_gemm_bench.c` pattern).
* Contended runs skewed a real campaign ~35%; rc=76 is a failed measurement,
  never a caveat.
* Wrong-cell publication is silent in both directions (the `tunedb-gemv`
  hardcoded-cell defect left the decode-GEMV census at zero records). Always
  run the `tune select` read-back.
* Calibration-only tiles (320x128, …) are measurements, not selectable facts —
  `gemm_rung_opcode` returns `None` for them; do not force them in.
* Stub kernels (`XFLASH_MERGE`, `XARGMAX_FIN`) benchmark at ~0 ns and win;
  only rank what `ProbedObject::executes` lists.
* Datasheet denominators overstate the gap (gfx942: 1307 vs 937 TF/s measured).
  Report % of the measured ceiling and say so.
* `plowc tune` does not run benchmarks itself (by design); do not wait for it
  to. The harnesses above are the measurement layer.

## When to stop and ask

* The family you must tune has **no correctness oracle** in-tree (e.g. MoE, or
  NVIDIA GEMM before the oracle/timer merge) — its winners can only be
  screening results; ask whether to build the oracle or proceed provisional.
* `plowc tune inventory` cannot derive an inventory (missing toolchain) or the
  inventory's defines do not match the object you are shipping.
* Every knob on the critical path classifies `Fixed`/asserted — the axis is a
  code change, not a sweep; surface it rather than editing kernels.
* You cannot get an uncontended card after repeated attempts.
* A published winner does not show up in `tune select` and the cell/digest
  audit does not explain why.
* The measured roofline % is far below ceiling and flat across every knob —
  that is a Stage 5/6 (block- or runtime-level) problem; report the per-kernel
  table and stop sweeping.

## Report back

* **Per-family verdict:** tunable / blocked (and on what) / alias no-op, per
  the inventory.
* **Shapes covered:** demand census size, HIT/MISS counts, explained misses.
* **Roofline table:** per hot kernel — achieved vs ceiling, which bound binds,
  which denominator (measured vs datasheet) was used.
* **Winners registered:** records qualified (cell, op cases, tiles) and/or the
  exact `-D` define string for the winning object; proof of consumption
  (`tune select` rationale/tier output).
* **Gate decision:** ready for Stage 5, or blocked (with the specific blocker).
* **Real-vs-ideal caveats:** contention retries, provisional-only families,
  oracle gaps, anything measured standalone rather than through the
  interpreter.
