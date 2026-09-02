# Agent — Stage 4: Kernel Tuning to the Roofline

## Target parameters — fill this in FIRST

This stage is target-specific. Read [`../target.md`](../target.md) and fill
every row before running anything. A row you cannot fill is a **blocker**, not
a default. Every command below is written in these names — **never substitute a
literal part name into a command.**

| param | value | source |
|---|---|---|
| `$VENDOR` | | `amd` or `nvidia` |
| `$ISA` | | the `--arch` string (`IsaLevel::arch_flag()`) |
| `$GPU` | | a name from `plowc --list-gpus` — do not guess one |
| `$NCU` | | `--n-cu`; leave `0` to take `$GPU`'s `sm_count` |
| `$NGPU` / `$PARALLEL` | | `--num-gpus` (default 1) / `--parallel` (must be `tp`) |
| `$MAXCTX` | | `--max-ctx` |
| `$TOOLCHAIN` | | `hipcc` or `nvcc`, + version |
| `$BUILD` | | `scripts/build_<isa>.sh` for `$ISA` |
| `$FEATURES` | | `--features hsa` or `--features cuda` |
| `$BW_BOUND` | | `MemorySpec::bandwidth_for_bound()` — **state whether it is measured or a datasheet fallback** |
| `$COMPUTE_CEIL` | | measured sustained matrix-core ceiling **on `$GPU`** (Step 1 below) |
| `$RESULTS` | | `perf-data/plow-<isa>/` if it exists, else `perf-data/` |

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
* The target block above is filled in, and `$TOOLCHAIN` is installed — it is
  required even for GPU-free probing, because capability is derived from a
  built object, never declared.
* You have `$GPU` itself and `perf-data/tools/gpulease` for measurement.
  Use `nix develop` for all cargo builds; `tuner` is a default `plowc` feature.

If any is missing, **stop and report** — a tuning campaign without the
toolchain or the lease produces numbers that cannot be trusted or filed.

## Inventory first — it determines the whole campaign

Before any GPU time, establish per family whether there is anything to tune:

```bash
plowc tune inventory --gpu "$GPU" --root .     # executable kernels + ALIASES
plowc tune status   --gpu "$GPU" --db tuning   # what the store already holds
```

* The **aliases** block is the most important output: opcodes listed together
  reach one body, and ranking within a group measures dispatch noise. On
  `$VENDOR = nvidia` the dense-GEMM opcodes alias and the tile is object-wide —
  the tuning axis is *which object is built* (a `-D` grid), not which opcode is
  emitted. On `$VENDOR = amd` the same opcodes are distinct and genuinely
  rankable.
* Check every macro you plan to sweep with `kernelcaps::classify_macro` —
  `Overridable` only. A `-D` on a hard `#define` is a redefinition; the numbers
  describe a tile that was never built.
* Check the family has a **correctness oracle** (`11-tuning-coverage.md`,
  oracle table). No oracle → nothing you measure can become a `qualified`
  record. On this tree the time-and-validate harnesses are on the AMD side
  (`interp_gemm_bench.c`, `qwen_interp_bench.c`, `dsa_gather_bench.c`,
  `tp_allreduce_bench.c`); on `$VENDOR = nvidia` timing and oracles live in
  separate files, so nothing there qualifies until they are merged.

Record the verdict per family (GEMM / GEMV / attention / MoE / collectives)
before proceeding: tunable now, blocked-on-oracle, blocked-on-knob, or alias
no-op.

For `$VENDOR = amd`, inspect the current [AITER](https://github.com/ROCm/aiter) implementation for the same
model/operator and hipBLASLt for GEMM before proposing a kernel body change.
Treat them as candidate schedules and measured ceilings, not code to copy or a
stored baseline. Record the upstream commit, toolchain, architecture, dtype and
complete live shape; a result from another shape or part is only a hypothesis.

## Procedure

### 1. Establish the ceilings

* Analytic: the cost model's tile candidates come from `SmSpec.shared_mem`
  minus `costmodel::kernel_reservation_bytes(arch)`, double-buffered; DMA cost
  from datasheet bandwidth (ranking only).
* Reported: fill `$BW_BOUND` and `$COMPUTE_CEIL` **for `$GPU`**, and record
  which kind each is. `MemorySpec::bandwidth_for_bound()` returns
  `bandwidth_measured` when the registry has it and **silently falls back to
  the datasheet peak when it does not** — which is the case for almost every
  entry. `$COMPUTE_CEIL` is a sustained figure measured here, not a datasheet
  peak; on one AMD part the two differ by 28%. Carrying either number from
  another part invents the result.
* Machine: single-CU roofline of the production op bodies (`$VENDOR = amd`;
  on nvidia use the `runtime/bench/nvidia/` occupancy probes instead):
  ```bash
  cd runtime && cmake -B build -DPLOW_ROCM=ON -DPLOW_UBENCH=ON -DPLOW_HIP_ARCH=$ISA
  cmake --build build --target ubench
  cd ubench && ./run_ubench_cu.sh --timing-only        # --decode for M=1 shapes
  ```
  Evidence only (no oracle) — use it to classify each kernel compute- vs
  bandwidth-bound and to size the gap.
* Vendor ceiling (`$VENDOR = amd`): measure each hot live shape against
  hipBLASLt or the matching AITER operator when one exists. Match ISA,
  dtype/scale layout, M/N/K, batch/context, head geometry, top-k and expert
  count. Verify both outputs against the same oracle. Record latency plus VGPR,
  AGPR, LDS and spills for the plow production object; a standalone wrapper
  with different occupancy is not evidence that the interpreter improved.

### 2. Derive the hot shapes from demand — never hand-author

```bash
plowc --hf-dir <ckpt> --arch $ISA --max-ctx $MAXCTX --n-cu $NCU --num-gpus $NGPU \
    tune shapes --gpu "$GPU"
PLOW_TUNE_DUMP=1 plowc --emit devblob ...     # TUNEDUMP_GEMV census for the decode side
```

The compile flags go **before** the word `tune` — a shape list is only correct
for one configuration. Hand-authored lists are how GLM-5.2 prefill ended up
100% unmeasured.

### 3. Sweep, under the lease

Every timed run: `perf-data/tools/gpulease <label> <cmd>`. Exit 76 =
contended = **failed measurement** (`tunedb::measurement_is_trustworthy`
accepts only 0). Wrap the run, not the build. Repeat until clean.

Which of the following applies is decided by `$VENDOR`; the other does not
exist on your target.

* **Prefill GEMM (`$VENDOR = amd`)** — one command measures, ingests, verifies:
  ```bash
  scripts/rebench_build_objs.sh <objdir>       # build test_kernels.elf objects (outside the lease)
  nix develop --command ./target/release/plowc \
      --hf-dir <ckpt> --arch $ISA --max-ctx $MAXCTX --n-cu $NCU --num-gpus $NGPU \
      tune gemm --gpu "$GPU" --root . --obj <objdir> --samples <out.jsonl> --lease
  ```
  Manual alternative: `gemm_tile_sweep <M> <N> <K> [label] [quant]` with
  `PLOW_GEMM_JSONL=<path>`, then `plowc tune ingest --samples <path>`.
* **Decode GEMV (`$VENDOR = amd`):** `gemv_row_sweep <N> <K>` with
  `PLOW_GEMV_JSONL` (shape list via `scripts/rebench_tune_gemv.sh` from the
  census), then `tunedb-gemv ingest --db tuning --gpu "$GPU" --samples <jsonl>`.
  `--gpu` is required — it decides the cell.
* **Decode knobs (`$VENDOR = nvidia`, object grid):** `scripts/tune_decode_sweep.sh`
  (sweeps `PLOW_NV_FORCE_MINBLK`, `GV_UNROLL*`, `GV_MM_MAX`, MoE knobs jointly,
  scored by end-to-end step TPOT), then `tunedb-decode ingest`.
* **Decode-attention split count (packet grid):** when the model exposes a
  split knob (K3: `PLOW_K3_NS`), emit matched packets across a geometric grid
  such as 16/32/64/128 while reusing the exact weights and interpreter object.
  Confirm only attention/merge packets and scratch extents differ. Measure the
  shortest context, longest context, and enough crossover points to derive a
  context table. Score with `plowrt serve` and the pinned `vllm bench serve`
  client, not an isolated kernel timer. The curve is U-shaped: more splits fill
  more CUs, but merge work grows with the partial count. Split boundaries change
  reduction association, so require rank/counter audits and the model quality
  gate rather than byte-identical cross-arm text. This is consumed as packet
  programs/runtime context selection, not a tune-store record. Screen the broad
  grid first with `K3_FULL=1 PLOW_K3_LAYERS=single:3` and a truncated
  real-model asset containing that MLA layer plus its production
  merge/output/residual tail; an isolated flash kernel omits the merge cost.
  Promote only the 2–3 finalists to full-model serving. Do not add a runtime
  context ladder unless matched crossover wins exceed noise; a fixed winner has
  less packet/scratch state.

Respect the couplings: `GV_MM_MAX` moves the register ceiling of every arm in
the object; MoE grouped GEMM shares the dense `PGM_*` tile; `PGM_BM` is
packet-layout-visible. These cannot be swept independently.

### 3a. Agent iteration loop (AMD)

Use [CUDA-Agent](https://arxiv.org/abs/2602.24286)'s implement → compile → verify → profile loop, with Plow's
stronger controls:

1. Freeze the oracle, workload generator, lease wrapper and timing parser before
   the baseline. If one changes, invalidate and re-run the baseline.
2. Write one hypothesis and expected counter movement before each candidate.
3. Change one algorithmic lever first; tile/unroll searches come only after the
   bottleneck is measured. Keep runtime-selectable candidates in one object only
   when that does not raise production-object registers or instruction footprint.
4. Compile, run the adversarial oracle grid, audit the production object, then
   measure repeated same-session A/B samples. Reject correctness failures,
   spills, route misses, contended samples and wins below `Stats::beats`.
5. Append every result, including losses, to the campaign record. Promote only
   the best qualified, current-digest candidate; remove dead candidate code after
   the record is complete.

A fixed 5% target is not proof of improvement. Precision, scale layout and
tolerance must be identical across arms. Stop at a decisive measured win or
exhausted hypotheses, not after a fixed number of agent turns.

### 3b. Prefill fusion promotion ladder (AMD)

Keep this procedure model-agnostic. Derive every operator, shape, dtype,
consumer count, parallel degree and prompt bucket from the emitted artifact and
the target workload. Exclude speculative/draft-model work from this campaign.
A fused kernel is not a candidate for network timing until it decisively beats
the exact unfused sequence it replaces.

#### Discover candidates; do not encode a model recipe

Audit the production trace for these general seams:

- projection + position transform, or related projections sharing an input;
- attention input projection + attention, including latent/absorbed forms;
- gate + up projection + activation;
- sparse routing/sort/align + grouped expert work, empty-bin skipping, and
  expert output + deterministic combine;
- collective + residual + normalization;
- recurrent/state update + gate/normalization;
- any producer/consumer pair whose intermediate is written to HBM and has one
  semantic consumer.

Reject a proposed fusion before coding when it duplicates a reduction across
consumers, changes a required association/order, extends buffer lifetime,
crosses workgroups without a valid communication mechanism, or makes the
production object's register/LDS ceiling worse. A decode-only fused opcode is
not a prefill implementation; absent dispatch arms must refuse loudly rather
than silently leave an output unwritten.

#### Fast-path and layout audit

For every live shape, record the selected packet opcode, object, device body and
fallback reason. Library divisibility/alignment constraints can route a valid
shape onto a much slower generic path. Test native arbitrary-shape support first;
padding is a candidate only when `fast_path(padded) + pad/unpad` beats the native
path despite the extra work. Compare against the current AITER/hipBLASLt route
for the same shape, but do not import its padding, preshuffle or quant layout
without an end-to-end conversion-cost measurement.

Treat a lower-bit weight/KV format as a system candidate, not a kernel flag:
include pack/quant/dequant work, scale traffic, persistent weight layout, KV
capacity/headroom and any change in the minimum feasible TP degree. Do not
credit memory saved unless the production allocator or topology consumes it.

#### Gates, in order

1. **Paired kernel gate.** Retain a same-revision switch for the exact unfused
   decomposition. Time that sequence and the fused candidate in one GPU timing
   region with identical buffers, inputs, scales, routing tables and warmup. The
   fusion must beat the sequence *as executed*, including launches and
   materialized intermediates. Verify every deleted intermediate boundary and
   final output.
2. **Shape gate.** Cover every emitted bucket plus ragged boundaries, shortest
   and longest served contexts, supported precisions, minimum/maximum batch,
   empty sparse partitions, maximum top-k and adversarial imbalance. Audit ISA,
   VGPR/AGPR/LDS, spills, achieved occupancy and the live route.
3. **Operator-chain gate.** Put the winner back beside its real producers,
   consumers and collectives. Reconcile the observed delta with packet counts,
   intermediate bytes removed and profiler counters. An unexplained inversion
   blocks promotion.
4. **Block gate.** Run a real block of each architecture class present in the
   artifact. A kernel winner that loses the block is rejected.
5. **Composition gate.** Enable only individually qualified fusions, first in
   pairs and then as a set. Re-run correctness and timing after every addition;
   object-wide register, instruction-cache and scheduling effects can reverse
   isolated wins.
6. **Network gate.** Emit the complete checkpoint and run cold/prefix-miss and
   warm/prefix-hit prefill through `plowrt serve` with the same artifacts and
   client on both arms. Report TTFT distributions, prefill tok/s, memory use and
   decode regression checks at workload-derived prompt lengths and concurrency.

At network scope, report collective time separately and evaluate the lowest TP
degree that fits. A topology change is a separate experimental cell, never
credited to a kernel fusion. Likewise, profile Plow's existing submission path
before proposing graph capture; capture is a separate runtime experiment, not
evidence that a fused body improved.

Record one row per `(candidate, scope, shape, artifact digest)`, with `scope` =
`kernel`, `operator-chain`, `block`, `composed-block`, or `network`. Adoption
requires every scope. Never multiply an isolated speedup by layer count to claim
a network win.

### 3c. General AMD candidate families

Derive these from the emitted graph and live trace; never key the harness on a model name.

1. **Long-sequence recurrence:** replace a serial token recurrence with a chunked parallel scan
   only when the state transition is associative or has an exact chunk composition. Sweep chunk
   and subchunk sizes. Carry sequence boundaries and reset metadata from the host; never rebuild
   them with a blocking device-to-host read. Keep the serial body as the oracle and fallback.
2. **Ownership-preserving recurrent fusion:** test one workgroup owning a complete recurrent head
   or state shard and fusing its convolution/state update, normalization and output gate. Account
   for all deleted intermediate traffic and packet edges. A partial fusion that leaves a grid
   rendezvous is a separate candidate, not evidence for the whole-head form.
3. **Skinny projection split-K:** sweep split count by exact `(arch, CU, object digest, op, M, N,
   K, dtype, layout)` and include the reduction kernel in the timed region. Do not copy a vendor
   table row or use one split for every rung. Missing/stale rows fall back loudly.
4. **Grouped sparse work:** tune both expert stages by token bucket and observed expert
   distribution, including persistent vs non-persistent and atomic vs reduction forms. Gate empty,
   maximally skewed and random expert bins; finite output and deterministic replay are mandatory.
5. **Attention split/persistence:** select candidates by decode rung and KV-length bucket. Measure
   split and merge together, including scratch traffic. Prefer deterministic table lookup at
   runtime; online timing is a separate experiment and must not enter request latency silently.

For every family, distinguish `compiled`, `qualified`, and `selected` in `build.json` or the
runtime result. A compiled arm or a populated database is not evidence that production selected
it. Refuse promotion when the artifact has no manifest, Lean was skipped without an accepted
reason, or any demanded tuned shape resolves to analytical fallback.

### 4. Interpret

* Compute the roofline % per family against the **binding** side: bytes/time
  vs `$BW_BOUND` for decode GEMV/attention; TF/s vs `$COMPUTE_CEIL` for prefill
  GEMM. Name which denominator you used and whether it is measured on `$GPU`.
* For small-M GEMM, read TILE COUNT / CU FILL — fill, not tile efficiency, is
  usually the limit at M=128. A tile that wins 3% while dropping fill has not
  won.
* A winner must beat under `Stats::beats` (median advantage exceeds jitter in
  both candidates). A 1 ns edge under 30 ns of spread is not a result.
* An occupancy delta is only real with a matched-grid control — the measured
  occ-2 "win" on decode flash was wave quantization
  (`perf-data/px16-decode-occupancy.md`).
* On AMD, collect profiler counters selected by the hypothesis, not every
  metric. Relate MFMA utilization, memory stalls/TCC traffic, LDS conflicts and
  achieved occupancy to the change. A resource limiter alone does not establish
  elapsed-time impact.

### 5. Register winners and verify consumption

* On `$VENDOR = amd`, GEMM/GEMV winners → `qualified` records (oracle passed,
  ≥ `Stats::MIN_SAMPLES`, decisive). On `$VENDOR = nvidia` the decode winner is
  an object — consume it as build defines through `$BUILD`:
  ```bash
  PLOW_EXTRA_DEFINES="$(tunedb-decode best --db tuning \
      --hardware $VENDOR/$ISA/<sku-slug> --print defines)" \
    $BUILD out/interp_$ISA.cubin
  ```
  `--print` refuses when the filter leaves more than one cell standing; take the
  `<sku-slug>` from the store layout `tuning/<vendor>/<isa>/<sku>/`.
* Then prove the compiler sees them:
  ```bash
  plowc tune status --gpu "$GPU" --db tuning
  plowc tune select --gpu "$GPU" --shape M,N,K   # expect rationale: measured, tier: sku-calibrated
  ```
  A miss is byte-identical to "never measured" — if a just-measured shape still
  reports `analytical cold start` / `portable`, the campaign published into the
  wrong cell or under a stale digest. This check is the difference between a
   working campaign and a green-gated no-op.

### 5a. Publish shape-keyed GEMV workgroup findings

For a GEMV workgroup cap that is not yet represented by a typed tuner record,
publish the measured result as an emit-time table while preserving the default:

```bash
PLOW_GEMV_WG_TUNING='896x7168=224,1536x7168=152' \
  plowc --emit devblob ...
```

The table is exact `(N,K)` matching and is consumed only by the K3 blocked-GEMV
emitter. An unset table must be the control packet; compare its packet census,
object resources, correctness output, and served TPOT before accepting a tuned
table. Keep the table in the campaign record with GPU/ISA/compiler/object
digests and the clean lease evidence.

The promotion path is:

1. Run the single-block sweep and reject contended or oracle-less samples.
2. Ingest the JSONL into `tunedb-gemv`; require a qualified, SKU/digest-matched
   record and a decisive `Stats::beats` result.
3. Generate the shape table from those qualified records, then re-emit and run
   the whole-model gate. The table is a reproducible projection of the database,
   not a second source of truth.
4. If no qualified record exists, leave the shape at the default. A heuristic
   may rank candidates for the next sweep, but it must not silently alter the
   emitted packet or become the default without measurements.

This keeps old performance as the hard control: unset means no extra parser
branch in emission and the normal `blocked_gemv_cus` mapping.

## Gate (into Stage 5)

Pass when **all** hold:

1. Every demanded hot-kernel shape is HIT in the store, or its MISS is
   explicitly explained (no oracle / no knob / alias group).
2. Winners are `qualified` records or recorded `-D` define strings; nothing
   selectable came from an oracle-less harness.
3. `plowc tune select` reports `measured` / `sku-calibrated` on covered shapes.
4. Per-family roofline % is written down against `$BW_BOUND` / `$COMPUTE_CEIL`
   as established for `$GPU`, with the denominator's kind stated and clean
   (`gpulease` rc=0) provenance.
5. Untunable families are documented as such — Stage 5 will re-check that
   "tuned" was real before trusting block latency.
6. No command in the campaign contained a literal part name; every `--gpu`,
   `--arch` and `--n-cu` came from the target block.
7. Every changed AMD object passed the resource/ISA audit: no spill regression,
   expected VGPR/AGPR/LDS and occupancy, the expected MFMA mnemonic for `$ISA`,
   and a live marker/routing proof that the measured arm executed.

## Pitfalls to actively guard against

* Ranking inside an alias group = dispatch noise; the "winner" will not
  reproduce. Body-level aliases (AMD MoE group ops wrap the expert ops) do not
  appear in the aliases block — known blind spot.
* Sweeping a hard `#define` measures a tile that was never built. Classify
  first. On `$VENDOR = nvidia` the attention tiles are template literals, not
  macros — that sweep is a code edit, not a parameter search.
* Standalone kernels do not transfer: on one measured part the same tile family
  scored 212 vs 770 TF/s standalone vs inlined in the interpreter. Measure
  through the interpreter (`interp_gemm_bench.c` pattern) on `$GPU`.
* Contended runs skewed a real campaign ~35%; rc=76 is a failed measurement,
  never a caveat.
* Wrong-cell publication is silent in both directions (the `tunedb-gemv`
  hardcoded-cell defect left the decode-GEMV census at zero records for the
  other ISA). Always run the `tune select` read-back.
* Calibration-only tiles (320x128, …) are measurements, not selectable facts —
  `gemm_rung_opcode` returns `None` for them; do not force them in.
* Stub kernels (`XFLASH_MERGE`, `XARGMAX_FIN`) benchmark at ~0 ns and win;
  only rank what `ProbedObject::executes` lists.
* Datasheet denominators overstate the gap — on one AMD part, 1307 datasheet vs
  937 TF/s measured. Report % of `$COMPUTE_CEIL` and say which kind it is; and
  remember `$BW_BOUND` is itself the datasheet peak unless `$GPU`'s registry
  entry carries a measured bandwidth.
* `plowc tune` does not run benchmarks itself (by design); do not wait for it
  to. The harnesses above are the measurement layer.

## When to stop and ask

* Any row of the target block cannot be filled — in particular `$GPU` is not in
  `plowc --list-gpus`, `$ISA` has no `$BUILD` script, or `$BW_BOUND` /
  `$COMPUTE_CEIL` cannot be measured on this box. Do not substitute another
  part's value.
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

* **The filled target block** — every row, with `$BW_BOUND` and `$COMPUTE_CEIL`
  labelled measured-here or datasheet-fallback.
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
