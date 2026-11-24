# Stage 4 — Kernel tuning to the roofline

Bring each hot kernel family (prefill GEMM, decode GEMV, attention, MoE) to a stated fraction of
its roofline on `$GPU`, and record the winners where the compiler reads them.

- **Precondition:** Stage 3 passes (the schedule you time is the schedule that ships). The
  [`target.md`](target.md) block is filled; every command uses `--gpu $GPU --arch $ISA`. Read the
  ISA reference for `$ISA` ([`isa/README.md`](isa/README.md)) before changing a kernel.
- **Gate out:** see [Gate](#gate-into-stage-5).
- **References:** [`docs/arch/11-tuning-coverage.md`](../arch/11-tuning-coverage.md) (what is
  tunable per family and what blocks it), [`docs/arch/12-using-the-tuner.md`](../arch/12-using-the-tuner.md)
  (how to take a measurement the store accepts).
- **Cross-cutting rules:** single-block screening [agent-tools.md](agent-tools.md) "HARD RULE";
  GPU access [agent-tools.md](agent-tools.md) "Queue"; noise [agent-tools.md](agent-tools.md) "A/B rule";
  kernel design/diff reviews [agent-tools.md](agent-tools.md) "Working rules".

## Principles

- **`compile` reads, `tune` writes.** `plowc` may consume qualified records; only an explicit
  tuning run publishes one.
- **Capability is probed from a built object, never declared.** `plowc tune inventory` answers
  "which kernels exist, which macros are knobs" with the vendor preprocessor, no GPU.
- **Correct before fast.** A measurement without a passing oracle is stored `provisional` and is
  never selected.

## Promotion order

1. Derive the exact production shape, dtype, scale layout, strides and output contract from the
   emitted packet and the reference framework's dispatch.
2. Beat the matching vendor/library kernel in a same-session A/B on the same inputs and oracle. A
   nearby shape or an unfused vendor fallback is not a comparator.
3. A fused kernel beats the exact unfused sequence it replaces.
4. Beat the reference framework at one isolated block ([05](05-single-block-sweep.md)).
5. Run the full network only as the final correctness and promotion gate.

Harness tiers: a production-header probe proves correctness and rejects gross losers; a
single-block or truncated-model sweep ranks the grid; only the 2–3 block winners run whole-model
`step_bench`; only a gated whole-model winner enters serving. Search `runtime/bench/`,
`runtime/tests/`, `scripts/` and `perf-data/tools/` first and extend the closest harness. Record
which tier produced each number.

Rank work by `block op share × measured improvement` from a short- and a long-context block trace.
A kernel still several times behind its library peer needs a different algorithm, reuse, tiling or
occupancy, not instruction-level variants.

## Step 0 — ceilings

- **Analytic (ranking only).** `costmodel::CostModel::new` takes `SmSpec`, subtracts
  `kernel_reservation_bytes(arch)` from shared memory and enumerates double-buffered tiles; DMA
  cost uses datasheet bandwidth. An absolute floor must use `MemorySpec::bandwidth_for_bound()`.
- **Reported.** `$BW_BOUND` and `$COMPUTE_CEIL` are measured on `$GPU`; name which kind of
  denominator every % uses. `bandwidth_for_bound()` falls back to the datasheet peak when the
  registry entry has no `bandwidth_measured`.
  - NVIDIA: `scripts/bench/calibrate_ceilings.py --out <scratch>/ceilings.json` (HBM read,
    BF16/FP8 GEMM), via the queue.
  - AMD: `runtime/tests/decode_bw_probe.hip`, median of three clean runs; land the value in the
    `$GPU` registry entry with a test asserting `bandwidth_for_bound()` returns it (pattern:
    `measured_bandwidth_governs_a_reported_bound_on_*` in `crates/hwspec/src/registry.rs`).
  - `gemm_tile_sweep.c` takes `PEAK_TFLOPS` / HBM constants as overridable macros for this.
- **Single unit (AMD, evidence only).** `runtime/ubench/` runs the production op headers on one CU:

  ```bash
  cd runtime && cmake -B build -DPLOW_ROCM=ON -DPLOW_UBENCH=ON -DPLOW_HIP_ARCH=$ISA
  cmake --build build --target ubench
  cd ubench && ./run_ubench_cu.sh --timing-only   # --decode for M=1, --kernel <k>, --json <f>
  ```

  No oracle → never a record. On NVIDIA use the occupancy probes in `runtime/bench/nvidia/`.
- **Vendor ceiling (AMD).** Time each hot live shape against hipBLASLt or the matching AITER
  operator: same ISA, dtype/scale layout, M/N/K, batch/context, head geometry, top-k, expert count,
  same oracle. Record upstream commit, toolchain and complete shape; another shape or part is a
  hypothesis. Trace the framework to the kernel it dispatches and include its padding,
  quantization, sorting, reductions and epilogues so both sides time the same boundary. Record
  latency plus VGPR/AGPR/LDS/spills of the plow production object.

## Step 1 — inventory

```bash
nix develop --command cargo build -p plowc --bin plowc   # `tuner` is a default feature
plowc --list-gpus                                          # recognized $GPU names
plowc tune inventory --gpu "$GPU" --root .                 # executable kernels + aliases
plowc tune select --gpu "$GPU" --shape M,N,K               # what a shape gets, and why
plowc tune status --gpu "$GPU" --db tuning                 # what the store holds
```

Per family decide: distinct kernel (not an alias)? a knob (`#ifndef`-guarded macro)? an oracle?
Record the verdict per family: tunable, blocked-on-oracle, blocked-on-knob, alias no-op.

- **Aliases first.** Opcodes listed together reach one body; ranking them measures dispatch noise.
  On NVIDIA the dense-GEMM opcodes alias and the tile is object-wide (`devgen::nvidia_prefill_gemm_op`
  emits one opcode per dtype; `PGM_*`/`PGM90_*` defines pick the tile), so the axis is which
  object is built. On AMD the opcodes are separate instantiations and rankable.
- **Knob check.** `kernelcaps::classify_macro(header, "<MACRO>") == Sweepable::Overridable`. A `-D`
  on a hard `#define` measures a tile that was never built. NVIDIA attention tiles are template
  literals: sweeping them is a code edit.

Condensed coverage (full table in `11-tuning-coverage.md`):

| family | knob | oracle | verdict |
|---|---|---|---|
| prefill GEMM | AMD `GM_BM`/`GM_BN`/`GM_BK`; NV `PGM_*`, `PGM90_*` | AMD yes | tunable (AMD); NV object grid |
| decode GEMV | `GV_*` unroll, `GV_MM_MAX`, `PLOW_GEMV_MM`/`WALK` | AMD yes | tunable |
| MoE | shares dense `PGM_*` tile | weak | blocked |
| attention / MLA / DSA | AMD `FA_*`; NV template literals | DSA strongest | partial |
| collectives | `PLOW_XR_CUS` | multi-GPU | needs ≥2 GPUs |

### Step 1b — audit the built object

A missing arm is usually a silent fallback, not a crash. Ask the built artifact:

| `$VENDOR` | entry symbols | resources / instruction mix |
|---|---|---|
| nvidia | `cuobjdump -symbols <obj>.cubin` | `cuobjdump -res-usage <obj>.cubin` |
| amd | `readelf -s <obj>.elf` | `scripts/asm_audit.py [--expect scripts/asm_expect_$ISA.json] <obj>` (MFMA choice, operand format, scratch spill) |

- An interpreter object's register count is the max over every inlined arm: a wide batching arm
  taxes `B=1` too.
- Spill is not slow by itself; measure before pruning on `ptxas -v`.
- A big body that underperforms in-model but not standalone: suspect the translation unit first.
  The lever is which object is built (an arm-stripped TU), which is why the NVIDIA axis is a
  define set.

## Step 2 — hot shapes from the compiler's demand

Never hand-author a shape list. Compile flags go before `tune`:

```bash
plowc --hf-dir <ckpt> --arch $ISA --max-ctx $MAXCTX --n-cu $NCU --num-gpus $NGPU tune shapes --gpu "$GPU"
PLOW_TUNE_DUMP=1 plowc --hf-dir <ckpt> --emit devblob ...   # TUNEDUMP_GEMV census (decode side)
```

## Step 3 — sweeps

GPU runs go through the queue ([agent-tools.md](agent-tools.md) "Queue"). `gpulease` exit 76 =
contended = failed measurement (`tunedb::measurement_is_trustworthy` accepts only 0). Build
outside the lease. The drivers marked *self-leasing* call `perf-data/tools/gpulease` per GPU command
themselves; gpulease does not nest, so do not wrap them in `gpuq.py submit`.

| sweep | `$VENDOR` | harness | consumer |
|---|---|---|---|
| prefill GEMM tiles | amd | `plowc … tune gemm` (measure + ingest + verify) or `gemm_tile_sweep <M> <N> <K> [label] [quant]` with `PLOW_GEMM_JSONL=<path>` then `plowc tune ingest --samples <path>` | `devgen::pick_tile` |
| decode GEMV rungs | amd (gfx950) | `TUNE_GPU=$GPU scripts/gemv_campaign_lease.sh <fresh OBJ> <jsonl> <campaign> -- <emit cmd>` (*self-leasing*; builds in `nix develop`, sweeps the census, ingests only clean, oracle-passing samples) | `tunedb-gemv` |
| decode knob grid | nvidia | `scripts/tune_decode_sweep.sh --block <L> --block-run <block_run>` (*self-leasing*): `PLOW_NV_FORCE_MINBLK`, `GV_UNROLL*`, `GV_MM_MAX`, MoE knobs; re-run the 2–3 winners without `--block` | `tunedb-decode ingest --db tuning --results <jsonl>` |
| decode-attention split count | both | matched packet arms (below) | packet/program choice, not the tune store |
| single-CU roofline | amd | `run_ubench_cu.sh` | none (evidence) |
| dispatch floor | both | `runtime/bench/dispatch/interp_dispatch_floor.hip` / `_nv.cu` | `provisional` only |

Prefill-GEMM campaign (AMD). `$BUILD <objdir>` writes `test_kernels.elf` beside the interpreter
objects (`PLOW_HSACO_CONFIG=<assets>` builds for one packet):

```bash
$BUILD <objdir>
python3 scripts/bench/gpuq.py submit tune-gemm 1 nix develop --command ./target/release/plowc \
  --hf-dir <ckpt> --arch $ISA --max-ctx $MAXCTX --n-cu $NCU --num-gpus $NGPU \
  tune gemm --gpu "$GPU" --root . --obj <objdir> --samples <scratch>/gemm.jsonl
```

`--shapes auto` (default) derives shapes from the compile; `--dry-run` stops after deriving;
`--no-ingest` measures without publishing.

Couplings: `GV_MM_MAX` moves the register ceiling of every arm in the object; MoE grouped GEMM
shares the dense `PGM_*` tile; `PGM_BM` is packet-layout-visible. Do not sweep them independently.

### Decode-attention split sweep

`nsplit` is a packet decomposition: more splits fill more CUs, the merge grows with the partial
count, so the curve is U-shaped and context-dependent.

- Re-emit only the packet per arm; reuse the exact weights and interpreter object. Prove by
  disassembly that only attention/merge packets and their scratch extents changed.
- Screen on a truncated real-model asset with one production attention layer plus its
  merge/output/residual tail (an isolated flash kernel omits the merge). Sweep shortest and longest
  served context, then crossover points. Only finalists go to whole-model serving.
- Split boundaries change reduction association: require finite outputs, rank/counter audits and
  the model quality gate, not byte-identical text.
- Add a context ladder only when matched crossover wins exceed noise; it selects a pre-emitted
  program by live KV length, never patches packet instructions or resizes scratch on the hot path.

### AMD iteration loop

1. Freeze oracle, workload generator, queue wrapper and timing parser before the baseline; if one
   changes, re-run the baseline.
2. Write one hypothesis and the expected counter movement before each candidate.
3. Change one algorithmic lever first; tile/unroll searches only after the bottleneck is measured.
   Keep runtime-selectable candidates in one object only if production registers and instruction
   footprint do not grow.
4. Compile, run the adversarial oracle grid, audit the production object, then same-session A/B.
   Reject correctness failures, spills, route misses, contended samples and wins below
   `Stats::beats`.
5. Record every result, losses included. Promote only the best qualified current-digest candidate;
   remove dead candidate code.

Precision, scale layout and tolerance are identical across arms. Stop at a decisive win or
exhausted hypotheses. On AMD collect only the counters the hypothesis names (MFMA utilization,
memory stalls/TCC traffic, LDS conflicts, achieved occupancy); a resource limiter alone does not
prove elapsed-time impact.

### Prefill fusion ladder (AMD)

Derive every operator, shape, dtype, consumer count, parallel degree and prompt bucket from the
emitted artifact and workload; never key a harness on a model name.

- **Seams to audit:** projection + position transform; projections sharing an input; attention input
  projection + attention (latent/absorbed forms); gate + up + activation; routing/sort/align +
  grouped experts + deterministic combine; collective + residual + norm; recurrent state update +
  gate/norm; any producer/consumer pair whose HBM intermediate has one consumer.
- **Reject before coding** a fusion that duplicates a reduction across consumers, changes required
  association/order, extends buffer lifetime, crosses workgroups without a valid mechanism, or
  raises the object's register/LDS ceiling. Absent dispatch arms refuse loudly.
- **Fast-path audit:** record selected opcode, object, body and fallback reason per live shape.
  Padding is a candidate only when `fast_path(padded) + pad/unpad` beats the native path. A
  lower-bit weight/KV format is a system candidate: include pack/quant/dequant, scale traffic,
  layout, KV headroom and minimum TP degree.

Gates, in order (one row per `(candidate, scope, shape, artifact digest)`; adoption needs every
scope; never multiply an isolated speedup by layer count):

1. **Kernel** — fused vs the exact unfused sequence, one timing region, identical buffers/inputs;
   verify every deleted intermediate and the final output.
2. **Shape** — every bucket, ragged boundaries, shortest/longest context, min/max batch, empty
   sparse partitions, max top-k, adversarial imbalance; ISA/VGPR/AGPR/LDS/spill/occupancy audit.
3. **Operator chain** — beside real producers, consumers and collectives; reconcile with packet
   counts, bytes removed and counters.
4. **Block** — a real block of each architecture class present.
5. **Composition** — qualified fusions in pairs, then as a set; re-gate after each addition.
6. **Network** — full checkpoint, cold and prefix-hit prefill through `plowrt serve`, same client
   both arms; report collective time separately. A topology change is its own cell.

Rows follow [`docs/schemas/fusion-ladder-row-v1.schema.json`](../schemas/fusion-ladder-row-v1.schema.json):

```bash
python3 perf-data/tools/fusion_ladder_results.py validate rows.jsonl
python3 perf-data/tools/fusion_ladder_results.py aggregate rows.jsonl --output promotion.json
python3 perf-data/tools/fusion_ladder_run.py manifest.json --output <scratch>/run-dir   # plow.fusion-ladder.run.v1
```

The runner needs ≥3 alternating paired rounds, one entry per scope, and a final
`plow.fusion-ladder.evidence.v1` JSON line per command; any missing field or non-decisive pair fails
closed.

### AMD candidate families

1. **Long-sequence recurrence** — chunked parallel scan only when the transition is associative or
   has an exact chunk composition; sweep chunk/subchunk; boundaries from the host, no blocking D2H;
   the serial body stays oracle and fallback.
2. **Ownership-preserving recurrent fusion** — one workgroup owns a full head/state shard and fuses
   conv/state update, norm and output gate; a partial fusion with a grid rendezvous is a separate
   candidate.
3. **Skinny projection split-K** — sweep split count per exact `(arch, CU, object digest, op, M, N,
   K, dtype, layout)` with the reduction in the timed region; missing rows fall back loudly.
4. **Grouped sparse work** — tune both expert stages by token bucket and observed distribution
   (persistent vs not, atomic vs reduction); gate empty, skewed and random bins; deterministic replay.
5. **Attention split/persistence** — by decode rung and KV bucket; time split + merge together;
   deterministic table lookup at runtime.

Distinguish `compiled`, `qualified` and `selected` in `build.json` or the runtime result: a compiled
arm or populated store is not evidence that production selected it.

## Step 4 — read the roofline %

- **Bandwidth-bound** (decode GEMV, decode attention): bytes moved / time vs `$BW_BOUND`.
  `gemv_row_sweep` reports per-(arm, MM) samples; the M curve prices `PLOW_GEMV_MM` / `WALK`.
- **Compute-bound** (prefill GEMM): TF/s vs `$COMPUTE_CEIL`. For small M read `gemm_tile_sweep`'s
  TILE COUNT / CU FILL: a tile that wins while dropping fill has not won.
- Records carry `median_ns`/`p10`/`p90`/`min`/`samples`; `Stats::beats` refuses a win inside the noise.
- Far from the bound: attribute before re-sweeping. Occupancy probes
  (`runtime/bench/nvidia/occ_probe.cu`, `px16_occ_bench.cu`, `runtime/ubench/gemm_occ1_bench.c`)
  separate residency from wave quantization; an occupancy delta needs a matched-grid control.

## Step 5 — record winners and read them back

- **Prefill GEMM (amd):** `plowc tune gemm` ingests as it measures; manual sweeps use
  `plowc tune ingest --samples <jsonl>`. Keys: cell (`tunedb::amd_tuning_cell`), op case
  (`gemm_op_case(m,n,k,quant)`), tile; `gemm_rung_opcode` returns `None` for calibration-only tiles.
- **Decode GEMV (amd):** `tunedb-gemv ingest --db tuning --gpu "$GPU" --samples <jsonl>`; `--gpu`
  decides the cell.
- **Decode knobs (nvidia):** the winner is an object; build it with the define string:

  ```bash
  PLOW_EXTRA_DEFINES="$(tunedb-decode best --db tuning --hardware $VENDOR/$ISA/<sku> --print defines)" \
    $BUILD <out>/interp_$ISA.cubin
  ```

  Take `<sku>` from `tuning/<vendor>/<isa>/<sku>/`. `--print` refuses when more than one cell matches.
- Cells are per SKU: two parts at one `$ISA` keep separate stores. Re-sweep on `$GPU`.
- Read back after every publish (a miss is byte-identical to "never measured"):

  ```bash
  plowc tune status --gpu "$GPU" --db tuning
  plowc tune select --gpu "$GPU" --shape M,N,K   # expect rationale measured, tier sku-calibrated
  ```

- Whether a production recipe uses a tuned record is a per-rung recipe decision
  ([agent-tools.md](agent-tools.md) "Per-rung gate and recipe selection"; the `PLOW_TUNEDB` hazard
  is in "Measurement hazards").

## Gate (into Stage 5)

1. Every demanded hot shape (`tune shapes`, `TUNEDUMP_GEMV`) is HIT, or its MISS is explained (no
   oracle, no knob, alias group).
2. Winners are `qualified` records (oracle passed, ≥ `Stats::MIN_SAMPLES`, decisive under
   `Stats::beats`) or exact `-D` define strings; nothing selectable came from an oracle-less harness.
3. Published records read back through `plowc tune select` with rationale `measured`.
4. Per-family roofline % against `$BW_BOUND` / `$COMPUTE_CEIL` for `$GPU`, binding side named,
   denominator kind stated.
5. Every timed run was uncontended (rc=0); rc=76 runs were discarded and re-run.
6. Untunable families are written down as such.
7. No command named a literal part; every `--gpu`/`--arch`/`--n-cu` came from the target block.
8. Every changed AMD object passed the resource/ISA audit: no spill regression, expected
   VGPR/AGPR/LDS and occupancy, the expected MFMA mnemonic for `$ISA`, and a live marker/route proof
   that the measured arm executed.

## Pitfalls

- Body-level aliases (AMD MoE group ops looping over expert ops) do not appear in the aliases block.
- Standalone kernel numbers do not transfer into the interpreter (different register allocation);
  measure through it (`runtime/bench/interp/interp_gemm_bench.c` builds a one-instruction program).
- Publishing into a cell nobody reads is silent; the `tune select` read-back is the only check.
- Calibration-only tiles are measurements, not selectable facts.
- Stub kernels (`XFLASH_MERGE`, `XARGMAX_FIN`) time at ~0 ns; rank only what
  `ProbedObject::executes` lists.
- On NVIDIA, timing benches and oracle tests live in separate files; NVIDIA GEMM winners stay
  screening results until merged.
- If every relevant opcode aliases and the tile is object-wide, the axis is a build define grid,
  not a per-op record.
- An exact-shape NVIDIA variant that changes block size, registers, shared memory, cluster or entry
  point is a new object ABI: expose its globals through a packet role; `plowrt` reads them and asks
  the driver for blocks/SM at load. On AMD `hwspec::ArchGeometry` gates the LDS tile and `pick_tile`
  selects only opcodes in the target inventory.

## Stop and ask

- A target-block row cannot be filled (`$GPU` not in `plowc --list-gpus`, no `$BUILD` for `$ISA`,
  ceilings not measurable).
- The family has no in-tree oracle: build one or proceed provisional?
- `tune inventory` cannot derive an inventory, or its defines do not match the shipping object.
- Every knob on the critical path is `Fixed`: the axis is a code change.
- No uncontended card after repeated attempts.
- A published winner does not appear in `tune select` and the cell/digest audit does not explain it.
- Roofline % far below ceiling and flat across all knobs: report the per-kernel table and move to
  [05](05-single-block-sweep.md)/[06](06-runtime-opt.md).

## Report

- The filled target block, `$BW_BOUND` / `$COMPUTE_CEIL` labelled measured or datasheet.
- Per-family verdict; census size, HIT/MISS, explained misses.
- Roofline table per hot kernel: achieved, ceiling, binding side, denominator kind.
- Winners: records (cell, op case, tile) or define string, plus the `tune select` read-back.
- Gate decision, contention retries, provisional-only families, standalone-only numbers.

Per-rung optimization after Stage 6: [08-rung-campaign.md](08-rung-campaign.md).

## Code pointers

| symbol / path | role |
|---|---|
| `plowc tune inventory\|select\|status\|shapes\|gemm\|ingest\|best\|regress` (`crates/plowc/src/tune/`) | probe, explain, campaign, ingest |
| `kernelcaps::select_kernel`, `probe::dispatch_arms`, `ProbedObject::executes`, `classify_macro` (`crates/kernelcaps/src/`) | selection, inventory, stub/alias detection, knob check |
| `tunedb::amd_tuning_cell`, `gemm_op_case`, `gemm_rung_opcode` (`crates/tunedb/src/gemm.rs`) | record keying |
| `tunedb::Stats`, `measurement_is_trustworthy` (`crates/tunedb/src/`) | noise gate, rc=76 |
| `tunedb-gemv`, `tunedb-decode` (`crates/tunedb/src/bin/`) | decode-side ingest / best |
| `costmodel::CostModel::new`, `kernel_reservation_bytes` (`crates/costmodel/src/`) | analytic ceiling |
| `hwspec::MemorySpec::bandwidth_for_bound` (`crates/hwspec/src/spec.rs`), `IsaLevel::geometry` (`isa.rs`) | measured floor; per-ISA geometry |
| `devgen::pick_tile` | emit-side consumer of GEMM records |
| `runtime/bench/gemm/` (`gemm_tile_sweep.c`, `gemv_row_sweep.c`), `runtime/bench/interp/` | sweeps and through-interpreter timing with oracles |
| `scripts/rebench_tune_gemm_all.sh`, `gemv_campaign_lease.sh`, `rebench_tune_gemv.sh`, `tune_decode_sweep.sh`, `gemma4_h100_kernel_tuner.py` | campaign drivers |
| `scripts/asm_audit.py`, `scripts/asm_expect_<isa>.json` | AMD disassembly assertions |
| `tuning/README.md` | record schema and store layout |
