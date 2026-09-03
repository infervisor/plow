# Stage 4 — Kernel Tuning to the Roofline

> Bring each hot kernel family of the new model — prefill GEMM, decode GEMV,
> attention, MoE — to a stated fraction of its roofline ceiling on the target
> GPU, and leave the winners recorded where the compiler actually reads them.
> Stage 3 proved the schedule correct; this stage makes the kernels inside it
> fast, *before* Stage 5 measures a whole block. The ceiling is defined by the
> cost model, not by feel: a kernel is "done" when its measured throughput sits
> against the min(compute, bandwidth) bound for its shape, and "tuned" when the
> winning tile/knob is a qualified record or a recorded build define.

**Precondition:** Stage 3 complete — every bucket verifies, so the schedule
being timed is the schedule that will ship. This is also the stage where the
target stops being abstract: fill in the parameter block in
[`target.md`](target.md) — `$VENDOR $ISA $GPU $NCU $NGPU $PARALLEL $MAXCTX
$TOOLCHAIN $BUILD $FEATURES $BW_BOUND $COMPUTE_CEIL $RESULTS` — before running
anything below, and write every command with `--gpu $GPU --arch $ISA` rather
than a literal part. You also need `$TOOLCHAIN` for probing, `$GPU` itself plus
`perf-data/tools/gpulease` for measuring, and `nix develop` for all cargo
builds.

**Gate out (into Stage 5, single-block sweep):** every demanded hot-kernel
shape has a measurement census entry (HIT or an explicit, explained MISS), the
winners are `qualified` records or recorded `-D` defines, `plowc tune --shape`
reports rationale `measured` on the shapes that have records, and the per-family
roofline % is written down with a clean (`gpulease` rc=0) provenance.

Authoritative references: [`docs/arch/11-tuning-coverage.md`](../arch/11-tuning-coverage.md)
(what is tunable per family and what blocks each) and
[`docs/arch/12-using-the-tuner.md`](../arch/12-using-the-tuner.md) (how to take
a measurement the database will accept). This stage is those two documents
turned into a bringup sequence.

---

## What "tuned" means here

Two load-bearing properties from the tuner design constrain everything below:

* **`compile` reads, `tune` writes.** `plowc compile` may consume qualified
  records; only an explicit tuning run publishes one. A build must never
  calibrate itself against its own output.
* **Capability is probed from a built object, never declared.** The same
  interpreter TU becomes different objects under different `-D` sets, so "which
  kernels exist" and "which macros are knobs" are questions about an object.
  `plowc tune inventory` answers them with the vendor preprocessor — no GPU
  needed.

And one from the store (`crates/tunedb/`): **correct before fast.** A
measurement without a passing correctness oracle can be stored (`provisional`)
but never selected. This single rule decides most of what is tunable today —
see the coverage table below.

## Step 0 — establish the ceiling before measuring anything

The roofline % is meaningless until the denominator is right.

* **The cost model defines the analytic ceiling.** `costmodel::CostModel::new`
  takes the GPU's `SmSpec`, subtracts `kernel_reservation_bytes(arch)` (2–4 KiB
  of barriers/TMA descriptors/LDS scratch that tiles cannot use) from
  `shared_mem`, and enumerates tile candidates under double-buffering. DMA cost
  comes from `sm_bytes_per_cycle` — the **datasheet** bandwidth divided across
  SMs, used deliberately for *ranking* (a constant derate cancels in a relative
  comparison). Anything that reports an **absolute** floor must use the measured
  `MemorySpec::bandwidth_for_bound()` instead (this is what `plowc
  --lean-oracle` does).
* **Use measured machine ceilings for the % you report.** `$COMPUTE_CEIL` is a
  *sustained* matrix-core figure measured on `$GPU`, and `$BW_BOUND` is
  `MemorySpec::bandwidth_for_bound()` for `$GPU` — which returns the **datasheet
  peak** whenever that registry entry has `bandwidth_measured: None`, i.e. on
  nearly every part. Establish both here, and say which kind of denominator you
  used in every % you later quote. `gemm_tile_sweep.c` makes its `PEAK_TFLOPS`
  / HBM constants overridable macros for exactly this reason. The gap is not
  cosmetic: one part's measured MFMA ceiling sat 28% below its own datasheet
  peak, and two AMD ISAs in this tree differ from each other by ~1.77× on
  compute — carrying either number across parts invents a result.
* **Measuring `$BW_BOUND` is a gate, not advice.** Worked precedent: MI325X
  shipped with `bandwidth_measured: None`, so `bandwidth_for_bound()` returned
  its 6000 GB/s datasheet peak. The measured sustained figure is **4164 GB/s —
  69% of the datasheet** — so every roofline % taken against the peak
  understated efficiency by ~44%. Measure it with
  `runtime/tests/decode_bw_probe.hip`, median of three clean leased runs, then
  **land the number in the registry entry for `$GPU` together with a test that
  asserts `bandwidth_for_bound()` returns it** (see
  `measured_bandwidth_governs_a_reported_bound_on_mi325` in
  `crates/hwspec/src/registry.rs`). That turns a one-off measurement into a
  standing invariant instead of a note someone has to remember. If you cannot
  measure it on `$GPU`, every reported % must say in words that its denominator
  is an unmeasured datasheet peak.
* **Single-unit rooflines come from `runtime/ubench/`.**
  `ubench_cu_roofline` runs the *production* op headers (same code, same
  register pressure) on one CU and prints efficiency vs the single-unit
  ceiling, with `--decode` switching to M=1 shapes:

  ```bash
  cd runtime
  cmake -B build -DPLOW_ROCM=ON -DPLOW_UBENCH=ON -DPLOW_HIP_ARCH=$ISA   # $VENDOR = amd
  cmake --build build --target ubench
  cd ubench && ./run_ubench_cu.sh --timing-only     # or --kernel d_gemm, --json out.json
  ```

  The single-CU harness is AMD-side; on `$VENDOR = nvidia` the equivalent
  evidence comes from the occupancy probes in `runtime/bench/nvidia/`.

  These numbers are **evidence, not selectable records** — `runtime/ubench/`
  has no correctness oracle, so nothing it produces can qualify in `tunedb`.
  Use it to decide whether a kernel is compute- or bandwidth-bound and how far
  from the ceiling it sits; use the oracle-bearing harnesses below to publish.

## Step 1 — inventory: what is actually tunable on this object

Do this before spending any GPU time. Three questions decide whether a family
is tunable at all (see `11-tuning-coverage.md` for the full derivation):

1. does the opcode reach a *distinct* kernel (or is it an alias)?
2. is there a parameter to vary (an `#ifndef`-guarded macro, not a hard
   `#define`)?
3. is there a correctness oracle?

```bash
nix develop --command cargo build -p plowc --bin plowc     # `tuner` is a default feature
plowc --list-gpus                                          # the recognized $GPU names — do not guess
plowc tune inventory --gpu "$GPU" --root .                 # executable kernels + aliases
plowc tune select --gpu "$GPU" --shape 4096,4096,4096      # what would this shape get, and why
plowc tune status --gpu "$GPU" --db tuning                 # what the store already holds
```

Read the **aliases** block first: opcodes listed together reach one function
body, and ranking within such a group measures dispatch noise. On NVIDIA the
three dense-GEMM opcodes are one body whose tile is an object-wide macro — the
tuning axis there is *which object is built*, not which opcode is emitted. On
AMD the same opcodes are separately compiled instantiations and genuinely
rankable. Before sweeping any macro, confirm it is a knob:
`kernelcaps::classify_macro(header, "PGM_BN") == Sweepable::Overridable`. A
`-D` on a hard-defined macro is a redefinition, and the numbers describe a tile
that was never built.

### Step 1b — prove the built object actually carries the arms

`tune inventory` answers "what does this source compile to under these defines".
The complementary question is **"does the object I am about to measure contain
the arms this model's geometry needs"**, and it must be asked of the built
artifact. A missing arm is rarely a crash: it is a silent fallback to a slower
correct path, or a feature that disables itself with a warning nobody read. One
recorded instance — a sampler object missing its multi-step advance entry point
— silently turned multi-step decode off and cost ~1.7× with no error.

The check is per `$VENDOR` because the object format is:

| `$VENDOR` | entry symbols present | register / spill / instruction mix |
|---|---|---|
| nvidia | `cuobjdump -symbols <obj>.cubin \| grep <mangled entry>` (`$BUILD` already fails hard when its own entry is missing) | `cuobjdump -res-usage <obj>.cubin` |
| amd | `readelf -s <obj>.hsaco` / `.elf` | `scripts/asm_audit.py [--expect scripts/asm_expect_$ISA.json] <obj>` — asserts on the *disassembly*, not just the symbol: which MFMA the backend picked, operand formats, spill to scratch |

`asm_audit.py` has no NVIDIA counterpart, and `cuobjdump -res-usage` has no AMD
counterpart; each side has the instrument the other lacks. Run whichever `$ISA`
gives you, and record which class of check went unmade.

**Reading register counts.** Two non-obvious properties, both of which have
inverted a conclusion in a real campaign:

* **An interpreter object's register count is the MAX over every instantiated
  arm**, because the megakernel inlines all of them. Adding a wide rung for
  batching therefore taxes the `B=1` path too — the same coupling
  `11-tuning-coverage.md` states for `GV_MM_MAX`, visible here as a number.
* **Spill is not the same as slow.** A latency-bound rung can tolerate spill and
  measure *faster* than a spill-free variant (recorded on a decode unroll
  sweep). Do not prune a variant on `ptxas -v` output; measure it.

**The TU-isolation law.** A heavyweight kernel body loses probe-grade register
allocation when it is compiled into a wide-armed interpreter TU. Recorded three
times on `$ISA = sm_90a`: a warp-specialized GEMM that deadlocked/spilled in the
fat TU ran clean in an arm-stripped one; an hd512 flash body was 1.75–2.5×
*worse* in the fat TU and won 10.5% in a flash-only TU; a 384-thread
producer/consumer GEMM was only expressible as its own object at all. The
general form: **when a big kernel underperforms in-model but not standalone,
suspect the translation unit before the kernel.** The lever is which object gets
built, which is why the NVIDIA tuning axis in this stage is a build-define set
rather than a per-op record.

Condensed coverage picture (full table in `11-tuning-coverage.md`):

| family | knob | oracle | verdict |
|---|---|---|---|
| prefill GEMM | AMD `GM_BM`/`GM_BN`/`GM_BK`; NV Ampere `PGM_BN`/stages; Hopper `PGM90_STAGES`/`PGM90_FP8_PROMOTE` | AMD yes | **tunable now (AMD)**; NV blocked on oracle merge |
| decode GEMV | `GV_*` unroll ladder, `GV_MM_MAX`, `PLOW_GEMV_MM`/`WALK` | AMD yes | tunable, high value |
| MoE | shares dense `PGM_*` tile | weak | blocked (oracle + shared tile) |
| attention / MLA / DSA | AMD `FA_*`; NV tiles are template literals, not macros | DSA strongest | partial |
| collectives | `PLOW_XR_CUS` | 2-GPU | park until N≥2 GPUs |

## Step 2 — identify the hot shapes from the compiler's own demand

Do not hand-author a shape list; that is exactly how GLM-5.2 prefill ended up
100% unmeasured. The demand comes from the compile itself:

```bash
# derive the demanded GEMM shapes from a real emit (the compile flags go BEFORE `tune`):
plowc --hf-dir <ckpt> --arch $ISA --max-ctx $MAXCTX --n-cu $NCU --num-gpus $NGPU \
    tune shapes --gpu "$GPU"

# current-source decode campaign: the supplied emit produces the TUNEDUMP_GEMV census
OBJ="$(mktemp -d /tmp/plow-gemv-obj.XXXXXX)"
rmdir "$OBJ"
TUNE_GPU=MI355X scripts/gemv_campaign_lease.sh "$OBJ" OUT.jsonl CAMPAIGN -- \
    plowc --emit devblob ...
```

The current production demand extends through B=128. Campaign objects compile
at MM<=16 and exercise wider M through the walk. Every demanded BF16 family,
including QKVG as its harness arm lands concurrently, must produce the complete
MM=1/2/4/8/16 row, including MM>M coverage. MXFP4 produces only its compiled
OBJ_MM row and is accepted only when M<=OBJ_MM. The campaign records the
physically reported MI350X or MI355X name and CU count separately from the
shared gfx950 cell. Its publication
gate binds the cell, interpreter, toolchain label, and oracle; it does not claim
that this identity is a digest of the temporary sweep object.
The production entrypoint obtains the toolchain label from the repository's
`nix develop` environment, then binds that exact value through the harness
build, both interpreter probes, and ingest.

The hot families for a transformer are known in advance — qkv/o and MLP
projections (GEMM at prefill, GEMV at decode), flash attention
(decode/prefill), MoE router/experts/combine — but the *shapes* are per-model
and per-bucket, and only the emit knows them.

## Step 3 — run the sweeps, under the lease

Every timed run goes through `perf-data/tools/gpulease <label> <cmd>`, which
audits for foreign compute processes and **exits 76 if the GPU was contended**.
`tunedb::measurement_is_trustworthy` accepts only exit 0 — a 76 is a failed
measurement, never a result with a caveat. Wrap the run, not the build.

| sweep | `$VENDOR` | harness | ingest / consumer |
|---|---|---|---|
| prefill GEMM tiles | amd | `plowc … tune gemm` (measure + ingest + verify, one command) or `gemm_tile_sweep <M> <N> <K> [label] [quant]` with `PLOW_GEMM_JSONL=<path>` | `plowc tune ingest --samples <jsonl>`; read back by `devgen::pick_tile` |
| decode GEMV rungs | amd | `gemv_campaign_lease.sh OBJ JSONL CAMPAIGN -- EMIT_COMMAND...`; requires a fresh OBJ, builds all harness components in `nix develop` outside one leased sweep, then derives/filter-checks the live `TUNEDUMP_GEMV` census | Ingests only passing uncontended samples after physical MI350X/MI355X detection and exact cell/interpreter/toolchain/oracle checks |
| decode attention split count | both | emit matched packet arms with the model's `nsplit` knob (K3: `PLOW_K3_NS`) and score with `plowrt serve` + `vllm bench serve`; keep weights and interpreter object fixed | packet/program selection, not the kernel tune store |
| decode knob grid | nvidia | `scripts/tune_decode_sweep.sh` — joint OBJECT knobs (`PLOW_NV_FORCE_MINBLK`, `GV_UNROLL*`, `GV_MM_MAX`, `PLOW_MOE_DOWN_SG`) scored by end-to-end step TPOT | `tunedb-decode ingest --db tuning --results <jsonl>` |
| single-CU roofline (evidence only) | amd | `run_ubench_cu.sh` | none — no oracle, never selectable |
| dispatch floor (evidence only) | both | `runtime/bench/dispatch/interp_dispatch_floor_nv.cu` / `.hip` | stored `provisional` with `reason_not_qualified` |

The vendor split here is real, not an accident of coverage: on AMD the dense
opcodes are separately compiled instantiations with per-op records to publish,
while on NVIDIA the tile is object-wide, so the sweep axis is the build define
set. Whichever side `$VENDOR` puts you on, the other column's harness has
nothing to say about `$GPU`.

The end-to-end prefill-GEMM campaign (`$VENDOR = amd`), in one command (objects
built first by `scripts/rebench_build_objs.sh` → `test_kernels.elf`):

```bash
nix develop --command ./target/release/plowc \
    --hf-dir <ckpt> --arch $ISA --max-ctx $MAXCTX --n-cu $NCU --num-gpus $NGPU \
    tune gemm --gpu "$GPU" --root . --obj <objdir> --samples <out.jsonl> --lease
```

`--shapes auto` (the default) derives the list from the compile's demand;
`--lease` wraps every GPU invocation in `gpulease`. Repeat until clean — one
clean run is still one sample of the run distribution.

Two knob couplings to respect in any sweep design (from `11-tuning-coverage.md`):
`GV_MM_MAX` moves the register ceiling of **every** arm in the object because
the interpreter inlines all of them, and MoE grouped GEMM shares the dense
`PGM_*` tile triple, so sweeping one moves the other. `PGM_BM` is additionally
packet-layout-visible (the MoE routing histogram pads to it).

### Decode-attention packet split sweep

Treat attention `nsplit` as a packet-decomposition sweep, not a kernel-tile
sweep. It divides one KV scan into independent ranges and increases the flash
grid, while the following merge grows linearly with the number of partials.
The result is context-dependent and U-shaped.

For every arm, re-emit only the packet, reuse the exact weights and interpreter
object, and prove by disassembly that only the attention and merge packets plus
their scratch extents changed. Sweep at the shortest and longest served context,
then add crossover points before choosing thresholds. K3 TP8 uses
`workgroups = (local_heads / head_group) * PLOW_K3_NS`; its measured 128K sweep
was ns16/ns32/ns64/ns128 = 81.400/67.417/60.569/60.683 ms TPOT, so ns64 won and
ns128 bracketed the merge-cost reversal.

Screen the full grid with a truncated real-model asset containing one production
attention layer and its merge/output/residual tail. This retains the interpreter,
packet counters, FP8 KV layout, and real weights while avoiding a full-model load
per arm. Use the isolated result to discard losers and locate approximate
crossovers; only the finalists proceed to whole-model serving. An isolated flash
kernel alone is insufficient because it omits the growing merge term.

For K3, zero-based layer 3 is an MLA layer. Emit the screen with the production
packet and object settings, changing only the layer set and split count:

```bash
for ns in 16 32 64 128; do
  nix develop --command env \
    K3_FULL=1 PLOW_K3_LAYERS=single:3 PLOW_K3_NS="$ns" \
    PLOW_FP8_KV=1 PLOW_MXFP4=1 PLOW_MLA_PF_V2=1 \
    PLOW_L2_PLACE=1 PLOW_DECODE_BATCH=1 PLOW_GEMV_MM=1 \
    ./target/release/plowc --hf-dir /home/lava/models/k3_farm \
      --emit devblob --arch gfx942 --gpu MI325X --num-gpus 8 --parallel tp \
      --max-ctx 131072 --n-cu 304 --out "build-amd/k3-ns${ns}-layer3"
done
```

Reuse one matching HSACO tree for every arm. Check that only the 24 decode
attention packets, their 24 merge packets, and scratch extents differ. The
single-layer screen ranks candidates; it is not a served-model or quality gate.

Changing split boundaries changes floating-point reduction association. Require
finite outputs, exact cross-rank/counter audits, the model quality gate, and
served `vllm bench serve` measurements; byte-identical text across arms is not a
valid requirement. Add a production context ladder only when matched crossover
wins exceed noise. It must select a pre-emitted program by live KV length, never
patch packet instructions or resize scratch in the hot path. K3's measured ns64
arm tied ns16 at short context and won at every measured context from 4K through
128K, so the current production choice is one fixed ns64 program, not a ladder.

## Step 4 — read the roofline %

For each family, compute the % against the bound that binds:

* **Bandwidth-bound (decode GEMV, decode attention):** bytes actually moved
  (weights + KV + activations) ÷ measured time, as a fraction of `$BW_BOUND`.
  `gemv_row_sweep` reports per-(arm, MM) samples;
  decode GEMV should sit near the weight-streaming bound, and the M curve is
  what prices the object-level choice of `PLOW_GEMV_MM` and `PLOW_GEMV_WALK`.
* **Compute-bound (prefill GEMM):** achieved TF/s as a fraction of
  `$COMPUTE_CEIL`. `gemm_tile_sweep` prints TF/s, and — for the
  small-M shapes — the **TILE COUNT and CU FILL** each tile achieves, because
  at M=128 the limit is usually fill, not the tile's own efficiency. A tile
  that "wins" by 3% while dropping CU fill has not won.
* **Timings are never a single number.** Records carry
  `median_ns`/`p10`/`p90`/`min`/`samples`, and `Stats::beats` refuses a win
  inside the noise. A 1 ns edge under 30 ns of spread is not a result.

When a family is far from its bound, attribute before re-sweeping: the
occupancy probes (`runtime/bench/nvidia/occ_probe.cu`, `px16_occ_bench.cu`,
`runtime/ubench/gemm_occ1_bench.c`) separate residency effects from wave
quantization — see the pitfalls.

## Step 5 — record the winners where the compiler will find them

* **Prefill GEMM (`$VENDOR = amd`):** `plowc tune gemm` ingests as it measures; a manual
  sweep goes through `plowc tune ingest --samples <jsonl>`. Records are keyed
  by cell (`tunedb::amd_tuning_cell(spec)` — the ONE rule shared by writer and
  reader), op case (`gemm_op_case(m,n,k,quant)` — quant is in the key so a bf16
  timing is never served for an fp8 op), and tile; `gemm_rung_opcode` maps a
  winning tile back to the opcode that carries it, and returns `None` for
  calibration-only tiles that must not become selectable facts.
* **Decode GEMV (`$VENDOR = amd`):** `tunedb-gemv ingest --db tuning --gpu
  "$GPU" --samples <jsonl>`. `--gpu` is required — it decides the cell.
* **Decode knobs (`$VENDOR = nvidia`):** the winner is an object, not a record
  the emitter reads; consume it as build defines through `$BUILD`:

  ```bash
  PLOW_EXTRA_DEFINES="$(tunedb-decode best --db tuning \
      --hardware $VENDOR/$ISA/<sku-slug> --print defines)" \
    $BUILD out/interp_$ISA.cubin
  ```

  The `--hardware` path is `<vendor>/<isa>/<sku>`; take the slug from the store
  layout (`tuning/<vendor>/<isa>/<sku>/`), not from `$GPU`'s display name.

  **The cell is per-SKU, so winners do not transfer between two parts at the
  same `$ISA`.** MI300X and MI325X are the same compute die — same CU count,
  same LDS, same `IsaLevel` — and still keep separate stores
  (`tuning/amd/gfx942/mi300x/` and `tuning/amd/gfx942/mi325x/`). They differ in
  the memory subsystem, which is exactly what a bandwidth-bound decode winner is
  selected against. Re-sweep on `$GPU`; do not copy a sibling SKU's store
  forward.

  `--print` refuses when the filter leaves more than one cell standing: a flag
  string names ONE object, and the union of two cells' winners is an object
  nobody measured.

Then **verify consumption** — a publish that lands in a cell nobody reads is
silent in both directions:

```bash
plowc tune status --gpu "$GPU" --db tuning      # the cell now holds qualified records
plowc tune select --gpu "$GPU" --shape M,N,K    # rationale: measured; tier: sku-calibrated
```

If the selection still says `analytical cold start` / tier `portable` for a
shape you just measured, the campaign published into the wrong cell or under a
stale digest — a miss is byte-identical to "never measured", so this check is
the only thing that distinguishes a working campaign from a no-op one.

## Success criteria

1. **Coverage census is explicit.** Every hot-kernel shape from the compiler's
   demand (`tune shapes`, `TUNEDUMP_GEMV`) is HIT in the store, or its MISS is
   explained (no oracle for the family, no knob, alias group).
2. **Winners are qualified.** Winning records are in state `qualified` — oracle
   passed, ≥ `Stats::MIN_SAMPLES`, decisive under `Stats::beats` — and object
   winners are recorded as the exact `-D` define string.
3. **The compiler sees them.** `plowc tune select` reports rationale `measured`
   and tier `sku-calibrated` (or better) on covered shapes.
4. **Roofline % recorded per family** against `$BW_BOUND` / `$COMPUTE_CEIL` as
   established for `$GPU` in Step 0, with the binding side (compute vs
   bandwidth) named and the denominator's provenance stated (measured on this
   part, or an unmeasured datasheet peak). A % whose denominator came from
   another part fails this criterion.
5. **Every timed run was uncontended** (`gpulease` rc=0); every rc=76 run was
   discarded and re-measured.
6. **Untunable is written down, not skipped over.** Families with no oracle or
   no knob on this target are recorded as such (they will resurface in
   Stage 5/6 as build-knob work, not tuner work).
7. **No command in the campaign names a literal part.** Every `--gpu`/`--arch`/
   `--n-cu` came from the `target.md` block; a script left pointing at another
   part's flags is the recurring bring-up failure this block exists to prevent.

## Pitfalls (from real campaigns)

* **Ranking an alias group measures dispatch noise.** On `$VENDOR = nvidia` the
  three dense-GEMM opcodes reach one body; a "winner" among them will not
  reproduce. Read the
  `aliases` block before designing any sweep. Body-level aliases (AMD MoE group
  ops are `k`-loops over the expert ops at default flags) do not appear there —
  known blind spot.
* **A hard `#define` is not a knob.** `-D` on it is a redefinition and the
  numbers describe a tile that was never built. `classify_macro` first, always.
  On `$VENDOR = nvidia` the attention tiles are not macros at all — `BQ`/`BKV`
  are template literals at the dispatch sites, so sweeping them is a code edit,
  not a parameter search.
* **Standalone kernel numbers do not transfer.** A standalone GEMM and the same
  GEMM inlined into the interpreter get different register allocations — on one
  AMD part the same tile family measured 212 vs 770 TF/s across that boundary.
  Measure *through* the interpreter (`interp_gemm_bench.c` builds a
  one-instruction `PlowProgram`); [Stage 7](07-perf-campaign.md) states the same
  probe law from the serving side.
* **Contended runs are quietly ~35% wrong.** The first campaign needed three
  attempts before the `gpulease` audit came back clean, and the contended runs
  differed from clean ones by ~35% on the gate term. rc=76 is a failed
  measurement, full stop.
* **Publishing into the wrong cell is silent in both directions.** `tunedb-gemv`
  once hardcoded one AMD ISA's cell while the reader keyed off `--gpu`, so a
  campaign on the other ISA would publish records nobody could read — the mechanical
  reason the decode-GEMV census had zero records. The reader finds nothing,
  falls back to the analytical model, and reports `portable`, which is exactly
  what it reports when nothing was ever measured. Always re-check
  `tune select` after ingest.
* **Calibration-only tiles must not become selectable.** The sweep also
  compiles tiles (320x128, 384x128, …) that no interpreter arm carries; they
  are legitimate measurements of a body and NOT selectable facts —
  `gemm_rung_opcode` returns `None` for them by design.
* **Stub kernels benchmark at ~0 ns and win.** AMD's `XFLASH_MERGE` /
  `XARGMAX_FIN` have live `case` labels with empty bodies.
  `ProbedObject::executes` excludes them; a hand-rolled harness will not.
* **An occupancy "win" can be wave quantization.** A per-cell occ-2 signal of
  1.1–1.4× on decode flash collapsed to 1.07× (and negative at the deployed
  config) once a matched-grid control was run
  (`perf-data/px16-decode-occupancy.md`). Never accept an occupancy delta
  without the matched-grid control.
* **Datasheet denominators flatter the kernel.** On one AMD part, a kernel
  sitting exactly at the measured MFMA issue ceiling looks 28% "slow" against
  that part's datasheet peak. Report % of `$COMPUTE_CEIL` and say which kind of
  ceiling it is — and remember `bandwidth_for_bound()` silently *is* the
  datasheet peak wherever `bandwidth_measured` is `None`.
* **No oracle → no record, however fast it runs.** `runtime/ubench/` numbers
  and the dispatch-floor probes are evidence and stay `provisional`. On
  `$VENDOR = nvidia` the timing benches and the oracle tests exist in separate
  files (the interp op tests validate but have no events; the `*_bw_*.cu`
  timers have no oracle); merging them is the shortest path to a qualifying
  measurement there — until then, NVIDIA GEMM winners are screening results.
* **The tuner can be a structural no-op on your path.** If every relevant
  opcode aliases to one body and the tile is object-wide, the real axis is a
  build knob (`-D` grid → `tune_decode_sweep.sh`), not a per-op record. Confirm
  the tuner selects something before reporting the family "tuned" (Stage 5's
  gate re-checks this).

## Code pointers

| symbol / path | role |
|---|---|
| `plowc tune` — `inventory\|select\|status\|shapes\|gemm\|ingest\|best\|regress` (`crates/plowc/src/main.rs`, `crates/plowc/src/tune/`) | probe, explain, campaign, ingest |
| `kernelcaps::select_kernel`, `OpSignature`, `KernelSpec`, `MeasuredCosts` (`crates/kernelcaps/src/select.rs`, `spec.rs`) | selection: measured → analytical fallback, alias collapse |
| `kernelcaps::probe::dispatch_arms`, `ProbedObject::executes` / `::stubs` (`crates/kernelcaps/src/probe.rs`) | derived kernel inventory, alias + stub detection |
| `kernelcaps::classify_macro`, `Sweepable` (`crates/kernelcaps/src/sweep.rs`) | is this macro a knob |
| `tunedb::amd_tuning_cell`, `gemm_op_case`, `gemm_rung_opcode`, `GEMM_ORACLE` (`crates/tunedb/src/gemm.rs`) | record keying — cell, op case, tile→opcode |
| `tunedb::gemv_case` / `gemv_sample_*` (`crates/tunedb/src/gemv.rs`); `rank_by_cell`, `DecodeKnobs` (`decode.rs`) | GEMV and decode-knob record shapes |
| `tunedb::Stats` (`beats`, `MIN_SAMPLES`), `TuneStore`, `RecordState`, `Digests` (`crates/tunedb/src/`) | noise gate, staleness, state machine |
| `tunedb::measurement_is_trustworthy`, `GPULEASE_CONTENDED` (`crates/tunedb/src/lib.rs`) | rc=76 is a failed measurement |
| `tunedb-gemv` / `tunedb-decode` (`crates/tunedb/src/bin/`) | ingest + `best` for the decode-side stores |
| `costmodel::CostModel::new`, `kernel_reservation_bytes` (`crates/costmodel/src/lib.rs`); `dma_cycles`, `sm_bytes_per_cycle` (`cost.rs`) | the analytic ceiling and SRAM budget |
| `hwspec::SmSpec`, `MemorySpec::bandwidth_for_bound` (`crates/hwspec/src/spec.rs`) | per-SM resources; measured-bandwidth floor |
| `devgen::pick_tile` (`crates/devgen/src/lib.rs`) | the emit-side consumer of qualified GEMM records |
| `runtime/bench/gemm/gemm_tile_sweep.c`, `gemm_bench_8k.c`, `gemv_row_sweep.c` | tile / row-bucket sweeps with oracle + JSONL |
| `runtime/bench/interp/interp_gemm_bench.c`, `qwen_interp_bench.c`, `dsa_gather_bench.c` | time-and-validate through the interpreter |
| `runtime/bench/dispatch/interp_dispatch_floor.hip` / `_nv.cu` | dispatch-floor calibration (provisional evidence) |
| `runtime/ubench/bench_cu_roofline.c`, `run_ubench_cu.sh`, `bench_cu_gfx950.hip` | single-CU roofline vs theoretical ceilings |
| `scripts/rebench_build_objs.sh`, `rebench_tune_gemm_all.sh`, `gemv_campaign_lease.sh`, `rebench_tune_gemv.sh`, `tune_decode_sweep.sh`, `$BUILD` | campaign orchestration |
| `scripts/asm_audit.py`, `scripts/asm_expect_<isa>.json` | assert on an AMD object's disassembly (MFMA choice, operand format, spill); NVIDIA's counterpart is `cuobjdump -symbols` / `-res-usage` |
| `perf-data/tools/gpulease` | the lease (rc=76 = contended = failed measurement) |
| `hwspec::IsaLevel::geometry` (`crates/hwspec/src/isa.rs`), `docs/arch/14-amd-arch-divergence.md` §3 | which layer owns a per-ISA difference |
| `tuning/README.md`, `tuning/<vendor>/<isa>/<sku>/` | the record schema and the store itself |

Related context: `docs/arch/11-tuning-coverage.md` (per-family verdicts this
stage executes against), `docs/arch/12-using-the-tuner.md` (reading inventories,
tiers, and the record schema), `perf-data/px16-decode-occupancy.md` /
`perf-data/px19-tile-graph.md` (measured occupancy / tile-granularity results
worth reading before re-deriving them).
# Shape-keyed GEMV workgroup tuning

For a measured GEMV shape, pass an emit-time table rather than changing the
kernel or runtime globally:

```text
PLOW_GEMV_WG_TUNING='896x7168=224,1536x7168=152' plowc ...
```

Entries are `N x K = workgroups`; matching is exact and the first matching
entry wins. Invalid or unrelated entries are ignored. The setting is currently
consumed by K3's blocked GEMV emitter only; unset output is byte-identical to
the default. It changes packet workgroup counts, so the corresponding object
must be rebuilt and audited, while the HSACO does not need a new kernel arm.

Keep tables tied to measured `(arch, model, compiler, object)` provenance. A
future tuner can emit this table from the existing block-sweep results; do not
make an unmeasured heuristic the default because workgroup caps trade occupancy
against bandwidth and can regress wide shapes.
