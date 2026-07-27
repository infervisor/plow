# tuning/ — calibrated kernel measurements

Kernel-level measurement records, keyed by hardware fingerprint. Written only by
an explicit tuning run; `plowc compile` may read qualified records and must
never publish, or the thing being measured and the thing doing the measuring
stop being separable.

This is **not** a replacement for `perf-data/`. That tree is the serving-level
record — its campaign files are keyed `(model, engine, precision, phase, tp,
ctx)` and answer "is plow faster than vLLM on this model". They have no column
for GPU, shape, tile, or kernel, and carry one hardware string per file. This
tree is the level beneath it: which kernel, on which silicon, under which
toolchain.

## Layout

```
tuning/<vendor>/<isa>/<sku>/kernel_measurement.jsonl
```

The path comes from `HardwareFingerprint::tuning_path()`, so two GPUs cannot
land in one file and be mistaken for one measurement population. The ISA segment
is the compiled `-arch`, not a marketing generation: `sm_90a` and `sm_120a` are
separate cells, and so are `sm_100a` and `sm_120a` despite both being
"Blackwell".

## Schema

One JSON object per line, appended, never rewritten. Fields are
`tunedb::KernelMeasurement`. The rules the store enforces:

- **No single-sample records.** A statistic without dispersion cannot
  distinguish a win from noise, so `Stats` requires a minimum sample count and
  keeps median, p10, p90, and min.
- **Correct before fast.** A record cannot reach `qualified` until its
  correctness oracle has passed. There is no path that promotes an unchecked
  kernel.
- **Atomic campaigns.** One unqualifiable record aborts the whole publication.
  An interrupted run leaves no selectable half-winner.
- **Specific staleness.** `digests` covers implementation, interpreter,
  toolchain, and oracle separately, so recompiling one kernel invalidates its own
  records and nothing else.
- **Negatives retained**, with their reason, so a campaign does not spend GPU
  time rediscovering a dead end.

## Measurement policy

Runs are serialized through `perf-data/harness/gpulease`, which audits for
foreign compute processes and **exits 76 if the GPU was contended**. A contended
run is discarded, not stored with a caveat — `tunedb::measurement_is_trustworthy`
accepts only exit 0. The first campaign in this tree needed three attempts
before the audit came back clean; the two contended runs were thrown away, and
the numbers they produced differed from the clean ones by ~35% on the gate term,
which is the whole argument for the check.

## Current cells

| cell | what is calibrated | status |
|---|---|---|
| `nvidia/sm_90a/h100-nvl` | interpreter dispatch floor | measured, see below |
| `nvidia/sm_120a/rtx-5090` | prefill GEMM tile (`prefill_tile_measurement.jsonl`) | measured, see below |

### `nvidia/sm_120a/rtx-5090`

`prefill_tile_measurement.jsonl` — the w8a8 prefill GEMM tile space of the shipped
prefill object, on Gemma-4-12B-it fp8, scored on the **end-to-end** conc-1 127k
prefill wall (`perf-data/px13-prefill-gemm.md`). Written by
`perf-data/px13_emit_tuning.py` from `perf-data/px13_sweep_{build,e2e}.sh`.

**Why a new record kind rather than `tunedb::KernelMeasurement`.** On NVIDIA the
GEMM tile is a compile-time macro of the *interpreter object*, and `plowc tune`
says so itself: all three dense-GEMM opcodes alias to one body, so "the real
tuning axis here is which object is built, not which opcode is emitted". A
`KernelMeasurement` is keyed by `op_case` + `kernel_id` and has no field that can
distinguish `BN=128` from `BN=64` — the two are the *same* kernel id in *different*
objects. Until that entity grows a build-identity column, a prefill-tile record
cannot be expressed in it, and this cell is therefore **not** loadable by
`TuneStore::load_kernels`. `plowc tune --status` will still report "no kernel
measurements for this cell", correctly.

Two things this cell exists to stop the next person rediscovering:

1. **The microbench and the runtime disagree in sign.** `PGM_GLU_BN=64` is +6.1%
   on the isolated `gate|up` GEMM at the M the runtime actually launches, bit-exact,
   at unchanged occupancy — and **2.3% slower** end to end. A tile tuner scored on
   the microbench ships a regression. Every row here carries its microbench number
   as *data*, never as the ranking key.
2. **Two arms in the space are undeployable, not slow.** `PGM_STAGES=5` and
   `PGM_BN=256` put the GEMM arena over the 101376 B dynamic-smem opt-in; plowrt
   then drops the packet's prefill buckets and consumes the prompt one decode step
   at a time. PX-9 measured both in an isolated bench and called them "noise".

### `nvidia/sm_90a/h100-nvl`

Measured with `runtime/bench/interp_dispatch_floor_nv.cu`, the NVIDIA
counterpart of `runtime/bench/interp_dispatch_floor.hip`. It reproduces the same
structure — an all-SM producer, then a single-block consumer gated on it — using
the `ld.acquire.gpu` / `red.release.gpu` lowering that `PLOW_NV_PTXSYNC=1`
(the default) actually emits, `PLOW_CTR_STRIDE`-strided counters, and a
cooperative launch at exactly the resident grid because co-residency is the
interpreter's safety condition.

Five clean runs, 359 post-warm-up samples each, 660 blocks (5/SM x 132 SMs):

| term | H100 NVL sm_90a | MI350X gfx950 (published in the HIP bench header) |
|---|---|---|
| GAP — gate wait | **1.088 us** (1.056–1.088) | ~3.8 us |
| WALL — body + successor signal | **0.928 us** (identical every run) | ~2.85 us |
| GAP + WALL | **2.016 us** (1.984–2.016) | ~6.65 us |
| period, op to op | 5.312 us | 6.5–6.9 us |

**Caveat on `period`:** this harness executes a `grid.sync()` per step that the
production persistent interpreter does not, so the period is an upper bound.
GAP and WALL are the directly comparable terms.

**What this says about `DECODE_DISPATCH_FLOOR_US`.**
`costmodel::cost::DECODE_DISPATCH_FLOOR_US = 4.6` was measured on MI350X and is
applied to every GPU in the registry, scaled only by `clock_boost`. On Hopper the
measured gate-plus-body cost is 2.0 us against the 4.6 us charged.

The gap is not just a calibration error. The HIP bench attributes roughly 3.0 us
of its 3.8 us gap to **cross-XCD 256-way producer convergence** — MI350X is a
chiplet part with 8 XCDs, and the bench shows `-DSOLO` collapsing that gap from
3.7 us to 0.5 us. GH100 is a monolithic die with no XCDs, so the dominant
physical cause of the AMD floor cannot exist here. A single universal constant
scaled by clock cannot represent both, which is the argument for keeping measured
constants in a calibration record keyed by hardware rather than in a shared
formula.

This record is evidence for that claim. It is deliberately **not** wired into the
cost model yet: replacing the constant changes how much every decode plan values
fusion, and that belongs behind a full model-level regression, not a drive-by
edit.
