# kx — the single-block kernel-experiment harness

`kx` answers "is this kernel idea faster, and is it still correct?" in **seconds**, against **one
workgroup**, without a blob, a server, or a serving benchmark.

```
scripts/kx.sh <experiment> [--grid=1|model|both] [--it=N] [--shape=NAME] [--isa] [--rebuild]
```

Measured on this box: `gemv_mm` cold (both code objects + the host driver + the full run) 8.9 s;
warm 6.7 s, `gemv_tile` 6.6 s, `fa_softmax` 2.8 s — each at `--it=41`, both geometries, all shapes.

## Why it exists

Every kernel question on this branch used to cost a campaign: rebuild 45 objects, emit a blob,
take a GPU lease, start a server, run a serving benchmark, then a separate greedy-identity run.
20–60 minutes for a one-line answer. Several agents spent hours to reach one. One whole result was
a **false null** because a candidate was measured against a stale binary, and another candidate
was reported as a **win while producing wrong output**.

The questions themselves are small and local — "does this tile spill?", "is R=2 better than R=4 at
this shape?", "how many `ds_bpermute` does this softmax issue?" — and they can be asked against
the body alone.

`kx` is the fast loop for exactly those. It is **test/bench infrastructure**: it never changes a
shipped kernel body or default, and it includes production bodies verbatim from the shipped
headers the way the benches beside it do.

## What one run gives you

```
kx gemv_mm — what does a batch-1 packet pay for running on an MM=4 decode object?
    device   AMD Instinct MI300X (gfx942:sramecc+:xnack-)  ROCR_VISIBLE_DEVICES=0
    launch   blockDim=512  median-of-41, palindromic interleave, all arms in ONE process
    variant  mm1      -DPLOW_GEMV_MM=1
    variant  mm4      -DPLOW_GEMV_MM=4
    ...
GEOM 1wg   ONE workgroup, doing exactly the share the model deals it (slice 0 of kx_nblk). ...
GEOM model  grid = the emitter's own workgroup count for the shape.
  per-token totals over the tabulated instance counts (ms): mm1:k_gemv 15.272  mm4:k_gemv 26.637
CORRECTNESS (device-side elementwise, vs mm1:k_gemv at the model grid)
A/A control (mm1:k_gemv_aa / mm1:k_gemv): 0.9970 .. 1.0053   tolerance 0.980 .. 1.020   PASS
OK: A/A in tolerance and every arm matches the reference.
```

### The two geometries, and why both are printed

* **`1wg`** — one workgroup, doing **exactly the share the model deals it**: `kx_nblk` is always
  the shape's model workgroup count, so a 1wg launch computes slice 0 and nothing else. (A
  workgroup made to do the *entire* shape is a different question, 300x the work, and minutes
  instead of seconds.) The idea measured on its own body, not on 304 CUs of scheduling noise.
* **`model`** — the grid is the emitter's own workgroup count for that shape, from
  `plowrt disasm <blob>/model.pkt --program N`. Not a CU count: the K3 router GEMV runs 224 and
  `b_proj` runs 12.

**They disagree, and that is the single most important thing this harness shows.** In
`gemv_tile`, R=2 reads 1.672x at `moe_down_lat` under `1wg` and 1.063x in the model — the same arm,
the same shape, a 16-point swing in the claim. In `gemv_mm` the MM=4 penalty is ~2.9x under `1wg`
and 1.2–2.4x in the model. A single-block win is a **hypothesis** about the model, never a result
about it: the megakernel wants a different tile than the standalone primitive, and on this branch
a 9% standalone spread became 16% in the model with the standalone winner losing.

Per-token totals are printed only under `model`, where they mean something.

### The correctness gate

Not optional and not skippable. Every arm that is not the reference is launched at the model grid
with `nrep=1` into its own buffer and compared **elementwise on device** against the experiment's
declared reference (the shipped body), reporting max error, rms error and an exact-match count.

An arm declared `KX_EXACT` must be **bit-identical** — that is the right bar whenever the change
only moves which wave does which work, and it catches a class of bug that a tolerance hides.

If any arm is wrong, the run prints `REFUSED AS A WIN` and exits non-zero. Its speed column is
still printed, but only so the trap stays visible.

### The A/A control

Every experiment declares an arm that is **byte-identical device code to the baseline under a
second name** — the harness gets this for free because `KX_BODY` is `__forceinline__`, so
`KX_ARM_OF(k_x, b)` and `KX_ARM_OF(k_x_aa, b)` differ only in the symbol.

If the A/A ratio falls outside tolerance, **the ratios are replaced by `REFUSED`** and the run
exits non-zero. An unpinned run on a contended box has read A/A between 0.27 and 4.57 on this
machine; without this control none of the small deltas here are readable.

`--aa-tol=lo:hi` tightens the gate, which is also how you check the gate itself still bites.

### `--isa`: what does this body actually issue?

Two sources, deliberately side by side:

* **ISA** — `llvm-objdump` of the code object, counted per kernel symbol. **Spill comes from
  here**, as `scratch_load_*` / `scratch_store_*`, because candidates in this tree have reported
  `.vgpr_spill_count: 0` in the metadata and spilled anyway. Instruction mix comes from here too:
  the prefill-attention win came from noticing one KV tile issues 160 `ds_bpermute` and 163
  `s_waitcnt` against 64 MFMAs.
* **META** — the compiler's own `-Rpass-analysis=kernel-resource-usage` remarks (VGPR/AGPR/SGPR/
  LDS/occupancy). Useful, but the row is flagged when its spill number disagrees with the ISA.

```
ISA  variant 'shfl'                     ISA  variant 'dpp'
arm      total bpermute swizzle waitcnt  arm      total bpermute swizzle waitcnt
k_soft     804      160       0     122  k_soft     748        0      64      47
```

## Adding an experiment: one file, one entry

Create `runtime/bench/amd/kx/exp_<name>.hip`. That is the whole change — the driver discovers it.

The file is compiled **twice**: by `hipcc --genco` for the arms, and by `g++ -DKX_HOST` for the
descriptor. Keeping both in one file is what stops a variant list and the arms it names from
drifting apart.

```c
#include "kx.h"

#ifndef KX_HOST                    /* ---------------- device: the arms */
#include "op_gemm.h"               /* include the SHIPPED body; never edit it */

/* A body is __forceinline__, so instantiating it twice gives byte-identical code. */
KX_BODY(b_base) {
    /* kx_out, kx_w, kx_x, kx_aux, kx_nrep, kx_nblk, kx_d0, kx_d1, kx_d2 are in scope. */
    for (unsigned e = 0; e < kx_nrep; e++) { /* walk a fresh slab out of the arena */ }
}
KX_ARM_OF(k_base, b_base)
KX_ARM_OF(k_base_aa, b_base)       /* the A/A control — free, and mandatory */

KX_BODY(b_cand) { /* the candidate */ }
KX_ARM_OF(k_cand, b_cand)

#else                              /* ---------------- host: the descriptor */

static const kx_variant VAR[] = {  /* one code object per entry; "" for no extra defines */
    {"a", "-DSOME_KNOB=0"},
    {"b", "-DSOME_KNOB=1"},
};
static const kx_arm ARM[] = {
    {"k_base",    0, KX_BASE | KX_REF, "the shipped body"},
    {"k_base_aa", 0, KX_AA,            "byte-identical device code to k_base"},
    {"k_cand",    1, KX_EXACT,         "what this arm changes, and nothing else"},
};
static const kx_shape SHP[] = {
    /* name, d0, d1, d2, grid, inst, slab_bytes, reps, out_elems */
    {"o_proj", 7168, 1536, 0, 256, 93, 7168.0 * 1536.0 * 2.0, 0, 7168},
};
static const kx_exp EXP = {
    "name", "the question in one line", "runtime/bench/amd/kx/exp_name.hip",
    "-DPLOW_BUCKET_DECODE=1 -DPLOW_WG_WAVES=8",   /* the SHIPPED object's defines */
    "gfx942", 512, KX_DT_BF16,
    VAR, 2, ARM, 3, SHP, 1,
    0.98, 1.02,      /* A/A tolerance */
    0.0,             /* max abs error for arms without KX_EXACT */
    "the known answer, so a reproduction is checkable at a glance",
};
extern "C" const kx_exp* kx_experiment(void) { return &EXP; }
#endif
```

Then: `scripts/kx.sh <name>`.

### Fields that matter

| field | why it is load-bearing |
|---|---|
| `kx_exp::defines` | The **shipped object's own defines**, copied from `scripts/build_gfx942.sh`. They gate which arms of `op_gemm.h` / `op_attention.h` instantiate at all, and they participate in the digest the tuning store keys on. A bench built without them measures a different kernel. The flash object is `PLOW_WG_WAVES=4`; the decode object is `8`. |
| `kx_shape::grid` | The **emitter's own** workgroup count for the shape, from `plowrt disasm`. Not the CU count. |
| `kx_shape::slab_bytes` | Bytes the in-kernel rep loop walks per rep. The driver picks `nrep` so the loop streams a fresh slab out of the arena — otherwise the number is an L2 hit, not a stream. Use `reps` instead for a body with no weight stream. |
| `kx_shape::out_elems` | How many elements the correctness gate compares. Zero disables the gate for that shape, which you should not do. |
| `KX_EXACT` | Declare it whenever the candidate is arithmetic-preserving by construction. Demanding bit-identity is strictly stronger than a tolerance and costs nothing. |
| `KX_NOCHECK` | Only for attribution arms with no comparable output (a stage-only arm, say). |
| `KX_SLOWOK` | The arm is expected to lose; a regression is the result, not a failure. |

### Arm signature

Every arm has the same shape so the driver can launch any of them without knowing what it does:

```
void* kx_out, const void* kx_w, const void* kx_x, const void* kx_aux,
unsigned kx_nrep, unsigned kx_nblk, unsigned kx_d0, unsigned kx_d1, unsigned kx_d2
```

`kx_nblk` is **always the shape's model workgroup count**, never `gridDim.x`. An arm must pass it
as the `nblk` its body partitions on, and `blockIdx.x` as the `slice` — that is what makes the
`1wg` geometry mean "one workgroup's real share" rather than "one workgroup doing everything".

`kx_w` is a 3 GiB arena of deterministic small-magnitude bf16 (`--arena-mb=` to resize); `kx_x`
and `kx_aux` are 64 MiB of the same. All are filled on device, so startup is not a PCIe copy.

## What it refuses, and how

| refusal | exit | trigger |
|---|---|---|
| `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` set | 64 | They **compose** with `ROCR_VISIBLE_DEVICES`; setting both makes a correctly targeted card report "no ROCm-capable device is detected". |
| `ROCR_VISIBLE_DEVICES` unset | 64 | The run is not pinned to an idle card. |
| more than one device visible | 64 | The lease must narrow to exactly one card. |
| no `KX_BASE` or no `KX_AA` arm | 67 | An experiment without an A/A control cannot defend a delta. |
| a shape's slab exceeds the arena | 68 | Clamping `nrep` would read out of bounds; it tells you the `--arena-mb` to pass. |
| A/A outside tolerance | 3 | Ratios are replaced by `REFUSED`. Nothing in that table may be reported. |
| an arm is WRONG | 4 | `REFUSED AS A WIN`. |
| wrong toolchain | 2 | The device objects must come from the flake's `rocm-7.14.0-nix` hipcc, the same one `scripts/build_gfx942.sh` uses. |

### Staleness

The rebuild key is a **content hash** of the source and of **every header the previous compile
actually opened** (read out of hipcc's own `-H` include trace), plus the defines, the arch and the
toolchain label. Not mtimes: a whole result on this branch was a false null because a candidate
was measured against a stale binary, and an `-nt` test would not catch a header restored from git
with an older timestamp. Verified by editing `amd_common.h` with its mtime set back to 2020 — kx
rebuilds.

`--rebuild` forces one anyway.

## What kx cannot answer

It measures **a body**. It cannot measure a **data-dependent branch** whose payoff depends on real
activations. `FA_LAZY_RESCALE` is exactly such a question — a wave-voted skip of the online-softmax
rescale — and it was ported into `exp_fa_softmax.hip` and then removed: against synthetic scores it
read 1.25x *faster* than plain DPP, the opposite of the measured tree result. That belongs in a
campaign against real activations. See the note in `exp_fa_softmax.hip`.

It also cannot tell you whether a body-level win survives the megakernel's register and LDS
budget, or its scheduling. The `model` geometry narrows that gap; it does not close it. A kx win
is the *cheap* half of the evidence, and the campaign is still the *sufficient* half.

## The experiments in the tree

| experiment | question | known answer it reproduces |
|---|---|---|
| `gemv_mm` | what does a batch-1 packet pay on an MM=4 decode object? | 15.52 vs 25.86 ms/token recorded; kx reads 15.27 vs 26.64, bit-identical output |
| `gemv_tile` | at short-K decode shapes, does R=2 beat R=4? | R=2 is the shipped choice (3.907 vs 4.039 ms/token); the ragged R=1 tail eats R=4 |
| `fa_softmax` | does the DPP row softmax cut the tile's LDS permutes, bit-identically? | 160 `ds_bpermute` → 64 `ds_swizzle`, `s_waitcnt` 122 → 47, bit-identical; plus a trap arm that must be rejected |

## Prior art in this directory

`kx` does not replace the hand-written benches beside it; it packages their discipline.
`k3_gemvbf16_bench.{hip,cpp}` is the model it was built from — production body included verbatim,
launch geometry mirroring the persistent interpreter, an in-kernel rep loop over a large arena,
hipEvent median-of-41, palindromic interleave, an A/A control arm and a device-side elementwise
check. Read it, plus `gemma31_gemv_decode_bench`, `k3_gemv_cohort_sweep`, `glm52_gemvblk_bench`
and `mla_prefill_8k_sweep`, before writing an experiment that does something new. The
lease/toolchain traps are written up in `scripts/gemv_one_shape.sh` and
`scripts/rebench_tune_gemm_gfx942.sh`.
