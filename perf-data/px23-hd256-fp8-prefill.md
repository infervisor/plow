# PX-23 — the hd256 fp8 prefill arm that did not exist: 3.8x on the op, and the naive retile does not fit

RTX 5090 (sm_120a, 170 SMs, **101,376 B dynamic-smem cap**) · 2026-07-26 ·
kernel `runtime/nvidia/op_attention.cuh` `d_flash_prefill_px23`, dispatch `interp_sm120.cu`
`PLOW_DOP_FLASH_PREFILL_FP8` · bench `perf-data/px23_hd256_fp8_bench.cu`, build
`perf-data/px23_build.sh`, raw `perf-data/px23-hd256-fp8-prefill-raw.txt` ·
**every GPU run under `perf-data/harness/gpulease`**.

Closes the finding PX-20 §5 called "the single most actionable in this note".

## The defect

`interp_sm120.cu` `case PLOW_DOP_FLASH_PREFILL_FP8` had three paths and a hole:

| build | hd512 (8 FULL layers) | hd256 (40 SLIDING layers) |
|---|---|---|
| `PIPE=1` + `FP8MMA` | px4 fp8-mma arm | **`__trap()`** |
| `PIPE=1` no `FP8MMA` | `__trap()` | `__trap()` |
| `PIPE=0` (the shipped default) | generic sync staging | generic sync staging |

Gemma-4-12B = 40 sliding (hd 256, window 1024, kv_heads 8) + 8 full (hd 512, kv_heads 1). An
**all-layer** e4m3 packet — the configuration vLLM ships by default — emits hd256
`FLASH_PREFILL_FP8`, so the entire packet, both head dims, is forced onto `PIPE=0`. PX-20 §4d
measures that at **176.4 s of prefill per 127k request** against 34.9 s for the same model at bf16,
and §4a attributes the whole matched-e4m3 gap to it: vLLM 42.55 vs plow 5.59 out tok/s, **7.61x**.

## Result 1 — it is a RETILE, and the naive retile does not fit. Measured before any body was written.

`d_flash_prefill_px4` is hd512-shaped throughout, so the first question was whether relaxing its
`static_assert` could ever have worked. It could not. From the real headers
(`FA_PX4_SMEM_FLOATS` compiled and printed, not hand-derived):

| arena claim | bytes |
|---|---|
| `FA_PX4_SMEM_FLOATS(512,32,16)` — what the fp8 object claims today | 89,104 |
| `FA_PRE_SMEM_FLOATS(256,64,32)` — the generic hd256 claim | 81,664 |
| **`FA_PX4_SMEM_FLOATS(256,64,32)` — the naive px4 retile** | **104,464** |
| `sharedMemPerBlockOptin` | **101,376** |
| **`FA_PX23_SMEM_FLOATS(256,64,32)` — what shipped** | **60,416** |

**The naive retile is 3,088 B over the cap.** It would not have failed at compile time — it fails
as a refused module load / `CUDA_ERROR_LAUNCH_FAILED` at serve time, which is the worst place to
find it. (This is the same class of error PX-8 Result 1 found in PX-7's budget: 89,104 B was being
compared against a figure that was really the *bf16* object's arena.)

## Result 2 — the tiling, and why BQ=64 is free here and was not for px4

**HD=256, BQ=64, BKV=32, 8 warps, uniform.**

`oacc` is `BQ*HD/THREADS` f32 per lane:

| arm | oacc f32/lane | verdict |
|---|---|---|
| px4 hd512 BQ=32 — ships today | 64 | 238 regs / 0 spills |
| **px23 hd256 BQ=64** | **64** | 240 regs / 0 spills |
| px8's rejected hd512 BQ=64 (PX-8 Result 3) | 128 | register wall, spilled accumulators |

**PX-8's BQ=64 register wall was an hd512 property, not a BQ property.** At hd256, BQ=64 costs
exactly what the shipped arm already pays. It doubles arithmetic intensity per KV byte (`2*BQ`
FLOP/B: 128 vs 64), which matters here in a way it did not for px4 — PX-8 Result 7 showed the
hd512 full-attention arm is not KV-traffic bound, but the sliding layers re-read a whole 1024-row
window per q-tile and are.

Three structural changes from px4, each forced:

1. **No hd split.** px4 splits HD in half across 2 warp groups into `SsA`/`SsB`, summed at every
   softmax read, *only* because at BQ=32/BKV=16 the query x kv tile is 2x2 = 4 warp-blocks and it
   needs 8. At BQ=64/BKV=32 the tile is 4x2 with `NJ_QK=2` sub-tiles per warp and already fills 8
   warps. `SsB` and its add are deleted. That is 8,192 B and one add per element per tile.
2. **The bf16 `Qs` tile is deleted.** In the fp8mma arm Q is staged bf16 only to be read straight
   back by the per-q-tile e4m3 quant, after which nothing reads it — px4 carries 33,792 B of dead
   tile at hd256. px23 loads Q from gmem into **registers** (8 lanes/row, `HD/8`=32 contiguous
   bf16 each = one coalesced 512 B burst per 8 lanes), amax-reduces with the same 3 shfls, and
   writes once in A-fragment order. It also deletes a `__syncthreads` px4 needed only to publish
   that tile. **This is what brings hd256 under the cap.**
3. **BKV=32 is two k16 P.V steps**, so each lane owns 8 P elements (`{c0,c0+1,c0+8,c0+9}` and the
   same +16) and two `af_pv` fragment sets. The row reduction is still two quad shfls — 4 lanes x
   8 elems = 32 cols = BKV. Halving the tile count halves the barriers and the loop floor, which
   PX-8 Result 10 measured at 18% (loop+barrier) and 24.5% (cp.async exposure) of the hd512 fp8 arm.

`qk_wm == pv_wm == warp>>1` and `qk_wn == pv_wn == warp&1`: a warp owns the same 16 query rows in
both phases, unlike px4 where the two grids differ.

### smem layout, byte budget against 101,376

| tile | shape | bytes |
|---|---|---|
| `Ss` | `[BQ][BKV]` f32 — one tile, not two | 8,192 |
| `Vs` | `[BKV][HD+8]` fp16, `v_scale` folded in | 16,896 |
| `qsc`/`ksc`/`vsc` | `[BQ]` + 2x`[BKV]` f32 | 512 |
| `Qs8` | `[BQ*HD]` e4m3, A-fragment order, **no row pad** | 16,384 |
| `Ks8` | `[BKV][HD+32]` e4m3 | 9,216 |
| `Vs8` | `[BKV][HD+32]` e4m3 | 9,216 |
| **total** | | **60,416** |

**60,416 B is 40% under the cap and below the fp8 object's existing 89,104 B px4 claim, so
`PLOW_NV_PRE_A` does not move at all.** The arena is unchanged, so there is no
`set_max_dynamic_smem` risk and no prefill-grid collapse. It leaves **28,688 B of headroom under
the current arena** for a real K/V double buffer (2x `Ks8`+`Vs8` = +18,432 B), which is PX-8
Result 10's named top remaining lever on this kernel. A `static_assert` in the body pins the claim
to the arena so it can never drift silently.

### Per-row K/V scales

Unchanged from the px4 fp8mma discipline, which is already validated against the PIPE=0 reference:
the **Q row scale** and the **K column scale** both factor out of the e4m3 dot and post-multiply
the score at the `Ss` store; the **V row scale** folds into V at the e4m3→fp16 dequant (`|V*vsc|`
is the real activation magnitude, comfortably fp16), so P stays unscaled in [0,1]. Scales are
prefetched one tile ahead into a register and published with an STS, so their gmem latency never
touches the critical path; out-of-range rows carry 0.

## Result 3 — uniform, NOT warp-specialized, and the reason is arithmetic

PX-22 measured warp specialization at 1.144x bit-exact on the w8a8 GEMM body, and its Result 3
notes 1-2 producers suffice **given a warp grid that divides**. Here it does not: QK needs
`BQ/16 = 4` query warp rows and P.V needs 4, so the consumer count must be a multiple of 4. The
only splits are:

* **4+4** — PX-22 Result 4 measured the forced 4-consumer retile as **4.5% worse** on its own, and
  that penalty is structural, not incidental;
* **2+6** — needs BQ=96, which puts `oacc` at 96 f32/lane and re-enters PX-8 Result 3's wall.

So the PX-22 follow-up that Result 3 called "the single most promising" does not reach this kernel
without a tiling change that costs more than it buys. Recorded as a **negative for warp
specialization on the hd256 flash prefill**, derived rather than measured — and the smem headroom
above is deliberately left so the cheaper attack on the same cost (K/V double buffering) stays open.

## Result 4 — correctness: 3.8x FASTER and 1.9x MORE ACCURATE than the arm it replaces

Both arms scored against the same in-bench f32 reference kernel (same e4m3 K/V, same bf16 Q, same
exp2/log2(e) softmax, same causal + window + ring semantics). Error measured only where the
reference exceeds 5% of its peak magnitude.

| gate shape | | `pipe0` maxrel | **`px23` maxrel** | `pipe0` rms | **`px23` rms** |
|---|---|---|---|---|---|
| `causal-nowin` | seq_q 256, kv 1024 | 3.88e-02 | **9.80e-03** | 2.56e-03 | **1.33e-03** |
| `sliding-1024` | seq_q 320, kv 2048, w 1024 | 2.40e-02 | **8.10e-03** | 2.59e-03 | **1.32e-03** |
| `ragged-qpos` | seq_q 200 (not a multiple of BQ), q_pos0 777 | 4.23e-02 | **1.40e-02** | 2.84e-03 | **1.37e-03** |
| `ring-wrap` | kv_stride 2048 < seq_kv 3000, so the ring wraps | 3.35e-02 | **1.04e-02** | 2.56e-03 | **1.31e-03** |
| `nsplit2` | nsplit 2, partials merged host-side as FLASH_MERGE does | 3.17e-02 | **9.05e-03** | 2.59e-03 | **1.34e-03** |

**px23 is 1.93x more accurate in RMS than the PIPE=0 arm it replaces**, despite quantizing Q to
e4m3, which is not what a mantissa-bit count predicts. The reason is the P.V, not the QK: PIPE=0
folds `v_scale` into P and keeps both P and V in **bf16** (7 mantissa bits); px23 folds it into V
and keeps both in **fp16** (10 bits). Same mechanism PX-8 Result 8 found for px4.

### Proof the gate is not vacuous

PX-22 bug 1 — `0x7f` is the sole E4M3 NaN encoding, and a `rand() & 0x7f` fill NaNs every output
plane so all arms hash alike — burned an agent on this campaign. Four independent defences, all
asserted per shape, all reported in the raw file:

| defence | evidence |
|---|---|
| operands are finite by construction | `rnd_e4m3()` restricts the exponent field to 5..9, every operand in [0.25, 7.5] |
| no NaN reached the kernel or left it | `nan_in 0 nan_out 0` on all 5 shapes x both arms, counted on host over every input and output buffer |
| output is not the zero/degenerate plane | `nz` = **full plane** on every shape (1048576/1048576 etc.); output hash asserted != the zero-plane hash, which is printed per shape and is distinct |
| output is not stuck across shapes | each shape's hash asserted != the previous shape's; all 10 arm hashes are distinct |
| **the two binaries really saw the same data** | the **reference hashes are byte-identical across the two binaries** on all five shapes (`ecd05c93…`, `fdaec681…`, `ddb375bf…`, `59728c72…`, `d180b7f6…`). This is the check that makes the two-binary comparison legitimate at all |
| reproducible | two runs under **separate leases**: every gate line bit-identical, every PERF cell within **1.2%** |

## Result 5 — the measurement: 3.8x at the production shape, up to 7.7x when the KV loop is long

`k_arm` is the arm called directly; both arms at `__launch_bounds__(256,1)` matching the real
object. Grid 170 = 1 block/SM. Real Gemma-4 sliding-layer shape: 1024-token prefill chunk,
`n_head 16 / n_kv_head 8` (the real gqa 2), hd 256. Mean of the two leases.

| # | shape | tiles/item | `pipe0` ms | **`px23` ms** | **speedup** | `px23` TFLOP/s | % of 518.5 |
|---|---|---|---|---|---|---|---|
| 1 | window 1024, q_pos0 64, kv 1,088 | 19 | 0.4140 | **0.1088** | **3.81x** | 93.8 | 18.1% |
| 2 | window 1024, q_pos0 7,232, kv 8,256 | 34 | 0.5910 | **0.1548** | **3.82x** | 118.0 | 22.8% |
| 3 | window 1024, q_pos0 31,808, kv 32,832 | 34 | 0.5922 | **0.1547** | **3.83x** | 118.0 | 22.8% |
| 4 | full causal, q_pos0 16,384, kv 17,408 | 529 | 9.298 | **2.040** | **4.56x** | 139.2 | 26.8% |
| 5 | full causal, q_pos0 32,768, kv 33,792 | 1041 | 30.906 | **4.008** | **7.71x** | 139.5 | 26.9% |

Three things fall out.

* **Rows 2 and 3 are flat to 0.2% across a 4x change in `seq_kv`** in both arms. That is the check
  that the sliding-window bound is respected: a chunk 31,808 tokens into the context costs the same
  as one 7,232 in. A window bug would show here first.
* **Row 1 is cheaper per tile than rows 2-3 in `px23` but the TFLOP/s is *lower*** (93.8 vs 118.0),
  because at 19 tiles per work item the per-q-tile prologue — the Q e4m3 quant, paid once per
  (q-tile, head) — is a larger share. Rows 4-5 are the asymptotic control: at 529 and 1041 tiles per
  item the prologue amortizes and the mainloop rate settles at **139 TFLOP/s = 26.9% of the e4m3
  ceiling**, flat between them.
* **`pipe0` *degrades* on row 5** (30.8 → 18.0 TFLOP/s) while `px23` holds 139. The synchronous arm
  re-reads KV from HBM with the mma stalled on it, so once the working set leaves L2 it falls
  further behind; the cp.async arm does not. That is why the speedup grows from 3.8x to 7.7x with
  KV-loop length rather than staying flat.

**Note on the honest denominator.** 26.9% of the fp8 ceiling is not a good absolute number, and
this file does not claim it is. The arm's job was to exist; PX-8 Result 10's decomposition (cp.async
exposure 24.5%, loop+barrier floor 18%) says where the remaining 73% is, and none of it is
tensor-core work.

## Result 6 — cost to the object: +2 registers, 0 spills, arena unchanged, shipped cubins byte-identical

`-Xptxas -v` on the real `_Z15interp_sm120_pf11PlowProgram`, not on the bench:

| prefill object | registers | spill | stack | arena |
|---|---|---|---|---|
| fp8-KV `PIPE=0` — the shipped default | 238 | 0 | 1024 B | 85,248 B |
| **fp8-KV `PIPE=1` — px4 (hd512) + px23 (hd256)** | **240** | **0** | 1024 B | 89,104 B |
| bf16 | 238 | 0 | 1024 B | 85,248 B |

Occupancy is 1 block/SM either way — 240 > 128 pins it on registers independently of smem, exactly
as PX-8 Result 1 established, so nothing about this changes occupancy.

**Byte-identity, md5 of the real cubins, this tree vs `git show HEAD`:**

| object | HEAD | this branch | |
|---|---|---|---|
| bf16 prefill | `3c48c945ab25d0f5…` | `3c48c945ab25d0f5…` | **identical** |
| fp8-KV prefill, `PIPE=0` (what ships today) | `6866da44e7b9ddc0…` | `6866da44e7b9ddc0…` | **identical** |
| fp8-KV prefill, `PIPE=1` | `256eabcb7803b1d0…` | `f8c2db96951c7897…` | changed, as intended |

Every shipped object is byte-identical. The only object that moves is the one that previously
contained a `__trap()` where the arm now is.

## Result 7 — wiring: there is no third state to mis-build

`FA_PX23_ELIGIBLE(HD)` is gated on exactly the same
`PLOW_NV_FA_PIPE && PLOW_NV_FA_PX4 && PLOW_NV_FA_FP8MMA` as the hd512 arm, so **no build exists in
which one arm is present and the other is not**. An all-layer e4m3 packet is served by the fast
path end to end or not at all; there is no configuration that serves hd512 fast and traps on hd256.
`scripts/build_sm120_cubin.sh` gains `PLOW_BUILD_FP8KV_FASTPF=1` to select it (default off, so the
shipped recipe is unchanged).

**On "fail loudly at load":** there is no `PLOW_PACKET_HASH` machinery in this tree. The real
load-time guard is `set_max_dynamic_smem` in `crates/plowrt/src/exec/gpu.rs`, which errors if the
arena exceeds `sharedMemPerBlockOptin`. Since px23's claim (60,416) is under the arena the object
already had (89,104), it cannot trip it — and the `static_assert` in the body makes an arena
overflow a **compile** error rather than a serve-time one. That is the guarantee that was actually
available; no new machinery was invented to look like more.

## Result 8 — the end-to-end gate could NOT be run on this branch base, and here is the proof it is not px23

This is a negative and it is recorded as one.

The e2e A/B was set up properly: an **all-layer e4m3 packet re-emitted by this worktree's own
`plowc`** (`perf-data/px23_emit.sh`, `PLOW_FP8_KV=1` without `_KV_FULL`, fp8 weights,
`PLOW_UNISEG=1 PLOW_NS_FULL_ABS=32`), this worktree's own decode/sampler cubins, and the prefill
object as the **only** variable across three arms (`perf-data/px23_e2e.sh`). Every arm failed:

| arm | prefill object | 66,901-token needle |
|---|---|---|
| `headfast` | HEAD `PIPE=1` — has px4, **no hd256 arm** | `CUDA_ERROR_LAUNCH_FAILED` |
| `pipe0` | `PIPE=0` — **no fp8 mma anywhere** | `CUDA_ERROR_LAUNCH_FAILED` |
| `px23` | `PIPE=1` + the new arm | `CUDA_ERROR_LAUNCH_FAILED` |

The attribution chain, each step a separate run under its own lease:

1. **`pipe0` fails too**, and it contains no fp8 mma at all. So the failure is upstream of every
   flash prefill arm — the same logic PX-8 Result 11 used to clear px8 of the needle failure.
2. **A 20-word prompt fails**, so it is not a long-context, bucket or chunking property.
3. **A pure bf16-KV packet, served with the bf16 prefill object that is byte-identical to HEAD,
   also fails at 20 words.** No fp8 anywhere in that configuration.

So **prefill serving of Gemma-4-12B is broken on this branch base (`main`) independently of this
change**, and no end-to-end number is obtainable here. PX-20 ran this exact model successfully on
`worktree-plowrt-max-completion-tokens`, which is **102 commits ahead** and carries PX-17's
prefill patch-site fix among others. The merge of that branch into this worktree was **refused by
the permission classifier**, twice, which is why this work sits on the older base.

**Nothing in this file may be scaled through a prefill budget to claim a wall-clock win.** The
honest statement of what is expected, clearly labelled as arithmetic rather than measurement: PX-20
§4d measures 176.4 s of all-layer e4m3 prefill against 34.9 s at bf16 for the same model, and this
file measures the hd256 op at 3.8x. Those two numbers are consistent with a large end-to-end win,
and this campaign has three times produced rankings that inverted when measured end to end
(PX-13 on `GLU_BN`, PX-8 on the px8 lever, PX-7 on the BQ=64 premise). **It must be measured on a
tree where prefill works.**

## Gates

| gate | result |
|---|---|
| smem claim verified from the real headers before any body was written | **PASS** — 60,416 B, and it caught the naive retile at 104,464 > 101,376 (Result 1) |
| the arm cannot outgrow its arena silently | **PASS** — `static_assert(FA_PX23_SMEM_FLOATS <= FA_PRE_SMEM_FLOATS)` in the body: compile error, not a serve-time launch failure |
| registers / spills from `-Xptxas -v` on the REAL object, not the bench | **PASS** — 240 regs, 0 spills, vs 238 for the PIPE=0 baseline |
| numerics vs an f32 reference on real shapes | **PASS** — and px23 is **1.93x better in RMS** than the arm it replaces, on all 5 shapes |
| ragged `seq_q`, `q_pos0 != 0`, sliding window, ring wrap, `nsplit > 1` all covered | **PASS** — 5 shapes, `nsplit2` merged host-side the way FLASH_MERGE does |
| **the correctness gate is not vacuous** | **PASS** — 4 independent defences; `nan_in 0 / nan_out 0`, full non-zero planes, distinct zero-plane and cross-shape hashes, and the **reference hash byte-identical across the two binaries** (Result 4) |
| reproducible under separate leases | **PASS** — gate lines bit-identical, PERF cells within 1.2% |
| no reading above the 518.5 TFLOP/s e4m3 ceiling | **PASS** — asserted per row by the bench; best is 139.8 (27.0%) |
| bf16 prefill cubin byte-identical to HEAD | **PASS** — `3c48c945…` both sides |
| shipped fp8-KV `PIPE=0` cubin byte-identical to HEAD | **PASS** — `6866da44…` both sides |
| the hd256 arm exists wherever the hd512 one does | **PASS** — same `FA_PX23_ELIGIBLE` gate; no build has one without the other (Result 7) |
| `cargo build --release -p plowrt --features cuda,hf-tokenizer` | **PASS** — clean, warnings only, all pre-existing |
| `cargo test --workspace` | **FAIL, PRE-EXISTING** — `crates/plowrt/src/asset/devblob.rs` test constructors miss `Program::{l2_domains,l2_sms}` and `Model::target`. Test-only struct drift on `main`; **this change touches 0 Rust files** and the `--features cuda` release build passes |
| **end-to-end prefill** | **NOT OBTAINABLE ON THIS BASE** — and proven not to be px23: the `pipe0` control (no fp8 mma) and a **bf16 packet on the HEAD-identical bf16 object** both fail at a 20-word prompt (Result 8) |
| GPU exclusive | **ENFORCED** — every run under `gpulease`; 8 leases |
| `ncu` counter attribution | **NOT RUN** — `ERR_NVGPUCTRPERM` in this container, as in PX-9/13/22. Every claim here is differential timing between arms differing in one thing |
| warp specialization (PX-22 follow-up) | **NOT APPLICABLE, derived** — the warp grid does not divide at any consumer count that keeps `oacc` under the wall (Result 3) |
| e4m3 P.V (PX-8's px8 lever) | **NOT DONE** — `d_flash_prefill_px8` is not in this worktree (the merge was refused). BKV=32 was chosen so it drops in without another retile |

### Bugs found mid-run, recorded

1. **A window sweep is not a control when `seq_q == window`.** The first perf table swept
   `window ∈ {1024, 0}` at `seq_q = 1024` and both settings produced identical times, because
   causal masking already caps the reach at 1024 — the "asymptotic" rows were measuring the same
   work as the production rows. The host-side tile counter also computed the window floor
   *unconditionally*, ignoring the kernel's `if (window)` guard, which silently mis-scaled the
   TFLOP/s column. Both fixed; the asymptotic control now uses a large `q_pos0` with `window = 0`,
   which really does give 529 and 1041 tiles per work item.
2. **`__launch_bounds__(256)` without the min-blocks argument caps ptxas at 128 registers** and made
   the bench arm spill 4 B where the real object does not. The real kernel is
   `__launch_bounds__(256, PLOW_NV_MINBLK)`; the bench now matches it, and both arms report 0 spills.

## What to do next, in order

1. **Merge to a tree where prefill works and re-run Result 8.** That is the only missing number and
   it is the one that decides whether PX-20's 7.61x actually closes. Nothing else should be built on
   top until it exists.
2. **A real K/V double buffer.** PX-8 Result 10 puts cp.async exposure at 24.5% of the hd512 fp8
   arm, and this layout deliberately leaves 28,688 B under the *current* arena — enough for
   `2x(Ks8+Vs8)` = 18,432 B — so it costs no arena growth at all.
3. **The e4m3 P.V** (PX-8's px8, 1.40x on the hd512 op). BKV=32 is already in place for it.
4. Not warp specialization on this kernel — see Result 3.
