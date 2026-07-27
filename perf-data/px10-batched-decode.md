# PX-10 — batched long-context decode, attributed on the single-block harness

RTX 5090 (sm_120a, 170 SM, 32 GiB, driver 580.159.03), Gemma-4-12B unified,
fp8 weights (W8A8) + bf16 sliding KV + fp8 full-layer KV — the config the
concurrency-8 127k measurement in `perf-data/gemma4-12b-longctx-5090.md` §8 ran.
All GPU runs under `perf-data/harness/gpulease`.

**Verdict: the gap is NOT long-context and NOT the sliding layers. It is a pure
BATCH effect, it appears as a single discontinuity at B=4 → B=8, and it is two
things in roughly equal parts —**

1. **the GEMV row-block ladder at M=8** — 21.0 ms of the 42.5 ms device step,
   reading the *same* 10.9 GB of weights as M=1 in **2.5× the time**. **Most of
   this is a build-flag bug, not a kernel limit**: the deployment cubin is built
   `-DGV_MM_MAX=16`, which routes M=8 through `gemv_walk`'s runtime-`rows`
   remainder arm instead of its compile-time-full loop, predicating every FMA in
   the hottest loop in the model. Dropping the flag is **−19.4% at 131k / −33.8%
   at 1k, free at B≤4** (§5a).
2. **the fp8-KV full-layer flash-decode** — 18.5 ms incl. its merge wait, running
   at **527 GB/s**, 2.5× off the bandwidth model, at *every* batch size. Not a
   batching regression; batching multiplies it by 8. bf16 KV is **worse** (§5b),
   so this needs a kernel, not a flag.

The sliding layers' flash is **at bandwidth** (1161 GB/s) and is 5% of the step —
candidate (a) is ruled out. Gate/counter overhead is 12% and is mostly
`FlashMerge` waiting on full-layer flash stragglers — real, but not the story.
`NS_FULL_ABS` is confirmed inert from the device side.

**The bandwidth model is right, not wrong.** plow's own M=1 GEMV streams fp8
weights at **1495 GB/s** (83% of the 1792 GB/s pin) and the whole B=1 step lands
1.13–1.30× off the 1.3 TB/s floor. 1.3 TB/s is attainable on this card; the B=8
step is 2.50× off it, and 2.01× after §5a.

## Method

`plowc --block l..r` extracts a layer range as a standalone asset;
`block_run <asset> bench` prefills B slots to T rows and times decode steps on it.

| asset | layers | content | ×N = the 48-layer model |
|---|---|---|---|
| `blk-slide` | L4 | 1 sliding | — |
| `blk-full` | L5 | 1 full | — |
| `blk-x6` | L6..11 | 5 sliding + 1 full | ×8 |
| `blk-x12` | L12..23 | 10 sliding + 2 full | **×4** |

`x12` is the unit of record: `2·x6 − x12 = 2 µs` at B=1 and 34 µs at B=8, i.e.
the per-launch fixed cost is **zero** — the block times are pure per-layer cost
and the ×4 reconstruction is linear.

Blobs are emitted at `PLOW_DECODE_BATCH = B` **matching the batch driven**. This
matters: a B=8 blob bakes M=8 into every GEMV and computes 8 rows *even when only
one slot is fed*, so "B=1 on the B=8 blob" is not a B=1 program (it is 6302 µs vs
the real B=1 blob's 3087 µs — a 2.0× difference at the same active work).

## Gates

| gate | result |
|---|---|
| finiteness (`block_run check`, both blocks) | **PASS** — no NaN/Inf |
| block additivity (`2·x6 − x12`) | **PASS** — 2 µs @B=1, 34 µs @B=8 (0.03%/0.3%) |
| single-layer vs multi-layer (`5·slide + full` vs `x6`) | **PASS** — 5437 vs 5341 µs @B=8/131k (−1.8%) |
| device-vs-host (`PLOW_STEP_TIME`) | **PASS** — `dev_interp_ms` 10.623 vs wall 10.649 ms; host is 20 µs |
| trace cycles ↔ wall linearity | **PASS** — 2.94 vs 2.97 GHz-equivalent across the B=1/B=8 traces |
| `NS_FULL_ABS` 32 vs compiler default (85) | **NOT A LEVER** — 2892 vs 2867 µs @B=8/131k (0.9%) |
| host-vs-device split of the *served* 66 ms | **NOT RUN** — owned by the parallel `PLOW_STEP_TIME` serving run |
| numeric parity vs HF/vLLM | **NOT RUN** — no `transformers` in this env (pre-existing harness scope) |
| `GV_MM_MAX=8` fix A/B | **WIN** — −19.4% @131k, −33.8% @1k, B≤4 unchanged (§5a) |
| bf16-KV fix A/B | **NEGATIVE** — +25% @B=8/131k; fp8 KV stays (§5b) |
| trace of the bf16-KV blob | **NOT RUN** — killed to release the lease (bf16-KV prefill is 11× slower; 22 min for a confirmatory number) |

**Bugs found mid-run** (both cost a full sweep):

1. `block_run bench` uploaded all `T` rows of `act.x` in one shot. Activations
   are sized by `MAX_CHUNK` (8192 rows), not by the context, so **every ctx >
   8192 failed** with `act.x: 62914560 f32 > tensor capacity 31457280 bf16` —
   i.e. the harness could not reach the long-context regime it exists to measure.
   Fixed here: `bench` now loops `prefill_chunk` and re-uploads `act.x` per chunk,
   with a `--pf-chunk` flag (default 8192, unchanged behaviour for T ≤ 8192).
2. With `PLOW_FP8_KV=1`, the **hd512 full-layer prefill crashes at bucket ≥ 4096**
   (`CUDA_ERROR_LAUNCH_FAILED`, rc 719) — T=1024/2048 fine, T=4096 dies. Worked
   around with `--pf-chunk 2048`; **not diagnosed further, not fixed.** This is a
   real defect in the fp8-KV prefill object on the hd512 shape.
3. The box's CUDA forward-compat driver (`/usr/local/cuda/compat`, 580.167.08) is
   NEWER than the kernel driver (580.159.03), so plowrt's dlopen order picks a
   libcuda that fails `cuInit` with `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`.
   Pin `PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1`.

## 1. The gap is BATCH, not context

48-layer decode step reconstructed as `4 × x12`, each blob compiled at its own
`PLOW_DECODE_BATCH`. Bandwidth floor = (10.9 GB fp8 weights + B × KV) / 1.3 TB/s;
KV/slot = 40 × 1024 × 8 × 256 × 2 × 2 B (sliding, bf16, window-clamped) +
8 × ctx × 512 × 2 × 1 B (full, fp8).

| ctx | B | step ms (deployed cubin) | step ms (§5 fix) | floor ms | **× floor** | × floor, fixed |
|---|---|---|---|---|---|---|
| 1024 | 1 | 9.74 | 9.73 | 8.65 | **1.13** | 1.13 |
| 1024 | 2 | 10.70 | 10.68 | 8.91 | 1.20 | 1.20 |
| 1024 | 4 | 12.92 | 12.89 | 9.44 | 1.37 | 1.37 |
| 1024 | 8 | 24.43 | **16.16** | 10.50 | **2.33** | **1.54** |
| 131072 | 1 | 12.35 | 12.33 | 9.47 | **1.30** | 1.30 |
| 131072 | 2 | 15.88 | 15.84 | 10.55 | 1.51 | 1.50 |
| 131072 | 4 | 23.29 | 23.23 | 12.72 | 1.83 | 1.83 |
| 131072 | 8 | **42.60** | **34.35** | 17.06 | **2.50** | **2.01** |

**At ctx 1024 — where there is essentially no KV — B=8 is already 2.33× off.**
The context barely moves the *ratio* (2.33 → 2.50). Whatever the "batched
long-context" deficit is, the *long-context* half of the name is wrong.

And the curve has a **discontinuity at B=4 → B=8**: at ctx 1024 the step jumps
12.92 → 24.43 ms, +89% for +14% of bytes, while B=1→2→4 tracks the floor almost
exactly. That single step is the whole gap, and §5 removes most of it with a
build flag.

## 2. Per-op attribution (PLOW_NV_TRACE=1, block 0, x12 @ ctx 131072)

This is the **deployed** cubin (`-DGV_MM_MAX=16`) — the configuration the ~66 ms
serving number was taken on. §5a then removes 8.2 ms of the GEMV row.

Trace cycles normalised per step, then scaled to the 48-layer step by the
measured device time (`dev_interp_ms` × 4). Read the shape; `clock64` serialises
on the recording thread, but the *same* factor holds across both columns
(2.94 vs 2.97 GHz-equivalent), so the comparison is sound.

| component | B=1 ms | B=8 ms | Δ | bytes @B=8 | **GB/s @B=8** | floor ms |
|---|---|---|---|---|---|---|
| **GEMV ladder** (`GemvFp8`+`GemvGluFp8`, incl. gate) | 9.47 | **21.00** | +11.53 | 10.9 GB | **519** | 8.38 |
| **full-layer flash** (`FlashDecodeFp8`) | 2.23 | **16.31** | +14.08 | 8.59 GB | **527** | 6.61 |
| full-layer `FlashMerge` (≈all gate) | 0.00 | 2.20 | +2.20 | — | — | — |
| sliding-layer flash (`FlashDecode`) | 0.53 | 2.31 | +1.78 | 2.68 GB | **1161** | 2.06 |
| norms / rope / residual | 0.05 | 0.66 | +0.61 | — | — | — |
| **total (device)** | **12.29** | **42.49** | +30.2 | 22.2 GB | 522 | 17.05 |

Three things fall straight out:

- **The GEMV grows +11.53 ms for ZERO extra bytes.** Weight traffic is identical
  at M=1 and M=8 (`gemv_walk` is weight-stationary, one pass for both). Per GEMV
  workgroup-packet the body goes **77.0 → 221.0 kcyc (2.87×)** for `GemvFp8` and
  **214.8 → 450.8 kcyc (2.10×)** for `GemvGluFp8`. This is the single largest
  avoidable term in the step.
- **The sliding-layer flash is fine.** 1161 GB/s, 5% of the step, and it scales
  with B at the bandwidth slope (+1.78 ms measured vs +1.81 ms floor). Candidate
  (a) from the ask is **ruled out**.
- **The full-layer flash is 2.5× off at *every* B** (2.23 ms vs 0.83 ms floor at
  B=1; 16.31 vs 6.61 at B=8). It is not a batching regression — it is a constant
  inefficiency that batching *multiplies* by 8. fp8 KV is still the right choice
  (§5b: bf16 is +25%), but `FlashDecodeFp8` at 527 GB/s against the bf16 arm's
  1161 GB/s is where the remaining headroom lives.

Gate/counter overhead (candidate (c)) is **12% of the B=8 step** — 2.66 ms of
`GemvFp8` gate + 2.20 ms of `FlashMerge` gate + 0.6 ms of norm gates. The
`FlashMerge` share is the full-layer split's stragglers, not scheduler tax. It is
real but it is not the story.

## 3. Why M=8 GEMV costs 2.5× M=1 for the same bytes

The decode GEMV is a **w8a16 FFMA** kernel (`op_gemm.cuh` `gemv_rows_fp8`): fp8
weight, bf16 activation, scalar FMA, per-column scale. Per weight *byte* it does
1 e4m3→float conversion and **M** FMAs. The megakernel is a cooperative launch
pinned at **1 block/SM** (`occ_per_sm=1`, confirmed in the load log), so an SM has
only its warps to hide that chain behind the weight stream.

On the **deployed** cubin, 10.9 GB of weights per step:

| | body ms | GB/s | TFLOP/s |
|---|---|---|---|
| M=1 | 7.29 | **1495** | 3.0 |
| M=8 | 17.97 | **607** | 9.7 |

Bandwidth fell 2.46× while the FLOP rate rose 3.2× — the batched arm left
bandwidth and became issue-bound.

**Two separable causes, and the bigger one is a build flag.** §5a shows
`-DGV_MM_MAX=16` predicates every FMA at M=8; removing it cuts the `GemvFp8` body
46% and `GemvGluFp8` 37%. What remains is the genuine M-rung cost — per
workgroup-packet body, same (correct) cubin, ctx 1024:

| op | M=1 | M=8 | genuine M-rung tax |
|---|---|---|---|
| `GemvFp8` | 77.6 kcyc | 113.7 kcyc | **1.47×** |
| `GemvGluFp8` | 215.7 kcyc | 251.3 kcyc | **1.17×** |

So of the ~2.5× M=8 GEMV penalty in the shipped build, **roughly ⅔ is the flag
and ⅓ is the FFMA arm's real ALU cost at M=8.**

For that residual third, `rtx19-E4` already built and gated the fix — a
`mma.sync.m16n8k32.e4m3` **w8a8 tensor-core** twin
(`runtime/tests/e4_tc_fp8_decode_sm120.cu`) that streams the weight once and does
the M-way accumulation on tensor cores (0 spill, 54/77 registers, oracle-gated,
1.38× over FFMA at 12B B=8 in a standalone full-occupancy kernel). E4 explicitly
left `op_gemm.cuh` and the megakernel untouched. That integration is the
follow-up — **after** §5a, which is free and larger.

## 4. `NS_FULL_ABS` and the sliding nsplit — confirmed non-levers

| variant | B=1 µs | B=8 µs |
|---|---|---|
| `blk-full` `NS_FULL_ABS=32` (deployed) | 885.9 | 2892.6 |
| `blk-full` compiler default (ns=85) | 987.9 | 2866.5 |
| `blk-slide` default (ns=22) | 480.3 | 508.8 |
| `blk-slide` `NS_ABS=16` | 468.8 | 505.7 |

Reproduces §8's flat sweep from the device side. One incidental find: the
windowed-layer nsplit cap in `devgen` (`ns.min(win/64)`, measured −0.30 ms/token)
is gated on the *per-layer* `fp8_kv`, so `PLOW_FP8_KV_FULL=1` — which sets fp8 on
the full layers and leaves the sliding rings bf16 — **disables it on the sliding
layers**. Worth ~0.5 ms/step (40 × 11.5 µs). Small, but free.

## 5. Fix candidates measured

### 5a. Drop `-DGV_MM_MAX=16` from the deployment cubin — **WIN, zero code**

Same B=8 blob, two cubins differing only in that flag (`x12`, wall median µs):

| ctx | `GV_MM_MAX=16` (deployed) | `GV_MM_MAX=8` (source default) | Δ |
|---|---|---|---|
| 1024 | 6106.5 | **4040.9** | **−33.8%** |
| 131072 | 10649.2 | **8588.1** | **−19.4%** |

B=1 / 2 / 4 are **unchanged** (2436→2432, 2675→2671, 3229→3223 µs at ctx 1024) —
this costs nothing anywhere and only pays at B=8. Scaled to the 48-layer step at
ctx 131k / B=8: **42.60 → 34.35 ms**, i.e. 2.50× → **2.01×** off the floor.

The trace says it is entirely the GEMV body (same blob, ctx 1024, B=8, kcyc per
workgroup-packet):

| op | MM_MAX=16 | MM_MAX=8 | Δ |
|---|---|---|---|
| `GemvFp8` body | 211.4 | **113.7** | **−46%** |
| `GemvGluFp8` body | 396.4 | **251.3** | **−37%** |
| `GemvFp8` gate | 56.8 | 26.8 | −53% |
| `FlashDecode` body | 151.3 | 149.7 | −1% |
| `FlashDecodeFp8` body | 327.8 | 329.2 | +0.4% |
| `NormResidualNorm` body | 9.3 | 9.0 | −3% |

**Root cause — `gemv_walk`, `op_gemm.cuh:118`.** The rung is `gv_mm<8>` in both
builds; what differs is how `rows` reaches it:

    for (; m0 + GV_MM_MAX <= M; m0 += GV_MM_MAX) f(gv_mm<GV_MM_MAX>{}, m0, (unsigned)GV_MM_MAX);
    ...
    #if GV_MM_MAX > 8
    else if (rem <= 8) f(gv_mm<8>{}, m0, rem);          // rem is a RUNTIME value

With `GV_MM_MAX=8`, M=8 goes through the loop and `rows` is the literal `8`, so
nvcc folds away the `if ((unsigned)m >= M) continue;` guard inside
`gemv_rows_fp8`'s innermost `#pragma unroll` and emits 8 clean `dot8_fp8` FMAs.
With `GV_MM_MAX=16`, M=8 falls to the `rem <= 8` arm and `rows` is opaque, so
**every one of the 8 FMAs in the hottest loop of the whole model carries a
runtime predicate.**

So the follow-up one-line kernel fix is to route an exact-multiple remainder
through a compile-time-full call (which would also fix M=24 under `GV_MM_MAX=16`,
where `rem` is again 8). Not implemented here — the build-flag fix already
captures it for the shipped B=8 config.

**Note the source already says this.** `op_gemm.cuh`'s own table has
`GV_MM_MAX=16` costing 17% at B=8 and recommends it only for deployments pinned
at B≥16. `perf-data/gemma4-12b-longctx-5090.md` §7 builds the B=8 asset with
`-DGV_MM_MAX=16` anyway.

### 5b. bf16 KV instead of fp8 full-layer KV — **NEGATIVE, rejected**

Hypothesis: `FlashDecodeFp8` runs at 527 GB/s vs `FlashDecode`'s 1161 GB/s, so
halving the bytes may not pay. Measured (`x12`, ctx 131072):

| KV dtype (full layers) | B=1 µs | B=8 µs |
|---|---|---|
| fp8 (`PLOW_FP8_KV_FULL=1`) | 6263* | **10649** |
| bf16 | 6263 | 13320 (**+25%**) |

fp8 KV **wins** — the marginal rate on the extra bf16 bytes is 805 GB/s, better
per byte than fp8's 527 but not enough to cover 2× the traffic. Keep fp8 KV.
(*the B=1 column is the B=8 blob with one slot fed, so the two are the same
program; they agree to 0.6%.) Side effect worth knowing: **bf16-KV prefill on the
`-DPLOW_NV_FA_PIPE=0` object is 11× slower** (54.9 s vs 4.8 s per 131k slot), so
this A/B is expensive to repeat.

### 5c. Not attempted

Integrating `rtx19-E4`'s w8a8 tensor-core decode GEMM into `op_gemm.cuh` — the
remaining GEMV headroom after 5a. Deliberately out of scope: the ask was
attribution, and this is a kernel integration, not a flag.

## 6. Reconciling with the served 66 ms

The device kernel is **42.5 ms** at B=8 / ctx 131k. §8's ~66 ms/step is derived by
subtracting an *estimated* 248 s prefill from a 385 s wall — a subtraction with
wide error bars — and it includes the whole serve path. The ~24 ms difference is
host / serve-layer / prefill-interleave and is exactly what the parallel
`PLOW_STEP_TIME` serving run measures; **this report claims only the device
side.** Independent support that the device number is right: at B=1 the same
reconstruction gives 12.35 ms against a served ITL of 16.21 ms — the same ~4 ms
of non-kernel time.

## 7. Reproduce

    # libcuda (see Gates bug 3)
    export PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1
    # cubin: deployment flavour
    PLOW_ROOT=$PWD nvcc -arch=sm_120a -O3 -cubin -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 \
      -DPLOW_NV_FA_GF_FULL=4 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_FP8_KV=1 \
      -DGV_MM_MAX=16 -DPLOW_NV_W8A8=1 [-DPLOW_NV_TRACE=1] ...
    # block asset (B is the point — emit at the batch you will drive)
    PLOW_SKIP_COVERAGE=1 PLOW_UNISEG=1 PLOW_DECODE_BATCH=$B PLOW_FP8=1 PLOW_W8A8=1 \
    PLOW_FP8_KV=1 PLOW_FP8_KV_FULL=1 PLOW_NS_FULL_ABS=32 \
      plowc --hf-dir /root/gemma4-fp8-ckpt --gpu rtx5090 --emit devblob \
            --max-ctx 132096 --weight-dtype fp8 --block 12..24 --out <dir>
    # sweep
    gpulease px10 block_run <dir> bench --batch $B --ctx 1024,131072 \
      --iters 100 --warmup 20 --prefill-iters 1 --pf-chunk 2048

Raw output: `/root/px10/out/` (transient).
