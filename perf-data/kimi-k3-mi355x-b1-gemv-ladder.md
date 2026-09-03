# Kimi-K3 MI355X B1 BF16 GEMV audit

Date: 2026-09-03. Hardware: one MI355X (gfx950, 256 CUs). Toolchain: repository Nix
ROCm 7.14. Trace: `/tmp/k3-d0-decode.trace`; packet disassembly from the matching
`/tmp/plow-k3-repro-seg-16384.3829231/model.pkt`.

## Finding

The best isolated kernel candidate found is a model-neutral short-K column ladder. The shipped BF16 row
kernel groups two independent output columns per wave. At the current 224/256-workgroup K3
geometry, each wave commonly owns three or four columns. Processing full groups of four, then two,
then one increases outstanding weight loads without changing column ownership or any dot-product
reduction order.

The ladder remains harness-only because its 1.4% gain does not justify invalidating the measured
TuneDB profile. A standalone lean object is safe (wave64, 512 threads, 116 VGPR, 64 SGPR, 16 KiB
LDS, zero private memory and zero VGPR/SGPR spills), but the production decode megakernel is
already at 256 VGPR with
624 B private memory, two VGPR spills and 86 SGPR spills. Promotion therefore requires a separate
opcode-only decode segment rather than adding the ladder to the megakernel's register union.
A production `d_gemv` dispatch compile with the ladder enabled is also clean: wave64, 512 threads,
173 VGPR, 68 SGPR, zero private memory and zero spills. Its 144 KiB interpreter-sized LDS arena
still limits it to one workgroup per CU, so this is a compile/resource gate rather than the final
lean-object shape.

## Current trace

The matched critical-chain trace attributes 15.603 ms over 816 plain `Gemv` packets. Its largest
cohorts are:

| N | K | blocks | packets | body ms | body us/packet | effective GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 7168 | 1536 | 256 | 93 | 2.263 | 24.330 | 905 |
| 7168 | 768 | 256 | 92 | 2.124 | 23.088 | 477 |
| 896 | 3584 | 224 | 92 | 2.095 | 22.774 | 282 |
| 3584 | 7168 | 256 | 92 | 2.018 | 21.932 | 2343 |
| 896 | 7168 | 224 | 92 | 2.005 | 21.798 | 589 |
| 1536 | 128 | 256 | 69 | 1.348 | 19.534 | 20 |

The trace rate includes persistent-dispatch skew and spill traffic. The exact isolated kernel moves
the same weights in 1.8--9.2 us for ordinary projections, so the 12--20 us small-packet floor cannot
be removed by changing GEMV arithmetic alone.

## Exact-shape isolated gate

The harness walks fresh weight slabs from a 3 GiB arena, uses the packet's actual grid, interleaves
all arms palindromically, and compares every BF16 output bit against the shipped body. The clean
production-header recheck is the authoritative timing. The new production-header wrapper
has the same normalized 2,277-instruction ISA as the cleanly measured wrapper
(`sha256 6ade118c046ac697dc993a2a80d6a4c3fb01875a4b2faeadb1165185f33a19d8`).

| arm | weighted body projection |
|---|---:|
| shipped packet body | 3.660 ms |
| shipped R=2 short-K grouping (`k_rA`) | 2.998 ms |
| R4→R2→R1 ladder | **2.956 ms** |

Ladder vs shipped R=2 = **-0.042 ms (-1.4%)**. It is also -0.704 ms vs the unspecialized packet
body. Every shape was bit-exact. The A/A range was 0.9972--1.0027. The similarly named `k_r2`
benchmark arm is a mixed R4/R2 experiment, not the shipped R2 arm, and is excluded from this
comparison.

The material per-shape wins vs the unspecialized packet were 1.53x at `(7168,1536)`, 1.68x at
`(7168,768)`, and 1.31x at `(6144,1536)`. Against the actual shipped R2 arm, the main incremental
win is 8.9% at `(7168,768)`; `(896,3584)` regresses 0.7%, and the remaining short-K shapes move by
less than 1.6%. Wide-K shapes remain on the existing single-column arm and measure 0.99--1.01x.

An independent clean grid sweep over blocks `{12,64,96,128,192,224,249,256}` found only 0.006 ms
of byte-weighted opportunity (3.690 ms emitted grids vs 3.683 ms per-shape best). The hot emitted
grids are already optimal: `(7168,1536)` b256 = 6.974 us / 3157 GB/s, `(896,3584)` b224 = 2.254 us /
2849 GB/s, `(7168,768)` b256 = 6.623 us / 1662 GB/s, and `(3584,7168)` b256 = 7.447 us / 6900 GB/s.
Only the one-per-token LM head moved materially, b256 341.003 us to b224 334.723 us, for 0.006 ms
total benefit.

## vLLM / AITER comparison

The pinned AITER `bf16_tuned_gemm.csv` has no exact K3 B1 rows. Large K3 projections therefore
fall back to PyTorch solution 0. Only outputs `N <= 512` meet AITER's default skinny predicate and
route to `wv_splitk_small_fp16_bf16` (solution 2). `LLMM1` is present as skinny solution 1 but is
not the default for these untuned shapes. AITER's skinny kernels also use row-major weights and
wave64; their main transferable technique is multiple output rows per wave/WG with deep independent
loads, which is the same memory-level-parallelism lever captured by this ladder without changing
Plow's weight layout.

Plow TuneDB likewise contains no qualified gfx950 BF16 GEMV records; its gfx950 database currently
contains only GEMM rows. The emitted B1 grids are analytical choices, not TuneDB selections.

## Reproduction

```sh
nix develop -c bash -lc '"$PLOW_HIPCC" --offload-arch=gfx950 -O3 -w \
  -std=c++17 --genco -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_K3=1 \
  -DGV_UNROLL=14 -DPLOW_FP8_KV=1 -DPLOW_GEMV_LG=1 \
  -I runtime/amd -I runtime/common runtime/bench/amd/k3_gemvbf16_bench.hip \
  -o /tmp/k3-gemv-ladder-gfx950.co'
```

Object SHA256: `b86e4c281a684a7118d29ee251f72a5eda509ac2fc11e65adcfea9ae0cf6f7b8`.

## Decision

Keep the ladder out of the production header. Next gate: compile `Gemv` as an opcode-only lean decode object,
emit exact-capability segments only for models that require plain BF16 GEMV, and repeat the
interpreter-through single-packet oracle. Do not route this through the current spilling megakernel.
