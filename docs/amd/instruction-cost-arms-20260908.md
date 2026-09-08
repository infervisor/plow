# AMD instruction-cost arms — measuring the 2026-09-08 static audit's candidates

Follow-up to [the GPU kernel static and instruction-cost audit](../../plans/gpu-kernel-static-audit-20260908.md),
"Costly instructions and repeated work". Four of that table's candidates are AMD-side; this is
what each one measured, what landed, and what deliberately did not.

**The default build is unchanged, byte for byte.** All 45 gfx942 objects built from this branch
are `cmp`-identical to the ones built from its parent with the same command, and the object
contract holds over all 45. Every arm below is default-off.

## Result in one table

| # | Candidate | Body (kx, model geometry) | Numerical price | Landed |
|---|---|---|---|---|
| 1 | mHC Sinkhorn divide → reciprocal | **1.48x** (IEEE rcp) / **1.65x** (`v_rcp_f32`) | ≤ 2.1e-7 abs vs the shipped arm; **no further from f64 truth** (0.97–1.12x its error) | `PLOW_HC_SINKHORN_RCP`, default 0 |
| 2 | DeltaNet per-head work hoisted off V rows | **1.47x** at batch 1, **1.74x** at batch 4 (VROWS=4); VROWS=8 **loses 20%** at batch 1 | bit-identical output | `PLOW_QWEN_GDN_VROWS`, default 1 |
| 3 | Quantization scale from the exponent | **1.05x** / **1.07x** | differs from the shipped arm at 37–38 of 130 power-of-two boundary cases | `PLOW_QUANT_SCALE_EXP`, default 0 |
| 4 | Hadamard helper's LDS extent | — | — | comment corrected; no live bug |
| 5 | Runtime integer divide/modulo in addressing | — | — | **nothing to do** — measured null |

## The thing that shapes every number below

**No blob on this box exercises three of these four bodies.** `plowrt disasm
build-glm53/tp4-long/model.pkt` is 1865 instructions over 33 opcodes and contains **no**
`HyperConnPre`, **no** `HyperConnPost` and **no** DSA-pool or compress op — GLM-5.3 `tp4-long` is
MLA + MoE. `PLOW_QWEN_GDN` is 0 in `scripts/build_gfx942.sh`, so op 137 is not compiled into any
shipped object at all.

So "measure the model" has no target for candidates 1, 2 and 3, and saying so is the result. What
IS measurable, and was measured, is the other half of the model question: **does touching these
bodies perturb the shipped megakernel?** The answer is no, and it is byte-exact — see
[Object A/B](#object-ab) — which is a stronger statement than a served-latency null would be.

## Candidate 1 — mHC Sinkhorn divide → reciprocal

`runtime/amd/op_hyperconn.h`, `PLOW_HC_SINKHORN_RCP` (0 shipped / 1 IEEE reciprocal / 2 `v_rcp_f32`).

### ISA evidence

The audit asked whether the production ISA already reuses the common denominator. **It does not.**
gfx942 static counts for a probe kernel wrapping `d_hyperconn_pre` at n=4, `sinkhorn_repeat=20`:

| | HEAD | RCP=0 | RCP=1 | RCP=2 |
|---|---:|---:|---:|---:|
| total instructions | 1932 | 1932 | 1396 | 1300 |
| `v_div_scale_f32` | 146 | 146 | 50 | 18 |
| `v_div_fmas_f32` | 73 | 73 | 25 | 9 |
| `v_div_fixup_f32` | 73 | 73 | 25 | 9 |
| `v_rcp_f32_e32` | 73 | 73 | 25 | 33 |
| `v_exp_f32_e32` | 24 | 24 | 24 | 24 |
| spill (`scratch_*`) | 0 | 0 | 0 | 0 |

73 divide expansions, none shared. The 73 decompose exactly as the source predicts: 16 (initial
softmax) + 16 (first column pass) + 32 (ONE rolled Sinkhorn iteration) + 8 (the pre/post sigmoids)
+ 1 (the `block_sum` mean) — and the rolled iteration runs 19 times, which is where the audit's
**640 divides per token** comes from. RCP=1 removes 48 of the 73 static ones.

**Arm 0's disassembly is byte-identical to HEAD's**, addresses and encodings included.

### Body measurement — `scripts/kx.sh mhc_sinkhorn`

hidden = 6144, n = 4, `sinkhorn_repeat` = 20, grid = `rows.min(n_cu)` per `emit_glm53_hc_pre`.

| shape | geom | base µs | A/A | RCP=1 | RCP=2 |
|---|---|---:|---:|---:|---:|
| decode T=1 | 1wg | 34.81 | 1.000 | **1.635** | **1.897** |
| pf304 | 1wg | 34.81 | 1.001 | 1.633 | 1.895 |
| pf2048 | 1wg | 243.05 | 1.000 | 1.637 | 1.898 |
| decode T=1 | model | 34.87 | 1.000 | **1.634** | **1.897** |
| pf304 | model | 41.43 | 1.000 | **1.477** | **1.648** |
| pf2048 | model | 289.36 | 1.000 | 1.478 | 1.647 |

The two geometries disagree by 16 points at prefill, as they usually do here — 1.63x standalone
reads as 1.48x once 304 workgroups compete. The model column is the claim.

### Numerical price — `runtime/tests/hyperconn_sinkhorn_gfx942_test.hip`

An f64 oracle of vLLM's `mhc_pre_torch` Sinkhorn (not a transliteration of the kernel), over three
logit spreads so the loop actually iterates rather than starting at its own fixed point.

| spread | arm | max err vs f64 | vs arm 0 | ratio to arm 0 |
|---|---|---:|---:|---:|
| 0.05 | 0 | 9.19e-08 | — | 1.00 |
| 0.05 | 1 | 1.03e-07 | 1.04e-07 | 1.12 |
| 0.05 | 2 | 1.03e-07 | 8.94e-08 | 1.12 |
| 1.50 | 0 | 1.32e-06 | — | 1.00 |
| 1.50 | 1 | 1.28e-06 | 2.09e-07 | 0.97 |
| 1.50 | 2 | 1.34e-06 | 1.79e-07 | 1.01 |
| 8.00 | 0 | 5.43e-06 | — | 1.00 |
| 8.00 | 1 | 5.34e-06 | 1.79e-07 | 0.98 |
| 8.00 | 2 | 5.40e-06 | 1.79e-07 | 0.99 |

The reciprocal arms are **no further from the truth than the divide they replace** (0.97–1.12x),
and their absolute disagreement with the shipped arm is ≤ 2.1e-7 on entries bounded by 1 — one to
two ULP. Note the shipped arm's own 5.4e-6 at spread 8: 20 passes of far-from-1 denominators in
f32 dominate anything these arms add.

**Landed default-off anyway.** It changes bits, and this branch's standing policy (the
`PLOW_GEMV_MFMA4` precedent) is that a rounding change is a flag with its price written down, not
a default — the more so for a body that no packet on this box executes, so nothing here can be
defended by a served number.

## Candidate 2 — DeltaNet per-head work hoisted off the V rows

`runtime/amd/op_qwen_gdn.h`, `PLOW_QWEN_GDN_VROWS` (1 shipped, N = V rows per wave). The AMD port
of NVIDIA's `PLOW_NV_GDN_STEP_VROWS8`.

Everything above the state loop — 2·PL bf16 loads of q and k, two `wave_sum`s over them, two
`sqrtf` plus two divides, `expf(-expf(a_log)·softplus(dt))`, and `sigmoid(b)` — depends on
(slot, head) and not on the V row. At vdim = 128 the shipped mapping recomputes it **128 times per
head**.

### ISA evidence

VROWS=1 is byte-identical to HEAD. Static totals barely move (2832 → 2934 for the four-rung
dispatch) and the per-mnemonic counts do not move at all: the saving is entirely dynamic, and so
is the loss. Nothing spills at any VROWS.

### Body measurement — `scripts/kx.sh gdn_vrows`

hk=16, hv=48, kdim=128, vdim=128, grid 304 (`b.all()`).

| shape | geom | base µs | A/A | VROWS=2 | VROWS=4 | VROWS=8 |
|---|---|---:|---:|---:|---:|---:|
| batch1 | 1wg | 6.19 | 1.000 | 1.089 | 1.472 | **0.919** |
| batch4 | 1wg | 21.86 | 1.000 | 1.332 | 1.817 | 1.667 |
| batch1 | model | 6.53 | 1.000 | 1.096 | **1.468** | **0.803** |
| batch4 | model | 23.24 | 0.998 | 1.328 | **1.735** | 1.486 |

**VROWS=8 — the NVIDIA arm's own choice — is a 20% REGRESSION at batch 1**, and it is a regression
for a reason that has nothing to do with instruction counts. batch 1 is 6144 rows over 304
workgroups × 8 waves = 2432 waves, i.e. 2.5 rows per wave before any tiling; VROWS=8 leaves 768
tiles and 68% of the waves idle. A 64-lane wave and an 8-wave workgroup are not a 32-lane warp,
and the tile that is right on one is not right on the other. **VROWS=4 is the AMD answer**, and it
wins at both batches and both geometries.

### Numerical price

None: bit-identical bf16 output, 6144/6144 and 24576/24576 exact under kx's device-side gate. The
f32 recurrence state is scored by the f64 oracle in `runtime/tests/qwen_gdn_gfx942_test.hip`
instead, where VROWS 1/2/4/8 all land inside the shipped arm's own run-to-run band (max rel
1.007e-7 .. 1.167e-7 over five runs of VROWS=1 — that test is not bit-reproducible run to run,
which is a property of the test and not of these arms).

### What did NOT land, and why

**VROWS=4 is not the default even though it is a bit-identical 1.47–1.74x win.** Op 137 is
compiled out of every shipped gfx942 object (`PLOW_QWEN_GDN=0`), so there is no served token to
defend the change with and nothing to gain by making it. The moment a Qwen3-Next blob exists on
this box, re-run `scripts/kx.sh gdn_vrows` at that blob's real batch and promote VROWS=4 if the
model column agrees — the measurement is already set up for it.

## Candidate 3 — quantization scale from the exponent

`runtime/amd/amd_common.h` (`plow_round_scale`, `PLOW_QUANT_SCALE_DIV`), used by
`op_dsa_pool.h`'s two sites and `op_compress.h`'s fake quant. `PLOW_QUANT_SCALE_EXP` (0 shipped,
1 exponent-based).

Arm 1 takes the exponent out of `frexpf` and rebuilds the scale with `ldexpf` — and gets the
**exact** reciprocal for free, turning the per-element IEEE divide into a multiply.

### ISA evidence, including the part that says "do not bother"

Arm 0 is byte-identical to HEAD. Arm 1 removes one `v_log_f32`, one `v_exp_f32` and one divide
expansion per site — and the **whole-kernel total goes UP** at three of four sites:

| kernel | arm 0 | arm 1 |
|---|---:|---:|
| `k_q_quant` | 371 | 383 |
| `k_pool_compress` | 569 | 581 |
| `k_compress_fp8` | 957 | 968 |
| `k_compress_fp4` | 1453 | **1417** |

The audit's cost table put `log2f` at 35 instructions and `exp2f` at 28; those are whole-probe
numbers, and in these bodies the compiler lowers each to a single `v_log_f32_e32` /
`v_exp_f32_e32` plus a short range fixup. The scale CONSTRUCTION is a wash. What is not a wash is
invisible to a static count: `cmp_fake_quant_block` divides `qblk` (64, or 32) elements by one
loop-invariant scale on ONE thread, and the loop is rolled, so that divide is counted once and
executed 64 times.

### Body measurement — `scripts/kx.sh quant_scale`

| shape | geom | base µs | A/A | arm 1 |
|---|---|---:|---:|---:|
| csa (d=512, qblk=64, fp8) | 1wg | 30.19 | 1.000 | 1.063 |
| indexer (d=128, qblk=32, fp4) | 1wg | 9.73 | 1.001 | 1.101 |
| csa | model | 34.96 | 1.000 | **1.054** |
| indexer | model | 14.58 | 1.000 | **1.065** |

5–7%, from the dynamic divide the static count could not see. The output was bit-identical on
kx's input (155648/155648 and 38912/38912 exact) — which is **not** a claim that the arms agree.

### Numerical price — `runtime/tests/quant_scale_gfx942_test.hip`

They do not agree, and the boundary is exactly where the audit said to look:

| top | arm 0 ≠ exact ceiling | arm 1 ≠ exact ceiling | arm 0 ≠ arm 1 | `scale*inv ≠ 1` | nonfinite |
|---|---:|---:|---:|---:|---:|
| 448 (e4m3) | **37 / 130** | 0 / 130 | 38 / 130 | 0 | 0 |
| 6 (e2m1) | **38 / 130** | 0 / 130 | 39 / 130 | 0 | 0 |

Every disagreement is the same shape: for `amax/top` **one ULP above a power of two**, `log2f`
rounds to the integer, `ceilf` returns the exponent below, and the shipped arm produces a scale a
**factor of two too small**. Arm 1 returns the exact ceiling everywhere, its reciprocal is exact
everywhere (which is what makes the multiply bit-identical to the divide), and neither arm
produces a nonfinite scale on any input including subnormal backstops below both amax floors.

**This is a finding about the shipped kernel, not only about the arm** — but it is not obviously a
bug. The reference (`fast_round_scale`, kernel.py:36-37) computes the same expression in f32, so
the shipped arm is plausibly bit-compatible with the vendor kernel and arm 1 would DIVERGE from
it. Promoting arm 1 needs a comparison against the vendor kernel's own output on those boundary
values, which needs a V4/GLM-5.3-Flash checkpoint this box does not have. Default-off, price
written down.

`compress_pool_gfx942_test.hip` (11 cases, bf16 round trips, fp8 and fp4, prefill and decode
boundaries) and `dsv4_ops_gfx942_test.hip` both PASS with the shipped arm after the refactor.

## Candidate 4 — the Hadamard helper's LDS extent

`dsa_hadamard128_stage` (`runtime/amd/op_dsa_pool.h`) documented "`lds` must be exactly 128 floats"
while writing `lds[threadIdx.x]` for **all** `blockDim.x` threads — 512 at `PLOW_THREADS`.

**No live bug.** All three call sites pass a buffer that is large enough: the interpreter passes
`sm->raw`, and the one standalone caller, `compress_pool_gfx942_test.hip`, already allocates
`CMP_LDS = 512`. `d_compress_pool`'s own doc comment already stated the real requirement
(`max(d, PLOW_THREADS)`).

So the **comment** was fixed, not the write extent. Suppressing the stores past 128 would only
leave the slots those same threads then read uninitialized, for a result that is discarded — a
change with a cost and no benefit. The corrected comment states `blockDim.x`, says why the threads
past 128 store at all, and records that a standalone caller believing the old line would have
corrupted 384 floats past its allocation.

## Candidate 5 — runtime integer divide/modulo in addressing: a measured null

The audit said "check generated ISA first; do not turn compile-time shifts into a proposed
optimization." Checked, on the **shipped** GLM-5.3 objects, counting `v_rcp_iflag_f32*` (the gfx942
tell for an unsigned runtime division) and locating each one inside its smallest enclosing loop:

| object | insn | runtime divides | not in any loop | innermost ≤ 64 insn | 65..256 | > 256 |
|---|---:|---:|---:|---:|---:|---:|
| `interp_decode_k3` | 231,243 | 197 | 126 | **0** | 5 | 66 |
| `interp_prefill_fp8kv_k3_moe_a4w4` | 125,946 | 165 | 108 | **0** | 3 | 54 |
| `interp_flash` | 24,182 | 20 | 15 | **0** | 0 | 5 |

**Not one runtime integer divide sits in an innermost loop** anywhere in the shipped GLM-5.3
object set. Two thirds are not inside a loop at all; the rest sit in the op-level grid-stride
loops, where an ~11-instruction divide amortizes over hundreds of instructions of real work. The
constant power-of-two divisors the audit anticipated have indeed already become shifts. There is
no hot runtime divisor to specialize, and no work was done here beyond establishing that.

## Object A/B

Built twice with the same command, once from this branch and once from its parent:

```
scripts/build_gfx942.sh <out>
```

* **45 / 45 objects `cmp`-identical.**
* `PASS  contract held over 45 object(s)` on both, and the two contract tables (VGPR / AGPR /
  SGPR / LDS / occupancy / ISA-counted spill, every row) are `diff`-identical.

That is the whole model-side claim for the defaults, and it is exact rather than statistical: no
served token can behave differently, because there is no different instruction to execute.

Getting there took one revision worth recording. The first version of candidate 1 routed **arm 0**
through a `__forceinline__` helper. It produced the same opcode histogram, the same VGPR count and
the same contract row — and still moved **7 register assignments** inside the outlined
`d_hyperconn_pre` of all 14 K3 objects, because adding an inline function to a translation unit
shifts a register-allocator tie-break. The standalone probe could not see it; only the object A/B
could. Arm 0 now keeps the shipped expression character for character under `#if`, at the cost of
twenty duplicated lines, and the objects are identical. **Equivalent-but-different codegen is not
what a default is allowed to be here.**

## Reproducing

```bash
# bodies (each 3-10 s, A/A control and device-side correctness gate included)
GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 kx scripts/kx.sh mhc_sinkhorn --isa
GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 kx scripts/kx.sh gdn_vrows
GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 kx scripts/kx.sh quant_scale

# numerics (build lines are in each test's file header)
runtime/tests/hyperconn_sinkhorn_gfx942_test.hip   # arms 0/1/2 vs an f64 mHC Sinkhorn oracle
runtime/tests/quant_scale_gfx942_test.hip          # 260 power-of-two / floor / subnormal cases
runtime/tests/qwen_gdn_gfx942_test.hip             # -DPLOW_QWEN_GDN_VROWS=N, f64 oracle
runtime/tests/compress_pool_gfx942_test.hip        # the quant path end to end, unchanged
runtime/tests/dsv4_ops_gfx942_test.hip             # the mHC head_only arm, unchanged

# objects
nix develop --command scripts/build_gfx942.sh /tmp/objs
```
