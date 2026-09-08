# Norm range and recurrent-gate conditioning on gfx942

Findings **F5** and **F6** of `plans/gpu-kernel-static-audit-20260908.md`, measured. Both were
static-analysis findings about *reachable* numerical failure; this is what the hardware and the
shipped checkpoints actually do. F5 is below; F6 lands with its own change.

**F5 in one line.** It needs no arithmetic change: the shipped bodies run 16-17 orders of magnitude
below the overflow point on both checkpoints, the audit's cancellation example is computed
*exactly* by the code the compiler emits, and the remedy costs 13-32%.

Everything below is on MI300X / gfx942, ROCm 7.14 from the flake, blobs
`build-glm53/tp4-long` (GLM-5.3 TP4, 93 layers) and `build-gemma31/assets-final-plain`
(Gemma-4 31B).

---
## 1. F5 — normalization range and conditioning

`runtime/amd/op_norm.h`. Every norm in the family reduces `sum(x_i^2)` in FP32 over bf16 inputs.
bf16 carries FP32's exponent range with 8 bits of mantissa, so a finite, exactly-representable
input can square to `+inf`: `x = 2^64` does. The row then normalizes to **zero** (`rsqrt(inf) = 0`)
with no error raised, or in `d_layernorm_bias` to **NaN** (`inf - inf`).

### 1.1 The contract

```
sum over the row of x_i^2  <  FLT_MAX        i.e.   max|x_i| < sqrt(FLT_MAX/feat)
```

| feat | where | per-element bound `sqrt(FLT_MAX)` | per-row bound `sqrt(FLT_MAX/feat)` |
|---|---|---:|---:|
| 6144 | GLM-5.3 hidden | 1.845e19 | 7.44e16 |
| 5376 | Gemma-4 hidden | 1.845e19 | 7.96e16 |
| 2048 | GLM `q_a_layernorm` | 1.845e19 | 1.29e17 |
| 512 | GLM `kv_a_layernorm` | 1.845e19 | 2.58e17 |
| 256 | Gemma per-head `q_norm`/`k_norm` | 1.845e19 | 3.65e17 |
| 128 | GLM-5.2 DSA `indexer.k_norm` (LayerNorm) | 1.845e19 | 5.16e17 |

### 1.2 Measured norm inputs

`PLOW_DUMP_ACT` on every act tensor that feeds a norm, at four prefill contexts plus decode steps,
reduced by max |x| and by max FP32 row sum-of-squares over the live rows only.

**GLM-5.3 TP4**, contexts 128 / 512 / 2048 / 8192 + 4 decode steps each:

| tensor | norm it feeds | max \|x\| | max row sum-sq | margin to element overflow |
|---|---|---:|---:|---:|
| `act.x` | `model.norm` (final residual) | **1080** | **2.883e6** | 1.71e16 x |
| `act.xmid`, `act.xnext` | layer input/post-attn residual | 225 | 1.379e6 | 8.20e16 x |
| `act.qlr` | `q_a_layernorm` (feat 2048) | 0.508 | 10.87 | 3.63e19 x |
| `act.ckvraw` | `kv_a_layernorm` (feat 512) | 6.84 | 175.6 | 2.70e18 x |

**Gemma-4 31B**, contexts 128 / 1024 / 8192 + 4 decode steps each:

| tensor | norm it feeds | max \|x\| | max row sum-sq | margin |
|---|---|---:|---:|---:|
| `act.kg` | `k_norm` (per head, hd 256) | **55** | 7431 | 3.35e17 x |
| `act.dg` | `post_feedforward_layernorm` addend | 35.5 | 3.351e4 | 5.20e17 x |
| `act.qg` | `q_norm` (per head, hd 256) | 27.1 | 2332 | 6.90e17 x |
| `act.og` | `post_attention_layernorm` addend | 16.4 | 1.461e4 | 1.13e18 x |
| `act.x` | residual / `input_layernorm` | 15.5 | 305.9 | 1.19e18 x |

Zero non-finite values in any dump. **The audit's own example value, bf16 `2^64` = 1.845e19, is
1.7e16 x larger than anything GLM-5.3 produces and 3.4e17 x larger than anything Gemma-4 produces.**

The act buffers alias across layers (GLM-5.3 ping-pongs `act.x`/`act.xmid`; Gemma updates `act.x`
in place), so a dump can only see the LAST layer's contents. That is the maximum-magnitude point of
a pre-norm transformer, but it is one point. §1.4 covers the other 92 layers.

### 1.3 The cancellation half is not reproduced by the shipped code

The audit's second example: 127 copies of bf16 255 and one 256, `feat = 128`, where an
**uncontracted** FP32 `msq - mean*mean` gives 0.0078125 against a true 0.00775146484375 — a 0.787%
variance error. The plan flags that FMA contraction changes this and asks for the emitted code to
be checked. It was:

```
v_div_fmas_f32 v2, v2, v7, v3          ; block_sum(ss)/feat  ->  msq
v_div_fixup_f32 v2, v2, v37, v5
v_fma_f32 v2, -v8, v8, v2              ; msq - mean*mean, ONE rounding
v_add_f32_e32 v2, s18, v2              ; + eps
v_rsq_f32_e32 v2, v2
```

`hipcc -O3` (no fast-math) contracts the subtraction. With one rounding instead of two the example
comes out **exactly right** — relative error 0, against 7.874e-3 uncontracted.

That is not a coincidence of one input. Simulating the kernel's own reduction (at `feat = 128` and
512 threads each thread owns exactly ONE element, so `block_sum` is a balanced butterfly, and
`/(float)feat` is a power-of-two division and exact) over near-constant bf16 rows from `|x| = 255`
to `2^20`, one to eight elements perturbed by one bf16 ulp: the contracted variance is exact in
every case and **never negative**, so `rsqrt` of a cancelled-negative variance is not reachable at
this shape either. The uncontracted form is wrong by up to 1.6% on the same rows.

The structural reason: severe cancellation needs a near-constant row, a near-constant row's bf16
squares share one exponent, and 128 of those sum exactly in FP32. Widen `feat`, drop the
power-of-two, or put more than one element per thread and the argument has to be re-made.

`d_layernorm_bias` is also the only non-RMS norm in any supported model (GLM-5.2 / GLM-5.3-Flash
DSA `indexer.k_norm`), and neither shipped blob emits it: `scripts/glm53_mi300x.sh` emits with
`PLOW_GLM_DSA=0`, and `build.json` for both blobs lists no `LayerNorm` opcode.

### 1.4 All 93 layers: the debug-build tripwire

`PLOW_NORM_RANGE_CHECK=1` (default 0) compiles a predicate on every reduced sum-of-squares in
`op_norm.h` and in `op_gemm.h`'s fused-norm GEMV; a violating workgroup traps.
`PLOW_NORM_SS_MAX` sets the ceiling — `FLT_MAX` by default, i.e. only real overflow. Lowering it
turns the facility into a *bound*: a campaign that completes proves nothing anywhere exceeded it.

`scripts/build_gfx942.sh` passes both through when they are set in the environment.

```
PLOW_DECODE_BATCH=4 PLOW_NORM_RANGE_CHECK=1 PLOW_NORM_SS_MAX=1e12f \
  scripts/build_gfx942.sh <objdir>
```

GLM-5.3 TP4 on those objects, contexts 128 / 2048 / 8192, 64 greedy decode steps each, all four
ranks token-identical, **no trap**:

```
prefill: 128 tokens in 1003.9 ms -> 46745 (all 4 ranks agree)   63 timed steps 64.652 ms/token
prefill: 2048 tokens in 1575.2 ms -> 73115                      63 timed steps 66.217 ms/token
prefill: 8192 tokens in 3599.9 ms -> 429                        63 timed steps 67.952 ms/token
```

So across **all 93 layers, every norm packet, every token of three contexts**, no row's
sum-of-squares reached 1e12 — **3.4e26 x below FLT_MAX**. That is the all-layer statement the
activation dumps cannot make.

**Negative control**, because a tripwire that cannot fire proves nothing: the same build at
`PLOW_NORM_SS_MAX=1e3f` (below the measured 2.883e6) does not complete. It **hangs** rather than
faulting, and that is worth knowing before using this facility: `__builtin_trap` inside the
persistent interpreter kills the trapping workgroup while the rest spin on its counter. Every other
`__builtin_trap` in `runtime/amd/` (`op_collective.h`, `packed_prefill.h`, `op_kda.h`) has the same
property. A run that stops making progress IS the signal; it needs a manual kill.

### 1.5 Cost of the remedy, and of the check

`scripts/kx.sh norm_ss` — `d_rmsnorm` verbatim against a scale-safe two-pass
(`block_max`, divide out, reduce the scaled squares, put the scale back), at GLM-5.3's own RmsNorm
shapes and workgroup counts. Ratios are >1 = faster than the shipped body.

| shape (grid, instances/token) | shipped | `PLOW_NORM_RANGE_CHECK=1` | scale-safe two-pass |
|---|---:|---:|---:|
| `glm_dec_qa` rows=1 feat=2048 (b=1, x78) | 1.158 us | 0.908 | 0.680 |
| `glm_dec_x` rows=1 feat=6144 (b=1, x2) | 1.530 us | 0.930 | 0.745 |
| `glm_pf_x` feat=6144 (b=304, x157) | 22.141 us | 0.981 | 0.870 |
| **per-token total over those instances** | **3.570 ms** | 3.646 ms (+2.1%) | **4.134 ms (+15.8%)** |

ISA, whole probe kernel: shipped 699 instructions, armed check 760 (+61), scale-safe 824 (+125,
and a second `block_max` + barrier per row). All three are bit-identical on the harness's data;
the two-pass is not bit-identical in general and would need its own identity campaign.

**Decision: no arithmetic change.** Paying 15.8% of the norm budget on every token of every layer
to buy a margin that is already 1.7e16 x is not a trade this evidence supports. What landed is the
contract, written down at the top of `op_norm.h` with these numbers, and the default-off assertion
that can prove it on demand.

### 1.6 What did NOT land, and why

* **Scale-safe normalization by default** — §1.5.
* **Two-pass centered variance in `d_layernorm_bias`** — §1.3: the error it fixes is measured at
  zero for the only shape that exists.
* **`fmaxf(var, 0.0f)` before the `rsqrt`** — one VALU, but it guards a negative variance that the
  simulation says is unreachable at `feat = 128`, and it would move every object's hash. If a wider
  LayerNorm shape ever ships, revisit both this and the two-pass together.

## Reproducing

```sh
# activation ranges (GLM-5.3 needs 4 GPUs, Gemma 1)
PLOW_DUMP_ACT="act.x:/tmp/x,act.xmid:/tmp/xmid,act.xnext:/tmp/xnext" \
  plowrt amd-bench --blob <blob>/model.pkt --hsaco <obj> --checkpoint <blob>/checkpoint \
                   --tp 4 --prompt <ids> --steps 4
# the all-layer tripwire
PLOW_DECODE_BATCH=4 PLOW_NORM_RANGE_CHECK=1 PLOW_NORM_SS_MAX=1e12f \
  scripts/build_gfx942.sh <objdir>
# body-level cost, seconds, 1 GPU
scripts/kx.sh norm_ss --isa
```

`PLOW_DUMP_ACT` used to be honoured only on the TP path, so every single-GPU model on this box —
Gemma-4 among them — dumped nothing and the caller got an empty range report rather than an error.
Both `amd-bench` closures honour it now.

## Blast radius

Full 45-object gfx942 build at `PLOW_DECODE_BATCH=4`, disassembled with `llvm-objdump` and
compared instruction-for-instruction against a build from the unmodified tree:
**all 45 ISA-identical, 0 differing.** `rn_ss` is an identity when `PLOW_NORM_RANGE_CHECK=0`, and
that is the check rather than the claim. `asm_audit.py --contract` PASSes unchanged over all 45,
so `scripts/obj_baseline_gfx942.json` is untouched — no `--bless`.

Greedy token streams therefore cannot move, and do not: base objects vs changed objects, 3 prompts
x 3 contexts x 64 tokens, full id streams compared — Gemma-4 31B 9/9 identical (128 / 1024 / 8192),
GLM-5.3 TP4 9/9 identical (128 / 2048 / 8192, all four ranks token-identical each step).
