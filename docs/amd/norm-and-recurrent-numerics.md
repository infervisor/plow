# Norm range and recurrent-gate conditioning on gfx942

Findings **F5** and **F6** of `plans/gpu-kernel-static-audit-20260908.md`, measured. Both were
static-analysis findings about *reachable* numerical failure; this is what the hardware and the
shipped checkpoints actually do.

**Outcome in one line each.** F5 needs no arithmetic change: the shipped bodies run 16-17 orders of
magnitude below the overflow point on both checkpoints, the audit's cancellation example is
computed *exactly* by the code the compiler emits, and the remedy costs 13-32%. F6 needed one: the
KDA softplus lost its whole tail, the loss compounds over a recurrence rather than perturbing one
output, and the fix is +14 instructions in `d_kda_gate`.

Everything below is on MI300X / gfx942, ROCm 7.14 from the flake, blobs
`build-glm53/tp4-long` (GLM-5.3 TP4, 93 layers) and `build-gemma31/assets-final-plain`
(Gemma-4 31B). No KDA checkpoint exists on this box — see §2.1.

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

---

## 2. F6 — the recurrent softplus

`runtime/amd/op_kda.h:111`, the `PLOW_KDA_GATE_SOFTPLUS` (unbounded) branch:

```
g[t,h,d] = -exp(A_log[h]) * softplus(g_raw[t,h,d] + dt_bias[h,d])       (log2 units, prefix-summed)
```

`softplus` was `__logf(1.0f + __expf(x))`, matching `[fla]`'s `tl.log(1 + tl.exp(x))`.

### 2.1 What breaks, and why it is not an ordinary rounding difference

FP32 has 24 significant bits, so `1.0f + e` rounds to **exactly 1.0f** for every `e < 2^-25`, i.e.
for every `x <= -16.6355`. The expression then returns exactly zero, the derived per-step decay is
`exp(-A*0) = 1`, and the gate stops forgetting. Just *above* the cliff the same expression is wrong
the other way: `1 + e` quantizes to `1 + 2^-23`, so the per-step decay is **2x too large**.

| x | true softplus | `__logf(1+__expf(x))` | rel err |
|---:|---:|---:|---:|
| -20 | 2.06115e-09 | **0** | 1.0 |
| -17 | 4.13994e-08 | **0** | 1.0 |
| -16.6355 | 5.96047e-08 | **0** | 1.0 |
| -16 | 1.12535e-07 | 1.19209e-07 | 5.93e-2 |
| -12 | 6.14419e-06 | 6.19886e-06 | 8.90e-3 |
| -10 | 4.53989e-05 | 4.54177e-05 | 4.14e-4 |

Over a persistent recurrence that is a missing operation repeated per token, not a perturbed
output. At the checkpoint-maximum `A = exp(A_log) = 11.776` (`docs/kimi-k3-kda.md` §3.1), the
fraction of state a channel should still hold:

| s | 8 192 steps | 131 072 steps | 1 048 576 steps (K3 `max_position_embeddings`) |
|---:|---|---|---|
| -17 | 0.99601 vs **1.00000** | 0.93810 vs **1.00000** | 0.59977 vs **1.00000** |
| -18 | 0.99853 vs **1.00000** | 0.97677 vs **1.00000** | 0.82856 vs **1.00000** |
| -20 | 0.99980 vs **1.00000** | 0.99682 vs **1.00000** | 0.97487 vs **1.00000** |
| -25 | 0.99999 vs **1.00000** | 0.99998 vs **1.00000** | 0.99983 vs **1.00000** |

**Scope, stated plainly.** The only KDA checkpoint that exists — Kimi-K3 — sets
`gate_lower_bound = -5.0`, and `crates/devgen/src/kda.rs` selects the gate mode from
`gate_lower_bound.is_some()`, so K3 takes `PLOW_KDA_GATE_LOWER_BOUND` and never calls this
function. The softplus branch serves a checkpoint without that field (Kimi-Linear-era `fla`
defaults). **No such checkpoint is on this box, and there is no K3 snapshot here either**, so the
`s <= -17` region is a contract, not a sampled observation: reaching it needs
`g_raw <= -9.1` given the measured `dt_bias >= -7.894`, and `g_raw` is an unmeasured projection
output.

### 2.2 Compatibility versus stability

Bit-compatibility with `[fla]` was **already not on the table**: this body calls `__logf`/`__expf`,
the fast intrinsics, where Triton's `tl.log`/`tl.exp` are the precise libdevice calls. The middle
region does not match `[fla]` bit-for-bit today and never did. What the guard preserves is the
*formula*, and the change preserves it too.

Against that: the sibling recurrent gate in this tree already made the opposite call and wrote down
why. `op_qwen_gdn.h:36`: *"TRANSCENDENTALS ARE THE PRECISE ONES. `expf`/`log1pf`, not `__expf` —
these bodies are memory bound, the gate feeds a multiplicative recurrence over a PERSISTENT state
where a relative error compounds across steps."* Same structure, same argument, opposite choice —
and KDA is the one whose formulation loses the tail entirely.

**Decision: take the stable branch.**

```c
if (x > 20.0f) return x;                       // [fla]'s guard, unchanged
const float e = __expf(x);
return x < -8.0f ? e * (1.0f - 0.5f * e)       // log(1+e) = e - e^2/2 + O(e^3)
                 : __logf(1.0f + e);           // [fla]'s expression, unchanged
```

`-8` is where the two are equal: below it the series' relative error is `e^2/3 <= 2^-24`, above it
the `1 + e` form is better, and at the switch point the series is already the closer of the two, so
the join has no step of its own. `PLOW_KDA_SOFTPLUS_FLA_COMPAT=1` restores the previous expression
exactly.

**Not** `log1pf(expf(x))`: correct in the tail too, and §2.4 prices it.

### 2.3 Oracle

`runtime/tests/kda_step_cdna3_test.hip` gates the BT64 chunk gate prefix against an f64 host
reference. Its existing case draws `dt_bias` from the K3-measured `[-7,-1]`, which cannot see this:
the accumulated prefix error stays under 7e-5 either way. A **tail case** was added at
`dt_bias in [-26,-20]` with its own bar; the bar follows the build, because the compat form cannot
pass a tight one by construction.

```
                                    FLA_COMPAT=1        FLA_COMPAT=0 (shipped)
chunk gate bounded    (K3's path)   1.859e-05           1.859e-05        unchanged
chunk gate softplus   (K3 range)    6.822e-05           6.999e-05        bar 2e-4, both pass
chunk gate softplus tail            2.833e-06           9.326e-13        3.0e6 x closer
```

Everything else in the file — BC16/BT64 intra, the W/U transform, the 4-chunk carry, prefill and
batched-decode state walks, the gated norm, the packed spans — is unchanged and PASSes in both.

The file also did not **build** on gfx942 before this: `k_intra_cached` asks for 114,688 B of LDS
against the part's 65,536 B limit, and the compiler refuses it, which made the only f64 oracle for
the KDA gate unusable on the arch its filename names. It is now compiled only where it fits and the
host skips the arm when the symbol is absent. (Its `KDA3_BENCH` timing path still assumes the arm;
that is a CDNA4 bench and was left alone.)

### 2.4 Cost

`scripts/kx.sh kda_gate --isa` — three formulations, three code objects, one source, with a device
**f64** reference so the error column is distance from the mathematics rather than from the
incumbent. The output is `exp2(accumulated log2 gate)`, i.e. the retained-state fraction, so max
error is directly "state kept that should have decayed".

| arm | ISA total | VALU | SALU | max err, 64 steps | max err, 8192 steps |
|---|---:|---:|---:|---:|---:|
| `fla` `__logf(1+__expf(x))` | 141 | 80 | 56 | 4.447e-05 | **5.644e-03** |
| `branch` (shipped now) | 203 | 83 | 115 | 4.888e-06 | **9.030e-06** |
| `log1pf(expf(x))` (`qwen_softplus`) | 525 | 191 | 329 | 2.384e-07 | 3.874e-06 |

The `fla` error grows 127 x from 64 to 8192 steps — it is a drift, linear in sequence length. The
branch's grows 1.85 x: ordinary rounding. At 8192 steps the branch is **625 x** closer to the
mathematics; `log1pf` buys a further 2.3 x for 2.6 x the instructions.

Timing (model grid, ratios >1 = faster than `fla`): branch 0.822 / 0.806, `log1pf` 0.294 / 0.270.
That 20% is on a probe whose entire body is the softplus. In the real op it is **+14 instructions
in `d_kda_gate` (538 -> 552, +2.6%)** and +10 in `d_kda_state_step_g`, +56 across all of
`interp_decode_k3.elf` (150 707 -> 150 763, +0.037%), with no VGPR, LDS, occupancy or spill change
— `asm_audit.py --contract` passes unchanged over all 45 objects.

### 2.5 What did NOT land

The nested `exp(A_log) * softplus(s)` is still a product of two separately-rounded FP32
intermediates, so extreme finite parameters — the audit's `A_log = 100, s = -100`, whose
mathematical product is ~1 — still overflow or underflow one factor. The measured K3 range is
`exp(A_log) in [0.471, 11.776]`, i.e. `A_log in [-0.75, 2.47]`. The contract the body now asserts
in its header is `|A_log| <= 80` with the product finite, not the general case; closing it means
the audit's 190-instruction nested precise gate, for parameters no checkpoint ships.

---

## Reproducing

```sh
# activation ranges (GLM-5.3 needs 4 GPUs, Gemma 1)
PLOW_DUMP_ACT="act.x:/tmp/x,act.xmid:/tmp/xmid,act.xnext:/tmp/xnext" \
  plowrt amd-bench --blob <blob>/model.pkt --hsaco <obj> --checkpoint <blob>/checkpoint \
                   --tp 4 --prompt <ids> --steps 4
# the all-layer tripwire
PLOW_DECODE_BATCH=4 PLOW_NORM_RANGE_CHECK=1 PLOW_NORM_SS_MAX=1e12f \
  scripts/build_gfx942.sh <objdir>
# body-level costs and errors, seconds each, 1 GPU
scripts/kx.sh norm_ss  --isa
scripts/kx.sh kda_gate --isa
# F6 oracle, both formulations
hipcc --offload-arch=gfx942 -O3 -w -DKDA3_DEVICE [-DPLOW_KDA_SOFTPLUS_FLA_COMPAT=1] --genco \
      runtime/tests/kda_step_cdna3_test.hip -o /tmp/kda3.co -Iruntime/amd -Iruntime/common
g++  -O2 -w -std=c++17 -x c++ -D__HIP_PLATFORM_AMD__=1 [-DPLOW_KDA_SOFTPLUS_FLA_COMPAT=1] \
     -I/opt/rocm-7.2.4/include runtime/tests/kda_step_cdna3_test.hip -o /tmp/kda3 \
     -L/opt/rocm-7.2.4/lib -lamdhip64
perf-data/tools/gpulease -n 1 kda3 /tmp/kda3 /tmp/kda3.co
```

The host half needs the SYSTEM `g++` and `/opt/rocm-7.2.4`, the way `scripts/kx.sh` builds its
driver: the flake's `clang++` links a libstdc++ that wants a newer glibc than this image has.

`PLOW_DUMP_ACT` used to be honoured only on the TP path, so every single-GPU model on this box —
Gemma-4 among them — dumped nothing and the caller got an empty range report rather than an error.
Both `amd-bench` closures honour it now.

## Blast radius

Full 45-object gfx942 builds at `PLOW_DECODE_BATCH=4`, disassembled with `llvm-objdump` and
compared instruction-for-instruction against a build from the unmodified tree:

| tree | ISA-identical | differing |
|---|---:|---|
| F5 only (`PLOW_KDA_SOFTPLUS_FLA_COMPAT=1`) | **45 / 45** | none |
| F5 + F6 | 31 / 45 | 14, every one a `*_k3*` object |

`rn_ss` is an identity when `PLOW_NORM_RANGE_CHECK=0`, and the first row is the check rather than
the claim. `asm_audit.py --contract` PASSes unchanged over all 45 in both, so
`scripts/obj_baseline_gfx942.json` is untouched — no `--bless`.

**GLM-5.3 TP4 loads `interp_prefill_fp8_mla_moe_gq` / `interp_decode_gq`; Gemma-4 31B loads
`interp_prefill_gq` / `interp_decode_gq`. All of those are in the ISA-identical set** — neither
model's served arithmetic can move. Greedy token streams, base objects vs changed objects, full id
streams compared:

| model | contexts | cells | result |
|---|---|---|---|
| Gemma-4 31B | 128 / 1024 / 8192, 3 prompts, 64 tokens | 9 | 9/9 identical |
| GLM-5.3 TP4 | 128 / 2048 / 8192, 3 prompts, 64 tokens | 9 | 9/9 identical, all 4 ranks token-identical each step |
