# GLM-5.2 MoE tail — two emitter changes, A/B'd separately (2026-07-28)

> **⚠️ SUPERSEDED (2026-07-29): `GLM_LINEAR_FP8` is a WIN, not a regression.** The `+0.39 ms`
> figure on this page was measured on an UNSTACKED blob. `perf-data/glm52-linear-fp8-reeval.md`
> re-measures it stacked at **−0.417 ± 0.175 ms, n=6 — 97 % of the −0.431 ms floor** (commit
> `b3f77fd`). Every "do not ship" verdict for this knob below is void; the other knobs on the
> page are unaffected.

**§0-BENCH.** Nothing here may be placed next to a vLLM number: these are plow-internal
device-side decode measurements, not a served comparison.

Two changes were proposed against the GLM-5.2 TP4 decode MoE tail. They are in the same code, so
one agent did both — but they are independent and are reported separately.

| change | verdict |
|---|---|
| **1. shared gate/up off `DenseGluFp8Blk` (47)** | BUILT (emitter-only, no new kernel, no weight change) and **MEASURED AT −0.017 ms = nothing.** Opt-in, not shipped. §1, §3. |
| **2. remove one of the two collectives per layer (vLLM K3 §3a)** | **NOT EXPRESSIBLE, and the premise is wrong** — vLLM's change ADDS a collective to remove compute, and GLM has neither the up-projection nor a spare collective. §2. No GPU spent. |

**The finding worth acting on is neither of them:** `GLM_LINEAR_FP8` measures **−0.44 ms** against
the current interpreter where it is on record as **+0.39 ms**, a 0.83 ms swing on an unchanged
knob. §3.1.

---

## 1. `GLM_SHARED_GLU_SPLIT` — the shared gate/up leaves op 47

### 1.1 The claim under test, split into its reliable and speculative halves

`GLM_LINEAR_FP8=1` is a **+0.39 ms regression** (`glm52-decode-emitter-abs.md` §2), and the culprit
is **not** op 44 (`o_proj`, shared down) but op 47 (`DenseGluFp8Blk`, the shared gate/up).
Isolated, GLM TP4 shape `(imoe_l=512, K=H=6144)`, 6200 GB/s denominator
(`perf-data/glm52_gemvblk_bench.hip`):

| shared gate/up | us | % of ceiling |
|---|--:|--:|
| bf16 `GemvGlu` (19), shipped today | 3.26 | 62.2 |
| fp8 `DenseGluFp8Blk` (47) | **7.17** | 14.2 |

**Half the weight bytes, 2.2x the wall time.** Over 75 sparse layers that is +0.29 ms of body
arithmetic on its own, which is most of the +0.39 ms the whole knob costs.

### 1.2 Mechanism — it is the WORK WALK, not the bytes and not the precision

`d_dense_glu_fp8_blk` (`runtime/amd/op_moe.h:748`) walks

```
for (n = slice*PLOW_WAVES + wave; n < N; n += nblk*PLOW_WAVES)
```

At `N = imoe_l = 512` over `nblk = 256` workgroups and `PLOW_WAVES = 8`, the stride is 2048 > N, so
only slices `0..63` ever get an output: **192 of 256 workgroups run empty**. The 512 waves that do
run use `wave_dot_fp8_blk`, which keeps **one load in flight** per wave and reads `x` from
**global**.

`gemv_rows_fp8_blk` (op 44, `op_gemm.h:1805`) instead splits `gv_per = ceil(N/nblk)` columns to
EVERY workgroup, keeps `UN` chunks in flight per column (`UN=3` at `nchunk=6`, i.e. K=6144) and
stages `x` in LDS. Same wave count, ~4x the CU spread and ~3x the in-flight weight bytes.

This is why the sibling agent's op-44 tail patch did not help op 47: **they are different kernels**,
and the patch (`op_gemm.h:1819`, the odd-column duplicate) only touches op 44.

### 1.3 What was built — and why NO weight concatenation was needed

The obvious fix is to concatenate gate and up into one `[2*imoe_l, H]` block-fp8 matrix and issue a
single op 44 at `N = 1024`. That is legal (`imoe_l = 512` is a multiple of 128, so the two `[16,48]`
`weight_scale_inv` grids stack into a legal `[32,48]` one) — but it needs a new tensor on disk in
`scripts/glm52_prep_fp8_linear.py` **and** a two-piece column shard in BOTH hosts, because rank `r`
needs rows `[r*512,(r+1)*512) ∪ [2048 + r*512, 2048 + (r+1)*512)` and `slice_for`'s column predicate
takes one contiguous row range.

**The concatenation's benefit is memory parallelism, and the CU set delivers it for free.** One
`N=1024` packet over 256 CUs is 1024 waves at 4 waves/CU. Two `N=512` packets on **disjoint CU
halves** are also 1024 waves at 4 waves/CU, running at the same time, because each half owns
`nblk = 128` and both are gated only on the post-attention norm. (Two `N=512` packets on the SAME
256 CUs would NOT do this: a workgroup walks the packet stream in order, so it would run one after
the other at 2 waves/CU.)

So the emitted arm is:

```
gate:  GemvFp8Blk (44)  shared_cus[..half]   xn2 -> shfu      N=imoe_l K=H   (w.shg, w.shg_s)
up:    GemvFp8Blk (44)  shared_cus[half..]   xn2 -> shfu_up   N=imoe_l K=H   (w.shu, w.shu_s)
glu:   Glu        (5)   1 CU                 silu(shfu)*shfu_up -> shfu
down:  GemvFp8Blk (44)  shared_cus           (unchanged)
```

Everything is an opcode the decode object already dispatches (`Glu` is in the unconditional switch,
`interp.hip:806`, and in `GFX950_DISPATCHED`). **No kernel change, no `.co`/`.elf` rebuild, no prep
change, no host change.** It is the same unfusing `emit_glm_moe_prefill_block` already does on the
MXFP4 arm, which is where `act.shfu_up` comes from.

Code: `crates/devgen/src/mla.rs` — `glm_shared_glu_split`, `glm_glu_halves`, the `c_shglu` arm.
Knob: default ON under `GLM_LINEAR_FP8=1`; `GLM_SHARED_GLU_SPLIT=0` restores op 47.

### 1.4 Structural verification from the blob (CPU, no GPU)

`scripts/glm52_glusplit_ab.sh` emits the three arms; `plowrt`'s `graphstat` example reads them.

| arm | ops | stream entries |
|---|--:|--:|
| `base` (bf16 `GemvGlu`) | 2756 | 297 569 |
| `linfp8_old` (op 47) | 2756 | 297 569 |
| `linfp8_split` (2x op 44 + `Glu`) | **2906** | **297 644** |

`+150 ops` = 75 sparse layers x (3 packets replacing 1). `+75 entries` = 75 x
`(128 + 128 + 1) - 256` — which is the **proof the halves are 128 CUs each and the `Glu` is one
workgroup**. Had the halves both been emitted on all 256 CUs the delta would have been `+75 x 257`.

`glm_glu_halves` is unit-tested for disjointness and total coverage
(`glm_shared_glu_halves_are_disjoint_and_total`): overlap is silent — the numbers stay right and
only the concurrency disappears — so the invariant is pinned in a test rather than in a comment.

### 1.5 What to believe about the projection, and what not to

`perf-data/glm52-weight-stream-split.md` prices the whole `GLM_LINEAR_FP8` byte saving at
**−0.431 ms of FLOOR**, and floor is a lower bound on time. The op-44 tail patch measured
**1.8-2.0x on the isolated kernel and 0.2% on the served token** — an isolated-kernel win is a
hypothesis about the token, and that one was falsified at the endpoint. Split the claim:

* **Reliable:** op 47 costs 0.538 ms of body arithmetic where bf16 costs 0.245 (75 layers x the
  isolated us above). Removing that **+0.29 ms penalty** is near-certain, because it is a
  *removal of a known cost*, not a projected gain.
* **Speculative:** that the remaining fp8 byte saving then materialises at the token. It did not for
  op 44 on the baseline blob. It may differ here because under `GLM_LINEAR_FP8` op 44 additionally
  carries `o_proj`, the shared down AND now the shared gate/up — but **that is reasoning, not a
  measurement.**

### 1.6 Bug found while doing this, NOT fixed: `GLM_LINEAR_FP8` is decode-only and nothing says so

Under `GLM_LINEAR_FP8=1`, `w.wo` / `w.shg` / `w.shu` / `w.shd` are re-declared as
`.weight_fp8` tensors at **1 B/elt** (`mla.rs:913-1005`). The DECODE emitter routes all four to
block-fp8 opcodes. The **PREFILL** emitter does not look at `lin_fp8` at all:

* `emit_glm_mla_prefill` — `gemm(b, n.og_tp, n.oat, w.wo, w.wo_s, …)` (`mla.rs:1943`)
* `emit_glm_block_prefill` — `GemmGlu` on `w.shg`/`w.shu` and `gemm(… w.shd …)` (`mla.rs:2039-2114`)

Those are bf16 `Gemm`/`GemmGlu` arms reading a tensor declared at **half** its bf16 byte size, so a
STACKED blob (`PLOW_MLA_PREFILL=…` + `GLM_LINEAR_FP8=1`) would read fp8 bytes as bf16 and run off
the end of every one of those tensors. It has never been hit because `GLM_LINEAR_FP8` has only ever
been measured on a decode-only packet — which is exactly §4's bug shape with the polarity reversed:
*the weight was swapped under an arm that was never told.* `scripts/rebench_emit_glm.sh` does not
set the knob, so nothing shipped is affected.

The fix is either a prefill block-fp8 arm (`GemmFp8Blk` does not exist) or a hard refusal when
`GLM_LINEAR_FP8` is set on a packet with prefill buckets. Left for the owner of that knob; recorded
rather than silently patched.

### 1.7 Measured

See §3.

---

## 2. The MoE-tail collective restructuring — the premise does not hold for GLM-5.2

### 2.1 vLLM's K3 change does not remove a collective. It ADDS one, to remove COMPUTE.

Read the source paragraph again (`docs/amd/kimi-k3-vllm-day0.md` §3a, quoting
<https://vllm.ai/blog/2026-07-27-k3>):

> "In the normal TP case, this requires **two all-reduces** on the routed and shared experts — or
> one all-reduce with concatenation — **and replicates the up-projection**. **To avoid redundant
> compute in the replicated linear projection**, vLLM instead performs reduce-scatter on the shared
> experts and keeps all-reduce on the routed experts ... Finally, the results are **all-gathered**
> onto each rank."

Count the phases:

| | collectives per LatentMoE tail |
|---|--:|
| vLLM baseline | 2 (two all-reduces) |
| **vLLM optimised** | **3** (reduce-scatter + all-reduce + all-gather) |

The `~20% off that step, ~7-8% end-to-end` comes from deleting **three quarters of a replicated
up-projection**, paid for with an *extra* collective. It is a compute lever, not a collective-count
lever. The brief's framing ("this removes one") is a misreading of the post.

### 2.2 GLM-5.2 has no up-projection in the MoE tail, so there is nothing to trade

`config.json`: `GlmMoeDsaForCausalLM`, `n_routed_experts=256`, `num_experts_per_tok=8`,
`n_shared_experts=1`, `moe_intermediate_size=2048`, `hidden_size=6144`. This is the
DeepSeek-V3-shaped MoE: every expert's `down_proj` writes **hidden** directly. There is no latent
expert space, no post-reduction RMSNorm, and no shared up-projection between the expert output and
the residual add. K3's LatentMoE has all three; that is the entire premise of vLLM's change.

### 2.3 plow already emits the alternative vLLM names and rejects

vLLM's own baseline lists "**or one all-reduce with concatenation**" and discards it because it
still replicates the up-projection. GLM has no up-projection, so concatenation is free — **and it is
what plow already does**. `crates/devgen/src/mla.rs:2880` (`emit_glm_moe_block`):

```
MoeCombine  shared partial + Σ_k gate_k · expert_k partial, residual = zero_h  -> dg_tp
XReduce     one-shot all-reduce of dg_tp                                       -> attn
Residual    attn + xmid                                                        -> x_out
```

One collective for shared **and** routed together. plow is already at the configuration vLLM's
optimisation is trying to approximate, one collective *cheaper* than vLLM's optimised path.

### 2.4 GLM's two collectives per layer are both structurally forced

`GRAPHSTAT_CP=1` confirms 156 collectives, all on the critical path, priced by `PLOW_NO_XREDUCE=1`
at **3.84 ms = 24.6 us each** (§6e-0). They sit after the only two **row-parallel** ops in the
layer:

| # | producer | why the reduction cannot be dropped |
|---|---|---|
| 1 | `o_proj` (head-sharded input) | feeds `AddNorm` -> `xn2`; the router, the shared gate/up and every expert gate/up read the WHOLE `xn2` |
| 2 | MoE combine (`imoe_l`-sharded input) | feeds `Residual` -> `RmsNorm` -> next layer's `q_a_proj` / `kv_a_latent`, all of which read the WHOLE vector |

With a **replicated residual stream**, each row-parallel op costs exactly one all-reduce, and GLM
has exactly two. That is already minimal.

The obvious escape — keep the residual stream **sharded** over H, all-reduce only the scalar
sum-of-squares for each RMSNorm, and all-gather before each column-parallel consumer — trades 2
one-shot rendezvous for **5** (AG(x) + RS(o_proj) + tiny-AR + AG(xn2) + RS(MoE)). At a
latency-bound 24.6 us per rendezvous on a 12 KB message that is **+3.7 ms/token**, not a saving.

The other escape — all-gather `oat` (8 KB) so `o_proj` can run **column-parallel** and need no
reduction — replaces one collective with two (the gather before, and a gather of the sharded
`xmid` after, because RMSNorm's consumers need all of H). Also worse.

### 2.5 The GLM-analogous lever — "redundant compute in a replicated projection" — is priced and DEAD

GLM does have replicated projections, and they are not small. Per rank per token, TP4, from the
checkpoint headers (`perf-data/glm52-weight-stream-split.md`):

| replicated tensor | MB/layer | x layers | MB/token |
|---|--:|--:|--:|
| `q_a_proj` `[2048, 6144]` | 24.00 | 78 | 1872.0 |
| `derived.kv_a_latent` `[512, 6144]` | 6.00 | 78 | 468.0 |
| `derived.k_rope` `[64, 6144]` | 0.75 | 78 | 58.5 |
| `mlp.gate` router `[256, 6144]` | 3.00 | 75 | 225.0 |
| **total** | | | **2623.5** |

Sharding all four 4-way removes 3/4 x 2623.5 MB = **1.97 GB = 0.317 ms of floor**. The cost:

* `q_a_proj` sharded -> `q_a_layernorm` needs the global sum of squares over `ql=2048`, and
  `q_absorb`/`q_rope` read the WHOLE `qlat`. Needs an all-gather (`qlat‖ckvraw‖krr` = 2624 elts,
  5.25 KB — it can carry `kv_a_latent` and `k_rope` in the same message).
* the router sharded -> `MoeRouterTopk` must select from all 256 logits and every rank must select
  the SAME 8. Needs a second collective, at a different point in the serial chain.

**+2 collectives/layer = +156 = +3.84 ms** (it doubles the collective bill) to save 0.317 ms.
Even the best case — gather only the q/kv/rope chain, leave the router replicated — is
**+1.92 ms for −0.29 ms**. Dead by 6.6x, on plow's own measured 24.6 us/collective.

### 2.6 If the collective work is ever revisited, this is the trap

A sibling agent found and fixed a live instance of it today: `count_xgates`
(`crates/plowrt/src/exec/amd_tp.rs:456`) never counted `XArgmaxFin`'s two ids, so the sharded-head
fold **wrote past the end of `xctr` with no fault, and the four ranks sampled four different
tokens**. Any change to the number or kind of collectives must update `count_xgates` in the same
commit. `emit_xreduce` (`crates/devgen/src/lib.rs:1713`) consumes **one** gate id at decode
(one-shot) and **two** at prefill (`XReduceTwoShot` = reduce-scatter + all-gather), and it is the
only place that increments `xgate` — a new collective that does not go through it is a silent
cross-GPU buffer overrun.

**Conclusion: change 2 cannot be expressed as a win for GLM-5.2 without a new op, and would not be
a win with one.** No GPU was spent on it.

---

## 3. Measured — the split buys NOTHING, and `GLM_LINEAR_FP8` is no longer a regression

GLM-5.2, TP4, 4x gfx950, real weights (`/home/lava/models/GLM-5.2-plow-q`), `plowrt amd-bench
--tp 4`, ctx 1024, 65 steps, `GLM_MOE_CORESIDENT=1`. Three arms **interleaved inside one lease**,
three folds, own contemporaneous control (§6b-STALE). Objects: `build-amd/hsaco-abi144` +
`interp_prefill_mla{,_gq}.elf`. Blobs from `scripts/glm52_glusplit_ab.sh`, run by
`scripts/glm52_glusplit_run.sh`.

| fold | `base` (bf16) | `linfp8_old` (op 47) | `linfp8_split` (2x op 44 + Glu) |
|---|--:|--:|--:|
| 1 | 30.625 † | 29.504 | 29.493 |
| 2 | 29.964 | 29.524 | 29.513 |
| 3 | 29.915 | 29.551 | 29.521 |
| **median** | **29.964** | **29.524** | **29.513** |

† fold 1 started with a foreign 203 GB/card process still resident (`gpulease` WARN
`foreign-before`, all four leased cards). It cleared during the run. Fold 1's `base` is the only
number that moved; the two fp8 arms in the same fold match folds 2-3 to 0.03 ms, so the contention
is visible in exactly one cell and is excluded from the medians below. Every one of the nine runs
reported **all 4 ranks token-identical on every step**.

**Change 1 (the split): −0.017 ms, i.e. nothing.** Consistent in sign across all three folds
(−0.011, −0.011, −0.030), and about 20x smaller than the 0.36 ms the isolated kernels project
(7.17 → ~2.4 us x 75 layers).

**The isolated ratio did not survive contact with the interpreter — for the third time.** Op 44's
odd-column patch measured 1.8-2.0x isolated and 0.2% at the token; `GemvFp8Blk` on `o_proj` + the
shared expert measured −0.05 ms pre-fold-rewrite and +0.31 ms post; and now a 3.8x kernel ratio
measures −0.017 ms. The rule these three share:

> **A packet body shrinking inside a window bounded by its neighbours' gates returns nothing.**
> Op 47's 192 empty workgroups are not idle *time* — they reach the packet, find no column, bump
> the counter and move on. The 64 that do work set the pace, and the split's 2x128 busy workgroups
> each still pay the same poll/bump. The isolated bench measures ONLY the body, and at 2756
> packets/token the body is not what the token is made of (§7: 21% gate stall on Gemma-4-31B, and
> GLM has 2.4x the packets).

**Verdict: not shipped, kept as an opt-in instrument.** `GLM_SHARED_GLU_SPLIT=1` opts in;
`GLM_LINEAR_FP8=1` alone keeps op 47. Two reasons rather than one: no measurable gain, **and** the
split rounds one extra time — op 47 keeps `g` and `u` in f32 registers through the SwiGLU
(`fg[n] = f2bf(moe_act(g,act) * u)`), whereas the split writes both halves to bf16 and re-reads
them in `Glu`. With nothing on the other side of the ledger there is no reason to pay it.

### 3.1 The result that DOES matter, and it is not the one under test

**`GLM_LINEAR_FP8` measures −0.44 ms here, not the +0.39 ms on record.**

| arm | median ms/token | vs `base` |
|---|--:|--:|
| `base` (bf16 shared expert + bf16 `o_proj`) | 29.964 | — |
| `GLM_LINEAR_FP8=1` | **29.524** | **−0.440** |
| `GLM_LINEAR_FP8=1 GLM_SHARED_GLU_SPLIT=1` | 29.513 | −0.451 |

That is a **0.83 ms swing** from the recorded verdict, on the same knob, same weights, same
hardware — the interpreter underneath it has changed. §6b-STALE says exactly this and it is now
demonstrated a second time in the opposite direction: a knob KILLED for costing +0.39 ms is worth
−0.44 ms against the current object. `scripts/rebench_emit_glm.sh` still says *"NOT set:
`GLM_LINEAR_FP8` (measured +0.39 ms, a regression)"* — **that comment is stale and the knob should
be re-evaluated for the shipping blob**, with the §1.6 prefill hazard fixed first if the blob is
stacked.

Note the whole −0.44 ms is the `o_proj` + shared-down + gate/up byte saving arriving at the token,
which is 102% of the −0.431 ms of floor `glm52-weight-stream-split.md` predicted. The floor
prediction was right; the earlier measurement of it was made on a different machine-state.

### 3.2 Token streams

`--prompt 100,264,6722,315,9822,374`, 24 greedy steps, all 4 ranks token-identical every step in
every arm:

```
base    5777 9125 1948  279 15742  315  458 3766  323  279 1196   13 ...
old     5777 9125 48376  990   315 1045 1290   13 1096  374  264 1140 ...
split   5777 9125 1948 1196   323 3150  315 3162   11 5310 3457 53707 ...
```

All three fork from each other within 3 tokens and none reproduces §6g's recorded stream past
index 3 — **including the bf16 `base` arm**, which is the control. So this is not evidence about
the change: greedy decode on this checkpoint forks on a near-tie almost immediately, and the branch
has moved under §6g's transcript (§6g already records one such move, `892` -> `429` at index 12).
**Token identity is the wrong gate for a precision change here** — the brief said so, and the right
one is `scripts/glm52_prefill_gate.sh`'s single-block B4 oracle. That gate was NOT run for these
arms, because the change is not being shipped; anyone who does ship `GLM_LINEAR_FP8` must run it.

What the runs DO prove: **every rank agrees on every step in all 12 runs**, so both collectives ran
in every arm. That is the check that a wrong reduction would fail, and it passed everywhere.
