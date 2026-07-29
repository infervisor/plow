# GLM-5.2 TP4 decode — where the gate stall actually goes

**Measured 2026-07-28** on `worktree-readme-build-instructions` HEAD (`e8003a0`), 4× gfx950
(MI355X), real weights (`/home/lava/models/GLM-5.2-plow`), ctx 1024, `--sweep 1024 --steps 65`.
Objects built from that source tree (`i_base.elf` 289496 B, byte-size-identical to
`build-amd/hsaco-abi144/interp_decode.elf`). Blob re-emitted by the same tree and **md5-identical
to `glm52_tp4_64k.pkt` / `glm52_a/c1.pkt`** — so the control is provably the shipping program.

`gpulease -n 4`, no contention warning on any run quoted.

**§0-BENCH.** Every number here is from the C harness, an EXPERIMENT instrument. Nothing in this
file may be placed next to a vLLM number.

New instruments, all checked in:
* `runtime/tests/glm52_decode.c` — `PLOW_TRACE_ALLRANKS=1` traces **every** rank, not just rank 0.
* `scripts/glm52_stall_attrib.py` — attributes the spin (`t_ready − t_arrive`) that
  `glm52_token_attrib.py` reports as one number and never breaks down.
* `runtime/amd/op_collective.h` — `-DPLOW_XR_NOWAIT=1`, a `PLOW_CHAIN_BYPASS`-style ceiling knob
  that deletes the cross-rank rendezvous (numerically wrong, timing real).

---

## 0. THE HEADLINE, AND A CORRECTION TO THE PREMISE

**The 48% gate stall no longer exists. It was 16.77 ms against a 34.68 ms token on the
pre-`6495efc` interpreter; on HEAD it is 8.77 ms against 28.76 ms — 30.5%.** The
wave-cooperative `MLA_MERGE_FOLD` rewrite took ~8 ms of stall out with it, because
`MLA_MERGE_FOLD`'s effective width went 28.3 → 129.6 and it stopped starving 228 CUs for 111 µs
78 times per token. Anyone still budgeting against 16.77 ms is budgeting against a dead object.

| per-CU decomposition | old fold (34.68 ms) | **HEAD (28.76 ms)** |
|---|--:|--:|
| body | 17.090 | **17.633** |
| **gate stall** | **16.772 (48.4%)** | **8.768 (30.5%)** |
| gap / launch | 0.819 | 2.363 |

## 1. THE ATTRIBUTION — the three candidates, separated

The sweep classifies every microsecond of spin by **how many of the 256 workgroups were executing
a packet body at that instant**. A workgroup at a closed gate is waiting either because nothing is
ready anywhere (chain bubble), because the thing that *is* running cannot use it (starvation), or
because everything is busy and its own input genuinely is not done (queueing). Only the last is a
gating question at all.

| CUs executing a body | ms/CU | % of stall | what it is |
|---|--:|--:|---|
| **0 — true idle** | 1.653 | **18.9%** | serial-chain bubble, nothing ready on the whole chip |
| **1** | 3.045 | **34.7%** | 1-workgroup spine op (1.918) + last straggler of a wide op (1.128) |
| 2–4 | 0.664 | 7.6% | |
| 5–32 | 1.833 | 20.9% | the 32-CU expert slices, `MoeCombine` |
| 33–128 | 1.293 | 14.7% | |
| 129–224 | 0.182 | 2.1% | |
| **225–256 — chip full** | **0.098** | **1.1%** | the only bucket a finer gate could ever address |

> **97% of the gate stall happens with ≤128 of 256 CUs doing anything, and 61% with ≤1.**

### The answer to the question that was asked

| candidate | size | % of the 8.77 ms stall | % of the 28.76 ms token | how measured |
|---|--:|--:|--:|---|
| **1. cross-rank arrival skew** | **≤0.68 ms** hard ceiling<br>(0.06 ms suggested, 0.94 ms loose bound) | **none of it** — the wait is inside the packet BODY | **≤2.4%** | `PLOW_XR_NOWAIT` deletes the rendezvous; in-packet reconstruction on **all four ranks**; inter-collective work-window spread |
| **2. narrow / one-workgroup ops** | **6.87 ms** | **78.4%** | **23.9%** | occupancy sweep, buckets 1…128 |
| **2b. true chain bubbles** (same cause, other side) | 1.65 ms | 18.9% | 5.7% | occupancy bucket 0 |
| **3. genuine local gate cost** | **0.098 ms** | **1.1%** | **0.34%** | occupancy bucket 225–256 |

**Candidate 3 — the only thing `Dep::Fine` was ever going to address — is 1.1% of the stall and
0.34% of the token.** That is an independent kill, and it does not rest on the collapse theorem at
all: there is essentially no moment in a GLM decode token when a workgroup is blocked *and* the
machine is saturated. A finer gate can only let a workgroup start earlier; if 224 CUs are already
idle, starting earlier buys nothing.

### Who causes it, and who suffers it

Two complementary views of the same 8.77 ms. `blame` charges the spin to the op that **was
running**; `blocked` charges it to the op the spinning workgroup **was waiting to run**.

| op | blame (ms/CU) | %stall | eff wg | | op | blocked (ms/CU) | %stall |
|---|--:|--:|--:|---|---|--:|--:|
| *idle — nothing live* | 1.653 | 18.9% | — | | `Gemv` | 3.187 | 36.3% |
| **`RmsNorm` ×313** | **1.219** | **13.9%** | **1.0** | | `GemvQkv` | 1.684 | 19.2% |
| `MlaMergeFold` ×78 | 1.003 | 11.4% | 129.6 | | `XReduce` | 1.127 | 12.9% |
| `GemvGlu` ×75 | 0.679 | 7.7% | 201.4 | | `HeadNormRope` | 0.766 | 8.7% |
| `GemvQkv` ×156 | 0.659 | 7.5% | 196.7 | | `FlashMlaDecode` | 0.626 | 7.1% |
| `Gemv` ×229 | 0.577 | 6.6% | 197.3 | | `MlaMergeFold` | 0.571 | 6.5% |
| **`Residual` ×156** | **0.575** | **6.6%** | **1.0** | | `MoeCombine` | 0.329 | 3.7% |
| `XReduce` ×156 | 0.462 | 5.3% | 172.7 | | `MoeExpertDown` | 0.233 | 2.7% |
| `HeadNormRope` ×156 | 0.400 | 4.6% | 162.9 | | **`RmsNorm`** | **0.001** | **0.0%** |
| **`MoeRouterTopk` ×75** | **0.241** | **2.8%** | **1.0** | | **`Residual`** | **0.001** | **0.0%** |

The 1-workgroup spine (`RmsNorm` + `Residual` + `MoeRouterTopk`, **2.035 ms/CU = 23.2% of the
stall**) is a pure cause: those packets never wait for anything themselves (0.001 ms/CU each) and
every other CU waits for them. The 256-wide GEMVs are pure victims: 4.87 ms/CU of the stall is a
GEMV workgroup waiting in front of a packet whose producer used one CU.

**The 1-CU bucket is not all spine, and that matters.** 3.046 ms/CU splits **1.918 (63%)
intrinsically-1-workgroup packets** and **1.128 (37%) the last straggler of a 256-wide packet** —
`GemvGlu` alone contributes 0.544 ms/CU of tail. Widening fixes the first; only the §6a straggler
tail explains the second, and §6b-i is the standing warning that widening *worsens* it.

---

## 2. CANDIDATE 1 IS DEAD, AND THE COLLECTIVE'S COST IS NOT WHERE ANYONE LOOKED

`d_xreduce_mega` waits for its peers **inside the packet body**, between `t_ready` and `t_end`. So
cross-rank skew is not part of the 8.77 ms stall at all — it is counted as *body*. Reconstructed
per collective on **all four ranks** (the earlier rank-0-only trace could not see this: tracing
makes the traced rank the systematic straggler and drives its measured peer-wait to ~0 **by
construction**):

| quantity, median over 156 collectives | rank 0 | rank 1 | rank 2 | rank 3 |
|---|--:|--:|--:|--:|
| peer-wait on the critical path, per token (ms) | 0.058 | 0.056 | 0.053 | 0.072 |
| per-collective peer-wait, median (µs) | 0.12 | 0.10 | 0.11 | 0.12 |
| `t_ready` spread across the packet's 256 wgs (µs) | 1.31 | 1.44 | 1.40 | 1.35 |
| per-workgroup **body** (µs) | 10.82 | 9.78 | 9.57 | 9.41 |

**By the time a rank announces, its peers have already announced.**

**Where that estimator is weak, stated rather than hidden.** It reconstructs the gate-open instant
as `min(t_end) − min(body)`, which degenerates toward `min(t_ready)` when *every* workgroup of the
packet arrives together — which they do (spread 1.3 µs). So read the 0.06 ms as *suggestive*, not
as the bound. The bound comes from the other two instruments, and they agree with it:

* **Inter-collective work windows**, which need no estimator and no cross-device clock alignment:
  over the 155 windows between consecutive collectives, `Σ max_r W − Σ mean_r W =`
  **0.943 ms/token (3.2%)**. Loose (each `W` also absorbs that rank's own wait), but independent.
* **The straggler is uniformly distributed** — rank counts 46 / 35 / 40 / 34, leader
  40 / 36 / 41 / 38. **The skew is jitter, not imbalance.** There is no slow rank to fix and no
  work to rebalance, which is the part that matters for what to build.

**The hard ceiling confirms it.** `-DPLOW_XR_NOWAIT=1` removes the rendezvous entirely — same
packets, same workgroups, same fabric traffic, same graph, only the *waiting* deleted:

| arm | sweep median (ms) | Δ vs its neighbouring control |
|---|--:|--:|
| base1 (all 4 ranks traced) | 28.786 | — |
| **`PLOW_XR_NOWAIT`** | **28.328** | **−0.683** vs base2, −0.458 vs base1 |
| base2 (control, drifting up) | 29.011 | — |

**Deleting the entire cross-rank rendezvous — protocol, skew and all — is worth ≤0.68 ms of a
28.8 ms token (2.4%).** `PLOW_XR_NOWAIT` removes skew and protocol together and so cannot separate
them; what it does settle is that their SUM is 0.68 ms, which caps candidate 1 whatever the split.
`Dep::Fine` per-slice gating of the collectives cannot reach even that: the rendezvous is inside
the body, behind the gate.

Per-workgroup body, baseline vs no-rendezvous, gives the split by subtraction:
**10.82 µs → 4.31 µs**, so the rendezvous is ~6.5 µs of body per workgroup per collective and the
4.31 µs that survives — in workgroups that compute nothing at all — is the system-scope acquire
fence. Note the body spread survives too (3.78 / 4.31 / 12.57 µs with no poll whatsoever), so the
5.5 µs exit spread is contention on the collective's own memory traffic, **not** 256 workgroups
being released one at a time by a late peer.

### But the collective IS expensive, for a reason nobody had named

`XReduce` is now the **#3 line in the entire token: 2.724 ms, 9.5%, 17.46 µs/packet** — against a
bare TP4 all-reduce of **0.626 µs**, a **28× premium**. It rose from 7.0% to 9.5% purely because
the fold rewrite shrank everything around it.

The premium is neither the fabric nor the peers. It is the dispatch:

```
n = 6144 elements,  PLOW_THREADS = 512,  blocks = 256
d_xreduce:  base = slice*512 + tid ;  base < n  only for slice < 12
```

**244 of 256 workgroups do ZERO arithmetic in every one of the 156 collectives** — they arrive,
poll a single system-scope counter line, take a system-scope acquire fence (a full L1/L2
invalidate on all 8 XCDs), and leave. Measured on rank 0:

| | median body (µs) | per token |
|---|--:|--:|
| slices 0–11 — do the reduce | 15.11 | 30.4 CU-ms |
| **slices 12–255 — compute nothing** | **10.82** | **455.7 CU-ms = 1.78 ms/CU** |

All 256 arrive within 1.3 µs and leave **5.5 µs apart** — they are not waiting for peers (0.1 µs),
they are serialising on one contended system-scope cache line. The `PLOW_XR_NOWAIT` arm removes the
poll and the body falls **10.82 → 4.31 µs**; the 3.8 µs that remains, in workgroups that compute
nothing at all, is the system-scope acquire fence.

**A token pays 156 × 256 = 39,936 system-scope L2 invalidates to reduce 6144 elements.**

### This is knob-contract §4's recurring bug shape, and the fix already shipped on the other emitter

| emitter | default `xr_cus` |
|---|---|
| `crates/devgen/src/lib.rs:1931` (dense — Gemma, Qwen, Llama) | **32**, with the measured win in the comment: 256 → 32 was **11.74 → 10.93 ms** at TP4 |
| **`crates/devgen/src/mla.rs:3138` (GLM full model)** | **`all` = 256** — honours `PLOW_XR_CUS` only when set explicitly |

*An arm exists, is correct, and nothing routes to it.* The change is documented bit-identical in
the dense emitter's own comment ("each element's sum still runs over the same N peer slots in the
same order; only the element→workgroup partition changes").

---

## 3. THE STALL IS NOT A POOL OF RECOVERABLE TIME

Before ranking anything: **8.77 ms/CU of spin is not 8.77 ms of prize.** Making a 3.7 µs
1-workgroup packet free returns 3.7 µs to the token, not 255 × 3.7 µs. The `blame` column says
*who causes the idleness*; the prize is the offender's **own** contribution to the token. Those
are different by a factor of 256/width, and confusing them is how "the gate stall is essentially
the whole gap" became a premise.

| offender | blame (ms/CU) | **own ms of the token** | reachable? |
|---|--:|--:|---|
| `XReduce` ×156 | 0.462 | **2.724** | yes — 244/256 workgroups do nothing (§2) |
| `RmsNorm` ×313 (1 wg) | 1.219 | **1.382** | partly — needs a cross-wg reduction, §7a-CHAIN's L3 shape |
| `MoeRouterTopk` ×75 (1 wg) | 0.241 | 0.596 | unexamined |
| `Residual` ×156 (1 wg) | 0.575 | **0.577** | yes — pure elementwise, no reduction |
| *chain bubbles* | 1.653 | — | §7a-CHAIN priced removing 44% of the chain at 0.35 ms |

**The addressable pool behind the whole 8.77 ms is ~5.3 ms of op-time, of which the collective is
by far the largest single piece.** The remaining 3.5 ms of blame belongs to the straggler tails of
256-wide GEMVs (§6a's ±19%), which no scheduling change touches.

Updated §7b census for GLM on HEAD (was 66% / 45.5% pre-rewrite):

| effective width | pkts | %pkts | ms | %token |
|---|--:|--:|--:|--:|
| ≤4 | 545 | 19.8% | 2.563 | 8.9% |
| ≤32 | 1745 | 63.3% | 7.383 | **25.7%** |
| ≤128 | 1834 | 66.5% | 9.199 | 32.0% |

## 4. WHAT THIS SAYS TO DO — measured, not projected

Every arm below is a devblob emitted by the same `plowc` as the control, run on the same
interpreter object, in one lease with the control **interleaved** (§6b-STALE), `--gen 24` on every
arm against the control's ids.

**The control blob is md5 `aba55146…` = `glm52_tp4_64k.pkt` = `glm52_a/c1.pkt`, and re-emitting it
from this tree *before* the `emit_xreduce` change reproduced that md5 exactly.** That is what makes
the deltas trustworthy: every env knob is provably inert when unset, and the blob format has not
drifted since the campaign's other GLM measurements.

**Lease 2 — six runs, control at positions 1 / 4 / 6: 28.862 / 28.955 / 28.901 — mean 28.906,
sd 0.047.** That is a tighter control than any GLM A/B in this campaign so far, so the deltas below
are not resting on drift. **All six arms produced the SAME 24 generated ids**, and 0 cross-rank
disagreements.

| # | arm | what it changes | ms/token | **Δ vs control** | ids |
|--:|---|---|--:|--:|---|
| 1 | control | — | 28.862 | — | ref |
| 2 | **`PLOW_XR_CUS=32`** | all 156 collectives 256 → 32 workgroups | **27.367** | **−1.539 (−5.3%)** | identical |
| 3 | `GLM_SPINE_CUS=32` | 156 `Residual` packets 1 → 32 workgroups | 28.728 | −0.178 | identical |
| 4 | control | — | 28.955 | — | ref |
| 5 | **both** | both of the above | **27.098** | **−1.808 (−6.3%)** | identical |
| 6 | control | — | 28.901 | — | ref |

**The collective's dispatch width is worth 1.54 ms of a 28.9 ms token, bit-identical, from one
number in the emitter.** The two arms compose (−1.539 + −0.178 = −1.717 against a measured
−1.808, i.e. additive within the control's own spread).

**Lease 3 — the shipping form, and it is BETTER than the flat 32.** `xrfit` is the
`emit_xreduce` sizing fix of §4.2 with **no environment knob set at all**; GLM decode lands on
`ceil(6144/512) = 12` workgroups:

This lease drifted **downward** (control 28.964 → 28.639), where lease 2 drifted up, so every row
carries **both** the mean-to-mean and the local-neighbour delta and no conclusion rests on the
difference (the `df8bc14` convention):

| arm | ms/token | Δ mean-to-mean | Δ interpolated | ids |
|---|--:|--:|--:|---|
| control (pos 1) | 28.964 | — | — | ref |
| **`xrfit` — sized to `ceil(elems/512)` = 12 wgs** | **27.041** | **−1.761** | **−1.815 (−6.3%)** | **identical** |
| `xrfit` + `PLOW_GLM_FUSE_B1=1` | 26.759 | −2.043 | −1.988 | **DIVERGES at index 3** |
| control (pos 4) | 28.639 | — | — | ref |

**`xrfit` beats the flat 32 by ~0.3 ms**, which is the whole argument for sizing over a constant:
the dense emitter's 32 was itself 2.7× wider than this reduction can use, and it was only ever a
guess that happened to be far better than 256. Call the shipped change **−1.8 ms, −6.3%,
token-identical**.

**Cross-lease reproducibility of the control: 28.862 / 28.955 / 28.901 (lease 2) and 28.964 /
28.639 (lease 3)** — five independent 4-minute loads of the same blob spanning 0.33 ms, which is
the honest noise floor for this harness and is 5× smaller than the effect.

### 4.0a The after-picture, and the reason "gate stall" is not the objective function

A separate traced run on the fixed blob (taken **after** the timing A/B, never inside it — the
trace store costs ~1.6%):

| | before | after |
|---|--:|--:|
| traced step | 28.764 | **27.213** |
| `XReduce` — ms of token | 2.724 (#3 line) | **1.158** |
| `XReduce` — µs/packet | 17.46 | **7.42** |
| `XReduce` — dispatched / effective wgs | 256 / 172.7 | **12 / 9.6** |
| `XReduce` — body CU-time | 1.898 ms/CU | **0.044 ms/CU** (43× less) |
| per-workgroup body (median / max) | 10.82 / 17.07 µs | **4.33 / 6.41 µs** |
| per-CU body | 17.633 | 16.045 |
| **per-CU gate stall** | **8.768** | **9.250 — UP** |

**The gate stall went UP by 0.48 ms while the token went DOWN by 1.55 ms.** That is not a paradox,
it is the clearest single statement of what this whole census found: the 244 idle workgroups were
always idle, they were just *inside the packet body* being counted as work, and narrowing the
dispatch moves them into the honestly-labelled stall column. **Optimising the gate-stall number
would have argued against the change that made the token 6% faster.** Use the token.

### 4.0b `PLOW_GLM_FUSE_B1` — real, small, and it FAILS the identity gate. Do not ship on this.

Merging the 1-workgroup `(Residual, RmsNorm)` pair into one `AddNorm` packet is §7a-CHAIN's
*serial producer→consumer* merge, not §7a's concurrent-sibling fusion: critical path **1400 →
1322** packets, 1-workgroup spine **391 → 313**. It is worth a further **−0.282 ms** on top of
`xrfit` — and it changes the token:

```
control  264 5777 9125 1948 279 15742 ...
b1       264 5777 9125 48376 990 315  ...   <- diverges at index 3
```

The emitter's own comment predicted this (`d_add_norm` reduces over the **un-rounded** `a+b` where
the split path norms the bf16-rounded `xmid`), so it is algebraically defensible and numerically a
different program. **0.28 ms does not buy a full HF-coherence gate**, which is what the fold
rewrite needed for the same reason. Left off by default; the number is recorded so nobody has to
spend a lease re-deriving it.

**`GLM_SPINE_CUS` is real but small, and it is the third confirmation of §6b-i.** Widening a
1-workgroup elementwise producer 1 → 32 recovers only 0.178 of the 0.577 ms the op costs, because
its consumer now waits on a max over 32 stragglers instead of 1. The dense emitter's own comment
predicted exactly this ("One workgroup (512 threads × 8) covers it"). **Not shipped as a default**
— it stays an opt-in ceiling instrument.

### 4.1 The dense emitter already knows all of this. The MLA emitter is a separate code path.

`crates/devgen/src/lib.rs:1950` carries a comment headed *"Elementwise ops sized to their ACTUAL
work, not handed the whole machine"*, and implements
`elem = |n| (0..n.div_ceil(512*8).max(1).min(n_cu))` — for hidden 6144 that is **2** workgroups.
It also defaults `PLOW_XR_CUS` to **32** with the measured 11.74 → 10.93 ms in the comment.

`crates/devgen/src/mla.rs` — the GLM/MLA path — has **neither**. It hard-codes `one = vec![0u32]`
for every spine op and `all` (256) for every collective. The sizing discipline was written down
once, on the emitter GLM does not use.

### 4.2 What shipped, and why it is not `PLOW_XR_CUS=32`

A flat 32 is the *dense emitter's* answer and it is wrong in both directions: 32 is still 2.7×
more than GLM decode's 12, and it would be a **cap on prefill**, where `xr_elems = t·hidden` and
the collective genuinely wants the whole machine. `PLOW_XR_CUS` is also shared between the decode
and prefill emits in both emitters, so raising or lowering it trades one phase against the other —
and TTFT is the #1 GLM item (§6g-FINAL), so silently narrowing prefill is not acceptable.

The change landed in `emit_xreduce` (`crates/devgen/src/lib.rs`), which already receives
`xr_elems`, as a **pure narrowing of whatever the caller allowed**:

```rust
let need = (xr_elems.div_ceil(512).max(1) as usize).min(xr_cus.len());
let xr_cus = &xr_cus[..need];
```

`ceil(xr_elems/512)` is the saturation point of both collective bodies (the one-shot grid-strides
the full `n` by `nblk·PLOW_THREADS`; the two-shot's all-gather does the same and its
reduce-scatter phase saturates earlier still at `n/nranks`). Consequences:

* GLM decode: 256 → **12** workgroups, automatically, no env knob.
* GLM/dense **prefill: unchanged** — `t·hidden` asks for more than 256 and still gets 256.
* Dense decode: the existing `PLOW_XR_CUS=32` default narrows further to `ceil(5376/512) = 11`.
* Fixes **both emitters and every model** rather than papering over one default.
* Bit-identical: same N peer slots, same order, only the element→workgroup partition changes.

**It is an EMIT-TIME change, so every already-built `.pkt` still carries the old width.** Anything
served from a stale blob — `plowrt`'s asset dirs, `glm52_tp4_64k.pkt`, the `glm52_a/*.pkt` A/B set
— keeps the 256-workgroup collective until it is re-emitted. Re-emit before quoting a TPOT number
against this change.

### 4.3 What is NOT the answer

* **`Dep::Fine` on the collectives.** Dead twice: the Lean `CounterGranularity.collapse` theorem
  kills it a priori, and the occupancy census kills it empirically — 1.1% of the stall happens
  with the chip full, which is the only condition under which starting a workgroup earlier helps.
  It is also aimed at the wrong half of the packet: the collective's cost is in its **body**.
* **Reducing rank skew.** There is nothing to reduce. The straggler is uniformly distributed over
  the four ranks and the whole rendezvous has a 0.68 ms ceiling.
* **`PLOW_GLM_FUSE_B1`** at 0.28 ms — see §4.0b, it changes the token.
* **Widening `RmsNorm`.** `crates/devgen/src/lib.rs:1957` and the comment above
  `d_norm_residual_norm` in `runtime/amd/op_norm.h` already record the measurement: a feature
  axis was added and it loses at every `k`, because the two extra counter-gated packets cost more
  than the whole op (§7a-CHAIN's L3, +1.28 ms). GLM's `RmsNorm` is the same shape.
* **Op-count reduction generally** (§6g-KNOBS: `GLM_GROUP=1` removed 38% of the ops and lost
  2.88 ms). The lever here is *width*, not *count* — `xrfit` removes zero ops.

---

## 5. THE `PLOW_NO_XREDUCE` DISCREPANCY, RESOLVED

§6e-0 could not reconcile two numbers: the collectives are "24.6 µs each" from the 39× premium,
which should imply ~10 ms, yet deleting all 156 saves only 3.84 ms. The 24.6 µs was **circular** —
it was 3.84 ms ÷ 156. There was never an independent 10 ms.

What the trace says instead, on HEAD: the `XReduce` packets are worth **2.724 ms** of the token
(9.5%), of which

| | ms/token |
|---|--:|
| irreducible: 12 workgroups reducing 4 × 6144 bf16 | ~0.10 |
| the rendezvous poll, paid by all 256 workgroups | ~0.68 (`PLOW_XR_NOWAIT` ceiling) |
| the system-scope acquire fence, paid by all 256 | remainder |
| **actually waiting for a peer** | **0.06** |

and `PLOW_NO_XREDUCE` also deletes 75 `Residual` packets and shortens the chain, which is the rest
of its 3.84 ms. The two instruments were never in conflict; one prices *the collective plus its
graph*, the other prices *the packet*.

---

## 6. REPRODUCTION

```bash
# objects + the patched harness (ROCm tooling OUTSIDE nix; Rust INSIDE)
bash /home/lava/models/glm52_skew/build.sh <worktree>     # i_base.elf, i_nowait.elf, glm52_decode

# blobs (control is md5-identical to glm52_tp4_64k.pkt when no knob is set)
nix develop -c bash /home/lava/models/glm52_skew/emit2.sh

# attribution run: ALL FOUR RANKS traced (rank-0-only drives peer-wait to 0 by construction)
gpulease -n 4 glm-stall sg render -c '
  cd /home/lava/models/glm52_skew &&
  PLOW_TRACE_RAW=tr/base1 PLOW_TRACE_ALLRANKS=1 PLOW_INTERP=i_base.elf \
    ./glm52_decode xr_base.pkt /home/lava/models/GLM-5.2-plow --tp 4 --sweep 1024 --steps 65 --gen 24'

python3 scripts/glm52_stall_attrib.py tr/base1.insts.txt \
        tr/base1.rk{0,1,2,3}.tp4.ctx1024.bin --traced-ms 28.764
python3 scripts/glm52_token_attrib.py tr/base1.insts.txt \
        tr/base1.rk0.tp4.ctx1024.bin --tp 4 --traced-ms 28.764
```

**Trap worth adding to §0a: `--gen` now composes with `--sweep` in one process** (the old §0a note
that they are mutually exclusive is stale — `1523ffc` fixed it), so token identity and timing come
out of the same 4-minute weight load.

Raw output of all three leases (14 sweeps): `perf-data/glm52-gate-stall-ab-raw.txt`. Driver:
`perf-data/glm52_stall_ab.sh`.

**Tile store.** This branch edits `runtime/amd/op_collective.h`, which moves the preprocessed build
digest and stales every tuning record — `cargo test -p devgen --test tuned_tile_selection` is the
only thing that catches it (§6g-STALE-2). The campaign was folded into lease 3's own cards:
`scripts/rebench_tune_gemm.sh` → 180 rows → `tunedb-gemm ingest` → 90 qualified records at
`gfx950-14811518192412b8`, and all four tests are green again.

**And the one that would have wasted this lease:** `PLOW_TRACE_ALLRANKS` did not exist. Every
GLM trace before today was rank-0-only, and rank 0 carries the trace store's per-(workgroup,
packet) write — which makes it the systematic straggler and drives its *measured* cross-rank
peer-wait toward zero **whatever the truth is**. A rank-0 trace cannot answer a cross-rank
question. It took four traced ranks to show the peer-wait really is ~0.1 µs.

---

# 7. THE COLLECTIVES, RE-PRICED ON THE FIXED BLOB (2026-07-28, later the same day)

Everything above §4 was measured against the **256-workgroup** collective. `xrfit` shipped, so
every collective number in this campaign — including the 3.84 ms of knob-contract §6e-0 — is now
stale. Re-derived from scratch on the CURRENT tree
(`worktree-readme-build-instructions` HEAD `f84ac5f`, which is 4 commits past §0's `e8003a0` and
whose `interp.hip` differs), objects rebuilt from it, control blob re-emitted and **md5
`e818c91b…` = `/home/lava/models/glm52_tp/glm52_tp4_64k.pkt`**, `op 24 blocks=12` × 156 confirmed
in the sidecar. `gpulease -n 4`, no contention on any run quoted. Raw:
`perf-data/glm52-xreduce-repricing-raw-lease{1,2}.txt`.

## 7.1 What the 156 collectives cost now

**Lease 1**, controls interleaved at 1 / 3 / 5 — **26.798 / 26.996 / 26.805, mean 26.866,
sd 0.112.** `noxr` = `PLOW_NO_XREDUCE=1`, md5 `c44cce03…`, which is **byte-identical to the blob
that produced the 3.84 ms**, so only the control moved:

| arm | ms/token | Δ vs interpolated control | ids |
|---|--:|--:|---|
| base (1 / 3 / 5) | 26.798 / 26.996 / 26.805 | — | ref, 0 cross-rank disagreements |
| `PLOW_NO_XREDUCE` (2) | 25.040 | **−1.857** | garbage (expected), 72 disagreements |
| `PLOW_NO_XREDUCE` (4) | 24.984 | **−1.917** | garbage, 72 disagreements |

| | old (256-wg collective) | **now (12-wg collective)** |
|---|--:|--:|
| token | 34.149 | **26.866** |
| all 156 collectives, by `PLOW_NO_XREDUCE` | 3.84 ms (11.2%) | **1.887 ms (7.0%)** |
| per collective, same (circular) derivation | 24.6 µs | **12.1 µs** |

**And the ÷156 is still circular, so here is the independent number.** `PLOW_NO_XREDUCE` does not
delete 156 packets, it deletes **231**: 2756 → 2525 ops = the 156 `XReduce` **plus 75 `Residual`**
(the MoE/dense tail's `no_xr` branch combines straight onto the residual, `mla.rs:2976`/`:3087`).
The trace prices those separately, and the two instruments close:

| | ms/token | µs/packet | source |
|---|--:|--:|---|
| `XReduce` × 156 | **1.321** (4.9%) | **8.47** | trace, `glm52_token_attrib.py` |
| `Residual` × 75 of 156 | 0.295 | 3.93 | trace |
| chain shortening + secondary | ~0.27 | — | 1.887 − 1.616 |
| **`PLOW_NO_XREDUCE` total** | **1.887** | — | A/B |

`XReduce` fell from the **#3 line of the token (2.724 ms, 9.5%, 17.46 µs/pkt)** to **#10
(1.321 ms, 4.9%, 8.47 µs/pkt)**, 12 dispatched / 10.0 effective workgroups.

## 7.2 Where the remaining 1.32 ms of collective sits — and why NO kernel can reach it

**Lease 2**, same blob every arm, only the interpreter's `d_xreduce`/rendezvous changing, so
nothing here is confounded by the graph the way base-vs-noxr is. Controls at 1 / 4 / 6 —
**26.795 / 27.038 / 27.055** (lease drifting up; every Δ is against the interpolated neighbour).

| arm | what it deletes | ms/token | **Δ** | ids |
|---|---|--:|--:|---|
| base (1) | — | 26.795 | — | ref |
| `-DPLOW_XR_NOWAIT=1` (2) | the whole cross-rank rendezvous | 26.529 | **−0.347** | garbage (expected) |
| **`-DPLOW_XR_NOREDUCE=1` (3)** | **the whole reduce body** | 26.931 | **−0.026** | garbage (expected) |
| base (4) | — | 27.038 | — | ref |
| **`-DPLOW_XR_VEC=1` (5)** | nothing — b128 `d_xreduce` | **27.427** | **+0.380 SLOWER** | **identical 24/24, 0 disagreements** |
| base (6) | — | 27.055 | — | ref |

| the 1.321 ms `XReduce` line splits as | ms/token | % of the line | reachable by a kernel? |
|---|--:|--:|---|
| system-scope acquire fence + packet dispatch/gate overhead | ~0.948 | 72% | no — protocol and packet count |
| cross-rank rendezvous (poll + wait + skew), `XR_NOWAIT` | 0.347 | 26% | no — protocol |
| **the ENTIRE reduce: 4 peer loads, 3 adds, 1 store, `XR_NOREDUCE`** | **0.026** | **2%** | this is the whole target |

> **Deleting every load, every add and every store `d_xreduce` performs — the absolute ceiling on
> any rewrite of that kernel, assembly or otherwise — is worth 0.026 ms of a 26.9 ms token
> (0.10%).** The trace agrees independently: reduce body **0.052 ms/CU**, local reduce median
> **5.49 µs** against a bare TP4 all-reduce of 0.626 µs, i.e. the premium is 28× and **none of it
> is arithmetic**.

**The b128 vectorisation was built and measured anyway, and it LOSES.** `d_xreduce` with
`global_load_dwordx4` (8 bf16/lane/rank, 4 loads in flight, `+4 global_load_dwordx4` confirmed in
the disassembly, tail path for non-divisible `n`): **+0.380 ms, token-identical over 24 ids.** It
is bit-identical and still slower, for the reason §6b-i keeps finding: at `nblk=12` the vector form
gives all the work to 1.5 workgroups instead of 12, so the packet's **last straggler lands later**.
Even the emitter change that would size the grid to `ceil(n/(512·VEC)) = 2` cannot rescue it —
`XR_NOREDUCE` caps the whole exercise at 0.026 ms.

**Why the premise was wrong, stated plainly:** a TP4 all-reduce of hidden=6144 moves
4 × 6144 × 2 = **49 KB**, which at the measured 6200 GB/s is **8 ns**. Memory-level parallelism is
a lever when a kernel is bandwidth-starved; this one is 49 KB behind a system-scope acquire fence.
The one element/thread that looks like a latency-hiding defect is **one HBM round trip either
way** — vectorising changes how many lanes wait through it, not how long it is.

## 7.3 The occupancy census is unchanged by the fix

Rank-0 trace on the fixed blob, `traced-ms 27.170`: per-CU **body 15.901 | gate stall 9.336 |
gap/launch 1.932**. The stall went 8.768 → 9.336 while the token went 28.76 → 26.87, which is
§4.0a's lesson repeating: **the token is the metric.**

| CUs in a body | §1 (256-wg blob) ms/CU | **now** ms/CU | % of stall |
|---|--:|--:|--:|
| 0 — true idle | 1.653 | 1.594 | 17.1% |
| 1 | 3.045 | 3.169 | 33.9% |
| 2–32 | 2.497 | 2.944 | 31.6% |
| 33–224 | 1.475 | 1.531 | 16.4% |
| **225–256 — chip full** | **0.098** | **0.098** | **1.0%** |

**The chip-full bucket is 0.098 ms/CU in both censuses**, so §4.3's kill for `Dep::Fine` on the
collectives survives the fix untouched. `XReduce` is now the **largest single blame op**
(1.266 ms/CU, 13.6%, ahead of `RmsNorm`'s 1.226) — but §3 applies: the prize is its **own** 1.321
ms, of which 0.026 is arithmetic.

## 7.4 What this says to do

* **Do not touch `d_xreduce`.** Ceiling 0.026 ms, measured candidate +0.380 ms.
* **Do not touch the rendezvous.** 0.347 ms, and it is correctness.
* The collective's residue is the **system-scope acquire fence × 12 workgroups × 156 packets**
  (~0.95 ms). The only levers on it are *fewer collectives* or *a weaker fence*, both of which are
  protocol changes with a correctness argument attached, not kernel work.
* The §7b census still says the money is in **1-workgroup packets (2.015 ms/CU of the 1-CU
  bucket)** and in `MlaMergeFold`/`GemvQkv`/`Gemv` straggler tails — `RmsNorm` 313 pkts / 1.411 ms
  and `MoeRouterTopk` 75 pkts / 0.629 ms at effective width **1.0** are the next L6-shaped items,
  and a sibling lease measured `GLM_ROPE_FIT` (`HeadNormRope` 256 → 2) at **25.591 vs a 26.768
  control** on this same blob.

**Method note for whoever re-runs this.** The two body instruments are `-DPLOW_XR_NOREDUCE=1` and
`-DPLOW_XR_VEC=1`, and they live in a **scratch copy** of `runtime/` at
`/home/lava/models/glm52_xr/rt` — deliberately NOT in the repo, because editing
`runtime/amd/op_collective.h` moves the preprocessed build digest and stales every tuned GEMM
record (§6g-STALE-2). The scratch tree with no instrument set reproduces `i_base.elf` to 43 bytes,
all of them build-id/source-path hex — identical code.


