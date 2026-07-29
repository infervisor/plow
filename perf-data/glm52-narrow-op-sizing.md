# GLM-5.2 TP4 decode — splitting the 1-CU bucket, and the two ops that were the XReduce bug again

**Measured 2026-07-28** on `worktree-readme-build-instructions` HEAD (`f84ac5f`), 4× gfx950
(MI355X), real weights (`/home/lava/models/GLM-5.2-plow`), ctx 1024, `--sweep 1024 --steps 65
--gen 24`. Objects built from this tree (`i_base.elf` 301072 B). Direct continuation of
`perf-data/glm52-gate-stall-attribution.md`, which named `MoeCombine` and `HeadNormRope` as "the
next two unexamined" — they are, and both are the same bug that `emit_xreduce` fixed.

**§0-BENCH.** Every number here is from the C harness, an EXPERIMENT instrument. Nothing in this
file may be placed next to a vLLM number.

---

## 0. THE PREMISE HAD MOVED — re-derive on the SHIPPED object, not the report's headline

The attribution report's census (stall 8.768, 1-CU bucket 3.045) was taken **before** `xrfit`
shipped. On the shipped blob the stall is **9.250** — it went UP as the token went DOWN, exactly
as §4.0a warned. Every number below is re-derived from `tr/xrfit`, the post-fix trace, so the
buckets are the ones a change today would actually attack.

---

## 1. THE 1-CU BUCKET, SPLIT — and it is worth splitting

`glm52_stall_attrib.py` §(1c) on the shipped trace:

| tenant of the "exactly 1 CU busy" bucket | ms/CU | share |
|---|--:|--:|
| **intrinsically 1-workgroup packets** | **1.927** | **62.3%** |
| `RmsNorm` ×313 | 1.177 | 38.0% |
| `Residual` ×156 | 0.591 | 19.1% |
| `MoeRouterTopk` ×75 | 0.159 | 5.1% |
| **straggler tail of a WIDE packet** | **1.169** | **37.7%** |
| `GemvGlu` | 0.542 | 17.5% |
| `HeadNormRope` | 0.254 | 8.2% |
| `XReduce` | 0.161 | 5.2% |
| `GemvQkv` / `MlaMergeFold` / `Gemv` / rest | 0.212 | 6.8% |

### The same split in WALL-CLOCK token time, which is the number that can be spent

`ms/CU` is blame, not prize (§3 of the attribution report). Sweeping the timeline for the wall
time during which *exactly one* workgroup is in a body:

| bucket | wall ms | % of the 27.213 ms token |
|---|--:|--:|
| 0 — true idle | 1.589 | 5.8% |
| **1 — total** | **3.168** | **11.6%** |
| — of which 1-workgroup packets | 1.973 | 7.2% |
| — of which straggler tails | 1.195 | 4.4% |
| 2–4 | 0.736 | 2.7% |
| 5–32 | 3.081 | 11.3% |
| 33–128 | 3.337 | 12.3% |
| 129–224 | 1.849 | 6.8% |
| **225–256 — chip full** | **13.452** | **49.4%** |

Two independent methods agree exactly on the narrow pool: this sweep gives
1.589 + 3.168 + 0.736 + 3.081 = **8.574 ms at ≤32 CUs busy**, and
`glm52_token_attrib.py`'s effective-width census gives **8.575 ms (31.5%)**. That is the
number to budget against — not 8.77 "ms/CU" of spin.

### Why the straggler lags: it starts late, it does not run slow

For every wide op, the last-finishing workgroup is **not** a fixed slice and **not** a fixed CU:

| op | pkts | tail µs/pkt | most-frequent last slice | body p50/p99 µs |
|---|--:|--:|---|--:|
| `GemvGlu` | 75 | **10.60** | 191 (4% of packets, uniform 0.4%) | 11.1 / 13.9 |
| `HeadNormRope` | 156 | 3.00 | 1 (26%, uniform 0.4%) | 6.3 / 8.5 |
| `MoeCombine` | 75 | 0.28 | 1 (27%) | 6.3 / 11.1 |
| `GemvQkv` | 156 | 0.49 | 252 (3%) | 17.7 / 26.9 |
| `Gemv` | 229 | 0.29 | 237 (3%) | 11.1 / 23.0 |

**`GemvGlu` is the whole straggler story and it is neither an imbalance nor a memory tail.**
Decomposing each packet's span into arrival spread, start spread and body:

| op | span µs | `t_arrive` spread | `t_ready` (start) spread | body p50 | body max |
|---|--:|--:|--:|--:|--:|
| **`GemvGlu`** | 26.64 | 20.37 | **19.43** | 11.14 | 13.02 |
| `GemvQkv` | 21.40 | 21.44 | **0.34** | 17.79 | 30.05 |
| `Gemv` | 13.82 | 26.28 | **0.36** | 11.06 | 14.66 |
| `HeadNormRope` | 14.24 | 14.18 | **11.00** | 6.37 | 8.29 |

`GemvQkv` and `Gemv` are healthy: their workgroups arrive over ~20 µs but all **start within
0.35 µs**, because they arrive *before* the gate opens and are released together. Their tails are
0.3–0.5 µs.

`GemvGlu` is the opposite: **start spread 19.4 µs on a uniform 11.1 µs body.** Its workgroups
arrive *after* their gate is already open, so each starts when its CU frees up. `GemvGlu` is the
shared expert, and it runs **co-resident with the routed MoE experts** — so its 10.6 µs tail is
the routed experts' tail, observed through it.

> **Consequence, and it kills two obvious proposals:** the straggler half of the 1-CU bucket is
> not fixed by re-slicing `GemvGlu`, by widening it, or by chasing an L2/HBM outlier. The
> workgroups are not slow; they are late, and they are late because of the co-residency
> schedule. Only a change to *that* touches this 1.195 ms.

`HeadNormRope` and `MoeCombine` also show a structural last-slice (slice 1 at 26–27% against a
0.4% uniform — 65× over-represented), and §2 explains it: slice 1 is one of the *only* slices in
those packets that does any work at all.

---

## 2. `HeadNormRope` AND `MoeCombine` ARE `emit_xreduce`'S BUG, ON THE SAME EMITTER

Both are dispatched over all 256 workgroups by `crates/devgen/src/mla.rs`, and both saturate
analytically far below that:

| op | kernel work walk | saturates at | dispatched | pkts/token |
|---|---|--:|--:|--:|
| `HeadNormRope` (q) | `w = slice*PLOW_WAVES + wave; w < ntok*nhead` | `ceil(1·16/8)` = **2** | 256 | 78 |
| `HeadNormRope` (k) | same, shared single head | `ceil(1·1/8)` = **1** | 256 | 78 |
| `MoeCombine` | `gid = slice*PLOW_THREADS + tid; gid < H` | `ceil(6144/512)` = **12** | 256 | 75 |

Confirmed empirically on the shipped trace — per-slice median body is bimodal, and the working
set is exactly the analytic one (2 of 256 for the ropes, 12 of 256 for the combine).

**The estimator that priced XReduce, applied to these.** For each op, the body CU-time burned by
workgroups past the saturation point — they poll the arrival counter, take the acquire fence, and
exit having done zero arithmetic:

| op | waste CU-ms/token | **waste ms/CU** | measured token Δ |
|---|--:|--:|--:|
| `XReduce`, pre-`xrfit` (calibration) | 455.5 | **1.779** | **−1.82 (shipped)** |
| **`HeadNormRope`** | 247.9 | **0.968** | *this document* |
| **`MoeCombine`** | 112.3 | **0.439** | *this document* |
| combined | 360.2 | **1.407** | |

The calibration point is the whole argument for trusting the estimate: on the one op where this
estimator has been checked against a shipped A/B, it predicted 1.78 and the token moved 1.82.

### Two corrections the naive fix would have got wrong

1. **The q and k ropes are CONCURRENT** — all 78 pairs overlap in the trace. Narrowing both onto
   `cus[..need]` puts them on the SAME workgroups, and the interpreter walks the packet stream
   per workgroup, so they would run one after the other. `mla.rs`'s own `glm_glu_halves` comment
   records this trap. They are given **disjoint** slices (`q` = `cus[0..2]`, `k` = `cus[2..3]`).
2. **Prefill needs no arm and gets none.** `ntok·nhead` there exceeds `PLOW_WAVES·len`, so `need`
   saturates at `cus.len()` and the packet is unchanged — the same property that let `xrfit` ship
   without capping TTFT.

Both changes are **pure narrowing and bit-identical**: slices `0..need` own exactly the work items
they own today, and only the empty slices go away.

### Structural verification, before any GPU time

`graphstat`, all four blobs:

| blob | wg-packets (`ents`) | ops | edges | gate polls |
|---|--:|--:|--:|--:|
| `n_ctl` (control) | 259,505 | 2756 | 3589 | 472,753 |
| `n_rope` | 219,803 | 2756 | 3589 | — |
| `n_cmb` | 241,205 | 2756 | 3589 | — |
| `n_both` | 201,503 | 2756 | 3589 | 268,351 |

`259,505 − 78·(254+255) = 219,803` and `219,803 − 75·244 = 201,503` — exact. **`ops` and `edges`
are unchanged in every arm: zero packets added or removed, zero graph change.** This is a WIDTH
change, which is the one lever §4.3 says is live (`GLM_GROUP=1` removed 38% of the ops and lost
2.88 ms; count is not the lever).

**The control blob is md5 `e818c91b…` = `/home/lava/models/glm52_tp/glm52_tp4_64k.pkt`,** the
shipping program byte for byte, so `GLM_ROPE_FIT` and `GLM_COMBINE_FIT` are provably inert unset.

---

## 3. THE A/B — one lease, control interleaved at positions 1 / 4 / 6

Same `plowc`, same interpreter object (`i_base.elf`, built from this tree), same weights, six
sweeps in one lease, no contention warning. `--gen 24` on every arm.

| # | arm | ms/token | **Δ vs control mean** | ids |
|--:|---|--:|--:|---|
| 1 | control | 26.768 | — | ref (§6g) |
| 2 | **`GLM_ROPE_FIT` — 156 ropes 256 → 2 / 1, disjoint** | **25.591** | **−1.148 (−4.3%)** | identical |
| 3 | **`GLM_COMBINE_FIT` — 75 combines 256 → 12** | **26.233** | **−0.506 (−1.9%)** | identical |
| 4 | control | 26.799 | — | ref |
| 5 | **both** | **25.065** | **−1.674 (−6.3%)** | identical |
| 6 | control | 26.650 | — | ref |

**Control: 26.768 / 26.799 / 26.650 — mean 26.739, sd 0.076.** The combined effect is **22× the
control's own spread**, and the controls do not drift monotonically, so no arm is resting on one.

**The two compose additively:** −1.148 + −0.506 = −1.654 predicted against **−1.674 measured**,
a 0.020 ms discrepancy against a 0.076 ms control sd.

**Every arm produced the same 24 generated ids as the control — the §6g GLM decode reference —
with 0 cross-rank disagreements on all four ranks.** That is the identity gate, and it is what
"bit-identical by construction" is supposed to look like when it is true.

### How the estimate held up

| op | predicted from `waste ms/CU` | measured Δ | ratio |
|---|--:|--:|--:|
| `XReduce` (the calibration, shipped as `xrfit`) | 1.779 | 1.82 | 1.02 |
| **`HeadNormRope`** | 0.968 | **1.148** | 1.19 |
| **`MoeCombine`** | 0.439 | **0.506** | 1.15 |

The estimator is a slight **under**-estimate for both — it counts only the body the empty
workgroups burn, and misses that the packet's own span also shrinks once 254 workgroups stop
arriving. Treat `waste ms/CU` as a lower bound on the prize, which makes it a good screen.

### Shipped

Both are now **unconditional in `crates/devgen/src/mla.rs`** (`rope_cus`, `elem_cus`), with no
environment knob — the `emit_xreduce` precedent. Re-emitting with **no knobs set** reproduces the
measured `n_both` blob **md5-identical** (`184b0b4b…`), so what ships is exactly what was measured.

**It is an EMIT-TIME change.** Every already-built `.pkt` — `plowrt`'s asset dirs,
`glm52_tp4_64k.pkt` (still md5 `e818c91b…`), the `glm52_a/*.pkt` A/B set — keeps the 256-workgroup
ropes and combine until re-emitted. **Not re-emitted here on purpose: two other agents are holding
live measurements against those exact blobs.** Re-emit before quoting a TPOT number against this.

Gates: `devgen --lib` 169, `packet --lib` 42, `plowrt --features hsa --lib` 122, `golden_blob` 9,
`--features cuda` compiles. `tuned_tile_selection` fails **on the base branch too** (records stale
against build digest `gfx950-29b635ca5d068435`); this change touches no runtime header and does
not move the digest.

### 3.1 CONTENTION AUDIT — the lease ended `rc=76`, and this is NOT §0a's false positive

`gpulease` flagged `foreign-during` on gpu4–7, and those were **my own leased cards**
(`ACQUIRED gpus=[4 5 6 7]`), so the §0a "known false positive" escape does not apply and the run
cannot be waved through. What settles it is the clock:

| event | time |
|---|---|
| lease acquired | 17:56:59 |
| arms 1–6 (the entire A/B) run | 17:57 → ~18:17:30 |
| traced after-picture starts loading | ~18:17:30 |
| **foreign `plowrt` (PID 3448565) starts** | **18:20:44** |
| trace dumped / lease released | 18:21:14 / 18:21:27 |

**The foreign process appeared 3+ minutes after the last A/B measurement.** Two independent
checks agree: the three interleaved controls do **not** degrade over the lease (26.768 → 26.799 →
**26.650**, the last one the fastest — a creeping contender would have made it the slowest), and
the traced arm at ~18:21 reported 25.049 against `n_both`'s clean 25.065 at ~18:14.

> **Verdict: the six-arm A/B is clean. The traced after-picture overlapped the contender for its
> last ~30 s and its ABSOLUTE TIMING IS NOT QUOTED anywhere in this document.** Its per-op widths,
> µs/packet and effective widths are used, because those are within-run structure rather than
> wall-clock, but nothing here rests on them.

### 3.2 The after-picture (structure only, per §3.1)

| | before (`xrfit`) | after (`n_both`) |
|---|--:|--:|
| `HeadNormRope` — dispatched / effective wgs | 256 / 162.0 | **2 / 1.5** |
| `HeadNormRope` — µs/packet | 14.13 | **6.43** |
| `HeadNormRope` — ms of token | 1.665 | **0.705** |
| `MoeCombine` — dispatched / effective wgs | 256 / 140.9 | **12 / 11.2** |
| `MoeCombine` — µs/packet | 11.76 | **5.92** |
| `MoeCombine` — ms of token | 0.882 | **0.444** |
| wg-packets | 259,505 | **201,503** |
| **per-CU body** | **16.045** | **14.309** |
| per-CU gate stall | 9.250 | **9.366 — UP** |

The two ops' own time falls 1.398 ms and the per-CU body floor falls 1.736 — against a measured
token delta of 1.674, i.e. the whole effect is accounted for by the ops themselves.

**And the gate stall went UP again**, for the third time in this campaign and for the same reason
(§4.0a): the empty workgroups were always idle, they were merely being counted inside the packet
body as work. Anyone optimising the stall number would have rejected this change too. **Use the
token.**

---

## 3b. THE 1-WORKGROUP SPINE IS LATENCY-BOUND, WHICH CAPS `GLM_SPINE_CUS` A PRIORI

The other 62% of the 1-CU bucket is the genuinely-1-workgroup spine, and the question standing
over it is whether `GLM_SPINE_CUS` (measured −0.178 ms for `Residual` 1 → 32) should ship and
whether `RmsNorm` deserves the same. Breaking GLM's 313 `RmsNorm` packets down by size settles it
without a lease:

| elements | packets | µs/packet | ms/token |
|--:|--:|--:|--:|
| 6144 (input / post-attn norm) | 157 | **4.87** | 0.764 |
| **512** (`kv_a_layernorm` → the latent cache) | 78 | **7.69** | 0.600 |
| 2048 (q_a_layernorm) | 78 | 4.10 | 0.320 |

**The 512-element norm takes 58% LONGER than the 12× larger 6144-element one.** Work is not what
sets the duration — a single workgroup's dependent-load latency is, and the 512 case is slowest
because it writes into the KV latent ring, a cold scattered HBM write, where the 6144 case writes
L2-resident `act.xn`.

> **A latency-bound op cannot be fixed by giving it more workgroups.** There is no work to spread.
> This is an independent, zero-cost kill for "widen `RmsNorm`", and it is a *different* argument
> from §4.3's (which killed it because the two extra counter-gated packets cost more than the op).
> It also explains the shape of the one datum we have: `Residual` 1 → 32 recovered only 0.178 of
> the 0.577 ms it costs — 31% — because ~2/3 of that 0.577 was never width-limited.

**Recommendation on `GLM_SPINE_CUS`: do not ship it on this evidence, and do not ship it blind on
top of this change.** It is real but small, its mechanism is capped by the above, and it now
*interacts* with the narrowing here — both change which CUs a narrow packet lands on. If it is
revisited it must be re-measured on top of `n_both`, not carried over from the lease that
produced −0.178 against a 256-wide rope.

---

## 4. WHAT THIS SAYS ABOUT THE OTHER TWO DIRECTIONS

### True idle / chain bubbles (1.589 ms, 5.8%) — already priced, and there is nothing to schedule

1437 bubbles per token, **median 0.88 µs**, against a critical path of 1400 packets. So there is
essentially **one ~1.1 µs bubble per critical-path packet**, and the pool is the chain depth times
the gate-open latency. That independently reproduces §7a-CHAIN's measured **−1.4 µs per removed
serial packet** from the other direction, and it kills the "schedule independent work into the
bubble" idea on its own terms: a 0.88 µs hole cannot absorb a packet whose dispatch alone costs
more than that. The lever remains chain-shortening, already priced at −0.35 ms for 4/layer.

### The 2–32 bucket (3.817 ms wall) — mostly the collective that was just narrowed

Charging each interval to the op holding the most live workgroups:

| bucket | top tenant | ms | share |
|---|---|--:|--:|
| 2–4 | `HeadNormRope` | 0.154 | 21.0% |
| **5–32** | **`XReduce`** | **0.948** | **30.8%** |
| 33–128 | **`MlaMergeFold`** | **1.650** | **49.5%** |

`XReduce`'s appearance here is the *cost side of `xrfit`* — narrowing 256 → 12 moved the
collective from "wide and wasteful" to "narrow and on the critical path". It is at its saturation
point (12) and cannot be narrowed further; making it cheaper now means making 6144 elements
reduce faster, not changing its width.

### The next one is `MlaMergeFold`, and it is the same arithmetic again

`d_mla_merge_fold` walks `w = slice; w < n_batch*n_head*vtiles`, and at GLM TP4 decode that is
`1 · 16 · (256/32)` = **128** — dispatched **256**, measured effective width **125.7**. It is the
#3 line of the token (2.701 ms, 9.9%) and owns half the 33–128 bucket. The kernel's own comment
states the 128 (*"VT=32 => 8 v-tiles => 128 workgroups at GLM tp4"*) and the emitter still passes
`all`.

Per-slice median body on the shipped trace is a clean step function at exactly the predicted
boundary:

| slices | count | median body |
|---|--:|--:|
| 0–127 | 128 | **27.8 µs** (they do the fold) |
| 128–255 | 128 | **5.2 µs** (gate + acquire fence, zero arithmetic) |

Waste by the same estimator: **0.203 ms/CU**. Not built here; it is the obvious next arm, and
`rope_cus`/`elem_cus` give it the shape to copy.
