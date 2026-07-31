# Batched K3 decode: what it actually needs, and the semantic trap in the middle of it

**The one piece of work standing between K3 and both remaining goals** — concurrency (aggregate
throughput) and speculative decoding (per-stream throughput) — is a slot-indexed, snapshottable
KDA recurrent state. This is the scoping note for it, written after reading the emitter and the
kernels rather than from the refusal message.

## 0. Why it gates BOTH

* **Concurrency**: `exec/amd.rs:3264` refuses `PLOW_DECODE_BATCH > 1` on any packet carrying a
  recurrent-state tensor — *"KDA state has no batch axis, so the per-slot stride below would alias
  every sequence's state onto every other's."* Independently, `serve/engine.rs:187` refuses
  TP × batch because `AmdTpGroup::submit_decode` is scalar.
* **Speculative decoding**: rejecting a speculated suffix needs a rollback. KV is append-only so
  rewinding `pos` works, but the KDA state is **read-modify-written in place with no snapshot**.
  69 of 93 layers are KDA. There is nothing to roll back to.

One state pool with per-slot addressing and a cheap snapshot serves both.

## 1. THE TRAP: K3's emitter is already `t`-parameterised, and `t` is NOT batch

This is the thing to get right before writing any code, because everything about the emitter
invites the wrong conclusion.

`crates/devgen/src/mla.rs:3878` emits one program per prefill bucket over `for &t in &pf`, using
*"the T-row emitters throughout"*. `k3.rs` threads `t` everywhere (`t * lat`, `t * hid`, `t == 1`).
So it looks as though a batched decode program is just another `t` value — emit decode at `t = B`.

**It is not, and for KDA it is silently wrong.** `runtime/amd/op_kda.h:332`:

```c
for (unsigned t = 0; t < T; t++) {
```

That loop is the RECURRENCE. It threads `T` tokens **sequentially through ONE state**, which is
exactly right for prefill (T consecutive tokens of one sequence) and exactly wrong for batch
(B tokens of B INDEPENDENT sequences). Emitting decode at `t = B` would run sequence 1's token
into sequence 0's state and produce fluent, plausible, wrong output — the failure mode this
codebase's gates exist to catch.

For the MLA layers `t` rows are fine either way: KV is per-sequence and `AmdEngine::kv_rebase`
already strides the `kv.*` table by `bytes / batch`. **KDA is the whole problem.**

## 2. What changes

| # | change | where | note |
|---|---|---|---|
| 1 | state gets a slot axis: `[B][H][D][D]` f32 | `kda.rs:265` `declare_kda_state` | one line; `state_elems() * 4` → `* batch` |
| 2 | conv state likewise: `[B][3][H*D][W]` | same | `cw * batch` |
| 3 | the recurrence takes a slot and a per-slot state base | `op_kda.h` `d_kda_state_step_g`, `d_kda_conv3` | the T loop stays for prefill; batch adds an OUTER slot dimension, it does not reuse T |
| 4 | the op must know which mode it is in | `DevOp` operand | a batch count in `i[]`, or a distinct opcode — do NOT overload `T` |
| 5 | emitter emits decode with B independent rows | `k3.rs` / `mla.rs` decode path | the non-KDA ops already take `t` rows correctly |
| 6 | lift the two refusals | `exec/amd.rs:3264`, `serve/engine.rs:187` | the second also needs rank-wise `submit_decode` |
| 7 | snapshot/restore for speculative rollback | new | only needed for spec decode, not for concurrency |

Item 3 is the real work. Items 1, 2 and 6 are small, and **must not be done alone**: a batched
state allocation with no kernel support is dead machinery, and lifting the refusal without item 3
converts a loud load-time error into silent cross-sequence corruption.

## 3. Cost, so the memory side is not a surprise

The KDA state is **6.5625 MiB per (sequence, layer)** and CONSTANT in context — which is the
architectural win (69 KDA layers cost 0.44 GiB at 1M tokens where 24 MLA layers cost 27 GiB) and
also a hard cost at ADMISSION rather than one that grows. So:

```
  B=1    0.44 GiB/rank      B=8    3.6 GiB/rank      B=16   7.2 GiB/rank
```

That is affordable next to 191 GiB of weights, and unlike the MLA KV it does not scale with
context — so K3 is, structurally, a *better* batching candidate than a dense model once the slot
axis exists.

## 4. What it buys, with the arithmetic

Decode is **protocol-bound**, not bandwidth-bound: 14 of 17 GEMV shapes move bytes that take less
time than an empty packet costs, and only **0.683 ms** of the token is bandwidth-recoverable
(`k3-75tps-program.md` §10). In that regime batch `B` divides **both** terms per emitted token —
the same weight bytes serve B sequences and the same packet floor amortises over B tokens. That is
why the MI355X roofline crossover is ~batch 312.

Against the shipped **28.876 ms (34.6 tok/s)**:

* **aggregate** throughput should scale close to linearly to B=8–16 (the ceiling is VRAM and the
  flat KV allocation, not compute), which clears 75 tok/s aggregate comfortably;
* **per-stream** is unchanged by batching — only speculative decoding moves it, and item 7 is what
  unblocks that.

**Decide which number the goal means before starting**, because items 1–6 give the first and item
7 is what gives the second.

## 5. Validation, which is not optional here

The KDA recurrence is the model's core and a wrong one is fluent rather than broken.

* `runtime/tests/k3_block_gfx950_test.c` is the numeric gate for a KDA block against a fixture.
* The TP8 token gate: prompt `1008,10484,318,15383,387` must continue *" Paris. The population is
  approximately 67 million people…"*, all 8 ranks identical.
* **The batch-specific gate that does not exist yet and must**: B copies of ONE prompt must
  produce B identical streams, AND B different prompts must each produce what they produce alone.
  `perf-data/batched-decode-amd-status.md:19-31` is the precedent — it caught ragged-position bugs
  on the dense path with exactly that test at lengths 3/5/7/4.
* `scripts/bench_gsm8k.sh` at B>1 is the end-to-end check: accuracy must not move.

---

# 6. PROGRESS — five of seven wired, all inert, all byte-identical

Landed on `worktree-gate-sc1-coverage`. Every one of these is a no-op for existing programs and
was gated that way before commit: the emitted K3 blob is **md5 `7db2fbb34230050f0508a4e706523a98`
at every step**, and the two kernel changes (which alter a signature and so cannot be
bit-identical) were gated instead by a bound TP8 run producing the control's exact 32-token
stream.

| # | item | state |
|---|---|---|
| 3a | recurrence takes a per-row state stride | **DONE** — `PLOW_KDA_F_SEQ_ROWS` in `KDA_STATE_STEP`'s flags word |
| 3b | conv window takes a per-row stride | **DONE** — carried in `CONV3`'s `fj[1].u` |
| 1,2 | KDA + conv state allocated per SLOT | **DONE** — `declare_kda_state(.., slots)` |
| 4 | `RowKind` and its wiring to both carriers | **DONE** — `emit_k3_model → emit_k3_block → emit_kda_mixer` |
| 6 | `in.kvlen` sized per slot | **DONE** — this is how the host learns the batch |
| 5 | **MLA `n_batch`** | **NOT DONE — bigger than it looks, see below** |
| 7 | batched TP `submit_decode` + lifting the refusals | **NOT DONE** |

## 6.1 Item 5 is not a one-line operand change

`emit_glm_mla` hardcodes `d.i[0] = 1`, which is the kernel's `n_batch`. Setting it to `B` is the
easy half. The hard half is that **the MLA KV cache must then carry a batch axis too** — the
kernel indexes `kv_scale[n_batch*kv_stride + b*kv_stride + row]` and takes `kv_len` as a
per-sequence `const int*`, so `n.ckv[slot]` / `n.krot[slot]` have to be allocated `B` wide and the
per-slot base resolved the way `AmdEngine::kv_rebase` already does for the dense path.

`emit_glm_mla` is also **shared with GLM-5.2**, so the parameter has to default to 1 there rather
than being threaded blindly.

## 6.2 What is genuinely left, in the order it has to happen

1. **Write the batch correctness gate first** (§5). Without it the rest is unverifiable, and the
   failure mode is fluent wrong output rather than a crash.
2. MLA KV batch axis + `n_batch` operand (item 5).
3. Batched TP `submit_decode` — factor a `decode_prepare_batched` out of the existing
   single-GPU `decode_step_batched`, then thread slices through `AmdTpGroup::submit_decode`,
   which today takes scalar `pos`/`kvlen`.
4. Lift `exec/amd.rs:3264` and `serve/engine.rs:187` — **only** once 1–3 are done. Lifting them
   earlier converts a loud load-time error into silent cross-sequence corruption, which is the
   single worst outcome available here.
5. Then measure: aggregate throughput at B=4/8/16, and GSM8K at B>1 to show accuracy has not moved.

## 6.3 One process note worth keeping

An earlier attempt at item 4 put the new parameter on `emit_k3_dense_mlp` instead of
`emit_k3_block`, because the signature tail it matched on (`t, n_cu, tp, deps`) is shared by
several functions in a 3000-line file. The compiler caught it; it was reset and redone against
explicit line numbers. **Blind string replacement is not a safe edit primitive on this emitter** —
match on the enclosing function, not on the parameter list.

---

# 7. FREEZE POINT (commit `a00daff`) — six of seven wired, campaign paused for merge

Frozen deliberately **before** item 7, so the branch can go to main and the campaign resume from
a known-good base rather than from a half-wired batch path.

## 7.1 What is in the freeze

| | |
|---|---|
| decode | **33.233 → 29.001 ms, −12.73%, 30.1 → 34.5 tok/s** (3 leased reps, sd 0.043) |
| token identity | prefill → 17374, all 8 ranks agree, 32-token stream identical to control |
| default blob | **byte-identical** to the pre-campaign control, md5 `7db2fbb3…` |
| tests | 979 passed / 0 failed |
| coverage gate | 391 scoped, 0 plain |
| canary | 248 VGPR / 0 AGPR / occ 2 / 0 spill |
| tracked build dirs | 0 (`ba_*/` ignored) |

Items 1–6 of §2 are wired and **every one is inert**: `slots == 1`, `seq_rows == false`, both
kernel carriers 0. That is why the blob is byte-identical — the batch path exists in the source
and is unreachable from any program the emitter currently produces.

## 7.2 What is deliberately NOT in the freeze

**Item 7 — a batched TP `submit_decode`, and lifting the two refusals.** The refusals at
`exec/amd.rs:3264` and `serve/engine.rs:187` are the only thing standing between a half-wired
batch path and silent cross-sequence corruption, and `scripts/k3_batch_gate.sh` must pass at
**B ≥ 4** before either moves. Freezing with the refusals intact means a merge to main cannot
regress anything: the batch path is code nothing can reach.

## 7.3 Resuming

The metric was settled: **aggregate throughput**, not per-stream. So speculative decoding is out
of scope for the resume, and the path is:

1. Factor `decode_prepare_batched` out of the existing single-GPU `decode_step_batched`.
2. Thread slices through `AmdTpGroup::submit_decode` (scalar `pos`/`kvlen` today).
3. Emit at `RowKind::Sequences` with `PLOW_DECODE_BATCH=B`, and check the blob is NO LONGER
   byte-identical — at that point the carriers are live and every claim above needs re-earning.
4. `scripts/k3_batch_gate.sh <blob> <hsaco> <ckpt> 4` — check A then check B. It refuses to pass
   at B=1, so a green run means something.
5. Only then lift the two refusals.
6. Measure aggregate at B=4/8/16 and re-run `scripts/bench_gsm8k.sh` at B>1: accuracy must not move.

Expected: ~34.5 tok/s/stream × B minus contention. The ceiling is VRAM — the KDA state is
0.44 GiB per slot and CONSTANT in context — not compute.


---

# 8. RESUMED AND LANDED — the gate passes, and 91.3 tok/s aggregate clears the goal

Item 7 is done. `scripts/k3_batch_gate.sh` **PASSES at B=4 on K3 at TP8**, both checks, and the
aggregate-throughput goal is met.

| batch | ms/step | aggregate tok/s | per stream |
|---|--:|--:|--:|
| 1 (shipped) | 29.0 | 34.5 | 34.5 |
| 4 | 76.1 | 52.6 | 13.1 |
| 8 | 110.4 | 72.4 | 9.1 |
| **16** | **175.3** | **91.3** | 5.7 |

**75 tok/s aggregate is cleared at B=16** (91.3), and B=8 is within 4% of it. Per-stream falls, as
§4 predicted: batching moves aggregate, never latency.

## 8.1 The whole bug class, in one sentence

**`t > 1` does not mean prefill.** A batched decode program has `t = B` rows that are INDEPENDENT
SEQUENCES, and every site that read `t == 1` as "am I a decode" routed it to a prefill arm. On this
interpreter a packet with no matching `case` falls through AMD's dispatch `default:` and WRITES
NOTHING — no fault, no diagnostic, output that is finite and plausible and wrong. Seven instances:

| # | site | what it did |
|---|---|---|
| 1 | `mla.rs` `prog_t` | hardcoded 1, so a B=4 blob refused itself against `in.kvlen` |
| 2 | K3 tail | sampled ONE row; a batched decode needs all B (Gemv `M=B`, and the `n_batch` both argmax kernels have always taken and no emitter ever set) |
| 3 | `nsplit` | flash split buffers sized at 1 while the arm wrote `n_split` |
| 4 | both KV writers | no batch axis; `n_batch_kv` + `out_stride` give row `t` its own ring at its own `pos[t]`, which the host-patched `out_row0` cannot express for B sequences |
| 5 | `emit_k3_linear` | **1276 packets per step** were tiled PREFILL GEMMs. `PLOW_GEMV_MM = next_pow2(PLOW_DECODE_BATCH)` exists precisely so `Gemv` carries B rows |
| 6 | `interp.hip` | the grouped expert arms sat behind `PLOW_MOE_PREFILL` **nested inside `PLOW_BUCKET_PREFILL`**, which a decode object never defines — so all 92 MoE layers wrote nothing |
| 7 | expert packing | the driver matched only `t[3]`/`i[1]`; `MoeGroupGluPf` carries its table at `t[2]`/`i[0]` |

Only #5 and #6 moved the output. The other five were real and individually invisible — which is
exactly why a fix has to be measured rather than reasoned about here.

## 8.2 The instrument that made it tractable

`PLOW_K3_SEQ_ROWS` forces the sequence-row carriers on at **B=1, where the answer is known**. That
blob reproduces the reference stream token for token, all 16 — proving every carrier (KDA stride,
conv stride, flash decode arm, batched KV ring, tail) sound and isolating the fault to multi-row
handling. B>1 alone cannot answer that question, because at B>1 there is no reference to compare to.

## 8.3 One correction to §5's validation plan

Check B as designed — "B different prompts must each produce what that prompt produces ALONE at
B=1" — **is not a valid criterion**, and demanding it made the gate fail a working batch. A batched
decode routes MoE through the GROUPED kernel and a B=1 decode through the per-slot one; they
accumulate in different orders, and greedy decoding turns any tie-break into a different token
within a few steps:

```
prompt   'The capital of France is'
B=1      ' Paris. The population is approximately 67 million people. The official language is French.'
B=4/8/16 ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The'
```

Both fluent, both correct, neither a defect. Check B now compares **two batched widths**, which
share a kernel and so legitimately owe token-identity, while still varying the per-slot strides,
positions and kvlens it exists to test. B=4, B=8 and B=16 agree exactly.

## 8.4 State of the refusals

* `exec/amd.rs` — **narrowed, not lifted, and that is the end state.** It no longer refuses any
  recurrent state at batch > 1 (false since `RowKind::Sequences`), and it cannot key on
  `PLOW_DECODE_BATCH` because `batch` is derived from `in.kvlen` and agrees with the emitter by
  construction. It keys on `PLOW_KDA_F_SEQ_ROWS` in the decode program — a carrier that cannot be
  set unless the emitter also sized the state per slot. An old B=1-shaped blob at batch > 1 is
  still refused, loudly.
* `serve/engine.rs` — **LIFTED.** `submit_decode_batched` prepares every rank before any rank
  launches, and `prefill_slot` rebases every rank onto the slot for a collective prefill. A B=4
  packet serves at TP8 and passes `bench_speed.sh`'s coherence gate.

## 8.5 What is NOT done, stated plainly

* ~~**GSM8K at B>1 has not been run.**~~ **RUN — see `perf-data/k3-gsm8k.md` §2.** Three runs:
  B=1/c1 **0.8100**, B=4/c1 **0.8400**, B=4/c4 **0.9031**. Accuracy HOLDS at B=4 — and the
  intermediate reading that the grouped MoE kernel is *better* was **refuted by its own control**:
  isolate the kernel (B=1/c1 -> B=4/c1) and the effect is +3.0pp at **z=0.79, not significant**.
  Only the doubly-confounded extremes reach z=2.67. Quote ~84%, not 90.3%.
  What the three runs DO expose is a 77.5-90.3% spread across nominally identical greedy runs,
  which §9's cross-rank race would explain by construction. Same-assets-twice is the open test.
* **The serving scheduler is the bottleneck, not the kernel.** Through `plowrt serve` at
  concurrency 16 on the B=16 packet: **49.3 tok/s**, against 91.3 from the same packet under
  `amd-bench`. TTFT is 3.3 s because prefills serialise and block the whole batch. Chunked
  prefill and prefill/decode interleave are what close that gap — the kernel side is already there.
* **Registers at B >= 8.** `interp_decode_fp8kv_k3_gq` is VGPR=248 / occ=2 / **spill=0** at B<=4
  and VGPR=256 / occ=2 / **spill=22** at B=8 and B=16. Occupancy holds, but the headroom is gone;
  `PLOW_GEMV_MM` is what costs it.
* **B=4 is slower per stream than a B=1 packet at concurrency 1** — every row runs whether or not
  its slot is live, because `t` is compiled, not passed. Serving a mixed load wants either a
  smaller batch or admission that keeps slots full.

# 8.6 THE KDA ROW AXIS IS NOW PARALLEL — 91.3 -> 94.8 tok/s

§2 item 3 asked for "an OUTER slot dimension"; what shipped first was the per-row STRIDE without
the axis. `d_kda_state_step_t`'s work-item map was `H * ntile` at every batch — 192 items at
TP8/BV=8 — so a B=16 decode ran its 16 rows **serially** inside a workgroup count that did not know
B existed, on **69 of K3's 93 layers**, with 64 of 256 CUs idle.

Rows are a parallel axis when they are INDEPENDENT SEQUENCES: nothing in row `t`'s recurrence reads
row `t-1`'s. So when `bstride != 0` the row is folded into the item map (`nitem = T * H * ntile`)
and the per-row loop collapses to one iteration; when `bstride == 0` — a prefill bucket, whose rows
are consecutive tokens of one sequence threading through one state — it stays exactly as it was.
`state_step_blocks_rows(n_cu, rows)` widens the packet to match.

MEASURED, K3 TP8 B=16, `amd-bench --batched`, 16 steps, bound:

| | ms/step | aggregate tok/s |
|---|--:|--:|
| serial row axis | 175.257 | 91.3 |
| **parallel row axis** | **168.784** | **94.8** |

All 8 ranks token-identical, `k3_batch_gate.sh` passes at B=4, B=1 blob byte-identical
(`7db2fbb34230050f0508a4e706523a98`).

## 8.6a RETRACTION — the "+3.7%" was NOT ESTABLISHED, and here is the measurement that says so

The two numbers above are SINGLE RUNS, and a single run at B=16 does not resolve 3.7%.

MEASURED afterwards, same blob, same object, same command, four consecutive runs at `--steps 64`:

```
177.207   178.579   182.085   260.533   ms/step
```

Excluding the outlier: **mean 179.29, sd 2.52, spread 2.7%** — so per-run noise is about
**+/-1.4% (1 sd)**, and roughly one run in four is a **+45% outlier**. Worse, the SAME code that
measured 167-169 in one session measured 177-182 in the next: a **6-8% session-to-session drift**
on top of the per-run noise. (The tuning DB was checked and is NOT stale —
`cargo test -p devgen --test tuned_tile_selection` passes 4/4 — so that documented trap is not the
cause. The cause is unidentified; thermal or power state are the obvious candidates.)

A 6.5 ms delta is inside that band. **The row-axis parallelisation's throughput effect is not
established by the data taken for it.** The change is still correct and still worth having — it
exposes a genuinely parallel axis that was being run serially on 69 of 93 layers, which the trace
(§8.7) independently confirms is now only 5.4% of the step — but it should not be quoted with a
percentage.

**The defensible B=16 number is 179.3 +/- 2.5 ms/step, i.e. ~89 tok/s aggregate** (n=3, one
outlier excluded). That still clears the 75 tok/s goal comfortably. Earlier single-run figures in
this document (91.3, 94.8, 95.7 tok/s) are all inside the same band and should be read as
"~89-95 tok/s", not as a progression.

## 8.6b THE HARNESS, AND ITS SELF-TEST

`scripts/k3_ab_bench.sh` is the fix: interleaved (A B B A ...), order-reversed every pair, median
rather than mean, per-arm ranges reported, and **no verdict when the arms overlap**.

**It was validated by pointing it at the same assets in both arms**, where the true answer is 0%:

```
        same-1  median 194.230  min 182.509  max 196.844  sd 7.634  n=3
        same-2  median 183.330  min 178.852  max 187.831  sd 4.490  n=3
  B vs A on the median: -5.61%
  VERDICT: NOT RESOLVED — the ranges overlap by 5.322 ms.
```

**A naive single-run comparison of a thing against ITSELF would have reported a 5.6% speedup.**
That is precisely the error §8.6a retracts, reproduced deliberately and caught.

Two practical consequences:

* **At `REPS=3` the instrument cannot resolve ~6%.** Do not pursue a change whose expected effect
  is under ~10% at B=16 without either many more reps or a fix for the variance itself.
* **Each run reloads 191 GiB of weights (~3-4 min)**, so a `REPS=3` A/B is ~25 minutes of leased
  GPU. That cost is the reason single runs were used, and it is not a good enough reason.

The variance's CAUSE is still unidentified and is the thing worth fixing next on this axis: a stale
tuning DB was ruled out (4/4 pass), the outlier is not a cold-start (it landed on run 4 of 4), and
thermal or power state remain the untested candidates.

## 8.6c THE CAUSE, most likely: THE BOX IS SHARED AND THE LEASE DOES NOT COOL IT

`gpulease` gives EXCLUSIVITY. It does not give a consistent STARTING STATE, and on this box that
distinction is the measurement. The lease log shows other sessions' jobs interleaved with this
campaign's, running back to back:

```
11:14:41  slabab  ACQUIRED ...  11:18:17  slabab RELEASED  held=216s
11:18:18  k3rep   ACQUIRED                                 <- one second later
```

`k3rep` is the run that produced **177 / 179 / 182 / 260 ms**, and it began on GPUs another job had
just driven hard for 216 seconds. MEASURED while a different session's job held the lease: junction
**43-51 C** and **311-322 W**. "Idle" on this box is not idle, and a measurement that does not
record its starting temperature cannot be compared against one taken at another time — which is
exactly the shape of the 6-8% inter-session drift.

This is a HYPOTHESIS with a mechanism and supporting measurements, not a proven cause: no run has
yet been taken with the temperature controlled, because the lease queue has been contended. What
has been ruled out is stale tuning data (`tuned_tile_selection` 4/4) and cold start (the outlier
landed on run 4 of 4).

`k3_ab_bench.sh` now records the hottest junction temperature at the start of every run and takes
`SETTLE_C` to wait for a threshold first. That makes the hypothesis testable by the next person
who runs it rather than leaving it as an assertion.

## 8.6d IS B=1 DECODE PERFORMANCE LOST? — the kernel A/B says no, and the box says why

The batching work changed KERNELS that B=1 also runs (`op_kda.h`: the parked mask and the row-axis
fold). The BLOB is byte-identical at B=1 (`7db2fbb34230050f0508a4e706523a98`, re-verified after
every change), but that does not clear the objects. So the objects were A/B'd directly: the
pre-batching `runtime/` was restored from commit `ed0faf4`, built into its own hsaco, and run
against the current one on the SAME blob.

MEASURED, `k3_ab_bench.sh`, B=1, `STEPS=64`, interleaved and order-reversed, n=3 per arm:

| arm | median | min | max | sd |
|---|--:|--:|--:|--:|
| pre-batching (`ed0faf4` objects) | **34.465** | 33.822 | 38.570 | 2.576 |
| current | 36.703 | 34.381 | 41.371 | 3.560 |

`B vs A: +6.49%` — **VERDICT: NOT RESOLVED**, ranges overlap by 4.189 ms.

**The finding that matters is the FIRST ROW.** The historical B=1 record is **28.876 ms**
(`k3-hier2-ceiling.md`), and today the PRE-BATCHING kernel measures **34.465 ms** — the same ~20%
above the record as the current one. So the B=1 gap against the record is **the box, not the
batching work**: this machine is shared and other sessions ran GPU jobs continuously through these
measurements (§8.6c).

On the remaining +6.5% between arms, the code says it is noise rather than cost. Every kernel change
is gated OFF at B=1:

* the parked mask needs `PLOW_KDA_F_SEQ_ROWS`, which a B=1 blob does not set, so the interpreter
  passes `nullptr` and the kernel does one scalar test per row;
* the row-axis fold is gated on `bstride != 0`, and B=1 has `bstride == 0`, so `trep = 1`,
  `nitem = items`, and the loop bounds are IDENTICAL to before;
* `state_step_blocks_rows(n_cu, 1)` is exactly the old `state_step_blocks(n_cu)`.

That is a handful of scalar branches across 69 KDA layers on ONE row — microseconds, not 2.2 ms of
a 34 ms token. ### The second A/B settles it: THE SIGN FLIPS

Repeated at `REPS=6` (12 runs per arm, same objects, same blob):

| arm | median | min | max | sd | n |
|---|--:|--:|--:|--:|--:|
| pre-batching | 38.511 | 34.570 | 40.918 | 2.472 | 5 |
| current | **36.306** | 33.829 | 38.944 | 2.496 | 4 |

`B vs A: -5.72%` — the CURRENT kernel is faster on the median this time. **NOT RESOLVED** again.

| A/B | delta | direction |
|---|--:|---|
| n=3 per arm | **+6.49%** | current slower |
| n=6 per arm | **-5.72%** | current faster |

**Two independent A/Bs disagree on the SIGN.** That is the signature of noise, not of a cost, and
together with the gating argument above it is enough to answer the question: **batch-1 decode
performance is not lost.** What the campaign cannot yet do is bound the effect below ~6%, and the
honest statement is that the effect is smaller than the instrument rather than that it is zero.

(Three runs of the twelve failed — the arms are n=5 and n=4 — because another session took the
lease mid-experiment. That is the same contention §8.6c documents, and it is why the harness reports
`n` per arm rather than assuming it got what it asked for.)

**Method note for anything that follows.** Every B=16 comparison in this campaign was a single run
against a single baseline. At +/-1.4% noise with +45% outliers and a 6-8% inter-session drift, that
resolves nothing under ~10%. Anything claiming less than that needs interleaved, order-reversed
repeats — which is what `plans/knob-contract.md` §0-BENCH already required and which this section
did not do.

# 8.7 THE B=16 STEP, ATTRIBUTED — the first trace ever taken at batch > 1

`PLOW_TRACE_RAW` produced NOTHING at B>1 until now: `amd_bench_tp`'s `--batched` arm returned
before the trace write at the end of the function. So the instrument was unavailable in exactly
the place the question lives — the B-sweep fits `43.6 + 8.245*B ms` and neither term had ever been
attributed. Fixed; the write now happens on that path too.

MEASURED, K3 TP8 B=16, 4 steps, bound, `k3_trace_report.py`: 2942 packets, 513,288
(workgroup, packet) records, **170.83 ms** accounted.

| subsystem | ms | % of step |
|---|--:|--:|
| **GEMV** (all shapes) | **96.50** | **56.5%** |
| **MoE PREFILL path** (GLU_PF + DOWN_PF + ALIGN_PF + COMBINE_PF) | **34.74** | **20.3%** |
| XREDUCE + XREDUCE2 | 16.50 | 9.7% |
| KDA (state + conv + gated norm) | 9.14 | 5.4% |
| ATTN_RES + RESIDUAL | 5.85 | 3.4% |
| FLASH_MLA_DECODE_FP8 | 1.99 | 1.2% |

## 8.7a The MoE prefill path is 20.3% of the step, and most of it is STRAGGLER

This is the measured consequence of `k3.rs`'s `if t == 1` — the last site that reads `t == 1` as
"am I a decode". At B>1 all 92 layers take the grouped PREFILL MoE arm.

| op | blocks | n | body/pk | **straggler/pk** |
|---|--:|--:|--:|--:|
| `MOE_GROUP_GLU_PF` | 256 | 92 | 181.7 us | **164.9 us** |
| `MOE_GROUP_DOWN_PF` | 256 | 92 | 108.6 us | **97.1 us** |
| `MOE_ALIGN_PF` | **1** | 92 | 50.2 us | 0 (single workgroup) |
| `MOE_COMBINE_PF` | 256 | 92 | 25.2 us | 15.1 us |

**Straggler is 91% of `GLU_PF`'s body and 89% of `DOWN_PF`'s — 24.10 ms of the 170.83 ms step is
workgroups waiting for their slowest peer.** The grouped GEMM tiles are `MPF_BM = 64` rows and a
B=16 decode gives ~63 live experts carrying about one real row each, so the tiles are ~1.6% full
and wildly unequal. This is load imbalance, not bandwidth.

`MOE_ALIGN_PF` is the other shape of the same problem: **one workgroup**, 92 times per token,
50.2 us each = **4.69 ms (2.7%) on a single CU while 255 sit idle**, doing a prefix scan over
`3*n_exp+1 = 2689` ints.

## 8.7b What this settles, and what it re-ranks

* **GEMV at 56.5% is the floor**, and §Q1 of the kernel review already established it is
  weight-stationary with no re-read and that MFMA would buy <=6%. It is bandwidth, not waste.
* **The MoE prefill path, at 20.3%, is the largest addressable item by a wide margin** — and it is
  addressable in three independent ways, cheapest first: widen `MOE_ALIGN_PF` off its single
  workgroup (2.7% on the table), shrink `MPF_BM` for the low-fill decode case, or route batched
  decode off the prefill arm entirely.
* **KDA is now 5.4%**, down from being the presumed dominant slope term. §8.6 parallelised its row
  axis for +3.7%; the trace confirms there is little left there — which retires the kernel review's
  estimate that it held 8-54 ms.
* `FLASH_MLA_DECODE_FP8` carries the **highest gate cost per packet in the model** (29.6 us) but
  only 1.2% of the step, so it is not worth chasing.

# 9. A RARE CROSS-RANK DIVERGENCE AT B=4, found by GSM8K and not by the gate

GSM8K at B=4 / CONC=4 (`N=200 SHOTS=8 MAXTOK=320`, greedy) completed **177/196 = 0.9031**, with
**4 of 200 requests failing outright**. The four failures are ONE event, not four:

```
WARN plowrt::serve::mux: amd: decode failed error=device error: TP ranks disagree on a
  batched decode step: rank 0 sampled [11, 28, 1288, 20], rank 1 sampled [11, 28, 5469, 20]
  — the all-reduce is wrong  fed=4
```

Exactly one distinct divergence across the run (5 log lines = 1 engine warning + 4 stream errors);
zero in either `bench_speed.sh` serve run. One bad step kills **every in-flight request at once**,
because at B>1 they share the decode step — which is why the errors arrive as a burst of B.

**This is the agreement check doing its job.** `decode_step_batched` compares the whole B-vector
across ranks (§8.4), so the step became an HTTP 500 instead of silently-wrong tokens. Without that
check this run would have reported a slightly lower accuracy and no fault at all.

## 9.1 What the two ids rule out

vocab 163840 / TP8 → `vocab_l` = 20480, so rank 0 owns ids `[0, 20480)`. Both `1288` and `5469`
are in **rank 0's shard**.

That REFUTES the obvious first theory, a `PLOW_XCTR_DEADLINE_TICKS` bail in `d_xargmax_fin_mega`
(`op_collective.h:344` — "the rank keeps its LOCAL argmax rather than hanging the queue ... the
same silent-wrongness"). A rank 1 that bailed would keep its shard-1 local max and report an id
**>= 20480**; it reported 5469. The 1 s deadline is also ~13x a 76 ms B=4 step.

So both ranks concluded the winner lies in shard 0 and disagreed on WHICH shard-0 token — i.e.
rank 1's read of **rank 0's published u64** did not match the value rank 0 itself folded. That
localises it to the publish/consume of the cross-rank slot in `d_xargmax_fin_mega`, not to the
per-layer all-reduces (whose failure would garble the residual stream and every slot).

## 9.2 Why the gate could not have caught it

`k3_batch_gate.sh` decodes 16 tokens. This is ~1 event in the tens of thousands of decode steps a
200-question GSM8K run performs. A 16-token gate is the wrong instrument for a 1e-4..1e-5 race;
it is a *semantic* gate, and it remains correct for what it tests.

## 9.2b SECOND SIGHTING — the same fold, in PREFILL, with a sharper signature

2026-07-31, the B=4 GSM8K re-run after the state-clear fix. 1 failure in 200 requests:

```
WARN plowrt::serve::mux: amd: prefill failed slot=0 error=device error: RANKS DISAGREE:
  rank 0 sampled 44208, rank 1 sampled 8735
  (all: [44208, 8735, 8735, 8735, 8735, 8735, 8735, 8735])
```

Two things this adds over §9.1:

1. **It is not decode-specific.** This is the PREFILL path's cross-rank agreement check, so the
   defect lives in the shared fold, not in anything `decode_step_batched` introduced.
2. **7-vs-1, and the outlier is rank 0.** Seven ranks fold to `8735`, which is in **rank 0's own**
   shard `[0, 20480)`; rank 0 alone folds to `44208`, which is in **shard 2's** range. So either
   rank 0 picked up a value from rank 2's published slot that no other rank saw, or the other seven
   all missed rank 2's contribution — and the former is far likelier, because a value only one rank
   observes is the signature of reading a slot that was written outside the intended
   publish/rendezvous window.

Combined rate across the two sightings: **~1 request in 200** hard-fails. That is a serving-visible
rate, not a curiosity, even though each event fails loudly rather than corrupting output.

## 9.3 The concrete next step, and one thing already worth fixing

`interp.hip:2508` passes `nullptr` for `d_xargmax_fin_mega`'s `status` out-param, so the op's own
`0xDEAD0000|rank` bail signal is **discarded**. Wiring it through would settle bail-vs-race on the
next occurrence instead of forcing the inference above. Two stale comments in the same op should
go with it: `op_collective.h:342` still says "decode here is B=1", which is no longer true.

Root-causing the race itself needs that instrument plus a soak; it is NOT fixed here, and B>1
should be considered to carry a rare hard-fail (not a rare wrong answer) until it is.

---

# 10. PROVING WHAT THE CROSS-GPU COLLECTIVES ACTUALLY DO

*"How do we prove that across GPUs the data moves as the counters unlock — op by op — rather than
the reduce simply being blocked?"* It is answerable, the instrument exists, and the answer is
measured rather than argued.

## 10.1 The mechanism, so the two hypotheses are distinguishable

plow's collectives are **ordinary counter-gated packets inside the persistent megakernel** — no
kernel launch, no host synchronisation (`exec/amd_tp.rs` module doc). The sequence per collective:

1. The **producing GEMV writes its partial straight into its own `peer_scratch` slot**, fused into
   the GEMV epilogue. There is no separate copy step and no staging buffer.
2. Each rank **signals every peer's counter** with a system-scope RELEASE RMW (`xctr_signal`).
3. Each rank **polls its OWN counter** relaxed until it reaches `n_gpu`, then takes exactly ONE
   system-scope acquire (`xctr_poll` / `xctr_acquire`). A relaxed poll is deliberate: a
   system-scope acquire load emits a full invalidate on every iteration.
4. `d_xreduce` then **READS all N peers' slots** — a PULL over XGMI — and sums them.

That gives a clean discriminator, because the waiting and the moving are in different phases:

| | if the reduce is BLOCKED (ranks desynchronised) | if data moves as counters unlock |
|---|---|---|
| `gate` (arrive -> ready) | **large** — peers spinning | **small** — everyone already there |
| `body` (ready -> end) | small | **large** — this is the bytes |

## 10.2 The measurement

`PLOW_TRACE_RAW` records `arrive / ready / end` per **(workgroup, packet)**, so the split is
directly observable. (It produced nothing at B>1 until §8.7 fixed the `--batched` arm.)

MEASURED, K3 TP8, B=16, bound, 2942 packets / 513,288 records:

| collective | blocks | n per token | gate/pk | body/pk | straggler/pk |
|---|--:|--:|--:|--:|--:|
| `XREDUCE2` | 224 | 94 | **2.96 us** | 76.93 us | 13.58 us |
| `XREDUCE2` | 112 | 92 | **1.64 us** | 64.34 us | 14.74 us |
| `XREDUCE` | 224 | 92 | **2.86 us** | 29.08 us | 11.41 us |

```
collectives total:  gate 0.66 ms   body 15.84 ms   -> gate is 4.0% of collective time
as a share of the 170.83 ms step:  gate 0.39%   body 9.27%
```

**The reduce is NOT blocked.** Ranks arrive within ~2-3 us of each other, and essentially all of
the 9.3% the collectives cost is bytes crossing the fabric.

## 10.3 It is genuinely op-by-op, and the packet count is the proof

There are **278 collective packets per token** (94 + 92 + 92), each with its own counter gate and
each unlocking independently. A design that blocked would show ONE rendezvous per layer or per
token, not 278 — and its gate would dominate.

Two further consequences visible in the same table:

* **The straggler (11-15 us) is 4-5x the gate.** So what unevenness exists is in the reduce ITSELF
  (workgroups finishing at different times), not in arrival. Arrival is the tight part.
* **`FLASH_MLA_DECODE_FP8` carries a 29.1 us gate — 10x any collective's.** That is where waiting
  actually happens in this model, and it is only 24 packets and 1.2% of the step. Worth knowing
  before optimising a collective that is already tight.

## 10.4 What would falsify this, and the alarms that already exist

The claim is falsifiable by the same instrument: if `gate/pk` grew to the scale of `body/pk`, the
ranks would be desynchronising. That failure is on record and is not hypothetical —
`runtime/tests/tp_decode.c`:

> *Per-rank-all-segments let the ranks desync — a lagging rank made peers time out and bail, giving
> a WRONG, 100x-slow reduction at TP>=4.*

Which is why prefill runs **per-segment, all-ranks, with a host barrier between segments**, and
decode runs **launch-all-then-drain-all**. Two independent alarms cover it:

* **`audit_xctr`** (12 KiB D2H per rank) confirms every gate reached its expected count.
* **The deadline bail.** A rank that spins past `PLOW_XCTR_DEADLINE_TICKS` (1 s) gives up and
  returns WITHOUT reducing, setting `0xDEAD0000 | rank`. Absence of bails is itself evidence the
  rendezvous is fast — though note §9: that status is passed `nullptr` at every call site, so the
  host-side agreement check is currently the only thing that surfaces it.

## 10.5 Reproduce

```bash
PLOW_TRACE_RAW=/tmp/trace.bin plowrt amd-bench --blob <b16>/model.pkt --hsaco <b16>/hsaco \
  --checkpoint <ckpt> --tp 8 --steps 4 --ctx 5 --prompt '<16 prompts>' --batched
python3 scripts/k3_trace_report.py /tmp/trace.bin --top 40
```

The `gate/pk` and `body/pk` columns are the answer. Read them per opcode, not in aggregate: the
aggregate hides that the collectives are tight while MLA's flash decode is not.
