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
