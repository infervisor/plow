# GLM-5.2 TP4 decode — three EMITTER/PREP levers, A/B'd separately

> **⚠️ SUPERSEDED (2026-07-29): `GLM_LINEAR_FP8` is a WIN, not a regression.** The `+0.39 ms`
> figure on this page was measured on an UNSTACKED blob. `perf-data/glm52-linear-fp8-reeval.md`
> re-measures it stacked at **−0.417 ± 0.175 ms, n=6 — 97 % of the −0.431 ms floor** (commit
> `b3f77fd`). Every "do not ship" verdict for this knob below is void; the other knobs on the
> page are unaffected.

**Measured 2026-07-28**, 4× gfx950 (MI355X), real weights (`/home/lava/models/GLM-5.2-plow`,
183 GiB/rank), `glm52_decode --tp 4 --sweep 1024,4096 --steps 65 --gen 256` under `gpulease -n 4`.

**TWO BASELINES, and every number says which.** The `MLA_MERGE_FOLD` rewrite (6495efc) landed
mid-session and moved the whole model:

* **PRE** — interpreter with `d_mla_merge_fold<512,32>` (the `interp_decode_fix.elf` dispatch fix
  only). Baseline blob = **32.35 ms/token** (5 runs, sd 0.15).
* **POST** — that dispatch fix plus the wave-cooperative fold BODY, i.e. the merged tree this
  branch now carries. Baseline blob = **28.329 ms/token** at ctx 1k (28.403 at 4k).

A lever's delta is only meaningful against its own baseline, and the two are not interchangeable:
the fold rewrite removes ~4 ms of the very serial chain that decides whether a byte-count change or
a concurrency change can show up at all. §1–§3 are re-derived POST; the PRE numbers are kept only
where they say something the POST run cannot.

**§0-BENCH.** The C harness is an EXPERIMENT instrument. **No number here may be placed next to a
vLLM number.** plow-internal A/B only.

Every blob is emitted by `plowc` from this worktree; with no knobs set the emission is
**byte-identical to `glm52_tp4_64k.pkt`**, which is what makes the deltas below attributable.

---

## 0. TL;DR — POST the fold rewrite, one lease, controls interleaved 4×

Run order was `c1 → b_head → c1 → a_linfp8 → c2_s32 → c2_s48 → c2_s48 → c1 → b_head`, so the
session drift shows up in the control instead of hiding in a delta. It is real and monotonic —
`c1` = 28.329 / 28.072 / 27.993 — so each row below gives BOTH the mean-to-mean delta and the
delta against the control's local neighbours, and no conclusion rests on the difference.

| lever | knob | vs control mean | vs local control | gate | verdict |
|---|---|--:|--:|---|---|
| **(c)** co-resident shared expert, s=48 | `GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48` | **−0.81** | −0.71 | **256/256 bit-identical** | **ship — but see §3.0, the sign is the interpreter's** |
| **(c)** same, at the shipped width s=32 | `GLM_MOE_CORESIDENT=2` | **−0.72** | −0.62 | **256/256 bit-identical** | " |
| **(b)** vocab-column-parallel `lm_head` + `XArgmaxFin` | `GLM_SHARD_HEAD=1` | **−0.26** | −0.16 / −0.28 | **256/256 bit-identical**, 0 cross-rank disagreements | **ship** |
| **(a)** `o_proj` + shared expert on block-fp8 | `GLM_LINEAR_FP8=1` | **+0.39** | +0.44 | weights provably identical to bf16 rounding | **do not ship — §2** |

(b)'s two samples sit at opposite ends of the session (positions 2 and 9) and are below their local
control in both places, so the drift does not explain it; and −0.26 mean-to-mean reproduces the
PRE-rewrite −0.26 exactly on a completely different baseline.

Full table:

| blob | runs @ ctx 1k | mean | @ 4k |
|---|---|--:|--:|
| `c1` control | 28.329 / 28.072 / 27.993 | 28.131 | 28.167 |
| `c2_s48` | 27.287 / 27.360 | **27.323** | 27.298 |
| `c2_s32` | 27.408 | **27.408** | 27.421 |
| `b_head` | 28.038 / 27.712 | **27.875** | 27.929 |
| `a_linfp8` | 28.516 | 28.516 | 28.656 |

**(b) and (c) compose and are both bit-identical, so the shippable pair is ~1.05 ms.** That is
arithmetic on two separately-measured deltas, not a measurement — the stacked blob has not been
re-run on this object.

---

## 1. (b) `lm_head` — the one that works

`mla.rs` bound `lm_head.weight` at the FULL `vocab × hidden` under TP and emitted the GEMV with
`i[1] = c.vocab`, so all four ranks streamed the same **1.903 GB every token** to compute the same
argmax. The kernel is at **106% of the 6200 GB/s ceiling** — purely a sharding gap.

It was deliberate. `crates/plowrt/src/asset/shard.rs`'s module note and
`crates/devgen/src/lib.rs:849` both record the reason: the column-parallel arm needs `XArgmaxFin`
to fold the per-rank maxima and **that op was a no-op stub** (`interp.hip`: "bodies land in
Phase-2"). Sharding without the fold gives every rank a quarter of the logits and they disagree on
the first token.

**So the work was the fold, and it is now implemented.** `d_xargmax_fin_mega`
(`runtime/amd/op_collective.h`) SUBSUMES `ArgmaxFin`: it does the local `AMAX_BLOCKS` fold, rebases
the winning index into global vocab space, publishes one u64 to every peer and takes the cross-rank
max. Three details that make it small:

* `amax_pack`'s key (`[63:32]` = order-preserving image of the bf16 value, `[31:0]` = `~index`) is
  already max-reducible, so the cross-rank fold is **one unsigned max** — no value/index pair to
  carry. The complement has to be re-formed around the rebase (`~(~i + off) != ~(i + off)`), which
  is the only arithmetic in the op.
* The published u64 rides a **dedicated xctr counter id**, not a `peer_scratch` partial slot.
  `PLOW_CTR_STRIDE` is 128 B per counter, the host zeroes the whole region every step, and the
  region is at the same offset on every rank — 8 peer-visible bytes with no host binding and no
  chance of aliasing a live all-reduce partial (the `partial_A`/`partial_B` parity only gives one
  collective of slack).
* Two ids are consumed: the arrival gate and the value slot. They must differ; the gate is an
  atomic counter.

| baseline | control runs | ctl mean | sharded runs | mean | delta |
|---|---|--:|---|--:|--:|
| **PRE** fold rewrite | 32.466 / 32.544 / 32.331 / 32.237 / 32.174 | 32.350 | 32.109 / 32.064 | 32.087 | **−0.264** |
| **POST** fold rewrite | 28.329 / 28.072 / 27.993 | 28.131 | 28.038 / 27.712 | 27.875 | **−0.256** |

**The same −0.26 on two baselines 4 ms apart.** That is the strongest form this evidence takes: a
pure bandwidth removal should be invariant to how much serial chain surrounds it, and it is. Bytes:
**1903 MB → 476 MB/rank/token**, −1428 MB = **−0.230 ms of floor**, and both measurements land on
it — which is what an op at 106 % of the HBM ceiling is supposed to do.

PRE, both sharded runs sat below the control's MINIMUM of five. POST, the control drifts downward
across the session (28.329 → 28.072 → 27.993) and the two sharded runs sit at opposite ends of it
(run positions 2 and 9), below their local neighbours in both places — −0.16 early, −0.28 late. So
the drift does not produce this either.

### The gate — and why "24 tokens match" would NOT have been enough

`vocab/tp = 38720`, and **all 24 reference ids are below it**, so the standard 24-token check
cannot distinguish a working fold from one that silently returns rank 0's local winner. Two
additions close that:

* **256-token generate, sequences compared BLOB-TO-BLOB on the same interpreter: 256/256
  IDENTICAL**, and **11 of those 256 ids live outside rank 0's shard** (52989, 57673, 64143 →
  rank 1; 84929 → rank 2; …). Those are only reachable through the cross-rank max *and* a correct
  global rebase. Blob-to-blob is the right gate and not a weaker one: the recorded 24-token
  reference string is no longer reproducible on any interpreter at all, because the fold rewrite's
  default (`PLOW_MLA_FOLD_MAP=1`) differs by 1 bf16 ulp on 2 of 4096 outputs and that is enough to
  move a greedy trajectory. A gate that compares the two arms against each other is immune to
  that; a gate that compares them to a frozen string is not.
* **Cross-rank agreement, checked every step**: the harness now reads `in.ids` from all four ranks
  after each token. **0 disagreements in 262 steps × 3 peer ranks.** A no-op fold would leave each
  rank holding its own shard's winner and this counts it.

`glm52_decode.c` also had `MAX_SHARD 128`, which silently truncated a weight dir with more than 128
shards (79 base + one `model-idx-*` per layer overflows it) into `MISSING WEIGHT` on the tail
layers. Raised to 256.

---

## 2. (a) the bf16 weight stream — the split, and why the reachable half buys nothing

Full accounting in **`perf-data/glm52-weight-stream-split.md`**. Summary of the three findings that
change the size of this lever before any work starts:

1. **`lm_head` is BF16 IN THE CHECKPOINT** (verified: `('BF16', [154880, 6144])`). Both prior notes
   counted its 1815 MB toward "the prep dequantised it". There is nothing to convert — it is a
   sharding problem, which is §1.
2. Of the 10 604 MB of bf16, **7492 MB is a whole fp8 checkpoint tensor** (or a 128-aligned slice)
   and can be republished verbatim; **312 MB (`q_rope`) and 312 MB (`v_absorb`) have verbatim fp8
   VALUES but a `[128,128]` grid that does not survive the slice/transpose**; **2496 MB
   (`q_absorb`) is a genuine einsum PRODUCT** with no fp8 form on disk; **2040 MB has no fp8 source
   at all**.
3. Only **5094 MB** of it sits behind an opcode that can read a `[128,128]` grid today —
   `o_proj` (`GEMV_FP8_BLK`), shared gate/up (`DENSE_GLU_FP8_BLK`), shared down (`GEMV_FP8_BLK`).
   The other 5206 MB is behind `GemvQkv` fusions A and G and `MlaMergeFold`, none of which has a
   block-fp8 arm.

So the version of (a) that needs **no kernel work** removes **2547 MB/rank/token = −0.431 ms of
floor**. Implemented (`GLM_LINEAR_FP8=1` + `scripts/glm52_prep_fp8_linear.py`, which publishes the
checkpoint's fp8 bytes and `weight_scale_inv` grids additively into a symlink-farm weight dir —
10.7 GB, no re-prep of the 715 GB base):

| blob | ctx 1k | ctx 4k |
|---|--:|--:|
| bf16 `o_proj` + shared expert (== `glm52_tp4_64k.pkt`) | 32.544 | 32.577 |
| block-fp8 `o_proj` + shared expert | 32.492 | 32.632 |
| **delta** | **−0.052** | **+0.055** |

**Removing 2.5 GB/rank/token — 13% of the whole weight stream, and 0.41 ms of the 3.08 ms floor —
moves the token by nothing.** The arm is definitely live: the harness fails loudly if the
`.weight_fp8` tensors are missing (it did, on the shard cap), and the output changes.

### POST the fold rewrite it is a small REGRESSION, and that is the more useful result

Re-measured on the current object: `a_linfp8` **28.516** against a control of
28.329 / 28.072 / 27.993, i.e. **+0.39 vs the control mean, +0.44 vs its local neighbour, and above
ALL THREE control runs** — and the control is drifting the other way. One sample against three
controls, so treat the magnitude loosely; the sign is not in doubt.

So converting an op from bf16 to block-fp8 **costs time while halving its bytes**, and the reason is
already in the campaign's own data: `GemvFp8Blk` (44) runs at **966 GB/s = 15.6 % of ceiling** where
the bf16 `Gemv` (10) on the same shape family runs at **1728 GB/s = 27.9 %**
(`glm52-decode-attribution.md`). Half the bytes through a kernel that is ~1.8× slower per byte is
break-even by construction; pre-rewrite there was enough serial slack to hide the difference and now
there is not.

**That relocates this lever.** The prep is not the problem and neither is the emitter — 
`gemv_rows_fp8_blk`'s memory-level parallelism is, which is the same "1 load in flight caps the
machine at 65 % of HBM peak" finding the kernel review already documented for ops 45/46. Until the
block-fp8 GEMV is worth its bytes, converting MORE of the stream to fp8 makes decode slower, not
faster — including the `GemvQkvFp8Blk` arm §4 of the split doc recommends. **Fix the kernel first;
the 0.90 ms of floor in the split doc is unreachable through this opcode family as it stands.**

This is §6a's "the decode gap is NOT bandwidth" holding, and it **contradicts the isolated-kernel
roofline**: `perf-data/glm52-kernel-review.md` measures `o_proj` at **83.4%** of the 6200 GB/s
ceiling on an idle GPU, which reads as bandwidth-bound; in the interpreter the same op is at
**27.9%** (`glm52-decode-attribution.md`), because it is waiting on gates. **An isolated-kernel
"% of roofline" is not evidence that a byte-count change will show up end-to-end.**

### The numeric gate: this is not a requantisation

`glm52_prep.py` writes `w_bf16 = round_bf16(fp8 * weight_scale_inv)`; `GEMV_FP8_BLK` computes
`fp8 * weight_scale_inv` in f32. Checked element-wise on six real tensors
(`perf-data/glm52_fp8_residual_check.py`): **`bf16_round(fp8·s)` equals the prepped bf16 tensor
bit for bit**, every element, every tensor. Max residual 2.8e-3 absolute, ≤1.9e-3 relative to
`max|w|` — exactly bf16 epsilon. The fp8 arm is the **un-rounded** form of the same weight, so it
is strictly more precise than what shipped and there is no accuracy question to answer.

Greedy token identity does move — the 256-token generate matches the reference for 4 tokens and
then takes a different (fluent, coherent) path, which is what one bf16 rounding removed from 78
layers of accumulation does to a greedy argmax. The single-block oracle was **not** run: the change
buys ~0 ms, so there is nothing to ship and nothing for the oracle to gate.

---

## 3. (c) the co-resident split — the premise was wrong, and the SIGN DEPENDS ON THE FOLD

Full write-up in `plans/glm52-coresident.md` §4b.

### 3.0 THE HEADLINE: this knob's answer is a property of the interpreter, not of the knob

The SAME byte-identical `cores=2` blob, measured against its own contemporaneous control:

| interpreter | `MLA_MERGE_FOLD` costs | control | `cores=2` s=32 | delta |
|---|--:|--:|--:|--:|
| `VT=256` (as the attribution agent found it) | 8.69 ms | 34.61 | 33.52 | **−1.09 WIN** |
| `VT=32` dispatch fix only | 6.66 ms | 32.35 | 33.67 | **+1.32 LOSS** |
| **+ wave-cooperative body (6495efc, current)** | **~1 ms** | **28.20** | **27.408** | **−0.79 WIN** |

`cores=2` overlaps the shared expert with the routed experts, and whether that pays depends
entirely on how much serial MLA chain is left to hide it in. **All three measurements are right
about their own object.** The campaign's unresolved −3.00 vs −1.09 disagreement needs no villain:
this knob simply cannot be inherited across a kernel change, and neither figure — nor mine —
transfers. Re-derive it after anything that moves the MLA chain, which is the third time this
particular knob has changed sign.

### 3.0b The ~2.2 ms run-to-run mode is GONE, and it was the old fold's

Pre-rewrite, the byte-identical `s=48` blob landed in one of two modes 2.2 ms apart
(31.46 / 31.50 / 31.52 vs 33.66 / 33.58), while the interleaved `cores=1` control stayed stable to
±0.15 — so the mode was specific to the co-resident arrangement, and a single-run A/B of this knob
measured the mode. **On the rewritten fold it is gone:** `s=48` twice, separate processes,
**27.287 / 27.360 — 0.073 ms apart**, with `s=32` at 27.408 between them.

That says what the mode was. The old fold gave 16 of 256 workgroups ~111 µs while 240 left in 4 µs;
a load imbalance that severe, interacting with a CU partition that also carves the chip unevenly, is
the shape that produces a schedule settled once per process. Spreading the fold removed the
interaction, not just the time. **Do not carry the "cannot be A/B'd in one run" caution forward to
this object** — it was correct for the data it was taken on and is now historical.

**The shipping arrangement is ALREADY 8-way.** `perf-data/glm52-kernel-review.md` §4 prices "the
shipping 9-way split" (`GLM_MOE_CORESIDENT=2`) at 0.63 ms/token against 8×32 CU. But `cores`
defaults to **1**, so the routed experts already get `split(8, ·)` = 32 CUs each. Proof: a blob
emitted with no knobs is byte-identical to `glm52_tp4_64k.pkt`. There is nothing to collect.

**`split(9, ·)` FLOORS**, so under `cores=2` the routed experts get 28 CUs and the shared expert
gets the 32-CU *remainder* — the co-resident arrangement was already non-uniform, and "8-way vs
9-way" is the wrong axis. The real one is the shared slice's WIDTH, now a knob (`GLM_SHARED_CUS`;
the default reproduces `split(tk+1, ·)` byte for byte).

Every run below is `--sweep 1024 --steps 65` (median of 65, within-run spread ±0.2 ms), separate
processes, `gpulease -n 4`, same interpreter object:

| variant | shared CUs | routed CUs | runs (ms/token @1k) |
|---|--:|--:|---|
| `cores=1` (default, == `glm52_tp4_64k.pkt`) | — | 32 | 32.466 / 32.544 / 32.331 / 32.237 / 32.174 |
| `cores=2`, s=32 (== `split(tk+1,·)`) | 32 | 28 | 33.658 / 33.673 |
| `cores=2`, s=40 | 40 | 27 | 33.755 |
| `cores=2`, s=48 | 48 | 26 | **31.461 / 31.502 / 31.517** / 33.664 / 33.584 |
| `cores=2`, s=56 | 56 | 25 | 31.769 / 33.693 |
| `cores=2`, s=64 | 64 | 24 | 31.546 |
| `cores=2`, s=32, router off the shared slice | 32 | 28 | 33.775 |

The interleaving is what makes this readable. One lease ran, back to back:
`c1 32.331 → s48 31.502 → c1 32.237 → s48 31.517 → s32 33.673 → s48 33.584 → c1 32.174`. **The same
`s=48` file went fast, fast, then slow, while the control on either side of the slow run measured
32.237 and 32.174** — its two best numbers of the session. Contention cannot produce that shape.

**Every `cores=2` number is drawn from one of exactly two modes: 31.49 ± 0.03 or 33.62 ± 0.06**, and
the same byte-identical `s=48` file has landed in both — three times fast, twice slow. Each number
is a median of 65 steps with a ±0.2 ms within-run spread, and the **`cores=1` control interleaved
between them is stable to ±0.15 ms (32.466 / 32.544 / 32.331 / 32.237 / 32.174)** across the very
same processes and the very same windows. So whatever the mode is, it is specific to the co-resident
arrangement and not a property of the machine at that moment, and it is decided once and then held
for all 65 steps.

**A single-run A/B of `GLM_MOE_CORESIDENT` or `GLM_SHARED_CUS` therefore measures the mode, not the
knob.** That is a sufficient explanation for the campaign's unresolved −3.00 vs −1.09 ms
disagreement about `cores=2`, with neither side having been careless.

**And the mode is not a function of the width.** `s=48` has gone both ways, and so has `s=56`
(31.769 / 33.693). Every width that has been run more than once is bimodal, with the same two
levels. So the promising-looking "s=48 and above are fast" reading from the first pass was the
sampling, not the knob: **the width evidence is exhausted, not encouraging.** The only asymmetry
left is that s≤40 has not been observed fast in 3 runs, which at ~40% fast-rate is worth
p≈0.2 — nothing. Separating width from mode needs ~10 runs per point, ~1 lease-hour each.

A mechanism that would have explained a width effect, recorded so nobody re-derives it from
scratch: the shared expert is **bf16** (18.9 MB/layer — the prep dequantises it) where a routed
expert is **fp8** (9.4 MB/layer), so co-resident it is the LONGER tenant and the MoE window is
`max(shared, routed)`. If that is real it also means the width is coupled to the shared expert's
precision and must be re-derived under `GLM_LINEAR_FP8`.

### 3.4 Where (c) actually lands on the current object

| variant | shared CUs | routed CUs | runs | mean | vs ctl mean | vs local ctl |
|---|--:|--:|---|--:|--:|--:|
| `cores=1` (default, == `glm52_tp4_64k.pkt`) | — | 32 | 28.329 / 28.072 / 27.993 | 28.131 | — | — |
| `cores=2`, s=32 (== `split(tk+1,·)`) | 32 | 28 | 27.408 | 27.408 | **−0.72** | −0.62 |
| **`cores=2`, s=48** | 48 | 26 | 27.287 / 27.360 | **27.323** | **−0.81** | **−0.71** |

Both bit-identical to `cores=1` over 256 generated tokens. The width still barely matters — s=48
beats s=32 by 0.09, inside the control's own spread — so the lever is the co-residency, not the
partition, and §4b's search over widths remains a dead end.

**The honest statement of (c) is therefore two separate claims:**

1. **The 0.63 ms the kernel review priced against a "shipping 9-way split" is not collectible**,
   because `cores` defaults to 1 and the routed experts already get 8×32 CUs. That is structural
   and needs no measurement — a knob-free blob is byte-identical to `glm52_tp4_64k.pkt`.
2. **A different −0.8 ms IS collectible right now** by turning `cores=2` on, bit-identically — but
   only on this interpreter, and §3.0 is the caveat that matters more than the number.

`GLM_ROUTER_OFF_SHARED=1` (keep the wave-starved router score GEMV off the shared expert's CUs so
the shared expert, gated only on `c_rn2`, can start ~11 us/layer earlier): **+0.12 ms. Nothing.**

---

## 4. Two things that will bite the next agent

**`cargo test -p devgen --test tuned_tile_selection` fails after ANY edit to `runtime/amd/interp.hip`.**
Isolated by stashing only the two runtime files: with them stashed the test passes 4/4, with them
applied it fails with *"no qualified measurement reached the compiler … every record is stale
against the new build digest"*. The tile-tuning records are keyed on a digest of the interpreter
build, so touching the interpreter invalidates all of them. It is a measurement-freshness gate, not
a correctness one, but it will fail for the merge-fold and prefill agents too, and it is not their
bug either. The fix is `tunedb-gemm ingest`, not a code change.

**`glm52_decode.c` had `MAX_SHARD 128` and overflowed SILENTLY.** 79 base shards plus one
append-only `model-idx-*` shard per layer (both the DSA indexer prep and the new fp8-linear prep add
one) exceeds it; `st_open` then just stops mmapping and the bind loop reports `MISSING WEIGHT` for
the tail layers — which is easy to filter out of a run log and mistake for a crash. Raised to 256.

## 5. Reproduction

```bash
# blobs (byte-identical to glm52_tp4_64k.pkt with no knobs set)
GLM_FULL=1 PLOW_FP8=1 plowc --hf-dir /home/lava/models/GLM-5.2-plow --emit devblob \
    --max-ctx 65536 --n-cu 256 --num-gpus 4 --no-rope-gen --out base.pkt
GLM_SHARD_HEAD=1  ... --out head.pkt          # (b)
GLM_LINEAR_FP8=1  ... --out linfp8.pkt        # (a), needs the -q weight dir below
GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 ... --out s48.pkt   # (c)

# (a) weight dir: symlinks the 79 prepped shards + 78 fp8 shards (10.7 GB, ~5 s)
python3 scripts/glm52_prep_fp8_linear.py \
    --src ~/.cache/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/<sha> \
    --base /home/lava/models/GLM-5.2-plow --out /home/lava/models/GLM-5.2-plow-q

# run (ROCm tooling OUTSIDE nix; ~4 min weight load per blob, not a hang)
gpulease -n 4 glm-ab sg render -c '
  PLOW_INTERP=i_wt.elf ./glm52_decode base.pkt /home/lava/models/GLM-5.2-plow \
      --tp 4 --sweep 1024,4096 --steps 65 --gen 256'
```

`--gen` now composes with `--sweep` in ONE process (the 4-minute weight load is the entire cost of
a run, and paying it twice per blob to get the ids separately prices an afternoon of lease time).

Before spending a lease on a new blob, run the offline pre-flight — it applies the harness's own
`glm_col`/`glm_row`/replicated size rules to every declared tensor against the weight dir and
reports a declare-vs-disk mismatch in seconds instead of after the load:

```bash
python3 perf-data/glm52_check_bind.py <blob.pkt> <weight-dir> 4
```
