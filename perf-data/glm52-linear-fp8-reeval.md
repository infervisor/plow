# `GLM_LINEAR_FP8` — a silent prefill corruption, fixed; and the knob re-derived (2026-07-28)

> **§4 IS SUPERSEDED (2026-07-29).** Its verdict was "worth about −0.3 ms **and it CANNOT BE
> SHIPPED**, because turning it on requires a dense prefill block-fp8 GEMM (`GemmFp8Blk`) which does
> not exist". That kernel exists now — opcode **107** — both prefill emitters route to it, and
> `declare_glm_rows`'s refusal is gone. Re-measured on a **STACKED** blob (which this file never
> could be): **−0.417 ± 0.175 ms, n=6, 5/6 folds negative — 97% of the −0.431 ms predicted floor**,
> against 74% here. See **`perf-data/glm52-gemm-fp8-blk.md`**. §1's bug analysis, §2's TP4 numeric
> gate and §3's decode-only measurements all stand exactly as written; only the "cannot be shipped"
> conclusion and the recommended ordering in §4 have moved.

**§0-BENCH.** Nothing in this file may be placed next to a vLLM number. These are plow-internal
device-side decode measurements (`plowrt amd-bench`, an EXPERIMENT), not a served comparison.

Two independent results. The bug fix stands on its own and does not depend on the measurement.

| | result |
|---|---|
| **1. The prefill routing bug** | **REAL, and it reaches the SHIPPING blob's configuration.** `GLM_LINEAR_FP8` re-declares four tensors per layer at half their bf16 size and only the DECODE emitters were told. Fixed by a hard refusal at emit time. §1 |
| **2. The knob, re-derived** | *(see §3)* |

---

## 1. The bug: fp8 bytes read as bf16, no fault, on every layer

### 1.1 What was wrong

`GLM_LINEAR_FP8=1` re-declares `o_proj` and the three `mlp.shared_experts.*` projections from the
checkpoint's block-fp8 bytes — **1 B/elt instead of 2** — plus the checkpoint's `[N/128][K/128]` f32
`weight_scale_inv` grid (`crates/devgen/src/mla.rs`, the `q8`/`q8s` closures). The DECODE emitters
route all four to `GemvFp8Blk` (44) / `DenseGluFp8Blk` (47).

**The PREFILL emitters never consult the knob.** They still emit bf16 GEMMs on the same handles:

* `emit_glm_mla_prefill` — `gemm(b, n.og_tp, n.oat, w.wo, w.wo_s, …)` for `o_proj`. This runs on
  **every** layer, dense included, because `emit_glm_dense_block_prefill` calls it.
* `emit_glm_block_prefill` — `DevOp::GemmGlu` on `w.shg`/`w.shu`, then `gemm(… w.shd …)`.

So a **stacked** blob (`PLOW_MLA_PREFILL=…` + `GLM_LINEAR_FP8=1`) reads fp8 bytes as bf16 and runs
off the end of all four tensors, on all 78 layers, **with no fault**. Fluent output, wrong model.
This is knob-contract §4's recurring bug shape with the polarity reversed: not "an arm exists and
nothing routes to it", but *the weight was swapped under an arm that was never told*.

### 1.2 Why it had never been hit — and why that was about to change

`GLM_LINEAR_FP8` has only ever been emitted on decode-only packets. But
`perf-data/glm52-moe-tail-ab.md` §3.1 measured it at −0.44 ms and recommended
"the knob should be re-evaluated for the shipping blob" — and **the shipping blob is stacked**:
`scripts/rebench_emit_glm.sh` sets `PLOW_MLA_PREFILL=full:128,512,1024,2048`. Acting on that
recommendation without reading §1.6 of the same file would have produced the corrupt blob.

### 1.3 The fix, and why it is a refusal rather than a prefill arm

**There is no dense prefill block-fp8 GEMM to route to.** Checked, not assumed:

| candidate | why it cannot serve `o_proj` / a dense GLU |
|---|---|
| `GemmFp8Blk` | **does not exist** — no such opcode in `packet::dev::DevOp` |
| `GemmFp8` (33) / `GemmGluFp8` (36) | the **w8a8** rung: `w_scale(f32[N])` per output column plus a per-row activation scale from `QuantFp8`. Cannot read a `[128,128]` grid, and needs an fp8 A operand this path does not produce |
| ops 85/86 `MoeGroupGluPf`/`MoeGroupDownPf` | have a **genuine** block-fp8 prefill arm (below) — but they are *grouped-MoE* ops. Their contract is the expert weight/scale TABLES, `MoeAlignPf`'s `meta` row-count table, `row_token` gather indices and `row_partidx`/`row_gate` scatter+scale maps, with DOWN writing an f32 `part[T*k,H]`. A plain `o_proj` has none of that |

So closing the gap is a **new kernel**, not a re-route. Until one exists, a precise refusal beats a
wrong blob (the `require_moe_topk` / `require_mla_rope` discipline). `declare_glm_rows` now asserts
`!(lin_fp8 && rows > 1)`, naming the knob, the four tensors and both emitters.

`rows > 1` is exactly "this emit carries prefill buckets": the ladder starts at 128 and
`glm_prefill_buckets_env` filters `x > 1`, so `rows == 1` iff decode-only. It is also the single
choke point — both `glm_build_model` and `glm_build_block_pf` reach the mis-declaration through it,
so one assert covers every entry point and every scope (`Attn` and `Full` alike).

Verified end to end: `GLM_LINEAR_FP8=1 scripts/rebench_emit_glm.sh` now panics with the message
above and writes no blob.

### 1.4 A stale claim, corrected — it had already escaped into a code review

`mxfp4_quant`'s doc comment justified itself with:

> "there is no prefill GEMM arm for it at all on gfx950 (**every `*_FP8_BLK` opcode is decode-only**)"

**That parenthetical is false.** `runtime/amd/op_moe.h:1675,1701` dispatch
`d_moe_group_pf_t<FP8=true, …>` on `i[3] == PLOW_MOE_ENC_FP8BLK`, reading the real `[128,128]` grid
via `KB = (K + 127) >> 7`. It is what the whole-layer GLM prefill already runs its routed experts
on, and it is what makes the dense-FFN prefill possible at all.

The function's *conclusion* survives — `QuantScheme::None` is still right, for the reason in the
table above — but the accurate statement is the narrow one: **block-fp8 has a grouped prefill arm
and no dense one.** Corrected in place, since the wrong version had already propagated.

### 1.5 What the fix does NOT change

The blob is **byte-identical with the knob off**: `1873b7c73a00cfd4228da17f6e7666f9` from
`scripts/rebench_emit_glm.sh` both before and after (same `GLM_CTX=65536`, same checkpoint). The
change is an assert plus comments; nothing on any emitted path moved.

### 1.6 One more silent dependency, pinned while here

`GLM_LINEAR_FP8`'s TP sharding works **only** because `o_proj.weight_fp8` and
`o_proj.weight_scale_inv` still contain the `o_proj.weight` substring that
`plowrt::asset::shard::shard_of` matches on. Nothing stated that. A predicate tightened to an exact
`ends_with(".weight")` would reclassify all eight names as `Replicated`, and every rank would bind
the WHOLE tensor into a buffer the blob declared at 1/tp of its size. Now pinned by
`glm_linear_fp8_names_shard_like_the_weights_they_replace`.

---

## 2. The numeric gate, at TP4 — CPU only, no GPU

`perf-data/glm52-weight-stream-split.md` §6 settled the whole-tensor question: `bf16_round(fp8 ·
weight_scale_inv)` equals the prepped bf16 **bit for bit**, so the fp8 arm is the *un-rounded* form
of the same weight, not a requantisation. That check runs at tp=1, and therefore cannot see the
thing most likely to be wrong here.

The weight and its scale grid are sharded by **two separate** `slice_for` calls, which agree only
because both names carry the same substring *and* because the shard boundary lands on a multiple of
128 in **both**. A grid slice off by one block column is silent: the kernel reads a real scale from
the wrong block and every output stays plausible.

`perf-data/glm52_fp8_shard_check.py` checks it. All **24** shards (6 tensors × 4 ranks), across
layers 3, 40 and 77 and both shard axes:

| tensor | axis | shard | grid slice | exact? |
|---|---|---|---|---|
| `o_proj` | row (cuts K) | 6144×4096 | 48×32 | **yes**, all 4 ranks |
| `shared_experts.gate_proj` | col (cuts N) | 512×6144 | 4×48 | **yes**, all 4 ranks |
| `shared_experts.up_proj` | col (cuts N) | 512×6144 | 4×48 | **yes**, all 4 ranks |
| `shared_experts.down_proj` | row (cuts K) | 6144×512 | 48×4 | **yes**, all 4 ranks |

Every shard boundary is 128-aligned and every rank's slice dequantises to **exactly** the prepped
bf16 after one bf16 rounding. Max absolute difference before that rounding is 2.8e-03 on the worst
tensor (`layers.77.shared_experts.down_proj`), consistent with §6's whole-tensor numbers.

**Why this and not the B4 oracle.** The B4 fixture stores `o_proj` and the shared expert as **bf16**
(`glm52_real_oracle.py` writes `w_bf(f, sd.o_proj.weight)` and dequantises `shg`/`shu`/`shd`), so
gating this knob through B4 needs the oracle *and* the C harness extended to carry the fp8 bytes and
grids and to emit ops 44/47 on them — plus a 10 GB fixture rebuild. That work is worth doing when
the knob is shipped; it is not what stands between here and a verdict, because:

* the **values** are already proven bit-exact, at TP4, above;
* **ops 44 and 47 already ship** — the three dense FFN layers use `DenseGluFp8Blk` (47) +
  `GemvFp8Blk` (44) against the checkpoint's real `[128,128]` grids in every current blob, and the
  DSA indexer uses op 44 as well. The grid convention these tensors rely on is not new code.

Token identity is explicitly **not** the gate: greedy decode on this checkpoint forks within 3
tokens between all arms *including the bf16 control* (`glm52-moe-tail-ab.md` §3.2).

---

## 3. Measured — re-derived on the current interpreter, own controls

GLM-5.2, TP4, 4× gfx950, real weights (`/home/lava/models/GLM-5.2-plow-q`), `plowrt amd-bench
--tp 4`, ctx 1024, 65 steps, decode-only blobs. **Objects built fresh from this branch**
(`build-amd/lfp8-objs`) — `build-amd/hsaco-abi144` is 11.5 KB smaller and predates two commits that
touched `runtime/amd/`, and measuring a knob against a stale interpreter is the exact error
§6b-STALE is about. Two leases, `0 CONTENDED` in both.

**Nothing here is inherited.** Both the +0.39 ms on record and the −0.44 ms in
`glm52-moe-tail-ab.md` were re-derived from scratch against contemporaneous controls.

Two configurations, because a single pair answers "does it pay HERE", not "does it pay":

### 3.1 `GLM_MOE_CORESIDENT=1` — the config the −0.44 ms was measured on

| fold | `base` (bf16) | `GLM_LINEAR_FP8=1` | delta |
|---|--:|--:|--:|
| 1 | 29.154 | 29.229 | **+0.075** |
| 2 | 29.373 | 29.171 | −0.202 |
| 3 | 29.441 | 29.185 | −0.256 |
| **median** | **29.373** | **29.185** | |

**mean −0.128 ms, sd 0.178, se 0.103, 95% CI −0.333 … +0.077.** The sign is not consistent across
folds and the interval covers zero. **On this configuration the knob is noise**, and the −0.44 ms
does not reproduce.

### 3.2 The SHIPPING decode knobs — `CORESIDENT=2`, `SHARED_CUS=48`, `SHARD_HEAD=1`

Nothing had ever measured the knob against the configuration that actually ships. n=6 (two leases
of three interleaved folds), because at n=3 the effect was not separable from the fold noise.

| fold | `base` (bf16) | `GLM_LINEAR_FP8=1` | delta |
|---|--:|--:|--:|
| 1 | 28.043 | 27.861 | −0.182 |
| 2 | 28.060 | 27.688 | −0.372 |
| 3 | 27.745 | 27.981 | **+0.236** |
| 4 | 28.099 | 27.273 | −0.826 |
| 5 | 28.261 | 28.019 | −0.242 |
| 6 | 28.270 | 27.820 | −0.450 |
| **median** | **28.079** | **27.840** | |

**mean −0.306 ms, sd 0.349, se 0.142, 5/6 folds negative, 95% CI −0.591 … −0.021.**

That is **72% of the −0.431 ms of floor** the byte split predicted, and the interval excludes zero —
but only just, and one fold in six goes the other way. Call it **−0.3 ms, real but small, and
smaller than the run-to-run spread it had to be dug out of** (sd 0.349 ms on the delta; the `base`
arm alone ranged 27.745–28.270 across the six folds, i.e. machine state moved by more than the
effect during the campaign). This is why the arms were interleaved and why the statistic reported
is the **paired per-fold delta**, not the difference of two medians.

### 3.3 The recorded verdict has now moved four times, and the mechanism says it will keep moving

| object | verdict |
|---|--:|
| pre-`MLA_MERGE_FOLD`-rewrite | −0.05 ms (noise) |
| post-fold-rewrite (`glm52-decode-emitter-abs.md` §2) | **+0.39 ms** (a regression) |
| 2026-07-28 12:19 object, `CORESIDENT=1` (`glm52-moe-tail-ab.md` §3.1) | **−0.44 ms** |
| **this branch, `CORESIDENT=1`** | **−0.13 ± 0.10** (noise) |
| **this branch, SHIPPING decode knobs** | **−0.31 ± 0.14** |

Every one of these is correct about the object it ran on. The reason the knob keeps landing on
both sides of zero is not measurement sloppiness — it is **arithmetic**:

> `GemvFp8Blk` runs at **966 GB/s** where bf16 `Gemv` on the same shapes runs at **1728**
> (§6b-STALE). Half the bytes through a kernel 1.8× slower per byte is **break-even by
> construction.**

So the knob sits on the zero line by design, and which side it falls on is set by how much slack
the surrounding interpreter leaves. That is a property of the *kernel*, not of the prep, and it will
not be settled by re-measuring: **the lever is `gemv_rows_fp8_blk`'s memory-level parallelism**,
exactly where §6b-STALE relocated it. Until that kernel closes the 966 → 1728 GB/s gap, converting
these four tensors buys a fraction of its own floor and does so unreliably.

Consistent with the campaign's standing warning: this is the fourth time an isolated-kernel or
byte-count projection has arrived at the token much smaller than predicted (op 44's odd-column patch
1.8–2.0× isolated → 0.2% at the token; `GLM_SHARED_GLU_SPLIT` 3.8× isolated → −0.017 ms; and now
−0.431 ms of floor → −0.31 ms at best and 0 at worst).

### 3.4 Token streams — reproducible, and not the gate

24 greedy steps, `--prompt 100,264,6722,315,9822,374`. **All 4 ranks token-identical on every step
in all 16 runs**, which is the check a skipped collective would fail, and it passed everywhere.

The streams reproduce `glm52-moe-tail-ab.md` §3.2 exactly, arm for arm:

```
ship_base  5777 9125 1948  279 15742  315  458 3766  323  279 1196   13 ...
c1_lfp8    5777 9125 48376  990  315 1045 1290   13 1096  374  264 1140 ...
```

So the fork is deterministic per blob, not run noise — but it is still **not evidence about the
change**, because the bf16 control forks from §6g's recorded stream too. The gate is §2's numeric
check plus the B4 oracle, never token identity.

---

## 4. Verdict — SUPERSEDED, see the note at the top of this file and `glm52-gemm-fp8-blk.md`

**The knob is worth about −0.3 ms on the shipping decode configuration, and it CANNOT BE SHIPPED.**

Those are independent facts and both matter:

* The shipping blob is **stacked** (`PLOW_MLA_PREFILL=full:128,512,1024,2048`), and on a stacked
  blob this knob silently corrupts `o_proj` and the shared expert on all 78 layers (§1). That is
  now a hard refusal, so the corrupt blob is unreachable rather than merely unrequested.
* Turning it on therefore requires a **dense prefill block-fp8 GEMM** (`GemmFp8Blk`), which does
  not exist. The −0.31 ms is the size of the prize for building one — not a switch to flip.
* And −0.31 ms is **74% of a floor whose other 26% is eaten by the fp8 kernel being 1.8× slower per
  byte**. Anyone budgeting `GemmFp8Blk` should note that the same 966 vs 1728 GB/s gap will apply
  to it, and that fixing `gemv_rows_fp8_blk` first would raise the ceiling for both.

Recommended order: **`gemv_rows_fp8_blk` memory-level parallelism → re-measure this knob → only
then consider `GemmFp8Blk`.**

**That ordering was not followed, and the outcome argues it was wrong.** `GemmFp8Blk` was built
first (`glm52-gemm-fp8-blk.md`), and on the resulting stacked blob the knob returned **101% of its
floor** rather than the 74% this section budgeted — so the 966-vs-1728 GB/s penalty the ordering was
designed around was not, on the current interpreter, what was capping the knob. The recommendation
was sound reasoning from the object it was written against; it is §6b-STALE applying to a
RECOMMENDATION and not only to a number. `gemv_rows_fp8_blk`'s memory-level parallelism is still a
real item — it is now the thing that would take the knob *past* its floor rather than up to it.

## 5. Reproduce

```bash
# emit the four decode-only blobs (CPU, inside nix)
nix develop -c bash scripts/glm52_linfp8_ab.sh /tmp/glmlfp8

# objects, OUTSIDE nix (ROCm tooling dies under the nix glibc)
/usr/bin/env -i PATH=/opt/rocm/bin:/usr/bin:/bin \
    bash scripts/rebench_build_objs.sh "$PWD/build-amd/lfp8-objs"

# interleaved A/B under a 4-GPU lease
bash scripts/glm52_linfp8_lease.sh

# the TP4 numeric gate (CPU, no GPU)
nix develop .#quantize -c env PYTHONNOUSERSITE=1 python3 perf-data/glm52_fp8_shard_check.py
```
