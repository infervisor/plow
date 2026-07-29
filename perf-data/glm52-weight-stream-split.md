# GLM-5.2 TP4 decode — what is actually bf16 in the weight stream, and which half of it is reachable

**Measured/derived 2026-07-28.** Byte counts are read off the **checkpoint headers** of
`zai-org/GLM-5.2-FP8` plus the emitted TP4 shard widths — no estimates. Reproduce with
`perf-data/glm52_weight_stream_audit.py`.

**§0-BENCH.** Nothing in this file may be placed next to a vLLM number. plow-internal accounting.

---

## 0. The claim under test

`perf-data/glm52-decode-attribution.md` and `perf-data/glm52-kernel-review.md` both found the active
weight stream at **19.2–19.8 GB/rank/token** on a *block-fp8 checkpoint*, and both attributed it to
`scripts/glm52_prep.py` **dequantising to bf16**. Attribution sized the prize at **−1.08 ms of
floor**, the kernel review at **−1.13 ms**.

Both numbers are **too high**, for two independent reasons, and the reachable part is smaller again.

## 1. The stream, classified by WHERE THE BF16 CAME FROM

Per rank per token, TP4, ctx 1k, 78 layers (3 dense + 75 sparse), `nh_l=16`, `imoe_l=512`,
`di_l=3072`:

| tensor | MB/layer | ×layers | MB/token | class |
|---|--:|--:|--:|---|
| `q_a_proj` `[QL,H]` | 24.00 | 78 | 1872.0 | **A** |
| `derived.kv_a_latent` `[DK,H]` | 6.00 | 78 | 468.0 | **A** |
| `derived.k_rope` `[DR,H]` | 0.75 | 78 | 58.5 | **A** |
| `derived.q_rope` `[nh_l*DR,QL]` | 4.00 | 78 | 312.0 | **A′** |
| `o_proj` `[H,nh_l*VD]` | 48.00 | 78 | 3744.0 | **A** |
| shared `gate+up` 2×`[imoe_l,H]` | 12.00 | 75 | 900.0 | **A** |
| shared `down` `[H,imoe_l]` | 6.00 | 75 | 450.0 | **A** |
| `derived.q_absorb` `[nh_l*DK,QL]` | 32.00 | 78 | 2496.0 | **B** |
| `derived.v_absorb` `[nh_l*DK,VD]` | 4.00 | 78 | 312.0 | **B′** |
| `mlp.gate` (router) `[E,H]` | 3.00 | 75 | 225.0 | **C** |
| `lm_head` `[V,H]` | 1815.00 | 1 | 1815.0 | **C** |
| routed experts, fp8 | 72.00 | 75 | 5400.0 | fp8 |
| dense FFN, fp8 | 54.00 | 3 | 162.0 | fp8 |
| **total** | | | **18214.5 MB = 19.10 GB** | |

* **A — fp8 on disk, dequantised only for convenience.** A WHOLE checkpoint tensor (or a
  128-aligned slice of one), so its fp8 bytes *and* its `[128,128]` `weight_scale_inv` grid can be
  republished **verbatim**. **7492.5 MB.**
* **A′ — fp8 bytes verbatim, grid does NOT survive the slice.** `q_rope` is `q_b_proj` rows
  `[h*256+192, h*256+256)`. Each 64-row slice lies inside ONE 128-row scale block (`2h+1`), so no
  value changes — but the *extracted* tensor's rows 0..127 come from two different scale rows, which
  is not a `[128,128]` grid. Needs a 64-row scale block or a requantisation. **312 MB.**
* **B — a genuine PRODUCT of two fp8 tensors.** `q_absorb = einsum(k_nope_w, q_b_nope)` has no
  native fp8 form on disk and cannot be produced without requantising. **2496 MB.**
* **B′ — a TRANSPOSE, not a product.** `v_absorb = value_w^T` where
  `value_w = kv_b_proj[:, 192:448, :]`. **Every fp8 value is verbatim**; only the grid breaks — the
  per-head slice starts at row 192 inside a 448-row head stride, so after the transpose the
  v-blocking boundaries land at 0/64/192/256 instead of 0/128/256. Values are exact; the scale
  layout is the whole problem. **312 MB.**
* **C — no fp8 source exists.** `mlp.gate` and **`lm_head` are BF16 IN THE CHECKPOINT**
  (`lm_head.weight ('BF16', [154880, 6144])`, verified). **2040 MB.**

## 2. Correction 1 — `lm_head` is not part of this lever at all

Both prior notes counted `lm_head`'s 1815 MB toward the "prep dequantised it" total. It is bf16 on
disk. There is nothing to convert. `lm_head`'s 1.9 GB is a **SHARDING** problem
(vocab-column-parallel + `XArgmaxFin`), not a precision one.

That removes **951 MB / −0.153 ms** from the −1.08 ms estimate before any work starts.

## 3. Correction 2 — the floor is not the token

Halving bytes only halves *time* where the op is bandwidth-bound. Measured today: `o_proj` is at
**83.4 %** of the 6200 GB/s ceiling (kernel review row 5) and the shared expert's GEMV is at
**~11 %**. Converting both removes the same bytes; only the first is guaranteed to remove the time.

## 4. Which of it an EXISTING opcode can consume

The conversion needs an opcode that reads a `[128,128]` f32 grid. At decode that is `GEMV_FP8_BLK`
(44) and `DENSE_GLU_FP8_BLK` (47).

| bf16 stream | today's opcode | block-fp8 arm? | MB/token |
|---|---|---|--:|
| `o_proj` | `Gemv` (10) | **yes — 44** | 3744.0 |
| shared gate/up | `GemvGlu` (19) | **yes — 47** | 900.0 |
| shared down | `Gemv` (10) | **yes — 44** | 450.0 |
| `q_a_proj` + `kv_a_latent` + `k_rope` | `GemvQkv` (22) fusion A | **NO** | 2398.5 |
| `q_absorb` + `q_rope` | `GemvQkv` (22) fusion G | **NO** | 2808.0 |
| `v_absorb` | `MlaMergeFold` (57) | **NO** | 312.0 |
| router, `lm_head` | `Gemv` (10) | n/a — bf16 on disk | 2040.0 |

So **5094 MB of the 10 604.5 MB of bf16 is reachable with the opcodes that exist**, saving
**2547 MB = −0.431 ms of floor**. The rest is blocked on:

* **`GemvQkv` has no block-fp8 arm** (5206.5 MB behind it). Un-fusing into three/two
  `GEMV_FP8_BLK` would reach it, but fusion A exists *precisely* to stop `kv_a` (N=512) and `k_rope`
  (N=64) starving CUs, and it measures **83.0 %** of ceiling as fused — the kernel review's
  explicit "do not un-fuse". The right fix is a `GemvQkvFp8Blk` arm, which is kernel work.
  *A cheaper alternative worth recording:* the three fusion-A weights CONCATENATE into one
  `[2624, H]` block-fp8 matrix whose grid is legal (2048 and 2560 are both multiples of 128), so a
  single `GEMV_FP8_BLK` would do — if the packet could express `qlr`/`ckvraw`/`krr` as offset views
  of one activation buffer, which it cannot today.
* **`MlaMergeFold` takes `W_uv` as `const bf16*`** with no encoding parameter (312 MB). NOTE for
  whoever picks this up: the wave-cooperative rewrite (6495efc) makes `W_uv` the op's dominant
  read and loads it `dwordx2`-wide, so an fp8 arm there is now both more valuable and more
  invasive than when this was written.
* **`q_absorb` (2496 MB) would have to be requantised** — a real numeric change, and the only part
  of this lever that has an accuracy question attached.

## 5. Ceiling, honestly

> **UNITS — read this before adding a row.** "MB" throughout this document means **MiB (2^20)**, not
> 1e6. It is provable from the table's own inputs: `q_a_proj` is `2048·6144·2 B = 24.000` MiB exactly
> (24.00 above), and §4 states "18214.5 MB = 19.10 GB", which holds only as MiB→GB(1e9).
>
> The rows below were WRONG until 2026-07-29 for exactly this reason: the three component rows had
> been computed at MB=1e6 while the total was computed at MiB, so the table did not sum to itself
> (0.411 + 0.395 + 0.050 = 0.856, against a stated total of 0.897). The bytes were never in doubt —
> only the divisor. Every component row is now ~4.8% larger, and the total is unchanged.
>
> This propagated: `-0.411` was quoted as the projected floor in five other documents and in
> `mla.rs`'s docstring table, which made the measured `-0.417 ± 0.175` look like **101% of floor**
> when it is **97%**. Corrected everywhere in the same commit. A measurement that appears to reach
> exactly 100% of a prediction deserves suspicion of the prediction, not confidence in the result.

| scope | MB removed | ms of floor |
|---|--:|--:|
| what the existing ISA can consume (`o_proj` + shared expert) | 2547 | **−0.431** |
| + a `GemvQkvFp8Blk` arm (fusions A and G, minus the `q_rope` grid problem) | +2447 | −0.414 |
| + requantising `q_absorb` and re-blocking `v_absorb`/`q_rope` | +312 | −0.053 |
| **all of class A+B** | **5306** | **−0.897** |
| `lm_head` — belongs to the SHARDING lever, not this one | (1428 at TP4) | (−0.242) |

The campaign's "−1.1 ms from fp8" is really **−0.90 ms of floor spread over three separate pieces
of work**, of which **−0.43 ms needs no new kernel**, and the floor is a lower bound on the time.

## 6. The numeric question, settled: this is NOT a requantisation

`glm52_prep.py` writes `w_bf16 = round_bf16(fp8 * weight_scale_inv)`. `GEMV_FP8_BLK` computes
`fp8 * weight_scale_inv` in f32. Checked element-wise against the real tensors
(`perf-data/glm52_fp8_residual_check.py`):

| tensor | max abs diff | rel. to max|w| | `bf16_round(fp8·s) == prep bf16`? |
|---|--:|--:|:--|
| `layers.3.self_attn.o_proj` | 4.883e-04 | 1.908e-03 | **exact** |
| `layers.3.shared_experts.gate_proj` | 8.371e-04 | 1.832e-03 | **exact** |
| `layers.3.shared_experts.up_proj` | 4.534e-04 | 9.249e-04 | **exact** |
| `layers.3.shared_experts.down_proj` | 8.371e-04 | 1.331e-03 | **exact** |
| `layers.40.self_attn.o_proj` | 8.371e-04 | 1.531e-03 | **exact** |
| `layers.77.shared_experts.down_proj` | 2.790e-03 | 1.693e-03 | **exact** |

The bf16 tensor on disk is **bit-for-bit** the bf16 rounding of the fp8 product. So the fp8 arm is
not an approximation of what shipped — it is the **un-rounded** form of the same weight, and the
only difference is that the kernel now multiplies in f32 instead of reading a pre-rounded bf16.
**Strictly more precise, with no requantisation anywhere.** Token identity may still move (different
rounding is still different arithmetic), which is why the gate for this change is the oracle and not
bit-identity — but there is no quality regression to hide.
