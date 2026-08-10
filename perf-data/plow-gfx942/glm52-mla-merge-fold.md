# `MlaMergeFold` (op 57) on gfx942 — anatomy, bound, and the token-blocked fold

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **CDNA3-SPECIFIC** — kernel anatomy and bound at this box's MFMA and HBM rates.

2026-08-08, branch `mla-merge` off `worktree-glm52-bringup` @ `daa894b`.

`MlaMergeFold` is the fourth-largest prefill component in GLM-5.2 (8.6% of a MoE
layer's CU budget at T=8192, ~120 ms of TTFT at 8k, ~242 ms at 16k) and both the
cost decomposition and the MoE fusion report flag it as **never examined**. This
is that examination.

**Headline, and it corrects the premise the campaign has been carrying:**
`MlaMergeFold` is **not** the price of the causal KV-split. It is a **batched
GEMM executed one M-row at a time**. The merge over `nsplit` partials is 21% of
it and the KV-split's *marginal* cost is **9%** — the other 91% is the `W_uv`
fold and would be there at `ns=1`. The fold re-reads a 256 KiB weight panel
**once per token**, which is 16.8 GB of L2 traffic per layer at T=8192 to do
8.59 GMAC, and that traffic is the bound.

---

## 0. What was measured, and with what

| | |
|---|---|
| kernel | `d_mla_merge_fold` (`runtime/amd/op_attention.h`), dispatched by `exec_mla_merge_fold` (`runtime/amd/interp.hip`) |
| emit | `crates/devgen/src/mla.rs`, the `MlaMergeFold` packet after `FlashMlaPrefill` |
| shape | GLM-5.2 TP8 prefill: `n_batch=T`, `n_head=nh_l=8`, `DK=512`, `V=256`, `nsplit=2` |
| objects | `env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1 scripts/build_gfx942.sh` |
| kernel instrument | `runtime/bench/amd/glm52_kbench_fold_pf.{hip,cpp}` (**new**), built by `perf-data/build_kbench_fold_pf.sh` |
| ISA | `llvm-objdump --mcpu=gfx942` on the unbundled bench object (same header, same flags) |
| serve instrument | `scripts/bench_speed.sh`, ports 8196, asset `glm52-tp8-final2`, `PLOW_MLA_PF_V2=1` |

The microbench exists because the shipped `glm52_kbench_fold.hip` is a **decode**
bench (`n_batch=1, nh=16, ns=16`, 16 rows of work). Prefill is a different
machine: the emitter folds the token axis into `n_batch`, so the packet is
`T x nh_l` rows — 65 536 at T=8192 — and nothing had ever run the body at that
shape. The new bench reproduces the traced packet cost to within 6%
(1624 µs measured vs 467 kCU-µs / 304 CU = 1536 µs from the census), which is
what licenses using it to iterate without the serve lock.

---

## 1. Anatomy

### 1.1 What the packet is

```
MlaMergeFold  t = [O, Opart, mlpart, Wuv]   i = [n_batch, n_head, V, _, nsplit]
```
with `n_batch = T` (the token axis folded in — `mla.rs`: "the token axis folds
into `n_batch`"), so a "row" is one **(token, head)** pair.

Per row the kernel does two things:

1. **merge** — online-softmax combine of `nsplit` latent partials:
   `olat[d] = (sum_s Opart[row][s][d] * exp(m_s - gm)) / gl`, `d = 0..DK`.
   `DK*nsplit` f32 loads, `DK*nsplit` FMA, result kept **f32 in LDS**.
2. **fold** — `o[v] = sum_l olat[l] * W_uv[h][l][v]`, `v = 0..V`.
   `DK*V` MAC against a `[DK][V]` bf16 panel.

The fold is the same arithmetic a separate `OUvFold` opcode used to do; the two
were fused so the `Olat` HBM round-trip and one dependency gate go away.

### 1.2 The work decomposition, and where it goes wrong

`n_work = n_batch * n_head * ceil(V/VT)`. At GLM TP8 prefill `bh = T*nh_l` is
65 536, so `bh*8 > nblk` and `exec_mla_merge_fold` takes the `VT = 256` arm →
`vtiles = 1` → **`n_work = 65 536` work items of one (token, head) each**, grid-
strided over the 304 workgroups `mla_fold_cus` hands the packet.

The fold map inside a work item (VT=256, VEC=4, `PLOW_THREADS`=512,
`PLOW_WAVES`=8): `NV = VT/VEC = 64` lanes cover the tile row, `LS = 512/64 = 8`
l-slices — **one wave per slice** — each owning a contiguous `BL = DK/LS = 64`
block of `l`, `UN = 4` W_uv rows in flight per group. So the 8 waves *together*
read the panel once per work item.

**Per work item that panel is `DK*V*2 = 256 KiB`, and every byte of it is used
exactly once, for one 256-wide output row.** The panel is 256 KiB against a
32 KiB vector L1, so nothing is caught on chip. `304 % n_head == 0`, so a
workgroup's grid-stride keeps it on ONE head for the whole packet — good for L2
residency (all 8 heads = 2 MiB, comfortably inside a 4 MiB per-XCD L2), and it
is exactly why the traffic is an **L2→CU** stream and not an HBM one.

### 1.3 Bytes and flops per layer, T=8192, TP8, ns=2

| stream | bytes/layer | where from |
|---|---:|---|
| **`W_uv` re-read** | **17.18 GB** | L2 (2 MiB panel set, re-read 8192×) |
| `Opart` read | 268 MB | HBM |
| `O` write | 33.5 MB | HBM |
| `mlpart` read | 1.05 MB | HBM |

| work | per layer |
|---|---:|
| fold MAC | 8.590 G (`T*nh_l*DK*V`) |
| merge MAC | 0.067 G (`T*nh_l*nsplit*DK`) |

The fold is **99.2% of the arithmetic and 98% of the bytes**. The `Opart`
round-trip — the thing the MoE work's analogous case turned out to be about — is
268 MB, which at this part's HBM ceiling is ~50 µs of a 1624 µs packet. **The
DRAM round-trip is not what pays here.** What pays is, exactly as in the MoE
case, the *shape* of the consuming op.

---

## 2. The bound: bandwidth, quantified

Measured, `grid=304`, `block=512`, 3 rounds (`us/pkt`, round spread ≤1.5%):

| shape | shipped body | `W_uv` rate | vector-f32 rate |
|---|---:|---:|---:|
| T=8192 ns=2 | **1624 µs** | 10.58 TB/s | 10.6 TF/s |
| T=8192 ns=1 | 1449 µs | 11.86 TB/s | 11.9 TF/s |
| T=4096 ns=2 | 806 µs | 10.66 TB/s | 10.7 TF/s |
| T=2048 ns=2 | 373 µs | 11.51 TB/s | 11.5 TF/s |

Three independent facts make this **bandwidth-bound on the L2→CU path**:

1. **The achieved `W_uv` rate is a flat 10.5–11.9 TB/s band across every T and
   every `nsplit`** — a rate ceiling, not a work quantity. Time is linear in the
   stream, not in the flops (T=8192 ns=1 and ns=2 do identical fold flops and
   differ by exactly the merge's extra 268 MB).
2. **It is not issue-bound.** The fold's inner group (below) is 4 loads + 1
   `ds_read_b128` + 8 `v_pk_fma_f32` + ~10 unpack/address ops ≈ 44 instructions
   for 16 f32 MAC. At 8 waves/WG = 2 per SIMD and 4 cycles per wave64 VALU op,
   the VALU-issue floor for the whole packet is ~290 µs against 1624 measured —
   **~18% VALU duty**.
3. **It is not latency-bound.** `UN=4` × 8 B × 512 threads × 304 WGs = 5 MB of
   `W_uv` in flight, against the ~1.7 MB Little's-law figure at 10.6 TB/s and
   L2 latency. Raising `UN` to 8 on the shipped body does not move it.

The merge half, priced separately (a fold-free kernel, same walk), behaves the
opposite way:

| | µs/packet | bytes moved | rate |
|---|---:|---:|---:|
| merge only, ns=2 | **348** | 288 MB | 828 GB/s |
| merge only, ns=1 | **201** | 160 MB | 795 GB/s |

800 GB/s is one sixth of this part's HBM ceiling, so the merge is **latency**-
bound, not bandwidth-bound: per token a thread issues one `mlpart` load that
gates the `exp`, one (`nsplit=2`) `Opart` load, an LDS store and a barrier, with
8 waves resident and nothing to overlap.

### 2.1 What the causal KV-split actually costs

`348 − 201 = **147 µs/layer**` — **11.5 ms of an ~1677 ms 8k TTFT, 0.7%**, and
9% of `MlaMergeFold`. The campaign accepted ~120 ms as "the price of ns2". The
real price is 11.5 ms, and it bought −7.7% at 8k. The trade is an order of
magnitude better than it was recorded as.

---

## 3. Stage 2 — is it necessary at this cost?

### (a) Is the kernel shaped for `ns=2`, or paying for generality?

It pays a little, and it costs registers rather than time. The merge is a
runtime `nsplit` loop with an `MS=8` blocked fast path that **prefill never
enters** (`nsplit` is 1 or 2), so every prefill packet runs the scalar tail. The
block is dead code at this shape but is live for decode (`ns=16`). Deleting it
from the *new* arm was worth 231→124 VGPR; deleting it from the shipped body
would be a decode regression. Not a time lever either way — the merge is 21% of
the op and the generality inside it is not what makes it slow (latency is).

### (b) `PLOW_GLM_OFOLD` — can the merge be folded away *and* keep ns2?

`W_ofold` makes the flash epilogue write **normalized bf16** rows into `Opart`
laid out `[t][head][DK]`, deletes the `MlaMergeFold` packet entirely, and lets
the o_proj GEMM absorb `W_uv` (its K grows `nh_l*V = 2048` → `nh_l*DK = 4096`).
It cannot compose with `ns>1` because normalizing in the epilogue needs the
un-split `l`. That much was known.

Re-examined with the cost now priced, the general idea does **not** dominate:

- What `W_ofold` deletes is the whole 1624 µs (at 4k: 806 µs), and what it buys
  back is a **doubled o_proj GEMM**. It measured **−4.0% @4k**. Token-blocking
  deletes 60% of the same op with **no o_proj change**, and projects **−3.9%
  @4k** — the same order, from the same place.
- The difference is what each gives up. `W_ofold` gives up `ns2` (worth −7.7% at
  8k, an order of magnitude more than the merge it removes) and is not
  bit-exact. Token-blocking gives up **nothing**: it composes with `ns2`, and it
  is bit-identical.
- A hypothetical "ofold that consumes 2 partials" would need a separate
  normalize+merge pass — i.e. the 348 µs merge — and would then still pay the
  doubled o_proj to delete a fold that token-blocking already made 2.6× cheaper.
  There is no version of the fusion that is worth building once the fold's
  traffic problem is fixed.

**Conclusion: the earlier "they don't compose" verdict stands, and is now moot —
the fusion was aiming at a cost that has a cheaper fix.**

### (c) Does it round-trip DRAM unnecessarily, and is it about bytes or shape?

`Opart` is 268 MB/layer written by the flash and read by the merge. That is
~50 µs at HBM rate — 3% of the packet. Halving it (the `ns=1` case) moves the
op by 175 µs, of which 147 is the merge's own latency chain, not bandwidth.

So the byte count is not the lever. **The access *shape* is**, and it is the
same finding the MoE work made in its own case (`k=8` strided → 1 contiguous was
−86.9%, halving the bytes only −12.8%): here the same 2 MiB of `W_uv` is fetched
8192 times because the consuming loop has no reuse dimension. Fix the shape and
the bytes follow.

---

## 4. What was built

`PLOW_MLA_FOLD_TB=<G>` — `d_mla_merge_fold_tb` in `op_attention.h`, routed by
`exec_mla_merge_fold`, exposed by `scripts/build_gfx942.sh`. **Default OFF.**

A workgroup takes **TB consecutive token-rows of one head** instead of one row.
The `W_uv` element a lane holds in a register is then consumed by TB
accumulators, so the panel is fetched once per TB tokens and the L2 stream
divides by TB. Nothing else moves: same `NV` lane→column map, same `LS`/`BL`
l-slice split, same `UN`, same shfl-then-LDS fold tree, same `VT=256` branch.

**Bit-identity is structural, not incidental.** For a fixed token `g` the
sequence of adds into `acc[g][k]` is exactly the sequence the shipped body makes
into `acc[k]`: same outer `i` order, same `u` order in a group, same l-block per
wave, same increasing-wave LDS sum. `TB` interleaves **independent** accumulator
chains; it never reassociates one. Verified by `memcmp` of the full 32 MB output
against the shipped body at every T and every `nsplit` tested.

`olds` grows to `TB*DK` floats; `red` does **not** grow to `TB*PLOW_WAVES*VT`
(64 KiB at TB=8, over the arena) — the cross-wave fold runs in chunks of `RB`
tokens, `RB` the largest power of two that fits `PLOW_MLA_FOLD_TB_LDS` = 12288
floats. At TB=8, VT=256 that is `RB=4`: 4096 + 8192 floats = 48 KiB, and the
barrier count is `2*TB/RB = 4` per work item rather than 16.

Guards, all correctness preconditions checked by the caller and none of them a
heuristic: not the `bh*8 <= nblk` decode arm (a different `VT` is a different
fold map and therefore different arithmetic); `V >= VT`, `V % VT == 0`,
`V % VEC == 0` (the TB body carries only the fast map, no scalar fallback);
`n_batch % TB == 0` (no tail); `nsplit < 8` (see below); and
`n_work >= nblk` (do not trade chip fill for locality — at the t=128 bucket
TB=8 would leave 128 of 304 workgroups idle).

### 4.1 The contraction trap, recorded because it is invisible

The natural way to interleave the merge is to hoist the per-split weight:

```c
wt[g] = (m == FA_NEG_INF) ? 0.0f : FA_EXP(m - gm[g]);
acc[g] += pv[g] * wt[g];                 /* -> ONE v_fmac_f32, one rounding */
```

The shipped body writes it the other way round:

```c
acc += (m == FA_NEG_INF) ? 0.0f : (opb[...] * FA_EXP(m - gm));
                                         /* select breaks contraction:      */
                                         /* v_mul + v_add, TWO roundings    */
```

These are not the same number. The hoisted form measured **IDENTICAL at ns=1**
(where `acc` is still 0 and `fma(a,b,0) == a*b`) and **DIFFERED at ns=2** — the
shipped prefill `ns`, i.e. exactly the case that matters. The TB merge therefore
mirrors the shipped expression shape verbatim, and carries **no `MS=8` block**:
the caller refuses `nsplit >= 8`, so both bodies are on their select-then-add
tail and agree bit for bit. (Keeping the block would also have cost `TB*MS*2`
live floats — measured 231 VGPR at TB=8 against 124 without — in a region
prefill never enters.)

---

## 5. ISA, before and after

Fold inner group (one `UN=4` step over `l`), from the unbundled bench object,
`llvm-objdump --mcpu=gfx942`:

| | shipped (TB=1) | **TB=2** | **TB=4** | **TB=8** |
|---|---:|---:|---:|---:|
| `global_load_dwordx2` (W_uv) | 4 | 4 | 4 | 4 |
| `ds_read_b128` (olat) | 1 | 2 | 4 | 8 |
| `v_pk_fma_f32` | 8 | 16 | 32 | 64 |
| f32 MAC per group | 16 | 32 | 64 | 128 |
| bf16→f32 unpack (`v_and`+`v_lshl`) | 8 | 8 | 8 | 8 |
| group size (instrs) | ~44 | 58 | 80 | 124 |
| **W_uv bytes per f32 MAC** | **2.00** | 1.00 | 0.50 | **0.25** |
| instrs per f32 MAC | 2.75 | 1.81 | 1.25 | **0.97** |

The load *count* is unchanged and the unpack count is unchanged — both are
amortised over TB times the arithmetic. The shipped body was already emitting
the right instructions (`global_load_dwordx2`, `ds_read_b128`, packed FMA); it
was simply issuing them once per token.

Kernel-level register/LDS (standalone bench, `__launch_bounds__(512)`):

| | VGPR | spill |
|---|---:|---:|
| shipped | 56 | 0 |
| TB=2 | 66 | 0 |
| TB=4 | 86 | 0 |
| TB=8 | 124 | 0 |

In the shipped **megakernel** the arm changes nothing: `interp_prefill_mla_moe`
stays **VGPR 256 / AGPR 0 / LDS 64 560 / spill 2**, identical to the control.

---

## 6. Kernel-level A/B (3 rounds, grid 304, block 512)

`us/packet`; `bitcmp` is `memcmp` of the whole output against the shipped body.

### T=8192, ns=2 (the 8k prefill chunk, and both chunks of a 16k prompt)

| variant | r1 | r2 | r3 | mean | vs shipped | bitcmp |
|---|---:|---:|---:|---:|---:|:--|
| shipped (TB=1) | 1621.6 | 1625.5 | 1625.9 | **1624.3** | — | REF |
| TB=2 | 931.9 | 947.0 | 939.6 | 939.5 | −42.2% | IDENTICAL |
| TB=4 | 690.2 | 702.4 | 692.3 | 695.0 | −57.2% | IDENTICAL |
| **TB=8** | 617.4 | 612.1 | 616.4 | **615.3** | **−62.1%** | IDENTICAL |
| TB=8, UN=8 | 601.0 | 603.1 | 600.8 | 601.6 | −63.0% | IDENTICAL |
| TB=4, serial merge | 768.5 | 764.1 | 771.5 | 768.0 | −52.7% | IDENTICAL |
| merge only, ns=2 | 347.4 | 348.8 | 348.6 | 348.3 | — | — |
| launch floor | 2.4 | 2.4 | 2.4 | 2.4 | — | — |

### The other buckets

| shape | shipped | TB=8 | ratio |
|---|---:|---:|---:|
| T=4096 ns=2 | 806.5 | 321.3 | 2.51× |
| T=2048 ns=2 | 373.3 | 159.0 | 2.35× |
| T=8192 ns=1 | 1448.9 | 501.4* | 2.89× |

\* the `ns=1` TB=8 datum is from the build one revision earlier, before the merge
half was rewritten into the shipped body's select-then-add expression shape (§4.1).
At `nsplit=1` the two forms are the same number (`fma(a,b,0) == a*b`) and the row is
quoted only for the `ns` scaling; every `ns=2` row in this file is the final kernel.

`TB=8, UN=8` is 2.3% better than `TB=8, UN=4` and costs 164 VGPR against 124.
**`UN=4` (the existing `PLOW_MLA_FOLD_UN` default) is adopted**: 2.3% of this op
is 0.1% of TTFT, and the megakernel is already at its 256-VGPR cap with 2 spills.

### Where the remaining time goes

At TB=8 the `W_uv` stream is 2.15 GB/layer at 3.5 TB/s — **no longer the
ceiling** (the shipped body's 10.6 TB/s was). Of the 615 µs, ~200 µs is merge
(latency-bound on `Opart`/`mlpart`, and the TB interleave already recovered
~78 µs of it: TB=4 serial 768 → interleaved 695) and the rest is fold running at
~41 TF/s of packed-f32 FMA, ~25% of the 163 TF/s `v_pk_fma_f32` peak. **The next
floor is VALU/issue, not memory** — attacking it means MFMA, which requires
`olat` in bf16 and is therefore **not** bit-identical and not in scope here.

---

## 7. Traced span + busy in the real packet stream, T=8192

`PLOW_TRACE_RAW` + `plowrt amd-bench --tp 8 --steps 4 --prompt <8192 ids>` on the real blob
and the real checkpoint, reduced by `scripts/glm52_layer_census.py --layers 6:74` (median over
the 69 MoE layers). **Same blob, same prompt, same runtime binary; only the `hsaco` directory
differs.** This is the wall-independent measurement.

| per MoE layer, T=8192 | control | `PLOW_MLA_FOLD_TB=8` | Δ |
|---|---:|---:|---:|
| **`MlaMergeFold` span** | **1589.1 µs** | **658.2 µs** | **−58.6%** |
| **`MlaMergeFold` busy** | **469 448 CU-µs** | **187 541 CU-µs** | **−60.0%** |
| `MlaMergeFold` busy / 304 CU | 1544 µs | 617 µs | (microbench said 1624 → 615) |
| `MlaMergeFold` share of layer span | 8.4% | 3.7% | |
| `MlaMergeFold` gate wait | 261 CU-µs (0.1%) | 247 CU-µs (0.1%) | — |
| layer span (median) | 18 040.9 µs | 16 827.2 µs | **−6.7%** |
| layer busy (all packets) | 5 009 638 CU-µs | 4 678 941 CU-µs | −6.6% |
| packing efficiency | 91.3% | 91.5% | — |

Every other packet is unmoved, which is the check that matters: `FlashMlaPrefill` 853 661 →
845 498 CU-µs, `MoeGroupDownPf` 860 845 → 846 881, `MoeGroupGluPf` 675 177 → 673 634,
`XReduceTwoShot` 658 346 → 655 605 (all ≤1.6%, i.e. run-to-run). The **entire** layer delta is
`MlaMergeFold`'s 281 907 CU-µs, and the layer-span delta (1213.7 µs) is 1.3× the packet-span
delta (930.9 µs) because the packet also stops being a straggler at the gate.

Layer span × 78 layers projects **−94.7 ms of device prefill at 8k**.

Two provenance notes:

- **The `amd-bench` prefill WALL of the tb8 arm is not usable and is not quoted.** It read
  12 487.8 ms against the control's 1403.7 ms because another workload's job was resident on the
  same 8 GPUs for the whole tb8 run (`fgn=1` in the log). Per-packet spans are medians over 69
  layers and survive that; a single wall does not. The direction is also self-evidently not a
  contention artefact — contention inflates spans, and tb8's are *lower*.
- **Both arms produced identical tokens** on that run: prefill first token `407` on all 8 ranks,
  and the 4 greedy decode steps `[154842, 40, 5293, 697]` on both arms, all 8 ranks
  token-identical. Decode TPOT 29.337 vs 29.561 ms/token — the decode objects are byte-identical
  between the arms, so that row is a null by construction and reads as one.
---

## 8. Interleaved served A/B

`scripts/bench_speed.sh <asset> 8196 auto`, `IN_LENS="4096 8192 16384"`, `CONCS=1`,
`NPROMPT=3`, `OUTLEN=16`, `PLOW_MLA_PF_V2=1`. Arms alternate order per round
(`ctl tb8` / `tb8 ctl` / `ctl tb8`), one server per arm, same blob and same
checkpoint — the arms differ **only** in which `hsaco` directory the asset's symlink
points at. `ttft_ms` (mean of 3).

| round | arm | 4096 | 8192 | 16384 |
|---|---|---:|---:|---:|
| 1 | ctl | 962.5 | 1667.4 | 3613.1 |
| 1 | **tb8** | *voided* | *voided* | *voided* |
| 2 | **tb8** | **929.2** | **1577.6** | **3389.4** |
| 2 | ctl | 972.0 | 1674.0 | 3618.7 |
| 3 | ctl | 963.4 | 1666.7 | 3610.4 |
| 3 | **tb8** | **929.6** | **1577.9** | **3389.1** |
| | **ctl mean** | **966.0** | **1669.4** | **3614.1** |
| | **tb8 mean** | **929.4** | **1577.8** | **3389.3** |
| | **Δ** | **−36.6 ms (−3.79%)** | **−91.6 ms (−5.49%)** | **−224.8 ms (−6.22%)** |
| | **control round-to-round spread** | 9.5 ms (**0.98%**) | 7.3 ms (**0.44%**) | 8.3 ms (**0.23%**) |
| | **Δ / control spread** | **3.9×** | **12.6×** | **27.1×** |

TPOT is unmoved, as it must be — the decode objects are byte-identical between the
arms: 29.01–29.04 / 29.30–29.35 / 29.74–29.80 ms/token, both arms, every round.

The **round-1 tb8 row is voided and not averaged**, for a reason worth recording
rather than hiding: `bench_speed.sh` printed `model: gemma-4-12b-it`. A sibling
agent's **Gemma** server was already bound to port 8196 when this arm's client
connected, so the row (TTFT 223/76/30 ms, `tpot 0.00`) is a measurement of someone
else's model. This is the third occurrence of the shared-port failure class in this
campaign's history and the first one caught **by the instrument** rather than after
the fact — `bench_speed.sh` echoes the model id it resolved off `/v1/models`, and
that line is the cheap assertion. Add it to the battery checklist next to
`pgrep -x plowrt`.

An **independent earlier battery** at `NPROMPT=4, OUTLEN=64`, both arms verified
`model: glm-5.2-fp8` and `fgn=0`, agrees: 8k **1678.4 → 1584.4 (−5.6%)**, 16k
**3664.2 → 3433.1 (−6.3%)**. (Its 4096 row is discarded: at `OUTLEN=64` the first
request of each cell carried a multi-second cold-start outlier — control mean 3147.3
against a median of 970.1 — which `OUTLEN=16` removed.)

### 8.1 Does the served delta match the arithmetic?

| ctx | predicted from the traced packet | measured served |
|---|---:|---:|
| 4096 | −37.8 ms | −36.6 ms |
| 8192 | −78.7 ms (packet) / −94.7 ms (layer span) | −91.6 ms |
| 16384 | −157 ms (2 chunks × packet) | −224.8 ms |

4k and 8k land between the packet-span and layer-span projections, which is what
should happen: removing 931 µs of a packet also removes it from the layer's critical
chain. 16k over-delivers against the 2×-the-8k-packet estimate by ~42 ms — the same
straggler effect, twice, plus the `MlaMergeFold` gap the census shows shrinking out of
the serial chain.

---

## 9. Gate

Five fixed temp-0 prompts, `max_tokens=200`, over `/v1/chat/completions`: Paris;
`17*23` with steps; a two-sentence free-form on gold; a **long free-form** (a 300-word
explanation of how a GPU does a matmul); and a **~9 000-token** document with a fact
buried at paragraph 89, asked to recall it (long-context retrieval, and it rides both a
T=8192 chunk at `ns=2` and a short second chunk at `ns=1` — i.e. it exercises the new
arm on both).

**The gate carries its own determinism control.** The control arm was run **twice**, in
two separate server lives (`ctlA`, `ctlB`), before the test arm, because "character-
identical" is only evidence if the serve path is character-stable to begin with.

```
-- determinism control ctlA vs ctlB (SAME object, two server lives):
   IDENTICAL -> serve path deterministic
-- arm ctlA vs tb8:
   CHARACTER-IDENTICAL
-- arm ctlB vs tb8:
   CHARACTER-IDENTICAL

3250cd8c075b153eb3a49ecf0964b40d  /tmp/mm_gate2_ctlA.txt
3250cd8c075b153eb3a49ecf0964b40d  /tmp/mm_gate2_ctlB.txt
3250cd8c075b153eb3a49ecf0964b40d  /tmp/mm_gate2_tb8.txt
```

One md5 for all three files. The long-context answer, verbatim and identical on every
arm: *"The calibration constant for the flowmeter on line 7 is **4.8173 units per
revolution**, and it is recorded in **Paragraph 89**."*

Plus, from the traced run of §7 (different instrument, real weights): prefill first
token `407` and greedy decode `[154842, 40, 5293, 697]`, all 8 ranks token-identical,
on **both** arms.

**This change is bit-identical.** Not "within tolerance", not "a small accumulation
class": the kernel's per-token accumulation order is unchanged by construction, the
32 MB microbench output `memcmp`s equal at every T and every `nsplit`, and the served
text is one hash. There is no numerics class to characterise.

### 9.1 A gate result that was NOT a gate result

A first gate attempt reported `q2_arith` diverging between arms. It was wrong, and the
mechanism is worth writing down because it looks exactly like a real numerics finding:

- the tb8 server had **died mid-gate** on
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES` from `hsa_amd_memory_pool_allocate(1.2 GB)` — a
  co-resident workload's GLM server was running, and two TP8 GLM engines do not fit in
  8×192 GiB. The diff was against a truncated file from a dying process.
- and `q2_arith` is a ~26-token prompt, i.e. the **t=128** bucket, where
  `n_work = (128/8)*8 = 128 < 304` and the TB arm's own chip-fill guard **refuses**
  it — the two arms run literally the same instructions there. A divergence at q2 was
  never attributable to this change, and the guard is what says so.

Re-run on a healthy pair with the determinism control: identical.

---

## 10. Verdict

**Adopt as an opt-in object knob; do not default it on in this commit.**

`PLOW_MLA_FOLD_TB=8` is **−3.79% / −5.49% / −6.22% TTFT at 4k / 8k / 16k** against a
control whose own round-to-round spread is 0.98% / 0.44% / 0.23%, TPOT unchanged, and
**bit-identical** output. It composes with everything currently shipped (`ns2`, V2
flash, `PLOW_MLA_PF_SV`), needs no blob, no emitter change, no manifest `requires` and
no host plumbing, and the default object is cmp-verified byte-identical without it.

To ship it, add `PLOW_MLA_FOLD_TB=8` to the canonical object recipe:

```
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1 PLOW_MLA_FOLD_TB=8 \
    bash scripts/build_gfx942.sh <dir>
```

Three things this leaves behind for whoever picks it up next:

1. **The campaign's cost model for `MlaMergeFold` was wrong and should be corrected
   wherever it is quoted.** The op is a batched GEMM, not a KV-split tax. The causal
   split costs 147 µs/layer — 11.5 ms of TTFT at 8k, 0.7% — for its −7.7%.
2. **`W_ofold` is not worth resurrecting.** It was aiming at the fold, which now costs
   2.6× less, and it pays a doubled o_proj and the loss of `ns2` to do it.
3. **The next floor on this op is VALU, not memory.** At TB=8 the fold runs at ~25% of
   the `v_pk_fma_f32` peak with the `W_uv` stream down to 3.5 TB/s. Getting past that
   means MFMA over a bf16 `olat`, which forfeits both the f32 latent (documented as
   "strictly MORE accurate" than the standalone path) and bit-identity. That is a
   different trade and should be argued on its own.
