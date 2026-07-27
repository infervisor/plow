# PX-14 — cross-request batched prefill cannot win the §2b cell, and here is the measurement

RTX 5090 (sm_120a, 170 SM, 31.36 GiB usable) · 2026-07-26 · Gemma-4-12B-it, fp8 W8A8, MIXED
fp8-KV packet (e4m3 on the 8 hd512 FULL layers, bf16 on the 40 sliding rings). Every GPU run under
`perf-data/harness/gpulease`. Companion to `px12-consolidated-baseline.md` (§3 is the cell) and
`px1-stage1.md` / `px1-stage2.md` (the PX-1 batched-prefill design).

**Question.** PX-1 (`PLOW_PF_BATCH=1`) packs several waiting requests' prefill chunks into ONE
launch so the GEMMs see one `M = Σ len`. It is force-disabled on fp8-KV packets (`gpu.rs:1109-1122`:
`FlashPrefillFp8` uses `t[6]`/`t[7]` for the k/v dequant scales, so no tensor-handle slot is left for
the request table, and the kernel has no fp8 mux arm). The cell is 8 × 126,976-token prefills that
plow runs one after another, 81% of its wall. **Does unblocking PX-1 for fp8 close any of the 1.42×?**

**Answer: no, and not because of the fp8 blocker.** On this cell the pack budget equals the chunk
the serialized path already uses, so a working fp8 batched arm is a *measured no-op*. The lever
underneath it — fewer, bigger prefill launches — is real but worth **12.5% of prefill at most**
(measured, four packets), i.e. **29.91 → ≤33.2 tok/s**, with vLLM still **1.28×** ahead. The
numerics gate that PX-1 claims it cannot fail, **fails**.

## 1. Why the fp8 arm changes nothing here: the pack budget IS the serial chunk

The mux's per-launch row budget is the largest prefill bucket:

* `mux.rs:1413` `budget_max = e.pf_max_rows()`; `gpu.rs:2257` `pf_max_rows()` = `prefill.last().t`.
* The ladder top is `max_chunk(window)` (`devgen/src/lib.rs:1041`), window-derived = **1024** for
  Gemma-4. Emitted ladder: `[128, 512, 1024]`.
* The serialized path's `pick_prefill_bucket` takes the largest rung whenever it fills
  (`gpu.rs:2627`), so a 126,976-token prompt already runs **124 launches of M = 1024**.
* A pack of 8 requests is capped at the same 1024 rows → 8 × 128 rows. **Same 992 launches, same
  M = 1024, same total flash work** for the cell (`126,976 = 124 × 1024` exactly — there is not even
  a tail chunk to co-pack).

Packing *cannot* raise `M` above what one request already reaches, because the same constant caps
both. The PX-1 win (+27% saturated, `px1-stage2.md`) is a win on shapes whose per-request chunk is
*small*; at 127k it is not.

## 2. The lever underneath: launch count. Measured, four packets

Four packets emitted with `PLOW_MAX_CHUNK` ∈ {1024, 2048, 4096, 8192}; **everything else identical**
(`PLOW_UNISEG=1 PLOW_NS_FULL_ABS=32 PLOW_FP8=1 PLOW_W8A8=1 PLOW_FP8_KV=1 PLOW_FP8_KV_FULL=1
--max-ctx 132096`, `PLOW_DECODE_BATCH=1` for VRAM). Cubins held fixed at px12 **arm E**. One
126,976-token prompt, 8 output tokens, conc 1 → wall ≈ prefill. Identical total rows, identical
attention work; only the launch count moves.

| chunk | launches | prefill buckets | KV/seq | wall (s) | vs 1024 |
|---|---|---|---|---|---|
| **1024** (deployed) | 124 | `[128,512,1024]` | 1.64 GiB | **27.51** | 1.00× |
| 2048 | 62 | `+2048` | 2.27 GiB | 26.32 | 1.045× |
| 4096 | 31 | `+4096` | 3.52 GiB | 24.63 | 1.117× |
| **8192** | 16 | `+8192` | 6.02 GiB | **24.08** | **1.142×** |

The chunk-1024 control reproduces px12's arm E conc-1 prefill (27.59 s) to **0.3%**, so the B=1
packet is a valid stand-in for the cell's B=8 packet on the prefill path.

**Per-launch fixed cost ≈ 32 ms** (`(27.51 − 24.08) / (124 − 16)`), not the 60.1 ms the chunk-cost
model regressed at 8k (`gpu.rs:38-46`) — at 127k a launch carries ~220 ms of real work, so the fixed
part is 13% of it. Total addressable: **3.43 s per 127k prefill = 12.5%**.

## 3. At B=8 the serial route to big chunks is VRAM-blocked — measured, not estimated

The chunk sizes the sliding ring (`ring = next_pow2(window + chunk − 1)`, `devgen/src/lib.rs:1062`),
so KV/seq scales with it. At the cell's B=8:

| chunk | KV (B=8) | + weights 12.0 + act | fits 31.36 GiB? |
|---|---|---|---|
| 1024 | 13.13 GiB | 25.6 | yes (deployed) |
| 2048 | **18.13 GiB** | **31.82** | **NO — measured refusal** |
| 4096 | 28.2 GiB | 41.6 | no |
| 8192 | 48.2 GiB | 61.9 | no |

Chunk 2048 at B=8 was emitted and served: `planner: no model fits the VRAM budget at startup`,
`need_gib=31.816 free_gib=30.861`. **So the serialized path on this GPU is hard-capped at M = 1024
for this cell.**

This is the real — and only — argument for cross-request packing, and it is not the one the brief
makes: **a pack of 8 × 1024 rows reaches M = 8192 while each request still writes ≤ 1024 rows into
its own 2048-row ring**, so KV stays 13.13 GiB. Packing buys the big-M launch that VRAM forbids
serially. That is worth §2's 1.142×, no more.

## 4. Ceiling of the whole idea, end to end

Upper bound, applying §2's *best* measured prefill to px12 arm E's cell:

| | px12 arm E | PX-14 upper bound |
|---|---|---|
| prefill (8 × conc-1) | 8 × 27.51 = 220.1 s | 8 × 24.08 = 192.6 s |
| wall | 273.9 s | 246.5 s |
| aggregate out tok/s | **29.91** | **≤ 33.2** |
| vLLM 42.49 ahead by | 1.42× | **1.28×** |

**It is an upper bound, not a projection.** Inside a pack the hd512 FULL layers still run
**serially per request** (`op_attention.cuh:2786-2818`: fused hd512 varlen measured
locality-negative), so those layers keep their chunk-1024 cost and only the GEMM share of the 1.142×
is captured. The true number is between 29.91 and 33.2. Per the campaign's own rule, no isolated
ratio is scaled through an assumed budget here: 24.08 s is a *measured whole-prefill wall*.

## 5. The numerics claim PX-1 relies on is false on this packet

`mux.rs:1400-1402`: *"Chunk boundaries are bit-invariant for the fused (nsplit=1) flash … so packing
decisions cannot change any request's tokens."* Packing changes every packed request's chunk
boundaries, so this is load-bearing. Tested directly — same 122,926-token prompt, `temperature 0`,
two packets differing **only** in `PLOW_MAX_CHUNK`:

| arm | chunk 1024 | chunk 8192 | first divergent completion token |
|---|---|---|---|
| **E** (FASTPF + FP8PV) | "…on a **still** riverbank in the late summer light." | "…on a **quiet** riverbank…" | **index 6** (`2036` → `12010`) |
| **C** (FASTPF, no FP8PV) | "…in the late summer light." (15 tok) | "…in the **slow gold light of late summer**." (18 tok) | **index 11** (`5226` → `5111`) |

Both continuations are fluent and semantically identical — this is rounding order, not corruption —
but the invariant does not hold, and it fails **without** FP8PV, so it is not px12's known FP8PV
issue. Most likely mechanism (**unverified**): the flash split count is derived from `seq_kv`, which
is `c0 + C` and therefore chunk-dependent, so the merge sums in a different order. Consequence for
the brief: its **Gate 2** (greedy parity, fp8 batched vs serialized) should be expected to FAIL for
the same reason, and "packing is numerics-neutral" must be retired as a claim.

## 6. What building it actually costs (scope, if the ≤11% is wanted anyway)

1. **Kernarg move** (as briefed) — move the request table + per-row slot map from `t[6]` into
   `DevProgram` (append two `u64`s, precedent: the TP fields at `dev.rs:896-907`). Mechanical.
   Frees the fp8 handle blocker outright.
2. **fp8 prefill arm — plumbing, NOT a kernel job.** A mixed packet emits fp8 prefill only for
   hd512 (`interp_sm120.cu:759`), and hd512 is exactly the class the mux *already* runs
   serially-per-request. So the arm is `d_flash_prefill_mux`'s `USE_SERIAL` loop wrapped around the
   existing `d_flash_prefill_px8`/`px4` call, plus a per-slot offset on the scale arrays
   (`k_scale + slot * n_kv_head * kv_stride`, mirroring `kvoff`), plus handing the slot map to
   `d_headnorm_rope_fp8` (its `t[7]` is free today). ~40 lines, no new kernel body.
3. **NOT IN THE BRIEF, AND LOAD-BEARING: devgen must decouple the ladder top from the per-request
   chunk.** `max_chunk` currently sets *both* the bucket ladder top (⇒ `pf_max_rows()`, the pack
   budget) *and* the sliding ring. Batched mode needs ladder 8192 with the ring still sized for a
   1024-row per-request cap — legal only because each packed request writes ≤ its cap. Without this
   step, steps 1 and 2 are a measured no-op (§1), and activations must also be re-sized for 8192
   rows (+1.3 GiB; 13.13 KV + 12.0 weights + 1.84 act = 26.9 GiB, fits).
4. Then a greedy gate that §5 says will fail, so the arm ships as a numerics-changing flag or not
   at all.

## 7. Gates

| gate | result |
|---|---|
| chunk-1024 control reproduces px12 arm E conc-1 prefill | **PASS** — 27.51 vs 27.59 s (0.3%) |
| launch-count sweep is a clean A/B | **PASS** — one emitter, one flag (`PLOW_MAX_CHUNK`), cubins byte-identical across all four arms |
| chunk 2048 at the cell's B=8 | **FAIL, and that is the finding** — `need_gib=31.816 free_gib=30.861`, planner refuses at startup |
| **greedy-token parity across the chunk, arm E** | **FAIL — diverges at completion token 6** |
| **greedy-token parity across the chunk, arm C (no FP8PV)** | **FAIL — diverges at completion token 11.** Not an FP8PV artifact |
| PX-10's "fp8-KV hd512 prefill crashes at bucket ≥ 4096" | **DID NOT REPRODUCE** — buckets 4096 and 8192 both served a 126,976-token prompt cleanly on the mixed packet with the FASTPF prefill object. Either the bug is specific to the all-layer packet / PIPE=0 object, or it is fixed. Numerics at those buckets are *not* clean (§5), but that is a different failure |
| kernarg move landed bit-exact | **NOT RUN** — not implemented. §1 shows it cannot pay on this cell; §6 prices it |
| fp8 batched prefill works | **NOT RUN** — same reason |
| end-to-end cell with a batched fp8 arm | **NOT RUN.** The one shippable-today variant (B=8, chunk 2048) does not fit VRAM, so there was nothing to run end to end |
| GPU exclusive | **ENFORCED** — every run under `gpulease`, rc=0, no foreign process |

## 8. Reproduce

    perf-data/px14_emit_chunks.sh                 # 4 packets, B=1, chunk 1024/2048/4096/8192
    perf-data/harness/gpulease px14-c1024 perf-data/px14_probe.sh 1024   # ... 2048 4096 8192
    perf-data/px14_emit_b8.sh 2048                # the B=8 chunk-2048 packet
    perf-data/harness/gpulease px14-cell perf-data/px14_cell.sh 2048     # -> VRAM refusal
    perf-data/harness/gpulease px14-par perf-data/px14_parity.sh 1024    # arm E parity, then 8192
    perf-data/harness/gpulease px14-parC perf-data/px14_parity_armC.sh 1024  # arm C, then 8192

Binaries: `plowc`/`plowrt` from the px12 worktree (code-identical to this branch — the only commit
ahead, 28cc330, touches docs only — and the exact binaries that produced px12 §3).

## 9. Verdict

**The brief's lever is switched off for a reason that does not matter, and switching it on would
change nothing measurable on this cell.** The fp8 blocker is real and cheap to fix; the pack budget
cap (§1) is what makes fixing it pointless here. The one thing packing can genuinely buy — a
1024-row-per-request cap with an 8192-row launch, which VRAM forbids serially (§3) — is worth
**≤12.5% of prefill and ≤11% end to end**, leaves vLLM **1.28×** ahead, and costs a devgen change
nobody has scoped plus a parity gate that already fails (§5).

`gemma4-12b-longctx-5090.md` §6's two items are still the only things that can close a 1.4×: the
prefill GEMM at 38% of fp8 peak, and the absence of prefill/decode overlap. **Launch count is not a
third one** — this note prices it at 12.5% of prefill and closes it.
