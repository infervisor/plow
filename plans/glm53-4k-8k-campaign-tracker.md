# GLM-5.3-FP8 4K/8K TTFT campaign tracker

Updated: 2026-09-13
Branch: `tp-bringup-mi300x`
Checkpoint: `zai-org/GLM-5.3` (FP8), TP8 on 8x AMD MI300X (gfx942)
Production serving set: `a7596da2` (packet `8b15f4a2…`, stamp `0xb47295d1df086e6b`)
Detailed log: `docs/bringup/tp-bringup-upstream-review-log.md` (rows #81–#98); live rung board `plans/rung-board.md` (local)

## Goal and rules

- Cut the 4K and 8K rungs' prefill TTFT by 50% with plow-native work. Library kernels (hipBLASLt, AITER) stay where they
  win on the same clock.
- Let those prefill rungs carry decode rows (decode band, KV address table).
- Follow `docs/bringup/agents/08-rung-campaign.md`: T1 CPU, T2 one GPU on the device clock, T3 rung ctrl/treat/ctrl2 in one
  job, T4 served A/B with a paired bootstrap CI. One rung at a time; one timing method per comparison.
- Every GPU process goes through the job queue. Raw logs and serving sets stay outside git.
- Values: cross-process runs are nondeterministic (#50), so gates use logit floors vs off/off2, byte identity only within one
  process, plus retrieval 39/39.
- A default flips only after T4 passes and checkpoint P certifies every touched rung (#90).

## Target matrix

| rung / metric | baseline | 50% target | current best (opt-in) |
|---|---:|---:|---|
| P8192-0: first 8192-row chunk, prior 0 | 725.9 ms | 363 ms | 575.1 ms (row split + native lower half, both default since 4cf2a87d) |
| P4096-0: first 4096-row chunk, prior 0 | 388.7 ms | 194 ms | 327.0 ms (row split + native lower half, both default since 4cf2a87d) |
| P8192-S: 8192-row chunk at prior 65536 | 669.8 ms | 335 ms | 641.4 ms (G4, T3; G1 648.3) |
| P4096-S: 4096-row tail in the sparse bucket | 390.1 ms | 195 ms | 376.4 ms (G4, T3; G1 379.9) |
| Served C1 TTFT, ISL 8192 | 845.8 ms | 423 ms | ≈ 716 ms median (row split + native lower half, both default; T4 −98.5 then −35.5 ms) |
| Served C1 TTFT, ISL 4096 | 412.6 ms | 206 ms | ≈ 363.5 ms median (row split + native lower half, both default; T4 −27.4 then −23.1 ms) |
| vLLM 0.28, same hardware, C1 TTFT (exact length, unprofiled) | 570.0 ms (8192), 310.6 ms (4096) | — | `vllm-8k-profile`; the older 449 ms (#81) used a different recipe |

## Where the time goes

8192-row chunk at prior 65536 (attribution v2, rank 0):

| part | ms |
|---|---:|
| attention + fold + o_proj | 231 |
| seam collectives (94.5 transfer) | 115 |
| AITER MoE + shared down | 114 |
| glue ops (non-GEMM, non-attention) | 66 |
| TP indexer | 61 |
| hipBLASLt GEMMs | 52 |
| interpreter GEMMs | 20 |

- At prior 0, attention runs in the interpreter (283 ms vs ~171 ms native). The native sparse route needs every row to hold
  2048 keys (`amd_sparse_mla.rs:17,168`).
- Served TTFT adds `begin_slot` ~100 ms on slots whose previous occupant published an 8192-token prefix, tokenize
  8.6–17 ms, and an untimed 10–16 ms handler remainder.

vLLM per-op profile (`reports/vllm-8k-profile.md`), 8192 at prior 0: GPU span 560.0 ms vs plow 721.3 ms. Attention is
+159.9 ms of the gap; host/admission adds +114.5 ms to TTFT (+16.4 with begin_slot v3). Plow is level or ahead on MoE,
collectives, indexer and glue. At 4096 the GPU gap is +53 ms. At 8192 over prior 65536 plow is ahead: 669.8 vs 860.3 ms,
with vLLM's indexer at 308.5 ms per chunk vs plow's 79.7. vLLM's attention speed comes from FP8 query and keys in place
(`pagedps` below: 20–60× plow's attention error), so it is a precision trade rather than a missing kernel.

## Levers

| lever | rung | knob | state | measured | next gate |
|---|---|---|---|---|---|
| Sparse tail floor 8192 | P4096-S | `PLOW_AMD_TAIL_SPARSE_CTX` | default (#82) | final chunks at prior ≥ 8192 run the sparse bucket; floor-only T3 (dense → sparse chunk ms): 128 rows +8.5…+10.7, 512 rows +0.8 (R 2112) / −8.4 (R 4160) / −17.8 (R 6208), 2048 rows −20.5 / −59.3 / −99.1; floors 2048 and 4096 give the same numbers where both move the tail; values PASS on the pooled rel-L2 gate, retrieval 21/21 | keep 8192 alone; see pairs threshold |
| Tail pairs threshold | P4096-S | `PLOW_AMD_TAIL_SPARSE_MIN_PAIRS` with `PLOW_AMD_TAIL_SPARSE_CTX=2048` | opt-in (#94, e8bbeed6) | sparse tail ≈ 102 ms + 0.061 ms/row at any prior; dense slope 0.085 / 0.107 / 0.124 ms/row at R 2112 / 4160 / 6208, so the crossover is prior × rows ≈ 1.2–1.3M; rule: sparse when prior ≥ 2048 and prior × rows ≥ 1.5M keeps every measured win and leaves every measured loss dense; 84 tests pass ; confirm T3 (chunk ms, ctl → treat): 2112×512 132.7 → 132.7 and 4160×256 120.2 → 120.2 stay dense; 2112×768 173.3 → 149.4 (−23.9), 2112×1024 182.8 → 161.0 (−21.8), 4160×512 142.8 → 134.3 (−8.5), 4160×1024 222.9 → 162.1 (−60.7), floors 0.2–1.0; every route as predicted; pooled logits on moved requests within 3× control | default flip needs a served T4; a bucket-aware threshold (dense 513–2048-row tails pay the 2048 bucket, ≈ −20 ms more at priors 2048–2930) is a separate lever |
| Row split | P8192-0, P4096-0 | `PLOW_MLA_PF_ROW_SPLIT` | **default** (#96, bf23f06f; rollback `=0`) | T3 −109 / −31 ms; T4 medians 856.6 → 752.7 ms C1, 1697 → 1488 ms C16, CIs touch 0 (#91); the +3.29 ms TPOT arm was single decode-tick stalls (276–2116 ms device drain) present in every arm, not row split | T4 `pfroute-t4d-*` on tip 38d97bac + stack, RETIRE unset, arms interleaved ctl/split/ctl2/split2, scored by paired medians against a same-job control floor (null check passes): FLIP. TTFT −98.5 ms (8192 C1), −211.8 (8192 C16), −27.4 (4096 C1), −61.6 (4096 C16); TPOT +0.02 / −11.65 / −0.04 / −3.29 ms; C1 E2E −95.2 / −28.6 ms; 0 failed. P certificate `perf-certs/rt.mla_pf_row_split.json` (stage C): P8192-0 730.6 → 630.3 ms (floor 2.3), P4096-0 388.9 → 357.6 (1.0), P8192-S neutral, 32 serving entries not worse |
| Row split native lower half | P8192-0, P4096-0 | `PLOW_MLA_PF_ROW_SPLIT_NATIVE_LO` | **default** (#98, 4cf2a87d; rollback `=0`) | T2: 2048 rows at context 8192 native 316.6 µs vs interpreter 744 µs per layer (≈ −33 ms / 8192 first chunk, −23 ms at 4096); ns=2 is wrong on ragged rows, path refuses anything but ns=1; one-launch whole chunk ≈ 9 ms slower; 2-layer values gate PASS (row 2047 scored against the measured single-row native floor); full-depth T3 with row split on in every arm: P8192-0 615.2 → 575.1 ms (−40.1, floor 5.0), P4096-0 356.0 → 327.0 ms (−28.9, floor 0.7), P8192-S unchanged; logits under the control floor, layer-0 KV byte-identical, retrieval 39/39 ; served T4 (row split vs row split + lower half) FLIP: TTFT −35.5 ms (8192 C1), −80.9 (8192 C16), −23.1 (4096 C1), −58.7 (4096 C16); TPOT +0.01 / −4.45 / −0.03 / −2.84 ms; 0 failed; one 4096 C16 arm held a device-drain decode stall (10 ticks of 186–2071 ms) absent from its twin; P certificate `perf-certs/rt.mla_pf_row_split_native_lo.json` accepted | — |
| DSA threshold select | P8192-S | `PLOW_DSA_SELECT_THRESHOLD` | parked | T2 32/32 cases set-identical; rank 7 at 8192 rows, prior 65536: 565.7 → 372.0 µs; in-flow indexer at 65536 = score 38.6 + select 17.8 + gather 10.9 ms; GPU unit test PASS (ties, all-equal, −inf, short rows, band subrange) ; T3 r1 measured nothing (job cleanup deleted its own wrappers); T3 r2 on a 3f901456 rebuild: P8192-S 650.7 / 651.0 / 651.4 ms (ctrl / treat / ctrl2, +0.0, floor 5.0), P4096-S +2.1 (floor 1.9), P8192-0 +0.3 (floor 6.2); values, selection identity, firing and retrieval all pass | parked with that number; patch `idxtier-select-thr-3f901456.patch` |
| Projection packing | D20, P8192-S | `PLOW_GLM_PACK_PROJ` | T3 FAIL, fix in T2b | T2 13/13 PASS; outputs copied by new op `ColSplit` (161) inside the following interpreter segment; hipBLASLt routes only; knob-off byte-identical; checkpoint S accepted | **T3 FAIL**: P8192-S +73.2 ms (floor 3.9), P4096-S +66.2, P8192-0 +49.8, D20 tick +4.9 ms, served TPOT +7…9 ms; values clean. Cause: the new `ColSplit` copy handed 512 rows per workgroup, so 2–16 of 304 workgroups did the copy. T2b of the fix, old → new µs: band 1024 rows 547 → 24 (Glu 19), bucket 8192 118 → 38, decode 20 rows at 20 workgroups 60 → 6. The bucket shape alone explains ≈ 18 ms of the 73, so the rest must be reconciled before a T3 rerun |
| GEMM + q RoPE fusion | P8192-S | `PLOW_GLM_FUSE_POST` | T3 near-miss | T2 bit-exact, 6/6 trials: 171.6 → 119.1 µs per q_rope GEMM at M=8192 (≈ −4.1 ms/chunk); other fusions rejected (no GEMM→pointwise adjacency); T3: P8192-S 668.9 → 666.0 ms (−2.9, floor 3.0, misses by 0.1), P4096-S −2.6 (floor 0.7), P8192-0 −3.4 (floor 6.8, not worse); layer-0 KV byte-identical, retrieval 18/18 + 21/21 | `fusepost-confirm-p8192s` (6 rounds of ctrl → treat → ctrl2, 2 counted reps per process, 12 samples per arm); patch rebased on 9ead3cfe (review-log #95), checkpoint S accepts; if it clears, certify with G4 in the regeneration |
| Parallel device queues | D20 | `PLOW_AMD_PAR_BRANCH` | built | T2b paired units −19.6 / −28.3 / −59.2 µs (K=2/3/5); predicted −2.6…−4 ms per D20 step in production terms; native GEMMs only, collectives on queue 0 | T3 `parbranch-t3-d20` |
| Dense decode at kv ≤ 2048 (exact) | D8, D16, D20 | `PLOW_GLM_DECODE_DENSE_EXACT` (emit) + `PLOW_AMD_DECODE_DENSE_EXACT` (runtime) | T3 + served C1 pass (P floor) | full depth kv 2048: D1 50.1 → 44.1 ms (−12%), D8 39.7 → 35.5 ms (−10.6%); net after indexer key writes ≈ −4.1…−4.6 / −2.3…−2.8 ms; top-2048 over ≤ 2048 keys selects every key, so the path is exact; full depth T3 (ms/step): D8 38.69 → 36.64 (−5.2%), D20 52.30 → 49.18 (−5.8%), D1 +12.95 from one intermittent ~1 s stall (scoped out: rungs ≥ 8 only); 1-layer logits byte-identical to sparse ≤ kv 2048, transition past 2048 within the off-vs-off floor; served C1 TPOT 37.72 → 35.93 ms (−4.7%, P floor 0.065 ms), TTFT unchanged; knob-on packet +16.2 MB, ≈ +50 MiB VRAM/rank; knob-off byte-identical; checkpoint S accepted | GSM8K + retrieval on the treat arm |
| G1 union skip | P8192-S | `PLOW_AMD_UNION_SKIP` | **default** (#92, 38d97bac; rollback `=0`) | T3 −18.7 ms (floor 3.7); P4096-S −11.1; T4 with G4: C1 TTFT −36.9 ms [−47.3, −29.8] at ISL 16384, −17.6 [−19.5, −15.8] at 12288; C16 TPOT −2.57 / −2.27 ms (CIs clear); C16 TTFT unresolvable (harness ±400–480 ms); retrieval 39/39 on all arms; P certificate `perf-certs/rt.union_skip.json` accepted (stage C) | — |
| G4 MoE shared seed | P8192-S | `PLOW_GLM_MOE_SHARED_SEED` | certified, waits on regeneration | T3 −7.3 ms (floor 2.3); P4096-S −3.3; P8192-0 −8.6 | P certificate accepted (stage C): P8192-S 648.7 → 641.4 (floor 3.7), P4096-S 379.7 → 376.4 (2.1), P8192-0 731.8 → 723.2 (7.4), P2048-0 199.3 → 195.9 (0.4), P2048-S2048 228.4 → 224.4 (1.8); flip-2 patch ready (default packet c2aeb002, `=false` gives production 8b15f4a2); lands with the next production serving-set regeneration (see Resume protocol) |
| begin_slot copy-out | P8192-0 admission | `PLOW_VMM_RELEASE_RETIRE` | opt-in; v4 T4 fails only 4096 C1; v5 in T4 | baseline clear 102.8 ms; v1 smoke FAIL (0.4–119 ms); spare probe: copy into mapped spares 11.15 ms serial / 2.13 ms parallel (8 GPUs × 78 blocks); v2 smoke2 FAIL (117 ms max: with no spare, 78 maps per rank serialize to 115–153 ms and lose the race to admission) ; v3 (spare pool mapped at load, 354 MiB/rank from the existing KV pool cap; copies outside the process lock; 5 ms bounded wait) smoke3 PASS: every clear 0.38–0.90 ms, 0 inline unmaps, all copies from mapped spares; **v3 T4 FAIL**: 8192 C1 clear 100 → 0.4 ms and TTFT −90 ms, but the copy-out moved into the prefill tick's deferred publish (38–47 → 60–118 ms on reused slots), so ITL[0] 71 → 107–130 ms at 4096 C1 and 67 → 115 ms at 8192 C16; E2E median −40…−50 ms (8192 C1), +50…+64 ms (4096 C1), +250…+300 ms (8192 C16); TPOT beyond the P floor in every cell; decode ticks identical across arms | v4 cause: with a full prefix cache every publish evicts a copied prefix, and v3 charged spares to the reuse-pool cap, so each eviction queued ~624 serialized unmaps under the pool lock; v4 copies outside the pool lock and gives spares their own budget (≤ 156 MiB/rank extra); lookup probe: ROCr call cost flat to 8192 live mappings, one range unmap of 78 granules 10.9 ms; zero-copy rejected (needs the same unmap, and a parked decode row may still read the mapping); rebased on 3f901456, smoke PASS (clears ≤ 0.57 ms, 0 inline unmaps, 0 spares dropped) ; v4 T4 (rebased on 3f901456): 8192 C1 TTFT 853 → 764 ms (−89), E2E −93 ms; 8192 C16 TTFT −178 ms, TPOT −9.1 ms, E2E −1.35 s; 4096 C1 FAIL: ITL[0] 71.5 → 108 ms (+37), E2E +34 ms, because each 4096 publish evicts a copied 8192 prefix and its 78 row-0 copies per rank overflow the spare slack (1638 drops per rank), and the unmaps hold ROCr's process-wide lock inside the publish; v5 defers spare drops, refill and reclaim to after the publish (smoke on 9f52fb73: clears ≤ 0.51 ms, 0 drops, 8192 C1 TTFT 643.5 ms with row split default, 4 requests, no control) | T4 `slotclear-t4-arms-v5` + retrieval guard; flip needs 4096 C1 ITL[0] within floor and no 8192 C16 decode-tick p99 or TPOT p99 regression |
| Prefix cache fix + fine rows | all | `PLOW_VMM_CACHE_MIN_FREE_MIB`, `PLOW_AMD_PREFIX_FINE_ROWS` | fix default, knobs opt-in (#83) | 0/0 mismatches over 63 attaches | — |
| Small-rung workgroup cap `auto` | P512, P128 | small-rung CUs | opt-in, T4 FAIL | T4 r2 ISL 128: C1 TTFT −12.5 ms (controls drifted), C16 TPOT −3.1 ms; ISL 512: C1 TTFT −3.2 ms [−4.7, −1.9], C16 TPOT −0.58 ms; C16 TTFT CIs cross 0 at both ISLs (harness ±110 ms); `wide` no better | C16-only rerun with more prompts, if pursued |
| Decode band on prefill rungs (status 23:10) | Band64 | `PLOW_GLM_ORDINARY_BAND`, `PLOW_AMD_DECODE_BAND_ROWS` | built | CPU gates; knob-off emit byte-identical; gate load refusals fixed in turn: decode row cap 20 → 64 (interpreter FP8 arm is row-count-general), decode objects compiled from the wrong source tree (bare relative build path after a cwd reset), missing vendor/adapter objects; v4 refused: `in.decode_slot` needs the shared-prefix VMM KV layout; fixed by sizing KV rows to max(decode ladder, band cap); v5 `band.` runtime prefix; v6 `in.kvlen` capacity check; v7 loads on TP8; v8 cross-rank band program choice tied on HashMap order (fixed); v9 decode reference padding; v9/v10 GPU fault at a 2 MiB VMM block boundary (probe read the unbacked kv-addr scratch slot; band staging never mapped rows) | v11 fault in bystander `prefill_slot(63)`: `kv_slot_stride` divided KV bytes by the decode batch (20) instead of `kv_row_capacity` (64), 3.2× too large (fixed), and the probe skipped `begin_slot` (fixed); v12: bystander prefill and snapshot pass, then GPU fault at a host-heap address (0x55555673e000) inside `advance_decode_band`; the band attention emitter writes no embed, norm, QKV or KV for band rows, so band rows do not compute a decode step yet ; v13: band rows gained their own `decode_slot` KV write, IndexScore and flash take `decode_slot`, `in.ids`/`in.pos` carry the band token (indexer-key write for band rows still missing); load refused: `--replay-knobs` does not restore the raw-env emit vars `PLOW_MLA_PF_V2`, `PLOW_MLA_PF_AITER`, `PLOW_UNISEG`, so FlashMlaPrefill lost its wave-class-4 segment; v14 (re-emitted with them) faults in the reference prefill before any band dispatch, at 2 MiB-aligned device VA 0x7eead5000000 ; v14 cause: the band KV write is compiled into every dispatch of a band-capable program, and `in.decode_slot` was only staged by a real band advance, so the first ordinary prefill wrote at an uninitialized slot index; v15 staged idle rows at load; review found a second bug: after a band advance, the next ordinary prefill reused the live `decode_slot` values and would have overwritten live decode KV, so v16 re-stages idle rows in `prefill_prepare` and adds a band advance → ordinary prefill → band advance KV-identity step; v16 faults during load (GPU node-12, 0x7efa18006000, inside a 2 MiB block), before any dispatch; every ordinary prefill of a band-capable program also pays the idle band rows' attention and KV write; row-table decode (`submit_decode_rows_at`) has no caller in `serve/mux.rs`, so no production path packs decode rows yet ; v16's load fault was a race (ranks load concurrently), so load-time staging was removed and a load-only probe mode added; v17 faulted in the reference prefill at a host-heap address: `stage_decode_band` passed plain `Vec` sources to the pinned-copy API, which requires pinned memory (likely also v12's real crash), fixed by staging through the pinned scratch with a debug assertion; v18 passed REF and BAND and faulted in ISOLATION (bystander slot 63) at an unmapped 2 MiB block: nothing mapped the kv-address scratch slot; v19 maps it on every staging call and asserts every staged row is mapped ; v19 still faulted at bystander slot 63; an isolation bracket (slots 19/20/21/32/63 with per-rank KV address dumps) showed the first ordinary prefill after load faults at slot 19 already, so it is not the ≥ 20-slot class; the fault is `kv.0.krot` at 65 × the per-slot stride (row 0 with idle pos 0, row 32 before), while every staged and device-read-back `decode_slot` is the scratch slot 128; the write lands at half of scratch's byte offset plus one stride, which points at the band KV write using the wrong element width or row bytes for krot | compare band, decode-ladder and prefill krot write variants and row widths, add a load-time byte-stride check, bracket v23, then the full gate; band-row indexer-key write still missing; production serving wiring and a served A/B after that |
| KV address table + dynamic slots | D20, Band64 | `PLOW_GLM_KV_ADDR`, `PLOW_KV_ADDR`, `PLOW_AMD_KV_SLOTS` | patch ready | floor gate a PASS (3.72e-2 vs 6.94e-2) | gates b/c/d |
| Admission path | served TTFT | — | unowned | `encode_fast` 9.6 → 7.2 ms (CPU, 8.5k tokens) | lever card |

## Rejected or parked

| candidate | reason |
|---|---|
| Plow GEMM tiles, split-K, grid (#87) | hipBLASLt wins every prefill shape on one clock; plow keeps GEMV at decode 1–2 |
| Decode kernel re-pick (#88) | T2 −7.7 ms predicted; T3 +0.5 ms measured |
| G2 router pre-all-gather | −1.2 ms, inside the 4.3 ms floor |
| FP8 keys in sparse attention | measured negative |
| FP8 block-scale prefill GEMMs | no end-to-end gain |
| Two micro-batches, GEMM+RS fusion, parallel decode enqueue, kernarg cache, XRN fusion | fixed cost or null |
| Token-batch bodies with seams | corrupt output |
| begin_slot: parallel ranks | 0.93x (driver serializes unmaps process-wide) |
| begin_slot: pre-retire at finish | 85 ms of serialized unmaps vs a ~35 ms gap |
| FP8 weights in decode GEMV (w8gemv) | T2 on the production object, M=1: q_a\|kv_a\|k_rope 60.0 → 67.2 µs (grid 304), 73.5 → 87.5 µs (grid 64); o_proj 37.4 → 56.0 µs (grid 304); only o_proj at grid 64 wins |
| Load-time weight pre-packing | nothing to pack: GEMV reads weights in place (`interp.hip:2322-2331`); GemmSmall's per-launch tile copy is the cache load itself |
| More projections on the sequence-parallel row shard (spseam) | SP seams and `_PROJ` are already default (−68 ms T3, #54–#69); four leftovers total −2.4…−4.0 ms, each below the P8192-S floor; FP8 MoE-input gather (≈ −13 ms) needs a W8A8 shared expert |
| GEMMs during reduce-scatter (parbranch) | 0.0 ms of independent work in the 8 segments after any collective on D20 and P8192-S |
| Paged in-place FP8 sparse attention (vLLM's AITER `mla_a8w8_qh16_qseqlen1_gqaratio16_ps`) | T2 numerics: attention rel-L2 1.8e-2…9.7e-2 at prior 65536 and 1.6e-2…9.9e-2 at prior 0 vs plow's bf16 floor 5.7e-4…1.5e-3; timing skipped |
| Per-row split for native sparse decode (decrowsplit) | native output on rows < 2048 keys is wrong (rel err ≈ 1.0), so the 2048 check stays; a split costs 170–191 µs per layer at D20 vs 108 µs for the interpreter rung (+5–6 ms/step); production decode never takes the native route (`PLOW_GLM_MLA_DEC_AITER` off) |
| Indexer span/tile tiers (idxtier) | span 2048/4096 identical on the slowest rank; tile 64 bit-identical and never faster |
| Prior-0 indexer rank skew | in flow ≈ 0.1 ms per 8192 chunk, below the floor |
| Live context params (livectx gap 6) | kernels already bound loops by live kv_len; decode stride fields 81920 → 4096 gave 1.162 vs 1.167 ms; prefill cap is a stride the load check pins |

## Current architecture finding

Each prefill bucket (128/512/2048/8192) is one compiled program; dense vs sparse attention is fixed at emit. Per chunk the
runtime picks a bucket, native vs interpreter attention (`prior >= 2047`), and patches row counts, prior and `in.kvlen`.
Top-k 2048 keeps attention flat with context; only the indexer grows. No object, wave or split count switches with context.

Gaps:
1. Chunks starting below 2047 run all attention in the interpreter. Row split fixes it opt-in.
2. Final chunks at prior 2047..8191 stay dense: the sparse move was measured only from 8192.
3. Native sparse decode needs every row ≥ 2048 keys (`amd_sparse_mla.rs:1110`): one short row sends the rung to the
   interpreter.
4. The indexer grows 18 ms (prior 0) → 61 ms (prior 65536) per 8192 chunk with one kernel for every length.
5. The decode dense/sparse switchover (65536) was measured on TP4 and is fixed per packet.
6. Cap, GF and the sparse decode split count are baked at maximum context.

## Next experiments (agents launched 2026-09-13)

| agent | lever | owner rung | first gate | result so far |
|---|---|---|---|---|
| packproj | one GEMM per group of projections sharing an input | D20 | T2 packed vs separate, launch cost included | T2 PASS 13/13 cells: decode −16 / −33 / −17 µs per group (3-proj attention, 5-proj attention, router+gate+up), band 1024 −42 / −61 µs, bucket −32 µs (4096) / −60 µs (8192); ≈ −2.9 ms per D20 step, ≈ −8 ms per 8192 chunk (kernel only); scoped to hipBLASLt routes; T1, then T3 |
| fusepost | GEMM plus following pointwise ops in one pass | P8192-S | T2 on the top glue ops | T2 PASS, bit-exact, ≈ −4.1 ms/chunk kernel; T3 queued |
| parbranch | independent branches on separate device queues; GEMMs during reduce-scatter | D20 | T0 multi-queue feasibility | T0 feasible; T2 probe decode chains 0.59–0.60× serial with GPU-side join, prefill 0.94–0.98×, parity PASS; in-model D20 net predicted −1.7…−3.6 ms; reduce-scatter overlap killed |
| w8gemv | per-call weight packing check; FP8 weights in decode 1–4 GEMV | D1 | accuracy gate before T4 | killed at T2 (see Rejected) |
| spseam | projections on the sequence-parallel row shard | P8192-S | SP status and pricing | killed at design (see Rejected) |
| pfroute | gaps 1–2: tail floor 2048/4096, row split T4 re-run, row split's interpreter half | P4096-S, P8192-0 | tail-floor T3 | tail smoke PASS (sparse tails 110 ms at 128 rows, ~134 ms at 512, 224–273 ms at 2048); native lower half built, gate PASS; tail T3 done (floor alone slows 128-row tails → pairs threshold, T3 queued); T4 re-run held (v3 base failed), requeue on the tip default; lower-half T3 queued |
| decrowsplit | gap 3: per-row split for native sparse decode | D20 | mixed-rung price and frequency | killed at T2 (see Rejected) |
| idxtier | gap 4: exact indexer tiers by length, then block pre-selection behind a quality gate | P8192-S | per-stage attribution | T2: score per layer at 8192 rows grows linearly with prior (54 → 370 → 729 → 1408 µs at 0 / 8192 / 32768 / 65536), select 6.7 → 509 µs; ≈ 40 ms over 21 layers at 65536; tile 64 bit-identical but no faster; select is 27–36% at long prior; rank 7 is 3× (score) and 17–30× (select) rank 0 at prior 0 |
| livectx | gaps 5–6: TP8 decode switchover, live cap/GF/split counts | D20 | crossover table | production decode is sparse at every context (max_ctx 81920 > 65536 gate); 1-layer crossover: D1/D8 dense wins to 80k, D20 crosses ~4.3k; full depth kv 2048 dense −12% (D1) / −10.6% (D8); exact ≤ 2048 part in T1, approximate part behind a quality gate; gap 6 closed |

Not measured yet: the stacked path. No served run combines row split, G1, G4 and the begin_slot fix, and production and the
real-world vLLM campaign serve none of them. G1 is now default (38d97bac) and G1+G4 passed T4; begin_slot v3 failed T4, so the stacked served A/B waits on begin_slot v4 or runs without it.

Serving-set regeneration (G4, plus fusepost if it certifies): `--replay-knobs` restores only `EmitConfig` knobs. Set `PLOW_MLA_PF_V2=1`, `PLOW_MLA_PF_AITER=1` and `PLOW_UNISEG=0` explicitly from the production `build.json` (a7596da2 records them), and check every program's segment count and wave classes against the old packet before rebuilding objects. Packets emitted from 5967d7a5 on record these in `unrecorded_env`. Keep the previous serving set until a gate passes on the new one.

Row-split attention (`PLOW_GLM_ROWSPLIT_ATTN`, rowsplit agent, −40…−75 ms priced on P8192-S) also removes the indexer gather + complete step (10.9 ms), the only exact indexer term left. Its fix2 repro passed at 07:24, but its T3s sat unrun in the backlog for 13 hours. The first T3 then measured nothing valid: `plowrt bench`'s PFCHUNK `chunk=` is host-side enqueue (15–20 ms per 8192-row chunk), and the 73728-token prompt overflowed argv. `rowsplit-t3b` reused fusepost's device-synchronized sweep and value harness and showed two further problems: the packets were single-layer bisect packets (`PLOW_LAYERS=1`, 8192-row chunk 12.9 ms), and the treat arm refused every chunk past c0=0 (`sparse MLA chunk exceeds its query/KV capacity`: `Route::rebase` compared bucket rows with the row-split instruction's 1/8 band). The capacity fix is in; the lever is scoped to whole-sparse chunks (prior ≥ 2047) so it composes with the default row split; a new owner is resolving 8-head vs 64-head instruction selection, then full-depth packets, a 2-layer gate at c0 ∈ {0, 8192, 65536}, and the T3.

packproj reconciliation: the band-width `ColSplit` calls (78 per 8192 program) cost 480–547 µs each, which explains ≈ 40 of the 50–73 ms. With the fixed kernel it predicts D20 −2.0 ms/step and prefill −3.5…+0.6 ms (unresolvable), so `packproj-t3b` decides it as a decode lever.

## Side findings

- MTP speculative decoding (full depth): gate B PASS after fixing draft steps 2+ reusing step 1's top-k over a longer
  kv_len (read a −1 selection row → GPU fault). Acceptance 739/815 drafted (90.7%), 3.68 committed tokens per verify
  at k=3. The 2/8 same-process divergence is benign: commits equal the verify argmax, and both flips sit on near-ties
  (margins 0.375 and 0.000) within the logit floor. T3 TPOT A/B with natural-text and random cells is queued.
- AITER attention objects: plow's pinned `mla_a16w16_qh8…v3.co` (and `mla_a8w8_qh8…v1.co`) carry a zero argument-block
  size in their descriptor, which plow's `load_attention` patches; loading them through AITER directly faults.
- `/dev/null` on the host was replaced by a regular file around 10:07 and restored at ~10:15; cause unknown.
- On the production decode object (MM=16) at one row, GEMV costs 5.6× the MM=1 build: q_a|kv_a|k_rope 60 vs 10.7 µs,
  o_proj 37.4 vs 11.0 µs (w8gemv T2). Relevant to the single-row decode program (b1) and to decode rungs 1–4.
- The fused shared gate|up GEMV costs 103 µs at one row against 31 µs for two plain GEMVs on the same object.

## Resume protocol

1. Pull `tp-bringup-mi300x`; read this tracker, the review log rows and `docs/bringup/agents/08-rung-campaign.md`.
2. Reuse the production set as control. Change one variable per candidate; keep ctrl and ctrl2 in the same job.
3. Register every new knob (`crates/devgen/src/knob_spec.rs`, `crates/plowrt/src/knob_spec.rs`), run the knob tests, and run
   checkpoint S for emit knobs.
4. Land qualified code opt-in with a review-log row; flip a default only with T4 and a checkpoint P certificate.
5. Update this tracker's lever and rejected rows so failed experiments are not repeated.
