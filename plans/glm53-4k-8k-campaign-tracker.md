# GLM-5.3-FP8 4K/8K TTFT campaign tracker

Updated: 2026-09-13
Branch: `tp-bringup-mi300x`
Checkpoint: `zai-org/GLM-5.3` (FP8), TP8 on 8x AMD MI300X (gfx942)
Production serving set: `a7596da2` (packet `8b15f4a2…`, stamp `0xb47295d1df086e6b`)
Detailed log: `docs/bringup/tp-bringup-upstream-review-log.md` (rows #81–#91); live rung board `plans/rung-board.md` (local)

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
| P8192-0: first 8192-row chunk, prior 0 | 725.9 ms | 363 ms | 616.8 ms (row split, T3) |
| P4096-0: first 4096-row chunk, prior 0 | 388.7 ms | 194 ms | 357.7 ms (row split, T3) |
| P8192-S: 8192-row chunk at prior 65536 | 669.8 ms | 335 ms | 641.4 ms (G4, T3; G1 648.3) |
| P4096-S: 4096-row tail in the sparse bucket | 390.1 ms | 195 ms | 376.4 ms (G4, T3; G1 379.9) |
| Served C1 TTFT, ISL 8192 | 845.8 ms | 423 ms | 752.7 ms median (row split, T4, CI not clear) |
| Served C1 TTFT, ISL 4096 | 412.6 ms | 206 ms | 386.2 ms median (row split, T4, CI not clear) |
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
| Sparse tail floor 8192 | P4096-S | `PLOW_AMD_TAIL_SPARSE_CTX` | default (#82) | final chunks at prior ≥ 8192 run the sparse bucket | floors 2048/4096 T3 |
| Row split | P8192-0, P4096-0 | `PLOW_MLA_PF_ROW_SPLIT` | opt-in (#84) | T3 −109 / −31 ms; T4 medians 856.6 → 752.7 ms C1, 1697 → 1488 ms C16, CIs touch 0 (#91); the +3.29 ms TPOT arm was single decode-tick stalls (276–2116 ms device drain) present in every arm, not row split | T4 re-run `pfroute-t4r3-*` on begin_slot v3 |
| Row split native lower half | P8192-0, P4096-0 | `PLOW_MLA_PF_ROW_SPLIT_NATIVE_LO` | built | T2: 2048 rows at context 8192 native 316.6 µs vs interpreter 744 µs per layer (≈ −33 ms / 8192 first chunk, −23 ms at 4096); ns=2 is wrong on ragged rows, path refuses anything but ns=1; one-launch whole chunk ≈ 9 ms slower; 2-layer values gate PASS (row 2047 scored against the measured single-row native floor) | T3 `pfroute-lo-t3-full` |
| DSA threshold select | P8192-S | `PLOW_DSA_SELECT_THRESHOLD` | built | T2 32/32 cases set-identical; rank 7 at 8192 rows, prior 65536: 565.7 → 372.0 µs; in-flow indexer at 65536 = score 38.6 + select 17.8 + gather 10.9 ms; GPU unit test PASS (ties, all-equal, −inf, short rows, band subrange) | T3 `idxtier-t3` (predicted −4.1…−6.1 ms P8192-S) |
| Projection packing | D20, P8192-S | `PLOW_GLM_PACK_PROJ` | built | T2 13/13 PASS; outputs copied by new op `ColSplit` (161) inside the following interpreter segment; hipBLASLt routes only; knob-off byte-identical; checkpoint S accepted | T3 `packproj-t3` (predicted −4.7…−6.6 ms per 8192 chunk, −0.6…−2.1 ms per D20 step) |
| GEMM + q RoPE fusion | P8192-S | `PLOW_GLM_FUSE_POST` | built | T2 bit-exact, 6/6 trials: 171.6 → 119.1 µs per q_rope GEMM at M=8192 (≈ −4.1 ms/chunk); other fusions rejected (no GEMM→pointwise adjacency) | T3 `fusepost-t3` |
| Parallel device queues | D20 | `PLOW_AMD_PAR_BRANCH` | built | T2b paired units −19.6 / −28.3 / −59.2 µs (K=2/3/5); predicted −2.6…−4 ms per D20 step in production terms; native GEMMs only, collectives on queue 0 | T3 `parbranch-t3-d20` |
| Dense decode at kv ≤ 2048 (exact) | D8, D16, D20 | `PLOW_GLM_DECODE_DENSE_EXACT` (emit) + `PLOW_AMD_DECODE_DENSE_EXACT` (runtime) | T3 + served C1 pass (P floor) | full depth kv 2048: D1 50.1 → 44.1 ms (−12%), D8 39.7 → 35.5 ms (−10.6%); net after indexer key writes ≈ −4.1…−4.6 / −2.3…−2.8 ms; top-2048 over ≤ 2048 keys selects every key, so the path is exact; full depth T3 (ms/step): D8 38.69 → 36.64 (−5.2%), D20 52.30 → 49.18 (−5.8%), D1 +12.95 from one intermittent ~1 s stall (scoped out: rungs ≥ 8 only); 1-layer logits byte-identical to sparse ≤ kv 2048, transition past 2048 within the off-vs-off floor; served C1 TPOT 37.72 → 35.93 ms (−4.7%, P floor 0.065 ms), TTFT unchanged; knob-on packet +16.2 MB, ≈ +50 MiB VRAM/rank; knob-off byte-identical; checkpoint S accepted | GSM8K + retrieval on the treat arm |
| G1 union skip | P8192-S | `PLOW_AMD_UNION_SKIP` | opt-in (#85), T4 pass on P floor | T3 −18.7 ms (floor 3.7); P4096-S −11.1; T4 with G4: C1 TTFT −36.9 ms [−47.3, −29.8] at ISL 16384, −17.6 [−19.5, −15.8] at 12288; C16 TPOT −2.57 / −2.27 ms (CIs clear); C16 TTFT unresolvable (harness ±400–480 ms); retrieval 39/39 on all arms | checkpoint P certificate, then runtime default flip |
| G4 MoE shared seed | P8192-S | `PLOW_GLM_MOE_SHARED_SEED` | opt-in (#86), T4 pass on P floor (with G1) | T3 −7.3 ms (floor 2.3); P4096-S −3.3; P8192-0 −8.6 | P certificate; emit default flips with the next production serving-set regeneration |
| begin_slot copy-out | P8192-0 admission | `PLOW_VMM_RELEASE_RETIRE` | v3 smoke PASS | baseline clear 102.8 ms; v1 smoke FAIL (0.4–119 ms); spare probe: copy into mapped spares 11.15 ms serial / 2.13 ms parallel (8 GPUs × 78 blocks); v2 smoke2 FAIL (117 ms max: with no spare, 78 maps per rank serialize to 115–153 ms and lose the race to admission) ; v3 (spare pool mapped at load, 354 MiB/rank from the existing KV pool cap; copies outside the process lock; 5 ms bounded wait) smoke3 PASS: every clear 0.38–0.90 ms, 0 inline unmaps, all copies from mapped spares | T4 `slotclear-t4-arms3` + retrieval guard |
| Prefix cache fix + fine rows | all | `PLOW_VMM_CACHE_MIN_FREE_MIB`, `PLOW_AMD_PREFIX_FINE_ROWS` | fix default, knobs opt-in (#83) | 0/0 mismatches over 63 attaches | — |
| Small-rung workgroup cap `auto` | P512, P128 | small-rung CUs | opt-in, T4 FAIL | T4 r2 ISL 128: C1 TTFT −12.5 ms (controls drifted), C16 TPOT −3.1 ms; ISL 512: C1 TTFT −3.2 ms [−4.7, −1.9], C16 TPOT −0.58 ms; C16 TTFT CIs cross 0 at both ISLs (harness ±110 ms); `wide` no better | C16-only rerun with more prompts, if pursued |
| Decode band on prefill rungs (status 15:40) | Band64 | `PLOW_GLM_ORDINARY_BAND`, `PLOW_AMD_DECODE_BAND_ROWS` | built | CPU gates; knob-off emit byte-identical; gate load refusals fixed in turn: decode row cap 20 → 64 (interpreter FP8 arm is row-count-general), decode objects compiled from the wrong source tree (bare relative build path after a cwd reset), missing vendor/adapter objects; v4 refused: `in.decode_slot` needs the shared-prefix VMM KV layout; fixed by sizing KV rows to max(decode ladder, band cap); v5 `band.` runtime prefix; v6 `in.kvlen` capacity check; v7 loads on TP8; v8 cross-rank band program choice tied on HashMap order (fixed); v9 decode reference padding; v9/v10 GPU fault at a 2 MiB VMM block boundary (probe read the unbacked kv-addr scratch slot; band staging never mapped rows) | v11: idle band rows must address the scratch slot, and the GPU bystander isolation check must always run |
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
| pfroute | gaps 1–2: tail floor 2048/4096, row split T4 re-run, row split's interpreter half | P4096-S, P8192-0 | tail-floor T3 | tail smoke PASS (sparse tails 110 ms at 128 rows, ~134 ms at 512, 224–273 ms at 2048); native lower half built, gate PASS; T3 tail, T4 re-run and lower-half T3 queued |
| decrowsplit | gap 3: per-row split for native sparse decode | D20 | mixed-rung price and frequency | killed at T2 (see Rejected) |
| idxtier | gap 4: exact indexer tiers by length, then block pre-selection behind a quality gate | P8192-S | per-stage attribution | T2: score per layer at 8192 rows grows linearly with prior (54 → 370 → 729 → 1408 µs at 0 / 8192 / 32768 / 65536), select 6.7 → 509 µs; ≈ 40 ms over 21 layers at 65536; tile 64 bit-identical but no faster; select is 27–36% at long prior; rank 7 is 3× (score) and 17–30× (select) rank 0 at prior 0 |
| livectx | gaps 5–6: TP8 decode switchover, live cap/GF/split counts | D20 | crossover table | production decode is sparse at every context (max_ctx 81920 > 65536 gate); 1-layer crossover: D1/D8 dense wins to 80k, D20 crosses ~4.3k; full depth kv 2048 dense −12% (D1) / −10.6% (D8); exact ≤ 2048 part in T1, approximate part behind a quality gate; gap 6 closed |

Not measured yet: the stacked path. No served run combines row split, G1, G4 and the begin_slot fix, and production and the
real-world vLLM campaign serve none of them. Run the stacked served A/B once the begin_slot fix and the G1+G4 T4 pass.

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
