# Kimi-K3 MI355X campaign summary (2026-09-02 → 2026-09-04)

One row per experiment from the gfx950 campaign on the `codex/amd-agent-harness`
line. This file replaces the per-experiment reports it folds (listed at the end);
the served baselines stay in `kimi-k3-plowrt-mi355x-baseline.md` and
`kimi-k3-vllm-mi355x-baseline.md`, the index in `kimi-k3-consolidated.md`, and the
raw data in the JSON/TSV/JSONL files and `tuning/`.

Conventions: TP8 on one 8×MI355X node, native MXFP4 experts, BF16 KV. Network
numbers are `plowrt bench` 8192→256 (or 8192→1 where noted), three
order-alternated folds, paired against a same-source control. "Exact" means the
output token array and `fnv1a64` checksum matched the control in every fold.
Isolated numbers are standalone-kernel screens, never throughput claims.

## Position

| Record | TTFT | TPOT | Notes |
|---|---:|---:|---|
| vLLM 0.28 reference (`kimi-k3-vllm-mi355x-baseline.md`) | 568.35 ms | 20.768 ms C1 | 1133.93 tok/s C128 |
| Plow kernel-gap inventory cell, 2026-09-02 (8192→1024) | 2276.89 ms | 55.63 ms | 17.30 tok/s = 4.01×/2.67×/2.70× behind |
| Plow prefill attribution, 2026-09-03 (8192→1) | 1783.545 → 1620.933 ms | — | device state clear promoted (−160.998 ms) |
| Plow served decode attribution, 2026-09-03 | — | 55.731 → 45.298 ms | compact TP audit (−18.7%), 14.94 → 17.69 tok/s |
| Plow current-default attribution, 2026-09-04 (no audit) | 1405.483 ms | 28.4830 ms | E2E 8668.645 ms |
| Plow with RS-U2 + MLA TR16, 2026-09-04 | 1340.797 ms | 28.5777 ms | 29.670 tok/s |
| Plow served C1 record (`kimi-k3-plowrt-mi355x-baseline.md`) | 1271.86 ms | 28.53 ms | 33.63 tok/s; 2.24×/1.37× behind vLLM |

Attribution at the current default (prefill, 693 ordered segments, 1370.556 ms):
primary interpreter 186 seg / 699.501 ms, lean MoE stage-1 92 / 213.543, raw KDA
138 / 194.538 (Wu 42.650 + Carry 151.888), MLA V2 24 / 92.441, lean stage-2 92 /
70.918, KDA intra 69 / 41.536, lean combine 92 / 39.038. Decode body per token:
GEMV 11.337 ms (40.72%), MoE 5.394, TP reductions 4.594, AttnRes 3.288.
`PLOW_PREFILL_SEG_TIMING=1` is the default-off per-segment diagnostic.

Sizing gates kept for reference: `PLOW_GATE_HIER` on vs off = 28.727 vs 34.659 ms
TPOT (−5.932 ms, −17.1%; keep on; older sc1 ceiling numbers are void). XReduce
workgroup sweep: 256 WGs best (38.487 / 72.976 / 96.434 µs; weighted 19.272 ms),
224 WGs +9.34%, 80 WGs (AITER-like) +143–195% — keep 256. Hybrid `devblob`
emitter is default-on (93 layers, 5,939 tensors, prefill 2,764 inst / 925 seg /
649 phases, decode 2,165 / 49 / 49); the EP asset it emits is experiment input
only.

Coverage/admission records (not performance baselines): production-path smoke
2026-09-02 (C1 64→2 B128 TPOT 100.716 ms; B1 low-rung tier 54.283 ms, −46.1%,
exact; C128 64→2 395.804 ms TPOT, 2.53 tok/s), bench↔HTTP parity smoke (output
IDs `[2598, 198, 5054, 220]` on both surfaces; B1 arm TPOT 76.112 → 62.450 ms,
one sample), throughput gates (66.500 GiB/rank resident; 8192→32 C128 1,304
tok/s total, TTFT 382,965 ms, TPOT 1,482.9 ms).

## Experiments

### Decode

| Experiment | Mechanism | Measured | Verdict | Switch / evidence |
|---|---|---|---|---|
| MLA specialist segments (09-03) | Adjacent `FlashMlaDecode`+`MlaMergeFold` pair in its own specialist segment (24 extra launches/rank/step) | TPOT −8.096 ms (−21.26%), SD 0.030, 3 folds, exact `6bdfaa7b84ee4e7e`; object 256 VGPR, 6 spills | promoted | `PLOW_SEG_DECODE_MLA=0` rollback |
| Decode inventory prune (09-03) | Drop unused dispatch arms from the decode interpreter object | 412,208 → 186,096 B; VGPR 256 → 248, VGPR spills 2 → 0; TPOT −6.259 ms traced / −4.925 ms untraced, exact 768/768 | promoted candidate; TTFT provenance open | pairing hash `0x866caa2fa6a1d6a5` |
| Compact TP counter audit (09-03) | Concurrent `plow_xctr_audit` kernels + one large-BAR status word instead of the 59,392 B copy audit | audit 11.350 → 1.160 ms/token; TPOT 55.731 → 45.298 ms (−18.7%); K4 multi-step neutral (45.290 vs 45.200) | promoted | `PLOW_TP_AUDIT_COMPACT=0` keeps the copy path |
| Wide-decode GEMV width (09-02) | GEMV MM8 vs MM16 objects, C128 2→16 | +10.0% tok/s (51.970 → 57.147), median ITL −32.6% | promoted | CMake `PLOW_GEMV_MM=8` when compiled decode capacity > B32 |
| Grouped-MoE standalone route (09-04) | Exact adjacent grouped GLU+DOWN pair in a pure decode segment at grid 768 | isolated 25.172 → 16.776 µs/layer; TPOT −0.716 ms (SD 0.052), exact; handoff 0.940–0.964 ms/token | qualified, default-off | `PLOW_MOE_DECODE_STANDALONE=1`; cooperative one-kernel variant 3.23× worse |
| MoE decode route rule (09-04) | Reroute only when `standalone + handoff <= 0.9 × interpreter` from `moe_decode_measurement.jsonl` (handoff 10.3 µs/layer) | TPOT 28.607 → 27.934 ms (−0.673, −2.35%); stacked with GLU UN=7 −0.834 ms (−2.92%), exact `b7682a38c151ac99` | promoted (rule) | `PLOW_SEG_DECODE_GROUPED_MOE`, `crates/tunedb/src/moe_decode.rs`, `scripts/tune_moe_decode_publish.py` |
| GEMV GLU UN=7 (09-04) | `K == 7168 → UN=7` unroll rung in `d_gemv_glu` | TPOT 28.574 → 28.460 ms (−0.114, −0.40%), exact; object resources unchanged | promoted, no flag | `runtime/amd/op_gemm.h`, commit c8c0c0a |
| Global-queue ASAP order (09-04) | Emit-time stable sort of each (segment, XCD) window by ASAP rank | TPOT 27.700 → 27.490 ms (−0.210, −0.76%), TTFT −6.1 ms, exact | promoted | default on in `packet::devbuild` (bb8cd21), `PLOW_GQ_ORDER=emit` rollback |
| `f_a_proj` GEMV 128 → 64 WGs | Exact-shape WG table for 69 KDA GEMVs | TPOT 45.1468 → 44.6793 ms (−1.04%), one pair | pending | `PLOW_GEMV_WG_TUNING=128x7168=64` |
| Fused `GemvQkv(Nv=0)` | One 140-WG launch for the 128+12 column pair | isolated 7.8879 → 3.5032 µs (−55.6%); network TPOT +0.097 ms | rejected | keep `--experiment-parallel-linear2` off; commit 62e7130 |
| B1 short-K GEMV column ladder | R4→R2→R1 column groups | weighted body 2.998 → 2.956 ms (−1.4%), exact | rejected for production header | `runtime/bench/amd/k3_gemvbf16_bench.hip` |
| lm_head row sharding (09-03) | 20480-row rank-local head + `XArgmaxFin` | TTFT +8.140 ms, TPOT −0.088 ms, E2E −14.372 ms, exact | rejected | do not enable `PLOW_K3_SHARD_HEAD` |
| D9 XReduce same-packet epilogues | Prefix add / routed RMSNorm inside the one-shot XReduce | hot −0.005 / −0.103 µs per site → 0.0104 ms/token | rejected | — |
| D6 XReduce+AttnRes gang | Leader-elected AttnRes phases inside the XReduce packet | 186 eligible sites; resource gate passes (VGPR 48, 0 spills, LDS 147,464 B); no timing | design-only | needs `PLOW_GQ_BATCH=1` + D6 marker |
| Tagged one-shot XReduce (09-04) | 8-byte tagged words, no counter, strict rank-order accumulate | in-network body 15.63 µs; tagged hot 3.716 vs one-shot 9.766 µs; projected ~2.5 ms/token; TP8 gate 3 alternating 8192→256 folds: TPOT 26.44/26.48/26.44 vs 28.59/28.57/28.57 ms (−2.12 ms/token, −7.4%), TTFT neutral, checksum identical | promoted, default on (bb8cd21) | CMake `PLOW_XR_TAGGED`, `PLOW_XR_TRACE_PHASES=1`, `scripts/k3_xr_decode_report.py` |
| MLA decode merge-fold split-tile (09-04) | (m,l) merge pass rewritten from 128 serial scalar round trips to a split-tile map; `PLOW_MLA_FOLD_DVT=8` | isolated 22.8 → 10.3 µs cold, bit-exact; TP8 gate 3 folds: TPOT 28.22/28.28/28.23 vs 28.56/28.53/28.57 ms (−0.31 ms/token), DVT16/32 28.23/28.20, checksum identical, TTFT neutral; decode-MLA specialist +1 VGPR spill (6 → 7) | promoted, default on (merge 37c141d) | `runtime/amd/op_attention.h`, `-DPLOW_MLA_FOLD_DVT=0` rollback |
| Sequence-parallel TP seams (09-04) | AttnRes / router / top-k / latent xe on the reduce-scatter-owned T/8 band, results all-gathered (`XReduceScatter` + `XAllGather`); no weight replication | TP8 gate 8192→256, 3 alternating folds: TTFT 961.8/963.2/963.1 vs 1071.8/1072.7/1072.4 ms (−109 ms), TPOT neutral, checksum identical, audited run exact | promoted, default on (8b2555d) | `PLOW_SEQ_PAR_SEAMS`, branch codex/seq-parallel-seams |
| Decode router top-k wave-parallel select (09-04) | Per-wave rank + merge instead of 48 serial rounds; lowest-id ties, same packed key | isolated 15.85 → 12.07 µs, byte-identical routing on 64 tables; TP8 with seams: TPOT 25.08/25.10 vs 25.28/25.24 ms (−0.18 ms/token), checksum identical | promoted, default (8e18b50) | `PLOW_MOE_ROUTER_SELECT=1`, `runtime/amd/op_moe.h` |
| Segment-relative ASAP window order (09-04) | `gq_asap_ranks` ranked over the whole program, so a ready `MoeCombine` sorted behind `GemvGlu` in the post-DOWN segment; ranks are now relative to each segment's launch | TP8 gate on the full stack, 8192→256: TTFT 955.2/956.2 vs 962.4 ms, TPOT 24.23/24.31 vs 25.06 ms/token (−0.8), checksum identical, order-only | promoted, default on (8c9c42c) | `PLOW_GQ_ORDER=asap` (program-wide) / `=emit` rollback; branch codex/decode-l2-seg-asap |
| KDA key-factor object pair ON (09-04 screen) | Build the exact standalone Wu/carry key-factor objects so the runtime routes the marked pair instead of the interpreter Wu + register-state carry | 8192→256 on the final stack: TTFT 997.2 vs 955-956 ms (+41 ms), TPOT 24.27 vs 24.2-24.3, checksum identical | rejected; keep `PLOW_HSACO_KDA_KEY_FACTOR=OFF` (the pair displaces the faster regstate carry) | CMake `PLOW_HSACO_KDA_KEY_FACTOR` |
| Deterministic DOWN→COMBINE tree | Balanced reduction tree, 2 slots/leaf, WG512 | 9.290 vs 14.148 µs → 0.445 ms/token vs 0.5 ms gate; 1 f32 ULP, BF16 exact | rejected | `runtime/bench/amd/k3_moe_grid_sweep.{hip,cpp}` |
| DOWN→COMBINE phase objects | XCD-local rendezvous instead of a packet boundary (3 prototypes) | +49.5% / +229.0% / +312.4% vs control | rejected | `k3_moe_down_combine_xcd` |
| KDA decode fused 256×16 (09-02) | Fused KDA decode block | 6.902 / 7.347 µs B1/B8 vs vLLM 8.40 / 9.24; fused-block gate 14.099 vs 14.030 ms/token (+0.49%) | benchmark only | `plow_kda_decode_fused_256x16_2` |
| MLA GF4 ns32 (09-02) | Decode MLA split rung | B1 38.532 µs (−17.3%, beats AITER 43.232); B8 85.913 µs (~2× AITER) | measured rung; GF6 rejected (4 spills, +31%) | `PLOW_K3_NS` |

### Prefill — MoE

| Experiment | Mechanism | Measured | Verdict | Switch / evidence |
|---|---|---|---|---|
| Lean deterministic stage-2 (09-03) | Wave64 `MoeGroupDownPf` writing fixed f32 `part[row_partidx]`, no atomics | TTFT 2515.296 → 1873.100 ms (−642.195, −25.53%), 3 folds exact; +637 MiB/rank | promoted | `PLOW_MOE_STAGE2_LEAN=0` rollback |
| Lean stage-1 BK256 (09-03) | Standalone `MoeGroupGluPf` object, BK256, 119,808 B LDS | TTFT 2688.504 → 2547.352 ms (−141.152, −5.25%), exact | promoted | `PLOW_MOE_STAGE1_LEAN=0` rollback |
| Stage-1 reusable A4 (09-04) | Quantize/sort once, reuse A4 tile across N tiles (2 launches) | isolated 2.119 → 1.727 ms (18.55%); TTFT 1368.667 → 1280.662 ms (−88.006, −6.43%), exact | promoted | `PLOW_MOE_STAGE1_A4_REUSE=0` rollback |
| Stage-1 schedule screen (09-04) | grid / priority / BN / BK / WGM / NT / epilogue sweep | best 1.17% (SiTU+alias); BN128 +44.7%, BK128 +27.5%, WGM +17–19% | rejected; schedule axis closed | AITER exact-shape ceiling 616.07 µs |
| Fixed-order lean Combine (09-03) | Pure `MoeCombinePf` segment, exact f32 order | TTFT −109.947 ms (6.789%), SD 1.244, exact | promoted | `PLOW_MOE_COMBINE_LEAN=0` rollback; commit edffb73 |
| Router + align parallel (09-03) | Per-token router slices (cap `n_cu`) + four align packets | TTFT −8.639 ms (−0.309%); router+align 82.334 → 70.956 ms, exact; stack-3 gate with the c8 tile on stack-2: TTFT 1072.3/1072.0/1072.0 vs 1095.1/1095.0/1094.7 ms (−22.9 ms), TPOT 25.28 vs 25.30, 1- and 256-token checksums identical | promoted, default on (stack-3 gate) | `PLOW_MOE_ALIGN_PAR=1` |
| Atomic prefill accumulate | f32 atomics in the DOWN epilogue | TTFT +29.421 ms (+0.758%); three different checksums | rejected | `PLOW_MOE_PF_ATOMIC` |
| Combined P1+P2+P3 gate (09-03) | lean stage-2 + XR2 gather + KDA qpre, BF16 KV | TTFT 3055.374 → 2064.309 ms (−991.065, −32.4%), exact; still 3.63× vLLM | P1/P2 promoted, P3 default-off at the time | first coherent BF16-KV gate; earlier P1/P3 cells were fp8-KV |
| 2D EP×TP expert layout (09-04) | Single-resident layout, existing all-rank reduction | prefill boundary 3.210 → 2.519 ms/layer (EP2×TP4, 63.6 ms/92 layers); decode +0.396 ms/token | pending prototype | `packet::moe_ep::Moe2dLayout` |
| EP prefill boundary design (09-04) | Whole-expert ownership for prefill only | EP8 1.774814 vs 3.209992 ms/layer (132.036 ms, 44.7%; net 104.90 ms after transport); P2P 308 GB/s/rank | design-only; K3 blocked by memory | `PLOW_MOE_PREFILL_EP_MAX_EXTRA_BYTES` |

### Prefill — collectives

| Experiment | Mechanism | Measured | Verdict | Switch / evidence |
|---|---|---|---|---|
| XR gate aggregation | Aggregate gate-arrival signalling in `d_xreduce_twoshot_mega` | T1024 19.007 → 12.931 µs (−32%); network −7.888 ms (−0.27%), exact | promoted | `PLOW_XR_AGG=1` |
| Folded-gather two-shot (09-03) | Column-partition gathers as reduce-scatter/all-gather | 330.068 → 94.424 µs/collective (3.50×); TTFT median −240.63 ms (−8.07%), exact | promoted | `PLOW_XR2_GATHER=0` rollback |
| RS-U2 reduce-scatter (09-04) | Independent-element U2 reduce-scatter (from the phase trace) | RS 104.313 → 70.708 ms; TTFT median −35.130 ms, TPOT neutral, exact | promoted (gfx950 default) | `PLOW_XR_RS_U=1` rollback; `PLOW_XR_TRACE_PHASES=1` |
| XReduce phase objects (09-04) | Every `XReduceTwoShot` in its own spill-free segment | TTFT 1262.286 → 1284.928 ms (+22.642, +1.79%), exact | rejected | `PLOW_PHASE_OBJECTS` stays off; fixes 97f3cba, e834351 kept |
| Wave-per-peer reduce-scatter (09-04) | Waves 0–7 load one peer each into LDS | weighted +3.04% plus +464 launches → +3.623 ms projected | rejected | `PLOW_XR_WAVE_RS=1` |
| Token-slice pipelining (09-04) | Split `o_proj → XReduceTwoShot` seams into K row bands | emit exact; expected +2..+8 ms TTFT; one WG/CU forbids overlap | design-only | `PLOW_XR_SLICES=K` |
| XReduce→AttnRes fusion (09-03) | AttnRes in phase 2 of the two-shot via a standalone segment | TTFT +91.738 ms (+5.677%); 188 extra drains at 0.488 ms | rejected | `PLOW_FUSE_XR_ATTNRES=1` |
| AITER custom-AR parity (09-04) | AITER registered/eager vs Plow production grids | after the unit fix (below): Plow 2.1× / 1.5× faster at 14 / 7 KiB, AITER 0.79–0.86× at 28–112 MiB; AITER fails strict rank order | rejected porting | `scripts/bench_aiter_custom_ar.py` |
| Phase-chain AQL replay (09-04) | Prebuilt AQL chains, one doorbell per chain | host enqueue 0.192 → 0.003 µs/packet (~118 µs/624); device period unchanged | design-only | `PLOW_PHASE_OBJECTS=1`, `runtime/bench/dispatch/aql_launch_floor` |
| TP prefill segment-major (09-04) | Segment-major submission, one drain per chunk | TTFT −10.958 ms, TPOT −0.022 ms, exact | promoted | `PLOW_TP_PREFILL_SEGMENT_MAJOR=1`; commit 7731f5a |

### Prefill — KDA / MLA / attention

| Experiment | Mechanism | Measured | Verdict | Switch / evidence |
|---|---|---|---|---|
| Device recurrent-state clear (09-03) | Replace 2,208 host SDMA copies per request | TTFT −160.998 ms (9.04%) | promoted | `PLOW_STATE_CLEAR_DEVICE=0` rollback; commit 91a6d24 |
| Materialized residual fusion (09-03) | `Residual → AttnRes` fused-input rewrite in `Builder::finish` | prefill −3.196 ms; TPOT −0.894 ms (−1.97%), exact | promoted (all models) | `PLOW_FUSE_RESIDUAL_INPUT=0` rollback |
| Chunk KDA prefill scan (09-03) | Four-op chunk path (BT64) vs serial recurrence | T8192 11.803 → 6.2225 ms (1.897×), T128 0.404×; output RMS 5.83e-3 | promoted for single-sequence T ≥ 512 (6786c4b) | `scripts/bench_kda_chunk_gfx950.sh` |
| KDA chunk schedule screen | grid / wave-count sweep | best 6.3175 ms at grid 256; 4 waves +26%; 6 waves invalid | rejected | `KDA_CHUNK_WAVES` |
| KDA qpre BF16 (09-03) | `d128_qpre` carry/Wu variants | TTFT −120.046 ms (−6.332%), SD 1.028, exact; Carry 265.683 → 143.193 ms | promoted | `PLOW_KDA_CHUNK_QPRE=0`, `PLOW_KDA_FAMILY_ROUTE=false` rollbacks |
| KDA intra 8-wave parallel solve (09-03) | Row-distributed forward substitution | 1.8775 → 0.4134 ms (4.54×); 8192→256 diverged 252/256 | rejected; cached bit-exact variant opt-in | `PLOW_KDA_INTRA_CACHED=1` |
| KDA intra wave-items (09-04) | Independent (chunk, head) items per wave, exact order | 1.741776 → 0.570285 ms (3.054×); TTFT −84.158 ms, exact | promoted | `PLOW_KDA_INTRA_WAVE_ITEMS=0` rollback; aab03bb |
| KDA carry register state (09-04) | Carry state in MFMA accumulators, prefetched key factors | 1.916 → 0.726 ms (2.64×), hwcvt 0.572 ms; 0 mismatches; TP8 gate 3 folds: TTFT 1144.5/1144.1/1144.1 vs 1256.3/1256.0/1256.4 ms (−112 ms), checksum identical, TPOT neutral | promoted, default on (bb8cd21) | `PLOW_KDA_CARRY_REGSTATE=1` |
| MLA prefill TR16 (09-04) | `ds_read_b64_tr_b16` PV-stage reads | 4006.959 → 3340.793 µs/layer; TTFT −17.9 / −18.9 ms, exact | promoted | `PLOW_MLA_PF_TR16` default on |
| Materialized MLA prefill, isolated (09-03) | AITER "Opus" 8-wave schedule | T8192 7438.902 → 349.083 µs (7.7×); RMSE 3.2e-7 | isolated accept; V1/template variants rejected | AITER 10b192f5 |
| Materialized MLA prefill, network (09-03) | Generic `D_QK=192/D_V=128` route | TTFT −8.63%; continuation diverged (255/256) | rejected, default-off | `PLOW_MLA_MATERIALIZED_PREFILL`; commit f5e3ec7 |
| GemmWide c8 tile (09-04) | Tagged 128×384×64 at `8192×1536×7168` | 164.398 vs 223.929 µs (26.6%); TTFT −11.462 ms, TPOT +0.04, E2E −1.327 ms, exact; gated jointly with align-parallel on stack-2 (−22.9 ms TTFT, exact) | promoted, default shape 8192x1536x7168 (stack-3 gate) | `--emit-gemm-wide-c8-shape`; a42b21b, fb017cd |
| AttnRes f32-mix, interpreter arm (09-04) | vLLM-order f32 mix feeding the output norm | 8–18% isolated, below the 10% gate at T8192; 42 SGPR spills | rejected | `runtime/bench/amd/attnres_f32_mix_norm` |
| AttnRes f32-mix phase object (09-04) | Standalone `plow_attn_res_f32mix_gfx950`, 768-WG grid | 0.260 vs 0.589 ms (−56%); relL2 ≤ 2.1e-7; 0 spills; TP8 gate 3 folds: TTFT 1209.0/1208.7/1208.3 vs 1255.9/1255.8/1257.7 ms (−48 ms), TPOT neutral; GSM8K n=200: 122 vs 124 correct | promoted, default on (bb8cd21; C3 contract, tokens differ by design) | `PLOW_ATTNRES_F32MIX=1`, `PLOW_HSACO_ATTN_RES_F32MIX=ON` |
| Native 16×16×128 stage-2 (09-02) | Lean stage-2 reference object | 0.180 ms vs AITER 0.170; 5.1–5.4× faster than BM32 control, exact | reference for the lean route | `runtime/bench/amd/lean_moe_stage2_ref/` |

## Closed mechanisms

Do not reopen without a new implementation hypothesis.

- Same-packet XReduce epilogues (D9); phase-packet scheduling for exact-order
  grouped-MoE seams; row-fragment ready queues; the GLU→DOWN cooperative kernel;
  partitioning expert weights by output rows.
- Same-input two-output GEMV fusion as a network lever; the GEMV ladder in the
  decode megakernel; grid re-selection for decode GEMV; MM4 GEMV width.
- lm_head sharding until the `XArgmaxFin` handoff is cheaper.
- Porting AITER's rank-rotated two-stage all-reduce; AITER's 80-WG cap;
  wave-per-peer reduce-scatter; packet-granularity token slicing; a co-resident
  lean XReduce object while the prefill object is fat; the interpreter register
  envelope as the collective-cost explanation.
- XReduce→AttnRes fusion through extra segment boundaries (both the raw segmented
  and the spilling interpreter arm).
- MoE stage-1 schedule-only tuning (grid, waves, priority, tile width, WGM, NT);
  BK64; preshuffled companions; dual TP+EP resident layout on K3; atomic prefill
  accumulation; router slices beyond resident CUs.
- Eight-wave parallel KDA forward substitution (order-divergent); chunk grid/wave
  tuning against the carry floor; force-inlined carry.
- Generic MLA prefill BKV64 / 8-wave / full-register prefetch / 32×32 MFMA
  variants; GF6 MLA; materialized MLA prefill without exact continuation.
- The interpreter f32-mix AttnRes arm; SLP-vectorized f32 mix; 128-VGPR budget.
- Scheduled-packet HF planner bridge for K3 (no lowering for the hybrid ops).
- The old pre-HIER gate ceiling and the 13.20 → 3.84 µs signal number for sizing
  `sc1`.

## Erratum: isolated collective benches before 2026-09-04

`runtime/tests/tp_allreduce_bench.c` and `tp_allreduce_prefill_bench.c` scaled
`s_memrealtime` by the 1 GHz `HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY`; the counter
is the 100 MHz REFCLK. Every number those harnesses printed before the fix is
10× understated: one-shot 14 KiB 0.981 → 9.81 µs, 7 KiB 0.964 → 9.64 µs,
two-shot 112 MiB 63.45 → 634.5 µs, and the RS-U2 focused projection 17.1 →
171 ms (so the "integrated body is 10.9× the isolated projection" gap does not
exist). The AITER parity ratios become 2.1× / 1.5× / 0.79–0.86× rather than
7–21×. `tp_p2p_bench.c`, `tp_tilegate_bench.c`, `tp_coherence_bench.c`, and
`tp_moe_combine_xreduce_bench.c` were not fixed. Network folds and in-network
trace attributions are unaffected.

Smaller corrections carried from the folded reports: the older grouped-MoE
microbenchmark defaulted to unsharded I=3072 (the TP8 packet uses I=384) and is
not a valid AITER comparison; the pre-09-03 GLU→DOWN cooperative probe counted
16,384 rather than 32 arrivals per XCD; the trace reporter previously summed
overlapping packet envelopes (an impossible 47.65 ms inside a 43.84 ms span);
the c8 gate at a42b21b reported 7326/7326 measured decisions (fb017cd fixes the
accounting to 7650/7650); standalone P1/P3 cells before the combined gate used
fp8-KV packets.

## Folded reports

Removed on 2026-09-04 in favour of this file (git history keeps the full text):
`gfx950-mla-prefill-tr16`, `gfx950-moe-2d-layout`, `gfx950-moe-ep-prefill-design`,
`gfx950-moe-stage1-a4-reuse`, `gfx950-moe-stage1-schedule-screen`,
`kda-chunk-gfx950-preprod`, `kda-chunk-gfx950-schedule-screen`, `kda-intra-gfx950`,
`kimi-k3-amd-kernel-gap-20260902`, `kimi-k3-decode-inventory-prune`,
`kimi-k3-hybrid-phase-emit`, `kimi-k3-mi355x-aiter-xreduce-parity`,
`kimi-k3-mi355x-atomic-prefill-screen`, `kimi-k3-mi355x-b1-gemv-ladder`,
`kimi-k3-mi355x-combined-prefill`, `kimi-k3-mi355x-current-attribution`,
`kimi-k3-mi355x-d6-gang-feasibility`, `kimi-k3-mi355x-d9-xreduce-epilogues`,
`kimi-k3-mi355x-decode-grouped-moe`, `kimi-k3-mi355x-decode-mla-segments`,
`kimi-k3-mi355x-fa64-screen`, `kimi-k3-mi355x-gemm-c8`, `kimi-k3-mi355x-gemv-qkv-nv0`,
`kimi-k3-mi355x-kda-qpre-bf16`, `kimi-k3-mi355x-lm-head-sharding`,
`kimi-k3-mi355x-materialized-mla-prefill`, `kimi-k3-mi355x-materialized-residual-fusion`,
`kimi-k3-mi355x-mla-materialized-prefill`, `kimi-k3-mi355x-moe-align-parallel`,
`kimi-k3-mi355x-moe-combine-fixed-order`, `kimi-k3-mi355x-moe-decode-route-rule`,
`kimi-k3-mi355x-moe-stage1-bk256`, `kimi-k3-mi355x-moe-stage2-deterministic`,
`kimi-k3-mi355x-prefill-attribution`, `kimi-k3-mi355x-served-decode-attribution`,
`kimi-k3-mi355x-xr-decode-tagged`, `kimi-k3-mi355x-xreduce-attnres-fusion`,
`kimi-k3-mi355x-xreduce-phase-trace`, `kimi-k3-mi355x-xreduce2-gather`,
`kimi-k3-plowrt-mi355x-parity-smoke`, `kimi-k3-plowrt-mi355x-smoke-20260902`,
`kimi-k3-plowrt-mi355x-throughput-gates`, `kimi-k3-plowrt-mi355x-wide-decode-20260902`,
`kimi-k3-xreduce-gfx950`, `mi355x-attnres-f32-mix-norm`, `mi355x-attnres-f32mix-object`,
`mi355x-gate-hier-decode-sizing`, `mi355x-gemv-glu-un7`, `mi355x-gq-asap-order`,
`mi355x-kda-carry-regstate`, `mi355x-kda-intra-wave-items`, `mi355x-moe-deterministic-tree`,
`mi355x-moe-down-combine-phase`, `mi355x-phase-chain-replay`,
`mi355x-tp-prefill-segment-major`, `mi355x-xreduce-nwg-sweep`,
`mi355x-xreduce-phase-object-gate`, `mi355x-xreduce-token-slice-pipelining`,
`mi355x-xreduce-wave-rs-design` (all `perf-data/*-2026090{2,3,4}.md` or undated
gfx950 files of the same campaign).
