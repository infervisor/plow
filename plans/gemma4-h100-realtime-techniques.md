# Gemma-4-12B H100 realtime (C1–C4 TTFT+TPOT): techniques, ranked

Written 2026-09-17 from code + records only; no new GPU runs. Paths are repo-relative
unless absolute. Measurement records live in the main checkout (`/home/lava/plow/perf-data`,
`/home/lava/plow/plans`), not this worktree.

Target (same H100 SXM5, vLLM 0.28, out 128, temp 0, C1): beat **BF16 TPOT 10.55–10.64 ms**
and **TTFT 28.2 / 46.7 / 170.2 ms** (in 128/1024/4096). Plow today (qualified roles):
TPOT 12.00 / 12.62 / 13.22, TTFT 42.2 / 89.8 / 213.6 (`plans/gemma4-dense-realtime-tracker.md`).
Needed: **−1.5 ms/step at ctx128, −2.6 ms at ctx4096; −14 ms TTFT at in128, −43 ms at in1024.**

## 1. What one C1 decode step is today (evidence)

- Program: **542 DevInsts, ≈32.9k workgroup-packets**, one cooperative launch
  `132 CTAs × 256 thr`, occ-1, **208 regs** (`crates/devgen/src/lib.rs:3729` `emit_phase`;
  `runtime/nvidia/interp_sm120.cu:714,2977-3009`; `crates/plowrt/src/exec/gpu.rs:6078-6097`).
- Per sliding layer (11 ops, `lib.rs:4677,4971,5001,5049,5234,5354,5418,5461,5646,5854,6340`):
  `GemvQkv`(132 blk) → 3×`HeadNormRope`(**2 blk each**, `lib.rs:4845`) → `FlashDecode`(132) →
  `FlashMerge`(16) → `Gemv` o(132) → `NormResidualNorm`(**1 blk**) → `GemvGlu`(132) →
  `Gemv` down(132) → `NormResidualNorm`(1). Full layers split q/k on 88/44 CTAs (`lib.rs:4694`).
  Tail: lm_head `Gemv`(132) → `SoftCap`(64) → `Argmax`(64) → `ArgmaxFin`(1) (`lib.rs:6497-6570`).
- Every projection is the FFMA `d_gemv` (`op_gemm.cuh:338-444`): 1 warp = 1 output column,
  128-bit `LDG`, `GV_UNROLL=8`, N blocked over 132 slices, no K split. sm90a `pick_tile`
  never consults tunedb (`lib.rs:601-606`). No tensor-core M=1 arm exists
  (`interp_sm90a.cu:8-11` states this is deliberate).
- Dependencies are per-packet counters (`ld.acquire`/`red.release`, `interp_sm120.cu:263-270`);
  **no grid.sync anywhere**. The cooperative launch only guarantees co-residency.
- Multistep: K=min(4, PLOW_MULTISTEP=8) at ≤2 live requests (`sched/multistep.rs:24-33`,
  `serve/mux.rs:936-942`); the host enqueues `[memset→decode→advance]×K`, one D2H + one sync,
  tokens surface only after the quantum (`gpu.rs:6151-6210`) ⇒ ITL median 0 / p99 4×TPOT.

### Where the ~5 ms/step gap to the 7.6 ms floor goes (3.154 TB/s sustained, 23.84 GB weights)

| Term | Evidence | Size at ctx128 |
|---|---|---:|
| GEMV bodies below HBM ceiling at occ-1 | block-0 trace gate/body/signal = 14.9/82.3/2.8 % (`/home/lava/plow/plans/h100-serving-omitted-block-cost-audit.md`); SXM ceiling 1 CTA/SM 2.752 vs 3 CTA/SM 3.154 TB/s (`perf-data/e0-hbm-ceiling-sxm-h100-20260905.csv`); Lt M1 GEMV 2.5–2.88 TB/s vs Plow 1.76–2.29 (`/home/lava/plow/plans/beat-vllm-h100.md:103`); 26B occ-1→occ-2 step 5.521→4.772 ms (`tuning/nvidia/sm_90a/h100-nvl/decode_measurement.jsonl`) | ≈2–3 ms |
| Serialized light stages: per layer NRN×2 (1 blk), HNR trio (2 blk), FlashMerge (16) ≈ 4 stages × 48 = ~200 fan-in→fan-out points; dispatch floor 2.0 µs/stage (`tuning/.../interpreter_calibration.jsonl`) + 3–6 µs single-block body | 200 × 5–8 µs | ≈1.0–1.5 ms |
| Decode attention (hd512 CUDA-core body, 208 regs, `op_attention.cuh:588`; "5.5–7.9× vs vLLM", `docs/bringup/tp-bringup-upstream-review-log.md:2811`) | Plow TPOT +1.22 ms from ctx128→4096 vs vLLM +0.09; KV bytes at 4K ≈0.4 GB = 0.13 ms | 0.2 → **1.1 ms at 4K** |
| Host: K launches + memsets + advance per quantum | one sync per 4 tokens | <0.1 ms |

Interpreter dispatch itself is **not** the lever: gate share includes producer imbalance; the
skeleton/ablate instruments exist but no absolute per-op µs is recorded (`interp_sm120.cu:329-354`).

### Direct per-role decode objects / "residency megakernel": evidence is against fragmentation
- GEMV512 role: Gemma12 C1 TPOT 12.64→13.07 (−3.4 %) (`/home/lava/plow/plans/h100-serving-omitted-block-cost-audit.md:32`).
- E2 512-thread role: 1.1–1.6 % block gain, killed for 5 extra graph nodes; cp.async weight ring 0.3–0.7 %; XREG 1.24× primitive → 1.3 % block (`/home/lava/plow/plans/h100-e0-e8-program-20260905.md`).
- cuBLASLt decode: 31B B1 fixed −8 %, adaptive +3.6 %, launch elision −6.4 % (`/home/lava/plow/perf-data/token-batch-main-h100-readiness.md`); C1 −6 % (review log H16); forces `multistep=0` (`gpu.rs:3150-3155`).
- Weights are already streamed exactly once per step; there is no "stream once" win left.
Verdict: keep one cooperative launch + counters; make the **per-rung object lean** (occupancy 2)
and **remove serialized stages**. Roles/library GEMV pay at B≥4, not at C1.

## 2. Ranked techniques

| # | Technique | Expected | Effort | Gate |
|---|---|---:|---|---|
| D1 | Occ-2 C1 decode object (`n_cu=264`, ≤128 regs, `GV_MM_MAX=1`, lean arms) | −1.0…−1.6 ms/step | 2–3 d | T1→T3 |
| P1 | Native split-K WS384 role for o/down/q at M≤512 | −5…−8 ms TTFT@128, −6…−10 @1024 | 4–5 d | T1→T3 |
| D2 | Fold NRN into consumer GEMV staging; widen HNR; `PLOW_FUSE_ARGMAX` | −0.4…−0.9 ms/step | 2–3 d | T1→T3 |
| D3 | Tensor-core hd512 flash-decode (GQA16 ⇒ M=16) + nsplit=33 law | −0.2 (128) … −1.0 ms (4K) | 3–4 d | T2→T3 |
| P2 | `PLOW_EMIT_PREFILL_CUBLASLT` A/B at 128/512/1024 (ceiling + fallback) | up to P1's number | 0.5 d | T1→T3 |
| D4 | Per-token ITL: per-step event + ring-row D2H (or `--multistep 0`) | ITL p99 4×→1×, TPOT ±0 | 0.5–1 d | T3 |
| P3 | Extend exact GLU role rows {128,512,1024,2048} | −0.2 (128) … −2 ms (1024) | 1 d | T1→T3 |
| D5 | W8A16 packet TPOT (free), then FP8 M1 arm audit on SXM5 clocks | toward vLLM FP8 7.15 | 0.5 d + 3 d | T2→T3 |
| C4 | B4–B16 TC GEMV (native TC / cuBLASLt) for C4 | C4 TPOT only | 5 d+ | T2→T4 |

### D1 — occupancy-2 decode object for the B1 rung
- Mechanism: compile the decode cubin so ptxas fits 2 CTA/SM (`__launch_bounds__(256,2)` ⇒ ≤128
  regs) and emit the decode programs with `n_cu=264` so every GEMV has 264 slices. All arms
  Gemma never runs (MLA/DSA/MoE/`gemv_rows<MM>` ladder) compiled out for the B1 object.
- Evidence: 26B fp8 B1 5.521→4.772 ms (−13.6 %) at `minblk 2, n_cu 264, gv_unroll 4, 128 regs`
  (`decode_measurement.jsonl`); hd512 GF4 flash decode is 104 regs standalone
  (`experiments/README.md:28`), so the 208 is the union of arms, not the hot path; SXM synthetic
  read 2.752→3.112 TB/s at 1→2 CTA/SM; lm_head 735→658 µs grid 1→2 (`beat-vllm-h100.md:464`).
- Arithmetic: 82 % of step ≈ 9.8 ms in GEMV bodies × 10–16 % ⇒ −1.0…−1.6 ms ⇒ 10.4–11.0 ms.
- Touch points: `scripts/build_sm90a_cubin.sh` (decode object: `PLOW_NV_LEAN_DECODE=1
  PLOW_NV_FORCE_MINBLK=2 GV_MM_MAX=1 GV_UNROLL=4`, `interp_sm120.cu:545-557,697-702`);
  devgen: decode-program `n_cu` decoupled from prefill `n_cu` (prefill roles assert
  `blocks == blob.n_cu` at `gpu.rs:598-608`, `gemma4_gemm_glu_role.rs:51`) — add
  `emit.decode_n_cu` in `knob_spec.rs` and thread it into `emit_phase` `all`/`split3`
  (`lib.rs:4694`); runtime decode grid check (`gpu.rs:6408-6411`) keyed per program.
  Keep `PLOW_GEMV_SPLIT=1` (S=2 measured +2.8 ms/tok at occ-1, `lib.rs:6609-6620`).
- Risks: spills at 128 regs (`PLOW_NV_FORCE_MINBLK` comment); `GemvGlu` unroll 4→2 changes
  nothing numerically (bit-exact unroll) but tunedb must sweep `GV_UNROLL{,_GLU}` at occ-2;
  ladder rungs 2–16 keep the fat object (multistep still selects per rung).

### P1 — split-K WS384 role for the under-filled small-M projections
- Facts: WS384 BM128/BN256 gives **15 tiles** for N=3840 (o, down), 16 for q N=4096, 32 for
  global q at M=128 (`op_gemm_sm90.cuh:1312`, tile math in
  `/home/lava/plow/plans/h100-short-prefill-tile-analysis.md`); measured M128 primitives
  down 158.2 µs (118 MB ⇒ 0.75 TB/s), o_local 50.2, o_full 89.7 (`beat-vllm-h100.md:278`).
  M64N64 (120 tiles) only reached 151.6/36.9/76.9 and **no serving win** (42.31→42.54 ms)
  because it re-reads A per N-tile and still runs one CTA/SM per tile.
  cuBLASLt beats ws384 **1.83–3.40× at M128**, 1.03–1.10× at 4096 (`beat-vllm-h100.md:25`);
  the tree has **no stream-K/split-K in any prefill body** (`op_gemm_splitk.cuh` is decode-only,
  `interp_sm120.cu:1788-1796`); the log reserves split-K for exactly this regime
  (`plans/gemma4-4k-8k-native-block.md:1073`).
- Mechanism: keep the BM128/BN256 tile and the 1P+2C ring; split K across S=⌈132/tiles⌉ CTAs
  (S=8 for N=3840 ⇒ 120 CTAs, each 1/8 of K); f32 partials to a `[S,M,N]` workspace; a
  **fixed-order** reduce+cast (reuse `ZeroF32`/`CastF32Bf16`/`GemmSplitK` opcodes,
  `crates/packet/src/dev.rs`) folded into the next light segment. A and B are read once total.
- Arithmetic (M=128): down 118 MB @3 TB/s ≈ 40 µs + 16 MB partials ≈ 50 µs vs 158 ⇒ −108 µs ×48
  = −5.2 ms; o_local −35 µs×40, o_full −60×8, q ≈ −35×40 ⇒ total ≈ −8 ms of the 14 ms gap.
  M=1024 (120 tiles) needs S=1; the 1024 gap is GLU (P3) + GEMM efficiency (P2 ceiling).
- Touch points: new body `d_gemm_sm90_tma_ws384_splitk_role` in `op_gemm_sm90.cuh` (fork of
  `:1312`, K-range from `entry.slice / S`); object `interp_sm90a_pfgemm_splitk.cu` on the
  `plow_ws384_role_loop` (`interp_sm120.cu:2430-2519`); devgen role predicate keyed on
  `tiles(m,n) < 132/2` (new `crates/devgen/src/gemma4_gemm_splitk_role.rs`, pattern of
  `gemma4_gemm_glu_role.rs:27-51`); `plow-asset/src/segment_roles.rs` new role id.
- Gate/risk: the packet numerics bound is tight (`act.fu` rel-L2 0.006252 vs 0.006 killed E3's
  last-arriver split-K, `h100-e0-e8-program-20260905.md`) ⇒ fixed-order f32 reduce only, then
  `scripts/gemma4_prefill_role_full_logits.sh`. Workspace `S·M·N·4` = 15.7 MB at M=128 fits.

### D2 — remove serialized light stages
- HNR: `hn_cus = ⌈t·heads/8⌉ = 2` blocks handle 16 heads of 256/512 (`lib.rs:4845`); one head
  per block (16/8/8 blocks) is a one-line change, bit-exact, ~4× shorter stage.
- NRN fold: each consumer GEMV block already stages `x` in smem (`op_gemm.cuh:353-355`);
  compute `resid = x + rms(o)·g_post` and `hn = rms(resid)·g_pre` redundantly per block
  (3840 elems, trivial), slice 0 writes `resid`. This is the AMD K3 `PLOW_K3_FUSE_NGEMV`
  pattern (default on, bit-exact, `docs/flags-reference.md:345`); the NV fused-norm GEMV arm
  is currently a trap (`interp_sm120.cu:1804-1808`). Removes 96 packets and 96 fan-in/out
  stages per step. Prefill `PF_GFUSE` fusion was rejected for changing checksums
  (`native-block.md:129`) — the decode fold must keep the BF16 rounding of `hn` identical.
- `PLOW_FUSE_ARGMAX` (`lib.rs:6470-6474`, `op_gemm.cuh:446`): replaces SoftCap/Argmax/ArgmaxFin
  (3 stages) with one epilogue; opt-in today; compatible with `plow_advance`.
- Arithmetic: 96 NRN stages × 5–8 µs ≈ 0.5–0.8 ms; HNR ≈ 0.1–0.2 ms; argmax ≈ 0.02 ms.
- Touch: `lib.rs:5461,6340` (NRN emission → GEMV `i3=norm_flag`), `op_gemm.cuh` x-staging
  path, `knob_spec.rs` opt-in `emit.nv_fuse_ngemv`. Gate T1 byte-identity of x/hn, T2 logits, T3.

### D3 — tensor-core hd512 flash-decode
- Global layers: 16 q heads share **one** KV head ⇒ S = Q[16×512]·Kᵀ is a natural
  `mma.sync m16n8k16` (heads as M) — the existing `gemv_sm90_transposed` trick
  (`op_gemv_transposed.cuh:8-9`) applied to attention. Current body is one KV row per thread,
  CUDA-core softmax, `FA_WPR` shipped (2.53×, `build_sm90a_cubin.sh:116-120`), GF_FULL=4.
- nsplit must follow `aligned = n_cu/gcd(n_grp,n_cu)` (ns=33 vs 48 = +41–47 %,
  `experiments/README.md:28`); the 12B rule at `lib.rs:4531-4542` is gated on `fp8_kv` and
  the GF rule on `kvh_full ≥ 4` (never fires on 12B, `experiments/README.md:28`) — fix both.
- Arithmetic: Plow ctx-slope 1.13 ms per 4K vs vLLM 0.09 ⇒ up to −1.0 ms at 4K, −0.2 at 128.
- Touch: `op_attention.cuh:588` (`d_flash_decode<512,GF>`), `lib.rs:4485-4542` nsplit
  gates. Gate: exact-order softmax (BKV16) T2 greedy agreement, T3 at ctx 128/4K/8K.

### P2 — cuBLASLt prefill at M∈{128,256,512,1024}
- `PLOW_EMIT_PREFILL_CUBLASLT` (`knob_spec.rs:927`, `dense_cublaslt.rs:175-213`) already routes
  M∈{128,256,512}×(3840,15360|8192) and M∈{1024..8192}×all 8 shapes
  (`plow-asset/src/segment_roles.rs:26-46`), each as its own segment (`gpu.rs:7881-7900`).
  Under pure-GEMM topology every GEMM is already a separate launch ⇒ **no extra boundaries**.
  Prefill Lt does not disable multistep (`gpu.rs:3132-3155` keys on decode roles only).
- Use: (a) the ceiling P1 must beat (campaign rule: promote native only ≥5 % over Lt,
  `native-block.md:195`); (b) pragmatic fallback for M=1024 where P1 does not apply.
  The H18 A/B (`review-log.md:2805`) is unrun. Extend `CUBLASLT_PREFILL_ROWS` shapes to q/o if
  the A/B says so. Gate: `gpu_prefill_roles_match_control_logits`, then T3 at 128/512/1024.

### D4 — per-token streaming without losing TPOT
- Option A (no TPOT change): in `multi_step_at_most` (`gpu.rs:6151-6210`) add per step
  `memcpy_dtoh_async(ring row)` + `cuEventRecord`; mux (`mux.rs:2107-2145`) polls events and
  emits tokens as they land; keep the single sync per quantum. Cost: K tiny D2H + events ≈ µs.
- Option B (zero code): `--multistep 0` for the realtime profile. Cost = one host turnaround
  per 12 ms step (unmeasured on 12B; the only record is 26B K=8 vs 32 = 3.6 %). Measure first:
  if `--multistep 0` costs <1 % TPOT, ship it and skip A.

### P3 — GLU role rows
- `interp_sm90a_pfgemm_glu_gemma4.cu:17-18,71` accepts only M∈{4096,8192}; ≤2048 rungs run
  gate, up, `Glu` separately (extra `act.gt/ut` write+read = 4·M·15360·2 B per layer). At M=1024:
  126 MB/layer × 48 = 6 GB ≈ 2 ms; the 4K role measured −3.3 % (`native-block.md`). Tiles at
  M=128 = 120 (91 %). Touch: `min_rows`/`exact_shape` (`gemma4_gemm_glu_role.rs:27-29`),
  object validator; same object, no kernel change. Gate T1 hashes, T3 at 128/512/1024.

### D5 — FP8 weights (to chase vLLM FP8 7.15 ms)
- FP8 weights 11.92 GB ⇒ 3.78 ms floor; vLLM FP8 sits at 53 % of it. The W8A8 packet's decode
  arms take bf16 activations (`flags-reference.md:575`) ⇒ its TPOT is a **free measurement**
  (packet exists; prefill 164.9 ms at 4K). The M1 fp8 arm was "compute-bound, 1046 GB/s" on
  the 600 MHz-throttled NVL (`op_gemm.cuh:2808-2811`); at SXM5 1.83 GHz the budget is
  ≈78 warp-inst per 8 B/lane vs the arm's ≈28 ⇒ should be bandwidth-bound; `PLOW_NV_FP8_RB`
  optimum flips with occupancy (`build_sm90a_cubin.sh:99-131`) ⇒ re-sweep with D1.
  FP8 WGMMA decode is B8/B16-only and not qualified (cos 0.9954, review log H17).

### C4 (secondary) — B4–B16 decode
- Rung 4 runs `gemv_rows<4>` FFMA; E3 split-K is closed (1.07–1.25× Lt, packet numerics);
  native TC 35.4 vs Lt 32.7 µs at B8; cuBLASLt route wins C4/C8 (31B B8 76.5→36.2 ms) but
  disables multistep and costs C1. Do after D1–D3; measure 12B C4 first (co-tenant must go).

## 3. Streaming summary (Q4)
`MultiStep::for_batch` caps K at 4 for ≤2 live; the device ring is drained once per quantum.
Per-token ITL = D4-A (event per step) or `--multistep 0`; either keeps the decode program and
counters untouched. `PLOW_MULTISTEP` is `rt.multistep` (`crates/plowrt/src/knob_spec.rs:376`).

## 4. Model-shape keying (Q5) — what must change for a new Gemma variant
- sm90a GEMM tile: fixed by `PGM90_*` macros; `pick_tile` returns one opcode per encoding
  (`lib.rs:601-633`); tunedb GEMM plans are gfx950-only (`lib.rs:550-565`).
- Exact roles: `gemma4_gemm_glu_role.rs:28` `(4096|8192, 15360, 3840)`, compile-time
  `N=15360,K=3840` in `op_gemm_sm90.cuh:1569-1572`; hd512 snake hard-coded `seq_q==4096`
  (`op_attention.cuh:1964-1979`); attention roles by hd/gqa (`attention_prefill_role.rs:24-69`);
  cuBLASLt shape table (`segment_roles.rs:29-38`); 31B gates `c.hidden==5376 && inter==21504`
  (`lib.rs:4152-4162`), 12B nsplit signature `kvh_full==1 && fp8_kv` (`lib.rs:4531`).
- Decode knobs (`n_cu, minblk, gv_unroll, ns_abs`) are tunedb cells keyed
  `(hardware, dtype, n_cu, ctx_bucket, model, batch)` (`decode_measurement.jsonl`) — only
  h100-nvl/26B cells exist; **no h100-sxm5 decode cell for 12B**.
- Required: (1) role predicates read `(hidden, inter, hd, gqa, window)` from the checkpoint
  config and objects take them via `plow_config.h` (`PLOW_CUBIN_CONFIG` already stamps
  packets, `build_sm90a_gemma4_segments.sh`); (2) a `tunedb` sm90a GEMM key
  `(arch, dtype, m, n, k)` → object SHA + tiles/occupancy envelope (the tracker's missing
  "compiled resource-aware devgen selection"); (3) an sm90a decode sweep cell per
  `(model, batch, ctx)` that D1's `n_cu/minblk/unroll` read from instead of hand `-D`s.

## 5. Two-week sequence (this H100; every run under `gpulease -n 1`, co-tenant gone)
1. **d1–2 attribution.** Full-ladder packet C1/C4 baseline; `PLOW_NV_TRACE=1` gate/body per
   opcode; `PLOW_PF_SEG_TIME=1` per-class at 128/1024 rungs; `--multistep 0` A/B; W8A8-packet
   TPOT. These four numbers decide the split between D1/D2/D3 and P1/P3.
2. **d3–5 D1.** Lean occ-2 B1 cubin + `emit.decode_n_cu`; T1 byte-identity of prefill
   programs, T2 logits, T3 TPOT at 128/1024/4096; tunedb sxm5 cells.
3. **d5–7 P2 then P1.** Lt A/B at 128/512/1024 (half a day) sets the ceiling; then the split-K
   role for o/down/q; full-logit gate; T3 at 128/512/1024.
4. **d7–9 D2.** HNR widen (trivial) → NRN fold → `PLOW_FUSE_ARGMAX`; one variable per A/B.
5. **d9–11 D3.** hd512 TC decode + nsplit/GF gate fixes; T2 greedy agreement, T3 at 4K/8K.
6. **d11–13 D4 + P3.** Streaming; GLU role rows.
7. **d14 T4.** Served C1/C4 grid vs the vLLM tables, `perf-certs/<id>.json` for every flip.

## 6. What cannot beat vLLM at parity precision (honest)
- **vLLM FP8 TPOT (7.15 ms) with BF16 weights**: impossible; the BF16 weight floor is 7.6 ms
  at 3.154 TB/s. Only D5 (FP8 weights) can approach it, and that is a different precision.
- **C1 BF16 TPOT**: D1+D2 project 10.3–11.0 ms — parity-to-slight-win, not a decisive win;
  ctx≥4K needs D3 as well. vLLM's 67 %-of-floor is a strong reference for a 542-op chain.
- **C4/C16 decode**: FFMA rung-4/8 GEMV cannot match TC GEMM at M≥4; the native TC and
  split-K work is closed at 1.08–1.25× Lt, and the Lt route removes multistep and costs C1.
- **TTFT at 4K** (1.25×): native GEMM is 0.87× Lt at wide M; ping-pong / TMA-store /
  multicast are all measured or structurally refused (`experiments/README.md:26,68`,
  `native-block.md` WS384 ping-pong row). Bounded ≈10 %.
- **TTFT at in128**: after P1, ≈483 launches × 3–5 µs drain/refill ≈ 2 ms remain structural
  (`PLOW_PF_SEG_GRAPH` removes host cost, not the GPU drain).
- Numerics: every reordering (split-K, fused norms) must pass the 0.6 % rel-L2 / greedy-checksum
  gates that already killed E3 and `PF_GFUSE`.
