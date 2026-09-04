# Materialized MLA prefill (P5): continuation fix + Q0 gate kit — status 2026-09-04

Branch `codex/prefill-mla-materialized` (on top of `codex/amd-agent-harness` 986200d). Flag
`PLOW_MLA_MATERIALIZED_PREFILL` stays **default off**; the flag-off packet is byte-identical to the
served bundle. TP8 gates are the lease owner's; the exact commands are in §5.

## 1. What was wrong

- The route refused every chunk that was not an exact initial bucket
  (`prefill_prepare`: "requires one exact initial bucket (c0=0, clen=T)"). Under the served
  config (`PLOW_RAGGED_CHUNK=1`, ladder 128/512/1024/2048/4096/8192) that is every prompt whose
  length is not a bucket width — 300 tokens (512-bucket, clen 300) as much as 8400 (`[8192, 512]`).
- The reason it had to refuse: the K/V projection read `kv.*.ckv` rows `[0, T)` and packed the
  chunk's raw `krot_raw`, so a continuation chunk would have attended over the *first* `T` cached
  rows instead of `[0, c0 + T)`.
- The "8192→256 diverged 255/256" result (f5e3ec7 gate) was **not** a continuation bug: a
  `--random-input-len 8192` prompt is one exact bucket. Token 0 matched and token 1 flipped — decode
  drift from the 1-ULP attention difference through the old BF16 residual seam, now replaced by the
  f32-mix AttnRes (bb8cd21). The Q0 oracle, not a checksum, decides that part.

## 2. The fix (flag-on only)

devgen (`crates/devgen/src/k3.rs`, inside `if materialized`):
- `act.pf.{kv,k,v}_materialized` are sized at the cache capacity (`ctx` = 16384 rows), not the
  bucket: one shared set per program, +225 MB/rank (100 + 75 + 50 MB) over the T8192 sizing.
- The pack reads its rope half from the cache row `kv.*.krot` (depends on the krot writer) instead
  of the chunk's `krot_raw`. K3's MLA is NoPE (identity table), so for the chunk's own rows `krot`
  is a bit-exact copy of `krot_raw`; rows `[0, c0)` exist only in the cache.
- The kv_b projection is emitted unchanged (`M = T`, family picked at `T`); the runtime patches `M`.

runtime (`crates/plowrt/src/exec/amd.rs`):
- `rebase_mla_materialized_routes` runs per chunk after `rebase_chunk_rows` with the same
  `(rows, kv_len)` pair `in.kvlen` carries (`rows = clen` under RAGGED-M, else the bucket; `kv_len =
  c0 + rows`): the `kv_materialized` GEMM gets `M = kv_len`, the pack `T = kv_len`, the attention
  `N = rows`, `N_KV = kv_len`, batch strides and the flat grid `ceil(rows/256)·H` re-derived. The
  Opus object aligns its causal mask bottom-right (`causal_offset = N_KV − N`), so query `i` sits at
  absolute position `c0 + i` — the same `qpos = kv_len − n_tok + t` as `d_flash_mla_prefill`.
- The pack route carries the `krot` tensor id and is rebased to the active KV slot at launch (same
  mechanism as the KDA carry-regstate route); the K/V transient capacity is checked at load and at
  every chunk (refuses `kv_len` past 16384 rows rather than overrunning).
- Packed-prefill spans still fail closed (unchanged).

## 3. What still differs from the absorbed path, and why

1. Attention contraction order: materialized `softmax(Q_192·K_192ᵀ)·V_128` vs absorbed latent math —
   1 BF16 ULP on the attention output (max 0.0039, RMSE ~8.7e-5 on real weights; README).
   Inherent to the formulation, and the formulation vLLM itself uses (`fmha_fwd_hd192_hd128`).
2. Continuation chunks re-materialize the prefix's K/V with the *continuation bucket's* GEMM family
   (family picked at the bucket's `T`; tunedb records exist for 128..8192 × 3072 × 512). A different
   tile family than the 8192 chunk's can round a prefix K/V element differently by 1 BF16 ULP.
   Same class as (1); the alternative (one family for all buckets or a persistent 3 GB/rank
   materialized cache) was not worth it.
3. Near-tie argmax flips downstream of (1)-(2) (gap ≤ max|Δ|), amplified by autoregressive decode.
   Whether the whole-model logits stay within 2× vLLM's repeat floor is the Q0 question (§5).

## 4. Evidence on this source

- Tests: `cargo test --release -p devgen -p packet -p plowrt --features plowrt/hsa` — all green
  except the two pre-existing `tuned_tile_selection::gfx942_*` cells. New: devgen
  `materialized_mla_prefill_is_generic_opt_in_and_has_pure_raw_boundaries` (krot source, ctx-sized
  transients), plowrt `materialized_mla_routes_follow_the_chunk` (patch, reset, capacity refusal).
  `cargo fmt --check` clean; `cargo build --release -p plowc`, `-p plowrt --features hsa` ok.
- Flag-off packet byte-identical to the served bundle `/tmp/k3-l2`: sha `f999fd5aaf6e48eb…`,
  pairing `0x6892b68e52f0e447` (`/tmp/k3-mlamat-off`). Flag-on: sha `8dbbfac6aa7c83bc…`, pairing
  `0x942062db69d0321b`, Lean verified, 62 objects (`/tmp/k3-mlamat-on`). 288/7650 tile lookups fall
  back to the analytical model: the f5e3ec7 tunedb records for the materialized projections are
  keyed to an older GEMM label — requal before a `PLOW_REQUIRE_TUNED=1` serve (see §6).
- 1-GPU microcheck (`runtime/bench/amd/mla_materialized_prefill/run.sh`, 9 samples, MI355X):
  Opus 354.4 µs at T8192 (33.96 µs T1024, 35.28 µs T1025); flat-grid vs 3D oracle 0 mismatches at
  1024/8192/1025; same-weight full-path oracle max abs 3.8e-6 / RMSE 1.2e-7 (T8192); Opus vs
  absorbed on the kernel-only oracle max abs 3.05e-5 (T8192), 0.0039 = 1 ULP (T1024); full path
  4.07 vs 12.99 ms. Resource gate: 254 VGPR / 88 SGPR / occ 2 / 0 spill. `/tmp/mla-mat-micro.log`.
- 1-GPU end-to-end continuation probe, real weights: `PLOW_K3_LAYERS=8` TP1 emit (MLA layers 3, 7)
  ×2 arms, `amd-bench --tp 1 --steps 8 --dump-logits`, deterministic random ids
  (`/tmp/k3-mlamat-probe`):

  | prompt | chunks (ragged) | greedy tokens on vs off | prefill-row centered relL2 | decode rows relL2 |
  |---:|---|---|---:|---:|
  | 300 | `[512]` clen 300 | flip at prefill (gap 0.000 vs 0.062, near tie) | 6.1e-2 | histories diverge |
  | 8192 | `[8192]` | 8/9 identical; step 1 flip at gap 0.000 vs 0.062 | 6.1e-2 | 9e-3 … 1.3e-2 before the flip |
  | 8400 | `[8192, 512]` clen 208 | **9/9 identical** | 5.7e-2 | 8e-3 … 6.4e-2 |
  | 8700 | `[8192, 512]` clen 508 | **5/5 identical** | 9.6e-2 | 9e-3 |
  | 9000 | `[8192, 1024]` | both arms NaN — see below | — | — |

  The continuation chunks (8400, 8700) run and reproduce the absorbed tokens; the row error is the
  same order as the exact 8192 bucket, i.e. no route error on top of the formulation difference.
  9000 (and a 1000-token initial chunk) produce NaN logits on **both** arms of this truncated TP1
  blob, so the 1024-bucket program of that diagnostic emit is broken independently of this change
  (the served TP8 blob passed 9000 exactly in the regstate probe, 0b04dd2). Not chased here; the
  TP8 probe below covers 9000 on the real bundle.

## 5. TP8 commands (lease owner)

`G=docs/k3-mi355x-20260904/scripts/mla_materialized_gate.sh` from the repo root on this branch;
control = the served bundle (`/tmp/k3-l2` or a fresh flag-off bundle — byte-identical packet).

```sh
$G bundle /tmp/k3-mlamat-on PLOW_MLA_MATERIALIZED_PREFILL=1        # candidate (already built once)
$G bundle /tmp/k3-mlamat-off                                        # control, == /tmp/k3-l2 packet
$G prompts /tmp/k3-q0                                               # 300/1024/8192/8400/9000-token GSM8K text
# (1) continuation exactness probe + logit dump, both arms (amd-bench --tp 8 --steps 16 --dump-logits)
$G dump materialized /tmp/k3-mlamat-on /tmp/k3-q0
$G dump absorbed     /tmp/k3-mlamat-off /tmp/k3-q0
$G tokens /tmp/k3-q0                                                # first greedy divergence per prompt
# (2) Q0 oracle: vLLM 0.28 raw logits on the union of both arms' histories (x2 = repeat floor)
$G cases  /tmp/k3-q0
$G oracle /tmp/k3-q0                                                # pinned image, TP8, needs `sg docker`
$G compare /tmp/k3-q0                                               # PASS = top-1 >= 99.5%, >= 90% rows within 2x floor, no severe flip
# (3) GSM8K n=200, both arms
$G gsm8k materialized /tmp/k3-mlamat-on 8101
$G gsm8k absorbed     /tmp/k3-mlamat-off 8102
# (4) 8192->256 TTFT gate vs the served bundle: 3 alternating 8192->1 folds + one 256 pair
$G ttft /tmp/k3-mlamat-on /tmp/k3-l2 /tmp/k3-mlamat-gate
```

Acceptance: (1) continuation prompts (8400, 9000) must run and their divergence, if any, must be a
near-tie class flip; (2) `Q0_PASS` for the materialized arm and not worse than the absorbed arm;
(3) GSM8K within noise of the absorbed arm (122 vs 124/200 was the f32-mix delta); (4) TTFT
−100..−115 ms, TPOT neutral. The 256-token checksum is *expected* to differ from
`fnv1a64:71a28c1449921c95` (formulation change) — the oracle, not the checksum, is the gate.

## 5b. TP8 results so far (lease owner, 2026-09-04 21:00)

TTFT gate 829.5/830.6/829.1 vs 953.8/954.9 ms (−125 ms), TPOT neutral. Token probe + rank-0
logit rows (`k3_q0_oracle.py diff`, materialized vs absorbed, `/tmp/k3-q0`):

| prompt | chunks | greedy divergence | margin at the flip (mat / abs) | row max\|Δ\| there | prefill-row relL2 | matching-history rows relL2 |
|---:|---|---|---:|---:|---:|---|
| 300 | `[512]` clen 300 | none / 17 | — | — | 0.158 | 0.056–0.171 |
| 1024 | `[1024]` exact | step 0 (276 vs 11) | 0.750 / 0.500 | 1.51 | 0.127 | 0.109 (only one shared row) |
| 8192 | `[8192]` exact | none / 17 | — | — | 0.179 | 0.055–0.167 |
| 8400 | `[8192, 512]` clen 208 | step 6 (32 vs 7683) | 0.500 / 0.500 | 1.64 | 0.133 | 0.076–0.195 |
| 9000 | `[8192, 1024]` clen 808 | both arms NaN | — | — | NaN | NaN |

Verdict: the two flips are near-tie flips (gap < the row's max|Δ|), and every shared-history row —
flipping or not, exact bucket or continuation — carries the same arm-vs-arm error (centered relL2
0.06–0.2, max|Δ| 1–3.5 logits). That is the 1-ULP attention difference amplified through 24 MLA
layers of the whole model, the same order as vLLM's own repeat floor (full-row max 0.099), not the
1e-3 class and not a route error: the 8400 continuation rows (0.08–0.19) sit exactly where the
exact-bucket rows sit, and a broken chunk would read ~1.0 (as the post-flip, different-history rows
do). Whether 0.06–0.2 is inside 2× vLLM's floor is the Q0 `compare` stage's call.

9000 tokens: NaN on **both** arms, i.e. on the byte-identical served packet, through the 1024
bucket as a continuation chunk. On the truncated TP1 diagnostic blob the same program NaN'd as a
ragged initial chunk (1000 tokens) and as a continuation with `PLOW_RAGGED_CHUNK=0`; the exact
1024 bucket is fine on TP8. Pre-existing and independent of this branch (the 0b04dd2 regstate
probe passed 9000 through `plowrt bench` before the seams merged at 8b2555d). GPU bisection for
the lease owner, control bundle only: `amd-bench --prompt <9000 ids>` on an emit with
`PLOW_SEQ_PAR_SEAMS=0`, then `PLOW_KDA_CARRY_REGSTATE=0`, then `PLOW_RAGGED_CHUNK=0` at runtime.

## 5c. Q0 oracle (vLLM 0.28 TP8, 109 repeat pairs) — parity, but neither arm passes

vLLM repeat floor (conservative max over 109 pairs): full-row relL2 0.2551 (median 0.0455, p90
0.094), head64 0.0573, min top-64 overlap 0.844, argmax flips 2/109. All 17 "severe" flips per arm
were the gsm9000 NaN rows (§5b); excluding them:

| arm | rows | top-1 | within 2× floor (full) | worst | head64 within 2× | top-64 overlap med/min |
|---|---:|---:|---:|---:|---:|---|
| materialized | 68 | 72.1% | 38.2% | 4.46× | 23.5% | 0.61 / 0.30 |
| absorbed | 68 | 73.5% | 25.0% | 4.52× | 17.6% | 0.63 / 0.30 |

Per prompt (top-1 mat/abs): 300 → 100/100%, 1024 → 88/94%, 8192 → 53/53% (identical flip pattern
in both arms), 8400 → 47/47%. Every disagreement is a near-tie by the compare tool's rule; min gap
median 0.5 logits. Prefill rows agree 4/4 on both arms; the loss is on decode steps after long
(≥ 8192) prompts, scattered, not drifting. On the 44 histories shared by both arms: full relL2
median 0.549 vs 0.563, head64 0.148 vs 0.146, top-64 0.609 vs 0.633, top-1 32 vs 33, same
verdict on 43/44, arm-vs-arm argmax 42/44. Alignment checks: dumped-row argmax == the token Plow
sampled on 68/68 rows per arm (full-vocab row, right index); vLLM argmax == its sampled token
67/68; bf16 dump quantisation (0.125 at |logit| 16–32) explains ≤ 7/19 of the flips, none of the
relL2. Verdict: the materialized arm is at parity with the absorbed arm on this oracle; the
~50% decode top-1 on 8192+ prompts is the pre-existing whole-model Plow-vs-vLLM long-context
divergence (both arms, the served packet included) and is not attributable to MLA formulation.

## 6. Follow-ups / risks

- tunedb: re-qualify the 12 materialized projection shapes (`{128..8192}×2304×1536`,
  `{128..8192}×3072×512`) on this source before a `PLOW_REQUIRE_TUNED=1` serve; the emit currently
  reports 288 analytical picks for them.
- The continuation projection recomputes the prefix (`kv_len × 512 × 3072` GEMM per MLA layer per
  continuation chunk, ~0.1 ms at 8448 rows) — negligible for the 8192+tail case; a 16K prompt pays
  it once on its second chunk.
- Packed-prefill spans stay refused on the materialized route.
