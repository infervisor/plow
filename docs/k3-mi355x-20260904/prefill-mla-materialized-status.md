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

## 6. Follow-ups / risks

- tunedb: re-qualify the 12 materialized projection shapes (`{128..8192}×2304×1536`,
  `{128..8192}×3072×512`) on this source before a `PLOW_REQUIRE_TUNED=1` serve; the emit currently
  reports 288 analytical picks for them.
- The continuation projection recomputes the prefix (`kv_len × 512 × 3072` GEMM per MLA layer per
  continuation chunk, ~0.1 ms at 8448 rows) — negligible for the 8192+tail case; a 16K prompt pays
  it once on its second chunk.
- Packed-prefill spans stay refused on the materialized route.
