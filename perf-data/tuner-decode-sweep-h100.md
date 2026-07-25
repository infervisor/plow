# Tuner decode sweep — first measured grid (H100 NVL)

Implements `perf-data/tuner-decode-sweep-design.md`. The design argued from the
26B campaign that the decode knobs need a *search*, not a constant, because the
optimum moves with occupancy and with context. This is the first run of that
search, and it measures both claims directly.

Harness: `scripts/tune_decode_sweep.sh` (driver) + `tunedb-decode ingest|best`
(record path). Machine-readable artifacts alongside this card:

| file | what |
|---|---|
| `tune-decode-h100-26b-fp8.jsonl` | the screening grid, 3 reps/cell |
| `tune-decode-h100-26b-fp8-confirm.jsonl` | winners + rivals re-measured at 5 reps |
| `tune-decode-h100-12b-bf16.jsonl` | 12B ctx curve, harness validation |
| `tuning/nvidia/sm_90a/h100-nvl/decode_measurement.jsonl` | the `tunedb` records |

## Read this first — the contention caveat

**No number in this card was taken on a verified-idle GPU.** This box carried a
foreign allocation of **52573 MiB** — a PID outside our namespace, which
`gpulease` cannot see and cannot evict — for the entire session, and a second
agent was measuring concurrently for part of it. Every row therefore records
`vram_before_mib` and `uncontended: false`, and `tunedb-decode ingest` **refuses
to publish them as qualified**; they are stored `provisional`, unselectable.
`tuning/README.md` is explicit that a contended run is discarded rather than
stored with a caveat, and that rule is now enforced in code rather than
remembered.

What can be said in their defence: the foreign holder sat at **0 % SM
utilisation** throughout, and rep-to-rep spread is **0.001–0.015 ms** on a
~5 ms step. What cannot: that is an argument for internal consistency, not the
verification the policy asks for. Treat every absolute number below as
provisional and every *comparison within a cell* as sound — all cells were
measured under the same holder, minutes apart, on the same binaries.

**bf16 could not be measured at all.** The 26B bf16 configuration needs 53.3 GiB
(47.0 weights + 4.38 KV + 1.88 activations) and the holder left 43.3 GiB. That
is arithmetic, not a judgement call. The bf16 grid is built and queued; fp8 is
the campaign's own second precision and keeps the MoE arms live, so the whole
knob set is still exercised.

## Harness validation

Three independent checks, all passing, before any number was trusted:

1. **The tuner hook is the shipped recipe.** `build_sm90a_cubin.sh` gained
   `PLOW_EXTRA_DEFINES` (empty by default). Built at the identity point — every
   knob at its source default — it reproduces the shipped decode cubin
   **byte-for-byte**, `5edf093b5086…`, and the prefill object likewise.
2. **The scorer reproduces a known point.** 26B fp8, occ-1, ctx 1024:
   **5.779 ms** measured against the campaign's shipped **5.720 ms** — 1.0 %.
   Separately the 12B measured 13.079/13.119 ms against an independently taken
   13.307 ms.
3. **sm_120 is untouched.** Decode `7c1b6708…`, prefill `9380f825…`, identical
   between a clean `git archive HEAD` tree and this one, and identical to the
   hashes the campaign card records.

## Result 1 — the optimum moves with OCCUPANCY

26B fp8, ctx 1024, median of 3, spread ≤ 0.015 ms. occ-1 carries the shipped
unrolls, occ-2 the campaign's retuned ones (`GV_UNROLL=4 GV_UNROLL_GLU=2`),
because unroll depth inverts with occupancy.

| `n_cu` | ns default | 8 | 16 | 32 | 48 | best | REG |
|---|---|---|---|---|---|---|---|
| **132** (1 blk/SM) | 5.779 | 5.688 | **5.520** | 5.582 | 5.787 | **16** | 182 |
| **264** (2 blk/SM) | 4.790 | 5.019 | 4.830 | **4.774** | 4.942 | **32** | 128 |

Both curves are clean U-shapes and **their minima differ — 16 at one block/SM,
32 at two.** This is the design's central claim measured directly: a single
`#define` cannot serve both occupancies. The knob is worth **0.259 ms (4.5 %)**
at occ-1 and **0.245 ms (5.1 %)** at occ-2, both against the emitter's own
CU-fill default.

The occ-2 optimum **independently reproduces** the `NS_ABS=32` the campaign
chose by hand for its occ-2 bf16 reference point — a knob found on a different
precision landing on the same value here is the strongest evidence available
that the coupling is to occupancy and not to noise.

### Confirmation at the store's sample floor

`tunedb`'s `Stats` refuses fewer than 5 samples at construction, so the 3-rep
grid above is a *screening* artifact and cannot become a record. The winner and
its nearest rival were re-measured at 5 reps:

| cell | knobs | TPOT ms | reps | spread | vs 3-rep screen |
|---|---|---|---|---|---|
| occ-1, 1k | `ns16` (winner) | **5.521** | 5 | 0.004 | 5.520 |
| occ-1, 1k | `ns32` (rival)  | 5.596 | 5 | 0.004 | 5.582 |
| occ-2, 1k | `ns32` (winner) | **4.772** | 5 | 0.012 | 4.774 |

The occ-1 margin is **+0.075 ms** and `Stats::beats` calls it **decisive** — the
gap exceeds the dispersion in both. Every 5-rep median reproduces its 3-rep
screening value to ≤0.014 ms, which is the evidence that 3 reps was an adequate
*screen* even though it is not an adequate *record*.

The occ-2 rival (`ns` default) did not finish before the GPU ran out, so that
cell currently stores a winner with no margin — reported as "only candidate"
rather than as a tie, because those are different claims.

## Result 2 — two things the campaign does not contain

**The recorded `PLOW_NS_ABS=8` is stale for fp8.** Round 5 recorded 8 as the
best emit-time setting. On fp8 at occ-1 it is beaten by 16 (5.520 vs 5.688,
3.0 %). That value was measured on **bf16** and **before** the round-9
warp-per-row flash rewrite that cut flash from 1.079 to 0.686 ms — so the
constant decayed underneath a kernel change, which is exactly the failure mode
round 12 documented for `GV_MOE_UN`. The tuner's first run rediscovering its own
motivating defect is the argument for it.

**fp8 at occ-2 was never measured.** Round 6's table lists it as "not emitted".
It is worth **5.520 → 4.774 ms, 1.16×** — the single largest knob in this grid,
and larger than the 1.09× the same pair buys on bf16 (6.196 → 5.746). Occupancy
pays *more* on fp8, plausibly because fp8 halves the weight traffic and so
leaves the step relatively more exposed to the latency-bound small ops that
round 8 showed occupancy helps least — but that mechanism is inferred here, not
measured, and the ablation to confirm it was not run.

## Result 3 — ctx

Incomplete. The ctx legs were starved by concurrent GPU use and only the 12B
carries a full curve:

| model | 1024 | 8192 | 32768 |
|---|---|---|---|
| 12B bf16, occ-1, ns16 | 13.119 | 13.725 | 15.080 |

That reproduces the *shape* the design relies on — decode is not flat in ctx,
and the 1k→8k leg (+4.6 %) is much flatter than 8k→32k (+9.9 %), which is why
the buckets are geometric. It does **not** establish that the *knob optimum*
moves with ctx; that needs the `NS_ABS` sweep repeated at 8k and 32k, which is
the first thing to run when the card frees up. **The design's ctx claim is
therefore supported for the TPOT curve and untested for the optimum.**

## What the record path stores

`tunedb::DecodeCell` keys on `(hardware, model, dtype, n_cu, ctx_bucket)` with
geometric buckets, and `DecodeKnobs` carries the entire define-set, so a record
rebuilds its own object:

```
-DPLOW_NV_FORCE_MINBLK=2 -DGV_UNROLL=4 -DGV_MOE_UN=2 -DPLOW_MOE_DOWN_SG=4u -DGV_UNROLL_GLU=2
PLOW_UNISEG=1 PLOW_NS_ABS=32 --n-cu 264
```

Ranking happens strictly *within* a cell — comparing a 32k number against a 1k
one would report "long context is slower" as "this knob set lost", which is the
mistake the ctx axis exists to prevent.

## Reproduce

```
# objects (nvcc needs no GPU; ~30 s each)
PLOW_ROOT=$PWD PLOW_EXTRA_DEFINES="-DPLOW_NV_FORCE_MINBLK=2 -DGV_UNROLL=4 \
  -DGV_MOE_UN=2 -DPLOW_MOE_DOWN_SG=4u -DGV_UNROLL_GLU=2" \
  scripts/build_sm90a_cubin.sh <dir>/interp_sm90a.cubin

# the grid
scripts/tune_decode_sweep.sh --occ "1:132 2:264" --ns-abs "0 8 16 32 48" \
  --ctx "1024 8192 32768" --reps 3 --results perf-data/<out>.jsonl

# winners at the store's 5-sample floor, then record
scripts/tune_decode_sweep.sh ... --reps 5 --results <confirm>.jsonl
tunedb-decode ingest --db tuning --results <confirm>.jsonl   # --provisional if caveated
tunedb-decode best --db tuning --hardware nvidia/sm_90a/h100-nvl --json <out>.json
```

## Open

1. **`NS_ABS` × ctx at both occupancies** — the one cell of the design's thesis
   still untested.
2. **bf16**, blocked on VRAM, not on the harness.
3. **The object knobs** (`GV_UNROLL`, `GV_MOE_UN`, `PLOW_MOE_DOWN_SG`) — all 32
   objects are built; only the packet knob was swept before the GPU ran out.
4. **A correctness oracle in the loop.** The sweep measures speed; nothing in
   it checks that a configuration still produces the right tokens. `ingest`
   therefore defaults to `Correctness::Unchecked` — which alone blocks
   qualification — and `--correctness pass` is an assertion the caller makes
   *after* running `gpu_lifecycle`. Wiring that oracle into the sweep per config
   is what would let any of this become selectable.
