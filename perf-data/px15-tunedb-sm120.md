# px15 — batch in the tunedb cell key, attention knobs wired, the sm_120a cell

Campaign `px15-tunedb-sm120`. RTX 5090 (sm_120a, 170 SMs, 32 GiB), driver 580.159.03,
CUDA 13.0. Model gemma-4-12B-it, fp8 weights (`/root/gemma4-fp8-ckpt`).

Three jobs: put `batch` in the cell key, make the attention knobs actually swept
rather than merely representable, and generate the first sm_120a cell. All three
are done. The tuner **rediscovered two of the four known winners, contradicted
one, and could not reach the fourth** — details and diagnosis in §5.

---

## 1. What changed in the schema

### `batch` is now part of `DecodeCell`

`{hardware, dtype, n_cu, ctx_bucket, model}` → `{hardware, dtype, n_cu, ctx_bucket,
model, batch}`.

This is not tidiness. `GV_MM_MAX` is the widest `gemv_*_rows<MM>` the object
instantiates; a batch of B costs `ceil(B/GV_MM_MAX)` weight passes, so the knob's
entire effect is a function of B. `op_gemm.cuh`'s own ladder has it inverting —
355 tok/s for `=8` vs 294 for `=16` at B=8, then 387 vs 520 at B=16 — and a cell
that cannot name its batch can hold only one of those two facts. A fully
populated batch-blind tunedb could not have caught the campaign asset that
shipped `=16` while serving B=8 (`px10`: −19.4 % @131k, −33.8 % @1k). **Measured
here at B=8 the penalty is −32.3 %** (§3), so the failure the axis exists to
prevent is live on this hardware, not inherited from another box.

The field has **no `serde(default)`**. A stored record with no batch is not a
batch-1 record, it is a record whose provenance was lost, and a default would
convert the second into the first silently. `a_cell_without_a_batch_refuses_to_load`
pins that.

### `gv_mm_max: Option<u32>` joins the typed `gv_*` family

`None` = "not overridden" = the source default 8. `Option` rather than `0` for the
same reason the flash knobs are: `-DGV_MM_MAX=0` is not the default, it is a
`gemv_walk` that instantiates no rung at all.

### `fa_gf_full` now renders on **both** sides

See §4 — it is a pair, not a define.

### Migration of the H100 rows: `batch = 1`, recovered not assumed

The three existing `nvidia/sm_90a/h100-nvl` records were rewritten in place with
`"batch":1`. The justification is mechanical, not editorial: until this campaign
`tune_decode_sweep.sh`'s only `step_bench` invocation was

    "$STEP_BENCH" "$adir" 1 "$ctx" "$STEPS"

— a **literal `1`** for the slot count. The harness could not have measured those
rows at any other batch. The value is read off the script that produced them.

The raw-sweep row type (`Row.batch`) *does* default to 1, for exactly that
reason and no other; the stored record type does not. The asymmetry is
deliberate and commented at both sites.

---

## 2. Bugs found on the way

Six, all pre-existing on `main`. Four blocked the campaign outright.

| # | bug | effect | fixed |
|---|---|---|---|
| 1 | `tunedb-decode best` has **no `--print`** and no cell filters | `tuning/README-decode-tuner.md` documents `best --model … --ctx … --print defines` as *the* way a build consumes the store. It never existed — the documented consumption path did not run. | yes |
| 2 | `scripts/build_sm120_cubin.sh` ignores `PLOW_EXTRA_DEFINES` | sm90a has the hook; sm120 did not. The sweep could *name* a knob but not build an sm_120a object with it — the reason `tuning/` had a sm_120a-shaped hole. | yes |
| 3 | `tune_decode_sweep.sh` calls bare `gpulease` | Not on PATH on this box. Does not fail the sweep — it fails every *run*, so the grid completes having recorded nothing and reads as "no trustworthy samples". | yes |
| 4 | `device::cuda` probes `/usr/local/cuda/compat` first | Right when the toolkit outruns the driver, wrong here: compat ships **580.167.08** against a **580.159.03** kernel driver, so every run died with `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`. Not fixable via `LD_LIBRARY_PATH` — `/usr/lib/x86_64-linux-gnu` on the path shadows nix's glibc and the binary dies in the loader instead. | worked around (`PLOW_LIBCUDA` pinned by both sweep scripts); the probe order itself left alone |
| 5 | `devgen`'s `const FA_GF_FULL = 2` vs the build script's `-DPLOW_NV_FA_GF_FULL=4` | The constant's own doc says it **must** equal the kernel's. Nothing enforces it, and they are already out of step on `main`. | override added (§4) |
| 6 | `cargo test --workspace` does not compile at `HEAD` | `plowrt` lib tests: missing `l2_domains`/`l2_sms` in `Program`, missing `target` in `Model`. Verified by `git stash` — **pre-dates this branch**. | **no** — not mine to fix, reported |

Two more traps that are not bugs but silently produce wrong records, now guarded:

- **`step_bench` clamps the batch silently.** `slots = want.min(engine.batch())`.
  A packet emitted for 1 slot answers a `--batch 8` request with a batch-1
  number and nothing says so. The sweep now parses the batch back off
  `RAW_STEP slots=` and refuses to record a point whose actual batch differs
  from the requested one.
- **`plowc --batch` is the *prefill bucket* list, not the decode batch.** The
  decode batch is `PLOW_DECODE_BATCH`, baked into the packet at emit (devgen
  sizes KV, activations, GEMV M, flash `n_batch` and per-sequence argmax from
  it). So batch is a **packet** axis; the sweep emits one packet per batch.

---

## 3. Block rankings (RANK only — see §6 for what was confirmed)

Method as directed: sweep wide and cheap on single-layer block assets, confirm
the shortlist end-to-end. A block asset is not the `gemv_lab` mistake — it drives
the **real interpreter** (same cubin, dispatch, counter protocol, register
footprint). It still cannot give magnitudes, only ordering.

Gemma-4-12B layer classes, and why both are needed:

| class | layers | q heads | kv heads | head_dim |
|---|---|---|---|---|
| sliding (win 1024) | all others | 16 | 8 | 256 |
| **full** | 5,11,17,23,29,35,41,47 | 16 | **1** | 512 |

### `GV_MM_MAX` × batch — sliding block (L0), decode µs median, lower better

| B | ctx | `=8` (default) | `=16` | winner | margin |
|---|---|---|---|---|---|
| 1 | 1024 | **208.8** | 210.2 | 8 | −0.7 % |
| 1 | 8192 | **208.9** | 210.2 | 8 | −0.6 % |
| 4 | 1024 | 275.7 | **274.3** | **16** | −0.5 % |
| 4 | 8192 | 275.4 | **273.4** | **16** | −0.7 % |
| 8 | 1024 | **343.8** | 508.1 | 8 | **−32.3 %** |
| 8 | 8192 | **343.9** | 507.8 | 8 | **−32.3 %** |

Registers: `=8` → 241, `=16` → 247. Both occ-1, smem 34432 either way.

This is the whole argument for the batch axis in one table. The winner is 8, then
16, then 8 again as B walks 1 → 4 → 8, and the B=4 flip is inside noise while the
B=8 one is a third of the step. Pooled into one cell the answer would be "8" for
every batch, and the 19–34 % that `px10` lost would be invisible.

Note the register story differs from `op_gemm.cuh`'s comment, which was measured
on an RTX PRO 6000 before the WS-batched shallow-unroll rungs (`GV_UN16`/`GV_UN32`)
landed and reports 212 → 255 + 72 B spill. On sm_120a today it is 241 → 247 with
no spill reported, so the modern penalty is **not** the register cliff the comment
describes. The 32 % is still there; its mechanism has moved and the comment is now
stale about the *why*. Worth a separate look — I did not chase it.

### `FA_GF_FULL` — FULL block (L5, kv_heads = 1), decode µs median

| B | ctx | `=2` | `=4` | `=8` | winner |
|---|---|---|---|---|---|
| 1 | 1024 | **236.9** | 247.0 | 269.9 | 2 |
| 1 | 8192 | 272.1 | **271.1** | 307.0 | 4 (tie) |
| 1 | 32768 | 417.0 | **358.3** | 443.5 | **4** (−14.1 %) |
| 1 | 130560 | 971.8 | **689.3** | 986.9 | **4** (−29.1 %) |
| 8 | 1024 | **382.5** | 384.3 | 418.7 | 2 (tie) |
| 8 | 8192 | 605.1 | **536.5** | 568.6 | **4** (−11.3 %) |
| 8 | 32768 | 1506.9 | **1132.5** | 1216.4 | **4** (−24.8 %) |
| 8 | 130560 | 5054.3 | **3480.3** | 3688.1 | **4** (−31.1 %) |

Registers 241 and smem 34432 for **all three** arms — so the H100 campaign's
"widening the arena bills every other op" explanation does **not** apply on
sm_120a. The arena does not move here.

`=4` wins every cell at ctx ≥ 8k, at both batches. `=8` never wins. See §5 for
the diagnosis against the validation gate.

---

## 4. `FA_GF_FULL` is a pair, and the pairing was unenforceable

`devgen` derives the full layers' `nsplit` from `n_grp = heads / FA_GF_FULL` so
that `n_grp × nsplit` fills the resident grid; the kernel derives how many query
heads one flash work item carries from `PLOW_NV_FA_GF_FULL`. The two are the same
number seen from opposite sides, and the emitter's constant carries a comment
saying so — but one is a Rust `const` and the other an `nvcc -D`, so nothing could
enforce it, and on `main` they disagree (2 vs 4).

For a tuner this is worse than a wrong constant: sweeping the define alone
re-splits work in the kernel while the packet keeps sizing for GF=2, so each arm
measures a **compiler/kernel disagreement** and the sweep reports it as the knob's
effect.

Fixed by making the emitter read `PLOW_FA_GF_FULL` (unset ⇒ byte-identical
packets), and by having `DecodeKnobs::emit_env` render the packet half whenever
`defines` renders the object half — the same treatment `(FORCE_MINBLK, --n-cu)`
already gets.

**But it did not change these numbers, and I checked rather than assumed.**
Emitting the full block at GF ∈ {2,4,8} gives **byte-identical packets** for this
shape:

- the `kvh_full >= 4` grid-alignment path is gated off (Gemma-4-12B is kvh_full=1);
- the `kvh_full == 1` path is gated on `fp8_kv`, and even with `PLOW_FP8_KV=1` the
  packets stay identical, because `aligned = n_cu / gcd(n_grp, n_cu)` and on
  **170 SMs** `gcd(16/GF, 170) = 2` for every GF ∈ {2,4,8}. 170 = 2·5·17, so the
  alignment is GF-independent *on this part*. It would not be on the 188-SM
  RTX PRO 6000 (188 = 4·47).

So the §3 `FA_GF_FULL` ranking is a clean object-only sweep, unconfounded — and
the pairing fix matters for the *next* part, not this one.

---

## 5. The validation gate

| knob | expected | tuner found | verdict |
|---|---|---|---|
| `GV_MM_MAX` | **8** at B=8 (not 16) | **8**, by **−32.3 %** at B=8 | **REDISCOVERED** |
| `PLOW_NV_FA_GF_FULL` | **8** | **4** at every ctx ≥ 8k and both batches; 8 never wins | **CONTRADICTED** — diagnosis below |
| `PLOW_FP8_LD16` + `PLOW_FP8_FAST` | both on, 1.61×, bit-exact | *not run* | **NOT RUN** |
| `NS_FULL_ABS` | 32 at long ctx | *not run* | **NOT RUN** |

### Diagnosing the `FA_GF_FULL` contradiction

Treating it as a tuner bug first, as instructed. Three candidate explanations,
two eliminated by measurement:

1. **Packet/kernel mismatch** (§4) — *eliminated*: packets are byte-identical
   across GF on this part, verified by sha256.
2. **Arena tax**, the H100 explanation — *eliminated*: smem is 34432 and registers
   241 for all three arms, and occupancy is 1 throughout. Nothing widens.
3. **Grid fill.** `n_grp = 16 / GF_FULL`, so GF 2/4/8 → 8/4/**2** work-item groups.
   Wider fusion buys fewer KV re-reads and pays in parallelism, and at GF=8 there
   are only 2 groups to spread over 170 SMs. `=4` is the turning point between the
   two, which is exactly the shape of the measured curve (2 → 4 improves, 4 → 8
   regresses, at every long-ctx cell).

What I think reconciles it with `px11`: the gate's evidence is *"1.52× on the
flash-decode op"* — an **op-level** number — while every number above is a whole
decode step through the real interpreter. That is precisely the substitution
`README-decode-tuner.md` §2 forbids, and the H100 round already caught the same
knob doing the same thing (`FA_GF_FULL=8`'s regression there was only ⅔ flash).
Second, the gate says "deployed is 2", which is true of `runtime/CMakeLists.txt`'s
gemma target but **not** of `scripts/build_sm120_cubin.sh`, which already ships
**4**. If `px11` compared 8 against a 2-baseline, then 4 was never in the running —
and against 2 alone, 8 *is* better at B=8/8k (568.6 vs 605.1), just nowhere near
1.52× and worse everywhere at B=1.

I am not claiming `px11` is wrong about the op. I am claiming the op result does
not survive being scored end-to-end, which is the one rule this tuner exists to
enforce. **The shipped `-DPLOW_NV_FA_GF_FULL=4` is correct and should not change.**

### Why the other two gates were not run

**`PLOW_FP8_LD16` / `PLOW_FP8_FAST` — attempted, blocked by a reproducible crash.**
Not a budget excuse; the stage is written (`px15_block_campaign.sh fp8`) and was
run twice. Findings, in order:

1. The arms live inside `else if constexpr (FP8KV)` in `op_attention.cuh`. On a
   weight-only-fp8 packet that branch is never taken, so sweeping them on the
   ordinary asset would faithfully report "no difference" for a knob never
   compiled into the executed path. Both the object (`-DPLOW_FP8_KV=1`) and the
   packet (`PLOW_FP8_KV=1 PLOW_FP8_KV_FULL=1`) must opt in. Emitted; KV halved
   0.25 → 0.13 GiB, so the packet genuinely took the fp8 path.
2. Every fp8-KV **block** asset then dies in prefill with
   `cuMemcpyHtoDAsync: CUDA_ERROR_LAUNCH_FAILED`, at both batches, on all four
   arms.
3. `build_sm120_cubin.sh` documents that fp8 prefill needs the synchronous
   staging arm (`-DPLOW_NV_FA_PIPE=0`, "cp.async cannot convert fp8 inline").
   Added to every arm. **Still faults**, identically.

So there is a live bug in fp8-KV on single-layer block assets on sm_120a,
independent of these two knobs. I did not diagnose further — it is a different
campaign's bug, and guessing a ranking from a crashed run is exactly what this
document is supposed to prevent. **NOT RUN**, with a reproducer.

**`NS_FULL_ABS` — not attempted.** GPU budget. The card was held by two other
agents for most of the window (27.4 GiB resident at the start, `gpulease` queue
depth 2–3) and the priority order was schema → tuner correctness → the batch
gate → the cell. The `--ns-full-abs` axis exists and is wired; it has no
measurements behind it. Marked NOT RUN rather than estimated.

One thing worth flagging for whoever picks it up: `NS_FULL_ABS` may not be the
lever it was on the H100. `devgen` now derives the full layers' split from a
**grid-alignment** rule, and on 170 SMs `aligned = n_cu / gcd(n_grp, n_cu) = 85`
for every GF — so the emitter's own value is already alignment-aware here, and a
hand-set 32 would *break* that alignment rather than improve it. Test the
emitter default as an arm, not just the constants.

---

## 6. The sm_120a cell

`tuning/nvidia/sm_120a/rtx-5090/decode_measurement.jsonl` — the first non-H100
cell in the store. gemma-4-12B-it, fp8 weights, `n_cu=170` (1 block/SM), ctx 1024,
B ∈ {1, 8}, 5 reps, **QUALIFIED**.

### End-to-end confirmation (full 48-layer model, `step_bench` TPOT)

| B | `GV_MM_MAX=8` | `=16` | winner | margin | block predicted |
|---|---|---|---|---|---|
| 1 | 10.650 ms | **10.634 ms** | 16 | −0.15 % | 8 by 0.7 % |
| 8 | **16.659 ms** | 25.557 ms | **8** | **−34.8 %** | 8 by 32.3 % |

Provenance for all four rows: `vram_before = 2 MiB` (card verifiably ours),
`stable = true` (relative rep spread 0.0006–0.0023, two orders under the 0.01
threshold), registers 241 / 247.

**The block ranking and the end-to-end result agree where it matters.** At B=8
the block predicted −32.3 % and the model gave −34.8 % — and `px10`, measuring
independently, got −33.8 % at 1k. Three methods, one answer. At B=1 the two
disagree in *sign* but both differences are ≤0.7 %, i.e. the block called a
0.7 % effect and the model called a 0.15 % one the other way; neither is a
reason to build anything.

That B=1 row is the one caveat I want on the record. The store marks it
`decisive`, and by its own rule it is — `Stats::beats` compares the gap against
the dispersion, and these runs were so quiet (0.0012 relative spread) that
0.15 % clears it. **Decisive is not the same as worth acting on.** The honest
reading of the B=1 cell is "these two objects are the same speed"; I would not
change a build for it, and I have not.

### Correctness

Not taken on trust. `scripts/px15_correctness.sh` runs the `gpu_lifecycle`
oracle against **each of the four measured assets** — load the real engine,
serve the canonical prompt, decode greedily:

    PASS  ..._mm16_..._b1   reply: "Paris"
    PASS  ..._mm16_..._b8   reply: "Paris"
    PASS  ..._mm8_..._b1    reply: "Paris"
    PASS  ..._mm8_..._b8    reply: "Paris"

Identical output from every arm, so `ingest --correctness pass` is an assertion
that was actually made rather than a flag that was passed. Without it the rows
would sit provisional and unselectable, which is the store working as designed.

### It reads back

    $ tunedb-decode best --db tuning --hardware nvidia/sm_120a/rtx-5090 \
        --batch 8 --ctx 1024 --print defines
    -DPLOW_NV_FORCE_MINBLK=1 -DGV_UNROLL=8 -DGV_MOE_UN=2 -DPLOW_MOE_DOWN_SG=4u -DGV_MM_MAX=8

    $ ... --print emit
    PLOW_UNISEG=1 --n-cu 170

`--print defines` refuses when the filter leaves more than one cell standing: a
flag string names one object, and the union of two cells' winners is an object
nobody measured.

---

## 7. Tuner pick vs today's hand-set constant

"Today" = what `scripts/build_sm120_cubin.sh` + the source defaults actually
build. Agreement is listed too, since a tuner that only reports disagreements
cannot be trusted when it stays quiet.

| knob | source default | sm120 build ships | tuner pick | agree? | evidence |
|---|---|---|---|---|---|
| `GV_MM_MAX` | 8 | 8 (unset) | **8 @ B=8**, 16 @ B=1 | **yes at B=8** | e2e −34.8 %; block −32.3 % |
| `PLOW_NV_FA_GF_FULL` | `= FA_GF` (2) | **4** | **4** | **yes** | block, 8 cells; `=4` wins every ctx ≥ 8k |
| `PLOW_NV_FORCE_MINBLK` | unset | unset (→ occ 1) | 1 | yes | only arm swept; 241 regs ⇒ occ-1 regardless |
| `GV_UNROLL` | 8 | 8 (unset) | 8 | yes (untested) | not swept — carried, not confirmed |
| `GV_UNROLL_GLU` | 4 | 4 (unset) | 4 | yes (untested) | not swept |
| `PLOW_NV_FA_GF` (sliding) | 4 | **2** | — | — | not swept |
| `PLOW_NV_FA_WPR` | 0 | 0 (unset) | — | — | not swept; note sm90a ships **1** |
| `PLOW_NV_FA_KUN` | 1 | 1 (unset) | — | — | not swept |
| `PLOW_NS_ABS` / `NS_FULL_ABS` | emitter-derived | emitter-derived | emitter default | — | not swept |
| `PLOW_FP8_LD16` / `PLOW_FP8_FAST` | off | off | — | — | fp8-KV path only; blocked (§5) |
| devgen `FA_GF_FULL` | 2 | — | **should be 4** | **NO** | §4 — disagrees with the object the build ships |

**What we are not currently building with:** nothing, at B=8. The tuner's B=8
pick is byte-identical to the shipped recipe, which is the most useful negative
result here — the hand-set `GV_MM_MAX=8` and `FA_GF_FULL=4` are both *correct*,
and now they are correct *on evidence* rather than by inheritance.

The one live discrepancy is not a cubin flag at all: **`devgen`'s `FA_GF_FULL`
constant is 2 while the object ships 4.** On a 170-SM part that is currently
inert (§4), so it is a latent bug rather than a present loss — but it will bite
the first time this model is tuned on a part whose SM count is not 2·5·17.

---

## 8. Gates

| gate | status |
|---|---|
| `cargo test -p tunedb` | **PASSED** — 41 tests, incl. 3 new ones pinning the batch axis |
| `cargo test --workspace --exclude plowrt` | **PASSED** |
| `cargo test --workspace` | **FAILS AT HEAD, NOT MINE** — `plowrt` lib tests do not compile; reproduced on a clean `git stash` of this branch |
| H100 rows migrated and still rank | **PASSED** — `best --all` prints both cells with `b1` in the key |
| `best --print defines` round-trips | **PASSED** — refuses when >1 cell matches, emits one line when narrowed |
| `PLOW_EXTRA_DEFINES` reaches the sm120 object | **PASSED** — cubin 2 804 312 B → 3 556 056 B under `-DGV_MM_MAX=16`, registers 241 → 247 |
| block ranking: `GV_MM_MAX` × batch | **PASSED** |
| block ranking: `FA_GF_FULL` | **PASSED** (contradicts the gate; diagnosed) |
| block ranking: fp8 arms | **NOT RUN** — attempted twice, `CUDA_ERROR_LAUNCH_FAILED` on every fp8-KV block asset (§5) |
| block ranking: `NS_FULL_ABS` | **NOT RUN** — GPU budget |
| end-to-end confirm, B ∈ {1,8} @ ctx 1k | **PASSED** — 5 reps, idle card, stable, block ranking agrees |
| correctness oracle on all 4 objects | **PASSED** — `gpu_lifecycle`, identical replies |
| cell published QUALIFIED | **PASSED** — `tuning/nvidia/sm_120a/rtx-5090/` |
| sm_120 default build stays byte-identical | **PASSED** — `git archive HEAD` vs tree: decode `7c1b6708…`, prefill `9380f825…`, both sides |
| ctx 32k / 131k end-to-end | **NOT RUN** — B=8 at max_ctx 8192 already needs 34.6 GiB (KV is not window-capped at allocation); needs a smaller batch or paged KV |
| B=4 end-to-end | **NOT RUN** — block ranking says it is the interesting cell (the flip), so this is the first gap to close |
