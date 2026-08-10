# GLM-5.2-FP8 TP8 on gfx942 — prefill chunk POLICY: the acceptance class, decided

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **PLOW-ARCHITECTURAL** — the acceptance class for a chunk-plan change. The policy is arch-independent; the bucket ladder it selects over is tuned per box.

> **RESOLVED, 2026-08-09.** §4's recommendations 2 and 3 landed together, and
> recommendation 4's precondition — "build that gate and the decision is
> unblocked" — was met first: `perf-data/probes/facts_gate.py` is the standing
> quality gate, it was **proven to fail on a deliberately degraded arm before it
> was trusted to pass** (exit 1, 14 regressions, p=0.0001), and ragged passed it
> (77/84 → 79/84, 2 regressions against 4 repairs, p=0.89). `PLOW_RAGGED_CHUNK`
> now defaults ON and the runtime's hardcoded `MAX_CHUNK` is gone in favour of
> the packet's own ladder. Restated headline TTFT **370 / 752 / 1418 / 3296 ms**
> at 1k/4k/8k/16k. See `glm52-facts-gate.md` and `docs/arch/13`.
>
> Recommendation 1 stands unchanged: the `LAUNCH_ROWS` reprice did NOT land, and
> `LAUNCH_ROWS` is still 416. Under ragged it is never read.

**One sentence: the output of this engine is determined by the LAST chunk's
executed row count and by nothing else about the chunk plan — identical plan gave
byte-identical text in 62 of 62 long free-form cells, and two DIFFERENT plans gave
byte-identical text whenever both ran a wide last chunk — so `PLOW_RAGGED_CHUNK`
and the `LAUNCH_ROWS` reprice do not introduce a numeric regime, they DELETE one
(the 128-row tail), which is why the acceptance question that has blocked this
area should be answered YES for ragged and NO for the reprice-on-its-own.**

| | |
|---|---|
| branch / base | `chunk-policy` off `worktree-glm52-bringup` @ `529654b` |
| blob (Task 1) | `/workspace/assets/gfx942/glm52-tp8-final2/model.pkt` — UNCHANGED, no re-emit |
| blob (Task 2) | `/workspace/assets/gfx942/cp-{ctl16,rung16}/model.pkt` — emitted here; `cp-ctl16` is `cmp`-IDENTICAL to `final2`, so the rung is the only difference |
| objects, every arm | `/root/.claude/jobs/b09a4bcc/tmp/hsaco_glm18` — UNCHANGED, including for the 16384 rung |
| serve env | `PLOW_MLA_PF_V2=1`; arms add `PLOW_LAUNCH_ROWS=1780` or `PLOW_RAGGED_CHUNK=1` |
| arm attribution | every arm's `prefill chunk policy` log line quoted in §6 — not inferred from the numbers |
| session | 2026-08-08 19:51–20:34 UTC, GPU lock held continuously; 5 server loads, coherence gate PASS on all 5 |
| harness | `perf-data/probes/chunk_policy_{battery.py,run.sh,analyze.py}`, `chunk_maxchunk_run.sh` |

---

## 1. TTFT, three arms, 14 exact prompt lengths

Served TTFT (first SSE content delta), **3 interleaved reps**, one server per arm,
same binary, same blob, same prompts. `128,512,1024,2048,4096,8192,8193` are
NULL cells — the plan is identical in all three arms — so their spread IS the
arm-to-arm bias, and every plan-changing delta below is reported after
subtracting it.

* control within-cell spread: **median 0.30%**, max 6.24% (the 128-token cell,
  the smallest and noisiest; every cell ≥512 is under 1.2%).
* arm bias on the null cells: reprice **+0.73%** median, ragged **+0.55%** median.

| tokens | ctrl plan | ctrl | reprice | Δ | ragged | Δ |
|--:|---|--:|--:|--:|--:|--:|
| 128 | `[128]` | 245.9 | 250.4 | *null* | 249.8 | *null* |
| 512 | `[512]` | 300.9 | 304.1 | *null* | 304.2 | *null* |
| 1024 | `[1024]` | 367.7 | 370.4 | *null* | 371.3 | *null* |
| **1025** | `[1024,128]` | 571.6 | **482.4** | **−16.3%** | **355.8** | **−38.3%** |
| 2048 | `[2048]` | 477.6 | 481.0 | *null* | 480.2 | *null* |
| **3073** | `[2048,1024,128]` | 1095.9 | **766.1** | **−30.8%** | **619.0** | **−44.1%** |
| 4096 | `[4096]` | 764.4 | 765.2 | *null* | 767.3 | *null* |
| **4097** | `[4096,128]` | 996.8 | 1001.5 | −0.3% *(same plan)* | **761.4** | **−24.2%** |
| **6145** | `[4096,2048,128]` | 1510.8 | **1447.5** | **−4.9%** | **1085.3** | **−28.7%** |
| 8192 | `[8192]` | 1445.7 | 1447.9 | *null* | 1448.6 | *null* |
| 8193 | `[8192,128]` | 1711.5 | 1717.3 | *null (same plan)* | 1684.0 | −2.2% |
| **10369** | `[8192,2048,512]` | 2511.0 | **2418.1** | **−4.4%** | **2014.9** | **−20.3%** |
| **12345** | `[8192,4096,128]` | 2714.0 | 2723.0 | −0.4% *(same plan)* | **2418.4** | **−11.4%** |
| **71808** | `[8192×8,4096,2048,128]` | 28509.1 | 28471.2 | −0.9% | **27589.5** | **−3.8%** |

### 1.1 The reprice is a real win, and the record said otherwise because of which lengths it sampled

`glm52-ragged-tail-chunk.md` §6 concluded *"repricing captures 44% of the win at
1025/1152 and **0% of it at 4097, 8193 and 12345**"*. That is true at those three
lengths and it is not the population. **4097, 8193 and 12345 are precisely the
lengths at which the reprice does not change the plan** — the earlier battery
sampled only the `+1`-past-a-rung family. Sampling lengths where the plan DOES
change gives −16.3% at 1025, **−30.8% at 3073**, −4.9% at 6145, −4.4% at 10369.

`LR=1780` never regressed, at any of the 14 lengths, including 71808.

**3073 is the worst cell in the shipped ladder and was never measured before: a
3073-token prompt takes 1095.9 ms while a 4096-token prompt takes 764.4 ms.**
1023 more tokens finish 331 ms sooner — a 43% penalty for being *shorter*.

### 1.2 What a launch and a padded row actually cost

Each ctrl→reprice pair trades launches for padded rows, which prices both:

| tokens | launches saved | padded rows added | bias-corrected Δ | implied padded-row price at L = 231 ms |
|--:|--:|--:|--:|--:|
| 1025 | 1 | 896 | −93.3 | 0.154 ms/row |
| 3073 | 2 | 896 | −337.8 | 0.139 |
| 6145 | 2 | 1920 | −74.3 | 0.202 |
| 10369 | 1 | 1536 | −111.2 | 0.078 |
| 71808 | 2 | 1920 | −244.7 | 0.113 |

**≈0.14 ms per padded row, with no trend in context** — 0.113 at `c0 ≈ 70k` sits
below 0.202 at `c0 ≈ 6k`. So the DP's linear-in-padded-rows cost is the right
SHAPE, and the implied launch price in row units is **L/p ≈ 231/0.14 ≈ 1650
rows** — i.e. **the shipped `LAUNCH_ROWS = 416` really is about 4× low**, as the
original reading of §3 said.

> **A model that said otherwise was built here and is recorded as refuted.**
> Fitting `chunk(r, c0) = FIX + r·(A + C·(c0 + r/2))` to the four measured
> single-chunk rung TTFTs reproduces them to 2%, reproduces four held-out
> multi-chunk cells to 5% — and then predicts **+105 ms at 10369 and +1163 ms at
> 71808** for the reprice. Both are wrong **in sign**; the measured values are
> −111 and −245. The within-chunk superlinearity in `T` that the rung ladder
> really does show **does not transfer to padded rows**. A cost model fitted to
> absolute TTFTs across very different plans cannot be trusted for the small
> DIFFERENCES a planner decides on. Commit `5d91c13` reverts what that model had
> been used to assert.

---

## 2. The acceptance class, measured

126 long free-form answers (6 questions × 7 lengths × 3 arms, ~200 completion
tokens and ~1150 characters each) plus 60 verifiable-fact answers (5 questions ×
4 lengths × 3 arms), greedy, exact prompt token counts.

### 2.1 Identical plan ⇒ byte-identical text. 62 of 62, zero exceptions

| pair | identical / total |
|---|--:|
| ctrl ~ reprice | **24/42** — and all 24 are the plan-identical cells, all 18 differences are plan-changing cells |
| ctrl ~ ragged | 12/42 — the 12 are 1024 and 4096, the only cells where ragged's plan matches |
| reprice ~ ragged | 30/42 |

The correspondence is exact in both directions at the cell level: **no
plan-identical cell ever differed, and — with the one structural exception in
§2.2 — no plan-changing cell was ever identical.** The same holds on the facts
battery (20/20 plan-identical cells byte-identical).

### 2.2 The determinant is narrower than "the plan": it is the LAST chunk's executed row count

Two arms with **different plans** produced **byte-identical** text whenever both
ran a wide last chunk:

| prompt | 8k-ladder plan (ragged) | +16384-rung plan | text |
|--:|---|---|---|
| 12345 | `[8192, 8192]` — last chunk 4153 real rows | `[16384]` — 12345 real rows | **IDENTICAL, 3/3 questions** |
| 8193 | `[8192, 128]` — last chunk **1** real row | `[16384]` — 8193 real rows | differs (char 96 / 145 / 101) |

Every divergence in the whole battery has one arm running a **128-row tail** and
the other running a wide chunk. Nothing else moved the text — not the number of
launches, not the rung widths, not padding *per se* (`[2048]` padded to 1023 dead
rows and `[2048]` run at 1025 real rows are byte-identical at 1025, 3073 and
6145: the `reprice ~ ragged` column).

**So the engine has two numeric regimes, not many: a wide chunk, and a narrow
tail chunk.** Which one a prompt gets is decided today by whether its length
happens to land on a rung. `PLOW_RAGGED_CHUNK` and the reprice both move prompts
OUT of the narrow-tail regime and into the wide-chunk regime **that every
exactly-on-rung prompt already uses**. They do not add a regime; they remove one.

### 2.3 How far into an answer, and whether it matters

Divergence position, over every differing cell: **median character 98–119 of a
~1150-character answer — about 11% in.** This is *not* a late cosmetic wobble;
the answers separate inside the first sentence or two and are then wholly
different fluent texts.

That is exactly why the identity instrument had to be long free-form and why the
verifiable-needle battery is the one that answers the question that matters:

| arm | verifiable answer correct |
|---|--:|
| ctrl | **20/20** |
| reprice | **20/20** |
| ragged | **20/20** |

**60/60**, including cells whose text diverges at character 39. The plan lottery
moves wording. On this battery it never moved an answer.

---

## 3. Task 2 — `MAX_CHUNK = 16384`: it builds, it runs, and memory is not the objection

### 3.1 Build

Emitting the exact `final2` recipe with
`PLOW_MLA_PREFILL="full:128,512,1024,2048,4096,8192,16384"` gives a seventh
prefill program of the same 2021 instructions. Three things that could have
blocked it do not:

* **Objects.** Manifest pairing hash **unchanged** (`0xe4fab0567679889a`), arm
  union identical ⇒ `hsaco_glm18` pairs with it. No rebuild.
* **Tuning.** `tuning/amd/gfx942/mi300x/kernel_measurement.jsonl` holds
  `M ∈ {1, 128, 512, 1024}` and nothing else — **8192 already falls back to the
  analytical tile model**, so 16384 adds no new gap. This retires the standing
  "the ladder and the tile campaign are coupled" warning for this change.
* **The KV ring does not bind.** `RING ≥ window + MAX_CHUNK − 1` is a
  *sliding-window* invariant; GLM-5.2's MLA is full-causal (`window = 0`), so
  `kv_ring` returns `(ctx, MASK_NONE)` and the chunk does not size the cache at
  all. The `dev_isa.h` static assert is Gemma-4's 1024-window bill, not GLM's.

`MAX_CHUNK` in `exec/amd.rs` is a plain `const` used for exactly one thing —
filtering the ladder in `plan_chunks_cfg`. It was raised in a throwaway build for
this measurement. **Landing it should take the cap from the packet's own
`shapes.max_chunk`, not add a second hardcoded copy.**

### 3.2 It runs

Coherence gate PASS, `prefill chunk policy … max_chunk=16384 buckets=[128, 512,
1024, 2048, 4096, 8192, 16384]`, and answers are ordinary.

### 3.3 Memory, measured on the card

| | 8k ladder | +16384 rung | delta |
|---|--:|--:|--:|
| `act.part` (tensor table) | 1.500 GiB | 3.000 GiB | +1.500 |
| `act.opart` | 0.250 | 0.500 | +0.250 |
| nine `[T, hidden]` bf16 buffers | 96 MiB ea | 192 MiB ea | +0.844 |
| rest, row-dimensioned | | | +0.315 |
| RUNTIME tensor total | 9.105 GiB | 12.015 GiB | +2.909 |
| **VRAM per rank, loaded and idle (`rocm-smi`)** | **113.30 GB = 105.52 GiB** | **116.56 GB = 108.56 GiB** | **+3.04 GiB** |
| blob | 275.8 MB | 405.0 MB | +129 MB |
| TP peer `slot_bytes` | 96 MiB | 192 MiB | ×2 |

**The card is 192 GiB and the shipped configuration uses 105.5 GiB of it. The
raise takes that to 108.6 and leaves 83.4 GiB free.** The doc's "act.part doubles
to 3.2 GB, which must be weighed against the launch saved" is right about the
tensor and wrong about the weighing: **+3.04 GiB against 83 GiB of headroom is
not a trade, and memory should stop being the reason this is not done.**

### 3.4 What it buys

Both arms RAGGED, same binary, blobs differing only by the rung:

| tokens | 8k-ladder plan | +16384 plan | ctl16 | rung16 | delta |
|--:|---|---|--:|--:|--:|
| 8192 | `[8192]` | `[8192]` | 1431.3 | 1430.3 | −0.1% *(null)* |
| **8193** | `[8192,128]` | `[16384]` | 1669.0 | **1421.5** | **−14.8%** |
| **12345** | `[8192,8192]` | `[16384]` | 2394.1 | **2240.8** | **−6.4%** |
| **16386** | `[8192,8192,128]` | `[16384,128]` | 3588.5 | **3303.9** | **−7.9%** |
| **24576** | `[8192,8192,8192]` | `[16384,8192]` | 5573.5 | **5286.2** | **−5.2%** |

**This is the 8k/16k residue that §2.1 of the architecture doc called
unaddressable.** It is worth ~247 ms per launch removed, which matches the
measured launch price.

**It is coupled to ragged-M.** Under the padded DP the rung is correctly unused
below 16384 — covering 8193 with one 16384-row chunk really does cost 8191 rows
of dead compute, which is worse than a second launch — so **without
`PLOW_RAGGED_CHUNK` the rung buys nothing in the 8k–16k band, the band it is
wanted for.** The two axes are one decision.

---

## 4. Recommendation

**1 — Do NOT land the `LAUNCH_ROWS` reprice as a separate change. Do not land
`LAUNCH_ROWS = 416` as "correct" either.** The constant IS about 4× low (§1.2),
but repricing it is strictly dominated by ragged-M on both axes that matter: it
is slower than ragged at **every** length where either changes the plan (−16.3
vs −38.3 at 1025, −30.8 vs −44.1 at 3073, −4.4 vs −20.3 at 10369, 0 vs −24.2 at
4097), and its output-visible blast radius is a **strict subset** of ragged's
(37.5% of lengths against 57.8%, zero exceptions, and on 83.3% of its own
changed lengths it picks *ragged's plan*). There is no state of the world in
which you want the reprice and not ragged. **Under `PLOW_RAGGED_CHUNK` the
constant is never read** (`plan_chunks_cfg` returns early), so landing ragged
retires the question rather than answering it.

**2 — Land `PLOW_RAGGED_CHUNK` default-ON.** The acceptance class is now
characterised rather than feared:

* it is **not** unbounded text drift — identical plan gives byte-identical text,
  62/62;
* the change **removes** a numeric regime rather than adding one — every
  divergence in the battery is a 128-row tail chunk against a wide chunk, and
  ragged moves prompts into the wide regime that on-rung prompts already use;
* it is **quality-neutral on the evidence available**: 60/60 verifiable answers
  correct in every arm, greedy ids identical at 1025/4097/8193 and over 64 steps
  at 4097 (prior report §5.1/§5.4);
* the TTFT is worth −38.3% at 1025, −44.1% at 3073, −24.2% at 4097, −28.7% at
  6145, −20.3% at 10369, −11.4% at 12345, −3.8% at 71808, and a **null at every
  exactly-on-rung length**.

What a reviewer should know they are accepting: **57.8% of prompt lengths will
produce different long-form wording than they do today, diverging ~11% into the
answer.** That is large, and it is the honest number. The counter-argument that
makes it acceptable is that the engine ALREADY assigns wording by prompt length
— 189 distinct plans over 1..73728 today — so this permutes an existing lottery
instead of creating one, and it permutes it toward the regime that is both faster
and already the majority case.

**3 — Raise `MAX_CHUNK` to 16384, but only together with (2), and source the cap
from the packet.** It builds, it runs on the existing objects, it needs no
re-tune, it costs 3.04 GiB/rank against 83 GiB of headroom, and it is worth
−14.8% at 8193 and −7.9% at 16386 — the residue nothing else reaches. Its
output-visible radius is larger again (95.3% of lengths in combination), and by
§2.2 much of that is text-neutral (`[8192,8192]` ≡ `[16384]` at 12345, 3/3), but
that has been checked at one length and should be checked at more before it
lands.

**4 — If (2) and (3) are refused, refuse them for the right reason.** The reason
is not TTFT, not memory, and not correctness; it is that plow does not currently
have a quality gate that could catch a plan change which degrades answers rather
than rewording them. `needle-in-a-long-answer` (this report's `facts` mode) is a
first one and it costs three minutes per arm. **Build that gate and the decision
is unblocked; leave it unbuilt and every future planner change re-runs this same
argument.**

---

## 5. Corrections this forces to the record

1. `glm52-ragged-tail-chunk.md` §6 — *"repricing captures 0% at 4097, 8193 and
   12345"* is a **sampling artifact**. Those three are the lengths where the
   reprice does not change the plan. It captures 16–31% at 1025/3073 and 4–5% at
   6145/10369.
2. `docs/arch/13` §2.1 — the `MAX_CHUNK` raise is no longer uncosted: **+3.04
   GiB/rank measured, 83 GiB of headroom, −14.8%/−7.9% at 8k/16k**, and it is
   worthless without ragged-M.
3. `glm52-ragged-tail-chunk.md` §5.4 — *"character identity across a plan change
   is not achievable on this engine"* is too strong. It is achievable and was
   observed: `[8192,8192]` and `[16384]` on the same 12345-token prompt are
   byte-identical on 3/3 long answers. The determinant is the **last chunk's
   executed row count**, not the plan.
4. The **0.31 ms/row marginal at ctx 8k** in that report's §2.1 does not describe
   padded rows: measured over five plan pairs, a padded row costs 0.078–0.202
   ms with no context trend (0.113 at `c0 ≈ 70k`).

---

## 6. Arm attribution, and reproducing this

Every arm logged its own policy at its first prefill — added here because an A/B
whose arms differ only by an environment variable had no positive signal that the
variable had reached the process:

```
ctrl     prefill chunk policy launch_rows=416  overridden=false ragged=false max_chunk=8192
reprice  prefill chunk policy launch_rows=1780 overridden=true  ragged=false max_chunk=8192
ragged   prefill chunk policy launch_rows=416  overridden=false ragged=true  max_chunk=8192
ctl16    prefill chunk policy launch_rows=416  overridden=false ragged=true  max_chunk=16384  buckets=[…8192]
rung16   prefill chunk policy launch_rows=416  overridden=false ragged=true  max_chunk=16384  buckets=[…8192, 16384]
```

```
# blobs (CPU only; cp-ctl16 comes out cmp-identical to glm52-tp8-final2)
env GLM_FULL=1 GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 \
    PLOW_GLM_DSA=0 PLOW_GLM_FUSE_B1=1 PLOW_GLM_GEMV_WG=152 PLOW_MLA_PF_V2=1 \
    PLOW_GLM_PF_NS=2 PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1 \
    PLOW_MLA_PREFILL="full:128,512,1024,2048,4096,8192,16384" \
  plowc --emit devblob --hf-dir /workspace/models/GLM-5.2-FP8 --gpu MI300X \
    --arch gfx942 --num-gpus 8 --max-ctx 73728 --out <dir>/model.pkt

# the battery (needs the box lock; both scripts release it in a trap that exits)
ASSETS=/workspace/assets/gfx942/glm52-tp8-final2 PORT=8195 \
  ARMS="ctrl reprice ragged" MODES="ttft ident facts" \
  bash perf-data/probes/chunk_policy_run.sh
PLOWRT_BIN=<a plowrt built with MAX_CHUNK >= 16384> \
  bash perf-data/probes/chunk_maxchunk_run.sh
# NOTE: chunk-policy-raw/ is NOT committed (9.3k lines of machine-generated per-cell JSON).
# Recreate it first with perf-data/probes/chunk_policy_run.sh, whose OUT= default is that dir.
python3 perf-data/probes/chunk_policy_analyze.py --dir perf-data/plow-gfx942/chunk-policy-raw
```

Two traps this run hit, recorded so the next one does not:

1. **The nix dev shell does not carry `/opt/rocm-*/lib`.** Without it the HSA
   probe fails, plowrt selects the CPU reference backend, and it *serves
   perfectly* — a whole battery of meaningless numbers, caught only by the
   coherence gate. `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-7.2.4/lib` inside
   the shell, not outside it.
2. **A cost model is not an instrument.** The one in §1.2's note fit every
   absolute TTFT it was shown, to 2–5%, and got the sign of the decision wrong at
   two of five plan pairs. Only the direct A/B settled it.
