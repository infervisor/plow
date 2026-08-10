# GLM-5.2-FP8 TP8 on gfx942 — the FACTS GATE: a quality instrument for chunk-plan changes

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **METHOD** — an answer-QUALITY instrument. Arch-independent by design.

**One sentence: plow now has a standing gate that answers the question character
identity cannot — *does a chunk-plan change DEGRADE answers or merely reword
them* — built as 12 machine-checkable items × 7 OFF-RUNG lengths compared
PAIRWISE between arms, with a baseline that scores 77/84 rather than 84/84 (so it
has room to move in both directions) and a demonstrated non-zero exit on a
deliberately degraded arm.**

| | |
|---|---|
| harness | `perf-data/probes/facts_gate.py` (`run` / `verdict` / `selftest`), `perf-data/probes/facts_gate_run.sh` |
| raw | NOT COMMITTED — 9.3k lines of machine-generated per-cell JSON. Regenerate with `perf-data/probes/facts_gate_run.sh`, which writes `facts-gate-raw/facts_{ctrl,ragged,ctrl-injected}.json`. |
| blob | `/workspace/assets/gfx942/glm52-tp8-final2/model.pkt` — UNCHANGED, no re-emit |
| objects | `/root/.claude/jobs/b09a4bcc/tmp/hsaco_glm18` — UNCHANGED |
| serve env | `PLOW_MLA_PF_V2=1`; the ragged arm adds `PLOW_RAGGED_CHUNK=1` |
| backend | `HSA backend selected device=gfx942`, `backend ready — GPU accelerated`, 8 ranks bound, VRAM 54% |
| cost | **12.2 min per arm** (84 cells, greedy, ~131 completion tokens each) |

---

## 1. Why the existing instruments could not answer this

Everything plow had was **character identity**. It fires on harmless
reassociation — 57.8% of prompt lengths reword under `PLOW_RAGGED_CHUNK` — and
says nothing at all about whether the reworded answer is still *right*. The one
quality probe that existed, `chunk_policy_battery.py --mode facts`, was 5
questions × 4 lengths and returned **60/60 correct in all three arms**: a battery
that cannot go red is not evidence of safety, it is an absence of evidence.

Three constraints follow from the measurements in `glm52-chunk-policy.md`, and
each one is enforced by the gate rather than left to the operator:

| constraint | why | enforced by |
|---|---|---|
| the checkable token must sit **LATE** | two plans diverge at median char ~100 of ~1150 (~11% in); a needle in the first sentence is on the *identical* side of the divergence | every item reasons first and answers on the last line; `verdict` ERRORS if the median answer position is under 25% |
| the battery must be **able to fail** | 60/60 across three arms proves nothing | 84 cells, adversarial classes, baseline 91.7% not 100% |
| it must run **OFF-RUNG** | on-rung lengths are byte-identical across arms by construction | `verdict` ERRORS on any cell whose ragged and padded covers agree *and* whose last chunk executes at bucket width |

---

## 2. What the battery is

**12 items × 7 off-rung lengths (1025, 3073, 4097, 6145, 8193, 10369, 12345) = 84
cells per arm**, greedy, at exact prompt token counts verified per cell against
the server's own `usage.prompt_tokens`.

| class | items | what a plan change would have to do to fail it |
|---|---|---|
| **multi-hop arithmetic** | `hop_train` `hop_tank` `hop_money` `hop_mod` `hop_avg` | perturb any link in a 3–4 step chain; the final token is the only thing graded |
| **retrieval at depth** | `ndl_early` (0.25) `ndl_mid` (0.55) `ndl_late` (0.94) `ndl_key` (3 records, wanted one at 0.50) `ndl_sum` (0.30 + 0.90, answer is their sum) | lose or misplace a token that sits **inside the prefill**, at a controlled depth — the region a chunk plan re-partitions |
| **structured output** | `str_num` (4 fields) `str_sq` (6 fields) | drop a field, or corrupt one of several; every key is checked for presence *and* value |

The retrieval items are the ones aimed squarely at chunking: the needle is an
opaque token (`QF7392`, `MX4185`, `ZD6031`) or a bare serial that cannot be
guessed, placed at a **measured token depth** in the haystack. `ndl_key` asks for
the *middle* record of three, so an arm that has only kept the most recent one
answers confidently and wrong. `ndl_sum` needs both ends of the context, so
losing either is fatal.

Ground truth for every arithmetic chain is recomputed independently by
`facts_gate.py selftest`, so a mistyped expectation fails the harness rather than
an arm.

### 2.1 The verdict is PAIRED, and a powerless gate is an ERROR

`verdict` compares candidate against baseline cell by cell, not against an
absolute score — "is the candidate worse", not "is the candidate good". A cell
both arms get wrong is a model limit, not a regression.

* **paired McNemar**, one-sided exact binomial on the discordant pairs;
* **a hard net-regression cap** (default 2), because McNemar alone needs 5
  one-sided discordants to reach p ≤ 0.05, which is lax for a *deterministic*
  engine where a differing cell is reproducible and not sampling noise. The cap
  is on the **net** so that reassociation flipping cells symmetrically both ways
  — which is rewording, not degradation — does not fire it;
* **a per-class trip**: a class the baseline can do (≥80%) and the candidate
  cannot (<50%) fails on its own, so a targeted failure cannot be diluted away by
  a large battery.

Exit **2 — GATE INVALID** (never PASS) if the baseline scores under 85%, if
format compliance drops under 90%, if the median answer position is not late, or
if any cell is at a length that cannot carry signal. **A gate that has silently
lost its power must not report green.** That is exactly how `PLOW_FUSE_QUANT`
shipped broken past a green gate for weeks, and it is the reason this branch
exists.

---

## 3. The gate was PROVEN TO FAIL before it was trusted to pass

A gate never demonstrated failing is exactly how `PLOW_FUSE_QUANT` shipped broken
past a green gate for weeks. So the gate was run against a deliberately degraded
arm first.

`run --inject drop-tail:800` replaces the 800 tokens immediately before the
question with filler. **The token count is preserved, so the chunk PLAN is
identical** — the only thing lost is the *content* of a region, which is exactly
what a chunk whose rows never executed costs the model. The question itself is
untouched.

```
baseline  ctrl           77/84 = 91.7%
candidate ctrl-injected  63/84 = 75.0%   [inject drop-tail:800]
paired: 14 regressions, 0 repairs, McNemar one-sided p = 0.0001

per class            base    cand   reg  rep
  hop                30/35  30/35     0    0
  needle             34/35  20/35    14    0
  struct             13/14  13/14     0    0

FAIL:
  x paired regression: 14 vs 0, p=0.0001 <= 0.05
  x net regressions 14 > 2
```

**Exit code 1.** Two independent rules fired, and the attribution is exact: all
14 regressions are in the retrieval class, and the arithmetic and structured
classes do not move by a single cell.

The regressed answers say what a lost context should make them say — `ndl_late`
returns *"Not provided in the record"*, *"None"*, *"CHARLIE"*; `ndl_sum` returns
*"Cannot be determined"* — rather than confabulating, which is why grading the
final line strictly rather than substring-matching the body matters.

**And the injection is surgical, which also re-proves the engine is
deterministic:**

| class | cells byte-identical to the clean arm |
|---|--:|
| hop | **35 / 35** |
| struct | **14 / 14** |
| needle | 19 / 35 |

An injection over pure filler is a genuine no-op — `hop_train@8193` returns the
same *wrong* answer (`13:15`) in both arms — so the 16 needle cells that moved
are the fault and nothing else is.

---

## 4. The flip gate: `PLOW_RAGGED_CHUNK` ctrl vs ragged — **PASS**

Same blob, same objects, same binary, one server per arm; the ragged arm logged
`PLOW_RAGGED_CHUNK: fewest-launch cover, last chunk runs at its real row count
buckets=[128, 512, 1024, 2048, 4096, 8192]` at its first prefill, so the cell is
attributed to a plan that was **logged, not assumed**.

```
baseline  ctrl       77/84 = 91.7%
candidate ragged     79/84 = 94.0%
paired: 2 regressions, 4 repairs, McNemar one-sided p = 0.8906

per class            base    cand   reg  rep
  hop                30/35  32/35     0    2
  needle             34/35  34/35     1    1
  struct             13/14  13/14     1    1

validity: baseline format compliance 100.0%, median answer position 98% into
          the answer, 84 cells at lens [1025, 3073, 4097, 6145, 8193, 10369, 12345]

PASS: no significant degradation
```

**Exit code 0.** Ragged is *nominally better* (79 vs 77), the discordant pairs
run 2:4 the wrong way for a degradation claim, and the net regression count is
**−2** against a cap of +2.

### 4.1 The two "regressions" are the mirror image of two of the four "repairs"

This is the most informative row in the whole battery, and it is what
distinguishes rewording from damage:

| item | fails in ctrl at | fails in ragged at | the wrong answer, both times |
|---|--:|--:|---|
| `ndl_key` | 10369 | 12345 | `3160` — FOXTROT's serial (depth 0.80) instead of ECHO's (depth 0.50) |
| `str_num` | 12345 | 10369 | `DIGITSUM=23` instead of 21 |

**The same two model weaknesses, at the same two lengths, swapped between the
arms.** The plan lottery moved *which length* lands on a pre-existing failure; it
did not create a failure. That is the acceptance class `glm52-chunk-policy.md`
argued for on the identity evidence, now confirmed on the quality axis by an
instrument that can go red.

Only **1 of 84** cells is byte-identical between the arms, confirming these are
genuinely plan-changing lengths and not a diluted battery: the text moved almost
everywhere and the *answers* did not.

### 4.2 What the battery costs, and why the baseline is not 84/84

**12.2 min per arm** (734 s ctrl, 655 s ragged), 84 cells, greedy, mean 131
completion tokens, `finish_reason=stop` on 84/84 — no truncation anywhere, so no
cell fails for want of room to answer.

The baseline scores **77/84, not 84/84**, and that is a feature. The 7 baseline
failures are all at 8193/10369/12345, the longest contexts:

```
hop_train@{8193,10369,12345}   13:15 / 09:55 / 13:10  (want 13:50)
hop_tank@{10369,12345}         2960 / 748             (want 1428)
ndl_key@10369                  3160                   (want 8074)
str_num@12345                  DIGITSUM=23            (want 21)
```

A battery pinned at the ceiling can only ever move one way and cannot show a
*repair*; this one has headroom in both directions, which is what let §4.1 be
observed at all.

---

## 5. Limits of this instrument, stated

1. **One model.** GLM-5.2-FP8 TP8 on gfx942 only. The items are model-agnostic
   but the 85% baseline floor is calibrated here, and a weaker model would trip
   `GATE INVALID` rather than silently reporting a meaningless PASS — which is
   the intended behaviour, but it does mean the battery is not portable as-is.
2. **The needle is near the last chunk, not straddling it.** `ndl_late` sits at
   0.94 of the haystack, which for a 4097-token prompt is ~250 tokens before the
   `[4096,128]` boundary rather than across it. Placing a needle to straddle each
   arm's own boundary would need the plan computed per arm per length; the covers
   are already mirrored in `facts_gate.py` (`cover_ragged` / `cover_padded`,
   cross-validated against the pinned Rust expectations), so this is a small
   extension and the obvious next sharpening.
3. **Power.** With 84 cells and a ~92% baseline, McNemar needs 5 one-sided
   discordants for p ≤ 0.05; the net cap of 2 is what actually binds for small
   regressions. A 3-cell regression fails on the cap and not on the p-value, and
   that is deliberate — but a 1-cell regression passes, and would not be caught.
4. **`drop-tail` is a content fault, not a kernel fault.** It proves the gate
   detects lost prefill *information*. It does not prove the gate detects a
   numerically subtle kernel error that shifts logits without erasing anything —
   that failure class is still only covered by the identity instrument.

---

## 6. The flip, and the headline ladder restated

The gate passed, so both defaults landed together.

### 6.1 What actually changed in the code

**`PLOW_RAGGED_CHUNK` now defaults ON** (`crates/plowrt/src/config.rs`).
`PLOW_RAGGED_CHUNK=0` restores the padding DP byte-identically and is now
mandatory on the control arm of any A/B here.

**The runtime `MAX_CHUNK` constant is gone**, and that *is* the "raise it to
16384". It was `8192`, and it filtered the compiled ladder — a second copy of a
number the packet already carries (`shapes.max_chunk` in the manifest is defined
as `max(prefill_buckets)`, and the same emit sizes the KV ring from it). A blob
built with a 16384 rung therefore **served as if the rung were absent**. The cap
is now the widest bucket the packet carries, and the policy line logs that
derived value — confirmed live:

```
ctl8   … launch_rows=416 overridden=false ragged=false max_chunk=8192  buckets=[…8192]
rag8   … launch_rows=416 overridden=false ragged=true  max_chunk=8192  buckets=[…8192]
rag16  … launch_rows=416 overridden=false ragged=true  max_chunk=16384 buckets=[…8192, 16384]
```

`ragged=true overridden=false` on `rag8`/`rag16` is the positive signal that the
**default** is what moved, not an environment variable.

`LAUNCH_ROWS` is untouched at 416, as instructed — under ragged
`plan_chunks_cfg` returns before it is read.

**Raising the cap for a model is now an EMIT decision.** The default MLA ladder
(`devgen::mla::glm_prefill_buckets`) still tops out at 8192 and was deliberately
left there: it is shared by GLM, Kimi K3 and DeepSeek, and the +3.04 GiB/rank is
measured on exactly one of them. The GLM-5.2 recipe gains
`PLOW_MLA_PREFILL=full:128,512,1024,2048,4096,8192,16384`.

### 6.2 The published TTFT table, restated

The four lengths plow's own harness prompts land on (`docs/arch/13` §7), 3
interleaved reps, same binary, one server per arm. **Control within-cell spread
0.35–1.21%**, so every delta below is far outside it.

| tokens | ctl8 (pre-flip) | rag8 (ragged, 8k ladder) | Δ | **rag16 (shipped: ragged + 16384)** | **Δ** |
|--:|--:|--:|--:|--:|--:|
| 1023 | 370.9 | 369.8 | −0.3% | **369.9** | **−0.3%** *(null)* |
| 4101 | 993.2 | 752.4 | −24.2% | **752.1** | **−24.3%** |
| 8196 | 1690.5 | 1661.1 | −1.7% | **1418.3** | **−16.1%** |
| 16386 | 3605.4 | 3571.9 | −0.9% | **3296.3** | **−8.6%** |

**These are the shipped numbers: 370 / 752 / 1418 / 3296 ms at 1k / 4k / 8k /
16k**, from 371 / 993 / 1691 / 3605.

### 6.3 The coupling, measured on the published lengths for the first time

The `rag8` column is the argument for landing the two together, and it is
starker here than the report that proposed it could show:

* **ragged alone buys −1.7% at 8196 and −0.9% at 16386.** At those lengths it
  does not change the plan at all — `[8192,128]` and `[8192,8192,128]` in both
  arms — and only shrinks the tail chunk's executed rows from 128 to 4 and to 2.
* **the 16384 rung turns those into −16.1% and −8.6%**, an extra 14.4 and 7.7
  points, because the plan finally collapses to `[16384]` and `[16384,128]`.
* and the rung is worth **nothing at all** without ragged: under the padded DP it
  is correctly declined below 16384 (8191 rows of dead compute cost more than a
  second launch), which the new unit test `the_ladder_is_its_own_cap` now pins.

**The 4k win is ragged's alone; the 8k/16k win is the rung's alone; and neither
is reachable by the other axis.** That is the whole case for one decision.

### 6.4 The gate was re-run on the SHIPPED configuration — **PASS**

`glm52-chunk-policy.md` §4.3 warned that the rung's output-visible radius is
larger again (95.3% of lengths in combination) and *"has been checked at one
length and should be checked at more before it lands."* It has now been checked
at all seven, and it is the largest plan change in the whole campaign: under the
16384 ladder **8193, 10369 and 12345 all collapse to a single `[16384]` chunk**
from three-chunk padded covers.

```
baseline  ctrl       77/84 = 91.7%
candidate rag16      78/84 = 92.9%
paired: 3 regressions, 4 repairs, McNemar one-sided p = 0.7734

per class            base    cand   reg  rep
  hop                30/35  32/35     0    2
  needle             34/35  34/35     1    1
  struct             13/14  12/14     2    1

PASS: no significant degradation
```

Only 1 of 84 cells is byte-identical to ctrl, and the discordants are again the
same handful of items sliding between lengths — `ndl_key` (`3160`, FOXTROT's
serial) fails at 10369 in ctrl and 12345 in rag16; `str_num`'s digit errors
(`DIGITSUM=23`, `REVERSED=4841`) move from 12345 to 8193/10369.

| arm | correct | reg | rep | p | verdict |
|---|--:|--:|--:|--:|---|
| ctrl (`PLOW_RAGGED_CHUNK=0`, 8k) | 77/84 | — | — | — | baseline |
| ragged, 8k ladder | 79/84 | 2 | 4 | 0.89 | **PASS** |
| **ragged + 16384 (shipped)** | **78/84** | 3 | 4 | 0.77 | **PASS** |
| ctrl + injected fault | 63/84 | 14 | 0 | 0.0001 | **FAIL (exit 1)** |

---

## 7. Cross-model exposure — the part of this flip with the least evidence

`plan_chunks` and `rebase_chunk_rows` are in the **shared AMD engine**
(`exec/amd.rs`), so a default flip turns ragged-M on for *every* AMD model, while
the entire evidence base — this report, `glm52-chunk-policy.md`,
`glm52-ragged-tail-chunk.md` — is GLM-5.2, a **full-causal MLA** model. A grep of
`perf-data/` and `docs/` finds **no ragged measurement on any sliding-window
model at all.**

Gemma-4-12B is the case that differs in kind: `sliding_window = 1024`, so its KV
cache is a *ring* sized `window + chunk - 1` rather than a linear cache.
`perf-data/probes/facts_gate_gemma.sh` runs it, and the answer is the best
available one:

```
baseline  gem-ctl (PLOW_RAGGED_CHUNK=0)   19/20 = 95.0%
candidate gem-rag (PLOW_RAGGED_CHUNK=1)   19/20 = 95.0%
paired: 0 regressions, 0 repairs, p = 1.0000    PASS
```

**All 20 cells are BYTE-IDENTICAL between the arms**, at four off-rung lengths
(600, 1025, 1500, 2100), with `ragged=true buckets=[128, 512, 1024]` logged. The
one failure (`hop_money@1500` → `255`) is the same wrong answer in both arms.

The reason is worth stating, because it *narrows* the acceptance class one more
notch. On Gemma's short `[128, 512, 1024]` ladder the padded DP and the ragged
cover pick the **same rungs at every one of these lengths**, so the only thing
ragged changes is the last chunk's executed row count — 600 real rows in a 1024
bucket instead of 1024 padded ones. And **that is inert**: `glm52-chunk-policy.md`
§2.2 had already observed that padding per se does not move the text, and this is
that property confirmed on a second model, a second architecture, and a second
attention regime.

So the sharper statement of the determinant is: **it is not "the last chunk's
executed row count" but "which BUCKET runs last" — narrow tail versus wide
chunk.** Shrinking a wide chunk's rows is invisible; replacing a 128-row tail
with a wide chunk is not.

Two honest limits remain on this axis:

* the ring requirement moves in the **safe direction** (ragged runs *fewer* rows
  than the bucket, never more), but that is an argument, not a measurement;
* 20 cells on one Gemma blob at four lengths is far thinner than the 84×3 on GLM,
  and no other AMD model was checked at all.

A separate, unrelated finding from setting this up: **`g12b-64k-mergefix` faults
the GPU on load** (`Memory access fault by GPU node-8`, during weight bind, on
the *control* arm with `PLOW_RAGGED_CHUNK=0`) — its `hsaco` still points at
`hsaco_static`. It is a stale asset/object pairing, it predates this branch, and
it is recorded here rather than fixed. `g12b-showdown` (on `hsaco_glm18`) is
healthy and is what the numbers above come from; it needs
`PLOW_FP8_DIR=/workspace/models/gemma-4-12B-it-fp8` to load at all.
