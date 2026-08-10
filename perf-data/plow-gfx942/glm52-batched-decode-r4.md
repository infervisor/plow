# GLM-5.2 batched decode (r4): the ladder GLM refused to carry — built, gated, measured

> **Scope:** 8x MI300X (gfx942, ROCm 7.2.4), GLM-5.2-FP8 TP8 · **CAMPAIGN RESULT + OPEN ITEMS**
> — the emitter-side batched decode the ledger's OPEN #2 scoped, landed end to end (`a0cbd1a`
> + fixes), with the five defects the road surfaced, two adopted-arm REVERTS, and the honest
> distance to vLLM that remains. Companion: `glm52-batched-decode-scope.md` (the scoping),
> `glm52-experiments.md` (the ledger this updates).

## What landed

One decode program per `PLOW_DECODE_BATCH_LADDER` rung; at `rows > 1` the MoE seam is emitted
with the grouped PREFILL family at `T = rows` and the dense FFN takes the degenerate
`n_exp = 1` route; both KV writers take the batch-major ring form (`i[6] = n_batch_kv`,
`j[0] = ctx` — row t writes ITS slot's ring at `pos[t]`); per-sequence lm_head/argmax tail.
The runtime needed **zero changes** — mux tick-sharing (one prefill chunk + one batched decode
dispatch per tick), ragged per-row `pos`/`kvlen`, parked masks and rung selection were all
built and inert. Serve banner on the shipped asset: `batch=4 decode_rungs=[1, 2, 4]`.

**Gates held:** the UNLADDERED emit is byte-identical (cmp on model.pkt) before and after every
emitter change; the 1-layer batch-determinism probe (`scripts/glm52_batch_determinism.sh`) is
byte-identical solo-vs-paired on the final objects; needle-content PASSES at 3000 and 6000
tokens; GSM8K 8-shot n=60: **0.9667 @ conc 2, 0.9833 @ conc 4, 0.9833 @ conc 8 on the B_max=4
asset** (vs 0.95–0.97 conc-1 baseline).

## The five defects between "it emits" and "it is right", in the order found

| defect | symptom | proof |
|---|---|---|
| asset `checkpoint/` left pointing at raw HF | GPU memory fault at load (with QNORM fold) or a loud shard-size refusal (without) | RECIPE §2 note; cost half a day of object-knob bisecting |
| `skip_norm` field on the batched latent writer (`i[4]`, copied from the rope packets) | unnormalized latent cached → '!!!' garbage within 2 tokens at rung ≥2 | one-field fix; rung-2 output goes coherent |
| grouped-MoE kernel at `MPF_BK=32` | wrong numerics at every `rows>1` (needed because the OCC4 arena is 30,720 B vs the tile's 40,960) | 1-layer determinism FAILS at BK32, byte-identical PASS at BK64/DBUF=1 → decode row builds OCC4=0 + BK64 (task open to fix BK32 and win OCC4 back) |
| **`patch_kvrow` keyed on BLOB batch, not the rung** | every rung-1 step writes KV into ring row 0; needle deterministically '741' (retrieval from prefill rows only); the one recorded needle 'PASS' had run concurrently with a smoke request — i.e. at rung 2, on the correct per-row writers | fixed: at `dbatch>1` every rung takes the ring form; needle PASSES 3000/6000 |
| rung-8/16 programs (rows 5+) | GSM conc-8 on the B16 asset = 0.7833, twice, deterministically; B_max=4 under the same 8 clients = 0.9833 (churn exonerated) | OPEN (task); B_max=4 is the validated ladder |

## Two adopted arms REVERTED by the needle-content gate

`PLOW_XR_AGG` (briefly default-on): an XR_AGG-only prefill build FAILS needle@3000 ('741' for
'7413') — the "no value is touched" claim does not order OTHER workgroups' stores; one closing
workgroup signalling for all nblk is a real release-ordering race. `PLOW_MLA_FOLD_TB=8`
(briefly default-8): fails in combination, never content-gated at length alone. Both back to
opt-in with the failure recorded at the knob. **The kernel-level bit-identity check and
short-prompt GSM both missed this; needle-content at 3000+ tokens is now the adoption bar for
prefill-object arms.**

## Numbers (validated `glm52-tp8-r4b4` asset: rungs 1,2,4 · MM=4/occ2/BK64 decode objects ·
## no-arm prefill objects · analytical tiles · FUSE_ROPE only · max-ctx 18432)

GSM8K-style serving (~1.1k-token prompts, 320-token answers), 60 questions:

| clients | acc | wall | aggregate out tok/s (approx) | vs flat-27 baseline |
|--:|--:|--:|--:|--:|
| 1 (r2 ctrl) | 0.97 | 316.9 s (n=100) | ~25 | 1.0× |
| 2 | 0.9667 | 328 s (n=60) | ~46 | 1.7× |
| 4 | 0.9833 | 185 s (n=60) | ~81 | 3.0× |
| 8 (4 slots + queue) | 0.9833 | 163 s (n=60) | ~92 | **3.4×** |

Single-stream TPOT by rung object: 26.5 ms (OCC4/MM1 shipped) → 37.4 ms (occ2/MM4) → 50.4 ms
(occ2/MM16). The OCC4 loss (BK32 task) and the MM tax are the rung-1 bill.

Streaming ladder at **in=8192 / out=128** (`speed.py`, distinct prompts, no prefix cache):

| conc | out tok/s | TTFT p50 | TPOT p50 |
|--:|--:|--:|--:|
| 1 | 17.1 | 2386 ms | 40.1 ms |
| 4 | 24.1 | 3585 ms | 135.5 ms |
| 16 | 24.3 | 64.0 s | 142.4 ms |

## Verdict vs vLLM, stated honestly

The recorded same-session vLLM figures (`r2-baseline/vllm-force-mqa.json`, no prefix cache):
TTFT 1179.5 ms @8k, 2504.1 @16k, TPOT 17.9 ms ⇒ **55.8 tok/s single-request at 8k**. No vLLM
≥8k concurrency ladder exists on this box.

* **Short-prompt serving throughput: plow now clears vLLM's recorded single-request rate** —
  ~92 tok/s aggregate at 8 clients (0.9833 quality) against a serving stack that was pinned at
  22.8–27 tok/s at every concurrency two days ago.
* **8k-input throughput: NOT beaten.** At 64:1 prefill:decode the workload is prefill-bound;
  plow's 8k prefill (~2.4 s on this blob via the streaming harness) caps admission, and the
  rung-4 decode step at 135 ms shows the structural fact: at B=4, top-8-of-256 routing almost
  never overlaps experts, so the grouped seam re-reads ~B× the expert weights — the
  amortisation the grouped form buys at prefill T does not exist at small B. The distance is
  (a) the prefill gap (the ledger's MoE GLU→DOWN fusion is the priced lever) and (b) expert
  overlap, which grows with B — exactly the rung-8/16 path that is quality-broken today.
* The 16384 row of the streaming long-context ladder is INVALID (17 ms "TTFT" = a server-side
  rejection measured as a response; max-ctx 18432 leaves no room for 16384 + template + out).

## Also in this campaign

* **Tuning store repaired** (`b67115d`): inventory-derived ingest files a tile timing under
  every dispatched opcode carrying that tile; 3708/3708 GLM tile lookups now BY MEASUREMENT
  (was 0 — the gfx950-geometry RUNGS map starved opcode `Gemm` of records on gfx942).
  **Adoption for GLM blobs is HELD** pending one open question: the r3 conc-1 arm (measured
  tiles + folds + the since-reverted objects) regressed TTFT +7.8% @8k while improving
  1k/4k ~3%; the reverted arms are confounded into that number and the tile-only re-measure
  has not been run. The r4 assets emit `PLOW_TUNEDB=""` (analytical) until it is.
* r3 conc-1 arm (all folds + QNORM): TTFT −3.2%/−3.1% @1k/4k, +7.8% @8k (confounded, above),
  −0.6% @16k, TPOT +1.9%, GSM 0.95 (n=100, within noise). Fold-by-fold rebisect on clean
  objects is queued behind the tile question.
* GPU-lease discipline extended to the ad-hoc serve scripts (`glm52_serve_smoke.sh`,
  `glm52_batch_determinism.sh`) after an unleased probe and a stale hand-made lock each cost
  a blocked/blocking run.

## r5: context to 64k on the validated ladder (2026-08-09, same day)

`glm52-tp8-r5b4-64k` — B_max=4, max-ctx 65536, analytical tiles, no-arm prefill objects,
MM=4/occ2/BK64 decode objects. **Every gate passes at length**: needle-content retrieval at
~5k, ~54k and **~61k tokens** ('7413' each), GSM 0.95 @ conc 4, coherence. Apple-ladder TTFT
(client.py, the r2 methodology): **8k 1609.7 / 16k 3525.1 / 32k 8667.6 / 48k 15571.8 /
64k 22766.8 ms** (spreads ≤0.8% past 8k). No vLLM record exists past 16k on this box; at
8k/16k plow is 1.36×/1.41× the vLLM recorded TTFT. Rung-1 TPOT 37.6 ms.

Streaming ladder at in=8192/out=128, conc 1→32: 17.1 → 24.0 → 24.6 → 24.4 → **25.0 tok/s**
(flat past 4 — queue-bound on 4 slots; TTFT p50 145 s at conc 32 is queueing, not batching).
The 64:1 prefill:decode workload is prefill-bound either way; the decode ladder's win shows on
decode-heavy serving (the GSM table above), not here.

## The burst-admission finding that reframes task "rung-8"

The conc-8 quality dip is NOT the rung-8 program: with 8 IDENTICAL prompts admitted in a
burst, ~3 slots produce deterministic per-slot garbage (membership varies run to run;
survives `PLOW_GEMV_LG=0`); the SAME 8 prompts admitted 1.5 s apart are **all byte-correct
through rung 8 on the same bits**. Bursts ≤4 are clean. Gemma's ladder — the only prior
batched-decode service — was TP1; the suspect is TP8 xctr/collective state across rapid rung
switches interleaved with prefill `zero_xctr` in the burst window. Until it is fixed,
B_max=4 is validated for any client count (admissions-on-completion are staggered by
construction), and B=8/16 only under staggered arrival.

r8 session facts (2026-08-10, r4b16 asset, one serve, `$CLAUDE_JOB_DIR/tmp/burst8.sh`):

* Reproduces on the r7 stack: burst-8 identical prompts → **6/8 diverge**, slots 7+8 in
  the SAME attractor ("oftoftoft…"), slot 6 in a second; slots 2/3/5 diverge subtly and
  late. Spaced 1.6 s, same serve, immediately after: **8/8 byte-identical**.
* **The every-token host xctr audit and the collective deadline are SILENT through the
  poisoning** — arrival counts are exactly right on every rank, so this is NOT a
  timed-out or miscounted collective. Whatever corrupts, it corrupts values while the
  gate protocol runs to spec (compare Lesson 19: counts certify arrival, not content).
* First token of EVERY slot is correct, including the attractor slots — prefill logits
  are sane, decay begins with decode steps.
* Even the burst's "OK" slots differ from the spaced consensus ~68 chars in, so burst
  admission may perturb ALL slots with severity graded by slot index — the next
  instrument should diff slot 1 burst-vs-spaced byte-by-byte, not only slot-vs-slot.
* Next: PLOW_TRACE_RAW on a 2-3 layer truncated blob under burst (full-size traces are
  impractical), and a host-side dump of the mux's slot→ring-row/position tables at each
  admission in the burst window — the graded-by-slot severity + one-time poisoning
  pattern points at admission-time slot state (ring rows / positions / kidx), not the
  per-step collective.

r8 session round 2 — "burst" was never the variable. **It is ROW INDEX ≥ 5 in the
rung-8/16 programs**, full stop:

* burst-5: 0/5 diverge. burst-6: exactly slot 6 (row 5), attractor A. burst-8: row 5 →
  attractor A, rows 6-7 → attractor B, rows 0-4 always clean. Deterministic per ROW.
  The 1.5 s "spacing threshold" and every burst framing were proxies for "did occupancy
  ever exceed 5 simultaneously"; the r4-era 'clean through rung 8 when spaced' run never
  held ≥6 concurrent. B_max=4 was always clean because rung 4 never has a row ≥ 5.
* `PLOW_PF_NO_INTERLEAVE=1`: rows 0-4 become byte-identical (the interleave explains
  only the earlier subtle drift of low rows); rows 5-7 unchanged garbage.
* `PLOW_L2_PLACE=0` re-emit, same objects: identical rows-5-7 attractors → placement and
  the GQ domain windows are exonerated.
* Program diff (disasm) rung-4 vs rung-8: every field scales correctly; the ONLY
  structural difference is MlaMergeFold's map switch — rungs 1-4 dispatch the VT=32 fold
  map (b = bh·8), rungs 8/16 the VT=256 map (b = bh·1, `d_mla_merge_fold<512,256>`)
  because bh·8 > 304 CUs. Clean rungs ↔ VT=32, broken rungs ↔ VT=256, perfect
  correlation. opart/mlpart sizing audited: fits exactly at every rung; hier_base carve
  audited: disjoint by construction (n_counter = insts·25).
* ~~Next session: audit `d_mla_merge_fold<512,256>`~~ — DONE, and the VT=256 correlation
  was a red herring. Round 3 facts:
  - The fold body derives every address from `w`; audited clean. Fine-gate theory dead
    too (these programs carry ONE coarse counter per packet; the ×25 is hierarchy
    scratch, disjoint by construction).
  - **Steady-state rung 8 is CORRECT for all 8 rows**: 8 streams admitted 2 s apart with
    220-token generations hold occupancy 8 through hundreds of rung-8 decode steps —
    every row coherent. The per-step compute (flash, fold, collectives, argmax) is fine.
  - A `PLOW_DECODE_BATCH_LADDER=1,8` asset (blob batch = 8) still breaks exactly ~3
    slots in a burst, but the MEMBERSHIP moves (5/6/8 instead of 6/7/8) — r4's
    "membership varies run to run" is the racy arrival→slot assignment, and blob-batch
    geometry (16 vs 8) is refuted as the mechanism.
  - Synthesis over every run: a slot is poisoned iff its PREFILL lands in the
    rapid-succession window at already-high occupancy; its first token (computed by the
    prefill itself) is always right and everything after reads corrupt history — the
    damage is PERSISTENT PER-SLOT KV RING STATE written at admission, not per-step
    compute. Burst-5 clean / burst-6 breaks the 6th / sustained-8 clean all fit.
  - ~~Next: host-side instrument~~ DONE (round 4). `dispatch_all`/`prefill_chunked`
    debug logs landed (serve/engine.rs, RUST_LOG=plowrt::serve=debug); the instrumented
    burst trace is unambiguous — **HOST STAGING IS CORRECT AND THE POISON IS IN THE
    FIRST DECODE STEP OF A JUST-ADMITTED SLOT NEAR THE RUNG-8 PROGRAM'S FIRST
    DISPATCHES**:
    - Every prefill agrees (token 1806 on all 8 slots). Every staged tick shows exactly
      right `pos/kvlen/parked/ids` for every row (trace preserved at
      `$CLAUDE_JOB_DIR/tmp/task9_instr.log`, "dstep stage" lines).
    - Slot 5, first decode tick: fed pos=24 kvlen=25 id=1806 (all correct) → produces
      785 where the healthy trajectory produces 749. Wrong from token 2 with perfect
      inputs ⇒ the defect is DEVICE-SIDE in that step's compute for that row.
    - Rows 6 and 7 emit CONSTANT token 14109 from their first step onward — the 'oft'
      attractor is a repeated argmax of a dead/garbage hidden state. Row 5's attractor
      is an evolving semi-garbage stream (785, 16, 6657, 311…).
    - Tick alignment: rung-8's first-ever dispatch was tick 5 (occupied=5). Slot 4 went
      live ON that dispatch — clean. Slots 5/6/7 went live 1, 2, 3 ticks later — all
      poisoned. In the sustained-2s run (clean), ~14 rung-8 ticks separated the
      program's first dispatch from the next admission.
    - Round 5, three more discriminators, two refutations and the surviving frame:
      1. Admit-5 → 30 s at rung 8 → burst 3 into rows 5-7: **ALL CLEAN.** The same
         burst shape that poisons those rows on a fresh ramp is byte-coherent once
         rung 8 has been running.
      2. Startup warm-up (3 idle dispatches of EVERY rung, incl. 8 and 16, before any
         request — implemented, validated to run, then REVERTED): burst still poisons
         ~3 slots. First-use-of-the-program is refuted.
      3. Fresh serve + 60 s IDLE + burst: still poisons. Wall-clock settling refuted.
      **The surviving pattern, consistent with every run to date: a slot is poisoned
      iff its FIRST decode step lands within ~1-3 ticks AFTER the ladder SWITCHES into
      the rung-8/16 program.** The switch tick itself is safe (slot 4, admitted on the
      4→8 switch, is always clean); the clean late-burst run is exactly the one where
      no switch occurred (rung already 8); the sustained-2s run put ~14 ticks between
      the switch and the next admission. Rung switches into 2 and 4 do NOT poison.
      Next session: find what is stale for 1-3 ticks after dp changes to the rung-8
      program while a row transitions parked→live — candidates: per-program device
      state the interpreter re-derives lazily (gq cursor bank? per-program counter
      bank `bank: Cell<u32>`?), the in.parked upload landing after the switch tick's
      dispatch, or the KDA-less parked-mask path. A host-side mitigation that needs no
      root cause: HOLD admission of a new slot for 3 ticks after any rung switch
      upward (delays TTFT by ~120 ms only on ramp ticks); implement behind an env
      knob, validate with burst-8 ×3, and take it as the interim unlock for B=8/16
      serving while the device-side cause is hunted.

## r6 (2026-08-10): three verdicts that reshape the ledger

* **The box drifts, and TTFT ratios are only valid same-session.** Untouched `final2`, served
  by the archived R2-ERA BINARY on its own objects, measures **1618.0 ms @8k** against its own
  r2-session record of 1372.2 — while TPOT is byte-stable (26.457 vs 26.503). The +18%
  8k-prefill drift is environmental (thermal/driver/box state), not a branch regression (the
  bisect binary predates every branch runtime commit). Consequence: every cross-day TTFT
  ratio in this file's earlier sections — including "1.36× vLLM @8k" — is METHODOLOGICALLY
  VOID; the r2 discipline (both engines, one session) is the only admissible form, now proven
  on plow alone. The r3 arm's "+7.8% @8k regression" dissolves into this drift too.
* **Measured tiles, unconfounded (same-session final2 vs r6-tiles A/B):** TTFT −0.4/−0.9/−0.7%
  at 4k/8k/16k — real but small — and **TPOT +5.1%** (26.485 → 27.836, tight reps). Net
  negative; adoption stays HELD. The TPOT coupling is unexplained (decode dispatches no tiled
  GEMM; suspicion: blob-layout/placement sensitivity) and is the open question on the tile
  task.
* **Both OCC4-compatible grouped-tile recuts are CLOSED**: BM=64/BK=32 stages NO A-tile
  (APASS truncates to 0 — silent wrong output, now a static_assert refusal) and BM=128/BK=32
  (SM=2, arena-fitting) hangs the megakernel's first batched dispatch. `build_gfx942.sh` now
  refuses `PLOW_OCC4=1` with `PLOW_DECODE_BATCH>1`; batched assets stay occ2/BK64 and OCC4's
  rung-1 TPOT win stays unrestored (task).
* Burst-poisoning (task 9) survived two more mitigations (admission pacing, warm-engine) —
  both measured null and reverted; the sharpened signature lives on the task.
* The op_moe.h static_assert RE-STALES the tuning store (preprocessed source changed); re-run
  `scripts/rebench_tune_gemm_gfx942.sh` before the next measured-tile emit. Held-adoption
  makes this non-urgent.

## r7 (2026-08-10): XR_AGG fixed, FOLD_TB exonerated, both re-adopted default-on

The 08-09 revert (686a3bf) had one real defect and one guilt-by-association in it.

**XR_AGG's ordering really was broken, and is now fixed** (`op_collective.h`): the failing
cut released with a fence and arrived with a *relaxed agent-scope* RMW on word 1 — which
orders nobody's stores for a remote observer, and which runs cached in the arriving XCD's
L2 on the same 128 B line the peers' system-scope signals update memory-side. The fix makes
the arrival RMW itself the release at SYSTEM scope (exactly `xctr_signal`'s form aimed at
word 1) and gives the closer the SYSTEM acquire (`xctr_acquire`) before the aggregated
cross-rank signal. Every edge in the visibility chain is now at the scope of the final
observer.

**The gate record**, all on the validated `glm52-tp8-r4b4` ladder asset, prefill-elf overlay
onto its shipping object set, one serve each:

| arm | needle 3000 | needle 8000 |
|---|---|---|
| XR_AGG fixed, solo | PASS ×2 | PASS ×2 |
| FOLD_TB=8 solo (first-ever solo gate) | PASS ×2 | PASS ×2 |
| XR_AGG fixed + FOLD_TB=8 (shipping combo) | PASS ×3 | PASS ×3 |
| **pre-fix XR_AGG control (rebuilt from old code)** | **PASS ×2** | — |

The control row is the honest asterisk: the original '741' failure did NOT reproduce on a
same-day rebuild of the old code, so it was INTERMITTENT — exactly what a memory-ordering
race looks like. A passing needle therefore cannot certify the old code, and only weakly
certifies the new; the fix stands on the memory-model argument, with 14 needle passes as
its non-refutation. (Lesson 19 updated with this.)

**FOLD_TB was never guilty on its own evidence**: 686a3bf's own message says it "was never
content-gated at length alone". Its first solo gate passed 4/4. TTFT −3.8/−5.5/−6.2%
@4k/8k/16k comes back for free.

Both defaults re-flipped in `build_gfx942.sh` (opt out: `PLOW_XR_AGG=0`,
`PLOW_MLA_FOLD_TB=0`). The op_collective.h edit re-staled the gfx942 tile store (expected);
`scripts/rebench_tune_gemm_gfx942.sh` must run against the new recipe before the next
tile-sensitive measurement.

## r8 (2026-08-10): the same-session two-engine measurement, and the honest margin

The first admissible vLLM comparison since the drift finding voided every cross-day ratio:
both engines, one session, one box state, identical client and prompts, no prefix cache,
both needle-gated before any number. plow = `glm52-tp8-r5b4-64k` + the r7 object stack
(XR_AGG fixed + FOLD_TB=8 prefill overlay on the validated B4 decode set). vLLM =
0.26.0+rocm723, AITER on, no force_mqa, max-model-len 73728. Drift bracket: a second plow
serve AFTER the vLLM hour repeated conc-1 within 0.3% (17.32→17.31 tok/s, TTFT
2281.7→2288.8) — the session is internally valid.

**Throughput ladder, in=8192 / out=128:**

| conc | plow tok/s | vLLM tok/s | plow/vLLM | plow TPOT p50 | vLLM TPOT p50 |
|--:|--:|--:|--:|--:|--:|
| 1 | 17.3 | 29.7 | 0.58× | 40.1 ms | 18.2 ms |
| 4 | 24.2 | 49.9 | 0.48× | 138.4 | 47.5 |
| 8 | 25.3 | 56.8 | 0.45× | 138.7 | 102.3 |
| 16 | 25.1 | 60.8 | 0.41× | 138.3 | 220.7 |
| 32 | 25.6 | 63.5 | 0.40× | 137.5 | 459.5 |

**Long-context TTFT** (row label = client words; tokens ≈1.5×, so "32768" is a ~49k-token
prefill; the 49152 row exceeded both engines' context and is invalid on both):

| row (~tokens) | plow | vLLM | ratio |
|--:|--:|--:|--:|
| 8192 (~12k) | 2316 ms | 1943 ms | 1.19× |
| 16384 (~24k) | 5711 | 4219 | 1.35× |
| 32768 (~49k) | 14745 | 9833 | 1.50× |

**Verdict: NOT beaten above 8k, in either mode, and the gap decomposes cleanly.**

1. **Concurrency is the structural loss.** plow is FLAT at ~25 tok/s from conc 4 up —
   the validated ladder caps at B_max=4 and everything beyond queues (TTFT p50 at conc 32
   is 141 s of queue) — while vLLM scales 2.5× further to 63.5. Interestingly plow's TPOT
   at its cap (137 ms) beats vLLM's at conc 16–32 (221–459 ms); the loss is admission, not
   the decode step. Task 9 (burst poisoning at rung ≥8) + rung-8/16/32 quality is the one
   lever that changes this row, and it also buys the expert-overlap amortisation the B=4
   seam cannot have.
2. **Single-stream decode: 40.1 vs 18.2 ms.** This asset serves rung 1 on the occ2/MM4
   batched decode objects; the OCC4/MM1 single-slot object measured 26.5 ms — task 7
   (grouped tile at OCC4) recovers ~13 ms of the 22 and the rest is the standing
   decode-attribution ledger.
3. **Prefill: 1.19× @~12k growing to 1.50× @~49k.** The r7 arms are already in this
   number. The priced lever is the MoE GLU→DOWN fusion (~8% @8k); the widening slope at
   length points at the flash path's scaling, unpriced.

The one mode plow holds: short-prompt (~1.1k) serving at 92 tok/s aggregate vs vLLM's
recorded 55.8 single-request — but no same-session short-prompt vLLM LADDER exists, so
that comparison is still not admissible-grade. The stored vLLM 8k record (1179.5 ms TTFT)
also failed to reproduce today (1943 same-config) — the drift rule cuts both ways.

## r9 (2026-08-10): BK=32 grouped tile FIXED; the OCC4 hang re-localized to the register
## ration; −12.3% rung-1 TPOT lands as the PLOW_DEC_SQUEEZE (WPE=3) recut

* **MPF_BK=32 at BM=64 had THREE BK=64 assumptions, not one.** (1) The recorded APASS=0
  truncation — now served by op_moe.h's `MPF_SUBQ` masked arm: the first BM*BK/8 threads
  (whole waves, wave-uniform predicate, no barrier inside the mask) stage one full 8-half
  vector each; loads stay 16-byte and the swizzled LDS cell map is unchanged. (2) The fp8
  promotion hardcoded BK=64 twice: `kb = kt>>1` charged the second half of every 128-element
  scale block to the NEXT block's scale, and the `(kt&1)` cadence cut the f32 accumulation
  chain mid-block — with the index fixed the outputs were still ulps away from BK=64 and the
  first probe serve saw it. Fixed by promoting at scale-block EDGES only; the MFMA chain then
  runs k=0..127 in the same order as BK=64 and the promotion is bit-identical to it. (3) The
  preshuffled-B address derived slab and byte-offset from `k0`/`el&63`, wrong when odd
  k-tiles start mid-slab; both now derive from the element k. PIPE/GH `#error` on sub-quantum
  tiles; the Gemma grouped twin (shared MPF macros, no sub-quantum arm) got the same
  static_asserts it was missing. **Proof:** occ2+BK32 batched objects serve rung-1
  trajectories BYTE-IDENTICAL to the shipped BK=64 objects over 48 tokens (independent
  serves; the paired trajectory also byte-matched a BK=64 paired serve), and the default
  BK=64 objects are byte-identical pre/post change (decode + prefill_fp8_mla_moe rows cmp'd).
* **The OCC4 hang was never the tile.** One axis at a time on the r4b4 asset, one serve each:
  occ2+BK32 PASS → +GM 128x256x32 PASS → +NO_MLA_DEC PASS → **+WPE=5 HANG**. WPE=4
  (128 VGPR, spill 20-26) also hangs; **WPE=3 (168 VGPR) serves**; GATE_HIER off (placement
  kept) still hangs; the B=1 OCC4 object (final2) serves on the same binary. The failing
  combination is {batched decode program × VGPR ration ≤ 128}, onset in (128, 168]. Two
  reframing facts for the "26.5 ms" prize: final2's blob is n_cu=304 and NOT oversubscribed,
  so at grid 304 every object runs ONE 8-wave workgroup per CU — the B=1 "OCC4" win was
  register-ration codegen, not occupancy — and the 26.5-vs-37.4 ledger delta also carries the
  B=1 blob's program shape and MM=1. The open lever is the WPE≤4 hang, now sharply bounded.
* **Landed: `PLOW_DEC_SQUEEZE=1`** (build_gfx942.sh, batched decode rows) — GM 128x256x32 +
  NO_MLA_DEC + MPF_BK=32/DBUF=1 + WPE=3: 168 VGPR / LDS 30,768 / spill 20-26. Same-session
  A/B on the r4b4 asset, both arms needle-gated (PASS @3000): rung-1 TPOT p50 **40.218 →
  35.291 ms (−12.3%)**, 17.06 → 18.58 tok/s at in=8192/out=128 conc 1; TTFT unmoved
  (2393.4 → 2390.8 ms — the correct negative control for a decode-only change).
* **Probe regression, recorded honestly:** the solo-vs-paired byte-identity probe FAILS today
  on the SHIPPED BK=64 objects (control serve, same binary/box) — X's paired trajectory is
  admission-order sensitive (r8's interleave drift), while solo and Y-paired stay stable
  across objects and serves. Cross-OBJECT byte-identity of served trajectories was the usable
  gate for this campaign and is the stronger form anyway. Separately, several determinism
  invocations wedged AFTER writing their artifacts (server kept a paired connection open past
  the complete JSON response); TERM-teardown recovered every time. Both belong to task 9's
  neighborhood, not the grouped tile.

## Next, in expected-value order

1. **Fix the burst-admission poisoning** (task; PLOW_TRACE_RAW + per-rank xctr audit, or an
   admission-pacing mitigation in the mux): unlocks B=8/16 under any arrival pattern — at
   B=16, 128 routing slots over 256 experts begin to overlap (~1.6–2× expert-weight
   amortisation), the only decode-side route to vLLM-class aggregate at long context.
2. **Fix `MPF_BK=32`** (task): wins OCC4 back for the decode object (26.5 → the rung-1 bill).
3. **Prefill @8k+**: the MoE GLU→DOWN fusion (`glm52-beat-vllm-experiment-plan.md` T1, ~8%
   @8k) and the tile-only 8k re-measure.
4. `PLOW_XR_AGG` with a correct intra-rank arrival counter (task) — the −1.7% TTFT is real
   if the ordering is.

Round 6 (counter audit, 2026-08-10): `PLOW_CTR_SNAP` differential audit landed
(engine.rs / amd.rs `ctr_word0_snapshot`) — rank-0 end-state of ALL 42,725 local
counters, every tick, diffed offline across a poisoned burst: **byte-identical on
every rung-8 tick including the poisoned ones**. Counters, counter addresses, gate
ordering, hierarchy scratch: all exonerated (the xctr audit was already silent).
The amd.rs `run()` dbuf-flip vs amd_tp sync-rearm divergence was audited: both
self-consistent, every phase drain-separated — no host sync bug found.

Consequence: the corruption is in DATA the device reads — KV ring content or
`in.*`/`act.*` bytes — for a row going live 1-3 ticks after the 4→8 switch, while
every ordering mechanism runs to spec. Next instrument (round 7): tensor snapshot
— after the newly-live slot's first decode tick, dump its KV ring rows [0..32) of
`kv.0.ckv`/`kv.0.krot` (layer 0) plus `act.qa`/`act.oat`/`act.attn` row slices,
in the failing burst AND in the clean late-admission shape, and diff. The first
tensor that differs names the op; walk its inputs backward. Design directive for
the eventual fix (user): every per-step mutable input (counters, pos, ids, kvlen,
parked) gets clean banked buffers so steps can pipeline across the entire launch.

Round 7 VERDICT (2026-08-10, tensor snapshots): **ROOT CAUSE LOCALIZED TO GEMV ROW
COVERAGE.** At the rung-8 program's first live tick per row, `act.qa` (q_absorb GemvQkv
output, M=8/Nq=4096) is: rows 0-4 fresh+correct, row 5 non-zero FOREIGN data (another
program's leftover layout), rows 6-7 ENTIRELY ZERO and byte-stale from the prior tick.
The M=8 GEMV never writes rows >= ~5. A zero query gives uniform attention and a
constant-token argmax — the 'oft' attractor, and rows 6==7 are byte-identical down the
whole chain (same zero input), which is exactly the two-attractor signature: attractor B
(rows 6-7) = zero-query, attractor A (row 5) = foreign-data query. Every earlier
observation reduces to this: 'burst vs spaced', 'switch windows', 'settling' were all
proxies for which rows ever go live at rung >= 8. Rows 0-4 work; rows 5+ never had a
working q-GEMV.

Also resolved en route: the earlier 'sustained-2s clean' result was on the B8ONLY
asset; on b16 the sustained shape poisons identically — asset/object difference, not
admission shape. The 30s/late-burst 'clean' run was invalid (story prompts EOS'd at
~40 tokens; slots freed; late requests landed low slots at low rungs).

Next (the fix): read the decode GEMV row map — op_gemm.h gemv_rows/GemvQkv M-handling
vs the object's PLOW_GEMV_MM bucket (these serves ran the r4b16 asset's own hsaco:
verify its MM) and find why coverage stops at ~5 of 8 rows at M=8: suspects are the
MM bucket vs runtime-M interaction (GEMV_MAXM=16 cap, MM buckets 4/16) and the
emitter's gemv_qkv_rows narrowing at M>1. Then the same audit for rung 16 (M=16).
Snapshots: $CLAUDE_JOB_DIR/tmp/tsnap_burst (846+ files, burst + late-admission +
sustained shapes, all on b16).

## TASK 9 ROOT CAUSE (2026-08-10, round 7 final): fused-QKV LDS staging overflow at rows ≥ 6

`d_gemv_qkvg` stages `M*K` halves of x in LDS unconditionally ("x is ALWAYS staged in
LDS here: plowc emits this op only when M*K fits GM_LDS_HALVES" — op_gemm.h). The GLM
batched-decode emit (mla.rs) NEVER CHECKS THAT FIT — the same separate-emitter-path
disease as the old dense-GQA-only L2_PLACE wiring. At the rung-8 program's packet #2
(fused q_a|kv_a|k_rope, K=hidden=6144, M=8): 8×6144 halves = 96 KiB against the 64 KiB
LDS hardware window. Rows 0-4 (60 KiB) stage inside the window and compute correctly;
ROW 5 stages PARTIALLY (attractor A: prompt-influenced semi-garbage); rows 6-7 read
fully past the window — zeros — so their q/k projections are EXACT ZEROS freshly
written every tick (attractor B, rows 6≡7 byte-identical downstream, constant-token
argmax). 64 KiB / 12 KiB-per-row = 5.33 rows = the observed 11111100 qa coverage.
B_max=4 = 48 KiB always fit — why rungs ≤4 never broke, why "burst vs spaced vs
switch vs settling" were all mirages (they only selected which rows went live), and
why the same bug class already bit gfx950 at §6g-BATCH (slots 13/14/15,
t*hidden 86016 > 73728, "exactly 13 of 16 rows fit").

The dense path's fix exists: the `gemv_staged_rows(t)*K <= gm_lds_halves()` fusion
gate (lib.rs, with the gfx942 OCC4 arena = 15,360 halves) + optional PLOW_GEMV_WALK
(staging inside the row loop, bound = min(MM,M)*K). THE FIX FOR GLM: wire the same
gate into mla.rs's batched emit — when rows*K exceeds the arena, split the fused
QKV into row-block packets at M_fit = arena/K, or fall back to the unfused per-stream
Gemv (global-x, correct at any M) exactly as the dense path does; consider
PLOW_GEMV_WALK=1 objects to keep the fusion at wide rungs (§6g-WALK's priced case:
B=16 at 142.4 vs B=8's 202.3 tok/s when fusion is lost). Validate: re-emit b16 ladder,
burst-8 (expect 8/8 byte-clean), GSM conc-8 on B16 (expect ≥0.98 vs the recorded
0.7833), then the vLLM margin re-run with B=8/16 admission unlocked.

## r10 (2026-08-10): the fix validated at quality, on the self-contained nix 7.14 stack

Full gate set on `glm52-b16-fixed` served with `hsaco-nix714-b16` (TheRock ROCm 7.14 /
clang-23 objects, campaign recipe, nix runtime libs): burst-8 coherent 8/8 (no attractors;
only the pre-existing interleave drift), needle 3000+8000 PASS, **GSM8K 8-shot conc-8:
0.9833 (59/60), errors=0, wall 2.0 min** — vs 0.7833 twice on the broken ladder and 2.7 min
wall for the same workload on B_max=4. B=8/16 admission is quality-validated; the rung-8
seam amortization is real (same accuracy, −26% wall at conc 8). Also merged this round:
worktree-nix-selfcontained (whole build+dev inside nix) and the OCC4 agent's BK32 fix
(r9: PLOW_DEC_SQUEEZE=1, rung-1 TPOT −12.3%).

## r11 (2026-08-10): the same-session rerun with the fixed ladder — gap narrowed, not closed

Both engines, one session, fixed B16 ladder (`glm52-b16-fixed-64k` + `hsaco-nix714-b16`,
ROCm 7.14 stack) vs vLLM 0.26 (first vLLM serve failed its needle gate with degenerate
output — refused and retried; the retry gated clean). in=8192/out=128:

| conc | plow tok/s (TPOT ms) | vLLM tok/s (TPOT ms) | plow/vLLM (r8 was) |
|--:|--:|--:|--:|
| 1  | 15.0 (49.3)  | 30.7 (18.2)  | 0.49× (0.58×) |
| 4  | 22.5 (146.9) | 51.3 (45.6)  | 0.44× (0.48×) |
| 8  | 30.1 (241.0) | 57.7 (101.0) | 0.52× (0.45×) |
| 16 | 36.7 (403.1) | 62.7 (215.1) | **0.59×** (0.41×) |
| 32 | 37.1 (410.5) | 64.3 (456.0) | **0.58×** (0.40×) |

LC TTFT (words; ~1.5× tokens): 2271/5585/14114 vs 1896/4165/9778 ms = 1.20/1.34/1.44×.

Verdict: **still not beaten, but the concurrency unlock is real** — plow now SCALES
(15→37.1 vs the old flat 25), conc-16 admission keeps up (TTFT 3.5 s vs 62 s queued), and
at conc 32 plow's TPOT (410 ms) is BETTER than vLLM's (456) — the remaining conc-32
deficit is pure admission width (ladder tops at B=16; TTFT p50 57 s is queue). The rows
that own the gap now: (1) conc-1 decode 49.3 vs 18.2 — the MM16 object tax
(PLOW_DEC_SQUEEZE −12.3% pending its 7.14 gate battery; objects built, cliff PASS) plus
the B=1-blob share; (2) mid-conc TPOT (the grouped-seam expert re-read at B=4-16); (3)
prefill 1.2-1.44×. A B=32 ladder rung is now a legitimate experiment (the LDS fit gate
makes it correct by construction; XArgmaxFin caps at 16 under GLM_SHARD_HEAD — needs the
cap addressed or the head unsharded at rung 32).

r11b — PLOW_DEC_SQUEEZE on the 7.14 stack: **correctness battery PASSES** (cliff asserts
under clang-23, burst-8 coherent, needle@3000, GSM conc-8 0.9833/60 errors=0, wall 2.3 min).
**Default-on stays PENDING one matched TPOT A/B**: the agent's −12.3% was measured on the
old toolchain and the MM4/B4 asset; today's crude short-prompt probe is not comparable to
the r11 conc-1 row. Next GPU session: run_speed CONCS=1 CTXS=8192 on the b16-fixed-64k
asset, squeeze vs non-squeeze objects, same serve shape — flip the default iff the win
reproduces. Objects staged: hsaco-nix714-b16-squeeze.

## r12 (2026-08-10): B=32 landed correct; the walk tax is the finding

Two-line XArgmaxFin + PLOW_GEMV_WALK objects + LDS-fit-gated emit: rungs [1..32] serve,
burst-32 coherent, needle PASS, GSM conc-32 0.9844 (63/64, errors=0, wall 1.5 min/64q).
Same-session vs vLLM at in=8192: conc-32 plow 40.0 tok/s (TPOT 760.5, TTFT 3.66 s — the
queue is GONE) vs vLLM 64.4 (453.4). NOT beaten: the rung-32 step pays the walk's
ceil(32/16)=2 weight passes on every GEMV — capacity bought with bandwidth, per the walk's
own cost model — nearly doubling TPOT over rung-16 (410→760). The walk also costs rung 1
~17% (57.9 vs 49.3 ms) from codegen alone, so walk objects must not serve low rungs.
Conclusion: admission width is SOLVED (TTFT flat to conc 32); the beat now runs entirely
through per-rung OBJECT selection (task 13: squeeze/OCC4-class for rungs 1-2, plain MM16
for 4-16, walk MM16 only for 32) + the wide-rung fusion recovery (task 16) + expert
amortization. Raw JSONs beside the note.

## r13 (2026-08-10): the two-tier final — TTFT beat lands, throughput does not

Same-session final, publication profile (in=8192/out=128, NMULT=4, needle-gated both
arms). plow arm: `glm52-b32` asset served two-tier — `hsaco-nix714-b16` (plain MM16, no
walk) for rungs ≤16 via `PLOW_HSACO_LOWRUNG`/`PLOW_LOWRUNG_MAX=16`, `hsaco-nix714-b32`
(walk) for rung 32 — on the nix ROCm 7.14 stack. vLLM 0.26, same box, same session.

| conc | plow tok/s (TPOT / TTFT ms) | vLLM tok/s (TPOT / TTFT ms) | ratio |
|--:|--:|--:|--:|
| 1  | 14.7 (49.6 / 2242)  | 30.6 (18.2 / 1850)  | 0.48× |
| 4  | 21.9 (154.4 / 3441) | 51.5 (44.9 / 4205)  | 0.43× |
| 8  | 29.9 (242.3 / 3467) | 58.0 (100.0 / 4425) | 0.51× |
| 16 | 36.5 (405.7 / 3536) | 61.6 (216.3 / 5239) | 0.59× |
| 32 | 39.9 (763.0 / 3672) | 64.5 (452.8 / 5678) | **0.62×** |

LC TTFT@8192: 2273 vs 1901 ms (1.20×). Raw JSONs beside the note
(`{plow,vllm}-final_speed.json`).

What LANDED: (1) **TTFT beats vLLM at every conc ≥ 4** — 3.4–3.7 s flat vs vLLM's
4.2–5.7 s climbing queue: the prefill+decode-in-step admission is now strictly better
under load, and the two-tier co-load costs it nothing. (2) Two-tier object selection
works in service: rung ≤16 TPOT 405.7 = pure-B16's 403–410, while rung 32 admits
without queuing. Best throughput ratios of the campaign at 16/32.

What DID NOT: decode TPOT. The rung-32 walk pays ceil(32/16)=2 weight passes
(763 ms vs 453 to match vLLM); and per op_gemm.h's own roofline (~9 FLOP/byte
scalar-FMA crossover), a scalar GEMV at M=32 is COMPUTE-bound even with one pass —
a wider MM is not the fix. The fix is task 16 re-scoped: at wide rungs route the
dense projections to the MFMA GEMM family (`pick_tile(rows, N, K)`, exactly what
prefill's T=128 bucket already does at M=128) — 1× weight traffic, matrix-core FLOPs,
no LDS M·K bound, and `check_gemv_capacity` pressure drops back to 16. Requires the
decode object to compile the Gemm arms (register-budget check against the decode
union) + pairing marker. Also open: serving-asset tiles are ALL analytical
(`tile_measured: 0` in build.json) — the 7.14 tune campaign has never run (task 6);
squeeze low-tier A/B; loader gather (task 17); DSA slope (task 14).

## r13a (2026-08-10): task-11 LL push — implementation dead-on-arrival, parked

The LL flag-in-payload implementation (from another worktree, f48f511) is complete and
carefully argued, but its dispatch guard requires
`PLOW_XR_LL_BYTES <= partial_bytes`: ~68 MiB of seq entries + packet regions against a
192–393 KiB slot. The LL path can NEVER execute — a built object (verified) is a
functional no-op, every collective falls back to legacy. Structural, not a tuning miss:
LL packets need n×128 B where slot 2 holds n×2 B. A real fix needs a dedicated peer-pool
region (exec/tp.rs), per-gate (not per-pair) seq counts, and a small-n gate (LL is a
latency protocol; 2× wire hurts at wide-rung message sizes). Prize is bounded — ~157
collective rendezvous/step ≈ 0.5–1.2 ms of rung-1's 49.6 — so parked behind 16/14/6.
Lesson (again): an unvalidated agent branch is a design note, not a result; the guard
that "falls back safely" is also the guard that falls back silently.

## r14 (2026-08-10): single-block ladder at in=4096 — squeeze refused, the flash slope unmasked

Iteration profile (NMULT=1, out=64, in=4096, same two-tier stack as r13), needle-gated.

Base ladder (conc → tok/s / TPOT ms): 1 → 15.6/48.8, 2 → 17.4/99.3, 4 → 25.0/128.1,
8 → 32.0/177.9, 16 → 38.4/267.8, 32 → 41.2/461.8.

**Squeeze low-tier REFUSED at the serve A/B**: with `hsaco-nix714-b16-squeeze` as
LOWRUNG (max 2), conc-1 TPOT 61.4 vs 48.8 (+26%), conc-2 107.5 vs 99.3 (+8%). The r9
−12.3% did not survive a matched serve measurement; PLOW_DEC_SQUEEZE stays opt-in, b16
stays the low tier. (JSONs: iter1/.)

**The context slope isolates the batched flash.** Same stack at in=4096 vs in=8192
(r13): rung 1 flat (48.8→49.6); rung 8 178→242; rung 16 268→406; rung 32 462→763 —
linear at ~8–9 ms *per slot per +4096 ctx* (rung-32 extrapolation 462+301=763, exact).
Bandwidth justifies ~75 µs of that (368 MB latent KV per slot per 4k across 78 layers at
~5 TB/s). FlashMlaDecode at batched rungs is ~100× off bandwidth; at 8k ctx it is ~600 ms
of the 763 ms rung-32 step — the walk tax and everything else together are ~160 ms.
**Task 18 opened; this is the #1 throughput lever**, ahead of wide-rung GEMM (16).
The beat arithmetic: fix the flash slope to near-bandwidth and rung-32 at 8k drops
toward ~170–200 ms — under vLLM's 453 with margin, at every context.

## r15 (2026-08-10): the N-tier object ladder — dead-lane recovery measured and adopted

The MM bucket's cost is per COMPILED MM, not per live row: the m-loop unrolls MM
accumulator lanes and predication does not retire the dead ones. r14's base served
rungs 1-16 on the MM16 object; serving each rung on its exact-MM object
(`PLOW_HSACO_LOWRUNG=b1:1,b2:2,b4:4,b8:8,b16:16`, loader extended at 50e6f2a)
measures, single-block in=4096 vs r14 base TPOT ms:

| rung | base | N-tier | Δ |
|--:|--:|--:|--:|
| 1  | 48.8  | **36.1** | −12.7 (−26%) |
| 2  | 99.3  | **85.2** | −14.1 |
| 4  | 128.1 | **119.1** | −9.1 |
| 8  | 177.9 | **170.0** | −7.9 |
| 16 | 267.8 | 267.3 | −0.5 (same object) |
| 32 | 461.8 | 461.6 | 0 (same object) |

Savings track dead-lane count (16−r), confirming the mechanism. ADOPTED as the
standard serve config; objects hsaco-nix714-b{1,2,4,8}. Squeeze A/B (r14) is
superseded by this ladder at the same rungs. Note for the census: rung 2 at 85.2
is still 2.4× rung 1 — two slots of a bandwidth-shared step should be nearly flat;
whatever splits 36→85 between one and two slots is the next base/slope question.
