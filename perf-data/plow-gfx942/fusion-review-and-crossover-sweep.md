# Fusion review, op 115, and the batch×ctx crossover sweep (Gemma-4-12B, MI300X)

Date: 2026-08-06. Objects: `PLOW_OCC4=1 PLOW_L2HIER=1 bash scripts/build_gfx942.sh`
at 4631d82 (the Q-staging fix — see below). Blob: the cap recipe
(`PLOW_FP8=1 PLOW_W8A8=1 PLOW_L2_PLACE=1`, `--seq 128,512,1024 --batch 1`).
Bench: `amd-bench --steps 48`, 2–3 interleaved reps; correctness by the serve
token gate (temperature 0, "Paris" + first 8 primes), NOT by `last id`.

## 0. A correctness bug shipped in 887d90e and this sweep found it

Commit 6748e5b ("Q staging as_glob+vectorise is null; **reverted**") deleted the
flash-decode Q-staging loop and kept only its comment: `qsm` was never filled,
so every object rebuilt from 887d90e decoded fluent garbage from token 2 while
timing exactly like a healthy build. It went unseen because the commit landed
AFTER the session's last serve validation, and every number in
`gemv-mlp-and-tensile.md` was measured on objects built BEFORE it (hsaco_pf2,
Aug 5 14:32) — those numbers stand. Fixed in 4631d82; the serve gate passes
again on a fresh rebuild. Bisection that found it: blob byte-identical (cmp),
kernels golden-clean, mixed-hsaco phase isolation → decode object → full-revert
variant still broken → post-14:32 commit diff.

## 1. Fusion review (the ask: "are the fundamental fusions landed and fusing properly")

Landed and verified correct (epilogues read against HF semantics):

| fusion | op | verdict |
|---|---|---|
| gate\|up GLU GEMV | GemvGluFp8 36 | correct: act=0 = gelu_pytorch_tanh, per-row fp8 scale applied before activation, situ trap NaNs, K-tail clamped, UN dispatch measured (UN=3 → rule: −3.3%) |
| sandwich norm pair | NormResidualNorm 23 | correct: `resid=(a+norm(b)·γb)·layer_scale` with bf16 rounding faithful to the unfused pair; Gemma-4 dropped `(1+w)` so raw γ is right |
| head norm + RoPE + KV write | HeadNormRope 3 | correct incl. batch>1 KV ring (`n_batch_kv`); v arm is a windowed-ring scatter |
| plain fp8 GEMV pair loop | GemvFp8 30 | correct: odd tail split out (`gemv_fp8_rows_1`), buffer-rsrc OOB = free bounds check |
| split-KV fold | FlashMerge 13 | correct; fold measured 0.60 µs |

The ONE missing fundamental was fused Q|K|V on the fp8 path ("opcode 26
deferred" in devgen). Landed as **PLOW_DOP_GEMV_QKV_FP8 = 115**: one body with
op 30 (Nk=Nv=0 → bit-identical plain GEMV; decode object VGPR/LDS/spill
unchanged at 104/30736/0), op 114's slot map (three f32 scale rows as tensor
handles in i5/i6/i7). Golden test `runtime/tests/gemv_qkv_fp8_gfx942_test.hip`
passes all four arms vs a CPU f64 oracle; serve tokens identical fused vs split.

**And it is OFF by default, because it measures slower** (48 steps, 3
interleaved reps, ms/token): split3 12.080 vs fused 12.232 @4k (+1.3%), 12.150
vs 12.273 @8k (+0.9%). Same verdict as T11's bf16 probe. The two deleted gates
were free — the traced q/k/v packets start within 0.3 µs on disjoint CU sets —
while fusing coarsens the gemv→headnorm dependency from the fine per-head
producer map to a whole-304-WG wait. `PLOW_FUSE_QKV_FP8=1` opts in (static
scheduler / batched decode are where the trade could flip).

## 2. ctx sweep, batch 1 (plow fp8+occ4, fixed objects; ms/token)

| ctx | plow | vLLM stored CSV | vLLM re-measured | plow/vLLM |
|---|---|---|---|---|
| 1k  | 11.96 | 6.80  | —     | 1.76× (stored) |
| 4k  | 12.08 | 7.57  | 10.03 | 1.20× (re-measured) |
| 8k  | 12.12 | 8.62  | 11.08 | 1.09× (re-measured) |
| 16k | 12.25 | 9.21  | —     | 1.33× stored / ~1.03× corrected |
| 32k | 12.81 | 11.18 | —     | 1.15× stored / **~0.89× corrected** |
| 64k | 13.70 | 12.70 | —     | 1.08× stored / **~0.84× corrected** |

"Corrected" scales the stored CSV by the factor the re-measurement moved it
where both exist (×1.325 @4k, ×1.285 @8k; ×1.28 applied ≥16k). plow's curve is
nearly FLAT (+15% over 64×ctx — 40 of 48 layers are 1k sliding windows; only
the 8 full k_eq_v layers scale), vLLM's rises +87%.

**The crossover, if it exists, is single-stream long context: ~16k on the
corrected baseline, plow ~1.1–1.2× ahead at 32k–64k — but on the raw stored
CSV plow never quite crosses (still 1.08× behind at 64k).** Which of those is
true is exactly task #9 (a fresh vLLM baseline on a working box); the stored
CSV is known to under-report by ~30% at the two points that could be checked.
64k blob passes the serve gate.

## 3. batch sweep, ctx 1024 (per-request ITL = decode-step ms)

| concurrency | plow best | plow objects | vLLM | plow aggregate | vLLM aggregate |
|---|---|---|---|---|---|
| 1  | 11.96 | occ4          | 6.84  | 84 tok/s  | 146 |
| 4  | 27.4  | occ4 (=occ2)  | 7.50  | 146 tok/s | 533 |
| 16 | 45.9  | occ2, MM=16   | 10.99 | 348 tok/s | 1456 |

(occ4 at MM=16 is 117 ms/step — the 104-VGPR cap is a batch-1 profile; batch
objects must be built without PLOW_OCC4. Prior-session 63.6 ms @b16 predates
this session's levers.) Batch blobs were emitted `PLOW_L2_PLACE=0`: batched
blobs L2-place prefill packets, which the prefill objects don't dispatch —
costs the −1.5% decode lever on these points only. Timing-only measurements
(amd-bench); the b=1 serve gate is the correctness anchor.

**Batching is decisively vLLM's region.** plow's decode step grows 12→27→46 ms
over b=1→4→16 (the packet-serial chain re-pays per-sequence work: 16 KV rings
through flash, wider staging) while vLLM amortizes to 10.99. plow's aggregate
scales only 4.2× at b=16.

## 4. Where plow gains — the answer

- **batch-1, long-context (≥16k) decode** is the only region with a plausible
  win, and it is plow's by DESIGN (flat sliding-window curve), not by kernel
  superiority: at 4k–8k plow is 1.09–1.20× behind on re-measured numbers.
- The trend is monotone: every doubling of ctx past 8k costs plow ~4% and
  vLLM ~15–30%. At 128k (both engines untested here) the gap would favor plow
  even against the uncorrected CSV.
- Batch ≥4 and short ctx are vLLM's, by 3–4×.
- Gate: task #9. No crossover claim should ship without a same-session vLLM
  baseline.

## 5. Addendum (same day, later): both NRN folds + the fp8 head land

Three levers found and landed after the sweep above (all serve-gated, ms/token,
48 steps × 3 interleaved reps, occ4 objects at 9c7c15c/edc2a18):

| config | 4k | 8k |
|---|---|---|
| session-start baseline | 12.09 | 12.15 |
| + NRN2→q/k/v fold (op 30 i3) | 11.91 | 12.01 |
| + NRN1→GLU fold (fj2.u + i-slot handles) | 11.76 | 11.86 |
| + fp8 lm_head (PLOW_FP8_HEAD=1, `scripts/quantize_fp8_head.py`) | **11.34** | **11.42** |
| vLLM re-measured | 10.03 | 11.08 |
| ratio | 1.13× | **1.031×** |

- Both Gemma sandwich norms now fold into their consuming GEMVs' staging
  (bit-exact, token-identical on the serve gate); ONE NormResidualNorm remains
  per token (the final norm). Decode program 622 → 527 packets.
- The i-slot-handle demotion (op 114's rule) is what unblocked the GLU side —
  the "needs 9 slots" note predated that precedent. The fold body had to be a
  SEPARATE function: a branch inside d_gemv_glu_fp8 cost +0.5 ms on blobs that
  never take it (hot-loop schedule), at identical resource stats.
- The fp8 head halves a 2 GB/token bf16 read. Own reporting row (vLLM's fp8
  recipe keeps lm_head bf16): greedy outputs stay factually identical; a
  free-form tail token can flip. The fp8 checkpoint was missing the embed
  table — PLOW_FP8_HEAD=1 memory-faulted until the script above wrote it, and
  plowrt binds the missing tensor without refusing (open task).
- Dead ends this round: PLOW_FUSE_ARGMAX (no AMD arm, and the tail is
  once-per-token ≈ 0.02 ms), 8k-capped blob for ns8 (ties, as the in-tree
  sweep said), fold FlashMerge into o_proj staging (78 MB of re-read).

At 8k the gap to the re-measured vLLM baseline is 0.34 ms (3.1%) — inside the
band where the baseline's own reproducibility (±0.3 ms drift observed across
re-measurements) decides the sign. 4k remains 1.13×. The one identified lever
left is the HeadNormRope→FlashDecode fold (~0.4 ms: deletes the hnr chain
level; slots JUST fit via a merged cos‖sin table, pos = kvlen−1, and handles
in i7/fj1/fj2) — a deep flash-decode surgery.

- One more null: PLOW_FP8_KV=1 on top of the h8 config measures +0.03 ms @4k /
  +0.08 @8k (n=2 interleaved) — the flash dequant VALU costs more than the
  ~170 µs/token of KV bandwidth it halves at these context lengths. It should
  win at 32k+ where the full-layer KV read scales; untested there.

## 6. The one designed-but-unlanded lever: HeadNormRope → FlashDecode fold

Deletes the hnr chain level (3 concurrent b=2 packets between qkv and flash),
estimated −0.4 ms — at 8k that is the difference between 1.03× and a tie. The
slot math JUST closes: t2 carries qg (raw) instead of Q, t6/t7 take kg/vg
(vg = kg on k_eq_v layers), fj1.u/fj2.u take the γq/γk handles, i4 takes a
MERGED cos‖sin generated table (both are plowc-generated, so merging is legal
where repacking checkpoint tensors is not), window moves into i0's high bits
(n_batch is a u8 at decode), pos = kvlen−1, and eps still needs a home (bake
into the merged table's header row, or a second fj-overlay rule). The kernel
side replicates d_headnorm_rope per element in the Q staging (wave-per-head,
same wave_sum order), substitutes the CURRENT KV row from locally-computed
values (never read back — L1 staleness), and gives the cache write to the one
(hkv, split) item whose range covers pos. Decode b=1 only; batch keeps the
packets. High-risk surgery in the kernel that shipped the 6748e5b bug — do it
with the serve gate + control blob discipline this file documents, and only
after task #9 says the 8k tie is worth buying.

## 7. The hnr→flash fold, landed and measured: a NULL, and why

Implemented (§6's design, d_flash_decode's NRF template arm + the packed-slot
wire): all 144 HeadNormRope packets leave the decode program (527 → 383
packets), serve tokens identical on the FIRST build. Three measurements, 48
steps × 3 interleaved reps @4k:

| variant | ctl | fold |
|---|---|---|
| coarse deps + agent-scope fence per owner | 11.36 | **11.99** |
| coarse deps + workgroup release (correct)  | 11.36 | 11.40 |
| FINE per-head deps (hnr's own map)         | 11.33 | 11.37 |

Two lessons, both now paid for:
1. The agent-scope fence's `buffer_inv` is a FULL L1+L2 invalidate; one per
   owning item destroyed the KV lines every other workgroup was streaming
   (+0.6 ms). The packet-gate acquire already guarantees no stale L1 line for
   the written row, so a workgroup release (waitcnt, no cache ops) is the
   entire requirement.
2. With that fixed the fold is +0.3% — a null, op 115's verdict again. The
   estimate ("delete a ~10 µs chain level") was wrong for the same reason:
   the hnr packets' FINE producer maps let them start before the slowest gemv
   workgroup, so the level's true wall-clock cost was the small post-producer
   tail. plow's fine-grained counter machinery has already collapsed every
   "delete a packet" win of this shape; the remaining floor is the per-packet
   claim/poll machinery itself and the flash/GEMV bodies.

Kept in-tree, OFF by default (PLOW_FUSE_HNR=1): correct, token-identical, and
the trade flips where fine deps do not exist (static scheduler, launch-per-op
backends). The decode object's resource envelope is unchanged (104/30736/0)
and the dormant arm costs the default path nothing (template arm, not a
branch — the d_gemv_glu_fp8_nrn lesson applied).

## 8. Final standings (2026-08-06 end of session)

| | 4k | 8k |
|---|---|---|
| plow (both NRN folds + fp8 head, serve-gated) | **11.34** | **11.42** |
| vLLM re-measured | 10.03 | 11.08 |
| ratio | 1.13× | 1.031× |
| session start (same day) | 12.09 | 12.15 |

Goal (beat vLLM at both) NOT met. The 8k gap (0.34 ms) is inside the
baseline's own observed reproducibility band; the 4k gap is real and bounded
by the per-packet floor + GEMV weight-stream rate, which the trace campaign
priced at more than the remaining kernel opportunity. Task #9 (a same-session
vLLM baseline) remains the gate on any final claim.

## 9. Oversubscribed cooperative launch: a decisive loss (+62%)

The last untested attack on the ~4 ms per-packet floor: launch 2 workgroups
per CU (the occ4 envelope fits: 2 × 104 VGPR ≤ 512/SIMD, 2 × 30.7 KB ≤ 64 KB)
so one workgroup's gate poll hides behind its sibling's body. plowrt gained
PLOW_OVERSUB=1 (expert env; the persistent kernel SPINS, so a non-resident
workgroup is a deadlock, and the L2-place wg→domain map assumes grid == CUs so
placement must be off in the blob). It launches and runs — and measures
22.86 ms/token vs its own 304-WG control's 14.08 (+62%, 3 reps): halved
per-WG work doubles every packet's fixed cost (claim, stream-entry load, gate
participation) and the claim contention swamps whatever hiding occurs. Same
family as the GQ_BATCH=2 lookahead (+32%): the claim path does not tolerate
more claimants.

Side observation worth its own follow-up: the no-L2-place control reads
14.08 vs 11.34 for the placed default — L2-domain placement is now worth
~24% in the full current config, far above the −1.5% measured when it first
landed (the folds and fp8 head shrank everything else, and GATE_HIER +
placement compound). Do not benchmark gfx942 decode with PLOW_L2_PLACE=0.

With this, every identified lever at the current design level is measured:
shipped (occ4, fp8 weights/head, L2 place, GATE_HIER, INST_PF, UN rules,
nsplit caps, both NRN folds, counter double-buffer) or null/negative (op 115,
hnr→flash, fp8-KV ≤8k, oversubscription, lookahead, static scheduler,
merge→o fold, 8k-cap blob, HN split). The remaining 1.31/0.34 ms at 4k/8k
prices out to the per-packet claim/poll machinery itself and the fp8 GEMV
marginal rate — the interpreter-redesign tier — plus a baseline (task #9)
that cannot be re-measured on this box.

## 10. Last two probes: WPE=4 (+15%) and the standalone-GEMV myth retired

- **PLOW_WPE=4** (128 VGPR instead of occ4's 104; same 4-waves/SIMD envelope on
  paper): 13.04 vs 11.36 ms/token (+15%, 3 reps) — 21 VGPR spills appear and
  the envelope sits exactly at the 512/SIMD boundary. WPE=5 was the right call.
- **The fp8 GEMV standalone probe** (runtime/bench/gemv/fp8_unroll.hip's tmp
  twin) finally ran. Raw it reads down=2970 GB/s vs the in-model ~2034 marginal
  — but the probe DROPS THE ODD TAIL (its own header says 10 of 13 columns at
  these shapes), and correcting for the ~23% uncounted work gives ~2290 GB/s:
  **the in-model and standalone rates are the same**. There is no megakernel
  context penalty on the GEMV bodies; the kernel itself caps at ~2.3–3.0 TB/s
  (45–56% of peak), UN-flat, at either occupancy. A doubled-grid probe run
  produced 18 TB/s "results" — impossible, and diagnostic: per-WG rows fall
  under PLOW_WAVES and the tail-less pair loop computes nothing. Do not quote
  any number from a tail-less GEMV probe.

Raising the GEMV family from ~2.4 to ~3.5+ TB/s is the one remaining lever
that would flip 4k (worth ~1.7 ms), and it is a from-scratch kernel campaign:
the UN sweep is flat, the register budget is measured-binding in both
directions (WPE=4 spills, the R-parameterised body loses 2.5–3% to scheduling),
and the dequant+dot path is ~5 VALU/element on CDNA3's bf16-dot-less ISA.

## 11. Correction to §10's closing claim — the GEMV bodies are ALREADY at roofline

§10 said raising the GEMV family "from ~2.4 to ~3.5+ TB/s" was the remaining
4k lever. That division was wrong, and the tree's own ablation ladder shows
it: at ABL=5 the four EMPTY plain-GEMV packets still cost 9.2–12.3 µs each
(≈2.0 ms/token of pure per-packet machinery, gemv-mlp-and-tensile.md §floor),
so of the 5.30 ms family total, the BODIES move their 12.91 GB in ≈3.3 ms —
≈3.9 TB/s marginal, i.e. at the practical roofline. The 2.4 TB/s figure
divides by time that includes the machinery. The tail-corrected standalone
probe (§10) agrees once its launch floor is subtracted.

So the residual 4k gap decomposes to: ~2.0 ms of per-packet machinery across
the GEMV family + the FlashDecode floor (1.42 ms) + the serial-chain gate
waits — the interpreter-design tier, attacked across two sessions with seven
shipped levers (occ4, L2 place, GATE_HIER, INST_PF, counter double-buffer,
both NRN folds, UN/nsplit rules) and five measured negatives (lookahead,
static schedule, oversubscription, hnr fold, op 115). There is no kernel-level
lever left; a goal-meeting effort requires redesigning the packet machinery
itself (per-layer mega-packets with in-kernel sequencing, or doorbell-style
gating) — and a re-runnable vLLM baseline (task #9) to know whether 8k even
needs it.

## 12. Mega-mode decode (level barriers on static streams): killed at the entry gate

The last interpreter-tier design: execute the topologically-ordered decode
stream level-by-level on the STATIC per-CU streams, one hierarchical barrier
and one buffer_inv per LEVEL instead of per packet (gate memoization: skip the
gate when consecutive entries share a wait list). The cost model against the
placed-GQ baseline said −2 ms. The entry measurement kills it:

  placed GQ (shipping)          11.34 ms/token
  unplaced GQ                   14.51        (L2 placement is worth −22% — §9)
  unplaced STATIC               15.34        (+5.7% over unplaced GQ)

Static mode is where the level machinery would live, and it starts 4.0 ms
BEHIND the shipping config: the "static is only 2.6% slower" datum predates
this session's folds and the L2-domain placement, which is a GQ-only feature
(domain-filtered claiming over the gq_stream). A levelization pass would have
to first re-create placement's producer-consumer L2 locality under fixed
assignment AND fix static's imbalance, before its own barrier savings count.
That is not a session lever; it is the full interpreter redesign, now with a
measured −4.0 ms starting handicap.

Also learned: PLOW_STATIC_DECODE=1 on an L2-PLACED blob DEADLOCKS — placed
entries carry seg=domain, and the static path's `seg != cur_seg` filter skips
7/8 of the packets, so their counters never signal. plowrt should refuse that
combination instead of hanging (open task).

## 13. Post-fold retune sweep: every remaining knob confirms the shipped value

The tunables were all chosen BEFORE this session's folds changed the packet
mix, so each was re-swept on the final stream (h8 config, 48 steps × 2 reps):

| knob | 4k | 8k | verdict |
|---|---|---|---|
| baseline (ns16, no VPIPE) | 11.35 | 11.44 | shipped |
| PLOW_NS_ABS=8 | 11.68 | 11.98 | the old "ns8 wins ≤8k" FLIPPED post-folds |
| PLOW_NS_ABS=12 | 11.42 | 11.72 | worse |
| FA_DEC_VPIPE=8 (V prefetch over softmax) | 11.57 | 11.67 | worse (104 VGPR held over the barriers) |

With this, the tunable surface is closed on the final stream: every knob is
either at its measured optimum or documented as a loss. plow's terminal
numbers on this box are 11.34/11.42 ms at 4k/8k against a vLLM baseline of
10.03/11.08 (re-measured) or 7.57/8.62 (stored CSV) — not beaten at either
point under either baseline.

## 14. The GEMV loop question, settled by construction and counter-measurement

§10 vs §11 flip-flopped on whether the fp8 GEMV loop leaves rate on the table.
A tail-correct standalone variant probe (fp8gemv3) found a real one: dequant
straight to f32 (skipping the bf16 pack that `dot2` immediately unpacks) is
−8..−11% standalone (down 25.67→23.62 µs, q 10.01→8.89). Wired into all three
megakernel bodies it measures **+1%** (11.32→11.42 @4k, 11.43→11.52 @8k) at
identical resource stats — REVERTED, probe and numbers recorded on the helper
(`plow_fp8x16_to_f32`, amd_arch.h). The resolution of the flip-flop: the
in-model GEMV is memory-LATENCY-bound at 2 waves/SIMD, so its dequant VALU is
free (hidden under load latency) and its rate ceiling (~2.3 TB/s standalone
AND in-model) is set by outstanding-load latency coverage at the occupancy the
megakernel's union register budget permits. The standalone probe's win existed
only because it ran at a 256-register budget. Raising the GEMV rate therefore
requires MORE WAVES — and every route there is measured closed: grid
oversubscription +62%, 16-wave WGs break the 512-thread contracts, WPE=4
spills, occ4 is the measured best register/wave point. The kernel-rate lever
is closed, this time with the mechanism, not just the number.

## 15. Task #9 executed — and plow BEATS vLLM at 8k in the same-session showdown

The "impossible" re-baseline fell to a direct pip install nobody had tried:
vLLM 0.26.0+rocm723 from wheels.vllm.ai/rocm (its torch 2.11 needed the ROCm
7.2.4 runtime, installed side-by-side via the already-configured apt repo;
libopenmpi3 for the wheel's MPI dep; the register-inspection SIGABRT was mixed
7.1/7.2 libraries until rocm-hip-libraries completed the 7.2.4 set). Serve
flags and bench methodology identical to the stored CSVs. Coherence gate:
'Paris'.

TWO discoveries changed plow's own numbers before the showdown:
1. **The ROCm 7.2.4 HSA runtime is worth −0.5 ms/token to plow** (11.32 →
   10.82 @4k on identical objects+blob) — the persistent kernel's host
   interface (doorbells/queues) got cheaper. plowrt now needs
   LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib (the apt install RESTRUCTURED /opt/rocm:
   lib/ is gone, hipcc needs ROCM_PATH=/opt/rocm-7.2.4 — see the ops note).
2. **The hnr→flash fold (§7's null) FLIPPED on the new runtime**: −0.04 ms
   @4k, −0.03 @8k, consistent — the cheaper dispatch changed the trade. The
   best config is now h8 + PLOW_FUSE_HNR=1 on clang-22 (7.2.4) objects,
   serve-gated ('Paris' + primes).

THE SHOWDOWN (strict alternation, same GPU 0, same minute-scale window,
vLLM server resident at 0.35 mem-util, 4 rounds × [vllm bench serve 3 prompts,
plow amd-bench 128 steps]):

| ctx | vLLM 0.26 (r1..r4) | plow (r1..r4) | verdict |
|---|---|---|---|
| 8192 | 10.93 10.91 10.94 10.95 (mean 10.933) | 10.836 10.857 10.839 10.847 (mean 10.845) | **plow WINS by 0.088 ms (0.8%) — all 4 rounds separated, zero overlap** |
| 4096 | 9.79 9.81 9.81 9.83 (mean 9.81) | 10.751 10.747 10.754 10.738 (mean 10.748) | vLLM wins by 0.94 ms (1.096×) |

The 8k half of the goal is MET, with the strongest measurement this project
has ever had: fresh engines, both measured minutes apart on one GPU. The 4k
half stands at 1.096× — vLLM's 4k attention is cheap while plow sits at its
context-independent floor; closing 0.94 ms remains the interpreter-redesign
tier (every session-scale lever measured, §0–§14).

Baseline history for the record: stored CSV 7.57/8.62 → old remeasure
10.03/11.08 → fresh 0.26.0 9.81/10.93. The baseline moved every time it was
measured; the same-session protocol is the only defensible one.

## 16. Placed STATIC decode, measured for the first time — the −4 ms handicap is the fixed assignment, not missing placement

§12 killed mega-mode at the entry gate on the argument that static started
−4.0 ms behind placed GQ, but the comparison had a confound: static had only
ever run UNPLACED, because PLOW_STATIC_DECODE=1 on a placed blob deadlocked
(the static window was selected by the host's wave-class `cur_seg`, which is 0,
while every placed entry carries seg=domain — 7/8 of each CU's stream emptied
and the rest spun on their counters).

The deadlock is now FIXED (interp.hip: a placed program's static window is the
CU's own domain, `cu % l2_domains` — the per-CU streams already encode the
exact placement, since slice s runs on CU op.cus[s] whose XCD is its assigned
domain). Token-correct: serve gate answers 'Paris' under PLOW_STATIC_DECODE=1
on the placed blob.

MEASURED (g12b-mergefix blob + PLOW_OCC4/L2HIER objects, 7.2.4 runtime,
48 steps × 3, ctx 4096):

  placed GQ (shipping)          10.63 ms/token
  placed STATIC                 14.61 (3 reps: 14.578 / 14.605 / 14.637)
  unplaced STATIC (§12, old rt) 15.34

Placement hands static ~nothing (−0.7, half of which is the 7.2.4 runtime).
The −4 ms is therefore NOT missing locality — it is the fixed assignment
itself: on the GQ any workgroup of a domain runs the next ready packet, and
that dynamic claiming is doing double duty as the load balancer. A static CU
whose next entry's gate is not yet satisfied idles while its domain-mates
could have run it.

CONSEQUENCE: the mega-mode/levelization route to 4k is now closed by
MEASUREMENT, not by model. Even a perfect −2 ms level pass from 14.61 lands at
~12.6 — 2.8 ms above vLLM's 9.81. Every static-only lever (GQ_BATCH lookahead,
op-115 QKV fusion, HN_SPLIT) starts from a floor that placement cannot rescue.
The 4k gap (10.63 vs 9.81) remains bounded below by the GQ claim/gate protocol
+ the fp8 weight stream, and closing it requires a scheduler that keeps dynamic
claiming while shedding protocol cost — the interpreter redesign, priced and
out of scope for kernel/tiling/wave-segmentation levers.

## 17. Long-context showdown: the ≥8k win region, measured same-session (2026-08-07)

Same protocol as §15 (vLLM 0.26 server resident on GPU 0 at util 0.35,
strict alternation vLLM→plow per round), extended past 8k with the 64k
best-config blob (g12b-64k-mergefix: PLOW_FP8/W8A8/L2_PLACE/FP8_HEAD/FUSE_HNR,
--max-ctx 65536, serve-gated 'Paris') on the 8d79a69 objects:

  ctx     vLLM 0.26 TPOT     plow ms/token     plow advantage
  8192    10.933 (§15)       10.72             +1.9%
  16384   11.68 / 11.67      10.870 / 10.873   +6.9%  (2/2 rounds separated)
  32768   14.17 / 14.10      11.325 / 11.315   +20%   (2/2 rounds separated)
  65408   16.12              12.292            +24%

plow's sliding-window curve is nearly flat (10.72 -> 12.29 over 8x the
context; only the 8 full-attention layers scale), while vLLM's TPOT climbs
~15-20% per doubling. The crossover sits just under 8k and the margin widens
monotonically — the win region is "8k and everything above", per the
standing goal's reframing to post-8k contexts. 4k remains vLLM's (§16 and
the floor decomposition close it for this interpreter).

Side data: vLLM's warm-prefix TTFT at 16k/32k is 109/186 ms vs cold 1859/2206;
plow prefill TTFT remains unmeasured on this blob (task #7).

## 18. Prefill TTFT + served-vs-kernel TPOT: the honest instrument table (2026-08-07)

Asked directly: is plow beating both prefill and decode? NO — decode only, and
only from ~16k once BOTH sides are measured through the same serving stack.
`vllm bench serve --backend openai-chat` against `plowrt serve`
(g12b-64k-mergefix), same client/flags as every vLLM row:

  DECODE (TPOT ms/token)         PREFILL (TTFT ms, cold)
  ctx   vLLM   plow-bench plow-served   vLLM-cold  plow    ratio
  4k    9.81   10.63      10.95         ~500(est)  644     ~1.3x slower
  8k    10.933 10.72      11.08         --         1285    --
  16k   11.67  10.87      11.31 WIN     1859       2807    1.51x slower
  32k   14.10  11.32      11.81 WIN     2206       6750    3.06x slower

Two corrections this table forces on earlier claims:

1. THE SERVING STACK COSTS ~0.35-0.5 ms/token (11.08 served vs 10.72 bench at
   8k): sampling + HTTP streaming + host loop. The 8k "win" of §15 was plow's
   KERNEL against vLLM's SERVER; server-vs-server the crossover is ~16k
   (11.31 vs 11.67, +3%), decisive by 32k (+16%) and beyond. §15/§17 bench
   numbers stand as kernel measurements; this row is the end-to-end truth.

2. PREFILL LOSES EVERYWHERE AND WIDENS: 1024-token chunk cap = 32 sequential
   launches at 32k, the 8 full-attention layers scale O(n^2), and the
   single-buffered CDNA3 prefill GEMM runs ~163 TF/s against vLLM's ~806.
   vLLM's prefix cache adds a ~20x repeat-prompt advantage (109/186 ms warm
   at 16k/32k) that plow has no counterpart for. Task #7 holds the attack
   order: serving overhead first (0.4 ms is 2x the 8k decode margin), then
   prefill GEMM rate, chunk ladder, prefix cache.

## 19. RETRACTION: the garbage-KV bench flattered decode by ~0.15–0.23 ms; the 8k "win" does not survive a real prefill

`amd-bench` without `--prompt` decodes over KV NOBODY WROTE (its own --help
says the tokens are meaningless; §instrument-notes already banned it for
CORRECTNESS). This section bans it for TIMING too: on the same blob + objects
+ ctx, 48 steps at 8k measure

  garbage KV (no prefill)   10.72 ms/token   (the number §15/§17 quoted)
  REAL KV  (--prompt 8150 real ids)  10.956
  serve GPU drain (real KV, dstep)   10.87
  served end-to-end (vllm bench client)  11.03

The ~0.15–0.23 ms is not host overhead — the dstep breakdown pins host phases
at ~106 µs and the serve DRAIN alone (10.87) exceeds the garbage-KV step
(10.72). Mechanism unproven; all-zero K/V rows toggle far fewer bits through
the dot/exp/accumulate path, consistent with a dynamic-power/clock effect.
Whatever the cause, the honest instrument is a REAL prefill.

CONSEQUENCES:
* The §15 8k kernel win (10.845 vs 10.933) is RETRACTED — real-KV decode at 8k
  is ~10.95, a tie with vLLM's SERVED 10.933 at best (and vLLM's number pays
  its serving stack; plow served is 11.03, behind by ~0.1).
* 4k moves further out (real-KV kernel ~10.8+ vs 9.81).
* The §17 LONG-CONTEXT WINS STAND: the served rows are real-KV by
  construction — 16k 11.31 vs 11.67 (+3%), 32k 11.81 vs 14.10 (+16%), and the
  64k kernel row re-checked served-side would still clear 16.12 by ~20%.
* Every historical ms/token in this directory measured via promptless
  amd-bench carries the same ~+0.2 ms correction before comparison to any
  served number.

Honest verdict against "beat vLLM at 4k and 8k": NEITHER holds under honest
instruments. The real crossover is between 8k and 16k, server-vs-server.

### §19 addendum: the knob surface re-swept on REAL KV — shipped values re-confirmed

The retraction invalidated the instrument every §13 knob was chosen on, so the
key knobs were re-swept with `--prompt` (real 8k prefill, 48 steps × 2, 16k-cap
blob, 8d79a69 objects):

  ns16 + hnr (shipped)  10.905 / 10.939   <- still the optimum
  ns24                  10.941 / 10.994
  no-hnr                11.066 / 10.958
  ns12                  11.156 / 11.147
  ns8                   11.381 / 11.314

Same ordering as the garbage-KV sweeps — the flattering instrument shifted the
LEVEL ~0.2 ms but not the RANKING, so no kernel recovery was hiding in the
retune. Terminal 8k arithmetic, served-vs-served: GPU tick ~10.92 + ~47 µs
exposed host + ~50 µs client ≈ 11.02 against vLLM's 10.933; the identified
host trims (device-side pos patch −19 µs, skip redundant seed −8, pinned id
readback −10, detok offload −6) close at most half the gap. 8k needs kernel
−60 µs on top of everything, which the measured floor denies. The 4k/8k goal
is closed at audit grade; the win region is 16k and above.

## 20. Flash→merge in-kernel fold (PLOW_FUSE_MERGE): measured −48 packets, +0.30 ms — the consumer-gate corollary, third confirmation

The last unfused candidate on the decode chain: the last-arriving split
workgroup of each (batch, head-group) merges the partials inside
d_flash_decode's epilogue ([MERGE-FOLD]) and the FlashMerge packet is never
emitted (383 → 335 packets/token). Rendezvous via a self-cleaning
per-head-group counter tensor (release fetch_add per arrival, ONE agent
acquire in the merger, reset before the packet's own completion signal).
Token-correct (serve gate: 'Paris'; primes exact); registers untouched
(104 VGPR / 0 spill).

MEASURED (real-KV --prompt, 8k, 48 steps × 3, interleaved, same objects):

  merge fold ON    11.208 / 11.230 / 11.253
  baseline (hnr)   10.933 / 10.932 / 10.927     -> fold is +0.30 ms, 3/3

The NRN-fold pricing (~4 µs × 48 deleted serial packets ≈ −0.19) did not
transfer, and the reason is §L1-WIDENING's corollary a third time: o_proj's
dense gate previously opened when the NARROW merge op (8–32 WGs, fine-mapped
onto exactly its producer slices) finished; with the fold it opens at the MAX
over all 304 flash workgroups, one of which now carries the serialized
GF-head merge as a tail. A deleted packet is only cheap when its consumer's
effective gate does not widen — the NRN folds deleted packets whose consumers
already waited on the same producers; this one substituted a wide packet's
straggler tail for a narrow one's.

The machinery stays in-tree, env-gated OFF (PLOW_FUSE_MERGE=1 + PLOW_FUSE_HNR=1),
same convention as op-115: the trade plausibly flips where fine deps do not
exist (static scheduler). Do not re-try on the global queue; re-read this.

With this, every fusion candidate on the Gemma-4 decode chain has a landed win
or a measured tombstone. The 4k/8k stopping condition remains closed (§19).
