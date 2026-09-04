# Kimi-K3 on 8x MI355X: plan v3 to beat vLLM 0.28 by a margin on all metrics (2026-09-04)

Supersedes v2 (`plans/k3-beat-vllm-0.28-v2.md`); v1 (`plans/k3-beat-vllm-0.28.md`, 1081 lines)
remains the execution log. Branch `codex/amd-agent-harness` at `15db91c`; the three sibling
branches (`-integration`, `decode-grouped-moe`, `gfx950-prefill-k3`) are behind HEAD with no unique
work. Goal (session): beat vLLM on every metric with a margin, using Plow's architecture advantages,
accepting small architecture changes where the gain is large.

---

## 1. Where we stand (published 2026-09-04, campaign `k3-showdown-c1-fe871e6-final`)

| metric | vLLM 0.28 | Plow | gap | **target (10% margin)** | needed |
|---|---:|---:|---:|---:|---:|
| median TTFT | 567.74 ms | 1271.86 ms | 2.24x | ≤ 510 ms | −760 ms |
| median TPOT | 20.81 ms | 28.53 ms | 1.37x | ≤ 18.7 ms | −9.8 ms |
| P99 ITL | 20.97 ms | 28.60 ms | 1.36x | ≤ 18.9 ms | — |
| output tok/s | 46.86 | 33.63 | 1.39x | ≥ 51.5 | follows from TPOT+TTFT |
| median E2E | 21.85 s | 30.45 s | 1.39x | ≤ 19.7 s | — |
| numerical quality | vLLM repeat floor | Plow absorbed is 4.1-5.7x outside vLLM's own repeat floor | — | within 2x floor, top-1 ≥ 99.5% teacher-forced | Q0 |

Source: `perf-data/kimi-k3-plowrt-mi355x-baseline.md:31-50` (3 folds x 10 requests, exact ids,
`--amd-tp-no-audit`, pairing hash `0x1df8ef184df9a71c`), `perf-data/kimi-k3-vllm-mi355x-c1.json`.
Engine-side post-A4-reuse 8192→1 sizing is 1280.7 ms (`gfx950-moe-stage1-a4-reuse-20260904.md:60`),
so the served cell already includes A4 reuse. "All metrics" includes quality: Plow cannot claim a
win while its whole-model logits sit 4-5x outside vLLM's repeat floor (`mi355x-moe-deterministic-tree-20260904.md:74-77`).

Since v1 (09-03): TTFT 3763 → 1272, TPOT 63.2 → 28.5. Fourteen promotions, all exact and default-on
(v2 §1.2 plus A4 reuse −88 ms). Nine mechanisms rejected with numbers (v2 §1.3 plus §3 below).

### 1.1 Attribution (current default; `kimi-k3-mi355x-current-attribution-20260904.md`, stage-1 adjusted for A4 reuse)

Prefill, per-segment all-rank timing, 925 prefill segments / 649 graph-derived phases:

| route / family | ms | note |
|---|---:|---|
| primary interpreter (186 segs) | ~700 | 256 VGPR, 1,348 B private, VGPR+SGPR spills |
| ↳ dense GEMM (GemmWide 146, GemmC5 83) | 243 | ~34% of bf16 peak |
| ↳ XReduce2 (RS 70.7 + AG 106.9) | 178 | **focused spill-free object: 17.1 ms** for the same calls (10x) |
| ↳ AttnRes | 89 | bf16-mix semantics differ from vLLM's f32-mix |
| ↳ routing/norm/other | 97 | |
| lean MoE stage-1 (A4 reuse) | ~125-160 | AITER exact-shape lower bound 137 ms; schedule axis closed at 1.17% |
| raw KDA Wu + Carry | 195 | Carry 152 (2.2 ms/layer); schedule levers closed (all slower) |
| raw MLA V2 (TR16) | ~75 | absorbed; materialized object 0.35 ms/layer exists, blocked on quality |
| lean stage-2 / intra / combine | 71 / 42 / 39 | at parity |
| host | 14 | |

Decode, device envelope 28.09 ms:

| family | ms | note |
|---|---:|---|
| dense GEMV | 11.34 | 468 `b=256` GEMVs 5.9 ms; latency-shaped |
| grouped MoE | 5.39 | opt-in standalone segment measured −0.72 ms/token; isolated ceiling −3.0 |
| TP reductions | 4.59 | 278 x 16.5 µs; **focused object 0.98 µs** per 14 KiB collective |
| AttnRes | 3.29 | 187 x 17.6 µs |
| KDA / MLA | 1.66 / 1.55 | MLA already in paired segments (−8.1 promoted) |
| protocol gate | ~2.3 | |

---

## 2. The four constraints, dispositioned

### C1. EP cannot ship with dual layouts (+224.52 GiB/GPU)
- Evidence: `gfx950-moe-ep-prefill-design-20260904.md:108-124`; `gfx950-moe-2d-layout-20260904.md:19-29, 41-68`.
- Facts: EP8 prefill needs whole experts per rank; decode needs 1/8-I slices of every expert; the two
  are not views of one allocation, and phase remapping would move 7/8 of 168 GiB per transition.
  The only single-resident forms are `EP x TPexpert = 8` canonical layouts (224.52 GiB, same as
  TP8). EP2xTP4: −63.6 ms TTFT, **+0.396 ms/token decode** (max routes per half 9.56 vs 8);
  EP4xTP2: −105.8 ms, +0.600 ms/token.
- **Verdict for the 8192→1024 C1 contract: EP is net-negative.** EP2: −64 ms + 1023 x 0.396 =
  +341 ms E2E; even composed with the deterministic tree (−0.44 ms/token) it is break-even on
  decode and only −64 ms TTFT. Weight virtualization over xGMI is dead by bytes (7/8 x 1.97 GB x 92
  = 159 GB per prefill at ≤ 0.5 TB/s inbound = 300+ ms). **Drop EP from this plan**; keep the
  default-off generic route for prefill-heavy contracts (short outputs) where −64..−106 ms pays.
- What replaces its 132 ms projection: stage-1 is already near the AITER exact-shape bound after A4
  reuse; the remaining MoE prefill lever is boundary/phase fusion (P2), not K depth.

### C2. Decode fixed-order MoE accumulation blocks profitable phase fusion
- Evidence: `mi355x-moe-down-combine-phase-20260904.md` (three exact-order arms +0.83/+2.98/+4.05
  ms/token), `mi355x-moe-deterministic-tree-20260904.md` (tree −0.445 ms/token < 0.5 gate; f32
  1-ULP, **0/3584 BF16 output words differ**; ordinary DOWN alone is 8.82 µs so a free combine's
  ceiling is 0.486 ms/token).
- **Architecture change adopted (small, large downstream gain): a compiler-defined reduction
  contract.** The combine contract becomes "compiler-fixed balanced tree over routed slots, arrival-
  order independent, all leaves of a 64-row tile on one XCD" with the oracle "f32 relL2 ≤ 1e-6 and
  BF16 output identical on the nonuniform-route oracle". This is a determinism contract, not a
  tolerance contract: outputs stay bit-reproducible run to run, and the BF16 boundary is unchanged.
  It permits reassociation inside the tree only.
- Under that contract the next candidate is the one the report names: carry the balanced root into
  the owning-rank one-shot `XReduce` epilogue (removes DOWN→COMBINE and COMBINE→XREDUCE edges),
  gate ≥ 0.75 ms/token, zero-spill (D2). A wider persistent expert/slot phase becomes possible
  later (D7).
- Explicit tolerance (reassociation beyond the tree) is **not** adopted until Q0 gives a usable
  whole-model oracle.

### C3. Materialized MLA must keep the f32 attention mix live through RMSNorm in one phase
- Evidence: `runtime/bench/amd/mla_materialized_prefill/README.md:56-69, 253-263`;
  `perf-data/mi355x-attnres-f32-mix-norm-20260904.md` (commit d32289e).
- Facts: materialized vs absorbed attention differs at 1 BF16 ULP (max 0.0039, RMSE 8.7e-5); the
  post-attention residual seam amplifies it to 0.047 because production `d_attn_res` rounds the
  softmax mix to BF16 before the output RMSNorm, while vLLM applies RMSNorm to the unrounded f32
  mix with separate epsilons. The f32-mix specialist fixes semantics (seam relL2 2.9e-3 → 5e-9)
  but is only 7.25% faster at the prefill boundary and keeps 42 SGPR spills at T8192; a forced
  one-shot specialization spilled 160-240 B/thread.
- **Architecture change adopted: AttnRes semantics move to vLLM's f32-mix + separate epsilons.**
  Rationale: it is required for quality parity (Q0) regardless of MLA; each of the 186 seams
  currently injects ~2.9e-3 relL2, which is the most plausible source of Plow's 4-5x excess over
  vLLM's repeat floor. The implementation target is a phase object (attention output → f32 mix →
  RMSNorm in one phase, no BF16 round-trip) that meets the spill gate; the interpreter arm is the
  fallback with the known spills. Materialized MLA (−130 ms TTFT, object ready) is promoted only
  after this and Q0 pass (P5).

### C4. Phase-object TP8 validation is blocked by incomplete hybrid-model emission
- Evidence: `mi355x-phase-chain-replay-20260904.md:74-79`; `kimi-k3-hybrid-phase-emit-20260904.md`
  (commit 5c2614c): production hybrid `devblob` emitter is now default-on and a full-model
  **manifest-only** gate emitted 93 layers, 925/649 prefill segments/phases, 49 decode phases.
  What is still missing is an **exact executable TP8 candidate** carrying phase objects (the
  scheduled-packet HF planner has no lowering for `Conv1dDepthwise`, `LinearAttention`, `SituGlu`,
  `BlockResidual`; the production emitter must be the one used, and the XReduce-only phase route
  (`PLOW_PHASE_OBJECTS=1`, 827e7f0/d5137a2) has correctness and host-publication evidence only).
- **This is the single prerequisite for both tiers** (P1, P2, D3, D5 all need it). Week-1 item.
- AQL chain prebuild is transport only (device 1.464 → 1.466 µs/packet); the device win must come
  from spill-free phase bodies, measured per family.

---

## 3. Plow's architecture advantages to lean on (and what they are worth)

| advantage | evidence | how the plan uses it |
|---|---|---|
| Exact strict-rank-order collectives 7-21x faster than AITER focused | `kimi-k3-mi355x-aiter-xreduce-parity-20260904.md:43-62` | P1/D5: put that object in a spill-free phase; the in-network 10x loss is the interpreter's register/spill envelope, not the algorithm |
| Single 8192 chunk prefill (vLLM pays 2 x 4096 with context-chunk attention and MoE fill twice) | `k3-prefill-attribution.md:103` | keep; do not add 32K rung |
| Lean per-family objects at AITER parity (stage-2 0.18 vs 0.17 ms; stage-1 at ~1.1x AITER bound after A4 reuse) | v1 §7.0, `gfx950-moe-stage1-schedule-screen-20260904.md:122-141` | done; remaining MoE lever is seams |
| Graph-derived phase partition + `dispatch_chains` refusal contract (compiler chooses boundaries by register/spill certificates) | `mi355x-phase-chain-replay-20260904.md:9-19` | the mechanism for every remaining interpreter family |
| Deterministic, reproducible outputs (32 reruns bit-identical) | tree report `:49-50` | quality argument vs vLLM's own nonzero repeat floor |
| Per-XCD device queue windows, HIER two-level signal | 83120a7, `interp.hip:3532-3560` | D4 sc1 sizing |

---

## 4. Quality gate (Q0) — required for "all metrics"

- **Facts**: pinned vLLM has a nonzero repeat floor (full-row relL2 max 0.099, head64 0.024,
  top-64 overlap ≥ 0.906, occasional argmax flips across 65 requests). Plow absorbed and
  materialized both sit 4.1-5.7x outside a 2x-floor acceptance on all 65 exact-history rows
  (`mla_materialized_prefill/README.md`, tree report `:74-77`). Tooling exists:
  `scripts/vllm_logit_oracle.py`, `logit_quality_compare.py`, `plow_logit_manifest.py`,
  `amd-bench --dump-logits`, `openai_correctness_gate.py`.
- **Do**: (1) bisect the whole-model divergence by seam: capture rank-0 tensors at the first KDA
  block, first MLA block, first MoE block on identical token ids (`PLOW_PF_CAPTURE`), compare to
  vLLM's captured tensors at the same boundaries (`tensor_boundary_compare.py`); (2) implement C3's
  f32-mix AttnRes contract and re-run; (3) define acceptance: teacher-forced top-1 ≥ 99.5%, full-row
  relL2 within 2x vLLM's repeat floor, GSM8K within noise.
- **Gate**: Plow within 2x of the vLLM repeat floor on ≥ 90% of rows; no severe argmax flips.
- Effort 3-5 days; unblocks P5 and any future reassociation tolerance.

---

## 5. Prefill plan (Tier 1: 1272 → ≤ 510 ms)

Budget: vLLM ≈ 6.1 ms/layer. Plow needs roughly MoE ≤ 220, dense ≤ 150, KDA ≤ 110, MLA ≤ 40,
collectives ≤ 80, AttnRes ≤ 40, other ≤ 50, host ≤ 10 → ~700 ms even at those figures; ≤ 510
requires collectives overlapped with compute and KDA carry at ~1 ms/layer. Order by (ms saved /
day) with prerequisites.

### P1. Spill-free XReduce phase object through the hybrid emitter — prerequisite C4; 3-5 days; −90..−140 ms
- **Facts**: RS 70.7 + AG 106.9 = 177.6 ms in-network vs 17.1 ms focused over the same 92/94/92
  calls; AITER is 7x slower than the focused object, so no external reference is faster. The loss
  is the primary interpreter's 256 VGPR / 1,348 B private / VGPR+SGPR spills around the collective
  body. Priced cost: +464 segment launches ≈ 3.0 ms.
- **Do**: emit the exact TP8 candidate with the production hybrid emitter; enable
  `PLOW_PHASE_OBJECTS=1` XReduce-only route; A/B 3 order-alternated 8192→1 folds; per-segment
  timing to confirm the collective body. Keep strict rank order (U2 is order-preserving).
- **Gate**: bit-exact; XReduce2 ≤ 80 ms; zero spill/private on the phase object; TPOT neutral.

### P2. Phase objects for the remaining interpreter families — after P1; 2-3 weeks; −150..−250 ms
- Order by family size: dense GEMM 243 (a 4-wave register-pipelined GEMM object at ≥ 55% of peak
  is worth −80..−120), AttnRes 89 (with C3's f32-mix semantics; −20..−40 plus quality),
  routing/norm/other 97 (−30..−50). Each phase fuses producer/consumer seams inside the phase so the
  925-segment count falls; boundary cost is priced at 0.49 ms/ordered boundary.
- **Gate per family**: exact (or C3 contract for AttnRes); family time ≤ 60% of today; occupancy ≥ 2;
  zero spill regression vs the ordinary object (`scripts/gfx950_objects.py` refusal contract).

### P3. KDA carry register-resident state (AITER K5 layout + `ds_read_b64_tr_b16`) — 5-8 days; −50..−90 ms
- **Facts**: 152 ms = 2.2 ms/layer; every wave/V-tile/schedule variant measured slower (+6.6..+55%),
  V8/V32 at WG256 spill SGPRs; the report names the K5 register-resident state layout as the
  remaining candidate (`kda_carry_schedule/README.md:64-72`). FLA's fwd_h keeps state in
  registers per (batch, head) and lands well under 1.5 ms/layer.
- **Do**: 4-wave lean object, 16x128 f32 state in VGPR (32/lane), chunk loop with transposed LDS
  reads for the k-factor, strict recurrence order preserved; route via the existing family route.
- **Gate**: standalone ≤ 1.3 ms/layer; state and outputs bit-exact; zero spill; network KDA family ≤ 120 ms.

### P4. Collective/compute overlap — after P1; 1-day screen then 5 days; −40..−80 ms
- After P1 the collective is xGMI-bound (≈ 0.4-0.6 ms per 117 MB full collective). Screen in-network
  `PLOW_XR_CUS` 64/128 (commit d86969c honors it) — the isolated sweep was MALL-warm and is not
  the in-network floor. If 128 WGs cost ≤ 10% in-network, slice producer GEMM + collective into 2
  token halves so GEMM(half 2) overlaps RS/AG(half 1). Bit-exact by construction.

### P5. Materialized MLA prefill — after Q0 + C3; 3 days; −100..−130 ms
- Object ready (0.353 ms/layer at T8192, byte-identical flat vs 3D grid); promotion under the Q0
  gate with continuation/packed-cache coverage (currently fails closed for continuation chunks).

### P6. Dense c8 tile default — 2 folds; −11 ms
- Two more accounting-fixed alternating folds; promote only if E2E is non-regressing.

### Prefill outcome model

| block | now | P1 | +P2 | +P3, P4 | +P5, P6 |
|---|---:|---:|---:|---:|---:|
| MoE (stage-1/2/combine/route) | ~300 | 300 | 260 | 260 | 260 |
| XReduce2 | 178 | 70 | 70 | 40 | 40 |
| dense | 243 | 243 | 150 | 150 | 139 |
| KDA | 237 | 237 | 237 | 160 | 160 |
| MLA | 75 | 75 | 75 | 75 | 35 |
| AttnRes + other + host | 200 | 200 | 130 | 130 | 130 |
| boundaries (implicit in above) | ~300 | ~290 | ~200 | ~180 | ~170 |
| **TTFT** | **1272** | **~1160** | **~920** | **~810** | **~760** |

**Honest verdict**: measured mechanisms reach ~750-800 ms served TTFT (1.35x vLLM), not 510. The
gap to the target is boundaries (~170 ms at ~350 segments) and KDA (160 vs vLLM's ~90). Reaching
≤ 510 needs P2 to collapse each layer into ≤ 3 phases (boundaries ≤ 60 ms) and P3 at ≤ 1 ms/layer.
Report Tier 1 progress per block; do not claim it until a served 3-fold cell shows it.

---

## 6. Decode plan (Tier 2: 28.53 → ≤ 18.7 ms)

### D1. Promote the grouped-MoE decode segment with a generic profitability rule — 1-2 days; −0.7 ms
- Qualified (3 folds, −0.716 ms/token, exact) but default-off pending a non-model predicate.
  Rule: enable when the isolated pair body / segmented pair body ratio from the packet's measured
  TuneDB rows exceeds a threshold, and the geometry has an audited spill contract. Rollback flag.

### D2. Deterministic tree carried into the one-shot XReduce epilogue — contract C2; 4-6 days; −0.75..−1.5 ms
- Removes DOWN→COMBINE and COMBINE→XREDUCE edges (2 x 5.7 µs + combine body 5 µs per layer ≈ 1.5 ms
  ceiling). Object must be zero-spill, tree-order compiler-fixed, BF16-identical oracle. Gate ≥ 0.75.

### D3. Ordinary decode object occupancy 3 — measured 2026-09-04; now a GEMV specialist item; −2..−3 ms
- **Measured** (`perf-data/kimi-k3-mi355x-decode-mla-segments-20260903.md:24-28`,
  `kimi-k3-decode-inventory-prune-20260903.md:6-27`): after the MLA split the ordinary decode
  object is still **248 VGPR / 106 SGPR / 84 SGPR spills / 216 B private / occupancy 2**. The
  retained-arm floor is `Gemv` alone at **215 VGPR** (MlaMergeFold 232 is already out; AttnRes 134,
  GemvGlu 127, MoeGroupDownFp8Blk 113, GemvQkvg 110). Occupancy 3 needs ≤ 168 VGPR, so moving
  arms out does not reach it while `Gemv` stays; the mega GEMV body itself must shrink.
- **Do**: a spill-free GEMV specialist object at ≤ 168 VGPR (occupancy 3) covering the 468
  `b=256` GEMV shapes, routed as paired segments like MLA (+1.46 µs AQL per launch, priced), with
  the ordinary object then re-measured for occupancy 3 without `Gemv`. This is the same object D7
  needs; do it once. Keep exact accumulation order per row.
- **Gate**: specialist 0 spills, occupancy ≥ 3; ordinary object ≤ 168 VGPR, 0 VGPR spill; exact
  ids; TPOT −2 ms net of launches.

### D4. HIER on/off sizing + `sc1` scoped stores — 1 + 4 days; −2..−4 ms
- Joint ceiling −5.04 ms; measure HIER on the pruned object first. sc1 needs local/peer scope split
  and every ragged scalar tail converted; stale-word oracle zero over 1000 tokens at TP8.

### D5. Decode XReduce spill-free phase object — after C4; 3 days; −2..−2.5 ms
- In-network 16.5 µs vs focused 0.98 µs per 14 KiB collective; with a ~5.7 µs handoff the phase
  body saves ~9 µs x 278. Same route as P1 at decode shapes.

### D6. Gang-admitted XReduce+AttnRes carrier — 1-2 weeks; −2..−3 ms
- CPU feasibility done (186/278 eligible, deadlock proof, 48 VGPR prototype). Note the cooperative
  grouped-MoE probe deadlocked at > 1 WG/CU: keep the carrier at exactly `n_cu` WGs and reject
  larger grids at load.

### D7. GEMV family phase object + cross-packet weight prefetch screen — after D3/P2; −2..−4 ms
- 468 small GEMVs at 12.6 µs in a spilling object; a spill-free GEMV phase with occupancy ≥ 3 and
  MALL-resident next-layer weights (per-layer weights ≈ 150 MB < 256 MB MALL) is the mechanism.
  Screen on one layer first; D11 showed isolated launch savings do not transfer, so the gate is
  network TPOT only.

### Decode outcome model

| step | ms |
|---|---:|
| now | 28.5 |
| D1 | 27.8 |
| D3 | 25.3 |
| D4 | 22.3 |
| D5 | 20.0 |
| D2 | 19.0 |
| D6 | 16.5 |
| D7 | 14 |

Tier 2 with margin (≤ 18.7) is reachable with D1-D5 landing; D6/D7 are the margin.

---

## 7.0 Live execution status

- **2026-09-04 08:00 — P1/C4 gate in flight.** Artifacts under `/tmp/k3-xr-phase-gate/`: control and
  candidate packets emitted from HEAD `937e41f` with the production hybrid emitter
  (`K3_FULL=1 --max-ctx 16384`), candidate with `PLOW_PHASE_OBJECTS=1` (isolates XReduceTwoShot
  segments); packet-paired gfx950 object sets built via cmake (`PLOW_HSACO_CONFIG`, inventory
  prune + decode MLA segments ON, candidate also `PLOW_XR_WAVE_RS=ON` to produce
  `interp_xreduce_gq.elf`, whose ordinary U2 arm the phase route executes). Stage 2 script
  (`xr_phase_stage2.sh`) runs alternating exact 8192→1 folds c1,p1,p2,c2,c3,p3 then one 8192→256
  pair, `--amd-phase-objects=true` on the candidate, under `gpulease -n 8`. A third object set
  `hsaco-control-nohier` (`PLOW_GATE_HIER=OFF`) is queued for the D4 decode pair.
- 08:05-08:20: control folds c1/c2 = 1262.39 / 1261.64 ms TTFT, checksum `fnv1a64:337f0f290d5ae157`.
  The candidate first refused to run ("AQL chain has 1157 packets but queue capacity is 1024")
  and, after the queue was raised to 4096, overran its reservation ("emitted more than its
  reserved 1157 packets") because the chain counted segments while the A4-reuse stage-1 route
  launches twice (EP align four times). Both fixed on branch `codex/d1-moe-decode-rule`
  (97f3cba, e834351); stage 2b reruns p1,c4,p2,p3,c5,p256,c256 with the fixed binary.
- D1 landed on the same branch (9cd623f): model-neutral `moe_decode_measurement.jsonl` cell
  records + selector charging the measured 10.3 µs handoff; K3 packet byte-identical with no
  records (`f1bf783d…`). Next: publish the two route measurements for the k16/H3584/I384/E896
  cell, then the 3-fold promotion cell.
- **08:50 RESULT — P1 rejected.** Three exact folds: control 1262.3 ms vs candidate 1284.9 ms
  (+22.6 ms); 256-token pair exact and TPOT-neutral (28.69 vs 28.65). Per-segment timing:
  the interpreter family fell 891 → 710 ms and the 278 isolated XReduce segments cost 203.4 ms
  (0.73 ms each), i.e. the spill-free object runs the collective at the same cost as the
  interpreter did. The collective is xGMI-bound (~205 MB inbound per rank per collective at
  ~280 GB/s); the focused 17.1 ms figure was Infinity-Cache-warm. Boundaries cost ~80 µs each
  with segment-major dispatch. Report: `perf-data/mi355x-xreduce-phase-object-gate-20260904.md`.
  Consequence: prefill collectives (~200 ms) are a fabric floor; only overlap (P-C) or fewer
  round trips can move them. Phase objects remain cheap to add where a body win exists.
- **D4 done**: `PLOW_GATE_HIER` on vs off on the pruned object = 28.73 vs 34.66 ms TPOT
  (−5.93 ms, exact). Keep on; size sc1 from a fresh decode trace, not the old ceiling.
  Report: `perf-data/mi355x-gate-hier-decode-sizing-20260904.md`.
- **P-C screen (09:08)**: `PLOW_XR_CUS=128` packet, exact, per-segment timing: interpreter family
  890.7 → 982.4 ms (+92 ms), TTFT 1275.6 → 1370.3 ms. The two-shot collective needs the full 256
  workgroups to reach its fabric floor, so overlap by CU partitioning is out; overlap must come
  from global-queue interleaving of independent packets (overlap agent informed).
- **D1 evidence (09:05)**: control vs standalone grouped-MoE decode, 3 exact folds each:
  28.607 → 27.934 ms TPOT (−0.673, −2.35%); interpreter pair body 34.70 µs (92 trace samples),
  network-derived standalone body 17.09 µs (isolated 16.78) → rule selects Standalone
  (gain 7.3 µs/layer = 21%). Records must be keyed to the shipping build digest: a kernel
  source edit (GLU rung) changed the label and staled all 5,886 GEMM records too (analytical
  tiles) — any kernel edit needs the TuneDB requalification campaign before its packet is gate-valid.
- **D3 closed (agent, branch `codex/d3-gemv-decode-specialist` ea57efb)**: ordinary decode object
  occupancy is pinned at 2 by the 147,512 B LDS arena (one 8-wave WG per CU) and
  `amdgpu_waves_per_eu(2,2)` (`interp.hip:4160`), not by VGPR: `-DPLOW_HAS_GEMV=0` → 228 VGPR,
  +AttnRes → 199, +one more arm → 149-151, all still occupancy 2. Occupancy 3 needs LDS ≤ ~53 KB,
  a WPE change and a grid > n_cu. A spill-free `gemv_decode_gfx950` specialist exists (90 VGPR /
  32 KB LDS / occ 3, bit-identical `d_gemv_t` body, CMake `PLOW_HSACO_GEMV_DECODE`), default-off;
  its only possible win is raw ordered launches replacing gate+convergence, which the grouped-MoE
  handoff (10.3 µs/segment) says is negative for 11 µs GEMVs. Do not route it. The decode-body
  lever is therefore per-op latency (straggler/tail, e.g. GLU UN=7) and packet count, not occupancy.
- **GLU GEMV UN=7 rung (09:22)**: exact, resource-neutral, 28.574 → 28.460 ms TPOT (−0.114,
  −0.40%) over 3 alternating folds. Promote; note any kernel-source change moves the TuneDB
  label and needs the dense-GEMM requalification before a packet from that source is gate-valid.
  Report: `perf-data/mi355x-gemv-glu-un7-20260904.md`.
- **D1 promoted by rule (09:25)**: records published against the emitter label
  `gfx950-e95f0a91a5a3c577`; flag-free emit selects Standalone and is byte-identical to the
  gated packet (`a1f7f6f7…`). Report `perf-data/kimi-k3-mi355x-moe-decode-route-rule-20260904.md`.
  Branch `codex/d1-moe-decode-rule` now carries: queue 4096, exact chain reservation, D1 rule +
  records, GLU UN=7 rung, and four perf-data reports (phase gate, HIER, GLU, D1).
- **P-C closed at packet level (agent, branch `worktree-agent-a60a34ba359c39966` ff3c8f8,
  note `perf-data/mi355x-xreduce-token-slice-pipelining-20260904.md`)**: token-slice pipelining
  of o_proj→XReduce2 is implemented (`PLOW_XR_SLICES=2`, exact, +93 packets, unflagged packet
  byte-identical) but cannot overlap in the current grid: 1 WG/CU from the 147.5 KB LDS arena,
  256 VGPR x 8 waves fills the register file, the collective needs all 256 WGs, and the global
  queue gives a blocked WG nothing else. Expected +2..+8 ms; not gated. Even with free
  concurrency the ceiling is Σmin(GEMM, XR) ≈ 118 ms, not 200. The only in-grid path is a fused
  lean "GEMM band i+1 + reduce-scatter band i" object issuing remote loads inside the GEMM
  workgroups (new kernel program). **Tier 1 consequence**: with collectives at ~200 ms and the
  remaining families at AITER-level, TTFT bottoms near 700-800 ms; ≤ 510 ms requires that fused
  object or a lower-LDS interpreter that admits co-resident work.
- **Stacked decode candidate (09:39)**: D1 route + GLU rung = 28.582 → 27.748 ms TPOT
  (−0.834, −2.92%), exact. Current engine-side stack: TTFT ~1262 ms, TPOT ~27.75 ms.
- **Decode tail attributed (agent, branch `worktree-agent-ae041b79dd38dfaf0`, merged)**: the GQ
  claims whole slices dynamically but slice→rows is static; the GEMV_GLU "straggler" is
  head-of-line blocking (GLU depends only on AttnRes but queues behind gated routed slices; the
  layer-closing XReduce waits on GEMV#130 in 186/186 layers: 24.2 µs/layer = 2.23 ms/token on the
  true critical path). Fix: emit-time `PLOW_GQ_ORDER=asap` window ordering (exact; A/B pending).
  Dynamic row claiming (`PLOW_GV_DYNCLAIM`) measured slower on all 16 shapes (closed).
  New item: `MLA_MERGE_FOLD` runs 48 slices at 21.5 µs and 208 at 1.6 µs (one WG per
  (token, head)x4 at 12 heads) — a parallelism bug worth ~0.45 ms/token on 24 packets.
- **C3 AttnRes f32-mix phase object delivered (agent, branch `worktree-agent-a6ab45125d8a5e8b2`
  bf5a5d0)**: `attn_res_f32mix_gfx950.hip`, 133 VGPR / 63 SGPR / occ 3 / 0 spill / 0 private /
  28.7 KB LDS; matches a CPU port of vLLM's `attn_res.py` to relL2 ≤ 2.1e-7 (production seam
  2.8e-3); T8192 site 0.589 → 0.260 ms (streaming floor 0.221). Route `PLOW_ATTNRES_F32MIX=1`
  default-off, prefill only (decode stays interpreter). TP8 gate pending: expect ~−60 ms TTFT
  minus 186 boundaries; tokens will differ from control by design (semantic change) → gate on
  the Q0 teacher-forced/logit oracle vs vLLM, not on checksum identity.
- **Label fragility**: the tunedb implementation label (`gfx950_gemm_inventory().build().label()`)
  moved on every merge today (870078 → 58fb → e95f → 92cc → fb5b), re-staling the grouped-MoE
  route records each time; the GEMM family key also re-stales on any op_gemm.h edit (requal +
  fill pass ≈ 7 min). Follow-up: key route records on the routed object's own source digest.
- **P2 dense-GEMM phase object closed (agent, branch `codex/p2-gemm-phase` 90c38cd)**: GEMM-only
  bucket object 200 VGPR / 0 spill / occ 2 is only 3-9% faster per packet than the ordinary
  prefill object in isolation (the ordinary object already reaches 45-61% of bf16 peak on the
  K3 M=8192 shapes: 8192x6144x7168 0.712 ms = 1014 TF/s); `PLOW_PHASE_GEMM=1` would add 1,162
  boundaries (+93 ms) for 12-17 ms of body. The in-network 34% is the network regime + boundaries.
  Remaining dense lever: a faster tile body (4-wave register-pipelined), screened in
  `gemm_tile_sweep` first. Report `perf-data/mi355x-gemm-phase-object-screen-20260904.md`.
- **ERRATUM (collective agent, branch `worktree-agent-a3db1e5a3e4fae120` 41e1279)**: isolated
  collective benches scaled `s_memrealtime` by 1 GHz instead of the 100 MHz REFCLK → all isolated
  collective numbers were 10x too small. Corrected: 14 KiB one-shot 9.8 µs (in-network 15.6 =
  hot protocol + ~6 µs last-rank/cold), 112 MiB two-shot 634 µs (in-network ~730). AITER parity:
  1.5-2.1x slower at decode sizes, **0.79-0.86x = faster at 28-112 MiB** → porting AITER's
  prefill data schedule under strict rank order is a ~30 ms TTFT lever. Decode: the one-shot's
  critical path is ~17 dependent fabric ops (8 serialized signal RTTs + poll + inv + 8 read
  RTTs); a tag-in-data collective (`d_xreduce_tagged_mega`, 64 VGPR / 0 spill, exact on the
  strict-order oracle, 1-GPU smoke 2.2 µs) has a ~1 RTT critical path → up to −2.8 ms/token if
  the TP8 microbench + all-rank trace show the protocol floor dominates. Harness fixed; §1.1/§3
  claims resting on the 0.98 µs / 17.1 ms figures are void.
- **Tagged decode collective, TP8 microbench (10:35)**: one-shot 14 KiB 9.77 µs hot / 11.36 cold;
  tagged 3.72 hot / 4.75 cold (7 KiB: 11.03 → 4.24 cold); strict-order bit-exact everywhere.
  ~−6.5 µs x 278 ≈ −1.8 ms/token if the in-network floor follows (all-rank trace pending).
  Agent proceeding to a flagged production decode arm.
- **ASAP order qualified (10:29)**: −0.210 ms/token, −6 ms TTFT, exact; in the showdown bundle
  (`/tmp/k3-showdown-stack`, Lean verified+oracle, 7650/7650 measured, packet `c2040cf8…`).
- **All-rank decode trace (10:32)**: XReduce protocol floor (min over ranks) 14.5 µs (b=14,
  14 KiB) / 12.9 µs (b=7); last-rank wait ~1 µs mean → the 15.6 µs body is protocol, not skew.
  Tagged collective (3.7-4.7 µs isolated at TP8) projects ~−9 µs x 278 ≈ −2.5 ms/token.
- **10:36 served showdown started** (`k3-showdown-c1-stack-20260904`, bundle
  `/tmp/k3-showdown-stack`: D1 standalone route + GLU rung + requalified TuneDB + ASAP order,
  Lean verified+oracle, packet `c2040cf8…`, 119 objects, pinned plowrt with queue fixes;
  same vLLM 0.28 image, 3 alternating rounds x 10 requests, 8192→1024, C1).
- **P3 KDA carry register-state delivered (agent, branch `worktree-agent-ac19cf50dcf202448`,
  merged 0f66fda)**: attribution shows the shipping carry is ~40 serialized memory round trips
  per chunk (exec-masked bounds branches around loads + `vmcnt(0)`), not MFMA/LDS/barriers.
  4-wave WG256 carry with the f32 state in MFMA accumulators, transposed LDS factor reads:
  1.916 → 0.726 ms/layer (2.64x), bit-exact (outputs, final state, tail chunk), 229 VGPR / 0
  spill / 0 private. Route `PLOW_KDA_CARRY_REGSTATE=1` default-off. Gate queued (stage 15):
  expect ~−80 ms TTFT. Follow-ups: feed the key-factor hi/lo pair (~0.4 ms/layer more), hw
  bf16 convert (3.36x, NaN payloads only).
- **MLA merge-fold decode (agent, branch `worktree-agent-a19227e3c64e77603` 37c141d)**: the 21 µs
  was the (m,l) merge pass (128 serial scalar round trips), not fill; rewrite 22.8 → 10.3 µs cold,
  bit-exact, but decode-MLA specialist spills 6 → 7 VGPR, private 28 → 32 B. Object sets
  `/tmp/k3-mf/hsaco{,-dvt16,-dvt32}`; A/B queued (stage 16); ~0.3 ms/token at stake.
- **Showdown attempt 1 aborted**: harness required served name `k3-gfx950-farm`, Plow serves the
  packet slug `kimi-k3`; rerun as `k3-showdown-c1-stack-20260904b` with MODEL_ID=kimi-k3.
- **Tagged decode collective integrated (agent commit 3629231, `-DPLOW_XR_TAGGED=1` default OFF)**:
  copy-publish tagged 8 B words, 8 system-scope peer loads in flight, strict rank order, relaxed
  xctr bump keeps audits; four 20 KiB tag slots in `PeerLayout` (per-token zeroing 12 → 92 KiB);
  PlowProgram unchanged, flag-off objects byte-identical. Decode object 248 VGPR / 0 VGPR spill /
  SGPR spill 84 → 110 / occ 2. A/B queued (stage 17) with `/tmp/xr-tag-hsaco/{control,tagged}-set`.
- GPU queue after the showdown: stage 14 AttnRes f32-mix (TTFT folds + GSM8K), stage 15 KDA carry
  regstate, stage 16 merge-fold, stage 17 tagged collective.
- Gate: candidate folds byte-exact vs control; XReduce2 body ≤ 80 ms in a per-segment trace;
  TTFT ≥ −80 ms net of the +464 launches; TPOT neutral on the 256 pair.

## 7. Execution order

Week 1 (all start now, in parallel where the lease allows):
1. **C4**: exact executable TP8 candidate through the hybrid emitter with the XReduce-only phase
   route; P1 A/B. This gates everything else.
2. **Q0 step 1**: seam bisection of the whole-model divergence on identical token ids.
3. **D1** generic rule + promotion; **D3** metadata read; **D4** HIER on/off.
4. **P3** carry register-resident screen (single kernel, no graph build).

Week 2: P1 promotion cell; C3 f32-mix AttnRes phase object; D3 act; D4 sc1.
Week 3: P2 dense GEMM phase; D5; Q0 gate rerun after C3.
Week 4: P2 AttnRes/routing phases; D2 tree-into-XReduce; P4 overlap screen.
Week 5-6: P5 materialized MLA under Q0; D6; P3 network gate.
Week 7: served 3-fold publication cell (both tiers + quality reported separately); D7.

Every promotion: 3 order-alternated exact folds, all-metrics non-regressing, rollback flag, dated
`perf-data/` report + JSON with digests and active routes. Served numbers only through
`perf-data/tools/bringup_showdown.sh` (`INPUT_MAP=8192 OUTLEN_MAP=default=1024
CONCURRENCY_MAP=default=1 PROMPT_MAP=default=10 WARMUP_MAP=default=1 ROUNDS=3 TP=8`).

---

## 8. Closed (cite before reopening)
v1/v2 lists plus: EP8/EP4/EP2 for this contract (C1), all fixed-order DOWN→COMBINE phase arms
(+0.83/+2.98/+4.05 ms/token), deterministic tree standalone (−0.445 < 0.5), cooperative grouped-MoE
handoff (3.23x slower; > 1 WG/CU deadlocks), KDA carry wave/V-tile schedules (all slower),
MoE stage-1 schedule/tile axis (1.17%; BK64 invalid), AITER collective port (7-21x slower, weaker
order contract), f32-mix AttnRes as interpreter arm (7.25%, 42 SGPR spills), `xr_cus` < 256
isolated, XReduce→AttnRes fusion (+91.7 ms), wave-per-peer RS (+3.6 ms), lm_head sharding,
GemvQkv(Nv=0), D9 epilogues, c8 as TTFT-only promotion.

## 9. Critical files
- `crates/devgen/src/manifest.rs` (`dispatch_chains`, phase refusal contract), `crates/devgen/src/k3.rs` (hybrid emit, `K3_FULL`), `scripts/gfx950_objects.py`
- `crates/plowrt/src/exec/amd.rs` (`PLOW_PHASE_OBJECTS`, EP route + capacity refusal, `prefill_segment_family`), `crates/plowrt/src/exec/amd_tp.rs`
- `runtime/amd/op_collective.h` (two-shot, U2, phase trace), `runtime/amd/op_k3.h` (AttnRes), `runtime/amd/op_kda.h` (carry), `runtime/amd/op_moe.h`, `runtime/amd/moe_decode_grouped_mxfp4.hip`
- `runtime/bench/amd/{kda_carry_schedule,moe_ep_boundary,mla_materialized_prefill,lean_moe_stage1_ref}/`
- `scripts/{vllm_logit_oracle.py,logit_quality_compare.py,tensor_boundary_compare.py,openai_correctness_gate.py,xreduce_phase_report.py,k3_trace_report.py}`
- `perf-data/tools/bringup_showdown.sh`, `perf-data/kimi-k3-plowrt-mi355x-baseline.md`

- **14:50 gates complete (stages 14-17, all 3 alternating folds)**: KDA carry regstate 1144 vs
  1256 ms TTFT (−112, exact); AttnRes f32-mix 1208.5 vs 1256.5 (−48, GSM8K 122 vs 124/200);
  tagged decode collective 26.44 vs 28.57 ms TPOT (−2.12, exact); MLA merge-fold 28.22 vs
  28.55 (−0.31, exact; DVT16/32 equal). All promoted to default (bb8cd21) with ASAP order.
  Served showdown d (decode stack only): 1284.84 / 27.54 / 34.76 vs vLLM 566.02 / 20.85 / 46.67.
  Stack-2 bundle (/tmp/k3-stack2) + exact-arm gate + showdown queued. Projection: TTFT ~1120,
  TPOT ~25.1.
- **TTFT lever research (agent, 12:55)**: boundaries are 5-9 µs (~5 ms total), not ~300 ms; the
  idle is structural (KDA carry 96/256 WGs 94 ms, MLA causal tail 25, MoE align 30, GEMM tails
  15). New lever: sequence-parallel seams (run AttnRes/router/top-k/latent on RS-owned T/8
  rows, all-gather results; −110..−140 ms, exact by construction, 8-12 d). Others: materialized
  MLA −100..−115 under Q0, router top-k body −40, AITER-rate collective schedule −30..−60,
  KDA key-factor/Wu chain −60..−80, MoE stage-1/2 bodies −55..−80, horizontal GEMM fusion
  −25..−40, host −10..−20, MOE_ALIGN_PAR −8.6 (exact, ready). Bottom line: ~600-680 ms engine
  reachable; ≤ 510 needs a second HSA queue + low-LDS interpreter (overlap), GEMM ≥ 1.3 PF/s,
  or relaxed MoE-combine exactness.
- **15:24 stack-2 served (2 folds, campaign `k3-showdown-c1-stack2-20260904`)**: Plow 1113.26 ms
  TTFT / 25.25 ms TPOT / 38.01 tok/s vs vLLM 566.36 / 20.88 / 46.70 (1.97x / 1.21x / 1.23x).
  Exact-arm packet (f32-mix off) bit-identical to the pre-stack control at 1144.2 / 25.25.
  Published 73c71c0; `codex/amd-agent-harness` fast-forwards onto main (merge-readiness agent:
  tests/fmt fixed, 59 verbose perf-data reports folded into
  `perf-data/kimi-k3-mi355x-campaign-summary-20260904.md`). Remaining gap: 547 ms TTFT, 4.4
  ms/token. Next: sequence-parallel seams, materialized MLA under Q0, MOE_ALIGN_PAR (+c8),
  tagged-collective follow-ups for decode (D2 tree-into-XReduce epilogue, D5), second HSA queue.
