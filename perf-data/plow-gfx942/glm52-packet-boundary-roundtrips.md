# GLM-5.2 decode: the serial per-packet boundary — round trips, poll granularity, and where the exposed time actually is

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **PLOW-ARCHITECTURAL** — the serial per-packet boundary is the execution model's own cost. Arch-independent in kind.

Box: gfx942 / 8x MI300X. Asset `/workspace/assets/gfx942/glm52-tp8-final3`, objects rebuilt with
`PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1`, env `PLOW_MLA_PF_V2=1`. Decode
rows only (`PLOW_ROWS_ONLY=interp_decode`), so every prefill/flash object in every arm is
byte-identical to the shipped one.

**Verdict: both commissioned experiments are NULL. One arm is a reproducible REGRESSION. The
value of the round is in three things the nulls establish:**

1. **The FRONT of the boundary — claim → stream entry → instruction header — is not on the
   exposed critical path.** Removing the instruction prefetch entirely (`PLOW_INST_PF=0`) is the
   *fastest* arm measured; halving it to one load is a null. The "merge / prefetch / widen the
   header loads" family is closed.
2. **An unsafe ceiling instrument prices the whole GATE_HIER follower observe hop at −2.6% TPOT.**
   That is the upper bound on the "collapse the two-hop observe" idea — ~0.7 ms of a 26.8 ms
   token, not the multi-ms the boundary model implies. Any sound version recovers less.
3. **The exposed boundary is CONCENTRATED, not uniform.** 77% of it sits behind four producer ops
   and 38% behind ONE (`GemvQkv`). A per-packet protocol saving of 100-200 ns cannot reach it;
   this is a schedule/shape problem on a handful of edges.

---

## 0. Proof the code under test is in the object the run opens

`Variant::detect` scans for `DevOp::GemvFp8` and the three fp8-KV flash ops. All four are absent
from this blob (`plowrt disasm … | grep -oE 'GemvFp8|FlashDecodeFp8|FlashMlaDecodeFp8|FlashMlaPrefillFp8'`
→ no output), so it detects `Variant::Bf16` and decode opens **`interp_decode_gq.elf`**. Confirmed
live: the run's own log prints `code object … variant=Bf16 prefill_arm=MlaMoe sched=GlobalQueue`.

Every arm was diffed at the ISA level against the control before any number was believed:

| check | result |
|---|---|
| CONTROL rebuild vs shipped `hsaco_glm18/interp_decode_gq.elf` | **instruction-identical** (objdump, byte-for-byte apart from the filename line) |
| CONTROL rebuilt again from the isolated source tree | identical again — the tree is sound |
| `PLOW_INST_PF=2` | probe is **one** `s_load_dword` + one `s_waitcnt`, control has **three** of each |
| `PLOW_INST_PF=0` | probe **gone**: first `s[56:57]` access moves from ISA line 278 to 552 (out of the boundary, into the body) |
| `PLOW_WAIT_PF=1` | `prog.waits` `global_load_dwordx2` moves from **after** the election atomic (ctl: atomic @390, load @439) to **before** it (arm: load @384, atomic @407) |
| `PLOW_HIER_SLEEP=1` / `PLOW_GATE_SLEEP=1` | the matching `s_sleep 8` → `s_sleep 1`, and only that one; the five `s_sleep 2` in `op_collective.h` untouched |
| `PLOW_HIER_NOWAIT=1` | third `s_sleep 8` and its poll loop gone (3 → 2 sites) |
| all knobs OFF | control still instruction-identical, i.e. every knob is inert by default |

---

## 1. A real defect found by reading the shipped ISA

`PLOW_INST_PF` (default ON for gfx942 decode, recorded −0.4% on Gemma) is documented as one
prefetch that overlaps the gate poll. In the object it is **three serialised scalar round trips**:

```
s_load_dword s0, s[56:57], 0x0   / s_waitcnt lgkmcnt(0) / v_writelane_b32 v72, s0, 59
s_load_dword s0, s[56:57], 0x20  / s_waitcnt lgkmcnt(0) / v_writelane_b32 v72, s0, 60
s_load_dword s0, s[56:57], 0x30  / s_waitcnt lgkmcnt(0) / v_writelane_b32 v72, s0, 62
```

Each word is spilled into the SGPR-spill VGPR immediately, which forces an `s_waitcnt` after every
load. Forty instructions later the SAME three fields are read for real as three issues under ONE
wait — so the compiler can schedule it properly; the `asm volatile` operand list is what prevents
it. And the extra two loads fetch nothing new: `PlowDevInst` is 64 B on a 64 B stride, i.e.
exactly one scalar-cache line, so the first load already brings the whole instruction.

`PLOW_INST_PF=2` keeps the prefetch and drops the two redundant round trips. It measures **null**
(§3) — which is the interesting part, see §5.

---

## 2. Instrument

`PLOW_TRACE_RAW` + `scripts/glm52_layer_census.py`, **grouped by `inst`**, `--last-dispatch`, MoE
layers 6..74, ctx 1024, `amd-bench --tp 8 --steps 8 --prompt <1024 real ids>`. Every run kept
135,834 records / 2523 packets / **33.00 packets per layer**.

The statistic is the **interval union** of every packet's `[min(ready), max(end)]` in a layer,
subtracted from the layer span: wall in which NO packet is executing anywhere. The positive-gap
sum used by earlier reports **double-counts ~2x** on this program — `GLM_MOE_CORESIDENT=2` runs 8
expert slots on disjoint CU partitions, so packet i+1 starts long before packet i ends and the
inst-order walk still charges the next positive gap in full. Both are printed. (Added to the
census as an `EXPOSED boundary` line plus a by-producer attribution; cross-checked against the
earlier ad-hoc `deadtime.py`: 33.3 vs 33.27 µs/layer on the same trace.)

**Control: 32.5 µs/layer exposed = 985 ns per boundary.** Four independent control traces read
32.6 / 32.8 / 32.5 / 32.1 → **spread 0.7 µs = 2.1%**, i.e. this instrument resolves ~2%, not 0.2%.
The positive-gap statistic reads 66-67 µs/layer = 2.02 µs/boundary on the same traces, reproducing
the campaign's 2.13 µs figure and its 2x correction.

---

## 3. Results — exposed boundary (median over MoE layers 6..74, ctx 1024)

| arm | change | EXPOSED µs/layer | ns/boundary | vs control | verdict |
|---|---|--:|--:|--:|---|
| control ×4 | shipped | 32.6 / 32.8 / 32.5 / 32.1 | **985** mean | — | spread 2.1% |
| `PLOW_INST_PF=0` | prefetch removed entirely | 31.7 | 961 | −2.4% | **NULL** (at/inside spread) |
| `PLOW_INST_PF=2` ×2 | 3 serialised scalar round trips → 1 | 32.4 / 32.0 | 976 | −0.9% | **NULL** |
| `PLOW_HIER_SLEEP=1` ×3 | follower poll 512 → 64 clk | 32.4 / 32.6 / 32.3 | 984 | −0.1% | **NULL** |
| `+PLOW_GATE_SLEEP=1` | both polls 512 → 64 clk | 33.0 | 1000 | +1.5% | **NULL** |
| `PLOW_WAIT_PF=1` ×2 | wait-list load hoisted above the election | 34.5 / 34.9 | 1052 | **+6.8%** | **REGRESSION** |
| all three stacked | | 34.1 | 1033 | +4.9% | carries WAIT_PF |
| `PLOW_HIER_NOWAIT=1` | **UNSAFE CEILING** — follower observe hop deleted | n/a | n/a | n/a | see §4 |

Traced `amd-bench` walls from the same runs (ctx 1024, 8 steps, **all 8 ranks token-identical in
every safe arm**): control 26.758 / 26.770; `INST_PF=0` 26.844; `INST_PF=2` 26.894 / 26.871;
`HIER_SLEEP=1` 26.741 / 26.735; both sleeps 26.726; all-stacked 26.663; `WAIT_PF` 26.881 / 26.883.
That is a 0.9% band with no ordering — the traced wall resolves nothing at this scale and is
reported only as a sanity check.

### Why `PLOW_WAIT_PF` loses — mechanism, not noise

Under `PLOW_GATE_HIER` **only the leader polls the wait list**; followers skip it entirely, and
that traffic reduction is a large part of why the hierarchy is worth −12% on GLM. The prefetch has
to sit *above* the election to overlap it, so it cannot be conditioned on `h_lead` — it therefore
re-introduces `nper-1` extra reads of `prog.waits` per (packet, domain), exactly the traffic the
hierarchy removed. Confirmed in the ISA: the prefetch load is guarded only by
`threadIdx.x < wait_len`, not by the leader test. There is no fix inside this shape: gating it on
`!h_on` makes it a no-op on GLM decode, where every multi-slice packet is `h_on`.

---

## 4. The ceiling: what the whole two-hop observe is worth

`PLOW_HIER_NOWAIT=1` deletes the follower's observe hop — the follower stops waiting for its XCD's
leader and goes straight to its own L1 invalidate. **Unsound by construction** (a follower can
read this XCD's L2 before the leader has invalidated it; the `PLOW_CTR_FENCE` probe measured that
race at a 100% hit rate). It cannot deadlock, so it runs. It prices the entire tail of the chain:
one dependent release atomic + one dependent observe load + one poll granularity.

**Result: 26.068 ms/token against a control band of 26.66-26.89 (n=9 traced runs across all safe
arms, of which 2 are the control itself) — `−2.6%`, cleanly outside that band.** Caveat stated
plainly: this is **n=1** for the ceiling arm. It is outside a 9-observation band, which is why it
is reported, but it is a single observation and should be re-taken before anyone plans work
against it. The trace census is not comparable for this arm (removing the
rendezvous makes packet intervals overlap, so the union statistic reads 0.0 and the layer span
blows up to 630 µs — the arm changes the structure the statistic measures); the wall is the
instrument here.

**So: collapsing the GATE_HIER two-hop observe into one — named in the prior review as the single
largest unbuilt term — has a hard ceiling of −2.6% TPOT, and a sound version recovers strictly
less than that.** Worth knowing before anyone spends days on it.

---

## 5. What the nulls actually establish

The commissioned framing was "five dependent global round trips plus two `s_sleep(8)`
granularities, 19% of the token, fully exposed". Three corrections come out of this round:

* **The exposure is 9.5%, not 19%** (the interval-union vs positive-gap correction, reproduced
  here independently: 985 ns/boundary exposed vs 2.02 µs/boundary by the old statistic).
* **The front three round trips are free.** `PLOW_INST_PF=0` removes the header prefetch outright
  and is the *fastest* exposed-boundary reading in the table; `=2` removes two demonstrably
  serialised `s_waitcnt lgkmcnt(0)` and is a null. Two independent ablations in opposite
  directions both move nothing. Whatever the boundary is made of, it is not the claim → entry →
  instruction chain.
* **The poll granularity is a null on GLM/GATE_HIER too.** The existing "sleep constant does not
  matter" sweep (4680dc6) predates `PLOW_GATE_HIER` (f91dc82) and so never covered the follower
  site. It does now: `HIER_SLEEP=1` over three traces is −0.1%, and tightening both sites is if
  anything worse (+1.5%, more polling traffic). **The record can now say the sweep holds on this
  model, at both sites, post-hierarchy.**

---

## 6. Where the exposed time actually is — the useful finding

The census now attributes each idle window to the packet that *ends* it. Across three independent
control traces the attribution is stable to ±0.2 µs:

| producer op | exposed µs/layer | share |
|---|--:|--:|
| `GemvQkv` | 12.4 / 12.5 / 12.6 | **38%** |
| `Gemv` | 4.9 / 5.1 / 5.2 | 16% |
| `FlashMlaDecode` | 4.9 / 4.9 / 4.9 | 15% |
| `MoeExpertDownFp8Blk` | 2.7 / 2.8 / 2.8 | 8% |
| everything else (29 packets) | ~7.5 | 23% |

**The exposed boundary is not 33 × 1 µs of uniform protocol tax. Four edges own 77% of it and one
owns 38%.** The layer-40 timeline shows the shape: `GemvQkv` (b=146) → `RmsNorm` (b=**1**, 5.3 µs)
→ `GemvQkv` (b=149) → `RmsNorm` (b=**1**, 5.3 µs) → `HeadNormRope` (b=**1**, 4.4 µs) →
`FlashMlaDecode`, with a 24.8 µs inst-order gap in front of the flash. Those b=1 packets are the
same defect `pf_wide_cus` fixed in PREFILL (b=1 norms were 73% of the prefill layer span there);
in DECODE they survive because the row count is genuinely 1, so they cannot be widened by rows —
they would have to be widened by FEATURE, folded into their neighbour, or moved off the serial
chain.

That is the shape of the next experiment on this axis, and it is a scheduling/emit question, not a
protocol one. A protocol change that saved 200 ns on all 33 boundaries would recover 6.6 µs of the
32.5; making the `GemvQkv → flash` chain not serialise would recover up to 12.4 on its own.

---

## 7. What shipped from this round

Nothing. Every arm is a null or a regression, and the honest disposition is:

* `PLOW_INST_PF=2` — **do not land.** The plan was to land it as a free cleanup ("strictly less
  code, measured null"), and the served A/B removed that basis: it is +0.16/+0.17% and the sign is
  the SAME in all six cells. Inside the control spread, so it is still a null and not a
  regression — but "smaller and not slower" is no longer supported, and there is no perf case.
  What survives is the finding, not the patch: the three-word probe's extra two loads are provably
  redundant (one 64 B line) and demonstrably serialised, and removing them changes nothing, which
  is §5's point.
* `PLOW_GATE_SLEEP` / `PLOW_HIER_SLEEP` — keep as documented knobs at their shipped default of 8,
  with the sweep result recorded at both sites. Do not spend on this again.
* `PLOW_WAIT_PF` — **reject.** Reproducible +6.8% on the exposed boundary, with a mechanism.
* `PLOW_HIER_NOWAIT` — ceiling instrument only, `#error`-worthy if anyone tries to ship it.

## 8. Served A/B — obtained

`bench_speed.sh`, port 8271, ctx 1024 + 4096, conc 1, 8 prompts/cell, 128 output tokens, **3
interleaved rounds** (ctl → INST_PF=2 → HIER_SLEEP=1, repeated). All 9 cells passed their
coherence gate, all 9 reported `model: glm-5.2-fp8`, and the port-collision guard never fired.

**TPOT ms/token**

| ctx | arm | R1 | R2 | R3 | mean | vs ctl | ranges |
|---|---|--:|--:|--:|--:|--:|---|
| 1024 | control | 26.80 | 26.85 | 26.84 | 26.830 | — | [26.80, 26.85] |
| 1024 | `PLOW_HIER_SLEEP=1` | 26.79 | 26.79 | 26.72 | 26.767 | **−0.24%** | [26.72, 26.79] disjoint |
| 1024 | `PLOW_INST_PF=2` | 26.85 | 26.91 | 26.86 | 26.873 | **+0.16%** | [26.85, 26.91] overlapping |
| 4096 | control | 29.04 | 29.10 | 29.09 | 29.077 | — | [29.04, 29.10] |
| 4096 | `PLOW_HIER_SLEEP=1` | 29.06 | 29.03 | 28.98 | 29.023 | **−0.18%** | [28.98, 29.06] overlapping |
| 4096 | `PLOW_INST_PF=2` | 29.09 | 29.17 | 29.12 | 29.127 | **+0.17%** | [29.09, 29.17] overlapping |

**Control's own round-to-round spread: 0.19% @1024, 0.21% @4096.** Every delta in the table is
inside it. **By the stated criterion these are NULLS**, and the served instrument agrees with the
trace instrument.

Two honest footnotes rather than one clean sentence:

* `HIER_SLEEP=1` at ctx 1024 has **disjoint ranges** from the control (26.72-26.79 vs 26.80-26.85)
  across three interleaved rounds. That is suggestive. But the effect is 0.24%, i.e. **1.3x the
  control spread** — against the 50x that `PLOW_MOE_DEC_LG` showed when it was real — and at ctx
  4096 the ranges overlap. Not established. If anyone wants to chase it, it needs many more rounds,
  not a different argument.
* `INST_PF=2` is consistently, marginally **WORSE** — +0.16/+0.17%, same sign in all six cells.
  Inside the spread, so still a null, but it removes the basis for landing it as a free cleanup
  (see §7).

**Character-identical gate: PASSES for all four arms** — control, `INST_PF=2`, `HIER_SLEEP=1` AND
`WAIT_PF=1` — on four prompts including a 2883-byte free-form technical answer. All four transcripts
are byte-identical (`md5 2c412dc132efac9a28fe2349654c3b3c`, 3377 bytes each).

Two co-tenancy hazards were found while queueing this, and both defeat the documented protocol:

* a sibling was serving **GLM-5.2 on port 8195 from a binary named `plowrt_stock`**. `pgrep -x
  plowrt` is comm-exact and does not see it, and because the co-tenant serves the SAME model,
  `bench_speed.sh`'s `model:` line matches and does **not** catch the collision either. This
  battery therefore used ports 8271/8272, asserted the port free (`ss -lptn`) before every run,
  and used `pgrep '^plowrt'` (comm PREFIX) for its idle check.
* an **unconditional `rmdir /tmp/plow_gpu.lock` in a signal trap deletes a SIBLING's lock** if the
  script is killed while still waiting for one. Every driver here sets `HAVE_LOCK=1` only after
  `mkdir` succeeds and installs the trap after that.

---

Objects, assets and traces: `/root/.claude/jobs/b09a4bcc/tmp/pb_*`,
`/workspace/assets/gfx942/pb-*`, `/tmp/pb/t*_<arm>_1024.bin`. Source diff (against
`worktree-glm52-bringup`): `/root/.claude/jobs/b09a4bcc/tmp/packet-boundary.patch`; isolated build
tree `/root/.claude/jobs/b09a4bcc/tmp/pbtree`.
