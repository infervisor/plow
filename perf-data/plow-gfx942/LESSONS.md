# Lessons from the gfx942 campaign

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **METHOD** — instrument and gate design. The lessons are about how to measure, not about this silicon, and transfer to any arch.

Method, not results. Every entry here cost real time on this box, and most of them cost it by
producing a **confident wrong answer** rather than an obvious failure. Results live in
`glm52-experiments.md`; the current state lives in `README.md`.

Ordered by how much they cost.

---

## 1. Prove the code under test is in the loaded object — before believing any result

`Variant::detect` picks the decode object by scanning opcodes for `DevOp::GemvFp8`. GLM's fp8 is
*block-scaled* `GemvFp8Blk`, so GLM decode loads `interp_decode_gq.elf`. An entire agent-round built
kernel arms **and their ablation** into an object the run never opens, measured "deleting 100% of
the work moves the token 0.0%", and concluded the kernel body was irrelevant. Nothing was being
deleted.

The correction reopened the whole decode kernel route and immediately paid: retiring one arm was
−11.8%, and a lane-group rewrite that had been recorded as a null was worth −7.6% TPOT.

> **A null has two explanations: the work is free, or the work was never removed. Serve-gating
> cannot tell them apart — code that never loads serves perfectly.**

Corollary: `PLOW_ROWS_ONLY=<stem>` silently matching **zero** rows still prints `ready (1 objects)`.
Check the row count, not the exit code.

---

## 2. A gate that never re-emits cannot catch an emitter regression

The standing Gemma cross-gate re-used a **stored** asset and rebuilt only the objects. That asset
predated the fused-quant fold, so for weeks the gate re-certified a blob that *could not contain*
the regression — while being cited as "no collateral" after every round.

The regression it could not see: a freshly-emitted Gemma-4 gfx942 blob answered "capital of France"
with `,1___....1.111111111111`.

> **Gate the artifact you ship, built the way you ship it.** If the gate rebuilds only half the
> pipeline, it only tests half.

---

## 3. Demonstrate that a gate can FAIL before trusting it to pass

Directly downstream of §2, and now enforced: every guard landed in this campaign was shown going
red on a known-bad input before being accepted.

- The facts gate: an injected context fault at *identical token count* ⇒ exit 1, all regressions in
  the retrieval class.
- The host/device geometry agreement test: perturbed from **both** sides.
- The gfx942 asm contract: pointed at a wrong-arch object, and with a perturbed expectation.
- The re-emitting Gemma gate: run against the known-bad blob, with the old stored-asset procedure
  passing alongside as a control.

> **A consistency check that has never gone red is not evidence that the things it compares agree.**

It also has to refuse to answer when it is powerless: the facts gate exits **2**, never PASS, if its
baseline is too weak or its answers land before the divergence point.

---

## 4. A tight control spread is not evidence you measured the right thing

Unbound `amd-bench` does not load the packed routed experts — the exact term that changes when you
re-shard. Pricing TP4 against TP8 unbound gave `alpha = 0.965` with a **0.1% spread**: beautifully
reproducible, and the **wrong sign**. Bound, it is 1.170.

The tell was in the log the whole time: the unbound TP4 arm's `memset_ms` was *smaller* than TP8's,
which cannot happen if the larger shard is resident.

> **Reproducibility measures the instrument, not the claim.** The object-selection failure in §1 was
> also perfectly reproducible.

Related instrument traps, all real here:

- `amd-bench`'s `last id` is **not a correctness signal** — it never prefills, so attention reads a
  KV cache that was never computed. Several historical "token-identical" claims rest on it.
- `nix develop` alone does not get you the GPU: the flake adds `/opt/rocm/lib`, which does not exist
  on this box, and `plowrt` then falls back to the **CPU interpreter and serves perfectly**. Two
  agents lost batteries to this. Assert `HSA backend selected` before trusting a number.
- `--dump-logits` writes nothing without `--prompt`.

---

## 5. When a kernel is declared closed, check which PART was measured

Five separate attacks closed the MoE grouped-prefill k-loop. The epilogue next door — same kernel,
never examined — was doing 1.77M serialized round trips, and hoisting it was −18.8%.

> "This kernel is optimised" is a claim about the code someone read.

The same shape recurred in the DSA indexer: the prime suspect (a materialised T×ctx score matrix)
cost ~0.13 ms/layer, while the real cost was operand re-fetch — 16 loads and 12 `s_waitcnt` drains
per 16 MFMA, running at 5–7% of peak.

---

## 6. A value duplicated is a bug with a delay fuse

Four instances found in one day, in four subsystems. In every case **the duplicate worked and the
original rotted**, and none was reachable by an existing test:

| duplicate | consequence |
|---|---|
| `GM_LDS_HALVES` mirrored as the CDNA4 value on every part | emitter fused batches onto an arena 4.8× smaller than it believed — **no batched-decode blob was ever correct on gfx942** |
| a GLU-into-quant kernel arm written on NVIDIA, never on AMD | the emitter deleted the packet computing `fu`; the kernel quantized an `fu` nothing wrote — fluent, confident, wrong |
| runtime `MAX_CHUNK = 8192` shadowing the packet's `shapes.max_chunk` | a wider-rung blob served **as if the rung were absent** |
| `RuntimeConfig.amd.trace_raw` parsed, read nowhere, env var read directly | `--trace-raw` was decoration that did nothing when set |

> **Derive it, or check it. Never restate it.** The guards that now exist —
> `device_header_agreement`, `every_field_has_a_reader`, `no_raw_env_reads` — all target this one
> shape, and one of them caught its first real regression on the very next branch to land.

---

## 7. Ablate in both directions before believing an attribution

The packet-boundary round deleted an instruction prefetch entirely **and** fixed its redundancy.
Both moved nothing. Two ablations in opposite directions both returning null is much stronger
evidence that a component is off the critical path than either alone — and it closed an idea family
rather than one patch.

Conversely, a planned landing was **reversed by its own data**: an arm that was "strictly less code,
measured null" on the trace instrument came back +0.16% with the same sign in all six served cells.
Still a null — but "smaller and not slower" was no longer supported, so it did not land.

---

## 8. Compilers defeat bit-identity claims

A "bit-identical by construction" rewrite of the MLA merge matched at `ns=1` and differed at `ns=2`
— the shipped prefill setting. Cause: the natural interleaved form `acc += pv*wt` contracts to one
`v_fmac_f32`, while the shipped `acc += select(0, pv*w)` cannot and stays mul+add.

> **Mirror the shipped expression verbatim, and gate at the shipped configuration**, not the
> simplest one.

Two more numerics traps from the same area:

- `ds_bpermute` honours EXEC on the **read** side, so cross-lane guards must be masks applied
  *after* the fetch, never EXEC. This has bitten twice.
- f32 addition is commutative but not associative. Making a fused MoE combine *order-independent*
  needed integer-valued f64 accumulation — and even then, bit-identity with the shipped
  slot-ordered sum is unattainable, with arithmetic: the shipped combine sums in **slot** order and
  any in-place accumulator receives contributions in **row** order.

---

## 9. Character identity and answer quality are different questions

Character-identity gates fire on harmless reassociation and say nothing about correctness. That is
why the ragged-chunk work sat opt-in for months: not evidence of harm, but the absence of a gate
that could distinguish *reworded* from *degraded*.

Building that gate took one agent-round and unblocked a −24% TTFT default. Design points that
mattered: needles must sit **late** in long answers (divergence lands ~11% in, so a short answer is
a useless instrument), cells must be **off-rung** (on-rung cells are byte-identical and carry no
signal), and the comparison must be **pairwise**, not two independent scores.

---

## 10. State the configuration a number belongs to

A recorded note — "tp4 infeasible (weights/card)" — was correct for **pure TP4** (173.6 GiB/rank)
and was read as covering **TP4×PP2**, which is at parity with TP8 (85.8/87.8 vs 86.8). That single
inference kept an option off the table for the whole campaign.

The option was eventually rejected anyway, on unrelated grounds — but by luck, not by process.

> Write the configuration into the note, not into the reader's head.

---

## 11. A model that fits absolute values can get the sign of a DIFFERENCE wrong

A cost model for chunk plans — `chunk(r, c0) = FIX + r*(A + C*(c0 + r/2))` — was fitted to the
engine's own measured single-chunk rung TTFTs. It reproduced those to **2%** and four held-out
multi-chunk cells to **5%**. On that basis it predicted that repricing `LAUNCH_ROWS` would
*regress* long context: +105 ms at 10369 tokens, +1163 at 71808.

Served A/B: **−111 and −245**. Both predictions wrong in sign.

The cause is precise and worth knowing: the within-chunk superlinearity in T that the rung ladder
really does show **does not transfer to padded rows**. Measured across five plan pairs, a padded row
costs 0.078–0.202 ms with no trend in context — so the DP's linear-in-padded-rows cost is the right
*shape*, and only its constant was wrong.

> **A planner only ever decides differences.** Fitting every absolute number it is shown is not
> evidence that a model ranks two plans correctly. Validate a cost model on the *deltas* it will be
> used to choose between, not on the levels it was fitted to.

Conclusions had already been committed off that model; they were reverted when measurement landed.

---

## 12. Check whether the work already exists — and whether the switch is simply off

Round 2 spent ~40 minutes producing two results that were already in the tree:

- A GEMM tile experiment was proposed, scoped at a day, and **refuted by one compile**. The XOR
  swizzle it wanted to add already existed at all eight sites; the blocker was a staging
  granularity assert, not LDS; and the whole ladder re-cut had already been built, oracle-passed
  and measured at **+11.7%** — documented ~600 lines below the comment that had been read.
- A narrow-K GEMV arm was written (129 lines), compiled, and verified present in the object —
  then found to target a body GLM emits **3 times out of 2523 ops**, at a K its guard rejects.
  The real fix existed on a merged branch, in the other body, measured at −1.6% TPOT.

Both were one command away: `git branch -a | grep`, and grep the flag name.

> **The opposite failure is worth more than the work it prevents.** That second fix was
> `default-off and passed by no recipe`, so a landed, character-gated, −1.61% TPOT win had been
> sitting unused. **Grep the tree for opt-in flags that carry a measured win and that no recipe
> passes.** It has now paid twice; `PLOW_MLA_PF_SV` and `PLOW_MOE_PF_EPI` are alive only because
> `build_gfx942.sh` passes them explicitly.

---

## 13. A decomposition bounds what you can WIN, not what you can LOSE

Prefill at 8k is 91.0% packed with 5.1% gate wait, so per-XCD prefill placement was predicted at
"low single-digit % of TTFT, either way". Measured: **−9.1% at 1k and +97.6% at 16k**, monotone
in context.

The 9%-of-CU-time budget describes how much time is *available to recover*. A scheduling change
that replaces dynamic load balancing with a **static partition** is not drawing on that budget —
it is changing how well the machine can *balance*, and the loss scales with the work, not with
the wait. Same family as "banding closed negative" and "on a saturated grid, overlap is a static
partition of the same CUs".

> Ask what a change can COST, on its own mechanism, before quoting a headroom figure as a bound.

---

## Battery discipline (shared box, many agents)

Every one of these was paid for at least once:

- **`pgrep -x plowrt` is not enough.** It is comm-exact and misses a sibling running a renamed
  binary (`plowrt_stock`). Use `pgrep '^plowrt'`, and assert the port free with `ss -lptn`.
  `pgrep -f "plowrt serve"` is worse — it **self-matches your own launcher** and spins forever while
  holding the lock.
- **Check `bench_speed.sh`'s `model:` line.** A sibling serving a *different model* on your port has
  corrupted an arm; a sibling serving the *same* model cannot be caught this way at all.
- **A signal trap MUST `exit`.** `trap 'release_lock' EXIT INT TERM` releases the lock and then lets
  the script keep running — unlocked — and its later EXIT deletes a lock that by then belongs to
  someone else. Set `HAVE_LOCK` only after `mkdir` succeeds.
- **SIGTERM, never SIGKILL, on `plowrt`.** `kill -9` leaves the persistent cooperative megakernel
  **resident**, corrupting later runs. Verify `rocm-smi --showuse` reads 0% before trusting an A/B.
- **`rocm-smi --showuse` is NOT enough — check VRAM too.** vLLM's TP workers rename themselves to
  `VLLM::Worker_TP*`, so `pkill -f 'vllm26/bin/vllm'` kills only the launcher and leaves all 8
  alive holding **184 GB each**. `rocm-smi --showpids` then lists them as dead `UNKNOWN` entries
  because it cannot resolve the renamed comm, and `--showuse` reads **0% busy** because they are
  idle. The next run fails to allocate **6 MB**. This is the `pgrep -x plowrt` / `plowrt_stock`
  trap one process-name convention over. Check `rocm-smi --showmemuse` reads ~0% VRAM, not just
  0% utilisation, and kill by the name the process actually has.
- **Idle GPUs are not evidence of a stale lock.** A legitimate holder does CPU-side emit/build work
  between device arms.
- **`build.json` travels with `model.pkt`.** It carries the `requires` set the arm-refusal chain
  reads; copying only `model.pkt` silently disables arm refusal, and a blob that should be refused
  then runs on an object missing the arm it needs.
- **±20% DVFS noise is ours, not the box's** — vLLM measured ≤1.0% across the same three reps where
  plow measured 17.9%. Always state the control's own round-to-round spread.

## 14. A missing compiler optimisation is a HYPOTHESIS about the hardware — test the hardware first

`glm52-flash-streamed-v.md` sec 8 carried "LLVM does not form `ds_read_u16_d16_hi` from the
current source; it needs gfx942-guarded inline asm" as a named, quantified next lever for a
full campaign round. Both halves of that sentence were wrong, and the cost of finding out was
one 20-line kernel:

    pool[4] = 0xab04, destination register seeded 0x00005a5a
      ds_read_u16_d16_hi -> 0xab040000     (preserve would have given 0xab045a5a)

On GFX940+ a D16 instruction writes the FULL 32-bit VGPR. The backend was not failing to
find the pattern; the pattern is unsound on this target. Writing the inline asm by hand
produced a kernel that compiled, disassembled exactly as intended, was 30% shorter — and
computed the wrong answer in all 8192 output words.

Three rules fall out of this:

1. **"The compiler won't emit X" is a claim about X's semantics, not about the compiler.**
   Before writing asm to force an instruction, spend five minutes proving the instruction
   does what you think on the actual silicon. The isolating test seeds the destination with
   a recognisable constant and issues exactly ONE instruction; anything larger confounds
   the semantics question with an addressing or scheduling question.
2. **Include an arm whose only job is to kill the convenient explanation.** The first
   failure looked exactly like a two-loads-in-flight hazard, which would have been fixable
   with a wait. Arm 5 put a full `lgkmcnt(0)` drain between the low and high loads and
   still failed — that is what turns "my scheduling is wrong" into "the instruction has no
   merge semantics", and it is the difference between closing the item and re-opening it
   next round.
3. **Also run the arms you expect to be redundant.** Arms 1 and 2 (source rewrites that
   "should" have handed the backend a v2i16 insert) came back byte-identical to the shipped
   code. That is the evidence that the backend had already considered and rejected the
   form — without it, the refutation would only cover hand-written asm.

Two numbers in the original claim were also wrong in the safe direction, and only surfaced
because the probe reproduced the real loop shape rather than the remembered one: the perms
are 23% of the PV body, not 8%, and the shipped double buffer was already scheduling
partial waits (29 x `lgkmcnt(8)`, 29 x `lgkmcnt(12)`), not the full drains a less faithful
probe had suggested. A ceiling quoted from memory is worth re-deriving before it is spent.

## 15. `cargo build` can silently disarm the thing you are measuring with

`target/release/plowrt` is only an accelerated server when it was built `--features hsa`.
Any *other* cargo invocation in the same tree — `cargo test --workspace`, `cargo build --bin
plowc`, anything that does not repeat the feature — relinks that path **without** the HSA
backend. The resulting binary is not broken. It serves the model on the CPU reference
interpreter: correct answers, coherence gate green, sensible-looking latencies, every one of
them fiction.

Measured 2026-08-09, and it cost a four-arm interleaved A/B: a `cargo test --workspace` run
started to audit the compiler quietly replaced the binary, and the next four serve runs came
back in 19 s each with `hsa=false`. Nothing about the *arms* was wrong.

Three things this establishes:

1. **The HSA gate earns its place, and it should be gate ZERO, not gate two.** Checking the
   log for `CPU reference backend active` works, but only after paying a 75 s model load and
   taking the GPU lock. `grep -aq libhsa-runtime64` on the binary is a millisecond and runs
   before either. Both `run_plow.sh` and `run_gsm_paired.sh` now do that first.
2. **A measurement harness must verify its own instrument, not just its subject.** Every gate
   in this directory was pointed at the model or the objects. None was pointed at the binary
   doing the serving, which is the one component every arm shares — so a fault there
   corrupts all arms *equally* and looks exactly like "no difference between arms".
3. **Audit work and measurement work contend even when they look independent.** Reading the
   compiler for bugs is a CPU task and the A/B was a GPU task, which is why running them
   concurrently seemed free. The shared mutable resource was neither CPU nor GPU; it was a
   file path.

## 16. A staleness guard that only covers one arch is invisible on the other

`crates/devgen/tests/tuned_tile_selection.rs` exists precisely to catch silent degradation:
when the tuning store goes stale, GEMM tile selection reverts to the analytical model and
reports tier `portable` — which is also what it reports when nothing was ever measured. The
test's own comment says this is "the failure mode that is INVISIBLE without this test".

It mentions `gfx950` eleven times and **`gfx942` zero times**.

So on the arch this campaign actually ships, the guard was never armed, and `plowc tune
status --gpu MI300X` says plainly what that cost:

    *** EVERY RECORD IN THIS CELL IS STALE. ***
    The store holds 196 record(s) and the compiler can use NONE of them: selection
    falls back to the analytical model for every shape and reports tier `portable`,
    while `tuned_tile_selection` keeps passing on whatever other cell has data.

The staleness key is `defines + toolchain + preprocessed-source digest`, so **one edit under
`runtime/amd/` re-stales the entire store at once** — and `runtime/amd/` changed in at least
five merges between the Aug 7 bring-up campaign and Aug 9. The gfx942 records have been
unusable for days, through every measurement this campaign published.

The transferable rule: **a guard is only a guard for the configurations it enumerates.** When
a test hardcodes a target, adding a second target does not extend it — and a test that passes
because *some other cell* has data is worse than no test, because it reads as coverage.

## 17. Static instruction count is not time, and "value-identical" is not "output-identical"

Making `f2bf` branchless deleted an exec-mask save/restore from a function on every store path
in the runtime. Every static measure said yes: **-5.0% instructions, -13.3% SALU, -30%
exec-mask ops** on the real prefill megakernel; **-55% instructions** in the isolated
64-conversion tile. It was value-identical over **all 2^32** float bit patterns, proved
exhaustively rather than by sampling, with a deliberately-wrong control the same gate caught
1.8 M times.

Served, it was **+4.5 / +5.3 / +7.0% TTFT** at 4k/8k/16k. The change was reverted.

Two rules, and the second is the one that nearly got past every gate in this directory:

1. **Fewer instructions can be more time.** The select form computes BOTH results on every
   conversion where the branch computed one. An exec-mask pair around a rarely-taken branch is
   not obviously a cost on this hardware, and a 6:1 VALU:MFMA ratio does not mean VALU is the
   limiter — plow's flash already ran leaner than aiter's fmha (6.7:1 *including* softmax), so
   the headroom the entry assumed was not there. Compile-time evidence licenses a MEASUREMENT,
   never an adoption.
2. **Value-identical in isolation does not imply output-identical in situ.** The two arms
   disagreed on GSM8K — 0.960 vs 0.970, reproducibly, in both rounds — while the function
   itself provably returns the same bits for every possible input. Changing a function this
   widely inlined perturbs surrounding codegen, and at -O3 that includes which fp contractions
   form, so downstream f32 arithmetic need not round the same way. The commit that landed the
   change claimed "logits byte-identical by construction". That was wrong.

**What caught it was the accuracy column on a speed experiment.** The A/B was run purely to
answer "is it faster", and accuracy was carried along only as a cheap identity check, on the
theory that a value-identical change must score identically. It did not — and that mismatch is
what falsified the byte-identity claim. A speed-only A/B would have measured the regression but
would have left the false "byte-identical" claim standing in the tree.


## 18. A host patch keyed on the wrong axis is invisible to every gate except content-at-length

`patch_kvrow` runs only when the ENGINE batch is 1 — the BLOB's widest rung, not the live rung
— so on a laddered blob the byte-identical rung-1 program wrote every decode step's KV into
ring row 0. Coherence passed (short answers survive on prefill rows), GSM at conc 2–4 passed
(those rungs use the per-row writers), and the one needle 'PASS' on record had silently run at
rung 2. Only the solo needle at 3000 tokens caught it, deterministically, as '741' for '7413'.
Two rules: (a) when a program is kept byte-identical for a gate, audit every HOST-side patch
that program depends on — byte-identity preserves the dependence too; (b) the needle-content
gate is cheap (~3 min) and catches what coherence + short-prompt accuracy structurally cannot.

## 19. "Bit-identical, signal-only" still has to pass the content gate

`PLOW_XR_AGG` touches no value — it defers per-workgroup release signals to one closing
workgroup — and an XR_AGG-only build FAILS needle@3000. A release RMW orders the ISSUING
workgroup's stores; one workgroup signalling for all nblk orders nobody else's. The arm was
ISA-audited, kbench-validated and served through short-prompt GSM before the needle refuted it.
Memory-ordering claims are content claims; gate them as such.

RESOLUTION (2026-08-10), and two corrections to the record above. (1) The fix: the arrival
RMW on word 1 is now itself the RELEASE at SYSTEM scope (xctr_signal's form), and the closer
takes the SYSTEM acquire before the aggregated signal — every edge in the visibility chain at
the scope of the final observer, and the counter line stays memory-side instead of splitting
between one XCD's L2 (agent RMW) and memory (the peers' system RMWs). Fixed arm PASSES needle
3000/8000 ×2. (2) A pre-fix control REBUILT the same day also passed 2/2 — the original
failure was INTERMITTENT, so a passing needle does not clear an ordering bug; only the model
argument does. The gate refutes; it cannot certify. (3) `PLOW_MLA_FOLD_TB` was condemned by
association: the 08-09 bisect never ran it solo (686a3bf says so). Solo gate 2026-08-10:
PASSES 4/4. Both arms re-adopted default-on.

## 20. A tile override is only as valid as the body's untested assumptions

The grouped-MoE kernel at MPF_BK=32 compiles clean, passes the cliff check, and computes
wrong numbers — some BK=64 assumption survives in the body. The 1-layer batch-determinism
probe (solo vs paired, cmp on the text) localized it in one serve each: FAIL at BK32,
byte-identical PASS at BK64. Determinism probes rank with oracles for cost-per-bit here.

## 21. The prepped checkpoint is part of the asset

`plowc --emit` leaves `<asset>/checkpoint` pointing at `--hf-dir`; serving needs the PREP dir
(`GLM-5.2-plow-lite`). A raw-HF checkpoint fails loud (shard size) or as a load-time GPU fault
(with a fold that changes the binding path) — the fault bisects like an object bug and is not
one. Re-emitting into an existing asset RESETS the symlink; re-link after every emit.
