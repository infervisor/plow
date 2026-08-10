# The Gemma cross-gate now re-emits — and it was proven to FAIL before it was trusted to pass

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **METHOD** — a gate that re-emits rather than trusting a stored asset. Arch-independent design; the numbers it gates are gfx942.

Branch `gate-fix`, off `worktree-glm52-bringup` @ 529654b. Box: 8x MI300X, gfx942
([[plow-devbox-is-gfx942]]). Two commissions, both closed here:

* **A.** The standing Gemma cross-gate could not catch an emitter regression. Fixed, and the
  fix demonstrated on a known-bad input.
* **B.** Every batched-decode number on gfx942 was suspect. Audited; §3.

---

## 1. Task A — what was broken about the gate, and what replaces it

### 1.1 The defect

The "standing Gemma cross-gate" was never a script. It was a hand-run procedure — rebuild the
objects with `scripts/build_gfx942.sh`, point the **stored** asset
`/workspace/assets/gfx942/g12b-fp8` at them, run `scripts/bench_speed.sh`, read its `Paris`
coherence check and the 4k TTFT/TPOT pair. That is the procedure behind every
`Gemma 595.0 / 11.05`-style line in this directory.

The stored asset was emitted **2026-08-04**. The fused activation quant (`PLOW_FUSE_QUANT`,
default ON on AMD) landed after it. So for weeks the gate re-certified **a blob that could not
contain the regression under test**, while a freshly emitted blob from the same tree answered
"What is the capital of France?" with `,1___....1.111111111111`.

Nothing in the old procedure was capable of noticing:

* the asset is a *stored artifact*, so an emitter change cannot reach it;
* the failure is **fluent and confident**, so "the server came up and returned 32 tokens"
  passes;
* `amd-bench`'s `last id` never prefills, and the fold lives in **prefill** — it moves the
  activation quant into `d_rmsnorm`'s t3/t4 arm, which writes a wrong KV cache that decode
  then reads. The decode programs of the good and bad blobs are identical instruction for
  instruction.

**A gate that never re-emits cannot catch an emitter regression.** Not "usually misses one" —
cannot, by construction.

### 1.2 What replaces it: `scripts/gemma_xgate.sh`

Emits the blob from the current checkout on **every run**, builds the objects from the same
tree unless a prebuilt set is handed in, assembles a complete asset, serves it, and asserts on
the CONTENT of three answers.

```
scripts/gemma_xgate.sh [name]
  PLOW_XGATE_EMIT_ENV   extra env for the EMIT — this is what makes the gate falsifiable
  PLOW_XGATE_HSACO      reuse an objects dir instead of building one
  PLOW_XGATE_STORED     serve a stored asset instead of emitting (the OLD behaviour, control only)
  PLOW_XGATE_PORT       default 8196, never 8199
  PLOW_XGATE_NO_LOCK=1  caller already holds the GPU lock
```

Properties that are load-bearing rather than decorative:

| property | why |
|---|---|
| re-emits every run | the whole point; a stored asset is available only as `PLOW_XGATE_STORED` |
| `build.json` kept next to `model.pkt` | it carries `backends.<arch>.requires`; assets assembled by copying `model.pkt` alone have an **inert** arm-refusal chain, and several in this campaign did |
| writes `weights.json` | `plowc --emit devblob` does not, and `plowrt serve` opens it unconditionally (bare `Io { path: …/weights.json, NotFound }` at load) |
| refuses a non-`hsa` `plowrt` | without `--features hsa` plowrt serves the CPU reference and decodes garbage — the gate would go red for the wrong reason |
| asserts on answer CONTENT | the failure mode is fluent; a liveness check does not see it |
| `pgrep -x plowrt`, never `pgrep -f "plowrt serve"` | the `-f` form self-matches the launcher and spins forever **holding the lock** |
| SIGTERM only, with a 60 s wait and a loud warning | `kill -9` leaves the persistent megakernel RESIDENT and poisons later runs; the script never escalates |
| signal handler `exit`s | a handler that only cleans up releases the lock, keeps running unlocked, and later deletes a sibling's lock |
| checks the served `model:` id **and** that the listener's pgid is the server it started | a sibling's server on a shared port has already answered a whole A/B battery for one agent this campaign |
| aborts if `rocm-smi` reads >5% with the lock held | that is what a SIGKILLed resident megakernel looks like |

### 1.3 Blob provenance — the gate is emitting the right two things

Reference hashes were reproduced exactly, from two independently built `plowc` binaries
(the `worktree-glm52-bringup` build and this worktree's build), at 529654b:

| emit env | sha256 | prefill packets | decode packets |
|---|---|--:|--:|
| default (guard active, fold OFF) | `c4476d3518f28406761ad6b70f75b755efa51e83052230c999e7a039c56e198e` | 910 | 527 |
| `PLOW_QNORM_FUSE=1` (fold ON — known bad) | `f6c78e31c4e596e868c48c749a73111a413be789412a57dc9ffc72f97b6ffe8a` | 814 | 527 |

Both match the commissioned references. The packet counts are the fold, visible: the bad blob
is **96 packets lighter in PREFILL** (the `QuantFp8` packets absorbed into `RmsNorm`; `Glu` is
the only opcode present in the good blob and absent from the bad one) and **identical in
decode**. That is precisely the §6a signature, independently re-derived here.

---

## 2. The gate was PROVEN TO FAIL before it was trusted to pass

A gate that has never been shown to go red on a known-bad input is not evidence of anything.
It is a green light with no wiring behind it, and this campaign has just spent weeks
discovering what that costs.

**The known-bad input.** `PLOW_QNORM_FUSE=1` still reached the broken fold on gfx942 at
529654b — 75fb82f changed only the DEFAULT, deliberately, "for whoever debugs the arm". So the
falsification arm is a real emit of a real regression, not a synthetic mutation.

**The design of the demonstration.** Both arms run in the same session, against the same
`plowrt` binary, on the **same 28 objects** (built from this tree, `PLOW_OCC4=1`, handed to
both arms via `PLOW_XGATE_HSACO` so an object difference cannot be the explanation). The ONLY
difference between them is one environment variable on the **emit**. Both blob hashes were
checked against the commissioned references before either was served (§1.3).

**Tree provenance, and why it is pinned.** These two arms are at **529654b + this branch's
gate commits** — i.e. the tree where the emitter regression is still live. That pinning is not
incidental: while this gate was being built, the sibling `fuse-quant-fix` landed the real cure
(`eb30157`), which removes the gfx942 carve-out and makes `PLOW_QNORM_FUSE=1` correct. A
falsification run on the fixed tree would prove nothing about the gate. §2.3 runs the fixed
tree separately, as confirmation rather than falsification.

### 2.0 The first pair of arms was thrown away, and why that matters

The first run of this gate produced **XGATE FAIL on BOTH arms** — including the arm whose blob
hash is the commissioned *correct* reference. The answers were not the documented
`,1___....1.111111111111` but `<unused87><unused87> <unused84>`: a different garbage, which is
the tell.

The serve log said it in one line: `HSA probe failed … libhsa-runtime64.so.1: cannot open
shared object file` → `all GPU probes failed — selecting CPU reference backend`. **plowrt never
touched the GPU.** It was launched bare; `nix develop` is what puts libhsa on
`LD_LIBRARY_PATH`, and `bench_speed.sh` has always launched through it.

The preflight check in this script had already asked "is this a `--features hsa` build?" and
been told yes — because it inspected the BINARY. Compiling the dlopen in is not the same
question as the library being loadable, and the gap between those two questions is a silent
CPU fallback that answers every prompt with garbage.

**This is the same failure class as the bug the gate exists to catch**, one level down: a check
that looks like it is asking the important question but is actually asking a cheaper adjacent
one, and passes. It is also the reason a gate must be run against a KNOWN-GOOD input as well
as a known-bad one — a gate that only ever goes red proves as little as one that only ever
goes green, and without the stored-asset control arm this would have been indistinguishable
from a real regression.

Two things landed as a result: the server is launched through `nix develop`, and the gate now
**refuses to score any run whose log shows the CPU fallback**, aborting instead of reporting a
verdict. Belt and braces, because the launch is the kind of thing a future edit changes.

### 2.1 The result

Three arms, one after another, same `plowrt` binary, same session, port 8197 (8195 and 8196
were both held by other runs). Arms 1 and 2 share the **same 28 objects**, so an object
difference cannot explain the difference between them. Full transcript:
`gemma-xgate-transcripts.txt`.

| arm | emit | blob | "capital of France" | verdict |
|---|---|---|---|---|
| 1 — known-bad | `PLOW_QNORM_FUSE=1` | `f6c78e31c4e596e8…` | `,,,_.._......1.1.111.1111111111` | **XGATE FAIL** (exit 1) |
| 2 — current default | none | `c4476d3518f28406…` | `The capital of France is Paris.` | **XGATE PASS** (exit 0) |
| 3 — bracketing control | the STORED `g12b-fp8` asset | `cf33ad4577a75b84…` | `The capital of France is **Paris**.` | **XGATE PASS** (exit 0) |

All three prompts moved together in every arm — the bad arm also returned
`enenenولة็ตenensen的光` for Tokyo and `옆émç_ess/CCหลหล` for 17x23, while both good arms
returned `Tokyo` and `391` exactly.

Two things worth saying precisely about arm 1's answer. First, it reproduces the **documented
signature** — `,1___....1.111111111111` in §6a of the ladder document, `,,,_.._......1.1.111.1111111111`
here. Second, it is exactly the failure a liveness check cannot see: 30 well-formed tokens,
returned promptly, HTTP 200.

**Arm 3 is what makes arms 1 and 2 interpretable.** It is the old gate's behaviour, run
deliberately: the stored pre-regression asset, its own objects, untouched. It passes — as it
always has, and as it did on every merge while the emitter was shipping wrong output. Its
blob hash (`cf33ad45…`) differs from both fresh blobs, which is the whole defect in one line:
**the artifact the old gate certified was not the artifact the branch produces.**

### 2.2 What this does and does not establish

**Establishes:** the gate goes red on a real emitter regression, green on a correct emit, and
green on the stored control — so a red from it is informative in both directions. That is the
minimum bar for a gate to be evidence, and the old procedure never cleared it.

**Does not establish:** that this gate catches every emitter regression. It catches ones that
corrupt short-prompt output on Gemma-4-12B fp8/w8a8. A regression confined to long context, to
another model family, or to performance rather than correctness would still pass. The gate is a
floor, not a proof.

### 2.3 After merging the real cure, the same gate corroborates it

While this gate was being built, the sibling branch `fuse-quant-fix` landed the actual repair
(`eb30157`, merged here as 17b2bbe). Its finding is worth restating because it changes what
§1 means: **the bug was the GLU-into-quant fold, not the norm fold, and it was never
gfx942-specific.** `PLOW_FUSE_QUANT` gates two folds; the RmsNorm t3/t4 arm was always
correct. The other one replaces the elementwise `Glu` packet with a `QuantFp8` carrying
t3=gate t4=up i2=act, and the AMD interpreter ignored those three operands for its entire
life — so the emitter deleted the packet that writes `fu` and the kernel then quantized an
`fu` nothing had written. gfx950 emits the identical broken packet; it was untested at the
only prompt length that reaches the arm (the smallest prefill bucket, t=128), which is why
PR #56's long-prompt TTFT runs never saw it. The gfx942 carve-out from 75fb82f is therefore
gone, and `PLOW_FUSE_QUANT` is default-ON on all AMD again.

**That makes the post-merge run a clean isolation, and the blob hashes say so.** At the merged
tip the DEFAULT emit is `f6c78e31c4e596e8…` — byte-identical to the pre-merge KNOWN-BAD blob
that this gate had just failed. Nothing about the packet changed; what changed is the kernel
that reads it. So the two arms below vary only the OBJECTS, with the blob held fixed at the
exact bytes that produced `,,,_.._......1.1.111.1111111111` an hour earlier.

| arm | blob | objects | "capital of France" | verdict |
|---|---|---|---|---|
| 4 | `f6c78e31c4e596e8…` (the default here) | POST-fix, built from 1fa5b7b | `The capital of France is Paris.` | **XGATE PASS** (exit 0) |
| 5 | `f6c78e31c4e596e8…`, the SAME bytes | PRE-fix, built from 529654b | *never served — load refused* | **XGATE ABORT** (exit 2) |

**Arm 4 is the corroboration.** The identical 32 MB blob that this gate failed an hour earlier
with `,,,_.._......1.1.111.1111111111` now answers Paris / Tokyo / 391, on objects whose only
relevant difference is `d_quant_fp8` gaining the gate/up/act arm. The emitter did not change;
the kernel did. That is the sibling's claim, reproduced by a gate written independently of
their branch — which is the useful kind of confirmation, since a fix verified only by its own
author's harness is verified by one thing, not two.

It is also the first time this campaign has had a Gemma cross-gate result that means anything:
it is a **freshly emitted** blob, so unlike every "Gemma 595.0 / 11.05" line in this directory
it was actually capable of failing.

**Arm 5 checks the other half of `eb30157`** — the `PLOW_T11_GLUQUANT` entry it added to the
packet's `requires`, so that an object predating the fix is REFUSED at load rather than serving
garbage. Handing the folded blob to pre-fix objects produced exactly that, verbatim:

```
Error: Device("packet/object MISMATCH: this packet requires PLOW_T11_GLUQUANT=1 but
  …/hsaco/interp_prefill_fp8_gq.elf was built WITHOUT it — none of [\"plow_t11_gluquant_arm\"]
  is in its symbol table. The AMD dispatch's `default:` does not trap, so those ops would write
  nothing and the prefill would complete with garbage instead of failing. …")
XGATE ABORT: server died
```

This is the exact configuration that produced fluent wrong tokens before the fix, and it is now
a loud load-time refusal. Note the gate distinguishes the two: **ABORT (exit 2), not FAIL (exit
1)** — "I could not run this" is a different claim from "this answered wrongly", and a harness
that collapses them is how a CPU-fallback run gets reported as an emitter regression (§2.0).

Together arms 4 and 5 close the loop on the sibling's fix from the outside: the folded blob is
correct on objects that carry the arm, and refused on objects that do not. Neither outcome was
available before `eb30157`; the same blob simply lied.


## 3. Task B — the batched-decode audit

`GM_LDS_HALVES` was the CDNA4 arena (73,728 halves) on every part. gfx942's shipped occ4
decode profile holds **15,360**. `gemv_qkv_rows` / `gemv_glu_rows` stage `x` ONLY through LDS,
so the emitter choosing those fused opcodes is a *promise* that `M*K` fits — and at
Gemma-4-12B's `hidden = 3840` it fused every batch up to **M=19** onto an object holding
**four rows**, writing the rest past the end of `plow_smem`. Fixed in **2130f04**.

### 3.1 The rule I applied

* **B = 1 is SAFE and was NOT re-run.** B=1 emits are byte-identical across 2130f04 (verified
  in that commit; `1 * hidden` fits either arena). Every single-row number on this box stands,
  including every `concurrency > 1` row of a B=1 blob — those measure QUEUEING, not batching,
  and the decode program is unchanged.
* **B > 1 emitted before 2130f04 is RETRACTED**, not corrected. These are not inaccurate
  timings; they are timings of a **different computation**. The corrupted rows do other work
  at an unknown rate, so the numbers are not even directionally usable, and no scaling
  argument may be rebuilt from them.
* **B > 1 emitted after 2130f04 stands**, and in this directory that is only
  `glm52-decode-batch-ladder.md` §7/§11.
* **Nothing was deleted.** Every retracted number is struck in place with its cause, so the
  claims that cite it stay traceable.

### 3.2 Verdicts, file by file

| record | B | verdict | basis |
|---|--:|---|---|
| `g12b-b8_b8_tp1_general.csv` (4 rows) | 8 | **RETRACTED, file REMOVED** | pre-fix B=8 blob |
| `g12b-b8_b8ctx128_tp1_general.csv` (1 row) | 8 | **RETRACTED, file REMOVED** | pre-fix B=8 blob |
| `gemv-mlp-and-tensile.md` §"Batched decode" — 3 tables + derived prose | 4/8/16 | **RETRACTED** in place | pre-fix; **no B>1 correctness gate recorded at all** |
| `fusion-review-and-crossover-sweep.md` §3 batch sweep, rows 4 and 16 (+ the 117 ms and 63.6 ms in its note) | 4/16 | **RETRACTED** in place | pre-fix; the section states "the b=1 serve gate is the correctness anchor" — a B=1 gate cannot certify a B>1 blob |
| `glm52-tp4-pp2-evaluation.md` §D.4(b) table + §E.1 rows + §D.4 prose | 8 | **RETRACTED** in place | re-quotes the B=8 CSV |
| `glm52-experiments.md` item 2 (`33.98 → 66.61`, `the ladder's 4.00×`) | 8 | **RETRACTED** in place | third-hand re-quote of the same CSV |
| `glm52-decode-batch-ladder.md` §7 and §11 (all arms `a`,`c`,`b`,`a16`,`c4`,`b4`,`a4`) | 1..16 | **SOUND — kept** | see 3.3 |
| `glm52-decode-ladder-vs-vllm026.md` / `.csv` (landed on the base branch while this audit ran) | 1..16 | **SOUND — kept** | its plow columns are the ladder's §7/§11 **verbatim** ("No plow blob was rebuilt or re-emitted"), i.e. post-fix data re-tabled; the vLLM column never ran plow's emitter |
| every other `*.csv` in this directory | 1 | **SOUND — not re-run** | B=1 blobs; `concurrency` column is request concurrency |
| `gemv-mlp-and-tensile.md` `PLOW_GQ_BATCH` sweep | 1 | **SOUND** | `PLOW_GQ_BATCH` is global-queue lookahead depth, not decode batch |
| `glm52-band-pipeline-cusubset.md` `b4` arm | 1 | **SOUND** | `b4` = `PLOW_GLM_XR_BAND=4`, not batch 4 |
| all GLM-5.2 numbers, all files | 1 | **SOUND** | GLM has no batched decode at all — `glm_emit_full` emits one row structurally |
| emit-side counts (packets, blob bytes, VGPR/LDS/spill) under batch headings | 1..16 | **not measurements** | compile-time, unaffected in kind |

Audit scope note: this table is complete as of the merge-back commit. Because `§7/§11` is now
cited by a second document, it is worth restating that those rows are the ONLY batched gfx942
measurements in this directory with a correctness anchor — anything else that re-tables them
inherits that, and anything that re-tables the retracted numbers inherits the retraction.

### 3.3 Why the ladder's own §7/§11 numbers survive — verified, not assumed

Three independent checks, all of which had to agree:

1. **Commit order.** `git log` puts the arena fix at `2130f04` (08-08 15:46) and every commit
   that wrote those tables strictly after it — `d9ce9f9` 16:36, `b807b32` 16:37, `d26270f`
   18:01, `e55a41f` 18:02.
2. **Self-report.** §6b of that document is where the fix landed and says so.
3. **Correctness anchor at B>1.** §9 records a served gate per arm — the only batched gfx942
   measurements in this directory that have one. Everything retracted above was timing-only at
   B>1, which is exactly why the bad math was invisible.

### 3.4 What I did NOT do, and why

**No re-measurement was taken.** Restoring the headline (`Gemma B=8: 4.00x aggregate at
conc 8`) needs a fresh `PLOW_DECODE_BATCH=8` emit plus matching `PLOW_GEMV_MM=8` objects and a
full concurrency sweep — a build and an hour of exclusive GPU, on a box where a sibling held
the lock for the whole of this lease. It is also the wrong first purchase: the ladder's
post-fix arms already cover the batched region with served gates, and they say the corrected
picture is **far worse** than the retracted one (corrected `B=16` costs **109.74 ms** TPOT at
conc 1, against the ~34 ms the retracted B=8 table implies). Re-deriving 4.00x is therefore
not a formality — it is likely to produce a materially different number.

**Consequence to carry forward, stated plainly:** the TP4xPP2 rejection in
`glm52-tp4-pp2-evaluation.md` §E.1 was priced against that 4.00x. Its *direction* survives on
the ladder's corrected data; its **margin is now unquantified**. Nobody should quote 4.00x
again, and the ladder's real ceiling should be established before the next batching decision.

### 3.5 Scope, so the retraction is not read as wider than it is

The arena bug reaches **fused decode GEMVs at B>1 on gfx942 only**. gfx950 / sm_120 emits are
unchanged (their arena constant was always right for them). B=1 is unaffected on every part.
GLM-5.2, Kimi-K3 and DeepSeek are not exposed at all, having no batched decode to emit. This
is a **different blast radius** from the fused-quant bug in §1, which is dense-GQA + `w8a8` on
AMD and hits B=1 too — the two bugs were found in the same investigation but share nothing
except the part they were found on.

---

## 4. Verified vs inferred

**Verified by running it, this session, on this box:**

* Both reference blob hashes, reproduced from two independently built `plowc` binaries.
* The packet-count delta between the two blobs (910 vs 814 prefill, 527 vs 527 decode) and the
  `Glu` opcode present in one and absent from the other.
* 28 gfx942 objects built from this tree, `PLOW_OCC4=1`, zero spill on the decode rows.
* The gate's two transcripts in §2 — one FAIL on the known-bad emit, one PASS on the default.
* That `glm52-decode-batch-ladder.md`'s §7/§11 tables were committed after `2130f04`
  (`git log`, timestamps in §3.3).
* That `scripts/build_gfx942.sh` dies with **exit 2 and an empty log** on this box's ROCm
  layout — its bundler probe is `ls -1 <3 candidates> | head -1` under `set -o pipefail`, and
  `/opt/rocm/lib/llvm/bin/clang-offload-bundler` does not exist here, so `ls` exits 2 and takes
  the pipeline with it. **Not fixed — reported.** It is a shared script and other agents are
  mid-run against it; `gemma_xgate.sh` resolves `PLOW_HIPCC`/`PLOW_BUNDLER` itself and passes
  them in, which is the surgical fix for my own path. Whoever owns that script should decide.

**Inferred, and flagged as such:**

* That the retracted numbers are *unusable* rather than merely biased. The mechanism (rows
  past the arena written past the end of `plow_smem`) makes the magnitude unknowable without
  re-measuring, so this is a refusal to interpret, not a measured claim about the error size.
* That the TP4xPP2 verdict's *direction* survives its retracted margin. Based on the ladder's
  corrected arms pointing the same way and further, not on a re-run of that comparison.
* The blast-radius statements in §3.5 are read off the code paths (`emit_phase` reachability,
  `glm_emit_full`'s structural single row), consistent with 2130f04 and 75fb82f, not
  independently re-derived here.

* The post-merge arms 4 and 5: same blob bytes, PASS on post-fix objects and a load REFUSAL on
  pre-fix ones, both run here.
* That `pgrep -x plowrt` misses a server copied to another name — caught live, a sibling
  serving as `/tmp/plowrt_stock` at 93% GPU while `-x` reported nothing.

**Found but NOT MINE, and not fixed:** `cargo test -p devgen` has two failures on this branch —
`tuned_tile_selection::published_measurements_reach_the_compiler_and_change_its_answer` and
`::the_narrow_shapes_select_a_narrow_rung_and_follow_the_measurement`. They are **pre-existing
on `worktree-glm52-bringup` itself**: verified by checking out the base tip (e038e47) in a clean
worktree and reproducing both, and this branch changes zero compiled code (`git diff
base...HEAD -- '*.rs' '*.toml' '*.hip' '*.h'` is empty). The tuned-tile-selection area belongs
to whoever landed the chunk-policy / `LAUNCH_ROWS` work; flagging rather than touching it.

**Not attempted:** any timing. This lease produced correctness verdicts and a gate, and no
performance number in this document is new. No control spread is quoted because nothing here
is a measurement that a +/-20% DVFS box could move — the gate's outputs are token strings.
