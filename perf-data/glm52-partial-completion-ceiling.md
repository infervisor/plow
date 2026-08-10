# Pricing tile-granular partial-completion signalling, before anyone builds it

**GLM-5.2 TP4 decode, 4× gfx950 (MI355X), real weights, `--sweep 1024 --steps 65 --gen 24`.**
Measured 2026-07-28 on `worktree-readme-build-instructions` HEAD + the ported `PLOW_CHAIN_BYPASS`.

**§0-BENCH.** Every number here comes from the C harness (`runtime/tests/glm52_decode.c`), an
EXPERIMENT instrument. Nothing in this file may be placed next to a vLLM number.

---

## 0. THE QUESTION, AND WHY BYPASS ANSWERS IT

The proposal is **tile-granular partial completion**: a consumer starts on tile `(i,j)` while the
producer is still computing `(i,j+1)`, signalled by a published tile id rather than a counter
threshold. It targets the one pool nothing has touched — §7c's **true idle, 1.653 ms/CU, 18.9% of
all gate stall**, the moments when zero of 256 workgroups are executing a body.

**`PLOW_CHAIN_BYPASS` is the strict upper bound on any such scheme.** It splices an op out of the
dependency chain — every consumer waits on that op's own predecessors instead — while the op still
runs on the same workgroups. That is the limit of *"the consumer never waits at all"*, which no
partial-completion protocol can beat: partial completion still makes the consumer wait for tile
`(i,j)`, bypass makes it wait for nothing.

> **If bypassing an edge buys nothing, tile-granularity on that edge buys nothing either.**

The instrument costs one env var and no kernel work, which is exactly §6b-i's rule: price the
consumer's gate with a cheap, numerically-wrong build before building the real thing.

## 1. THE INSTRUMENT IS EXACT ABOUT WHAT IT CHANGES

`graphstat` on every emitted arm, GLM-5.2 TP4 decode program (T=1):

| arm | bypassed opcodes | ops | wg-packets | counter bumps | critical path |
|---|---|--:|--:|--:|--:|
| `ctl` | — | 2756 | 259,505 | 259,505 | 1400 (50.8%) |
| `b_resid` | 4 `Residual` | 2756 | 259,505 | 259,505 | 1244 (45.1%) |
| `b_rms` | 1 `RmsNorm` | 2756 | 259,505 | 259,505 | 1165 (42.3%) |
| `b_spine` | 1, 4 | 2756 | 259,505 | 259,505 | 1009 (36.6%) |
| `b_spine3` | 1, 4, 56 `MoeRouterTopk` | 2756 | 259,505 | 259,505 | 934 (33.9%) |
| `b_all4` | 1, 4, 56, 24 `XReduce` | 2756 | 259,505 | 259,505 | **778 (28.2%)** |
| `b_comb` | 43 `MoeCombine` | 2756 | 259,505 | 259,505 | 1325 (48.1%) |
| `b_xr` | 24 `XReduce` | 2756 | 259,505 | 259,505 | 1244 (45.1%) |
| `b_rope` | 3 `HeadNormRope` | 2756 | 259,505 | 259,505 | 1322 (48.0%) |

Packet count, workgroup-packet count and **atomic bump count are identical in every arm** — every
op unconditionally bumps its own counter (`devbuild.rs`, `succs.push(op.counter); succ_len = 1`),
so a bypassed op simply becomes a dead counter. The only delta is the wait lists.

**The control blob is `cmp`-identical to the shipping `/home/lava/models/glm52_tp/glm52_tp4_64k.pkt`
(md5 `e818c91b…`)**, which is the post-`xrfit` object — so the knob is provably inert when unset and
the control is provably the shipping program, collective already narrowed to 12 workgroups.

## 2. THE EDGE STRUCTURE — what could even consume partially

The GLM decode spine is a **strictly linear chain of single-dep packets** (`graphstat -V`):

```
Gemv(o_proj, 256) -> XReduce(12) -> Residual(1) -> RmsNorm(1) -> DenseGluFp8Blk(256)
   -> GemvFp8Blk(256) -> XReduce(12) -> Residual(1) -> RmsNorm(1) -> GemvQkv(256)
   -> RmsNorm(1) -> GemvQkv(256) -> HeadNormRope(256) -> FlashMlaDecode(256)
   -> MlaMergeFold(256) -> Gemv(o_proj, 256) -> ...
```

Per-op token attribution on the shipping object (traced step 27.213 ms, `glm52_token_attrib.py`):

| op | pkts | ms of token | %tok | dispatched wg | eff wg |
|---|--:|--:|--:|--:|--:|
| `Gemv` | 229 | 3.982 | 14.6% | 256 | 209.8 |
| `GemvQkv` | 156 | 3.782 | 13.9% | 256 | 195.2 |
| `MlaMergeFold` | 78 | 2.701 | 9.9% | 256 | 125.7 |
| `MoeExpertDownFp8Blk` | 600 | 2.607 | 9.6% | 32 | 28.8 |
| `FlashMlaDecode` | 78 | 2.592 | 9.5% | 256 | 204.5 |
| `MoeExpertGluFp8Blk` | 600 | 2.221 | 8.2% | 32 | 28.7 |
| `HeadNormRope` | 156 | 1.665 | 6.1% | 256 | 162.0 |
| `GemvGlu` | 75 | 1.403 | 5.2% | 256 | 197.2 |
| **`RmsNorm`** | 313 | **1.384** | 5.1% | **1** | 1.0 |
| `XReduce` | 156 | 1.158 | 4.3% | 12 | 9.6 |
| `MoeCombine` | 75 | 0.882 | 3.2% | 256 | 140.9 |
| **`MoeRouterTopk`** | 75 | **0.605** | 2.2% | **1** | 1.0 |
| **`Residual`** | 156 | **0.594** | 2.2% | **1** | 1.0 |

The 1-workgroup spine is **2.583 ms = 9.5% of the token**, and that is the a-priori cap on making
it free. Which of these edges a partial-completion scheme could actually exploit:

| edge (producer -> consumer) | partially consumable? | why |
|---|---|---|
| `XReduce` -> `Residual` | **yes** | elementwise; element `h` needs only element `h` |
| `Residual` -> `RmsNorm` | **partly** | `RmsNorm` can accumulate Σx² as tiles arrive, but cannot emit any output before the last one |
| `RmsNorm` -> `GemvQkv`/`GemvGlu` | **no** | `out = W @ x` needs the whole K-reduction of `x` (§6b-i) |
| `MoeExpertDown` -> `MoeCombine` | **yes** | `acc += part[j]` in fixed slot order — consuming slots in order preserves the bit-exactness obligation |
| `HeadNormRope` -> `FlashMlaDecode` | **yes** | head-parallel; head `h` needs only head `h` |
| `MlaMergeFold` -> `Gemv(o_proj)` | **no** | dense GEMV edge; §7a-CHAIN's C3 already measured **+0.123 ms**, i.e. negative value |


---

## 3. THE EDGE → CEILING TABLE

**One lease, twelve runs, control interleaved at positions 1 / 5 / 9 / 12:
26.715 / 26.759 / 26.771 / 26.735 — mean 26.745, sd 0.025.** That is the tightest control in this
campaign (lease 2 of the attribution run was sd 0.047), so every delta below is many times the
noise floor. Raw: `perf-data/glm52-bypass-ceiling-raw.txt`.

| # | arm | edge taken out of the chain | ms/token | **Δ vs control** | 24 gen ids |
|--:|---|---|--:|--:|---|
| 10 | `b_rope` | `HeadNormRope` -> `FlashMlaDecode` | 26.236 | **−0.509 (−1.9%)** | identical |
| 4 | `b_rms` | `RmsNorm` -> `GemvQkv` / `GemvGlu` | 26.463 | **−0.282 (−1.1%)** | diverges |
| 8 | `b_xr` | `XReduce` -> `Residual` | 26.643 | −0.102 | diverges |
| 7 | `b_comb` | `MoeCombine` -> `XReduce` | 26.658 | −0.087 | **identical** |
| 11 | `b_spine` | `RmsNorm` + `Residual` | 26.683 | −0.062 | diverges |
| 3 | `b_resid` | `Residual` -> `RmsNorm` | 26.737 | **−0.008 (nothing)** | **identical** |
| 2 | `b_spine3` | + `MoeRouterTopk` (all 1-wg spine) | 26.800 | **+0.055** | diverges |
| 6 | `b_all4` | + `XReduce` (all narrow producers) | 27.491 | **+0.746** | diverges |
| 1,5,9,12 | control | — | **26.745** (sd 0.025) | — | ref |

**gpulease flagged `foreign-before` and `foreign-during` on this lease and both are the known false
positive** (§0a: ROCR device index != `rocm-smi` card index, so the audit compares lease ids against
different card labels). The evidence is the control itself: four positions spanning **0.056 ms**
across 42 minutes, straddling both warnings.

## 4. WHAT IT SAYS: DO NOT BUILD TILE SIGNALLING

### 4.1 The elementwise spine ceiling is ZERO, and the reason is structural

`b_resid` — the purest case in the proposal, an elementwise producer feeding an elementwise
consumer — is **−0.008 ms, inside a 0.025 ms noise floor, and its 24 generated ids are
byte-identical to the control's** on an arm that is numerically wrong by construction.

That identity is the finding. **`crates/devgen/src/mla.rs` places every 1-workgroup packet on CU 0**
(`let one = vec![0u32]`, six sites), so `XReduce -> Residual -> RmsNorm -> MoeRouterTopk` are
consecutive entries in **one CU's own stream**. CU 0 executes them in program order whether or not
a gate exists, so the gate between two 1-workgroup packets is redundant with stream order — removing
it cannot let anything start earlier, and the consumer still reads correct data.

> **A consumer cannot start on tile `(i,j)` "early" when it is queued behind the producer on the
> same CU. Partial-completion signalling has nothing to signal across.**

Widening the spine does not rescue this: §7c already measured `GLM_SPINE_CUS=32` recovering only
0.178 of `Residual`'s 0.577 ms, because the consumer then waits on a max over 32 stragglers instead
of 1 (§6b-i, third confirmation).

### 4.2 The two edges worth more than 0.25 ms cannot be reached by partial completion

* **`b_rms` −0.282 ms.** `RmsNorm`'s consumers are `GemvQkv` / `GemvGlu`, and `out = W @ x` needs
  the whole K-reduction of `x`. A GEMV workgroup cannot start on half a normalised vector without
  being restructured into partial-sum accumulation — a much larger change than tile signalling, and
  one that must fund itself out of 0.282 ms (1.1% of the token) at **zero implementation cost**.
* **`b_rope` −0.509 ms**, the largest ceiling measured, is **slice-granular, not tile-granular**.
  `HeadNormRope` is head-parallel and `Dep::Fine` already expresses exactly this ("slice `s` blocks
  only on the producer slices that actually feed it", `dev_isa.h` `PLOW_SE_FINE`), with
  `PLOW_FINE_FORCE=1` as the standing override for `CounterGranularity.collapse`. **No new tile id,
  no new ABI field, no new protocol is needed to try it.** What it removes is the §6a straggler
  barrier — a max over 256 rope workgroups — not a chain bubble.

### 4.3 Ceilings do not compose, and they can change sign

| bypass set | Δ |
|---|--:|
| `RmsNorm` | −0.282 |
| + `Residual` | −0.062 |
| + `MoeRouterTopk` | **+0.055** |
| + `XReduce` | **+0.746** |

Each addition makes it worse, and the full set is **0.75 ms SLOWER than the control**. Unchaining
the 12-workgroup collective from its 1-workgroup consumer makes them contend instead of overlap —
the same shape as §7a-CHAIN's C3 (`FlashMerge` -> `o_proj`, ceiling **+0.123 ms**) and §6b-i's L1
kill. This also serves as the positive control for the instrument: it moves the token in **both**
directions, so a null result on the spine arms is a real null, not an insensitive probe.

### 4.4 The bottom line

| candidate edge | ceiling | partial-consumable? | verdict |
|---|--:|---|---|
| `Residual` -> `RmsNorm` | −0.008 | yes | **dead** — gate is redundant with CU-0 stream order |
| `MoeCombine` -> `XReduce` | −0.087 | yes | **dead** — 0.3% of token at zero cost |
| `XReduce` -> `Residual` | −0.102 | yes | **dead** — `xrfit` already took its cost out |
| `RmsNorm` -> GEMV | −0.282 | **no** (dense K-reduction) | not reachable by this mechanism |
| `HeadNormRope` -> `FlashMlaDecode` | −0.509 | yes, but per-SLICE | **use `Dep::Fine`, which exists** |
| all narrow producers together | **+0.746** | — | actively harmful |

**Every edge on which a consumer could genuinely start early prices at ≤0.11 ms — under 0.4% of the
token — with implementation cost set to zero. Tile-granular partial-completion signalling should not
be built.** The one number worth following up needs none of it: price `PLOW_FINE_FORCE=1` restricted
to the `HeadNormRope -> FlashMlaDecode` edge against its −0.509 ms ceiling.

### 4.5 Two warnings for whoever picks up the `HeadNormRope` edge

1. **`b_rope`'s identical tokens are NOT evidence that dropping the barrier is safe.** `FlashMlaDecode`
   slice `s` does not provably read only what `HeadNormRope` slice `s` wrote, and `interp.hip:1388`
   records the workgroup-scope acquire firing at a **100% stale-read rate** in a stress harness while
   the full model still produces correct tokens — because the 57 GiB/token weight stream evicts stale
   lines first. That is a hardware accident. The safe form is a `Dep::Fine` map that names the
   producing slices, not a removed edge.
2. **`dev_isa.h`'s `PLOW_SE_FINE` deadlock argument depends on stream order**: "for any dependency
   A -> B EVERY slice of A precedes EVERY slice of B in EVERY CU's stream… the moment a scheduler is
   added that INTERLEAVES tiles across ops, that argument dies." A tile-granular scheme is exactly
   that interleaving, and the design notes it points at **do not
   exist in this tree**. Slice-granular `Dep::Fine` stays inside the proof; tile-granular does not.
