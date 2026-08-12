# Kimi-K3 decode: folding the narrow `b=1` gates into their wide consumers

Follows `perf-data/archive/k3/k3-decode-counter-graph.md`, whose recommendation 2 was:

> **Fuse the narrow gates into their consumers.** `AttnRes`/`RmsNorm`/`MoeRouterTopk` at `b=1` in
> front of `b=256` consumers are 19% of counter traffic AND they lengthen the chain by one level
> each. Fusing removes the packet, the edge and the chain level together.

Taken for `RmsNorm`, **refused for the other two**, and one of that paragraph's two justifications
turns out not to survive measurement. Everything below is read off emitted blobs
(`plowrt disasm --program 1 --counters`) or measured on gfx950 under `gpulease`.

**STATUS: adopted and ON by default on gfx942 TP8 as of 2026-08-11.** Sections 1--5 preserve the
original implementation and its failed gate; §6 records the corrected current-kernel logit and
serving gates. The fan-out census, the arithmetic that rejects `AttnRes`, and the correction to
this doc's counter-traffic claim remain applicable.

---

## 1. Check N first — the rule that decides all three

`runtime/amd/op_gemm.h` had this fusion once, as GEMV `norm` mode 2, and deleted it: measured
**22.4 -> 24.4 ms/token on Gemma**, because the norm feeding attention has FIVE consumers, so
folding turned one shared reduction into five redundant ones. The rule it left behind:

> Fusion that duplicates a reduction across N consumers costs (N-1) extra reductions — check N
> first.

There are really TWO multiplicities and the Gemma note only names one. A fused producer is redone
once per consumer PACKET **and once per consumer WORKGROUP**, because workgroups cannot share a
reduction. The cost is `sum over consumers of consumer blocks`. Measured over the 93-layer TP8
decode blob, from full operand lists:

| producer | packets | fan-out | consumer WGs that each redo it | verdict |
|---|---|---|---|---|
| `RmsNorm` | 116 | 1 (x92), 2 (x24) | 309 mean | **TAKEN** |
| `AttnRes` | 187 | 3 (x161), 4 (x24) | 619 mean | refused |
| `MoeRouterTopk` | 92 | 2 (x92) | 512 mean | deferred |

The fan-out was read from full operand lists, never a truncated disasm — `k3.rs fuse_block_resid`
records what truncating one costs (a fluent wrong model, prefill token 261 instead of 17374).

### Why `AttnRes` is refused — it is a loss by arithmetic

It is the biggest prize on paper (187 chain levels, more than `RmsNorm` and the router together)
and it is the one that cannot be paid for. A GEMV stages ONE row of `x`; the `AttnRes` mix spans
`nb+1` rows of 7168, `nb` running 0->8 with depth (measured mean 4.36, so 5.36 rows). A fused
consumer workgroup has to read all of them:

    619 consumer WGs x 5.36 rows x 7168 x 2 B  = 47.6 MB per site
    x 187 sites                                = 8.9 GB per token of extra reads

against a token that already streams 57 GiB in 36 ms. It buys 187 levels ~ 1.07 ms and spends
several times that. It would also stage ~126 KB of LDS per workgroup, which on its own costs the
occupancy the GEMV depends on. Not attempted.

### Why `MoeRouterTopk` is deferred, not refused

The source doc lists ONE consumer for it (`MoeGroupGluFp8Blk`, 23,552 polls). The blob has **two**
— `MoeGroupDownFp8Blk` reads `route_tab` as well — so it is fan=2 and both consumers are `b=256`.
Folding replicates a top-16-of-896 selection into 512 workgroups per site and puts it on the
critical path of both. The re-read is small (896 f32 logits + bias, ~7 KB per WG), so unlike
`AttnRes` this is not obviously a loss — but it is a different kernel and it wants its own A/B.
92 levels are still on the table.

---

## 2. What was built

`RmsNorm(b=1) -> Gemv(b=256)`, at both K3 decode sites (behind `PLOW_K3_FUSE_NGEMV=1`):

    routed_expert_norm  feat=3584  fan=1  -> routed_expert_up_proj     x92
    q_a_layernorm       feat=1536  fan=2  -> q_absorb, q_rope          x24

92 of the 116 cost ZERO extra reductions. All 116 RMSNORM packets leave the decode program.

**Bit-exact by construction — and the isolated gate agrees, while the whole model does not (§4).** The deleted packet stored `f2bf(x*inv*gamma)`
to HBM and the GEMV re-read those bf16 values. So the fused arm does NOT multiply through the
k-loop the way `norm` mode 1 does — that would be a different number. It normalizes the
LDS-staged copy IN PLACE, rounding to bf16, walking the row with `d_rmsnorm`'s exact per-thread
element map and `block_sum`, and then runs the ORDINARY un-normed hot loop over it. The bytes the
k-loop reads are the bytes the unfused pair wrote. `RN_REG`/`RN_VEC` moved to `amd_common.h` so
the two paths cannot drift apart.

The hot loop is untouched, and the register canary confirms it: `interp_decode_fp8kv_k3_gq` still
builds at **248 VGPR / 0 AGPR / occ 2 / 0 spill**, the value `kimi-k3-README.md` §1 pins.

---

## 3. Static effect on the decode graph — measured (with the fold forced on)

Two blobs from the same binary, `PLOW_K3_FUSE_NGEMV` on and off, identical 5411-tensor table:

| | control | fused | delta |
|---|---|---|---|
| decode packets | 2459 | 2343 | **-116** |
| coarse edges | 2969 | 2853 | **-116** |
| **critical path (chain levels)** | **1831** | **1715** | **-116 (-6.3%)** |
| polls | 400,110 | 399,994 | -116 (-0.03%) |
| `RmsNorm` packets | 116 | 0 | -116 |

Every deleted packet deleted a chain level. That is the whole win, and it is the one the source
doc named.

### The poll claim does not survive

That doc's other justification — these gates are "19% of counter traffic", 151,598 polls — implies
fusing recovers that traffic. **It does not.** Polls fell by 116, i.e. only the deleted packets'
own single waits: 0.03%, not 19%.

The reason is structural and worth keeping: fusing a narrow gate into a wide consumer does not
delete the consumer's polls, it REDIRECTS them. The consumer previously polled the `b=1` norm with
all 256 workgroups; now it polls the norm's producer with all 256 workgroups instead. Same width,
same count. Poll traffic is a property of the CONSUMER's width and of how many edges it has, and
this fusion changes neither.

So the lever is the chain level, at ~5.7 us of measured per-packet protocol cost, and it should be
argued for on that basis alone. `perf-data/archive/k3/k3-decode-counter-graph.md` §"Free wins" recommendation
1 (transitive reduction) has since landed — the emitter now reports `69 ... 207 wait entries
removed` at emit time and the disassembler finds 0 remaining redundant edges.

---

## 4. The bug this found: `plow_smem` is a UNION, and the standalone test could not see it

Worth reading before writing any other op that wants a block reduction inside a GEMV.

`interp.hip`'s shared memory is one union:

```c
typedef union __align__(16) {
    float part[PLOW_WAVES];     /* block_sum scratch */
    bf16  gm[PLOW_GM_ARENA];    /* the GEMM/GEMV arena */
    ...
} plow_smem;
```

`sm->part` and `sm->gm` are **the same address**. The first version of this fold took `part` from
the caller — i.e. `exec_gemv(in, T, slice, nblk, sm->gm, sm->part)` — so `block_sum` wrote
`part[0..PLOW_WAVES)` straight through **the first 16 halves of the very activation row the fold
was reducing**.

It did not fail loudly. The write-back repairs those 16 halves from registers, so the damage is
partial and data-dependent, and the model stayed fluent: over 32 decode steps the token stream was
IDENTICAL and only the logits drifted. `op_collective.h`'s own note prices exactly this scale —
"a 1-bf16-ULP difference per element per layer ... 92 MoE layers of it moves the logits by ~0.03
and greedy decode then flips a token" — and the measured median was 0.047.

**The standalone test PASSED throughout.** It declared

```c
__shared__ bf16 lds[GM_LDS_HALVES];
__shared__ float part[PLOW_WAVES];   /* two separate arrays — NOT what production does */
```

so in the test the two buffers did not alias and the fused arm was genuinely bit-exact. A test
whose memory layout is friendlier than production cannot gate production. `k_gemv` in
`runtime/tests/gemv_fusednorm_gfx950_test.hip` now declares the arena as the interpreter's union.

The fix is not to pass a better pointer, it is to make the bad pointer unreachable: the scratch is
DERIVED from the top of the arena (`GV_NORM_SCRATCH` halves, past the row) and the parameter is
gone from `d_gemv`/`d_gemv_t` entirely, so no caller can hand the fold an aliasing buffer again.

### How it was localised

Whole-model A/B could not say which of the two rewrites moved the model, so the knob learned to
name one: `PLOW_K3_FUSE_NGEMV=lat|q`. At `--ctx 5`, against the same control dumps:

    q   (24 sites, K=1536)   33/33 logit files identical
    lat (92 sites, K=3584)   diverged from step 14

Two other controls were needed to read that at all, and both are worth keeping:

* **`--ctx N` decodes from position N over KV NOBODY EVER PREFILLED** (`plowrt/src/main.rs:391`).
  At `--ctx 32000` with a 5-token prompt every decode step attends over ~32k rows of uninitialised
  memory, so those step logits are not a function of the program alone. Correctness A/Bs must run
  at `--ctx <prompt length>`.
* **The same blob reproduces itself exactly** (33/33, at both ctx settings), so a difference
  between two blobs IS a real difference and not run-to-run noise. Establish this first; without
  it the divergence above is unreadable.

## 5. Measured decode cost

NOT MEASURED on this branch, and the reason is worth recording rather than papering over: every
timing window available was contended by another agent's K3 runs, and gpulease flagged it
(`foreign-before`/`foreign-during`) on every attempt. The same blob measured **32.977 and 38.479
ms/token** in two runs minutes apart — a 17% spread against an effect predicted at ~2%. No number
taken under that is worth publishing.

What the change is expected to be worth, from the static result and this tree's own per-packet
figure: 116 chain levels at ~5.7 us of protocol cost is ~0.66 ms/token, against a ~36 ms token.
`scripts/k3_block_sweep.sh` with `PLOW_K3_FUSE_NGEMV=0` as the control arm is the A/B to run on a
quiet box; use unbound-weight runs (`kimi-k3-README.md` §5 trap 2) since the effect is sub-1 ms.

## 6. Current gfx942 TP8 gate and adoption

The current interpreter derives the reduction scratch from the arena top, so the original
aliased pointer cannot be passed. Control and candidate were re-emitted from the same binary with
identical 5,411-tensor tables and the same ns64 packet/object settings. The only emit difference
was `PLOW_K3_FUSE_NGEMV=0` versus `=1`.

| Decode structure | Control | Fused |
|---|---:|---:|
| packets | 2,459 | 2,343 |
| counters | 61,475 | 58,575 |
| critical path | 1,831 | 1,715 |

Packet SHA256:

```text
e17025ab76237f6d7f5c6006a982be14e20803431f8bab49760b1051ece09805  control
d2b058e2a701ba182db6bea3228c2a04fbcf85759355759f1206695e61a3695d  fused
```

Two matched vLLM 0.27 `bench serve` cells used C1, actual input 149, output 512,
one warmup, TP8 compact exact counter audit, counter double buffering, device state clear,
FP8 MLA V2, and the same gfx942 object inventory:

| repetition | Control TPOT | Fused TPOT | Delta |
|---:|---:|---:|---:|
| 1 | 55.118 ms | 53.668 ms | -1.450 ms (-2.63%) |
| 2 | 54.994 ms | 53.663 ms | -1.331 ms (-2.42%) |

Every cell completed 1/1 with 512 output tokens and empty errors. Generated text is byte-identical
for each matched pair. Detailed JSON is retained under `/tmp/k3-ngemv-result/`.

The stronger gate used real weights, prompt ids `1008,10484,318,15383,387`, `ctx=5`, and dumped
rank 0's complete BF16 vocabulary row after prefill and each of 32 decode steps. All 33 vectors
are byte-identical: relative error zero, `maxabs=0`, and the same argmax at every step. All eight
ranks also emitted the same token stream. Evidence directories are
`/tmp/k3-ngemv-logits-{control,candidate}`.

Decision: enable the fusion by default for K3 B1 decode. `PLOW_K3_FUSE_NGEMV=0` preserves the
unfused control; `lat` and `q` retain the site-level bisect.
