# SE-FINE-decode — per-slice fine-gate straggler recovery on DECODE (Gemma-4 12B + 31B)

**Verdict: NO-GO. SE_FINE recovers ~0% of the claimed 15.6% decode-straggler wait.**
On the fine-safe scheduler the forced-fine schedule is byte-identical to coarse
and **0.25–0.67% SLOWER**; on the production (global-queue) scheduler fine gates
break correctness outright. The 2.63 ms straggler wait dev.rs measures is real,
but it sits on GEMV→reduction barriers that SE_FINE structurally cannot narrow.

RTX PRO 6000 Blackwell (sm_120, 188 SMs, CUDA 13.0), TP1, batch 1, bf16.
Harness `gemma4_sm120_chat`, prompt `prompt_tpot.ids` (ctx 3603→3715), 128 gen
tok, 16 warmup discarded, **112 timed**, sd < 0.02 ms. Branch off main `b953a7b`.

## What the lever is

`StreamEnt.flags |= SE_FINE` makes an entry carry its OWN wait/succ lists, so
consumer slice *s* blocks only on the producer slices that feed it, instead of
the full N-way `blocks` barrier. The builder already emits fine maps; the
`collapse` theorem (`Plow/CounterGranularity.lean`) then downgrades every one to
coarse because per-slice **modelled** work is uniform (makespan provably
identical). The claim under test: real hardware has a *diffuse* straggler
(dev.rs "Per-slice gates": 256 identical gemv CUs finish across 9.6–16.6 µs,
2.63 ms/16.9 ms token) the model doesn't see, so keeping the gates fine could
recover it. `PLOW_FINE_FORCE=1` (added here) overrides the downgrade for
genuinely-sparse edges only, to measure the real delta.

## AUDIT — every decode gate, coarse/fine + recoverable

Per-layer producer→consumer gates for the gemv-family decode program. "Recoverable"
= consumer slice provably reads a strict SUBSET of producer slices (SE_FINE helps);
a full reduction over the contracted/feature axis needs ALL producers (SE_FINE gives
nothing).

| gate (producer → consumer) | class | recoverable? | why |
|---|---|---|---|
| Embed → input RmsNorm | coarse | no | 1-wg producer, nothing to narrow |
| RmsNorm → q/k/v GEMV | coarse | no | GEMV contracts K=hidden; each out col needs the WHOLE normed vector |
| **q/k GEMV → HeadNormRope** (FULL layers only) | **fine** | **YES** | head *h* = cols [h·hd,(h+1)·hd); GV_BLOCKED gemv → few producer wgs |
| GemvQkv → HeadNormRope (SLIDING layers, bf16) | coarse | no | fused q\|k\|v concatenation, per-head fan-in = all wgs (not GV_BLOCKED-mappable) |
| **HeadNormRope{q,k,v} → FlashDecode** | **fine** | **YES** | flash item (b,head,split) reads only its own head's headnorm wg |
| **FlashDecode → FlashMerge** | **fine** | **YES** | merge of (row,head) folds only the `ns` splits of that head (~8 of 256) |
| FlashMerge → o_proj GEMV | coarse | no | o_proj contracts K=qd (all heads) → needs every merge slice |
| o_proj → NormResidualNorm | coarse | no | RMS reduction over the full H feature axis |
| NormResidualNorm → GemvGlu | coarse | no | GLU gemv contracts K=hidden → needs whole normed vector |
| GemvGlu → down_proj GEMV | coarse | no | down contracts K=inter (all GLU lanes) → full reduction |
| down_proj → end-of-layer NormResidualNorm | coarse | no | RMS reduction over H |
| final RmsNorm → lm_head GEMV | coarse | no | contracts K=hidden → full reduction |

**Recoverable set = the attention subgraph only** (q/k→headnorm on full layers,
headnorm→flash, flash→merge). Everything else is a GEMV whose consumer contracts
over the full producer output, or an RMS reduction over the feature axis — both
genuinely need ALL producer slices, so SE_FINE gives nothing there **by
construction**. This is the crux: the 2.63 ms straggler dev.rs measured lives on
the big all-CU GEMV barriers (qkv/o/glu/down/lm_head), and those feed reductions.
SE_FINE is the wrong lever for a reduction.

Declared-fine edges (all pass the sparsity filter, all kept under `PLOW_FINE_FORCE=1`):

| model | layers | sliding×4 | full×7 | total fine edges | downgraded (default) |
|---|---|---|---|---|---|
| 12B | 48 (40 sl / 8 full) | 160 | 56 | **216** | 216 → coarse |
| 31B | 60 (50 sl / 10 full) | 200 | 70 | **270** | 270 → coarse |

(Per full layer: 3 gemv→headnorm + 3 headnorm→flash + 1 flash→merge = 7.
Per sliding layer: fused qkv is coarse, so 3 headnorm→flash + 1 flash→merge = 4.)

## Un-stub (what was stubbed like the UNISEG prefill trap)

- `runtime/nvidia/interp_sm120.cu`: the interp **trapped** on any `SE_FINE`
  entry ("coarse path only"). Now honors it: `wait/succ_{ofs,len}` come from the
  ENTRY when `SE_FINE`, from the instruction otherwise. `SE_XCTR` (cross-GPU)
  still traps — single-GPU decode never sets it.
- `runtime/tests/gemma4_sm120_chat.cu`: host guard rejected fine pkts; now only
  `SE_XCTR` is fatal.
- `crates/packet/src/devbuild.rs`: `PLOW_FINE_FORCE=1` keeps a fine edge iff a
  consumer slice waits on strictly fewer than all producer slices (isolates the
  recoverable gates; never pays the 256×256-atomic all-to-all cost). Default
  unset = byte-identical (0 fine edges emitted).

## RESULTS — decode TPOT ms/token (ctx 3603, 112 timed)

Scheduler is compile-time (`PLOW_NV_SCHED`). Production default = 1 (global-queue,
op-major reordered). The topological deadlock-safety argument in
The design notes hold for the **per-CU in-order** stream
(sched=0), NOT for a reordering scheduler — and that is exactly what we observe.

| model | sched | baseline (coarse) | SE_FINE forced | Δ | parity |
|---|---|---|---|---|---|
| 12B | 0 (per-CU, fine-safe) | 18.644 | 18.769 | **+0.67% (slower)** | **byte-identical** |
| 12B | 1 (gq, production) | 18.511 | 18.527 | +0.09% | **BROKEN** (garbage tokens) |
| 31B | 0 (per-CU, fine-safe) | 45.999 | 46.114 | **+0.25% (slower)** | **byte-identical** |

- **Parity (sched=0): byte-identical greedy tokens, both models.** The fine maps
  are correct; narrowing the wait set changed no output — proving no producer was
  dropped. (Confirms the recoverable edges are genuinely sparse and correctly mapped.)
- **TPOT: fine is 0.25–0.67% SLOWER.** The extra per-slice counter, the extra
  producer atomic, and the wider consumer wait list cost more than the diffuse
  straggler on the (small) attention subgraph saves. Net negative.
- **gq (production) + fine = wrong output.** The global-queue reorders the
  op-major stream across a single shared cursor; that is the "tile-interleaving
  scheduler" the deadlock doc warns kills the topological-order safety. SE_FINE
  is fundamentally incompatible with the production scheduler without the relay
  machinery that doc describes (not built).

## Did 31B flip? No.

31B decode is 1.4–5.4% behind vLLM (44.67 ms@1k → 55.46 ms@128k). SE_FINE makes it
**0.25% slower**, so it moves the wrong way. No flip. The 12B lead does not widen
either (fine is 0.67% slower). Both were bounded to fail by the audit before the
run: the recoverable edges are a small attention subgraph, and the dominant
straggler cost is on reduction-fed GEMV barriers SE_FINE cannot touch.

## Honest recovered vs claimed

- **Claimed recoverable: 15.6%** (2.63 ms / 16.9 ms).
- **Actually recovered: 0%** (−0.25% to −0.67%, i.e. slightly negative).
- The claim mis-targets the lever. The straggler wait is real, but it is the wait
  of reduction consumers on all-CU GEMV producers — irreducible, because the
  consumer reads every element of the producer's output. SE_FINE only narrows the
  attention subgraph (headnorm→flash→merge), which is too small a slice of the
  token for its diffuse straggler to beat the fine-gate bookkeeping overhead.

Recommendation: do not ship SE_FINE for decode. If straggler wait on the big
GEMV barriers is to be attacked, the lever is load-balancing / persistent-CU
scheduling of the GEMV itself, not per-slice gates on a reduction edge.
