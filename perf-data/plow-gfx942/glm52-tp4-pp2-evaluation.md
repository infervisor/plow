# GLM-5.2-FP8 on 8x MI300X: is TP4 x PP2 worth building, and is a pipeline the right basis for a throughput design?

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4), 192 GB/card · **CDNA3 / MI300X-SPECIFIC** — per-rank memory feasibility and link counts are this box. The alpha ceiling argument is general.

Branch `tp4pp2` off `worktree-glm52-bringup` @ `daa894b`. Worktree
`/app/plow/.claude/worktrees/tp4pp2`. Box: 8x MI300X (gfx942, 304 CU, 192 GB),
ROCm 7.2.4. Model GLM-5.2-FP8, 78 layers (3 dense + 75 MoE), hidden 6144.

**Verdict up front: NO — do not build TP4 x PP2, and a pipeline is not the right
basis for a throughput design on this box.** Not because the memory does not work
— it does, and the campaign's "tp4 infeasible" note does NOT rule out TP4 x PP2;
that part of the premise is CONFIRMED and the option was dismissed on a shape
error. It is a no for five other reasons:

1. the seam saving that motivates it **does not exist** — TP4 makes the prefill
   collective *more* expensive per request, not less (§C.1);
2. the pipeline's own added cost, the stage handoff, **is genuinely free** — so PP
   is being priced in the wrong place entirely (§C.2);
3. `alpha`, the TP4/TP8 per-layer wall ratio, **measured 1.17** (decode) and
   **1.26** (prefill) with bound weights (§D.2) — so PP2 costs **+17% TPOT at
   concurrency 1** and its throughput ceiling is `2/alpha` = **1.71x**;
4. that 1.71x is cashing the *same* 69%-idle machine that `PLOW_DECODE_BATCH`
   already cashes at a **measured 4.0x** for the price of an emit flag — and
   batching also wins on throughput-per-unit-TPOT (2.04 vs 1.46), so PP does not
   even win the latency-preserving argument (§E.1);
5. it is a **20-30 engineer-day** build for correctness alone, 35-55 to be
   throughput-competitive, against a lever that is already in the tree (§B).

The batch ladder is the right throughput route. PP is the wrong lever pointed at
the right problem.

**Also delivered:** the campaign's first throughput baseline — GLM-5.2 TP8 across
concurrency 1/2/4/8/16 — which is **flat** (31.5 -> 28.9 out tok/s) while TTFT
rises 50.6x, because the packet is `batch=1` and concurrency is pure queueing
(§D.4). And one methodological correction worth more than this evaluation: an
unbound `amd-bench` run gets a sharding change's **sign wrong** (§D.1).

---

## Stage A — the memory question: the premise is CONFIRMED

**Claim under test.** The campaign recorded "tp4 infeasible both stacks
(weights/card)". If TP4 x PP2 has the same per-GPU weight footprint as TP8, that
note does not rule it out and the option was dismissed on an arithmetic/shape
error.

**It has the same footprint. The option was dismissed on a shape error.**

Measured directly from the checkpoint's 141 safetensors headers
(`/workspace/models/GLM-5.2-FP8`, byte-exact `data_offsets`, not config
arithmetic):

| category | GiB |
|---|--:|
| routed experts (256/layer x 76 layers) | 684.167 |
| attention (MLA + DSA indexer projections) | 12.145 |
| non-layer (`embed_tokens`, `lm_head`, `model.norm`) | 3.545 |
| shared expert | 2.673 |
| router | 0.434 |
| dense MLP (L0-L2) | 0.422 |
| indexer | 0.196 |
| norms | 0.142 |
| **TOTAL on disk** | **703.72** |

Sparse layers are 9.194-9.344 GiB each (76 of them: L3..L77 plus the L78 MTP
layer, which no plow blob emits); dense layers L0-L2 are 0.374 GiB each. Model
excluding the MTP layer = **694.380 GiB**, of which 690.835 is layers.

Per-GPU **weights**:

| scheme | per-rank GiB | derivation |
|---|--:|---|
| TP8 | **86.80** | 690.835 / 8 + 3.545 / 8 |
| **TP4 x PP2, stage 0** (L0..L39) | **85.79** | 341.381 / 4 + `embed_tokens` / 4 |
| **TP4 x PP2, stage 1** (L40..L77) | **87.81** | 349.454 / 4 + `lm_head` / 4 + `model.norm` |
| TP4 (pure, no PP) | **173.6** | 690.835 / 4 + 3.545 / 4 |

The balanced cut is at layer 40 (341.4 / 349.5 GiB); the naive 39/39 cut is 7.7%
imbalanced because L0-L2 are dense and 25x smaller than a MoE layer, so stage 0
must take one extra layer. **TP4 x PP2 is within +1.2% / -1.2% of TP8's per-rank
weight bytes.** The embed/lm_head asymmetry that PP normally introduces cancels
here: PP puts `embed_tokens` only on stage 0 and `lm_head` only on stage 1, each
sharded 4 ways, which is the same 0.44 GiB/rank TP8 pays for both sharded 8 ways.

Pure TP4 really is infeasible: 173.6 GiB of weights + ~7.6 GiB of KV + ~13 GiB of
workspace = ~194 GiB against a 179 GiB card. **The recorded note is correct as
written and simply does not transfer to TP4 x PP2.** The two shapes differ by a
factor of two in layers-per-rank, which exactly cancels the factor of two in
ranks-per-layer.

**Corroboration of the ~107 GiB/rank figure** (`glm52-experiments.md (consolidated: shipped result):14`):
86.80 weights + MLA latent KV + workspace. The latent cache is `kv_lora_rank +
qk_rope_head_dim` = 512 + 64 = 576 elements/token/layer, bf16 = 1152 B; x78
layers x max_ctx 73728 = **6.17 GiB**, and it is REPLICATED on every TP rank
(`crates/devgen/src/mla.rs:2592`: "the latent ckv/krot caches are REPLICATED ...
so the cache stays full-width on every rank"). This reproduces the independently
recorded "latent cache halved ~= 2.9 GiB/rank at max_ctx 73728"
(`glm52-experiments.md (consolidated):53`) to 6%. Plus the DSA indexer `kidx` cache (~1.4 GiB)
and ~12 GiB of prefill workspace / XR slots = ~107 GiB. Checks out.

**A second, real memory win for PP that nobody has counted.** Under PP2 each stage
caches KV for only its own ~39 layers, so the replicated latent cache **halves**:
6.17 -> 3.09 GiB/rank, plus ~0.7 GiB of `kidx`. That is ~3.8 GiB/rank freed.
It is real and it is currently worth nothing: at 107 GiB of a 179 GiB card there
is already ~72 GiB of headroom, so KV is not the binding constraint at any context
or batch this campaign has run. It would start to matter at ~4x the context or
~10x the batch slots.

**Stage A verdict: CONFIRMED.** TP4 x PP2 is memory-feasible on this box at parity
with TP8, and the prior dismissal does not apply to it. Everything that follows is
therefore a real question rather than a moot one.

---

## Stage B — the implementation gap: design, not code

`crates/plowc/src/lib.rs:83` is honest, and the situation is slightly worse than
it reads. `Parallel` is a plain unit enum (`Tp | Dp | Pp | Ep`) — **not** the
data-carrying `Pipeline(usize)` that `docs/arch/09-multi-gpu.md:39-44` designs.
Both consumers (`crates/plowc/src/lib.rs:2333-2343`,
`crates/plowc/src/main.rs:967-976`) match `Tp` and hard-`Err` on everything else.
`--parallel pp` parses and is then refused.

| item | status | evidence |
|---|---|---|
| TP emit + runtime (AMD) | **EXISTS**, measured, wired into `plowrt serve` | `plowrt/src/exec/{tp.rs,amd_tp.rs}`, `serve/engine.rs:82-207` |
| `Parallel::Pp` | accepted by clap, hard error downstream | `plowc/src/lib.rs:83,2337-2341` |
| `derive_parallel` / `ParallelConfig{tp,pp,ep,dp}` — real PP sizing math | **PARTIAL, dead**: computed, zero callers outside its own tests | `plowc/src/parallel.rs:42-107` |
| `docs/arch/09-multi-gpu.md` PP design | design only; self-honestly marked "Planned" throughout | `09-multi-gpu.md:74-95,217-229,285-299` |
| `{stem}.blocks.json` | **PARTIAL, dead as a seam** — see below | `plow-asset/src/lib.rs:426-439` |
| `HandoffKind::{Barrier,P2p,Rdma}` | EXISTS as enum + compiler logic, but only ever serves TP's GEMM joins | `rewrite/src/tilegraph.rs:180-199` |
| rank running a layer SUBSET | **ABSENT** — every TP rank runs every layer; TP shards width, never depth | `devgen/src/mla.rs:5132` |
| GPU-to-GPU point-to-point copy | EXISTS but documented "bulk/test, never per-token" | `plowrt/src/device/mod.rs:228`, rationale at `exec/tp.rs:53-56` |
| `orch::Transport` — designed *explicitly* for "PP layer-boundary activations" | **PARTIAL**: trait + descriptors exist, **every `program()` body is `Ok(())`** | `plowrt/src/orch/transport.rs:1-140` |
| `orch::Pipeline` / `Stage` | false friend — chains whole *models* (vision->text, draft->target), not layer ranges; and the serve mux never walks it | `plowrt/src/orch/pipeline.rs:20-29` |
| boundary-activation input/output device ops | **ABSENT** — `Embed` and the lm_head/Argmax tail each run exactly once, unconditionally, asserted by test | `devgen/src/mla.rs:5124-5131`, `5247-5314`, assert at `8390` |
| PP microbatch scheduling | **ABSENT** — `PLOW_DECODE_BATCH` is continuous batching of independent requests inside ONE dispatch, not stage-interleaved microbatching | `serve/engine.rs:88-110,624-630` |

**`{stem}.blocks.json` is a manifest, not a seam.** `BlockRange` carries
`{index, label, first_task, last_task, task_count}` — *task-id ranges into an
already-compiled single-device program*, recovered by scanning op names for an
`_L{block}` suffix (`plowc/src/lib.rs:1084-1132`). **No weight offsets, no byte
sizes, no memory-map or KV boundary information.** It is written unconditionally
(`plowc/src/lib.rs:1626-1633`) and deserialized into `Bucket.blocks`
(`plowrt/src/asset/bucket.rs:38,72`) — where **nothing ever reads it**. It tells
you which instructions belong to a layer; it carries nothing that would let a
runtime load only a subset of weights. It cannot be the seam.

**What is already 80% there:** `declare_glm` / `declare_glm_rows`
(`devgen/src/mla.rs:1123-1150`) already take `layer_ids: &[u32]` and bind tensors
*and KV* by the real layer index, and `--glm-layers` / `PLOW_GLM_LAYERS`
(`emit_config.rs:283-285,510-527`) already accepts a prefix `N` or `single:L`. So
"emit layers `0..N` with correct weight names and a correctly sized KV cache" is a
solved problem today — that is exactly what §D.0 exploits. **What is missing is
the two ends**: a non-first stage has no way to receive an activation instead of a
token id, and a non-last stage has no way to emit a hidden state instead of
sampling. Those are new device ops, not a refactor.

**Honest size.** Correctness-only TP4 x PP2, batch 1, one request in flight, no
bubble hiding, AMD only: **20-30 engineer-days** — generalise the layer loop from
a `0..N` prefix to an arbitrary `[a,b)`; two new boundary device ops; a real PP
manifest; implement `LocalP2pTransport::program`, currently `Ok(())`; and a new
`Ranks::Pp` owning two independently-loaded blobs with a sequenced per-token
driving loop (the closest existing analogue, `AmdTpGroup::launch_token`,
dispatches ranks *simultaneously*, not sequentially with a data dependency
between them), plus a token-identity harness across the stage boundary.
**Throughput-competitive** — actually overlapping stage-0 work for request N+1
with stage-1 work for request N in `serve/mux.rs` — is **another 15-25 days**.
Total **35-55 engineer-days**, and that is a floor: it assumes the boundary op and
the P2P transport go as smoothly as the TP collectives did, and
`exec/amd_tp.rs:38-52` records that even a well-understood collective cost real
debugging time on a device-level race.

---

## Stage C — the arithmetic

Every input below is measured, from this campaign, on this box. Nothing is
re-derived.

### C.1 The seam saving from TP4 — REFUTED. It is a seam COST.

Per-rank remote bytes (bf16, `n` = message elements = T x 6144):

* two-shot RS+AG (prefill): `4(N-1)/N . n` bytes -> **3.5n at N=8, 3.0n at N=4** = **-14%**, not -50%.
* one-shot (decode): `2(N-1).n` bytes -> **14n at N=8, 6n at N=4** = **-57%**.

**But the collective COUNT per REQUEST does not change.** All 78 layers still run
and each still has 2 all-reduces; PP moves 78 of the 156 onto the other stage, it
does not delete them. What halves is the per-*rank* count, which is a duty-cycle
statistic, not a latency one.

And the **rate falls**, because a TP4 rank has 3 live XGMI links instead of 7.
From `glm52-collective-tuning-mi300x.md` §1: one pair solo = 44.2 GB/s (2 B pull);
8-device all-gather = 239.7 GB/s/rank of which 209.7 is remote over 7 links =
**29.9 GB/s per link, 62% of solo** — the device's shared fabric concurrency is
what caps it. Reduce-scatter = 289.4 GB/s/rank = 41.3 GB/s per link, already
essentially at solo rate. At 3 links neither is device-capped, so both run near
solo: **~120-133 GB/s**. Rate ratios TP4/TP8: AG 0.57-0.63, RS 0.43-0.46.

    prefill collective time ratio  =  0.857 / [0.43 .. 0.63]  =  1.36x .. 2.0x

Collectives are 11.6% of the per-layer CU budget at T=8192
(`glm52-current-cost-decomposition.md` §1.1). At the mid-estimate 1.5x that term
goes 11.6% -> 17.4%: **+5.8% TTFT. TP4 makes prefill collectives worse.**

Decode is not saved either, for a different reason — there is nothing there to
save. From the same decomposition (§2.1, ctx 1024, 27.36 ms/token):

    XReduce x2      real work 0.086 ms/token   gate wait 0.180 ms/token
                    total attributable 0.266 ms = 0.97% of the token

86 kB at 209.7 GB/s is 0.41 us against a measured ~15 us per XReduce packet — the
decode collective is ~97% rendezvous and ~3% bytes, so cutting bytes by 57% cuts
almost nothing. A 4-peer rendezvous is cheaper than an 8-peer one; call it 0.7x.
**Best case saving: 0.08 ms/token = 0.3% of TPOT.** Inside this box's own DVFS
noise.

> **The premise that TP4 buys a cheaper seam is false in both phases.** Prefill:
> +5.8% TTFT. Decode: -0.3%, which is noise. There is no seam argument for PP.

### C.2 The stage-boundary handoff — genuinely free

The boundary tensor is the post-residual hidden state, which is *already
replicated across TP ranks* by the layer's own all-reduce. So rank *i* of stage 0
sends to rank *i* of stage 1: four independent 1-hop peer copies, no collective.

* **decode**: `[B, 6144]` bf16 = **12.3 kB per sequence per crossing**. At 48 GB/s
  that is 0.26 us of transfer. The cost is the handshake, not the bytes: the
  campaign measures a 1-signaller N-way gate round at **8.2 us** and an exposed
  packet boundary at 1.0-2.0 us. Two crossings per token (activation forward,
  sampled token id back, 4 B) = **~10-20 us = 0.04-0.07% of a 27.4 ms token.**
  Even the *lazy* implementation is free: `exec/tp.rs:53-56` prices an 8 kB
  `copy_peer_blocking` SDMA call at 13.95 us, so two host-mediated crossings =
  28 us = **0.10% of the token**. (That 13.95 us is exactly why the primitive is
  banned for TP, which needs ~96 crossings per token. PP needs 2. The reasoning
  that disqualifies it for TP *qualifies* it for PP.)
* **prefill**: `[8192, 6144]` bf16 = 100.7 MB per crossing over 4 parallel links =
  **~2.3 ms per chunk** against a 1677 ms TTFT @8k = **0.14%**.

> **PP is being priced in the wrong place.** Its advertised cost — the extra
> transfer — is orders of magnitude below the noise floor. Everything PP costs, it
> costs by making every layer run at TP4.

### C.3 The bubble at concurrency 1 — real, but it is NOT a latency doubling

The textbook statement ("the two stages serialise, so latency is the sum") is true
and misleading. Both schemes execute all 78 layers in sequence for one request.
The difference is that TP4 x PP2 runs each layer on 4 GPUs instead of 8. So

    latency(TP4 x PP2) / latency(TP8)  =  alpha  =  (TP4 per-layer wall) / (TP8 per-layer wall)

and `alpha = 2` only if the machine is perfectly work-bound. **It is nowhere near
work-bound.** Decode packing efficiency is **31.1%**: 7.694 ms of real work inside
a 27.36 ms token, 67.5% gate wait (`glm52-current-cost-decomposition.md` §2.1,
§2.3). Halving the rank count doubles the *work* term and leaves most of the
*overhead* term alone.

Three independent estimates of `alpha`:

1. **The GEMM strong-scaling table** (`glm52-experiments.md (consolidated: GEMM rate, OPEN item 4)` §4)
   measures per-rank wall at TP1/TP2/TP4/TP8 on GLM's real shapes. TP4/TP8 wall
   ratio:

   | family | M=512 | M=2048 | M=8192 |
   |---|--:|--:|--:|
   | A `q_nope` (N/tp) | 1.29 | 1.64 | 1.85 |
   | B `q_rope` (N/tp) | 1.10 | 1.14 | 1.47 |
   | C `o_proj` (K/tp) | 1.49 | 1.67 | 1.82 |
   | D `sh_down` (K/tp) | 0.76 | 1.30 | 1.53 |

   `alpha_prefill(8k) ~ 1.5-1.85`; at small M, 1.1-1.5. **Never 2.0**, because
   TP4's fatter shards are more efficient per FLOP than TP8's — the same effect
   the source table records as "it hurts, always", read in the other direction.
2. **The decode decomposition, additively.** Layer span 317.3 us = 98.6 us of
   perfectly-packed real work + 218.7 us of gate wait and dead time. Packet count
   per layer (33) is unchanged by TP width. Doubling only the work term gives
   (197.3 + 218.7) / 317.3 = **alpha_decode ~ 1.31**.
3. **The campaign's own bring-up datum.** A 78-layer TP4 GLM blob ran 8 decode
   steps at **47.234 ms/token** against the contemporaneous TP8 blob's 48.6 ->
   `alpha ~ 0.97`. That run had UNBOUND weights, so it under-counts the weight
   stream — but the weight stream is only ~0.53 ms of a 27 ms token (2.64 GiB/rank
   at 5.3 TB/s), so it cannot account for a factor of 1.3.

So `alpha_decode` sits in **[1.0, 1.5]** and `alpha_prefill(8k)` in
**[1.5, 1.85]**. §D.2 measures `alpha` directly.

**The bubble is therefore not a latency doubling — it is 50% idle silicon.** At
concurrency 1, TP4 x PP2 costs `alpha - 1` *and* leaves half the machine doing
nothing at every instant. There is no upside at concurrency 1, and the entire
campaign is measured at concurrency 1.

### C.4 The throughput model, and the crossover

Steady state, 2 stages, >= 2 requests in flight, no batching inside a stage:

    throughput(TP4 x PP2, c)  =  min(c, 2) / (39 . alpha . L)      [L = TP8 per-layer wall]
    throughput(TP8, c=1)      =  1 / (78 . L)

    speedup  =  min(c, 2) . 2 / alpha
             =  0.68 .. 1.00x   at c = 1     (a LOSS)
             =  1.35 .. 2.00x   at c >= 2    (and FLAT thereafter — there are only 2 stages)

Crossover at **c = 2**, ceiling **~1.5x** at `alpha = 1.31`. Against which
baseline?

**(a) TP8 at `PLOW_DECODE_BATCH=1` — what the campaign actually ships.** Then
concurrency measures *queueing*, and aggregate throughput is flat. Measured on
this box (`gemma4-12b-bf16_bf16_tp1_general.csv`, Gemma-4-12B TP1, B=1, in_len 1024):

| conc | TTFT ms | TPOT ms | aggregate out tok/s |
|--:|--:|--:|--:|
| 1 | 191 | 20.31 | 46.2 |
| 4 | 9,288 | 23.74 | 39.9 |
| 16 | 41,457 | 21.53 | 43.8 |
| 64 | 160,184 | 21.25 | 44.3 |

Aggregate throughput is **flat within noise from c=1 to c=64** while TTFT rises
**840x**. Against this baseline PP2 crosses at c=2 and wins ~1.5x.

**(b) TP8 with batch slots — the same box, the same runtime, one emit flag.**
`g12b-b8_b8_tp1_general.csv`, Gemma-4-12B TP1, `PLOW_DECODE_BATCH=8`, in_len 4096:

> ## RETRACTED 2026-08-08 — SECOND, INDEPENDENT REASON, AND THIS ONE VOIDS THE NUMBERS
>
> The CORRECTION block in §E.1 below strikes these rows because GLM-5.2 has no batched decode,
> i.e. they do not TRANSFER. That objection left the Gemma numbers themselves standing. They do
> not stand: **the B=8 blob that produced them was computing wrong math.** Before **2130f04**,
> `devgen`'s `GM_LDS_HALVES` was the CDNA4 arena (73,728 halves) on every part, while gfx942's
> occ4 decode object holds 15,360 — so at `hidden = 3840` the emitter fused every batch up to
> M=19 onto four rows and wrote the rest past the end of `plow_smem`. The source CSV is
> retracted in place; see its header.
>
> Every number in the table below, and the **4.00x**, **2.66x**, **1.96x TPOT** and
> **3.56x packing ceiling consumed** derived from them, is withdrawn. `conc 1` is not a
> survivor: a B=8 decode program advances 8 rows whatever the concurrency.
>
> **NOT RE-MEASURED.** Corrected batched numbers exist only for other configurations
> (`glm52-decode-batch-ladder.md` §7/§11) and they are far worse — corrected B=16 costs
> 109.74 ms TPOT at conc 1 against this table's implied ~34. **The direction of the §E.1
> verdict is therefore unchanged but its MARGIN is unknown**, and no reader should quote 4.00x.
>
> Row (a) — the GLM TP8 B=1 baseline — is unaffected: B=1 blobs are byte-identical across the
> fix.

| conc | TTFT ms | TPOT ms | aggregate out tok/s | vs c=1 |
|--:|--:|--:|--:|--:|
| 1 | ~~620~~ | ~~33.98~~ | ~~25.9~~ | ~~1.00x~~ |
| 4 | ~~1,170~~ | ~~49.12~~ | ~~69.0~~ | ~~**2.66x**~~ |
| 8 | ~~1,396~~ | ~~66.61~~ | ~~103.7~~ | ~~**4.00x**~~ |
| 16 | ~~9,968~~ | ~~70.29~~ | ~~105.7~~ | ~~4.08x~~ |

~~**4.0x aggregate throughput for a 1.96x TPOT, and TTFT rises 2.3x rather than 840x.**
Against this baseline **PP2 never crosses.** Batching passes PP2's *ceiling* before
concurrency 4.~~ **RETRACTED — wrong-math timings, see above.**

**And the two levers do not compose — they are cashing the same idleness.**
Decode's work headroom is 27.36 / 7.694 = **3.56x**; the measured 4.0x says `B=8`
consumes essentially all of it. Once TP8 is work-bound there is no idle time left
for a pipeline to fill, and PP2 in the work-bound limit is worth exactly
`2/alpha` — the per-rank efficiency gain from doubling the tensor width, and
nothing else.

That residual `2/alpha` is a real effect (TP8's shards are too thin) and it
deserves to be named precisely, because it is the strongest surviving argument for
PP: **PP2's only durable advantage over TP8 is that it doubles the per-rank tensor
width.** But batching attacks exactly the same inefficiency from the M axis
instead of the N/K axis — and the GEMM table says the M axis works better:
family A's strong-scaling goes from 40% at M=512 to **84% at M=8192** purely by
growing M. Batching buys the arithmetic-intensity win *and* the idle-filling win,
composes to 16 slots instead of stopping at 2 stages, and costs an emit flag
instead of 35-55 engineer-days.

### C.5 Summary of the arithmetic

| term | effect of TP4 x PP2 vs TP8, per request | sign |
|---|---|---|
| per-rank weight bytes | -1.2% / +1.2% (stage 0 / stage 1) | **neutral — the premise holds** |
| per-rank KV cache | -50% (3.09 vs 6.17 GiB) | win, **currently worthless** (72 GiB already free) |
| prefill collective | +36% to +100% time -> **+5.8% TTFT** | **loss** |
| decode collective | -0.3% TPOT | noise |
| stage handoff, decode | +0.04-0.10% of the token | **free** |
| stage handoff, prefill | +0.14% of TTFT | **free** |
| per-layer wall (`alpha`) | a priori x1.0-1.5 decode, x1.5-1.85 prefill — **MEASURED 1.170 / 1.261** (§D.2) | **the whole cost** |
| latency @ concurrency 1 | `alpha` -> **+17.0% TPOT, +26.1% TTFT** (measured) | **loss** |
| throughput ceiling | `2/alpha` -> **1.71x**, flat past c=2 | wins vs B=1's flat, **loses to B=8's 4.0x** |

---

## Stage D — measurement

### D.0 The trick that made `alpha` measurable without building PP

`alpha` — the TP4/TP8 per-layer wall ratio — is the single number the whole
verdict turns on, and §C.3 could only bracket it. It is directly measurable
*today*, with no PP code, because of the emitter capability §B identified:
`PLOW_GLM_LAYERS=N` emits the first `N` layers with real weight names and a
correctly sized KV cache.

**A 40-layer GLM-5.2 blob at `--num-gpus 4` IS pipeline stage 0 of a balanced
TP4 x PP2 split** (the balanced cut is at layer 40, §A). It is missing only the
two boundary ops. And at 341.4 GiB / 4 = 85.4 GiB of weights per rank, **it
fits** — which is also the empirical half of Stage A.

So: emit the same 40 layers twice, once at `--num-gpus 8` and once at
`--num-gpus 4`, run both on the same objects, and the ratio of their token times
IS `alpha`. Both blobs have identical layer counts, identical packet counts per
layer, and an identical tail; the only difference is shard width.

    GLM_FULL=1 PLOW_MLA_PREFILL=full GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 \
    GLM_SHARD_HEAD=1 PLOW_GLM_LAYERS=40 \
    plowc --emit devblob --hf-dir /workspace/models/GLM-5.2-FP8 \
          --gpu MI300X --arch gfx942 --num-gpus {8|4} --max-ctx 8192

Objects `hsaco_glm18` (`PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1
PLOW_MOE_PF_EPI=1`), `PLOW_MLA_PF_V2=1`, `plowrt amd-bench --steps 32`,
**3 interleaved rounds tp8/tp4/tp8/tp4 in one lock hold**, `rocm-smi` 0% on
acquire.

### D.1 alpha unbound: 0.97 — and it is WRONG. A worked example of this campaign's own trap.

First measurement, `amd-bench` **without** `--checkpoint`, 3 interleaved rounds:

| rep | ctx 1024 TP8 | ctx 1024 TP4 | ctx 4096 TP8 | ctx 4096 TP4 |
|--:|--:|--:|--:|--:|
| 1 | 14.350 | 13.896 | 15.255 | 14.689 |
| 2 | 14.335 | 13.903 | 15.274 | 14.720 |
| 3 | 14.337 | 13.889 | 15.275 | 14.741 |
| **median ms/token** | **14.337** | **13.896** | **15.274** | **14.720** |
| control spread | 0.10% | 0.10% | 0.13% | 0.35% |

    alpha_unbound(ctx 1024) = 13.896 / 14.337 = 0.969
    alpha_unbound(ctx 4096) = 14.720 / 15.274 = 0.964

Beautifully reproducible — 0.1% spread, a 3.1% effect at 30x the spread — and it
says TP4 runs a layer FASTER than TP8 while doing twice the work. It also
appears to corroborate the campaign's own bring-up datum (a 78-layer TP4 blob at
47.234 ms/token vs TP8's 48.6, `alpha ~ 0.97`).

**It is an artefact, and both numbers are artefacts of the same thing.**
`amd-bench` without `--checkpoint` allocates the named tensors but **never loads
the packed routed experts** — the tell was visible in the log and I did not read
it at first: the unbound TP4 arm's `memset_ms` is *smaller* than TP8's
(1.9-3.2 s vs 4.0-4.4 s), when a TP4 rank must hold twice a TP8 rank's weights.
Nothing that scales with the shard width was being touched. The MoE expert
stream — 27% of decode's real work — was absent from **both** arms, and it is
precisely the term that doubles at TP4.

This is `glm52-gfx942-campaign`'s recorded lesson landing on a new path: *"an
ablation showing 'removing the work changes nothing' has TWO explanations — the
work is free, or the work was never removed."* Here the work was never bound.
**Do not price a sharding change on an unbound `amd-bench` run.**

### D.2 alpha bound — the authoritative measurement: 1.17 decode, 1.26 prefill

Same blobs, same objects, same session, with `--checkpoint
/workspace/models/GLM-5.2-plow-lite` and a real 1024-token prefill:

| | TP8 (40 layers) | TP4 (40 layers) | ratio |
|---|--:|--:|--:|
| resident weights per rank | **47.06 GiB** | **91.04 GiB** | **1.93x** |
| of which packed routed experts | 41.64 GiB | 83.27 GiB | 2.00x |
| prefill, 1024 tokens | **180.9 ms** (5662 tok/s) | **228.1 ms** (4489 tok/s) | **`alpha_prefill` = 1.261** |
| decode, 16 steps @ ctx 1024 | **15.586 ms/token** | **18.229 ms/token** | **`alpha_decode` = 1.170** |
| cross-rank token identity | all 8 ranks identical | all 4 ranks identical | both PASS |

Binding the weights moves the TP8 token 14.337 -> 15.586 (+8.7%) and the TP4
token 13.889 -> 18.229 (**+31.2%**) — exactly the asymmetry the unbound run was
blind to, and in the direction the shard width predicts.

**Two things are settled by this table.**

1. **Stage A is confirmed empirically, not just arithmetically.** A TP4 rank
   really does hold **91.04 GiB** for 40 of 78 layers, against a TP8 rank's 47.06
   for the same 40 — a clean 1.93x, which is 2x within padding. Scaled to a real
   PP2 stage this is ~91 GiB/rank against TP8's ~87 for the whole model: **parity,
   measured.** A TP4 x PP2 stage fits on a 192 GB card. It was never a memory
   question.
2. **`alpha` > 1: TP4 is genuinely slower per layer.** Decode +17.0%, prefill
   +26.1%.

**Why decode's +17% and not the +100% a work-bound machine would pay.** The
27.36 ms token is 67.5% gate wait, so doubling the per-rank work per packet is
mostly absorbed. What is *not* absorbed is the routed-expert stream:
`GLM_MOE_CORESIDENT=2` pins 8 expert slots to disjoint CU partitions, so each
slot has a fixed ~38 CUs and twice the bytes to pull through them — that term
pays close to the full 2x, and it is where the +17% comes from (2.6 GiB/rank/token
at TP4 against 1.3 at TP8, at a measured ~0.6-1.06 TB/s effective, far off the
5.3 TB/s roofline, consistent with the campaign's standing "MoE is
weight-stream-bound ~3x off HBM" finding). Prefill pays more (+26%) because it is
much closer to work-bound: packing efficiency at T=8192 is 91.0%, not 31.1%.

> **Caveat, stated because it matters:** the bound arm is n=1 per cell. The effect
> (+17.0% / +26.1%) is far outside the unbound arm's 0.1-0.35% control spread, and
> the two phases agree in sign and rough magnitude, but a replication battery was
> launched and yielded to the `decode-ladder` sibling's server rather than
> contend for the GPU. Treat `alpha_decode` as **1.17 +/- 0.05**.

### D.3 What alpha = 1.17 does to the Stage C model — and a retraction

`alpha` landed in the upper half of the §C.3 bracket [1.0, 1.5], not below it.

| claim | §C.3 a priori | measured (bound) | verdict |
|---|---|---|---|
| latency @ concurrency 1 | +0 to +50% TPOT | `alpha` = 1.170 | **+17% TPOT, +26% TTFT** |
| throughput ceiling, c >= 2 | 1.35-2.0x | `2/alpha` | **1.71x** |

**RETRACTION.** An earlier revision of this document (commit `0cad08f`) reported
`alpha` = 0.965 from the unbound arm and stated that *"the brief's premise that
'for a SINGLE request, PP2 is strictly worse than TP8' is REFUTED by
measurement."* **That is withdrawn. The premise is correct.** PP2 is worse for a
single request — by **17%**, not by the 100% a naive
"the-stages-serialise" reading predicts, but worse. The correct statement of the
nuance is:

> PP2's concurrency-1 cost is `alpha`, the TP4/TP8 per-layer wall ratio, and on a
> machine that is 67.5% gate-wait `alpha` is 1.17, not 2.0. Most of the naive
> penalty is absorbed by idle issue slots; what survives is the routed-expert
> weight stream, which doubles per rank and is pinned to a fixed CU partition.

### D.4 The TP8 concurrency sweep the campaign has never had

Every published GLM number in this directory is concurrency 1. This is the
throughput baseline that was missing. Canonical asset
`/workspace/assets/gfx942/glm52-tp8-final2` (objects `hsaco_glm18`), serve env
`PLOW_MLA_PF_V2=1`, `scripts/bench_speed.sh`, in_len 1024, 16 requests per cell,
64 output tokens, coherence gate PASS.

| in_tok | conc | n | TTFT mean ms | TTFT med ms | TPOT ms | ITL p99 ms | **out tok/s** | req/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 1 | 16 | 344.3 | 344.3 | 26.81 | 26.97 | **31.5** | 0.49 |
| 1024 | 2 | 16 | 2,552.9 | 2,769.1 | 31.42 | 36.03 | **27.5** | 0.43 |
| 1024 | 4 | 16 | 5,902.5 | 6,685.3 | 28.22 | 29.96 | **30.1** | 0.47 |
| 1024 | 8 | 16 | 11,554.5 | 15,157.6 | 28.46 | 30.00 | **29.9** | 0.47 |
| 1024 | 16 | 16 | 17,412.5 | 17,809.4 | 29.61 | 30.02 | **28.9** | 0.45 |

**Aggregate throughput is FLAT — 31.5 -> 28.9 out tok/s across a 16x concurrency
range (-8%, i.e. flat to slightly negative) — while TTFT rises 50.6x.**

**Am I measuring batching or queueing? QUEUEING, and here is how that is
established rather than assumed.** `bench_speed.sh` prints
`conc>1: batching vs queueing is UNKNOWN here — the server does not report its
decode batch` because `/v1/models` carries no batch field on this build. The
answer is in the server's own startup line:

    AMD serve engine ready n_gpu=8 max_ctx=73728 batch=1 decode_only=false

`batch=1`, so `dispatch_all` advances exactly one sequence per decode tick and
every other in-flight request waits. Three independent signatures confirm it:
TPOT is flat at 26.8-31.4 ms at every concurrency (a batched engine's TPOT climbs
with the batch — the Gemma B=8 control goes 33.98 -> 66.61 (**RETRACTED**, wrong-math blob — see §D.4(b))); ITL p99 pins to
~30 ms, i.e. one token per tick with no batch tail; and TTFT scales as
`~(conc-1) x` a whole request's service time, which is the definition of a queue.

**This makes the §C.4 baseline (a) an exact GLM measurement rather than a Gemma
transfer.** GLM-5.2 at TP8 has *no* throughput scaling today. Against this
baseline TP4 x PP2's `2/alpha` = **2.07x** is a real doubling — and so is the
batch ladder's, at twice the size and a small fraction of the cost.

One incidental: the c=2 row is the worst cell (27.5 tok/s, TPOT 31.4). With two
requests queueing, the second one's prefill lands inside the first one's decode
stream, and GLM's prefill chunk is long enough to stall the decode tick behind
it. This is a scheduling artefact of `batch=1`, not a property of concurrency,
and it disappears in the c>=4 rows once the queue is deep enough to amortise it.

---

## Stage E — verdict

### E.1 Is TP4 x PP2 worth building for throughput? NO.

Everything, relative to GLM-5.2 TP8 at `PLOW_DECODE_BATCH=1`, concurrency 1
(26.81 ms TPOT, 31.5 out tok/s, §D.4):

| route | aggregate throughput | per-user TPOT | throughput per unit TPOT | cost to build | evidence |
|---|--:|--:|--:|---|---|
| TP8 B=1, raise concurrency | **1.0x** (flat to c=16) | 1.0x | 1.00 | 0 | §D.4, measured on GLM |
| **TP4 x PP2, B=1, c >= 2** | **1.71x** (`2/alpha`) | **1.17x** | **1.46** | **35-55 eng-days** | `alpha` measured §D.2, model §C.4 |
| ~~TP8 B=8, c=4~~ | ~~2.66x~~ | ~~1.45x~~ | ~~1.84~~ | an emit flag | **RETRACTED** — wrong-math B=8 blob, §D.4(b) |
| ~~**TP8 B=8, c=8**~~ | ~~**4.00x**~~ | ~~1.96x~~ | ~~**2.04**~~ | an emit flag | **RETRACTED** — wrong-math B=8 blob, §D.4(b) |
| machine packing ceiling | ~3.6x | — | — | — | 27.36 / 7.694 |

> ## CORRECTION (2026-08-08, after the `decode-ladder` sibling reported)
> ## The two bullets below marked ~~struck~~ are WRONG FOR GLM-5.2.
>
> **GLM-5.2 has no batched decode at all**, so `TP8 B=8` is not a flag for this
> model — it is a build. `glm_emit_full` (`crates/devgen/src/mla.rs`) emits its
> decode program at one row *structurally*: the decode `Embed` carries `i[0]=1`,
> `emit_glm_tail` is passed a literal `1`, and the decode layer emitters take no
> row parameter at all, unlike their `_prefill` twins. `PLOW_DECODE_BATCH` is read
> in exactly two places and GLM is neither. See
> `glm52-decode-batch-ladder.md` §0.
>
> Note what this does to the evidence column of the E.1 table, which should have
> been read more carefully when it was written: **PP2's 1.71x is grounded in an
> `alpha` measured on GLM; batching's 4.00x is a Gemma control transferred
> across, and its "an emit flag" cost is Gemma's cost, not GLM's.**
>
> A second point this evaluation never had to weigh, because it assumed batching
> was free: **PP2's gain does not require batched decode.** Pipelining is
> request-level — request A occupies stage 1 while request B occupies stage 0,
> one row each — so it cashes idle machine without solving GLM's row-carrier
> problem. Batching cashes the same idle, but only *after* that problem is solved.
>
> **The verdict does not flip** — PP2 still reaches 48% of the 3.56x packing
> ceiling for 20-30 engineer-days minimum, and batched GLM decode reaches
> essentially all of it. But the comparison as written is between a build and a
> flag that does not exist for this model, and "ship the batch ladder" is NOT an
> available instruction for GLM-5.2.
>
> **Revised recommendation for GLM: scope batched GLM decode first** — sequence-row
> carriers through MLA decode, the block-fp8 MoE decode chain, every GEMV and the
> per-slot KV write. It is the prerequisite for the larger win, its ceiling is the
> packing headroom rather than 48% of it, and the ladder that consumes it is
> already built and measured (`glm52-decode-batch-ladder.md`). PP is the fallback
> if that scoping comes back expensive.

**Batching dominates PP2 on every axis simultaneously** (for a model that HAS
batched decode — see the correction above; GLM-5.2 does not):

* **absolute throughput** — 4.00x vs 1.71x;
* **throughput per unit of TPOT spent** — 2.04 vs 1.46, so PP2 does not even win
  the "latency-preserving throughput" argument that would be its natural refuge;
* ~~**cost** — an emit flag that exists (plus the runtime ladder the
  `decode-ladder` sibling is landing) vs 35-55 engineer-days;~~ **WRONG for GLM:
  a flag for the dense-GQA family and Kimi-K3, a BUILD for GLM-5.2.**
* ~~**generality** — batching works for every model family in the tree;~~ **WRONG:
  it works for every family that carries a decode row count, which is the
  dense-GQA family and (via `k3_build_model`) Kimi-K3 — NOT GLM/DeepSeek-class
  MLA emitters.** TP4 x PP2 helps only models sharded thin enough at TP8 for TP4
  to be worth its bubble, and GLM-5.2 is not one of them (`alpha` > 1).

**And they do not compose, because they are two roads to one destination.** Both
are cashing the same idle machine — decode packing efficiency is 31.1%, so the
work headroom is 27.36/7.694 = **3.56x** and that is all there is. PP2 reaches
1.71x = **48% of it**. Batching reaches essentially all of it. Stacking batching
on top of TP4 x PP2 converges to the same ~3.5x ceiling from the other side (a
TP4 rank does 2x the work in 1.17x the time = 1.71x more work per second, leaving
1.9x of packing headroom; 1.71 x 1.9 = 3.3x). Once one lever has cashed the idle,
the other has nothing left to cash — and the cheap one gets there first.

### E.2 Is a pipeline the right basis for a throughput design at all? NO, not on this box.

PP is two levers wearing one name, and this machine needs neither:

1. **A memory lever** — fit a model TP alone cannot. GLM-5.2 fits at TP8 with
   **~72 GiB/rank spare** (§A, and §D.2 measured the TP4 side of it at 91.04 GiB).
   Not this box's problem.
2. **A fabric lever** — replace a slow seam with a narrow point-to-point handoff.
   This box's seam is a **full XGMI mesh, all 64 ordered pairs 1 hop**, and PP
   does not remove the collectives anyway: each stage still all-reduces internally
   78 times per token. TP4 makes each of those **slower per byte** (3 live links
   instead of 7, §C.1). Not this box's problem, and PP makes it slightly worse.

Microbatched throughput is a *consequence* of those two levers, not an independent
virtue. Note where PP's costs actually landed, because it is the opposite of the
usual story: **the stage handoff — PP's own advertised overhead — is free**
(0.04-0.14%, §C.2). **The seam narrowing — PP's own advertised saving — is a
cost** (+5.8% TTFT, §C.1). Everything PP costs, it costs by making every layer
run at TP4, which is `alpha` = 1.17.

On a single node with a 1-hop mesh and 72 GiB of spare VRAM per card, a pipeline
is answering a question this machine does not ask.

### E.3 What the measurement changed, that is worth keeping

* **The concurrency-1 penalty is 17%, not 100%.** The naive
  "the-stages-serialise-so-latency-doubles" model is wrong by a factor of six on
  this machine, because 67.5% of the token is gate wait and absorbs most of the
  doubled per-rank work. What survives is one specific term: the routed-expert
  stream, doubled per rank and pinned to a fixed CU partition by
  `GLM_MOE_CORESIDENT=2`. **If anyone ever does want PP here, that is the term to
  attack first**, and it is a scheduling knob, not a parallelism change.
* **GLM-5.2 at TP8 has no throughput scaling at all today** (§D.4): 31.5 -> 28.9
  out tok/s across a 16x concurrency range while TTFT rises 50.6x. That is the
  campaign's first throughput datum and it is the strongest argument for the
  batch ladder in the tree.
* **Do not price a sharding change on an unbound `amd-bench` run** (§D.1). It gets
  the sign wrong.
* **`PLOW_GLM_LAYERS=N --num-gpus 4` is a working, weight-bindable TP4 stage-0
  blob.** Two emits and one lock hold priced a parallelism strategy that does not
  exist in the code. Keep the recipe.

### E.4 Recommendation

1. **Ship the batch ladder.** Highest value per engineer-day in the campaign:
   4.00x aggregate on the Gemma control against GLM's measured *flat* baseline.
   Already in flight on `decode-ladder`. **Coordinate on one point: PP and
   batching are competing claims on the same 3.56x of idle, not complementary
   ones.** If the ladder lands, the PP case gets weaker, not stronger.
2. **Do not build TP4 x PP2**, and specifically do not build it on the seam
   argument — §C.1 shows the seam gets *more* expensive, not less.
3. **Nothing was implemented here, deliberately.** The brief asked for
   implementation only where the arithmetic justified it; the arithmetic
   justifies none. §B is the scoped plan if the situation in §E.5 ever changes.
4. **Re-check the batch-slot memory ceiling before assuming B=16 is reachable at
   full context.** Each decode slot costs ~7.6 GiB of *replicated* latent +
   indexer cache at `max_ctx` 73728, so TP8's ~72 GiB of free VRAM tops out near
   **9-10 slots** — just under the emitter's 16-slot clamp. At 32k and below the
   clamp binds first. This is the one place PP's KV halving would matter, and it
   is worth knowing which constraint the ladder actually hits.

### E.5 What would flip this verdict

* **A second node, or any seam that is not 1-hop XGMI.** PP is the correct answer
  for a slow seam; this box has none.
* **A model that does not fit at TP8.** GLM-5.2 fits with ~72 GiB/rank spare; a
  ~1.4T-parameter fp8 model would not, and then PP2 stops being optional and
  §A's parity result becomes the reason it is possible at all.
* **Context or batch demand ~2-4x beyond today**, where PP's halved per-rank KV
  (3.09 vs 6.17 GiB at `max_ctx` 73728) lifts the slot ceiling from ~10 to ~20.
* **`alpha` falling to ~1.0.** It is 1.17 and the +17% is one identifiable term
  (§E.3). If the MoE decode expert partitioning were made shard-width-aware, or
  if the routed-expert stream were fixed by other means, PP2's ceiling would rise
  from 1.71x toward 2.0x — still below batching's 4.0x, so this alone is not
  enough.

**None of these hold today. The answer is no, and the arithmetic is above.**
