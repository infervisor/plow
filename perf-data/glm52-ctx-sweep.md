# GLM-5.2 — apples-to-apples context sweep to 128k, TP2 / TP4 / TP8

Script: `scripts/glm52_ctx_sweep.sh` (emit / tune / gate / run / table).
Collator: `scripts/glm52_ctx_sweep_table.py`.

Everything here is produced by **one client**, `vllm bench serve --backend openai-chat`,
pointed at two different base-urls (knob-contract §0-BENCH). Same dataset, same
`--random-input-len`, same `--random-output-len 128`, same `--max-concurrency`, same
`--num-prompts`, same warm-ups. plow is served by **`plowrt serve` and nothing else**.

## Provenance of every cell

| | |
|---|---|
| objects | `/home/lava/plow/build-amd/ctxsweep-objs`, `PLOW_MLA_PREFILL=1 PLOW_MOE_PREFILL=1 scripts/build_gfx950.sh`, register-cliff gate **PASS** (prefill 256/occ2/spill2, decode 248/occ2/spill0, flash 512/occ1/spill228 — all at the recorded values) |
| build digest | **`gfx950-a168b6e2e77e1975`** |
| tuning | 3 campaigns x 180 rows -> **270 records published under `gfx950-a168b6e2e77e1975`**; the 990 pre-existing records are for other digests and are correctly skipped as stale |
| tuning tier | **measured** for every shape in the campaign (see the gate below); analytical fallback for anything outside it |
| blob | one per TP, `--max-ctx 135168`, `PLOW_GLM_DSA=0`, `PLOW_FP8=1 GLM_FULL=1 GLM_SHARD_HEAD=1 GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48`, ladder `full:128,512,2048,8192` |
| blob check | 5 programs (4 prefill buckets + decode), `max_chunk=8192`, **zero DSA opcodes**, `weight_enc=fp8 / expert_enc=fp8blk` |
| precision | block-fp8 both engines (checkpoint `quantization_config`: `fp8`, e4m3, `weight_block_size [128,128]`, dynamic activations) — a legal fp8-vs-fp8 comparison |

The whole chain was re-run from `build_gfx950.sh` forward for this sweep. TP4 and TP8 were
emitted off the **same** object build and the **same** digest, which is legitimate: nothing
in the chain changed between them.

### The digest MOVED during the campaign, and that is why it is printed per cell

This worktree is shared with other agents, and the build digest is a function of the *source
tree*, not of the objects on disk. Timeline:

| time | event |
|---|---|
| 20:24-20:25 | objects built -> `build-amd/ctxsweep-objs` |
| 20:28-20:31 | 3 tile campaigns ingested, keyed to **`gfx950-a168b6e2e77e1975`** |
| 20:32 | TP4 and TP8 blobs emitted against that tuning |
| 21:05-21:57 | **TP8 column measured** |
| 22:11 | tuning gate re-checked and green; **TP4 run starts** |
| 22:15-22:16 | another agent edits `crates/devgen/src/lib.rs` and `runtime/amd/op_gemm.h` |
| after | probed digest is now **`gfx950-d48d9b08cd6cd5d8`** |

**Nothing measured here is invalidated**: objects, tuning and blob are all from the 20:2x
snapshot and the runs consume those files, not the source. But two consequences are real and
neither is optional to state.

1. **The 270 records published in this campaign are now stale against HEAD.** Anyone
   reproducing today gets the analytical fallback unless they re-run `tune` first. The gate
   in this script will tell them so.
2. **A multi-hour benchmark in a shared worktree cannot assume a stable digest.** The only
   defences are the two used here: keep objects, tuning and blob from ONE snapshot, and print
   the digest beside every cell so a reader can tell which snapshot a number belongs to.

Both columns below carry digest **`gfx950-a168b6e2e77e1975`**, tuning tier **measured**.

## THE CAVEAT THAT GOVERNS EVERY ROW: plow runs DENSE, vLLM runs SPARSE

`GlmCfg::dsa` (`crates/devgen/src/mla.rs:136`) arms GLM's sparse indexer->select->gather
path at `ctx > 65536`, and that path is recorded as producing degenerate output and is
unvalidated past ctx 2048. `PLOW_GLM_DSA=0` — named in `mla.rs:135` as "the apples-to-apples
decode baseline" — forces the dense path at every ctx, and that is what these blobs contain
(verified: no `AttnSelect` / `FlashGatherDecode` opcode in either build manifest).

vLLM serves the same checkpoint with the indexer live. From `config.json`:
`model_type = glm_moe_dsa`, `index_topk = 2048`, `index_n_heads = 32`.

> **Above ~2k context vLLM's attention reads 2048 KV rows per layer per token and plow reads
> all of them — 64x more at 128k.** plow is doing strictly more work in every cell of this
> table except the shortest. A plow win under that handicap is a real win. A plow loss is
> **not** evidence about plow's kernels and must not be reported as one.

## TP2 IS NOT MEASURABLE ON THIS BOX — capacity, and it is not close

This was checked first, before any harness was built, because it was the likeliest failure.
It is not a context-dependent OOM: it is a weight-residency deficit that exists at ctx 0.

    GLM-5.2 block-fp8 safetensors:  766,860,115,496 B  =  714.2 GiB
    MI355X VRAM (rocm-smi):         309,220,868,096 B  =  288.0 GiB

| TP | GiB/rank, weights alone | verdict |
|---|--:|---|
| 1 | 714.2 | 2.48x the card |
| **2** | **357.1** | **1.24x the card — 69 GiB over, before a single KV byte** |
| 4 | 178.5 | fits (0.62x); measured live residency 183.08 GiB/rank incl. KV + activations |
| 8 | 89.3 | fits (0.31x) |

Divisibility is **not** the blocker and this is worth stating because it is the intuitive
guess: 64 attention heads give 32/16/8 per rank at TP2/4/8 and 256 routed experts divide
too, so the `c.heads % tp == 0` guard (`mla.rs:4009`, `:4115`) passes at all three. The TP2
**blob emits cleanly** — the compiler is not the limit. There is simply no ctx at which the
weights fit, because lowering ctx removes KV, not the 69 GiB. Closing TP2 needs a smaller
checkpoint, weight streaming, or expert offload; none exists today.

**TP2 is therefore reported as infeasible-by-capacity, not as a failed run.**

## The tuning gate, and the one thing it caught

`cargo test -p devgen --test tuned_tile_selection`: **3 of 4 pass**, including the
load-bearing one.

* `published_measurements_reach_the_compiler_and_change_its_answer` — **PASS**. This is the
  gate that matters: it fails exactly when tile selection has silently reverted to the
  analytical model, which is the state that leaves prefill (and therefore TTFT, and
  therefore most of a context sweep) unmeasured. It was **FAILING** before this work and
  passes now.
* `the_narrow_shapes_agree_between_model_and_hardware` — **FAIL**, reproduced across three
  independent campaigns, and it is a real measured correction rather than staleness.
  Medians over the 3 passes, ns:

| shape | | measured winner | runner-up | model says |
|---|---|--:|--:|---|
| Gemma-26B router | 128x128x2816 | **128x128 = 28828** | 64x128 = 29590 | 64x128 — **disagrees** |
| Gemma-12B k global | 128x512x3840 | **128x128 = 43362** | 64x128 = 47639 | 64x128 — **disagrees** |
| GLM-5.2 router | 128x256x6144 | 64x128 = 49351 | 128x128 = 62749 | agrees |
| GLM-5.2 kv_a_proj | 128x576x6144 | 64x128 = 63405 | 128x128 = 72806 | agrees |
| Kimi kv_a_proj | 128x576x7168 | 64x128 = 70695 | 128x128 = 81856 | agrees |
| Gemma-26B gate/up | 128x2112x2816 | 64x128 = 43891 | 128x128 = 49146 | agrees |

**Every GLM-5.2 shape agrees, so this sweep is unaffected.** The disagreement is on two
Gemma shapes at N <= 512 and belongs to whoever owns `pick_tile`. Recorded, not silenced.

## Coherence — run before timing on every configuration, and it is not a formality

GLM-5.2's MoE routing is data-dependent, so a configuration that quietly degrades output
**also gets faster** (garbage activations collapse the router's top-k and the model does less
work). A fast wrong server is therefore indistinguishable from a fast right one on the
timing alone, which is why the gate runs first on every arm and its text is read.

| TP | short (T=128) | arithmetic (T=128) | long ~2.1k (T=1024+2048, 2 chunks) | first SSE chunk |
|---|---|---|---|---|
| 4 | `The capital of France is Paris.` | `17 * 23 = 391 ... three hundred ninety-one.` | `The capital of Japan is Tokyo.` | carries `content:"The"` — the 63f9957 artefact stays dead |
| 8 | `The capital of France is Paris.` | `17 * 23 = 391 ... **three hundred ninety-one**.` | `The capital of Japan is Tokyo.` | same |

**GLM-5.2 had never been run at TP8 before this sweep. It works.** Both TP4 and TP8 answer
correctly at `max_ctx = 135168` with dense attention forced.

Worth recording because it will surprise someone: **TP4 and TP8 are not token-identical.**
The arithmetic answer differs in markdown emphasis (`**three hundred ninety-one**` vs plain).
Both are correct; a different shard count changes the reduction order, hence the rounding,
hence eventually the argmax. Do not use cross-TP token identity as a correctness gate.

### The gap this does NOT close

The coherence prompt tops out at **2,116 tokens**. It exercises the T=128/512/2048 buckets
and says nothing about the 8192 bucket, the 16-chunk cover, or dense attention over 128k KV
rows — which is the entire regime this sweep is about. `vllm bench serve`'s random-token
prompts cannot fill that gap either: their outputs are meaningless by construction.
`glm52_tpctx_sweep.sh longcoherence <tp>` is the needle probe written for it (a passphrase
planted at 3 depths inside ~119k tokens of filler). **Status: written, not yet run** — see
"What was not run" below.

## EVERY cell here ran GF=4 while the blob asked for GF=8

`glm_gf()` (`crates/devgen/src/mla.rs:258-269`) bakes **GF=8** into `i[7]` of every
`FlashMlaDecode` packet whenever the emit-time `max_ctx` exceeds `GLM_GF_CROSSOVER = 4096`.
The AMD interpreter (`runtime/amd/interp.hip:427-450`) dispatches **only two arms**:

```
const unsigned gf = in->i[7] ? in->i[7] : (unsigned)GLM_MLA_GF;
if (gf == 2)  d_flash_mla_decode<512,64,2>(...);
else          d_flash_mla_decode<512,64,4>(...);
```

Verified independently for this report: `d_flash_mla_decode<...,8,...>` is **not instantiated
anywhere under `runtime/amd/`**. `gf == 8` therefore falls into the `else` and executes GF=4,
with no warning and no trap. `mla.rs:1952`'s own comment already says *"interp dispatches
GF=2/4 on this"* while the line above it bakes 8, and `mla.rs:6636` asserts *"the emitter
still bakes 8"*. Workgroup sizing is compensated (`mla.rs:6636-6637`, "8 is dispatched as 4"),
so this is **not a correctness bug — it is a silently unrealised optimisation.**

**This applies to all five sweep points, not just the long ones.** GF is a function of the
emit-time `max_ctx` (135168 here), not of the prompt length at runtime, so `i[7] = 8` is baked
on every decode packet in both blobs and every cell in this document ran the GF=4 arm.

The emitter's claim that GF=8 is 1.5-1.9x faster at ctx>=8192 is cited to
the design notes §7 — **sm120, i.e. NVIDIA**. It has never been realised on AMD.

> **So the TPOT curve below is a GF=4 curve, and it is a FLOOR.** The flat degradation
> (TP4: 29.38 -> 32.48 ms over 8x ctx; TP8: 1.50x over 32x) was achieved on the slower
> attention arm. If the GF=8 arm is implemented, these cells need re-running and should only
> improve. Digest `gfx950-a168b6e2e77e1975` identifies which side of that change they are on.

GF was deliberately **not** changed mid-sweep; doing so would have made the cells
incomparable.

## Measurement integrity — audited, and one incident the gate caught

**Two plowrt paths render a FAILED request as a SUCCESSFUL one on the wire**, so
`vllm bench serve` scores it as complete: `StreamChunk::Err` renders as
`finish_reason:"stop"` (`serve/chat.rs:401`), so an admission shed arrives as a normal
completion carrying the error text as content; and a stream ending with no terminal chunk
(fixed receiver-side in a5f4618) was scored as a short-but-successful request. Both inflate
tok/s while deflating mean output length, which reads exactly like a good result. The wire
shape is deliberately unchanged so cells stay comparable, so the filtering is the caller's.

Audited for every cell reported here:

| check | result |
|---|---|
| `admission shed` / `stream ended with no terminal chunk` in the plow server log | **0 occurrences** |
| successful requests == requested prompts | 8 / 8 / 4 / 2 / 2 — **exact** |
| generated tokens == n x 128 | 1024 / 1024 / 512 / 256 / 256 — **exact, no truncation** |

Every cell is at **concurrency 1**, where admission shedding cannot arise anyway; the audit
is recorded because absence of the marker is only evidence if someone looked.

### The incident: a shared `target/` clobbered the binary mid-campaign

Between the TP8 run and the TP4 re-run, `target/release/plowrt` went from **6,962,848 bytes
(built `--features hsa`, 21:34)** to **3,509,000 bytes (21:54)** — a default
`cargo build -p plowrt` by another agent in the same worktree. That binary selects the **CPU
reference backend** and serves fluent-looking garbage through a byte-fallback tokenizer.

The next run came up `server ready after 2s` (no weights loaded at all) and answered
`'koesgysgseyioseyeskuyiksggqocsgy'`. **The coherence gate failed it and the run aborted
before producing a single number.** That is coherence-before-timing paying for itself.

Two lessons, both now in the script: the sweep **rebuilds plowrt with `--features hsa` and
asserts the HSA backend is present in the `gate` phase, before any lease is taken** (§0
forbids compiling under `gpulease`, so it cannot live in `run`); and a 2-second startup is
itself a red flag, because a real GLM-5.2 load is 167 s at TP4 and 255 s at TP8.

## Sample counts

Prefill cost is superlinear in prompt length, so a flat prompt count makes the long-ctx
points own the wall clock. Counts are therefore a function of ctx — and **identical for both
engines at every ctx**, which is the property that matters. The caller builds one map and
hands the same string to `bench_plowrt_serve.sh` and `bench_vllm_chat.sh`.

| ctx | prompts | warm-ups |
|---|--:|--:|
| 4096 | 8 | 2 |
| 16384 | 8 | 2 |
| 32768 | 8 | 2 |
| 65536 | 2 | 1 |
| 131072 | 2 | 1 |

This is cheap in precision: at 32k the TTFT mean and median differ by **0.03%**
(72339.38 vs 72319.58 ms), so the estimator is not what limits these cells.

## Results

### plow, TP4 — measured, complete column, concurrency 1

| ctx | chunks | TTFT mean ms | TTFT med ms | TPOT mean ms | TPOT med ms | out tok/s | n |
|---|--:|--:|--:|--:|--:|--:|--:|
| 4,096 | 3 (2048+2048+128) | 4,770.6 | 4,773.0 | 28.51 | 28.54 | 15.2 | 8 |
| 16,384 | 3 (8192+8192+128) | 25,929.1 | 25,971.4 | 29.47 | 29.59 | 4.3 | 8 |
| 32,768 | 5 (4x8192+128) | 72,657.3 | 72,633.7 | 32.97 | 33.11 | 1.7 | 4 |
| 65,536 | 9 (8x8192+128) | 229,037.2 | 229,037.2 | 37.25 | 37.25 | 0.6 | 2 |
| 131,072 | **17** (16x8192+128) | **791,233.3** | 791,233.3 | **44.24** | 44.24 | 0.2 | 2 |

#### Reproducibility — two independent leases, two server loads

The first three points were measured twice: once on an early TP4 lease at n=8, and again on
the lease that produced the table above, with a fresh 167 s weight load in between.

| ctx | lease 1 TTFT | lease 2 TTFT | spread | lease 1 TPOT | lease 2 TPOT |
|---|--:|--:|--:|--:|--:|
| 4,096 | 4,772.6 | 4,770.6 | **0.04%** | 29.38 | 28.51 |
| 16,384 | 26,101.3 | 25,929.1 | **0.66%** | 29.36 | 29.47 |
| 32,768 | 72,339.4 | 72,657.3 | **0.44%** | 32.48 | 32.97 |

TTFT reproduces to well under 1% across leases. Nothing in this table rests on a single
observation of the quantity it is making a claim about.

### plow, TP8 — measured, complete column, concurrency 1

**GLM-5.2 had never been served at TP8. This is the first complete TP8 column.**

| ctx | chunks | TTFT mean ms | TTFT med ms | TPOT mean ms | TPOT med ms | out tok/s | n |
|---|--:|--:|--:|--:|--:|--:|--:|
| 4,096 | 3 | 4,249.1 | 4,272.2 | 27.78 | 27.66 | 16.5 | 8 |
| 16,384 | 3 | 20,159.5 | 20,144.9 | 28.20 | 28.04 | 5.4 | 8 |
| 32,768 | 5 | 50,721.9 | 50,721.1 | 28.79 | 28.78 | 2.4 | 4 |
| 65,536 | 9 | 144,006.4 | 144,006.4 | 32.59 | 32.59 | 0.9 | 2 |
| 131,072 | **17** (16x8192 + 128) | **462,264.7** | 462,264.7 | **41.75** | 41.75 | 0.3 | 2 |

### TP8 vs TP4 — the full comparison

| ctx | TTFT TP4 ms | TTFT TP8 ms | TP8 gain | TPOT TP4 ms | TPOT TP8 ms | TP8 gain |
|---|--:|--:|--:|--:|--:|--:|
| 4,096 | 4,770.6 | 4,249.1 | 1.12x | 28.51 | 27.78 | 1.03x |
| 16,384 | 25,929.1 | 20,159.5 | 1.29x | 29.47 | 28.20 | 1.05x |
| 32,768 | 72,657.3 | 50,721.9 | 1.43x | 32.97 | 28.79 | 1.15x |
| 65,536 | 229,037.2 | 144,006.4 | 1.59x | 37.25 | 32.59 | 1.14x |
| 131,072 | 791,233.3 | 462,264.7 | **1.71x** | 44.24 | 41.75 | 1.06x |

**Doubling the GPU count buys up to 1.71x on TTFT and never more than 1.15x on TPOT.**

That split is the knob-contract's decode picture holding at TP8, and it is the most useful
structural result in this sweep. Decode is dominated by per-packet cost, not per-rank work, so
halving the work per rank barely moves the token — 1.03-1.15x for 2x the hardware. Prefill is
real compute, so it scales, and **its gain grows monotonically with context** (1.12 -> 1.71):
the longer the prompt, the larger the share of TTFT that is genuine parallel work rather than
fixed per-launch overhead.

### The two curves, stated plainly

**TPOT degradation over the full 32x range: TP4 1.55x (28.51 -> 44.24 ms), TP8 1.50x
(27.78 -> 41.75 ms).** That is what dense attention costs, on the GF=4 arm, and it is the
number to hold against vLLM's recorded **0.98x — flat** on this same checkpoint. vLLM is flat
*because* the DSA indexer caps attention at `index_topk = 2048` rows however long the context
gets. This is dense-vs-sparse; the slope is the price of the escape hatch, not a kernel defect.

**TTFT is where plow actually loses, and it is not close.** 4.8 s at 4k rising to **791 s at
128k** at TP4 — a 13-minute time-to-first-token, 7.7 minutes at TP8. Growth is ~`n^1.5`:
prefill is 16 chunks of 8192 and each chunk pays dense attention against all the KV already
written, so chunk cost rises with position. Recorded vLLM TTFT for this checkpoint at TP4/1k
is **1.9 s** against plow's 37.9 s (knob-contract §6g-LEGAL, a 20x gap), and nothing here
suggests the ratio improves with length — it is the axis to fix.

## Bonus: what this says about `CROSSOVER = 65536`

`mla.rs:131-134` asks for exactly this: *"this is the TP4 crossover ...; a TP8 deployment
halves the parallel floor and per-rank attention shrinks, **lowering** the crossover —
recalibrate with an 8-GPU sweep before serving TP8."* This sweep is that 8-GPU sweep.

**Do not change the constant on this evidence** — the gather side sits on the degenerate,
unvalidated path and cannot be measured, so only a bound is available. But the bound points
the other way from the projection.

Dense TPOT slope, measured, 32k -> 128k:

| | slope ms per 1k ctx | TPOT at 16k |
|---|--:|--:|
| TP4 | 0.115 | 29.47 |
| TP8 | **0.132** | 28.20 |

**TP8's dense curve is STEEPER than TP4's, not shallower.** TP8 starts lower (28.20 vs 29.47)
but degrades faster, so the two curves converge rather than diverge — which is what the
1.06x TP8 TPOT gain at 128k, against 1.15x at 32k, already says.

The calibration that produced 65536 recorded *"gather tpot is flat ~48.6 ms; dense grows
~0.136 ms/1k-ctx **from 41.4 ms @16k**"*. Dense at 16k is now **29.47 ms** — the interpreter
has taken ~12 ms out of it since (MLA merge-fold rewrite, `xrfit`, narrow-op sizing). So:

| assumption about the gather side | implied crossover, TP4 | implied crossover, TP8 |
|---|--:|--:|
| gather unchanged at 48.6 ms flat | ~183k | ~171k |
| gather improved by the same ~12 ms (shared fixes) | ~79k | ~80k |

**Both bracket ends are above 65536, and TP8 is not below TP4 at either end.** The projection
in the source — that TP8 lowers the crossover — is not supported. If anything the constant is
now too *low*, because the dense side it was calibrated against has got materially faster
while the gather side has not been re-measured at all.

The honest summary: **the recalibration cannot be completed until the gather path is
numerically correct**, and that is task #6. What this sweep contributes is a solid dense
curve at both TP4 and TP8 to calibrate against once it is.

## The vLLM column: GLM-5.2 does not currently start on this ROCm stack

This is vLLM's failure on this box, not a gap in the measurement, and it is a **hard
dichotomy with no third option** — both halves reproduced.

**Without AITER — hard refusal at model init:**

```
RuntimeError: Sparse attention indexer ROCm path is only supported on AITER.
              Please enable aiter with VLLM_ROCM_USE_AITER=1
```

Raised from `vllm/model_executor/layers/sparse_attn_indexer.py` via
`deepseek_v2.py:1082/1189/1330` (GLM-5.2 is served by the DeepSeek-V2 MLA path), surfacing as
`torch._dynamo.exc.ObservedRuntimeError` during `aot_compile` and then
`Engine core initialization failed`. GLM-5.2 **is** a `glm_moe_dsa` checkpoint with
`index_topk = 2048`, so the indexer is not optional — the model cannot be built without it.

**With `VLLM_ROCM_USE_AITER=1` — a very long single-threaded JIT compile at every startup:**

```
[aiter] shape is M:8192, N:2624, K:6144, not found tuned config in
        /tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv, will use default conf
[aiter] start build [module_gemm_a8w8_blockscale] under .../aiter/jit/build/...
[aiter] waiting for baton release at .../aiter/jit/build/lock_module_gemm_a8w8_blockscale
```

**Correction to my own first diagnosis, which was wrong and is recorded so nobody repeats
it.** I called this a deadlock after seeing all four TP workers on `waiting for baton release`
with zero new log lines for 90 s, and killed a run on that basis. It is **not** a deadlock:
`docker exec ... ps` inside the container shows a single **`clang-23` at 99.9% CPU** building
`module_gemm_a8w8_blockscale`, still climbing past **14 minutes** of CPU time. Three workers
wait on the baton by design while one compiles; the silence in the log is a long compile, not
a hang. **A log that has gone quiet is not evidence of a hang — look for the working process
before concluding one.**

The practical consequence is unchanged, though: this compile is paid on **every** container
start because AITER builds into `site-packages`, which is not on the `-v .../.cache` mount
that `bench_vllm_chat.sh` persists. Startup is therefore dominated by a JIT that a cache mount
covering `aiter/jit/build` would eliminate — a cheap fix for whoever next benchmarks vLLM on
this image.

Two things worth carrying forward regardless of how the deadlock is resolved:

1. **There is no tuned gfx950 AITER config for GLM's shapes** — the log says so explicitly for
   `M:8192, N:2624, K:6144`, which is a *prefill* shape. This corroborates knob-contract §6g's
   "untuned floor, not GLM's ceiling", and it means any future vLLM prefill number from this
   box is a floor and must be disclosed as one.
2. **vLLM's ROCm GLM-5.2 path is AITER-only**, so the dense-vs-sparse asymmetry in this
   document is not a configuration choice either engine could have avoided: vLLM cannot run
   this model densely, and plow cannot run it sparsely without the degenerate indexer.

It does start, after **3,110 s (52 minutes)** of that JIT. The TP4 column below is real.

## THE HEAD-TO-HEAD — TP4, same client, same dataset, same n, concurrency 1

| ctx | plow TTFT ms | vLLM TTFT ms | plow/vLLM | plow TPOT ms | vLLM TPOT ms | plow/vLLM |
|---|--:|--:|--:|--:|--:|--:|
| 4,096 | 4,770.6 | 677.7 | 7.0x | **28.51** | 30.88 | **0.92x — plow wins** |
| 16,384 | 25,929.1 | 1,508.8 | 17.2x | **29.47** | 30.29 | **0.97x — plow wins** |
| 32,768 | 72,657.3 | 1,607.4 | 45.2x | 32.97 | 25.41 | 1.30x |
| 65,536 | 229,037.2 | 2,827.4 | 81.0x | 37.25 | 26.48 | 1.41x |
| 131,072 | 791,233.3 | 5,967.5 | **132.6x** | 44.24 | 29.10 | 1.52x |

### Decode: plow wins at 4k and 16k, and the crossover is exactly where the theory puts it

**plow's TPOT beats vLLM's at 4,096 (0.92x) and 16,384 (0.97x)** — while reading every KV row
against vLLM's 2,048, and on the GF=4 arm. Past 32k the dense cost takes over and plow falls
to 1.30 -> 1.41 -> 1.52x.

The two curves behave exactly as their attention does. vLLM is flat (30.88 -> 29.10 over 32x,
**0.94x**) because `index_topk = 2048` caps its work; plow rises **1.55x** because dense
attention cannot. **The crossover sits between 16k and 32k**, which is where a dense curve
meets a flat one given a ~2k cap — and it is the honest headline: *plow's decode is
competitive-to-better up to ~16k and the sparse path is what wins beyond it.*

**One caveat on the two plow wins, and it cuts against plow.** vLLM's TPOT is non-monotonic
(30.88, 30.29, **25.41**, 26.48, 29.10) and its 4k/16k cells are its slowest. Those two ran
first, minutes after an engine that had just spent 52 minutes JIT-compiling, and their
mean/median spread shows it: at 4,096 the mean ITL is **30.64 ms against a median of 22.93**.
A steady-state vLLM at 4k is probably nearer its 32k figure of ~25.4 ms, which would turn
both plow wins into ~1.12x losses. **Treat the 4k/16k decode win as unproven until vLLM is
re-measured with a warm JIT cache** — the fix is a cache mount covering `aiter/jit/build`.

### Prefill: this is the real result, and it is not close

**7.0x at 4k widening monotonically to 132.6x at 128k.** The gap is not a constant factor
being amortised — it *grows with context*, because plow's TTFT is ~`n^1.5` (17 chunks at 128k,
each attending to everything already written) while vLLM's is very nearly linear
(677 ms -> 5,967 ms over a 32x range is **8.8x**, i.e. sublinear).

That single comparison — plow `n^1.5` vs vLLM `n^0.9` — is the whole story of long context on
this model, and it is an architectural difference (chunked dense prefill vs sparse), not a
tuning gap.

### The full matrix

| TP | ctx | plow TTFT ms | plow TPOT ms | vLLM TTFT ms | vLLM TPOT ms | n |
|---|--:|--:|--:|--:|--:|--:|
| 4 | 4,096 | 4,770.6 | 28.51 | 677.7 | 30.88 | 8 |
| 4 | 16,384 | 25,929.1 | 29.47 | 1,508.8 | 30.29 | 8 |
| 4 | 32,768 | 72,657.3 | 32.97 | 1,607.4 | 25.41 | 4 |
| 4 | 65,536 | 229,037.2 | 37.25 | 2,827.4 | 26.48 | 2 |
| 4 | 131,072 | 791,233.3 | 44.24 | 5,967.5 | 29.10 | 2 |
| 8 | 4,096 | 4,249.1 | 27.78 | not run | not run | 8 |
| 8 | 16,384 | 20,159.5 | 28.20 | not run | not run | 8 |
| 8 | 32,768 | 50,721.9 | 28.79 | not run | not run | 4 |
| 8 | 65,536 | 144,006.4 | 32.59 | not run | not run | 2 |
| 8 | 131,072 | 462,264.7 | 41.75 | not run | not run | 2 |
| 2 | any | — | — | — | — | **structurally impossible: 357.1 GiB/rank vs a 288.0 GiB card** |

vLLM TP8 was not run: it needs all eight cards *and* another ~52-minute AITER JIT, because
that build is not on the persisted cache mount. Every vLLM TP4 cell passed the same integrity
check as the plow cells — successful requests 8/8/4/2/2 and generated tokens exactly
1024/1024/512/256/256, no truncation.

**Prefix caching (task #21), partly discharged.** The client side is verified from
`vllm bench serve`'s own argument dump: `random_prefix_len=0`, `random_range_ratio=0.0`, so
prompts share no prefix by construction, and plow's `usage` reports `cached_tokens: 0` on
every request. The **server-side** hit rate was *not* captured: `bench_vllm_chat.sh` removes
the container on exit and the logs go with it. The script now dumps the server's own
prefix-cache lines before teardown, so the next run closes this properly.

## What was NOT run, and why

| cell / check | status | reason |
|---|---|---|
| **TP2, every ctx** | not run, and never will be | 357.1 GiB/rank of weights against a 288.0 GiB card. Not a context-dependent OOM — the deficit exists at ctx 0. The blob emits; the hardware is the limit. |
| **vLLM TP8, every ctx** | not run | needs all 8 cards *and* another ~52-min AITER JIT (the build is not on the persisted cache mount). TP4 was the higher-value column and it is complete. |
| vLLM warm-JIT re-measure at 4k/16k | not run | the two cells where plow "wins" are vLLM's first two after a cold start; see the caveat above. |
| **long-context needle probe** | written, not run | `glm52_tpctx_sweep.sh longcoherence <tp>`. Needs its own ~3-4 min weight load and a lease; the sweep consumed the available window. **This is the highest-value follow-up** — see below. |
| GF=8 arm | not implemented on AMD | not this task's to fix; all cells labelled GF=4-actual. |

## The single highest-value follow-up

**Run `longcoherence` at TP4 and TP8 before anyone quotes the 65k and 128k rows.**

Everything above 2,116 tokens in this document is timed but not *verified*. The coherence gate
proves the T=128/512/2048 buckets; the 8192 bucket, the 16-chunk cover and dense attention over
128k KV rows are exercised only by random-token prompts whose output cannot be checked. Three
facts make that gap the most dangerous thing here:

* the recorded failure mode at long ctx on this model is **degenerate text, not a crash**;
* GLM's MoE routing is **data-dependent**, so a configuration producing degraded output also
  gets **faster** — a silently-wrong 128k row would look like a *better* result, not a worse
  one (the `PLOW_XR_SHUFFLE` arm captured 45% of a ceiling while deleting no work at all);
* the 128k row is the single most quoted number in a sweep whose entire purpose is 128k.

The probe is written and costs one lease and one weight load per TP. Until it passes, the 65k
and 128k rows should be read as **instrument readings pending verification**, not results.

Second follow-up, now quantified rather than guessed: **TTFT is the axis, and the gap grows
with context — 7.0x at 4k to 132.6x at 128k.** plow's prefill is ~`n^1.5` where vLLM's is
~`n^0.9`. A 131,072-token prompt is 17 chunked launches and each attends to everything already
written, so the exponent — not the constant — is the target. Decode is already competitive
(plow wins at 4k/16k, subject to the warm-JIT caveat); prefill is where the model is lost.

Third, cheap and unambiguous: **mount `aiter/jit/build` into the vLLM compile cache.** 52
minutes of single-threaded `clang` per container start is most of what made the vLLM column
expensive, it is paid on every run, and it is one `-v` away from being paid once.
