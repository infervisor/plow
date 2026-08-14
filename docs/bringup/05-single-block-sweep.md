# Stage 5 — Single-Block Sweep

> Prove that **one** transformer block is numerically correct and tuned before
> scaling to the whole model. A block is the repeating unit; if the block is
> wrong or slow, the model is wrong or slow N times over — and debugging it in
> isolation is an order of magnitude cheaper than debugging it inside a 90-layer
> serve loop.

**Precondition:** Stages 1–4 complete — the model's config is understood, its
weights load, and the emitter produces a full-model `model.pkt` without aborting.
The [`target.md`](target.md) block is filled in and every command below is
written against it (`--gpu $GPU --arch $ISA --n-cu $NCU`), never a literal part.

**Gate out (into Stage 6, runtime optimization):** one block matches a host
reference within tolerance **and** its isolated per-step latency is at or near
the per-block target derived from a reference framework's TPOT **measured on
`$GPU` itself**. Only then is it worth optimizing the runtime around the whole
model.

---

## Why a single block

* **A full-model run is expensive and noisy.** It mixes in embed, final norm,
  lm_head and sampling — none of which repeat per layer — and a full emit + load
  + step loop is minutes per data point, almost all of it weight-table
  allocation and bind. One block loads in ~1 s.
* **Per-block × num_layers reconstructs the model number.** This is the anchor
  used in both directions: forward, `block_us × layers ≈ decode body`; reverse,
  `per_block_us ≈ (TPOT_ms − fixed_overhead_ms) × 1000 / num_layers`. If the two
  agree, the isolated measurement is trustworthy.
* **It isolates the thing being optimized.** A per-op cycle profile of one block
  attributes latency to `qkv_proj`, `moe_experts`, `attn`, etc. — impossible to
  read cleanly out of a whole-model trace.

There are two distinct questions, and they use different tools:

| question | tool | reference |
|---|---|---|
| **Is the block correct?** | `plowrt amd-block` (A/B, `$VENDOR = amd`), or a C oracle harness | host/HF f64 or bf16 reference, bit-for-bit within tol |
| **Is the block fast enough?** | `block_run bench` (Gemma path) or a truncated-model sweep (MLA path) | a reference framework's per-block floor, measured on `$GPU` |

Correctness is target-independent — the oracle is a host reference, and a block
that matches it on one part matches it on another at the same precision.
Latency is not: every number in Step 3 is a statement about `$GPU`.

Do correctness first. A fast wrong block is worthless.

---

## Where this stage sits: the four-tier measurement ladder

Every knob in the tree is scored at one of four tiers, and **which tier a knob
belongs to is a property of the knob, not of your budget.** Scoring a knob at
the wrong tier produces a confident wrong answer, not a noisy one. This stage is
tier 2 — the one you should be doing most of your sweeping on.

| tier | harness | cost per config | what it is for |
|---|---|---|---|
| 1 — prune | standalone kernel probes: `runtime/tests/*.c[u]`, `runtime/bench/**`, `runtime/nvidia/experiments/*.cu` | seconds, no blob | **correctness-gating a rung** and killing obviously-bad shapes. **Never trust its ranking.** |
| 2 — sweep | `plowc --emit devblob --block` + `block_run bench`, or the truncated-model sweep for the MLA families (this stage) | seconds to ~a minute | the wide sweep — real `GpuEngine`, real megakernel, reproduces full-model ratios |
| 3 — confirm | `examples/step_bench` — whole blob, real layer mix, no HTTP/mux | ~a minute | final scoring before serving |
| 4 — accept | a serving client against `plowrt serve` ([Stage 7](07-perf-campaign.md)) | minutes | end-to-end, and the **only** tier that sees host-gap knobs |

```bash
nix develop --command bash -c 'cargo build --release -p plowrt $FEATURES --example step_bench'
PLOW_STEP_TIME=1 ./target/release/examples/step_bench <assets> [slots] [ctx] [steps]
```

`scripts/tune_block_sweep.sh` automates tier 2; `scripts/tune_decode_sweep.sh`
covers tier 1 → 3 (both are written against `$VENDOR = nvidia` — read their
headers and re-point them at `$ISA`/`$GPU` before quoting a number).

**Why tier 1 cannot score.** A standalone probe re-runs one kernel on hot inputs;
its operands and outputs become cache-resident in ways a real forward pass never
sees, and the ops that would co-schedule against it are absent. Recorded in both
directions on measured campaigns: a probe win of +15% was −3 ms in-model, and a
decode-GEMV lever scored **1.00× isolated where the megakernel measured 1.43×**
— the same lever tier 2 put at 1.45×, i.e. within ~1.4% of the full model. A
probe may motivate a variant or prune one; only an in-model gate + bench decides
whether it lands. [Stage 4](04-kernel-tuning.md) and
[Stage 7](07-perf-campaign.md) state the same law from their own sides.

### What tier 2 cannot score

A block sweep is not universal, and these four categories are the ones that
silently return a null instead of an answer:

* **Prefill dispatch/classing knobs** — per-object block size, segment classing,
  segment graphs. The block harness drives the decode/single-segment path, so on
  `$VENDOR = nvidia` the segmented prefill stack is only exercised at tier 4.
* **Prefill bucket policy** (`PLOW_PF_COVER`, `PLOW_PF_CHUNK_COST`,
  `PLOW_PF_LADDER_APPEND`) — needs *varied prompt lengths* to have any effect at
  all, which means a client, i.e. tier 4.
* **Host-gap knobs** (`PLOW_DEV_SAMPLE`, `PLOW_MULTISTEP`, `--slo-ms`) — their
  effect is *between* kernels. No single-engine harness at any tier below 4 can
  see them.
* **`PLOW_MAX_CHUNK`** needs no benchmark at all: it is an analytic
  memory-sizing decision derived from the model's window (the ring formula in
  [Stage 6](06-runtime-opt.md)). Benchmarking it measures the wrong axis.

Note also the harness caveat that makes tier 2 mis-report if ignored:
`block_run --batch` selects *active slots*, not kernel width, so scoring a decode
batch means emitting one block asset per `PLOW_DECODE_BATCH` you want to score.
And layer kinds are not interchangeable — emit one block per kind and score the
kind-weighted sum of **marginal** per-layer cost
(`score = N_slide·L_slide + N_full·L_full`); the fixed per-block overhead and the
lm_head cancel in a ranking.

---

## The `--block` emit path

A single block is emitted by the devblob compiler, `plowc`, with `--block`:

```bash
plowc --emit devblob --block <l>        # one block, layer l
plowc --emit devblob --block <l..r>     # a half-open layer range [l, r)
```

`--block` (env `PLOW_BLOCK` is the unchanged fallback) is parsed by
`devgen::block::parse_block` into a bounds-checked half-open range, and each
family's emitter writes a sibling `block.json` descriptor next to `model.pkt`
via `devgen::block::write_block_descriptor`. The descriptor records the block's
geometry (hidden, head counts, head_dim, intermediate size, expert layout) so a
reference harness cannot drift from what plow compiled.

The block emitters live in `crates/devgen/`:

| family | emitter symbol | file | block program shape |
|---|---|---|---|
| Gemma-4 (dense / MoE) | dense-attn `--block` path | `crates/devgen/src/lib.rs` | **carries** `in.ids`/`embed_tokens` + lm_head → model-shaped, runnable end-to-end |
| GLM (`glm_moe_dsa`, GLM-5.2) | `glm_emit_block` → `glm_build_block` / `glm_build_block_pf` | `crates/devgen/src/mla.rs` | `act.x`-in / `act.x`-out, **no embed, no lm_head, no `act.logits`** |
| Kimi-K3 | `kimi_emit_block` (reuses `glm_build_block_pf`) | `crates/devgen/src/mla.rs` | `act.x`-in / `act.x`-out |
| Nemotron-3 (Mamba-2 hybrid) | `nemotron_emit_block` | `crates/devgen/src/mla.rs`, dispatched in `lib.rs` | `act.x`-in / `act.x`-out (M4: only the `--block` device path is wired) |

**This distinction drives everything below.** A Gemma-4 `--block` asset still
declares embed and lm_head, so `plowrt amd-bench` and `examples/block_run` can
drive it directly. The **MLA-family** `--block` assets (GLM / Kimi / DeepSeek /
Nemotron) emit only `act.x`-in / `act.x`-out programs — there is no token entry
point — so they are driven either by `plowrt amd-block` (act.x in, act.x out) or
by a per-family C oracle harness under `runtime/tests/`.

### Reading and disassembling the emitted block

`plowrt disasm` is static, offline inspection — no GPU, no driver:

```bash
plowrt disasm <asset-dir-or-model.pkt>            # named operands, kernargs, counters
plowrt disasm <model.pkt> --program 1             # just the decode program
plowrt disasm <model.pkt> --range 0..40 --format json
```

Use it to confirm the block contains the ops you expect (e.g. the MLA read path,
the MoE router/experts/combine) before you spend a GPU on it.

---

## Step 1 — CPU sim (no GPU): does the block plan?

Before any hardware, walk the compiled block through the CPU simulator. This
checks that the block *plans* — packets fire, counters resolve, the makespan is
finite — and needs no checkpoint if you drive it from a plow-native descriptor.

```bash
# plow-native NetConfig descriptor (weight-free, runs anywhere):
./scripts/block_sim.sh crates/plowc/examples/transformer_block_gemma4_12b.json prefill:1:128

# or a real HF checkpoint directory:
./scripts/block_sim.sh /path/to/gemma-hf-dir decode:1:128
```

`block_sim.sh` compiles one block (`plowc --net <desc>` or `--hf-dir <dir>`) and
runs `plowrt simulate --assets <out> --bucket <phase>:<batch>:<seq>`, printing
packets fired, makespan, and per-op-family counts. `simulate --math golden` runs
reference numerics instead of a dry walk.

A block that will not sim will not run. Fix planning failures here.

---

## Step 2 — correctness: block output vs a host reference

This is the gate. Two independent mechanisms exist; use whichever fits the family.

### 2a. `amd-block` A/B (precision deltas, weight-free)

`plowrt amd-block` runs a block asset **act.x in → act.x out** on the AMD engine.
It is the A/B vehicle for numerics: compile the *same* block at two precisions
(same weights, same input), dump both outputs, and diff them.

```bash
plowrt amd-block --blob <block>/model.pkt --hsaco <hsaco-dir> \
  --checkpoint <ckpt> --inspect act.x,act.hn --dump /tmp/out_bf16.bin

# list the block's tensors (sanity: are embed/lm_head present or not?)
plowrt amd-block --blob <block>/model.pkt --hsaco <hsaco-dir> --list-tensors
```

`--inspect` reports zero / non-zero statistics per tensor — the "zero vs merely
wrong" instrument that catches a silently-unwritten output. `--dump` writes the
raw `act.x` bytes for a bit-exact diff between two precision builds. This tells
you *bf16 vs fp8* deltas cheaply, but it does not compare against an external
oracle.

### 2b. C oracle harness (bit-for-bit vs HF/host reference)

The authoritative correctness check diffs plow's block output against an
independent host reference (an HF-transformers oracle fixture, or an f64/bf16 CPU
reference). One harness per family lives in `runtime/tests/`:

The `_gfx950_` in these filenames is historical — it names the ISA they were
first brought up on, not a restriction. The oracle they diff against is a host
reference and is target-independent; what is target-specific is the interpreter
object they load (`$BUILD`) and the device they run on.

| family | harness | oracle |
|---|---|---|
| GLM-5.2 (`glm_moe_dsa`) | `runtime/tests/glm52_block_gfx950_test.c` | `glm52_oracle.py` fixture (HF transformers) |
| GLM (base MoE) | `runtime/tests/glm_block_gfx950_test.c` | HF fixture |
| Kimi-K3 | `runtime/tests/k3_block_gfx950_test.c`, `k3_mla_block_gfx950_test.c`, `k3_moe_block_gfx950_test.c` | `k3_*_oracle.py` |
| DeepSeek | `runtime/tests/deepseek_block_gfx950_test.c` | oracle fixture |
| MoE (generic) | `runtime/tests/moe_block_gfx950_test.c` | oracle fixture |
| block-fp8 | `runtime/tests/block_fp8_gfx950_test.c` | bf16 vs fp8 A/B |
| Gemma net block | `runtime/tests/net_gemma_block_test.c` | plow-native reference |

Example — build and run the GLM-5.2 single-block de-risk:

```bash
# build the decode interpreter, the HF oracle fixture, and the host harness:
nix develop --command bash -c './scripts/build_glm52_block.sh /tmp/glm52b'

# run on ONE GPU (system gcc + clean env; see the script header):
sg render -c 'cd /tmp/glm52b && /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME \
    LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=6 \
    ./glm52_test interp_decode.elf glm52_fixture.bin'
```

The GLM-5.2 harness assembles the full block at **real** dims and diffs plow's
last-token output against HF **per substep** (`attn_out`, `x_mid`, `xn2`, the
router top-8 set + gates, `x_next`), not just at the block boundary — so a wrong
op is localized, not merely detected. It runs twice: bf16 routed experts first
(isolate correctness), then block-fp8 (measure the fp8-vs-bf16 delta).

> **Note on `block_run check`.** `plowrt`'s `examples/block_run … check` feeds a
> hidden-state into `act.x`, runs one prefill bucket, reads `act.x` back, and
> reports shape / min / max / mean / NaN-Inf. Its scope is **shape + finiteness +
> self-consistency only** — there is no in-tree HF parity in `block_run`. Use it
> as a smoke test; use the C oracle harness (or `amd-block` A/B) for real parity.

---

## Step 3 — latency: block-level bench and sweep

Once the block is correct, measure it. The metric is per-step **decode** us and
per-pass **prefill** ms, both over a batch × context (B × T) grid, timed with
device events (pure GPU time). Numerics are irrelevant to the *timing* — the
per-step kernel time is data-independent — so a bench needs no correct weights.

### Gemma path (model-shaped block, `block_run bench`)

```bash
# emit one Gemma-4 block:
plowc --emit devblob --block 0 --arch $ISA --gpu "$GPU" --n-cu $NCU \
  --max-ctx $MAXCTX --hf-dir /path/to/gemma-hf-dir --out <asset>/

# bench decode B×T and prefill:
./target/release/examples/block_run <asset> bench \
  --batch 1,4 --ctx 128,1024,4096 --iters 100 --warmup 20 --prefill-iters 10
```

`block_run bench` prefills B slots to T rows, times N decode steps per (B,T), and
writes `sweep.json`. Both phases are timed with the same warmup / median / p95
treatment; the decode step is held at a fixed context so shapes stay static and
the step is graph-capturable on `$VENDOR = nvidia`.

### MLA family (GLM / Kimi / DeepSeek): truncated-model sweep

The MLA-family `--block` asset has no embed / lm_head, so `amd-bench` cannot
drive it. Two options:

* **Truncated-model TP sweep** (`PLOW_GLM_LAYERS=N` / `K3_NLAYERS=N`): emit a
  few real layers of the real megakernel at the served TP geometry, and
  difference the fixed overhead out. This keeps the megakernel context an
  isolated kernel loses.

```bash
# Kimi-K3 fast decode loop: a 5-layer TP8 asset swept over ctx.
./scripts/k3_block_sweep.sh                    # baseline
./scripts/k3_block_sweep.sh PLOW_K3_FUSE_A=1   # any emit knob, applied to the emit

# GLM-5.2 few-layer prefill sweep, kind-weighted per-layer score.
# NOTE: this script is pinned to one ISA in its name and its flags — read its
# header and re-point it at $ISA/$GPU before trusting a number from it.
./scripts/glm52_block_sweep_gfx942.sh
```

`K3_NLAYERS=5` is the smallest span containing both mixers (4 KDA + 1 MLA); it
loads in ~1 s and sweeps three contexts in under two minutes. The GLM sweep
times three spans and differences them into a `3·L_dense + 75·L_moe` comparator
(GLM-5.2 is 3 dense + 75 MoE via `first_k_dense_replace = 3`).

### Anchoring against a reference framework

A single block cannot be *served* by a reference framework in isolation, so the
per-block floor is derived from that framework's measured full-model decode
latency:

```
per_block_us ≈ (TPOT_ms − fixed_overhead_ms) × 1000 / num_layers
```

The block-baseline harness (`scripts/block_baseline.py`,
`scripts/block_layer_bench.py`) times the **same** single block through a tuned
reference path — either a hand-written torch/cuBLAS block, or the reference
framework's own decoder-layer class driven standalone (no engine, no server, no
checkpoint) — and writes a `sweep.json` with the **identical row schema**, so the
two diff directly via `scripts/block_compare.py --phase {decode,prefill}`. Pass
`--layers` and `--vllm-tpot-ms` and the harness prints its per-block number next
to the implied floor; if `harness × layers ≈ TPOT`, the isolated baseline is
trustworthy.

Per-op attribution (the target-to-beat by kernel, not just end-to-end):

```bash
# per-op device time + GB/s + TFLOP/s, CUDA-graph-per-op + L2 flush, both phases:
python3 scripts/block_op_bench.py <block-config>.json --phases decode,prefill --ctx 1024,4096
```

For plow's own per-op breakdown on `$VENDOR = nvidia`, build the device object
with `-DPLOW_NV_TRACE=1` and read `trace_reset` / `trace_summary` in `block_run`.

---

## Success criteria

The block passes Stage 5 and gates into Stage 6 when **all** hold:

1. **Correct.** Block output matches the host / HF reference within the
   family's tolerance:
   * bf16 + MoE: ~1e-2
   * block-fp8: ~3e-2
   * a bit-exact A/B is expected only between two builds that share precision
     and weights (the `amd-block --dump` diff).
   Per-substep diffs (attn_out, x_mid, xn2, router set, x_next for MLA) all pass,
   not just the block boundary.

2. **No silent zeros.** `--inspect` shows the expected tensors non-zero; no
   NaN/Inf from `block_run check`.

3. **Latency at target.** Isolated per-block decode us (and prefill ms) is at or
   near the per-block floor derived from the reference-framework TPOT **measured
   on `$GPU`**, and `block_us × layers ≈ decode body` reconciles forward and
   reverse. The floor is a number for this part; a floor carried over from
   another part does not satisfy this criterion.

4. **Sweep is uncontended.** Every timed run is on a card to itself (`gpulease`
   rc=0). A contended run (rc=76) is discarded and re-measured, never reported.

5. **The asset under test was built for this target.** The emit's `--gpu`/
   `--arch`/`--n-cu` and the loaded device object (`$BUILD`) all name `$GPU` /
   `$ISA`; a block benched against a foreign object measures the wrong machine.

---

## Pitfalls (from real block campaigns)

* **Block ms does not scale to model ms.** There is one embed and one lm_head in
  both a block and the model, so the fixed term is amortized over N layers in the
  model but over just the block's layers in isolation. A truncated-model score is
  a **comparator/ranking**, not a predicted TTFT — treat it as one.

* **Isolated single-block decode over-states the norm floor.** A whole isolated
  block launches ~10 separate RMSNorm/RoPE/residual/combine kernels, each paying
  a launch-latency floor at M=1 (on one measured part, ~180 us of a 356 us
  whole-block number — re-measure the split on `$GPU`, do not reuse the ratio).
  plow fuses these into the packet, so that overhead is an artifact of
  running one block in isolation, not a real per-block cost in a full-model graph.
  **The transferable numbers are the per-op GEMV/attn rows, not the whole-block
  us.**

* **An eager reference baseline flatters plow.** At decode (M=1) a block is
  ~15–20 tiny ops; eager mode launches each separately, and on one measured part
  that pure launch overhead was ~100–150 us — enough to flatter plow ~2.5×.
  Capture the decode step into a device graph (the block-baseline harness does
  this by default) before comparing — an eager number is not a real target.

* **A few-instances-per-block op lands in noise.** A block has only ~4–5 MoE
  layers, so ablating `MoeRouterTopk` or `MoeCombine` on a block sweep is inside
  timer noise (one measured campaign: base 3.184, router-ablated 3.152,
  combine-ablated 3.230 ms/token — one spread apart). Those need the full
  network. The spread is per-target; re-establish it on `$GPU` before calling an
  ablation null.

* **A tune knob measured on an isolated kernel can invert in the megakernel.**
  A GEMV harness once timed one `gemv_rows<16>` as equal to two `gemv_rows<8>`,
  while in the real megakernel on the same part the knob went 41.17 → 28.8 ms. A truncated
  *model* blob runs the real megakernel and keeps the context an isolated kernel
  loses — prefer it for ranking runtime knobs.

* **Reference geometry can be an approximation.** The plow-native example
  descriptors (`crates/plowc/examples/*.json`) may linearize a MoE (e.g.
  8-expert/top-2 vs a real 128-expert/top-8). Expert count and top-k drive fused
  efficiency directly, so fix the descriptor to the real geometry before
  comparing MoE against a served number.

* **Cross-GPU block numbers are not comparable.** A per-block floor is only valid
  when the reference framework was measured on the *same* card — and this holds
  between two parts at the same `$ISA` too, since they can differ in memory
  subsystem and CU/SM count while emitting identical code. Re-anchor before
  claiming any ratio.

* **The tune system is not always a lever.** On some GPU/precision paths all GEMM
  opcodes alias to one body with a fixed compile-time tile, and the decode GEMV
  has no tunable kernel at all. Confirm the tuner actually selects something
  before treating "tuned" as a step (see Stage 6). The real axis on such paths is
  a build knob (rebuilding the device object via `$BUILD` with different `-D`
  macros), outside the tune system.

* **Driver / toolchain mismatch fails both halves silently-ish.** Prebuilt device
  objects and a reference venv's runtime must match the installed driver
  (`CUDA_ERROR_INVALID_IMAGE` / "driver too old", or the HSA-side equivalent).
  Confirm `$TOOLCHAIN` and the driver agree before blaming the block.

* **Differencing measures MARGINAL cost, and the critical path is what you
  want.** Ablating an op and taking the delta prices what removing it saves
  *given everything else still runs* — which is ~0 for anything that overlaps.
  One V4 campaign measured `V4GroupedLinear` and `V4MoeRoute` at ~0 this way and
  spent days elsewhere; the packet trace then showed them closing **61.3%** and
  **11.2%** of the critical path, and fixing them gave 2.2× and 1.36×. Use
  differencing to confirm a hypothesis, never to generate one.

* **A measurement taken with an env knob set is not a measurement of the
  default.** An entire V4 branch was measured with `PLOW_V4_NCU=128` exported
  while the committed default was 304 — a program 2.1× slower than every number
  in the log. Re-emit clean (`env -u`) before believing a committed default, and
  put the measured value in the code, not in your shell history.

---

## The per-packet floor, and per-op dispatch width

The single most transferable finding of the V4 bring-up, and it applies to any
architecture with many layers and many small ops per layer.

**A packet costs something even when it computes nothing.** In the persistent
interpreter that floor is *per-workgroup cache maintenance*: an agent-scope
release emits `buffer_wbl2` and an acquire emits `buffer_inv`, both **per-L2**,
so every workgroup makes its XCD's L2 repeat the same writeback and invalidate
and they serialise. `runtime/bench/ctr_convergence.hip` prices an empty
256-workgroup packet at **13.2 µs**, of which the counter and the atomic are
0.07–0.14 — it is nearly all maintenance.

Measured on the shipped V4 program (`PLOW_TRACE_RAW`, one block's
`t_end - t_ready`, tick = 10 ns):

| packet | what it computes | at 128 CUs |
|---|---|---|
| `V4HcZero` | writes **25 floats** | 34.6 µs |
| `RmsNorm` | one row | 32.6 µs |
| `V4HcExpand` | 4096 elements | 31.8 µs |
| `GemvFp8Blk` | a real projection | 46.5 µs |

A kernel that writes 25 floats cost three quarters of a GEMV. **That is the
floor, not the work** — and it is why a many-layer decode program can sit at 6%
of peak bandwidth with every kernel individually well tuned.

### The check to run

1. **Roofline first, before any kernel work.** `perf-data/tools/*_decode_budget.py`
   (write one for the new arch — it is an afternoon) gives bytes/token and
   therefore the ceiling. V4's said 2.72 ms/token = 367 tok/s while the program
   ran at 45 ms. A 16× gap is never kernel arithmetic; it is structural.
2. **Take a packet trace and compute one block's execution per instruction.**
   If `sum over instructions of (one block's exec)` ≈ the total span, the program
   is **serial** and its cost is `packets × floor`. V4: 101.4%.
3. **Divide the step by the packet count.** V4 was 1505 packets against K3's
   302 at a comparable size. If mean-µs-per-packet is flat across ops that do
   wildly different amounts of work, you are floor-bound, and no kernel change
   will move it.

### The lever: give each packet only the CUs its kernel can use

This is a property of the **kernel's grid mapping**, not of the model, so the
rules are shared. `crates/devgen/src/k3.rs` has them and they are meant to be
reused verbatim by new architectures:

| helper | rule | because |
|---|---|---|
| `norm_cus(cus, rows)` | `rows` | `d_rmsnorm` reduces a row inside ONE workgroup; at decode `rows = 1` |
| `k3_rope_cus(cus, start, ntok, nhead)` | `ceil(ntok*nhead / waves)` | one wave per (token, head); `start` keeps a concurrent q/k pair on **disjoint** slices |
| `combine_cus(cus, n)` | `ceil(n / threads)` | one element per thread |
| `vec8_cus(cus, n)` | `ceil(n / (threads*8))` | the `bf16v8` bodies stride by 8 |

Read the kernel before picking one — the mapping is not always the obvious one.
`d_hc_expand` looks like it covers `T*HC*D` but is one element per thread over
`T*D` (each thread writes all `HC` streams for its depth), so it takes the
*combine* rule. `d_v4_hc_mix` runs its Sinkhorn tail on thread 0 of **every**
block with an unstrided token loop, and only its output pass takes a slice — so
56 of 64 workgroups re-ran a 20-iteration Sinkhorn to discard it.

Narrowing is **bit-identical** whenever the dropped slices wrote nothing, which
is exactly the condition that makes it a win.

### The trap: the global width and the per-op rules are coupled

Both are U-shaped and they interact. Swept on V4 twice:

```
   uniform width, every packet     with per-op widths applied
      4  396.8 ms/token                 64  44.6
     16  112.9                          96  40.2
     32   69.3                         128  38.2   <- min
     64   52.1  <- min                 192  44.3
    128   53.5                         256  47.8
    304   81.0                         304  51.6
```

**The knee moved, 64 → 128.** Below it the wide ops lose bandwidth they could
use; above it the maintenance round costs more than the extra CUs return.
Pinning the ops that cannot use width takes their share out of the second
effect, so the wide ops get to keep the CUs the narrow ones were wasting.

Consequence: **re-sweep the global width every time the per-op rules change.**
Measured against the stale width the V4 narrowing read as 1.17×; at the width
its own knee had moved to, it was 1.40×.

### When the floor itself is the target

If packets × floor still dominates after widths are right, the remaining moves
are, in increasing order of cost:

* **Fuse packets.** `k3::fuse_norm_gemv` folds an `RmsNorm` into the GEMV that
  consumes it (`norm = 2`), deleting a packet, an edge and a chain level at
  once. Check the op actually carries a norm immediate first — the block-fp8
  decode GEMV does not.
* **Cut the maintenance itself.** `PLOW_GATE_HIER` does the writeback/invalidate
  **once per XCD** instead of once per workgroup: 13.2 → 3.46 µs on the
  microbench, a 3.8×. It requires `PLOW_L2_PLACE_DISPATCH` and per-domain
  windows (`PLOW_SE_NPER`) from the emitter, without which it silently reads as
  "no hierarchy" and does nothing.
* **Do not fuse blindly.** V4's fused `d_hc_reduce` exists and passes its gate,
  but is token-parallel — at `T=1` it collapses onto one CU and is *worse* than
  the three packets it replaces. Check the fused arm's grid mapping at the
  batch size you actually run.

---

## Code pointers

| symbol / path | role |
|---|---|
| `devgen::block::parse_block` / `write_block_descriptor` (`crates/devgen/src/block.rs`) | `--block l[..r]` parsing + sibling `block.json` |
| `glm_emit_block`, `kimi_emit_block`, `nemotron_emit_block` (`crates/devgen/src/mla.rs`) | MLA-family block extraction |
| `glm_build_block`, `glm_build_block_pf` (`crates/devgen/src/mla.rs`) | GLM/Kimi decode + prefill block builders |
| `plowc --emit devblob --block` (`crates/plowc/src/main.rs`) | the block emit entry point |
| `plowrt amd-block` (`crates/plowrt/src/main.rs::amd_block`) | act.x-in / act.x-out A/B numerics runner |
| `plowrt disasm` (`crates/plowrt/src/main.rs::disasm_cmd`, `crates/plowrt/src/disasm.rs`) | static blob inspection |
| `plowrt simulate` (`crates/plowrt/src/main.rs`) | CPU sim, `--math dry|golden` |
| `examples/block_run` (`crates/plowrt/examples/block_run.rs`) | Gemma-path block `check` + `bench` |
| `scripts/block_sim.sh` | compile → CPU sim one block |
| `scripts/block_e2e.sh` | end-to-end compile → plow bench → reference bench → diff (see caveats) |
| `scripts/block_baseline.py`, `scripts/block_layer_bench.py`, `scripts/block_op_bench.py` | reference-side single-block harnesses |
| `scripts/block_compare.py` | join plow `sweep.json` vs reference `sweep.json` |
| `scripts/build_glm52_block.sh`, `scripts/k3_block_sweep.sh`, `scripts/glm52_block_sweep_gfx942.sh` | family-specific build / sweep |
| `runtime/tests/*_block_*_test.c` + `runtime/tests/*_oracle.py` | C oracle correctness harnesses |
| `perf-data/block-baseline-harness.md`, `perf-data/block-decode-baseline-26b.md` | recorded methodology + reference numbers |

Related context: `docs/arch/13-prefill-chunking.md` (why a prompt is a sum of
compiled rungs — the block's prefill buckets), `docs/arch/03-scheduler.md` (how
a block's tasks map onto executors).
