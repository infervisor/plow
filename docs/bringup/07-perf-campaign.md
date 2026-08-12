# Stage 7 — End-to-End Performance Campaign

> Turn a serving recipe into a **defensible, reproducible number**. Measure
> latency and throughput versus concurrency and context on the whole served
> model, behind a correctness battery, and write it up in the established
> `perf-data/` format so a reader can rebuild every cell. This is the final
> stage: its output is a committed campaign write-up, not another tuning round.

**Precondition:** Stage 6 complete — the model serves at its target concurrency
with a recorded recipe (`plowrt serve` command line + `PLOW_*` knobs + measured
TTFT/TPOT/throughput/memory), memory fits, no spurious shedding, and every
numerics-changing lever was re-verified. The [`target.md`](target.md) block is
filled in and is part of the campaign header: a campaign is a statement about
`$GPU` at `$ISA`, `$NGPU`/`$PARALLEL`, `$MAXCTX`, and it is written to
`$RESULTS`. Nothing here recompiles or re-tunes; Stage 7 only *measures* and
*documents*. A number that moves a lever belongs in Stage 5/6, not here.

**Gate out (bringup complete):** the correctness battery passes (token-identity
gate + at least one accuracy number), the performance targets are met and
measured — same-session, uncontended, concurrency- and context-swept — and the
whole campaign is written up in `perf-data/` with every cell traceable to a
source file and a reproduction command. When this gate passes, the model is
brought up.

This stage carries one non-negotiable convention from every historical campaign:
**the tone is neutral and research-grade.** Report what the box did. There is no
"win/loss" framing, no leaderboard voice — a comparison against a reference
framework is stated as a ratio with its caveats (`plow leads TPOT from ~16k up,
trails prefill ~2×`), never as a victory or a defeat. Every number that could be
read as flattering carries the caveat that makes it honest.

---

## What a campaign is (and is not)

A campaign is a **fixed, pre-declared measurement plan** run once, cleanly, and
transcribed verbatim. It is not an optimization loop and not a place to discover
new levers — if a run suggests a lever, note it as open work and finish the
campaign at the recipe you gated on.

Three properties separate a campaign number from a benchmark you can't trust:

* **Same-session.** Both engines (plow and any comparator) measured in **one
  lease, one session, on one box**, interleaved. A stored CSV is not a baseline:
  one recorded campaign has a comparator number moving **33% (7.57 → 10.03 ms)**
  on re-measure, which silently re-rated every ratio in that directory
  (`perf-data/plow-gfx942/README.md`). Re-baselining both engines in one session
  is repeatedly called out as *the single most valuable missing measurement*.
  This applies across parts too: a baseline measured on another GPU is not a
  baseline for `$GPU`.
* **Behind a correctness gate.** No performance cell is recorded for an arm that
  has not passed the token-identity gate that same build, plus at least one
  accuracy number for the campaign (`perf-data/archive/k3/k3-gsm8k.md`). A fast wrong kernel
  is a wrong kernel.
* **Swept, not single-point.** plow's fixed-width decode makes its position
  strongly concurrency- and context-dependent — in one recorded campaign it led
  at c4, tied at c8 and trailed at c16 on the same model
  (`perf-data/b2-concurrency-family.md`). One point is not a capacity claim; a
  curve is. Where the crossovers land is a property of `$GPU`.

---

## The harness

Two client families, both driving the running `plowrt serve` over its OpenAI
route (`/v1/chat/completions` — the only route plowrt implements). Pick per what
you are measuring; **use the same client for every arm of one campaign.**

| harness | what it drives | where |
|---|---|---|
| `huggingface/inference-benchmarker` (pinned rev) | multi-user **capacity** sweeps: fixed virtual users (ConstantVUs), warm+measure windows, aggregate tok/s + TTFT/ITL percentiles | invoked via `perf-data/bench_ib.sh` / `bench_b2_ib.sh` (parameterized: `CAMPAIGN`, `PROMPT_TOKS`, `VUS`, `MODEL_NAME`, `TOKENIZER`, `ASSETS`) |
| `vllm bench serve` (`--backend openai-chat`) | single-stream + low-concurrency latency, and the like-for-like comparator (same client both engines) | `perf-data/tools/bringup_bench.sh`, orchestrated by `bringup_showdown.sh` |

Supporting scripts in `perf-data/tools/`:

| script | role |
|---|---|
| `gpulease <label> <cmd>` | advisory GPU lease + contention audit. Wraps the **run**, not the build. Exits **76** if the GPU was contended — that run's timings are void. See the contention pitfall below. |
| `bringup_gate.sh <assets> <tag> <port>` | token-identity correctness gate: serves the bundle, runs 4 fixed greedy prompts (temp 0, 32 tok), dumps to `$BRINGUP_OUT/gate-out/<tag>.txt` for `diff` against the reference arm |
| `bringup_bench.sh <tag> <url> <model> <tokenizer> [round]` | one client pass over one endpoint; env `IN_LENS`/`NPROMPT`/`OUTLEN`; appends one row per input length to `cells.tsv` |
| `bringup_showdown.sh` | sequential-**exclusive** multi-arm template — one server at a time, kill+drain between arms, medians over ≥5 rounds |
| `bringup_ceiling.py` | vendor-BLAS/torch fp8+bf16 GEMM ceiling at your model's exact prefill shapes — the roofline reference (edit `SHAPES`); written against cuBLASLt, so on `$VENDOR = amd` it needs the hipBLASLt equivalent or Stage 4's `$COMPUTE_CEIL` instead |
| `consolidate_b2_ib.py`, `b2-ib/{slo_capacity,summarize}.py` | ingest the tool's raw report JSON into `b2-concurrency-*.json`; derive max-users-under-SLO. Scratch reducers — the markdown table is typed by hand from their output |

Accuracy battery: `scripts/bench_gsm8k.sh` (GSM8K 8-shot greedy) is the
whole-stack accuracy gate — it exercises chat template, channel stop, tokenizer,
prefill, on-device sampling and SSE in a way `amd-bench` cannot.

> **`bench_ib.sh` may not be in the tree.** The `inference-benchmarker` driver is
> referenced by the historical campaigns (`b2-concurrency-family.md`,
> `px1-stage1.md`) but is not always committed here. If it is absent, use
> `bringup_bench.sh` / `bringup_showdown.sh` with `vllm bench serve` for the
> like-for-like arm, and drive the pinned `inference-benchmarker` directly for
> the capacity sweep — the profile (warm+measure windows, ConstantVUs, greedy,
> fixed output tokens) is what matters, not the wrapper.

---

## Step by step

Fix the campaign plan first, in writing: **models, precisions, prompt lengths,
output length, the concurrency ladder, the context ladder, the SLOs, and the
comparator (if any).** Everything below measures against it. Commands assume a
built `plowrt` (`nix develop`; `cargo build -p plowrt --release $FEATURES`) and
the Stage 6 recipe.

### 0. Environment sanity (invalidates everything downstream)

Before any number, confirm the box:

* **GPU actually used** — grep the serve log for `backend ready — GPU
  accelerated`, every arm, every run. This is the single cheapest check in the
  campaign and the most expensive one to skip. When the device backend cannot
  initialize, `plowrt` falls back to the **CPU reference interpreter**, which is
  a `WARNING`, not an error, and which **still serves correct text** — so the
  bench completes, the tokens are right, the gate passes, and the number
  describes the CPU path. Nothing downstream catches it. Confirm the
  GPU-accelerated line before you trust any number, on `$VENDOR` either way.

  The *cause* of a silent fallback is vendor-specific, so check it per `$VENDOR`
  once per box:

  | `$VENDOR` | what fails silently | fix |
  |---|---|---|
  | nvidia | a CUDA **compat** `libcuda` newer than the installed driver fails `cuInit` (`CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`); `plowrt` warns and takes the CPU path | point `--libcuda` / `PLOW_LIBCUDA` at the system library rather than the compat one, and check `nvidia-smi --query-gpu=driver_version` against it |
  | amd | ROCr/`hsa` init failure or a missing `.hsaco` directory | `--rt-hsaco` / `PLOW_HSACO`; confirm the ROCm runtime and the driver agree |

* No resident foreign compute (the `$VENDOR` SMI tool — `nvidia-smi
  --query-compute-apps=...` or `rocm-smi --showuse` — reads 0% before any A/B;
  and it must cover all `$NGPU` devices) — the persistent cooperative
  megakernel outlives its host process, so a bench started while a prior `serve`
  tears down reads a contended box.
* `$TOOLCHAIN` and the driver agree; `nix develop` for the host side; profiler
  availability for `$VENDOR` (`ncu`/`nsys`, or `rocprof`) checked once per box.
* **Comparator availability — decide this now, not at step 3.** A same-session
  baseline needs a comparison engine that can *soundly serve this model on
  `$ISA`*, on this box. Establish that it exists before the campaign starts,
  because discovering it at step 3 wastes the whole run. Two failure modes have
  already cost real campaigns: a client-only benchmark shell being mistaken for
  an engine (a CPU-only build measures nothing about the GPU), and the vendor's
  supported engine shipping only as a container image on a host with no
  container runtime. If no sound comparator exists, say so **here**, scope the
  campaign to self-comparison against plow's own prior revision, and do not let
  a stored or cross-hardware number stand in for it later.

### 1. Correctness battery (before ANY performance cell)

Two levels; both required for a campaign.

**Token-identity gate** — for every serving arm you will bench, that same build:

```bash
perf-data/tools/gpulease gate \
  perf-data/tools/bringup_gate.sh $ASSETS plow-fp8 8080
# reference arm, then:
diff $BRINGUP_OUT/gate-out/plow-bf16.txt $BRINGUP_OUT/gate-out/plow-fp8.txt
```

Classify: **token-identical** (required for refactors / staging / reordering),
**coherent-but-shifted** (acceptable only for a *documented* numerics change —
fp8 accumulation, reduction order — and called out), or garbage/truncation/wrong
first token (**reject** — a corrupt first token means the lm_head path is broken
even if decode "recovers"). On TP, every rank must emit the identical stream.

> `amd-bench`'s `last id` is **not** a correctness signal — it never prefills, so
> attention reads a KV cache that was never computed. Gate on the *serve* path
> with a real prompt (`perf-data/plow-gfx942/README.md`).

**Four checks the fixed-prompt gate cannot make.** The gate runs four short
greedy prompts; each of the following is a distinct way to be silently wrong
that those four prompts never reach. Run each **once per bring-up**, on any
`$VENDOR` — none of them is target-specific:

1. **Multi-chunk prefill.** Send a prompt longer than the emitted max chunk, so
   the prefill runs as more than one chunk and the sliding-layer KV ring
   wraps. The gate's prompts fit in one chunk; ring wraparound is only exercised
   here.
2. **Stop behavior.** Confirm generation actually stops at eos, *and* that
   `"ignore_eos": true` runs to the cap and reports `finish_reason: "length"`.
   Stage 3's cross-engine token comparison is meaningless if this is wrong.
3. **The output cap is honored.** Send `max_completion_tokens` and check the
   response respects it. OpenAI renamed `max_tokens`; a server binding only the
   old spelling silently runs to eos, which reads as a throughput result rather
   than a bug.
4. **Lossy modes are opt-in and labelled.** A lossy config (fp8 KV, fp8
   activations, any non-parity-preserving flash arm) diverges from greedy bf16
   by design — one recorded fp8-KV path diverges after ~21 tokens. Never compare
   a lossy plow config against a bf16 baseline without saying so in the write-up.

**Accuracy number** — at least one per campaign:

```bash
perf-data/tools/gpulease gsm8k scripts/bench_gsm8k.sh   # 8-shot greedy, n=200
```

A throughput number without an accuracy number is not publishable against a
comparator (`perf-data/archive/k3/k3-gsm8k.md`): token-identity proves self-consistency
only — a blob wrong the same way on every rank passes it.

### 2. Roofline sanity (so "slow" vs "the box is the wall" is decided by arithmetic)

Measure the practical ceiling at *your* shapes before quoting any efficiency
claim:

```bash
# $VENDOR = nvidia only — see the gap note below.
perf-data/tools/gpulease ceil python3 perf-data/tools/bringup_ceiling.py
```

The output is `$GPU`'s practical prefill ceiling at your shapes — that is what
separates "the kernel is slow" from "the box is saturated". (For calibration of
what the output looks like, one recorded run on GH200/12B measured fp8
1324–1468 / bf16 804–861 TF/s, `perf-data/gemma12b-gh200-prefill-campaign.md`;
that is *that* box's ceiling, not a target for yours.)

> **Gap: there is no in-tree ceiling harness for `$VENDOR = amd`.**
> `bringup_ceiling.py` is written against cuBLASLt and nothing in the tree ports
> it to hipBLASLt. On AMD this step has no drop-in: either write the hipBLASLt
> equivalent for your shapes, or carry Stage 4's `$COMPUTE_CEIL` forward and say
> in the write-up that the prefill ceiling is a Stage-4 single-unit figure rather
> than a vendor-BLAS measurement at the campaign's exact GEMM shapes. Do not
> quote an efficiency % against a denominator you did not establish — that is the
> failure [`target.md`](target.md) exists to prevent.

Decode capacity is usually **HBM-bandwidth-bound**: state the bytes moved and
the achieved GB/s, and the % against `$BW_BOUND` — naming whether `$BW_BOUND`
is a measured figure or the datasheet peak `bandwidth_for_bound()` falls back to
(see [`target.md`](target.md)). One recorded campaign derived plow at ~970 GB/s
= 18% of that part's peak vs a comparator at ~2.4 TB/s = 45%
(`perf-data/plow-gfx942/README.md`) — an example of the arithmetic, not a bar.

### 3. Same-session baseline (both engines, interleaved)

One server at a time, on the same box, same client, medians over ≥5 rounds
(9 for a headline). `bringup_showdown.sh` is the template — edit the arm list:

```bash
SNAP=$HF_SNAPSHOT MODEL_ID=$SERVED_ID BUNDLES=$BUNDLE_DIR CUBINS=$CUBIN_DIR \
IN_LENS="1024 4096" NPROMPT=9 \
  perf-data/tools/gpulease showdown perf-data/tools/bringup_showdown.sh
```

Mandatory cross-engine checks:

* **Identical input tokens** — compare `Total input tokens` across engines.
* **Identical output tokens** — `Total generated tokens` must match; plowrt must
  honor `ignore_eos` or it emits far fewer tokens/request (measured 161 vs 512 →
  3.2× distortion).
* **Raise `--slo-ms` for throughput arms** — the default 250 ms sheds requests
  once predicted wait exceeds it, and the bench counts those 429s as
  **successful** with ~12 tokens each → a fake "2592 tok/s". Always re-check
  `Total generated tokens`.
* **The first request after start pays a one-time warmup** (~57 ms observed);
  medians absorb it, means do not.

### 4. Concurrency sweep (capacity)

The capacity contest. Fixed virtual users (ConstantVUs), warm window + measure
window, greedy, fixed output tokens, TTFT **including** server-side queueing (the
capacity convention). Sweep the ladder — do not quote one point:

```bash
CAMPAIGN=b2-12b PROMPT_TOKS=4096 VUS="1 2 4 8 16 32" \
MODEL_NAME=$SERVED_ID TOKENIZER=$HF_SNAPSHOT ASSETS=$BUNDLE_DIR \
  perf-data/tools/gpulease b2 perf-data/bench_b2_ib.sh
python3 perf-data/consolidate_b2_ib.py           # raw JSON -> b2-concurrency-*.json
python3 perf-data/tools/b2-ib/slo_capacity.py  # max-users under each SLO
```

Report per VU: aggregate tok/s, TTFT avg/p99, ITL(=TPOT) avg/p99, and tok/req
(to catch a shed truncation). Derive **max sustainable users** = highest VU with
**zero** failed requests under each stated SLO (e.g. ITL p99 ≤ 50 ms, TTFT p99 ≤
5 s), and the unconstrained peak-throughput point separately. Read the shape: a
flat decode ITL with an exploding TTFT is **mux queueing above `B`**, not a
decode regression — that IS the capacity answer for a fixed-width engine.

### 5. Context sweep (context scaling)

At concurrency 1, sweep the prompt length ladder (`IN_LENS="1024 4096 8192 16384
32768 65536"`, every point ≤ `$MAXCTX`), reporting TTFT and TPOT at each. This
is where plow's sliding-window / flash-decode path is characterized: TPOT is
often close to context-flat while a comparator's grows (one recorded 12B
campaign: plow TPOT +7.5% over 1k→64k vs the comparator's +87%,
`perf-data/plow-gfx942/README.md`). State the crossover if you compute one, and
state that it is a trend, not a served point.

> Watch the chunk ladder vs your benchmark lengths. The chat template pushes a
> "4096-token" prompt to ~4110 rows; if the ladder splits it `[4096, 128]`, the
> 128-tail is a second full-model pass (~30–36 ms) the comparator does not pay —
> credit it when reading a TTFT ratio
> (`perf-data/plow-gfx942/README.md` caveat 3). A ctx blob that cannot hold
> `input + output` refuses the request and reports `tok/req` far below target and
> `TPOT 0.000` — that is a **compile-time size error, not a measurement**;
> recompile at a larger `--max-ctx` (and update `$MAXCTX`), do not re-run.

### 6. Consolidate and write the results doc

Transcribe every number verbatim from the tool's own report JSON — nothing
interpolated, nothing projected. The consolidators (`consolidate_b2_ib.py`)
build the machine-readable `*.json`; **the markdown companion is written by hand
from those JSONs** and carries the prose, the caveats, and the honesty banner.

**The honesty banner is a required artifact, not a style choice.** Every results
doc opens with an explicit two-list statement of what was and was not run:

```text
## Honesty banner

Ran: <the cells that actually executed, with the gates they passed>
Not run: <every cell that did not — comparator, concurrency ladder, context
ladder, accuracy — and why>
```

Then the campaign reports **only** what the "Ran" list covers. A cell that did
not run is not a gap to be filled by a stored number, a projection, or a result
from another box: if there was no sound same-session comparator, the doc says so
and makes no comparative claim at all. A campaign that ends blocked and says
which gate blocked it is a completed campaign — it is a weakened gate that
corrupts the record, not a red one.

Two things the banner has to be honest about specifically, because both have
been recorded wrongly before:

* **What a gate actually executed**, not merely that it returned green. A gate
  that reported success while silently skipping the audit it was asked to
  perform is a failed gate that looked like a passed one; quote what it printed.
* **Which denominator every percentage used** — measured on `$GPU`, or an
  unmeasured datasheet peak (Stage 4, Step 0).

---

## Where results live and how they're written

Committed under `$RESULTS` — `perf-data/plow-<isa>/` if such a directory exists
for `$ISA` (today only `perf-data/plow-gfx942/` does, linked from the README's
perf section), else the `perf-data/` root. A campaign targeting one GPU family
lands in the per-ISA home; cross-arch capacity reports live at the `perf-data/`
root (e.g. `b2-concurrency-family.md`, `serving-capacity-report.md`). If `$ISA`
has no home yet, create one on the same pattern rather than filing an
ISA-specific result under another ISA's directory.

Structure a campaign write-up to the established format:

* **Header line**: date, branch @ commit, box (`$GPU`, `$ISA`, `$NCU`,
  `$NGPU`/`$PARALLEL`, `$MAXCTX`, driver, `$TOOLCHAIN`), engine configs (both),
  harness (tool + pinned rev), profile (warm + measure durations, greedy, output
  tokens), the TTFT convention, and which denominators `$BW_BOUND` /
  `$COMPUTE_CEIL` were (measured here, or datasheet).
* **Scope-actually-measured banner** (an *honesty banner*): state plainly what
  ran and, explicitly, **what did not** — a campaign stopped early by operator
  request labels the un-run rows and does not project them. A per-ISA results
  home additionally requires a `Scope:` class line — how far a finding
  generalizes (this µarch only / this vendor / plow-architectural / method /
  …) — so a finding cannot silently inherit the directory's arch claim.
  `perf-data/plow-gfx942/README.md` shows the convention in use.
* **Tables**: one row per (tag, VU) or (tag, ctx); percentiles included; a
  `valid: false` flag on any row whose serving config failed its gate (kept for
  the record, never used for a claim).
* **"What holds each model back," per model** — the honest limiter, with the
  roofline arithmetic.
* **"What's still open"** — staged, one lease each; every driver parameterized so
  dropping a missing JSON in lights up the row with no code change.
* **Data / reproduction pointers** — the `*.json`, the driver script + its env
  vars, the reducers.

Neutral voice throughout. A comparator ratio is a fact with a caveat
(`1.096× at 8k — inside the range a fresh baseline could move`), never a verdict.

---

## Success criteria

The model completes bringup when **all** hold:

1. **Correctness battery passes.** Every benched arm is token-identity gated that
   build (or its shift is documented), and at least one accuracy number
   (GSM8K-class) is recorded. On TP, rank token-identity holds.
2. **Performance targets met and measured** at the target concurrency and
   context — not projected. TTFT/TPOT within budget; capacity (max users under
   SLO) and peak throughput both stated; context scaling characterized.
3. **Same-session and uncontended.** Both engines measured in one lease; every
   timed run is `gpulease` rc=0; a stored baseline is not quoted as a
   same-session number.
4. **Swept, not single-point.** A concurrency ladder and a context ladder, not
   one cell.
5. **Documented in `$RESULTS`.** A committed write-up in the established format
   — header, honesty banner, tables with `valid` flags, per-model limiter,
   open-work list, reproduction pointers — every cell traceable to a source JSON
   and a command. Neutral tone; no win/loss framing.
6. **The target is on the record.** The write-up names `$GPU`, `$ISA`, `$NCU`,
   `$NGPU`/`$PARALLEL`, `$MAXCTX` and the provenance of every denominator, and
   its `Scope:` line says how far the finding generalizes. An unscoped number
   will be reused on a part it was never measured on.

---

## Pitfalls (from real campaigns)

* **A contended run silently invalidates every number.** `gpulease` rc=76 means a
  foreign compute process was resident; discard and re-run. The lease is
  **advisory** (flock) — it cannot stop a process that never called it, so audit
  the box before *and* after. `rocm-smi --showuse` / `nvidia-smi` must read 0%
  first; the persistent megakernel outlives its host, so a bench started during a
  prior `serve` teardown reads contention (13.5 ms outliers against a 12.0 ms
  baseline, `perf-data/plow-gfx942/README.md`).
* **A stored baseline is not a same-session number.** A recorded vLLM CSV moved
  33% on re-measure and re-rated a whole directory of ratios. Re-baseline both
  engines in one session or label the ratio as stored-baseline.
* **A/B order / session drift bias.** Arms benched under different background load
  are not comparable; when a foreign server appears or exits, **re-baseline
  everything** (`perf-data/amd-bench-ab-order-bias.md`).
* **Shed requests bench as "successful"** with ~12 tokens → fake tok/s. Raise
  `--slo-ms` for throughput runs and re-check `Total generated tokens`.
* **`ignore_eos` mismatch distorts throughput** (measured 161 vs 512 tok/req →
  3.2×). Compare `Total generated tokens` across engines every arm.
* **A wide fixed `B` doubles per-token latency for the same throughput** — decode
  is bandwidth-bound; the SLO-bounded capacity is set by bandwidth and prefill,
  not the batch size (`serving-capacity-report.md`, `b2-concurrency-family.md`).
* **Only concurrency-1 rows compare kernels on a single-sequence engine.** On
  `$VENDOR = amd`, plow's serve is `batch=1` per rank, so its concurrency ladder
  measures requests *queueing* while a continuous-batching comparator climbs —
  the ratio columns there are not a kernel comparison
  (`perf-data/plow-gfx942/README.md` caveat 2).
* **Chat-route token inflation.** The template's extra tokens push a prompt past
  its bucket → an extra tail chunk of prefill the comparator did not pay; credit
  the ~11% when reading a TTFT ratio.
* **A ctx/profile mismatch is a refusal, not a slow run.** `input + output >
  max-ctx` → request rejected, `tok/req` collapses, `TPOT 0.000`. Flag the row
  `valid: false`; recompile, don't re-run.
* **A gate that never re-emits cannot catch an emitter regression.** If the
  campaign re-uses a stored asset and rebuilds only objects, an emitter change
  cannot reach it (`perf-data/plow-gfx942/README.md`: a stored asset kept
  certifying a blob that could not contain the regression under test). Emit from
  the current checkout for the gate.
* **Never launch a battery twice / kill with SIGTERM.** Two copies on one port →
  one server answers both arms, delta meaningless. `kill -9` leaves the
  persistent megakernel resident, corrupting later runs; send SIGTERM and let it
  tear down.
* **Standalone probes overstate** (THE PROBE LAW): in recorded cases a +15%
  probe was −3 ms in-model, and an isolated GEMV scored a lever 1.00× where the
  megakernel measured 1.43×. A campaign quotes only in-model, in-serve numbers.
* **A number from another part is not a target.** Prior campaigns in
  `perf-data/` are attributed to the box they ran on; reuse their *method*, and
  re-measure their *values* on `$GPU`. Two parts at the same `$ISA` differ here
  too.

---

## Code / harness pointers

| path | role |
|---|---|
| `perf-data/tools/gpulease` | advisory lease + contention audit (rc=76 = contended) |
| `perf-data/tools/bringup_gate.sh` | token-identity correctness gate |
| `perf-data/tools/bringup_bench.sh` | one client pass (`vllm bench serve`), appends `cells.tsv` |
| `perf-data/tools/bringup_showdown.sh` | sequential-exclusive multi-arm showdown template |
| `perf-data/tools/bringup_ceiling.py` | vendor-BLAS/torch GEMM ceiling at your exact shapes (cuBLASLt-written) |
| `perf-data/tools/README.md` | what each harness script is and how to drive it |
| `perf-data/bench_ib.sh` / `bench_b2_ib.sh` | `inference-benchmarker` capacity driver (parameterized; may be absent — see note) |
| `perf-data/consolidate_b2_ib.py`, `b2-ib/{slo_capacity,summarize}.py` | raw report JSON → `b2-concurrency-*.json`, max-users-under-SLO |
| `scripts/bench_gsm8k.sh` | whole-stack accuracy battery (GSM8K 8-shot greedy) |
| `$RESULTS` (`perf-data/plow-gfx942/` is the one per-ISA home in the tree today) | the per-ISA results home (README = index, LESSONS = method) |
| `perf-data/b2-concurrency-family.md`, `serving-capacity-report.md` | the model campaign template (tables, honesty banner, per-model limiter, open-work) |
| `perf-data/archive/k3/k3-gsm8k.md`, `perf-data/gemma12b-gh200-prefill-campaign.md` | accuracy-number method; roofline / prefill campaign |

Architecture reading: `docs/arch/06-runtime.md` (execution model),
`docs/arch/11-tuning-coverage.md` and `docs/arch/12-using-the-tuner.md` (what the
tuner can and cannot select — most large wins are structural, not tuner-axis).
Prior stage: `docs/bringup/06-runtime-opt.md` (the recipe this campaign measures).
