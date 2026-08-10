# Agent — Stage 7: End-to-End Performance Campaign

You are executing **Stage 7** of the model-bringup playbook — the final stage.
Your job: run one clean, reproducible end-to-end performance campaign for a model
that already has a serving recipe (Stage 6), behind a correctness battery, and
write it up honestly in `perf-data/`. Read
[`docs/bringup/07-perf-campaign.md`](../07-perf-campaign.md) and
[`perf-data/harness/BRINGUP.md`](../../../perf-data/harness/BRINGUP.md) first — they
are authoritative and hold the full harness, commands, and pitfalls. This prompt
is the executable checklist.

**Nothing here recompiles or re-tunes.** A campaign measures the gated recipe and
documents it. If a run suggests a new lever, record it as open work and finish
the campaign at the recipe you gated on — do not start a Stage 5/6 loop.

The tone of everything you produce is **neutral and research-grade.** Report what
the box did. No win/loss framing, no leaderboard voice — a comparison is a ratio
with its caveats, never a victory or defeat. Every flattering-looking number
carries the caveat that makes it honest.

## Preconditions (from Stage 6)

* A recorded serving recipe: the `plowrt serve` command line + `PLOW_*` knobs +
  measured TTFT/TPOT/throughput/memory at the target concurrency.
* Memory fits at the target concurrency (admitted, not KV-OOM shed); no spurious
  shedding; every numerics-changing lever already re-verified.
* Assets in `$ASSETS` (bundle dir: `.pkt` + `weights.json` + sidecars; `hsaco`
  dir for AMD). A built `plowrt` (`nix develop`; `cargo build -p plowrt --release
  --features cuda` or `--features hsa`).
* A GPU you can lease exclusively, and — for a comparator arm — a working
  reference framework (e.g. vLLM) **on the same box**.

If any is missing, **stop and report**. Do not run a campaign on a recipe that
has not gated out of Stage 6, and do not quote a comparator ratio you cannot
produce in the same session.

## Fix the campaign plan first — everything is measured against it

Before leasing a GPU, write down: **models + precisions, prompt lengths, output
length, the concurrency ladder (VUs), the context ladder, the SLOs (e.g. ITL/TPOT
p99 ≤ 50 ms, TTFT p99 ≤ 5 s), and the comparator (if any).** Note the engine
shape — it changes what the concurrency sweep means:

* **CUDA / sm_120 slotted** (`B` mux slots): a real capacity sweep up to `B`, then
  mux queueing.
* **AMD single-sequence per rank** (`batch=1`, optional TP): only the
  concurrency-1 rows compare kernels; higher VUs measure requests *queueing*, not
  a batched engine. Say so; do not read those ratio columns as a kernel contest.

## Procedure

### 0. Environment sanity (invalidates everything downstream)

* Confirm `backend ready — GPU accelerated` in the serve log — CPU fallback is a
  warning, not an error, and benches the CPU path.
* `rocm-smi --showuse` / `nvidia-smi --query-compute-apps` reads **0%** before any
  A/B. The persistent megakernel outlives its host process; a bench started while
  a prior `serve` tears down reads a contended box.
* Everything through `nix develop`. Every GPU run through
  `perf-data/harness/gpulease <label> <cmd>` — rc=76 = contended, discard and
  re-run.

### 1. Correctness battery — before ANY performance cell

**Token-identity gate**, for each arm you will bench, that same build:

```bash
perf-data/harness/gpulease gate perf-data/harness/bringup_gate.sh $ASSETS <tag> 8080
diff $BRINGUP_OUT/gate-out/<reference>.txt $BRINGUP_OUT/gate-out/<tag>.txt
```

Token-identical (required for refactors/staging/reordering), coherent-but-shifted
(only for a *documented* numerics change, called out), or garbage / truncation /
wrong first token (**reject**). On TP, every rank must emit the identical stream.
`amd-bench`'s `last id` is **not** a correctness signal — gate on the serve path
with a real prompt.

**Accuracy number**, at least one:

```bash
perf-data/harness/gpulease gsm8k scripts/bench_gsm8k.sh   # 8-shot greedy, n=200
```

A throughput number without an accuracy number is not publishable — token
identity proves self-consistency only.

### 2. Roofline sanity (decide "slow" vs "box is the wall" by arithmetic)

```bash
perf-data/harness/gpulease ceil python3 perf-data/harness/bringup_ceiling.py
```

Edit `SHAPES` to your model's prefill GEMMs. Decode capacity is usually
HBM-bandwidth-bound — state the bytes moved and the achieved GB/s next to the ms,
not the ms alone.

### 3. Same-session baseline (both engines, interleaved)

One server at a time, same box, same client, medians over ≥5 rounds (9 for a
headline). `bringup_showdown.sh` is the template — edit the arm list:

```bash
SNAP=$HF_SNAPSHOT MODEL_ID=$SERVED_ID BUNDLES=$BUNDLE_DIR CUBINS=$CUBIN_DIR \
IN_LENS="1024 4096" NPROMPT=9 \
  perf-data/harness/gpulease showdown perf-data/harness/bringup_showdown.sh
```

Cross-engine checks, mandatory: identical `Total input tokens`; identical `Total
generated tokens` (plowrt must honor `ignore_eos`); **raise `--slo-ms`** for
throughput arms or 429'd requests bench as successful → fake tok/s; medians (not
means) to absorb the one-time warmup.

### 4. Concurrency sweep (capacity)

Fixed VUs, warm+measure windows, greedy, fixed output tokens, TTFT including
queueing. Sweep the ladder — never one point:

```bash
CAMPAIGN=<name> PROMPT_TOKS=4096 VUS="1 2 4 8 16 32" \
MODEL_NAME=$SERVED_ID TOKENIZER=$HF_SNAPSHOT ASSETS=$BUNDLE_DIR \
  perf-data/harness/gpulease b2 perf-data/bench_b2_ib.sh
python3 perf-data/consolidate_b2_ib.py
python3 perf-data/harness/b2-ib/slo_capacity.py
```

If `bench_ib.sh`/`bench_b2_ib.sh` is not in the tree, drive the pinned
`inference-benchmarker` directly with the same profile, or fall back to
`bringup_bench.sh` at each VU. Per VU report tok/s, TTFT avg/p99, ITL(=TPOT)
avg/p99, tok/req. Derive **max users** = highest VU with **zero** failed requests
under each SLO, and the unconstrained peak separately. Flat decode ITL + exploding
TTFT = mux queueing above `B` (the capacity answer), not a decode regression.

### 5. Context sweep (context scaling)

At concurrency 1, sweep `IN_LENS="1024 4096 8192 16384 32768 65536"`; report TTFT
and TPOT at each. Characterize TPOT-vs-context (plow's decode is often close to
context-flat). Watch the chunk ladder: a template-inflated prompt that splits
`[4096,128]` pays a second full-model tail pass — credit it in a TTFT ratio. A
ctx blob that cannot hold `input+output` **refuses** the request (`tok/req`
collapses, `TPOT 0.000`) — that is a size error, flag the row `valid: false` and
recompile, do not re-run.

### 6. Consolidate + write the results doc

Transcribe every number verbatim from the tool's report JSON. Consolidators build
the `*.json`; **you write the markdown by hand** from those JSONs with the prose
and caveats. Commit under `perf-data/` (per-arch home `perf-data/plow-gfx942/`;
cross-arch capacity reports at the `perf-data/` root). Match the established
structure:

* header (date, branch@commit, box, both engine configs, harness+pinned rev,
  profile, TTFT convention);
* **honesty banner** — what ran and, explicitly, what did **not** (label, never
  project); in `plow-gfx942/`, add the `Scope:` class line;
* tables, one row per (tag, VU) / (tag, ctx), percentiles, `valid: false` on any
  ungated row;
* per-model "what holds it back" with the roofline arithmetic;
* "what's still open" (staged, one lease each);
* data/reproduction pointers.

## The gate — bringup complete

Passes when **all** hold; otherwise the model is blocked with a specific blocker:

1. Correctness battery passes: every benched arm token-identity gated that build
   (or its shift documented), ≥1 accuracy number recorded, TP rank-identity holds.
2. Performance targets met and **measured** (not projected) at target concurrency
   and context; capacity (max users/SLO) + peak throughput + context scaling all
   stated.
3. Same-session and uncontended: both engines one lease, every run `gpulease`
   rc=0, no stored baseline quoted as same-session.
4. Swept: a concurrency ladder and a context ladder, not one cell.
5. Documented in `perf-data/` in the established format, every cell traceable to
   a JSON + a command, neutral tone, no win/loss framing.

## Pitfalls to actively guard against

* **Contention voids the run.** rc=76 = discard. Lease is advisory (flock) — audit
  before and after; the persistent megakernel outlives its host, so confirm 0%
  GPU first.
* **A stored baseline is not same-session.** A recorded comparator CSV moved 33%
  on re-measure. Re-baseline both engines in one session or label the ratio.
* **A/B order / session drift** — when background load changes, re-baseline
  everything; do not compare arms measured under different contention.
* **Shed requests bench as successful** (~12 tok) → fake tok/s. Raise `--slo-ms`
  for throughput; re-check `Total generated tokens`.
* **`ignore_eos` mismatch** distorts throughput 3×+ — compare `Total generated
  tokens` every arm.
* **A wide fixed `B` doubles latency for the same throughput** — capacity is
  bandwidth/prefill-bound, not batch-size-bound.
* **Single-sequence engine**: only concurrency-1 compares kernels; higher VUs are
  queueing, not batching.
* **Chat-route token inflation** (~11% extra prefill) — credit it in a TTFT ratio.
* **ctx/profile mismatch is a refusal, not a slow run** — flag `valid: false`,
  recompile.
* **A gate that never re-emits cannot catch an emitter regression** — emit from
  the current checkout for the gate, not a stored asset.
* **Never launch a battery twice; kill with SIGTERM.** Two servers on one port →
  one answers both arms (delta meaningless); `kill -9` leaves the megakernel
  resident and corrupts later runs.
* **Standalone probes overstate** (probe law) — quote only in-serve numbers.

## When to stop and ask

* The correctness gate fails (garbage, wrong first token, TP ranks disagree, or
  accuracy far below the reference) → a correctness blocker; stop, do not report
  a perf number behind it.
* The comparator cannot be produced on this box (framework absent / version
  drift / no docker) → do not difference against a stored CSV as if same-session;
  report the ratio as stored-baseline with the caveat, and flag re-baselining as
  the missing measurement.
* The recipe cannot meet the performance target at the planned concurrency/context
  and the fix is structural (a different `B`, TP, faster prefill, a different
  card) → that is a Stage 5/6 decision or a hardware call; report the limiter with
  its roofline arithmetic, do not hand-tune inside the campaign.
* The campaign is cut short (operator stop, lease expiry) → write up what ran,
  label the un-run rows in the honesty banner, and stage the rest as open work.
  Never project a row you did not measure.

## Report back

* **Campaign scope**: models, precisions, box (GPU/arch/CU-or-SM/driver), engine
  configs, harness + pinned rev, profile, and — explicitly — what ran vs what did
  not.
* **Correctness**: token-identity gate result per arm, the accuracy number(s), TP
  rank-identity where applicable.
* **Capacity**: max users under each SLO and peak throughput per model, with the
  concurrency curve's shape (where TTFT vs ITL breaks).
* **Context scaling**: TPOT/TTFT vs context, and any crossover (stated as a
  trend, not a served point).
* **Roofline**: measured GEMM ceiling and the achieved fraction; the honest
  per-model limiter.
* **The write-up**: path to the committed `perf-data/` doc + its JSONs, and the
  reproduction command per table.
* **Real-vs-ideal caveats**: contention, stored-vs-same-session baselines,
  chat-route inflation, any comparator that could not be reproduced — everything
  that bounds how far a reader should trust a number.
