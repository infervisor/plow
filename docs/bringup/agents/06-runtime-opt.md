# Agent — Stage 6: Runtime Optimization

## Target parameters — fill this in FIRST

This stage is target-specific, and on this stage `$VENDOR` decides **which
levers exist at all**. Read [`../target.md`](../target.md) and fill every row
before touching a flag. A row you cannot fill is a **blocker**, not a default.
Every command below is written in these names — **never substitute a literal
part name into a command.**

| param | value | source |
|---|---|---|
| `$VENDOR` | | `amd` or `nvidia` — selects the engine and the lever set |
| `$ISA` | | the `--arch` string (`IsaLevel::arch_flag()`) |
| `$GPU` | | a name from `plowc --list-gpus` — do not guess one |
| `$NCU` | | `--n-cu`; leave `0` to take `$GPU`'s `sm_count` |
| `$NGPU` / `$PARALLEL` | | `--num-gpus` / `--parallel`; **must be `tp`** — the devblob path errors on dp/pp/ep |
| `$MAXCTX` | | `--max-ctx`; must hold `input + output` of every request you serve |
| `$TOOLCHAIN` | | `hipcc` or `nvcc`, + version |
| `$BUILD` | | `scripts/build_<isa>.sh` for `$ISA` |
| `$FEATURES` | | `--features hsa` (amd) or `--features cuda` (nvidia) |
| `$BW_BOUND` | | carried from Stage 4 — decode is bandwidth-bound against it |
| `$RESULTS` | | `perf-data/plow-<isa>/` if it exists, else `perf-data/` |

You are executing **Stage 6** of the model-bringup playbook. Your job: take a
model whose single block is correct and tuned (Stage 5) and make it **serve the
whole model efficiently** on `plowrt` — profile the runtime, apply the levers,
measure, and decide whether it is ready for Stage 7 (the end-to-end campaign).
Read `docs/bringup/06-runtime-opt.md` first — it has the full lever set,
commands, and pitfalls. This prompt is the executable checklist.

Nothing here recompiles the model. Every lever is a `plowrt` flag or a `PLOW_*`
env var read at serve time — except the decode batch width `B`, which is
compiled into the blob and may send you back to Stage 5.

## Preconditions (from Stage 5)

* One block is numerically correct within tolerance **and** at its per-block
  latency target.
* `plowc --emit devblob` produces a full-model `model.pkt` that loads, with the
  Stage-5 assets in `$ASSETS` (`.pkt` + `weights.json` + sidecars; `hsaco` dir
  on `$VENDOR = amd`) — emitted for `$GPU`/`$ISA`, not another part.
* The target block above is filled in and you have `$GPU` to run on. Use
  `nix develop` for all build/cargo commands:
  `cargo build -p plowrt --release $FEATURES`.

If any is missing, **stop and report** — do not optimize a runtime around a block
that has not gated out of Stage 5.

## Fix the target first — everything is measured against it

Before touching a flag, write down: **model + family, prompt length, target
concurrency, TTFT budget, TPOT budget** — all of them for `$GPU`. A "faster"
number that ignores the TPOT budget at the target concurrency is not progress,
and a budget inherited from another part is not a budget. `$VENDOR` selects the
engine, and with it the lever set:

* **`$VENDOR = nvidia`** (`ServeEngine::Cuda`, slotted): B sequence slots,
  chunked prefill, VMM prefix sharing, device sampling, multi-model residency.
* **`$VENDOR = amd`** (`ServeEngine::Amd`, fixed-width slots with batch-ladder
  rungs, optional TP): one model per process, no paging or multi-model
  residency; ragged-tail chunking + same-slot prefix cache for recurrent models
  are the levers on this side.

The seam is per vendor, not per ISA — every `$ISA` of a vendor reaches the same
engine. A lever listed for the other vendor **does not exist** on your target;
it is not merely untuned.

## Procedure

### 1. Sanity — does the schedule run without a GPU?

```bash
plowrt simulate --assets $ASSETS --all-buckets --chrome sim.json
```

No deadlocks, all dependencies honored, finite makespan. `--bucket decode:1:128`
isolates a bucket; `--math golden` runs reference numerics. A schedule that will
not sim will not serve — fix it here.

### 2. Production-engine latency and throughput

`plowrt bench` is the performance authority. It drives the same `ServeEngine`
and mux as `serve`, excludes model load and warmup from the timed interval, and
fails without JSON on partial output, shedding, TP rank disagreement, or CPU
fallback. Start at concurrency 1, then repeat at the target concurrency:

```bash
plowrt bench --assets $ASSETS --prompt-ids 1,2,3,4 \
    --concurrency 1 --requests 8 --warmup-requests 1 --output-len 64
```

The JSON records TTFT/TPOT/ITL/E2E distributions, throughput, scheduler rungs,
TP width, active runtime settings, packet/object checksums, and checkpoint
layout. Preserve each JSON result with the exact command and environment.

`amd-bench` drives `AmdEngine` directly and bypasses production scheduling. Use
`plowrt bench --trace-raw PATH` for decode packet traces. Keep `amd-bench` only
for tensor/logit snapshots, repeated-prefill sweeps, synthetic kernel floors,
and diagnosing a TP disagreement reported by `plowrt bench`. Never publish it
as served-model performance.

### 2a. AMD benchmark harness convergence

Track `amd-bench` removal as a staged migration, not a flag deletion:

- [x] Inventory and classify every consumer: performance, correctness,
  trace/dump, TP audit, prefill sweep, or synthetic unbound probe.
- [x] Add a vendor-neutral in-process `plowrt bench` backed by the production
  engines, with token-id direct mode and bound weights by default.
- [ ] Add endpoint mode; Stage 7 remains the endpoint acceptance authority.
- [x] Emit fail-closed structured results with artifact digests, target/TP,
  shape/concurrency, warmup/request counts, latency distribution, throughput, and
  active knobs.
- [x] Expose raw AMD decode packet traces through the production `AmdServe`
  path (`plowrt bench --trace-raw PATH`).
- [ ] Expose tensor/logit snapshot, TP-rank audit, bucket timing,
  repeated-prefill, and ragged-batch diagnostics without reaching through
  `AmdServe` into `AmdEngine`.
- [ ] Move unbound weights and unwritten-KV timing to an explicitly synthetic
  `amd-probe`; never report it as served-model performance.
- [ ] Add bench-vs-serve parity tests for tokens, buckets, chunk boundaries, slot
  lifecycle, batch-ladder rungs, and TP rank agreement.
- [x] Migrate the headline production A/B script and runtime-optimization docs.
- [ ] Migrate remaining consumers where production scheduling is the intent;
  keep Stage 7 endpoint-only, warn for one release, then remove `amd-bench` only
  when repository search finds no diagnostic consumers.

### 3. Bring up serving and baseline

```bash
plowrt serve --assets $ASSETS --port 8080 --executors 8 --max-hold-ms 8 --slo-ms 250
```

Drive at target concurrency + prompt length; scrape `/metrics`:

* `rate(plowrt_batch_size_sum[5m]) / rate(plowrt_batch_count_total[5m])` — mean
  batch actually run.
* `plowrt_admit_shed_total` vs `plowrt_rejected_total` — controller shed vs
  capacity.
* `plowrt_utilization` (ρ = λ/μ), `plowrt_arrival_rate` (λ).

Record baseline TTFT / TPOT / throughput. **If you see shedding, raise `--slo-ms`
and re-measure before concluding anything about the blob** (see pitfalls).

### 4. Fit memory to the target concurrency

If `kv arena OOM` sheds (typed OOM to the client): shrink the prefill chunk / KV
ring so the target concurrency fits. Confirm the budget with `PLOW_LOAD_PROFILE=1
plowrt serve ...`. Do **not** grow the slot table past the compiled `B` — mux
slot *i* is engine slot *i*.

### 5. Apply levers, one at a time, re-measuring after each

Highest value first; keep only what helps *your* target:

1. **Multi-step + device sampling**: NVIDIA supports `PLOW_MULTISTEP` up to 64
   with `PLOW_DEV_SAMPLE=1`. AMD TP supports a bounded K≤4 capture ring only
   when `PLOW_TP_AGREE_EVERY>1`, while retaining each token's counter audit.
   The AMD path is correctness-gated but has no resolved speedup yet; keep it
   only after a matched A/B on `$GPU`.
2. **Batch width `B`** from the **TPOT budget**, not peak throughput. Decode is
   bandwidth-bound against `$BW_BOUND`, so the crossover is a `$GPU` property —
   derive it, do not inherit it. If a different `B` is needed, recompile in
   Stage 5 (`PLOW_DECODE_BATCH`).
3. **Prefill chunking / interleave** only if the TTFT-under-load tail is the
   problem (`--pf-interleave`, `--pf-chunk`; both vendors). Measure —
   finer often loses throughput; and re-fit `--pf-chunk-cost` on `$GPU`, its
   default is another part's launch cost. AMD TP selects from its compiled
   ladder, re-splits a pending chunk when decode lowers the cap, and keeps
   recurrent requests isolated; `PLOW_PF_BATCH=1` is fair rotation, not
   co-packing. Its ragged-tail chunk (`PLOW_RAGGED_CHUNK`) is on by default.
   To implement AMD co-packing, extend the packet ABI with per-request row range,
   slot, KV base/length and recurrent reset metadata; form a token-budgeted ragged
   batch from compatible compiled rungs; execute with parked-row masks and per-slot
   state addressing. Partition or prove non-aliasing prefill/decode scratch before
   overlap. Gate single-request parity, adversarial ragged batches, block latency,
   then matched C1/C8/C32 serving before changing the statement above.
4. **Prefix cache** if traffic shares prefixes: `--vmm-prefix` (nvidia;
   `--vmm-block-mib` is the real knob — 64 MiB won at long ctx on the measured
   part) or `--prefix-cache` (amd, recurrent families).
   **Always report the hit rate** — a low-hit cache is a net loss.
5. **TP** if one `$GPU` cannot hold the model / hit the budget; `$PARALLEL` must
   be `tp`. Check peer visibility with `plowrt devices --tp $NGPU`, then gate
   performance and rank agreement with `plowrt bench` on the TP assets. The
   default agreement interval checks every token; use `amd-bench` only to
   diagnose a reported disagreement rank by rank.
6. **Load / cold start** (`--rt-prefetch-threads`, weight-slab knobs) only if
   time-to-serving matters. `PLOW_WEIGHT_VMM` helps nvidia, hurts amd — leave
   the amd default off.

### 6. Diagnose off numbers

`PLOW_TTFT_LOG=1` (conc. 1), `PLOW_DSTEP_LOG=1`, `PLOW_PFX_LOG=1`,
`PLOW_PF_PACKLOG=1`; `--trace` + `GET /trace` or `PLOW_TRACE_RAW=path` (amd);
`plowrt disasm $ASSETS --counters --kernargs --tensors` for static analysis.

## The gate before Stage 7

Gate passes when **all** hold; otherwise the model is blocked with a specific
blocker:

1. TTFT within budget at conc. 1 and not blown past at target concurrency.
2. TPOT within budget at target concurrency (and monotone up to it, not just at
   conc. 1).
3. Memory fits — target concurrency admitted, not KV-OOM shed;
   `PLOW_LOAD_PROFILE=1` shows weights + KV + scratch resident with headroom.
4. `plowrt_admit_shed_total` ~0 at target load; any 429s are genuine capacity.
5. Any numerics-changing lever (batched prefill, ragged tail, prefix sharing)
   re-verified against a correctness gate. On TP, rank token-identity holds.
6. A recorded recipe: `plowrt serve` command + `PLOW_*` knobs + measured
   numbers, **stamped with `$GPU`, `$ISA`, `$NGPU`/`$PARALLEL`, `$MAXCTX`**. An
   unstamped recipe gets read as a target on the next part.
7. Every lever kept was measured on `$GPU`; nothing was carried over from
   another part's campaign as "already tuned".
8. `build.json` exists and reports `lean.verified=true`; every demanded tuned
   shape reports selected measurement provenance. A stale or wrong-SKU store is
   a blocked tuning campaign, not an analytical result that can be called tuned.

## Pitfalls to actively guard against

* **The SLO shed looks like an emitter bug.** A flat `--slo-ms` 429s every live
  slot on a wide blob while the kernel is correct. Raise `--slo-ms`, use the
  `ceil(live/batch)` floor, and check `admit_shed` vs `rejected` before blaming
  the blob.
* **Shed requests bench as "successful"** with ~12 tokens → fake tok/s. Raise
  `--slo-ms` for throughput runs.
* **A wide `B` doubles per-token latency** (decode is bandwidth-bound); a fixed
  large `B` overpays at partial load. Pick `B` from TPOT.
* **Serve width must match compiled width** — a mismatched `PLOW_DECODE_BATCH` /
  `-DGV_MM_MAX` runs *slower* through a predicated remainder arm.
* **Cross-request batched prefill is not numerics-neutral** and is often a no-op
  or force-off (fp8-KV). **Finer chunking usually loses throughput** — it is a
  tail tool.
* **Prefix cache: report the hit rate.** The VMM `BlockHash` collision check is
  not implemented — treat sharing as measured-good, not hardened.
* **Do not reach for unwired features:** no `Parallel { Dp, Pp, Ep }` selector
  (only TP is wired; EP is a MoE remap, disagg is a skeleton), no S2
  multi-tenancy, no host watchdog, `Streamer::execute_reclaim` is a no-op.
* **`gpulease` rc=76** is a false positive on a shared box; discard and re-run.
* Multi-model (`$VENDOR = nvidia`): duplicate network names silently drop a
  bundle; install-by-slug vs lookup-by-network can fall to the CPU reference
  path.
* **An asset built for another part still serves.** `--arch` is packet metadata
  and its default disagrees with `--gpu`'s default by design; `PLOW_UNISEG` on
  amd is warn-and-ignore, not a refusal. Check `build.json` names `$GPU`/`$ISA`
  before trusting a serving number.

## When to stop and ask

* Any row of the target block cannot be filled — in particular `$PARALLEL` is
  not `tp`, or `$GPU` is not in `plowc --list-gpus`.
* The full-model emit aborts, or `simulate` deadlocks / never terminates → a
  Stage 1–5 defect, not a runtime one.
* KV cannot be made to fit the target concurrency at any sane chunk size on
  `$GPU`'s VRAM → a memory-budget decision (smaller `B`, TP, or a different
  part), not a knob.
* The target needs a mode that is **not wired** (DP/PP GPU parallelism, prefill/
  decode disaggregation, multi-tenancy, a watchdog) → report the gap; do not
  hand-roll it.
* A lever your plan depends on exists only on the other `$VENDOR` — report it
  as a target-capability gap, do not substitute an unrelated knob.
* A numerics-changing lever helps latency but you cannot get a correctness gate
  to confirm it (no facts probe / oracle for the family) → surface it and ask.
* TP ranks disagree on tokens → a collective is not running; this is a
  correctness blocker, stop.

## Report back

* **The filled target block** — every row.
* **Engine + geometry**: nvidia slotted vs amd single-sequence/TP; compiled `B`,
  `$NGPU`, prompt length, target concurrency.
* **Baseline vs tuned**: TTFT, TPOT, throughput at the target concurrency, and
  which levers moved which number (with the measured delta).
* **Memory budget**: weights / KV / scratch, and the concurrency that fits.
* **Correctness**: which numerics-changing levers were enabled and how each was
  re-verified (facts probe, oracle, TP token-identity). Prefix-cache hit rate if
  used.
* **Gate decision**: ready for Stage 7, or blocked (with the specific blocker).
* **The recipe**: the exact `plowrt serve` command line + `PLOW_*` knobs,
  stamped with `$GPU` / `$ISA` / `$NGPU` / `$MAXCTX`.
* **Real-vs-ideal caveats** affecting trust: contention, benched-on-shed
  throughput, any lever left off because it was unwired or unverified.
