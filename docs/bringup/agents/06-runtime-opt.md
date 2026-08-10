# Agent — Stage 6: Runtime Optimization

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
  for AMD).
* You know the target arch/GPU and have a GPU to run on. Use `nix develop` for
  all build/cargo commands: `cargo build -p plowrt --release --features cuda`
  (sm_120) or `--features hsa` (gfx950).

If any is missing, **stop and report** — do not optimize a runtime around a block
that has not gated out of Stage 5.

## Fix the target first — everything is measured against it

Before touching a flag, write down: **model + family, prompt length, target
concurrency, TTFT budget, TPOT budget.** A "faster" number that ignores the TPOT
budget at the target concurrency is not progress. Also note which engine you are
on — the lever set differs:

* **CUDA / sm_120** (`ServeEngine::Cuda`, slotted): B sequence slots, chunked
  prefill, VMM prefix sharing, device sampling, multi-model residency.
* **AMD / gfx950** (`ServeEngine::Amd`, single-sequence per rank, optional TP):
  one model per process, no paging, no residency; ragged-tail chunking + same-
  slot prefix cache for recurrent models are the AMD-side levers.

## Procedure

### 1. Sanity — does the schedule run without a GPU?

```bash
plowrt simulate --assets $ASSETS --all-buckets --chrome sim.json
```

No deadlocks, all dependencies honored, finite makespan. `--bucket decode:1:128`
isolates a bucket; `--math golden` runs reference numerics. A schedule that will
not sim will not serve — fix it here.

### 2. Single-stream latency floor (device, one sequence)

```bash
plowrt amd-bench --blob $ASSETS/model.pkt --rt-hsaco $ASSETS/hsaco \
    --checkpoint $CKPT --prompt 1,2,3,4 --steps 64 --ctx 1024
```

Records the TPOT floor (per-token decode ms) and, with a multi-token `--prompt`,
the TTFT floor (prefill ms / tok/s). **Without `--checkpoint` the ids are noise —
timing only.** On CUDA, measure the single-stream floor through `serve` at
concurrency 1 instead. For TP prefill scaling: `--tp N --prefill-sweep
512,1024,2048,4096,8192 --prefill-reps 3`.

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

1. **Multi-step + device sampling** (sm_120): `PLOW_MULTISTEP=8`,
   `PLOW_DEV_SAMPLE=1`. Largest single decode win (~1.74×, free).
2. **Batch width `B`** from the **TPOT budget**, not peak throughput. If a
   different `B` is needed, recompile in Stage 5 (`PLOW_DECODE_BATCH`).
3. **Prefill chunking / interleave** only if the TTFT-under-load tail is the
   problem (`--pf-interleave`, `--pf-chunk`). Measure — finer often loses
   throughput. AMD ragged-tail (`PLOW_RAGGED_CHUNK`) is on by default.
4. **Prefix cache** if traffic shares prefixes: `--vmm-prefix` (sm_120,
   `--vmm-block-mib 64` at long ctx) or `--prefix-cache` (AMD recurrent).
   **Always report the hit rate** — a low-hit cache is a net loss.
5. **TP** if one GPU cannot hold the model / hit the budget. Gate on rank
   token-identity: `plowrt devices --tp N`, `plowrt amd-bench --tp N` (every rank
   must emit the identical stream).
6. **Load / cold start** (`--rt-prefetch-threads`, weight-slab knobs) only if
   time-to-serving matters. `PLOW_WEIGHT_VMM` helps CUDA, hurts AMD — leave AMD's
   default off.

### 6. Diagnose off numbers

`PLOW_TTFT_LOG=1` (conc. 1), `PLOW_DSTEP_LOG=1`, `PLOW_PFX_LOG=1`,
`PLOW_PF_PACKLOG=1`; `--trace` + `GET /trace` or `PLOW_TRACE_RAW=path` (AMD);
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
6. A recorded recipe: `plowrt serve` command + `PLOW_*` knobs + measured numbers.

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
* Multi-model (CUDA): duplicate network names silently drop a bundle; install-by-
  slug vs lookup-by-network can fall to the CPU reference path.

## When to stop and ask

* The full-model emit aborts, or `simulate` deadlocks / never terminates → a
  Stage 1–5 defect, not a runtime one.
* KV cannot be made to fit the target concurrency at any sane chunk size on the
  available VRAM → a memory-budget decision (smaller `B`, TP, or a different
  card), not a knob.
* The target needs a mode that is **not wired** (DP/PP GPU parallelism, prefill/
  decode disaggregation, multi-tenancy, a watchdog) → report the gap; do not
  hand-roll it.
* A numerics-changing lever helps latency but you cannot get a correctness gate
  to confirm it (no facts probe / oracle for the family) → surface it and ask.
* TP ranks disagree on tokens → a collective is not running; this is a
  correctness blocker, stop.

## Report back

* **Engine + geometry**: CUDA slotted vs AMD single-sequence/TP; compiled `B`,
  TP degree, prompt length, target concurrency.
* **Baseline vs tuned**: TTFT, TPOT, throughput at the target concurrency, and
  which levers moved which number (with the measured delta).
* **Memory budget**: weights / KV / scratch, and the concurrency that fits.
* **Correctness**: which numerics-changing levers were enabled and how each was
  re-verified (facts probe, oracle, TP token-identity). Prefix-cache hit rate if
  used.
* **Gate decision**: ready for Stage 7, or blocked (with the specific blocker).
* **The recipe**: the exact `plowrt serve` command line + `PLOW_*` knobs.
* **Real-vs-ideal caveats** affecting trust: contention, benched-on-shed
  throughput, any lever left off because it was unwired or unverified.
