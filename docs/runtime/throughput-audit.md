# Throughput audit: where plowrt loses output tok/s to vLLM while leading on TTFT

Gemma-4 E4B on one H100 SXM, ISL 1000 / OSL 128, `vllm bench serve` (same client, completions,
`ignore_eos`), vLLM 0.28 (`--max-num-seqs 256`, `--max-model-len 8192`, defaults otherwise:
chunked prefill with an 8192-token budget, async scheduling, FULL_AND_PIECEWISE CUDA graphs,
FlashAttention on sm90, prefix caching on). The second model is Veena (Llama-3.2-3B, hd128) served as
a plain LLM. plow runs HEAD 5eea79c4 on the E4B packet with the KV-shared-tail skip, with
`--multistep-adaptive`.

## Verdict

* **The published gap was half measurement.** Every cell of the old grid reused the previous
  cell's prompts. vLLM served ~50% of c64/c128 prompts from its prefix cache; plow served ~0%.
  With unique prompts per cell, greedy output tok/s is c64 3438 vs 4187 (1.22x) and c128 3973 vs
  5103 (1.28x), not 3525 vs 5302 and 4381 vs 6786.
* **Throughput is set by device time per request, and plow spends more of it in decode.** The
  scheduler itself costs ≤ 2.5% (host gap 0.4%, bucket padding 1.0-1.6%). Admission's VMM mapping
  added 2.7% (runtime; fixed by the stale reserve, §6). The rest is kernel time:
  * the batched decode step's per-row marginal cost: 94 µs/row vs 27 at B 1→64, ctx 1024;
  * the rider tail, which runs on the same kernels;
  * plow's prefill is 21% faster than vLLM's and pays back part of it.
* **TTFT vs throughput is a redistribution, not a leak.** Prefill-first packing plus riding decode
  rows gives c64 TTFT p50 101 vs 620 ms and TPOT 17.0 vs 10.4 ms. E2E = TTFT + 127·TPOT is
  2261 vs 1936 ms. That ratio (1.17) equals the device-time-per-request ratio; the tick policy
  only moves time between TTFT and TPOT. plow also keeps a larger decode batch:
  * c128: ~105 decoding rows per step vs vLLM's 82 running + 29 waiting.
* **The same pattern holds on Veena.** B=1 decode is at parity (3.44 vs 3.42 ms). B=64 is 7.44 vs
  4.83 ms and B=128 10.03 vs 6.06. c128 output is 11618 vs 19972 tok/s, and TTFT still leads
  (c64 25.6 vs 142 ms).

## 1. Measurement correction (fixed: `audit_2.patch`)

`pb_bench` passed `--seed 8193` to every cell. With `--random-range-ratio 0`, numpy's generator
draws the same offsets, so cell N's first prompts are cell N−1's prompts. This was checked:
`offsets[:96]` of the 192-prompt draw equals the 96-prompt draw.

vLLM retains all of them: 956k tokens of KV, and its log shows a cumulative prefix hit rate of
41% after c32, 47% at c64 and 58% at c128. plow's 4 GiB VMM cache hits almost nothing: c64
prefilled 193,000 rows for 193 prompts, and c128 averaged 990 rows per prompt.

The fix makes the default seed a function of `(conc, isl, osl)`. `PB_SEED` still pins it, and the
GLM scripts and campaign.py already set it.

| cell (greedy) | vLLM old grid | vLLM unique prompts | plow unique prompts |
|---|---|---|---|
| c64 tok/s | 5302 | **4187** | 3438 |
| c128 tok/s | 6786 | **5103** | 3973 |
| c64 TTFT p50 ms | 254 | 620 | 101 |
| c128 TTFT p50 ms | 381 | 1150 | 127 |

Sampled rows (T=1, top_k 64, top_p 0.95), plow vs vLLM:

* c64: 3350 vs 4135 tok/s
* c128: 3943 vs 5060 tok/s

## 2. Accounting method

* **plow.** `PLOW_PF_PACKLOG=1` with the TICK line extended by `steps= tokens= live= prefilling=`,
  and an `ADMIT` line per KV admission. `scripts/bench/packlog_audit.py` splits wall time per cell
  into four buckets:
  * mixed ticks (prefill launch plus riders);
  * prefill-only ticks;
  * decode ticks (steps, rows per step);
  * host gap between ticks.

  It also measures launch padding (`(bucket − rows)/bucket` of each launch) and prices riders
  against the pure-launch cost.
* **vLLM.** `/metrics` polled every 50 ms: `iteration_tokens_total` gives the step count and tokens
  per step, plus the running and waiting gauges.
* **Both.** One nsys trace each at c64, with CUDA graph launches counted as busy.
* **Kernel split.** `step_bench` at fixed context, and instruction-cap sweeps (`--sweep`; the bench
  rewind now works with VMM KV).

## 3. Step-for-step at equal concurrency (E4B, greedy)

| | plow c64 | vLLM c64 | plow c128 | vLLM c128 |
|---|---|---|---|---|
| wall for 3×c requests | 7.13 s | 5.87 s | 12.34 s | 9.61 s |
| device busy (nsys) | 94.9% | 95.7% | — | — |
| decode rows per step | 60.5 (decode ticks), 50.6 riders | 58.1 | 105.6, 104.4 riders | 107.1 |
| ms per step | 11.94 decode / 31.9 mixed | 13.87 (every step mixed) | 16.05 / 39.3 | 20.94 |
| prefill tokens per step | 1883 per mixed launch | 454 | 1872 | 830 |
| running (+ waiting) | 61.3 live, 5.4 prefilling | 47.6 (+8.0) | 116.4 live, 8.8 prefilling | 82.3 (+28.7) |

The nsys trace shows vLLM's c64 structure: 360 full-graph decode steps averaging 6.73 ms, plus
about 60 mixed steps that carry close to 8192 prompt tokens each. It effectively prefills a wave
together, which puts time into TTFT. plow prefills ~2 prompts per 2048-row launch and lets every
live row ride.

## 4. Waterfall: vLLM → plow, device ms per request

| item | c64 | c128 | kind | owner |
|---|---|---|---|---|
| vLLM (wall / requests) | 30.57 | 25.03 | | |
| prefill rows: 12.2 µs (plow, full 2048-row launch) vs 15.4 µs (vLLM) | **−3.19** | **−3.19** | kernel (plow ahead) | — |
| decode-tick steps: 0.197 vs 0.119 ms/token (c64), 0.152 vs 0.076 (c128) | **+7.88** | **+5.58** | kernel: batched decode marginal | decode + attention agents |
| riders (tail window + terminal) vs vLLM's per-token decode cost | **+1.51** | **+4.44** | kernel (same GEMV/attention), plus host admission inside the tick | decode agent; VMM owner |
| — of which VMM KV mapping at admission (GPU idle) | 1.0 | 1.0 | runtime | fixed: `PLOW_VMM_STALE_RESERVE` (§6) |
| partial or padded prefill-only launches | +0.29 | +0.20 | scheduler (≤1%) | — |
| host gap between ticks | +0.13 | +0.13 | scheduler (0.4%) | — |
| **plow (wall / requests)** | **37.14** | **32.14** | | |

The rider rows show that plow's shared-tail design costs the same as its decode. A rider costs
136 µs: the tail window, which is 18 of 42 layers plus lm_head, at 6.0-6.4 ms device per mixed
tick for ~51 rows. A plow decode step costs 197 µs per row, and vLLM's decode costs 119 µs per
token. The riding choice (`sched::ride`) is right at both concurrencies:

* c64: riding 51 rows costs 6.9 ms vs a ~10.2 ms standalone step.
* c128: riding 104 rows costs 14.3 ms vs 16.0 ms.

## 5. The decode step at matched context (the kernel leak)

plow uses `step_bench` (kernel-only; served decode ticks land within 1% of it). vLLM uses decode-only
serving phases timed through `/metrics` (one wave, ISL = ctx − 128, OSL 256). All values are ms per
step.

| E4B | B=1 | B=64 | B=128 |
|---|---|---|---|
| ctx 512 plow / vLLM | 5.86 / 5.70 | 11.49 / 7.03 | 15.76 / 8.81 |
| ctx 1024 | 5.88 / 5.70 | 11.82 / 7.39 | 16.41 / 9.43 |
| ctx 2048 | 5.88 / 5.73 | 12.48 / 7.93 | 17.74 / 10.50 |

* Marginal cost per row at ctx 1024:
  * B 1→64: plow 94 µs, vLLM 27 µs.
  * B 64→128: plow 71 µs, vLLM 32 µs.
* The ctx-dependent part is at parity: going from ctx 512 to 2048 adds 0.99 vs 0.90 ms at B=64
  and 1.98 vs 1.69 ms at B=128.

Per-class split at ctx 1024. plow comes from instruction-cap sweeps; vLLM from an nsys node trace
of its full-graph decode step (615 kernels, 6.78 ms span under nsys):

| ms/step | plow B=1 | plow B=64 | plow B=128 | vLLM B=64 |
|---|---|---|---|---|
| projections (GEMV / GEMM) | 3.04 | **6.88** | **8.27** | 3.66 incl. lm_head |
| lm_head | 0.43 | 0.52 | 0.72 | (in GEMM) |
| attention (FlashDecode + merge) | 0.50 | 2.02 | 4.28 | 1.90 |
| interpreter skeleton (cap 0) | 1.21 | 1.56 | 1.77 | — (graph replay) |
| other (norms, PLE glu, rope, argmax) | 0.69 | 0.83 | 1.33 | 0.92 |
| total | 5.86 | 11.82 | 16.37 | 6.78 |

At B=64 about 80% of plow's excess is the projections: 7.40 ms of GEMV against 3.66 ms of cuBLAS
GEMM. Attention accounts for +0.1 ms and skeleton plus other for +1.2 ms. The largest B=64 ops:

| op | B=64 ms/step | B=1 ms/step |
|---|---|---|
| GemvGlu | 3.24 | 1.55 |
| down_proj | 1.91 | 0.88 |
| FlashDecode | 1.85 | 0.37 |
| o_proj | 0.54 | 0.16 |
| q / qkv | 0.51 | — |
| PLE input gate (`[2560→256]`) | 0.41; 1.23 at B=128 | 0.12 |

The PLE input gate grows 10x from B=1 to B=128.

Veena at ctx 512 / 1024 / 1792, plow vs vLLM:

| Veena | B=1 | B=64 | B=128 |
|---|---|---|---|
| ctx 512 | 3.47 / 3.41 | 7.76 / 4.93 | 10.61 / 6.38 |
| ctx 1024 | 3.52 / 3.42 | 9.04 / 6.10 | 13.19 / 8.77 |
| ctx 1792 | 3.62 / 3.47 | 11.00 / 7.80 | 17.07 / 11.86 |

plow's B=64 split at ctx 1024:

* GEMV 4.25 + lm_head 0.39 (B=1: 2.44 + 0.32)
* FlashDecode + merge 3.49, against a KV floor of ~2.24 ms (117 MB per row)
* skeleton 0.53, other 0.36

That is roughly GEMV +2.3 ms and hd128 attention +0.9 ms over vLLM.

### Recommendations to kernel owners (numbers above)

* **Decode agent: batched projections at B ≥ 32.** Target a tensor-core GEMM cost: E4B
  projections ~3.2 ms at B=64 vs 6.9 today, 4.3 ms of it in GemvGlu and down_proj. The
  `[2560→256]` PLE gate should not grow 10x with B. One option: route B ≥ 32 through the
  cuBLASLt decode graph (`DecodeRung::library`) for the dense projections. That closes ~3.7 ms of
  the 4.4 ms step gap, and at c64 roughly +6.5 of the +9.4 ms per request in decode and riders.
* **Attention agent.** E4B hd256/512 flash decode is at vLLM's level: 2.02 vs 1.90 ms at B=64,
  with the same ctx slope. Veena hd128 is 3.49 ms vs a 2.24 ms floor, the target for that model.
* **Interpreter skeleton.** The skeleton costs 1.2 ms at B=1 and 1.77 ms at B=128, plus a
  ~10 ms prefill launch floor: 64 rows take 11.0 ms and 2048 rows 25.0 ms. The launch floor does
  not cost throughput at ISL 1000, where launches are full. It sets short-prompt and session TTFT
  (prefill-kernel agent).
* **Sampler.** Sampled rows add 0.2 ms per step at B=64: decode ticks run 12.14 vs 11.94 ms, and
  output drops 2.6% at c64. vLLM loses 1.2%.

## 6. Scheduler and runtime leaks checked

| candidate | measured | verdict |
|---|---|---|
| host gap between ticks (dispatcher, handoff) | 0.4% of wall at c64 and c128 | not a leak |
| GPU idle inside ticks (nsys) | 5.1% (vLLM 4.3%); ~2.8 ms at each mixed-tick boundary, none between decode steps | see the next row |
| KV admission VMM mapping (`admit_packed_slot` → `ensure_rows(total + max(block, 2048))`) | 1.0 ms per new prompt (p50 1018 µs, 192 per cell), serialized before the launch; 2.7% of c64 wall | **runtime leak, fixed** (below) |
| launch staging, emit and detok, retire loop | 30 µs, 136 µs (54 tokens), 78 µs per tick | not a leak |
| bucket padding (a 1061-row pack in a 2048 bucket) | 1.0-1.6% of wall | minor |
| decode rung padding (rung chosen by highest slot) | rows ≈ extent at c64; at c128 106 rows run rung 128 because the other slots are prefilling | inherent to fixed slots |
| multistep quantum delaying admission | adaptive: 1 step while prefill is pending, 8-step quanta only when no prompt waits | not a leak |
| prefill-first starving decode | every live row rides the launch unless `sched::ride` prices a separate step cheaper; TPOT is set by mixed-tick length | policy (TTFT ↔ TPOT) |
| ride vs step | per-launch EWMA picks the cheaper arm (§4) | correct |

### The admission leak and its fix (`PLOW_VMM_STALE_RESERVE`, default on)

What admission cost, per new E4B prompt at c64 (a probe on `ensure_rows`):

* It maps 16 fresh 2 MiB blocks: column 1 of 8 full-layer tracks × 2 KV heads. `block_rows` is
  2048 and the window is `1128 + 2048` rows.
* The cost splits into `cuMemSetAccess` ~56 µs per block (~0.9 ms), creates ~0.4 ms (the pool
  was usually dry) and maps ~0.07 ms.
* In-place reuse never happened: 0 of ~27 blocks per admission. The pool thread had already
  unmapped the previous occupant's private columns (`reclaim_stale`), so the next occupant of
  the same slot mapped them again.

Two changes (`memory/vmm.rs`):

* **Coalesced grants (shared, both backends).** `ensure_rows` maps each (track, head) window
  first, then grants access once per VA-contiguous run instead of once per block, as
  `try_attach` already did. A failed grant unmaps its run and leaves the frontier unchanged.
  E4B's admission grows one column, so this alone measures neutral: c64 3403 vs 3405 tok/s. It
  pays off for multi-column growth: long prompts, and AMD.
* **Stale reserve (CUDA only; `enable_stale_reserve`, AMD untouched).**
  * The reclaimer leaves a retired window's private blocks mapped while stale plus pooled
    blocks fit the `--kv-pool-mib` cap. Past the cap it unmaps as before.
  * The slot's next occupant reuses them in place, with no driver call.
  * Reserved blocks are never an OOM: before `create_block` evicts the prefix cache or reports
    OOM, it unmaps a kept block of any slot and reuses the handle (`steal_stale`).
  * Idle HBM stays bounded by the same cap the pool already had.

A/B on E4B H100, unique prompts per cell, greedy, cells g32 → g64 → g128 on a fresh server.
Arms are TTFT p50 ms / tok/s:

| cell | HEAD | HEAD run 2 | reserve | reserve run 2 | grants only |
|---|---|---|---|---|---|
| g32 | 100 / 2288 | 142 / 2158 | **56 / 2535** | **56 / 2528** | 61 / 2497 |
| g64 | 102 / 3405 | 143 / 3010 | **95 / 3482** | **96 / 3482** | 102 / 3403 |
| g128 | 169 / 3947 | 210 / 3558 | 210 / 3633 | 123 / 4050 | 129 / 3948 |

* **Admission `ensure_rows`** goes from p50 1154 µs to 1 µs. p90 stays ~1.3 ms from the first
  wave, before any slot retires.
* **c64 mixed tick** goes from 32.28 to 30.76 ms. Decode ticks are unchanged at 11.82 ms/step.
* **nsys c64 on a warmed server:**
  * GPU idle 5.0% → 3.0%.
  * Idle in 1-10 ms gaps 233 → 45 ms.
  * 3427 → 3503 tok/s (+2.2%).
* **Spreads.** The treatment's spread is under 0.3% at g32/g64; HEAD's run 2 was a slow arm.
  g128 is bimodal in both arms (TTFT 123-210 ms), so not convictable.
* **Sessions (64 calls, `session_bench.py`):**
  * later-turn TTFT p50 39.8 → 34.1 ms (p99 126 → 97);
  * TPOT 15.3 → 14.9 ms;
  * 1927 → 1964 tok/s;
  * cached fraction unchanged at 0.87.

`PLOW_VMM_STALE_RESERVE=0` is the rollback.

A first version counted kept blocks against the pool's cap in `unref_block` too. That released
freed blocks instead of parking them, so later admissions paid `cuMemCreate` (p90 4.8 ms), and
c64 lost 10.6%. The shipped rule gates only the reclaimer's decision.

## 7. What vLLM and SGLang do that plow does not

* **vLLM V1** (`v1/core/sched/scheduler.py`):
  * One forward per step over a flat token budget (`max_num_batched_tokens` 8192). Running
    requests go first at 1 token each, then waiting prompts FCFS, with the last one chunked to
    fill the budget.
  * Async scheduling overlaps the next step's scheduling with the current step on the device.
  * Pure-decode batches replay a FULL CUDA graph per batch size (35 sizes up to 512). Mixed batches
    use piecewise graphs, with attention outside them.
  * Projections are cuBLAS / nvjet tensor-core GEMMs at any M. At M=64 they cost 3.66 ms for all of
    E4B, near the weight-read floor.
  * FlashAttention sm90 for decode and prefill; KV is paged, so admitting a prompt is block-table
    bookkeeping with no driver calls.
  * Gumbel sampling at 42 µs per step, FlashInfer top-k/top-p.
  * Its wave-prefill behaviour costs it TTFT.
* **SGLang** (server args, v0.5): chunked prefill (`--chunked-prefill-size`, auto-sized from GPU memory),
  with prefill and decode batches alternating by default (`--enable-mixed-chunk` off), which is
  prefill-prioritised like plow. Other mechanisms:
  * the overlap scheduler (CPU scheduling of step n+1 during step n, on by default);
  * CUDA-graph decode up to `--cuda-graph-max-bs`;
  * FA3 / FlashInfer attention with split-KV;
  * a radix-tree prefix cache on paged KV;
  * `--max-prefill-tokens` 16384.
* **Against plow.** plow already has most of the scheduling mechanisms:
  * unified token batch (mixed steps);
  * device multistep and a lookahead pipeline instead of async scheduling;
  * a persistent interpreter instead of CUDA graphs: the same B=1 cost, 1.2-1.8 ms of skeleton;
  * a per-launch ride policy.

  The differences that cost throughput:
  * GEMV-walk projections instead of tensor-core GEMM at B ≥ 32 (§5);
  * VMM driver calls instead of paged block tables at admission (§6).

## 8. Reproduce

```sh
# unique prompts per cell (audit_2); PACKLOG accounting (audit_3)
PLOW_PF_PACKLOG=1 plowrt serve --assets $E4B ...
python3 scripts/bench/packlog_audit.py server.log          # per cell: ticks, steps, riders, padding, occupancy
step_bench $E4B 64 1024 10 --warmup 4 --sweep 0..604        # per-op decode cost (VMM KV now supported)
plowrt disasm $E4B --program 1                               # op names for the sweep
```

Raw data from this audit (H100 box): `/opt/dlami/nvme/lava-tts/audit/res/`:

* `r1-plow`, `r1-vllm`: grid and metrics
* `r2-*`: Veena
* `n-*`: nsys c64
* `r6`: matched-ctx steps, sweeps, vLLM decode trace
* `r5-host`, `r7-host`: tick host phases
