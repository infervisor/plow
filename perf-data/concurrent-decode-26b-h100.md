# Concurrent / batched DECODE for Gemma-4-26B-A4B on H100 NVL — plow vs vLLM at matched concurrency

GPU: **H100 NVL** (sm_90a, 132 SMs, HBM3 ~3350 GB/s spec), bf16, TP-1. Date 2026-07-24.
All GPU runs under `gpulease` (rc=0, uncontended; VRAM verified 0 MiB between runs).
Both engines correctness-gated → **Paris** (see §Correctness).

## The thesis under test (mission)

plow's decode megakernel is a **persistent cooperative kernel**: one grid reads each
weight **once** and reuses it across all B tokens in the batch. The README claims this
should let plow win **multi-user aggregate** decode even where it loses B=1. Prior data
only had B=4 on the un-optimized occ-1 megakernel (244 tok/s aggregate) vs vLLM's *peak*
1,850 — not an apples-to-apples point. This measures plow's batched decode **at multiple
B with the best config**, against vLLM at **exactly matched concurrency** (not vLLM's peak).

## VERDICT — plow loses at EVERY concurrency, and the gap WIDENS with B

The shared-weight-read thesis **does not hold** for this MoE model. plow is beaten on
**both** aggregate throughput **and** per-user latency-under-load at every batch size, and
the deficit grows monotonically from **1.8× (B=1) to 4.6× (B=32)**. There is **no batch
where plow wins**. Root cause: **the MoE expert union saturates the weight read** — the
one lever plow is built to exploit (weight reuse across the batch) is exhausted by B=16 on
a 128-expert / top-8 model, exactly where vLLM's tensor-core batching keeps scaling.

## The headline table (matched concurrency = matched engine batch)

plow: main sm90a cubin (occ-1 megakernel), `step_bench <assets> B 1024 128`, prompt
ctx=1024, 128 timed decode steps (16 warmup dropped). vLLM 0.25.1 bf16, `vllm bench serve`
`--max-concurrency B --max-num-seqs 32 --no-enable-prefix-caching`, random in=1024 out=128,
`--ignore-eos --temperature 0`. Aggregate = total output tok/s; per-user = 1000/TPOT.

| B / concurrency | plow agg tok/s | vLLM agg tok/s | **vLLM/plow agg** | plow per-user TPOT | vLLM per-user TPOT |
|---:|---:|---:|---:|---:|---:|
| 1  | 107.1 | 190.7  | **1.78×** | 9.33 ms  (107.1 tok/s) | 4.84 ms  (206.6 tok/s) |
| 4  | 243.8 | 447.5  | **1.84×** | 16.41 ms (60.9 tok/s)  | 7.94 ms  (125.9 tok/s) |
| 8  | 296.2 | 776.4  | **2.62×** | 27.01 ms (37.0 tok/s)  | 8.50 ms  (117.6 tok/s) |
| 16 | 310.1 | 1103.7 | **3.56×** | 51.60 ms (19.4 tok/s)  | 11.76 ms (85.0 tok/s)  |
| 32 | 305.8 | 1407.3 | **4.60×** | 104.66 ms (9.6 tok/s)  | 18.70 ms (53.5 tok/s)  |

- **Aggregate: plow loses everywhere, gap widens 1.78×→4.60×.** vLLM's aggregate scales
  almost linearly with concurrency (190→1407, ~7.4× over 32×). plow's **plateaus at
  ~310 tok/s (B=16) then declines** (B=32: 306).
- **Per-user latency-under-load: plow also loses everywhere**, and worse — 1.93× slower
  TPOT at B=1 widening to **5.60× at B=32** (104.7 ms vs 18.7 ms). vLLM holds per-user TPOT
  low under load (7.9→18.7 ms) via continuous batching; plow's whole batched step is slow,
  so per-user latency collapses as B grows.
- vLLM's single-user TPOT (4.84 ms) reproduces the campaign roofline (161 µs/layer ×30 =
  4.83 ms) — the measurement method is sound.

## Why plow's aggregate plateaus — the MoE expert-union physics (the key finding)

plow's aggregate scaling **per batch step**:

| step | plow agg | scaling |
|---|---:|---:|
| B=1→4 (4×) | 107→244 | **2.28×** (shared weight read working) |
| B=4→8 (2×) | 244→296 | 1.21× (diminishing) |
| B=8→16 (2×) | 296→310 | 1.05× (nearly flat) |
| B=16→32 (2×) | 310→306 | **0.99× (declining)** |

**The dense projections ARE shared across B** (q/k/v/o, the shared expert) — that is why
B=1→8 still scales somewhat. **But the MoE — the single biggest decode op (56 µs, ~30 % of
the layer) — is NOT.** With 128 experts / top-8, B tokens route to **B×8 expert slots**; the
**union of active experts grows with B**:

- B=1: 8 experts read. B=8: up to 64. **B=16: 16×8 = 128 draws → the union approaches ALL
  128 experts.** Past that, every decode step already reads the **entire** MoE weight, so
  there is **zero remaining weight-reuse headroom** in the MoE.
- Beyond B≈16 batching only **adds token rows** to a full-weight read → per-step time grows
  ~linearly (51.6→104.7 ms, 2.03× for 2× batch) and **aggregate flatlines then declines**.

So plow's persistent-kernel weight-sharing advantage is **real but bounded**: it applies to
the dense path (fully shared) and to the MoE only until the active-expert union saturates
the expert weights — which, on a 128-expert top-8 model, happens by **B=16**. That is
precisely the concurrency region where vLLM's grouped-GEMM MoE + tensor cores keep scaling,
so plow's window to win never opens.

## occupancy (occ2) helps batched — more than B=1 — but nowhere near enough

occ2 = the `PLOW_NV_FORCE_MINBLK`/occ2 decode cubin (2 blocks/SM, n_cu=264 packet). Measured
occ2-vs-main at matched short ctx=128 (occ2's decode grid 264 has no matching prefill cubin,
so prompt setup falls back to per-token decode — ctx kept small; decode-step timing is the
valid comparison and main@ctx128 reproduces main@ctx1024 within 1 %):

| B | main agg (ctx128) | occ2 agg | occ2 lift | occ2 vs vLLM agg |
|---:|---:|---:|---:|---:|
| 1  | 108.2 | 115.7 | +6.9 % | 0.61× |
| 4  | 248.1 | 258.5 | +4.2 % | 0.58× |
| 8  | 300.6 | 328.4 | +9.2 % | 0.42× |
| 16 | 313.6 | 352.4 | **+12.4 %** | 0.32× |
| 32 | 309.5 | ~340 (extrap; not measured¹) | ~+10 % | ~0.24× |

- occ2's lift **grows with B** (+7 %→+12 %): higher occupancy most helps the large MoE-union
  weight read, the batched bottleneck. Consistent with the segmented-decode finding that
  occupancy is the real lever for the memory-bound GEMV/MoE.
- **But it does not change the verdict.** Even occ2's best point (B=16, 352 tok/s) is still
  **3.1× behind** vLLM's matched 1104 tok/s. Occupancy narrows the constant factor; it does
  not touch the expert-union saturation that caps plow's aggregate.

¹ occ2 B=32 not measured: with no occ2 prefill cubin, 32-slot prompt setup via decode-only
consumption is ~10.8 s/slot × 32 ≈ 340 s — impractical. Extrapolated from the B=16 +12 % trend.

## Correctness gate (→ Paris)

- **plow** (b1 asset: emitted packet + main sm90a cubin + bf16 checkpoint), `plowrt serve`,
  greedy: *"What is the capital of France?"* → **"The capital of France is \*\*Paris\*\*."** ✓
- **vLLM** (same model dir), same prompt → **"The capital of France is \*\*Paris\*\*."** ✓

## Method notes / repro

- plow packets emitted at **max_ctx=2048** (covers the 1024-prompt + 128-decode window;
  max_ctx sizes only KV allocation, not per-step decode cost — verified: main agg at ctx=128
  vs ctx=1024 differ <1 %). All B fit in 94 GiB (B=32: 47 GiB weights + 13.75 GiB KV +
  0.57 GiB act ≈ 61 GiB). At the mission's max_ctx=8192 the KV is 1.72 GiB/B, so B=16 fits
  (74.5 GiB) but **B=32 does not** (102 GiB) — the ctx=8192 batch ceiling is B=16.
- Emit: `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33 PLOW_DECODE_BATCH=B gemma4 <model> 2048 <out>/model.pkt 132`
  (occ2: n_cu=264, cubin `/workspace/assets/cubin-sm90a-occ2`). Assets: `/workspace/assets/cc/{b,occ2-b}{1,4,8,16,32}`.
- plow bench: `PLOW_LIBCUDA=… LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat step_bench <assets> B 1024 128` → RAW_STEP.
- vLLM: venv `/workspace/venvs/vllm-blk` (torch 2.11+cu130; the cu128 venv's flash-attn is
  ABI-broken). Driver `perf-data/cc_vllm.sh` (needs venv-bin + cuda-bin on PATH for
  flashinfer's ninja JIT). Serve once `--max-num-seqs 32`, sweep `--max-concurrency ∈{1,4,8,16,32}`;
  in-flight = concurrency = engine batch, so one serve == per-B `--max-num-seqs`.
- Raw: `/dev/shm/cc-vllm/summary.csv`; scripts `perf-data/cc_vllm.sh`, `perf-data/cc_plow_rest.sh`, `perf-data/cc_paris.sh`.

## Bottom line

plow's persistent-kernel shared-weight-read is a genuine mechanism, but on **26B-A4B
(128-expert MoE)** it **cannot beat vLLM at any concurrency**. It loses B=1 (1.8×) and
loses *more* as you add users (4.6× at B=32) on aggregate, and its per-user latency-under-load
degrades ~2× faster than vLLM's. The advertised "wins multi-user aggregate" claim is
**refuted for this model**: the dense path is shared but the MoE expert union saturates the
weight read by B=16, exactly cancelling the batching headroom right where vLLM's tensor-core
grouped-MoE keeps scaling. occupancy (occ2) adds a growing but small margin (+12 % at B=16),
not the 3–4× needed. A concurrent-decode win would require a batched-MoE kernel that keeps
per-token cost sub-linear as the expert union fills (grouped/sorted expert GEMM with
tensor cores) — i.e. matching vLLM's MoE path, not just the occupancy fix.
