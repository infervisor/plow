# plow on MI300X (gfx942) — first baseline

The first plow inference numbers on CDNA3. Produced by `scripts/bench_plow_rocm.sh`,
which is a deliberate mirror of `scripts/bench_vllm_rocm.sh`: the same
`vllm bench serve` client, the same regex parse, the same phases and the same CSV
columns as `perf-data/vllm-rocm/`. Regenerate the comparison with
`scripts/plow_vs_vllm_rocm.py`.

Toolchain: ROCm 7.14 / clang-23, 8x MI300X (304 CU), objects from
`scripts/build_gfx942.sh`.

## CURRENT STATE (2026-08-04) — read this before the tables below

The sweep tables further down are the FIRST baseline and are now superseded for
Gemma-4-12B decode. Current, built from HEAD by `scripts/build_gfx942.sh` with
`PLOW_OCC4=1 PLOW_L2HIER=1` against an L2-placed fp8 blob:

    ctx 4096   12.077 ms/token      ctx 8192   12.144

    vs vLLM re-measured on this box (10.03 @4k, 11.08 @8k):  1.204x / 1.096x — BEHIND

    (measured --steps 48, 3 reps, stock defaults: per-XCD queues + counter double-buffer,
     NO env flags. Supersedes the 12.379/12.494 line, which predates both.)

That is down from 20.9 ms (bf16) and 15.32 ms (fp8+occ4). The full derivation, the
ablation that located the bottleneck, and every rejected transform are in
`gemv-mlp-and-tensile.md`. The single largest lever was NOT a kernel: per-packet L2
cache maintenance (`buffer_wbl2`/`buffer_inv` issued once per workgroup, serialising
per XCD) was 16% of the token, fixed by plow's own `PLOW_GATE_HIER`.

WHY plow is still behind, stated plainly: vLLM moves 24 GB of bf16 weights in
10.03 ms = ~2.4 TB/s, a bandwidth-bound decode at ~45% of peak. plow moves 12 GB of
fp8 in 12.38 ms = ~970 GB/s, 18% of peak. plow is already spending HALF the bytes
and is still slower, so the deficit is not memory — it is ~9 serial
latency-bound packet steps per layer x 48 layers, each with a floor no amount of
bandwidth removes.

ALSO: `amd-bench`'s `last id` is NOT a correctness signal — it never prefills, so
attention reads a KV cache that was never computed. Use the serve path
(`plowrt serve` + a real prompt) as the gate; it answers 'Paris' on the current
build. Several "token-identical" claims made earlier in this directory's history
rest on that meaningless signal and are corrected in `gemv-mlp-and-tensile.md`.

## Read these caveats before quoting a number

1. **SUPERSEDED (the cell now exists).** When these tables were recorded there was
   no `tuning/amd/gfx942/mi300x` cell, so `plowc` reported
   `3080 record(s) skipped as STALE ... tile selection fell back to the analytical
   model` on all four models — tiles in THOSE rows were chosen by the cost model.
   The cell was seeded later in the same campaign (96 records, prefill GEMM tiles),
   and `gemv-mlp-and-tensile.md` records the outcome: pulling the lever was a NULL
   for Gemma-4 decode, whose GEMVs run fixed rungs rather than the tuned ladder.

2. **Only the concurrency-1 rows compare kernels.** plow's AMD serve is
   `batch=1`, so the concurrency ladder measures requests QUEUEING while vLLM
   batches them continuously. plow's throughput stays flat (~45-58 tok/s) as vLLM
   climbs past 1400 tok/s. That gap is the absence of continuous batching, not a
   kernel deficit, and the ratio columns for those rows should not be read as one.

3. **plow is measured over `/v1/chat/completions`; the vLLM baseline used
   `/v1/completions`.** Not a choice: `plowrt` implements only the chat route.
   The template's ~14 extra tokens push a 1024-token prompt past the 1024 bucket,
   so the engine plans `chunks=[1024, 128]` and prefills 1152 tokens' worth of
   work for 1038 real ones — about **11% more prefill than the baseline paid**.
   Credit plow that 11% when reading a TTFT ratio. TPOT is unaffected.

4. **The 65536 ctxsweep point is not a measurement.** The blobs were compiled
   `--max-ctx 65536`, which cannot hold 65536 input + 128 output, so the requests
   were refused: `gen_toks` comes back at 81 instead of `3 x 128 = 384`, TPOT
   reads 0.000 and the harness flags the row. Fix is a recompile at
   `--max-ctx 73728`, not a re-run. The row is left in the CSV rather than
   deleted so the failure is visible.

5. **`gemma-4-26b-a4b` FAILED its coherence gate — do not quote its numbers.**
   See below.

## Status per model

| model | tp | compiles | loads | coherent | swept |
|---|--:|:--:|:--:|:--:|:--:|
| Gemma-4 12B | 1 | yes | yes | **yes** ("Paris") | yes |
| Gemma-4 31B | 1 | yes | yes | **yes** ("Paris") | yes |
| Gemma-4 26B-A4B | 1 | yes | yes | **NO** | numbers invalid |
| GLM-5.2-FP8 | 4 | yes | **no** | - | no |

### Gemma-4 26B-A4B is numerically wrong on gfx942

It answers `_Step-re_s_s_s_1_0_1-0-1- France-1` to "capital of France". The fault
is NOT in the Gemma-MoE kernels, which were checked directly on this hardware:

- `runtime/tests/moe_gemma_gfx950_test.c` against the CPU oracle: **25/25 pass**,
  every op 61-77 and 81/82, bf16 and fp8 arms, decode and prefill.
- `runtime/tests/moe_gemma_interp_gfx950_test.c`, the same ops driven through the
  real persistent interpreter: **13/13 pass**.

It is also not the flash-object degrade. The first 26B run logged
`no flash object — flash segments run on the 8-wave interpreter` (plowrt's
`check_moe_gemma_arms` is a blanket check over every object it loads, so a flash
object without the marker symbol is rejected at `info!` level). Rebuilding the
flash row with the arms made the degrade go away and left the output
**byte-identical** — so that path is benign exactly as its comment claims, and
the cause is elsewhere. Unresolved.

### GLM-5.2 does not fit at TP4

`--num-gpus 4` on the default path asserts `GLM TP sharding is milestone-3`. The
`GLM_FULL=1 PLOW_MLA_PREFILL=full` path has no such assert and emits a TP4 blob
(78 layers, 2756 decode ops, 6 prefill buckets x 2021), which brings up all four
ranks and runs 8 decode steps at 47.234 ms/token with UNBOUND weights. Binding the
real checkpoint takes a GPU memory access fault, and it is not a race — it
reproduces on rank 0 alone under `PLOW_TP_SERIAL_LOAD=1`.

The cause is size. `plowrt disasm --tensors` shows the blob's named weight table
carries **no routed-expert tensors at all** (14.29 GiB at TP4, 10.20 GiB at TP8,
all of it non-expert); the 256 experts per layer are bound as packed experts at
load time. GLM-5.2 is ~700B parameters — 256 experts x 3 x 6144 x 2048 per layer
x 75 MoE layers is ~725 GB of fp8 routed-expert weights — so TP4 puts ~181 GB of
experts on a 192 GB card before anything else. A TP8 blob is compiled and
untested (it needs all 8 cards free).

Worth noting separately: an allocation that does not fit should fail at alloc
time, not fault on the device.

## 1k → 64k context sweep (concurrency 1)

Re-run at `--max-ctx 73728`, which is what makes the 65536 point a real
measurement rather than the refusal recorded above (65536 input + 128 output does
not fit a 65536 blob; `gen_toks` came back 81 instead of 384). Both models pass
the coherence gate. 26B-A4B is included for completeness and its numbers remain
INVALID — it still fails the gate.

**The finding: plow's decode is far more context-scalable than vLLM's.** TPOT
ratio narrows monotonically, and it is not a small effect.

### Gemma-4 12B

| ctx | plow TTFT | vLLM TTFT | TTFT | plow TPOT | vLLM TPOT | TPOT |
|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 191 | 28 | 6.8x | 20.29 | 6.80 | 2.98x |
| 4096 | 622 | 121 | 5.1x | 20.39 | 7.57 | 2.69x |
| 8192 | 1256 | 197 | 6.4x | 20.49 | 8.62 | 2.38x |
| 16384 | 2762 | 471 | 5.9x | 20.68 | 9.21 | 2.25x |
| 32768 | 6690 | 1337 | 5.0x | 21.07 | 11.18 | 1.88x |
| 65536 | 18237 | 4279 | 4.3x | 21.81 | 12.70 | **1.72x** |

plow's TPOT rises **+7.5%** from 1k to 64k (20.29 -> 21.81). vLLM's rises
**+87%** (6.80 -> 12.70).

### Gemma-4 31B

| ctx | plow TTFT | vLLM TTFT | TTFT | plow TPOT | vLLM TPOT | TPOT |
|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 391 | 80 | 4.9x | 31.84 | 13.51 | 2.36x |
| 4096 | 1328 | 244 | 5.4x | 32.02 | 14.40 | 2.22x |
| 8192 | 2730 | 423 | 6.4x | 32.30 | 15.57 | 2.07x |
| 16384 | 6055 | 1045 | 5.8x | 32.84 | 16.42 | 2.00x |
| 32768 | 14790 | 3033 | 4.9x | 33.90 | 19.00 | 1.78x |
| 65536 | 40529 | 10200 | 4.0x | 35.94 | 20.79 | **1.73x** |

plow +12.9% from 1k to 64k; vLLM +53.9%.

Mechanism: plow's per-token cost is dominated by a FLAT weight-streaming term
(the GEMV family, 69% of the decode span and independent of context), while its
attention term barely grows — the flash-decode path and KV layout are doing their
job. vLLM starts far ahead on the flat term and gives it back to attention. The
crossover is far out (the slopes differ by ~2.4e-5 ms/token, so ~380k context at
this rate), but the direction is real and it is the one axis where plow is
already winning on trend.

## The vLLM baseline does not reproduce on this box — 10.03 ms, not 7.57

Re-measured vLLM directly rather than trusting `perf-data/vllm-rocm/*.csv`, on the
same machine, with the SAME vLLM version the recorded harness used
(`0.23.0+rocm714`; `scripts/bench_vllm_rocm.sh` pins
`rocm/vllm:rocm7.14.0_..._vllm_0.23.0`). Warm (a discarded warm-up point first,
as that harness does), single stream, `vllm bench serve`, default backend:

| ctx | recorded CSV | re-measured here | plow (fp8+occ4) | gap vs re-measured |
|--:|--:|--:|--:|--:|
| 4096 | 7.57 ms | **10.03 / 10.07 / 10.10** | 15.27 | **1.52x** |
| 8192 | 8.62 | **11.08** | 15.34 | **1.38x** |

The run is healthy, not cold: TTFT at 4k is 54.96 ms against the CSV's 120.98 —
BETTER — so this is not a warm-up artifact, and the three TPOT repeats span
0.07 ms. vLLM reports `Model loading took 22.73 GiB`, which also confirms the
weight accounting used throughout this file.

**I am not claiming the CSV is wrong.** I could not reproduce its serve
configuration exactly: that harness runs in the ROCm vLLM *container* with
`--max-num-batched-tokens 8192` and no explicit `--gpu-memory-utilization`, while
this ran natively with `--gpu-memory-utilization 0.85` and a smaller
`--max-model-len`. Neither of those should move decode TPOT by 33%, but the
difference is unexplained and it is 33% on the number every comparison in this
directory is measured against.

What it means concretely: **the deficit is 1.52x at 4k and 1.38x at 8k against a
vLLM this machine actually produces**, not the 2.02x/1.78x quoted earlier. Both
are recorded. Anyone re-running this should measure the baseline in the SAME
session as the plow numbers rather than differencing against a stored CSV.

## vLLM CANNOT be re-measured on this box (2026-08-04)

Every ratio in this directory compares against a vLLM number that cannot currently be
reproduced here, and anyone continuing this work should know exactly why:

  * the 0.23.0+rocm714 install used for the 10.03 ms @4k / 11.08 @8k re-measurement is
    GONE from the image;
  * `vllm` now resolves to 0.7.4.dev388, which predates Gemma-4 support;
  * that build's torch is 2.7.0a0 compiled against **HIP 6.3.42133** while the system
    ROCm is **7.14**, so `torch.cuda.is_available()` is False and vLLM reports
    "No platform detected, vLLM is running on UnspecifiedPlatform";
  * there is no docker daemon, so `scripts/bench_vllm_rocm.sh` (which runs the
    rocm/vllm container) cannot be used either.

So the comparison rests on numbers taken hours earlier in the same session, and the
STORED CSV in perf-data/vllm-rocm/ already failed to reproduce once (7.57 -> 10.03,
a 33% move). RE-BASELINING BOTH ENGINES IN ONE SESSION REMAINS THE SINGLE MOST
VALUABLE MISSING MEASUREMENT, and it needs a box with a working ROCm vLLM.

## Per-XCD queues are now the DEFAULT on gfx942 (2026-08-05)

They were the largest win of the port and were sitting behind two opt-in flags. Now:

  * `plowc` places gfx942 blobs by DEFAULT (`crates/plowc/src/main.rs`, opt out with
    PLOW_L2_PLACE=0). SCOPED to gfx942: a placed blob requires objects built with
    -DPLOW_L2_PLACE_DISPATCH, build_gfx942.sh passes it and build_gfx950.sh does not, so
    defaulting it for all AMD would break gfx950 pipelines. gfx950 blobs verified BYTE-IDENTICAL.
  * `scripts/build_gfx942.sh` passes -DPLOW_L2_PLACE_DISPATCH -DPLOW_GATE_HIER on the decode rows
    by DEFAULT (opt out with PLOW_L2HIER=0). Safe on an unplaced blob -- verified, not assumed:
    those objects run one at 15.60/15.71 ms vs 15.67 for objects built without, i.e. identical.
  * `PLOW_L2_PLACE_DISPATCH=1` IS NO LONGER REQUIRED. The runtime now CHECKS the code object for
    `plow_l2_place_dispatch_1` instead of having the operator assert it, which is strictly
    stronger -- a genuinely mismatched pairing is still refused, by inspection. The env var still
    works for anyone scripting it.

STOCK RUN, no env flags, default blob + default objects (Gemma-4-12B fp8, ctx 4096):

    12.205 / 12.303 / 12.333 ms/token
    coherence gate: 'Paris'  and  'Three prime numbers are **2, 3, and 5**.'
    negative case: objects without the axis are REFUSED, with the rebuild instruction

WHY THE GUARD MOVED. The parse-time refusal ran before any object was loaded, so it could only
ask the operator to promise. It also fired on METADATA-ONLY readers -- `serve`'s memory planner,
the TP fan-out probe, and `disasm` -- none of which dispatch a packet; that is why reading a
placed blob used to need the env var set. Those three now use `DevBlob::parse_l2(.., true)` and
the real check lives in the AMD engine, per phase: `Builder::finish` skips placement on segmented
programs, so a normal blob has a PLACED DECODE program and UNPLACED prefill ones, and requiring
the axis on every object would reject the stock build over its (correctly) plain prefill objects.

## Canonical numbers, 2026-08-05 (supersede everything above)

Stock build, NO env flags -- `PLOW_OCC4=1 bash scripts/build_gfx942.sh` + a default-compiled
blob. `amd-bench --steps 48`, 3 reps:

    ctx 4096   12.050 / 12.087 / 12.093   mean 12.077 ms/token   vs vLLM 10.03   1.204x
    ctx 8192   12.104 / 12.150 / 12.179        12.144            vs vLLM 11.08   1.096x

The 8k deficit is now UNDER 10%. Session arc at 4k: 20.9 (bf16) -> 15.32 (fp8+occ4) ->
12.077, i.e. -42% overall and -21% in this session.

CAVEAT THAT OUTWEIGHS THE LAST DECIMAL: the vLLM column cannot be re-measured on this box
(0.23.0 gone, torch built against HIP 6.3 vs system ROCm 7.14, no docker) and the STORED
baseline already moved 33% once under re-measurement (7.57 -> 10.03 @4k). At 1.096x the 8k
result is INSIDE the range a fresh vLLM baseline could plausibly move. Re-baselining both
engines in one session is still the highest-value missing measurement in this directory.


## Canonical, 2026-08-05 (final for this session)

Stock build, NO env flags, everything default: per-XCD queues, counter double-buffer,
instruction prefetch. `amd-bench --steps 48`, 4 reps, GPU verified idle first:

    ctx 4096   12.015 / 12.033 / 12.076 / 12.009   mean 12.033 ms/token   vs vLLM 10.03   1.200x
    ctx 8192   12.133 / 12.113 / 12.077 / 12.086        12.102            vs vLLM 11.08   1.092x

Coherence gate (serve, real prefill): 'Paris', and a correct, fluent two-sentence technical
answer on GPU matrix multiplication.

Session arc at 4k: 20.9 (bf16) -> 15.32 (fp8+occ4) -> 12.033. -42% overall, -21% this session.
8k deficit is 9.2%.

MEASUREMENT NOTE worth more than the last decimal: two runs in this session produced 13.50 and
13.59 ms outliers against a ~12.05 baseline. Both came from a bench started while a `serve`
process was still tearing down -- the PERSISTENT COOPERATIVE MEGAKERNEL outlives its host
process. `rocm-smi --showuse` must read 0% before any A/B on this box, and that is task #10.
