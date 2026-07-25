# Gemma-4-12B-it on H100 NVL — where plow lands (bf16 + fp8, prefill + decode)

Date: 2026-07-25. Branch: `perf-12b-h100-landing`, a single commit on top of
`worktree-beat26b-h100-campaign` @ `e434603`, adding only this card and
`vllm_tpot_12b_h100.py`.

Companion to `gemma26b-h100-beat-vllm-campaign.md`: same GPU, same runtime, same
method of record, but the **dense 12B** instead of the 26B MoE. This card is a
*landing* measurement — stock configuration, no tuning knobs — so it is the
baseline any 12B optimisation work should be measured against.

## Headline

| | bf16 | fp8 (w8a16) |
|---|---|---|
| decode TPOT @ctx1024 | **13.307 ms** (75.1 tok/s) | **10.443 ms** (95.8 tok/s) |
| decode TPOT @ctx4096 | **13.652 ms** (73.2 tok/s) | **10.832 ms** (92.3 tok/s) |
| prefill @ctx1024 | **0.197 s** (5,198 tok/s) | 0.292 s (3,507 tok/s) |
| prefill @ctx4096 | **0.789 s** (5,191 tok/s) | 1.100 s (3,724 tok/s) |
| weights resident | 22.18 GiB | 12.04 GiB |

vLLM 12B bf16 on the same box, same day: **11.058 ms @ctx1024, 11.147 ms @ctx4096.**
So plow bf16 decode is 1.20–1.23× slower, and **plow fp8 decode is 1.03–1.06×
faster than vLLM bf16.** Full comparison and its caveats below.

Two results worth flagging:

- **fp8 buys 21–22% on decode** (13.307 → 10.443 @1k, 13.652 → 10.832 @4k) and
  halves the weight footprint. Decode is weight-bandwidth-bound, so this tracks
  the 22.18 → 12.04 GiB drop closely.
- **fp8 *costs* 33–39% on prefill** (0.197 → 0.292 s @1k). Prefill is
  compute-bound and the fp8 weights are dequantised into the bf16 GEMM path —
  there is no w8a8 tensor-core prefill program in this build, so fp8 pays the
  dequant without getting the tensor-core win. Same capability gap the 26B card
  records ("fp8 has NO prefill program"). `PLOW_W8A8=1` exists in the emitter
  (`crates/devgen/src/lib.rs:6192`) and is the lever to close this.
- **Decode is nearly flat in context** (+2.6% bf16, +3.7% fp8 from 1k → 4k),
  consistent with the campaign's "plow decode is weight-bound, ~flat in ctx"
  finding. Prefill is linear in ctx (5.2k tok/s bf16 at both 1k and 4k).

## Model

The 12B is **not** under `/workspace/models/` — that directory holds only
`gemma-4-26B-A4B-it` and `gemma-4-E4B-it`. The HF cache had a
`models--google--gemma-4-12B-it` entry containing `config.json` **only** (no
weight blobs). `/` is 100% full (~485 MB free), so the checkpoint was fetched
into `/dev/shm` (RAM-backed, 70 GB, host has 1.5 TB):

- repo `google/gemma-4-12B-it`, sha `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`,
  public/ungated, single `model.safetensors`, 11,959,730,224 params BF16 (23.9 GB)
- fetched to `/dev/shm/models/gemma-4-12B-it`
- fp8 twin: `perf-data/harness/quantize_fp8.py` → `/dev/shm/models/gemma-4-12B-it-fp8`
  (10.91 GB over 328 projections; per-output-channel e4m3, scale = amax/448),
  linked in as `<model>/fp8-full-plow`

Geometry as the emitter reports it:

```
48 layers (8 full)  hidden=3840 inter=15360  heads=16  hd=256/512  kvh=8/1  vocab=262144
```

**These artefacts live in `/dev/shm` and do not survive a reboot.** Re-create with
the commands below; everything is scripted and takes ~4 minutes end to end.

## Provenance

| artefact | value |
|---|---|
| GPU | NVIDIA H100 NVL, 132 SM, driver 570.133.20 |
| runtime env | `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat` |
| nvcc | CUDA 13.0 V13.0.88 |
| `interp_sm90a.cubin` | sha256 `5edf093b…52b`, 1,325,272 B |
| `interp_sm90a_pf.cubin` | sha256 `5baf3396…c86`, 666,584 B |
| bf16 `model.pkt` | sha256 `37cffe6e…94c`, 25,573,864 B |
| fp8 `model.pkt` | sha256 `d9735482…551`, 25,611,432 B |
| `plowc` | sha256 `f1b46fba…67f` |
| `step_bench` | sha256 `6587e49b…6cf` |

The cubins were built **from this worktree's own `runtime/nvidia/`** at
`e434603` and are **byte-identical** to the pre-existing
`/workspace/assets/cubin-sm90a-rb/` pair — so the numbers here are directly
comparable to the sibling `rb` runs, and the source state is pinned.

`plowc` / `step_bench` are the pre-built release binaries from
`/root/plow/.claude/worktrees/beat26b-h100-campaign/target/release/`; that
worktree's `crates/` and `runtime/` were byte-identical to this branch
(`diff -rq`, no output) when the binaries were taken, so no rebuild was needed.
(`/` had no room for a from-scratch target dir.)

Two provenance caveats, both checked and both benign:

1. `scripts/build_sm90a_cubin.sh` was edited *by another agent* at 02:57, two
   minutes before this cubin build, to add a `$EXTRA` tuner hook. The hook is
   `EXTRA="${PLOW_EXTRA_DEFINES:-}"` — empty unless that variable is set, and the
   build here ran under `env -i` with it unset. Proof rather than argument: the
   resulting cubins are **byte-identical to `/workspace/assets/cubin-sm90a-rb/`,
   which was built at 02:29, before the edit.** The shipped recipe was compiled.
2. A re-check of `diff -rq` at the end of the task shows `runtime/` still
   byte-identical, and `crates/` differing **only** under `crates/tunedb/`
   (another agent's in-flight tuner work, which appeared inside this worktree
   mid-task). `tunedb` is reached only via `plowc --tuning-db`, which was not
   passed, and cannot affect a pre-built binary in any case. Nothing on the
   emit or execute path moved.

### Reproduce

```sh
# cubins (outside nix, clean env)
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin TMPDIR=/dev/shm/tmp PLOW_ROOT=$PWD \
  bash scripts/build_sm90a_cubin.sh /dev/shm/cubin12b/interp_sm90a.cubin

# fp8 twin
/workspace/venvs/torch128/bin/python perf-data/harness/quantize_fp8.py \
  /dev/shm/models/gemma-4-12B-it /dev/shm/models/gemma-4-12B-it-fp8
ln -sfn /dev/shm/models/gemma-4-12B-it-fp8 /dev/shm/models/gemma-4-12B-it/fp8-full-plow

# packets
PLOW_UNISEG=1 plowc --hf-dir /dev/shm/models/gemma-4-12B-it --emit devblob \
  --max-ctx 8192 --n-cu 132 --out /dev/shm/assets12b/bf16
PLOW_UNISEG=1 PLOW_FP8=1 plowc --hf-dir /dev/shm/models/gemma-4-12B-it --emit devblob \
  --max-ctx 8192 --n-cu 132 --out /dev/shm/assets12b/fp8
```

> **`--weight-dtype fp8` is inert on the `devblob` path.** Emitting with
> `--weight-dtype fp8` produced a `model.pkt` **byte-identical** to the bf16 one
> (same 22.2 GiB weight footprint, same 542-packet decode program). The knob the
> devblob emitter actually reads is the env var **`PLOW_FP8=1`**
> (`crates/devgen/src/lib.rs:5019,6191`), which gives weights 12.0 GiB and a
> 622-packet decode program. `--weight-dtype` only reaches the `packets`
> pipeline (`crates/plowc/src/lib.rs:522`). Anyone scripting fp8 devblobs off
> the `--help` text will silently benchmark bf16 twice.

The fp8 asset dir needs a `checkpoint/` carrying **both** tensor sets under
non-colliding *file* names (the engine maps tensors by name: bf16 `model.*`,
fp8 `fp8/model.*`):

```
fp8/checkpoint/bf16-model.safetensors -> <model>/model.safetensors
fp8/checkpoint/fp8-model.safetensors  -> <model>/fp8-full-plow/model.safetensors
```

## Method

`crates/plowrt/examples/step_bench.rs`, which drives `GpuEngine` directly — no
HTTP, no mux, no SSE — so these are kernel-level numbers with no serving overhead.

```
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat \
gpulease <label> step_bench /dev/shm/assets12b/<prec> 1 <ctx> 128
```

- **Decode**: `RAW_STEP … mean_ms=`, B=1, 128 timed steps after 16 discarded
  warmup steps (repo convention). Reported cell = **median of 5 process runs**;
  "run sd" is the sd across those 5 means, "step sd" is the median within-run sd
  over the 128 steps.
- **Prefill**: `slot 0: prompt consumed in <s>` — one `prefill_slot` call on a
  `ctx`-token synthetic prompt. **This is the fallback the task allowed**: the
  repo has no dedicated kernel-only prefill bench (`scripts/bench_plowrt_sweep.sh`
  is an HTTP/served TTFT sweep, which would fold in the serving layer). It is a
  *cold, single, un-warmed* call per process, so it is the weakest number on this
  card — but it is reproducible to ±0.5% across 5 processes.
- Every GPU command ran under `gpulease`.

## Results (n=5 process runs per cell)

### Decode — TPOT, B=1

| precision | ctx | TPOT median-of-5 | run sd | step sd (median within-run) | tok/s |
|---|---|---|---|---|---|
| bf16 | 1024 | **13.307 ms** | 0.018 | 0.580 | 75.1 |
| bf16 | 4096 | **13.652 ms** | 0.023 | 0.391 | 73.2 |
| fp8  | 1024 | **10.443 ms** | 0.013 | 0.014 | 95.8 |
| fp8  | 4096 | **10.832 ms** | 0.005 | 0.020 | 92.3 |

fp8 is markedly *steadier* step-to-step (step sd 0.014–0.020 ms vs bf16's
0.391–0.580 ms). The bf16 step sd is ~4% of the mean — worth a look on its own;
it is not run-to-run drift (run sd is 0.02 ms), so it is per-step jitter inside
the bf16 decode program.

### Prefill — one `prefill_slot` on a ctx-token prompt

| precision | ctx | prefill median-of-5 | spread | tok/s |
|---|---|---|---|---|
| bf16 | 1024 | **0.197 s** | 0.195–0.198 | 5,198 |
| bf16 | 4096 | **0.789 s** | 0.779–0.792 | 5,191 |
| fp8  | 1024 | **0.292 s** | 0.292–0.292 | 3,507 |
| fp8  | 4096 | **1.100 s** | 1.099–1.102 | 3,724 |

### Raw

```
prec	ctx	run	vram_before_mib	prefill_s	mean_ms	median_ms	sd_ms
bf16	1024	1	52573	0.195	13.307	13.260	0.592
bf16	1024	2	52573	0.197	13.303	13.199	0.566
bf16	1024	3	52573	0.198	13.295	13.175	0.564
bf16	1024	4	52573	0.197	13.315	13.205	0.580
bf16	1024	5	52573	0.197	13.342	13.408	0.591
bf16	4096	1	52573	0.788	13.614	13.479	0.396
bf16	4096	2	52573	0.779	13.651	13.498	0.391
bf16	4096	3	52573	0.789	13.652	13.559	0.323
bf16	4096	4	52573	0.792	13.670	13.649	0.203
bf16	4096	5	52573	0.792	13.669	13.692	0.400
fp8	1024	1	52573	0.292	10.467	10.467	0.014
fp8	1024	2	52573	0.292	10.443	10.441	0.015
fp8	1024	3	52573	0.292	10.465	10.464	0.015
fp8	1024	4	52573	0.292	10.442	10.439	0.014
fp8	1024	5	52573	0.292	10.442	10.442	0.014
fp8	4096	1	52573	1.100	10.830	10.825	0.024
fp8	4096	2	52573	1.099	10.828	10.827	0.017
fp8	4096	3	52573	1.100	10.832	10.828	0.025
fp8	4096	4	52573	1.102	10.840	10.838	0.020
fp8	4096	5	52573	1.102	10.832	10.831	0.017
```

## GPU contention — read this before trusting the numbers

**No run on this card was taken with `memory.used < 2000 MiB.`** A foreign
process (host pid 651779, outside our PID namespace, unattributable and
unstoppable by `gpulease` — its `foreign()` audit degrades to lock-only in a
container) held **52,564 MiB for the entire task window** (02:41 → 03:2x, never
dropping). `vram_before_mib = 52573` on every single row above records this
honestly.

What can be said in its defence, measured rather than assumed:

- **It was doing no compute.** `utilization.gpu` sampled at 3 s intervals for
  60 s immediately before the measurement set: **0% on all 20 samples**, SM
  clocks pinned at 1785 MHz, `used_memory` constant at 52,564 MiB throughout. It
  is a memory-resident, idle occupant — it does not contend for SMs or HBM
  bandwidth, only for capacity.
- **The timings show no contention signature.** Run-to-run sd is 0.005–0.023 ms
  on a ~10–14 ms measurement (≤0.2%), and the fp8 within-run sd is 0.014 ms over
  128 steps. A process competing for bandwidth would not leave that.
- Capacity was never the binding constraint either: 95,830 − 52,573 = 43,257 MiB
  free vs bf16's 26.5 GiB peak (22.18 weights + 2.62 KV + 1.63 activations).

**So: treat these as high-confidence but formally unverified-uncontended.** They
should be re-taken on a clean GPU before being quoted as the 12B baseline of
record. The re-run is `bash` over `perf-data/harness/` in ~3 minutes; the assets
are all that matter and they are hash-pinned above.

## Tuning check — `PLOW_NS_FULL_ABS`

`scripts/build_sm90a_cubin.sh` prescribes `PLOW_NS_FULL_ABS=33` as the H100
"runtime companion". For the 12B, `n_grp = heads/FA_GF_FULL = 16/4 = 4`, so the
grid-aligned split is `132/gcd(4,132) = 33` — the same value as the 26B. A/B at
3 runs per arm:

| precision | ctx | default | ns=33 | delta |
|---|---|---|---|---|
| bf16 | 1024 | 13.350 / 13.375 / 13.385 | 13.347 / 13.383 / 13.382 | ~0 |
| bf16 | 4096 | 13.710 / 13.751 / 13.720 | 13.729 / 13.739 / 13.740 | ~0 |
| fp8  | 1024 | 10.434 / 10.427 / 10.434 | 10.425 / 10.429 / 10.428 | ~0 |
| fp8  | 4096 | 10.838 / 10.841 / 10.835 | 10.849 / 10.834 / 10.841 | ~0 |

**Neutral at ctx ≤ 4096** (all deltas < 0.2%, inside run-to-run spread), which
matches the knob's own documentation — it is a long-context lever (the cited win
is at 32k–128k, and "@1k … free"). The headline numbers are therefore stock-default
and lose nothing to it at these contexts. Note also that the two automatic
grid-alignment rules in `crates/devgen/src/lib.rs` are both gated on `ctx > 8192`
and so never fire here; the 12B's `kvh_full = 1` signature additionally gates the
second one on `fp8_kv`, which this build does not set.

## vLLM comparison — decode only

Obtained, via the driver added here as `perf-data/vllm_tpot_12b_h100.py` (takes
`<ctx> [model_dir] [dtype]`, honours `VLLM_GPU_UTIL`). Two deviations from the
26B recipe were forced by the box: `ninja` is not on the default `PATH` (it
lives in `/workspace/venvs/vllm-blk/bin`, and without it flashinfer's JIT dies
with `FileNotFoundError: ninja`), and `gpu_memory_utilization` had to drop to
**0.33** — at 0.42 vLLM OOMed against the foreign process's 51.33 GiB.

| ctx | plow bf16 | plow fp8 | vLLM bf16 | plow bf16 vs vLLM | plow fp8 vs vLLM |
|---|---|---|---|---|---|
| 1024 | 13.307 ms | 10.443 ms | **11.058 ms** | 1.20× slower | **1.06× faster** |
| 4096 | 13.652 ms | 10.832 ms | **11.147 ms** | 1.23× slower | **1.03× faster** |

- **plow bf16 loses decode by ~20–23%** — a much narrower gap than the 26B MoE,
  where plow is 1.93× slower (9.340 vs 4.833).
- **plow's fp8 decode beats vLLM's bf16 decode** at both contexts. This is *not*
  an equal-precision comparison — it is w8a16 against bf16 — but it is the
  configuration a user would actually deploy, and vLLM fp8 was not measured (see
  below), so treat it as "plow's best 12B decode beats vLLM's stock 12B decode",
  not as a kernel-vs-kernel win.
- Both engines are near-flat in context here (vLLM 11.058 → 11.147), confirming
  B=1 decode at ≤4k is weight-bandwidth-bound for both.

**vLLM fp8 was NOT measured** — it needs either a pre-quantised
compressed-tensors checkpoint or vLLM's on-the-fly `quantization="fp8"`, which
is a separate calibration path and was out of budget here. The fp8 column above
is plow-only.

**The vLLM `t1` value is NOT a valid TTFT and no vLLM prefill number is quoted.**
`t1` @ctx1024 was 20.69 ms; subtracting one ~11.06 ms decode step leaves ~9.6 ms
for a 1024-token prefill, i.e. 2,552 TFLOP/s against the H100's ~1,979 TFLOP/s
dense bf16 peak — physically impossible. vLLM V1 enables automatic prefix
caching by default and the driver re-sends an identical prompt after a warm-up
call, so the prefill is served from cache. The same caveat applies to the 26B
card's `t1`. Comparing prefill against vLLM needs APC explicitly disabled.

Caveat: the vLLM runs were taken under the same resident foreign process as
everything else on this card, *and* additionally squeezed into a 0.33 memory
fraction, which shrinks the KV cache. At B=1 and ctx ≤ 4k that should not move
TPOT, but it is one more reason to re-take the whole comparison on a clean GPU.

## Roofline

| path | bytes/token | measured | achieved BW | % of 3.9 TB/s |
|---|---|---|---|---|
| decode bf16 @1k | 22.18 GiB | 13.307 ms | 1.79 TB/s | **45.9%** |
| decode fp8 @1k | 12.04 GiB | 10.443 ms | 1.24 TB/s | **31.7%** |

(H100 NVL = HBM3, ~3.9 TB/s. Weights only; KV adds a little at these contexts.)

- The dense 12B decode reaches **45.9% of peak bandwidth in bf16** — far
  healthier than the 26B MoE decode path's 20.9% recorded in the campaign card,
  and the reason the 12B's gap to vLLM is 1.20× rather than 1.93×.
- **fp8 decode is the weaker of the two in roofline terms** (31.7%): halving the
  bytes bought only 21% of time, so the fp8 path is leaving ~30% on the table
  and is the better decode target of the two.
- **Prefill is the real weak axis.** 2·12e9·1024 FLOP / 0.197 s = **125 TFLOP/s,
  ~6% of the ~1,979 TFLOP/s dense bf16 peak.** fp8 prefill is worse still
  (84 TFLOP/s). Whatever the decode story, prefill is where the order-of-magnitude
  headroom is on this model.

For scale, the 26B MoE on the same box and runtime
(`gemma26b-h100-beat-vllm-campaign.md`): decode TPOT bf16 @ctx1024 9.340 ms,
fp8 7.395 ms, vLLM 4.833 / 4.417 ms. The dense 12B is *slower per token* than
the 26B MoE (13.307 vs 9.340 bf16) simply because the MoE gathers only top-8
experts (~7.6 GB/token) while the dense 12B streams all 22.18 GiB every token —
but it is much closer to its own roofline, and much closer to vLLM.
