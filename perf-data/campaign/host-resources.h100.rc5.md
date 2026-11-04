# Host resources: plowrt vs vLLM 0.29, H100 RC5 (2026-11-01)

The host-side cost of serving the same production mix with the qualified RC5 plowrt and with
vLLM 0.29, in two configurations:

1. **Host tier on.** This is plowrt's production setting: a pinned prefix-cache host tier
   beside the GPU prefix cache.
2. **Host tier off** (`PLOW_VMM_HOST_CACHE_MIB=0`). This is the apples-to-apples case: GPU
   prefix cache only, as in vLLM 0.29 with these flags. See "Apples to apples" below. Both arms ran on one box with the same client, prompts and seeds. The data is in
`host-resources.h100.rc5.csv`, one row per model per arm per lease. The `lease` column is
`run2-tier-on`, `ctl-tier-on-sampler-off` or `run3-tier-off`. It was recorded by `llm_grid.sh --prod`
with `scripts/bench/host_res.py`, which samples the server's whole process tree every 2 s.

## Setup

| | |
|---|---|
| Box | gpu03 (lab): 1x NVIDIA H100 80GB HBM3, AMD EPYC 7R13 16 vCPU, 249 GiB RAM, Linux 7.0 |
| Date | 2026-11-01. Tier-on: 14:04-14:37 UTC, four arms in one GPU lease. Tier-off: 16:45-17:56 UTC, six arms in one lease (plowrt tier-off, a fresh vLLM, plowrt tier-off with the sampler off, per model) |
| plowrt | RC5g build, commit `2ee27f888`, binary sha256 `b29db01f163d...`; cuBLASLt 13.4 |
| Packets | 26B BF16 `model.pkt` sha256 `7ce782987bc0...`; 12B FP8 `159093417a1a...` (recipe env from `recipes/infervisor/gemma-4-{26b-a4b,12b}/sm90a-h100-tp1*.toml`) |
| vLLM | 0.29.0, `--gpu-memory-utilization 0.93 --max-num-seqs 256 --max-model-len 16384 --tensor-parallel-size 1`; 12B also `--kv-cache-dtype fp8` and the BF16 tokenizer copy; `VLLM_USE_FLASHINFER_SAMPLER=0`; cutlass DSL on `PYTHONPATH` |
| Load | open-loop production mix (`agentic_turns.py --open-loop --prod-fidelity`), 300 s, of which 75 s warm-up and 25 s cool-down are unmeasured, greedy, 1 rep. 26B BF16 at 1.8 sessions/s (`q1800.g.r1`); 12B FP8 at 2.7 sessions/s (`q2700.g.r1`) |
| Prefix cache | on for both arms; plowrt host KV tier at the recipe default (26B: auto = min(avail - reserve, RAM/4) = 63.9 GiB cap; 12B: `PLOW_VMM_HOST_CACHE_MIB=65536`) |

"Cell" values cover the measured cell, from `CELL_BEGIN` to `CELL_END`. "Session" values cover
everything from server start to exit, model load included. Steady is the median over the cell.

## Apples to apples: plowrt host tier off vs vLLM 0.29 (GPU prefix cache only)

plowrt runs with `PLOW_VMM_HOST_CACHE_MIB=0`, appended after the recipe env, so it overrides
the 12B recipe's 65536 and the 26B auto sizing. The serve log shows `kv store: armed ...
device_cap_mib=20270 host_cap_mib=0`. `plowrt_kv_store_host_bytes` read 0 in every sample of
both cells. The GPU prefix cache and the device KV block store stay on (`PLOW_PREFIX_CACHE=1`,
`PLOW_VMM_PREFIX=1`). The vLLM arms are fresh, measured in the same lease right after plowrt.
Latency rows come from the sampler-off plowrt arm (`PB_HOST_MS=0`) of the same lease; resource
rows come from the sampled arm.

| metric | 26B plowrt | 26B vLLM | ratio | 12B plowrt | 12B vLLM | ratio |
|---|---|---|---|---|---|---|
| Host RSS peak (cell) | 2,906 MiB | 6,925 MiB | 0.42x | 23,929 MiB | 5,680 MiB | 4.21x |
| Host RSS steady (cell median) | 2,886 MiB | 6,918 MiB | 0.42x | 23,908 MiB | 5,663 MiB | 4.22x |
| Pinned KV host tier | 0 | - | | 0 | - | |
| Host RSS peak (whole session, load included) | 49,594 MiB | 51,228 MiB | 0.97x | 35,638 MiB | 6,457 MiB | 5.52x |
| Anon RSS peak (session) | 2,723 MiB | 6,113 MiB | 0.45x | 23,691 MiB | 4,641 MiB | 5.11x |
| VmLck / VmPin peak | 0 / 0 | 0 / 0 | | 0 / 0 | 0 / 0 | |
| System MemAvailable min | 226,303 MiB | 224,223 MiB | 1.01x | 206,716 MiB | 230,323 MiB | 0.90x |
| CPU cores busy avg / peak (cell) | 0.84 / 1.19 | 0.57 / 1.00 | 1.47x | 0.78 / 1.19 | 0.44 / 1.02 | 1.79x |
| Client CPU cores avg | 0.06 | 0.06 | | 0.07 | 0.07 | |
| Threads / processes (cell peak) | 40 / 1 | 193 / 3 | | 40 / 1 | 188 / 3 | |
| Binary vs site-packages | 0.02 GiB | 11.11 GiB | | 0.02 GiB | 11.11 GiB | |
| Mapped shared libraries | 0.56 GiB | 4.98 GiB | 0.11x | 0.56 GiB | 4.94 GiB | 0.11x |
| GPU memory peak (cell) | 79,828 MiB | 76,322 MiB | 1.05x | 79,504 MiB | 76,800 MiB | 1.04x |
| GPU prefix/KV capacity | device store 19.8 GiB | KV cache 22.9 GiB (109k tokens) | 0.86x | device store 19.8 GiB | KV cache 58.9 GiB (367k tokens) | 0.34x |
| Prefix token hit (`/metrics`) | 31.5% | 56.2% | | 32.6% | 75.7% | |
| Server ready time | 25.6 s | 190.6 s | 0.13x | 56.9 s | 102.3 s | 0.56x |
| Total throughput (sampler off) | 40,167 tok/s | 40,551 tok/s | 0.99x | 44,865 tok/s | 52,107 tok/s | 0.86x |
| TTFT P99 (sampler off) | 2,392.5 ms | 399.9 ms | 5.98x | 15,380.0 ms | 956.7 ms | 16.1x |
| TPOT P99 (sampler off) | 84.38 ms | 39.56 ms | 2.13x | 73.75 ms | 117.99 ms | 0.63x |
| SLO attainment (sampler off) | 98.13% | 100.0% | | 26.2% | 97.75% | |

The sampled tier-off arm is within run-to-run noise of the sampler-off arm:
- 26B: 40,073 tok/s, TTFT P99 2,238 ms, TPOT P99 85.9 ms, 30.3% hit.
- 12B: 44,350 tok/s, TTFT P99 17,373 ms, TPOT P99 75.3 ms, 32.6% hit.

**What the host tier buys.** The comparison is plowrt tier-on (sampler-off control, same
binary, packets and mix, earlier lease) vs plowrt tier-off (sampler-off):

| | 26B tier on | 26B tier off | 12B tier on | 12B tier off |
|---|---|---|---|---|
| Prefix token hit | 77.0% | 31.5% | 76.6% | 32.6% |
| Host RSS (cell median, sampled arm) | 65.2 GiB | 2.8 GiB | 87.4 GiB | 23.3 GiB |
| Throughput | 40,409 tok/s | 40,167 tok/s | 52,179 tok/s | 44,865 tok/s |
| TTFT P99 | 616 ms | 2,393 ms | 706 ms | 15,380 ms |
| TPOT P99 | 55.4 ms | 84.4 ms | 52.2 ms | 73.8 ms |
| SLO attainment | 99.9% | 98.1% | 98.9% | 26.2% |

What this shows:

* The tier costs about 64 GiB of pinned host RAM. For that it more than doubles plowrt's prefix
  hit rate on this agentic mix. It is what keeps plowrt ahead of vLLM on 12B (TTFT P99 706 vs
  944 ms) and within 1.5x on 26B (616 vs 411 ms).
* Without the tier, plowrt's GPU-only prefix cache hits far less than vLLM's GPU-only cache:
  31% vs 56% on 26B, and 33% vs 76% on 12B. Two causes:
  * plowrt's device block store is capped at 19.8 GiB (`device_cap_mib=20270`).
  * vLLM gives every byte left after weights at 0.93 utilization to one paged KV pool that is
    both working set and prefix cache: 22.9 GiB on 26B and 58.9 GiB on 12B.
* On 12B, plowrt therefore re-prefills most context. Prefill then saturates, and the open loop
  backs up: TTFT P50 8.8 s, SLO 26%.
* As a GPU-only server, plowrt would need a device prefix capacity comparable to vLLM's KV pool
  to match it. The host tier currently makes up for the smaller device store.

## Host tier on (production setting): per-model detail

The two tables below are the tier-on lease. plowrt's host RSS there is mostly the pinned
prefix-cache host tier.

## Gemma-4 26B-A4B BF16, q1800 (1.8 sessions/s)

| metric | plowrt | vLLM | plowrt / vLLM |
|---|---|---|---|
| Host RSS peak (cell) | 66,800 MiB | 6,854 MiB | 9.75x |
| Host RSS steady (cell median) | 66,780 MiB | 6,849 MiB | 9.75x |
|   of which pinned KV host tier (peak, `/metrics`) | 63,876 MiB | - | - |
|   RSS excluding the tier | ~2,900 MiB | 6,854 MiB | 0.42x |
| Host RSS peak (whole session, load included) | 66,800 MiB | 46,491 MiB | 1.44x |
| Anon RSS peak (session) | 66,617 MiB | 5,906 MiB | 11.28x |
| VmLck / VmPin peak | 0 / 0 MiB | 0 / 0 MiB | - |
| System MemAvailable min | 162,312 MiB | 227,343 MiB | 0.71x |
| CPU cores busy avg (cell) | 1.00 | 0.49 | 2.02x |
| CPU cores busy peak (2 s interval) | 1.65 | 0.84 | 1.95x |
| Client CPU cores avg | 0.06 | 0.06 | 1.04x |
| Threads (cell peak) | 43 | 193 | 0.22x |
| Processes (cell peak; session peak) | 1; 1 | 3; 5 | 0.33x |
| Server install: binary vs site-packages | 0.02 GiB | 11.11 GiB | 0.002x |
| Mapped shared libraries (count, size) | 9, 0.56 GiB | 387, 4.98 GiB | 0.11x |
| Packet dir vs vLLM/inductor/triton caches | 0.15 GiB | 1.83 GiB | 0.08x |
| Checkpoint (both read the same HF dir) | 48.1 GiB | 48.1 GiB | 1.00x |
| GPU memory peak (cell) | 79,974 MiB | 76,232 MiB | 1.05x |
| Server ready time | 27.6 s | 182.3 s | 0.15x |
| Total throughput | 40,441 tok/s | 40,558 tok/s | 1.00x |
| TTFT P99 | 1,138.9 ms | 411.0 ms | 2.77x |
| TPOT P99 | 58.33 ms | 42.12 ms | 1.38x |
| SLO attainment | 99.86% | 100.0% | - |

## Gemma-4 12B FP8, q2700 (2.7 sessions/s)

| metric | plowrt | vLLM | plowrt / vLLM |
|---|---|---|---|
| Host RSS peak (cell) | 89,498 MiB | 5,691 MiB | 15.73x |
| Host RSS steady (cell median) | 89,476 MiB | 5,675 MiB | 15.77x |
|   of which pinned KV host tier (peak, `/metrics`) | 65,536 MiB | - | - |
|   RSS excluding the tier | ~23,960 MiB | 5,691 MiB | 4.21x |
| Host RSS peak (whole session, load included) | 89,498 MiB | 16,184 MiB | 5.53x |
| Anon RSS peak (session) | 89,251 MiB | 4,663 MiB | 19.14x |
| VmLck / VmPin peak | 0 / 0 MiB | 0 / 0 MiB | - |
| System MemAvailable min | 146,674 MiB | 231,925 MiB | 0.63x |
| CPU cores busy avg (cell) | 1.02 | 0.40 | 2.58x |
| CPU cores busy peak (2 s interval) | 1.67 | 0.77 | 2.18x |
| Client CPU cores avg | 0.08 | 0.07 | 1.10x |
| Threads (cell peak) | 43 | 188 | 0.23x |
| Processes (cell peak; session peak) | 1; 1 | 3; 5 | 0.33x |
| Server install: binary vs site-packages | 0.02 GiB | 11.11 GiB | 0.002x |
| Mapped shared libraries (count, size) | 9, 0.56 GiB | 385, 4.94 GiB | 0.11x |
| Packet dir vs vLLM/inductor/triton caches | 0.19 GiB | 1.83 GiB | 0.11x |
| Checkpoint (both read the same HF dir) | 12.2 GiB | 12.2 GiB | 1.00x |
| GPU memory peak (cell) | 79,320 MiB | 76,800 MiB | 1.03x |
| Server ready time | 55.9 s | 124.8 s | 0.45x |
| Total throughput | 52,303 tok/s | 52,083 tok/s | 1.00x |
| TTFT P99 | 660.8 ms | 943.7 ms | 0.70x |
| TPOT P99 | 53.94 ms | 113.54 ms | 0.48x |
| SLO attainment | 98.85% | 97.75% | - |

## Software footprint detail

* plowrt: a 23.9 MB static binary. At runtime it maps 9 libraries: glibc's libc, libm,
  libdl, libpthread, librt and ld.so, plus libgcc_s, libcuda (87 MiB) and the dlopened
  cuBLASLt 13.4 (485 MiB). The packet dir is 0.15-0.19 GiB: `model.pkt`, cubins, tokenizer and the cuBLASLt algorithm table. The
  checkpoint is a link and is not counted.
* vLLM 0.29: the venv holds 0.82 GiB (the vllm package). It inherits the system PyTorch
  site-packages at 10.9 GiB: nvidia libs 3.9, torch 1.1, vllm 0.8, triton 0.7,
  transformer_engine 0.5 and cutlass DSL 0.5 GiB. Together that is 11.1 GiB, plus 0.25 GiB of
  system dist-packages on `sys.path`. At runtime it maps 385-387 libraries (4.9-5.0 GiB): the
  largest are cuBLASLt (517 MiB), libtorch_cuda, libtriton, libtorch_cpu, vLLM FA2 and cuFFT.
* Compile and JIT caches: vLLM 1.84 GiB, inductor 0.10 GiB, triton 0.02 GiB. They were already
  warm from earlier runs, so the vLLM ready times are warm-cache times. A cold start compiles
  for several minutes more.

## Checks

* **Sampler vs ps, during each measured cell.** The `ps` RSS sum over the hand-walked process
  tree (`/proc/<pid>/task/*/children`) matched the sampler row taken within 2 s:

  | arm | ps RSS sum | sampler RSS | ps threads / sampler | tree |
  |---|---|---|---|---|
  | 26B plowrt | 58,369 MiB | 58,330 MiB (1 s earlier, tier growing) | 43 / 43 | plowrt |
  | 26B vLLM | 6,846 MiB | 6,846 MiB | 193 / 193 | API server, resource tracker, `VLLM::EngineCore` |
  | 12B plowrt | 89,466 MiB | 89,407 MiB | 43 / 43 | plowrt |
  | 12B vLLM | 5,665 MiB | 5,665 MiB | 188 / 188 | API server, resource tracker, `VLLM::EngineCore` |
  | 26B plowrt tier-off | 2,877 MiB | 2,877 MiB | 40 / 40 | plowrt |
  | 26B vLLM (tier-off lease) | 6,914 MiB | 6,914 MiB | 193 / 193 | API server, resource tracker, `VLLM::EngineCore` |
  | 12B plowrt tier-off | 23,889 MiB | 23,889 MiB | 40 / 40 | plowrt |
  | 12B vLLM (tier-off lease) | 5,656 MiB | 5,656 MiB | 188 / 188 | API server, resource tracker, `VLLM::EngineCore` |

  The vLLM tree includes EngineCore (3.5-3.8 GiB RSS, about 115-120 threads in the cell). Its
  load-time peak was 44.6 GiB RSS on 26B while loading weights. During load the session sampler
  also caught two short-lived JIT helper processes (`humming` nvrtc, ~0.33 GiB each): the
  session peak is 5 processes. VmLck
  and VmPin read 0 in `/proc/<pid>/status` on both servers. CUDA pinned host memory
  (`cuMemHostAlloc`) counts as anonymous RSS, not VmPin. So plowrt's pinned host tier shows up
  as RSS and is reported from plowrt's own `plowrt_kv_store_host_bytes`.
* **Sampler overhead.** The first version summed Pss from `/proc/<pid>/smaps_rollup`. That walks
  every page-table entry under the target's mmap lock: 0.32 s per 16 GiB resident (measured),
  so about 1.5 s per sample on plowrt's 60-90 GiB. Its run is discarded. In that run, 26B
  plowrt TTFT P99 was 2.1 s and TPOT P99 104 ms, and 12B plowrt fell to 26.5% SLO attainment
  (TTFT P50 6.5 s). The sampler now reads only `status`, `stat` and `maps`; Pss is opt-in.
  With that sampler, a sampler-off control of plowrt in a follow-on lease gave:

  | arm, sampler | throughput | TTFT P99 | TPOT P99 | SLO |
  |---|---|---|---|---|
  | 26B plowrt, on | 40,441 | 1,138.9 ms | 58.33 ms | 99.86% |
  | 26B plowrt, off | 40,409 | 615.6 ms | 55.38 ms | 99.86% |
  | 12B plowrt, on | 52,303 | 660.8 ms | 53.94 ms | 98.85% |
  | 12B plowrt, off | 52,179 | 706.1 ms | 52.17 ms | 98.85% |

  12B shows no effect. On 26B, throughput, TPOT and SLO match, but TTFT P99 differs by
  0.5 s. That P99 covers 14 requests at 1 rep, and this cell has had r1 tail outliers before.
  The RC5 qualification ran without this sampler and gave 592/569 ms. So a residual sampler
  effect on the 26B TTFT tail is not excluded. Read the 26B plowrt TTFT P99 above as an upper
  bound, and use a sampler-off run (`PB_HOST_MS=0`) for latency evidence.

## Caveats

* plowrt's host RSS is mostly the pinned prefix-cache host tier: 63.9 GiB on 26B and 64 GiB on
  12B, at the recipe default, prefix cache on. It is a deliberate cache sized to free RAM, not
  a working set, and it fills to its cap within about 3 minutes of the mix. vLLM keeps no host
  KV tier with these flags. Compare plowrt RSS excluding the tier: 26B ~2.9 GiB vs vLLM 6.7 GiB.
* Odd: 12B FP8 plowrt holds about 23.4 GiB of anonymous RSS beyond the tier. The tier-off run
  confirms it: 23.3 GiB with the tier at 0. It is already ~25 GiB before the tier starts
  growing, so it is load-time host memory that is never released. 26B BF16 holds only
  ~2.8 GiB. This looks like host copies kept by the FP8 checkpoint load path and is worth a
  look in plowrt.
* vLLM's session RSS peak (46.5 GiB on 26B, 16.2 GiB on 12B) is EngineCore while it loads
  weights. Its file-backed pages come from the safetensors mmap. The tree's steady state is
  5.7-6.9 GiB.
* CPU is the server tree's utime+stime over wall time on a 16-vCPU box. plowrt's ~1 core is its
  engine thread plus HTTP/tokenizer. vLLM's ~0.4-0.5 core excludes GPU-side time in both cases.
* Results are 1 rep (screen budget). The serving metrics are context for the resource numbers,
  not a qualified comparison: those come from `serving_comparison.py` / `campaign.py report`.
* Raw evidence is kept in lab scratch under `host-res/`:
  * `run2/<model>/<arm>/`: `host.tsv`, `host.json`, `<cell>.host.json`, `footprint.json`, `server.log`, client JSON;
  * `run2/handcheck.log`: the ps cross-checks;
  * `ctl/`: the sampler-off control;
  * `run3/`: the tier-off lease (`plow-tieroff`, `vllm`, `plow-nosampler` per model, plus `handcheck.log`);
  * the discarded Pss run sits at the top level.
