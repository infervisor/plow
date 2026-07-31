# Cold start — plowrt vs vLLM, gemma-4 12B and 31B, GH200 (2026-07-31)

Box: NVIDIA GH200 480GB (sm_90a, 132 SM, 97871 MiB), 573 GB host RAM, checkpoint
on NVMe (ext4 on LVM). plowrt = this branch, `--release --features cuda`;
vLLM = 0.26.0 installed in the host python.

Model: `google/gemma-4-12b-it`, bf16, **one** 22.28 GiB safetensors shard
(22.18 GiB of it weights plowrt binds; vLLM counts 22.83 GiB including what it
keeps for the multimodal tower). plowrt bundle
`plow-out/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (ctx 131072, batch 1,
`interp_sm90a`); vLLM served with the flags this box was already using —
`--dtype bfloat16 --tensor-parallel-size 1 --max-model-len 131072
--gpu-memory-utilization 0.50`.

"cold" = `echo 3 > /proc/sys/vm/drop_caches` immediately before the run, so the
checkpoint comes off the drive. "warm" = the shard pre-read into page cache.

## 1. Cold start to serving — the number that matters

Process launch to the server accepting a request with the model live on the GPU
(`t_listen`, /healthz resp. /health returning 200), and to the first completion
coming back (`t_first`, which additionally runs prefill+decode kernels, so it
proves the GPU path rather than just an open socket).

| engine | state | t_listen | t_first |
|---|---|---|---|
| plowrt, prefetch OFF (`PLOW_PREFETCH_THREADS=0`) | cold | 17.55 s | 17.59 s |
| **plowrt, prefetch ON (16 threads, default)** | cold | **11.19 s** | **11.23 s** |
| plowrt, prefetch ON | warm | 11.29 s | 11.31 s |
| vLLM 0.26.0 | cold | 97.04 s | 97.13 s |
| vLLM 0.26.0 | warm | 103.36 s | 103.45 s |

With the pool on, plowrt's cold start equals its warm start (11.19 vs 11.29 s) —
the page-cache state stops mattering, which is the whole point. vLLM is ~97–103 s
either way and if anything noisier warm than cold: its startup is dominated by
init work that the page cache cannot help, so the 6 s spread there is run-to-run
variance in compile/capture, not an I/O effect.

- The prefetch pool takes **1.57×** off plowrt's cold start (17.55 → 11.19 s).
- Against vLLM, plowrt is **8.7×** faster to serving.
- `t_first − t_listen` is **33 ms** for plowrt and **90 ms** for vLLM: both open
  the listener only after the weights are resident, so neither is cheating by
  accepting connections it cannot serve.

vLLM's 97 s is with its `torch.compile` cache already populated. The FIRST cold
start on this box — cache empty — took **129.6 s** to /health, of which the
engine reported 71.0 s for "init engine (profile, create kv cache, warmup model)"
and 32.2 s of that was compilation. plowrt does that work AOT in `plowc`, which
is where most of the 8.7× lives; the weight upload is the smaller half.

## 1b. Same thing at 31B

`gemma-4-31b-it`, 57.18 GiB of weights, bundle emitted here with
`plowc --hf-dir … --emit devblob --gpu h100 --arch sm_90a --max-ctx 8192`
(60 layers, KV 2.19 GiB). vLLM at `--gpu-memory-utilization 0.80`: 0.90 OOMs on
this box because ~5.5 GiB is already held by unrelated processes and the
capture phase overshoots.

| engine | t_listen | t_first |
|---|---|---|
| **plowrt, prefetch ON** | **13.35 s** | **13.38 s** |
| vLLM 0.26.0 | 171.68 s | 171.77 s |

**12.9×.** The shape is identical to 12B and the split is the same: vLLM spends
22.27 s on weights and then 78.86 s in "init engine (profile, create kv cache,
warmup model)", 35.95 s of it compilation — work plowc did ahead of time.

## 2. Weight upload alone (engine-internal timer)

plowrt's `checkpoint weights uploaded to GPU`; vLLM's `Model loading took`.

| state | engine | seconds | GiB/s |
|---|---|---|---|
| cold | plowrt, prefetch OFF | 12.4 | 1.80 |
| cold | plowrt, 16 threads | **5.8** | 3.80 |
| cold | plowrt, 32 threads | 5.8 | 3.84 |
| cold | vLLM | 9.67 | 2.36 |
| warm | plowrt, prefetch OFF | 6.0 | 3.70 |
| warm | plowrt, 16 threads | 5.8 | 3.80 |
| warm | plowrt, 32 threads | 5.9 | 3.76 |
| warm | vLLM | 6.02 | 3.79 |

At 31B (57.18 GiB), cold:

| engine | seconds | GiB/s |
|---|---|---|
| plowrt, prefetch OFF | 23.9 | 2.39 |
| **plowrt, 16 threads** | **12.6** | 4.55 |
| vLLM | 22.27 | 2.65 |

The finding that prompted the patch, and it holds at both sizes: **cold and
single-threaded, plowrt was LOSING this term to vLLM** — 12.4 s against 9.67 s
at 12B, 23.9 s against 22.27 s at 31B. With the pool it wins both: 5.8 s
(1.67×) and 12.6 s (1.77×). The pool's own speedup is 2.14× at 12B and 1.90×
at 31B.

Two more things fall out:

1. **With the pool, cold == warm** (5.8 s either way). The I/O is fully hidden;
   what is left is memcpy + DMA + per-tensor allocs.
2. **Warm, every engine lands on ~3.8 GiB/s** — plowrt 5.8, vLLM 6.02. That is
   nowhere near the link (this is NVLink-C2C) and nowhere near the 12.7 GiB/s a
   bare single-threaded memcpy of this file achieves (§3), so the warm floor is
   allocation and DMA submission, not bandwidth. Parallelising the *copy* side is
   the next lever; this patch only fixes the *fault* side.

## 3. Host read path in isolation (no GPU)

`$CLAUDE_JOB_DIR/tmp/loadbench.c` — mmap the real 22.28 GiB shard, either one
thread copying it out in 64 MiB chunks (what `UploadPipe` does) or N threads
issuing `MADV_POPULATE_READ` over disjoint spans first (what `Prefetcher` does),
caches dropped between runs.

| mode | cold total | cold GiB/s | warm total |
|---|---|---|---|
| serial, 1 thread | 7.17 s | 3.11 | 1.76 s |
| pool, 4 | 6.41 s | 3.47 | — |
| pool, 16 | 6.04 s | 3.69 | 1.54 s |
| pool, 32 | 5.88 s | 3.79 | 1.59 s |

The pool rows pay populate and copy sequentially; the loader overlaps them, so
the term that matters is populate alone — **4.45 s at 32 threads vs 7.17 s
serial**. Warm the pool is worth ~0.2 s, because warm the path is memcpy-bound
rather than fault-bound. This drive saturates near 16 readers, which is why the
default is 16 and why 32 buys nothing measurable end to end.

## 4. What was wrong

`exec/gpu.rs::UploadPipe` ran page faults, staging memcpy and DMA submission on
ONE thread. The double buffer overlapped the memcpy with the DMA but never
addressed the faults, and `Checkpoint::populate`'s own scaling table already said
one thread cannot exceed ~2.9 GiB/s here. `asset::checkpoint::Prefetcher` existed
for exactly this problem and was wired into `exec/amd.rs` only — the CUDA path
was never given it.

Not fixed by sharding across files: this checkpoint is a **single** 22.28 GiB
safetensors shard, so file-level parallelism has nothing to divide. The
parallelism has to come from populating spans *within* one mapping.

## 4b. Allocation cost is by BYTES, not by call count

Carving every tensor out of one allocation instead of asking the driver per
tensor does not make the load faster, which is worth recording so it is not
tried again:

| | allocations | alloc time |
|---|---|---|
| 12B, per-tensor | 737 | 1.97 s |
| 12B, one slab (25 GiB) | 1 | 1.92 s |
| 31B, one slab (59.8 GiB) | 1 | 4.45 s |

Two sizes, one rate: 13.0 and 13.4 GiB/s of *committed* memory. The driver is
building page tables, and batching the requests does not reduce how much it has
to build. What the slab does buy is memory — the per-allocation rounding waste
goes from 322 MiB to 21 MiB on a 12B model, which the co-residency planner
spends directly.

Killing the term needs the commit avoided rather than batched: VMM with lazy
mapping (what `VmmKv` already does for KV), or holding the slab across loads so
S1 model switches stop re-paying it.

## 5. Caveat

`--features cuda` did not compile on `main` at d8459bf — `hier_base` was added to
`DevProgram` without updating the two CUDA initializers. Fixed in the same commit
as the prefetch wiring; every number above is from the fixed build.
