# Weight slab on AMD — MI355X, Kimi-K3 TP8 (2026-07-31)

Companion to `coldstart-plow-vs-vllm-gh200.md`, which measured the same change on
an NVIDIA GH200. The conclusion there does **not** carry over, in the direction
that matters: on CUDA the slab is a memory win and not a speed win; on ROCr it is
both, and the speed half is large.

Box: 8×AMD Instinct MI355X (gfx950, 288 GiB each), ROCm 7.2.4. Bundle
`/home/lava/models/k3_b1` (Kimi-K3, `--num-gpus 8 --parallel tp`, ctx 32768),
checkpoint `k3_farm` (97 shards, 497 316 tensors). Binary built
`--release --features hsa`. Both arms are the same binary and the same run of
`plowrt amd-bench --tp 8`, differing only in `PLOW_WEIGHT_SLAB`.

Per rank: 5408 carved tensors, 3 views (the TP peer slots), 22.84 GiB of named
weights, then 168.39 GiB of packed experts.

## 1. The two arms

| | slab (default) | per-tensor (`PLOW_WEIGHT_SLAB=0`) |
|---|---|---|
| peak VRAM per card | **204 579 MiB** | 205 904 MiB |
| peak VRAM, 8 cards | **1 636 638 MiB** | 1 647 236 MiB |
| `alloc_ms`, named tensors | **96–266** | 6410–8802 |
| wall, named tensors | **6.01–6.65 s** | 12.96–16.72 s |
| `alloc_ms`, packed experts | 1491–2144 | 1491–1830 |

**Memory: 1325 MiB per card, 10 598 MiB (10.35 GiB) across the eight.**
**Time: ~7–8.5 s per rank, which halves the named-tensor phase.**

The packed-expert phase is unchanged, as it should be: `bind_packed_experts`
already carves all of a layer's experts out of two allocations, so it never paid
the per-tensor term this removes.

## 1b. Cold start — the same saving, a twentieth of the proportion

Everything in §1 is **warm**: this box has 2.2 TiB of RAM against a 1.5 TiB
checkpoint, so an unprepared run measures memcpy out of page cache. With
`echo 3 > /proc/sys/vm/drop_caches` immediately before each arm:

| | slab | per-tensor | delta |
|---|---|---|---|
| named-tensor `alloc_ms` | **13 ms** | 7560 ms | **−7.5 s** |
| named-tensor wall | 28.2 s | 33.4 s | −5.2 s |
| packed-expert wall | 683.1 s | 695.0 s | −11.9 s |
| packed-expert `fault_ms` | 617.1 s | 646.0 s | (noise) |
| **total per rank** | **710.0 s** | 726.9 s | −16.9 s |

The allocation saving is the same ~7.5 s either way — it is driver time and does
not care about the page cache. What changes is what it is a fraction OF: **~19%
of a 40 s warm load, ~1% of a 710 s cold one.** The 11.9 s of the expert-phase
difference is fault variance between runs (617 vs 646 s), not the slab.

**So: this change is a warm-start optimisation.** Cold start is 17.75× warm and
90% of it is page-fault stall (638 s of 710 s), which nothing here touches. The
cold bottleneck is the packed-expert bind issuing ~494 592 separate ~357 KiB
reads per rank — 168.39 GiB at **0.25 GiB/s per rank**, ~2 GB/s aggregate,
against **9.7 GB/s** for a sequential `dd` on this array. That is a read-pattern
problem, and the first thing to try is the prefetch depth: `prefetch_depth()`
defaults to 256 TENSORS and its doc comment sizes that against a "~1.4 MiB expert
projection", which is the **tp=1** figure. Under TP8 the per-rank read is
357 KiB, so the real lookahead is 89 MiB — about a third of a second of work.
See the design notes §0.

## 2. Why the memory moves — ROCr's rounding

`hsa_amd_memory_pool_allocate` on the device's coarse-grained pool reports
`RUNTIME_ALLOC_GRANULE` = 4096 and `REC_GRANULE` = 2 MiB, and then rounds far
harder than either. Measured directly (one process per size, so a previous
probe's deferred frees cannot contaminate the delta):

| request | committed | waste |
|---|---|---|
| 1 KiB | 32 KiB | 3100% |
| 4 KiB | 32 KiB | 700% |
| 16 KiB | 32 KiB | 100% |
| 64 KiB | 64 KiB | 0% |
| 1 MiB | 1 MiB | 0% |
| **1 468 006 B** (expert projection) | **2 MiB** | **42.9%** |
| 1.5 MiB | 2 MiB | 33.3% |
| 2 MiB + 1 B | 4 MiB | 100% |
| 3 MiB | 4 MiB | 33.3% |
| 5 MiB | 6 MiB | 20% |
| 30 MiB + 4 KiB | 32 MiB | 6.7% |
| 100 MiB + 4 KiB | 102 MiB | 2.0% |

The rule: **under 2 MiB, round up to the next power of two with a 32 KiB floor;
at or above 2 MiB, round up to a 2 MiB multiple.** Sub-2 MiB requests are
suballocated out of 2 MiB blocks — 64 × 1 KiB consumes exactly 2 MiB.

CUDA's equivalent waste on a 12B model was 322 MiB → 21 MiB. ROCr's is worse
because of the power-of-two step, and MoE checkpoints sit right on it: a 1.4 MiB
expert projection is the single most common tensor size in this model and it
loses 42.9% every time.

## 3. Why the TIME moves — and why a microbenchmark says it does not

This is the part worth recording, because the obvious experiment gives the wrong
answer and it was believed for most of a day.

Allocating 737 uniform 30 MiB buffers on an **idle** MI355X costs 8.8 ms total
(11.9 µs each) against 0.395 ms for one allocation of the same bytes. Read alone,
that says ROCr barely charges per call — three orders of magnitude cheaper than
the ~2.0 s CUDA pays for 737 `cuMemAlloc`s — and therefore that the slab can only
be a memory optimisation on AMD. That was the first conclusion here and it was
wrong.

The real load is not that shape. It carves 5408 unevenly sized tensors, most of
them small, interleaved with 168 GiB of expert buffers, with eight ranks issuing
against one driver concurrently. Under those conditions the per-call cost rises
to ~1.3–1.6 ms amortised (6410–8802 ms over 5408 tensors), and the slab removes
essentially all of it.

**Do not re-derive this from a standalone allocation benchmark.** An idle card
with uniform large allocations is the one configuration where the term vanishes.

## 4. Correctness

`amd-bench --tp 8` checks that every rank emits the same token ids each step —
the ranks hold a replicated residual and a full-vocab lm_head, so a carve that
overlapped two tensors or handed a rank an address inside its neighbour's span
shows up as diverging streams rather than silently. Both arms completed with
exit 0 and no divergence.

Structurally, the carve is guarded by a `debug_assert_eq!(slab_off, slab_bytes)`
after the loop: the sizing pass and the carve walk the same list with the same
filter, so the cursor must land exactly on the total. Short wastes the tail; long
means two tensors were aliased and the weights are quietly wrong.

The AMD filter has two exclusions CUDA has no analogue of, and `views=3` in the
log is the check that they are right:

* **TP peer slots** (`act.og_tp`/`act.dg_tp`/`act.ug_tp`) — storage owned by the
  `TpRank` peer region, which `XReduce` reads over XGMI. Carving these out of
  local VRAM would have every rank reduce slots its peers never wrote.
* **full-layer KV under VMM** — the pool's VA reservation, mapped lazily at the
  per-sequence frontier. (Off in this run: `vmm=false`.)

## 5. Reproducing

```bash
perf-data/harness/gpulease -n 8 slabab sg render -c \
  "PLOW_WEIGHT_SLAB=1 PLOW_LOAD_PROFILE=1 nix develop --command \
     ./target/release/plowrt amd-bench \
       --blob /home/lava/models/k3_b1/model.pkt \
       --hsaco /home/lava/models/k3_b1/hsaco \
       --checkpoint /home/lava/models/k3_b1/checkpoint \
       --tp 8 --steps 4 --ctx 2048"
```

`PLOW_WEIGHT_SLAB=0` for the other arm. Peak VRAM read by sampling
`rocm-smi --showmeminfo vram --csv` at 2 Hz across the run; the GPUs were leased
and otherwise idle, so per-card total-used is the rank's own footprint.

Caveat on this bundle: its `hsaco` lacks a `PLOW_K3` flash object, so flash
segments fall back to the 8-wave interpreter. That is orthogonal to the loader —
it changes what runs after the weights are bound, not how they are allocated —
but it is why the log carries a wall of `no flash object` lines.
