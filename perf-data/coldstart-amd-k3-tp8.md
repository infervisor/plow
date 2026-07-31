# Cold start on AMD — Kimi-K3 TP8, 8×MI355X (2026-07-31)

Companion to `coldstart-plow-vs-vllm-gh200.md`, which measured cold start on an
NVIDIA GH200 and found the prefetch pool worth 1.57×. **That framing does not
transfer.** On this box cold start is 17–18× warm, ~90% of it is spent waiting on
the drive, and both of the prefetch pool's tuning knobs are already at or past
their optimum. Recorded mainly so the two obvious fixes are not re-attempted.

Box: 8×MI355X (gfx950), ROCm 7.2.4, dual EPYC 9565 (2 NUMA nodes), **2.2 TiB RAM**,
checkpoint on NVMe (ext4 on LVM, 14 T). Bundle `/home/lava/models/k3_b1`
(`--num-gpus 8 --parallel tp`, ctx 32768), checkpoint `k3_farm` — 97 shards,
497 316 tensors, **1.5 TiB**. Binary `--release --features hsa`.

"cold" = `echo 3 > /proc/sys/vm/drop_caches` immediately before the run. This is
not optional here: 2.2 TiB of RAM against a 1.5 TiB checkpoint means an
unprepared run is fully cached and measures memcpy. Dropping 2.1 TiB of cache
itself takes ~33 s.

Per rank: 5408 named tensors (22.84 GiB) + 168.39 GiB of packed experts =
191.23 GiB.

## 1. Cold vs warm

| | cold | warm |
|---|---|---|
| total upload, slowest rank | **710–724 s** | **~40 s** |
| named tensors | 28–34 s | 6.0–6.7 s |
| packed experts | 677–691 s | 31–33 s |
| experts throughput | **0.25 GiB/s** | 5.2–5.4 GiB/s |

**17.75×.** Aggregate across 8 ranks the cold expert bind runs at ~2 GB/s, while
a plain sequential `dd` on this array does **9.7 GB/s**. The gap is the access
pattern: the expert bind issues **~494 592 separate ~357 KiB reads per rank**
(one per expert tensor slice, well under the 64 MiB staging chunk), and eight
ranks seek independently.

## 2. What the weight slab is worth cold — about 1%

`PLOW_WEIGHT_SLAB=1` vs `=0`, both cold, both with prefetch at defaults:

| | slab | per-tensor |
|---|---|---|
| named-tensor `alloc_ms` | **13 ms** | 7560 ms |
| named-tensor wall | 28.2 s | 33.4 s |
| packed-expert wall | 683.1 s | 695.0 s |
| **total** | **710.0 s** | 726.9 s |

The ~7.5 s allocation saving is identical warm or cold — it is driver time and
does not care about the page cache. What changes is the denominator: **~19% of a
40 s warm load, ~1% of a 710 s cold one.** The 11.9 s of expert-phase difference
is fault variance between runs, not the slab, and is not claimed. See
`weight-slab-amd-mi355x.md` for the warm case, which is what that change is for.

## 3. Both prefetch knobs are dead ends

### 3a. Depth does nothing — 128× buys 2%

`PLOW_PREFETCH` is denominated in TENSORS and defaults to 256. Its doc comment
sizes that against "a ~1.4 MiB expert projection", which is the **tp=1** figure;
under TP8 each rank's read is 357 KiB, so the real lookahead is 89 MiB — about a
third of a second of work. That looked like the bug. It is not:

| `PLOW_PREFETCH` | lookahead | total | experts GiB/s |
|---|---|---|---|
| **256 (default)** | 89 MiB | 712.7 s | 0.25 |
| 2048 | 714 MiB | 695.3 s | 0.25 |
| 8192 | 2.8 GiB | 694.0 s | 0.25 |
| 32768 | 11.2 GiB | 714.1 s | 0.24 |

Flat inside run-to-run noise, deepest arm second-worst.

**Why: depth is not concurrency.** `Prefetcher`'s workers each call a *blocking*
`madvise(MADV_POPULATE_READ)` on one span (`checkpoint.rs::populate` is a single
`advise_range`). The number of I/Os in flight is therefore exactly `threads`;
depth only lengthens the queue those same threads drain and cannot change the
number of outstanding requests by one.

*(The tp=1 mis-sizing is still a latent flaw — a depth in TENSORS silently
mis-sizes whenever `tp` changes bytes-per-tensor, and a depth in BYTES would not.
It is simply not what costs the 600 s.)*

### 3b. Concurrency is worse than a no-op — 8× threads is 48% slower

| threads/rank | in flight (8 ranks) | expert bind | total |
|---|---|---|---|
| **16 (default)** | 128 reads, ~45 MiB | **690.9 s** | **723.9 s** |
| 128 | 1024 reads, ~360 MiB | 1035.1 s | 1071.9 s |

`checkpoint.rs`'s "the measured knee is ~16" holds for this pattern too. The pool
is already sized correctly, and whatever caps the load at ~2 GB/s aggregate is
not the prefetcher. **The remaining lever is the read pattern** — coalescing
~500k small scattered reads into large sequential ones, and/or having ranks
cooperate on shared reads rather than each seeking independently.

## 4. `PLOW_LOAD_PROFILE=1` relabels the wait, it does not add it

Worth stating because it nearly invalidated everything above. The flag is not
passive: `prefault()` (`exec/amd.rs:1816`, `:3117`) runs a serial,
single-threaded pass touching one byte per page of each span, inline on the
loading thread, *ahead of* the copy — and its own comment warns it is "a second
pass over the source ... this is the load path, not a benchmark". Cold, that
could plausibly have manufactured the entire fault term. Measured both ways, on
the expert bind:

| | wall | fault | gather | memcpy |
|---|---|---|---|---|
| profile ON | 677.3 s | **610.6 s** | 11.0 s | 92.9 s |
| profile OFF | 690.9 s | — | **646.5 s** | 66.4 s |

Same total within noise. With profiling on the disk wait is charged to `fault`;
with it off the identical wait is charged to `gather`, because `slice_for` is
then the first thing to touch the page. So the numbers in §1–§3 are sound, and
"90% of cold start is waiting on the drive" is a statement about the drive, not
about the instrument.

## 5. Reproducing

```bash
# any arm: drop caches immediately before, or the run is warm and meaningless
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
perf-data/harness/gpulease -n 8 cold sg render -c \
  "PLOW_WEIGHT_SLAB=1 PLOW_PREFETCH=256 PLOW_PREFETCH_THREADS=16 \
   nix develop --command ./target/release/plowrt amd-bench \
     --blob /home/lava/models/k3_b1/model.pkt \
     --hsaco /home/lava/models/k3_b1/hsaco \
     --checkpoint /home/lava/models/k3_b1/checkpoint \
     --tp 8 --steps 2 --ctx 2048"
```

Read the slowest rank's `checkpoint weights uploaded ... secs=`. Add
`PLOW_LOAD_PROFILE=1` for the phase split, knowing §4.

Caveat on this bundle: its `hsaco` has no `PLOW_K3` flash object, so flash
segments fall back to the 8-wave interpreter. Orthogonal to the loader — it
changes what runs after the weights are bound — but it is why the logs carry a
wall of `no flash object` lines.
