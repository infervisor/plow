# P0: topology, isolation and baseline (Xeon 6975P-C, BF16, single socket)

Checklist phase P0 of the L2-resident BF16 experiment. Raw evidence: `/tmp/g4c/l2r/results/p0*`
(`inventory.txt`, `lscpu_e.txt`, `meta.json`, `p0_bw/{tiers.jsonl,imc.csv,turbostat.mode*.txt}`,
`p0_serve/`). Benches: `runtime/cpu/bench/l2r/`. Git `fef89819` (main after PR #63).

## P0.1 Inventory

| item | value |
|---|---|
| host | AWS c8i.metal-48xl class, Intel Xeon 6975P-C, family 6 model 173 stepping 1, microcode 0x1000434 |
| kernel | 6.18.51-120.163.amzn2023.x86_64, `CONFIG_HZ=100`, no `CONFIG_X86_CPU_RESCTRL` (no resctrl fs) |
| cores | 1 socket, 96 physical cores, SMT on (sibling = cpu + 96) |
| NUMA | SNC3, 3 nodes: node0 cpus 0-31 (+96-127), node1 32-63 (+128-159), node2 64-95 (+160-191); ~126 GiB each |
| caches | L1d 48 KiB 12-way, L2 2 MiB 16-way private per core (2048 sets), L3 480 MiB 16-way shared, non-inclusive |
| ISA | `amx_tile amx_bf16 amx_int8 avx512_bf16 avx512_vnni cldemote movdir64b serialize waitpkg` |
| RDT | `cat_l2 cat_l3 cdp_l2 cdp_l3 mba cqm cqm_llc cqm_mbm_total rdt_a` |
| CAT control | out-of-tree `pseudo_lock_sram` (ABI v3, `/dev/pseudo_lock`), loaded `l2_ways=12 l3_ways=0`: L2 lock cbm `0xfff`, 1,376,256 B/core lockable (7/8 of 12 ways); no L3 lock |
| power / freq | `intel_pstate` active, governor `performance`, turbo on; `numa_balancing=0`; THP `madvise`; no hugetlb pages |
| tools | perf 6.1 (kernel 6.18), turbostat, cpupower, taskset; numactl/numastat via `nix develop`; no pqos/msr-tools (MSRs read through `/dev/cpu/N/msr` as root) |
| uncore PMUs | `uncore_imc_0..11` (`cas_count_{read,write}_sch{0,1}`), `uncore_cha_*`, `uncore_b2cmi_*` |

Prefetch control: the driver writes `MSR_MISC_FEATURE_CONTROL` (0x1A4) = 0xF (bits 0-3: L2 streamer, L2
adjacent line, DCU, DCU IP) only while it fills or measures a region, with interrupts off on the owner
core, and restores the saved value (0x0 on every core sampled). No other prefetch MSR is written.

## P0.2 Isolation (reboot-free, software only)

No reboot (`isolcpus` / `nohz_full` are boot parameters). `runtime/cpu/bench/l2r/isolate.sh apply|restore|status`:

| item | setting |
|---|---|
| housekeeping | 2 physical cores per SNC node + SMT siblings: 0,1,32,33,64,65 (+96,97,128,129,160,161) |
| inference workers | 90 physical cores 2-31, 34-63, 66-95, one worker per core; SMT siblings left idle |
| device IRQs | irqbalance stopped; 161 of 165 IRQs moved to housekeeping cpus (4 managed per-cpu IRQs refuse) |
| kernel work | workqueue cpumask = housekeeping cpus |
| processes | `system.slice`, `init.scope` AllowedCPUs = housekeeping; `user.slice` not restricted (other sessions); benchmark processes pinned with `taskset` / `sched_setaffinity` |
| idle | `/dev/cpu_dma_latency` = 0 held (no deep C-states: C6 would flush L2) |

Measured on an idle inference core over 10 s: device IRQs 0/s; local timer (LOC) 100/s (`HZ=100`, only
`nohz_full` removes it); CAL ~2/s, TLB < 1/s. Residual noise is therefore one 100 Hz tick per core.

Under load (`perf stat -C <inference cpus>`):

* 20 s AMX-BF16 L2 stream on all 90 cores: 3,241 context switches (1.8/s per cpu), 180 migrations
  (benchmark start-up), 180,633 local-timer interrupts (100/s per cpu), 225 page faults. Stream held
  138.85 GB/s per core (min 138.23).
* One 256-token E4B request on a running `plowrt serve`: ~270,000 context switches on the inference cpus.
  Per-thread `/proc/<pid>/task/*/status` deltas attribute them to plowrt's own workers: 90 `plow-cpu-*`
  threads 615,725 voluntary / 3,395 nonvoluntary. Workers spin `PLOW_CPU_SPIN_US` (2 ms), yield, then
  `park_timeout(200 µs)`; an idle server costs ~8,000 switches/s per worker cpu (7.16 M in 10 s). Preemption
  by foreign tasks is ~6.6/s per worker. Parking does not cost L2: with `cpu_dma_latency=0` an idle core
  stays in POLL and never reaches C6.

NUMA placement (`numastat -p`, E4B serve, `PLOW_CPU_WEIGHT_AFFINE=1`): private 2,997 / 2,975 / 3,045 MB
on nodes 0 / 1 / 2 (weights split per node), host heap 251 MB on node 2. **No weight page is huge-page
backed** (`AnonHugePages` 0): with more than one NUMA node plowrt defaults to `MADV_NOHUGEPAGE`
(`PLOW_CPU_HUGE_PAGES` unset → `nodes <= 1`) because THP allocation may fall back to another node. P0.3
shows 4 KiB pages lose up to 25% of L2 capacity to set conflicts, so P1/P5 must run with
`PLOW_CPU_HUGE_PAGES=1` and re-verify per-node placement.

### Host-thread placement (found while reproducing the source baseline)

Restricting plowrt to exactly the worker cpus costs **30% decode time**. Topology pins one worker per
physical core; the engine (`plow-eng-*`), tokio and encode threads share the same cpuset, so with no spare
logical cpu the engine thread time-slices against a spinning worker on every token. E2B, ISL 1000, OSL 128,
c1, 2 reps each, TPOT p50:

| cell | packet | plowrt cpuset | TPOT p50 ms |
|---|---|---|---:|
| source (`final/`, 59a36e3d) | n_cu 96 / 2K | unrestricted | 10.65 |
| g | n_cu 96 / 2K, current plowc | unrestricted | 10.62 |
| l | n_cu 96 / 2K | `0-95` (no siblings) | 13.51 |
| j | n_cu 90 / 2K | `0-29,32-61,64-93` | 13.86 |
| h | n_cu 90 / 2K | `2-31,34-63,66-95` (isolated, no siblings) | 13.83 |
| **k** | n_cu 90 / 2K | isolated cores **+ their SMT siblings** | **10.47** |

`perf record` on the 90 workers (h): `gemv_rows` 48-54% and the worker wait loop 37-40% on every worker, no
straggler; with n_cu 96 unrestricted the wait loop is 24.5%. The plowc target (`rtx6000pro`/`sm_120a` as in
the production recipe vs `xeon6975p`/`amx`) and the plowrt build make no difference (≤ 0.5%).

Adopted for every following phase: workers on the 90 isolated cores, the inference cores' SMT siblings
added to plowrt's cpuset so the host threads have somewhere to run
(`taskset -c 2-31,34-63,66-95,98-127,130-159,162-191`). The siblings share the worker cores' L2; under a
CAT partition their CLOS must exclude the locked ways (P1). A housekeeping-core affinity for plowrt's
non-worker threads is a P5 item.

## P0.3 Bandwidth tiers on the isolated worker set (90 cores, 3 reps each)

`l2r_bw` (`runtime/cpu/bench/l2r`), buffers allocated after pinning (node-local), 2 s per rep. Totals in GB/s
of buffer bytes consumed; rep spread < 2% everywhere.

| tier | bytes/core | pages | AVX-512 read | AMX-BF16 stream | AVX-512 BF16 GEMV | slowest core (AMX) |
|---|---:|---|---:|---:|---:|---:|
| L2 | 1.0 MiB | 4 KiB | 15,010 | 12,247 | 8,796 | 103.2 (cpu 7) |
| L2 | 1.3 MiB | 4 KiB | 13,395 | 11,274 | 8,110 | 98.3 |
| L2 | 1.5 MiB | 4 KiB | 10,780 | 9,359 | 7,093 | 87.4 |
| L2 | 1.0 MiB | THP | 15,445 | 12,496 | 9,102 | 138.6 |
| L2 | 1.3 MiB | THP | 15,396 | 12,508 | 9,103 | 138.8 |
| L2 | 1.5 MiB | THP | 15,400 | 12,521 | 9,106 | 139.0 |
| L3 | 3 MiB | THP | 1,361 | 1,358 | 1,376 | 10.9 |
| L3 | 4 MiB | THP | 1,304 | 1,300 | 1,318 | 10.4 |
| DRAM | 64 MiB | THP | 669 | 657 | 607 | 7.2 |

* **Page size decides L2 residency.** With 4 KiB pages the physical page colouring maps a 1.0-1.5 MiB
  buffer unevenly over the 2,048 L2 sets: conflict misses appear already at 1.0 MiB on some cores and
  the 1.5 MiB mean falls 25%. With THP (2 MiB pages cover all sets uniformly) every core holds 1.5 MiB at
  full rate (slowest/fastest within 0.2%). Weight slices must be huge-page backed.
* DRAM: IMC CAS counters during the 64 MiB read: 612 GB/s read at the controllers (client 626 GB/s),
  writes 0.5%.
* Per-core L2 rates: AVX-512 read 171 GB/s, AMX-BF16 B-tile stream 139 GB/s, AVX-512 VDPBF16PS GEMV
  101 GB/s. The batch-1 VDPBF16PS stream consumes L2 at ~29 B/cycle (0.45 instructions/cycle at
  3.5 GHz) against 1 instruction/cycle compute-only: open item for P2 (load-port / uop limit).

### Sustained frequency (10 s, 90 workers, turbostat; mean worker Bzy_MHz)

| load | MHz (p5 / min) | package W | DRAM W | core temp max |
|---|---:|---:|---:|---:|
| AVX-512 BF16 compute only (VDPBF16PS) | 3,800 (3,800 / 3,784) | 602 | 19.9 | 89 °C |
| AVX-512 BF16 GEMV stream, 1 MiB L2 | 3,502 (3,498 / 3,497) | 601 | 19.7 | 88 °C |
| AMX-BF16 compute only (TDPBF16PS) | 3,197 (3,187 / 3,124) | 570 | 19.8 | 86 °C |
| AMX-BF16 stream, 1 MiB L2 | 2,901 (2,901 / 2,901) | 581 | 19.8 | 86 °C |

Throughput: AMX-BF16 3.27 Tflop/s per core (294 Tflop/s socket), AVX-512 BF16 218 Gflop/s per core. No
throttling or frequency collapse; every load holds its license frequency for the whole run.

## P0.3 Serving baseline (E2B / E4B, isolated set)

plowrt `fef89819` CPU build, 90 workers on the inference cores (`PLOW_CPU_THREADS=90`, cpuset = inference
cores + SMT siblings, see host-thread placement above), n_cu 90 packets (`plowc --gpu xeon6975p --arch amx
--n-cu 90 --max-ctx 16384`, ladder 1-32, `PLOW_MAX_CHUNK=2048`; E4B `PLOW_DENSE_PF_NS_MIN=2`),
`PLOW_CPU_WEIGHT_AFFINE=1`, `PLOW_SESSION_SLACK=32`, greedy, OSL 128, 3 reps per cell, `llm_grid.sh`
(vLLM bench client). TTFT/TPOT percentiles pooled over the 3 reps; spread = (max - min) / mean of the
per-rep value.

| model | ISL | conc | reps | TTFT p50 / p95 / p99 ms | TPOT p50 / p95 / p99 ms | out tok/s (spread) | median TPOT spread | failed |
|---|---:|---:|---:|---|---|---|---:|---:|
| E2B | 1900 | 1 | 3 | 261 / 284 / 315 | 10.62 / 10.64 / 10.65 | 79 (0.4%) | 0.2% | 0 |
| E2B | 1900 | 4 | 3 | 537 / 1,074 / 1,093 | 17.05 / 19.12 / 19.14 | 188 (0.6%) | 6.1% | 0 |
| E2B | 1900 | 16 | 3 | 757 / 4,052 / 4,479 | 50.47 / 53.80 / 54.06 | 285 (0.3%) | 4.4% | 0 |
| E2B | 15900 | 1 | 3 | 1,672 / 1,725 / 1,813 | 11.60 / 11.63 / 11.69 | 41 (1.0%) | 0.1% | 0 |
| E2B | 15900 | 4 | 3 | 2,620 / 6,798 / 6,871 | 50.84 / 58.06 / 58.71 | 56 (0.4%) | 0.2% | 0 |
| E2B | 15900 | 16 | 3 | 3,488 / 24,547 / 28,120 | 248.73 / 257.13 / 261.28 | 58 (0.1%) | 2.1% | 0 |
| E4B | 1900 | 1 | 3 | 395 / 415 / 453 | 19.38 / 19.40 / 19.41 | 45 (0.3%) | 0.1% | 0 |
| E4B | 1900 | 4 | 3 | 1,044 / 1,668 / 1,668 | 25.14 / 30.23 / 31.52 | 120 (3.1%) | 0.2% | 0 |
| E4B | 1900 | 16 | 3 | 2,651 / 5,370 / 5,665 | 63.16 / 79.55 / 80.48 | 191 (1.3%) | 0.3% | 0 |
| E4B | 15900 | 1 | 3 | 3,164 / 3,221 / 3,412 | 20.44 / 20.50 / 20.66 | 22 (0.8%) | 0.0% | 0 |
| E4B | 15900 | 4 | 3 | 4,955 / 12,808 / 12,969 | 93.59 / 108.06 / 108.54 | 30 (0.6%) | 1.0% | 0 |
| E4B | 15900 | 16 | 3 | 4,998 / 46,702 / 52,263 | 436.31 / 453.88 / 454.91 | 33 (0.6%) | 1.0% | 0 |

Against the source campaign (`gemma4-xeon6-bf16-20261006`, n_cu 96, unrestricted cpuset, ISL 1000): c1 TPOT
p99 E2B 10.69 ms / E4B 19.22 ms there vs 10.65 / 19.41 ms here at ISL 1900 on 90 cores. The first P0 pass,
with plowrt confined to the 90 worker cpus, measured 14.23 / 22.62 ms (kept as `p0_serve.nosmt`); the
host-thread placement above accounts for all of it.

Correctness: FP32-reference gate (`campaign.py gate --only llm_fp32_ref`, same prompt set and vLLM capture
as the source campaign) on these exact packets, plowrt on the isolated cores:

| model | packet | kl_mean (vLLM 0.30) | kl_p99 (vLLM) | top1_decisive | needle_acc | gate |
|---|---|---:|---:|---:|---:|---|
| E2B | `312766240c72` | 0.000431 (0.000512) | 0.00591 (0.00555) | 1 | 1 | PASS |
| E4B | `634f11dc0869` | 0.000673 (0.001404) | 0.00538 (0.00816) | 1 | 1 | PASS |

## Exit gate P0

| criterion | result |
|---|---|
| reproducible topology and affinity | `isolate.sh apply` / `status` scripted; one worker per physical core, verified per thread (`Cpus_allowed_list`) |
| repeat variance < 5% | output tok/s spread ≤ 3.1% in all 12 cells; median TPOT spread < 5% in 11 of 12. E2B 2K c4 6.1% (per-rep median 17.04 / 17.05 / 18.10 ms; TPOT p99 spread 0.2%, tok/s 0.6%): closed-loop c4 phase alignment of prefill chunks against decode steps, not noise; same cell 2.6% in the first pass |
| no unexplained migrations | 180 migrations in a 20 s stream are benchmark start-up; serving switches are plowrt worker parking (attributed per thread) |
| no frequency collapse | AVX-512 3.50-3.80 GHz, AMX 2.90-3.20 GHz held for 10 s at 570-602 W |
| correctness | FP32-reference gate PASS on both packets |

**P0 exit: PASS.** Carried into P1: huge-page-backed weights (`PLOW_CPU_HUGE_PAGES=1`), plowrt cpuset with the
inference cores' SMT siblings, the 100 Hz tick as the residual per-core noise.
