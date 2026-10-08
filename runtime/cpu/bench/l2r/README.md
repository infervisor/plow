# L2-resident BF16 stage benches (Xeon 6)

Microbenchmarks for the single-socket L2-residency experiment (`plans/xeon-l2-resident-inference.md`,
checklist phases P0-P5). Raw output goes to campaign scratch, never into the repo.

| file | what | build |
|---|---|---|
| `l2r_bw.c` | per-core L2 / L3 / DRAM bandwidth and sustained BF16 compute on an explicit worker set (`L2R_CPUS`, default the 90 isolated inference cores 2-31,34-63,66-95); modes 0 AVX-512 read, 1 AMX-BF16 stream, 2 AVX-512 BF16 GEMV, 3/4 compute only; `L2R_HUGE=1` backs buffers with THP | `cc -O3 -march=sapphirerapids -mamx-tile -mamx-bf16 -mavx512bf16 -pthread l2r_bw.c -o l2r_bw` |
| `p0_bw.sh` | P0.3 tier sweep (1.0/1.3/1.5/3/4/64 MiB per core, 4 KiB and THP, 3 reps), IMC CAS counters, turbostat frequency traces | `p0_bw.sh <outdir>` |
| `isolate.sh` | reboot-free P0.2 isolation: IRQs, workqueue cpumask, `system.slice`/`init.scope` AllowedCPUs onto 2 housekeeping cores + siblings per SNC node; saves and restores the original state | `sudo -v; isolate.sh apply\|restore\|status` |
| `l2r_lock.c` | P1 residency probe: per-core weight slice streamed by AMX or AVX-512, optionally locked in L2 through `/dev/pseudo_lock` (`L2R_LOCK=1`, MEASURE before/after), under interference scenarios A-E (none, node slot broadcast, DRAM KV stream, LLC KV + housekeeping DRAM streamers, long run with q block); per-step p50/p99/max | `cc -O3 -march=sapphirerapids -mamx-tile -mamx-bf16 -mavx512bf16 -pthread l2r_lock.c -o l2r_lock` |
| `p1_matrix.sh` | P1.2 capacity x interference matrix over L2 lock way counts (reloads the driver only when no region is live, restores its parameters); `P1_LONG=1` adds 10 / 30 min runs | `p1_matrix.sh <outdir> [ways...]` |
| `stage.c` | synthetic INT8-shaped E4B layer (4 GEMV phases + dissemination barriers; the plan's 23.4 µs figure); thread i on cpu i | `cc -O3 -march=sapphirerapids -pthread stage.c -o stage` |
| `sync.c` | one-way core-to-core latency and 2,560-byte vector exchange (dissemination, per-SNC hierarchical, flag-in-data, UMWAIT, CLDEMOTE) | `cc -O2 -march=sapphirerapids -mwaitpkg -mcldemote -pthread [-DNT=32] sync.c -o sync` |
| `catinfo.c` | CPUID 0x10 CAT capability widths | `cc -O2 catinfo.c -o catinfo` |
| `iso_demo.c` | isolated-core mailbox round trip with involuntary-preemption count | `cc -O2 -pthread iso_demo.c -o iso_demo` |
| `project.py` | the plan's per-layer latency projection model | `python3 project.py` |

L2 locking uses the out-of-tree driver in `runtime/cpu/driver/` (`/dev/pseudo_lock`). Reloading it
with other way counts affects every process on the host; check `/sys/class/misc/pseudo_lock/caps`
and `regions` first.
