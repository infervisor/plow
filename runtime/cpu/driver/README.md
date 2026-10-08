# Intel RDT cache pseudo-locking for the CPU engine

`pseudo_lock_sram.ko` partitions L2 and L3 with Intel CAT at load, then locks memory that
**plowrt owns** into those ways: a range is held in one core's private L2 (level 2) or in the
L3 slices of one SNC node (level 3). plowrt allocates the buffers itself (node-bound, 2 MiB
THP-backed) and drives everything through `/dev/pseudo_lock`; locks are released when the file
descriptor closes, so a crashed server cannot leak pinned memory.

This is the out-of-tree route. The running Amazon Linux kernel is built without
`CONFIG_X86_CPU_RESCTRL`, so the in-kernel resctrl pseudo-lock (and
`linux-resctrl-xeon-pseudo-lock.patch`, which adds Granite/Emerald/Sapphire Rapids to its
model list) needs a rebuilt kernel and a reboot; nothing here depends on it.

## Load

```bash
make
sudo insmod pseudo_lock_sram.ko            # l2_ways=14 l3_ways=14 (defaults)
sudo chgrp "$(id -g)" /dev/pseudo_lock     # the device is 0660 root
cat /sys/class/misc/pseudo_lock/caps       # masks and lockable bytes per core / node
cat /sys/class/misc/pseudo_lock/regions    # live locks and their last measurement
```

| param | default | meaning |
|---|---|---|
| `l2_ways` | 14 | L2 ways reserved for locking on every core (16-way L2: 14 = 1.75 MiB, 1.53 MiB lockable) |
| `l3_ways` | 14 | L3 ways reserved for locking (14 of 16 = 420 MiB socket, 122.5 MiB lockable per SNC node) |
| `l3_io_ways` | 0 | allow locked L3 ways to overlap the I/O-shared ways (CPUID 0x10: ways 14-15), e.g. `l3_ways=15`; DMA fills can then evict locked lines |

Lockable bytes are 7/8 of the reserved ways (set-conflict headroom). Every mask comes from CPUID
leaf 0x10; a CPU without L2 and L3 CAT refuses to load.

## Interface (`/dev/pseudo_lock`, ABI v3)

| ioctl | argument | effect |
|---|---|---|
| `PL_IOC_CAPS` (`'P'`, 10) | `struct pl_caps` out | masks, lockable bytes per core / per node |
| `PL_IOC_LOCK` (11) | `struct pl_lock_req` | pin `[addr, addr+len)`, load it from `cpu` in the lock CLOS, measure; returns an id |
| `PL_IOC_UNLOCK` (12) | id | unpin, return the budget |
| `PL_IOC_MEASURE` (13) | `struct pl_measure` | lines by calibrated L1/L2, L3, DRAM latency class, read from the owner |
| `PL_IOC_RELOAD` (14) | id | re-load and re-measure |
| `PL_IOC_WORKER` (15) | `{cpu, on}` | run `cpu` in the worker CLOS until the fd closes |

Level 3 ranges must be on the owner's node (`-EXDEV` otherwise); budgets are enforced per core
(L2) and per node (L3) (`-ENOSPC`).

## Classes of service

| CLOS | L3 mask | L2 mask | used by |
|---|---|---|---|
| 0 default | normal ways | normal ways | everything else |
| 1 L2 fill | normal ways | locked ways | `PL_IOC_LOCK` level 2 |
| 2 L3 fill | locked ways | normal ways | `PL_IOC_LOCK` level 3 |
| 3 worker | locked ways | normal ways | plowrt worker cpus |

## Why each piece exists (measured on Xeon 6975P-C, SNC3)

* **Non-inclusive L3.** A read hit moves the line into the reader's L2; its eviction refills L3
  through the *reader's* CLOS. Readers in CLOS 0 therefore unlock what they read: 122.5 MiB/node
  locked, re-read by 31 streaming threads in CLOS 0, fell to 30-47% held. The same threads in the
  worker CLOS kept 60-93% under load and 89-90% idle. Lock fills and measurements also
  `CLDEMOTE` each line so the last L2-resident lines land in the locked ways.
* **SNC.** A line is cached only in its home node's slices, so an L3 lock is per node, loaded
  from a cpu of that node, from pages on that node.
* **L2.** Each core's private L2 holds its own range: 100% of 1.53 MiB per core held through 31
  threads streaming 1 GiB each.
* **Idle states.** Core C6 flushes L2; a CPU-latency QoS request keeps cores out of deep idle
  while the module is loaded.

The v2 driver (one fixed vmalloc region per level) measured ~98% DRAM for both devices: its
L3-fill CLOS carried a full L2 mask, so the L3 preload on CPU 0 evicted CPU 0's locked L2; its
L3 region was not node-local; and it classified by a fixed 150-cycle threshold.

## Verify

```bash
cc -O2 -pthread sram_bench.c -o sram_bench
./sram_bench 10                                   # lock, stream 31 x 1 GiB, re-measure
READ_LOCKED=1 ./sram_bench 10                     # streamers also re-read the L3 locks (CLOS 0)
READ_LOCKED=1 WORKER_CLOS=1 ./sram_bench 10       # ... from the worker CLOS
```

plowrt uses it with `PLOW_CPU_SRAM=1` (`--cpu-sram`): worker scratch heads in L2, decode-hot
layer weights in L3, workers in the worker CLOS (`crates/plowrt/src/memory/sram.rs`).
