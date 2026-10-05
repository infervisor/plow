# Intel RDT Cache Pseudo-Locking SRAM Driver

This directory contains the out-of-tree Linux kernel driver (`pseudo_lock_sram.ko`) that transforms Intel L2 and L3 CPU caches into deterministic, zero-eviction on-chip SRAM for **Intel Xeon Scalable Processors** (Sapphire Rapids, Emerald Rapids, and Granite Rapids).

---

## 1. Supported Architectures & Hardware Capabilities

| Architecture | Model Code (VFM) | L2 Cache / Core | L2 Max SRAM | L3 Cache / Socket | L3 Max SRAM |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Granite Rapids (Xeon 6th Gen P)** | `0xAD` (`INTEL_GRANITERAPIDS_X`) | 2.0 MiB (16 ways) | **1.75 MiB (14 ways)** | 480 MiB (16 ways) | **420 MiB (14 ways)** |
| **Emerald Rapids (Xeon 5th Gen)** | `0xCF` (`INTEL_EMERALDRAPIDS_X`) | 2.0 MiB (16 ways) | **1.75 MiB (14 ways)** | ~300 MiB (15–20 ways) | ~260 MiB |
| **Sapphire Rapids (Xeon 4th Gen)** | `0x8F` (`INTEL_SAPPHIRERAPIDS_X`) | 2.0 MiB (16 ways) | **1.75 MiB (14 ways)** | ~105 MiB (15 ways) | ~90 MiB |

---

## 2. Compilation Instructions

### Prerequisites
Ensure kernel build headers are installed:
```bash
# On Amazon Linux 2023 / Fedora / RHEL
sudo dnf install -y kernel-devel-$(uname -r) gcc make
```

### Build the Kernel Module
```bash
cd /home/ec2-user/plow/runtime/cpu/driver
make clean
make
```
This produces `pseudo_lock_sram.ko`.

---

## 3. Loading the Driver with High-Capacity SRAM Limits

By default, Intel CAT requires CLOS 0 (OS / default) to retain at least 1 bit in its Capacity Bitmask (CBM).
The driver supports dynamic parameters to push L2 and L3 reservations to maximum limits:

### Recommended Maximum Safe Configuration (14 Ways L2 + 14 Ways L3):
```bash
# Unload previous module instance if present
sudo rmmod pseudo_lock_sram 2>/dev/null || true

# Load with 1.75 MiB L2 SRAM per core and 420 MiB L3 SRAM across socket
sudo insmod pseudo_lock_sram.ko \
    l2_ways=14 \
    l3_ways=14 \
    l3_size_mb=420
```

### Module Parameters:
- `l2_target_cpu`: Target logical CPU for core-private L2 SRAM (default: `0`).
- `l2_ways`: Number of L2 ways to lock into SRAM (`1..15`, default: `14` = 1,792 KiB / 1.75 MiB).
- `l3_target_cpu`: Target CPU for socket-shared L3 SRAM (default: `0`).
- `l3_ways`: Number of L3 ways to lock into SRAM (`1..15`, default: `1` to `14`, 1 way = 30 MB on Granite Rapids).
- `l3_size_mb`: Memory size in megabytes for L3 SRAM allocation (e.g. `30`, `60`, `120`, `240`, `420`, `450`).

---

## 4. Character Device Interfaces

Upon loading, the driver exposes world-accessible (`0666`) character devices:
- `/dev/pseudo_lock_l2` — Core-local private L2 SRAM (1.75 MiB).
- `/dev/pseudo_lock_l3` — Socket-wide shared L3 SRAM (up to 420 MiB).

### Userspace Access & Memory Mapping:
```c
#include <fcntl.h>
#include <sys/mman.h>

// Memory-map the 1.75 MB L2 SRAM buffer
int fd = open("/dev/pseudo_lock_l2", O_RDWR);
void *l2_sram = mmap(NULL, 1792 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

// Memory-map the 420 MB L3 SRAM buffer
int fd_l3 = open("/dev/pseudo_lock_l3", O_RDWR);
void *l3_sram = mmap(NULL, 420ULL * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd_l3, 0);
```

### IOCTL Operations:
- `PSEUDO_LOCK_IOC_GET_INFO` (`_IOR('P', 1, struct pseudo_lock_info)`): Query level, size, CBM, line size, and physical address.
- `PSEUDO_LOCK_IOC_MEASURE` (`_IOWR('P', 2, struct pseudo_lock_latency)`): Read in-kernel latency distribution (L1/L2 hits, L3 hits, DRAM misses).
- `PSEUDO_LOCK_IOC_RELOAD` (`_IO('P', 3)`): Trigger hardware preloading and cache warming.

---

## 5. Automated Verification Test Suite

Run the full verification benchmark and contention stress test:
```bash
./run_all_tests.sh
```
This tests:
1. Data Read/Write verification (100% data integrity).
2. Cache-line stride latency distribution across every 64-byte line.
3. Eviction resistance under active 16-thread background memory thrashing.

---

## 6. Upstream Linux Kernel Patch

For environments utilizing upstream `/sys/fs/resctrl` rather than this dedicated driver:
The upstream Linux kernel resctrl subsystem has two known limitations:
1. It whitelists only Broadwell and Goldmont Atom CPUs in `arch/x86/kernel/cpu/resctrl/pseudo_lock.c`.
2. It enforces a 4 MiB allocation ceiling via `KMALLOC_MAX_SIZE` in `fs/resctrl/pseudo_lock.c`.

Apply the provided patch to an upstream Linux kernel source tree:
```bash
patch -p1 < linux-resctrl-xeon-pseudo-lock.patch
```
This adds Sapphire Rapids (`0x8F`), Emerald Rapids (`0xCF`), Granite Rapids (`0xAD`), and Sierra Forest (`0xAF`) to the hardware whitelist and replaces `kzalloc`/`kfree` with `kvzalloc`/`kvfree` to support large multi-way L3 allocations.
