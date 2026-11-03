# x86-64 AVX-512 + AMX (Intel Xeon 6 P-core, Granite Rapids)

| item | value |
|---|---|
| target | x86-64 host ISA run by the plowrt CPU engine (no device object; the packet is the NVIDIA-format devblob) |
| `IsaLevel` (`crates/hwspec/src/isa.rs`) | `Amx` (`arch_flag` `amx`), `Avx512` (`avx512`), `CpuRef` (`cpu`, scalar golden) |
| runtime tier enum (`runtime/cpu/dev/cpu_dev.h`) | `PLOW_CPU_ISA_SCALAR=0`, `PLOW_CPU_ISA_AVX512=1`, `PLOW_CPU_ISA_AMX=2` |
| `GpuSpec` | `XEON_6975P_C` in `crates/hwspec/src/intel/xeon6.rs`; `Arch::GraniteRapids`, `compute_cap (6,0)`; lookup keys `xeon6975p`, `xeon6975p-c`, `xeon6` (`crates/hwspec/src/registry.rs`) |
| SKUs plow ships | Xeon 6975P-C (AWS c8i.metal-48xl class), recipes `recipes/infervisor/{gemma-4-e2b,gemma-4-e4b,gemma-4-12b,gemma-4-26b-a4b,gemma-4-31b}/xeon6-amx-bf16.toml` (31B = `emits`, the rest `validated`) |
| other hosts the tier runs on | AMD EPYC 9654 (Zen 4): AVX-512 tier only, verified in `docs/runtime/cpu.md` "Verification on EPYC 9654". aarch64 NEON tier (`runtime/cpu/dev/neon/`) is out of scope here |
| kernels | `runtime/cpu/dev/{amx,avx512,golden}/`, dispatch `runtime/cpu/dev/{isa_detect,dispatch}.c`; host engine `crates/plowrt/src/exec/cpu/{engine,workers,interp,topology,ffi}.rs`. `runtime/cpu/*.c` at the top level is the old correctness-only scalar runtime, not the serving path |
| runtime docs | `docs/runtime/cpu.md`; campaign evidence `docs/bringup/results/gemma4-xeon6-bf16-20261006/`, `docs/bringup/results/xeon6-l2r-bf16-20261008/` |

Verified against: Intel 64 and IA-32 Architectures Optimization Reference Manual Vol. 1 (248966-049US) ch. 2.3 and ch. 20; Intel SDM instruction pages for TDPBF16PS, TILELOADD/TILELOADDT1, VDPBF16PS, VCVTNE2PS2BF16 (felixcloutier.com mirror); Intel product spec page for Xeon 6972P; Linux kernel `Documentation/arch/x86/xstate.rst`; fetched 2026-11-03. ORM 049 describes Sapphire Rapids / Golden Cove. It has no Granite Rapids (Redwood Cove) chapter, so its AMX and port numbers are applied to GNR as "same core family" (UNVERIFIED for GNR specifically, but consistent with plow's GNR measurements below).

## 2. Execution model

The "SM" in `GpuSpec` is one physical core. The "warp" is one 512-bit vector of 16 fp32 lanes. The "tensor core" is one AMX TMUL unit per core. A packet's `n_cu` is a count of virtual executors. Kernels compute "slice `s` of `nblk`", and the worker pool maps any thread count onto them (`docs/runtime/cpu.md`).

| resource | Xeon 6975P-C value | source |
|---|---|---|
| cores / SMT | 96 P-cores, 2 threads each (sibling = cpu + 96) | `xeon6.rs` `sm_count`, `GNR_CORE.max_threads`; P0 lscpu [measured in plow: docs/bringup/results/xeon6-l2r-bf16-20261008/p0_baseline.md] |
| dies / NUMA | 3 compute dies, SNC3 = 3 NUMA nodes of 32 cores (node0 cpus 0-31, node1 32-63, node2 64-95), ~126 GiB each | `xeon6.rs` `ChipletGrouping`; p0_baseline.md |
| vector registers | 32 zmm x 512 bit + 8 opmask (k0-k7) | `GNR_CORE.regs_32bit = 32*16` |
| 512-bit FP FMA | 2 ports (p0, which fuses p0+p1 for 512-bit ops, and p5) | [vendor documented: ORM 049 Table 2-1/2-2, Golden Cove server]; Intel lists 2 AVX-512 FMA units for the sibling 6972P |
| 512-bit shuffles | p5 only (p1 shuffles are 256-bit only) | [vendor documented: ORM 049 Table 2-1 note 1] |
| AMX tiles | 8 tiles TMM0-7, max 16 rows x 64 B = 1 KiB each, 8 KiB XTILEDATA state | [vendor documented: ORM 049 §20.5.2] |
| AMX TMUL rate | 512 bf16 / 512 fp16 / 1024 int8 MAC per cycle per core; no fp8/fp4 | `xeon6.rs` `GNR_AMX`; [measured in plow: p0_baseline.md, 3.27 TF/core at 3.197 GHz = 1023 flop/cycle] |
| L1D | 48 KiB, 12-way (64 sets, so the set stride is 4 KiB) | P0 lscpu; `GNR_CORE.l1_shared_total` |
| L2 | 2 MiB private, 16-way, 2048 sets (set stride 128 KiB) | P0 lscpu; `GNR_CORE.shared_mem` |
| L3 | 480 MiB shared, non-inclusive, 160 MiB per die | P0 lscpu; `xeon6.rs` `l2` / `l2_per_chiplet` (the L3 is stored in the `l2` field) |
| memory | 12-channel DDR5, 768-bit bus; datasheet 614.4 GB/s (DDR5-6400); the platform also supports MRDIMM-8800 | `xeon6.rs` `MemorySpec`; [vendor documented: Intel spec page, Xeon 6972P, the 96-core public sibling] |
| capacity | 384 GiB in hwspec; 377 GiB visible on the lab box | `xeon6.rs`; comparison.md |
| clocks | hwspec boost 3.9 GHz; the 6975P-C SKU sheet was not found (UNVERIFIED). The 6972P sibling is 2.4 base / 3.5 all-core / 3.9 max | [vendor documented: 6972P spec page] |
| sustained clocks, 90 cores | AVX-512 BF16 compute 3.80 GHz; AVX-512 GEMV stream 3.50; AMX compute 3.20; AMX L2 stream 2.90 | [measured in plow: p0_baseline.md "Sustained frequency"] |
| copy engines | DSA exists, but the engine copies with cores (`copy_engines: 1`) | `xeon6.rs` |

Occupancy analogue: one kernel worker per physical core. SMT siblings are left for host threads (§7). The AMX tile set is per logical thread and is saved and restored by XSAVE on context switch.

## 3. Matrix instructions

### AMX (TMUL)

| mnemonic (intrinsic) | operation | CPUID | plow use |
|---|---|---|---|
| `LDTILECFG m512` (`_tile_loadconfig`) | Loads the 64-byte config: byte 0 palette (1), byte 1 `start_row`, bytes 2-15 reserved (0), bytes 16-31 `colsb[0..7]` (u16), bytes 32-47 reserved, bytes 48-55 `rows[0..7]` (u8), bytes 56-63 reserved. Palette 1 has 8 usable slots; nonzero reserved bytes fault. plow's `tilecfg_t` declares 16-entry arrays but zeroes them and sets only 0-7 | AMX-TILE `(7,0):EDX[24]` | `gemm_amx.c:plow_cpu_thread_init_amx`. Called once per worker thread: all 8 tiles are 16 x 64 B |
| `TILELOADD` (`_tile_loadd(t, p, stride)`) | Loads `rows` rows of `colsb` bytes from `base + row*stride`. Needs SIB addressing (the stride is the index register). Restartable through `start_row`. Trailing bytes are zeroed | AMX-TILE | Everywhere |
| `TILELOADDT1` (`_tile_stream_loadd`) | Same, with a hint that the data need not stay in the near caches | AMX-TILE | `gemm_amx.c` (A operand in the strip driver, x tiles in `wm_run`) |
| `TILESTORED` (`_tile_stored`) | Stores a tile with a row stride | AMX-TILE | Epilogues, fp32 partials between K panels |
| `TILEZERO` (`_tile_zero`) | Zeroes a tile | AMX-TILE | Accumulator init |
| `TILERELEASE` (`_tile_release`) | Returns the tile state to INIT | AMX-TILE | **Not used** (grep finds no call) |
| `TDPBF16PS` (`_tile_dpbf16ps(c,a,b)`) | C[16x16 f32] += A[16 rows x 32 bf16] . B[16 rows = k-pairs x 16 dword(bf16 pair)] | AMX-BF16 `(7,0):EDX[22]` | gemm/gemv/moe/attention AMX kernels |
| `TDPFP16PS` | Same shape, fp16 inputs | AMX-FP16 `(7,1):EAX[21]`, GNR and later | **Not used**: no `-mamx-fp16`, not detected |
| `TDPBSSD` / `TDPBSUD` / `TDPBUSD` / `TDPBUUD` | C[16x16 i32] += A[16 x 64 int8] . B[16 rows x 16 dword(4 int8)] (s/u selects signedness of A/B) | AMX-INT8 `(7,0):EDX[25]` | `TDPBSSD` in `moe_amx.c:dot_block_i8` (optional int8 MoE path) |

Semantics that matter [vendor documented: SDM TDPBF16PS page]: MXCSR is neither read nor written. Each accumulation rounds to nearest even. Denormal inputs are treated as zero and denormal outputs are flushed (DAZ=FTZ=1 behaviour regardless of MXCSR). TDPFP16PS differs on inputs: FP16 subnormal inputs are kept, not treated as zero; outputs are still flushed (per a review against the SDM TDPFP16PS page, which was not fetched here: UNVERIFIED).

B ("VNNI") layout: one B tile row holds one k-pair across 16 output columns, as 16 dwords of (b[2p][n], b[2p+1][n]). For int8 it is a k-quad per dword. plow builds it with one in-register 16x16 dword transpose (`amx_common.h:plow_amx_tr16x16`) per 1 KiB tile. Stage 1 uses shift+blend so that 64 lane moves split across p0/p5 instead of all landing on p5.

Throughput/latency in cycles [vendor documented: ORM 049 Table 20-2, SPR]:

| instr | throughput | latency |
|---|---|---|
| TDP* | 16 | 52 |
| TILELOADD | 8 | 45 |
| TILELOADDT1 | 33 | 48 |
| TILESTORED | 16 | n/a |
| TILEZERO | 0 | 16 |
| LDTILECFG | n/a | 204 |
| TILERELEASE | n/a | 13 |

A 16-cycle TDP over 16x16x32 = 8192 MAC gives 512 MAC/cycle, which matches hwspec and the measured 3.27 TF/core. Because of the 52-cycle latency, at least 4 independent accumulators are needed. plow's standard block is **2x2**: C in tmm0-3, A in tmm4/5, B in tmm6/7, with each A and B load feeding 2 TDPs. A 2x2 block consumes 4 KiB of tiles per 64 TDP cycles, which is the L2-to-L1 fill rate, so operands must already be in L1 [measured in plow: `runtime/cpu/dev/amx/gemm_amx.c` header, Xeon 8581C].

Tile indices are instruction immediates, so the intrinsics need compile-time-constant tile numbers.

### AVX-512 dot products

| mnemonic (intrinsic) | operation | CPUID | plow use |
|---|---|---|---|
| `VDPBF16PS zmm` (`_mm512_dpbf16_ps`) | 16 f32 lanes += 2 bf16 products per lane | AVX512F + AVX512_BF16 `(7,1):EAX[5]` | avx512 gemv/gemm/attention/fp8/moe_gemma, `gemv_amx.c` tails, `mxfp4_common.h` |
| `VCVTNE2PS2BF16` (`_mm512_cvtne2ps_pbh(hi, lo)`) | 2x16 f32 to 32 bf16. `hi` (src1) fills the upper half. RNE, DAZ/FTZ always, MXCSR ignored | AVX512_BF16 | A staging, norm folds, attention P |
| `VCVTNEPS2BF16` (`_mm512_cvtneps_pbh`) | 16 f32 to 16 bf16, same rounding rules | AVX512_BF16 | `avx512.h:v_store_bf16` |
| `VPDPBUSD` (`_mm512_dpbusd_epi32`) | u8 x s8, 4-way, i32 accumulate | AVX512_VNNI `(7,0):ECX[11]` | Required by detection, but no kernel calls the intrinsic |
| AVX512-FP16 (`VFMADD*PH`, `VCVTPH2PSX`...; plain `VCVTPH2PS` is F16C/AVX512F, not FP16) | native fp16 arithmetic | AVX512_FP16 `(7,0):EDX[23]` | **Never enabled**: keeps Zen 4 on the same tier (`docs/runtime/cpu.md` "ISA coverage") |

VDPBF16PS denormal rules are the same as TDPBF16PS: DAZ, FTZ and RNE are fixed and MXCSR is ignored [vendor documented: SDM VDPBF16PS page]. Measured compute-only issue rate on GNR is about 1 VDPBF16PS zmm per cycle (218 GF/core at 3.8 GHz), not 2 [measured in plow: p0_baseline.md]. The exact port is not documented in ORM 049 (UNVERIFIED).

AVX10: GNR is the first AVX10.1 part. AVX10.1/512 is a re-enumeration of the AVX-512 set GNR already has, and GCC's `-march=graniterapids` maps to it [vendor documented: GCC AVX10.1 enablement, per Phoronix reporting]. plow does not test the AVX10 CPUID leaf; it detects the individual AVX-512 bits. The exact AVX10 enumeration leaf and bit were not confirmed (UNVERIFIED).

## 4. Memory movement

* **Vector width**: one zmm is 64 B, which is one cache line. Unaligned `loadu/storeu` is the norm in plow; `engine.rs` allocates every tensor 64 B aligned (`ALIGN`), and tensors of 2 MiB or more are 2 MiB aligned.
* **Alignment for AMX operands**: align to 64 B, but **not** to 128/256 B multiples [vendor documented: ORM §20.4]. On this L1 (64 sets), a row stride that is a multiple of 4 KiB (bf16 K = 2048) puts all 16 rows of one TILELOADD in the same L1 set, which is 12-way. This has not been measured in plow (UNVERIFIED for plow shapes).
* **TILELOADD cannot forward from the store buffer**: it waits until the store reaches L1. TILESTORED-to-vector forwarding only works for full 64 B lines and has outliers. Keep tens of cycles between them [vendor documented: ORM §20.15]. plow's `stage_a` and the pack loops write kb-outer so the first chunk a TILELOADD reads is the oldest, and fp32 partials go TILESTORED-to-TILELOADD, never through AVX stores.
* **Temporal vs T1 loads**: the streamed operand should use TILELOADDT1 so it does not evict the reused one from L1 ("priority inversion") [vendor documented: ORM §20.8.1]. plow keeps a 32 KiB packed B strip (K panel 512) L1-resident and streams A with T1 (`gemm_amx.c`, `KP`).
* **Prefetch**: `_mm_prefetch(..., _MM_HINT_T1)` into L2, about 512 B to 1 KiB ahead per streamed weight row (`PACK_PF_B`, `XGV_PF`, `XM_PF`). Results:
  * T0 software L1 prefetch two K steps ahead measured −13 to −18% single-core and is not used [measured in plow: a25488237].
  * NTA prefetch is worse than none: the line still allocates in L2 and only gets a shorter life [measured in plow: docs/bringup/results/xeon6-l2r-bf16-20261008/p4_kv_policy.md].
  * Prefetching KV d rows ahead hides DRAM latency (257 to 193 µs per step). A bulk early prefetch burst stalls the GEMVs behind it [measured in plow: p4_kv_policy.md].
* **Non-temporal stores**: `plowrt` kernels issue none. In the l2r stage bench, per-SNC-node replicas written with full-line NT stores plus `sfence` cut the all-gather cost by about 83% (E2B layer 147.8 to 51.8 µs) [measured in plow: p3_sync.md]. This is not ported into plowrt.
* **CLDEMOTE**: hand-encoded (`.byte 0x0f,0x1c,0x07`) in `gemv_amx.c` because the build passes no `-mcldemote`. It is behind `PLOW_AMX_DEBUG=demote` and measured neutral [measured in plow: a25488237].
* **Pages**: with 4 KiB pages, a 1.0-1.5 MiB L2-resident slice loses up to 25% to set conflicts (page colouring over 2048 sets). THP restores full rate on every core [measured in plow: p0_baseline.md P0.3].
* **NUMA** (`engine.rs`, `docs/runtime/cpu.md` "NUMA policy"):
  * Tensors of 256 KiB or more get a fresh 2 MiB-aligned anonymous mapping, with `mbind` (BIND on one node, INTERLEAVE on several) applied **before first touch**.
  * Multi-node interleave defaults to `MADV_NOHUGEPAGE`. Override with `PLOW_CPU_HUGE_PAGES`.
  * Per-worker scratch (`PLOW_CPU_SCRATCH_BYTES` = 4 MiB) is first-touched after pinning.
  * `PLOW_CPU_WEIGHT_AFFINE=1` (every Xeon recipe) binds each decode GEMV slice's weight rows to its executor's node.
* **Measured bandwidth tiers**, 90 isolated cores, THP, GB/s total [measured in plow: p0_baseline.md]:

| tier | AVX-512 read | AMX-BF16 tile stream | AVX-512 BF16 GEMV | slowest core (AMX), GB/s |
|---|---:|---:|---:|---:|
| L2 (1.0-1.5 MiB/core) | 15,400 | 12,500 | 9,100 | 139 |
| L3 (3-4 MiB/core) | 1,300-1,360 | 1,300-1,360 | 1,320-1,380 | 10.4-10.9 |
| DRAM (64 MiB/core) | 669 | 657 | 607 | 7.2 |

  IMC CAS counters read 612 GB/s during the DRAM test. The client figure is higher because part of the buffer hits L3.

## 5. Synchronization and ordering

* x86 is TSO: ordinary stores are seen in order and a release needs no fence for WB memory. **NT stores are weakly ordered**: issue `sfence` after them, before the flag or barrier arrival (the l2r stage does this) [measured in plow: p3_sync.md "Memory ordering"].
* AMX has no async completion to wait for. TILESTORED is an ordinary store as far as ordering goes. The hazard is TILELOADD after a recent store to the same lines (§4).
* plowrt workers synchronise through per-line atomic counters (`interp.rs`, `control.rs`). They wait spin, then yield, then park (`PLOW_CPU_SPIN_US`, default 2000 µs; parking at 50 µs measured +17% TPOT). Every flag and counter has its own 64 B line.
* Completion counters bumped by many slices contend on one line (2 µs median, up to 16 µs for the last bump). `PLOW_CPU_COMBINE=16` (E2B/E4B recipes) gives TPOT −7 to −10% at c1 [measured in plow: p5_serving_report.md, `docs/runtime/cpu.md`].
* Cross-die handoff costs about 0.58 µs inside one SNC node and more across dies. Hierarchical barriers lost 1.8-4.4 µs per layer under load. UMWAIT/CLDEMOTE moved idle handoff by 11 ns or less [measured in plow: p3_sync.md].
* ORM recommends that idle pool threads wait with UMWAIT/TPAUSE C0.2 rather than PAUSE or a raw spin when they share a core with an AMX thread [vendor documented: ORM §20.17.2]. plowrt spins with `spin_loop` (PAUSE).

## 6. Toolchain as plow uses it

* Compiler: the nix devShell's C compiler, driven by the `cc` crate from `crates/plowrt/build.rs` (`cpu_dev` build, `--features cpu`). `runtime/CMakeLists.txt` mirrors it (target `plow_cpu_dev`, ctest `cpu_dev_*`).
* Base flags: `-std=c11 -O2`, warnings on. ISO C11 (not gnu11) is deliberate, because gnu11 lets GCC contract `a*b+c` into FMA and the tier tests would then check different rounding than plowrt ships (`runtime/CMakeLists.txt`, `C_EXTENSIONS OFF`).
* Per-directory flags. There is **no `-march`**: tiers are chosen at run time by cpuid, so the base library must run on a non-AVX-512 host.

| sources | flags |
|---|---|
| `runtime/cpu/dev/*.c`, `golden/*.c` | base only |
| `runtime/cpu/dev/avx512/*.c` | `-mavx512f -mavx512bw -mavx512vl -mavx512bf16 -mavx512vnni` |
| `runtime/cpu/dev/amx/*.c` | the above + `-mamx-tile -mamx-bf16 -mamx-int8` |
| `runtime/cpu/dev/neon/*.c` (aarch64 only) | `-march=armv8.6-a+bf16+i8mm+fp16` |

  Scratch benches in `runtime/cpu/bench/l2r/` build standalone with `-O3 -march=sapphirerapids -mamx-tile -mamx-bf16 -mavx512bf16 [-mcldemote -mwaitpkg]` (`runtime/cpu/bench/l2r/README.md`).
* Adding AMX-FP16 or AVX512-FP16 code needs `-mamx-fp16` / `-mavx512fp16` on that file only, a new cpuid check in `isa_detect.c:detect_isa` (AMX-FP16 is `(7,1):EAX[21]`), and a fallback for SPR and Zen 4. Do not add `-march=graniterapids` to a tier directory: it would let the compiler emit GNR-only instructions in code shared with SPR and Zen 4.
* `crates/kernelcaps` has no x86 CPU entries. `plowc --gpu xeon6975p --arch amx|avx512` only labels the packet. The emitted packet takes the `sm_120a` decisions and differs from an `--gpu rtx6000pro --arch sm_120a` packet only in its target hash and one CPU default: the NVIDIA decode merge-fold (`PLOW_NV_FA_FOLD_WIDE`) is off, because the CPU flash-decode kernels need FLASH_MERGE (`docs/runtime/cpu.md`). `plowc`'s `--intel-cpu` profiles (`xeon6/gnr`, `xeon5/emr`, `xeon4/spr`, `client`) set only pseudo-lock SRAM budgets (`crates/plowc/src/main.rs:resolve_intel_sram_budgets`; GNR 1920 KiB L2 / 420 MiB L3).
* Runtime tier: `--cpu-isa scalar|avx512|amx` / `PLOW_CPU_ISA` is a ceiling and never goes above what cpuid and the OS allow. `PLOW_CPU_GOLDEN_OPS=1,16,...` pins listed ops back to golden for bisection (`isa_detect.c`).
* Detection order (`isa_detect.c:detect_isa`):
  1. OSXSAVE, then XCR0 bits 1, 2, 5, 6 and 7 (`0xE6`).
  2. AVX-512 F/BW/VL + BF16 + VNNI, else scalar.
  3. AMX TILE+BF16+INT8, then `arch_prctl(ARCH_REQ_XCOMP_PERM=0x1023, XFEATURE_XTILEDATA=18)`, then XCR0 bits 17 and 18, else AVX-512.
* Per-thread init (`plow_cpu_thread_init`) sets MXCSR FTZ+DAZ (`|= 0x8040`), then `LDTILECFG` palette 1 with all tiles 16 x 64.
* Intrinsic style (matches `gemm_amx.c`, 2x2 block over one K step of 32 bf16):

```c
_tile_stream_loadd(4, A0 + kb * 32u, lda);   /* A rows 0-15, streamed (T1) */
_tile_loadd(6, bstrip, 64);                  /* B cols 0-15, VNNI, L1-resident */
_tile_dpbf16ps(0, 4, 6);
_tile_loadd(7, bstrip + 1024, 64);           /* B cols 16-31 */
_tile_dpbf16ps(1, 4, 7);                     /* tmm2/3 likewise with A rows 16-31 in tmm5 */
```

* Inspect objects:
  * `objdump -d --no-show-raw-insn target/release/build/plowrt-*/out/*.o | grep -cE 'tdpbf16ps|tileloadd'`, or `llvm-objdump -d` on `libplow_cpu_dev.a`.
  * Check that the 32-register transpose did not spill (look for `rsp`-relative `vmovdqu64` inside `plow_amx_tr16x16` callers).
  * `scripts/plow_isa.py` and `scripts/asm_audit.py` are AMD code-object tools and do not read x86 objects.
* Tests and benches:
  * `ctest -R cpu_dev` (golden vs tier on random inputs). `cpu_dev_amx_test` self-skips without AMX.
  * `cpu_dev_amx_test --bench` gives single-thread GEMM/MoE GFLOPS.
  * `l2r_bw` gives per-core L2/L3/DRAM bandwidth and compute ceilings.

## 7. Pitfalls

* **No permission, no AMX.** The first instruction that touches XTILEDATA (TILELOADD, TILEZERO, TDP*...) without `ARCH_REQ_XCOMP_PERM` raises SIGILL through the XFD trap. LDTILECFG, STTILECFG and TILERELEASE do not touch tile data and do not trap, so a successful LDTILECFG proves nothing; with permission, a failed XSAVE buffer allocation raises SIGSEGV. The permission is per process, is inherited across fork and cleared on exec, and the request fails if an existing sigaltstack is too small for the larger frame. Request it before any worker thread starts [vendor documented: Linux `Documentation/arch/x86/xstate.rst`; ORM §20.20.2]. plow does this in `detect_isa` and falls back to AVX-512 on failure [measured in plow: `runtime/cpu/dev/isa_detect.c`].
* **Tile config is ABI-volatile.** The psABI treats tile config and tmm registers as caller-clobbered: every function is meant to configure its own tiles [vendor documented: ORM §20.19.1]. plow configures **once per worker thread** and relies on no other code in the process running LDTILECFG or TILERELEASE. Embedding a library that uses AMX (oneDNN and similar) in the same worker threads breaks this silently. Re-issue `plow_cpu_thread_init_amx` after foreign code runs [measured in plow: `gemm_amx.c:plow_cpu_thread_init_amx`].
* **TILERELEASE is never issued.** A thread with valid tile state cannot reach core C6 on SPR; it is demoted to C1. The Linux idle task holds no AMX state, so this does not apply to ordinary idling [vendor documented: ORM §20.19]. plow hosts hold `/dev/cpu_dma_latency=0` anyway (no C6, because C6 would flush L2) [measured in plow: p0_baseline.md].
* **Denormals.** bf16 activations widen to fp32 denormals, and each AVX-512 op on them takes a microcode assist. A 1024 x 5376 RMSNORM measured 38 ms vs 0.95 ms with FTZ/DAZ on. Every worker sets MXCSR FTZ+DAZ [measured in plow: 0a2d38a78, `isa_detect.c:plow_cpu_thread_init`]. TDPBF16PS, VDPBF16PS and VCVTNE* flush regardless of MXCSR [vendor documented: SDM pages].
* **VCVTNEPS2BF16 is not plow_f2bf bit for bit.** It flushes denormals, while the scalar `plow_f2bf` keeps them and NaN payloads differ. Where an epilogue must match golden bits, open-code the round (MoE epilogues do) [measured in plow: 0d07dfa66].
* **Contraction changes rounding.** Build ISO C11 so GCC does not fuse `a*b+c`; gnu11 makes the tests check different numerics than plowrt ships [measured in plow: `runtime/CMakeLists.txt` comment].
* **Attention summation order is accuracy-visible.**
  * TMUL FLASH_PREFILL with bf16-rounded P failed the Gemma-4-26B-A4B FP32 gate (kl_mean 0.118-0.128 vs limit 0.112, a router near-tie).
  * `PLOW_CPU_AMX_ATTN_SPLIT_P=1` (P = hi + lo bf16, two TDPs per key step) passes at 0.0999 and is set in that recipe [measured in plow: eebd6ba06, b69242293].
  * `PLOW_CPU_AMX_ATTN=0` restores the AVX-512 kernel.
  * KV split width: 256-key splits gave 12B KL p99 0.79 vs 0.33 at 2048 [measured in plow: `docs/runtime/cpu.md` "Packed prefill"].
* **AMX loses at M < 5.** At M=4 the AMX GEMV took 276 ms vs 250 ms for AVX-512 RB=4 dots, so the crossover is `XGV_MIN_M = 5`. At batch 1, AMX uses 1/16 of each tile op and is not faster [measured in plow: `gemv_amx.c`; p2_real_bf16_layer.md].
* **Falling off a tier cliff.** A GEMV shape no fast tier accepts falls to golden scalar, which re-reads the weights per row. E2B rung-32 took 554 ms (vs 29.8 ms once AMX handled M up to 32), and E4B rung-64 took 2137 ms (vs 90 ms). Check `plow_cpu_tier_of(op)` for every rung you add [measured in plow: da46693f4, e2b813ba7].
* **Activation pack replicated per slice.** Splitting prefill GEMM over N only made every slice pack all of M x K (E2B prefill 300 ms, 108 ms with the pack skipped). Use 2D (m, n) slice grids. Narrow N (router N=32) serialised onto slice 0 [measured in plow: da46693f4, 0e3d2eccf].
* **L2 budget for the token chunk.** Past about 640 tokens, the x panel (2 KiB/token) plus partials overflowed L2 and the TDP loop went 4x for 2x tokens. Chunk tokens to 3/4 of L2 (`WM_L2_BUDGET`) [measured in plow: 6ad987ac7]. x tiles served from L3 cap the K loop at about 40% of TMUL; both operands from L2 reach about 94% [measured in plow: `gemm_amx.c` `wm_run` header].
* **Single-core vs in-model.** The same GEMM ran 813 GFLOPS single-thread from L3 but about 17 GFLOPS/thread in-model, because 16 weight-row streams per thread are not covered by the L2 streamer. Software T1 prefetch of each row is required [measured in plow: `gemm_amx.c` `PACK_PF_B`]. Prefetch effects reverse between 1 and 96 threads (dropping the strip prefetch is +12-40% on one core and neutral to −4% at 96), so always A/B at full width [measured in plow: a25488237].
* **4 KiB pages lose L2 capacity**: −25% at 1.5 MiB per core. Weight slices meant to stay L2-resident need THP, and multi-node interleave disables THP by default [measured in plow: p0_baseline.md].
* **Host threads need a sibling.** Restricting plowrt to exactly the worker cores cost 30% decode time: engine and tokio threads time-slice against spinning workers. Give the cpuset the workers' SMT siblings [measured in plow: p0_baseline.md "Host-thread placement"]. Do not put a second AMX thread on a sibling [vendor documented: ORM §20.17.2].
* **Surplus workers.** Threads beyond `n_cu` own no work but still spin, measured 8.3x slower decode at 240 idle workers [measured in plow: 0cb9adb48, `docs/runtime/cpu.md`].
* **Placement ideas that lost.** Placing executors by L2 domain measured 1.5x slower (`--cpu-l2-place`). The op-major global queue measured about 2x slower on the EPYC 9654 (`--cpu-global-queue`). Restricting placement to node 0 measured 2.3-2.6x slower than distributed placement at 24 workers [measured in plow: `docs/runtime/cpu.md`].
* **L2 pseudo-lock helps only 31B.** `PLOW_CPU_SRAM` with `pseudo_lock_sram.ko l2_ways=12` gave 31B prefill −9% but made 12B/E2B slower (12B prefill 606 to 817-982 ms). Under KV-stream interference, plain L2 beat locked [measured in plow: 0a2d38a78, 6fccf559a, p1_residency.md].
* **AMX license clock.** AMX compute holds 3.2 GHz and AMX streaming 2.9 GHz, vs 3.5-3.8 GHz for AVX-512 [measured in plow: p0_baseline.md]. AMX has its own lower frequency license, but at under 15% AMX utilisation the core may exceed it [vendor documented: ORM §20.2.1]. Mixing a little AMX into an AVX-512 phase can still drop the clock.
* **Recipe .md emit blocks omit the CPU target flags.** `scripts/render_recipe.py` prints no `--gpu xeon6975p --arch amx --n-cu 96 --max-ctx`; `campaign.py build` passes them from `[cell]`. Emit through `campaign.py`, not by pasting the `.md` block [measured in plow: `recipes/infervisor/gemma-4-12b/xeon6-amx-bf16.md` vs `scripts/campaign/campaign.py`].
* **Benchmark host state.** Every qualified Xeon number assumes `kernel.numa_balancing=0`, `/dev/cpu_dma_latency=0` held, THP `madvise`, and no pseudo-lock unless the recipe sets one [measured in plow: recipe headers].

## 8. Roofline numbers

| quantity | datasheet / derived peak | plow-measured ceiling | where it lives |
|---|---|---|---|
| DRAM read, socket | 614.4 GB/s (DDR5-6400 x 12) | 612 GB/s at the IMC, 607-669 client, 500-626 in long-context attention | `xeon6.rs` `bandwidth` (`bandwidth_measured: None`); p0_baseline.md; p4_kv_policy.md |
| DRAM in decode (E2B, whole model streamed) | n/a | about 445 GB/s achieved | p5_serving_report.md |
| L3, 90 cores | n/a | 1.30-1.38 TB/s | p0_baseline.md |
| L2 per core | n/a | 171 (AVX-512 read) / 139 (AMX stream) / 101 (VDPBF16PS GEMV) GB/s | p0_baseline.md |
| AMX BF16 | 96 x 1024 flop x 3.9 GHz = 383 TF; at the measured 3.2 GHz = 314 TF | 3.27 TF/core, 294 TF on 90 cores | `xeon6.rs` `GNR_AMX`; p0_baseline.md |
| AMX INT8 | 2x BF16 (1024 MAC/cycle) = 766 TOPS at 3.9 GHz (derived) | not measured | `xeon6.rs` |
| AMX FP16 | same as BF16 (hwspec) | not used, not measured | `xeon6.rs` |
| AVX-512 BF16 (VDPBF16PS) | 64 flop/instr; the 2-port assumption gives 128 flop/cycle (UNVERIFIED) | 218 GF/core (about 1 instr/cycle) | p0_baseline.md |
| single-thread AMX GEMM | n/a | 813 GFLOPS from L3 | `gemm_amx.c` comment |
| serving throughput c32, 1000/128 | n/a | E2B 4046, E4B 2497, 12B 1321, 26B-A4B 1881 tok/s | recipe `[measured]` tables |

* No Xeon recipe carries a `[roofline]` table. `tuning/` has no Intel directory, and `crates/costmodel` models only the AMX 16x16x32 MMA shape (`mma.rs`) with zero SRAM carve-out (`lib.rs`).
* `scripts/bench/calibrate_ceilings.py` is torch/CUDA-only and `scripts/campaign/block_roofline.py` has no CPU path. Re-measure CPU ceilings with `runtime/cpu/bench/l2r/l2r_bw.c` (modes 0-4, `L2R_CPUS`, `L2R_HUGE=1`), with turbostat for clocks and the IMC CAS counters for DRAM, as in p0_baseline.md P0.3.
* `bandwidth_for_bound()` falls back to the 614.4 datasheet figure because no measured value is registered.

## 9. Decision checklist

* If you are about to write a GEMM/GEMV for M ≥ 5 bf16 rows: use TDPBF16PS with the 2x2 accumulator block (tmm0-3 C, 4/5 A, 6/7 B). For M ≤ 4, use AVX-512 VDPBF16PS row dots.
* If you are about to choose which operand is the tile A: make the large, once-streamed matrix (the weights) the row-major A operand loaded in place, and pack the small one (the activations) to VNNI B once per call or panel (`wm_run`, `gemv_amx.c`, `moe_amx.c`).
* If you are about to size a K panel or token chunk: keep the reused operand in L1 (32 KiB strip, K panel 512) and the chunk under 3/4 of L2 (1.5 MiB). Stream the other operand with TILELOADDT1.
* If you are about to read weights from DRAM inside the tile loop: add a T1 software prefetch per streamed row (about 512 B-1 KiB ahead) and A/B it at 96 threads, not on one core.
* If you are about to store and then TILELOADD the same lines: put tens of cycles between them, or march stores so the first-loaded chunk is the oldest. Never feed TILELOADD from fresh AVX stores in the next instruction.
* If you are about to add a new rung, shape or op: check `plow_cpu_tier_of(op)` at that M. A miss falls to golden scalar and costs 10-50x.
* If you are about to partition work across slices: use a 2D (m, n) grid that avoids replicating the activation pack, and split narrow N over M.
* If you are about to round f32 to bf16 in an epilogue that must match golden bits: open-code RNE, because VCVTNE* flushes denormals.
* If you are about to change summation order in attention or a router-adjacent op: re-run the FP32-reference gate (`scripts/llm/fp32_ref_gate.py`), since 26B-A4B sits on a near-tie edge.
* If you are about to use a new x86 extension (AMX-FP16, AVX512-FP16, CLDEMOTE, WAITPKG): add the `-m` flag per file, a cpuid (+XCR0) check in `detect_isa`, and keep Zen 4 / SPR on a fallback. Never add `-march` to a shared tier.
* If you are about to allocate a large buffer: allocate it through the engine (fresh mapping, `mbind` before first touch, 2 MiB aligned). Want L2 residency? Then force THP. First-touch scratch after pinning.
* If you are about to exchange activations across SNC nodes: per-node replicas with full-line NT stores plus `sfence` beat remote snoops of the producer's L2 (measured in the l2r bench, not yet in plowrt).
* If you are about to benchmark: hold `cpu_dma_latency=0`, set `numa_balancing=0`, give plowrt the workers' SMT siblings, keep `--cpu-threads` ≤ `n_cu`, and record turbostat clocks (AMX vs AVX-512 licenses).

## 10. Sources

| topic | document | section | version |
|---|---|---|---|
| AMX detection, latencies, alignment, T1 loads, store forwarding, HT, TILERELEASE/C-state, ABI, XFD/sigaltstack | Intel 64 and IA-32 Architectures Optimization Reference Manual Vol. 1, https://cdrdv2-public.intel.com/814198/248966-Optimization-Reference-Manual-V1-049.pdf | ch. 20 (§20.1-20.4, 20.8, 20.14-20.15, 20.17.2, 20.19-20.20) | 248966-049US |
| Golden Cove ports and 512-bit units | same | ch. 2.3, Tables 2-1/2-2 | 049 |
| TDPBF16PS semantics | Intel SDM Vol. 2 (mirror https://www.felixcloutier.com/x86/tdpbf16ps) | instruction page | fetched 2026-11-03 |
| TILELOADD/TILELOADDT1 | Intel SDM Vol. 2 (https://www.felixcloutier.com/x86/tileloadd:tileloaddt1) | instruction page | fetched 2026-11-03 |
| VDPBF16PS, VCVTNE2PS2BF16 | Intel SDM Vol. 2 (https://www.felixcloutier.com/x86/vdpbf16ps, .../vcvtne2ps2bf16) | instruction pages | fetched 2026-11-03 |
| Tile architecture, CPUID leaves, XCR0 components | Intel SDM Vol. 1 ch. 18 (AMX) and Vol. 2 CPUID; https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html | ch. 18; CPUID leaf 7 | current SDM |
| AMX permission on Linux | https://docs.kernel.org/arch/x86/xstate.html | "Using dynamically enabled XSTATE features" | fetched 2026-11-03 |
| Intrinsic signatures | Intel Intrinsics Guide, https://www.intel.com/content/www/us/en/docs/intrinsics-guide/ | AMX-TILE/BF16/INT8/FP16, AVX512_BF16 | current |
| Xeon 6 platform (cores, cache, memory types, FMA units) | https://www.intel.com/content/www/us/en/products/sku/240785/intel-xeon-6972p-processor-480m-cache-2-40-ghz/specifications.html | specifications | fetched 2026-11-03 (6975P-C itself not published) |
| AVX10.1 on Granite Rapids, AMX-FP16 enumeration | Intel AVX10 Architecture Specification (https://www.intel.com/content/www/us/en/content-details/784267/intel-advanced-vector-extensions-10-intel-avx10-architecture-specification.html); GCC/LLVM enablement notes | version enumeration | not fetched in full (UNVERIFIED details) |
| psABI tile register rules | System V ABI Intel386/x86-64 supplement (AMX) | tile registers | via ORM §20.19.1 |
