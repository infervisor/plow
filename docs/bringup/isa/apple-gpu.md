# Apple GPU (Apple family 9, MSL 3.2) — `metal3`

| Field | Value |
|---|---|
| Target | Apple silicon GPU, Metal GPU family **Apple9** (M3, M4 series; also A17 Pro, A18) |
| Arch strings | `--arch metal3` (`IsaLevel::Metal3.arch_flag() == "metal3"`); `plowc --gpu m4pro` resolves it (`crates/hwspec/src/registry.rs` `("m4pro", "Apple M4 Pro")`) |
| SKUs plow ships | Apple M4 Pro, 16 GPU cores: `recipes/infervisor/qwen3-0.6b/metal3-apple-m4-pro-tp1-2k.toml`, `recipes/infervisor/smollm2-360m/metal3-apple-m4-pro-tp1-2k.toml` (`[target] units = 16, mem_bytes = 19069665280`; status `emits` in the recipe `.md`) |
| hwspec | `IsaLevel::Metal3` (`isa.rs`); `APPLE_M4{,_PRO,_MAX}` in `crates/hwspec/src/apple/m4.rs` |
| Kernels | `runtime/apple/*.metal` (`interp.metal` = device-ISA interpreter; `q8.metal`, `asr*.metal`, `mlp_channel.metal`); probes `runtime/apple/probe/*.m` |
| Host engine | `crates/plowrt/src/exec/apple/` (feature `metal`; `ane` adds CoreML), via `objc2-metal 0.3` |
| Shipping form | MSL **source** inside `plowrt` (`include_str!`), compiled on the serving Mac at load. Apple bundles ship no objset (`docs/DISTRIBUTION.md` "Runtime-compiled targets") |

Apple publishes no GPU machine ISA. This file documents the level kernel authors actually
program: the Metal Shading Language (MSL) and the Metal feature-set limits for Apple9.

## 1. Execution model

| Item | Apple9 / M4 Pro | Source |
|---|---|---|
| SIMD-group width | 32 threads; the interpreter, q8, ASR and conformer/RNNT loaders assert `threadExecutionWidth() == 32` at load (not every pipeline: `asr_causal_conv3_f16` is unchecked) | [measured in plow: `crates/plowrt/src/exec/apple/mod.rs`, `q8.rs`, `asr_subsampling.rs`]; width per pipeline is a runtime property [vendor documented: MSL 4.1 §4.4.1 / `MTLComputePipelineState.threadExecutionWidth`] |
| Max threads per threadgroup | 1024 (theoretical; real limit per pipeline = `maxTotalThreadsPerThreadgroup`, lowered by register use) | [vendor documented: Feature Set Tables, "GPU implementation limits", footnote 5] |
| Threadgroup memory | 32 KiB total per threadgroup, length aligned to 16 B | [vendor documented: Feature Set Tables, "Maximum total threadgroup memory allocation", Apple4–Apple11] |
| Threadgroup-memory arguments | max 31 entries per kernel | [vendor documented: Feature Set Tables] |
| `setBytes` inline constants | 4 KiB max | [vendor documented: Feature Set Tables] |
| GPU cores | M4 = 8 or 10; M4 Pro = 16 or 20; M4 Max = 32 or 40 | [vendor documented: Apple Newsroom M4 / M4 Pro-Max; Apple iMac (M4, two ports) tech specs, 8-core GPU] |
| hwspec cores | `APPLE_M4` 10, `APPLE_M4_PRO` 16, `APPLE_M4_MAX` 32 (smaller SKU for Pro/Max; the 8-core M4 is not listed). Registry lookup is by device **name**, so a 20-core M4 Pro resolves to 16 | `crates/hwspec/src/apple/m4.rs` |
| FP32 lanes per core | 128 (hwspec `APPLE_G16_MMA` = 128 MACs/clk/core) (UNVERIFIED: Apple publishes no ALU count) | `m4.rs` |
| Registers per core | hwspec `regs_32bit = 32768`, `max_threads = 1536`, `max_warps = 48`, `max_blocks = 32` — flagged in-code as conservative guesses (UNVERIFIED) | `m4.rs` `APPLE_M4_CORE` |
| Dynamic Caching | Registers, threadgroup and tile memory, stack and buffer cache are allocated dynamically out of one on-chip store; occupancy adapts per shader (M3 generation onward, M4 shares it) | [vendor documented: Apple Newsroom M4 "Dynamic Caching"; tech talk "Explore GPU advancements in M3 and A17 Pro"] |
| Occupancy query | None exposed. The runtime launches the **packet's `n_cu`** threadgroups and compares it with the name-based registry core count (`hwspec::registry::lookup(device.name())`); no co-residency or core-count probe runs at load. The `m4.rs` comment saying the grid is "sized by the co-residency probe at load" does not match the code; `probe.m` (b) `coresident` is a bring-up tool | `crates/plowrt/src/exec/apple/mod.rs` `matching_geometry`, dispatch `width: n_cu`; `m4.rs` |
| L1 / shared total | hwspec `l1_shared_total = 64 KiB` (UNVERIFIED) | `m4.rs` |
| Last-level (SLC) | hwspec `l2`: M4 8 MiB, M4 Pro 24 MiB, M4 Max 32 MiB (UNVERIFIED: Apple publishes no SLC size) | `m4.rs` |
| Memory | Unified LPDDR5X; M4 120 GB/s, M4 Pro 273 GB/s (256-bit), M4 Max 410 GB/s (32-core) / 546 GB/s (40-core) | 120/273/546 [vendor documented: Newsroom M4 Pro-Max]; LPDDR5X and bus widths (UNVERIFIED) |
| GPU clock | hwspec `clock_boost` M4 1470 MHz, M4 Pro/Max 1578 MHz (UNVERIFIED) | `m4.rs` |
| Neural Engine | 16 cores, 38 TOPS (int8 per hwspec) per M4 chip | [vendor documented: Newsroom M4 "38 trillion operations per second"]; fp16 measured 4.05 TFLOPS [measured in plow: 8682fc356] |
| Forward progress | **Not guaranteed** for compute on Apple9; "Compute shader forward progress guarantees" starts at Apple11 | [vendor documented: Feature Set Tables, feature availability] |
| FP32 denormals | Flushed on Apple9; "FP32 denormalized numbers" starts at Apple11 | [vendor documented: Feature Set Tables; MSL 4.1 §8.1] |

Plow's shape on this GPU: **one threadgroup per GPU core, 1024 threads = 32 SIMD-groups**,
grid = `n_cu` (`runtime/apple/interp.metal` header, `NT = 1024`, `NSG = 32`;
`crates/plowrt/src/exec/apple/mod.rs` `THREADS = 1024`). The packet's `n_cu` must equal the
registry core count for the device **name** or the engine falls back to instruction-ordered
dispatch (§6 pitfall 3). On a 20-core M4 Pro the name resolves to 16, so a 16-core packet runs the
persistent walk on 16 threadgroups and 4 cores idle.

Family map (Feature Set Tables, Oct 2026): M1 = Apple7, M2 = Apple8, M3/M4 = Apple9,
M5 = Apple10, M6 = Apple11. hwspec has one `IsaLevel::Metal3` for all of them;
`HardwareFingerprint::from_live_named` describes an unlisted M-series chip (e.g. M5) with the
per-core resources of the nearest listed Apple part.

## 2. Matrix / tensor instructions

### 2.1 `simdgroup_matrix` (what plow uses)

| Item | Fact |
|---|---|
| Header | `<metal_simdgroup_matrix>` (pulled in by `<metal_stdlib>`) |
| Types | `simdgroup_half8x8`, `simdgroup_float8x8`, `simdgroup_bfloat8x8` (bfloat needs MSL **3.1+**). Only 8x8. [vendor documented: MSL 4.1 §2.4] |
| Functions | `simdgroup_load(d, src, elements_per_row = Cols, ulong2 origin = 0, bool transpose = false)` from `threadgroup` or `device`; `simdgroup_store(...)` same shape; `simdgroup_multiply_accumulate(d, a, b, c)` (d = a*b + c); `simdgroup_multiply(d, a, b)`; `make_filled_simdgroup_matrix(v)`; diagonal ctor `simdgroup_matrix<T,8,8>(v)` [vendor documented: MSL 4.1 §6.8, Tables 6.9/6.10] |
| Element-to-lane map | Unspecified by Apple. Never index fragment registers; go through `simdgroup_store` to threadgroup scratch. [vendor documented: MSL 4.1 §2.4] |
| Control flow | All 32 lanes must execute each op under uniform control flow, else undefined [vendor documented: MSL 4.1 §2.4] |
| Availability | "SIMD-scoped matrix multiply operations" from Apple7 [vendor documented: Feature Set Tables] |
| Hardware | No dedicated matrix unit on M1–M4. hwspec assumes matrix rate = FP32 FMA rate (`APPLE_G16_MMA` fp16 = bf16 = 128 MAC/core/clk, fp8/fp4/int8 = `None`, `m4.rs`); uncalibrated (UNVERIFIED: measure fp16 and bf16 ceilings separately). Cost model shape 8x8x8 only (`crates/costmodel/src/mma.rs` `APPLE`). Latency/throughput per op: not published (UNVERIFIED) |
| Mixed precision | Plow multiplies `simdgroup_bfloat8x8` operands into a `simdgroup_float8x8` accumulator. MSL Table 6.10 writes one `T` for all operands; mixed bf16→f32 compiles and is lockstep-exact in plow [measured in plow: 56ec6e6e9]. Spec coverage of the mixed form (UNVERIFIED) |

Plow call sites (`runtime/apple/interp.metal`):

* `gemm_sub<SM,SN,RB,CB>` + `frag_load` — prefill GEMM. Tile per simdgroup RB x CB 8x8 f32
  accumulators (128-row sub-tile: 2x4), bf16 fragments from threadgroup staging, B loaded with
  `transpose = true`; K staged `GK = 32` per chunk, double-buffered.
* `gemm_sub_glu` — gate/up GLU on the same machinery (64-wide halves).
* `epilogue8x8` — `simdgroup_store` one accumulator to a 64-float per-simdgroup scratch, then
  `simdgroup_barrier(mem_threadgroup)`, lane l finishes elements 2l, 2l+1 (scales, bias, row map).
* `op_flash_prefill<T>` (bf16 and e4m3 KV; op 39 = the fp8 instance) — QK^T and PV on 8x8 tiles, hd 64/128/256/512
  [measured in plow: a27ea350a].
* `q8.metal`, `asr.metal`, `asr_subsampling.metal` — `simdgroup_float8x8` tiles for ASR encoders.

Pattern (from `gemm_sub`):

```metal
simdgroup_float8x8 acc(0.0f);
simdgroup_bfloat8x8 a, b;
simdgroup_load(a, As + 64u * kb, 8u);
simdgroup_load(b, Bs + 64u * kb, 8u, ulong2(0, 0), true);  // B stored [N][K]
simdgroup_multiply_accumulate(acc, a, b, acc);
```

### 2.2 Metal 4 tensors and `tensor_ops::matmul2d` (not used by plow)

* MSL 4.0+ `mpp::tensor_ops::matmul2d` / `cooperative_tensor` [vendor documented: MSL 4.1 §2.22,
  §7]. The engine pins `Version3_2`, so they do not compile today. Using them needs `Version4_0`
  and macOS 26; the locked `objc2-metal 0.3.2` already exposes `Version4_0`.
* Available from Apple7 [vendor documented: Feature Set Tables]; the per-core Neural Accelerators
  are M5 (Apple10) [vendor documented: Newsroom M5]. Performance on M4 vs `simdgroup_matrix` is
  (UNVERIFIED): needs a matched M4 benchmark.
* dtype support per OS: MSL 4.1 Table 7.3.

### 2.3 Precision support

| dtype | Storage | Arithmetic on Apple9 | Plow handling |
|---|---|---|---|
| f32 | yes | native | accumulators, softmax, norms |
| half | yes | native | ASR paths |
| bfloat | `bfloat`, `bfloatN`, `packed_bfloatN` MSL 3.1+; `MTLDataType.bfloat` Apple6+ [vendor documented: MSL 4.1 §2.1, Feature Set Tables] | Native bf16 ALU rate (UNVERIFIED) | weights/activations held as `ushort`, widened with `bf2f`/`bf4` (`<< 16`), rounded back with `f2bf` (RNE); `bfloat` only as `simdgroup_bfloat8x8` operands |
| fp8 e4m3/e5m2 | no shader scalar type; TensorOp formats only (Metal 4.1) | **no FP8 hardware** | software decode `e4m3_exact`, `e4m3bf4` (exact to bf16), encode `f2e4m3` (`interp.metal`) |
| fp4 e2m1 (MXFP4) | no shader type | none | LUT `E2M1[]` + E8M0 scale folded at staging (`load_b16`) [measured in plow: 97efc7d82] |
| int8 | `char`/`uchar` | integer ALU; no int8 dot instruction documented | w8 paths in `q8.metal` |

## 3. Memory movement

* No async copy engine visible to shaders (no TMA / `cp.async` / LDS-DMA analogue). Overlap is
  done in registers: issue next chunk's `device` loads, compute on current `threadgroup` buffer,
  then store the registers into the other buffer (`gemm_sub`: `ra/rb` prefetch, `stage_put`)
  [measured in plow: interp.metal GEMM comment, 56ec6e6e9].
* Vector width: plow streams weights as `ushort4` (8 B) / `uchar4` / `float4` per lane; GEMV
  lanes take 8 bf16 per step with a 256-element simdgroup stride (`dot_glu_bf16_pair`,
  `k = lane * 8; k += 256`). The unnormalized bf16 GEMV main loop takes four weight rows per
  simdgroup; tails and normed paths go one row at a time (`interp.metal` `dot4_bf16`, `norm == 0`)
  [measured in plow: 8682fc356]. Widest efficient load width: not published (UNVERIFIED).
* Alignment: the compiler assumes pointer arguments are aligned to their type
  [vendor documented: MSL 4.1 §2.5] — casting `ushort*` to `ushort4*` requires 8 B alignment
  and `K % 4 == 0`; plow guards with `(K & 3u) == 0u` and a scalar tail.
* `simdgroup_load/store` take `elements_per_row` (stride) and `transpose_matrix`. GEMM staging
  stores 8x8 blocks contiguously (stride 8); flash prefill and `op_q8_gemm_f32` load with stride 32/64. Pass
  the real row stride.
* Threadgroup-memory banking: not documented by Apple (UNVERIFIED). Plow avoids relying on a bank
  model: stage in 8x8 blocks and keep per-simdgroup scratch disjoint.
* Unified memory, zero copy: model tensors are host allocations wrapped with
  `newBufferWithBytesNoCopy` (`StorageModeShared`) when the pointer is 16 KiB page-aligned and the
  length is rounded to a page (`PAGE = 16384`); otherwise a shared copy [measured in plow:
  `crates/plowrt/src/exec/apple/mod.rs`]. Tensor buffers (reached indirectly) go into one
  `MTLResidencySet` on the queue, fallback per-encoder `useResource`; directly bound buffers are
  tracked by the encoder (residency sets: Apple6+ [vendor documented: Feature
  Set Tables]).
* Kernels reach tensors through a `device const ulong* tab` of GPU addresses indexed by handle
  (`ten<T>()`), not bound buffers.
* Cache hints / non-temporal loads: no MSL-level control (UNVERIFIED).

## 4. Synchronization and ordering

| Primitive | Meaning | Plow use |
|---|---|---|
| `threadgroup_barrier(mem_flags::mem_threadgroup)` | execution barrier + orders threadgroup memory for the threadgroup | after staging, reductions (`tg_sum`), GEMM chunk swaps |
| `threadgroup_barrier(mem_flags::mem_device)` | also orders device memory **within the threadgroup only** | end of every walked op in `plow_interp` |
| `simdgroup_barrier(mem_flags::mem_threadgroup)` | SIMD-group-scope barrier (Apple2+) | `epilogue8x8` after `simdgroup_store` |
| `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)` | device-scope fence; MSL **3.2+** on Apple silicon | acquire after the wait loop and release before bumping successors in `plow_interp` |
| `atomic_load_explicit / atomic_fetch_add_explicit(..., memory_order_relaxed)` on `device atomic_uint` | relaxed atomics | wait counters and successor bumps |

Vendor rules [vendor documented: MSL 4.1 §6.10.1, §6.16]:

* Every thread of the threadgroup (SIMD-group) reaches every barrier; keep barriers on uniform
  control flow.
* Atomics take only `memory_order_relaxed` before MSL 4.1; device-scope
  `atomic_thread_fence(..., seq_cst, thread_scope_device)` needs MSL 3.2.
* Atomics: 32-bit int on device/threadgroup; `atomic_float` on device (Apple7+). 64-bit: device
  `atomic_ulong` with `atomic_min/max_explicit` only; no 64-bit fetch-add or CAS [vendor
  documented: MSL 4.1 §6.16.4, Tables 6.27–6.28].
* SIMD-group functions (`simd_shuffle*`, `simd_broadcast`, `simd_sum/max/min/prefix_*`,
  `simd_ballot/all/any`) need no barrier and take no `bfloat`/`bool`/`long`: widen to float
  [vendor documented: MSL 4.1 §6.10.2].

Plow's cross-threadgroup protocol inside one dispatch (`plow_interp`):

```metal
while (atomic_load_explicit(ctr + wt.id, memory_order_relaxed) < wt.threshold) { if (++it > P.spin_max) ... }
threadgroup_barrier(mem_flags::mem_threadgroup);          // publish the gate flag
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);  // acquire
/* op */
threadgroup_barrier(mem_flags::mem_device);
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);  // release
if (lid == 0) atomic_fetch_add_explicit(ctr + succ, 1u, memory_order_relaxed);
```

Plow's payload pointers are unqualified `device T*`; that the fences make them visible across
threadgroups is measured (pitfall 1), not spec-guaranteed: MSL 4.1 §4.8 gives plain `device`
memory threadgroup coherence and offers `coherent(device)` (UNVERIFIED which is required; settle
with probe (f) on a Metal toolchain).

Host side: one command buffer per GPU-only program run (CPU/ANE offload submits one per GPU
range, `mod.rs` `dispatch_range`), host sync with `waitUntilCompleted` (Event mode);
the host never spins on GPU state mid-kernel (`exec/apple/mod.rs` header). Kernel boundaries make
all stores visible (`plow_single` diagnostic relies on it).

## 5. Toolchain as plow uses it

* **No offline Metal compile.** No `xcrun metal`, no `.metallib`, no Metal entry in
  `runtime/CMakeLists.txt` or `crates/kernelcaps`. MSL is embedded with `include_str!` and
  compiled at engine load by `MTLDevice::newLibraryWithSource_options_error`.
* Compile options (every Apple loader: `exec/apple/mod.rs`, `q8.rs`, `asr.rs`, `asr_conformer.rs`,
  `asr_subsampling.rs`):
  * `setMathMode(MTLMathMode::Safe)` — fast math is the compiler default [vendor documented:
    MSL 4.1 §1.6.3 `-fmetal-math-mode`, default fast].
  * `setLanguageVersion(MTLLanguageVersion::Version3_2)` — needed for device-scope
    `atomic_thread_fence`; the default follows the SDK the binary links against, and the nix SDK
    is older [measured in plow: `exec/apple/mod.rs` comment]. MSL 3.2 = macOS 15 minimum
    [vendor documented: MSL 4.1 §1.6.10].
* Specialization by source prefix, not function constants: `#define PLOW_QKV_DOT4 1`,
  `PLOW_DECODE_HEADS`, `PLOW_GLU_PAIR`, `PLOW_BF16_M2/M4/M8` are prepended per
  `PLOW_METAL_*` runtime knob (registered in `crates/plowrt/src/knob_spec.rs`).
  `mlp_channel.metal` is concatenated when `--ane-mlp` is on.
* Load-time pipeline checks are loader-specific. The interpreter (`mod.rs`) checks
  `threadExecutionWidth() == 32`, `maxTotalThreadsPerThreadgroup() >= 1024` and
  `staticThreadgroupMemoryLength() <= device.maxThreadgroupMemoryLength()` and returns an
  **error** on failure; the optional `plow_mx4_prefill` pipeline is dropped instead; conformer/RNNT
  packet pipelines fall back to the interpreter; ASR encoder tiles fall back 32x64 → 16x32 → 8x8
  (`docs/runtime/asr.md`); `asr_subsampling.rs` checks only the pointwise pipeline, not
  `asr_causal_conv3_f16`. Add width/threads/threadgroup-memory checks to any new loader.
* Build: `cargo build -p plowrt --features metal` (or `metal,dist`, `ane`) on macOS inside
  `nix develop`; `flake.nix` `plowrt-release` adds `metal` on Darwin. Emit:
  `plowc --gpu m4pro ...` (recipe header comments).
* Probes (plain clang, Objective-C, compile MSL at runtime, no Xcode Metal toolchain needed):
  * `runtime/apple/probe/probe.m` — `clang -fobjc-arc -O2 -framework Metal -framework Foundation probe.m -o probe`;
    modes `coresident` (b), `counters` (c), `live` (a), `events` (d), `gemv` (e), visibility (f).
  * `runtime/apple/probe/mslcheck.m` — compile a `.metal` with the system compiler and print
    `maxThreads` and static threadgroup memory: `./mslcheck runtime/apple/interp.metal plow_interp`.
    It sets `MTLMathModeSafe` but no language version (SDK default, not the engine's `Version3_2`),
    so its verdict does not transfer to the engine compile.
  * `dtypecheck.m`, `fp8castcheck.m`, `qkvcheck.m`, `mx4*check.m`, `glucheck.m` — per-kernel
    numerics checks against the CPU golden C (`-I../../cpu/dev -I../../common`).
* Correctness/perf harnesses: `crates/plowrt/examples/apple_lockstep.rs` (per-op vs CPU kernels,
  `--time` per-op GPU profile), `apple_diff.rs`, `metal_gemm_bench.rs` (real-weight GEMM/GEMV
  cells vs CPU tier), `apple_prefill_calibrate.rs`, `apple_split_bench.rs`.
* Inspecting the object: Apple ships no disassembler; a community reverse-engineered one
  (dougallj/applegpu, G13 = M1) exists, Apple9 coverage (UNVERIFIED). Use pipeline reflection
  (above) for resources, Xcode GPU frame capture / Metal System Trace for timing and occupancy
  (UNVERIFIED as plow workflow: not referenced in the repo).

## 6. Pitfalls

1. Plain stores from one threadgroup are invisible to other threadgroups for the rest of the
   dispatch unless the producer issues a device-scope release fence; `threadgroup_barrier(mem_device)`
   alone is not enough, and the consumer must fence too or it reads a stale L1 line for any address
   its core touched earlier in the dispatch. [measured in plow: 8682fc356, `runtime/apple/interp.metal` `plow_interp` "probe (f)"]
2. Device-scope `atomic_thread_fence` needs MSL 3.2; the compile's default language version
   follows the linked SDK (older under nix), so always set `Version3_2` explicitly.
   [measured in plow: `crates/plowrt/src/exec/apple/mod.rs`] [vendor documented: MSL 4.1 §6.16.3]
3. No forward-progress guarantee before Apple11: a persistent grid that spins on other
   threadgroups deadlocks if they are not co-resident. Grid = packet `n_cu`; when that differs from the
   registry core count for the device name, dispatch is instruction-ordered (`serial`). No
   co-residency probe guards the walk at load. [vendor documented: Feature Set
   Tables "Compute shader forward progress guarantees" Apple11] [measured in plow: `exec/apple/mod.rs` `matching_geometry`; `docs/DISTRIBUTION.md`]
4. The cooperative walk stalled on a real ASR packet (recording 52 of 100) and left a residual
   wait timeout; packets that must progress without concurrent workgroups set
   `ordered_dispatch`. [measured in plow: `docs/runtime/asr.md`, `exec/apple/mod.rs` `set_ordered_dispatch`]
5. A wait that never resolves holds every GPU core, and the same GPU drives the display: at the old
   `1<<26` spin budget a bad packet froze the Mac for 8 s (120 ns per iteration). Default is
   `1<<22` (~0.5 s), `--metal-spin-max` / `PLOW_METAL_SPIN_MAX`; faults report `0x80000000 | inst`.
   [measured in plow: 8c97a41e4]
6. The wait-gate flag written by thread 0 needs a threadgroup-scope barrier before other threads
   read it. [measured in plow: 199e01e2e]
7. FP32 denormals flush on Apple9: a bit-trick e4m3 decode that rebiases after multiplying loses
   subnormal products; decode to an exact float first (`e4m3_exact`). [measured in plow: a27ea350a, `interp.metal` comment] [vendor documented: MSL 4.1 §8.1; Feature Set Tables "FP32 denormalized numbers" Apple11]
8. Library `tanh` returns NaN (inf/inf) for |x| > ~44 in safe-math mode; Gemma GELU arguments
   reach 1e3. Use `tanhs(x) = 1 - 2/(1+exp(2x))`. [measured in plow: `interp.metal` `tanhs`, 8682fc356]
9. Fast math is the compiler default and assumes no NaN/Inf; plow kernels rely on NaN sentinels
   (`NEG_INF`, unused-expert `NAN`). Keep `MTLMathMode::Safe`. [vendor documented: MSL 4.1 §1.6.3, §8.1]
10. SIMD-group functions reject `bfloat`; keep bf16 as `ushort` and reduce in f32.
    [vendor documented: MSL 4.1 §6.10.2]
11. `simdgroup_matrix` ops under divergent control flow are undefined; tail handling must be by
    zero-fill in staging (`load_b16` returns zeros past K/N), not by skipping the op per lane.
    [vendor documented: MSL 4.1 §2.4]
12. Threadgroup memory is one 32 KiB `tile` (`TILE_FLOATS = 8192`). Aliases: `red` = last 32
    floats, `gate` = the float below it, `keys` (argmax) = first 8 KiB. An op that reduces must not
    touch the last 33 floats; an op that does argmax must not touch the first 8 KiB. The cost model
    reserves 2 KiB. [measured in plow: 56ec6e6e9; `interp.metal` `plow_interp`; `crates/costmodel/src/lib.rs`]
13. GEMM rate came from `simdgroup_matrix` 64x64 sub-tiles with vectorized K staging (128-row
    prefill 608 → 286 ms, 1bff9a4e5), then GEMM v3: bf16 8x8-block staging, K=32 chunks, 2x4
    accumulators, pipelined fragment loads (512x3072x3072 1.79 → 0.94 ms, 56ec6e6e9).
14. 128-wide output tiles leave cores idle on narrow projections: E4B 128x512x2560 1.02 → 0.53 ms
    with 64-wide sub-tiles. The emitter picks 128x128 (`GemmMed`) when that grid fills `n_cu`,
    else the small tile; never 256x256 (`devgen` `apple_prefill_tile`). [measured in plow: 9264aeed3, 031338ca1]
15. Flash decode on a single simdgroup cost 0.18 ms per key; spreading over all 32 simdgroups
    → 0.005 ms. [measured in plow: 48bd43c52]
16. bf16/fp8 GEMV K tails were wrong for `K % 256 >= 8` (the 256-element simdgroup stride).
    Test K values that are not multiples of 256. [measured in plow: fe97514e2]
17. The GPU and CPU share one memory bus: GPU 255 GB/s alone, CPU 239 GB/s alone, both together
    247.5 GB/s, so CPU column shares add nothing to decode (share = 0); ANE prefill offload is
    slower past ~4 ops (program swaps + segment boundaries). [measured in plow: `m4.rs` probe (e), eae8964d8, 1bff9a4e5]
18. A pipeline may come back with `maxTotalThreadsPerThreadgroup` below 1024 (register pressure);
    check it at load, never assume the 1024 limit. [vendor documented: Feature Set Tables footnote 5] [measured in plow: loader checks in `exec/apple/mod.rs`, `q8.rs`, `asr*.rs` (not uniform, see §5)]
19. Zero-copy wrapping needs page-aligned host tensors (16 KiB on Apple silicon); unaligned ones
    are silently copied, doubling memory. [measured in plow: `exec/apple/mod.rs` `PAGE`, `copied`]
20. Metal does not host-resample: the engine is device argmax only (`x_plow_sampling: "device_argmax"`).
    [measured in plow: d5456c109]
21. Metal 4 tensor ops and M5 Neural Accelerators sit behind MSL 4.0 / macOS 26; an M5 still maps
    to `IsaLevel::Metal3` in hwspec, so packets get no matrix-unit route on M5.
    [vendor documented: Newsroom M5; MSL 4.1 §7] [measured in plow: `crates/hwspec/src/isa.rs` `from_live_named`]

## 7. Roofline numbers (M4 Pro, 16-core)

| Ceiling | Datasheet | Plow value | Where |
|---|---|---|---|
| DRAM bandwidth, GPU | 273 GB/s | 255 GB/s measured (bf16 GEMV over 2 GiB, probe (e)) | `m4.rs` `bandwidth_measured`; `scripts/campaign/roofline.py` `m4pro` `bandwidth_measured_gbps = None` |
| DRAM bandwidth, CPU | n/a | 239 GB/s measured (8 P-cores) ; GPU+CPU 247.5 | `m4.rs` `cpu_bandwidth_measured` |
| bf16/fp16 matrix | not published by Apple | hwspec-derived 16 x 128 MAC x 2 x 1.578 GHz = 6.46 TFLOPS (UNVERIFIED clock and ALU count); `roofline.py` 7.0 TFLOPS | `m4.rs`, `scripts/campaign/roofline.py` |
| bf16 GEMM measured | — | 2.5 TFLOPS (M=128), 3.8 (M=512), 4.1 (GLU), Llama-3.2-3B w8a16 shapes | [measured in plow: 56ec6e6e9] |
| fp8 / int8 / fp4 | no hardware | `roofline.py` sets fp8 = bf16 = 7.0 (decode is bandwidth-bound; FP8 only halves bytes) | `roofline.py` |
| ANE | 38 TOPS (precision unstated; hwspec models int8) | 4.05 TFLOPS fp16 measured (CoreML GEMM 128x3072x8192) | `m4.rs` `ane_fp16_tflops_measured` |
| Decode reference | — | Llama-3.2-3B w8a16 19.4 ms/tok; Gemma-4-12B w8a16 68 ms/tok | [measured in plow: eae8964d8] |

No recipe carries a `[roofline]` table for `metal3`; `tuning/` has no Apple records (dropped as
unread in 3a030b408). `scripts/bench/calibrate_ceilings.py` uses torch/CUDA and does not run on
Metal: measure bandwidth with `probe gemv` and matrix rate with `metal_gemm_bench`, then hand a
ceilings JSON to `scripts/campaign/block_roofline.py --ceilings` (`roofline.py` `lookup_gpu`
maps `m4pro`/`apple`/`metal` to the M4 Pro entry).

Repo-vs-repo/vendor disagreements (flagged, not edited):

* `roofline.py` `m4pro` 7.0 TFLOPS at 1500 MHz vs hwspec 128 MAC/clk at 1578 MHz = 6.46 TFLOPS.
* `roofline.py` `m4max` = 40 cores / 546 GB/s; hwspec `APPLE_M4_MAX` = 32 cores / 410 GB/s.
  Both SKUs exist [vendor documented: Newsroom M4 Pro-Max]; they describe different parts.
* `roofline.py` `m4` 4.3 TFLOPS at 1450 MHz vs hwspec 1470 MHz (10 x 128 x 2 x 1.47 = 3.76).

## 8. Decision checklist

* Writing a new Metal kernel → add it to an existing `runtime/apple/*.metal`, compile it through
  the same `MTLCompileOptions` (Safe, `Version3_2`), and assert width 32 / max threads / static
  threadgroup memory on the pipeline before admitting it.
* Matrix work at M ≥ 8 → `simdgroup_bfloat8x8` operands, `simdgroup_float8x8` accumulators,
  RB x CB accumulators per simdgroup (2x4 at 128 rows), K chunk 32, register double-buffering.
* Decode GEMV (M ≤ 8) → no matrix ops; lane-strided `ushort4` loads, four weight rows per
  simdgroup, `simd_sum`; for batched decode reuse each weight load across rows
  (`PLOW_METAL_BF16_M2/M4/M8`). Quote % of the measured 255 GB/s.
* Sizing threadgroup memory → reuse the interpreter's 32 KiB `tile`; keep staging off the live
  `red`/`gate`/`keys` aliases (pitfall 12). If a tile needs more, shrink it; never add a second array.
* Cross-threadgroup handoff inside a dispatch → relaxed counter + device-scope seq_cst fence on
  both sides, exactly as `plow_interp` (measured, see §4); otherwise split into separate dispatches.
* Anything that spins → bound it by `spin_max` and write a fault code; never an unbounded loop.
* Need FP8/FP4 math → decode to bf16/f32 in staging (`e4m3bf4`, E2M1 LUT); expect bandwidth win only.
* Need bf16 in a SIMD reduction or shuffle → widen to float first.
* Considering Metal 4 TensorOps → benchmark against the `simdgroup_matrix` GEMM on M4, target M5
  separately; needs `Version4_0` plus macOS 26.
* Packet built for another core count → expect serial dispatch. Persistent dispatch needs the
  registry core count for the device name to equal packet `n_cu` (`mod.rs` `matching_geometry`);
  a 20-core M4 Pro resolves to 16, so a 20-core packet alone stays serial.
* Verifying a change → `apple_lockstep` (0 mismatches vs CPU golden), then `metal_gemm_bench` on
  the model's real shapes, then end-to-end ms/tok.
* Offloading to CPU or ANE → assume no gain for decode (shared bus); re-measure with
  `apple_split_bench` / `apple_calibrate` before enabling.

## 9. Sources

| Topic | Document | Section | Version / date |
|---|---|---|---|
| Family map, feature availability, limits | Metal Feature Set Tables, https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf | "Metal GPUs", feature availability, "GPU implementation limits by family", tensor limits | October 23, 2026 |
| MSL language | Metal Shading Language Specification, https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf | §1.6.3 math modes, §1.6.10 language versions, §2.4 simdgroup_matrix types, §2.22 tensors, §6.8 simdgroup matrix functions, §6.10 barriers and SIMD/quad functions, §6.16 atomics/memory model, §7 MPP TensorOps, §8 numerics | Version 4.1, 2026-06-04 |
| Metal API | https://developer.apple.com/documentation/metal | `MTLComputePipelineState`, `MTLCompileOptions`, `newBufferWithBytesNoCopy`, `MTLResidencySet` | fetched 2026-11-03 |
| M4 | https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/ | GPU cores, Dynamic Caching, Neural Engine 38 TOPS | 2024-05 |
| M4 Pro / Max | https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/ | GPU cores, 273 / 546 GB/s | 2024-10 |
| M5 Neural Accelerators | https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/ | per-core Neural Accelerator, Metal 4 Tensor APIs | 2025-10 |
| Dynamic Caching / occupancy | Apple tech talk "Explore GPU advancements in M3 and A17 Pro", https://developer.apple.com/videos/play/tech-talks/111375 | shader core memory, occupancy | 2023-11 |
