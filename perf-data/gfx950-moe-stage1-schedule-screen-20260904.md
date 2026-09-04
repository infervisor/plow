# gfx950 MXFP4 MoE stage-1 schedule screen

Date: 2026-09-04

## Decision

Close the schedule-only axis. No candidate cleared the required 10% isolated gate, so none is
routed through `plowrt` and no TP8 endpoint run was spent on it. The best exact candidate was the
SiTU-specialized epilogue at 2.091247 ms mean versus 2.115963 ms paired control, only 1.17% faster
and about 2.27 ms over 92 layers. The current network attribution is 212.579 ms over 92 stage-1
segments (2.311 ms/layer).

The next stage-1 experiment must change data movement: produce a sorted A4 activation tile once,
reuse it across all three local-N tiles, and use a native four-wave kernel with A-only LDS
ping-pong and register-pipelined preshuffled weights. Keep Plow's fused SiTU-to-MXFP4 epilogue.

## Exact isolated contract

All valid rows below use one leased MI355X, gfx950 wave64, T=8192, H=3584, local I=384,
E=896, top-k=16, BM64 expert padding, production row-major MXFP4 weights/E8M0 scales, and
151,232 sorted padded rows. Every timed candidate first matched all 29,036,544 output payload
bytes and 1,814,784 scale bytes. Each mean is 31 alternating, compute-cache-flushed samples.
The control and candidate means shown on one row came from the same process.

### Persistent-grid and priority screen

The control grid is 256 workgroups. The candidate uses the shipping BM64/BN256/BK256 body and
linear persistent schedule.

| candidate | grid | control mean (ms) | candidate mean (ms) |
|---|---:|---:|---:|
| priority on | 64 | 2.173091 | 8.170254 |
| priority on | 128 | 2.151002 | 4.147616 |
| priority on | 192 | 2.119418 | 2.788708 |
| priority on | 256 | 2.108619 | **2.108430** |
| priority on | 320 | 2.135521 | 3.267379 |
| priority on | 384 | 2.127318 | 2.778709 |
| priority on | 512 | 2.111673 | 2.119243 |
| priority on | 768 | 2.111634 | 2.142157 |
| priority on | 1024 | 2.113497 | 2.125387 |
| priority off | 128 | 2.154199 | 4.154888 |
| priority off | 256 | 2.111097 | 2.115244 |
| priority off | 384 | 2.132629 | 2.782436 |
| priority off | 512 | 2.114113 | 2.121680 |

Grid 256 is the only optimum. Non-multiples of 256 leave a tail residency wave, while smaller
grids execute multiple persistent rounds. Removing the MFMA priority window is neutral-to-worse.

### Tile-width screen

| tile / implementation | best grid | control mean (ms) | candidate mean (ms) | result |
|---|---:|---:|---:|---|
| BN128/BK256, fully overlapped | 512 | 2.119813 | 3.066635 | +44.7% |
| BN384/BK256, low-register | 256 | 2.117759 | 2.240055 | +5.8% |
| BN512/BK256, low-register | 512 | 2.125811 | 2.634920 | +23.9% |
| BN256/BK128 | 512 | 2.118075 | 2.700206 | +27.5% |
| BN256/BK64 | — | — | — | invalid layout |

For completeness, the wider-N mean matrix was BN128: 9.849610/5.028789/4.043879/3.066635 ms,
BN384: 4.269435/2.240055/2.841509/2.246030 ms, and BN512:
4.858798/2.637012/3.283594/2.634920 ms at grids 128/256/384/512 respectively. BK64 produced
11,899,223 wrong payload bytes and 2,088 wrong scale bytes because the current MXFP4 XOR row
swizzle requires at least a 64-byte row (BK >= 128); it was not timed.

### XCD/WGM, cache hint, and epilogue screen

These runs use grid 512 for both arms.

| candidate | control mean (ms) | candidate mean (ms) |
|---|---:|---:|
| linear, ordinary loads | 2.118289 | **2.116053** |
| linear, non-temporal loads | 2.118609 | 2.135387 |
| WGM1, ordinary / non-temporal | 2.119084 / 2.118874 | 2.491708 / 2.477135 |
| WGM2, ordinary / non-temporal | 2.117804 / 2.117975 | 2.513976 / 2.507976 |
| WGM4, ordinary / non-temporal | 2.116302 / 2.119449 | 2.507913 / 2.512266 |
| WGM8, ordinary / non-temporal | 2.119835 / 2.119661 | 2.514299 / 2.514330 |
| bridge LDS alias only | 2.115556 | 2.115565 |
| SiTU specialization | 2.115802 | 2.091917 |
| SiTU specialization + alias | 2.115963 | **2.091247** |

The AITER-inspired WGM/XCD reorder regresses 16.9-18.7%. Plow's linear persistent order better
preserves reuse for its current row-major expert weights. Non-temporal loads are also harmful.

## Compiler resource gate

All timed objects are wave64 with zero private segment, zero VGPR spills, and zero SGPR spills.
Occupancy is the compiler's wave/SIMD ceiling; LDS may reduce effective workgroup residency.

| candidate | threads | dynamic LDS | VGPR | SGPR | occupancy |
|---|---:|---:|---:|---:|---:|
| shipping / priority off | 512 | 119,808 B | 190 | 90 | 2 |
| WGM1 | 512 | 119,808 B | 190 | 93 | 2 |
| WGM2/4/8 | 512 | 119,808 B | 190 | 94 | 2 |
| BN128 fully overlapped | 256 | 52,224 B | 205 | 89 | 2 |
| BN384 low-register | 768 | 121,856 B | 117 | 78 | 4 |
| BN512 low-register | 1024 | 156,672 B | 117 | 72 | 4 |
| BN256/BK128 | 512 | 76,288 B | 158 | 85 | 3 |
| BN256/BK64, invalid | 512 | 54,528 B | 140 | 88 | 3 |
| SiTU specialized | 512 | 87,040 B | 188 | 83 | 2 |

BN384 raises the register occupancy ceiling but its 12-wave workgroup and 121,856-byte LDS
allocation still permit only one workgroup/CU. The result confirms that reduced register count
alone does not replace the overlap lost by the low-register pipeline. BN128 retains overlap but
doubles the N work and raises VGPR pressure.

## Why the 2.11 ms floor remains

- The harness executes 7,089 `(expert M tile, N tile)` tasks over 151,232 padded rows. Logical
  rows are 131,072, so routing/padding fill is 86.7% before kernel inefficiencies.
- Nominal padded GEMM work is about 832.5 GFLOP. At 2.116 ms this is about 393 TFLOP/s, only
  about 6% of the repository's 6.6 PFLOP/s scaled-MFMA ceiling. MFMA issue is not independently
  saturated.
- Per task, Plow gathers BF16 activation and quantizes it to MXFP4. Because local I=384 and
  NB=128, that activation work is repeated for three N tiles. Logical activation traffic is
  about 3.25 GB and logical weight/scale traffic about 3.46 GB per launch, or roughly 3.2 TB/s
  at the measured time, before LDS traffic and cache refetches.
- The current kernel double-buffers activation, weight, and both scale tiles in LDS (87,040 B),
  then reserves another 32 KiB bridge unless aliased. Grid 256 versus 512 is effectively tied,
  and XCD/WGM reordering is substantially worse, so idle CUs and grid ownership are not the
  remaining schedule lever.

## Exact-shape AITER comparison

Pinned AITER's gfx950 T8192 row selects
`flydsl_moe1_afp8_wfp4_bf16_t64x256x256_w3_bnt0_gui_xcd4` at 616.0674 us. The vLLM production
4096-token rung selects T64x128x256 at 409.5168 us, so two stage-1 bodies are about 0.819 ms per
layer, or 75.35 ms over 92 layers. This is not an apples-to-apples kernel number:

- AITER receives already sorted MXFP8 activation plus E8M0 scales and writes BF16 stage-1
  output. Its later stage-2 preparation runs a separate fused dynamic MXFP8 quant/sort.
- It uses 256 threads/four waves, T64xN256xK256, 32 KiB of X-only ping-pong LDS, and overlays
  the 32 KiB cshuffle output on that allocation. Weight tiles are prefetched global-to-register;
  they are not staged through LDS.
- Plow receives BF16 activation, performs its input A4 quantization inside this kernel, stages
  A and B through LDS, and fuses SiTU plus exact MXFP4/E8M0 output quantization for stage 2.

Even treating AITER's two 4096 bodies as a lower bound gives a category ceiling of about
137.2 ms versus Plow's current 212.579 ms stage-1 attribution, before charging AITER's separate
quant/sort work. That gap cannot be reached by the screened grid, wave, priority, or tile-width
switches. It requires the A-only-LDS/register-B architecture above, or the larger generic
expert-parallel prefill design that improves both stages' K depth and weight footprint.
