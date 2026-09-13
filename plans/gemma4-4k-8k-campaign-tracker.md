# Gemma-4-12B 4K/8K native campaign tracker

Updated: 2026-09-13
Branch: `tp-bringup-mi300x`
Checkpoint: `google/gemma-4-12B-it`
Detailed experiment log: `plans/gemma4-4k-8k-native-block.md`

## Goal and rules

- Cut exact 4K and 8K cold prefill TTFT by at least 50% on each target GPU.
- Use Plow-native kernels for the target result. Library kernels remain comparison ceilings.
- Compare identical model, precision, prompt tokens, output tokens, cache policy, TP, concurrency, and GPU clocks.
- Optimize and qualify individual rung blocks before full-model serving.
- Run every GPU measurement through `perf-data/tools/gpulease -n 1`.
- Keep raw logs and generated assets outside git. Record commands, hashes, medians, correctness, and decisions here or in the detailed log.
- Never transfer a performance result between architectures. Portable runtime changes still require a driver run on each backend.

## Target matrix

| GPU | ISA | Precision | 4K | 8K | Current state |
|---|---|---|---|---|---|
| NVIDIA H100 80GB | SM90a | BF16 | open | open | Exact HD256 and fused-GLU roles qualified; 50% full-model gate not met |
| NVIDIA H100 80GB | SM90a | W8A8/FP8 weights+activations | ~157.5 ms warmed p50 | ~351.2 ms warmed p50 | Latest qualified snapshot; correctness matched in accepted A/Bs; 50% gate not met |
| AMD MI300X | gfx942 | BF16 | unmeasured in this fixed protocol | unmeasured in this fixed protocol | Establish native block and serving baseline |
| AMD MI300X | gfx942 | FP8 | unmeasured in this fixed protocol | unmeasured in this fixed protocol | Establish dtype-correct block and serving baseline |

The H100 W8A8 values are the latest exact-rung snapshot after the accepted HD512
row-cooperative TMA issue path. They are not an apples-to-apples vLLM result and
must not be compared with another precision or cache policy.

## Portable packet/runtime state

| Item | State | Cross-device gate |
|---|---|---|
| Continuous batching | present | C1/C8/heavy-concurrency trace on H100 and MI300X |
| Packed cross-request prefill | present | single, homogeneous, ragged, prefix hit/miss |
| Unified prefill/decode token batching | present | route-fired trace and exact completion counts |
| Prefix cache | present | hit/miss/cancel/reuse correctness per backend |
| Demand-backed live-KV VMM | CUDA and gfx942 paths present | zero initial physical KV, slab growth, retirement/reuse |
| Segment packet counters | preserved by direct roles | queue equality, successor count, fault recovery |
| Live-KV attention object switching | missing | authenticated variant registry and mixed-packet policy |
| Compiled resource-aware devgen selection | missing | object SHA + registers/spills/smem/occupancy envelope |

## Qualified H100 changes

| Kernel/route | Exact scope | Result | Correctness |
|---|---|---|---|
| HD256 paired GQA2 direct role | M4096/M8192, HD256, window1024 | attention subtotal -3.70%; full-rung -0.47%/-0.32% | packed R2 and hashes pass |
| W8A8 fused-GLU direct role | M4096/M8192, N15360/K3840 | 48-site subtotal -1.15%/-2.32%; full-rung -0.78%/-0.72% | packed R2 and hashes pass |
| HD512 direct role | M4096/M8192, HD512, GQA16 | direct boundary removed; prior 8K sites -15.7% | three seeds and packed R2 pass |
| HD512 row-cooperative TMA issue | BQ64/BKV16, 512 threads | HD512 subtotal -6.12%/-6.82%; full-rung -1.23%/-2.12% | three seeds and packed R2 pass |
| HD512 score-tile bank swizzle | BQ64/BKV16, score stride 20 + row-parity column swizzle | HD512 subtotal -2.03%/-1.23%; full-rung -0.27%/-0.38% | three seeds, final ABI v4 driver run, and packed R2 pass |
| BF16 fused gate/up+GeGLU role | M4096/M8192, N15360/K3840 | five-seed full-rung -3.31%/-3.27% | matching hashes |

## Rejected H100 changes

| Candidate | Reason |
|---|---|
| HD512 WGMMA single/two-chain | large speedup, but only 72.40%/78.65% token agreement overall |
| HD512 TMA issue restricted to warp 0 | registers 117→121; HD512 subtotal +3.41%/+4.09% |
| HD256 BKV64 promotion | standalone win did not transfer; checksums changed |
| W8A8 GLU two WGMMA groups in flight | 48-site subtotal +1.59% |
| W8A8 GLU raster band 8 | packet subtotal +2.04%/+1.53% |
| W8A8 down projection NS3 | packet subtotal +4.29%/+0.94% |
| Generic coalesced GEMM epilogue | packet result +0.41%/-0.11% |
| Fused residual+next-norm opcode | slower and greedy checksums changed |

## Current architecture finding

The packet chooses exact 4K/8K programs, while the promoted attention roles use
one fixed object across all live-KV histories. Packed metadata supplies each
request's `qlen`, slot, and `kvlen`, so arithmetic is correct, but H100 HD512
still uses one BQ64/BKV16, 512-thread, 132-CTA, `nsplit=1` object. The runtime
does not switch object, split count, or launch geometry as global KV grows.
HD256 sliding attention saturates at its 1024-token window and needs less history
adaptation.

Required attention key:

`(arch, dtype, kv_dtype, query_rung, live_kv_bucket, head_dim, gqa, window, packed_topology)`

Each selected record must bind object SHA, program digest, BQ/BKV, threads/warps,
`nsplit`, shared memory, registers, spills, occupancy, and timing certificate.
The packet's dependency and counter ABI remains common across variants.

## Current H100 bottleneck evidence

The exact accepted HD512 direct object was profiled at M8192 with its production
grid (132 CTAs), block size (512 threads), and 108,048-byte arena. The standalone
11.91 ms duration matches the approximately 12 ms production site time.

| Signal | Result | Decision |
|---|---:|---|
| Registers / occupancy | accepted score-swizzle object uses 122 registers/thread; 25% occupancy; one register-limited block/SM | Reject growth that lowers residency or fails full-rung transfer |
| Compute / memory | 27.35% compute; 62.47% memory; 0.71% DRAM; 98.99% L2 hit | Do not prioritize HBM bandwidth or GQA multicast for the full-query cell |
| Scheduler | eligible in 29.26% of cycles; 0.54 eligible warps/scheduler | Reduce dependency and synchronization gaps |
| Stall per issued instruction | short scoreboard 4.71; barrier 2.85; wait 1.98; MIO throttle 0.91 | Repair shared-memory access and phase scheduling first |
| Shared conflicts | score swizzle cuts total bank conflicts from 438,612,278 to 1,073,696 and measured load conflicts from 403,046,400 to zero | Move the next screen to Q/K and V `LDSM` dependencies |

Source counters localize the largest barrier sample to the Q/K `LDSM.16.M88.2`
load and the largest MIO samples to the scalar score-tile `LDS` sequence. Any
candidate must preserve the accepted BKV16 score/PV reduction order.

## Next experiments

1. H100 HD512: screen the Q/K `LDSM` layout, then V's transposed `LDSM`; preserve reduction order and one block/SM.
2. H100 HD512: test a non-divergent producer/consumer phase schedule after the shared-memory screen.
3. H100 HD512: qualify live-KV bucket variants and `nsplit` only where the merge pass repays shorter slices; defer GQA multicast until a history-heavy cell shows DRAM pressure.
4. H100 GEMM: test ping-pong consumers, `stmatrix` + TMA output store, and operand multicast on exact Gemma dimensions.
5. Devgen/plowrt: add authenticated resource envelopes and live-KV attention variant selection.
6. MI300X: run the same four exact block cells and attribute GEMM/attention/light/dispatch before changing kernels.
7. Run matched H100 and MI300X BF16/FP8 full-model C1/C8/heavy-concurrency gates only after block winners qualify.

## Resume protocol

1. Pull `tp-bringup-mi300x` and read this tracker plus the detailed experiment log.
2. Confirm branch HEAD and GPU identity; acquire `gpulease`.
3. Reuse the last qualified object as control. Change one variable per candidate.
4. Require zero stack/spills, exact packet role/hash validation, numerical gate, three rotated timing seeds, packed ragged coverage, and full-rung transfer.
5. Commit and push only qualified code wins. Update rejected rows so failed experiments are not repeated.
