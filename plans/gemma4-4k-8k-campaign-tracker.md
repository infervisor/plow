# Gemma-4-12B 4K/8K native campaign tracker

Updated: 2026-09-13
Branch: `tp-bringup-mi300x`
Checkpoint: `google/gemma-4-12B-it`
Protocol: `g4-4k8k-v1`
Last qualified kernel commit: `68f72fe1`
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
| NVIDIA H100 80GB | SM90a | BF16 | 208.6 ms current-schema p50 | 459.8 ms current-schema p50 | Pure-GEMM production default, exact HD256, descriptor-TMA HD512, and fused-GLU roles qualified; 50% gate not met |
| NVIDIA H100 80GB | SM90a | W8A8/FP8 weights+activations | 168.3 ms current-schema p50 | 368.0 ms current-schema p50 | Pure-GEMM production default and ABI v5 descriptor TMA qualified; 50% gate not met |
| AMD MI300X | gfx942 | BF16 | unmeasured in this fixed protocol | unmeasured in this fixed protocol | Establish native block and serving baseline |
| AMD MI300X | gfx942 | FP8 | unmeasured in this fixed protocol | unmeasured in this fixed protocol | Establish dtype-correct block and serving baseline |

The H100 BF16 and W8A8 values are three-seed mean p50s for current-schema ABI v5
packets with their native pure-GEMM topology. They are not apples-to-apples vLLM
results and must not be compared with another precision or cache policy. An earlier
157.5/351.2 ms snapshot remains the best recorded packet, but predates the
current schema/default build and is not the control for new source changes.

## Cross-GPU checkpoint ledger

| Backend cell | Native kernels | Packet/runtime | Comparable baseline | Promotion state | Next gate |
|---|---|---|---|---|---|
| H100 SM90a BF16 4K/8K | HD256, descriptor-TMA HD512, fused gate/up+GeGLU qualified | pure-GEMM default, exact-rung roles, and packed R2 qualified | Plow production C1 recorded; matched vLLM pending | native production topology promoted; 50% TTFT open | non-divergent HD512 phase overlap, then matched cold C1/C8 |
| H100 SM90a FP8 4K/8K | W8A8 fused-GLU and descriptor-TMA HD512 qualified | ABI v5 role/hash, mixed BKV views, packed R2, and recorded pure-GEMM default qualified | current-schema native C1 recorded; matched vLLM pending | native production topology promoted | numerical FP8 gate, then cold C1/C8 |
| MI300X gfx942 BF16 4K/8K | no result under `g4-4k8k-v1` | shared planner/packing/VMM code present; driver gate pending | missing | unmeasured | capture four block cells and cold C1/C8 baseline |
| MI300X gfx942 FP8 4K/8K | no result under `g4-4k8k-v1` | shared planner/packing/VMM code present; driver gate pending | missing | unmeasured | qualify dtype path, then four block cells and cold C1/C8 |

`qualified` is architecture-local. A portable scheduler or packet change may be
shared, but its correctness and performance state remains pending until that
backend completes its own driver run.

### Required run record

Every new row or promotion must record:

- UTC run ID, branch commit, GPU SKU/UUID, architecture, driver, compiler, clocks, and power mode.
- Model revision, precision/quantization, TP, concurrency, exact query rung, live-KV bucket, cache state, and packed topology.
- Program digest, object SHA, kernel shape/config, registers, spills, shared memory, occupancy, and launch count.
- Median/p95, prompt/output hashes, numerical metric, rotated-seed result, and `gpulease` command/log location.

Do not enter a cross-GPU speedup without a same-cell control under protocol
`g4-4k8k-v1`. Missing fields keep the cell `unmeasured` or `provisional`.

### Durable evidence checkpoints

| Checkpoint | Architecture | Evidence |
|---|---|---|
| `6f5d9fe7` | H100 SM90a | accepted HD512 score-tile swizzle, ABI v4, role authentication |
| `51f77652` | H100 SM90a | post-swizzle NCU attribution, Q/K/V padding rejection, single-thread poll rejection |
| `c43f539dbae0137da1dc38ce3ec65096c8b6c8b7dbcdf9349cadd3f97fb64817` | H100 SM90a | qualified HD512 direct cubin SHA256, 122 registers, zero stack/spills, 110,096-byte arena |
| `68f72fe1` | H100 SM90a | exact 4K/8K HD512 K/V descriptor-TMA layout, ABI v5, mixed BKV16/BKV32 packet views |
| `ecd3feeaa853ad8547d8e05c6c61d2ccf09ba7b3317fb1b7147b21dbd5349072` | H100 SM90a | qualified ABI v5 direct cubin SHA256, 128 registers, zero direct-entry stack/spills, 110,592-byte arena |

Raw profiler reports, generated assets, and timing logs stay outside git. The
tracker stores enough identity to reject stale or cross-architecture evidence.

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
| HD512 descriptor-backed K/V TMA | M4096/M8192, BQ64/BKV16, rank-3 128B-swizzled maps | direct kernel -8.60%/-9.83% versus accepted ABI v4 object | six exact seed/rung checks, mixed 2K/4K/8K packet, and packed R2 pass |
| Pure-GEMM production topology | Gemma BF16 and W8A8 SM90a TP1, all prefill rungs | BF16 4K/8K -30.00%/-25.96%; W8A8 -45.09%/-42.04% versus accidental mixed packets | recorded defaults, explicit `=0` rollback, checkpoint K, exact default/explicit routing, and driver load pass |
| HD512 ABI v5 full-rung transfer | current-schema pure packet, M4096/M8192 | three-seed mean -1.51%/-2.21% versus ABI v5 row-TMA control | every paired prompt/output checksum matches |
| BF16 fused gate/up+GeGLU role | M4096/M8192, N15360/K3840 | five-seed full-rung -3.31%/-3.27% | matching hashes |

## Rejected H100 changes

| Candidate | Reason |
|---|---|
| HD512 WGMMA single/two-chain | large speedup, but only 72.40%/78.65% token agreement overall |
| HD512 TMA issue restricted to warp 0 | registers 117→121; HD512 subtotal +3.41%/+4.09% |
| HD512 Q/K/V row padding 0/16/24/32 | pad 24 is neutral (-0.01%/-0.12%); other strides regress 37–277% |
| HD512 single-thread TMA completion poll | registers 122→125; direct kernel +8.6%/+9.9% |
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

The descriptor-TMA HD512 direct object was profiled at M8192 with its production
grid (132 CTAs), block size (512 threads), and 110,592-byte arena. Direct A/B
timing against the accepted ABI v4 object is authoritative because the profiler
harness has a different compile context.

| Signal | Result | Decision |
|---|---:|---|
| Registers / occupancy | descriptor-TMA direct entry uses 128 registers/thread; 25% occupancy; one block/SM; zero stack/spills | Reject growth that lowers residency or fails full-rung transfer |
| Compute / memory | 44.14% compute; 46.70% memory; 0.69% DRAM; 98.20% L2 hit | Do not prioritize HBM bandwidth or GQA multicast for the full-query cell |
| Scheduler | eligible in 47.16% of cycles; 0.90 eligible warps/scheduler | The swizzled fragment layout removed a material dependency gap; phase overlap remains next |
| Direct timing | 4K 3.207861 -> 2.932128 ms; 8K 11.698603 -> 10.548309 ms | Promote for exact 4K/8K and require comparable full-rung transfer |
| Shared conflicts | score swizzle cuts total bank conflicts from 438,612,278 to 1,073,696 and measured load conflicts from 403,046,400 to zero | Move the next screen to Q/K and V `LDSM` dependencies |

The prior ABI v4 source counters localize the largest barrier sample to the Q/K
`LDSM.16.M88.2` load and the largest MIO samples to the scalar score-tile `LDS`
sequence. ABI v5 improves scheduler eligibility materially; its next source
screen must still preserve the accepted BKV16 score/PV reduction order.

## Next experiments

1. H100 HD512: test a non-divergent producer/consumer phase schedule after the qualified descriptor-TMA full-rung transfer.
2. H100 W8A8: qualify real-prompt logits/greedy agreement for the WS384 pure-GEMM numerics, then record the packet performance certificate.
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
