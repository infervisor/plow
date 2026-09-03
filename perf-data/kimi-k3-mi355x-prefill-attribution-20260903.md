# Kimi-K3 BF16 prefill attribution — MI355X TP8

## Result

This is the current committed production graph at
`91a6d24fbacf45d0bfcb4a9700789d25b896287c`, not the rejected KDA intra-segment
candidate. One 8192-token prompt and one generated token took **1783.545 ms**.
The slowest of eight post-completion device traces spans **1608.117 ms**. The
default host-staged recurrent-state clear accounts for another **161.231 ms**;
their 1769.348 ms sum is within **14.197 ms (0.796%)** of endpoint TTFT and passes
the 2% accounting gate.

Device recurrent-state clear is exact and removes **160.998 ms (9.04%)** from
TTFT, so it is now the AMD runtime default. `PLOW_STATE_CLEAR_DEVICE=0` is the
rollback.

| 8192 -> 1, C1/N1 | serial clear | device clear |
|---|---:|---:|
| TTFT mean, 3 alternating folds | 1781.931 ms | **1620.933 ms** |
| TTFT standard deviation | 1.710 ms | 0.285 ms |
| `begin_slot` mean | 161.231 ms | **0.064 ms** |
| output checksum, all folds | `fnv1a64:7d749e3b002fafa7` | same |

The retained vLLM BF16 TP8 baseline is 568.35 ms, so the new default remains
**2.85x / 1052.58 ms** behind. The gap is now device work, not admission clear.

## Current device attribution

All ranks contain 2787 traced ops. Rank spans are 1608.088, 1608.117, 1608.092,
1607.939, 1607.905, 1607.908, 1607.717, and 1607.717 ms. Rank 1 is critical;
max-minus-min skew is 0.400 ms (0.025%). Its critical envelope contains 6.268 ms
gate time, 1309.472 ms traced bodies, and 292.377 ms between traced interpreter
segments. The latter is principally the separately launched lean MoE experts.

| category | critical time | share |
|---|---:|---:|
| KDA scan + conv + norm | 368.129 ms | 22.89% |
| standalone MoE expert kernels | 290.018 ms | 18.03% |
| dense GEMM/GEMV | 246.783 ms | 15.35% |
| MoE routing/align/combine | 228.952 ms | 14.24% |
| TP reductions | 218.377 ms | 13.58% |
| MLA attention/merge/gate | 145.614 ms | 9.05% |
| AttnRes | 89.738 ms | 5.58% |
| other traced ops | 18.146 ms | 1.13% |
| other segment gap | 2.359 ms | 0.15% |

| op | count | gate | body | total |
|---|---:|---:|---:|---:|
| XReduce2 | 278 | 0.864 ms | 217.513 ms | 218.377 ms |
| KdaChunkCarry | 69 | 0.053 ms | 154.376 ms | 154.429 ms |
| GemmWide | 740 | 1.582 ms | 147.234 ms | 148.816 ms |
| MoeCombinePf | 92 | 0.123 ms | 146.532 ms | 146.655 ms |
| KdaChunkIntra | 69 | 0.186 ms | 122.167 ms | 122.354 ms |
| FlashMlaPrefill | 24 | 0.039 ms | 110.541 ms | 110.580 ms |
| AttnRes | 187 | 0.499 ms | 89.239 ms | 89.738 ms |
| GemmC5 | 349 | 0.857 ms | 83.197 ms | 84.054 ms |
| MoeRouterTopkPf | 92 | 0.095 ms | 52.109 ms | 52.204 ms |
| KdaChunkWu | 69 | 0.194 ms | 43.634 ms | 43.828 ms |
| MlaMergeFold | 24 | 0.163 ms | 33.580 ms | 33.743 ms |
| MoeAlignPf | 92 | 0.048 ms | 30.046 ms | 30.094 ms |
| KdaConv3 | 69 | 0.097 ms | 20.878 ms | 20.975 ms |
| KdaChunkPrepare | 69 | 0.192 ms | 17.004 ms | 17.196 ms |
| remaining traced ops | 901 | 2.271 ms | 41.024 ms | 43.295 ms |

## Configuration and routing audit

- Exact BF16 TP8: weights/activations/KV are BF16; experts are checkpoint MXFP4.
- TuneDB fingerprint: `gfx950-76ef5b9982d04cbd`; 7650/7650 dense-GEMM tile
  lookups are measured. All seven programs have Lean ordering/LDS certificates;
  `lean.verified=true`, `lean.oracle=true`.
- T8192 has 325 ordered segments: 92 singleton `MoeGroupGluPf`, 92 singleton
  `MoeGroupDownPf`, 24 singleton `FlashMlaPrefill`, and 117 mixed segments.
- Actual defaults do not emit `PLOW_SEG_PACKED_PREFILL`. All 69
  `KdaChunkCarry/d128_qpre` and `KdaChunkWu/d128_qpre` sites remain in mixed
  primary-interpreter segments. The KDA family route cannot select the packed
  object despite being enabled at runtime.
- Primary `interp_prefill_k3_moe_a4w4_gq.elf`: wave64, 256 VGPR, occupancy 2,
  8 VGPR spills, 74 SGPR spills, 1348 B private/thread, 147504 B LDS.
  `interp_packed_kda_gq.elf`: wave64, 248 VGPR, occupancy 2, zero VGPR spills,
  25 SGPR spills, 440 B private/thread, 147504 B LDS. Routing qpre out of the
  primary object remains a material promotion gap.
- Raw MLA V2+SV object is wave64/4-wave, zero VGPR spill and zero private bytes.
  Required L2 placement, V2/raw, qpre, residual-input, packet ABI, lean MoE ABI,
  no-spill, and state-clear object markers were resolved before measurement.

Artifacts: packet SHA-256 `a4d8bee5d861daaa8b8db67e8bd8ff316480bf41c1d6d0c1ac4cc7dccf74db02`;
primary prefill object `56dd617a3ebb24de3efd884786abf3c2650c21a15924055f23ec2ff0959d6a1e`;
raw MLA object `28d4667dd68721b177c16335feaaa683992358a96aad04c75e5a276f1e80fb71`.

## Recurrent-state clear gate

Each rank has 276 carried tensors totaling 59,351,040 B (56.601562 MiB): 69
786432-byte recurrent matrices (54,263,808 B) and 207 24576-byte convolution
windows (5,087,232 B). The old path issued 276 blocking host-staged HSA SDMA
copies per rank, serially across TP8: 2208 blocking copies per request.

The device path creates 414 load-time descriptors per rank (three 256 KiB chunks
for each recurrent matrix plus one range per conv window), launches one 414-block
clear kernel concurrently on every rank, then drains. The 896 MiB `kv.blkres`
scratch tensor is correctly excluded.

Correctness gates:

- Three alternating 8192 -> 1 folds: six identical output checksums, 6/6
  complete, zero failures.
- 8192 -> 256 full generation: serial/device output IDs are byte-identical,
  checksum `fnv1a64:6bdfaa7b84ee4e7e`; TPOT is 44.342 vs 44.312 ms.
- True same-slot reuse, two requests, 8192 -> 16: both requests and both arms
  produce identical token rows, checksum `fnv1a64:31983f6844275241`; 4/4
  complete, zero failures. Mean TTFT is 1781.490 vs 1619.220 ms and TPOT is
  44.344 vs 44.304 ms.

The all-rank trace dump and bench phase dump were post-completion measurement
instrumentation in a private worktree. They changed no dispatch or timed path and
are not part of the production commit.

## Counter protocol reference

The current decode object carries sound per-XCD hierarchy and L2-placement
markers. The existing isolated gfx950 measurement remains applicable: one empty
b=256 packet is 13.16 us with one release/acquire counter, 8.95 us with
contention removed, 5.49 us with the unsound writeback deletion, and **3.46 us**
with one elected maintenance leader per XCD. No new empty-packet run was made in
this attribution campaign; the current prefill objects use L2 placement but not
the decode-only hierarchy marker.
