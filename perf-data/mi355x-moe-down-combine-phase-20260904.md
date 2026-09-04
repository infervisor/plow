# MI355X grouped-DOWN → deterministic-COMBINE phase screen (2026-09-04)

## Decision

Reject the phase object. Keep production routing unchanged.

The compiler-visible eligibility contract is model-independent: an adjacent
`MoeGroupDownFp8Blk → MoeCombine` edge, unique `part` ownership, identical routing table/top-k and
hidden dimensions, fixed-slot-order combine, gfx950, wave64, and a 256-workgroup placed grid.
There is no model or layer predicate. The current T1 graph contains 92 such seams. Removing one
dependency level at every seam has a 92 × 5.72 us = 0.526 ms/token protocol ceiling.

The larger `GemvQkvg → KdaConv3 → KdaStateStepG → KdaGatedNorm` candidate was considered first:
69 repetitions and three removed levels give a 1.184 ms/token protocol ceiling. It fails the
resource-compatible phase predicate before timing. Production `GemvQkvg` is WG512/grid256 with a
147,456-byte LDS arena; the qualified fused KDA object is WG256/grid12 and deliberately traps on a
different block/grid contract. One kernel cannot change workgroup width between phases. WG256
makes QKVG incomplete, while a new WG512 KDA implementation changes the numerical schedule and
reopens the already rejected fused-KDA kernel experiment. The KDA-only three-op form is that same
closed standalone route. This candidate therefore fails closed rather than being timed or routed.

## Safety and resources

`k3_moe_down_combine_xcd` retains the unchanged 256-workgroup grouped DOWN body. Each workgroup
reads `HW_REG_XCC_ID`; the host preflight refuses unless the device has 256 CUs and the launch maps
exactly 32 workgroups to each of eight physical XCDs. The last workgroup per XCD publishes that
partition, all eight leaders meet the unchanged global threshold, and only then do seven
workgroups run the fixed-order combine. A grid larger than one workgroup per CU is never admitted.

ROCm 7.14/gfx950 metadata:

| kernel | wave | WG | VGPR | SGPR | occupancy | private | spills |
|---|---:|---:|---:|---:|---:|---:|---:|
| grouped DOWN control | 64 | 512 | 91 | 58 | 5 | 0 B | 0/0 |
| DOWN→COMBINE phase | 64 | 512 | 94 | 63 | 5 | 0 B | 0/0 |

The candidate does not reduce occupancy and adds no private memory or register spills.

The follow-up XCD-owned object is 58 VGPR / 57 SGPR, occupancy 8 waves/SIMD, 4 B LDS, and zero
private memory or spills. A WG512 consumes eight waves, so metadata admits four workgroups/CU and
the grid768 launch requires three. Correctness does not depend on that admission: producer
workgroups never wait. Each exits after its local relaxed arrival; whichever workgroup arrives
96th on an XCD becomes the combine leader. Thus an arbitrary scheduling order still makes forward
progress. The host separately observes 96/96 workgroups on all eight physical XCD IDs.

## Hardware gate

Full emitted decode geometry: top-16, H=3584, I=384, E=896, MXFP4. Fifty-six rotating expert
sets keep the 1.83 GiB weight arena cold. The control is the two-launch grid-256 DOWN plus the
seven-workgroup deterministic combine. Timing is the median of 12 sets after warmup.

| arm | complete boundary | BF16 output differences |
|---|---:|---:|
| control | 18.247 us | — |
| phase | 27.286 us | 0 |
| phase - control | **+9.038 us (+49.5%)** | 0 |

Projected over 92 graph sites, the phase regresses TPOT by 0.832 ms. An earlier independent build
also regressed by 6.471 us/site (+0.595 ms/token), so this is not promoted and no TP8 network gate
is justified.

### XCD-owned grid768 follow-up

The second prototype remaps every eight-row hidden stripe to one physical XCD. Its 96 local
workgroups compute that stripe for every routed slot; followers retire, and the last local producer
combines only its disjoint rows in fixed slot order. No workgroup polls a global counter. Kernel/AQL
completion is the only global publication.

| arm | complete boundary | BF16 output differences |
|---|---:|---:|
| ordinary grid768 DOWN + combine | 14.126 us | — |
| XCD-owned grid768 phase | 46.467 us | 0 |
| phase - control | **+32.341 us (+229.0%)** | 0 |

The projected regression is +2.975 ms/token. Although it eliminates the global rendezvous and is
safe at grid768, hidden-stripe ownership destroys the production flattened `(slot,row)` schedule's
weight-stream locality. It is also far beyond the grouped standalone route's measured 0.94–0.96
ms/token segment-handoff tax. No TP8 network run is warranted.

### Flattened-owner ready queues

The final prototype preserves the ordinary flattened producer map exactly. For WG512/grid768,
fragment `f` executes on physical XCD `floor(f/8) % 8`. The narrow MXFP4 body has one fragment per
eight hidden rows and `ng=ceil(H/8)=448`; `ng % 64 == 0` proves that changing routed slot does not
change the fragment's owner XCD. Unsupported geometry fails closed.

There are exactly `ng/8=56` XCD-local ready counters. Producers never wait and issue no global
atomic. After all writes in their workgroup retire, each wave increments its row group's local
counter. The 16th fragment invalidates its own L1, reloads all 16 partials from the shared XCD L2,
and combines the eight rows in original slot order. Kernel completion is the only global release.

The object is wave64/WG512, 60 VGPR / 53 SGPR, occupancy 8, zero LDS/private/spills. The full-shape
output is byte-exact.

| arm | complete boundary | BF16 output differences |
|---|---:|---:|
| ordinary grid768 DOWN + combine | 14.093 us | — |
| flattened-owner ready phase | 58.116 us | 0 |
| phase - control | **+44.023 us (+312.4%)** | 0 |

The projection is +4.050 ms/token. Preserving weight locality fixes the prior ownership defect,
but 16 serialized arrivals for each of 448 row groups create 7,168 local atomics per layer. This
is much worse than both the 0.526 ms packet ceiling and the measured 0.94–0.96 ms standalone
segment-handoff loss. Stop this line; do not runtime-route or TP8-gate it.

## Consequence

One cross-XCD rendezvous costs more than one ordinary dependency boundary. A useful phase must
amortize a rendezvous across at least three adjacent levels and retain enough work per XCD to avoid
the grid-256 body penalty. The next architecture should assign tensor/expert stripes to persistent
XCD-local workers across several ops, keep local producer/consumer handoffs inside one L2, and
reconcile globally only at semantic reductions or collectives. Repackaging one edge at a time
cannot close the 7.72 ms/token vLLM gap.

The XCD-owned follow-up further narrows that recommendation: do not partition the expert weight
stream by output rows merely to make the consumer local. A credible redesign needs persistent
expert/slot ownership and a consumer layout compatible with it, or a device scheduler that lets the
existing flattened producer tiles enqueue ready row fragments without a global phase barrier.

The ready-queue result closes the latter option at row-fragment granularity. Any queue protocol
must publish a coarse tile at most once per XCD, not once per routed-slot fragment. That requires
producer tiles to accumulate several slots before publication or changes the DOWN kernel's work
decomposition; it is no longer a small phase-packet adaptation.

These gates close phase-packet scheduling for the exact-order grouped-MoE seams. The next credible
decode architecture changes the output accumulation/layout contract: for example, a compiler-
defined deterministic reduction tree with a matching oracle, or a higher-level quality tolerance
that permits reassociation. Another exact fixed-slot rendezvous schedule cannot recover its own
coordination cost.

The audit also found a measurement defect in the older, already rejected GLU→DOWN cooperative
probe: its local-block census increments once per thread, producing 16,384 rather than 32 arrivals
per XCD (`sync_diff=8`). Its outputs happened to remain exact through redundant coverage and it was
already rejected at 3.23x slower, so no promotion decision changes. Do not reuse that carrier as a
barrier proof without fixing and repeating it.

Artifacts: `/tmp/moe-down-combine-phase-final.out`,
`/tmp/k3_moe_phase.compile.log`, and lease `moe-down-combine-phase-final`.
