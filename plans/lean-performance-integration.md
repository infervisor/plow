# Lean performance integration

Worktree: `/home/lava/plow-lean-perf`, branch `lean/performance-certificates`.
Inherited dirty baseline is recorded by `inherited-working-tree.patch` and
`inherited-untracked.sha256`. No changes to the main campaign worktree.

## Stages and acceptance

1. Close proof-boundary holes: malformed/cyclic lower-bound queries rejected;
   certificate envelopes preserved; empty/weak empirical evidence rejected;
   four-arm, min-30 and both-spread campaign requirements retained. Negative
   tests accompany each boundary. Use proof-backed D checker on certificate path.
2. Separate semantic, structural-cost and empirical records. Bind exact artifact,
   precision, rung, runtime/hardware and protocol identities; fail closed for
   qualification while retaining explicitly unverified bringup. Validate on load,
   never add certificate traversal to the token loop.
3. Feed original dependencies/address lifetimes into plowc checks and retain
   optimization witnesses. Cover scheduling, buffer generations/retirement,
   specialization domains and fusion precision contracts. Never advertise a
   rewrite-name catalog as a machine-code floating-point proof.
4. Implement opt-in BF16 attention metadata reuse and stride-aware WV candidates.
   Layer-local selected indices/KV remain local. Preserve rounding/scale/reduction
   boundaries. CPU guards first; queued exact replay and matched four-arm GPU
   qualification require parent coordination. No promotion before those pass.
5. Correct roofline accounting from actual selected kernels/model metadata,
   separating logical traffic and physical HBM assumptions; use measured costs
   for policy selection, not proof-derived speed predictions.

## Status

- Lower-bound malformed/cyclic/rate inputs fail closed; envelope retained. P now
  requires four distinct reciprocal arms, >=30 samples, both spreads and the
  treatment-spread <=3*control-drift stability gate, nonempty measured scope and
  passing gate facts. Campaign producer and ledger selection emit the links.
- D now verifies supplied dependency edges with a proven checker and accepts
  checked path witnesses for removed edges; malformed address access arrays are
  rejected. The pre-reduction coarse graph and final retained paths are now
  retained by Builder and sent to Lean by plowc. This does NOT yet prove original
  tensor-effect completeness, fine-grained dependencies or allocation lifetimes.
  GQ issue ordering must never be substituted for completion ordering.
- 21 Lean bridge tests pass (10 D/F/batch, 2 lower-bound, 9 P); Lean builds pass.
  Runtime/devgen knob checks passed before the second experimental knob; rerun
  after remaining edits. Unit-depth/allocation-surrogate oracle outputs no longer
  advertise a performance certificate merely because the envelope was retained.
- Metadata-hoist candidate is default off and passes CPU guards plus queued
  gfx950 replay job1790131548614414475-3a7eb78f (done0). Frozen artifacts:
  `/opt/models/plow-glm53-metadata-hoist-20260923`. Test binary sha256
  d54d4683a13ce0e58786239346444a84bcee77de04f77c65061f8ca115d0c6aa.
  16 sparse +10 dense variants retain exact pinned-oracle results. Changed-layer
  Q/KV/index inputs compare fresh5 launches vs producer5->reuse4 in one9-packet
  chain, bitwise twice; reset-before-reuse rejects. No metadata reuse across
  replay boundaries, no host allocations or waits in enqueue.
- Padded BF16 output -> stride-aware WV candidate implemented, default off;
  exact M<32 domain, sole-consumer/capacity and compiled-object ABI checks.
  CPU emitter, route and object checks pass; queued standalone GPU replay passed
  26 cases with synthetic WV weights. Connected captured attention→WV exact
  replay now passes all8ranks at B16ctx512; modular packet/T4 remain pending.
- No speedup, T4 promotion, full-model qualification or claim that all requested
  integration work is complete. Artifact/rung binding, full effect/lifetime
  witnesses, scalable proof-backed address checking and roofline accounting remain.

## Recommendation tracking

| Recommendation | Implementation / evidence | Remaining gap |
| --- | --- | --- |
| Lower-bound input/envelope boundary | Strict DAG parsing, Kahn cycle rejection, envelope preservation; 2 negative/positive tests | Conditional arithmetic is not hardware optimality |
| P campaign gate | Four reciprocal arms, >=30, both spreads, stability and gate facts; 9 tests. Campaign cert now retains raw client JSON/run records, full packet SHA, replays exact ledger samples/statistics, rejects missing cells/reused runs/protocol and repeat-identity mismatches; portable evidence schema2 | Runtime/full-object identity producer, actual numerical/artifact-gate evidence and provenance authentication beyond raw-file integrity; historical schema1 remains arithmetic-only |
| D/F acceptance TCB | FastCheckD is rejection-only; D and F both require proven reference/path acceptance. Existing Rust schedule/address-map bridges now produce dependency/reclamation paths; Lean independently checks every used path. D/F10 + Effects3 tests pass | Complete devblob effects/lifetimes producer; address-pair enumeration still quadratic |
| Removed dependency witnesses | Builder retains original coarse edges and paths through reduced graph; plowc sends proven checkPaths | Fine dependencies, native segment counter deletion, full RAW/WAR/WAW and completion/fence model |
| Certificate cache / batch | Exact checkpoint + payload bytes + verifier content SHA256; retained envelope; one checkpoint batch for all coarse dependency/policy/layout witnesses and one GQ/LDS batch per program. Linux executes sealed memfd snapshots; AOT/load receipts bind exact executed bytes. Source-replacement/seal and malformed/reordered tests pass. D and F cache scopes separated to preserve checkpoint-specific notes | External dynamic-library/environment identity is not authenticated by executable SHA alone; immutable bound receipts Linux-only |
| Semantic vs structural vs empirical | Distinct typed records and scopes; no aggregate verified bit; AOT coarse-D receipts retain request, envelope and verifier hash in packet/compiler-bound lean-checks.json, replayed once at AMD/CUDA/CPU load | Full ExecutionIdentity producer/admission, actual object/resource and precision contract binding; these scoped receipts are NOT full semantic or empirical qualification |
| Identity/rung/resource bindings | ExecutionIdentity binds model/revision/checkpoint/precision/packet/DAG/objects/ABI/launch/resources/HW/topology/runtime/compiler/oracle and domains; tunedb Digests staleness includes identity; scoped compile receipt packet/request/verifier binding is now consumed at load | Populate full real identities at compile/campaign/load (including runtime configuration); authoritative original-graph/effect completeness remains a compiler obligation; no token-loop traversal |
| Runtime P presence hole | Requires complete scoped knob-rung stamps, nonempty unique rungs, accepted P and evidence hash; runtime always reports absence of execution-bound qualification even when historical scoped claims exist | Historical scoped stamps are not packet-wide qualification |
| Runtime physical provenance and retirement | Allocator-issued physical/mapping generations propagated through owned/subview/VMM allocations and actual AMD/CUDA tensor tables; stale bind/alias/recycled-handle tests. Generation-guarded HSA/CUDA owned upload staging. HSA kernarg completion-backed admission, all-rank preflight in TP chain paths and immediate-publication per-segment admission for audited non-chain TP paths; partial failure cancels every rank | CSR-restaging/native-timing routes require split preparation; dynamic KV/peer provenance, complete scratch/counter-bank/VMM release admission and full fault teardown remain uncertified. No automatic capacity retry; not a complete physical Effects certificate |
| Metadata reuse | Default-off route; 26 exact kernel-chain cases, changed layer data, replay reset refusal; job1790131548614414475-3a7eb78f. Actual captured attention→WV chain all8ranks B16ctx512 passes twice with producer/reuser on different rank data, job1790139833245823384-f7259993 | Regenerated connected modular packet + matched T4; inactive/M32+ remain unsupported |
| Remove unpad | Default-off padded-output/stride-WV ABI/sole-consumer/capacity guards; 26 exact cases job1790132861993475983-4e128c15, including poisoned odd heads + metadata reuse. Actual-weight WV job1790137638876873657-18cccdf6. Connected actual captured attention→WV all8ranks B16ctx512, contiguous/stride1024 × metadata off/on × two poisons, both olat/oat bitwise exact, job1790139833245823384-f7259993 | Regenerated connected modular packet and matched T4 not done; no expansion to M32+/inactive |
| Lifetime/generation/cancellation | Checkpoint-D memory_effects checker with proven RAW/WAR/WAW ordering, live owner/generation/bounds, retired physical overlap, and cancellation retirement; strict thresholds/fence declarations. Packet-derived logical effects now reconstruct real coarse wire counters/slice coverage and audited whole-tensor operands, with AOT batch producer and exact load reconstruction; missing RAW/WAR/WAW ordering rejects, concurrent reads pass. Compiler hook3/load3 tests pass | Producer currently covers only audited ordinary scalar/dense/FP8 operations, not full GLM/TP/fine/indirect kernels. Logical distinct tensor pools and virtual entry/exit lifetime do NOT establish physical aliases, kernel bounds, allocator generations/cancellation or actual release/acquire fences. Full runtime producer and doublebuffer admission remain |
| Precision fusion/specialization | Proven L padded-address bounds, odd-head exclusion, identical consumer input bits and unchanged opaque consumer output. Exact M<32 domain/precision guard; actual plowc producer and runtime reconstruct packet shapes/capacity before replay. AOT and load tamper tests plus 11 Lean negatives pass; runtime orphan-stride guard rejects malformed unverified packets | General fixed M/N/K/head/tail and precision-aware fusion witnesses; L is not a machine-code/FP implementation proof |
| Measured rung policies | Proven checkpoint R checks exact supplied domains and qualified minima. Actual attention, grouped-MoE and generic dense-GEMM selectors retain eligible exact-cell populations through AOT batch and load replay. Audited unfused BF16 prefill GEMMs additionally bind final instructions, operands/immediates and balanced full-CU placement; load reconstructs bindings before cached replay | Remaining quantized/fused/attention/MoE final-route reconstruction; full tail/inactive/maxlive domains from ExecutionIdentity and authenticated raw measurement provenance; current kernel-store qualification is not T4 |
| Roofline | Fake unit-duration cycle/us bounds removed (zero unavailable + topological depth only); selected-op TP-local FLOPs/weight+scale+activation bytes, sparse KV and collective payload inputs;5 tests; WV scalar scales and native FP8 partial writes corrected; incorrect GLM40-layer preset removed. Campaign roofline now labels logical traffic explicitly, leaves physical HBM/spills unknown. Optional recorded-trace priorities consume real disassembled full counters, complete workgroup/op coverage and explicit calibrated clock; packet/runtime identities match supplied run record; 27 campaign tests including CLI and 8 trace negatives pass | Actual selected object/launch/spill costs, calibrated physical HBM and complete opcode coverage. Trace tool rejects fine/partial/native-untraced coverage; no actual full campaign trace analyzed yet. Counter-chain body envelopes are not end-to-end critical path or speedup predictions; capture provenance is not authenticated |

Stride-WV frozen artifacts: `/opt/models/plow-glm53-stride-wv-20260923`;
binary78e0103ae6d8f977ab493e2af22aceb0c434bb34e6b0c32ee5684e79a408011a,
object652f9021da05451f5b06bdf3d44ccb043c317ff8e0d5d583ca186dc8ee7f2c94.
The standalone WV object uses512 threads,81VGPR,0private bytes,2SGPR spill slots;
this is not the final interpreter resource envelope or a performance promotion.

Conditioned actual-weight WV frozen artifacts:
`/opt/models/plow-glm53-stride-wv-captured-20260923`.
Job1790137638876873657-18cccdf6 DONE0; binary
ee8f3cc7b46fbbbc3dfe05208dd6e2f5a51f0bb1ca76e55d71d86abd69fb7905;
unchanged WV object652f9021da05451f5b06bdf3d44ccb043c317ff8e0d5d583ca186dc8ee7f2c94.
Pinned vLLM0.29 MLA report
31eeafdb8fa74e0a05e1cb355aef0eb22efb5dfdb5e52d8b4116aa2679f3acf8.
Every input/weight/scalar-scale/output hash checked; all8ranks actual original WV
weights, captured BF16 input, M16/H8/K512/N256, poisoned output and odd heads.
This is one conditioned operation, not full-model or empirical qualification.

Latest CPU regression: 26 real Lean bridge tests, 2 actual compiler hook tests,
2 load receipt replay/tamper tests, devgen knob13, runtime knob19 (+1 ignored),
campaign Python24 (including evidence5) passed. Linux verifier snapshots are sealed;
D/F caches are checkpoint-separated to preserve their different envelopes.

Additional producer regression: shared logical-effects2 tests; real Lean test
covers ordered/missing RAW+WAR, WAW, transitive dependencies and independent reads;
actual compiler hook3 and runtime receipt3 tests pass. Unsupported effect domains
are recorded in manifest `logical_effect_gaps`, never turned into an empty effect
certificate. The scope remains distinct logical tensors with program entry/exit
sentinels, not a physical runtime allocation/lifetime proof.
Padded-WV rejection-only guards now include AttnRes demoted residual/scratch,
FlashDecode NRF hidden operands and indirect KDA state descriptors. These CPU
guards do not change frozen GPU replay artifacts or broaden their qualification.

Broad `lean_verify --tests --include-ignored` found two unchanged knob-fixture
assumptions: the inherited production recipe enables token_batch_tp, so enabling
seq_par rejects at that constraint before the fixture's expected seam constraint;
recorded seq_par=false is no longer stale under that recipe. Registry/runtime knob
tests pass. These unrelated fixture updates are not included; broad suite is not
claimed fully passing. Targeted production-integration tests remain passing.

Connected actual attention→WV frozen evidence:
`/opt/models/plow-glm53-attention-wv-captured-chain-20260923`.
Job1790139833245823384-f7259993 DONE0; binary
c45cfd68b5dda4647b978a0d9ab6ffd21c4e6a10fb6f655d3c1961215405cacb.
Pinned attention report7636e6606b199edc6315aab6f7683446125a83a174878b449a4475ce02c48fbf
and MLA report31eeafdb8fa74e0a05e1cb355aef0eb22efb5dfdb5e52d8b4116aa2679f3acf8.
All8ranks × stride0/1024 × metadata off/on × scratch/output poisons255/85.
Combined Q/KV and selected-index identities, original WV/scalar and both exact
intermediate/final reference outputs checked. Hoisted metadata producer uses the
previous rank's differing Q/KV; reuser uses current rank data. Frozen objects
unchanged. This remains connected kernel-chain evidence, not a regenerated
modular packet, long-context, full-model, uninstrumented timing or T4 claim.

Actual HSA object identity producer now captures the bytes accepted by the loader,
not a later reread of a path. Load-resolved symbol ABI/static LDS/private-byte
queries now check every ROCr status; failed queries cannot silently become zero
resource claims. Existing AMD block reports include a per-rank loaded-object
inventory, explicitly not selected dispatch/launch geometry or register-count
qualification. Inventory work is confined to load/get_function/unload/report;
HsaKernel remains Copy and dispatch is unchanged. Two CPU tests cover stable
handle-independent identities, duplicate resolution, changed resources/bytes,
unknown handles and unload/recycled-handle invalidation. Live ROCr integration
has not yet been replayed. Full execution identity/admission remains pending.
Targeted remaining Lean regression:27 tests passed across logical effects,
lower bounds, measured policy, memory effects, MLA layout, P and scope.

Coarse-D receipts now normalize the final wire's complete producer counters,
slice coverage and identical GQ gates at AOT; they no longer reuse the builder's
possibly stale retained protocol after native/fine rewriting. Original dependency
requirements and path witnesses remain independently checked against this final
protocol. Unsupported counter domains emit `dependency_binding_gaps`, not stale
receipts. Runtime reconstructs the same protocol before cached replay, including
when someone updates a sidecar packet SHA. Compiler hook3/load3 tests pass with
removed-wire-wait rejection and native-counter coverage-gap cases. This does not
add a native HSA segment-order or fine-grained completion proof.

MoE exact-cell policy now retains a checkpoint-R witness through the existing AOT
batch/receipt path. Shared selector/witness eligibility rejects undersampled or
malformed distributions, stale identity, invalid geometry and invalid policy
parameters. Witness costs explicitly include externally supplied segment handoff
and policy margin; they do not turn the historical handoff estimate into a new
measurement or establish serving speedup. Attention and MoE policy producers are
covered; projection, whole-rung inactive/tail coverage, authenticated
raw measurement provenance and load-time route reconstruction remain pending.

Generic AMD dense GEMM selection now emits checkpoint-R requests from its actual
legal registry population and strict-current exact M/N/K costs, through the same
AOT batch/receipt path. Full hardware CU count only; legacy stale-digest fallback
and partial/analytical populations receive no witness. The store/loader rechecks
correctness, samples and finite ordered distributions rather than trusting a
serialized Qualified state. Generic selector rejects nonfinite/nonpositive costs
and breaks measured ties by opcode. This certifies minimum supplied costs for a
selection decision, not final packet-route reconstruction, FP implementation,
hardware optimality, arbitrary inactive/tail safety or empirical T4 performance.
Tests: selector9, emitter tile14, TuneDB86, actual GEMM and MoE producers against
Lean (correct/opposite choices, geometry changes and bad/missing costs), manifest3.
Roofline trace malformed-input coverage now also rejects boolean/nonfinite clocks,
boolean indices/thresholds, counter IDs outside their exact declared domain and
incomplete counter inventories; packet-roofline11 tests pass.

Final-wire ordinary BF16 GEMM policy scope is now distinct from decision-only R.
The compiler binds only actual matching unfused prefill instructions, exact
M/N/K, opcode, operand identities/capacities, immediate/float bits, complete GQ
slices and balanced declared full-CU placement. No speculative query becomes a
final-route claim without a match. Load reconstructs the binding before cache
lookup, including after a changed sidecar packet SHA. Tagged tiles, norm/RoPE/
bias fusion, quantized/packed/token-batch/reduced-CU cases retain decision-only
scope. This does not authenticate costs, object implementation, post-load runtime
rewrites or FP arithmetic, and introduces no token-loop work.
Tests: asset192 passed/2 ignored; runtime load/replay4; actual GEMM
producer+wire binding→Lean1. Thirteen packet/domain refusal mutations and five
load mutations retain failure even when the outer packet hash is updated.

The legacy expanded-schedule D/F producer now rejects missing placements,
starts/stream/packet disagreement, unknown/duplicated/zero-threshold counters,
missing or altered tensor reader/writer sets, missing/undersized/overflowing
address entries. It reconstructs the task sets from the actual expanded task
vocabulary before submission instead of filling missing values with empty sets,
zero starts or a fake shared resource. This is completeness relative to that
expanded graph, not full machine-code effects or runtime allocator lifetimes.
API returns Result; both real plowc D/F callers propagate rejection. Fifteen
source mutations pass the CPU rejection test; real schedule negative tests2 pass.
An initial strict check correctly exposed the compiler's additional KV-state
effects, which are not represented by task.tensor. The shared Flash semantic
producer now explicitly supplies those effects to both map construction and
verification; unknown extra entries cannot inherit empty effect sets. Two CPU
tests cover base and supplemental input mutations. Full real example-bucket Lean
compile suite passes (91s), plus both real corruption tests. plowc feature check,
runtime CUDA/HSA bin checks and diff whitespace check pass. Supplemental kernel
footprints, physical aliasing and final runtime rebinding remain outside scope.

Checkpoint A now has an actual-body mode. The producer uses egglog's own parser,
preserving variables versus literal kinds and all actual constructor operands;
conditional/bidirectional/subsuming rewrites are refused by this scope. New Lean
RewriteBody.check_sound proves equality of depth-bounded full-arity expanded
syntax, retaining output dimensions, scales, activation kinds, and nested
residual nodes. Both plowc paths use actual bodies; devblob retains source hash,
request, executed-verifier hash and envelope in RewriteBodyExpansion receipts
for cached load replay. Legacy name-only clients remain explicitly catalog-only;
manifest reports which scope was checked. This is NOT a BF16/FP8 machine-code,
actual rounding implementation, full instantiated graph or backend lowering proof.
Lean builds; actual engine parser test and actual source→Lean body test pass,
including unchanged-name altered operand/dimension/activation/scale/tree cases.
Latest body scope regression: seven body mutations plus four malformed catalog
mutations reject; actual devblob hook3 and load/replay4 pass (load replay includes
altered body with recomputed request hash); manifest57 pass. No token-loop work.

Physical producer milestone (explicitly authorized by parent): allocator-issued
process-local physical IDs and mapping generations now flow through owned
CPU/HSA/CUDA DeviceMem allocations, audited parent-derived subviews and VMM slab
carves. The metadata does not own device storage; owner/drop/free behavior is
unchanged. Public base/length mutation, stale owners and stale map generations
cannot produce valid evidence. Raw-address views remain explicitly unknown.
AMD/CUDA VmmOps record actual successful physical-handle creation and mapping,
ordinary snapshot allocations, unmap/release/free invalidation. A pooled handle
retains physical identity; every remap receives a new binding generation.
Distinct VAs with the same physical range fail allocation_disjoint. Multi-chunk
slab carves carry all intersecting physical ranges. CUDA attaches these after
its existing mapping join, preserving its chunk-upload/commit overlap.

| Physical boundary deliverable | Implemented/tested | Remaining |
| --- | --- | --- |
| Owned allocation identity | CPU/HSA/CUDA allocator-issued IDs; nested subviews; stale owner, reused address, public range mutation and bounds tests | Foreign/imported aliases and unaudited raw-pointer views remain unknown |
| VMM physical aliases and generations | Real backend hooks, shared tracker tests, actual VmmSlab mock lifecycle; pool roundtrip preserves physical ID but invalidates old binding | Reservation/access-permission proof, actual GPU run and fault-path evidence |
| Source-to-runtime provenance | AMD flat/band/twin/VMM slab views; CUDA flat/counter/prefill/slab views; TP bulk helper; actual AMD load/KV/band and CUDA base/per-slot table binding ledgers | Dynamic KV/peer views and cancellation/retirement admission; CPU/other table producers not covered by this ledger |
| Lifetime proof admission | Stale owner/map evidence rejects; metadata-only Arc never pins physical allocation | Actual completion/release-acquire facts and physical Effects producer; no physical-lifetime certificate claimed |

Implementation: new device/provenance.rs; device/{mod,cpu,hsa,cuda}.rs;
memory/vmm.rs; exec/{amd,gpu,tp}.rs. No PLOW knob, no dispatch/decode-loop
certificate calls. DeviceMem grows 40→48 bytes (one optional metadata pointer),
not 40→72; identity records and view metadata are cold-path allocations.
VMM driver mutations and their records share a backend-local lock to prevent
concurrent unmap/remap from resurrecting stale evidence. This can affect VMM
mapping/admission concurrency; no end-to-end overhead claim without GPU replay.
The inventory uses ordered-map predecessor/range queries, not a full mapping
scan per operation. Partial/unknown/overlapping mappings fail evidence closed
without changing the existing driver error/ownership behavior.

CPU metadata microbenchmark (optimized standalone test of production module,
7×100000 iterations, unpinned host): allocation metadata+drop median57.60ns,
subview metadata+drop23.12ns, create/map/unmap/release metadata with four actual
parking_lot uncontended locks140.45ns. No driver calls, not serving latency,
not T4, not evidence that contention or larger handle tables are free. Reproduce
with rustc --edition=2021 --test -O device/provenance.rs, existing serde and
parking_lot rlibs, then provenance_metadata_cost --ignored --nocapture.
Final focused regression: provenance10 passed/1 microbenchmark ignored (includes
the existing disassembler provenance test); device ownership/CPU4 passed;
VMM100 passed, including actual slab map/access failure unwinds and existing
release/readmission races. CUDA+HSA library/binary checks and git diff --check
passed. No GPU replay or numerical/performance qualification was run for this
milestone. Mapping-generation nonces are not the VMM request-generation counters
and do not replace their cancellation checks or supply completion evidence.
Tensor-table binding milestone: new exec/tensor_bindings.rs retains actual
allocation/mapping lease metadata per table entry and advances a table generation
only after successful upload. Known stale entries reject before driver upload;
public owner-range corruption and out-of-bounds spans cannot become a known
binding. Failed uploads leave the ledger invalid; neither hardware rollback nor
completion is claimed. Unknown raw-pointer entries remain usable for bringup,
explicitly outside fully tracked coverage. Intentional aliases remain supported;
the explicit disjointness helper rejects physically aliased ranges, including
different VAs mapped to one handle. Scheduling has not yet supplied/consumed a
complete physical-disjointness obligation set.

Actual producers/consumers: AMD initial table creation, prefill KV rebase/restore,
and ragged-band rebinding now use the ledger; CUDA base and immutable per-slot
prefill tables capture owners, appended prefill buffers and slot-specific
descriptor allocations before upload. CUDA checks exact equality with its
independently built pointer arrays. AMD preserves the old host mirror until a
successful upload. Shared checked slot geometry is used by both engines and
tested for valid slots/restoration, invalid slot/handle, and arithmetic overflow.
Band views bind through their actual parent allocation rather than inventing a
new allocation identity for the shifted address.

Cost/scope: O(table entries + referenced physical spans) validation at load and
actual prefill rebind; O(table entries) pointer serialization plus update-vector
storage. Updates are sorted in place for duplicate detection (no hash-set
allocation); liveness checks allocate nothing. CUDA keeps one host ledger per
immutable slot table. Existing no-op rebase fast paths and all dispatch/decode
paths are unchanged; these are not recurring token-loop checks. No additional
GPU launch, wait, fence or device table allocation. Post-upload failure recovery,
future stale-generation detection without rebinding, dynamic KV/peer views and
actual queue completion/retirement remain outside this scope. In particular a
caller must still honor upload errors; this ledger is not an execution lease.
Focused verification: seven binding tests pass, including actual shared slot and
band producers, pointer-array mutation, physical aliasing at distinct VAs,
recycled handles, remap without release, owner drop, range corruption, duplicate
updates, and upload failure. Provenance10 pass/1 CPU microbenchmark ignored;
VMM100 pass. CUDA/HSA binary check and diff whitespace check pass. No GPU replay
or measured serving-overhead claim. Files: exec/{tensor_bindings,mod,amd,gpu}.rs,
device/{mod,provenance}.rs; isolated branch/base and inherited provenance unchanged.

Next dependency remains binding these records to actual queue retirement and
the complete dynamic-view producers before physical Effects admission.

Completion/retirement trace (next bounded milestone):

| Actual path | Completion observation | Reuse/remaining boundary |
| --- | --- | --- |
| HSA AQL kernels | synchronize waits on counting done_signal, not queue read index | dispatch/begin_dispatch_chain kernarg-slot reuse still follows AQL read-index capacity; this does not establish kernel retirement. No reproduced corruption or changed queue algorithm in this stage |
| AMD counter banks | inactive_ready tracks successful counter/cursor clearing | Clear completion is not a dispatch-generation retirement proof; whole-program/TP completion admission remains |
| HSA upload staging | Per-slot SDMA completion signal | New generation guard accepts exact zero for the matching outstanding ticket before refill |
| CUDA upload staging | Per-slot end event; whole upload-stream synchronize | New generation guard blocks failed/unrecorded event reuse; only successful stream quiescence can clear uncertain submission failure |
| CUDA execution | Stream synchronize per public step; H2D event gates prompt-staging overwrite | Kernel scratch/step staging, graph/event rerecord generations and fault retirement are not yet admitted by this new guard |
| VMM prefix reuse | Sequence-generation checks on queued CopyOut/Premap; caller release/flush paths | Request generation and queue job completion are not GPU kernel retirement; dynamic KV/peer provenance stays unknown |

New device/retirement.rs is an allocation-free, lock-free per-resource state
machine with slot identity and monotonic generation tickets. Prepared operations
may be canceled or rejected as unsubmitted; submitted cancellation remains
pending. Stale/foreign completion cannot release a newer generation. Uncertain
submission or end-event failures require actual stream quiescence, not a reused
event that may still describe an earlier transfer. This is a runtime guard with
backend completion calls in its TCB, not a new Lean/hardware completion theorem.

Actual integration is deliberately bounded to HSA UploadRing and CUDA UploadPipe
owned pinned staging. No kernel dispatch/decode path changed. Normal waits,
signals/events and GPU copy counts are unchanged. On unresolved teardown failure,
unretired HSA pinned slots/signals or CUDA pinned buffers are retained (intentional
fault-path leak) rather than explicitly freed while DMA may still read them.
CUDA Drop supplies a stream-drain backstop when normal finish did not establish
quiescence. This does NOT retain/certify destination allocations, direct borrowed
checkpoint mappings, arbitrary VMM mappings or full engine fault teardown.

Optimized standalone production guard benchmark: 9×1000000 submit/event-ticket/
complete cycles, median5.03ns on this unpinned host. No per-copy heap allocation
or locking; only slot creation allocates an identity using a global atomic.
Measurement excludes driver calls, integration branches and contention; not a
serving overhead or speedup claim. Existing kernel queue/doorbell path untouched.
Verified: retirement4 tests pass (stale/foreign generations, reuse while busy,
cancellation, uncertain submission and missing-end-event failure, exact-zero
HSA observation); one CPU microbenchmark ignored in the normal suite and run
separately optimized. Binding7 and VMM100 regressions pass; CUDA/HSA binaries
check and diff whitespace check pass. No GPU replay. Initial generic From-error
conversion introduced inference ambiguities elsewhere; removed it in favor of
local explicit conversion without editing unrelated callers.
Changed files: device/{retirement,mod,hsa}.rs and exec/gpu.rs. Next remaining
retirement consumers are kernel scratch/counter-bank generations, kernarg-ring
wrap, table/VMM release admission and complete fault teardown; none are claimed
covered by the upload-staging guard.

Post-body regression:27 Lean bridge tests pass (D/F, lower bounds, effects,
layout, measured policy, P); compiler catalog/growable/tile suites12 pass;
asset192 pass/2 ignored. Growable negative fixture now recognizes the proven
reference checker's rejection message (the failure verdict remains required).
Inherited unrelated knob-fixture failures recorded earlier are not claimed fixed.
Baseline provenance files remain SHA256:
`inherited-working-tree.patch`64c4334c3484ab69c7f93ef4b74e52a604e0b8b76423d1dc119a970082690ad2;
`inherited-untracked.sha256`3b83fbf6cc4955811f9946ddb7b176baa448e215544022c83e843363b00d1776.

HSA kernarg-ring retirement milestone (2026-09-23):

Contract finding: AMDGPU execution directly reads kernarg memory; the dispatch
completion signal is updated after execution. Therefore AQL read-index progress
alone does not establish kernarg retirement. Primary reference:
https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.1.1/LLVM/llvm/html/AMDGPUUsage.html#kernel-dispatch
This fixes an unsound reuse condition, not a reproduced GLM numerical failure.

New device/kernarg_retirement.rs keeps issued/published/retired batch generations
in three atomics. Ordinary dispatch and whole-chain reservation admit the entire
range BEFORE advancing the actual AQL write index; actual indices must agree.
Every index is reserved through these two audited paths. Ring reuse requires
exact-zero counting-signal observation for the matching fully published batch.
Partial preparation, failed publication, stale zero observations and canceled
chains cannot retire a generation. Integer exhaustion rejects before the HSA
chain sentinel values; ordinary modulo ring wrap remains supported.
Existing synchronize now validates its returned signal value and retires the
matching generation; it rejects unpublished chains instead of waiting forever.
Explicit chain cancellation and incomplete/overfull commit poison the queue:
reserved AQL positions cannot be safely rolled back. On unretired backend Drop,
the kernarg ring, counting signal and driver reference are intentionally retained.
Other operands, modules, VMM mappings and full engine fault teardown are NOT
protected by that scoped retention.

Cost/admission policy: no per-dispatch allocation, lock, extra packet, completion
signal, doorbell or blocking wait. Three u64 atomics per backend. Reaching ring
capacity performs a single completion-signal load. If prior kernels remain in
flight, admission rejects before this queue's reservation rather than blocking
one TP rank while peers still need publication. Existing successful drain
boundaries replenish capacity. Coordinated TP preflight/retry is NOT implemented:
another rank may already have a reservation when this rank rejects; callers must
not blindly retry or claim the multi-rank operation was untouched. Sustained
unsynchronized >4096-dispatch producers may now get a deliberate admission error.
HSA_QUEUE_TYPE_SINGLE / one host producer remains a caller invariant, not a new
multithreaded submission proof.

Verification: kernarg4 tests pass (wrap, delayed completion, whole-chain capacity,
partial/failed/canceled chain, stale completion and integer exhaustion); HSA6 and
upload retirement4 and binding7 regressions pass; CUDA/HSA binary check and whitespace check
pass. Optimized standalone production-state microbenchmark,9×1000000 iterations,
median0.80ns/dispatch on this unpinned CPU, amortizing one zero completion per4096
reservations. This measures metadata only, excludes actual driver calls/queue
publication and is NOT a measured serving overhead or speedup. GPU queue replay
and TP pressure/fault recovery remain unqualified; no GPU was used for this stage.
Logs: /tmp/plow-lean-kernarg-{tests-final,check,hsa-regression,upload-regression}.log.
Changed: device/{kernarg_retirement,hsa,mod}.rs and this tracker. Branch/base and
both inherited-provenance hashes rechecked unchanged; no main writes or merge.

Coordinated TP chain-admission milestone (2026-09-23):

Supersedes the prior missing all-rank preflight note above. Actual decode replay,
graph-phase segment-major prefill, and token-batch graph-phase body now use one
reserve_all helper: first non-mutating preflight of EVERY rank, then generation-
checked reservation. Scratch ticket slots allocate once at TP group load, not per
token. Tickets bind a unique queue identity, issued/retired generations, packet
count and capacity; a needed exact-zero completion observation is captured before
reservation and checked against unchanged generation state. Preflight does not
retire slots or advance AQL indices. A rank exhausting capacity rejects the whole
admission without any new reservation. Foreign/replayed/stale tickets reject.

Failure after reservation begins cancels ALL ranks, including the failing and
not-yet-visited rank, because the failed driver step may already have mutated its
queue. Actual fill errors and commit errors likewise poison every rank; all ranks
validate commit readiness before the first doorbell. Decode joins every fill
thread even after an error/panic so host writers finish before cancellation.
This is fail-closed admission, NOT transactional rollback or proof that poisoned
GPU work stopped: AQL headers may already be visible, and submitted collectives
can still be in flight. Full operand/VMM/module retention on faults remains open.
Admission occurs after existing prepare/rearm work, so the no-mutation guarantee
is specifically queue reservation/retirement, not the entire token-step state.
No blind retry or blocking per-rank drain was introduced. Non-chain ordinary
multi-rank launch paths remain outside this bounded collective-chain mechanism.

New normal-path work is fixed-size atomic metadata checks and preallocated ticket
stores per rank, with no new lock, heap allocation, GPU event/packet or syscall.
The old decode fill threads/handle-vector allocation are unchanged, so this does
not claim the entire existing submission path is allocation-free. Optimized CPU
metadata benchmark9×100000 TP8 iterations:20.37ns/collective vs6.80ns for the prior
reserve/publish metadata path, delta13.57ns. Unpinned sequential samples, no driver
or GPU work, not a serving-overhead or speedup claim.

Verification: five new coordinator tests pass (exhausted rank leaves all queues
unchanged; successful all-preflight-before-reserve ordering and wrap; changed
generation; failure after actual reservation; foreign/stale/concurrent-completion
tickets). Four underlying ring tests also pass;16 actual TP orchestration tests
pass; CUDA/HSA binary check and whitespace check pass. Standalone optimized
metadata benchmark ran; no GPU use. Logs /tmp/plow-lean-tp-admission-{tests,check,
orchestration}.log and /tmp/plow-lean-tp-kernarg-regression.log.
Files: device/{kernarg_retirement,hsa}.rs, exec/{chain_admission,mod,amd,amd_tp}.rs.
Unknown dynamic KV/peer effects remain uncertified. No main synchronization,
commit or merge; reviewable work stays in the isolated branch.

Non-chain TP admission audit and implementation (2026-09-23):

| Path | Action / boundary |
| --- | --- |
| Non-phase segment-major prefill | Exact per-segment packet budget preflighted and reserved on every rank before enqueue; preserve each original per-dispatch doorbell |
| Per-segment timing/capture prefill | Same admission; retain existing post-all-rank completion/timing/capture boundary |
| Packed prefill | Same admission using actual span-dependent packet count; no new per-rank drain |
| Non-phase token-batch body / segment timing | Same admission; existing body launch ordering retained; zero-launch skip must agree on every rank |
| Compact post-decode audit | Already preceded by full all-rank completion; rank-local work has no peer launch dependency. Keep ordinary one-packet path; attempt every rank's completion even if a later audit enqueue fails |
| Device recurrent-state clear | Rank-local, not collective; no artificial chain conversion. Attempt every rank's completion after partial enqueue failure before returning the first error |
| Rank-0 deferred token capture | No multi-rank admission needed; enqueue failure now poisons all rank backends before returning with the preceding decode potentially in flight |
| Sparse native-lo / row-band CSR restaging | Explicitly refused by the new immediate-batch path before reservation; existing private LoCsr::stage may synchronize and overwrite shared CSR inside enqueue. Needs an all-rank quiescence/preparation split |
| Sparse/indexer native per-launch timing | Explicitly refused when its SplitTimer would synchronize inside enqueue; ordinary interpreter and other native routes are not refused merely because that diagnostic knob is enabled |

HSA reservations now distinguish deferred chains from immediate-publication
batches. Both reserve the whole admitted segment before enqueue; immediate batches
ring once per original dispatch and do NOT add a final doorbell. Kernarg generation
publication still waits until the whole reserved batch is prepared. Active
reservations add one relaxed publication-mode load per dispatch and one at commit;
ordinary unreserved dispatch short-circuits without that load. No new GPU packet,
lock, allocation, syscall, blocking admission wait or blind retry. The existing
single-producer HSA invariant still applies.

Concrete producer fixes: SparseMla active FP8 single-pass rows>=512 emitted2
packets while Route::active_launches counted3; mixed native-upper/interpreter-lower
splits fell through the reservation producer's default1 despite emitting3 or4.
Actual producer methods and enqueue_window now share window_launches; accounting
also follows loaded pack-object availability, FP8 scale presence and CSR mode.
The enqueue diagnostic counter now uses the actual returned packet count.
Tests construct real Route values and cover511/512 tails, unavailable pack object,
BF16/FP8, CSR mode, row-band5 and mixed split3/4. No arithmetic/precision change.
Mixed native-upper/interpreter-lower splits remain supported; only native-lo and
row-band variants that may restage CSR need the preparation split.

Failure scope: low-level collective preflight still does not reserve/retire any
queue on a later-rank refusal. The non-chain segment caller additionally poisons
all rank backends before propagating ANY admission/skip/enqueue error: previous
segments may still read tensor tables, so a caller must not restore/rebase those
tables or blindly retry after refusal. Existing full-rank completion boundaries
now attempt all ranks before returning the first error, then poison all backends
if completion failed. This is fail-closed host admission, NOT proof of GPU
cessation or permission to free operands. Full model/module/VMM fault teardown
and dynamic KV/peer ownership remain explicitly uncertified.

Compatibility limitation: the CSR/native-timing refusals affect previously
accepted ordinary execution in this isolated implementation. They are NOT a
shipping/performance promotion. Safely splitting current private preparation APIs
is remaining work; no main runtime was changed. Newer main native FP8 prefill
counts/opcodes were not copied into this inherited branch.

CPU metadata-only TP8 benchmark including immediate publication decisions:
9×100000 iterations,23.67ns/collective vs6.86ns prior reserve/publish guard,
delta16.81ns. Unpinned host; excludes driver calls, packet preparation, route/span
count walks, HSA polling and GPU work. No serving-overhead or speedup claim.
Admission tests7 pass (including later-rank exhaustion/staleness/partial failure,
immediate doorbell policy and all-rank teardown-attempt order); underlying ring4
pass; sparse CPU13 (including concrete producer1), HSA6 and TP orchestration16 pass; CUDA/HSA check and
whitespace check pass. No GPU used. Logs /tmp/plow-lean-nonchain-*.log.
Main-relevant changed files: device/{hsa,kernarg_retirement}.rs,
exec/{amd,amd_tp,amd_sparse_mla,chain_admission}.rs. Tracker and baseline provenance
remain in /home/lava/plow-lean-perf; no commit, main synchronization or merge.
