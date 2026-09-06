# E3 ordinary BF16 split-K packet integration

CPU design, 2026-09-06. No production edit, GPU execution, new runtime/model knob, or promotion. Authority: current packet/builder/loader sources and the qualified frozen `zg0-body.cuh` + `performance-analysis.json` in this directory.

## Decision and first scope

Use three explicit local instructions: zero an FP32 partial matrix, accumulate every K split, then convert the completed matrix to BF16. Retain the interpreter-owned static packet stream, physical132CTAs and existing counter protocol. Do not use peer XReduce, CUDA child launches, host callbacks, host memset between instructions, or a private global barrier.

First integration is an ordinary BF16 projection, M=16 single-rung block/full-model packet, with no folded RMSNorm, bias, residual, GLU, QKV fusion, quantization, TP collective or KV write. The same body supports measured M4/8/16. Select one exact ordinary projection shape through an artifact compiler fixture before enabling an offline planner. A useful first shape is N3840,K15360,M16, already48 ordinary Gemv instructions in actual Gemma12R16; this is a generic shape fixture, not model dispatch. The output is act.dg and one reusable FP32 plane is245760B. All downstream NormResidual/normalization semantics remain existing packets.

The current standalone candidate uses up to528physical CTAs. A packet body uses132; no throughput result from the528-grid experiment qualifies the integration. A132-grid rerun of the existing comparator and actual broad resource build are prerequisites to selecting a split. BM16 uses82944B shared and256threads; the broad object must declare max(existing arena,82944), keep the old GEMV staging threshold16448B, and pass actual cooperative occupancy>=132 on132SMs. Full broad registers are unknown until compiled; standalone40registers are not the broad resource claim.

## Proposed v1 instruction contract

New enum slots147/148/149 are currently unused; allocate only when implementation begins and coordinate the global registry. The following symbolic contract is the design; never repurpose existing opcodes or spare Gemv mode bits.

| Instruction | Tensor operands | Immediates | Effect |
|---|---|---|---|
| ZeroF32 | t0=P, FP32[M,N] | i0=M,i1=N; all remaining words0 | Write positive FP32 zero to exactly M*N elements |
| GemmSplitK | t0=P read/write; t1=A BF16[M,K]; t2=W BF16[N,K] | i0=M,i1=N,i2=K,i3=S; i4..i7=0, floating/overlaid words0 | BM16/BN128/BK64/NW8/STAGES4 atomic FP32 split accumulation |
| CastF32Bf16 | t0=C BF16[M,N]; t1=P FP32[M,N] | i0=M,i1=N; remaining words0 | Exactly one round-to-nearest-even BF16 conversion per completed element |

Unused tensor operands are TENSOR_NONE. No activation, alpha/beta scaling, epilogue or active-mask modes are implicit. P is distinct from A/W/C and from every live tensor. A/W/C cannot alias in v1. M is exactly4,8or16; N,K positive, K%8=0; S in1/2/4/8/16. N tails and empty/short K splits are supported by the measured body's predicates. Checked64-bit sizes validate4MN,2MK,2NK,2MN against declared tensor allocations; reject overflow, unsupported target/dtype/shape, zero blocks, wrong reserved words or unsupported capability before launch.

Zero and cast index by `i=slice*THREADS+tid; i<M*N; i+=nblk*THREADS`. The producer changes only its outer ownership/staging parameter plumbing: `job=slice; job<ceil(N/128)*S; job+=nblk`, arena supplied by interpreter. No blockIdx/gridDim access is reachable in the device body. M rows belong to one BM16 tile. K split boundaries remain `tot=ceil(K/64)`, `per=ceil(tot/S)`, `[sp*per,min((sp+1)*per,tot))`. Preserve existing cp.async four-stage schedule, FP32 accumulation and atomicAdd order within each tile. Drain asynchronous copies before returning/aliasing the shared arena. Empty work still reaches normal interpreter retirement. Porting requires a GPU comparator rerun even though arithmetic is retained.

The complete projection is `C=BF16(sum_s partial_s)`; FP32 atomics can reorder. Primitive error limits remain relL2<=.006 and maxabs<=.05+.02*maxref, with repeated error reported. No bitexact claim for S>1. S1 repeat remains exact. Deterministic [S,M,N] scratch is a later alternative that changes the qualified atomic epilogue and needs4SMN bytes; do not silently substitute it into this first integration.

## Dependency and lifetime contract

For original `D=proj(out,A,W,deps)` emit:

1. `Z=ZeroF32(P)` waits on the last finalizer using P, when any.
2. `G=GemmSplitK(P,A,W)` waits on ALL original deps plus Z.
3. `F=CastF32Bf16(out,P)` waits on ALL G slices. Return F from the projection helper.

Every old consumer waits on F. Z and G must never inherit/retire the original projection's successor list. Each instruction owns its ordinary counter; F alone is the completion of the logical projection. `Builder::emit` uses Coarse dependencies, whose expected count is producer.blocks. One producer instruction covers allS splits and allN tiles by the job-stride loop; all producer slices retire even when they own no nonempty split. No fine-grained edge shortcut in v1. The existing end-of-body CTA barrier and release counter signal, followed by consumer acquire, publish all FP32 atomic stores before cast. This retains the ordinary scheduler fences; do not invent a host synchronization requirement.

Use runtime tensor allocation at engine load, not temporary per-step allocation. `Builder::tensor` is a persistent name/byte declaration; it has no automatic read/write lifetime inference. Declarations MUST be hoisted before `adopt_tensors(tensors.clone())`, or merged through a shared registry as TmapMint does. Declaring P only inside an individual rung's proj closure loses it from the model tensor table.

For the initial single selected output, one plane at max emitted M*N is enough. Generalization allocates one plane per ordinary output handle, sized to the largest selected rung extent, with an explicit previous-finalizer edge for reuse. This preserves overlap between independent Q/K projections; a single universal plane would accidentally serialize them. For all current Gemma12 ordinary Gemv output handles at R16 the union is278528columns and17825792B scratch, dominated by the16MiB full-vocabulary head. First down-only plane is245760B. Prefill programs do not touch these planes. All M rows, including idle physical slots within the selected rung, are zeroed/produced/cast exactly as the original ordinary Gemv computes them. No KV slots are compacted or renumbered.

A new engine step/rung replay resets its own counters using existing upload/reset logic; Z is executed on every invocation. Loading an allocation zero once is insufficient. Begin_slot/reset does not replace Z. All program launches remain on the existing engine stream, so cross-step reuse cannot race; keep existing multistep/callback restrictions until the new instructions are qualified there.

## Ladder and planner integration

`validate_decode_ladder` currently whitelists known ops and compares normalized instruction vectors. New instructions are opaque to it; M1 Gemv vs M16 three-op expansion also changes length. Merely adding the new names to its whitelist still disables narrow selection. Do not work around this by emitting changed math at M1 or by reporting feeds.len as row count.

Phase A: single-rung M16 block/full assets; no runtime rung extension required. Phase B: add a narrowly validated canonical projection representation to the loader: an adjacent Z/G/F triple can normalize to the same plain Gemv(t0=C,t1=A,t2=W,i0=1,i1=N,i2=K,all modes0) only after verifying matching M/N, unique scratch triple, full coarse Z->G->F edges, no other P reader/writer, no consumer of unfinished result, and exact per-rung row/capacity checks. Nontriple instructions and all physical KV addressing remain under existing validation. Plain Gemv at R1/R2 and triples at selected R4/R8/R16 then compare as the same semantic sequence. Malformed triples fail; valid opaque unrelated programs retain widest fallback. Sparse/high-slot transition tests must confirm rung width=maxfedslot+1 and scratchM equals actual selectedrung, not activefeedcount.

Offline planner keys must include hardware fingerprint, BF16 A/W/output, M/N/K, complete path identity, physicalgrid, blockthreads, compiled broad source/object hashes, shared arena, and cache/timing contract. Values include splitK and verified tile body; S is packet-directed. Existing tunedb/gemv and tunedb/gemm machinery supplies key/correctness/provenance conventions, but neither a scalar Gemv opcode record nor a prefill tile-only record represents this three-op latency. Add a local projection plan record/variant and rank total Z+G+F plus packet gate overhead. Reject stale/missing/wrong-grid records; fallback is existing Gemv. Do not file standalone528-grid measurements as production132-grid records or promote posthoc best-split rows without held-shape/repeat checks and full serving qualification. No model-name branch or runtime tuning switch.

## M32 and fused variants are separate

The verified M32 primitive is BM32/BN64/NW4,128threads,55296B shared. It cannot be called unchanged by all256threads of a broad CTA: warp indices4..7 overrun the layout and barriers become unsafe if half the CTA simply returns. Do not split M32 into twoBM16 passes and claim one weight pass. Leave M32 on the existing path for first integration. A later exact128-thread segmented object needs generic multi-rung role support and correct shared/cursor metadata, or a separately validated NW8/BM32 body; existing role3 is512threads and hard-requires oneM1 rung, so it is not that capability.

GemvGlu is NOT covered: completed gate and up projections must retain their prior BF16 rounding boundary and activation choice before existing Glu executes. Decomposing fused GEMV can alter rounding/math and weight reuse, so it needs its own oracle/performance gate. GemvQkv is NOT covered: each Q/K/V projection must finish before its HeadNormRope/cache writer; tied K=V and optional RMSNorm semantics must remain exact. Existing separate plain q/k Gemv sites are ordinary candidates when their norm mode is0. TP output must finalize locally before unchanged XReduce; first phase excludes TP to avoid widening qualification. Fused lm_head Argmax is excluded; ordinary BF16 lm_head may later finish before unchanged SoftCap/Argmax. No KV cache or recurrent state is written by these instructions.

## Minimal edit map if implementation is authorized

- `crates/packet/src/dev.rs`: append3opcodes, ALL/c_name registry, operand semantics; `slots.rs`: human-readable contracts; generated `runtime/common/dev_isa.h` and existing ABI tests. Do not change DevInst64 or section ABI.
- Small `runtime/nvidia/op_gemm_splitk.cuh`: parameterized device extraction of qualified body plus zero/cast; `interp_sm120.cu`: three dispatch cases, packet-presence compilation, maxarena/capability. Unused packets/builds compile no new body and preserve default text. Preserve GEMV staging independently of arena.
- `crates/devgen/src/lib.rs`: hoisted scratch declaration, minimal proj-helper branch returning F; small helper module if it keeps the large closure readable. `manifest.rs`: explicit opcode inventory and required generic splitK capability. Existing build/cubin capability loader must reject old objects rather than letting an unknown-op trap occur later.
- `crates/plowrt/src/exec/gpu.rs`: narrow CPU triple/geometry/capability validation; phaseB canonical rung equivalence only. No allocation/selection in hot steps, new runtime switches, or kernel bypass. Tests adjacent to existing gpu_decode_rung_tests.rs.
- `crates/tunedb/src/{gemv,gemm,record}.rs` conventions reused for a complete-path variant and evidence gate, only once interpreter measurements exist. No premature global planner refactor.
- Existing packet/devgen/Lean schedule and allocation verification tests must describe FP32 scratch writes and all completion edges. Extend opcode-aware validators/cost tables only where exhaustive dispatch requires it; Lean graph-order certificates do not prove atomic numerical error.

## Existing harness gates and stop conditions

1. Reuse E3 `probe.cu` in a new immutable artifact folder: outer jobs use packet-style slice/nblk132; scalar/reference unchanged; rerun tails/empty splits/M4/8/16 and the same complete zero+producer+cast timing. No new harness. This also checks whether the prior standalone win survives lower physicalgrid.
2. CPU tests: all three op fields/bytes, partial/result disjointness, no declaration lost across cloned builders; allzero slices precede any producer, every producer precedes any finalizer, only F feeds old successors; negative missingZ, missingproduceredge, undersizedP, K%8!=0, badS, wrongM, reusedPbeforeF, unknowncapability and oldobject rejection. Model all possible scheduling orders on a small nontrivial2zero/3producer/2finalizer example. Existing full packet schedule verifier must pass.
3. Existing `block_run ... bench --batch 16 --ctx 1024,8192,32768 --iters 40 --warmup 5 --prefill-iters 1 --pf-chunk 1024` measures the complete isolated block, with frozen control/candidate assets and same packet context policy. Its check verb is prefill-only, so it does NOT validate new decode arithmetic. Use existing decode_dump block mode with fixed --input-f32 sequence and raw act.dg/P dumps for decode checkpoints; no new runtime API. Full-model decode_dump/consume_prompt gates provide realistic history+decode quality.
4. Existing `gpu_consume_prompt` allslot interleaving/reset/full-logit gates compare compiled candidate with widest ordinary baseline; for atomic splitK use an explicitly predeclared bounded comparison instead of weakening an existing bitexact test in-place. Dedicated new test in that existing harness is appropriate. Rung transitions0->3->15 and reset/replay are necessary before phaseB. Run memcheck on partial/output canaries in the existing primitive and on integrated block where feasible.
5. Full-model teacher-forced logits at1K/8K/32K plus full serving15cells, NP32/warm16/output512, two repeats, actual clock/power logs and symmetric L2 eviction for primitive A/B. Freeze bounds before results; existing full-logit relL2<=.01+argmax is a smoke gate, not independent model-quality certification. No production promotion from scalar projection timings alone. Reject if broad spills/arena/gates erase the gain; do not replace a failed body with a new model-specific switch.
