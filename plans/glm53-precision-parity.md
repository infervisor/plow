# GLM-5.3 / vLLM 0.29 precision parity

User requirement, 2026-09-22: match vLLM precision for all dtypes. The original
3000 output tok/s and full context/concurrency matrix remain unchanged.

Reference: full revision aca966e4e02791568aa6a4ced368624b3d897f42, TP8 MI350X,
Docker sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1,
VLLM_ROCM_USE_AITER=1, BF16 model dtype, KV auto, prefix cache off.
Do not weaken the reference or switch either arm to MXFP4 to close this gap.

User performance floor added 2026-09-22: >=800 aggregate output tok/s at 8192
input tokens, concurrency16. The broader3000 tok/s/matrix objective remains.
User reports SGLang616 tok/s at that cell. Supplied command is single-nodeTP8
zai-org/GLM-5.2-FP8 (NOT GLM-5.3), TileLang DSA prefill/decode,
chunked-prefill-size131072, mem-fraction-static.80, watchdog1200, port30000.
GPU model and benchmark output length remain unknown. Not a matched baseline.
For C16,800 aggregate outputtok/s requires <=20ms steady decode iterations,
with additional headroom for prefill and request scheduling in end-to-end scoring.

## Evidence and current gaps

CPU-only inspection of the pinned container's installed source, together with
the completed reference server log at
/opt/models/plow-glm53-full-vllm-qual8k-aiter-20260922/server.log:

| Path | Reference evidence | Current Plow | Status |
|---|---|---|---|
| FP8 linear | AiterFp8BlockScaledMMKernel selected; scaled_mm/aiter.py requires dynamic activation groups (1,128), weight blocks (128,128), float32 scales | many projections pre-dequantized/absorbed BF16; no activation quant | mismatch |
| Routed/shared MoE | AITER Fp8 backend selected; rocm_aiter_moe.py block W8A8 maps to per_128x128, aiter/fused_moe.py remaps to per_1x128 | routed W8A16, shared BF16 in current packet | mismatch |
| MLA absorption BMM | mla_attention.py selects FP8 BMM with AITER enabled; server log confirms FP8 BMM precompile. Derived W_K/W_V dynamically quantized per batched tensor | BF16 prep-derived q_absorb and BF16 merge/fold | mismatch; fusion changes quantization boundaries too |
| Main MLA cache | ROCM_AITER_MLA_SPARSE selected; auto cache resolves to BF16 for BF16 model | BF16 ckv/krot | storage dtype agrees; kernel accumulators not fully audited |
| Indexer wq_b | deepseek_v2.py Indexer constructs ReplicatedLinear with quant_config; not in checkpoint modules_to_not_convert | dequantized BF16 | mismatch |
| Indexer wk/weights_proj | deepseek_v2.py fused projection explicitly quant_config=None, checkpoint wk dequantized BF16 | BF16 projections | storage dtype agrees; fusion/rounding differs |
| Indexer query/key | query per-token-group FP8 quant after RoPE; key cache uint8 storage packs FP8 plus FP32 scales, scale_fmt=ue8m0 | BF16 query/key cache for nonpooled GLM | mismatch |
| Norm, router, reductions, logits | model-specific unquantized exceptions and kernel-dependent accumulation | mixed BF16/FP32 | pending exact active-path audit |

Sources under /usr/local/lib/python3.12/dist-packages/vllm:
model_executor/kernels/linear/scaled_mm/aiter.py (372+),
model_executor/layers/fused_moe/experts/rocm_aiter_moe.py (310+),
model_executor/layers/attention/mla_attention.py (629,903,1142+),
model_executor/models/deepseek_v2.py (Indexer, 665+),
v1/attention/backends/mla/rocm_aiter_mla_sparse.py (422+).

No claim of matched accumulation or intermediate rounding from a top-level FP8
label. Static route evidence is not a complete runtime tensor-dtype trace.
Source comments previously generalized BF16 wk to wq_b; corrected, no behavior
change. The pinned reference's wq_b keeps quant_config.

## Campaign consequence

Existing 39.12 vs134.03 tok/s cell is same workload, different arithmetic: retain
as diagnostic, NOT strict apples-to-apples evidence. Prior block wins qualify
only their current W8A16/BF16 paths, not the new precision contract. Same top-1
tokens and the large reference-repeat logit floor cannot establish dtype parity.
New serve-bench records explicitly carry precision_qualified=false and manifest
axes, alongside the existing numerics_qualified=false; no automatic promotion.

Next work: capture active per-op dtypes and quantization boundaries from the
pinned reference, including all unquantized exceptions and accumulation. Adapt
gfx950 W8A8/block-scale linear and MoE paths (existing native adapters are gfx942
only), then align MLA BMM/indexer. Match prefill and every decode rung before
rerunning quality gates and the timed matrix. Preserve scales, clipping/rounding
rules, outputs and collective dtypes; do not fuse across a reference quantization
boundary without reproducing its rounding.

## Device evidence, 2026-09-22

- Named worker-extension inventory completed job1790099599948719131-c3974253:
  /opt/models/plow-glm53-precision-audit-v2-20260922/reference/precision.json,
  sha256 f4b3b00f164dcf2aadfa175c62943d52a622e13d0fced14ff0c301ffa293973e.
  All8 ranks have78 layers and1914 named modules. No serialized callbacks or
  insecure-serialization override. Loaded metadata, not a runtime arithmetic trace.
  Confirmed W_K/W_V FP8 withFP32 scales; mainKV BF16; indexer cache packeduint8;
  router weightBF16, outputFP32, correctionbiasFP32 (distinguish weight vs output).
- Native W8A8/block128 GEMM isolated test passes14/14 shapes ongfx950:
  /opt/models/plow-glm53-a8w8-block-kernel-20260922,
  job1790100010721016155-e6ce9f3d done0. Alloutputs vs f64; maxrowrelativeL2
  <=.001961. Rungs1/8/16/32/64, realGLM N/K pairs, zero rows andraggedtails.
  Assembly has native unscaled v_mfma_f32_32x32x64_f8f6f4; originalFP32 scales
  appliedper128K. No MXFP8 requantization. No activationquant/serving routeyet.
- Sameoperands pinnedAITER check11/12 alignedshapes pass;2raggedshapes f64only.
  /opt/models/plow-glm53-a8w8-block-aiter-20260922 job1790100176107615504-b8f8293a
  FAIL due reference repeat instability atM8N6144K2048. Auditrerun
  /opt/models/plow-glm53-a8w8-block-aiter-audit-20260922,
  job1790100325366762742-62b2221f, confirms CK tunedkernelId8 splitK2;
  reference-repeat maxrowrelL2=.00192760, candidate/reference=.00365027.
  Other11shapes repeatbitwise, candidate/reference maxrowrelL2<=.000044264.
  Saved bothreference BF16outputs; do not promote this as whole-kernel parity.
  Need audit splitK intermediate/output rounding before reproducing the route.
  GenericFP32partialaccumulation alone does not prove matching every dtypeboundary.
- Activationquant source: QuantFP8.forward_hip usesAITER group_fp8_quant when
  is_linear_fp8_enabled; get_hip_quant(per_1x128) calls per_group_quant_hip,
  reshapesinputinto128elements andcalls dynamic_per_token_scaled_quant.
  Do not copy the Tritonfallback epsilon/rounding withoutchecking activeHIPpath.
- Native d_quant_fp8_block128 now implemented as4threads/group,32elements/thread,
  scale=max(amax,1e-10)/448,FP32scalegroup-major forGEMM. Isolatedtestonly; caller
  mustenforce K%128=0 whenaddingpacketroute. gfx950 HIP build passes.
  /opt/models/plow-glm53-quant128-20260922 job1790100660281787609-db3cdacc done0:
  15/15 rungs1/8/16/32/64 xK256/2048/6144 bit-identicalFP8bytesAND FP32scales
  vs pinnedvLLM rocm_aiter_ops.group_fp8_quant, referencesrepeatbitwise.
  Inputscoverzero,small/tinyvalues,broadexponentrange,mixedsignsandordinaryBF16.
  Syntheticboundaryqualificationonly; no realactivation/servingperfclaim.
  ComparatorCPUtests3pass, exactshapecompletenessrequired, diffcheckclean.
- Opt-in `PLOW_GLM_OPROJ_W8A8=1` plus `GLM_LINEAR_FP8=1` now emits explicit
  QuantFp8Block128(184) -> GemmFp8Block128(185) for decode and prefill o_proj.
  Original FP8 weights, per128 FP32 activation scales, per128x128 FP32 weight
  scales, FP32 accumulation, BF16 output. Scratch sized from local head*V width,
  not hidden. TP1/8 and decode1/8/16/64 + prefill128 emitter tests pass, including
  complete quant dependency and same-segment constraint. Default route unchanged.
  Manifest says mixed activations, not whole-packet W8A8; 53 manifest tests pass.
  Full78-layer linear overlay copied verbatim to
  /opt/models/GLM-5.3-full-aca966e4-plow-linear-fp8 (10.69GB, base untouched).
- First B16 packet trial job1790101659220226736-e663c585 failed before execution:
  mixed MLA/MoE selects interp_decode_gq, not the separate PLOW_FP8 object.
  Compiled-opcode marker correctly refused absent QuantFp8Block128 handler.
  Preserve /opt/models/plow-glm53-oproj-w8a8-b16-check-20260922 as failed evidence.
  Revised integration: block128 arms independently compiled ONLY when paired
  PLOW_HAS_* inventory declares them, outside the per-row PLOW_FP8 object axis.
  Keep explicit per-op markers; do not broaden Variant::detect and select a
  nonexistent fp8+MLA/MoE object. Fresh object/runtime rebuild pending GPU check.
- Revised objects built successfully (decode/GQ256VGPR,occ2,6spills). Trial
  job1790102098933141862-dc11d29f executes but fails the unchanged block oracle:
  act.attn row4 relL2=.8734 (limit.03). Boundary comparison against prior B16
  shows unchanged router IDs, xmid rowrelL2<=.000031, xn2<=.000161, but shared
  expert output rows4..15 are ALL ZERO. Existing d_gemv_fp8_blk uses fixed MM4
  without walking M16. GLM_LINEAR_FP8 exposes this for shared projections;
  not an o_proj quantization error. Fix shared entry by iterating bounded MM
  bands with LDS reuse barrier, keep M<=MM fast path. Extend existing f64
  kernel test to all rows at M1/5/8/16/32/64 on shared gate/down dimensions.
  Fresh v3 campaign build pending; no threshold change or performance promotion.
- v3 DONE: /opt/models/plow-glm53-oproj-w8a8-b16-v3-check-20260922,
  job1790102421549270458-353e47b0 done0. B16/ctx512/TP8 all stages and rows
  pass unchanged .03 gate; final residual maxrowrelL2=.0003425801 on all8
  ranks, rank identity/finiteness pass. Median1509.283us from30 samples,
  host prepare-dispatch-drain-audit clock. Single diagnostic, NO accepted
  speedup, NO whole-block/reference precision qualification, NO8K serving claim.
  Old standalone kernel reproduces missing rows (10/12FAIL), job
  1790102322020010259-62210e53. Fixed same host regression passes12/12, job
  1790102421785737398-30f7e8a4 (all elements, max normalized error<=.0038).
  Decode/GQ remains256VGPR/occ2/6compiler spills. CPU checks: releaseHSA,
  devgenknob12, runtimeknob19+1ignored, manifest53, projectionrouting2,
  ABI1/opcodes4, failclosedmarkers1, TPweight/scale-sharding1. Index hash
  remains cce53d1bfa6e92fdc53a65e735538a8c26355a3eb98dbb22f9eb51f351870cbf.
  Next: real o_proj input/quant/scale/output boundary capture vs pinned AITER,
  including splitK intermediate rounding; prefill GPU validation and remaining
  rungs, then remaining W8A8/MLA/indexer precision paths before serving promotion.
- Real o_proj capture at /opt/models/plow-glm53-oproj-boundary-b16-20260922:
  all8 ranks activation FP8 bytes and FP32 scales match pinned AITER bitwise.
  Original weight mismatch is signed zero canonicalization (rank0:84 bytes of
  12582912, all0x80->0x00); native-only gfx950 operands now bypass legacy scrub.
  Shared legacy consumers retain canonicalization. CPU regression passes;
  fresh GPU capture of this loader fix remains pending.
- Pinned CK source audit: gemm_a8w8_blockscale.cu computes KBatch=1<<splitK.
  kernelId8/splitK2 at M8/16,N6144,K2048 means FOUR512-K partitions.
  FP32 accumulation -> BF16 partial -> BF16 atomic adds. Source chain:
  device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp ->
  gridwise_gemm_xdl_cshuffle_v3_multi_d_ab_scale.hpp ->
  threadwise_tensor_slice_transfer_v7r3.hpp -> amd_buffer_addressing.hpp
  (__builtin_amdgcn_global_atomic_fadd_v2bf16). Reference repeats differ;
  do not relax the current comparator or claim bitwise equivalence.
- User requests CDNA4-specialized kernels, assembly where justified. Candidate
  GemmFp8Block128Split4(186) emits independent BF16 partials followed by three
  ordered BF16 Residual adds in the SAME AOT segment. Only known M8/16 TP8
  o_proj shapes select it under the existing opt-in. 48->192 independent tiles;
  no claimed performance win, GPU/precision-matched T4 validation pending.
  Deterministic addition order differs from reference atomic ordering; dtype
  boundaries alone do not establish full arithmetic or serving qualification.
  CPU checks: projection routing/dependency tests2, ABI1, devgen knob12,
  runtime cuda+hsa knob19+1ignored, manifest1 pass. gfx950 standalone wrappers
  compile successfully (/tmp/plow-split4-gfx950-4D0zNB/test_kernels.co).
  No GPU execution or timing of split4 yet; existing frozen v3 objects unchanged.
- Topology question: keep TP8/C16 as control; test two independent TP4/C8
  replicas before a fixed TP4-prefill + TP4-decode split for 8K throughput.
  This is a hypothesis, not a measured recommendation to deploy. Built-in
  vLLM MoE DP may span experts across TP*DP, so distinguish it from independent
  replicas and pin EP/sharding on both arms. TP4 changes o_proj K2048->4096
  and local attention heads, requiring a separate AOT/precision audit.
- Split4 GPU validation DONE, job1790103884259270645-b12b35c9:
  /opt/models/plow-glm53-split4-kernel-20260922. All10 cases/fourpartials/every
  element pass f64 rowrelL2<.004 (max.002134), including M8/16,N6144,K2048,
  raggedN130, M65, K512/6144; all14 unsplit regressions pass. Disassembly
  confirms native v_mfma_f32_32x32x64_f8f6f4. No MX scales substituted.
- Fresh native-only weight loader capture DONE, job1790104094532937302-1199b0d9:
  /opt/models/plow-glm53-oproj-boundary-b16-v2-20260922. Named CK reference
  job1790104223217665700-22969c94 verifies original FP8 weight bytes, scalegrid,
  activationFP8 and activationFP32 scales bitwise on all8 ranks. Fullblock
  gate passes. Strict comparator remains FAIL because active CK repeats differ.
  Earlier reference job1790104094923596205-a38e3bab was intentionally stopped
  during unnecessary tune-module JIT; its frozen script/log remain. Replaced
  audit with already-compiled gemm_a8w8_blockscale_ck using the exact selected
  kernelName, splitK0 perK/4slice. No reference-serving route override.
- Split4 fullblock DONE, job1790104241176156178-4c30c959:
  /opt/models/plow-glm53-oproj-split4-b16-check-20260922 uses fresh build
  /opt/models/plow-glm53-oproj-split4-b16-v2-20260922. B16ctx512TP8 allstage
  gates pass, finalmaxrowrelL2=.0003422509663;1563.522us diagnostic median
  (30samples,host-prepare-dispatch-drain-audit), NO speedup claim. Decode/GQ
  resources unchanged256VGPR/occ2/6spills. The earlier non-v2 build used old
  plowc during a concurrent rebuild; marked NOT_QUALIFIED.md, never GPU-scored.
- Split4 pinned CK boundary audit job1790104266776425523-a090c684 remains
  FAILrc1 under unchanged strict reference-repeat-bitwise gate. All8ranks
  weights/quant/scales match bitwise. Same selected CK kernel on fourK/4slices
  with ordered BF16 adds: maxrowrelL2 drops from.00394425(unsplit) to.0000839905
  (split4); rank1bitwise. Active atomic route difference<=.00305712 while its
  repeatdifference<=.00264961. This isolates the intermediate rounding effect,
  NOT a full arithmetic/serving qualification. Six comparator CPUtests pass,
  including a BF16 intermediate-rounding regression. git index unchanged.
  Next perf candidate: combine three Residual ops into one ordered BF16
  reduction, preserving EVERY intermediate round; T4 against this split4
  packet (not unsplit, whose intermediate precision differs). Then remaining
  projection/MoE/MLA/indexer precision paths and prefill/rung coverage.
- User clarifies: retain megakernel when measured data shows an advantage.
  No architecture-wide preference for megakernel, segmented, or vendor paths;
  select by shape/rung under the same precision contract and end-to-end gates.
- Sum4Bf16(187) candidate replaces three Residuals with one in-place ordered
  four-input addition in the SAME AOT segment. BF16 round after EACH addition;
  vector16B loads/stores, tail-safe; logicaltensortraffic18->10B/element.
  No new runtime knob; explicit object inventory/marker, knob define registered.
  Only existing opt-in native o_proj split4 shapes select it. GPU standalone
  job1790104624315062014-a80909c5 passes16/16 bitwise against three existing
  Residual GPU calls, including signedzero, rounding-sensitive cancellation,
  broadexponents, n1/7/8/9/2049/49152/98304/131073, blocks1/32, guardtails.
  Frozen /opt/models/plow-glm53-sum4-kernel-20260922. Disassembly confirms16B
  global loads; explicit rounding remains integer-conversion/FP32 arithmetic.
  CPUchecks: projectionrouting2, ABI1, manifest53, devgenknob12,
  runtimecuda+hsa knob19+1ignored, failclosedmarkers1; releaseplowc/HSA pass.
  Paired candidate build /opt/models/plow-glm53-oproj-sum4-b16-20260922 DONE.
  Four-arm comparison /opt/models/plow-glm53-oproj-sum4-b16-ab-20260922,
  job1790105082003640852-8c88e500 DONE0, B16/ctx512/TP8, 200 samples/arm,
  20 warmups, same lease/runtime/inputs: medians1560.573/1523.733/1556.633/
  1523.563us. Mean arm medians1558.603->1523.648us, saving34.955us=2.24271%;
  noise11.290us, drift3.940us, treatmentspread.170us, stable gatePASS.
  All120 captured output files bit-identical between every arm; all numerical
  gates pass. Retain fused reduction in existing opt-in path, no default/full-
  model promotion. This compares two same-segment megakernel packets, NOT
  megakernel versus segmented/vendor execution; no architecture-wide verdict.
  B8 and larger-context coverage remain pending, as does whole-model precision.
- Next precision gap: shared expert. Pinned loaded layer3 has merged gate/up
  FP8[512,6144] and FP32[4,48] scales, down FP8[6144,256] and FP32[48,2]
  atTP8. DeepseekV2MLP uses quantized gate/up -> BF16 SiluAndMul -> quantized
  down; shared experts are NOT folded into routed experts in this inventory.
  New opt-in PLOW_GLM_SHARED_W8A8 reuses QuantFp8Block128/GemmFp8Block128
  for gate/up/down, retains explicit BF16 gate/up and SiLU boundaries, same
  AOT segment, independent of the o_proj knob. Requires gfx950/GLM_LINEAR_FP8,
  aligned local K and no shared folding. Existing routes/defaults unchanged.
  CPU routing/dependency tests cover decode1/8/16/32/64 and prefill128 atTP1/4/8;
  both tests pass. Original-checkpoint/capture comparator tests7 pass; devgen
  knob12 and runtimecuda+hsa knob19+1ignored pass. GPU validation pending;
  no arithmetic-parity or speedup claim. Compare merged reference gate/up,
  not two independently selected reference GEMMs, then isolated SiLU/down
  and the complete shared chain. Keep strict reference-repeat/0.004 gates.
- Shared W8A8 paired build DONE, /opt/models/plow-glm53-shared-w8a8-b16-20260922.
  First GPU capture job1790105801799116121-75231019 FAILrc1: act.attn row5
  relL2=.0301740236 > unchanged .03 gate. Capture retained at
  /opt/models/plow-glm53-shared-w8a8-b16-check-20260922; only rank0 dumped before
  old harness's early return. No timing qualifies. The oracle still uses the
  old dequantized-weight/unquantized-activation shared path; this is a possible
  explanation, NOT established until original-weight AITER boundaries are checked.
  Extend failure diagnostics to dump every rank and write oracle_verified=false
  with errors and NO timing samples; comparator may inspect these boundaries
  but MUST keep overall passed=false when the block oracle failed. No gate change.
  CPU GLM suite77, manifest53, block-oracle2 pass. Roofline now costs nativeFP8
  GEMMs at an explicitFP8 ceiling (split4 does not quadruple GEMM work);11
  roofline/T4 tests pass. Quant/activation/partial-reduction traffic still excluded.
- All-rank diagnostic capture job1790106141362955663-6a67e234 FAILrc1,
  /opt/models/plow-glm53-shared-w8a8-b16-audit-20260922. Same row5 gate failure
  on all8 ranks, oracle_verified=false and no timing samples. Pinned reference
  v2 job1790106215122112125-0645a6fb completes audit but FAILrc1: originalFP8
  weights/scales and both quantization boundaries match bitwise on all8 ranks;
  gate/up maxrowrelL2<=.000456260, down<=.000088690, reference repeats bitwise.
  SiLU maxrowrelL2=.004143748; complete shared chain<=.032572588, fails. The
  first reference job1790106141757240336-bffe9e5c failed at CustomOp construction
  without a vLLMConfig. Retained; v2 calls the exact torch.ops._C.silu_and_mul
  used by installed SiluAndMul.forward_cuda, no synthetic config or route override.
- Rounding isolated from saved reference outputs: CPU FP32 SiLU -> BF16 ->
  multiply BF16 up -> BF16 matches ALL32768 reference SiLU elements bitwise.
  Single final round differs on1071..1166 elements/rank. This is a real missing
  arithmetic boundary, not merely the older oracle's precision mismatch.
  Candidate Glu act5 adds that intermediate BF16 round only for sharedW8A8;
  legacy act1 unchanged. Explicit plow_glu_silu_bf16_1 object marker required
  by runtime for act5; failclosed CPUtest passes. Fresh v2 paired build and
  runtime rebuild in progress; do not promote until GPU/chain gates are rerun.
- Corrected v2 build/runtime DONE. Capture job1790106695033558979-e38f4dde
  FAILrc1, /opt/models/plow-glm53-shared-w8a8-b16-v2-check-20260922: unchanged
  act.attn row5 gate=.0301293812>.03 on all8 ranks, no timing samples.
  Reference job1790106695426506969-3cecfd8d FAILrc1 with completed audit:
  every captured original weight/scale and both activation quantizers bitwise;
  corrected SiLU bitwise on all8 ranks/32768 values. Isolated gate/up and down
  pass on every rank, stable references. Complete chain passes7/8 ranks, but
  rank0 maxrowrelL2=.00490594155>.004; strict gate unchanged, NOT qualified.
  Rank0 merged gate/up differs at only(row,column)=(0,500),(2,434),(15,129):
  candidate/ref .03173828125/.031982421875, -.0004444122314453125/
  -.0004425048828125, -.00084686279296875/-.000850677490234375. Next audit is
  the GEMM accumulation/rounding feeding FP8 requantization; do not attribute
  all remaining failure to the old oracle or relax the threshold. Then align
  the independent block oracle to audited precision, other rungs/prefill, and
  routed MoE/MLA/indexer. No perf/default/fullmodel promotion.
  Decode/GQ256VGPR/occ2/6spills unchanged. Projection tests2, failclosed SiLU
  marker1, devgenknob12, runtimeknob19+1ignored, comparatorCPU8, roofline/T4
  CPU11 pass; earlier fullGLMemitter77 and manifest53 pass. Index unchanged.
- Remaining shared-chain audit: pinned CK default uses native16x16x128 MFMA
  and combines activation/weight scales before accumulation. CPU reconstruction
  of the three rank0 differing outputs shows scale association alone does NOT
  explain their BF16 differences. Added unselected CDNA4-only native16x16x128
  diagnostic helper/wrapper and a8w8-m16 mode in the existing standalone tester.
  No packet/interpreter dispatch change. GPU job1790107364843424413-7829bc78
  DONErc0: all14 f64 tests pass, rungs1/8/16/32/64 plus ragged dimensions;
  maxrowrelL2=.001961<.004. Frozen source/object/test/captures at
  /opt/models/plow-glm53-fp8-m16-kernel-20260922. These synthetic results do
  NOT resolve captured shared-chain parity or establish a performance win.
  Next: replay real shared operands through this candidate, compare pinned
  merged gate/up reference, then matched-precision block T4 only if qualified.
- Real-input replay job1790107603747806682-7c0c4e70 DONErc0,
  /opt/models/plow-glm53-shared-m16-replay-20260922: native16x16 candidate
  bitwise identical to pinned reference on ALL24 GEMMs (gate/up/down x8 ranks),
  twice per input, including the three previously differing rank0 outputs.
  Existing32x32 replay also runs as control; numerically passes but differs
  bitwise. New reusable capture exporter verifies original checkpoint shards,
  audit operand hashes, quantization gates and reference repetition; no full
  block/serving qualification inferred. Host tester now replays captured inputs.
  Candidate selection added ONLY to existing sharedW8A8 opt-in, rows16/H6144/
  localinter256. GemmFp8Block128.i3=16 requires CDNA4 m16 object marker;
  unknown selectors/old objects fail closed. Other shapes/defaults unchanged.
  Shared emitter2, marker1 and comparator/exporter9 CPUtests pass. Compiler
  release rebuild DONE; runtime and paired-object builds also DONE.
  /opt/models/plow-glm53-shared-w8a8-b16-m16-20260922; decode/GQ remains
  256VGPR/occ2/6spills. No perf promotion.
- M16 full-block capture job1790108095764886085-ef9f76ba FAILrc1 at unchanged
  old-oracle act.attn row5=.0301293812>.03, all8 ranks; no timing samples.
  /opt/models/plow-glm53-shared-w8a8-b16-m16-check-20260922. Pinned boundary
  job1790108121482952189-435dec61 completes FAILrc1 ONLY because block oracle
  failed: boundaries_passed=true; ALL8 ranks gate/up, SiLU, down and FULLCHAIN
  bitwise, both quantizers/original weights/scales bitwise, repeat-bitwise.
- Added explicit --shared-w8a8-tp to existing glm52_real_oracle.py, inputs-only:
  pinned vLLM0.29/AITER merged gate/up, BF16 SiLU, nativeFP8 down per TP shard,
  two repeated evaluations; FP32 sum of BF16 TP reference partials. Inputs
  remain independent HF post-attention normalization, not candidate outputs.
  No gate relaxation and no change to legacy default shared math. First job
  1790108286658021896-c5d1dc2a failed missing frozen glm52_prep.py dependency;
  retained. v2 job1790108341437807285-8c6b7ec9 DONErc0 at
  /opt/models/plow-glm53-shared-w8a8-oracle-b16-v2-20260922. All4 input files,
  xmid and xn2 references byte-identical to old inputs; only FFN/residual
  reference changed. Metadata states exact shared-reference precision/TP.
- New-oracle block job1790108406186786648-15b533ae FAILrc1 at act.attn
  row11=.03017873095>.03; no timings. At
  /opt/models/plow-glm53-shared-w8a8-b16-m16-parity-check-20260922. Reference
  job1790108425081296476-977828da again confirms ALLshared boundaries/fullchain
  bitwise on8 ranks but overallfailed. Independent HF upstream xn2 differs
  .00278..00312 perrow; FFN=.02194..03018, residual<=.000414. Need investigate
  normalization/routed precision rather than relax threshold.
- New --block-norm diagnostic: standalone AITER RMSNorm on captured rounded
  xmid differs from Plow xn2 by .001477 worstrow (10014elements), whereas CPU
  BF16 intermediate-before-weight model differs .003011 (28070elements).
  job1790108599660106096-9c7b4e44 DONErc0, but this does not match the packet's
  fused AddNorm boundary. Now reconstruct ordered FP32 TP oproj sum->BF16,
  feed original residual + attention to installed vLLM fused AITER norm, and
  require reconstructed residual bitwise matches captured xmid. First fused
  v2 job1790108719671643129-bc65d8ab failed because decorated IrOpImpl requires
  .impl_fn, not direct call; fixed. v3 job1790108754627978539-04887ff9 DONErc0
  at /opt/models/plow-glm53-norm-audit-b16-v3-20260922: fused AITER norm AND
  residual match Plow BITWISE on all8 ranks, repeated stable. Standalone norm
  result above is not the packet's executed arithmetic; keep distinction.
  Oracle now has explicit --vllm-post-norm (inputs-only), applying installed
  vLLM0.29 fused norm to independent HF attention output + original residual,
  not candidate operands. Requires repeat-bitwise and unchanged BF16 residual.
  v3 oracle job1790108862583830934-6d116ab8 DONErc0 at
  /opt/models/plow-glm53-shared-w8a8-oracle-b16-v3-20260922. Next full-block
  capture uses this new reference with unchanged thresholds and same packet.
- v3 block job1790108924515301492-971074bb DONErc0 at
  /opt/models/plow-glm53-shared-w8a8-b16-m16-parity-v3-check-20260922:
  all unchanged rowwise gates PASS, all8 ranks identical/finite. MaxrowrelL2:
  xmid=.000311047, xn2=.001527648, FFN=.019942062, residual=.000381612.
  Inputs and HF residual reference remain byte-identical to original fixture;
  postnorm and shared reference arithmetic now explicitly aligned as audited.
  Diagnostic30sample/5warmup median2232.806us; NOT an A/B performance win.
  Pinned reference job1790108924901751596-48a57393 DONErc0: passed=true and
  boundaries_passed=true, all8 ranks gate/up/SiLU/down/fullchain BITWISE,
  original weights/scales and both quantization boundaries BITWISE. Overall
  precision_qualified=false still: other rungs/prefill/routedMoE/MLA/indexer
  and full-model serving have not met the reference contract. Defaults stay off.
  ComparatorCPU9 and Pythoncompile pass; git diff check/index preservation pass.
  All build/GPU jobs above are terminal; no outstanding process to wait on.
  Next perf baseline is this qualified B16 single-block candidate. Inspect
  native16x16 load instructions and test aligned vector loads/pipelining under
  bitwise replay + four-arm same-precision block A/B; no old W8A16/failed-chain
  candidate as a claimed apples-to-apples performance control. Then finish
  routed W8A8 / MLA / indexer precision and expand rungs/context/serving matrix.
- Native16x16 disassembly confirms scalar global_load_ubyte operand loads.
  Candidate adds guarded 128-bit loads (two per32-byte operand) for K%128=0
  and16-byte-aligned A/B. Scalar bounded fallback retained for ragged/unaligned
  input; accumulation/scale association/BF16 rounding unchanged. Existing
  unit mode extended with packed-K/ragged-N case(3,130,256), outside the fixed
  14-capture matrix. Fresh paired build DONE at
  /opt/models/plow-glm53-shared-w8a8-b16-m16-vec-20260922. Decode/GQ remains
  256VGPR/occ2/6spills. Disassembly confirms FOUR global_load_dwordx4 plus
  native16x16x128 MFMA in packed path; original scalar fallback retained.
  Unit/replay job1790109297709904620-de9b9c8f DONErc0 at
  /opt/models/plow-glm53-shared-m16-vec-kernel-20260922:15/15 f64 cases pass,
  all24 captured GEMMs bitwise to pinned reference AND scalar-m16 baseline,
  repeated stable. Frozen compiled header precedes whitespace-only indentation
  cleanup; device-only syntax check of final source passes.
- T4 job1790109327142953395-7595732e DONErc0 at
  /opt/models/plow-glm53-shared-m16-vec-b16-ab-20260922, B16/ctx512/TP8,
  200samples/20warmup each arm. ctl2231.666,treat1563.023,ctl22233.457,
  treat21558.673us. Mean-arm2232.5615->1560.848us, savings671.7135us=30.08712%.
  Drift1.791us, treatmentspread4.350us, noisefloor14.479us, stability/gain gates
  PASS. All216 outputfiles/arm BITWISE; control also bitwise to pinned-audited
  v3 baseline capture216files. Identical packet hash c0e120f11b3044f9, same
  runtime/inputs/full linear-FP8 checkpoint. RETAIN vector load change on
  experimental native16x16 route; no default/full-model/serving promotion.
- Separate instrumented trace job1790109437534395002-a2c3dce7 DONErc0 at
  /opt/models/plow-glm53-shared-m16-vec-b16-trace-20260922, all8 traces captured.
  Existing glm53_trace_attrib.py now accepts dev_isa.h enum/#define names as
  well as CSV; CLIregression coversall3formats. Rank0/rank7 similar envelopes:
  op86 MoEdown span.309/.300ms, shared185 aggregate.273/.279ms,
  op85 MoEgateup .221/.213ms, Gemv aggregate.239/.236ms. Envelopes overlap and
  include gate/arrival effects: NOT per-op wall-time attribution or critical
  path proof. Trace wall1.278/1.285ms at100MHz, instrumented not perf evidence.
  Alljobs terminal; next prioritize pinned routedMoE A8/rounding audit and
  native/shared scheduling based on this passed baseline. Remaining precision,
  other rungs, 8K..70K, C8..64 and full-model3000tok/s goal remain OPEN.
  TraceCLItest1(3formats), device syntax and diff/index preservation pass.
- Routed-MoE audit added to existing block_fp8_aiter_compare.py (--block-routed),
  CPUtests12 pass. Original checkpoint gate/up TP rows and down TP columns,
  FP8 bytes (including negative zero) and FP32 scales preserved; captures use
  Plow xn2/routes and weighted FP32 part, so this isolates routed experts and
  does NOT qualify router/shared-add/TP reduction/full-model precision.
  Job1790110238791980258-c9c6f19f DONErc0 at
  /opt/models/plow-glm53-routed-audit-b16-20260922; boundary-repeat extension
  job1790110364302618898-78967c37 DONErc0 at
  /opt/models/plow-glm53-routed-audit-b16-v2-20260922. All8 ranks captured using
  pinned vLLM0.29.0 image, E256/H6144/I256/top8/B16/TP8 actual original shards.
  audit_complete=true, passed=false, precision_qualified=false intentionally.
- Active route is CK TWO-STAGE DEFAULT, block_m16, splitk0, no fused-quant,
  not flat/one-stage ASM. Stage1 input FP8E4M3 [16,6144], scales FP32[16,48];
  stage1 output BF16[16,8,256]; stage2 input FP8E4M3 same shape, scales
  FP32[16,8,2]; final routed output BF16[16,6144]. All stage1 input/scales/output
  and stage2 input/scales repeat BITWISE on all8 ranks. Only stage2 output
  differs on every rank; maxrow reference-repeat relL2 .00423446..00455637
  in v2. Original audit Plow routed-only maxrow relL2 .0395533..0463962;
  old W8A16 route is materially mismatched, no accuracy gate relaxed.
- Pinned CK source clarifies non-shared rounding: fused stage1 uses FP32
  gate/up accumulators -> FP32 SiLU(gate)*up -> BF16 output. Do NOT reuse
  shared-expert act5's BF16 SiLU/intermediate boundary. Stage2 multiplies
  FP32 accumulator by FP32 route weight, then BF16 atomic output accumulation.
  Source: aiter_meta/csrc/ck_gemm_moe_2stages_codegen/
  gemm_moe_ck2stages_common_blockscale.cuh and composable_kernel/include/ck/
  tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp (1590+,1648+),
  device/impl/device_moe_gemm_blockscale.hpp (MemoryDataOp at318+).
  Exact internal heuristic CK instance name remains unobserved (kernelName="").
- CPU bitwise check confirms captured existing native shared-input quantizer
  act.sh_xq and act.sh_xs (transpose group-major scales) equal routed AITER
  stage1 FP8 bytes and FP32 scales on all8 ranks. Reuse this xn2 quantization
  for routed input; implement native grouped W8A8 gate/up with the audited
  FP32 activation epilogue, then group128 quant and weighted BF16 down/reduce.
  v2 saved both repeats for every boundary to replay native kernels without
  conditioning expected stage1 output on a candidate. No serving/default or
  megakernel-vs-vendor promotion from this audit. Full3000tok/s and matrix OPEN.
- Native routed gate/up core implemented as GLU specialization of the validated
  d_gemm_fp8_block128_m16. Native CDNA4 16x16x128 FP8 MFMA, shared activation
  loads, separate FP32 block128 gate/up scales, FP32 SiLU*up then BF16 store.
  No packet/runtime route yet. Unit wrapper consumes contiguous [gate;up]
  weight and scale halves; production grouped adapter must handle separate
  expert weight-table pointers and token gathers, not assume contiguous prep.
- Existing comparator adds --export-routed: verifies pinned audit, repeat and
  checkpoint shard hashes, groups captured token/slot rows by expert, exports
  all224 expert cases/all1024 routed rows/eight ranks (M1..13,N256,K6144).
  Replay format GLU mode stores two weight/scale halves and one output half;
  original GEMM mode unchanged. First CPU export stopped on PyTorch singleton
  transpose stride (M1 scales [48,1], contiguous but last stride48). Writer now
  flattens before byte view; regression added, old partial cases retained.
  Valid cases at /opt/models/plow-glm53-routed-glu-m16-20260922/cases-v2,
  manifest sha256 4640cb25219e8c117a15b13b71c2657e8e70013a7bf34dafc9a8d481d2eab68d.
- Initial replay job1790110882559775502-3501b9b4 DONErc0,224/224 numeric PASS
  but3 BF16 elements differed (r4e98,r5e163,r7e41). Pinned CK Silu source is
  x*(1/(1+exp(-x))), NOT x/(1+exp(-x)): reciprocal rounding matters.
  Corrected only new GLU specialization to CK expression; OCML exp preserved.
  Replay job1790111049703708687-9269e031 DONErc0 at
  /opt/models/plow-glm53-routed-glu-m16-v2-20260922: all224 outputs BITWISE to
  pinned stable CK stage1, all262144 BF16 elements; every replay repeat bitwise.
  Original15 non-GLU f64 cases still pass, all14 saved captures byte-identical
  to retained vector-load baseline. Frozen ELF sha256
  95a5198d6ebf6a6ac0e9f93c6a099402b92e5c02c340e21d0c850e770dffb7be.
- GLU f64 rung/tail job1790111190001798112-ae87d150 DONErc0:13/13PASS,
  M1/8/16/32/64 x N256/512,K6144 plus(3,130,256),(3,130,260),(65,129,129),
  zero rows and packed/scalar tails, worst maxrowrelL2 .002204 < unchanged.004.
  CPUtests14PASS, device syntax/diff checks pass, staged-index hash unchanged.
  Final header differs from frozen build only by comments. Next implement
  grouped gather/scatter adapter and routed down FP8 quant/weighted BF16
  reduction, then independent full-block oracle and matched T4 before
  promotion. No throughput/latency claim from isolated correctness replays;
  full-model3000tok/s,8K..70K,C8..64,other precision gaps remain OPEN.
- Native grouped gate/up adapter added in op_moe_fp8_block128.h: existing
  alignment metadata, token gathers, separate expert gate/up weight and scale
  pointers, explicit padded-row zeroing. No production packet/runtime route.
  Artifact /opt/models/plow-glm53-routed-grouped-glu-20260922;
  job1790111810922701306-f91bbf97 DONErc0:48/48 checks bitwise to captured CK
  stage1, all8 ranks x B16/replicatedB96 x grids1/7/256. Padding and repeat
  checks pass. B96 is replicated input for tiling validation, NOT a B96 CK
  dispatch/performance result. Regression1790111897967954919-5d336dde passes
  28 f64 cases,14 unchanged normal captures and224 prior dense GLU replays.
- Weighted-down isolation added to pinned comparator: preserve selected CK
  stage2 callable, shapes and quantized operands; zero only other experts'
  route weights. Full reference and its repeat retained before isolation.
  Artifact /opt/models/plow-glm53-routed-down-isolate-20260922;
  job1790112051283743033-7ccb85e1 DONErc0:224 isolated expert cases across8
  ranks are finite and repeat-bitwise. Whole routed reduction remains
  nondeterministic; isolated references do not qualify full MoE precision.
- Native weighted-down m16 specialization applies FP32 route weight AFTER
  FP32 accumulation, then stores BF16. Artifact
  /opt/models/plow-glm53-routed-down-native-20260922;
  job1790112283712583856-060efbba DONErc0:224/224 cases bitwise to isolated
  pinned CK, all6291456 BF16 elements, repeats bitwise. Independent CPU replay
  comparison confirms every element. Final regression
  job1790112462523146196-f9712dbc DONErc0:28 f64 cases,14 normal captures
  unchanged, grouped B16/B96 rank0 at grids1/7/256 unchanged. Logs archived.
  CPUtests16 pass; final device syntax, diff check and staged-index hash pass.
- Remaining: grouped down/scatter, inter-stage block128 FP8 quantization,
  matching BF16 reduction and experimental AOT integration, then independent
  block qualification and matched latency A/B. No speed claim or promotion
  from these correctness tests. Keep megakernel wherever matched measurements
  win per shape/rung; retained 30.08712% B16/ctx512 block gain compares two
  megakernel variants, NOT megakernel versus vendor. Full-model goal OPEN.
- Grouped down/scatter implemented in op_moe_fp8_block128.h using native m16
  GATHER+WEIGHTED+SCATTER specialization. Gather FP8 rows and FP32 block scales
  by row_partidx, multiply completed FP32 accumulator by row_gate, store BF16
  directly to token/slot. Separate down weight/scale tables preserved. Missing
  resident expert writes zero to its live destination slots; padding skipped.
  Slot-major quant output chosen so quantization itself performs the layout
  conversion; no separate transpose/scatter copy needed between quant/down.
- Existing replay exporter now verifies original checkpoint hashes, repeated
  stage2 input/scale boundaries, every isolated expert capture/hash/route and
  complete route coverage before exporting grouped down. B96 is six copies
  of B16 reference rows, still ONLY a tiling test, not B96 active dispatch.
  Artifact /opt/models/plow-glm53-routed-grouped-down-20260922;
  job1790113057820370803-c5f01d8f DONErc0:48/48 bitwise checks, all8 ranks x
  B16/B96 x grids1/7/256; output guards, align metadata and repeats pass.
- Group128 quantizer gains compile-time scatter specialization: reads sorted
  BF16 hidden rows, skips padded row_partidx, writes slot-major FP8 and
  group-major FP32 scales directly. Existing default specialization unchanged.
  Grouped helper uses actual padded-row extent from alignment metadata, not
  worst-case capacity. No new knobs or production route enabled.
  Artifact /opt/models/plow-glm53-routed-quant-down-20260922;
  job1790113230700242018-1115eac0 DONErc0:48/48 quant boundaries BITWISE to
  pinned CK stage2 FP8 bytes AND FP32 scales,48/48 downstream weighted BF16
  part outputs BITWISE. Harness feeds captured CK stage1 BF16 in sorted
  layout; this is quant->down validation, NOT a connected native gate/up
  through reduction or AOT block run. Padded hidden rows deliberately NaN.
- Regression1790113266344332592-ea62282e DONErc0:15 normal+13 GLU f64 cases,
  all14 normal saved captures unchanged,15 quant captures byte-identical to
  pinned-qualified original quantizer,48 grouped gate/up checks unchanged.
  All16 quant->down output files equal standalone grouped-down outputs.
  CPUtests16 pass including grouped down export/coverage/hidden hash; final
  device syntax, diff check and staged-index hash pass. Alljobs terminal.
  Next: matching BF16 reduction, experimental AOT route with fail-closed
  object/packet checks, full-block oracle then matched T4. Existing op85 i7
  means PER-TOKEN A8 and op86 i7 means PART16; do not silently reinterpret
  these fields as block128 A8 or change reduction precision. No speed claim,
  no default promotion; full-model3000tok/s and entire matrix remain OPEN.
- Native routed reduction now uses CDNA4 packed BF16 atomic add via compiler
  builtin; disassembly confirms global_atomic_pk_add_bf16 and native FP8 MFMA.
  No CAS loop or hand-written ASM required. Artifact
  /opt/models/plow-glm53-routed-bf16-atomic-20260922;
  job1790113689701577132-d5a187f8 DONErc0:72 checks pass (16 pinned CK bound
  checks,8 ordered native-atomic bitwise checks,24 quantization bitwise checks,
  24 native grouped atomic bound checks). CPU subset-DP computes exact extrema
  over all BF16 addition orders; brute-force permutation selftest topk1..8
  passes. Bounds do not prove every interior value is reachable.
  Native-vs-CK maxrowrelL2 .00432365..00468244; CK repeats .00425288..00449082.
  Expected atomic order nondeterminism, not overall bitwise qualification.
- Regression1790114405149359858-1c2ce805 DONErc0:28 f64 cases,14 normal saved
  captures and15 quant captures unchanged; B16/replicatedB96 all8ranks x3grids
  pass48 quant and48 non-atomic down checks. All16 part outputs unchanged.
- Opt-in PLOW_GLM_ROUTED_W8A8 adds AOT opcodes188/189/190: grouped FP8 gate/up
  -> BF16 hidden -> group128 quant/scatter and reduction-buffer zero -> weighted
  BF16 atomic down. Original FP8 weights/FP32 scales retained. Combine receives
  one BF16 routed result per token; shared add then existing TP seam. All stages
  stay in the modular megakernel segment. No default change. Reject non-gfx950,
  unaligned geometry, BF16 weights, EP/shared fold/vendor/fixed-point routes.
  Registered knob/object flags, slot specs/classes, packed table binding,
  runtime geometry checks and compiled-opcode markers. Emitter tests cover
  TP1/4/8,decode1/8/16/32/64,prefill128 and reject BF16 weights.
- Independent block oracle extended with pinned vLLM/AITER routed W8A8 and
  BF16 shared addition per rank before FP32 TP sum/BF16 output. No candidate
  inputs used. Artifact /opt/models/plow-glm53-routed-w8a8-oracle-b16-20260922,
  job1790114548104130944-83e121dd DONErc0, B16/layer3/ctx512/TP8 original full
  checkpoint. Reference repeat FFN maxrowrelL2 .00244264. Supplied selected keys,
  not learned-indexer qualification; tolerance remains .03. Frozen oracle,
  prep/helper scripts, reference hashes and queue logs retained.
- First paired AOT build /opt/models/plow-glm53-routed-w8a8-b16-aot-20260922
  completed, but block job1790114877830901033-b3e21501 FAILED unchanged gate:
  all ranks FFN zero, relL2=1. Routing metadata and routed hidden zero, shared
  output nonzero. Cause: decode inventory armed PLOW_MOE_PREFILL only from
  old MoeGroupGluPf; replacing it removed align/combine dispatch too. Corrected
  inference to any of the five grouped-family ops. Loader now requires the
  existing family capability marker for each op, so incomplete decode objects
  fail closed. Regression test and all53 manifest tests pass; marker tests pass.
  Rebuild v2 completed with unchanged packet sha60b330c151b056fe; decode/GQ
  remains256VGPR/occ2/6spills. Binary symbols confirm missing family capability
  in v1 and present in v2. No failed-run timing retained.
- CPU verification: packet155, routed emitter2, shared emitter2, devgen knob12,
  runtime knob19 (1 fixture-dependent ignored), routed geometry1, marker1 pass.
  Python compile, device syntax and release build pass; corrected runtime/object
  rebuilds complete. Comparator CPU16 also pass. Prior staged
  index preserved. Megakernel retained where matched per-rung T4 wins; old
  routed W8A16 is not a same-precision speed baseline for new routed W8A8.
- Corrected block job1790115212767708745-672f6710 DONErc0 at
  /opt/models/plow-glm53-routed-w8a8-b16-block-v2-20260922. B16/ctx512/TP8,
  30samples/5warmup: median1550.195us, all rank residuals identical and finite,
  independent oracle passes unchanged .03 gate. All-rank maxrowrelL2:
  xmid .000311047, xn2 .001527648, FFN .02111965, residual .000369461.
  Initial/final block states checked; not every timing iteration captured.
  Experimental AOT route remains off by default. This connected native chain
  passes a modular-block numerics gate; it does NOT close MLA/indexer/other
  precision gaps, prove all reference boundaries or establish full-model parity.
  Fresh baseline job1790115275585757702-53e9ae71 DONErc0 at
  /opt/models/plow-glm53-routed-w8a8-b16-baseline-20260922:200samples/20warmup,
  median1558.284us, same residual error/rank identity and all oracle gates pass.
  This is the new connected routed-W8A8 modular baseline, not a speedup claim
  against old W8A16 or a vendor-vs-megakernel comparison. All jobs terminal.
  Next: audit connected routed boundaries using these captures, then matched
  T4 tuning and other rungs; remaining full-model precision/matrix goal OPEN.
- Connected routed audit now checks actual AOT captures: validates expert counts,
  offsets, full token/slot coverage, FP32 gate bits and padding; unsorts BF16
  hidden rows before comparison. First audit1790115548611364055-824c5e05 failed
  in the reader, not kernels: emitter reserves128-row padding while AMD uses64.
  Reader now derives reserved capacity from consistent capture sizes and checks
  the actual metadata extent. No allocation/kernel behavior changed.
  Corrected job1790115622591185682-69262314 DONErc0 at
  /opt/models/plow-glm53-routed-w8a8-connected-audit-v2-20260922. All8ranks x5
  stable boundaries BITWISE to pinned vLLM/AITER: input FP8, input F32 scales,
  hidden BF16, down-input FP8, down-input F32 scales. Native/CK/repeat routed
  outputs all within exact BF16 addition-order extrema from224 isolated CK
  weighted-down cases. Native/CK maxrowrelL2 .00435830..00451140; CKrepeat
  .00429599..00444017. Audit passed=true but precision_qualified=false:
  candidate-conditioned input/routes, no independent router or whole-model claim.
- Separate trace1790115577247136842-6e51ca70 DONErc0 at
  /opt/models/plow-glm53-routed-w8a8-b16-trace-20260922, all8rank traces retained.
  Rank0 span envelopes: shared GEMM185 aggregate.277ms, quant184 aggregate.274ms,
  routed down190 .155ms, quant189 .150ms, GLU188 .141ms. Tracewall1.276ms;
  spans overlap and include arrival/wait, not critical-path attribution.
  Roofline parser extended for op188/190 with native FP8 ceiling and exact
  align geometry. Artifact /opt/models/plow-glm53-routed-w8a8-b16-roofline-20260922:
  captured28expert union,215256704B,2805989376 useful FLOPs, optimistic34.7188us
  floor at configured6200GB/s/4600FP8TFLOPS. Activation/atomic/fabric/launch costs,
  padded MFMA and repeated tile reads excluded; NOT measured hardware efficiency.
- B16-only quant scheduling retained: non-scatter group128 quant uses four lanes
  per128 values, so bounded workgroup widths2/6/1/6 replace four256-WG packets.
  Total quant workgroups1024->15, same32 instruction / single-megakernel segment,
  no changes to arithmetic, scatter quant, other rungs or default-off FP8 knobs.
  Candidate /opt/models/plow-glm53-routed-w8a8-b16-quantfit-20260922, paired object
  build and release compiler pass. Decode/GQ register budget remains unchanged.
- Atomic-aware T4 gate added to existing comparator/scorer/campaign, selected by
  --routed-reference. Freezes pinned audit, checker and CPU Docker image. Before
  scoring, validates all4arms x8ranks against captured stable boundaries and
  isolated-part BF16 bounds, plus exact per-rank shared-add/FP32 TP sum/BF16
  residual rounding. Every other output must match across all arms. Certificate
  is bound to every raw output hash; stale/incomplete certificates and coexistence
  with --require-bitwise refused. Does not waive the independent .03 block gate.
  Mapped row order and atomic-dependent outputs legitimately vary; blanket
  same-arm raw-byte repeatability would reject the matched reference contract.
- T4 job1790116261693475890-407e4cdc DONErc0, quiet lease and final audit passed,
  /opt/models/plow-glm53-routed-w8a8-b16-quantfit-ab-20260922. B16/ctx512/TP8,
  200samples/20warmup: ctl1552.774,treat1413.785,ctl21556.074,treat21411.345us.
  Mean-arm1554.424->1412.565us =141.859us /9.12614576% reduction. Drift3.300us,
  treatmentspread2.440us,noisefloor10.181us, both gates PASS.288files/arm,
  40control/candidate differences confined to validated mapped/atomic-dependent
  outputs;32 connected arm/rank audits pass, all remaining outputs bitwise.
  RETAIN B16 sizing; no whole-model or megakernel-vs-vendor promotion.
  ComparatorCPU19, campaignCPU18 (includes roofline8, T4scorer5), W8A8emitter17,
  Pythoncompile/diff checks pass. Staged index unchanged. AllGPUjobs terminal.
  Next baseline is quantfit. Other precision gaps (MLA/indexer/projections),
  rung expansion and8K..70K/C8..64 full-model3000tok/s goal remain OPEN.
- QKV-A reference pinned from loaded layer3 inventory and installed vLLM source:
  DeepSeekV2FusedQkvAProjLinear disables TP; concatenates Q2048 + KV512/rope64.
  Original FP8 weights [2624,6144], F32 scales [21,48], loaded row stride6400;
  BF16 input norm/output, group128 dynamic FP8 activation, original continuous
  F32 scales. All8 ranks select AiterFp8BlockScaledMMKernel with use_triton=false.
  Added --export-qkva to the existing comparator, validating checkpoint bytes,
  scale geometry, loaded backend/stride and repeated norm/quant/GEMM boundaries.
  CPU test preserves negative zero and the final64-row scale tail and rejects
  a non-block-aligned concatenation boundary. No MX requantization introduced.
- Reference job1790116707549574685-8b567e64 DONErc0 at
  /opt/models/plow-glm53-qkva-reference-20260922: M1/8/16/32/64/128, N2624,K6144.
  All norm/quant/scale/output repeats bitwise. Pinned CK has no tuned CSV entry
  for these six shapes and uses its default route. Inputs are independent B16
  block fixture rows, sliced/repeated for shape coverage, not six serving runs.
  Native flat replay1790116804729593641-5a6794af DONErc0 at
  /opt/models/plow-glm53-qkva-native-20260922 matches all six outputs BITWISE.
- Added compile-time SPLIT3 epilogue to the native m16 FP8 core and standalone
  wrapper/replay mode. Same arithmetic and rounding; directly stores Q, KV and
  rotary-K to separate row-major buffers, avoiding a subsequent split copy.
  No production opcode/emitter route or new knob. Compiled gfx950 ELF/host test
  frozen at /opt/models/plow-glm53-qkva-split-20260922.
  Job1790117243612922749-0f3086a5 DONErc0: all six split-output replays match
  both native flat outputs and pinned reference hashes BITWISE, including tails.
  Full existing standalone GEMM/GLU, quant, routed down and BF16 atomic regressions
  pass (202 logged PASS checks total); baseline capture comparisons unchanged.
  Comparator CPU20, Python compile, shell syntax and git diff checks pass.
  Frozen sources and all three queue records/logs retained; staged index unchanged.
- QKV split remains a correctness-qualified component, NOT a speed win or full
  precision qualification. Next: preserve original fused weights in checkpoint
  prep, connect three-output AOT dispatch, independently audit the connected
  norm/quant/output boundaries, then matched T4 timing. Keep megakernel or vendor
  segments per measured same-precision block/rung advantage; no global preference.
  B16 quantfit remains the measured production-experimental baseline (9.126%
  modular-block latency reduction). Full-model throughput/matrix goal stays OPEN.
- Connected QKV-A implementation added behind default-off PLOW_GLM_QKVA_W8A8:
  registered emit knob, requires gfx950/GLM_LINEAR_FP8 and aligned geometry,
  refuses seam norm folding. Original fused FP8/F32 tensors replicated on each
  TP rank. Decode and prefill emit group128 quant then op185 with m16 and
  optional split widths i4/i5, BF16 outputs t0/t5/t6. No new opcode or launch
  boundary; existing GEMM modes unchanged. Prefill band-projection shortcut is
  disabled for this route so it cannot silently use the old BF16 projections.
  Loader validates split geometry/operands and dedicated split3 object marker.
- Prep --qkva extends the existing additive linear overlay, streaming original
  Q-A/KV-A records and scales at the aligned concatenation boundary. Fresh
  /opt/models/GLM-5.3-full-aca966e4-plow-qkva-fp8: all156 added tensors across78
  layers hash-verified against original concatenated records,1257819264 bytes.
  Base/original shards untouched. Prep tests2 pass (negative-zero bytes, scale
  tail, shape refusal, legacy single-record writer and resume-header check).
- Block captures extended for norm, quant/scales, Q/KV/rotary-K and fused weights.
  Comparator --check-qkva is CPU-only and checks all8 boundaries per rank against
  the independently seeded pinned reference, binding input and reference hashes.
  Tests include changed-output refusal, wrong input and stale reference rejection.
  ComparatorCPU21, W8A8emitter19, packet155, devgen knob12 and loader block-FP8
  tests5 pass. Initial test field typo and packet slot-doc mismatch corrected.
  Release build completed; frozen runtime
  /opt/models/plow-glm53-qkva-w8a8-runtime-20260922/plowrt sha256
  555e51470aa83fc8796da6972df05e3a416466d7b8b2bf3b403c6ca437c6ac43.
  Paired object build /opt/models/plow-glm53-qkva-w8a8-b16-aot-20260922 completed,
  packet sha3662ecb1f28eb3d1cd474aa2a7ceec3c8e8d32e048d54b16f69ae4793d3f5a9f.
  Decode/GQ now248VGPR/occ2/0spills (previous256/occ2/6); both m16/split3 capability
  symbols present. Runtime knob19 pass (1 fixture-dependent ignored); replication
  test and downstream Q/KV/rotary-K dependency threshold assertions pass.
- Block job1790118126205946567-bf12c20e DONErc0 at
  /opt/models/plow-glm53-qkva-w8a8-b16-block-20260922, doctor passed, B16/ctx512/TP8,
  30samples/5warmup. Median1309.744us, all rank residuals identical and finite,
  existing independent HF-attention/shared+routed-W8A8 oracle passes unchanged
  .03 gate; residual maxrowrelL2 .0003699504. This oracle still has BF16 upstream
  projections and supplies selected keys; it is a broad block numerics check,
  not proof of matching full vLLM attention/indexer arithmetic.
- Independent connected QKV-A audit PASS at
  /opt/models/plow-glm53-qkva-w8a8-connected-audit-20260922/aiter-comparison.json:
  all8 ranks x8 boundaries BITWISE to previously frozen pinned vLLM reference:
  input BF16 norm, activation FP8, activation F32 scales, loaded original FP8
  weight/F32 scales, Q2048, latentKV512, rotaryK64. Input hash matches independent
  B16 fixture exactly. CPU-only checker; source snapshot and queue logs retained.
  Qualified tested QKV-A boundary chain, precision_qualified=false for whole model.
  No timing speedup claim vs quantfit: QKV precision changed, and this was not T4.
  All jobs terminal, staged index preserved. Goal OPEN.
  Next: new baseline is this QKV-A block; regenerate the routed boundary audit
  before T4 because upstream values changed (old routed certificate is NOT reusable).
  Candidate sizing: QKV m16 B16 has164 tiles/8waves, only21 useful workgroups of256;
  test matched width reduction with unchanged arithmetic. Then remaining q_b/MLA
  FP8 absorption, indexer and full serving/rung matrix precision/performance gaps.
- Refreshed routed audit job1790118261712869932-f358e778 DONErc0 at
  /opt/models/plow-glm53-qkva-routed-audit-20260922. New QKV-A upstream captures:
  all8ranks stable boundaries bitwise, native/CK/repeat reductions in isolated
  BF16 addition-order bounds. Use this audit for the next same-precision T4.
  B16-only QKV scheduling candidate narrows256->21 workgroups (164 output tiles,
  8waves/WG). Other rungs untouched. Emitter dependency/width tests and all19
  W8A8 tests pass; release compiler done, paired object build in progress at
  /opt/models/plow-glm53-qkva-w8a8-b16-fit-20260922. No retention decision yet.
- Width candidate paired build completed with packet
  3a3e9d16e03e593a49b6e0d61a7d020587cda23a88bc1d06ce3b0a30228787a8;
  decode/GQ remains248VGPR/occ2/0spills. T4 job1790118644590954082-2b5a8aaa
  DONErc0 at /opt/models/plow-glm53-qkva-w8a8-b16-fit-ab-20260922:
  ctl1311.016,treat1298.966,ctl21314.116,treat21299.986us. Mean-arm saving13.090us
  /0.997283%, noise10.450us, control drift3.100us, treatment spread1.020us; PASS.
  All32 routed arm/rank chains pass,352outputs/arm;52 changed files restricted
  to canonicalized maps and atomic-dependent outputs. Separate independent QKV
  checks pass all4arms x8ranks x8boundaries bitwise to pinned vLLM.
- Independent repeat with identical builds/workload/samples/warmup,
  job1790118738138819577-7cf7a364 FAILEDrc1 at
  /opt/models/plow-glm53-qkva-w8a8-b16-fit-ab2-20260922: timing gate only.
  ctl1304.246,treat1297.666,ctl21306.216,treat21299.086us. Mean-arm saving6.855us
  /0.525194%, below8.480us noise; drift1.970us, spread1.420us, stable=true,
  gate_pass=false. All32 routed chains and independent block gates still pass.
  Both sessions quiet/final audited. DO NOT RETAIN: apparent gain did not clear
  repeated measurement noise. Reverted only this turn's sizing/test edits;
  baseline remains QKV-A build3662ecb1...,256 QKV workgroups. Frozen candidate
  and both result sets retained, no full-model speed claim. Restored emitter
  tests pass; release compiler rebuilt successfully to match restored source.
  Fresh verified emit at /opt/models/plow-glm53-qkva-w8a8-restored-20260922/assets
  reproduces the baseline packet3662ecb1... BYTE-FOR-BYTE;31ops, existing Lean
  ordering/LDS/rewrite checks pass. Source matches frozen pre-experiment emitter.
  Staged index unchanged. All GPU/build jobs terminal; goal remains OPEN.
- Remaining MLA reference source inspected in pinned Docker (CPU-only files):
  mla_attention.py:885..913,1080..1150,1232..1252,1404..1412. W_K [heads,512,192]
  and W_V [heads,256,512] derive from BF16-dequantized kv_b_proj, not absorbed
  q_b weights. dynamic_per_batched_tensor_quant actually uses ONE scalar across
  the whole local head batch (aminmax without dims), not one scale per head.
  AITER triton/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py
  delegates to _triton_kernels/gemm/batched/ same basename: input activation group128
  quant is fused into BMM, masked K tail supported (192 ->128+64), amaxfloor1e-10,
  accumulator += dot*a_scale per group, then ONE final weight-scale multiply,
  BF16 output [M,heads,N]. SplitK unsupported. This association differs from
  block128 linear GEMM's per-group as*ws product: cannot blindly reuse that core.
  The wrapper's INT8 prose is stale; active WQ dtype supplies FP8 limits/convert.
  With BF16 dequantized input, weight amax, reciprocal quant multiplier and
  scaled/clamped values remain BF16 before the final FP8 cast; only the saved
  inverse scale uses F32. CPU dtype probe confirms this. Preserve these rounding
  steps, not a cleaner all-F32 quantizer. Frozen exact sources at
  /opt/models/plow-glm53-mla-fp8-contract-20260922; kernel assembly/replay still
  required before treating Triton expression association as fully qualified.
  Loaded inventory confirms layer3 local W_K [8,512,192], W_V [8,256,512],
  both FP8 with scalar (shape[]) F32 scales. kv_b_proj [3584,512], stride[768,1],
  scale grid[28,4]. get_and_maybe_dequant_weights at quant_utils.py:498 uses
  scaled_dequantize(weight,scales,group_shape=[128,128],out_dtype=BF16) for this
  non-Marlin/non-DeepGEMM Fp8LinearMethod, not the generic identity-GEMM fallback.
  scaled_dequantize explicitly broadcasts F32 scales, multiplies FP8-promoted-F32
  weights in F32, then casts BF16. quant_utils.py also frozen with the BMM sources.
  Next: pinned q_b/MLA BMM component fixtures and native equivalent preserving
  these dtype boundaries; full8K..70K/C8..64 throughput/quality matrix OPEN.
- 2026-09-22 MLA FP8 BMM component baseline now implemented and qualified against
  pinned vLLM0.29.0 (NOT connected block/full-model parity). Extended existing
  block_fp8_aiter_compare.py with --export-mla/--check-mla, original TP kv_b
  byte/scale extraction, strict binary replay format and hash-bound coverage.
  Scalar hashing now flattens before uint8 view (old nonscalar hashes unchanged).
  Reference uses installed scaled_dequantize and dynamic_per_batched_tensor_quant,
  preserving BF16 weight-preparation rounding and one F32 scalar per local batch.
  Both W_K[8,512,192] and W_V[8,256,512], all8 TP shards, M1/8/16/32/64/128:
  96 seeded BF16 component cases (zero/tiny rows included), real layer3 checkpoint
  weights, all finite and reference repeats BITWISE. Not serving workload inputs.
  Reference /opt/models/plow-glm53-mla-reference-20260922/reference.json;
  queued job1790119495968137798-aff1eb75 DONE rc0. Frozen comparator, job/log and
  Triton cache include actual LLVM/AMDGPU assembly. Assembly confirms native
  16x16x128 MFMA, group dot*a_scale FMA, full reciprocal division sequence, and
  final scalar weight multiply (LLVM source fmul/fadd alone hid backend fusion).
- Added test-only d_mla_bmm_fp8_m16 in op_gemm_common.h and HIP wrapper:
  one wave/16x16 tile, fused BF16->FP8 group128 input quant, masked K192 tail,
  vector FP8 weight loads, head-interleaved BF16 output and preserved association.
  No new opcode/knob/serving dispatch. Existing block_fp8_gfx950_test.c now has
  mla-replay, strict finite+repeat+BITWISE gate (no tolerance relaxation).
  /opt/models/plow-glm53-mla-native-20260922/test_kernels.elf
  sha7731a7952bde386fffdd17bd240131cdaf0cff00fecb82f5bf96577eb477f806;
  host sha ce8c5e400fd46297329a5eb8b6f1d9eddd28726d773320d79882d5c1a4a142c6.
  Native job1790119754129417006-6c5a88f0 DONE rc0: all96 BITWISE, repeat-bitwise,
  max-row-rel-L2=0. Independent CPU comparison.json hashes every fixture/input/
  weight/scalar/reference/output and requires complete rank/projection/rung set:
  passed=true, precision_qualified=false. Frozen source, assembly and notes.
  Native metadata81VGPR,105SGPR,0VGPRspills,2SGPRspills intoVGPR lanes,0scratch.
  This is a correctness baseline, NOT a tuned/performance-qualified kernel.
- Regression job1790119795436910816-fd722b57 DONE rc0,202 PASS checks: prior QKV
  split replays bitwise, plain/GLU/quant captures, B16/B96 routed quant/down and
  all8-rank atomic order bounds.24CPU unit tests pass including MLA tail/signed0,
  TP byte slices, scalar shape/finite checks, stale hashes/missing rung rejection
  and changed native-output failure. git diff --check passes; stagedindex remains
  cce53d1bfa6e92fdc53a65e735538a8c26355a3eb98dbb22f9eb51f351870cbf.
  All GPU jobs/compiles terminal. Serving baseline packet3662ecb1 unchanged.
  Retain megakernel where repeatable block/T4 data supports it; no scheduling
  default changed this turn. Next: q_b FP8 native boundary + MLA integration,
  then shape specialization/SGPR pressure and block-level timing. Full goal OPEN.
- 2026-09-22 next goal turn: previous turn classified PROGRESS (native MLA
  implementation + pinned replay evidence). Q-B source inspection changed the
  next action: loaded layer3 q_b_proj is AiterFp8BlockScaledMMKernel, original
  E4M3[2048,2048], padded stride[2304,1], F32[16,16], UE8M0=false. Unlike QKV-A,
  the pinned tuning table selects CK kernel8 with splitK exponent3 at M1/8/16,
  exponent2 at M32/64, exponent0 at M128. It uses BF16 atomic accumulation into
  a zeroed BF16 output, not a final FP32 sum. First strict-repeat capture
  /opt/models/plow-glm53-qb-reference-20260922 failed correctly at atomic rungs
  (job1790120101811298959-bb32402b rc1); retained, not silently retuned.
  Frozen installed Python/C++/CK source at ...qb-split-contract-20260922 confirms
  KBatch=1<<splitK, contiguous KRead partitions, atomic vs Set dispatch, zero init.
- Extended existing comparator with qb_weights, --export-qb and --check-qb.
  Q-A norm inputs are independent QKV-A reference outputs, not random data.
  Full Q-B reference preserves pinned default dispatch; additional isolated CK
  calls use the SAME named kernel with splitK0 on exact K256/K512/K2048 slices
  solely to establish partial-rounding and atomic-order bounds, not as a serving
  override. Every isolated FP8 A/W/F32 scale slice is checked against full inputs.
  /opt/models/plow-glm53-qb-split-reference-20260922/reference.json:
  job1790120325883835199-983ff665 DONE0,48 rank/rung cases,264 parts; norms/quant/
  scales and isolated parts repeat BITWISE; both full reference outputs within
  exact BF16 addition-order extrema. audit_complete=true,precision_qualified=false.
  Existing native M16 core replayed all264 isolated parts BITWISE in
  ...qb-native-20260922,job1790120370285631849-6de089a9 DONE0.
- Added test-only split-K4/8 specializations to d_gemm_fp8_block128_m16, with
  separate diagnostic partial-store and BF16-atomic output wrappers. Default
  SPLITK1 retains existing logic. Split variants require K divisible128*SPLITK
  and even N; host replay deliberately restricts to actual Q-B N=K=2048/rungs.
  No new knob/opcode/emitter/serving dispatch. Full-stride A/W/scales consumed
  directly, each partition rounds BF16 before native packed BF16 atomic addition.
  ...qb-split-native-20260922/test_kernels.elf
  sha5226ab23ff56ca99ae863a38501d47b514296b16be85a135a35a5c6a4407c47a;
  host sha549046e3c9ddc9090e2bb7ca378e0904dd5da9693889cddd65c276be00bdd2b0.
  All4 variants47VGPR,56/58SGPR,0spills,0scratch. Frozen assembly confirms native
  16x16x128 FP8 MFMA and global_atomic_pk_add_bf16 (no CAS/locks).
  job1790120586355088555-75a0d0e5 DONE0:48 cases, every partial BITWISE (two runs),
  reference + four native atomic outputs per case inside exact order bounds.
  CPU comparison.json independently rechecks all raw hashes/partition operands/
  output bounds: passed=true,precision_qualified=false. Native atomic repeats
  are NOT required bitwise; bounds are from exact partitions, not an L2 tolerance.
- Existing regression jobs1790120586596777042-28766be7 and
  1790120586846295834-3d521537 DONE0:202 previous checks +96 MLA bitwise replays.
 26CPU tests pass incl. Q-B TP bytes/scales/gamma, changed native parts, changed
  atomic result and incorrect partition rejection. git diff --check passes.
- Extended --export-mla optionally feeds W_K from the frozen, bounds-qualified
  Q-B reference snapshot. ...qb-mla-reference-20260922/reference.json,
  job1790120685140047033-50baf0b4 DONE0; W_K input retains reference strided
  [head,M,192] view of [M,head,256] Q-B output; W_V inputs still seeded. All96
  reference cases stable. Native job1790120751196434898-91292c7e DONE0 and CPU
  ...qb-mla-native-20260922/comparison.json pass BITWISE for all96 cases.
  Native replay currently packs this strided input into fixture-contiguous K192:
  this is a component arithmetic gate, NOT proof of no-copy connected data flow.
  Source Q-B atomic order is one frozen reference snapshot, not native-produced
  atomic output. No full-block/serving speed or parity claim.
  Next: original Q-B/derived MLA weights overlay, AOT q_b+MLA dispatch (including
  no-copy Q head stride256 and raw rope extraction), Q-A norm/quant connected
  boundaries, separate FP8 W_V after MLA merge, then block gates and T4 timing.
  Baseline packet3662ecb1 and stagedindexcce53d1... unchanged. All jobs/compiles
  terminal; megakernel choice remains measured per rung. Full goal OPEN.
- 2026-09-22/23 next goal turn: previous turn PROGRESS (Q-B split-K native
  implementation + exact partition/atomic-order evidence). Full prepared base
  already retains original q_b_proj.weight FP8 and weight_scale_inv F32; do NOT
  duplicate/requantize them. It has no HF index JSON (Plow scans shard headers).
  Added --mla-tp to existing glm52_prep_fp8_linear.py: CPU dequant in F32 -> BF16,
  split UK/UV, BF16 global-local-head amax/multiplier/scaled values -> FP8, saved
  reciprocal F32. Only four derived tensors/layer are new; original Q-B binds
  its existing names. Names self_attn.derived.mla_fp8_tp8.{wk,wv}.weight and
  .weight_scale; full weight shapes[64,512,192]/[64,256,512], scale shapes[8,1].
  Each rank's single scalar is over its local head batch; TP-specific prep is
  necessary, not a different precision. Added Column classification for original
  self_attn.q_b_proj.weight* and derived.mla_fp8_tp*; slice_for refuses encoded
  TP != runtime TP even when byte-size/replication could otherwise permit it.
 28 asset::shard tests pass, including 3-D head slices, 2-D scalar slices and
  wrong TP1/4/16 rejection.3 prep tests pass, including TP-dependent scale,
  exact resume (unchanged mtime), corrupted-file regeneration and invalid geometry.
  Existing mmap ResourceWarning is preexisting and unchanged.
- Full78-layer overlay produced CPU-only at
  /opt/models/GLM-5.3-full-aca966e4-plow-mla-fp8-tp8;312 new derived tensors,
  ~1.15GB new payload, base symlinks intact. Existing --qkva overlay is its base.
  Added --check-mla-weights to comparator, using pinned installed GPU
  scaled_dequantize and dynamic_per_batched_tensor_quant plus loaded metadata.
  Every layer/rank/W_K/W_V weight AND scalar BITWISE:1248 pair comparisons.
  Job1790121292604913451-01e9abb5 DONE0; full audit at
  /opt/models/plow-glm53-mla-weight-prep-20260922/reference.json,passed=true,
  precision_qualified=false (weight preparation only, not serving). Source/prep
  log/queue job frozen. The CPU implementation was not assumed equivalent.
- d_mla_bmm_fp8_m16 now accepts input head stride; optional COPY_ROPE template
  reads [M,H,256] Q-B output directly for K192 and copies raw tail64 to [M,H,64]
  only on n0==0 tiles. No separate Q-nope packing kernel. Existing generic calls
  keep strideK and no rope writes. New test-only mla_bmm_fp8_qrope wrapper;
  existing host test gains mla-qb-replay, reads actual full Q-B reference output,
  checks Q-nope input bytes against MLA fixture, verifies projected output and
  copied raw rope exactly on both repeats. CPU --check-mla --qb-reference also
  hash-binds Q-B provenance, no-RoPE slice and rope output; tests reject stale
  provenance and changed rope.26 comparator CPU tests pass.
  /opt/models/plow-glm53-mla-strided-native-20260922/test_kernels.elf
  sha454f9ab7b0b31381fa9f9a5632c8800e3e647695d4f474355eff0072bf6a8b3a;
  host sha2cfd67497fc618faf71fd6be1e88e954c3af610898d34491015283960da83666.
  Generic qrope variant88VGPR,106SGPR,13SGPRspills intoVGPR lanes,0VGPRspills,
  0scratch: correctness baseline, NOT tuned/performance-qualified. Shape
  specialization and measurement are needed before claiming this fusion wins.
- Job1790121492164905901-cbf449dc DONE0:96 MLA outputs BITWISE and48 raw Q-rope
  copies BITWISE; comparison.json CPU checks all hashes/provenance/outputs.
  Regression1790121492419378783-c5fddcb9 DONE0 (202 checks) and Q-B full-stride/
  atomic regression1790121492662548199-ba59f9a3 DONE0 (48 cases, exact parts and
  bounds). All jobs/compiles terminal. No opcode/knob/emitter change yet; serving
  packet3662ecb1 and stagedindexcce53d1... unchanged. git diff --check passes.
  NEXT: actually wire AOT attention packet (new MLA BMM op, Q-B split-K selector
  and zero dependency, direct original q_b bindings, TP-specific WK/WV tensors),
  replace absorbed q projections and BF16 merge+value fold without moving dtype
  boundaries; connected norm/quant/attention gates then block T4/roofline tuning.
  Full8K..70K/C8..64 same-precision serving/throughput goal remains OPEN.
- 2026-09-23 AOT MLA wiring: opt-in PLOW_GLM_MLA_W8A8 (gfx950, exact GLM TP8,
  requires QKV-A W8A8) now declares original q_b FP8/scales and per-rank scalar
  WK/WV prepared tensors. Quant128 -> zero BF16 Q-B -> live-row split-K GEMM ->
  MlaBmmFp8 opcode191 (strided Q-B input + raw rope64 copy). FlashMerge retains
  BF16 latent output before separate FP8 value BMM. Absorbed norm/value and
  token/row-band variants are rejected. Same modular packet dependencies; no
  extra host launch. ZeroF32 now has an AMD arm with coherent activation stores.
  New opcode/selector markers and shape checks fail closed on stale objects;
  native OCP weight detection preserves WK/WV signed-zero bytes. Capture includes
  Q-A/Q-B/MLA activations and weights. Roofline counts both FP8 BMMs with one
  scalar scale per local-head batch, using FP8 rather than BF16 compute ceiling.
  Q-B selector checked against all 234 relevant rows in the pinned image's
  merged gfx950/256CU table + getPaddedM search for every M=1..131072: zero
  mismatches. Besides M<=16 split8 and M17..64 split4, split4 applies at exact
  M72/M88 and M225..240 except exact232 (first exact then ceil16 lookup).
  Emitter tests cover decode1/8/16/32/64 and prefill128/2048, clear+quant producer
  gates, original weight sizes, no absorbed reads. Devgen knob12, runtime knob19
  (+1 pre-existing ignored), AMD loader161, packet155, C/Rust ISA4, manifest53,
  roofline9, comparator26 tests pass. Packet slot-doc guard and test-only clone
  compile failure were fixed; no test waivers. New --block-mla comparator checks
  independent norm/quant/Q-B exact atomic bounds and captured-input-conditioned
  pinned query/value BMM outputs, explicitly NOT merge/full-serving qualification.
  Build /opt/models/plow-glm53-mla-w8a8-b16-aot-20260923 is in progress;
  emitted35 decode ops. Private runtime /opt/models/plow-glm53-mla-w8a8-runtime-20260923/plowrt
  sha321acc0dca603e286be5f3e307bf66f03897b33192d551e1d1560f102140b190.
  Megakernel remains eligible only by repeatable matched-precision block/rung
  measurements above both control drift and treatment spread. No performance
  promotion; previous measured baseline and staged index remain unchanged.
- Initial AOT b95c4d6158119372c96fec1b1f51577eac3168d96ec98aa726c948894f3766c9
  built successfully, decode/GQ248VGPR/occ2/0spills. First block job
  1790123033535741090-245f5ab6 failed before execution: root-owned generated
  MLA shards were mode0600. Repaired exactly78 generated files to0644; prep
  writer now makes new/resumed MLA shards readable, with mode/resume tests (3PASS).
  Retry1790123097678143623-e6e3d190 ran with finite agreeing ranks and residual
  maxrowrelL2.000357918, median1392.345us, but is INVALID: connected audit
  1790123125630609186-ea28ff61 found act.olat still allocated for one row.
  Whole-block tolerance did not catch the out-of-bounds latent merge. INVALID.json
  added to capture; neither correctness nor timing from this build is evidence.
  Emitter now allocates rows*local_heads*latent*BF16 only for the new opt-in
  route, preserving old default allocation; rung test now checks olat bytes too.
  Rebuilding compiler and a fresh packet/object set; need fresh block+audit.
- Replacement /opt/models/plow-glm53-mla-w8a8-b16-aot-v2-20260923 completed:
  packet7fd85a4b110fbe413502e44f16b5c09ae23756cbf4debe9d60999dfb29c2ca87.
  Static packet.json confirms act.olat131072B atB16 and every one of35 ops in
  segment0. Decode/GQ248VGPR/occ2/0spills. DoctorPASS. New block capture
  /opt/models/plow-glm53-mla-w8a8-b16-block-v2-20260923,
  job1790123502634541940-eea190e8 DONE0:30samples5warmup, median1389.796us,
  finite identical final residuals on all8ranks, maxrowrelL2.000358946842.
  Diagnostic only: NOT a matched T4 or a same-precision comparison with old
  absorbed attention; no performance promotion. Old measured baseline retained.
  /opt/models/plow-glm53-mla-w8a8-connected-audit-v2-20260923:
  independent QKV-A8boundaries/rank BITWISE; connected MLA job
  1790123538321703625-eca6aeeb DONE0, all8ranks PASS. Q-A norm, Q-B quant/scales,
  original/prepared weights and rawrope BITWISE; captured Q-B within exact
  BF16 atomic-addition-order bounds; query and value BMM outputs BITWISE vs
  installed pinned vLLM0.29 on captured BF16 inputs, repeats BITWISE. Value
  latent merge itself remains unqualified against pinned attention; full
  precision_qualified=false. Initial invalid capture/audit retained separately.
  Updated roofline /opt/models/plow-glm53-mla-w8a8-b16-roofline-v2-20260923
  uses captured28-expert union, optimistic29.709us aggregate floor; excluded
  activation/launch/fabric/reduction traffic, NOT achievable or measured speed.
  All compiles/GPUjobs terminal. Stagedindexcce53d1... preserved. git diff --check
  PASS. NEXT: qualify attention merge/remaining indexer/prefill precision; tune
  query/value dataflow within this FP8 route with matched repeated block T4.
  Q-B BF16 atomics can change downstream inputs between runs: do not reuse the
  pre-MLA routed certificate or assume one captured routed-input hash is a
  valid certificate for every future arm. Use per-arm connected gates/exact
  atomic bounds or prove the downstream stable boundaries before timing claims.
  Keep megakernel only where measured matched-rung advantage survives noise.
- Attention boundary audit (2026-09-23): extended existing block dump outside
  timing with post-write slot KV, opart/mlpart and original KV norm gamma. Reused
  dump_slot_kv (only actual context rows), no new runtime knobs or GPU scheduling
  changes. Private runtime /opt/models/plow-glm53-attention-capture-runtime-20260923/plowrt
  sha91be3aa6d3338bf1b3ddf7efbdffabb917ac64dd73572b976b7f6f528358e0c2.
  Same valid AOT v2 packet7fd85a4b..., no object rebuild. DoctorPASS. Capture
  /opt/models/plow-glm53-attention-boundary-capture-20260923,
  job1790124323399756428-585787fa DONE0: B16ctx512TP8layer3,30/5 samples,
  finite identical residuals, maxrowrelL2.000353837391. Median1390.996us is
  diagnostic only, not matched T4 or serving performance qualification.
  Existing comparator now has --block-attention. Uses installed pinned sparse
  forward_mqa, query concatenation/head padding, selected-key conversion,
  persistent work metadata + reducer. Only constructor/workspace plumbing is
  replaced with captured-input allocations. Uses inventory BF16 cache page16,
  real sparse split heuristic (4 atctx512), scale.0625 and selected512keys.
  Validates selected-key prefix/padding/uniqueness and exact global mapping;
  all carried KV rows unchanged; includes new row written by Plow.
  Audit /opt/models/plow-glm53-attention-boundary-audit-20260923,
  job1790124356066621586-4ab7934d FAILED1 as intended by strict diagnostic gate:
  audit_complete=true, passed=false, precision_qualified=false. KV RMSNorm is
  BITWISE on all8ranks, repeats stable. Attention finite/repeat-bitwise but
  NOT bitwise vs Plow on all8ranks: maxrowrelL2 .002520168649, maxabs1.5258789e-5.
  No tolerance change or performance promotion. Frozen outputs, comparator,
  backend/helper/wrapper source, job logs and installed loaded persistent ELF.
  ELF b6d4181c3ed19750b22a02dc0d290727272ed53091678c5e75c39f45d9832cfd
  disassembly shows FP32 exp/sum then v_cvt_pk_bf16_f32 at0x3c28..0x3c40.
  Plow scalar FlashGatherDecode keeps probabilities FP32 through PV (source
  op_attention_common.h Ssm/pw); actual packet uses8splits vs pinned4. This is
  a concrete intermediate-precision difference, not yet a complete causal
  attribution of every output mismatch. NEXT: match BF16 probability rounding
  and inspect split/tile/reducer semantics before tuning/promotion. RoPE and
  learned indexer/prefill remain unqualified; this attention audit conditions
  on captured Q/post-write KV, not an independent end-to-end model oracle.
  Release hsa,cuda buildPASS; comparator28 CPUtestsPASS; git diff --checkPASS;
  stagedindexcce53d1... unchanged. All builds/GPUjobs terminal. Megakernel policy
  unchanged: retain only repeated matched-rung advantage above both arm spreads.
- Scalar attention rounding experiment (2026-09-23): new default-OFF paired
  env/object define PLOW_MLA_P_BF16 rounds only BF16-cache scalar MLA PV operands
  to BF16; denominator/softmax stay FP32, FP8-cache arm unchanged. Two PV loops
  covered; shared golden wrappers get the same build define via INC. Exports
  plow_mla_p_bf16_1. Registry env+define and flags reference updated. No automatic
  serving/emitter selection: this is an experimental precision arm, NOT parity
  qualification. Existing decode MFMA32 prototype is not a drop-in replacement:
  it still walks top_k rather than min(top_k,kv_len) on short-context gather, and
  prior batch1 measurements were slower. Do not promote it based on its name.
  /opt/models/plow-glm53-attention-pbf16-aot-20260923 built; exact same packet
  7fd85a4b... as control, decode/static+GQ248VGPR/occ2/spill0. Header/build script
  frozen beside objects; decode-gq.asm retained; BF16-P marker verified in ELF.
  Capture /opt/models/plow-glm53-attention-pbf16-capture-20260923,
  job1790124823289252529-7ee40455 DONE0: all8ranks finite/identical residuals,
  maxrowrelL2.000379405753. 30/5 diagnostic median1202.328us vs old1390.996us
  is NOT a performance claim: no matched repeated T4, precision path differs,
  Q-B atomics can also change connected inputs. Old measured baseline retained.
  /opt/models/plow-glm53-attention-pbf16-audit-20260923,
  job1790124866901668833-5c334445 FAILED1 diagnostic gate: attention still differs
  on all8ranks, worstrowrelL2.002614896835; KV norm remains bitwise/repeat-stable.
  BF16 probability conversion alone is not sufficient.
  Extended existing comparator with explicitly non-qualifying FP64 QK/exp/accum
  rounding model and persistent metadata dumps. CPU tests cover denominator
  staying unrounded, online rescale, partition/shape rejection. Comparator29PASS,
  devgenknob12PASS, runtimeknob19PASS+1preexistingignored, bash-n/diff-checkPASS.
  Initial model audit /opt/models/plow-glm53-attention-rounding-model-audit-20260923,
  job1790124755979955499-a4baa952: uniform4/8 splits insufficient. Important
  correction to prior shorthand: pinned max_split_per_batch=4 is a CAP, NOT
  actual4equal splits. Captured work_indptr ends48; MlaWorkInfo8-int rows and
  reduce_indptr prove THREE ranges per query [0,192),[192,384),[384,512).
  Installed metadata v1_2_device.cuh/v1_comm.cuh/mla.h frozen for provenance.
  Pinned assembly advances s83 by32 (s67=1 at0x2968, s84=s67*32 at0x2a50),
  with BF16 probability conversion every32-key online update. Its max is raw
  dot-domain, scale/subtraction fused before exp2; Plow scales before max.
  /opt/models/plow-glm53-attention-worktiles-audit-20260923,
  job1790125016966448486-17b27920: on same native BF16-P capture, diagnostic model
  using ACTUAL captured work ranges + online32 tiles reduces model-vs-pinned
  mismatches from20678..21283 (whole-partition max) to7..109 of65536 perrank,
  worstrowrelL2.000450890696. This is evidence for numerical operation placement,
  NOT a native kernel parity result. Actual native attention still fails strict
  gate; no tolerance loosening/promotion. Remaining small differences may involve
  MFMA accumulation, FP32 scale/exp/rescale and reducer order; not yet isolated.
  NEXT: implement/qualify native32-key online BF16-P path and live work partition
  contract (or adapt pinned persistent ASM into a Plow segment), then connect
  projection/routed gates before matched-rung perf T4. Do NOT hardcode ctx512's
  three ranges into serving; work distribution varies with batch/length/CU count.
  All GPUjobs/builds terminal; stagedindexcce53d1... preserved.
- Pinned persistent attention HSA replay (2026-09-23): existing Python comparator
  now exports stage1 FP32 partials/LSE only after direct installed ASM+reduce
  matches the unmodified serving reference bitwise and repeats bitwise. Router
  must independently select persistent mode; no routing override. Reducer uses
  installed get_mla_decode_fwd_max_splits, not None (initial export failed there).
  /opt/models/plow-glm53-attention-ps-reference-v2-20260923,
  job1790125786014100670-3c9a6645: all8 exports finite/bitwise/repeat-stable,
  B16/ctx512/padded16heads/48work-items. Ordinary native-attention strict gate
  remains FAILED1; persistent_exports_complete=true is a separate scoped gate.
  Reference JSON sha10c7d254d5662310fc7dd22594dc627a302420843d4303ad4de7a33c4f16f2aa.
  Existing block_fp8_gfx950_test.c attention-ps-replay loads pinned persistent
  ELF via Plow C HSA backend/AQL queue (no HIP launch or graph), validates live
  work ranges, uses captured Q/KV/indices/metadata, and compares FP32 partials
  and LSE across two launches against the export. Installed wrapper supplies
  368 argument bytes; metadata reserves384, with trailing padding. Descriptor
  size is zero (same vendor-toolchain issue as existing runtime MLA adapters).
  Readelf confirms .kd atfile0x1700, LDS163840/private0/WG256. Harness checks
  exact descriptor and68768B image, normalizes only in-memory size at0x1708 to384,
  supplies zero-padded384B arguments, then verifies loaded resource contract.
  Runner enforces full ELF sha b6d4181c3ed19750b22a02dc0d290727272ed53091678c5e75c39f45d9832cfd.
  Initial unnormalized replay failed safely before dispatch (kernarg0).
  /opt/models/plow-glm53-attention-ps-native-v2-20260923,
  job1790125950254945841-a0b5fcb1 DONE0: all8ranks finite, ZERO partial/LSE bit
  mismatches, repeat-bitwise. Hostsha e8093c428c8939f066fd6ff8c8bb173aef5d34a0766f3aaa41401e126f52e7d0.
  Sources/ELF/outputs/queue logs frozen. No serving integration or performance
  qualification: isolated stage1 replay only, not a full attention replacement.
  NEXT: qualify persistent reducer via HSA and runtime segment data-layout /
  live metadata contract, including short/no-split and nonpersistent B32+ routes.
  Do not hardcode this fixture's48work-items or promote its kernel to every rung.
  Comparator30CPUtestsPASS (new test rejects route mismatch, unstable/nonfinite
  partials and changed serving output), hostbuildPASS, BF16boundsCPUselftestPASS,
  doctor/bash-n/gitdiff-checkPASS. No new knobs. Stagedindexcce53d1... unchanged.
  Megakernel remains per-rung: retain only numerically qualified, matched T4
  advantage larger than BOTH arm spreads; no performance promotion this turn.
- Persistent reducer HSA qualification (2026-09-23): extracted gfx950 device ELF
  from immutable vLLM0.29 module_mla_reduce.so .hip_fatbin with llvm-objcopy and
  clang-offload-bundler; no recompilation or precision substitutions.
  /opt/models/plow-glm53-attention-reduce-pinned-20260923 freezes installed source,
  .so (sha d3bf630bea13426809cd9f0747edb5c9298509920a04a223b24e2358f8cd9791),
  fatbin and reduce_gfx950.elf (sha401e7dd9c9714650361a87bba36b216e5b491d90fa10e8fc9cda72712e62f383).
  Installed source selects online merge for2/3splits and a different massive
  path at4+; don't replace with a generic max-then-sum reduction without a gate.
  Qualified symbol kn_mla_reduce_v1<Traits<512,16,1>,float,bf16>: args84B
  (params80B+LDSconfig4B), staticLDS0/private0, WG128, grid(16,1,M), dynamicLDS2048.
  Params=max_splits256, lds_scale_cap256, output_lse=false, use_final_map=true,
  BF16outputstrides8192/512. Installed ELF uses direct workgroup IDs, no hidden
  args; rawHSA kernarg84B verified against metadata and loaded descriptor.
  Python persistent exporter adds .reduce.bin with captured reduction maps and
  padded16head BF16 serving output, only after exact/repeat/finite reference gate.
  Existing C harness attention-reduce-replay consumes ACTUAL HSA stage1 F32 files,
  validates map coverage/ranges and finite inputs, poisons BF16 output before EACH
  launch, and checks bitwise reference/repeat. This is a staged-file component
  chain, NOT a same-queue resident two-kernel pipeline or serving measurement.
  /opt/models/plow-glm53-attention-reduce-reference-20260923,
  job1790126263254675084-5d7a8a9e:8exports complete, JSONsha
  4b8adc93860d6ba3a04c2be2449c17dd7c4ec20892a288825c2a588f184096fe.
  Native scalar-attention audit stillFAILED1; no strict-gate loosening.
  /opt/models/plow-glm53-attention-reduce-native-20260923,
  job1790126301878366173-a28a06cd DONE0; final bounds-hardened rebuild/replay
  /opt/models/plow-glm53-attention-reduce-native-v2-20260923,
  job1790126443172945587-8c2006df DONE0. All8ranks stage1 partial/LSE and reducer
  finalBF16 ZERO bit mismatches, finite, repeat-bitwise atB16/ctx512/3splits.
  Hostsha f266b84f1b2f1244ac422ccb84419d37912bc3a96febbe91f8c7be023969bfb1.
  Comparator30CPUtestsPASS including reduce-header/layout export, hostbuild and
  BF16boundsCPUselftestPASS, doctor/bash-n/diff-checkPASS. No new knobs; no
  production routing changes. Stagedindexcce53d1... unchanged. All jobs terminal.
  Runtime review: existing DecodeRoute matches only FP8-KV FlashMlaDecodeFp8,
  top2048/full rows, <=20; it converts FP8cache and relayouts partials for native
  FlashMerge. It cannot substitute for pinned BF16cache attention+BF16 merge.
  NEXT: live metadata + BF16 Q/KV layout adapter / dedicated segment route;
  preserve original precision and paddedhead16->8 mapping. Qualify 4+split path,
  short/no-split, ragged live lengths, and nonpersistent B32+ installed routing.
  Keep measured megakernel controls; no perf promotion until matched-rung T4.
- BF16 layout adapter and live metadata (2026-09-23): added plow_mla_bf16_pack /
  plow_mla_bf16_unpad to existing runtime/amd/mla_sparse_adapter.hip. Inputs are
  original BF16 QA/QR/CK/KR, live kvlen and optional2048-wide selected indices;
  packs padded16head queries (repeat each of8heads twice), gathers only selected
  BF16KV into576wide rows, builds live CSR onGPU via wave64 prefix scan, emits
  identity indices. Output unpad selects even heads bitwise. No dtype conversion;
  invalid lengths/indices trap instead of silently clamping to a different key.
  Build script --bf16-gfx950 validates persistentELF hash and emits gfx950 adapter;
  default gfx942/single-pass contract retained. New kernels not runtime-routed yet.
  /opt/models/plow-glm53-bf16-adapter-20260923 job1790126795763109498-376105d6 DONE0,
  final geometry-guard rebuild /opt/models/plow-glm53-bf16-adapter-v2-20260923
  job1790127668103413065-d820fdee DONE0:32exact-byte cases, M1/8/16/31/32/64,
  ctx16/512/2048/8192/71680, dense identity/permuted sparse/ragged/zero-length
  rows, poisoned unused tails/256Bguards unchanged. These are layout tests, NOT
  end-to-end attention or empty-request serving qualification. gfx950ELFsha
  ed1426279c1b8228334315570830fac9bd57d4c43adc3066466afcda0649a35b.
  Pack18VGPR/unpad12VGPR/private0/spill0. gfx942 default build regressionPASS.
  Pinned module_mla_metadata.so extracted to
  /opt/models/plow-glm53-attention-metadata-pinned-20260923/metadata_gfx950.elf,
  sha3ade36a825eb249bc7dd8168e2338119173c595f1a1a427622733b73d9bd4081.
  Important: sparse backend supplies already-selected CSR and leaves topk=-1
  in get_mla_metadata_v1, so actual planner is parallel non-sparse template
  Traits<128,false,1,true,false>, not the sparse serial template. Params136B,
  COv5kernarg392B, WG512, grid1, dynamicLDS163840. Params.num_splits derives from
  selected length andbatch with installed128-key power-of-two cap, <=256.
  New existing-harness attention-metadata-replay generates metadata from CSR,
  compares fullworkptr/reduceptr, validwork7fields (padding ignored), valid
  final/partialmaps, and GPU pointers. Poisoned outputs before each of2runs.
  /opt/models/plow-glm53-attention-metadata-native-20260923,
  job1790127615064431947-1c18e3ef DONE0: bothruns ZERO mismatches B16ctx512works48.
  Hostsha8e8e6939127ccf0d6f401a6d207f1f3bb60641fbfd3153de6a7771e259fd3553.
  All builds/jobs terminal, doctor/diff-check/bash-n/BF16boundsselftestPASS.
  NEXT: same-queue resident pack->metadata->stage1->reduce->unpad chain; broader
  live metadata/short/no-split and4+split qualification, then dedicated segment
  route. Existing C loader replaces its sole executable handle on repeated loads;
  don't leak modules when constructing the chain (Rust backend already owns them).
- Requested pull (2026-09-23): fetched4commits5a6d7a3e..96a0fc56 on current branch.
  ff-only/no-autostash pull safelyABORTED due overlapping local mux.rs,
  devgen/knob_spec.rs, flags-reference.md. Asked permission to recoverably stash
  only conflicts, fast-forward, reapply/reconcile preserving index. No reply yet;
  HEAD5a6d7a3e/stagedindexcce53d1... unchanged. Do not infer approval from automatic
  goal continuations or filesystem permission changes. Independent attention
  work continues. Upstream decode-lookahead1 is candidate for AMD; prefill overlap
  explicitlyFAULTS upstream at retirement/level2 and remainsOFF. Shared policy
  thresholds are Gemma/H100-derived, not GLM/AMD-certified. AMD already has split
  submit/complete and device token feedback; requires double-buffered readback,
  staging lifetime and retirement/frontier checks before lookahead port promotion.
- Resident attention chain (2026-09-23): existing block_fp8_gfx950_test.c adds
  attention-pipeline-replay. Five ordered dispatches onONE Plow HSA/AQLqueue:
  BF16pack -> pinnedparallelmetadata -> pinnedpersistentstage1 -> pinnedreduce
  -> even-headunpad. No host wait/readback between stages, ONE wait after unpad.
  Starts from separate QA/QR/CK/KR plus localindices/kvlen, reconstructed without
  dtype conversion from frozen reference fixtures. GPU regenerates packedKV,
  identityindices, liveCSR, metadata. All intermediates poisoned before each run,
  so inherited captured metadata/outputs cannot hide a missing producer.
  Stage1 F32partials/LSE still compared to original frozen AITER; final unpadded
  BF16output checked against original serving reference. This tests connected
  device data flow, not full model, independent learned selection or serving speed.
  /opt/models/plow-glm53-attention-pipeline-20260923,
  job1790127962919777972-1a8175b4 DONE0: all8capturedranks B16ctx512 ZERO stage1
  andfinaloutputbit mismatches, finite, repeat-bitwise. Hostsha
  118a3093b63ca0f245d79c512f82ccb7ed1f9a9279b0bf742034fd2a5a379e52.
  Runnerhashes all4codeobjects and8input/reducefixtures. Sources/logs/outputs frozen.
  C HSA backend now retains all loaded executable modules until shutdown; kernel
  lookup still targets latest module, cached handles remain valid. Loader handles
  allocation/load/freeze failures without replacing last successful executable.
  No dispatch-path change or extra hot-path allocations. Needed to avoid leaking
  earlier modules in the four-object chain; Rust runtime already has module owners.
  Build/doctor/diff-check/bash-n/BF16boundsCPUselftestPASS, stagedindex unchanged.
  Existing regression.sh + atomic.sh rerun with new Cbackend/host and unchanged
  previouslyqualified test_kernels.elf in
  /opt/models/plow-glm53-attention-pipeline-regression-20260923,
  job1790128018046593465-db1b14aa DONE0: QKV-A replay, m16A8W8, GLU, quant128,
  routed-down and exact BF16atomic-order gates PASS; bytecomparisons to previous
  qualified captures PASS. Atomic repeats can differ within exact permitted
  accumulation-order bounds (not falsely reported bitwise). All jobs terminal.
  NEXT: broader live/short/no-split/4+split gates, then dedicated BF16 runtime
  segment route and matched end-to-end block T4; no performance promotion.
- Resident attention sweep (2026-09-23): existing comparator now has
  --export-attention-sweep, pinned vLLM0.29/source/TP8 geometry gates. Synthetic
  seeded BF16 Q/KV with unique permuted per-request selected indices. Calls the
  installed serving forward_mqa twice before exporting persistent stage/reducer
  fixtures; no replacement arithmetic or forced persistent route. Explicitly NOT
  full-model precision or performance qualification.
  Reference /opt/models/plow-glm53-attention-sweep-reference-20260923,
  job1790128188657691140-bbc2a9a6 DONE0, reportsha
  881fc0e3a023da6a54abe1eed5f1ffcaf9179cbac017b0c87136ca2b658ee5c0.
  Native /opt/models/plow-glm53-attention-sweep-native-20260923,
  job1790128501628702167-0927a1f0 DONE0, unchanged qualified pipeline host118a3093.
  All8 cases PASS: M1/M31 ctx512 (3 splits/row), M8/M16 each ctx2048/8192/71680
  (15 splits/row). Five same-queue launches with one final wait, poisoned outputs;
  every F32 stage partial/LSE and final BF16 element exact, finite, repeat-bitwise.
  This qualifies the 4+split reducer path and long-context gather addressing,
  not end-to-end serving or 800/3000tok/s. Runners hash all fixtures/codeobjects;
  queue records/logs frozen alongside outputs. Existing30CPU tests, doctor,
  bash-n and diff-check PASS; stagedindex cce53d1... unchanged. No jobs live.
  NEXT: no-split/short/ragged qualification, then dedicated BF16 runtime segment
  route. Default installed persistent route excludes M>=32; do not force it to
  claim coverage at C32/C64. Those require the matching nonpersistent route.
- Short/ragged persistent chain (2026-09-23): expanded existing sweep to16cases;
  added v2stagefixture magic0x41505332 with per-request KV lengths, v1 remains
  readable. Work count and partial count are distinct: direct-output work has
  partial_qo_loc=-1; split partial slots remain a compact prefix. Python and C
  poison all partial/output buffers; finite gates cover live partials and every
  final BF16 value, and untouched partial slots must retain exact poison bits.
  C reconstructs ragged CSR and per-request local indices, checks causal bounds,
  work coverage and direct-vs-split consistency, and regenerates metadata onGPU.
  New CPU regression covers ragged/direct export, invalid lengths/CSR and a
  missing direct-output write (must fail before fixture creation).
  Reference /opt/models/plow-glm53-attention-ragged-reference-20260923,
  job1790128841738907967-dbbcb11d DONE0, reportsha
  b12dfaf100144103d0b56a6d6a84b5c6288a7d8afd76fedc1ca4757d6b258c96.
  Native /opt/models/plow-glm53-attention-ragged-native-20260923,
  job1790128878992373773-ab14f3e4 DONE0, hostsha
  0c2ca5fbb541143d3b55be672d17fc1c84d1f48cedc8c4fe91713ef2956a5903.
  All16cases bitwisePASS, finite/repeat-bitwise: previous8cases plus M1ctx16,
  M8ctx16, M16ctx128, M31ctx128, M8ctx256, and ragged M8ctx2048,
  M16ctx8192/M16ctx71680 (per-request lengths include1,15/16,127,128,129,
  512,2048 and context ceiling). Direct-output rows coexist with 2/3/4+split rows.
  IMPORTANT: max_split_per_batch16 yielded actual43splits for one ragged request;
  workspace/reducer cap must follow the pinned global work capacity, NOT the
  per-request split-cap hint. Runtime metadata cap must use max LIVE selected
  length, not allocated context ceiling. Current fixtures have a ceiling-length
  row, so the harness uses their actual max length correctly.
  31CPU tests, coherentNix Cbuild, BF16bounds selftest, doctor, bash-n, diffcheck
  PASS. Sources/codeobject hashes/fixtures/joblogs frozen. Index cce53d1 unchanged.
  No running jobs; no runtime integration/performance claim yet. NEXT: dedicated
  BF16 runtime segment route for qualified M<32 chain, then connected block gates
  and matched T4; M32+ matching route and end-to-end model gates remain required.
- Rust runtime BF16 PS integration (2026-09-23), defaultOFF:
  PLOW_GLM_MLA_BF16_PS emit knob registered/documented, requires MLA_W8A8 and
  non-FP8 KV (unset is BF16). Cross-knob constraint+ASSERT_SITES+generated mirror
  updated; positive unset/false KV and negative FP8/missing projection tests added.
  Builder isolates adjacent FlashMlaDecode/FlashGatherDecode + FlashMerge pair
  in class27, strips its internal/external counter obligations, denies uniseg.
  Existing opcodes/math fallback retained; no new device opcode. Emit only M<32,
  no DCP/fused RoPE, sparse top2048 or short dense. Defaults unchanged.
  IMPORTANT parity correction under this knob: legacy c.dsa used dense attention
  through64K even with DSA=1. Opt-in now arms DSA above model.index_topk2048;
  M32+ also uses that selection boundary but still interpreter attention. Prefill
  route is unchanged. Learned selection/RoPE still need separate qualification.
  New exec/amd_mla_bf16.rs validates pure pair ownership in BOTH streams,
  shape/capacity/nonaliasing, exact scale and live nonempty KV lengths. Hash-pinned
  four objects; stage descriptor normalized only after exact hash. Scratch sized
  at256+max_rows-1, not max_split_per_batch. No allocation/readback/sync within
  enqueue. Same5launches integrate AmdEngine load/route/arm/accounting; HsaBackend
  launch_3d_lds adds dynamic LDS for reducer, existing launch_3d delegates at0.
  Uses stack args + tensor pointers. Empty/inactive rows currently rejected;
  do not enable general serving until that contract is qualified.
  Ignored Rust GPU replay (existing PLOW_TEST_AITER_DIR fixture knob) loads all16
  synthetic short/ragged/long reference fixtures, hashes them, poisons scratch,
  runs twice via begin_dispatch_chain(5)/commit, ONE doorbell and final wait,
  checks exact partials/LSE and final BF16. ALL16PASS. This exercises the actual
  Rust kernel loader/queue/ABI, not AmdEngine whole-block execution yet. GPU tests
  use sparse selections, including short ones; short DENSE identity runtime arm
  still needs captured-rank replay or whole-block gate.
  /opt/models/plow-glm53-attention-rust-20260923,
  job1790129620338526541-5f20052e DONE0. Frozen test binarysha
  8051165c8d34a8e47aad31088fdfac35400de3f2778c7b7d7a9ba60f0ca8449a;
  module source subsequently rustfmt-only; frozen original source beside binary.
  CPU:3packet segmentation,3devgen MLA,13devgen knob,19plowrt knob (1fixtureignore),
  161AMD runtime +1new route test PASS. Initial compiler/constraint/test errors
  fixed. Generated mirror regenerated again after allowing unset BF16 cache;
  final logs knob-final.log/devgen-knob-final.log PASS. Index cce53d1 unchanged.
  RELEASE BUILD LIVE session93974: builds plowrt(cuda,hsa), copies it then builds
  plowc/copies to /opt/models/plow-glm53-bf16-ps-runtime-20260923. Poll handle;
  do not restart on timeout. build.log there. Registry generation changed during
  build; rerun incremental release build after terminal and re-freeze BOTH binaries
  before use, ensuring no stale generated constraint. All GPU jobs terminal.
  NEXT: dense captured-rank check + emit private single-block packet with new knob,
  validate pure native pair, build packet-matched AOT objects and install4pinned
  files (metadata name mla_metadata_gfx950.elf, reducer mla_reduce_gfx950.elf),
  run TP8 connected block precision gate; only then matched T4. Current helper
  scripts/build_mla_sparse_aiter.sh --bf16-gfx950 copies stage+adapter, NOT the
  metadata/reducer yet. No throughput/full-model/serving promotion.
