# GLM-5.3 / vLLM 0.29 / MI350X campaign

User goal remains full-model apples-to-apples vLLM Docker 0.29 vs plow,
8K–70K context, C8–C64, 3000 output tok/s and wins on all metrics.
Latest direction: use campaign harness and single-block modular packets while downloading.

## Authoritative state, 2026-09-22

- Host: eight MI350X, gfx950, 256 CU each (amd-smi static).
- Docker image locally verified: vllm/vllm-openai-rocm:v0.29.0,
  digest sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1;
  installed vLLM 0.29.0+rocm723.
- Full HF revision aca966e4e02791568aa6a4ced368624b3d897f42 downloading via hf CLI
  PID 1276071 (exec session 79710), /opt/models/GLM-5.3-full-aca966e4.
  Recheck process and byte growth before deciding whether to resume/restart.
- /opt/models/GLM-5.3 has only nine of 141 shards. Prepared
  /opt/models/GLM-5.3-plow-lite contains layers 0–3 (1627 tensors, 15 GB).
  Its running server PID 1270838 on port 8123 is not full-model evidence.
  User approved stopping that exact server; SIGTERM sent, PID exited on Sep 22 15:15 UTC.
- Existing modular layer-3 packet /tmp/glm53_tp8/block3_modular.pkt is TP8,
  B1 only. No numerical verification/serving qualification may be inferred from its filename.
  Existing build manifests skipped Lean and knob verification.

## Current rung / gate

- First gate: layer 3, B1, ctx512, real weights, seeded CPU HF block reference.
- Oracle now runs in the exact 0.29 image, no GPU devices. Its modern Transformers
  rotary class reads head_dim; use a separate rotary config with the checkpoint's
  qk_rope_head_dim rather than the non-rotary head_dim.
- Fixture /opt/plow-campaign/glm53-blocks/layer3-ctx512.bin (10.15 GB), inputs
  /opt/plow-campaign/glm53-blocks/layer3-ctx512-inputs. FP32/BF16 router top8 matches.
  GPU residual gate pending, rel-L2 <= 0.03 plus finite outputs and TP rank identity.
- amd-block --input-dir now runs block activations without model-only sampled IDs.
  Timings explicitly host dispatch/drain (TP includes preparation/audit); not T2 device timings.
- campaign block-roofline uses packet TP geometry, reports excluded ops and an optimistic
  lower bound. At B1/8K shared-index layer3: 131,081,216 bytes/rank => 21.142 us at
  configured 6200 GB/s. QKV projections dominate modeled traffic. No measured efficiency yet.
- campaign block-bench freezes artifacts and queues via scripts/bench/gpuq.py;
  worker needs render group or GPU audit incorrectly reports idle.
- Control queued: 1790089622366707210-7420f9dd, runner PID 1288976,
  /opt/models/plow-campaign-block3-control-20260922. Doctor passed with a missing
  block build.json warning; this old packet is diagnostic only, not publishable evidence.
  Binary, objects, inputs and packet are frozen and hashed. Queue ran after approved server stop.
  First attempt failed before GPU execution: harness incorrectly rejected an unused act.logits
  declaration. Check changed to reject actual logits references, not declarations. Rebuild underway.
  Retry 1790090292638438960-9c6d41d4 loaded all ranks and completed its first dispatch.
  Residual/xmid/xn2 gates passed but final FFN act.attn failed rel-L2=0.24806 vs 0.03.
  Timing correctly withheld. Added failure-side intermediate dumps for localization.
  Diagnostic job 1790090466936384516-c42768f5 reproduced identical error.
  All 256 act.rlogit values were zero; selected gates were all 0.3125. HF on the captured
  xn2 still selects the original correct eight experts, so upstream rounding is not the cause.
  runtime/amd/interp.hip's sole GemmF32 arm was inside PLOW_BUCKET_PREFILL, despite the block
  decode containing GemmF32. Moved this arm to common dispatch; decode uses existing scalar
  FP32-output GEMV, retaining MFMA only in prefill. Regression test passes.
  Fresh general MM1 objects building in /opt/models/plow-block3-gfx950-mm1-20260922;
  unused MLA prefill pair copied from /tmp/build_b32 because loader selects it even for B1-only.
  Build log /tmp/plow-block3-mm1-build.log. Not a qualified paired object set yet.
  Build passed register gates (decode/GQ 256 VGPR, 0 AGPR, occupancy 2, spill 2).
  Routerfix job 1790090776668016632-16dfb59b and repeat 1790090820467769975-b841e53b PASSED:
  every stage/rank and residual identity; FFN rel-L2 now 0.0046377, residual 0.00011636.
  Host medians 758.912 / 759.983 us (30 samples each, warmup5).
  Artifacts /opt/models/plow-campaign-block3-routerfix{,-repeat}-20260922.
  Separate trace job 1790090820865281963-7f1802cf PASSED; rank0 device envelope 392 us at
  100MHz; /opt/models/plow-campaign-block3-routerfix-trace-20260922/trace.bin.rk[0-7].
  Largest overlapping packet-span sums are expert down / expert GLU; these are NOT
  serialized time shares. Packet floor at ctx512 = 20.8567 us, configured 6200GB/s,
  excludes launch/fabric/etc. Do not divide host latency into a claimed HBM efficiency.
  T4 static scheduler experiment (100 samples, warmup10):
  ctl 757.702, treat 779.071, ctl2 753.632, treat2 780.962 us.
  Control drift 4.070us, treatment spread 1.891us; static +24.3495us (+3.22%), REJECTED.
  Outputs /opt/models/plow-block3-sched-t4-20260922-{ctl,treat,ctl2,treat2}.
  Keep global-queue default. All four passed numerics. Packet/runtime/input hashes held fixed.
  Lean verifier build completed successfully. Fresh paired blocks built through campaign:
  /opt/models/plow-block3-paired-c{1,2}-20260922, GLM_MOE_CORESIDENT=1 vs2.
  Both Lean ordering/rewrite/LDS and K verified; pairing hash 0x81161507f35b3d09.
  New exact-inventory decode objects: 248 VGPR, 0 AGPR, occupancy2, zero spills.
  Co-residency T4 (100 samples, warmup10) all numerics/TP identity passed:
  ctl715.104 / treat705.334 / ctl2 714.314 / treat2 707.334 us.
  Drift0.790us, treatment spread2.000us (<3x drift), saving8.375us =1.17%.
  Qualifies for this B1 block only, not C8-C64 serving. No runtime default promoted.
  Outputs /opt/models/plow-block3-cores-t4-20260922-{ctl,treat,ctl2,treat2}.
  Block emitter now honors existing decode ladders, with batched KV geometry.
  Serial MLA suite 164 passed; targeted block tests 9 passed. Initial parallel test
  attempt hit shared environment races; the new test now passes rungs explicitly.
  Replay supports fixed capture batch, strided KV upload, and per-row stage/residual
  checks (max row rel-L2, not an average). Two oracle rejection tests passed.
  B8 CPU capture: /opt/plow-campaign/glm53-blocks/layer3-b8-ctx512-inputs;
  eight distinct seeded sequences, varied expert selections. Row0 input and residual
  are byte-identical to the B1 fixture. CPU-only vLLM0.29 Docker oracle completed.
  Paired B8 build /opt/models/plow-block3-b8-paired-20260922; Lean and K passed;
  MM4/walk1, same maxctx131072 DSA geometry. Build complete: decode/GQ 256 VGPR,
  zero AGPR, occupancy2, six spills. HSA runner release build passed.
  Batched grouped-MoE dispatch was not armed in decode objects. It now derives
  PLOW_MOE_PREFILL from the paired decode inventory (before capabilities/LDS).
  Inventory define registered; regression passed. New recipe fp8-block3-b8.toml.
  B8 ctx512 packet floor 21.5227us / 133440512 bytes per rank, with optimistic
  maximum expert reuse; launch/fabric/activation overhead still excluded.
  Roofline parser now obtains grouped T/k from preceding MoeAlignPf (not GEMM ABI).
  Campaign tests 7 passed; knob suites 12 / 19 passed (one runtime test ignored).
  B8 GPU gate PASSED, /opt/models/plow-block3-b8-gate-20260922:
  all rows/stages/ranks; worst residual row rel-L2 0.000337355; rank identity.
  Initial 30-sample median1349.006us. Separate trace /opt/models/plow-block3-b8-trace-20260922:
  rank0 envelope1.061ms; FP32 router packet tail224us, overlapping span285us.
  d_gemv_bf16_f32 previously distributed N only, looping M serially: N256 uses
  32 of256 CUs even at B8. Changed to distribute independent flattened M*N outputs;
  each dot product's reduction order unchanged. No new tuning flag/default.
  Treatment /opt/models/plow-block3-b8-routerflat-paired-20260922 is fresh verified
  campaign build, identical packet/config to control, same six decode spills.
  First T4 (100/warm10):1346.356/1205.288/1346.226/1199.668us.
  INCONCLUSIVE under drift rule: control drift0.130us, treatment spread5.620us.
  Retest T4b (300/warm30):1343.107/1196.378/1345.256/1193.498us.
  Drift2.149us, treatment spread2.880us (<3x drift); average medians
  1344.1815 vs1194.938us =>149.2435us /11.10% reduction. QUALIFIES for B8/ctx512
  single-block only. All captured intermediate/output files byte-identical across
  four arms. Runtime, packet, input hashes fixed. All numerical/rank gates passed.
  Outputs /opt/models/plow-block3-b8-routerflat-t4{,b}-20260922-{ctl,treat,ctl2,treat2}.
  Separate treatment trace /opt/models/plow-block3-b8-routerflat-trace-20260922:
  rank0 envelope907us, router tail61us/span121us. Not serialized time attribution.
  Tests after treatment: devgen knobs12, runtime knobs19 (one ignored), campaign7,
  queue2, oracle2, diff-check passed. Full-model download live at503GiB ~16:05 UTC.
  Wider batches, sparse long-context and full serving remain unqualified.
  Follow-up: B8 shared-index block PASSED at actual ctx8192 and71680, using2048
  evenly spaced supplied logical positions (not learned-indexer validation).
  Oracle --block-context scatters selected KV into logical slots and rotates HF
  keys/queries at the same positions. Every row/stage/rank passed; max residual
  rel-L2 .0002201255 (8K), .0002183958 (70K). Only3 timing samples, no perf claim.
  /opt/models/plow-block3-b8-ctx{8192,71680}-gate-20260922.
  B64 verified paired block and CPU capture completed; GPU gate PASSED all64 rows,
  all stages/all8ranks, max residual rel-L2 .000428408, 30-sample median3047.510us.
  /opt/models/plow-block3-b64-{routerflat-paired,gate}-20260922; capture
  /opt/plow-campaign/glm53-blocks/layer3-b64-ctx512-inputs. B8 recipe reused with
  recorded build override PLOW_DECODE_BATCH_LADDER=64; actual batch in measurement64.
  Full78-layer verified packet/object set built: /opt/models/plow-glm53-full-initial-20260922,
  recipe glm53.mi350x.fp8-full.toml, B1/8/16 decode, full128/512/2048 prefill, maxctx73728,
  DSA on, BF16 KV, sharded head. Emit used /opt/models/GLM-5.3 CONFIG via recorded --hf-dir
  override; checkpoint points to future full-aca966e4-plow. No full weights loaded yet.
  Full C64/71680 memory geometry (per rank, before weights/scratch): BF16 latent+rope
  383.906GiB plus index22.969GiB. FP8 latent+rope+scale214.614GiB plus index22.969GiB.
  Existing DCP2/8 reduces the latter to107.307/26.827GiB, index stays replicated.
  Need DCP qualification or explicitly queued admission; replicated B64 BF16 cannot fit.
  Added campaign serve-bench: freezes artifacts, same pinned0.29 Docker client for
  plow/vLLM, queue only, missing-shard refusal, coherence smoke and complete-cell
  metric/token checks. Not run on full model yet. PB teardown now uses foreground
  timeout and exact PID (no process-group kills); graceful/TERM-ignoring tests passed.
  Client wrapper CLI tested in pinned image. Full download still live595GiB ~16:18 UTC.
- Verified: release HSA build, one Rust oracle rejection test, four packet roofline
  tests, two FIFO/foreign-process queue tests, Python compilation, shell syntax,
  git diff --check. CPU oracle also exports intermediate xmid/xn2/final FFN references;
  all intermediate gates must pass before the first timed block launch.
  After router fix: decode-dispatch regression and inventory test pass;
  devgen knob tests 12 passed; plowrt cuda,hsa knob tests 19 passed, 1 ignored.
  Full-model download at 275GiB around 15:29 UTC; original download PID still alive.
- NVIDIA-to-AMD review: host timing instrumentation is backend-independent; inline
  mux port changes the AMD DEFAULT (config.mux_inline_tick defaults to its backend bool),
  so it needs a served gate. AMD multistep only defers readback and retains each token's
  drain/audit, unlike a device-resident graph quantum; don't assume CUDA gains transfer.
  Existing changes are preserved, not certified. main.rs's prior change to sampled
  amd-bench TP agreement weakens its oracle contract and must not be used for numerics.

## Next

### 2026-09-22 B64 follow-up

- B64/71680 supplied-index block gate PASSED all rows/stages/ranks, max residual
  rel-L2 .0002340786495. Captures use 2048 evenly spaced logical positions, not
  the learned indexer. `/opt/models/plow-block3-b64-ctx71680-gate-20260922`.
- Existing `PLOW_GLM_DECODE_NORM_ROWS=1` qualified at B64/512: T4 (300/warm30)
  ctl3045.760 / treat2805.392 / ctl23047.109 / treat22803.321 us.
  Control drift1.349us, treatment spread2.071us (<3x drift). Average medians
  3046.4345 ->2804.3565us =7.946% reduction. All72 output/intermediate files
  byte-identical across all four arms. All numerical/rank gates passed.
  `/opt/models/plow-block3-b64-normrows-t4-20260922-{ctl,treat,ctl2,treat2}`.
  Treatment also passed B64/71680 supplied-index gate. No global default promoted.
- Separate B64 normrows trace: rank0 envelope2.510ms; 9 GEMVs have summed
  overlapping tails1.413ms, FP32 router243us. Not serialized attribution.
  Optimistic B64/71680 packet floor45.116us /279716864bytes per rank, maximum
  expert reuse and one weight stream assumed. Excludes repeated MM4 walks,
  launch/fabric/activation overhead; not a measured roofline efficiency.
- Next isolated candidate: MM8/walk1 vs MM4/walk1 at B64/normrows1. New recipe
  `glm53.mi350x.fp8-block3-b64-mm8.toml`; paired build in progress. This targets
  repeated GEMV weight passes, with resource and numerical checks before timing.
- Full raw download live690GiB at~16:32UTC. Finished raw config SHA matches the
  config used to emit the initial full packet. Checksum verification still pending.
- Pure host port tests passed: multistep scheduler/runtime/output caps, AMD quantum
  at batch>8, deferred token-ring row-major bounds. Served AMD port remains unqualified.
- MM8 paired build completed: decode/GQ256VGPR/0AGPR/occ2/four spills (MM4 six).
  Packet byte-identical to B64 normrows control. T4 300/warm30:
  2809.981/2677.073/2812.042/2672.893us; drift2.061, treatment spread4.180 (<3x drift).
  Average medians2811.0115->2674.983us =4.839% lower. All72 output/intermediate
  files byte-identical across arms and all numerical/rank gates passed. Raw checksum
  hashing ran concurrently; preparation started before the final arm completed.
  Keep this exploratory until a quiet-box retest. No global default promoted.
- Full download process exited0; all141 raw shards present,755617140416weight bytes.
  HF checksum verification session53894 still running, JSON `/tmp/plow-full-download-verify.json`.
  CPU all78-layer prep completed session17180: `/opt/models/GLM-5.3-full-aca966e4-plow`.
  Added prep-lite --verify-only with explicit requested layers (defaultall78), not
  inferred from files present. Full prepared-name check116601/116601,13shape/dtype
  checks, verbatim expert check and q_a_proj dequant check all PASSED.
- Existing needle probe now has --exact-lengths using /tokenize + token-ID prompts,
  achieved length assertion and per-prompt SHA256. Campaign serve-bench accepts
  --quality-lens (freezes/hash-records probe and runs it before timing). Two pure
  needle tests and nine campaign tests passed. Retrieval is not full-logit numerics.
- HF checksum verification exited0: all155 remote files checked, no missing/mismatch;
  313 extra files are HF local download metadata/cache, not extra model weights.
- First full qualification launches were setup failures, no timed results:
  plow specialized MLA/MoE prefill GQ object omitted L2 dispatch despite the full
  packet being placed. Fixed build_gfx950.sh for MLA and MLA+MoE GQ variants AND
  their resource checks. Fresh full paired build `/opt/models/plow-glm53-full-l2fix-20260922`
  passed; symbol `plow_l2_place_dispatch_1` verified in specialized object; 256VGPR,
  occ2, eight spills for prefill_mla{,_moe}_gq. No packet/default changes.
  vLLM wrapper incorrectly required ROCR_VISIBLE_DEVICES even for full-machine
  leases (gpulease intentionally omits it there). Fixed optional visibility args,
  guarded startup cleanup PID; CPU tests cover full/partial visibility cases.
- vLLM v2 loaded all141shards then refused sparse-indexer warmup: requires
  VLLM_ROCM_USE_AITER=1. Added recorded [reference.env] in full recipe and harness.
  New vLLM rerun uses AITER and localhost bind. Job1790095595742083436-a5d59f8b,
  `/opt/models/plow-glm53-full-vllm-qual8k-aiter-20260922` in progress.
  Plow L2-fixed full C8/8192 qualification queued behind it, job1790095607059610473-37f82610,
  `/opt/models/plow-glm53-full-plow-qual8k-l2fix-20260922`. Both exact8192 needle probes,
  same client16prompts/128out/C8. Still numerics_qualified=false.
- MM8 B64/71680 supplied-index gate passed too, byte-consistent numerical error
  .0002340786495. Quiet T4b queued after serving jobs:
  `/opt/models/plow-block3-b64-mm8-t4b-20260922-{ctl,treat,ctl2,treat2}`.

1. B1/B8/B64 block and B8 supplied sparse-position gates are DONE above. Learned
   indexer, full prefill and full serving gates remain. Full initial build is ready
   for all-layer prepared checkpoint after download verification.
2. Control/control and static T4 are DONE above; do not repeat without a new question.
3. Tune top expert stages one lever at a time (e.g. co-residency via verified emit). Wider production decode
   ladders and sparse long-context captures remain missing from this block runner.
4. Validate full HF download with hf cache verify --revision <above> --local-dir
   <above> --fail-on-missing-files. Prepare ALL layers using matching GLM-MoE prep
   (glm52_prep_lite.py); glm53_prep.py currently targets another KDA-shaped checkpoint.
5. Re-emit verified full packets/objects and run the full matched matrix. Neither
   3000 tok/s nor any vLLM win has been established.

### 2026-09-22 full qualification and per-rung campaign continuation

- Full raw checksum verification and all78-layer preparation are DONE (above).
  Initial full8K/C8,16prompts,128output completed on both engines, all2048 output
  tokens and9/9 exact8192 needle cases; all prompt hashes match. Plow39.1205 vs
  vLLM134.0328 outputtok/s; meanTTFT5603.700 vs2775.776ms; meanTPOT157.4225 vs
  38.1228ms. Separate queued lifetimes, diagnostic first cell, not report-grade
  interleaved comparison or full matrix. Paths full-plow-qual8k-l2fix and
  full-vllm-qual8k-aiter under `/opt/models/plow-glm53-*-20260922`.
- Full-vocabulary teacher-forced logits captured at8192/71680 +4decode steps,
  all8 shard rows assembled (vocab154880), strict TP agreement. Exact pinned vLLM
  raw-logit oracle completed12 cases with no invalid cases, including two repeated
  prefill histories. `/opt/models/plow-glm53-full-logits-20260922/comparison.md`:
  all10 top tokens match, but8/10 rows FAIL the existing2x repeat-floor gate
  (3/5 at8K,5/5 at70K). Reference max full/head64 floors .109267/.0219146.
  Do not loosen tolerance or claim full-model numerical qualification.
- Quiet B64 MM8 T4b:2817.223/2686.604/2809.242/2685.434us,4.522% reduction,
  all72 outputs identical. B64 MFMA4 (existing kernel adapted via gfx950 object
  flag, samepacket/MM8/normrows1):2698.013/2300.657/2692.133/2296.177us,
 14.72% exploratory block reduction, allrow/stage/rank gates pass;63/72 output
  files differ from VALU (reassociation), no default promotion. These historical
  four-arm series used separate FIFO submissions; new block-ab uses one lease.
- Latest user steer: one modular block for EACH rung; inspect fusion, scheduling,
  kernel roofline; use hipBLAS or adapt kernels into plow segments. Owner rung
  initiallyB8 (currentservingmin), B64regression; thenB16/B32, thenprefill128/512/2048.
  Prioritize measured workload share and gap, one lever per rung. Existing block
  runner has no prefill capture support; do not label decode results prefill.
- `campaign.py block-ab`: freeze four arms, doctor, one GPU queue job/lease,
  CPUquietlock inside lease, numerical/repeat gates and robust drift+MAD scorer.
  B8control/control followed by B8MFMA queued17:10UTC. Scorer must reproduce null
  forcontrol/control; only blockcandidate PASS, never serving/default qualification.
- Fusion/scheduling inspection: B64normrows packet has27ops:9BF16Gemv +FP32router,
  AddNorm alreadyfused, routedexpert gate/up+activation alreadyMoeGroupGluPf,
  MLAmerge/value projection alreadyMlaMergeFold. Shared gate/up+Glu remain3ops;
  router/topk/align precede independent shared path. Most ops own256CUs; router
  topk256CUs for64rows. Existing `PLOW_GLM_DECODE_GLUE_CUS` is an isolated
  schedule-only candidate; test aftermatrix-core attribution, not simultaneously.
  B1-only norm/seam/Qnorm fusions cannot be applied blindly tobatchedrungs.
- hipBLASLt runtime route currently pins `glm_lt_gfx942.elf` plusgfx942descriptors
  andobject hashes; cannotreuseonMI350X. Startwith adaptedMFMA inside existing
  segments (no newnativeboundary); library alternative requires exactshape gfx950
  kernel/resource/ABI and numericalqualification includingboundarycosts.
  Analytical6.2TB/s/2300TF/s packetceilings remain optimistic, notmeasuredroofline.
- Same-lease B8MFMA qualification DONE (`plow-block3-b8-mfma4-ab-20260922`):
  ctl1196.218/treat1167.528/ctl21197.698/treat21167.899us; delta29.2445us=2.4432%,
  noise7.910us, drift1.480us, treatmentspread.371us. PASS as blockcandidateonly;
  allrow/stage/rank gates pass,72outputsrepeatidenticallywithinvariant,16files
  differVALUvsMFMA. Precedingcontrol/control measured -4.8945us vs9.689usnoise,
  correctlyno performancePASS.13then14campaignunit tests passed.
- `block-roofline --router-table` now validates/hashes the exact captured routing
  table and models its selectedexpert union. B8/51225experts,213676160bytes/rank,
  optimistic34.4639us; B64/51235experts,293903744bytes/rank,47.4038us. Actual
  arithmeticintensityreported perop; stillnoactualHBMutilizationclaim.
- B8fusion detail:2GemvQkv and1GemvGlu remainVALU; MFMA4changesonlyunfusedGemv.
  B8MFMA allranktrace (`plow-block3-b8-mfma4-trace-20260922`),rankenvelopes
  877.7–884.9us; groupedGLUlast-readytail202.3–208.3us andbalance-surplus141.9–148us.
  Trace/counter-derived diagnostics, not mutuallyexclusivewalltime attribution.
  NextprioritygroupedGLUlookup/staging, notnewQKVfusionbyhunch.
- Glue-CU isolatedcandidate testedtwice,samelease. First1167.098/1140.998/
  1166.208/1147.368us; second1166.288/1143.559/1166.818/1147.289us. Bothall72
  outputsbyteidenticalandnumericsPASS, buttreatmentspread>3*controldrift BOTHtimes.
  PARK,no accepted1.8–1.9%gain. Do not keep retestinguntilapass.
- NextlevercardB8: existing `mpf_expert_of_tile` linearlyscansupto256prefixentries,
  includingduplicatesforemptyexperts. Exposeexistingbinaryupperboundsearch through
  gfx950decode-object-only `PLOW_MOE_TILE_BINSEARCH=1`,default0. Samepacket,MFMA4/MM4,
  noglueCUSchange. HypothesisreducegroupedGLUstragglertail; expected20–100us/block
  (less thandiagnostic142–148usimbalance); rejectifbitwisediff, increasedscratch,
  ornobenefitabovedrift+MAD. Existingdefinealreadyregistered; addedRawEnvBoolentry.
  devgenknob12pass,plowrtknob19pass1ignored; shellsyntax/diffcheckpass.
- B16/B32independentCPUcaptures fromverifiedfullraw DONE. Separatepacketbuilds
  viarecordedPLOW_DECODE_BATCH_LADDER=16/32 override, MM4andnormrowsunchanged.
  B16VALUready; initialB16MFMAobjectbuildfailed becausebuildscriptwasedited while
  Bashstillreadit (syntaxoffset, notkernel/resourcefailure). Do notusepartialset.
  FreshB16MFMA-v2 thenB32VALU/MFMA buildsstarted; nofurtherbuildscripteditsduringthem.
- B8expertbinarysearch samelease PASS:1169.667/1124.237/1167.076/1127.137us;
  1168.3715->1125.687us,42.6845us=3.65333%,noise9.038us,drift2.591/spread2.900.
  `plow-block3-b8-binsearch-ab-20260922`,job1790097660575541206-7b8ef0b8done0.
  `--require-bitwise` machinegate passedall72files,no changes; allrow/stage/rank
  oracle gatespassed. Samepacket; decode/GQ256VGPR,occ2,sixcompiler spills;
  plow_execstatic scratch474->474,MFMA128->128. Separateinstrumentedtrace queued
  job1790097733676794237-0dd724dd tochecktargetedtail,not scoreagainsthosttimings.
  LandOPT-INonly. B16MFMA-v2buildDONE; rung16AB queued1790097706985192815-874712de.
  B32pairedbuildsinprogress(session50351); noB32timingsyet.
- B16MFMA samelease PASS, job1790097706985192815-874712de done0:
  1517.074/1484.413/1521.073/1484.393us;2.28235%reduction,noise11.569us;
  58/72outputschangeacrossarithmeticvariants,repeatidenticalwithinvariant,
  allrow/stage/rankgatespassed. B32bothbuildsDONE; ownMFMA ABqueued.
- Binarysearch B8trace confirmslookupbenefit,but mostlyDOWN,notGLU: rank0
  groupedGLU207.25->197.49us,DOWN98.38->65.40us; envelope877.9->843.2us.
  Both kernels callthechangedownerlookup;noexclusiveutilizationclaim. B64
  binarysearchpairedcandidatebuilding(session68030),notqualifiedyet.
- Fullnumericaldiagnosticcorrection: initialreference rebuilt everyappendeddecode
  history asPREFILL. ExistingunmodifiedpublicvLLMoracle supports incremental
  generation; queuedfourrequests(twooriginal8192/71680prompts+repeats),5outputs
  each, samepinnedimage/TP8/chunkbudget16384/AITER. Job1790097869297922815-943f0bb7
  done0,20validfullvocabrows,10exacthistoryrepeatpairs; allfourtokenstreams
  [220,22,101961,24,7671] matchplow. Newgeneration-run-record.json andgeneration-comparison.md
  underfull-logitscapture. DoctorpassedafterexplicitDockerclient+nixlibrarypaths.
- Incrementalcomparison passes existing2xGLOBALMAXreference-repeat gate for all10
  matchedsame-phaseplowrows; no missinghistories. BUT observedreferencefull/head64
  repeatfloorisnow .448868/.0781548 (high; containsdecodevariation), so this is
  limitedfloor-bounded evidence, NOT broadfull-modelnumericalqualification.
  Originalcross-phasefailureevidenceretained. Do not interpret largermeasured
  floor as proofdifferencesvanished. CandidatefullrelL2range.138–.616;top-token10/10.
- Hardenedlogitcomparison: nonfinitevectorsrefused, missingcandidatehistories
  failqualitygate, optional --require-same-phase + --require-pass producefailure
  exitcodes. Plowmanifest nowrecordsphase/generationstep. CPUtestscoverNaN/Inf,
  missinghistory,crossphase,completeidenticalcase. No numericalthresholdchanged.
- Pulled upstream fast-forward 6b12a05a -> 5a6d7a3e; tracked edits restored with
  original index byte-identical to /tmp/plow-prepull-QfgEbC/index.patch. Retained
  backup stash 6de280e7d592ba8908ed4a8c300d04031ba277f5. Incoming CUDA/Gemma MoE
  tail fusion is not a direct GLM port: hidden6144 exceeds its3072 bound and
  GLM has a TP reduction boundary. Per-member resource pruning remains relevant.
- B32 MFMA same-lease PASS, job1790098104042390870-b7f892ed done0:
  2013.269/1956.659/2016.939/1958.870us, 2.84549% reduction, noise12.479us.
  All numerical gates pass;61/72 output files differ between arithmetic variants,
  repeat-identical within each variant. No full-model/default promotion.
- B64 expert binary search first same-lease trial:2291.806/2246.877/2292.636/
  2243.327us; all72 outputs identical, but treatmentspread3.550 >3*drift0.830.
  Performance gate FAIL; do not report the apparent2.06% as an accepted win.
- Launch-overhead next diagnostic reuses DSTEP in the modular runner. Wire its
  token-window close and expose block-bench --dstep-log; mark instrumented records
  so block-ab refuses them. Host drain is remaining wait, not total GPU time;
  inactive-bank clears and submission can overlap GPU work. No new runtime knob.
- User now explicitly requires all dtype axes match pinned vLLM. See
  plans/glm53-precision-parity.md, read before further qualification. Existing
  serving cell is workload-matched but arithmetic-mismatched, diagnostic only.
  Reference W8A8, FP8 MLA BMM and FP8 indexer query/key differ from Plow BF16
  paths. In particular vLLM's BF16 wk exception does NOT extend to wq_b; corrected
  misleading emitter comments. Pause promotion of W8A16 wins while closing parity.
  Full 3000 tok/s/context/concurrency objective unchanged. No reference weakening.
- Diagnostic plumbing verified: release HSA build passed, devgen knob12,
  runtime cuda+hsa knob19+1ignored, block oracle2 tests, campaign15 tests,
  diffcheck clean. Initial block test filter matched0; corrected filter
  amd_block_cmd matched2 and both passed. New serve records explicitly mark
  precision_qualified=false and include declared Plow precision; no matched-dtype
  performance result exists yet. No diagnostic GPU job queued after parity request.
- Precision inventory now DONE on all8 ranks/78layers; use
  plans/glm53-precision-parity.md for captured axes and next numericalgates.
  NativeFP8/block128 GEMM passes14f64cases. PinnedAITERsameoperandcomparison
  passes11/12alignedshapes; outputprojection M8N6144K2048 repeatunstable inCK
  splitK2, repeatedandsaved. Not promoted. Nativeper128 activationquant added
  in isolatedtestpath; bitwiseAITERboundarytestqueued1790100660281787609-db3cdacc.
- Useradds minimum800outtok/s at8K C16, broader3000/matrixgoalunchanged.
  SuppliedSGLang616baselinecommand isGLM-5.2-FP8 TP8,TileLangDSA,prefillbudget131072,
  memfraction.80; notsameGLM-5.3modelandGPU/outputlenunknown. Preserveasreference
  claim, not apples-to-apples result. No droppingvLLMprecisioncontract.
- Nativeactivationquant job1790100660281787609-db3cdacc DONE0:15/15 tests
  bitwiseFP8andFP32scale parity withpinnedAITER acrossrungs1/8/16/32/64,
  K256/2048/6144. Isolateddeviceprimitive, notserving. ComparatorCPU3pass.
  Nextconnectquant+block128GEMM inone modularpacket andauditoutputprojection
  splitKrounding; donotdeclareentireblockmatchedbeforeMLA/indexer/MoEalignment.
- Opt-in native o_proj quant+W8A8 block128 packet now passes B16/ctx512/TP8
  modular gate: job1790102421549270458-353e47b0, v3-check artifact,1509.283us
  diagnostic only. Fixed exposed shared FP8 GEMV truncation above compiledMM;
  standalone all-row regression12/12 passes (old kernel10/12fails).
  Details/failed trials in glm53-precision-parity.md. No accepted performance
  promotion or matched-vLLM dtype claim; prefill GPU and realboundary audit next.
