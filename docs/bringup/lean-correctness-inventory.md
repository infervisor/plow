# Lean correctness inventory (gemma12b-next)

Phase 0 of the lean-correctness plan. It records what the verifier and receipt system cover today,
derived from the code rather than from `03-lean-verify.md`.

Baseline: `gemma12b-next` 4978bf03, compared with `main` fef89819 (124 commits, 310 files). The
branch does not change `lean-plow/`. The production bundles were built at 4de53bf6, which has the
same compiler, `plow-asset` and Lean sources as 4978bf03, and their verifier is `becacc81…`. That is
byte-identical to a fresh `lake build plow_verify` of the baseline sources.

## 1. What the branch changes

| Area | Change | Main files |
| --- | --- | --- |
| Emitters | Gemma-4 vision/audio towers as `mm_vision.pkt`/`mm_audio.pkt` sidecars (`PLOW_EMIT_MULTIMODAL`), new ops RmsNormF32/RopeAxialF32/ChunkAttentionF32/MmRowsBf16, Embed pad row | `devgen/src/mm.rs`, `devgen/src/lib.rs`, `packet/src/dev.rs` |
| Emitters | 26B MoE cuBLASLt prefill/W8A8 chains, MoE decode cap 128, sm_89 cuBLASLt gates | `devgen/src/dense_cublaslt.rs`, `devgen/src/lib.rs` |
| Emitters | Silero VAD v5 packet, Nemotron cache-aware streaming conformer, Parakeet TDT, qwen3-asr dims from checkpoint, Orpheus voices | `devgen/src/{vad,conformer,rnnt,asr/qwen,tts}.rs` |
| Schemas | `plow.multimodal.v1` contract (`MmContract`), speech contract (`asr_vocabulary.json`, VAD/ASR contract 1), `runtime_requires {cublaslt, plowrt_contract}` | `plow-asset/src/{multimodal,speech_contract}.rs`, `plowrt/src/asset/serve.rs` |
| Object routes | 31B and qwen3-0.6B cuBLASLt prefill shape lists, MoE decode cuBLASLt via `moe_lt::decode_segments`, `_routed_fp8kv` cubin, generated hd256/hd512 attention cubins (recipe only) | `plow-asset/src/segment_roles.rs`, `plowrt/src/exec/gpu/{moe_lt,cublaslt,decode_rung}.rs` |
| Runtime patches | Bit-31 soft-token ids (mm table), optional `vmm.kv` (slot-granular rings), prefix-cache relieve on OOM, KV OOM admission retry, `in.pos_base` reset, ordered mm slab writes | `exec/gpu/{token_batch,mixed_step,prefix}.rs`, `memory/vmm.rs`, `exec/gpu.rs` |
| Serving | `serve/mm` slab/encoder per engine (51ee0d9a), `--dp` router, manager load/unload, tools, streaming, ASR realtime/VAD gating, TTS realtime | `serve/{mm,dp,manager,mux,tools,stream}`, `asr/serving`, `tts` |

No new emitter path has CPU golden tests for the four new ops. Several runtime paths have no
tests: bit-31 admission, MoE decode routing, the KV OOM retry, `serve/mm/encoder.rs` and
`manager.rs` unload. These are numerical/serving gaps outside the Lean scope, listed here so the
GPU lane can select them.

## 2. Production support matrix

All 14 H100 production recipes target `sm_90a`, 132 CUs, TP1. Bundles live in
`/opt/dlami/nvme/lava-tts/allbuild/b/` (`campaign.py build --no-probe`).

| Bundle | Network / precision | Decode ladder; cuBLASLt | Builder | Packets (programs) | Receipts on disk |
| --- | --- | --- | --- | --- | --- |
| gemma-4-12b | gemma4, W8A8 + FP8 KV, 16K, packed prefill | 1..32,128; decode LT ≥128, prefill LT, gen hd256/hd512 attn | plowc devblob | model (19) | A + 19 coarse D |
| gemma-4-26b-fp8 | gemma4 MoE, W8A8 + FP8 KV, 128K | 1..128; LT ≥64 | plowc | model (20) | A + 20 coarse D |
| gemma-4-26b-bf16 | gemma4 MoE, BF16, 16K | 1..32; LT ≥16 | plowc | model (18) | A + 18 coarse D |
| gemma-4-31b-fp8 | gemma4, W8A8 + FP8 KV, 16K | 1..32; decode LT off, prefill LT | plowc | model (18) | A + 18 coarse D |
| gemma-4-31b-bf16 | gemma4, BF16, 16K | 1..8; LT ≥4 | plowc | model (16) | A + 16 coarse D |
| gemma-4-e4b | gemma4 E4B, BF16 | 1..128; LT ≥1 | plowc | model (18) | A + 18 coarse D |
| qwen3-asr | qwen3 audio-LM | 1..128; LT ≥32 | plowc | model (14), encoder (460) | A + 14 coarse D; encoder 46 distinct LTE, 460/460 mapped |
| qwen3-asr-0.6b | qwen3 audio-LM | 1..128 | plowc | model (14), encoder (360) | A + 14 coarse D; encoder 36 LTE, 360/360 |
| veena | llama TTS | 1..128; LT ≥32 | plowc | model (14), codec (19) | A + 14 coarse D; codec 1 LTE, 19/19 |
| orpheus | llama TTS | 1..128; LT ≥32 | plowc | model (14), codec (19) | as veena |
| chatterbox | llama TTS + S3Gen | 1..128 | plowc | model (14), s3gen (1931) | A + 14 coarse D; s3gen 143 LTE, 1931/1931 |
| chatterbox-mtl | as chatterbox | 1..128 | plowc | model (14), s3gen (1931) | as chatterbox |
| nemotron-3.5-asr | FastConformer RNNT, streaming | n/a | `examples/asr/nemotron_pipeline_compile.rs` | nemotron.pkt | none |
| silero-vad | Silero v5 | n/a | `examples/asr/silero_vad_compile.rs` | silero_vad.pkt | none |

No production recipe sets `PLOW_EMIT_MULTIMODAL`. The `mm_*.pkt` route goes through the same sidecar
verifier hook, but it has no production bundle to qualify.

Every `packet_sha256` on disk matches its packet. Every receipt names verifier `becacc81…`.

## 3. Checkpoint call paths (actual)

| CP | Production caller | Runs when | Skipped when | Persisted |
| --- | --- | --- | --- | --- |
| A | `plowc main.rs devblob_verify_hook` (catalog + bodies) | every devblob emit with verify on | verifier unusable (incl. non-Linux `call_batch_bound`) | `RewriteBodyExpansion` receipt |
| A,B,D,E,F | `plowc lib.rs run_lean_verify` (schedule path) | `--lean-verify` (opt-in) | no binary (silently forced off) | build.json only. E sends a fixed 2-frame sample, not the program |
| C | none (tests only) | — | always | — |
| D coarse | devblob hook, programs with `reduction_witness` | default | unusable; wire protocol not bindable → `dependency_binding_gaps` | `CoarseDependencyPreservation` |
| D effects | devblob hook, every program where `logical_effects::obligation` is `Ok` | default | first unaudited op → `logical_effect_gaps` (log only) | `LogicalTensorEffects` |
| D ordering + G | devblob hook per program (`address_map: []`, GQ topological order, staged-LDS fit) | default | unusable | build.json `verified`, `lds_fit_verified`; no receipt |
| D sidecar | devgen `write_sidecar_packet` (codec, s3gen, encoder, mm towers) | sidecar verifier installed | not installed / unusable → receipts deleted | `<stem>.lean-checks.json` |
| K | devgen `knob_spec::gate` on every devblob emit | default | `--no-knob-verify`; unusable binary is fatal | build.json `knobs.K` |
| S | `plowrt knob-scope` via `scripts/knob_scope_ci.sh` | CI, knob changed | env unset → exit 0 with warning | report file |
| P | `scripts/perf_cert.py` via `perf_gate_ci.sh` | CI, default flip | no verifier → exit 0 with warning | `perf-certs/*.json` |
| R | devblob hook, measured/GEMM policies | policies recorded | unusable | `MeasuredPolicy` / `SelectedGemmPolicy` |
| L | devblob hook, `FlashMerge i4=1024` (padded MLA) | GLM-class packets | unusable | `LayoutMapping` |

Runtime replay (`plowrt certificate_checks::check_packet`, called by the CUDA, HSA and CPU engines):

- It loads silently with no receipts, with zero checks, with a missing or changed verifier, or with
  an unusable binary.
- It rejects a binding mismatch, a rejected certificate or a changed envelope.
- Sidecar receipts are never read at runtime.
- `CudaPacketRuntime` (encoder/codec/s3gen on CUDA) never calls it.
- Nemotron, Parakeet and Silero packets bypass every gate, including K.
- Hazard: `check_packet` always reads `<dir>/lean-checks.json`. A CPU/Metal load of `encoder.pkt`
  therefore reads `model.pkt`'s receipts and fails the hash binding.

### Logical-effects coverage of the main packets

`logical_effects::obligation` covers 46 opcodes (the FP32 speech set plus Gemm/Gemv/norm/residual
basics). It rejects a whole program at the first unaudited op. It also rejects TP, packed/token-batch
siblings, `l2_domains`, fine counters and `@` aliases. Unaudited opcodes in production main packets
(from each `build.json` `opcodes`):

| Bundle | Unaudited opcodes |
| --- | --- |
| gemma-4-12b, 31b-fp8 | Argmax ArgmaxFin Embed FlashDecodeFp8 FlashMerge FlashPrefillFp8 GemmFp8 GemvArgmax GemvFp8 GemvGluFp8 HeadNormRope HeadNormRopeFp8 QuantFp8 SoftCap |
| gemma-4-26b-fp8 | the 12B set without GemvArgmax, plus GemmGluFp8 MoeAlignGemmaPf MoeCombineNormGemma(Pf) MoeExpertDownGemmaFp8 MoeExpertGluGemmaFp8 MoeGroupDown/GluGemmaPfW8a8 MoeRouterGemmaPf/ScoreFast/Topk |
| gemma-4-26b-bf16 | Argmax ArgmaxFin Embed FlashDecode FlashMerge FlashPrefill GemvGlu GemvQkv HeadNormRope SoftCap + 10 BF16 MoE ops |
| gemma-4-31b-bf16 | Argmax ArgmaxFin Embed FlashDecode FlashMerge FlashPrefill GemvArgmax GemvGlu GemvQkv HeadNormRope SoftCap |
| gemma-4-e4b | Argmax ArgmaxFin Embed FlashDecode FlashPrefill HeadNormRope SoftCap |
| qwen3-asr(-0.6b), veena, orpheus | Argmax ArgmaxFin Embed FlashDecode FlashPrefill GemvGlu GemvQkv HeadNormRope |
| chatterbox(-mtl) | Argmax ArgmaxFin EmbedPosBf16 FlashDecode FlashMerge FlashPrefill GemvGlu GemvQkv HeadNormRope |

That is 34 distinct opcodes. Every main-packet program is a logical-effects gap today. Since the 3d0b6948 merge, op 210 `MmSpanExtent` (multimodal Gemma main packets) is a 35th. The sidecars
(encoder, codec, s3gen) are fully covered.

## 4. Obligations per program

"Complete" means a receipt exists, binds to the loaded bytes and replays. A skipped verifier or an
empty input is not complete.

| Packet kind | Required obligations | Today |
| --- | --- | --- |
| plowc main packet | A once; per program: coarse D, logical-effects D; L per padded-MLA producer | A + coarse D complete; logical effects 0 of N |
| plowc sidecar | per program: logical-effects D via `program_checks` | complete for all 4 production sidecar kinds; checked at load under `PLOW_LEAN_QUALIFY` |
| multimodal sidecars (`mm_vision.pkt`, `mm_audio.pkt`, merged from mm-vision at 3d0b6948) | sidecar route (not named `model`), receipts through `write_sidecar_packet`; `media_geometry.v1` multimodal family on the main packet | no built bundle yet; the 4de53bf6 baseline predates them |
| policy receipts (R) | none required. They bind when present (selection evidence, not structure) | — |
| example-built ASR/VAD packets | per program: logical-effects D, or a recorded exemption | no gate, no receipts |
| runtime transforms (decode rung, prefill patch, packed descriptors, mixed step) | separate execution subjects (Phase 3) | none |
| KV ring / VMM lifecycle | `kv_ring.v1` per packet with packed prefill over a sliding ring; `vmm_trace.v1` on runtime traces (CPU lane) | done (§11) |
| multimodal slab lifecycle | reserve/stage/release/unload invariants (Phase 4) | Rust tests only |
| speech fusions | `speech_fusion.v1` per packet with a fused site | done (§12); no production packet has one |

## 5. Proof roots

`lean-plow/proof-manifest.json` maps every dispatched endpoint to its acceptance theorems.
`lake exe proof_audit` checks them. It verifies that the manifest ids equal
`Plow.CLI.Dispatch.endpointIds`, that each listed theorem exists, and that each theorem's axiom
closure is inside {propext, Classical.choice, Quot.sound}. Where a checker is named, the theorem's
statement must mention it. A self-test shows that the audit rejects `sorryAx` and a fresh axiom.

| Endpoint | Roots |
| --- | --- |
| A | `RewriteBody.check_sound`, all `Rewrite.rule_*` |
| B | `TilePartition.check_sound` |
| C | `Sram.occupancy_le_of_temporal_fit` |
| D | `Verify.{verifyDependencies,checkPaths,verifyAddressMapVia,treeBefore}_sound`, `verifyAddressMap_sound_strict`, `Effects.{checkConflicts,checkAccesses,checkReuse}_sound`, `cancellation_retired` |
| E | `Wire.decodeProgram_encodeProgram` |
| F | `verifyAddressMap_sound_strict`, `verifyAddressMapVia_sound` |
| G | `LdsFit.fits_of_check_ok` |
| K / S / P | `Knobs.checkK_sound`; `Scope.checkS_sound`, `diff_complete`, `untouched_rungs`, `route_untouched`; `Ledger.checkP_sound` |
| R | `MeasuredPolicy.check_sound` |
| L | `MlaLayout.address_in_bounds`, `ignored_padding`, `same_consumer_output` |

D/F acceptance always runs the proven reference checker. `FastCheckD` (`implemented_by` unsafe
kernels) can only reject early. `Plow/MlpInterleave.lean` was not imported by any root and so was
never built. It is now imported.

## 6. Ignored tests

196 ignored tests in total: 88 CPU-verifier, about 35 artifact, about 60 GPU, and 13 other
(microbenchmarks, simulations, permissions). The CPU lane runs the explicit list in
`scripts/lean_correctness_ci.sh`; adding an ignored CPU-verifier test means adding it there.

- **CPU-verifier:**
  - all ignored tests in `lean_verify` (tests/*);
  - all ignored tests in `plowc` integration targets, plus `plowc --bin plowc cli_tests::actual_devblob_hook_*`;
  - `plowrt --lib certificate_checks::tests::*`.
- **Artifact:** `plowrt` `vmm_packet_geometry`, `vmm_ring_tests::actual_*`, `knob_scope::incident_fixtures`,
  `exec/gpu/*_tests::actual_*`, `cpu_serve_live`, `plow-asset packed_prefill::fp8_*`. These need bundle
  paths in env vars, so they belong to the production artifact lane.
- **GPU:** `exec/gpu/decode_rung_tests::gpu_*`, `token_batch::unified_cuda_*`, `gpu_consume_prompt`,
  and the AMD `amd_*` replays.

Baseline failures (4978bf03): `devgen --lib` `gfx950_coverage` (2 tests, also on main). They are
unrelated to this work and are not chased.

## 7. Corrections to `03-lean-verify.md`

- 12 endpoints are dispatched (A–G, K, S, P, R, L), not seven.
- No pipeline runs all of A–G. The schedule path (opt-in) runs A/B/D/E/F. The devblob path runs
  A/D/G/R/L.
- A receives the whole rule catalog and bodies, not the fired rules.
- C has no production caller.
- E checks a fixed sample.
- The D/F cache key includes the checkpoint letter.
- Devblob D sends an empty address map.
- `lake build` only warned on `sorry`. Warnings are now errors (`warningAsError`), and
  `proof_audit` is the axiom gate.
- Binary lookup order: `PLOW_VERIFY_BIN`, then crate-relative, then `PATH`.
- Skips are not all recorded:
  - non-Linux skips every batch;
  - effect gaps are log-only;
  - ASR example packets bypass every gate;
  - missing runtime receipts load silently.

## 8. Gap → task

| Gap | Task |
| --- | --- |
| No completeness or verifier policy at load or qualification | Strict policy, `plowrt qualify`, `PLOW_LEAN_QUALIFY` (Phase 1) |
| Sidecar receipts unchecked at load; encoder receipt-file hazard | Sidecar route in `certificate_checks`, `program_checks` completeness (Phase 1) |
| 34 unaudited opcodes in main packets | Extend `logical_effects` footprints (Phase 2 follow-up). Until then a listed gap per bundle |
| Example-built ASR/VAD packets have no receipts | Route their builders through the sidecar verifier hook, or record an exemption |
| KV ring / shared backing lifecycle | Done: `kv_ring.v1`, `vmm_trace.v1` (§11) |
| mm slab lifecycle | `Plow/Multimodal.lean` + `serve/mm` traces (Phase 4) |
| ASR/TTS/mm capacity and shape contracts | Done: `media_geometry.v1` (§10) |
| Speech fusion preconditions unbound | Done: `speech_fusion.v1` (§12) |
| Runtime patch/rung transforms unchecked | Phase 3 (not started) |
| Object capability model | Phase 2 support checker (not started) |
| Serving trace / tool chunking | Phase 7 (not started) |
| CI: no Lean build, audit or ignored CPU tests | Done: `scripts/lean_correctness_ci.sh` step in `.github/workflows/build.yml` (nix-build job) |

## 9. Strict qualification policy

`plow_asset::certificates` derives each packet's required obligations from its programs. It then
matches receipts to them and records a `Qualification`: required and satisfied obligations, policy
receipts, receipt verifiers, and gaps. `plowrt certificate_checks` adds the replay.

- **Required set.**
  - Compiler packet (`model.pkt`, or any packet whose `lean-checks.json` binds its bytes): A once;
    coarse D and logical-effects D for every program; L for every padded-MLA producer.
  - Any other packet is a sidecar: logical-effects D for every program, through `program_checks`.
  - Policy receipts (R) bind to the wire when present. They are never required.
- **Gaps.** Each of these is a gap:
  - receipts absent, or `packet_sha256` not matching the loaded bytes;
  - a receipt that does not rebind (request ≠ reconstruction from the loaded packet);
  - a wrong or absent program, a duplicate or unused receipt, or an unsupported scope;
  - a `None` or out-of-range `program_checks` entry, or `program_checks` shorter than the
    program count;
  - a receipt verifier that is not approved;
  - a current verifier that is unavailable or not approved, a replay rejection, or a changed
    envelope.
- **Verifier identity.** `lean-plow/approved-verifiers.json` pairs each approved `plow_verify`
  sha256 with the digest of the Lean sources it was built from (`lean_verify::lean_sources_sha256`).
  The commit is recorded but not matched. `lean_verify` tests fail when the current sources have no
  entry, and (ignored, CPU lane) when the built binary is not the entry for them. Receipts from
  `becacc81…` (the baseline that signed the production bundles) replay on the current verifier with
  identical envelopes.
- **Cache.** Replay results are cached per (receipts digest, verifier, required-scope-set digest),
  so a verdict is never reused for a different scope set.
- **Entry points.**
  - `plowrt qualify --assets DIR...` qualifies every `*.pkt` with replay (`--no-replay` skips the
    verifier and then cannot qualify). It exits non-zero on any gap.
  - At load, `PLOW_LEAN_QUALIFY` applies to `check_packet` (CUDA/HSA/CPU engines) and to
    `check_sidecar` (`CudaPacketRuntime`, i.e. encoder/codec/s3gen on CUDA).
- **Knob.** `PLOW_LEAN_QUALIFY` (`rt.lean_qualify`, `--lean-qualify`, opt-in) takes `off`, `report` or
  `strict`. The default is `off`.
  - `off` is the pre-policy load path, with one fix: a sidecar now reads `<stem>.lean-checks.json`,
    never the compiler packet's receipts.
  - `report` also logs every gap.
  - `strict` refuses any packet with a gap.
  - The default stays `off` because `strict` refuses all 14 production bundles: 12 have the
    logical-effects gap and 2 have no receipts. `report` re-reads sidecar obligations at load
    (s3gen has 1931 programs), so it is not free either.

Strict results for the 14 H100 production bundles (`plowrt qualify`, replay on), at 4de53bf6:

| Bundle | Packet | Verdict | Gap |
| --- | --- | --- | --- |
| gemma-4-12b | model.pkt | 21/40 | logical effects 0/19 (first: Embed) |
| gemma-4-26b-fp8 | model.pkt | 22/42 | logical effects 0/20 (Embed) |
| gemma-4-26b-bf16 | model.pkt | 20/38 | logical effects 0/18 (Embed) |
| gemma-4-31b-fp8 | model.pkt | 20/38 | logical effects 0/18 (Embed) |
| gemma-4-31b-bf16 | model.pkt | 18/34 | logical effects 0/16 (Embed) |
| gemma-4-e4b | model.pkt | 20/38 | logical effects 0/18 (Embed) |
| qwen3-asr | model.pkt / encoder.pkt | 16/30 / **qualified 460/460** | logical effects 0/14 (HeadNormRope) |
| qwen3-asr-0.6b | model.pkt / encoder.pkt | 16/30 / **qualified 360/360** | logical effects 0/14 (HeadNormRope) |
| veena | model.pkt / codec.pkt | 16/30 / **qualified 19/19** | logical effects 0/14 (Embed) |
| orpheus | model.pkt / codec.pkt | 16/30 / **qualified 19/19** | logical effects 0/14 (Embed) |
| chatterbox | model.pkt / s3gen.pkt | 16/30 / **qualified 1931/1931** | logical effects 0/14 (HeadNormRope) |
| chatterbox-mtl | model.pkt / s3gen.pkt | 16/30 / **qualified 1931/1931** | logical effects 0/14 (HeadNormRope) |
| nemotron-3.5-asr | nemotron.pkt | 1/82 | no receipts. Fixed: the builder now writes them, and a rebuild is byte-identical and qualifies 82/82 |
| silero-vad | silero_vad.pkt | 1/3 | no receipts. Fixed the same way: rebuild byte-identical, 3/3 |

Every receipt that is present binds and replays. Counts include the `media_geometry.v1`
obligation (§10), which every speech bundle satisfies, and the `kv_ring.v1` obligation (§11),
which every Gemma packet satisfies. The remaining gaps are the 34 unaudited
opcodes in §3, and a rebuild for nemotron and silero. Their builders
(`examples/asr/{nemotron_pipeline_compile,silero_vad_compile}.rs`) now write through
`devgen::write_sidecar_packet`. Rebuilt into scratch from the same inputs, both packets are
byte-identical to production, and their new receipts qualify. Parakeet uses the same pattern but
has no production recipe, so it is not converted.

## 10. Media geometry (`media_geometry.v1`)

`Plow/MediaGeometry.lean` defines one contract per speech/multimodal family over `Nat`. The acceptance
theorem is `check_sound`. The lemmas give what an accepted contract implies for every input up to the
declared maxima.

| Family | Contract (abridged) | Lemmas |
| --- | --- | --- |
| audio_lm (qwen3-asr) | Encoder and LM agree on sample rate and max samples; LM hidden equals encoder width; rows of the longest recording ≤ overlay rows and ≤ encoder output rows; rows + reserve < context; rows + max tokens ≤ context; audio/stop ids < 2^31 and distinct | `AudioLm.every_recording_fits` (via `audioRows_mono`) |
| rnnt (nemotron) | `max_samples/hop + 1 ≤ input_frames`; the strided-conv chain maps `input_frames` to `frames ≤ joint_rows`; transforms well formed; blank ≤ vocab < 2^31; positive symbol bound | `Rnnt.every_input_fits` (via `chain_mono`), `Rnnt.decode_steps_bounded` |
| codec_lm (veena/orpheus) | LM and codec agree on rate, codebook, frame codes and samples; code tensor holds whole frames; PCM ≥ frames × samples; window + lookahead ≤ frames; audio token range < 2^31, with stops outside it; max new + fixed prompt < context | `CodecLm.addresses_in_bounds`, `CodecLm.token_decodes` |
| guided_lm (chatterbox) | speech/text control ids inside their vocabularies; `max_speech_tokens + 2 ≤` speech positions; overlay ≤ context | — |
| vad (silero) | frame + context = input elements; 2 × banks equal, positive state tensors; min speech ≤ max duration | — |
| multimodal (`plow.multimodal.v1`) | contract hidden equals encoder output width; pad/placeholder/begin/end < 2^31; table capacity a power of two ≥ slab rows; slab and table tensor bytes match | `Multimodal.soft_ids_disjoint` |

`plow_asset::media_geometry::request` derives the input from the bundle's packet pipeline sections,
which `PacketAsset` validates against the tensor bytes, and from `asr_vocabulary.json` and the
multimodal contract. A missing parameter, an absent referenced sidecar, or an overflowing shape
product rejects. Unknown fields and kinds are rejected on the Lean side. `plowrt qualify` adds the
obligation to the packet that owns the pipeline. Load-time `strict` does not re-run it.

## 11. KV ring and VMM lifecycle (`kv_ring.v1`, `vmm_trace.v1`)

`Plow/KvRing.lean` holds both endpoints.

- **`kv_ring.v1`.**
  - **Input.** Launches of the form `{ring_log, window, capacity, spans:[{slot, start, len}]}`.
  - **Contract.** `0 < window`, `ring_log < 32`, `capacity < 2^32`. Every non-empty span has
    `window + len ≤ ring + 1` and `start + len ≤ capacity`. Non-idle slots are distinct.
  - **Theorems.**
    - `checkLaunch_sound`
    - `launch_safe`: no row a span writes lands on a ring row that one of its own queries reads,
      for absolute positions across any number of wraps.
    - `launch_safe_masked`: the same for the kernel's `pos & (ring − 1)`, via `mask_eq_mod`.
    - `slots_distinct`
- **Derivation.** `plow_asset::kv_ring` builds launches from the packet's `live_kv` caches: sliding
  rings, power-of-two stride, `mask = stride − 1`. It reads the request limit and stage rows from the
  `packed_prefill` section. Launches come from the runtime planner itself (`plan_with_limit`,
  `stage_slots`), run on boundary mixes:
  - one full request ending at the context;
  - staggered requests that wrap the ring;
  - one one-row request, whose padding writes the rest of the bucket on an unmasked plan.
  A span is a maximal run of one slot at consecutive positions. A negative position rejects. A slot
  split into two runs is rejected by Lean. `plowrt qualify` adds the obligation to every packet with
  packed prefill over a sliding ring. Production: all six Gemma packets pass.
  - 12B, 26B and 31B: ring 2048, window 1024, staged 1024 of 4096.
  - E4B: ring 4096, window 512, unstaged 2048.
- **Rejected mutations.** `lean_verify` test `kv_ring` checks that these are rejected:
  - 12B unstaged;
  - 12B unmasked;
  - E4B at half the ring;
  - E4B with a wider window;
  - a split slot;
  - a context overrun.
- **`vmm_trace.v1`.**
  - **Input.** Driver events: `reserve`, `address_free`, `create`, `release`, `map`, `unmap`,
    `access`, plus `quiesce`.
  - **Transition relation.** `step`.
  - **Invariant (`State.Inv`).** Mappings reference live handles, are pairwise disjoint, and lie
    inside reservations.
  - **Theorems.**
    - `step_preserves`, `run_preserves`.
    - `traceOk_sound`: every prefix of an accepted trace satisfies the invariant. With `quiesce`,
      everything is returned.
- **Trace source.** Real `VmmRings` traces, from `memory::vmm::ring_tests` (the mock `VmmOps`
  records every successful call). They cover:
  - sub-granularity shared backing, released out of order and remapped;
  - injected create, map and access failures with prefix growth afterwards;
  - a failure on the second tensor;
  - reserve failures in the constructor.
  All are accepted. These mutations are rejected:
  - release while mapped;
  - double release;
  - map outside the reservation;
  - access before map;
  - a leaked reservation.
- **Out of scope.** Device completion before reuse: `VmmOps` carries no retirement events, and
  checkpoint D memory effects cover retirement. Live-context growth beyond the packet `max_ctx` is
  not covered either: capacity is the packet's `max_ctx`, and the ring bound does not depend on it.

## 12. Speech fusion preconditions (`speech_fusion.v1`)

`Plow/SpeechFusion.lean` binds the hypotheses of `Plow.Speech` to emitted instructions.
`plow_asset::speech_fusion` derives one site per fused instruction.

- **LayerNorm prologue** (`DenseGemmF32`, flag bit 5).
  - **Contract (`LnSite.Contract`).**
    - The last earlier writer of `stats`, in (program, pc) order, is `RowStatsF32`.
    - It sits in an earlier program, and its `x` is the GEMM's A.
    - It covers `a_row0 + M ≤ rows`, with `feat = K`.
    - The stats tensor holds `8 × rows` bytes. gamma and beta are absent or `4K` bytes.
    - No write of A or of stats occurs from the writer's program through the GEMM's. Same-program
      writes count whatever their pc.
  - **`ln_site_stats`.** Turns the contract into `ln_prologue_eq`'s hypothesis `hstats`, given two
    instruction-semantics premises: RowStatsF32 writes its input's statistics, and an unwritten
    tensor is unchanged.
  - **`ln_site_gemm_eq`.** Gives the full GEMM equivalence.
- **Conv1dF32 `row_scale`.**
  - **Contract (`ConvSite.Contract`).** Lean recomputes `out_rows` from the raw fields `in_rows`,
    `pads` (`j0`), `kernel`, `dilation` and `stride`. Then:
    - `row_scale = 4·batch·out_rows` bytes;
    - `out = 4·batch·out_rows·out_channels` bytes;
    - the residual is present at the output's size;
    - the scale aliases neither the output nor the residual.
  - **`conv_site_in_bounds`.** Every `(b, t, c)` index into the scale and the residual is in bounds.
  - **`conv_site_eq`.** Lifts `conv_row_scale_residual_eq` to every element.
- **Qualification.** `plowrt qualify` adds the obligation to any packet with a fused site.
  - None of the 14 production bundles has one: Qwen's `fuse_layer_norm` is off, and no production
    builder emits `row_scale`.
  - The ignored test `certificate_checks::tests::emitted_speech_fusion_sites_meet_the_lean_preconditions`
    checks devgen's actual emitters: a two-layer fused Qwen audio tower and `conv1d_f32_row_scaled`.
    Both are accepted.
  - It rejects nine single-field mutations, one per bound precondition. It also checks that a GEMM
    whose stats writer is removed cannot be derived.
- **Assumptions.**
  - Programs run in index order, one after another. This holds for forward-pipeline sequences.
  - The arithmetic premises are those of `Plow.Speech`.
