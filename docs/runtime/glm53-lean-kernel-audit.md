# GLM-5.3 Lean and compiled-kernel audit

Offline audit at `d413e331`, 2026-09-10. [Machine-readable evidence](../../runtime/bench/amd/glm_projection/lean-compiled-audit.json) pins the qualified packet, manifest, runtime, verifier and existing diagnostic trace. No new GPU benchmark was run. Lean can constrain legal transformations; it does not predict which kernel is fastest.

## Concrete missing optimization path

The ordinary GLM decode projection closure in [`mla.rs`](../../crates/devgen/src/mla.rs) emits `Gemv` on `all.clone()`. It does not call `blocked_gemv_cus_tuned`. That helper's only production callers are K3/KDA. Consequently `PLOW_GEMV_WG_TUNING` cannot tune these GLM projections. `PLOW_GLM_GEMV_WG` reaches the fused QKV projections, but their staged-LDS fit gates disable them at B8 on these assets.

The qualified B8 packet confirms 304 workgroups for **every** ordinary GEMV. With the current blocked mapping `per=ceil(N/304)`, N=64 leaves 240 groups without output columns; N=256 leaves 48. This is a coverage gap in the tuning path, not evidence that an opcode is missing from the loaded interpreter.

Joining instruction IDs from offline disassembly to the existing 65k-context decode trace prioritizes these shapes:

| M×N×K | Instructions | Aggregate body span, ms | First experiment |
|---|---:|---:|---|
| 8×256×6144 | 225 | 12.037 | Fixed-point removal of empty groups; then measured width sweep |
| 8×2048×6144 | 78 | 4.399 | Compare blocked GEMV widths / a qualified small-M matrix kernel |
| 8×6144×2048 | 78 | 4.175 | Same, retaining exact output layout |
| 8×64×6144 | 78 | 4.003 | Remove 240 empty output groups before tuning arithmetic |
| 8×512×6144 | 78 | 3.742 | Measure width and staged-input behavior |
| 8×4096×2048 | 99 | 2.872 | Measure matrix-kernel crossover |

These spans can overlap; their sum is **not** a wall-time speedup estimate. The trace has 106.742 ms wall span. Real serving also pays prefill interference and scheduling costs.

Start with the fixed-point transform already implemented by `blocked_gemv_cus`: preserve `ceil(N/groups)` while removing empty groups. Test all B1/2/4/8 rungs, inactive slots and placement, then run retrieval and paired serving measurements. Further caps alter ownership and need fresh numeric/performance qualification. No width or assembly change is adopted by this audit.

All four decode rungs are a single interpreter segment. They contain neither `GemmLtPf` nor `MoeAiterFp8Pf`; existing native hipBLASLt/AITER prefill improvements do not accelerate these decode instructions. The Hopper projection rewrite also explicitly rejects architectures other than `sm_90a`; enabling that knob cannot silently provide an AMD small-M path.

## Tiling coverage

The qualified manifest reports **0 measured / 2,046 tile lookups**, `tile_source=analytical`. Its emit log reports 4,823 stale tuning records against `gfx942-b9197aecac22cd3c`. This remains true even though earlier, differently built assets had measured tiles.

The evidence JSON lists every residual prefill projection shape and selected tile. Examples worth measuring after a prefill trace identifies critical-path cost:

| Residual shape | Manifest-reported tile | Instructions per program |
|---|---|---:|
| 8192×64×6144 | GemmSmall 64×128 | 78 |
| 8192×32×6144 | GemmSmall 64×128 | 21 |
| 8192×128×6144 | GemmSmall 64×128 | 21 |
| 2048×256×6144 | GemmSmall 64×128 | 75 |
| 2048/8192×6144×2048 | Gemm 256×256 | 78 |

The tally counts selector lookups, not necessarily final surviving opcodes after native-route substitution. Do not infer that hipBLASLt's internal choice is unmeasured from this tally. Re-run tuning under the exact compiler/object digest and verify emitted choices; measure smaller tiles or split-K only with matching LDS/register/arm contracts. Manifest `occupancy` is tile-wave utilization, not hardware residency, and its `waste_bytes` is an estimate rather than measured HBM traffic.

There is also a concrete geometry-reporting mismatch: `gemm_tile_of` reads the `GFX950_RUNGS` table even for gfx942, reporting ordinary `Gemm` as 256×256. Reading ELF `plow_geom_GM_BM/BN` gives **192×256** in the qualified ordinary prefill object and **64×128** in the flash object. Ordinary `exec_gemm` uses the compiled defaults. Bind audit rows to the actual selected object before interpreting their tile count/occupancy; the table above deliberately reports what the manifest says. The smaller `GemmSmall` markers match 64×128. All four discovered low-rung decoder objects have the expected MM=1/2/4/8; the root MM=16 object alone would give a misleading ceiling audit.

## What Lean actually checks

The devblob hook in [`plowc/main.rs`](../../crates/plowc/src/main.rs) submits GQ waits, successors, thresholds and queue ordering to checkpoint D. It submits **empty task-graph data edges and an empty address map**. Thus this path certifies ordering of supplied counters, not independent completeness of tensor dependencies or allocation alias safety. Checkpoint G checks selected always-staged GEMV demands against a 15,360-half arena. `lean.verified=true` is not all A–G checkpoints over concrete kernel semantics.

Neither payload includes cooperative barrier participants, generation state, CU admission, compiled register occupancy, or internal kernel read/write ranges. [`GlmTileIR.glm_deadlock_free`](../../lean-plow/Plow/GlmTileIR.lean) models a chain of single-workgroup stages. It does not establish progress of the multi-workgroup radix selector. The previously rejected parallel selector passed packet verification and standalone correctness but stalled in serving; [failure evidence](../../runtime/bench/amd/dsa_select_parallel/integration-rejected.json) remains the authoritative integration result.

To qualify another cooperative selector, model/check logical participant coverage, scratch disjointness and generation reuse **plus** actual object residency/queue admission. A DAG certificate alone cannot rule out its observed stall. This audit does not identify the stall's exact cause.

## Oracle limitations found by replay

The CLI exposes only `counter_granularity` and `lower_bound`; it has no tile-search query. TilePartition proves arithmetic coverage under a positive regular partition, not that assembly implements that partition or wins on hardware.

Replayed `lower_bound` requests with durations `[3,5]` returned:

| Edges | Result |
|---|---|
| `[[0,1]]` | `ok=true`, critical path 8 |
| `[[0,1],[1,0]]` | `ok=true`, critical path 19 |
| `[[0,99]]` | `ok=true`, critical path 5 |

The oracle does not reject cycles or out-of-range edges before its bounded relaxation. Its certificate string therefore must not be read as validation of arbitrary input graphs. The Rust wrapper also discards the envelope certificate while parsing `answer`; that explains `certified=false` in the devblob log despite a top-level CLI certificate string. Attaching that string alone would not strengthen the proof boundary.

For these GLM assets the oracle uses unit instruction durations, zero FLOPs, only the final decode rung, deduplicated non-KV tensor declarations, and no indirect expert traffic model. Its reported 2.778 ms bandwidth floor cannot identify the real 106.742 ms trace bottleneck or predict 70k/C20 performance. Use the measured instruction/shape table above to select experiments; improve oracle input validation and cost extraction before using it to stop an optimization campaign.

The H200 target remains unmet. These are prioritized experiments and verification gaps, not new serving performance claims.

## Follow-up: disabled batched matrix-core GEMV

The existing `gemv_rows_mfma4` implementation in
[`op_gemm.h`](../../runtime/amd/op_gemm.h) is another concrete decode experiment.
[`build_gfx942.sh`](../../scripts/build_gfx942.sh) enables it with
`PLOW_GEMV_MFMA4=1`; it defaults off. It uses
`__builtin_amdgcn_mfma_f32_4x4x4bf16_1k` (LLVM disassembles it as
`v_mfma_f32_4x4x4_16b_bf16`) for ordinary BF16 GEMV (`norm != 1`) at compiled batch widths ≥2,
including the global-activation path used when the batch exceeds staged LDS.
It changes reduction order and cannot promise VALU-reference bit identity.

Inspection of the qualified B1/2/4/8 `interp_decode_gq.elf` objects finds **zero**
instances of the MFMA4 instruction family in each. This was rechecked with
the installed LLVM spelling and `plow_geom_GV_MFMA4=0` ELF markers; the original
builtin-style mnemonic search alone was insufficient. Their recorded build
defines omit `GV_MFMA4`, consistent with the default-off source. This establishes a path
absent from the loaded assets; it does not establish its performance or model
quality on GLM. The [earlier Gemma measurements](../amd/gemma4-31b-mi300x.md)
also show that the best standalone unroll/column tile can lose inside the full
interpreter. Qualify real GLM operands, reduction error, routing, inactive rows,
object resources and serving behavior before adoption.

The ordinary-GEMV width experiment and pinned MFMA4 inspection are recorded in
[`mi300x-gemv-width.json`](../../runtime/bench/amd/glm_projection/mi300x-gemv-width.json).
That experiment uses unchanged GPU objects and does not enable MFMA4.

The subsequent [GLM MFMA qualification](../../runtime/bench/amd/glm_gemv_mfma/README.md)
passes 80 real-operand numerical cases and 18/18 retrieval checks. Restricting the
path to K≤2048 avoids primitive regressions at narrow K=6144 sites. One paired
serving screen improves throughput 1.82% and mean TPOT 1.49%, but worsens mean
TTFT 1.12% and P99 TPOT 5.81%. It remains opt-in; Lean ordering certificates do
not qualify its changed floating-point reduction order.

## Follow-up: ragged prefill loses token-blocked folding

The refreshed native-Lt profile finds a second missing fast path in actual
execution: `exec_mla_merge_fold` required all token rows to be divisible by
`PLOW_MLA_FOLD_TB`. A final 8,191-row chunk therefore sent every row through the
scalar fold. The [ragged-fold experiment](../../runtime/bench/amd/glm_fold_tail/README.md)
keeps complete eight-token groups on the existing blocked body and sends only
the remainder through the scalar body. This requires no new packet or Lean
ordering certificate. Numerical identity and compiled resource checks are the
relevant gates; the first implementation failed the scratch budget despite
passing primitive correctness.

The same trace exposes an attribution trap: native sparse attention bypasses
interpreter trace writes, leaving first-chunk `FlashMlaPrefill` records in the
buffer. Restricting timestamps to the final chunk removes 23,712 stale entries.
Only then does the trace describe that chunk's work. Two-shot all-reduce remains
a larger body aggregate than folding; these overlapping spans are priorities
for measurement, not additive performance forecasts.

For further library adaptation, AMD's [vLLM optimization guide](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)
describes AITER FP8 batched matmul for MLA and RCCL channel tuning. Those settings
are not controls for Plow's native HSA/custom-collective paths. Compare the actual
kernel/data-movement boundary before adopting them. In particular, the local
`PLOW_XR_MLP` record already rejects peer-batched scalar loads on MI300X; its
source-level appearance of extra parallelism is not evidence of a win.
