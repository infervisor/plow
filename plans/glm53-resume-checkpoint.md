# GLM-5.3 resume checkpoint — 2026-09-23

## Objective

Original full zai-org/GLM-5.3, pinned revision aca966e4e02791568aa6a4ced368624b3d897f42;
vLLM Docker 0.29 vs Plow with identical precision, prompts and client metrics.
Context 8192 through 71680, concurrency 8 through 64. Target 3000 aggregate output
tokens/s and improvement across all requested serving metrics. This is incomplete.

## Source state

Main /home/lava/plow, branch worktree-gemma4-26b-beat-vllm, base
5a6d7a3e7ac118bf30249ec70c19ac3493208e95. Existing staged index digest
cce53d1bfa6e92fdc53a65e735538a8c26355a3eb98dbb22f9eb51f351870cbf.
Isolated Lean implementation: /home/lava/plow-lean-perf, branch
lean/performance-certificates. Its source includes a recorded inherited dirty
baseline; do not merge its whole snapshot as an implementation-only diff.

Checkpoint backup refs and incremental bundles are generated under
/opt/models/plow-resume-checkpoint-20260923. They preserve tracked working files,
untracked source under crates/runtime/scripts/docs/lean-plow and ignored plans.
They are archival snapshots of mixed existing work, not reviewed release commits.
Live branches and indexes remain in place. Generated binaries, external vendor
checkout, virtual environments, model weights and large captures stay in their
existing paths; frozen job artifacts remain authoritative for measurements.

## Newly closed gates

- Native selector job 1790149036822646308-f925d4c3: done, rc=0. All 68 cases pass
  exact top-k thresholds, range/count/uniqueness, short identity, inactive padding,
  input/guard preservation, stable native selected sets, and agreement with vLLM
  on all unambiguous rows. Native output order changes in 56 cases. Seven cases
  differ in selected sets at ambiguous boundaries: three repeated real-score
  shapes at capacity 131072 and four adversarial boundary-tie capacities.
  Artifact /opt/models/plow-glm53-indexer-selection-native-20260923/comparison.json,
  SHA256 3a5f235cadbf1140ec7243d262b27cce8a5cd7379b4b7bfa67aff538c40a6033.
  Object SHA256 823d24842f52cc9a81789d886339eb7f82a5ce3da8fe953b54c5c819a1ac12f0.
  Wrapper runtime/tests/indexer_select_gfx950.hip reuses d_index_select_pf<true>.
  No model dispatch promotion or attention qualification follows from this test.
- Lean worktree HSA job 1790149361490409752-e5b2d194: done, rc=0. One GPU test,
  8194 checked outputs; deferred and immediate publication each wrap 4096 packets.
  Capacity refusal with pending signal, stale ticket rejection, and upload refusal
  after error/quiescence pass. No GPU overhead measurement, TP8 or fault teardown
  qualification. Frozen artifacts /opt/models/plow-hsa-retirement-20260923;
  authoritative result /tmp/plow-gpuq/1790149361490409752-e5b2d194.log.
  Agent ended after submission; root inspected terminal status and actual results.

## Already qualified components

See plans/glm53-precision-parity.md for complete provenance. Decode headroom has
150 native passes; actual-live prefill has 84 native passes across 21 cases,
two modes and two poisons. Runtime opcode 192 handles native FP8 indexer decode;
193 handles ordinary single-slot prefill with live-dependent scoring. Model
emission is not yet wired. Default ragged chunking already patches selector and
union row count; arm_for_program rejects mismatched executed/live row counts.
BF16 native main attention is qualified on earlier M1..31 fixtures only.

Full original model: /opt/models/GLM-5.3-full-aca966e4 (141 shards).
Prepared TP8 model: /opt/models/GLM-5.3-full-aca966e4-plow-mla-fp8-tp8.
Separate official AMD MXFP4 download is complete and checksum/layout verified:
/opt/models/GLM-5.3-MXFP4-AttnFP8-amd-4992911b. See its preparation plan;
vLLM 0.29 scale-name loader compatibility remains unqualified.

## Resume in this order

1. Read this file, precision-parity plan and docs/bringup/agent-tools.md. Inspect
   current git status and queue jobs before starting anything. Both jobs above
   are terminal; do not resubmit them merely to recover context.
2. Extend runtime/tests/block_fp8_aiter_compare.py using existing
   export_attention_sweep metadata construction and actual installed forward_mqa.
   Replay both frozen reference and native selector index arrays into identical
   BF16 query/KV inputs. Preserve their original ordering; quantify effects of
   order changes separately from ambiguous tied sets. Hash all inputs and outputs.
   Start with decode active rows; explicitly cover inactive rows and M32/64
   before claiming production coverage. Prefill requires its actual backend.
3. Resolve selection/attention precision acceptance from evidence; then wire
   native FP8 indexer cache and opcodes into modular model emission and run the
   connected projection→RoPE→indexer→selection→attention block gate.
4. Close remaining attention rungs, prefill and full-model precision/retrieval
   gates. Run full model with the same Docker/client/prompt precision contract.
   Long context C64 needs a verified memory/DCP plan with BF16 main KV.
5. Measure four arms ctl/treat/ctl2/treat2; accept improvements only beyond both
   spreads. Optimize measured critical paths and expand serving matrix. Neither
   full serving parity nor 3000 tok/s has been established.
6. Lean worktree: record the GPU pass, resolve documented CSR/native timing
   compatibility refusals, review changes against inherited baseline, and measure
   TP8 overhead before integration. Keep semantic, structural and empirical
   certificates distinct. Runtime completion/ownership checks must stay cheap.

All builds run inside nix develop. Run the CPU doctor before leasing. Every GPU
process goes through scripts/bench/gpuq.py. Freeze binaries before submission.
Pinned Docker: vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1.
No automatic dirty-tree pull, reset, stash, index staging or Lean whole-tree merge.
