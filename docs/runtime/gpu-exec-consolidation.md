# GPU execution consolidation — 2026-09-10

Merged `perf/gemma31-prefix-h100` at `4f830ec6` into `tp-bringup-mi300x`.
The merge was textually clean. Review covered prefix-cache ownership, unified
batch admission, terminal sampling, backend dispatch and shared memory layouts.

## Findings fixed

- The C ISA enum omitted native GLM `IndexTpPf` (157) and `GemmLtPf` (158),
  although Rust already emitted them. Added the declarations; the opcode-count
  and ABI checks pass. Existing opcode values and structure layouts are unchanged.
- A config integration assertion still read `nv.pf_interleave` after the merge
  moved that setting to the shared runtime config. Updated the assertion and
  reran the integration test.

## Shared execution code

`exec/kv_layout.rs` now owns the KV tensor-name parser and the two-span
sliding-window calculation used by AMD prefix snapshots and CUDA KV/scale
copies. `memory::slab_pad` supplies the common weight-carving stride.

AMD retains fixed-window snapshot head strides and distinct allocation for
zero-byte tensors. CUDA retains compact short-prefix head strides, pitched KV
copies, FP8 scale copies and its zero-byte slab policy. Full-attention-only
CUDA snapshots skip absent sliding-ring geometry. The shared helpers add no
allocation, locking or dynamic dispatch.

The existing shared token-batch planner and staging remain the common row and
frontier contract. AMD HSA dispatch and CUDA VMM/stream ownership remain in
their adapters: these have different completion and memory-lifetime rules.

## Verification

- Combined CUDA/HSA runtime library: 666 passed, 29 ignored.
- Assets: 103 passed, 1 ignored. Packet library: 120 passed.
- C/Rust ABI and opcode integrations: 5 passed.
- CPU compact terminal: 4 passed. New emission integrations: 2 passed.
- Config integration: 1 passed. CMake token-batch configuration: 2 passed.
- Combined CUDA/HSA compile check and HSA release build passed.
- Merged GLM-5.3 TP8/B8 serving: 18/18 retrieval cases passed at concurrency 8;
  14/18 responses match the prior final runtime text exactly.
  [Runtime, packet, object and response evidence](../../runtime/bench/amd/glm_projection/mi300x-merge-review.json).
- Full compiler suite: 401 passed, 2 failed. The existing raw-environment-read
  guard fails; the Kimi whole-model assertion fails in the combined suite but
  passes in isolation. Neither is a demonstrated merge regression.
- H100 GPU tests were not executed on this MI300X host; mock/compile tests are
  not hardware qualification.

The GLM parallel-selection experiment is retained as
[benchmark and rejected integration evidence](../../runtime/bench/amd/dsa_select_parallel/README.md).
Its standalone result does not establish a serving gain; its production path
was removed after both global-queue and static serving stalled. The previous
GLM serving screen remains 33.40 output tokens/s versus the supplied H200
reference of 273.67; the workloads differ in request count and parity is unmet.

## Execution-module organization

A follow-up extraction separates cold validation and cache ownership from the
engine entry points:

| Module | Responsibility |
|---|---|
| `exec/amd.rs` | HSA engine ownership, loading and dispatch |
| `exec/amd_object.rs` | Object naming, opcode capabilities, geometry and packet pairing |
| `exec/gpu.rs` | CUDA engine ownership, loading and dispatch |
| `exec/gpu_prefix.rs` | KV mappings, prefix snapshots, attachment and publication |
| Backend `*_tests.rs` files | Existing test suites, preserving module/test names |

The existing public AMD object-selection functions/types remain re-exported.
CUDA prefix methods remain direct `GpuEngine` methods. No forwarding traits,
new allocation or locking were introduced. Stale module introductions claiming
single-sequence/decode-only support were replaced with current responsibilities.

AMD's engine file falls from 19,855 to 12,590 lines; CUDA's from 9,038 to 7,564.
These are organizational changes, not measured inference speedups.

Verification: 666 runtime tests passed, 29 ignored; the complete 695-entry test
inventory is identical. CUDA/HSA compile checks pass. An offline Rust AST
comparison covers all 602 original function/method definitions: signatures and
bodies match after normalizing formatting and string literal spelling. Module
visibility, imports and documentation are deliberately outside that comparison.

The [Lean/compiled-asset audit](glm53-lean-kernel-audit.md) identifies the next
performance experiments and the limits of the current verifier.
