/-
# Plow.FusionSavings — DMA-count reduction theorems for op fusion.

For every fusion `F(g(x)) → Fused(F, g)(x)`, the fused kernel eliminates
one HBM round-trip of the intermediate `g(x)` — the producer's DmaOut and
the consumer's DmaIn are both dropped. Total HBM bytes saved =
`2 × sizeof(g_output)`.

This module certifies the DMA-count reduction for the Rewrite-side fusion
rules, so the compiler can rank fusion candidates by *provable* DMA savings
rather than an empirical heuristic.
-/
import Plow.Basic

namespace Plow.FusionSavings

/-- Per-op HBM traffic model: a producer writes `size` bytes to HBM; a
    consumer reads `size` bytes from HBM. -/
structure OpTraffic where
  /-- Weight bytes read (independent of fusion). -/
  weight : Nat
  /-- Activation output bytes written (0 for terminal consumers). -/
  output : Nat
  /-- Activation input bytes read (0 for graph inputs / fused predecessors). -/
  input  : Nat

/-- Total HBM bytes an unfused op does. -/
def opBytes (t : OpTraffic) : Nat := t.weight + t.output + t.input

/-- Two-op chain `producer → consumer`. Unfused: producer writes output +
    consumer reads producer's output as its input. -/
structure Chain where
  producer : OpTraffic
  consumer : OpTraffic
  /-- Byte size of the shared intermediate tensor (producer.output ==
      consumer.input in the unfused case). -/
  intermediate : Nat
  /-- Invariant: the producer's output = the consumer's input = intermediate. -/
  h_shared_p : producer.output = intermediate
  h_shared_c : consumer.input = intermediate

/-- Unfused HBM bytes for the chain. -/
def unfusedBytes (c : Chain) : Nat := opBytes c.producer + opBytes c.consumer

/-- Fused HBM bytes: the intermediate never touches HBM.
    Producer's weight + no-output-write + consumer's weight + no-input-read. -/
def fusedBytes (c : Chain) : Nat :=
  c.producer.weight + c.consumer.weight
    -- The output of the fused kernel is whatever the consumer would have
    -- written; the input is whatever the producer would have read.
    + c.producer.input + c.consumer.output

/-! ## F1 — Fusion saves 2× intermediate bytes.

    The intermediate tensor is written by producer (`+intermediate`) and
    read by consumer (`+intermediate`); fusion eliminates both. -/

/-- **F1**: fusing a producer→consumer chain reduces HBM bytes by exactly
    `2 × intermediate` (one write + one read eliminated). -/
theorem fusion_saves_intermediate_bytes (c : Chain) :
    unfusedBytes c = fusedBytes c + 2 * c.intermediate := by
  unfold unfusedBytes fusedBytes opBytes
  rw [c.h_shared_p, c.h_shared_c]
  omega

/-- **F1** corollary: `unfusedBytes ≥ fusedBytes`, so fusion never
    increases HBM traffic. -/
theorem fused_bytes_le_unfused (c : Chain) :
    fusedBytes c ≤ unfusedBytes c := by
  rw [fusion_saves_intermediate_bytes c]
  omega

/-! ## F2 — Ranking fusion candidates by intermediate size. -/

/-- The HBM bytes a chain's fusion actually saves. -/
def savedBytes (c : Chain) : Nat := unfusedBytes c - fusedBytes c

/-- The savings equal `2 × intermediate` exactly (write + read eliminated).
    Follows from **F1**; the subtraction is exact, not clamped. -/
theorem savedBytes_eq (c : Chain) :
    savedBytes c = 2 * c.intermediate := by
  unfold savedBytes
  rw [fusion_saves_intermediate_bytes c]
  omega

/-- **F2**: fusion savings — derived from each chain's actual traffic via
    F1, not assumed — are monotone in intermediate size. A candidate with a
    larger intermediate tensor saves more HBM bytes. Certifies the
    extractor's heuristic "fuse biggest intermediates first". -/
theorem savings_monotone_in_intermediate
    (c1 c2 : Chain)
    (h_bigger : c2.intermediate ≥ c1.intermediate) :
    savedBytes c1 ≤ savedBytes c2 := by
  rw [savedBytes_eq, savedBytes_eq]
  omega

/-! ## F3 — Attention-o_proj savings.

    The specific case that motivated this module: attention's output is
    `[batch, seq, heads·head_dim]`. On Gemma 4 31B with batch=1, seq=512,
    heads=16, head_dim=256, elem_bytes=2: intermediate = 4 MB.
    Fusion saves 8 MB of HBM bytes per attention op. -/

/-- **F3** (corollary of F1 by substitution — no proof content beyond F1;
    recorded to pin the concrete attention-shape instantiation the ranking
    pass evaluates): for an attention chain whose intermediate is the
    `[batch, seq, heads·head_dim]` output tensor, fusion savings equal
    twice that byte count. -/
theorem attention_out_savings (batch seq heads head_dim elem : Nat)
    (c : Chain)
    (h_int : c.intermediate = batch * seq * heads * head_dim * elem) :
    unfusedBytes c - fusedBytes c = 2 * (batch * seq * heads * head_dim * elem) := by
  rw [fusion_saves_intermediate_bytes c, h_int]
  omega

/-! ## F4 — Split-KV / FlashDecode merge-fold savings.

    In split-KV attention with `splits` partitions across the sequence length,
    the unfused kernel writes `splits` partial outputs and logsumexp statistics
    to HBM, which `FlashMerge` then reads from HBM before writing the merged
    output.

    Folding the merge into the attention kernel (using in-SRAM / atomic
    reduction across splits) eliminates the HBM round-trip of the partial
    accumulators for all splits.
-/

/-- Multi-producer reduction chain: `splits` partial tiles feed one merge op.
    `part_bytes` is the byte footprint of one split's partial output + stats. -/
structure SplitKvChain where
  splits : Nat
  part_bytes : Nat
  out_bytes : Nat
  h_splits_pos : splits ≥ 1

/-- Unfused split-KV HBM bytes:
    - Each split writes its partial accumulators to HBM (`splits * part_bytes`)
    - The merge op reads all partial accumulators from HBM (`splits * part_bytes`)
    - The merge op writes the final output tensor (`out_bytes`) -/
def unfusedSplitKvBytes (c : SplitKvChain) : Nat :=
  2 * c.splits * c.part_bytes + c.out_bytes

/-- Fused split-KV HBM bytes:
    - Merged output is written directly to HBM without intermediate HBM writes/reads (`out_bytes`). -/
def fusedSplitKvBytes (c : SplitKvChain) : Nat :=
  c.out_bytes

/-- **F4**: Folding split-KV merge eliminates all partial accumulator round-trips
    to HBM, saving exactly `2 * splits * part_bytes` HBM traffic. -/
theorem split_kv_merge_fold_savings (c : SplitKvChain) :
    unfusedSplitKvBytes c = fusedSplitKvBytes c + 2 * c.splits * c.part_bytes := by
  unfold unfusedSplitKvBytes fusedSplitKvBytes
  omega

/-- Saved HBM bytes by folding split-KV merge. -/
def savedSplitKvBytes (c : SplitKvChain) : Nat :=
  unfusedSplitKvBytes c - fusedSplitKvBytes c

theorem savedSplitKvBytes_eq (c : SplitKvChain) :
    savedSplitKvBytes c = 2 * c.splits * c.part_bytes := by
  unfold savedSplitKvBytes
  rw [split_kv_merge_fold_savings c]
  omega

/-- Monotonicity of merge-fold savings: as split count increases, the savings
    scale linearly. -/
theorem split_kv_savings_monotone (c1 c2 : SplitKvChain)
    (h_bytes : c1.part_bytes = c2.part_bytes)
    (h_splits : c1.splits ≤ c2.splits) :
    savedSplitKvBytes c1 ≤ savedSplitKvBytes c2 := by
  rw [savedSplitKvBytes_eq, savedSplitKvBytes_eq, h_bytes]
  have h : 2 * c1.splits ≤ 2 * c2.splits := by omega
  exact Nat.mul_le_mul_right c2.part_bytes h

/-! ## F5 — MoE Combine Epilogue Fold savings.

    In Mixture of Experts (MoE) down projection, `k` experts each produce a
    partial down-projection tensor of shape `[tokens, hidden]`.
    Unfused:
    - `MoeDown` writes `tokens * k * hidden * 4` bytes of float32 partials.
    - `MoeCombine` reads `tokens * k * hidden * 4` bytes of float32 partials.
    - `MoeCombine` writes `tokens * hidden * 2` bytes of bf16 combined output (`moe`).
    - `AddNorm` reads `tokens * hidden * 2` bytes of `moe`, reads residual `x`,
      and writes updated residual and normalized output.

    Epilogue folding `MoeCombine` into `AddNorm` eliminates the intermediate
    tensor `moe`:
    - `AddNorm` accumulates the `k` partials on the fly in registers.
    - The intermediate tensor `moe` is neither written nor read from HBM.
    - HBM savings = `2 × sizeof(moe) = 4 * tokens * hidden` bytes.
    - Eliminates one kernel dispatch, one claim loop, and one global barrier.
-/

structure MoeCombineFoldChain where
  tokens : Nat
  hidden : Nat
  k : Nat
  elem_bytes : Nat
  h_k_pos : k ≥ 1

/-- Intermediate `moe` tensor footprint in bytes. -/
def moeIntermediateBytes (c : MoeCombineFoldChain) : Nat :=
  c.tokens * c.hidden * c.elem_bytes

/-- Unfused traffic for the combine + residual norm phase:
    `MoeCombine` writes `moe` and `AddNorm` reads `moe` (`2 * moeIntermediateBytes`),
    in addition to partials input and residual I/O. -/
def unfusedMoeCombineBytes (c : MoeCombineFoldChain) (part_bytes resid_io : Nat) : Nat :=
  part_bytes + 2 * moeIntermediateBytes c + resid_io

/-- Fused traffic: `moe` is accumulated in registers and never written/read to/from HBM. -/
def fusedMoeCombineBytes (_ : MoeCombineFoldChain) (part_bytes resid_io : Nat) : Nat :=
  part_bytes + resid_io

/-- **F5**: Epilogue folding `MoeCombine` into `AddNorm` saves exactly
    `2 * sizeof(moe) = 2 * tokens * hidden * elem_bytes` of HBM traffic. -/
theorem moe_combine_fold_savings (c : MoeCombineFoldChain) (part_bytes resid_io : Nat) :
    unfusedMoeCombineBytes c part_bytes resid_io =
    fusedMoeCombineBytes c part_bytes resid_io + 2 * moeIntermediateBytes c := by
  unfold unfusedMoeCombineBytes fusedMoeCombineBytes
  omega

/-- Saved bytes from MoE combine fold. -/
def savedMoeCombineBytes (c : MoeCombineFoldChain) : Nat :=
  2 * moeIntermediateBytes c

theorem savedMoeCombineBytes_eq (c : MoeCombineFoldChain) (part_bytes resid_io : Nat) :
    unfusedMoeCombineBytes c part_bytes resid_io - fusedMoeCombineBytes c part_bytes resid_io =
    savedMoeCombineBytes c := by
  unfold savedMoeCombineBytes
  rw [moe_combine_fold_savings c part_bytes resid_io]
  omega

/-! ## F6 — MoE Router GEMV + TopK Fusion savings.

    In Mixture of Experts routing:
    Unfused:
    - Router GEMV (`Gemv: mlp.gate.weight`) reads activation `x` ([tokens, hidden]),
      reads router weights `wr` ([experts, hidden]), and writes `rlogit`
      ([tokens, experts]) to HBM.
    - Global barrier / claim loop (1 instruction dispatch).
    - `MoeRouterTopkPf` reads `rlogit` from HBM, performs activation/softmax and top-k,
      and writes `topk_ids` and `topk_weights` to HBM.
    - Total router instructions = 2, total barriers = 2.
    - Intermediate HBM roundtrip: `rlogit` is written then read: `2 × tokens × experts × elem_bytes`.

    Fused (`DevOp::MoeRouter`):
    - Computes GEMV in registers, performs in-place softmax and top-k selection
      across warp lanes without writing or reading `rlogit` to/from HBM.
    - Total router instructions = 1, total barriers = 1.
    - Intermediate HBM savings = `2 × tokens × experts × elem_bytes`.
    - Barrier / dispatch reduction = 1 instruction and 1 barrier per MoE layer pass.
-/

structure MoeRouterFoldChain where
  tokens : Nat
  hidden : Nat
  experts : Nat
  k : Nat
  elem_bytes : Nat
  h_k_le_experts : k ≤ experts
  h_k_pos : k ≥ 1

/-- Intermediate `rlogit` tensor footprint in bytes. -/
def rlogitIntermediateBytes (c : MoeRouterFoldChain) : Nat :=
  c.tokens * c.experts * c.elem_bytes

/-- Unfused traffic for router GEMV + Top-k:
    input activation and weight reads, output `topk` structures, plus
    `rlogit` written by GEMV and read by Top-k (`2 * rlogitIntermediateBytes`). -/
def unfusedMoeRouterBytes (c : MoeRouterFoldChain) (weight_and_io : Nat) : Nat :=
  weight_and_io + 2 * rlogitIntermediateBytes c

/-- Fused traffic: `rlogit` is maintained in registers and never touches HBM. -/
def fusedMoeRouterBytes (_ : MoeRouterFoldChain) (weight_and_io : Nat) : Nat :=
  weight_and_io

/-- **F6**: Fusing router GEMV into Top-K saves exactly
    `2 * sizeof(rlogit) = 2 * tokens * experts * elem_bytes` of HBM traffic. -/
theorem moe_router_fold_savings (c : MoeRouterFoldChain) (weight_and_io : Nat) :
    unfusedMoeRouterBytes c weight_and_io =
    fusedMoeRouterBytes c weight_and_io + 2 * rlogitIntermediateBytes c := by
  unfold unfusedMoeRouterBytes fusedMoeRouterBytes
  omega

/-- Saved bytes from MoE router fold. -/
def savedMoeRouterBytes (c : MoeRouterFoldChain) : Nat :=
  2 * rlogitIntermediateBytes c

theorem savedMoeRouterBytes_eq (c : MoeRouterFoldChain) (weight_and_io : Nat) :
    unfusedMoeRouterBytes c weight_and_io - fusedMoeRouterBytes c weight_and_io =
    savedMoeRouterBytes c := by
  unfold savedMoeRouterBytes
  rw [moe_router_fold_savings c weight_and_io]
  omega

/-- Barrier and instruction reduction:
    Unfused routing requires 2 instructions per layer pass (GEMV + TopK).
    Fused routing requires 1 instruction (`MoeRouter`).
    Across `L` MoE layer passes, exactly `L` instructions and `L` global barriers are eliminated. -/
def unfusedRouterInstructions (layers : Nat) : Nat := 2 * layers
def fusedRouterInstructions (layers : Nat) : Nat := layers

theorem router_fusion_instruction_reduction (layers : Nat) :
    unfusedRouterInstructions layers = fusedRouterInstructions layers + layers := by
  unfold unfusedRouterInstructions fusedRouterInstructions
  omega

theorem router_fusion_eliminates_layers_barriers (layers : Nat) :
    unfusedRouterInstructions layers - fusedRouterInstructions layers = layers := by
  rw [router_fusion_instruction_reduction layers]
  omega

end Plow.FusionSavings
