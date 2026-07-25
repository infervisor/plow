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

end Plow.FusionSavings
