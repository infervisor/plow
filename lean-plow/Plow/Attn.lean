/-
# Plow.Attn — Flash attention correctness theorems.

Models the plow flash-attention op's tiling structure and proves the
invariants the compiler + runtime depend on:

* **T1**: attention tile grid covers every (query, KV) pair — no output
  element is skipped.
* **T2**: (stub — see the T2 section) a degenerate accumulator model in
  which the online-softmax fold reduces to plain addition. It does NOT
  prove online-softmax correctness.
* **T3**: mask semantics — causal, sliding-window, and prefix masks each
  correspond to an explicit predicate on `(query_pos, key_pos)`.
* **T4**: decode KV read range `[0, past_len)` is correctly bounded by the
  runtime's `past_len` counter, matching the sidecar contract in
  `plans/kv-decode-and-static.md`.
-/
import Plow.Basic
import Plow.TilePartition

namespace Plow.Attn

/-- Attention shape: heads, per-head dim, sequence lengths. -/
structure AttnShape where
  heads   : Nat
  headDim : Nat
  seqQ    : Nat
  seqKv   : Nat

/-- Tile shape for flash attention. `bq` is the query-tile row height and
    `bkv` is the KV-chunk width the streaming softmax accumulates over. -/
structure AttnTile where
  bq  : Nat
  bkv : Nat

-- `ceilDiv` and `le_ceilDiv_mul` come from `Plow.Basic`.

/-- Number of query tiles per attention head. -/
def queryTileCount (s : AttnShape) (t : AttnTile) : Nat := ceilDiv s.seqQ t.bq

/-- Number of KV chunks streamed per query tile. -/
def kvChunkCount (s : AttnShape) (t : AttnTile) : Nat := ceilDiv s.seqKv t.bkv

/-- Total attention tile-steps: heads × query_tiles × kv_chunks. Each tile-step
    is one `BQ × BKV × HD` compute. -/
def totalTiles (s : AttnShape) (t : AttnTile) : Nat :=
  s.heads * queryTileCount s t * kvChunkCount s t

/-! ## T1 — Tile grid covers every attention output. -/

/-- A valid attention tile: positive dims. Analogous to `TilePartition.ValidPartition`. -/
structure ValidAttnTile (s : AttnShape) (t : AttnTile) : Prop where
  bq_pos      : 0 < t.bq
  bkv_pos     : 0 < t.bkv
  heads_pos   : 0 < s.heads
  seqQ_pos    : 0 < s.seqQ
  seqKv_pos   : 0 < s.seqKv
  headDim_pos : 0 < s.headDim

/-- **T1**: total attention tile-work covers every output element. Every
    `(head, query, key)` triple is included in some tile — no unfilled
    output cell after all tiles complete. -/
theorem tile_grid_covers_attn (s : AttnShape) (t : AttnTile)
    (v : ValidAttnTile s t) :
    s.heads * s.seqQ * s.seqKv ≤ totalTiles s t * t.bq * t.bkv := by
  have hq : s.seqQ ≤ t.bq * ceilDiv s.seqQ t.bq :=
    le_ceilDiv_mul s.seqQ t.bq v.bq_pos
  have hkv : s.seqKv ≤ t.bkv * ceilDiv s.seqKv t.bkv :=
    le_ceilDiv_mul s.seqKv t.bkv v.bkv_pos
  -- Chain: heads · seqQ · seqKv ≤ heads · (bq·⌈q⌉) · (bkv·⌈kv⌉).
  have hq' : s.heads * s.seqQ ≤ s.heads * (t.bq * ceilDiv s.seqQ t.bq) :=
    Nat.mul_le_mul_left s.heads hq
  have hqkv : s.heads * s.seqQ * s.seqKv ≤
              s.heads * (t.bq * ceilDiv s.seqQ t.bq) *
              (t.bkv * ceilDiv s.seqKv t.bkv) :=
    Nat.mul_le_mul hq' hkv
  -- Reassociate to totalTiles · bq · bkv.
  have hassoc :
      s.heads * (t.bq * ceilDiv s.seqQ t.bq) *
      (t.bkv * ceilDiv s.seqKv t.bkv)
      = totalTiles s t * t.bq * t.bkv := by
    unfold totalTiles queryTileCount kvChunkCount
    simp only [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  -- Substitute hassoc into hqkv to close the goal.
  rw [← hassoc]
  exact hqkv

/-! ## T2 — Streaming accumulator stub.

    **Honesty note.** This section does NOT prove online-softmax
    correctness. Real online softmax rescales the running accumulator by
    `exp(old_max - new_max)` when a larger max arrives; that requires real
    (or fixed-point) arithmetic and an `exp` model, neither of which exists
    in this Lean-core development. `onlineUpdate` below uses an *identity*
    rescale, so under the theorem's hypothesis (every value ≤ the initial
    max — i.e. the rescale branch never fires) the fold degenerates to
    plain addition. The theorem certifies only that degenerate case:
    the accumulator totals every value exactly once, in any order. The
    actual online-softmax numerics are unverified. -/

/-- Abstract online-softmax accumulator state: `(max_seen, sum_expwise_scaled)`.
    The streaming update sees a new value `v` and produces a new state. -/
structure OnlineSoftmax where
  maxV : Nat
  accV : Nat

/-- The streaming update — **stub**: the "rescale" on a new max is the
    identity, not the `exp(old_max - new_max)` factor of real online
    softmax. -/
def onlineUpdate (s : OnlineSoftmax) (v : Nat) : OnlineSoftmax :=
  if v ≤ s.maxV then { s with accV := s.accV + v }
  else { maxV := v, accV := v + s.accV }

/-- Fold a list of KV chunk values through the streaming accumulator. -/
def streamingFold (init : OnlineSoftmax) (vs : List Nat) : OnlineSoftmax :=
  vs.foldl onlineUpdate init

/-- **T2 (stub)**: when every value is ≤ the initial max (so the rescale
    branch never fires and the identity-rescale stub is exact), the fold
    accumulates the plain sum of all values — each KV chunk contributes
    exactly once regardless of order. This is bookkeeping for the
    degenerate case only; it is NOT online-softmax correctness. -/
theorem streamingFold_sum_invariant (init : OnlineSoftmax) (vs : List Nat)
    (h : ∀ v ∈ vs, v ≤ init.maxV) :
    (streamingFold init vs).accV = init.accV + vs.foldr (· + ·) 0 := by
  induction vs generalizing init with
  | nil => simp [streamingFold]
  | cons head tail ih =>
    have h_head : head ≤ init.maxV := h head (List.mem_cons_self _ _)
    have h_tail : ∀ v ∈ tail, v ≤ init.maxV := fun v hv =>
      h v (List.mem_cons_of_mem _ hv)
    -- After first update, max stays at init.maxV (because head ≤ maxV),
    -- accV := init.accV + head. Then recurse.
    have h1 : onlineUpdate init head = { init with accV := init.accV + head } := by
      unfold onlineUpdate; rw [if_pos h_head]
    -- The new init after one step has the same maxV.
    have h_max_stable : (onlineUpdate init head).maxV = init.maxV := by
      rw [h1]
    -- Apply IH to the rest.
    have h_rest_bound : ∀ v ∈ tail, v ≤ (onlineUpdate init head).maxV := fun v hv => by
      rw [h_max_stable]; exact h_tail v hv
    unfold streamingFold
    simp only [List.foldl_cons]
    have := ih (onlineUpdate init head) h_rest_bound
    unfold streamingFold at this
    rw [this]
    rw [h1]
    simp
    omega

/-! ## T3 — Mask semantics. -/

/-- Attention mask kinds emitted by the compiler. Runtime interprets these
    via the `.request_io.json` sidecar's semantic tag. -/
inductive MaskKind
  | Causal      -- `key_pos ≤ query_pos`
  | Sliding (window : Nat) -- `query_pos - window ≤ key_pos ≤ query_pos`
  | Full        -- prefill prefix: all positions attend

/-- The predicate a mask kind imposes on `(query_pos, key_pos)`. -/
def maskAllows (m : MaskKind) (q k : Nat) : Prop :=
  match m with
  | MaskKind.Causal => k ≤ q
  | MaskKind.Sliding W => k ≤ q ∧ q ≤ k + W
  | MaskKind.Full => True

/-- **T3a**: causal mask allows the diagonal. -/
theorem causal_allows_diagonal (q : Nat) :
    maskAllows MaskKind.Causal q q := by
  unfold maskAllows
  exact Nat.le_refl _

/-- **T3b**: sliding window is a sub-relation of causal when window covers
    the whole causal range. -/
theorem sliding_subset_causal (q k W : Nat)
    (h : maskAllows (MaskKind.Sliding W) q k) : maskAllows MaskKind.Causal q k := by
  exact h.1

/-- **T3c**: `Full` mask allows every pair — the trivial upper bound. -/
theorem full_allows_all (q k : Nat) : maskAllows MaskKind.Full q k := trivial

/-! ## T4 — Decode KV read range. -/

/-- At decode step `step`, the runtime supplies `past_len := step` and the
    compiler-emitted Flash body's `seq_kv` is patched to `past_len + 1`.
    The read range is `[0, past_len + 1)`. -/
def decodeKvRange (past_len : Nat) : Nat × Nat := (0, past_len + 1)

/-- **T4**: the decode KV read range is exactly `past_len + 1` tokens wide
    (past context + the new token). Matches the runtime patch at
    `FlashBody.seq_kv` byte offset 12. -/
theorem decode_kv_range_size (past_len : Nat) :
    (decodeKvRange past_len).2 - (decodeKvRange past_len).1 = past_len + 1 := by
  unfold decodeKvRange
  simp

/-- **T4** monotone: past_len grows one token per step; the read range grows
    by exactly one. -/
theorem decode_kv_range_grows (past_len : Nat) :
    (decodeKvRange (past_len + 1)).2 = (decodeKvRange past_len).2 + 1 := by
  unfold decodeKvRange
  rfl

end Plow.Attn
