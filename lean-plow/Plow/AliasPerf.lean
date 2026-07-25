/-
# Plow.AliasPerf — aliasing / streaming performance theorems (§B5–B8).

Proves four transformations the compiler applies to reduce memory traffic:

* **B5**: reshape/transpose elision — a pure-view op that produces
  byte-identical output can be elided.
* **B6**: concat sub-region aliasing — disjoint-byte-range sub-writes into
  a concat output move zero bytes.
* **B7**: in-place activation update — read-then-write into the same slot
  is safe when no other reader sits between.
* **B8**: streaming fusion — producer→consumer rowwise pipelining equals
  materializing the intermediate then re-reading.

Scope note: B6 is a real theorem (disjoint sub-writes into a shared buffer
commute, with a frame property). B5, B7 and B8 are *definitional
bookkeeping* — the byte-view model equates both sides by construction, so
those theorems record modeling decisions rather than prove transformations;
see the per-section honesty notes.
-/
import Plow.Basic

namespace Plow.AliasPerf

-- `ByteView`, `viewEq`, `viewEq_refl/symm/trans` come from `Plow.Basic`.

/-! ## B5 — Reshape / transpose alias elision.

    **Honesty note.** `reshape` is *defined* as the identity on byte
    views, so B5 is definitional bookkeeping: it records the modeling
    decision "a reshape is a pure view", it does not prove anything about
    an index-remapping reshape. A real proof would model strides/shapes
    and show the remap is a bijection on bytes. -/

/-- A reshape is a pure view — same underlying bytes, different logical
    shape. Modeled as identity on the byte view (modeling decision, not a
    derived fact). -/
def reshape (v : ByteView) : ByteView := v

/-- **B5** (definitional): eliding a reshape produces the same byte view.
    True by definition of the model above; carries no proof content. -/
theorem reshape_alias_elides (v : ByteView) : viewEq (reshape v) v :=
  fun _ => rfl

/-! ## B6 — Concat sub-region aliasing. -/

/-- Write `src` into `[base, base + bytes)` of an existing view `dst`,
    leaving all other bytes of `dst` intact. This is the actual concat
    write primitive: sub-writes land in a shared output buffer. -/
def writeInto (base bytes : Nat) (src dst : ByteView) : ByteView :=
  fun i => if base ≤ i ∧ i < base + bytes then src (i - base) else dst i

/-- Two write ranges are disjoint iff their `[base, base+bytes)` intervals
    don't overlap. -/
def rangesDisjoint (b1 s1 b2 s2 : Nat) : Prop :=
  b1 + s1 ≤ b2 ∨ b2 + s2 ≤ b1

/-- **B6**: disjoint-range sub-writes into a shared output commute — the
    concat result is independent of the order the parts are written in.
    Holds for arbitrary source bytes (zero bytes included). -/
theorem disjoint_writes_commute (b1 s1 b2 s2 : Nat)
    (src1 src2 dst : ByteView)
    (h : rangesDisjoint b1 s1 b2 s2) :
    viewEq (writeInto b2 s2 src2 (writeInto b1 s1 src1 dst))
           (writeInto b1 s1 src1 (writeInto b2 s2 src2 dst)) := by
  intro i
  unfold writeInto
  -- Case analysis on which range `i` falls into.
  by_cases h1 : b1 ≤ i ∧ i < b1 + s1
  · by_cases h2 : b2 ≤ i ∧ i < b2 + s2
    · -- Both ranges cover i ⇒ contradiction with disjointness.
      exfalso
      rcases h with hd | hd
      · omega
      · omega
    · -- Only range 1 covers i: both orders yield src1 (i - b1).
      simp [h1, h2]
  · by_cases h2 : b2 ≤ i ∧ i < b2 + s2
    · -- Only range 2 covers i: both orders yield src2 (i - b2).
      simp [h1, h2]
    · -- Neither range covers i: both orders yield dst i.
      simp [h1, h2]

/-- **B6** (byte accounting): a sub-write leaves every byte outside its
    range untouched — writing a concat part moves exactly its own `bytes`,
    zero bytes of the neighboring parts. -/
theorem writeInto_frame (base bytes : Nat) (src dst : ByteView)
    (i : Nat) (h_out : i < base ∨ base + bytes ≤ i) :
    writeInto base bytes src dst i = dst i := by
  unfold writeInto
  have : ¬ (base ≤ i ∧ i < base + bytes) := by omega
  simp [this]

/-! ## B7 — In-place activation update.

    **Honesty note.** Definitional bookkeeping. The byte-view model has no
    notion of *where* a view is stored, so "in-place" and "out-of-place"
    are the same term and B7 is `rfl`. The genuine safety condition — no
    other reader observes `x`'s slot between the read and the overwrite —
    lives in the scheduler's happens-before checks (`Plow.Memory`,
    verified per-schedule by `Plow/Verify.lean`), not here. -/

/-- A residual op: `out = f(x) + x`. Modeled: the byte range of `out`
    equals `x + delta` (delta ≡ f(x)). -/
def residualUpdate (x delta : ByteView) : ByteView :=
  fun i => x i + delta i

/-- **B7** (definitional): records that the in-place value written to `x`'s
    slot is the same value the out-of-place op computes. Carries no proof
    content; the aliasing-safety condition is checked in `Plow.Memory`. -/
theorem inPlace_equals_outOfPlace (x delta : ByteView) :
    viewEq (residualUpdate x delta) (fun i => x i + delta i) :=
  fun _ => rfl

/-! ## B8 — Streaming fusion.

    **Honesty note.** Definitional bookkeeping. Both sides are *defined* as
    `g (f x)` — the model cannot distinguish streaming from materializing,
    so B8 is `rfl`. What fusion actually buys (the eliminated HBM
    round-trip) is quantified for real in `Plow.FusionSavings`. -/

/-- Materialize-then-consume: produce a full intermediate byte view, then
    apply the consumer. -/
def materializeConsume (f g : ByteView → ByteView) (x : ByteView) : ByteView :=
  g (f x)

/-- Streaming: apply the consumer directly to the producer's output view. -/
def stream (f g : ByteView → ByteView) (x : ByteView) : ByteView :=
  g (f x)

/-- **B8** (definitional): streaming and materializing are the same term in
    this model; the equality carries no proof content. See
    `Plow.FusionSavings` for the non-trivial byte-savings statement. -/
theorem streaming_equals_materializing (f g : ByteView → ByteView) (x : ByteView) :
    viewEq (stream f g x) (materializeConsume f g x) :=
  fun _ => rfl

end Plow.AliasPerf
