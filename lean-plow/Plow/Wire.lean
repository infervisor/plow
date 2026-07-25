/-
# Plow.Wire — packet wire-format round-trip (§5.10-E).

Models the shape-level property that the plow packet emitter must satisfy:
the serialized byte stream `decode` inverts `encode` on every well-formed
program.

## Scope

**Abstract, not byte-identical.** The actual `packet::Program::to_bytes` in
`crates/packet` carries a `MAGIC` (u32) + `VERSION` (u16) header, per-body
POD structs with opcode-specific fields, per-instruction u16 wait/succ length
prefixes, etc. This module does **not** re-derive that whole layout — it
would be a large undertaking with little proof value, since `packet` has its
own encode/decode round-trip test suite in Rust.

What this module *does* prove is the underlying framing invariant every
framing scheme (including packet's) must satisfy:

    ∀ well-formed program p, decode (encode p) = some p.

The Lean model uses a simplified frame layout — u16 opcode + u16 payload_len
+ payload bytes — that shares the "sized-length-prefixed opcode-tagged
frame" shape with `packet::Inst`. The checkpoint-E CLI submits pairs
`(raw, frames)` in this abstract shape; Rust-side integration tests
(`crates/plowc/tests/lean_verify_wire.rs`, `crates/packet/tests/*`)
independently exercise the concrete `packet::Program::to_bytes` /
`Program::decode` pair. Together they close the loop: Lean proves the
framing shape is bijective on well-formed input; Rust proves the concrete
packet encoder satisfies that framing shape.

## Wire format modeled here

    Frame := u16 opcode | u16 payload_len | payload : Bytes[payload_len]
    Program := Frame*

A checkpoint-E request carries `(raw_bytes, decoded_frames)` and the checker
returns success iff both directions round-trip. The universal theorem below
means either direction suffices (since encode/decode is a bijection on
well-formed input), but the CLI checks both to catch schema drift on either
side of the bridge.
-/
import Plow.Basic

namespace Plow.Wire

open Plow

/-- A single wire frame: a 16-bit opcode tag and a variable-length payload.
    Payload length is encoded as a leading Nat (u16 in the on-wire form). -/
structure Frame where
  opcode  : Nat        -- 0..65535
  payload : List Nat   -- each byte 0..255
  deriving Repr, DecidableEq

/-- A packet program is a list of frames. -/
abbrev Program : Type := List Frame

/-! ## Encoding — deterministic, bijective on well-formed input. -/

/-- Encode a single `Nat` as two big-endian bytes. Values > 65535 wrap. -/
def encodeU16 (n : Nat) : List Nat := [n / 256 % 256, n % 256]

/-- Encode a single frame: opcode (u16) + payload_len (u16) + payload bytes. -/
def encodeFrame (f : Frame) : List Nat :=
  encodeU16 f.opcode ++ encodeU16 f.payload.length ++ f.payload

/-- Encode a full program by concatenating frame encodings. -/
def encodeProgram (p : Program) : List Nat :=
  p.foldr (fun f acc => encodeFrame f ++ acc) []

/-! ## Decoding — inverse of encode on well-formed input. -/

/-- Read two bytes off the front, big-endian to Nat. Returns the value +
    remaining tail. Errors when < 2 bytes are available. -/
def decodeU16 : List Nat → Option (Nat × List Nat)
  | hi :: lo :: rest => some (hi * 256 + lo, rest)
  | _ => none

/-- Split the first `n` items off a list. Errors when the list is shorter. -/
def take? : Nat → List Nat → Option (List Nat × List Nat)
  | 0,     rest        => some ([], rest)
  | _ + 1, []          => none
  | k + 1, x :: rest   =>
    match take? k rest with
    | some (taken, tail) => some (x :: taken, tail)
    | none => none

/-- Decode one frame off the front, returning the frame + remaining bytes. -/
def decodeFrame (bytes : List Nat) : Option (Frame × List Nat) := do
  let (op, r1) ← decodeU16 bytes
  let (len, r2) ← decodeU16 r1
  let (payload, tail) ← take? len r2
  some ({ opcode := op, payload := payload }, tail)

/-- Decode a full program, using `fuel` steps as an upper bound (equal to the
    frame count of the well-formed program). Returns `some p` iff every byte
    decodes cleanly and no trailing bytes remain. -/
def decodeProgramAux : Nat → List Nat → Option Program
  | 0,       []     => some []
  | 0,       _::_   => none
  | _+1,     []     => some []
  | fuel+1,  b::bs  => do
    let (frame, rest) ← decodeFrame (b :: bs)
    let more ← decodeProgramAux fuel rest
    some (frame :: more)

def decodeProgram (bytes : List Nat) : Option Program :=
  decodeProgramAux bytes.length bytes

/-! ## Well-formed frames.

    A frame is well-formed when the opcode fits in u16, every payload byte
    fits in u8, and the payload length itself fits in u16 (so it can be
    round-tripped through the leading `encodeU16 f.payload.length`). -/

def WellFormedFrame (f : Frame) : Prop :=
  f.opcode < 65536 ∧ f.payload.length < 65536 ∧ ∀ b ∈ f.payload, b < 256

def WellFormed (p : Program) : Prop :=
  ∀ f ∈ p, WellFormedFrame f

/-! ## Round-trip lemmas. -/

/-- `decodeU16` inverts `encodeU16` on values < 65536. -/
theorem decodeU16_encodeU16 (n : Nat) (h : n < 65536) (rest : List Nat) :
    decodeU16 (encodeU16 n ++ rest) = some (n, rest) := by
  unfold encodeU16 decodeU16
  simp
  omega

/-- `take?` peels back the exact prefix `encodeFrame` appended. -/
theorem take?_append (xs rest : List Nat) :
    take? xs.length (xs ++ rest) = some (xs, rest) := by
  induction xs with
  | nil => simp [take?]
  | cons x xs ih =>
    simp [take?, List.length_cons]
    rw [ih]

/-- `decodeFrame` inverts `encodeFrame` on a well-formed frame. -/
theorem decodeFrame_encodeFrame (f : Frame) (wf : WellFormedFrame f)
    (rest : List Nat) :
    decodeFrame (encodeFrame f ++ rest) = some (f, rest) := by
  obtain ⟨h_op, h_len, _h_payload⟩ := wf
  unfold encodeFrame decodeFrame
  simp only [List.append_assoc]
  rw [decodeU16_encodeU16 f.opcode h_op]
  simp
  rw [decodeU16_encodeU16 f.payload.length h_len]
  simp
  rw [take?_append f.payload rest]
  cases f
  simp

/-! ## Program-level round-trip. -/

/-- Encoding a program lays exactly `Σ (framecount contributions)` bytes; on a
    non-empty well-formed program, the encoded stream is non-empty. -/
theorem encodeProgram_cons (f : Frame) (p : Program) :
    encodeProgram (f :: p) = encodeFrame f ++ encodeProgram p := by
  unfold encodeProgram
  simp

/-- Auxiliary: `decodeProgramAux` inverts `encodeProgram` when given enough
    fuel. The fuel we hand it in the top-level `decodeProgram` is the byte
    count, which is at least the frame count for any well-formed program. -/
theorem decodeProgramAux_encodeProgram (p : Program) (wf : WellFormed p)
    (fuel : Nat) (hfuel : p.length ≤ fuel) :
    decodeProgramAux fuel (encodeProgram p) = some p := by
  induction p generalizing fuel with
  | nil =>
    cases fuel with
    | zero => rfl
    | succ n =>
      unfold encodeProgram
      simp [decodeProgramAux]
  | cons f rest ih =>
    have wf_f : WellFormedFrame f := wf f (List.mem_cons_self f rest)
    have wf_rest : WellFormed rest := fun g hg =>
      wf g (List.mem_cons_of_mem f hg)
    cases fuel with
    | zero => simp [List.length] at hfuel
    | succ n =>
      have hn : rest.length ≤ n := by
        simp [List.length_cons] at hfuel
        omega
      rw [encodeProgram_cons]
      have hstep : decodeFrame (encodeFrame f ++ encodeProgram rest)
                   = some (f, encodeProgram rest) :=
        decodeFrame_encodeFrame f wf_f _
      -- The encoded stream begins with a non-empty frame prefix.
      have hnonempty : encodeFrame f ++ encodeProgram rest ≠ [] := by
        unfold encodeFrame encodeU16
        simp
      match hcase : encodeFrame f ++ encodeProgram rest with
      | [] => exact absurd hcase hnonempty
      | b :: bs =>
        unfold decodeProgramAux
        simp only [hcase] at hstep
        rw [hstep]
        simp
        rw [ih wf_rest n hn]
        rfl

/-- Byte count is at least frame count — every frame contributes ≥ 4 bytes
    for its two header words. -/
theorem encodeProgram_length_ge (p : Program) :
    p.length ≤ (encodeProgram p).length := by
  induction p with
  | nil => simp [encodeProgram]
  | cons f rest ih =>
    rw [encodeProgram_cons]
    simp only [List.length_cons, List.length_append]
    have hf : 1 ≤ (encodeFrame f).length := by
      unfold encodeFrame encodeU16
      simp
    omega

/-- **Main theorem**: the emitter's `encode` is a right-inverse of `decode`
    on well-formed programs. -/
theorem decodeProgram_encodeProgram (p : Program) (wf : WellFormed p) :
    decodeProgram (encodeProgram p) = some p := by
  unfold decodeProgram
  exact decodeProgramAux_encodeProgram p wf _ (encodeProgram_length_ge p)

end Plow.Wire
