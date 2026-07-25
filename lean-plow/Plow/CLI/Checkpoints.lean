/-
# Plow.CLI.Checkpoints — per-checkpoint dispatch handlers for the CLI.

One handler per checkpoint A..F. A and F are the fully-proven paths (they
delegate to `Plow.Verify.verifyAddressMap`). The rest return `notImplemented`
until their universal proofs land — the wiring is here so callers see the
shape of the eventual API.
-/
import Lean.Data.Json
import Plow.CLI.Schema
import Plow.CLI.Payload
import Plow.Verify
import Plow.Sram
import Plow.Wire
import Plow.Rewrite
import Plow.TilePartition

namespace Plow.CLI.Checkpoints

open Lean (Json)
open Plow.CLI Plow.Verify

/-! ## Checkpoint A: Rewrite rule soundness (§5.10-A). -/

/-- Verify a rewrite rule is in the sound-rules table. Every rule the compiler
    fires must appear in `Plow.Rewrite.soundRules` — the sound-rules table is
    a *closed enumeration* backed by definitional-equality proofs in
    `Plow.Rewrite.rule_*`.

    Payload shape: `{ "rules": [String, ...] }` — the list of rule names the
    egglog engine reports as fired for this bucket. Rejection names the first
    unknown rule. -/
def checkA (payload : Json) : Certificate :=
  match payload.getObjVal? "rules" with
  | .error _ => reject "A" "payload missing 'rules' field"
  | .ok j =>
    match j with
    | .arr arr =>
      let rules := arr.foldr (init := ([] : List String)) fun x acc =>
        match x.getStr? with
        | .ok s => s :: acc
        | _ => acc
      match rules.find? (fun r => ¬ Plow.Rewrite.isSoundRule r) with
      | some bad =>
        reject "A" s!"rule '{bad}' is not in the sound-rules table; \
                     add it to Plow.Rewrite.soundRules with a proof"
      | none =>
        ok "A" s!"{rules.length} rules verified sound"
    | _ => reject "A" "payload 'rules' must be an array of strings"

/-! ## Checkpoint B: Tile partition + cost bounds (§5.10-B). -/

/-- Verify every tile candidate: (a) partition is valid (positive tile dims,
    each ≤ its GEMM dim), and (b) the caller's cost bound is not exceeded by
    the tile-work sum `tileCount · bm · bn · bk`. Backed by
    `Plow.TilePartition.tile_partition_covers` (completeness) and
    `check_sound` (partition validity from the executable check). -/
def checkTileCandidate (idx : Nat) (c : Payload.TileCandidate) :
    Except String Unit := do
  match Plow.TilePartition.checkPartition c.gemm c.tile with
  | .error msg => throw s!"candidate[{idx}]: partition invalid: {msg}"
  | .ok _ =>
    let work := Plow.TilePartition.tileCount c.gemm c.tile *
                c.tile.bm * c.tile.bn * c.tile.bk
    if work > c.costBound then
      throw s!"candidate[{idx}]: tile-work {work} > cost_bound {c.costBound}"
    else
      .ok ()

def checkAllCandidates : Nat → List Payload.TileCandidate → Except String Unit
  | _, [] => .ok ()
  | i, c :: rest => do
    checkTileCandidate i c
    checkAllCandidates (i + 1) rest

def checkB (payload : Json) : Certificate :=
  match Payload.parseTilePartition payload with
  | .error msg => reject "B" s!"payload parse error: {msg}"
  | .ok d =>
    match checkAllCandidates 0 d.candidates with
    | .ok _ =>
      ok "B" s!"tile-partition + cost bound verified: {d.candidates.length} candidates"
    | .error msg => reject "B" msg

/-! ## Checkpoint C: SRAM temporal fit (§5.10-C). -/

/-- Executable check: every submitted hand-off must satisfy `temporalFitSafe`
    against the shared page budget. The Rust `sram_fit::analyze_temporal_fit`
    pass filters candidates against this rule already; the Lean side
    re-checks so the promotion story is closed by the universal theorem
    `Plow.Sram.occupancy_le_of_temporal_fit`. -/
def checkSramFit (b : Nat) : List Plow.Sram.Handoff → Except (Nat × String) Unit
  | [] => .ok ()
  | h :: rest => do
    if ¬ (h.producerRelease ≤ h.consumerAcquire) then
      throw (rest.length, "producer_release > consumer_acquire (temporally overlapping)")
    else if ¬ (h.producerPages ≤ b) then
      throw (rest.length, s!"producer_pages {h.producerPages} > budget {b}")
    else if ¬ (h.consumerPages ≤ b) then
      throw (rest.length, s!"consumer_pages {h.consumerPages} > budget {b}")
    else
      checkSramFit b rest

/-- Verify every hand-off in a bucket fits its SRAM budget by the temporal-
    disjointness rule. Backed by `Plow.Sram.occupancy_le_of_temporal_fit`. -/
def checkC (payload : Json) : Certificate :=
  match Payload.parseSramFit payload with
  | .error msg => reject "C" s!"payload parse error: {msg}"
  | .ok d =>
    match checkSramFit d.budget d.handoffs with
    | .ok () =>
      ok "C" s!"temporal-fit safe: {d.handoffs.length} hand-offs verified against budget {d.budget}"
    | .error (i, reason) =>
      reject "C" s!"handoff[{d.handoffs.length - 1 - i}]: {reason}"

/-! ## Checkpoint D: Counter protocol + reclamation (§5.10-D). -/

/-- Verify a concrete `(TaskGraph, CounterProtocol, AddressMap)` produced by
    `plowc`. Runs the executable verifier `verifyAddressMap` and additionally
    checks reader/writer disjointness — together these give the **strict**
    `AddressMapSound` guarantee (via `verifyAddressMap_sound_strict`), not
    just the loose form. -/
def checkD (payload : Json) : Certificate :=
  match Payload.parse payload with
  | .error msg => reject "D" s!"payload parse error: {msg}"
  | .ok d =>
    if ¬ verifyAddressMap d.protocol d.entries then
      reject "D" "verifyAddressMap rejected — some byte-overlapping pair is not counter-ordered"
    else if ¬ readersWritersDisjointB d.entries then
      reject "D" "reader/writer sets overlap — strict AddressMapSound not derivable"
    else
      ok "D" s!"verifyAddressMap accepted {d.entries.length} entries (strict)"

/-! ## Checkpoint E: Wire-format round-trip (§5.10-E). -/

/-- Compare two byte lists — returns `none` on match, `some idx` on first
    divergence. Used for descriptive rejection reasons. -/
def firstDiff : List Nat → List Nat → Nat → Option Nat
  | [],    [],     _   => none
  | [],    _::_,   idx => some idx
  | _::_,  [],     idx => some idx
  | a::as, b::bs,  idx => if a = b then firstDiff as bs (idx + 1) else some idx

/-- Verify the wire round-trip. Success requires both directions to check:
    `encode(frames) = raw` and `decode(raw) = some frames`. The universal
    theorem `Plow.Wire.decodeProgram_encodeProgram` proves either direction
    implies the other on well-formed input; asking for both catches schema
    drift on either side of the bridge. -/
def checkE (payload : Json) : Certificate :=
  match Payload.parseWire payload with
  | .error msg => reject "E" s!"payload parse error: {msg}"
  | .ok w =>
    let re := Wire.encodeProgram w.frames
    if re ≠ w.raw then
      match firstDiff re w.raw 0 with
      | some idx => reject "E" s!"encode(frames) ≠ raw: first divergence at byte {idx}"
      | none     => reject "E" "encode(frames) ≠ raw: length mismatch"
    else
      match Wire.decodeProgram w.raw with
      | none        => reject "E" "decode(raw) failed (malformed stream)"
      | some frames =>
        if frames ≠ w.frames then
          reject "E" "decode(raw) ≠ frames (round-trip mismatch)"
        else
          ok "E" s!"wire round-trip: {w.frames.length} frames, {w.raw.length} bytes verified"

/-! ## Checkpoint F: Allocation safety (§5.10-F). -/

/-- Address-map allocation safety. Same underlying verifier + disjointness
    check as D — F is conceptually "post-emit" verification, but it's the
    same math (strict `AddressMapSound`). -/
def checkF (payload : Json) : Certificate :=
  match Payload.parse payload with
  | .error msg => reject "F" s!"payload parse error: {msg}"
  | .ok d =>
    if ¬ verifyAddressMap d.protocol d.entries then
      reject "F" "allocation unsafe: two byte-overlapping entries have no counter or resource ordering"
    else if ¬ readersWritersDisjointB d.entries then
      reject "F" "allocation unsafe: reader/writer sets overlap — cannot derive strict safety"
    else
      ok "F" s!"allocation safe: {d.entries.length} entries checked, strict AddressMapSound"

end Plow.CLI.Checkpoints
