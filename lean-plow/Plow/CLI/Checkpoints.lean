/-
# Plow.CLI.Checkpoints — per-checkpoint dispatch handlers for the CLI.

Handlers return certificates scoped to the supplied abstract obligations.
They do not establish completeness of kernel access declarations or machine-code semantics.
-/
import Lean.Data.Json
import Plow.CLI.Schema
import Plow.CLI.Payload
import Plow.CLI.FastCheckD
import Plow.CLI.Effects
import Plow.Verify
import Plow.Sram
import Plow.Wire
import Plow.Rewrite
import Plow.RewriteBody
import Plow.TilePartition
import Plow.Knobs.Consistency
import Plow.Knobs.Scope
import Plow.Knobs.Ledger
import Plow.MeasuredPolicy
import Plow.MlaLayout

namespace Plow.CLI.Checkpoints

open Lean (Json)
open Plow.CLI Plow.Verify

def checkR (payload : Json) : Certificate :=
  match Plow.MeasuredPolicy.run payload with
  | .ok notes => ok "R" notes
  | .error msg => reject "R" msg

def checkL (payload : Json) : Certificate :=
  match Plow.MlaLayout.run payload with
  | .ok notes => ok "L" notes
  | .error msg => reject "L" msg

/-! ## Checkpoint K: knob consistency. -/

/-- Resolve the knob sources against the registry and target, and check every constraint.
    Backed by `Plow.Knobs.checkK_sound`; the registry-consistency and record-agreement checks
    around it are instance checks. -/
def checkK (payload : Json) : Certificate :=
  match Plow.Knobs.runK payload with
  | .ok notes => ok "K" notes
  | .error msg => reject "K" msg

/-! ## Checkpoint S: knob scope. -/

/-- Compare a base and a variant packet program by program against a knob's declared scope.
    Backed by `Plow.Knobs.Scope.checkS_sound`, `diff_complete`, `off_identity`,
    `untouched_rungs` and `route_untouched`. -/
def checkS (payload : Json) : Certificate :=
  match Plow.Knobs.Scope.runS payload with
  | .ok notes => ok "S" notes
  | .error msg => reject "S" msg

/-! ## Checkpoint P: performance certificate for a default flip. -/

/-- Decide a flip from ledger measurements: untouched rungs keep their digests, touched rungs
    improve beyond the control-vs-control floor, tier 4 is not worse, numeric changes carry
    passing facts. A floor that cannot be computed is `insufficient_evidence`, never a pass.
    Backed by `Plow.Knobs.Ledger.checkP_sound`, `insufficient_blocks`, `flip_non_regression`,
    `carry_over` and `per_rung_argmin`. -/
def checkP (payload : Json) : Certificate :=
  match Plow.Knobs.Ledger.runP payload with
  | .ok notes => ok "P" notes
  | .error msg => reject "P" msg

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
      match arr.toList.mapM Json.getStr? with
      | .error _ => reject "A" "payload 'rules' must contain only strings"
      | .ok rules =>
      match rules.find? (fun r => ¬ Plow.Rewrite.isSoundRule r) with
      | some bad =>
        reject "A" s!"rule '{bad}' is not in the sound-rules table; \
                     add it to Plow.Rewrite.soundRules with a proof"
      | none =>
        match payload.getObjVal? "bodies" with
        | .error _ => ok "A" s!"{rules.length} rewrite names in the proven syntax catalog; no floating-point or machine-code implementation claim"
        | .ok bodies => match Plow.RewriteBody.run rules bodies with
          | .ok notes => ok "A" notes
          | .error reason => reject "A" reason
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

/-! ## Checkpoint G: staged-LDS fit (LdsFitSound). -/

/-- Verify every always-staged GEMV instance in a program fits the decode-object
    LDS arena. Backed by `Plow.LdsFit.fits_of_check_ok`; the demand model is the
    kernel's own `rows*K + scratch` halves (op_gemm.h staged-x contract), the
    arena comes from hwspec via the Rust caller. A rejection names the first
    violating instance — the task-9 bug class caught at emit. -/
def checkG (payload : Json) : Certificate :=
  match Payload.parseLdsFit payload with
  | .error msg => reject "G" s!"payload parse error: {msg}"
  | .ok d =>
    match Plow.LdsFit.checkLdsFit d.arena d.ops with
    | .ok () =>
      ok "G" s!"staged-LDS fit: {d.ops.length} staged instances verified against arena {d.arena} halves"
    | .error s =>
      reject "G" s!"inst {s.idx} ({s.op}): staged demand rows={s.rows} * k={s.k} + scratch={s.scratch} = {Plow.LdsFit.demand s} halves exceeds arena {d.arena} — the always-staged kernel would read past the LDS window (task-9 class)"

/-! ## Checkpoint D: Counter protocol + reclamation (§5.10-D). -/


/-- Verify a concrete `(TaskGraph, CounterProtocol, AddressMap)` produced by
    `plowc`. Runs the executable verifier `verifyAddressMap` and additionally
    checks reader/writer disjointness — together these give the **strict**
    `AddressMapSound` guarantee (via `verifyAddressMap_sound_strict`), not
    just the loose form. -/
def checkD (payload : Json) : IO Certificate := do
  match Payload.parse payload with
  | .error msg => return reject "D" s!"payload parse error: {msg}"
  | .ok d =>
    let dependenciesOk ← match payload.getObjVal? "dependency_paths" with
      | .error _ => pure (verifyDependencies d.protocol)
      | .ok paths =>
        let parsed : Except String (List (List (Fin d.taskGraph.n))) := do
          let raw ← Lean.fromJson? (α := List Json) paths
          raw.mapM fun path => do
            let ids ← Payload.parseNatArrayStrict "dependency_paths" path
            ids.mapM (Payload.strictFin "dependency_paths" d.taskGraph.n)
        match parsed with
        | .error msg => return reject "D" s!"dependency witness parse error: {msg}"
        | .ok paths => pure (checkPaths d.protocol d.taskGraph.edges paths)
    if !dependenciesOk then
      return reject "D" "data dependency is not counter-ordered by the supplied protocol"
    let addressPaths : Option (List (PathWitness d.taskGraph)) ←
      match payload.getObjVal? "address_paths" with
      | .error _ => pure none
      | .ok paths =>
        let parsed : Except String (List (PathWitness d.taskGraph)) := do
          let raw ← Lean.fromJson? (α := List Json) paths
          raw.mapM fun path => do
            let source ← path.getObjValAs? Nat "source"
            let target ← path.getObjValAs? Nat "target"
            let via ← path.getObjVal? "via"
            let via ← Payload.parseNatArrayStrict "address_paths.via" via
            return { source := ← Payload.strictFin "address_paths.source" d.taskGraph.n source,
                     target := ← Payload.strictFin "address_paths.target" d.taskGraph.n target,
                     via := ← via.mapM (Payload.strictFin "address_paths.via" d.taskGraph.n) }
        match parsed with
        | .error msg => return reject "D" s!"address witness parse error: {msg}"
        | .ok paths => pure (some paths)
    -- FastCheckD is an early rejection filter, not a proof-backed acceptance path.
    match ← FastCheckD.run (if addressPaths.isSome then { d with entries := [] } else d) with
    | .error msg => return reject "D" s!"ordering-graph check failed: {msg}"
    | .ok (amOk, djOk) =>
      if ¬ amOk then
        return reject "D" "verifyAddressMap rejected — some byte-overlapping pair is not counter-ordered"
      else if ¬ djOk then
        return reject "D" "reader/writer sets overlap — strict AddressMapSound not derivable"
      else if !(match addressPaths with
          | some paths => verifyAddressMapVia d.protocol d.entries paths
          | none => verifyAddressMap d.protocol d.entries && readersWritersDisjointB d.entries) then
        return reject "D" "proven reference address checker rejected"
      else
        let mut notes := s!"proven verifyAddressMap accepted {d.entries.length} entries (strict); supplied graph/address scope only"
        if let .ok effects := payload.getObjVal? "memory_effects" then
          match addressPaths with
          | none => return reject "D" "memory effects require explicit address_paths"
          | some paths =>
            match Effects.run d paths effects with
            | .error msg => return reject "D" s!"memory effects rejected: {msg}"
            | .ok scope => notes := notes ++ "; " ++ scope
        return ok "D" notes

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
def checkF (payload : Json) : IO Certificate := do
  let cert ← checkD payload
  if cert.ok then
    return ok "F" s!"strict AddressMapSound; {cert.notes.getD ""}"
  else
    return reject "F" (cert.reason.getD "allocation check rejected")

end Plow.CLI.Checkpoints
