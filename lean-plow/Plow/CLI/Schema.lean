/-
# Plow.CLI.Schema — JSON request/response types for the verifier CLI.

Kept independent of the theorem-carrying modules so the CLI can build even if
future refactors change the internal representation. `Certificate` is the wire
type the Rust side deserializes.
-/
import Lean.Data.Json

namespace Plow.CLI

open Lean (Json)

/-- The verifier's response for a single checkpoint call. -/
structure Certificate where
  ok         : Bool
  checkpoint : String
  notes      : Option String := none
  reason     : Option String := none

def Certificate.toJson (c : Certificate) : Json :=
  let baseFields : List (String × Json) :=
    [("ok", Json.bool c.ok), ("checkpoint", Json.str c.checkpoint)]
  let withNotes := match c.notes with
    | some n => baseFields ++ [("notes", Json.str n)]
    | none   => baseFields
  let withReason := match c.reason with
    | some r => withNotes ++ [("reason", Json.str r)]
    | none   => withNotes
  Json.mkObj withReason

/-- Handy helpers for building responses. -/
def ok (cp : String) (notes : String) : Certificate :=
  { ok := true, checkpoint := cp, notes := some notes, reason := none }

def reject (cp : String) (reason : String) : Certificate :=
  { ok := false, checkpoint := cp, notes := none, reason := some reason }

def notImplemented (cp : String) : Certificate :=
  { ok := false, checkpoint := cp, notes := none,
    reason := some s!"checkpoint {cp} not yet proven; universal lemma pending" }

end Plow.CLI
