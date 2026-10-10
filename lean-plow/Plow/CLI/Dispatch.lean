/-
# Plow.CLI.Dispatch — the one checkpoint table.

`plow_verify` dispatches only through `endpoints`. `ProofAudit` requires every id here to have
a soundness entry in `proof-manifest.json`, and every manifest entry to name an id here.
-/
import Lean.Data.Json
import Plow.CLI.Schema
import Plow.CLI.Checkpoints

namespace Plow.CLI.Dispatch

open Lean (Json)
open Plow.CLI

def endpoints : List (String × (Json → IO Certificate)) := [
  ("A", fun p => return Checkpoints.checkA p),
  ("B", fun p => return Checkpoints.checkB p),
  ("C", fun p => return Checkpoints.checkC p),
  ("D", Checkpoints.checkD),
  ("E", fun p => return Checkpoints.checkE p),
  ("F", Checkpoints.checkF),
  ("G", fun p => return Checkpoints.checkG p),
  ("K", fun p => return Checkpoints.checkK p),
  ("S", fun p => return Checkpoints.checkS p),
  ("P", fun p => return Checkpoints.checkP p),
  ("R", fun p => return Checkpoints.checkR p),
  ("L", fun p => return Checkpoints.checkL p)
]

def endpointIds : List String := endpoints.map (·.1)

def run (cp : String) (payload : Json) : IO Certificate :=
  match endpoints.find? (·.1 == cp) with
  | some (_, check) => check payload
  | none => return { ok := false, checkpoint := cp,
                     notes := none, reason := some s!"unknown checkpoint '{cp}'" }

end Plow.CLI.Dispatch
