import Plow.Effects
import Plow.CLI.Payload

namespace Plow.CLI.Effects

open Lean (Json)
open Plow Plow.Protocol Plow.Verify Plow.Effects

def parseLease (tg : TaskGraph) (j : Json) : Except String (Lease tg) := do
  let acquire ← j.getObjValAs? Nat "acquire"
  let retire ← j.getObjValAs? Nat "retire"
  let cancel ← match ← j.getObjVal? "cancel" with
    | .null => pure none
    | value => do
      let task ← value.getNat?
      pure (some (← Payload.strictFin "lease.cancel" tg.n task))
  return {
    pool := ← j.getObjValAs? Nat "pool"
    allocation := ← j.getObjValAs? Nat "allocation"
    generation := ← j.getObjValAs? Nat "generation"
    owner := ← j.getObjValAs? Nat "owner"
    offset := ← j.getObjValAs? Nat "offset"
    size := ← j.getObjValAs? Nat "size"
    acquire := ← Payload.strictFin "lease.acquire" tg.n acquire
    retire := ← Payload.strictFin "lease.retire" tg.n retire
    cancel := cancel }

def parseAccess (tg : TaskGraph) (j : Json) : Except String (Access tg) := do
  let task ← j.getObjValAs? Nat "task"
  return {
    task := ← Payload.strictFin "access.task" tg.n task
    pool := ← j.getObjValAs? Nat "pool"
    allocation := ← j.getObjValAs? Nat "allocation"
    generation := ← j.getObjValAs? Nat "generation"
    owner := ← j.getObjValAs? Nat "owner"
    offset := ← j.getObjValAs? Nat "offset"
    size := ← j.getObjValAs? Nat "size"
    write := ← j.getObjValAs? Bool "write" }

def checkCompletionCounters {tg : TaskGraph} (p : CounterProtocol tg)
    (fences : List Nat) : Bool :=
  let tasks := List.finRange tg.n
  -- Resource cursor order is issue order, not completion. This scope uses only
  -- counters whose declared release/acquire fence covers every producer.
  (tasks.all fun a => (p.succs a).all fun counter =>
    decide (((p.succs a).filter (· == counter)).length = 1)) &&
  (tasks.all fun a => tasks.all fun b => decide (a = b) || decide (p.resource a ≠ p.resource b)) &&
  (tasks.all fun t => (p.waits t).all fun counter =>
    let producers := tasks.filter fun source => decide (counter ∈ p.succs source)
    decide (counter ∈ fences) && decide (0 < producers.length) &&
    decide (p.threshold counter = producers.length))

def run (d : Payload.Deserialized) (paths : List (PathWitness d.taskGraph))
    (j : Json) : Except String String := do
  let schema ← j.getObjValAs? Nat "schema"
  if schema != 1 then throw "unsupported memory effects schema"
  let leasesRaw ← j.getObjValAs? (List Json) "leases"
  let accessesRaw ← j.getObjValAs? (List Json) "accesses"
  let leases ← leasesRaw.mapM (parseLease d.taskGraph)
  let accesses ← accessesRaw.mapM (parseAccess d.taskGraph)
  if leases.isEmpty || accesses.isEmpty then throw "empty memory effect scope"
  let fences ← j.getObjVal? "fence_counters"
  let fences ← Payload.parseNatArrayStrict "fence_counters" fences
  if !checkCompletionCounters d.protocol fences then
    throw "effects require completion counters, full producer thresholds and release/acquire fence declarations"
  for a in leases do
    if (leases.filter fun b => sameGeneration a b).length != 1 then
      throw "duplicate allocation generation"
    for b in leases do
      if !sameGeneration a b && a.pool == b.pool && a.allocation == b.allocation then
        if a.generation < b.generation then
          if !witnessedBefore d.protocol paths a.retire b.acquire then
            throw "allocation generation reused before prior retirement"
        else if !witnessedBefore d.protocol paths b.retire a.acquire then
          throw "allocation generation reused before prior retirement"
  if !checkRetirements d.protocol paths leases then throw "acquire/cancellation is not ordered before retirement"
  if !checkAccesses d.protocol paths leases accesses then throw "access has stale generation, wrong owner, bounds or lifetime"
  if !checkReuse d.protocol paths leases then throw "overlapping allocation reuse is not retired"
  if !checkConflicts d.protocol paths accesses then throw "RAW/WAR/WAW conflict is not completion-ordered"
  return s!"{accesses.length} supplied effects/{leases.length} leases checked by checkConflicts_sound, checkAccesses_sound, checkReuse_sound, cancellation_retired; kernel effect completeness and fence implementation are external obligations"

end Plow.CLI.Effects
