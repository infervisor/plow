/-
# Plow.CLI.Payload — parse a JSON payload into `Plow.Protocol` / `Plow.Memory`
                     types used by the verifier.

The parser is strict: unknown enum values, out-of-range indices, and type
mismatches produce descriptive errors (returned as a rejection `Certificate` by
`Checkpoints.checkD/F`). This matters because a permissive parser can silently
drop the very entries that would have exposed an unsafe map — turning a real
bug into a false green.
-/
import Lean.Data.Json
import Plow.Basic
import Plow.Protocol
import Plow.Memory
import Plow.Sram
import Plow.Wire
import Plow.TilePartition

namespace Plow.CLI.Payload

open Lean (Json)
open Plow Plow.Protocol Plow.Memory

/-- Strict: parse a JSON value as an array of Nat. Errors on non-array input
    or non-Nat elements (naming the offending index). -/
def parseNatArrayStrict (ctx : String) (j : Json) : Except String (List Nat) :=
  match j with
  | .arr xs => do
    let arr ← xs.mapIdxM fun i x =>
      match x.getNat? with
      | .ok n => pure n
      | _ => throw s!"{ctx}[{i}]: expected Nat"
    pure arr.toList
  | _ => throw s!"{ctx}: expected array of Nat"

/-- Legacy forgiving version — used by tolerant call sites (per-entry
    reader/writer lists, threshold values), where a soft default is fine. -/
def parseNatArray (j : Json) : List Nat :=
  match j with
  | .arr xs => xs.foldr (init := []) fun x acc =>
      match x.getNat? with
      | .ok n => n :: acc
      | _ => acc
  | _ => []

/-- The de-serialized D-checkpoint payload. -/
structure Deserialized where
  taskGraph     : TaskGraph
  protocol      : CounterProtocol taskGraph
  scheduleOrder : Fin taskGraph.n → Nat
  entries       : List (AddrEntry taskGraph)

/-- Convert a Nat to a `Fin n` — returns `none` when out of range. Callers
    that require presence turn this into a strict error. -/
def clampFin (n : Nat) (v : Nat) : Option (Fin n) :=
  if h : v < n then some ⟨v, h⟩ else none

/-- Strict fin: like `clampFin`, but returns a descriptive error when v ≥ n. -/
def strictFin (ctx : String) (n : Nat) (v : Nat) : Except String (Fin n) :=
  if h : v < n then .ok ⟨v, h⟩ else .error s!"{ctx}: index {v} out of range for n={n}"

/-- Strict: read an object field as a Nat array of length exactly `n`.
    Missing field → all zeros (waits/succs/resource are optional in the wire
    format). Present but malformed → error. Present with wrong length → error. -/
def getStrictNatArrayLenN (obj : Json) (name : String) (n : Nat)
    (allowMissing : Bool) : Except String (List Nat) := do
  match obj.getObjVal? name with
  | .error _ =>
    if allowMissing then pure (List.replicate n 0)
    else throw s!"protocol.{name}: missing"
  | .ok v => do
    let xs ← parseNatArrayStrict s!"protocol.{name}" v
    if xs.length ≠ n then
      throw s!"protocol.{name}: length {xs.length} ≠ n={n}"
    else
      pure xs

/-- Parse a rows-of-Nat-array field (used for `waits` / `succs`). Missing →
    all-empty rows; present but malformed / wrong length → error. -/
def parseRowsOfNat (obj : Json) (name : String) (n : Nat) :
    Except String (Array (List CounterId)) := do
  match obj.getObjVal? name with
  | .error _ => pure ((List.replicate n ([] : List CounterId)).toArray)
  | .ok j =>
    match j with
    | .arr rows =>
      if rows.size ≠ n then
        throw s!"protocol.{name}: length {rows.size} ≠ n={n}"
      else
        rows.mapIdxM fun i row => parseNatArrayStrict s!"protocol.{name}[{i}]" row
    | _ => throw s!"protocol.{name}: expected array of arrays"

/-- Assemble a `CounterProtocol` from the vector data (already validated). -/
def assembleProtocol (tg : TaskGraph)
    (waitsRows succsRows : Array (List CounterId))
    (resArr strArr : Array Nat)
    (thrJson : Json) : CounterProtocol tg :=
  { waits := fun t => waitsRows.get! t.val
    succs := fun t => succsRows.get! t.val
    threshold := fun c =>
      (match thrJson.getObjVal? (toString c) with
       | .ok v => (v.getNat?.toOption).getD 0
       | _ => 0)
    resource := fun t => resArr.get! t.val
    streamIdx := fun t => strArr.get! t.val }

/-- Build the `CounterProtocol` from a strictly-typed object. Every array is
    checked for length = n (or missing → defaulted to n zeros). Per-task waits
    and succs are each parsed as arrays-of-arrays. Threshold entries out of the
    JSON object return 0 (thresholds are strictly a hint to the checker). -/
def buildProtocol (tg : TaskGraph) (obj : Json) : Except String (CounterProtocol tg) := do
  let n := tg.n
  let resVec ← getStrictNatArrayLenN obj "resource"   n (allowMissing := true)
  let strVec ← getStrictNatArrayLenN obj "stream_idx" n (allowMissing := true)
  let waitsRows ← parseRowsOfNat obj "waits" n
  let succsRows ← parseRowsOfNat obj "succs" n
  let thrJson : Json := (obj.getObjVal? "threshold").toOption.getD (Json.mkObj [])
  pure (assembleProtocol tg waitsRows succsRows resVec.toArray strVec.toArray thrJson)

/-- Strict: parse one address-map entry. Unknown `cls` values, missing fields,
    and out-of-range reader/writer indices all error (naming the entry). -/
def parseEntry (tg : TaskGraph) (idx : Nat) (eJson : Json) :
    Except String (AddrEntry tg) := do
  let name := (eJson.getObjValAs? String "name").toOption.getD s!"entry[{idx}]"
  let ctx := s!"address_map[{idx}] ({name})"
  let offset ← match eJson.getObjVal? "offset" with
    | .ok v => match v.getNat? with
      | .ok n => pure n
      | _ => throw s!"{ctx}.offset: expected Nat"
    | .error _ => throw s!"{ctx}: missing offset"
  let size ← match eJson.getObjVal? "size" with
    | .ok v => match v.getNat? with
      | .ok n => pure n
      | _ => throw s!"{ctx}.size: expected Nat"
    | .error _ => throw s!"{ctx}: missing size"
  let clsStr ← (match eJson.getObjValAs? String "cls" with
    | .ok s => pure s
    | .error _ => throw s!"{ctx}: missing cls" : Except String String)
  let cls : BufClass ← (match clsStr with
    | "Persistent" => pure BufClass.Persistent
    | "RequestIo"  => pure BufClass.RequestIo
    | "Growable"   => pure BufClass.Growable
    | "Scratch"    => pure BufClass.Scratch
    | other        => throw s!"{ctx}.cls: unknown value '{other}' (expected Persistent/RequestIo/Growable/Scratch)"
    : Except String BufClass)
  -- Readers / writers are optional (some entries have no readers, e.g. terminal outputs).
  let rawWriters := ((eJson.getObjVal? "writers").toOption.map parseNatArray).getD []
  let rawReaders := ((eJson.getObjVal? "readers").toOption.map parseNatArray).getD []
  let writers ← rawWriters.mapM (strictFin s!"{ctx}.writers" tg.n)
  let readers ← rawReaders.mapM (strictFin s!"{ctx}.readers" tg.n)
  return { name := name, offset := offset, size := size, cls := cls,
           writers := writers, readers := readers }

/-- Strict: build every address-map entry. Errors carry the offending index +
    entry name. -/
def buildEntries (tg : TaskGraph) (entriesJson : Json) :
    Except String (List (AddrEntry tg)) := do
  match entriesJson with
  | .arr arr =>
    let ls ← arr.mapIdxM fun i eJson => parseEntry tg i eJson
    pure ls.toList
  | _ => throw "address_map: expected array of entries"

/-- Strict: build the schedule-order function from a length-n Nat array. Missing
    field defaults to the identity map; malformed → error. -/
def buildScheduleOrder (tg : TaskGraph) (obj : Json) :
    Except String (Fin tg.n → Nat) := do
  match obj.getObjVal? "schedule_order" with
  | .error _ => pure (fun t => t.val)
  | .ok j =>
    let xs ← parseNatArrayStrict "schedule_order" j
    if xs.length ≠ tg.n then
      throw s!"schedule_order: length {xs.length} ≠ n={tg.n}"
    else
      let arr := xs.toArray
      pure (fun t => arr[t.val]!)

/-- Parse the full D-checkpoint payload. Every failure returns a descriptive
    `.error` naming the offending field — the caller in `Checkpoints` turns
    that into a rejection certificate. -/
def parse (payload : Json) : Except String Deserialized := do
  let tgJson ← payload.getObjVal? "task_graph"
  let n ← tgJson.getObjVal? "n" >>= (·.getNat?)
  -- Edges: strict — malformed pair or out-of-range endpoint is an error.
  let edgesJson := (tgJson.getObjVal? "edges").toOption.getD (Json.arr #[])
  let edges : List (Fin n × Fin n) ← (match edgesJson with
    | .arr arr => do
      let out ← arr.mapIdxM fun i eJson => do
        match eJson with
        | .arr #[a, b] =>
          match a.getNat?, b.getNat? with
          | .ok av, .ok bv => do
            let af ← strictFin s!"task_graph.edges[{i}].0" n av
            let bf ← strictFin s!"task_graph.edges[{i}].1" n bv
            pure (af, bf)
          | _, _ => throw s!"task_graph.edges[{i}]: expected [Nat, Nat]"
        | _ => throw s!"task_graph.edges[{i}]: expected 2-element array"
      pure out.toList
    | _ => throw "task_graph.edges: expected array"
    : Except String (List (Fin n × Fin n)))
  let tg : TaskGraph := { n := n, edges := edges }
  let protoJson := (payload.getObjVal? "protocol").toOption.getD (Json.mkObj [])
  let entriesJson := (payload.getObjVal? "address_map").toOption.getD (Json.arr #[])
  let protocol ← buildProtocol tg protoJson
  let scheduleOrder ← buildScheduleOrder tg payload
  let entries ← buildEntries tg entriesJson
  return {
    taskGraph := tg
    protocol := protocol
    scheduleOrder := scheduleOrder
    entries := entries
  }

/-! ## Checkpoint C — SRAM temporal-fit payload. -/

/-- The de-serialized C-checkpoint payload. Carries a page-budget and a list
    of hand-off descriptors (one per relaxable that the caller wants to prove
    fits temporally). -/
structure SramFitPayload where
  budget   : Nat
  handoffs : List Sram.Handoff

/-- Strict-read one `Handoff` from JSON. All five fields required. -/
def parseHandoff (idx : Nat) (j : Json) : Except String Sram.Handoff := do
  let ctx := s!"handoffs[{idx}]"
  let getNat (name : String) : Except String Nat := do
    match j.getObjVal? name with
    | .error _ => throw s!"{ctx}: missing {name}"
    | .ok v => match v.getNat? with
      | .ok n => pure n
      | _ => throw s!"{ctx}.{name}: expected Nat"
  let producerPages   ← getNat "producer_pages"
  let consumerPages   ← getNat "consumer_pages"
  let producerRelease ← getNat "producer_release"
  let consumerAcquire ← getNat "consumer_acquire"
  let consumerRelease ← getNat "consumer_release"
  return {
    producerPages := producerPages
    consumerPages := consumerPages
    producerRelease := producerRelease
    consumerAcquire := consumerAcquire
    consumerRelease := consumerRelease
  }

/-- Parse the full C-checkpoint payload. Structure:
    `{ "budget": Nat, "handoffs": [ Handoff, ... ] }`. -/
def parseSramFit (payload : Json) : Except String SramFitPayload := do
  let budget ← match payload.getObjVal? "budget" with
    | .error _ => throw "missing budget"
    | .ok v => match v.getNat? with
      | .ok n => pure n
      | _ => throw "budget: expected Nat"
  let handoffsJson := (payload.getObjVal? "handoffs").toOption.getD (Json.arr #[])
  let hos ← (match handoffsJson with
    | .arr arr => do
      let out ← arr.mapIdxM fun i eJson => parseHandoff i eJson
      pure out.toList
    | _ => throw "handoffs: expected array"
    : Except String (List Sram.Handoff))
  return { budget := budget, handoffs := hos }

/-! ## Checkpoint E — wire-format round-trip payload. -/

structure WirePayload where
  raw     : List Nat            -- byte stream as list of Nats (0..255)
  frames  : Wire.Program        -- the caller's decoded view

/-- Parse a JSON array as a list of Nats. -/
def parseByteList (ctx : String) (j : Json) : Except String (List Nat) :=
  parseNatArrayStrict ctx j

/-- Parse a single frame: `{"opcode": Nat, "payload": [Nat, ...]}`. -/
def parseWireFrame (idx : Nat) (j : Json) : Except String Wire.Frame := do
  let ctx := s!"frames[{idx}]"
  let opcode ← match j.getObjVal? "opcode" with
    | .error _ => throw s!"{ctx}: missing opcode"
    | .ok v => match v.getNat? with
      | .ok n => pure n
      | _ => throw s!"{ctx}.opcode: expected Nat"
  let payload ← match j.getObjVal? "payload" with
    | .error _ => throw s!"{ctx}: missing payload"
    | .ok v => parseByteList s!"{ctx}.payload" v
  return { opcode := opcode, payload := payload }

/-- Parse the full E-checkpoint payload:
    `{ "raw": [Nat, ...], "frames": [ { opcode, payload }, ... ] }`. -/
def parseWire (payload : Json) : Except String WirePayload := do
  let rawJson ← match payload.getObjVal? "raw" with
    | .ok v => pure v
    | .error _ => throw "missing raw"
  let raw ← parseByteList "raw" rawJson
  let framesJson := (payload.getObjVal? "frames").toOption.getD (Json.arr #[])
  let frames ← (match framesJson with
    | .arr arr => do
      let out ← arr.mapIdxM fun i eJson => parseWireFrame i eJson
      pure out.toList
    | _ => throw "frames: expected array"
    : Except String Wire.Program)
  return { raw := raw, frames := frames }

/-! ## Checkpoint B — tile partition + cost bound payload. -/

/-- One tile candidate: the target GEMM shape, the tile it will use, and the
    caller-supplied cost bound (upper). Checkpoint B verifies both partition
    validity and the cost bound. -/
structure TileCandidate where
  gemm : TilePartition.Gemm
  tile : TilePartition.Tile
  costBound : Nat

structure TilePartitionPayload where
  candidates : List TileCandidate

def getNatField (ctx : String) (j : Json) (name : String) : Except String Nat := do
  match j.getObjVal? name with
  | .error _ => throw s!"{ctx}: missing {name}"
  | .ok v => match v.getNat? with
    | .ok n => pure n
    | _ => throw s!"{ctx}.{name}: expected Nat"

def parseTileCandidate (idx : Nat) (j : Json) : Except String TileCandidate := do
  let ctx := s!"candidates[{idx}]"
  let g ← match j.getObjVal? "gemm" with
    | .ok gj => do
      let m ← getNatField s!"{ctx}.gemm" gj "m"
      let n ← getNatField s!"{ctx}.gemm" gj "n"
      let k ← getNatField s!"{ctx}.gemm" gj "k"
      pure ({ m := m, n := n, k := k } : TilePartition.Gemm)
    | .error _ => throw s!"{ctx}: missing gemm"
  let t ← match j.getObjVal? "tile" with
    | .ok tj => do
      let bm ← getNatField s!"{ctx}.tile" tj "bm"
      let bn ← getNatField s!"{ctx}.tile" tj "bn"
      let bk ← getNatField s!"{ctx}.tile" tj "bk"
      pure ({ bm := bm, bn := bn, bk := bk } : TilePartition.Tile)
    | .error _ => throw s!"{ctx}: missing tile"
  let costBound ← getNatField ctx j "cost_bound"
  return { gemm := g, tile := t, costBound := costBound }

def parseTilePartition (payload : Json) : Except String TilePartitionPayload := do
  let arr := (payload.getObjVal? "candidates").toOption.getD (Json.arr #[])
  let cs ← (match arr with
    | .arr xs => do
      let out ← xs.mapIdxM fun i eJson => parseTileCandidate i eJson
      pure out.toList
    | _ => throw "candidates: expected array"
    : Except String (List TileCandidate))
  return { candidates := cs }

end Plow.CLI.Payload
