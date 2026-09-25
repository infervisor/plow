/-
# Plow.CLI.Queries — performance query handlers for the CLI.

One handler per query type. Unlike checkpoints (accept/reject), queries
compute an answer and return it with a correctness certificate.

Dispatched from `Main.lean` via the `"query"` key in the JSON request.
-/
import Lean.Data.Json
import Plow.CLI.Schema
import Plow.CounterGranularity
import Plow.CostBounds

namespace Plow.CLI.Queries

open Lean (Json toJson)
open Plow.CLI

/-! ## Query result structure. -/

structure QueryResultJ where
  ok : Bool
  query : String
  answer : Json
  certificate : Option String := none
  error : Option String := none
  time_ms : Option Nat := none
  deriving Inhabited

def QueryResultJ.toJson (r : QueryResultJ) : Json :=
  -- `Lean.toJson` must stay qualified in this body: the unqualified name
  -- resolves to this very def (recursive self-reference), not the class method.
  let base := Json.mkObj [
    ("ok", Lean.toJson r.ok),
    ("query", Lean.toJson r.query),
    ("answer", r.answer)]
  let withCert := match r.certificate with
    | some c => base.setObjVal! "certificate" (Lean.toJson c)
    | none => base
  let withErr := match r.error with
    | some e => withCert.setObjVal! "error" (Lean.toJson e)
    | none => withCert
  match r.time_ms with
  | some t => withErr.setObjVal! "time_ms" (Lean.toJson t)
  | none => withErr

def okResult (qt : String) (answer : Json) (cert : String) : QueryResultJ :=
  { ok := true, query := qt, answer := answer, certificate := some cert }

def errResult (qt : String) (reason : String) : QueryResultJ :=
  { ok := false, query := qt, answer := Json.null, error := some reason }

/-! ## Counter Granularity query.

    Evaluates `CounterGranularity.fineCanPay` per edge. -/

private def parseEdge (j : Json) : Except String (Nat × List Nat × List Nat) := do
  let id ← j.getObjValAs? Nat "id"
  let slices ← j.getObjValAs? (List Nat) "consumer_slices"
  let work ← j.getObjValAs? (List Nat) "work"
  return (id, slices, work)

private def evalEdge (id : Nat) (cons : List Nat) (work : List Nat) : Json :=
  -- Build the work function from the parallel lists.
  let workFn : Nat → Nat := fun v =>
    match cons.indexOf? v with
    | some idx => work.getD idx 0
    | none => 0
  let useFine := Plow.CounterGranularity.fineCanPay cons workFn
  Json.mkObj [
    ("id", toJson id),
    ("use_fine", toJson useFine),
    ("reason", toJson (if useFine then "non-uniform work: fine can pay"
                        else "uniform work: collapse theorem applies"))]

def counterGranularity (payload : Json) : IO QueryResultJ := do
  match payload.getObjValAs? (List Json) "edges" with
  | .error msg => return errResult "counter_granularity" s!"missing 'edges': {msg}"
  | .ok edgesJ =>
    let mut decisions : List Json := []
    for ej in edgesJ do
      match parseEdge ej with
      | .error msg => return errResult "counter_granularity" s!"edge parse error: {msg}"
      | .ok (id, cons, work) =>
        decisions := decisions ++ [evalEdge id cons work]
    return okResult "counter_granularity"
      (Json.mkObj [("decisions", Json.arr decisions.toArray)])
      s!"fineCanPay evaluated on {edgesJ.length} edges (CounterGranularity.collapse)"

/-! ## Lower Bound query.

    Computes max(critical_path, bw_bound, compute_bound). -/

private def longestPath (edges : List (Nat × Nat)) (durations : List Nat) : Except String Nat := do
  let n := durations.length
  let durArr := durations.toArray
  let mut successors : Array (Array Nat) := Array.mkArray n #[]
  let mut indegree := Array.mkArray n 0
  for (a, b) in edges do
    unless a < n && b < n do
      throw s!"edge ({a}, {b}) is outside {n} tasks"
    successors := successors.setD a ((successors.getD a #[]).push b)
    indegree := indegree.setD b (indegree.getD b 0 + 1)
  let mut ready := (Array.range n).filter fun i => indegree.getD i 0 == 0
  let mut cursor := 0
  let mut finish := durArr
  let mut ms := 0
  while cursor < ready.size do
    let a := ready.getD cursor 0
    cursor := cursor + 1
    ms := Nat.max ms (finish.getD a 0)
    for b in successors.getD a #[] do
      finish := finish.setD b (Nat.max (finish.getD b 0)
        (finish.getD a 0 + durArr.getD b 0))
      let remaining := indegree.getD b 0 - 1
      indegree := indegree.setD b remaining
      if remaining == 0 then ready := ready.push b
  unless cursor == n do throw "task graph contains a cycle"
  return ms

def lowerBound (payload : Json) : IO QueryResultJ := do
  let edges ← match payload.getObjValAs? (List (List Nat)) "edges" with
    | .ok es =>
      match es.mapM (fun l => match l with
        | [a, b] => Except.ok (a, b)
        | _ => Except.error "each edge must contain exactly two task indices") with
      | .ok pairs => pure pairs
      | .error msg => return errResult "lower_bound" msg
    | .error msg => return errResult "lower_bound" s!"missing 'edges': {msg}"
  let durations ← match payload.getObjValAs? (List Nat) "durations" with
    | .ok ds => pure ds
    | .error msg => return errResult "lower_bound" s!"missing 'durations': {msg}"
  let totalHbm ← match payload.getObjValAs? Nat "total_hbm_bytes" with
    | .ok v => pure v
    | .error msg => return errResult "lower_bound" s!"missing 'total_hbm_bytes': {msg}"
  let peakBw ← match payload.getObjValAs? Nat "peak_bw_bytes_per_cycle" with
    | .ok v => pure v
    | .error msg => return errResult "lower_bound" s!"missing 'peak_bw_bytes_per_cycle': {msg}"
  let totalFlops ← match payload.getObjValAs? Nat "total_flops" with
    | .ok v => pure v
    | .error msg => return errResult "lower_bound" s!"missing 'total_flops': {msg}"
  let peakFlops ← match payload.getObjValAs? Nat "peak_flops_per_cycle" with
    | .ok v => pure v
    | .error msg => return errResult "lower_bound" s!"missing 'peak_flops_per_cycle': {msg}"

  let cp ← match longestPath edges durations with
    | .ok v => pure v
    | .error msg => return errResult "lower_bound" msg
  if totalHbm > 0 && peakBw == 0 then
    return errResult "lower_bound" "positive HBM work requires positive bandwidth"
  if totalFlops > 0 && peakFlops == 0 then
    return errResult "lower_bound" "positive compute work requires positive throughput"
  let bwBound := if peakBw > 0 then (totalHbm + peakBw - 1) / peakBw else 0
  let compBound := if peakFlops > 0 then (totalFlops + peakFlops - 1) / peakFlops else 0
  let lb := Nat.max cp (Nat.max bwBound compBound)
  let binding := if lb == cp then "critical_path"
                 else if lb == bwBound then "hbm_bandwidth"
                 else "compute_throughput"

  let answer := Json.mkObj [
    ("lower_bound", toJson lb),
    ("binding_constraint", toJson binding),
    ("critical_path", toJson cp),
    ("bw_bound", toJson bwBound),
    ("compute_bound", toJson compBound)]

  return okResult "lower_bound" answer
    s!"validated DAG; evaluated max(E1={cp}, E2={bwBound}, E3={compBound}) = {lb}; binding={binding}; conditional on supplied durations, work and channel rates; not a measured hardware guarantee"

end Plow.CLI.Queries
