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
  let base := Json.mkObj [
    ("ok", toJson r.ok),
    ("query", toJson r.query),
    ("answer", r.answer)]
  let withCert := match r.certificate with
    | some c => base.setObjVal! "certificate" (toJson c)
    | none => base
  let withErr := match r.error with
    | some e => withCert.setObjVal! "error" (toJson e)
    | none => withCert
  match r.time_ms with
  | some t => withErr.setObjVal! "time_ms" (toJson t)
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

private def longestPath (n : Nat) (edges : List (Nat × Nat)) (durations : List Nat) : Nat :=
  -- Simple O(V+E) via topological relaxation. We compute finish times.
  let durArr := durations.toArray
  let mut finish := Array.mkArray n 0
  -- Iterate edges in a fixed-point loop (at most n iterations for a DAG).
  let mut changed := true
  let mut iters := 0
  while changed && iters < n do
    changed := false
    iters := iters + 1
    for (a, b) in edges do
      let fa := finish.getD a 0
      let da := durArr.getD a 0
      let fb := finish.getD b 0
      let db := durArr.getD b 0
      let candidate := fa + da + db
      if candidate > fb + db then
        -- Update start of b so its finish = max over predecessors + own duration
        -- Actually: finish[b] = max(finish[b], finish[a] + dur[a]) + dur[b] on first
        -- visit. Simpler: compute as longest-path to each node.
        let newFinB := Nat.max (finish.getD b 0) (fa + da)
        if newFinB > finish.getD b 0 then
          finish := finish.setD b newFinB
          changed := true
  -- Makespan = max(finish[t] + dur[t]) over all t.
  let mut ms := 0
  for t in List.range n do
    let f := finish.getD t 0
    let d := durArr.getD t 0
    ms := Nat.max ms (f + d)
  ms

def lowerBound (payload : Json) : IO QueryResultJ := do
  let edges ← match payload.getObjValAs? (List (List Nat)) "edges" with
    | .ok es => pure (es.filterMap fun l => match l with | [a, b] => some (a, b) | _ => none)
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

  let n := durations.length
  let cp := longestPath n edges durations
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
    s!"max(E1={cp}, E2={bwBound}, E3={compBound}) = {lb}; binding={binding} (CostBounds.makespan_dominates_lower_bounds)"

end Plow.CLI.Queries
