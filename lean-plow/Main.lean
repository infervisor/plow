/-
# `plow_verify` — JSON-IPC CLI for the Rust ↔ Lean bridge.

Reads one JSON request from stdin. The request is dispatched based on its
top-level key:

* `{"checkpoint": "D", "payload": {...}}` → verification (accept/reject)
* `{"query": "counter_granularity", "payload": {...}}` → performance oracle

## Checkpoint output:
  { "ok": true,  "checkpoint": "D", "notes": "…" }
  { "ok": false, "checkpoint": "D", "reason": "…" }

## Query output:
  { "ok": true,  "query": "counter_granularity", "answer": {...}, "certificate": "…" }
  { "ok": false, "query": "counter_granularity", "error": "…" }

Exit code is 0 iff `ok = true`. The Rust wrapper treats a non-zero exit as a
verifier rejection / query failure.
-/
import Lean.Data.Json
import Plow.CLI.Schema
import Plow.CLI.Checkpoints
import Plow.CLI.Queries

open Lean (Json)
open Plow.CLI

def runCheckpoint (cp : String) (payload : Json) : IO Certificate := do
  match cp with
  | "A" => return Checkpoints.checkA payload
  | "B" => return Checkpoints.checkB payload
  | "C" => return Checkpoints.checkC payload
  | "D" => Checkpoints.checkD payload
  | "E" => return Checkpoints.checkE payload
  | "F" => Checkpoints.checkF payload
  | _   => return { ok := false, checkpoint := cp,
                    notes := none, reason := some s!"unknown checkpoint '{cp}'" }

def runQuery (qt : String) (payload : Json) : IO Queries.QueryResultJ := do
  match qt with
  | "counter_granularity" => Queries.counterGranularity payload
  | "lower_bound"         => Queries.lowerBound payload
  | _                     => return Queries.errResult qt s!"unknown query type '{qt}'"

partial def readAllStdin (acc : String) : IO String := do
  let stdin ← IO.getStdin
  let line ← stdin.getLine
  if line.isEmpty then return acc
  readAllStdin (acc ++ line)

def main (_args : List String) : IO UInt32 := do
  let input ← readAllStdin ""
  match Json.parse input with
  | .error e => do
    let cert : Certificate :=
      { ok := false, checkpoint := "?", notes := none,
        reason := some s!"invalid JSON request: {e}" }
    IO.println cert.toJson.compress
    return 1
  | .ok request =>
    -- Dispatch: "query" key → performance oracle; "checkpoint" key → verification.
    match request.getObjValAs? String "query" with
    | .ok qt =>
      let payload := (request.getObjVal? "payload").toOption.getD Json.null
      let result ← runQuery qt payload
      IO.println result.toJson.compress
      return if result.ok then 0 else 1
    | .error _ =>
      -- Fall through to checkpoint dispatch (existing behavior).
      let cp := (request.getObjValAs? String "checkpoint").toOption.getD "?"
      let payload := (request.getObjVal? "payload").toOption.getD Json.null
      let cert ← runCheckpoint cp payload
      IO.println cert.toJson.compress
      return if cert.ok then 0 else 1
