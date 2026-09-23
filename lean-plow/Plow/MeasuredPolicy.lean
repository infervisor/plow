import Plow.Knobs.Ledger

namespace Plow.MeasuredPolicy

open Plow.Knobs.Ledger

structure Candidate where
  domain : String
  key : String
  cost : Q
  qualified : Bool

structure Choice where
  domain : String
  key : String

def minimumFor (candidates : List Candidate) (selected : Candidate) : Bool :=
  candidates.all fun candidate =>
    !(candidate.qualified && decide (candidate.domain = selected.domain)) || Q.le selected.cost candidate.cost

def check (required : List String) (candidates : List Candidate) (choices : List Choice) : Bool :=
  required.all fun domain => candidates.any fun selected =>
    decide (selected.domain = domain) && selected.qualified &&
    choices.any (fun choice => decide (choice.domain = domain) && decide (choice.key = selected.key)) &&
    minimumFor candidates selected

theorem minimumFor_sound (candidates : List Candidate) (selected : Candidate)
    (h : minimumFor candidates selected = true) :
    ∀ candidate ∈ candidates, candidate.qualified = true → candidate.domain = selected.domain →
      selected.cost.num * candidate.cost.den ≤ candidate.cost.num * selected.cost.den := by
  intro candidate hm qualified domain
  have checked := List.all_eq_true.mp h candidate hm
  simpa [qualified, domain, Q.le] using checked

theorem check_sound (required : List String) (candidates : List Candidate) (choices : List Choice)
    (h : check required candidates choices = true) :
    ∀ domain ∈ required, ∃ selected ∈ candidates, selected.domain = domain ∧ selected.qualified = true ∧
      (∃ choice ∈ choices, choice.domain = domain ∧ choice.key = selected.key) ∧
      (∀ candidate ∈ candidates, candidate.qualified = true → candidate.domain = selected.domain →
        selected.cost.num * candidate.cost.den ≤ candidate.cost.num * selected.cost.den) := by
  intro domain hd
  have checked := List.all_eq_true.mp h domain hd
  simp only [List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq] at checked
  obtain ⟨selected, hs, ⟨⟨same, qualified⟩, choice⟩, minimum⟩ := checked
  exact ⟨selected, hs, same, qualified, choice, minimumFor_sound candidates selected minimum⟩

open Lean (Json)

def run (j : Json) : Except String String := do
  let required ← j.getObjValAs? (List String) "required"
  let candidateJson ← j.getObjValAs? (List Json) "candidates"
  let choiceJson ← j.getObjValAs? (List Json) "choices"
  let candidates ← candidateJson.mapM fun raw => do
    let cost ← parseQ "candidate.cost" (← raw.getObjVal? "cost")
    if !Q.lt (Q.ofNat 0) cost then throw "candidate cost must be positive"
    let domain ← raw.getObjValAs? String "domain"
    let key ← raw.getObjValAs? String "key"
    if domain.isEmpty || key.isEmpty then throw "empty candidate identity"
    let qualified ← raw.getObjValAs? Bool "qualified"
    pure ({domain, key, cost, qualified} : Candidate)
  let choices ← choiceJson.mapM fun raw => do
    pure ({domain := ← raw.getObjValAs? String "domain", key := ← raw.getObjValAs? String "key"} : Choice)
  if required.isEmpty || required.any String.isEmpty then throw "empty required domain coverage"
  for domain in required do
    if (required.filter (· == domain)).length != 1 then throw "duplicate required domain"
  for candidate in candidates do
    if (candidates.filter fun c => c.domain == candidate.domain && c.key == candidate.key).length != 1 then
      throw "duplicate candidate identity"
  if choices.length != required.length then throw "incomplete policy domain coverage"
  for choice in choices do
    if !required.contains choice.domain || (choices.filter (·.domain == choice.domain)).length != 1 then
      throw "unknown or duplicate choice domain"
  if !check required candidates choices then throw "policy does not select a qualified supplied minimum for every required domain"
  return s!"check_sound: {required.length} required domains covered; selected costs minimal among supplied qualified measurements only; no hardware optimality or evidence-authenticity claim"

end Plow.MeasuredPolicy
