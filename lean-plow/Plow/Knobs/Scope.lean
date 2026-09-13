/-
# Plow.Knobs.Scope — checkpoint S, the scope certificate.

A packet is compared per program pair. The Rust extractor pairs base and variant programs by key,
aligns their instructions by opcode (an inserted or removed instruction pairs with `none`), names
operands by tensor rather than by handle (a renumbered table moves no operand), and sends a pair
whose digests agree without a body. Those three are the trusted steps; everything below is about
what was sent.

`diff` names every difference as an element: the programs it touches, the field, the opcodes and,
for object facts and the tensor table, the facts that differ. A scope is a list of allowances;
S accepts when every element is allowed.

* `diff_complete`: `diff r = []` iff every pair and every route step is unchanged.
* `checkS_sound`: accepted ⇒ every element is inside the scope.
* `off_identity`: with no knob delta the scope is empty, so acceptance means identity.
* `untouched_rungs` / `route_untouched`: accepted ⇒ a program (or route step) no allowance selects
  is unchanged.
-/
import Lean.Data.Json
import Plow.Knobs.Formula

namespace Plow.Knobs.Scope

open Lean (Json)
open Plow.Knobs (field arrOf strOf natOf strsOf memB)

inductive Field where
  | cus | operands | shape | op | segments | tensorBytes | objectFacts | programSet | route
  deriving DecidableEq, Repr, Inhabited

structure Key where
  kind : String
  rows : Nat
  topology : String
  sparse : Bool
  deriving DecidableEq, Repr, Inhabited

/-- `t` are tensor ids shared by both packets, `seg` the instruction's segment, `tb` the byte size
    of each operand. -/
structure Inst where
  op : Nat
  blocks : Nat
  fj : List Nat
  t : List Nat
  i : List Nat
  seg : Nat
  tb : List Nat
  deriving DecidableEq, Repr, Inhabited

structure Body where
  insts : List (Option Inst × Option Inst)
  tensors : List String × List String
  facts : List String × List String
  deriving DecidableEq, Repr, Inhabited

/-- `body = none`: the two programs are identical, by digest. -/
structure Pair where
  a : Option Key
  b : Option Key
  body : Option Body
  deriving DecidableEq, Repr, Inhabited

structure RouteStep where
  label : String
  off : Option Key
  on : Option Key
  offSkip : List Nat
  onSkip : List Nat
  deriving DecidableEq, Repr, Inhabited

structure Elem where
  keys : List Key
  field : Field
  ops : List Nat
  facts : List String
  deriving DecidableEq, Repr, Inhabited

def Body.Unchanged (x : Body) : Prop :=
  (∀ q ∈ x.insts, q.1 = q.2) ∧ x.tensors.1 = x.tensors.2 ∧ x.facts.1 = x.facts.2

def Pair.Unchanged (p : Pair) : Prop :=
  p.a = p.b ∧ ∀ x, p.body = some x → x.Unchanged

def RouteStep.Unchanged (s : RouteStep) : Prop :=
  s.off = s.on ∧ s.offSkip = s.onSkip

def pairKeys (p : Pair) : List Key := p.a.toList ++ p.b.toList

/-! ## Diff -/

def symDiff (a b : List String) : List String :=
  a.filter (fun x => !memB x b) ++ b.filter (fun x => !memB x a)

def diffInst (ks : List Key) (a b : Inst) : List Elem :=
  let ops := if a.op = b.op then [a.op] else [a.op, b.op]
  (if a.op = b.op then [] else [⟨ks, .op, ops, []⟩]) ++
  (if a.blocks = b.blocks then [] else [⟨ks, .cus, ops, []⟩]) ++
  (if a.t = b.t then [] else [⟨ks, .operands, ops, []⟩]) ++
  (if a.i = b.i ∧ a.fj = b.fj then [] else [⟨ks, .shape, ops, []⟩]) ++
  (if a.seg = b.seg then [] else [⟨ks, .segments, ops, []⟩]) ++
  (if a.tb = b.tb then [] else [⟨ks, .tensorBytes, ops, []⟩])

def diffAligned (ks : List Key) : List (Option Inst × Option Inst) → List Elem
  | [] => []
  | (some x, some y) :: r => diffInst ks x y ++ diffAligned ks r
  | (some x, none) :: r => ⟨ks, .op, [x.op], []⟩ :: diffAligned ks r
  | (none, some y) :: r => ⟨ks, .op, [y.op], []⟩ :: diffAligned ks r
  | (none, none) :: r => diffAligned ks r

def diffBody (ks : List Key) (x : Body) : List Elem :=
  diffAligned ks x.insts ++
  (if x.tensors.1 = x.tensors.2 then [] else
    [⟨ks, .tensorBytes, [], symDiff x.tensors.1 x.tensors.2⟩]) ++
  (if x.facts.1 = x.facts.2 then [] else [⟨ks, .objectFacts, [], symDiff x.facts.1 x.facts.2⟩])

def diffPair (p : Pair) : List Elem :=
  (if p.a = p.b then [] else [⟨pairKeys p, .programSet, [], []⟩]) ++
  (match p.body with
   | none => []
   | some x => diffBody (pairKeys p) x)

def diffRoute (s : RouteStep) : List Elem :=
  if s.off = s.on ∧ s.offSkip = s.onSkip then [] else [⟨s.off.toList ++ s.on.toList, .route, [], []⟩]

def diffPairs : List Pair → List Elem
  | [] => []
  | p :: ps => diffPair p ++ diffPairs ps

def diffRoutes : List RouteStep → List Elem
  | [] => []
  | s :: ss => diffRoute s ++ diffRoutes ss

/-! ## Scope -/

inductive OpSel where
  | any
  | inClasses (cs : List String)
  | notIn (cs : List String)
  deriving Repr, Inhabited

/-- `facts`: when non-empty, every fact an element names must contain one of these strings. -/
structure Allow where
  kinds : List String
  rowsLo : Nat
  rowsHi : Nat
  topology : Option String
  sparse : Option Bool
  model : Option String
  ops : OpSel
  fields : List Field
  facts : List String
  deriving Repr, Inhabited

structure Req where
  model : String
  classes : List (Nat × List String)
  delta : List String
  scope : List Allow
  pairs : List Pair
  routes : List RouteStep

def selects (model : String) (al : Allow) (k : Key) : Bool :=
  (al.kinds.isEmpty || memB k.kind al.kinds) && Nat.ble al.rowsLo k.rows && Nat.ble k.rows al.rowsHi &&
  (match al.topology with | none => true | some t => decide (t = k.topology)) &&
  (match al.sparse with | none => true | some s => decide (s = k.sparse)) &&
  (match al.model with | none => true | some m => decide (m = model))

def classesOf (tbl : List (Nat × List String)) (op : Nat) : List String :=
  s!"op:{op}" :: ((tbl.find? (·.1 = op)).map (·.2)).getD []

def opOk (tbl : List (Nat × List String)) : OpSel → Nat → Bool
  | .any, _ => true
  | .inClasses cs, op => (classesOf tbl op).any (memB · cs)
  | .notIn cs, op => !(classesOf tbl op).any (memB · cs)

def hasInfix (needle hay : String) : Bool :=
  !needle.isEmpty && decide ((hay.splitOn needle).length > 1)

/-- An element with no program is never allowed: every difference must be attributed. -/
def allowedBy (model : String) (tbl : List (Nat × List String)) (al : Allow) (e : Elem) : Bool :=
  !e.keys.isEmpty && e.keys.all (selects model al) && al.fields.any (fun f => decide (f = e.field)) &&
  e.ops.all (opOk tbl al.ops) &&
  (al.facts.isEmpty || e.facts.all fun f => al.facts.any (hasInfix · f))

def allowed (model : String) (tbl : List (Nat × List String)) (scope : List Allow) (e : Elem) : Bool :=
  scope.any fun al => allowedBy model tbl al e

/-- A request without a knob delta is checked against the empty scope. -/
def effScope (r : Req) : List Allow := if r.delta.isEmpty then [] else r.scope

def diff (r : Req) : List Elem := diffPairs r.pairs ++ diffRoutes r.routes

def checkS (r : Req) : Except Elem (List Elem) :=
  if (diff r).all (allowed r.model r.classes (effScope r)) then .ok (diff r)
  else .error (((diff r).find? fun e => !allowed r.model r.classes (effScope r) e).getD default)

/-! ## Completeness -/

theorem diffInst_nil (ks : List Key) (a b : Inst) : diffInst ks a b = [] ↔ a = b := by
  cases a; cases b
  simp only [diffInst, List.append_eq_nil, Inst.mk.injEq]
  constructor
  · intro ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩
    split at h1 <;> split at h2 <;> split at h3 <;> split at h4 <;> split at h5 <;> split at h6 <;>
      simp_all
  · intro ⟨h1, h2, h3, h4, h5, h6, h7⟩
    simp [h1, h2, h3, h4, h5, h6, h7]

theorem diffAligned_nil (ks : List Key) : ∀ qs : List (Option Inst × Option Inst),
    diffAligned ks qs = [] ↔ ∀ q ∈ qs, q.1 = q.2
  | [] => by simp [diffAligned]
  | (some x, some y) :: r => by
    simp only [diffAligned, List.append_eq_nil, diffInst_nil, diffAligned_nil ks r, List.mem_cons,
      forall_eq_or_imp, Option.some.injEq]
  | (some x, none) :: r => by simp [diffAligned]
  | (none, some y) :: r => by simp [diffAligned]
  | (none, none) :: r => by simp [diffAligned, diffAligned_nil ks r]

theorem diffBody_nil (ks : List Key) (x : Body) : diffBody ks x = [] ↔ x.Unchanged := by
  unfold diffBody Body.Unchanged
  rw [List.append_eq_nil, List.append_eq_nil, diffAligned_nil]
  constructor
  · rintro ⟨⟨h1, h2⟩, h3⟩
    refine ⟨h1, ?_, ?_⟩
    · by_cases ht : x.tensors.1 = x.tensors.2
      · exact ht
      · simp [ht] at h2
    · by_cases hf : x.facts.1 = x.facts.2
      · exact hf
      · simp [hf] at h3
  · rintro ⟨h1, h2, h3⟩
    exact ⟨⟨h1, by simp [h2]⟩, by simp [h3]⟩

theorem diffPair_nil (p : Pair) : diffPair p = [] ↔ p.Unchanged := by
  cases p with
  | mk a b body =>
    cases body with
    | none =>
      simp only [diffPair, List.append_nil, Pair.Unchanged]
      constructor
      · intro h; split at h <;> simp_all
      · intro ⟨h, _⟩; simp [h]
    | some x =>
      simp only [diffPair, List.append_eq_nil, diffBody_nil, Pair.Unchanged, Option.some.injEq]
      constructor
      · intro ⟨h1, h2⟩
        split at h1
        · exact ⟨by assumption, fun y hy => hy ▸ h2⟩
        · simp at h1
      · intro ⟨h1, h2⟩
        exact ⟨by simp [h1], h2 x rfl⟩

theorem diffPairs_nil : ∀ ps : List Pair, diffPairs ps = [] ↔ ∀ p ∈ ps, p.Unchanged
  | [] => by simp [diffPairs]
  | p :: ps => by simp [diffPairs, List.append_eq_nil, diffPair_nil, diffPairs_nil ps]

theorem diffRoute_nil (s : RouteStep) : diffRoute s = [] ↔ s.Unchanged := by
  simp only [diffRoute, RouteStep.Unchanged]
  split <;> simp_all

theorem diffRoutes_nil : ∀ ss : List RouteStep, diffRoutes ss = [] ↔ ∀ s ∈ ss, s.Unchanged
  | [] => by simp [diffRoutes]
  | s :: ss => by simp [diffRoutes, List.append_eq_nil, diffRoute_nil, diffRoutes_nil ss]

/-- The diff misses nothing: it is empty exactly when every program pair and route step is
    unchanged. -/
theorem diff_complete (r : Req) :
    diff r = [] ↔ (∀ p ∈ r.pairs, p.Unchanged) ∧ ∀ s ∈ r.routes, s.Unchanged := by
  simp [diff, List.append_eq_nil, diffPairs_nil, diffRoutes_nil]

/-! ## Soundness -/

theorem checkS_sound (r : Req) (d : List Elem) (h : checkS r = .ok d) :
    d = diff r ∧ ∀ e ∈ d, allowed r.model r.classes (effScope r) e = true := by
  unfold checkS at h
  split at h
  · rename_i hall
    cases h
    exact ⟨rfl, List.all_eq_true.mp hall⟩
  · cases h

/-- With no knob delta, S accepts only the identical packet. -/
theorem off_identity (r : Req) (d : List Elem) (hd : r.delta = []) (h : checkS r = .ok d) :
    d = [] ∧ (∀ p ∈ r.pairs, p.Unchanged) ∧ ∀ s ∈ r.routes, s.Unchanged := by
  obtain ⟨rfl, hall⟩ := checkS_sound r d h
  have hnil : diff r = [] := by
    cases hdiff : diff r with
    | nil => rfl
    | cons e es =>
      have := hall e (by simp [hdiff])
      simp [allowed, effScope, hd] at this
  exact ⟨hnil, (diff_complete r).mp hnil⟩

theorem mem_diffPairs (e : Elem) (p : Pair) :
    ∀ ps : List Pair, p ∈ ps → e ∈ diffPair p → e ∈ diffPairs ps
  | [], hp, _ => absurd hp (List.not_mem_nil _)
  | q :: qs, hp, he => by
    rcases List.mem_cons.mp hp with rfl | hp
    · simp [diffPairs, he]
    · simp [diffPairs, mem_diffPairs e p qs hp he]

theorem mem_diffRoutes (e : Elem) (s : RouteStep) :
    ∀ ss : List RouteStep, s ∈ ss → e ∈ diffRoute s → e ∈ diffRoutes ss
  | [], hs, _ => absurd hs (List.not_mem_nil _)
  | q :: qs, hs, he => by
    rcases List.mem_cons.mp hs with rfl | hs
    · simp [diffRoutes, he]
    · simp [diffRoutes, mem_diffRoutes e s qs hs he]

theorem keys_diffInst (ks : List Key) (a b : Inst) : ∀ e ∈ diffInst ks a b, e.keys = ks := by
  intro e he
  simp only [diffInst, List.mem_append] at he
  rcases he with ((((h | h) | h) | h) | h) | h <;> (split at h <;> simp_all)

theorem keys_diffAligned (ks : List Key) : ∀ qs : List (Option Inst × Option Inst),
    ∀ e ∈ diffAligned ks qs, e.keys = ks
  | [] => by simp [diffAligned]
  | (some x, some y) :: r => by
    intro e he
    simp only [diffAligned, List.mem_append] at he
    rcases he with h | h
    · exact keys_diffInst ks x y e h
    · exact keys_diffAligned ks r e h
  | (some x, none) :: r => by
    intro e he
    simp only [diffAligned, List.mem_cons] at he
    rcases he with rfl | h
    · rfl
    · exact keys_diffAligned ks r e h
  | (none, some y) :: r => by
    intro e he
    simp only [diffAligned, List.mem_cons] at he
    rcases he with rfl | h
    · rfl
    · exact keys_diffAligned ks r e h
  | (none, none) :: r => by
    intro e he
    exact keys_diffAligned ks r e (by simpa [diffAligned] using he)

theorem keys_diffBody (ks : List Key) (x : Body) : ∀ e ∈ diffBody ks x, e.keys = ks := by
  intro e he
  simp only [diffBody, List.mem_append] at he
  rcases he with (h | h) | h
  · exact keys_diffAligned ks _ e h
  · split at h <;> simp_all
  · split at h <;> simp_all

theorem keys_diffPair (p : Pair) : ∀ e ∈ diffPair p, e.keys = pairKeys p := by
  intro e he
  simp only [diffPair, List.mem_append] at he
  rcases he with h | h
  · split at h <;> simp_all
  · split at h
    · simp at h
    · exact keys_diffBody _ _ e h

/-- An allowed element touches at least one program, and every program it touches is selected. -/
theorem allowed_selects (r : Req) (e : Elem) (h : allowed r.model r.classes (effScope r) e = true) :
    ∃ al ∈ effScope r, ∃ k ∈ e.keys, selects r.model al k = true := by
  simp only [allowed, List.any_eq_true] at h
  obtain ⟨al, hal, hby⟩ := h
  simp only [allowedBy, Bool.and_eq_true, List.all_eq_true] at hby
  obtain ⟨⟨⟨⟨hne, hkeys⟩, _⟩, _⟩, _⟩ := hby
  cases hek : e.keys with
  | nil => simp [hek] at hne
  | cons k ks => exact ⟨al, hal, k, by simp [hek], hkeys k (by simp [hek])⟩

/-- A program pair no allowance selects is unchanged in an accepted variant. -/
theorem untouched_rungs (r : Req) (d : List Elem) (h : checkS r = .ok d)
    (p : Pair) (hp : p ∈ r.pairs)
    (hsel : ∀ k ∈ pairKeys p, ∀ al ∈ effScope r, selects r.model al k = false) : p.Unchanged := by
  obtain ⟨rfl, hall⟩ := checkS_sound r d h
  refine (diffPair_nil p).mp ?_
  cases hdp : diffPair p with
  | nil => rfl
  | cons e es =>
    exfalso
    have he : e ∈ diffPair p := by simp [hdp]
    have hin : e ∈ diff r := by simp [diff, mem_diffPairs e p r.pairs hp he]
    obtain ⟨al, hal, k, hk, hs⟩ := allowed_selects r e (hall e hin)
    rw [keys_diffPair p e he] at hk
    rw [hsel k hk al hal] at hs
    exact Bool.false_ne_true hs

/-- A route step whose programs no allowance selects is unchanged in an accepted variant. -/
theorem route_untouched (r : Req) (d : List Elem) (h : checkS r = .ok d)
    (s : RouteStep) (hs : s ∈ r.routes)
    (hsel : ∀ k ∈ s.off.toList ++ s.on.toList, ∀ al ∈ effScope r, selects r.model al k = false) :
    s.Unchanged := by
  obtain ⟨rfl, hall⟩ := checkS_sound r d h
  refine (diffRoute_nil s).mp ?_
  cases hdr : diffRoute s with
  | nil => rfl
  | cons e es =>
    exfalso
    have he : e ∈ diffRoute s := by simp [hdr]
    have hin : e ∈ diff r := by
      simp only [diff, List.mem_append]
      exact Or.inr (mem_diffRoutes e s r.routes hs he)
    obtain ⟨al, hal, k, hk, hsl⟩ := allowed_selects r e (hall e hin)
    have hkeys : e.keys = s.off.toList ++ s.on.toList := by
      simp only [diffRoute] at he
      split at he <;> simp_all
    rw [hkeys] at hk
    rw [hsel k hk al hal] at hsl
    exact Bool.false_ne_true hsl

/-! ## JSON -/

def parseField : String → Except String Field
  | "cus" => .ok .cus | "operands" => .ok .operands | "shape" => .ok .shape | "op" => .ok .op
  | "segments" => .ok .segments | "tensor_bytes" => .ok .tensorBytes
  | "object_facts" => .ok .objectFacts | "program_set" => .ok .programSet | "route" => .ok .route
  | s => .error s!"unknown scope field '{s}'"

def fieldName : Field → String
  | .cus => "cus" | .operands => "operands" | .shape => "shape" | .op => "op"
  | .segments => "segments" | .tensorBytes => "tensor_bytes" | .objectFacts => "object_facts"
  | .programSet => "program_set" | .route => "route"

def natsOf (ctx : String) (j : Json) : Except String (List Nat) := do
  (← arrOf ctx j).mapM (natOf ctx)

def parseKey (j : Json) : Except String Key := do
  pure { kind := ← strOf "kind" (← field j "kind"), rows := ← natOf "rows" (← field j "rows"),
         topology := ← strOf "topology" (← field j "topology"),
         sparse := ← match ← field j "sparse" with
           | .bool b => pure b
           | _ => throw "sparse: expected a bool" }

def optKey (j : Json) : Except String (Option Key) :=
  if j.isNull then .ok none else (parseKey j).map some

def parseInst (j : Json) : Except String (Option Inst) := do
  if j.isNull then return none
  match ← arrOf "inst" j with
  | [op, blocks, fj, t, i, seg, tb] =>
    pure (some { op := ← natOf "op" op, blocks := ← natOf "blocks" blocks, fj := ← natsOf "fj" fj,
                 t := ← natsOf "t" t, i := ← natsOf "i" i, seg := ← natOf "seg" seg,
                 tb := ← natsOf "tb" tb })
  | _ => throw "inst: expected [op, blocks, fj, t, i, seg, tb]"

def parseSides (ctx : String) (j : Json) : Except String (List String × List String) := do
  match ← arrOf ctx j with
  | [a, b] => pure (← strsOf ctx a, ← strsOf ctx b)
  | _ => throw s!"{ctx}: expected [base, variant]"

def parsePair (j : Json) : Except String Pair := do
  let a ← optKey (← field j "a")
  let b ← optKey (← field j "b")
  let body ← match j.getObjVal? "body" with
    | .error _ | .ok .null => pure none
    | .ok x => pure (some {
        insts := ← (← arrOf "insts" (← field x "insts")).mapM fun q => do
          match ← arrOf "aligned inst" q with
          | [p, v] => pure (← parseInst p, ← parseInst v)
          | _ => throw "aligned inst: expected [base, variant]",
        tensors := ← parseSides "tensors" (← field x "tensors"),
        facts := ← parseSides "facts" (← field x "facts") })
  pure { a, b, body }

def parseAllow (j : Json) : Except String Allow := do
  let optStr (k : String) : Except String (Option String) :=
    match j.getObjVal? k with | .ok (.str s) => .ok (some s) | _ => .ok none
  let ops ← match j.getObjVal? "ops" with
    | .error _ | .ok .null => pure OpSel.any
    | .ok o => match o.getObjVal? "in", o.getObjVal? "not_in" with
      | .ok cs, _ => pure (.inClasses (← strsOf "ops.in" cs))
      | _, .ok cs => pure (.notIn (← strsOf "ops.not_in" cs))
      | _, _ => throw "ops: expected {in: [..]} or {not_in: [..]}"
  pure { kinds := ← match j.getObjVal? "kinds" with
           | .ok ks => strsOf "kinds" ks
           | .error _ => pure []
         rowsLo := (j.getObjValAs? Nat "rows_min").toOption.getD 0
         rowsHi := (j.getObjValAs? Nat "rows_max").toOption.getD (2 ^ 64)
         topology := ← optStr "topology"
         sparse := (j.getObjValAs? Bool "sparse").toOption
         model := ← optStr "model"
         ops
         fields := ← (← arrOf "fields" (← field j "fields")).mapM fun f => do
           parseField (← strOf "field" f)
         facts := ← match j.getObjVal? "facts" with
           | .ok fs => strsOf "facts" fs
           | .error _ => pure [] }

def parseReq (j : Json) : Except String Req := do
  let classes ← (← arrOf "classes" (← field j "classes")).mapM fun c => do
    match ← arrOf "class entry" c with
    | [op, cs] => pure (← natOf "class op" op, ← strsOf "classes" cs)
    | _ => throw "class entry: expected [op, [class..]]"
  let routes ← match j.getObjVal? "routes" with
    | .error _ => pure []
    | .ok rs => (← arrOf "routes" rs).mapM fun s => do
      pure { label := ← strOf "label" (← field s "label"),
             off := ← optKey (← field s "off"), on := ← optKey (← field s "on"),
             offSkip := ← natsOf "off_skip" (← field s "off_skip"),
             onSkip := ← natsOf "on_skip" (← field s "on_skip") }
  pure { model := ← strOf "model" (← field j "model"), classes,
         delta := ← strsOf "delta" (← field j "delta"),
         scope := ← (← arrOf "scope" (← field j "scope")).mapM parseAllow,
         pairs := ← (← arrOf "pairs" (← field j "pairs")).mapM parsePair,
         routes }

def renderKey (k : Key) : String :=
  s!"{k.kind}/{k.rows}/{k.topology}{if k.sparse then "/sparse" else ""}"

def renderElem (e : Elem) : String :=
  s!"{fieldName e.field} on {e.keys.map renderKey} ops {e.ops}" ++
  (if e.facts.isEmpty then "" else s!" facts {e.facts.take 8}")

/-- Counts per (programs, field, opcodes), for the certificate notes. -/
def summarize (d : List Elem) : List (String × Nat) :=
  d.foldl (fun acc e =>
    let label := s!"{(e.keys.map renderKey)}:{fieldName e.field}:{e.ops}"
    match acc.find? (·.1 = label) with
    | some _ => acc.map fun (l, n) => if l = label then (l, n + 1) else (l, n)
    | none => acc ++ [(label, 1)]) []

def runS (payload : Json) : Except String String := do
  let r ← parseReq payload
  match checkS r with
  | .error e =>
    throw s!"diff element outside the declared scope: {renderElem e}{if r.delta.isEmpty then " (no knob delta: the scope is empty and the packets must be identical)" else ""}"
  | .ok d =>
    let unchanged := (r.pairs.filter fun p => (diffPair p).isEmpty).length
    let emptyEffect := !r.delta.isEmpty && d.isEmpty
    let slack := (effScope r).enum.filterMap fun (i, al) =>
      if d.any (allowedBy r.model r.classes al) then none else some i
    let warn := (if emptyEffect then ["empty_effect"] else []) ++
      (if slack.isEmpty then [] else [s!"scope_slack: allowances {slack} matched no difference"])
    let facts := d.foldl (fun acc e => acc ++ e.facts.filter (fun f => !memB f acc)) []
    return s!"{d.length} differences, all in scope; {unchanged}/{r.pairs.length} programs unchanged; \
      warnings={warn}; changed={summarize d}{if facts.isEmpty then "" else s!"; facts={facts.take 16}"}"

end Plow.Knobs.Scope
