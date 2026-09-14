/-
# Plow.Knobs.Formula — constraint evaluation over a resolved config and a target.

`eval_congr` is the one fact the rest leans on: a formula reads the config only through its
`vars`, so two configs that agree there evaluate it identically.
-/
import Plow.Knobs.Schema

namespace Plow.Knobs

def find : Config → String → Option Val
  | [], _ => none
  | (k, v) :: c, x => if x = k then some v else find c x

def get (c : Config) (x : String) : Val := (find c x).getD .unset

def memB (x : String) : List String → Bool
  | [] => false
  | y :: ys => decide (x = y) || memB x ys

theorem memB_iff (x : String) : ∀ l : List String, memB x l = true ↔ x ∈ l
  | [] => by simp [memB]
  | y :: ys => by simp [memB, memB_iff x ys]

theorem memB_eq_false_iff (x : String) (l : List String) : memB x l = false ↔ x ∉ l := by
  rw [← memB_iff]; simp

/-- `unset` is admitted by every domain: it is "no value given", which an optional knob is. -/
def admits : Domain → Val → Bool
  | _, .unset => true
  | .bool, .bool _ => true
  | .nat lo hi, .nat n => Nat.ble lo n && Nat.ble n hi
  | .enum vs, .str s => memB s vs
  | .list vs, .str s => (s.splitOn ",").all (fun x => memB x vs)
  | .str, .str _ => true
  | _, _ => false

/-- Ordering comparisons hold only between two naturals; `unset > 1` is false. -/
def cmpVal : Cmp → Val → Val → Bool
  | .eq, a, b => decide (a = b)
  | .ne, a, b => !decide (a = b)
  | .lt, .nat a, .nat b => Nat.blt a b
  | .le, .nat a, .nat b => Nat.ble a b
  | .gt, .nat a, .nat b => Nat.blt b a
  | .ge, .nat a, .nat b => Nat.ble b a
  | _, _, _ => false

def evalT (t : Target) : TAtom → Bool
  | .arch s => decide (t.arch = s)
  | .tp n => decide (t.tp = n)
  | .nCu n => decide (t.nCu = n)
  | .model s => decide (t.model = s)
  | .cap s => memB s t.caps

def eval (c : Config) (t : Target) : Formula → Bool
  | .tt => true
  | .atom k op v => cmpVal op (get c k) v
  | .tgt a => evalT t a
  | .not f => !eval c t f
  | .and a b => eval c t a && eval c t b
  | .or a b => eval c t a || eval c t b
  | .implies a b => !eval c t a || eval c t b

def vars : Formula → List String
  | .tt => []
  | .atom k _ _ => [k]
  | .tgt _ => []
  | .not f => vars f
  | .and a b => vars a ++ vars b
  | .or a b => vars a ++ vars b
  | .implies a b => vars a ++ vars b

theorem eval_congr (t : Target) (c₁ c₂ : Config) :
    ∀ f : Formula, (∀ x ∈ vars f, get c₁ x = get c₂ x) → eval c₁ t f = eval c₂ t f
  | .tt, _ => rfl
  | .atom k op v, h => by simp only [eval]; rw [h k (by simp [vars])]
  | .tgt _, _ => rfl
  | .not f, h => by
      simp only [eval]; rw [eval_congr t c₁ c₂ f (fun x hx => h x (by simpa [vars] using hx))]
  | .and a b, h => by
      simp only [eval]
      rw [eval_congr t c₁ c₂ a (fun x hx => h x (by simp [vars, hx])),
          eval_congr t c₁ c₂ b (fun x hx => h x (by simp [vars, hx]))]
  | .or a b, h => by
      simp only [eval]
      rw [eval_congr t c₁ c₂ a (fun x hx => h x (by simp [vars, hx])),
          eval_congr t c₁ c₂ b (fun x hx => h x (by simp [vars, hx]))]
  | .implies a b, h => by
      simp only [eval]
      rw [eval_congr t c₁ c₂ a (fun x hx => h x (by simp [vars, hx])),
          eval_congr t c₁ c₂ b (fun x hx => h x (by simp [vars, hx]))]

/-- The constants a formula compares `x` against — the boundaries of `x`'s finite abstraction. -/
def constsFor (x : String) : Formula → List Val
  | .atom k _ v => if k = x then [v] else []
  | .not f => constsFor x f
  | .and a b => constsFor x a ++ constsFor x b
  | .or a b => constsFor x a ++ constsFor x b
  | .implies a b => constsFor x a ++ constsFor x b
  | _ => []

end Plow.Knobs
