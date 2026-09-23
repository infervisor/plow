import Lean.Data.Json

namespace Plow.RewriteBody

open Lean (Json)

inductive Term where
  | atom (kind value : String)
  | node (head : String) (args : List Term)
  deriving Repr

mutual
  def termEq : (a b : Term) → Decidable (a = b)
    | .atom ka va, .atom kb vb =>
      match decEq ka kb, decEq va vb with
      | .isTrue hk, .isTrue hv => .isTrue (by cases hk; cases hv; rfl)
      | .isFalse hk, _ => .isFalse (fun h => hk (Term.atom.inj h).1)
      | _, .isFalse hv => .isFalse (fun h => hv (Term.atom.inj h).2)
    | .node ha aa, .node hb ab =>
      match decEq ha hb, termsEq aa ab with
      | .isTrue hh, .isTrue hl => .isTrue (by cases hh; cases hl; rfl)
      | .isFalse hh, _ => .isFalse (fun h => hh (Term.node.inj h).1)
      | _, .isFalse hl => .isFalse (fun h => hl (Term.node.inj h).2)
    | .atom _ _, .node _ _ => .isFalse (by intro h; cases h)
    | .node _ _, .atom _ _ => .isFalse (by intro h; cases h)
  termination_by a b => sizeOf a + sizeOf b

  def termsEq : (a b : List Term) → Decidable (a = b)
    | [], [] => .isTrue rfl
    | a :: aa, b :: bb =>
      match termEq a b, termsEq aa bb with
      | .isTrue hh, .isTrue ht => .isTrue (by cases hh; cases ht; rfl)
      | .isFalse hh, _ => .isFalse (fun h => hh (List.cons.inj h).1)
      | _, .isFalse ht => .isFalse (fun h => ht (List.cons.inj h).2)
    | [], _ :: _ => .isFalse (by intro h; cases h)
    | _ :: _, [] => .isFalse (by intro h; cases h)
  termination_by a b => sizeOf a + sizeOf b
end

instance : DecidableEq Term := termEq

private def str (value : String) : Term := .atom "string" value
private def ew (kind : String) (a b : Term) : Term := .node "Ew" [str kind, a, b]
private def act (kind : Term) (x : Term) : Term := .node "Act" [kind, x]
private def rms (x w eps : Term) : Term := .node "RmsNorm" [x, w, eps]

-- Full engine arities: shape/scale/activation operands and nested residual
-- materialization nodes are retained. No arithmetic reassociation is modeled.
def unfold (head : String) (args : List Term) : Term :=
  match head, args with
  | "FusedNormLinear", [x,w,wl,e,n] => .node "Linear" [rms x w e,wl,n]
  | "FusedZeroCenteredNormLinear", [x,w,wl,e,n] =>
      .node "Linear" [.node "ZeroCenteredRmsNorm" [x,w,e],wl,n]
  | "FusedNormLinearBias", [x,w,wl,bl,e,n] => .node "LinearBias" [rms x w e,wl,bl,n]
  | "FusedLayerNormLinear", [x,w,b,wl,e,n] => .node "Linear" [.node "LayerNorm" [x,w,b,e],wl,n]
  | "FusedLayerNormLinearBias", [x,w,b,wl,bl,e,n] =>
      .node "LinearBias" [.node "LayerNorm" [x,w,b,e],wl,bl,n]
  | "SwiGLU", [k,g,u] => ew "mul" (act k g) u
  | "FusedGroupNormAct", [x,w,b,g,e,k] => act k (.node "GroupNorm" [x,w,b,g,e])
  | "FusedAdaLN", [x,s,b] => ew "add" (ew "add" (ew "mul" x s) x) b
  | "FusedGatedResidual", [x,y,g] => ew "add" x (ew "mul" y g)
  | "FusedNormRope", [x,w,e,d,t] => .node "Rope" [rms x w e,d,t]
  | "FusedZeroCenteredNormRope", [x,w,e,d,t] =>
      .node "Rope" [.node "ZeroCenteredRmsNorm" [x,w,e],d,t]
  | "FusedNormRopeScale", [x,w,e,d,t,s] => .node "Scale" [.node "Rope" [rms x w e,d,t],s]
  | "FusedResidualNorm", [a,b,w,e] => rms (ew "add" a b) w e
  | "FusedResidualZeroCenteredNorm", [a,b,w,e] => .node "ZeroCenteredRmsNorm" [ew "add" a b,w,e]
  | "FusedResidualLayerNorm", [a,b,w,bias,e] => .node "LayerNorm" [ew "add" a b,w,bias,e]
  | "FusedResidual3Norm", [x,a,b,w,e] => rms (ew "add" x (ew "add" a b)) w e
  | "FusedNormResidualNorm", [a,b,w1,e1,w2,e2] => rms (ew "add" a (rms b w1 e1)) w2 e2
  | "FusedNormResidualScaleNorm", [a,b,w1,e1,s,w2,e2] =>
      rms (ew "mul" (ew "add" a (rms b w1 e1)) s) w2 e2
  | "FusedGroupNormActConv3d", [x,w,b,g,e,k,cw,s,p] =>
      .node "Conv3d" [act k (.node "GroupNorm" [x,w,b,g,e]),cw,s,p]
  | "FusedGroupNormActConv3dBias", [x,w,b,g,e,k,cw,cb,s,p] =>
      .node "Conv3dBias" [act k (.node "GroupNorm" [x,w,b,g,e]),cw,cb,s,p]
  | "FusedLinearAct", [x,w,n,k] => act k (.node "Linear" [x,w,n])
  | "FusedLinearBiasAct", [x,w,b,n,k] => act k (.node "LinearBias" [x,w,b,n])
  | "FusedEmbeddingScale", [ids,table,s] => .node "Scale" [.node "Embedding" [ids,table],s]
  | "FusedKdaGatedNorm", [o,nw,e,x,gw,n,shape] =>
      ew "mul" (rms o nw e) (act (str "sigmoid") (.node "Reshape" [.node "Linear" [x,gw,n],shape]))
  | "FusedMlaOutGate", [attn,x,gw,n] => ew "mul" attn (act (str "sigmoid") (.node "Linear" [x,gw,n]))
  | "FusedRmsNormSiluGate", [x,nw,e,g,shape] => ew "mul" (rms x nw e) (act (str "silu") (.node "Reshape" [g,shape]))
  | "FusedPackedAttnGate", [m,x,qw,n,ps,axis,start,len,gs] =>
      ew "mul" m (act (str "sigmoid") (.node "Reshape" [
        .node "Slice" [.node "Reshape" [.node "Linear" [x,qw,n],ps],axis,start,len],gs]))
  | "FusedMaterializedResidualBlock", [a,b,s,nw,pw,max] =>
      .node "BlockResidual" [ew "add" a b,s,nw,pw,max]
  | "FusedMaterializedResidual3Block", [pre,a,b,s,nw,pw,max] =>
      .node "BlockResidual" [ew "add" pre (ew "add" a b),s,nw,pw,max]
  | _, _ => .node head args

def normalize : Nat → Term → Term
  | 0, t => t
  | _ + 1, .atom k v => .atom k v
  | fuel + 1, .node h args => unfold h (args.map (normalize fuel))

def check (lhs rhs : Term) : Bool := decide (normalize 64 lhs = normalize 64 rhs)

theorem check_sound (lhs rhs : Term) (h : check lhs rhs = true) :
    normalize 64 lhs = normalize 64 rhs := of_decide_eq_true h

def parse : Nat → Json → Except String Term
  | 0, _ => throw "rewrite term exceeds supported nesting depth"
  | fuel + 1, value => do
    let fields ← value.getArr?
    if fields.size != 3 then throw "rewrite term requires kind, value and operands"
    let kind ← fields[0]!.getStr?
    let text ← fields[1]!.getStr?
    let args ← fields[2]!.getArr?
    if kind == "call" then
      return .node text (← args.toList.mapM (parse fuel))
    if !args.isEmpty || !( ["variable", "string", "integer", "float_bits", "boolean", "unit"].contains kind) then
      throw "invalid rewrite atom"
    return .atom kind text

def run (rules : List String) (value : Json) : Except String String := do
  let bodies ← value.getArr?
  if bodies.size != rules.length || bodies.isEmpty then throw "rewrite body catalog is incomplete"
  for (name, body) in rules.zip bodies.toList do
    if (← body.getObjValAs? String "name") != name then throw "rewrite name/body order mismatch"
    let lhs ← parse 64 (← body.getObjVal? "lhs")
    let rhs ← parse 64 (← body.getObjVal? "rhs")
    if !check lhs rhs then throw s!"rewrite '{name}' changes its full-arity expanded syntax"
  return s!"{bodies.size} actual rewrite bodies checked by RewriteBody.check_sound; retained expression tree/attributes only, not floating-point kernel implementation"

end Plow.RewriteBody
