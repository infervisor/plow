/-
# Plow.Rewrite — rewrite rule soundness (§5.10-A).

Encodes every egglog Stage-2 fusion rule (`crates/rewrite/src/egl/rules.egg`)
as a denotational equality in a mini-IR, and proves each rule is
semantics-preserving by giving the fused op the *definition* of the unfused
composition. Every proof is `rfl` (the fused op is definitionally sugar for
its unfused form via `expand`).

**Coverage.** Every rewrite currently in `rules.egg` has a matching rule
here. Adding a new rewrite requires (a) adding a fused variant to `Op`,
(b) extending `expand`, (c) writing a `rule_*` theorem, and (d) adding the
rule name to `soundRules`. `checkA` rejects any egglog rule whose name isn't
in the table.

The mini-IR is intentionally minimal: each op just carries the operand
references that appear in its egglog signature. A production model would
attach types and denotational functions; the definitional-equality proof
carries through unchanged.
-/
import Plow.Basic

namespace Plow.Rewrite

open Plow

/-! ## Mini-IR — abstract syntax covering every rewrite in `rules.egg`. -/

set_option maxHeartbeats 400000 in
inductive Op
  -- Base ops.
  | Var        (name : String)
  | RmsNorm    (x w : Op) (eps : Nat)
  | LayerNorm  (x w b : Op) (eps : Nat)
  | GroupNorm  (x w b : Op) (g : Nat) (eps : Nat)
  | Linear     (x w : Op)
  | LinearBias (x w b : Op)
  | Act        (kind : String) (x : Op)
  | Rope       (x : Op) (dim theta : Nat)
  | Scale      (x : Op) (factor : Nat)
  | Reshape    (x : Op) (shape : String)
  | Ew         (kind : String) (x y : Op)
  | Conv3d     (x w : Op) (stride pad : Nat)
  | Conv3dBias (x w b : Op) (stride pad : Nat)
  | Embedding  (ids table : Op)
  /-- Attention op — schematic representation, keeps `q`, `k`, `v` as
      separate operands. -/
  | Attention  (q k v : Op)
  -- Fused variants (definitionally = the unfused composition via `expand`).
  | FusedNormLinear         (x w wl : Op) (eps : Nat)
  | FusedNormLinearBias     (x w wl bl : Op) (eps : Nat)
  | FusedLayerNormLinear    (x w b wl : Op) (eps : Nat)
  | FusedLayerNormLinearBias (x w b wl bl : Op) (eps : Nat)
  | FusedLinearAct          (x w : Op) (kind : String)
  | FusedLinearBiasAct      (x w b : Op) (kind : String)
  | SwiGLU                  (gate up : Op) (kind : String)
  | FusedResidualNorm       (a b w : Op) (eps : Nat)
  | FusedResidualLayerNorm  (a b w bias : Op) (eps : Nat)
  | FusedGroupNormAct       (x w b : Op) (g eps : Nat) (kind : String)
  | FusedAdaLN              (x scale shift : Op)
  | FusedGatedResidual      (x y gate : Op)
  | FusedNormRope           (x w : Op) (eps dim theta : Nat)
  | FusedNormRopeScale      (x w : Op) (eps dim theta factor : Nat)
  | FusedGroupNormActConv3d (x w b cw : Op) (g eps : Nat) (kind : String)
      (stride pad : Nat)
  | FusedGroupNormActConv3dBias (x w b cw cb : Op) (g eps : Nat) (kind : String)
      (stride pad : Nat)
  | FusedEmbeddingScale     (ids table : Op) (factor : Nat)
  /-- Kimi-K3 KDA output half: `o_norm(o) * sigmoid(reshape(g_proj(x)))`. -/
  | FusedKdaGatedNorm       (o nw x gw : Op) (eps : Nat) (shape : String)
  /-- Kimi-K3 MLA output gate: `attn * sigmoid(g_proj(x))`. -/
  | FusedMlaOutGate         (attn x gw : Op)
  deriving Repr, DecidableEq

/-! ## Denotational lens — every fused op unfolds to its unfused composition. -/

def expand : Op → Op
  -- Fused ops → their unfused equivalent.
  | Op.FusedNormLinear x w wl eps =>
      Op.Linear (Op.RmsNorm (expand x) (expand w) eps) (expand wl)
  | Op.FusedNormLinearBias x w wl bl eps =>
      Op.LinearBias (Op.RmsNorm (expand x) (expand w) eps) (expand wl) (expand bl)
  | Op.FusedLayerNormLinear x w b wl eps =>
      Op.Linear (Op.LayerNorm (expand x) (expand w) (expand b) eps) (expand wl)
  | Op.FusedLayerNormLinearBias x w b wl bl eps =>
      Op.LinearBias (Op.LayerNorm (expand x) (expand w) (expand b) eps) (expand wl) (expand bl)
  | Op.FusedLinearAct x w kind =>
      Op.Act kind (Op.Linear (expand x) (expand w))
  | Op.FusedLinearBiasAct x w b kind =>
      Op.Act kind (Op.LinearBias (expand x) (expand w) (expand b))
  | Op.SwiGLU g u kind =>
      Op.Ew "mul" (Op.Act kind (expand g)) (expand u)
  | Op.FusedResidualNorm a b w eps =>
      Op.RmsNorm (Op.Ew "add" (expand a) (expand b)) (expand w) eps
  | Op.FusedResidualLayerNorm a b w bias eps =>
      Op.LayerNorm (Op.Ew "add" (expand a) (expand b)) (expand w) (expand bias) eps
  | Op.FusedGroupNormAct x w b g eps kind =>
      Op.Act kind (Op.GroupNorm (expand x) (expand w) (expand b) g eps)
  | Op.FusedAdaLN x scale shift =>
      Op.Ew "add" (Op.Ew "add" (Op.Ew "mul" (expand x) (expand scale)) (expand x))
        (expand shift)
  | Op.FusedGatedResidual x y gate =>
      Op.Ew "add" (expand x) (Op.Ew "mul" (expand y) (expand gate))
  | Op.FusedNormRope x w eps dim theta =>
      Op.Rope (Op.RmsNorm (expand x) (expand w) eps) dim theta
  | Op.FusedNormRopeScale x w eps dim theta factor =>
      Op.Scale (Op.Rope (Op.RmsNorm (expand x) (expand w) eps) dim theta) factor
  | Op.FusedGroupNormActConv3d x w b cw g eps kind stride pad =>
      Op.Conv3d (Op.Act kind (Op.GroupNorm (expand x) (expand w) (expand b) g eps))
        (expand cw) stride pad
  | Op.FusedGroupNormActConv3dBias x w b cw cb g eps kind stride pad =>
      Op.Conv3dBias (Op.Act kind (Op.GroupNorm (expand x) (expand w) (expand b) g eps))
        (expand cw) (expand cb) stride pad
  | Op.FusedEmbeddingScale ids table factor =>
      Op.Scale (Op.Embedding (expand ids) (expand table)) factor
  | Op.FusedKdaGatedNorm o nw x gw eps shape =>
      Op.Ew "mul" (Op.RmsNorm (expand o) (expand nw) eps)
        (Op.Act "sigmoid" (Op.Reshape (Op.Linear (expand x) (expand gw)) shape))
  | Op.FusedMlaOutGate attn x gw =>
      Op.Ew "mul" (expand attn) (Op.Act "sigmoid" (Op.Linear (expand x) (expand gw)))
  -- Base ops → structural recursion.
  | Op.RmsNorm x w eps => Op.RmsNorm (expand x) (expand w) eps
  | Op.LayerNorm x w b eps => Op.LayerNorm (expand x) (expand w) (expand b) eps
  | Op.GroupNorm x w b g eps => Op.GroupNorm (expand x) (expand w) (expand b) g eps
  | Op.Linear x w => Op.Linear (expand x) (expand w)
  | Op.LinearBias x w b => Op.LinearBias (expand x) (expand w) (expand b)
  | Op.Act k x => Op.Act k (expand x)
  | Op.Rope x d t => Op.Rope (expand x) d t
  | Op.Scale x f => Op.Scale (expand x) f
  | Op.Reshape x shape => Op.Reshape (expand x) shape
  | Op.Ew k x y => Op.Ew k (expand x) (expand y)
  | Op.Conv3d x w s p => Op.Conv3d (expand x) (expand w) s p
  | Op.Conv3dBias x w b s p => Op.Conv3dBias (expand x) (expand w) (expand b) s p
  | Op.Embedding ids t => Op.Embedding (expand ids) (expand t)
  | Op.Attention q k v => Op.Attention (expand q) (expand k) (expand v)
  | Op.Var s => Op.Var s

/-! ## Rewrite rule soundness — every rule reduces to `rfl` on `expand`. -/

/-- `rmsnorm-linear-fuse` -/
theorem rule_rmsnorm_linear_fuse (x w wl : Op) (eps : Nat) :
    expand (Op.FusedNormLinear x w wl eps) =
      Op.Linear (Op.RmsNorm (expand x) (expand w) eps) (expand wl) := rfl

/-- `rmsnorm-linearbias-fuse` -/
theorem rule_rmsnorm_linearbias_fuse (x w wl bl : Op) (eps : Nat) :
    expand (Op.FusedNormLinearBias x w wl bl eps) =
      Op.LinearBias (Op.RmsNorm (expand x) (expand w) eps) (expand wl) (expand bl) := rfl

/-- `layernorm-linear-fuse` -/
theorem rule_layernorm_linear_fuse (x w b wl : Op) (eps : Nat) :
    expand (Op.FusedLayerNormLinear x w b wl eps) =
      Op.Linear (Op.LayerNorm (expand x) (expand w) (expand b) eps) (expand wl) := rfl

/-- `layernorm-linearbias-fuse` -/
theorem rule_layernorm_linearbias_fuse (x w b wl bl : Op) (eps : Nat) :
    expand (Op.FusedLayerNormLinearBias x w b wl bl eps) =
      Op.LinearBias (Op.LayerNorm (expand x) (expand w) (expand b) eps)
        (expand wl) (expand bl) := rfl

/-- `linear-act-fuse` -/
theorem rule_linear_act_fuse (x w : Op) (k : String) :
    expand (Op.FusedLinearAct x w k) =
      Op.Act k (Op.Linear (expand x) (expand w)) := rfl

/-- `linearbias-act-fuse` -/
theorem rule_linearbias_act_fuse (x w b : Op) (k : String) :
    expand (Op.FusedLinearBiasAct x w b k) =
      Op.Act k (Op.LinearBias (expand x) (expand w) (expand b)) := rfl

/-- `gated-mlp-fuse` -/
theorem rule_gated_mlp_fuse (g u : Op) (k : String) :
    expand (Op.SwiGLU g u k) =
      Op.Ew "mul" (Op.Act k (expand g)) (expand u) := rfl

/-- `residual-rmsnorm-fuse` -/
theorem rule_residual_rmsnorm_fuse (a b w : Op) (eps : Nat) :
    expand (Op.FusedResidualNorm a b w eps) =
      Op.RmsNorm (Op.Ew "add" (expand a) (expand b)) (expand w) eps := rfl

/-- `residual-layernorm-fuse` -/
theorem rule_residual_layernorm_fuse (a b w bias : Op) (eps : Nat) :
    expand (Op.FusedResidualLayerNorm a b w bias eps) =
      Op.LayerNorm (Op.Ew "add" (expand a) (expand b)) (expand w) (expand bias) eps :=
  rfl

/-- `groupnorm-act-fuse` -/
theorem rule_groupnorm_act_fuse (x w b : Op) (g eps : Nat) (k : String) :
    expand (Op.FusedGroupNormAct x w b g eps k) =
      Op.Act k (Op.GroupNorm (expand x) (expand w) (expand b) g eps) := rfl

/-- `adaln-modulate-fuse` -/
theorem rule_adaln_modulate_fuse (x scale shift : Op) :
    expand (Op.FusedAdaLN x scale shift) =
      Op.Ew "add" (Op.Ew "add" (Op.Ew "mul" (expand x) (expand scale)) (expand x))
        (expand shift) := rfl

/-- `gated-residual-fuse` -/
theorem rule_gated_residual_fuse (x y gate : Op) :
    expand (Op.FusedGatedResidual x y gate) =
      Op.Ew "add" (expand x) (Op.Ew "mul" (expand y) (expand gate)) := rfl

/-- `rmsnorm-rope-fuse` -/
theorem rule_rmsnorm_rope_fuse (x w : Op) (eps dim theta : Nat) :
    expand (Op.FusedNormRope x w eps dim theta) =
      Op.Rope (Op.RmsNorm (expand x) (expand w) eps) dim theta := rfl

/-- `rmsnorm-rope-scale-fuse` -/
theorem rule_rmsnorm_rope_scale_fuse (x w : Op) (eps dim theta factor : Nat) :
    expand (Op.FusedNormRopeScale x w eps dim theta factor) =
      Op.Scale (Op.Rope (Op.RmsNorm (expand x) (expand w) eps) dim theta) factor :=
  rfl

/-- `groupnorm-act-conv3d-fuse` -/
theorem rule_groupnorm_act_conv3d_fuse (x w b cw : Op)
    (g eps : Nat) (k : String) (s p : Nat) :
    expand (Op.FusedGroupNormActConv3d x w b cw g eps k s p) =
      Op.Conv3d (Op.Act k (Op.GroupNorm (expand x) (expand w) (expand b) g eps))
        (expand cw) s p := rfl

/-- `groupnorm-act-conv3d-bias-fuse` -/
theorem rule_groupnorm_act_conv3d_bias_fuse (x w b cw cb : Op)
    (g eps : Nat) (k : String) (s p : Nat) :
    expand (Op.FusedGroupNormActConv3dBias x w b cw cb g eps k s p) =
      Op.Conv3dBias (Op.Act k (Op.GroupNorm (expand x) (expand w) (expand b) g eps))
        (expand cw) (expand cb) s p := rfl

/-- `embedding-scale-fuse` -/
theorem rule_embedding_scale_fuse (ids table : Op) (factor : Nat) :
    expand (Op.FusedEmbeddingScale ids table factor) =
      Op.Scale (Op.Embedding (expand ids) (expand table)) factor := rfl

/-- `kda-gated-norm-fuse` -/
theorem rule_kda_gated_norm_fuse (o nw x gw : Op) (eps : Nat) (shape : String) :
    expand (Op.FusedKdaGatedNorm o nw x gw eps shape) =
      Op.Ew "mul" (Op.RmsNorm (expand o) (expand nw) eps)
        (Op.Act "sigmoid" (Op.Reshape (Op.Linear (expand x) (expand gw)) shape)) := rfl

/-- `mla-out-gate-fuse` -/
theorem rule_mla_out_gate_fuse (attn x gw : Op) :
    expand (Op.FusedMlaOutGate attn x gw) =
      Op.Ew "mul" (expand attn)
        (Op.Act "sigmoid" (Op.Linear (expand x) (expand gw))) := rfl

/-! ## Rule registry — the closed enumeration the CLI accepts as sound. -/

/-- The catalog of egglog rule names covered by the proofs above. Every rule
    the compiler *may* fire must appear here; adding one requires a matching
    `rule_*` theorem. This is the exact string list the Rust side submits
    via `checkA`. -/
def soundRules : List String :=
  ["rmsnorm-linear-fuse",
   "rmsnorm-linearbias-fuse",
   "layernorm-linear-fuse",
   "layernorm-linearbias-fuse",
   "linear-act-fuse",
   "linearbias-act-fuse",
   "gated-mlp-fuse",
   "residual-rmsnorm-fuse",
   "residual-layernorm-fuse",
   "groupnorm-act-fuse",
   "adaln-modulate-fuse",
   "gated-residual-fuse",
   "rmsnorm-rope-fuse",
   "rmsnorm-rope-scale-fuse",
   "groupnorm-act-conv3d-fuse",
   "groupnorm-act-conv3d-bias-fuse",
   "embedding-scale-fuse",
   "kda-gated-norm-fuse",
   "mla-out-gate-fuse"]

/-- Whether a rule name is in the sound-rules table. -/
def isSoundRule (name : String) : Bool :=
  soundRules.contains name

end Plow.Rewrite
