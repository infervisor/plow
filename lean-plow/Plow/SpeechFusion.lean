/-
# Plow.SpeechFusion — endpoint `speech_fusion.v1`.

Binds the hypotheses of `Plow.Speech`'s fusion equivalences to the instructions a packet
actually emits (`plow_asset::speech_fusion` derives the input).

* LayerNorm prologue (`DenseGemmF32` flag bit 5): `ln_prologue_eq` assumes the stats tensor
  holds LayerNormF32's statistics of every A row the GEMM reads. `LnSite.Contract` states the
  emitted facts that give it: the last earlier writer of `stats` is RowStatsF32 over the GEMM's
  own A tensor, in an earlier program, covering rows `[a_row0, a_row0 + M)` at width K, with no
  write of A or stats from that writer's program through the GEMM's. `ln_site_stats` derives
  the hypothesis from that contract plus the instruction semantics (RowStatsF32 writes the
  statistics of its input; a tensor no instruction writes keeps its contents), and
  `ln_site_gemm_eq` closes the GEMM equivalence.
* Conv1dF32 `row_scale`: `conv_row_scale_residual_eq` holds per element. `ConvSite.Contract`
  recomputes `out_rows` from the raw fields and checks the scale is `[batch][out_rows]`, the
  residual has the output's size and neither aliases the scale; `conv_site_in_bounds` gives
  every `(b, t, c)` element's scale and residual index in bounds.
-/
import Lean.Data.Json
import Plow.Speech

namespace Plow.SpeechFusion

open Plow.Speech

/-! ## LayerNorm prologue -/

structure LnSite where
  writerProgram : Nat
  gemmProgram : Nat
  writerRowStats : Bool
  writerX : Nat
  gemmA : Nat
  writerRows : Nat
  writerFeat : Nat
  gemmK : Nat
  aRow0 : Nat
  m : Nat
  statsBytes : Nat
  gammaBytes : Nat
  betaBytes : Nat
  aWritesBetween : Nat
  statsWritesBetween : Nat
  deriving Repr

def LnSite.Contract (s : LnSite) : Prop :=
  s.writerRowStats = true ∧ s.writerProgram < s.gemmProgram ∧ s.writerX = s.gemmA ∧
  0 < s.m ∧ s.aRow0 + s.m ≤ s.writerRows ∧ s.writerFeat = s.gemmK ∧
  8 * s.writerRows ≤ s.statsBytes ∧
  (s.gammaBytes = 0 ∨ s.gammaBytes = 4 * s.gemmK) ∧ (s.betaBytes = 0 ∨ s.betaBytes = 4 * s.gemmK) ∧
  s.aWritesBetween = 0 ∧ s.statsWritesBetween = 0

instance (s : LnSite) : Decidable s.Contract := by unfold LnSite.Contract; infer_instance

/-- `rowStats` is LayerNormF32's statistics of one row (RowStatsF32 runs that code with the
    same eps/order flags). `xw`/`statsW` are A and stats right after the writer; `x`/`statsT`
    when the GEMM runs. The semantic premises: RowStatsF32 writes the statistics of its input
    rows, and a tensor with no intervening writer is unchanged. -/
theorem ln_site_stats {α : Type} (s : LnSite) (h : s.Contract)
    (rowStats : (Nat → α) → α × α) (xw x : Nat → Nat → α) (statsW statsT : Nat → α × α)
    (hwriter : ∀ r, r < s.writerRows → statsW r = rowStats (xw r))
    (haFrame : s.writerX = s.gemmA → s.aWritesBetween = 0 → xw = x)
    (hsFrame : s.statsWritesBetween = 0 → statsT = statsW) :
    ∀ r, r < s.m → statsT (s.aRow0 + r) = rowStats (x (s.aRow0 + r)) := by
  obtain ⟨_, _, hx, _, hrows, _, _, _, _, ha, hs⟩ := h
  intro r hr
  rw [hsFrame hs, hwriter _ (by omega), haFrame hx ha]

/-- The fused GEMM equals the GEMM over materialized LayerNormF32 output (with the writer's
    statistics and the prologue's rounding bit), for every accepted site. -/
theorem ln_site_gemm_eq {α β : Type} (A : Arith α) (s : LnSite) (h : s.Contract)
    (rowStats : (Nat → α) → α × α) (xw x : Nat → Nat → α) (statsW statsT : Nat → α × α)
    (hwriter : ∀ r, r < s.writerRows → statsW r = rowStats (xw r))
    (haFrame : s.writerX = s.gemmA → s.aWritesBetween = 0 → xw = x)
    (hsFrame : s.statsWritesBetween = 0 → statsT = statsW)
    (round : Bool) (gamma beta : Nat → Option α)
    (gemm : (Nat → Nat → α) → Nat → β) (hlocal : ∀ a a' : Nat → Nat → α,
      (∀ r k, r < s.m → a r k = a' r k) → ∀ n, gemm a n = gemm a' n) :
    ∀ n, gemm (prologueA A statsT round x gamma beta s.aRow0) n
      = gemm (fun r k => layerNorm A (fun r => rowStats (x r)) round x gamma beta (s.aRow0 + r) k) n :=
  ln_prologue_gemm_eq A (fun r => rowStats (x r)) statsT round round x gamma beta s.aRow0 s.m
    (ln_site_stats s h rowStats xw x statsW statsT hwriter haFrame hsFrame) rfl gemm hlocal

/-! ## Conv1dF32 row_scale -/

structure ConvSite where
  batch : Nat
  inRows : Nat
  outChannels : Nat
  kernel : Nat
  stride : Nat
  dilation : Nat
  padBefore : Nat
  padAfter : Nat
  outBytes : Nat
  residualBytes : Nat
  rowScaleBytes : Nat
  rowScaleIsOut : Bool
  rowScaleIsResidual : Bool
  deriving Repr

/-- `out_rows = (in_rows + pad_before + pad_after - dilation*(kernel-1) - 1) / stride + 1`. -/
def ConvSite.outRows (s : ConvSite) : Nat :=
  (s.inRows + s.padBefore + s.padAfter - s.dilation * (s.kernel - 1) - 1) / s.stride + 1

def ConvSite.Contract (s : ConvSite) : Prop :=
  0 < s.batch ∧ 0 < s.outChannels ∧ 0 < s.kernel ∧ 0 < s.stride ∧ 0 < s.dilation ∧
  s.dilation * (s.kernel - 1) + 1 ≤ s.inRows + s.padBefore + s.padAfter ∧
  s.rowScaleBytes = 4 * (s.batch * s.outRows) ∧
  s.outBytes = 4 * (s.batch * s.outRows * s.outChannels) ∧
  s.residualBytes = s.outBytes ∧ s.rowScaleIsOut = false ∧ s.rowScaleIsResidual = false

instance (s : ConvSite) : Decidable s.Contract := by unfold ConvSite.Contract; infer_instance

theorem row_lt {b t rows batch : Nat} (hb : b < batch) (ht : t < rows) : b * rows + t < batch * rows := by
  have : (b + 1) * rows ≤ batch * rows := Nat.mul_le_mul_right _ hb
  rw [Nat.succ_mul] at this
  omega

/-- Every output element `(b, t, c)` reads `row_scale[b][t]` and `residual[b][t][c]` in bounds. -/
theorem conv_site_in_bounds (s : ConvSite) (h : s.Contract) (b t c : Nat)
    (hb : b < s.batch) (ht : t < s.outRows) (hc : c < s.outChannels) :
    4 * (b * s.outRows + t) + 4 ≤ s.rowScaleBytes ∧
    4 * ((b * s.outRows + t) * s.outChannels + c) + 4 ≤ s.residualBytes := by
  obtain ⟨_, _, _, _, _, _, hscale, hout, hres, _, _⟩ := h
  have hrow := row_lt hb ht
  have hel := row_lt hrow hc
  constructor <;> omega

/-- The fused epilogue equals the unfused BinaryF32 mul then add, for every accepted site. -/
theorem conv_site_eq {α : Type} (A : Arith α) (s : ConvSite) (_h : s.Contract)
    (mulComm : ∀ a b, A.mul a b = A.mul b a) (addComm : ∀ a b, A.add a b = A.add b a)
    (post : α → α) (y : Nat → Nat → Nat → α) (scale : Nat → Nat → α) (residual : Nat → Nat → Nat → α) :
    ∀ b t c, convFused A post (y b t c) (scale b t) (residual b t c)
      = convUnfused A post (y b t c) (scale b t) (residual b t c) :=
  fun _ _ _ => conv_row_scale_residual_eq A mulComm addComm post _ _ _

/-! ## Checker -/

inductive Site where
  | ln (s : LnSite)
  | conv (s : ConvSite)
  deriving Repr

def Site.Contract : Site → Prop
  | .ln s => s.Contract
  | .conv s => s.Contract

instance (s : Site) : Decidable s.Contract := by cases s <;> unfold Site.Contract <;> infer_instance

def check (sites : List Site) : Bool :=
  !sites.isEmpty && sites.all fun s => decide s.Contract

theorem check_sound (sites : List Site) (h : check sites = true) :
    sites ≠ [] ∧ ∀ s ∈ sites, s.Contract := by
  simp only [check, Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false, List.all_eq_true,
    decide_eq_true_eq] at h
  exact ⟨by intro e; simp [e] at h, h.2⟩

/-! ## JSON -/

open Lean (Json)

def exactKeys (j : Json) (keys : List String) : Except String Unit := do
  let obj ← j.getObj?
  let present := obj.toArray.map (·.1) |>.toList
  for k in present do
    if !keys.contains k then throw s!"unknown field {k}"
  for k in keys do
    if !present.contains k then throw s!"missing field {k}"

def nat (j : Json) (k : String) : Except String Nat := j.getObjValAs? Nat k
def bool (j : Json) (k : String) : Except String Bool := j.getObjValAs? Bool k

def parseSite (j : Json) : Except String Site := do
  match ← j.getObjValAs? String "kind" with
  | "ln_prologue" =>
    exactKeys j ["kind", "writer_program", "gemm_program", "writer_row_stats", "writer_x", "gemm_a",
      "writer_rows", "writer_feat", "gemm_k", "a_row0", "m", "stats_bytes", "gamma_bytes",
      "beta_bytes", "a_writes_between", "stats_writes_between"]
    return .ln {
      writerProgram := ← nat j "writer_program", gemmProgram := ← nat j "gemm_program",
      writerRowStats := ← bool j "writer_row_stats", writerX := ← nat j "writer_x",
      gemmA := ← nat j "gemm_a", writerRows := ← nat j "writer_rows",
      writerFeat := ← nat j "writer_feat", gemmK := ← nat j "gemm_k", aRow0 := ← nat j "a_row0",
      m := ← nat j "m", statsBytes := ← nat j "stats_bytes", gammaBytes := ← nat j "gamma_bytes",
      betaBytes := ← nat j "beta_bytes", aWritesBetween := ← nat j "a_writes_between",
      statsWritesBetween := ← nat j "stats_writes_between" }
  | "conv_row_scale" =>
    exactKeys j ["kind", "batch", "in_rows", "out_channels", "kernel", "stride", "dilation",
      "pad_before", "pad_after", "out_bytes", "residual_bytes", "row_scale_bytes",
      "row_scale_is_out", "row_scale_is_residual"]
    return .conv {
      batch := ← nat j "batch", inRows := ← nat j "in_rows", outChannels := ← nat j "out_channels",
      kernel := ← nat j "kernel", stride := ← nat j "stride", dilation := ← nat j "dilation",
      padBefore := ← nat j "pad_before", padAfter := ← nat j "pad_after",
      outBytes := ← nat j "out_bytes", residualBytes := ← nat j "residual_bytes",
      rowScaleBytes := ← nat j "row_scale_bytes", rowScaleIsOut := ← bool j "row_scale_is_out",
      rowScaleIsResidual := ← bool j "row_scale_is_residual" }
  | other => throw s!"unknown fused site kind {other}"

def run (j : Json) : Except String String := do
  exactKeys j ["schema", "sites"]
  if (← nat j "schema") != 1 then throw "unsupported speech_fusion schema"
  let raw ← j.getObjValAs? (List Json) "sites"
  let sites ← raw.mapM parseSite
  for (s, i) in sites.zip (List.range sites.length) do
    if !decide s.Contract then throw s!"site {i} precondition violated: {repr s}"
  if !check sites then throw "no fused sites"
  return s!"check_sound + ln_site_gemm_eq / conv_site_eq, conv_site_in_bounds: {sites.length} fused sites meet the Plow.Speech preconditions; instruction semantics and program order assumed"

end Plow.SpeechFusion
