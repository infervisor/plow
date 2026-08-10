/-
# Plow.LdsFit — staged-LDS fit obligation (LdsFitSound).

The always-staged GEMV family (`gemv_qkv_rows` / `gemv_glu_rows`, op_gemm.h)
reads `x` ONLY through LDS: the kernel stages `M*K` halves unconditionally and
"plowc emits this op only when M*K fits GM_LDS_HALVES" — choosing the fused
opcode IS that promise. Task 9 (perf-data/plow-gfx942/glm52-batched-decode-r4.md,
TASK 9 ROOT CAUSE) is what a broken promise serves: at rows=8, K=6144 the stage
runs past the LDS window, rows past the fit read zero/garbage x, and the model
answers fluently and wrongly. The emitter now carries a fusion gate (mla.rs);
THIS module is the machine-checked half — every emitted instance of a staged op
is re-checked here, so a future emitter path that forgets the gate is rejected
at emit rather than found by seven rounds of GPU discriminators.

## Role

`checkLdsFit` is the executable checker the CLI runs per program; the theorem
`fits_of_check_ok` is its soundness: an accepted list contains no instance
whose staged demand exceeds the arena. The demand model is deliberately the
kernel's own arithmetic (`rows * k + scratch` halves — scratch is
GV_NORM_SCRATCH when the q-norm fold rides the packet, else 0); the arena is
supplied by the Rust side from `hwspec` (single source of truth, held against
the device headers by `device_header_agreement.rs`).
-/
import Plow.Basic

namespace Plow.LdsFit

/-- One emitted instance of an always-staged op: which op (for the report),
    its instruction index, and the demand parameters. -/
structure StagedOp where
  op      : String
  idx     : Nat
  rows    : Nat
  k       : Nat
  scratch : Nat
  deriving Repr

/-- Staged-LDS demand in halves — the kernel's own arithmetic. -/
def demand (s : StagedOp) : Nat := s.rows * s.k + s.scratch

/-- Executable checker: first instance whose demand exceeds `arena`, or `ok`. -/
def checkLdsFit (arena : Nat) : List StagedOp → Except StagedOp Unit
  | [] => .ok ()
  | s :: rest =>
    if demand s ≤ arena then checkLdsFit arena rest else .error s

/-- Soundness: an accepted list has no over-demand instance. -/
theorem fits_of_check_ok (arena : Nat) (ops : List StagedOp)
    (h : checkLdsFit arena ops = .ok ()) :
    ∀ s ∈ ops, demand s ≤ arena := by
  induction ops with
  | nil => intro s hs; cases hs
  | cons a rest ih =>
    intro s hs
    unfold checkLdsFit at h
    by_cases hd : demand a ≤ arena
    · simp [hd] at h
      cases hs with
      | head => exact hd
      | tail _ hmem => exact ih h s hmem
    · simp [hd] at h

/-- The check is not vacuous: an over-demand instance is reported, and it is
    the first one. -/
theorem rejects_over_demand (arena : Nat) (s : StagedOp) (rest : List StagedOp)
    (h : arena < demand s) :
    checkLdsFit arena (s :: rest) = .error s := by
  unfold checkLdsFit
  simp [Nat.not_le.mpr h]

end Plow.LdsFit
