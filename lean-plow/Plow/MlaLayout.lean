import Lean.Data.Json
import Plow.Basic

namespace Plow.MlaLayout

def paddedAddress (row head column : Nat) : Nat := ((row * 8 + head) * 1024) + column

def unpad {rows : Nat} (source : Nat → UInt16) : Fin rows → Fin 8 → Fin 512 → UInt16 :=
  fun row head column => source (paddedAddress row.val head.val column.val)

def directInput {rows : Nat} (source : Nat → UInt16) : Fin rows → Fin 8 → Fin 512 → UInt16 :=
  fun row head column => source (((row.val * 8 + head.val) * 1024) + column.val)

theorem address_in_bounds (rows row head column : Nat)
    (hr : row < rows) (hh : head < 8) (hc : column < 512) :
    paddedAddress row head column < rows * 8192 := by
  unfold paddedAddress
  omega

theorem odd_heads_not_read (row head column : Nat) (hc : column < 512) :
    paddedAddress row head column % 1024 < 512 := by
  simp [paddedAddress, Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_lt (by omega : column < 1024)]
  exact hc

theorem consumer_input_bits_unchanged {rows : Nat} (source : Nat → UInt16) :
    unpad (rows := rows) source = directInput source := by rfl

theorem same_consumer_output {rows : Nat} {Result : Type}
    (consumer : (Fin rows → Fin 8 → Fin 512 → UInt16) → Result) (source : Nat → UInt16) :
    consumer (unpad source) = consumer (directInput source) :=
  congrArg consumer (consumer_input_bits_unchanged source)

theorem ignored_padding {rows : Nat} (a b : Nat → UInt16)
    (equal : ∀ row : Fin rows, ∀ head : Fin 8, ∀ column : Fin 512,
      a (paddedAddress row.val head.val column.val) = b (paddedAddress row.val head.val column.val)) :
    directInput (rows := rows) a = directInput b := by
  funext row head column
  exact equal row head column

open Lean (Json)

def run (j : Json) : Except String String := do
  let rows ← j.getObjValAs? Nat "rows"
  let heads ← j.getObjValAs? Nat "heads"
  let n ← j.getObjValAs? Nat "n"
  let k ← j.getObjValAs? Nat "k"
  let stride ← j.getObjValAs? Nat "head_stride"
  let capacity ← j.getObjValAs? Nat "capacity_elements"
  let boundary ← j.getObjValAs? String "boundary"
  let scale ← j.getObjValAs? String "weight_scale"
  let group ← j.getObjValAs? Nat "activation_group"
  let inactive ← j.getObjValAs? Bool "inactive_rows"
  if rows == 0 || 32 ≤ rows || heads != 8 || n != 256 || k != 512 || stride != 1024 ||
      capacity < rows * 8192 || boundary != "bf16_rne" || scale != "scalar_after_group_sum" ||
      group != 128 || inactive then
    throw "unsupported MLA layout/precision domain"
  return "address_in_bounds, odd_heads_not_read, consumer_input_bits_unchanged: same BF16 input bits for an unchanged consumer; not a floating-point kernel or machine-code implementation proof"

end Plow.MlaLayout
