/-
# Plow.PageEnvelope — the read envelope of a paged attention loader's page table.

Model of FlashAttention-4's SM90 paged producer (`flash_fwd_sm90.py`, the TMA KV branch, and
`block_info.get_n_block_min_max`): for request `r`, query tile `m` and KV split `s` of `S`, it
reads page-table columns `hi − 1` (first K), every column of `[lo, hi)` (the loop), and `lo`
(the trailing V), where `nmax ≤ ceil(kv_len / 64)` is the tile's causal block limit,
`per = ceil(nmax / S)`, `lo = s · per`, `hi = min(lo + per, nmax)`. An empty split (`lo ≥ nmax`)
still reads `hi − 1` and `lo`; the consumer skips them, so the values never matter, only that the
loaded page id names mapped memory.

`read_in_envelope`: with `S ≤ pad + 1` and a non-empty request (`0 < N`), every column read lies in `[−1, width + pad)`, and `−1`
only when `nmax = 0`. A table written per launch as `[N_r own pages | pad copies of the last own
page]` per row (stride `width + pad`), plus one word before row 0, therefore never hands the
loader a page id it did not write this launch (`padded_entry_own`).

`unpadded_overreads`: with no padding (`pad = 0`, the table plowrt uploaded before the fix) a
one-page request at `S = 2` reads column 1, one past its row: the stale entry of bug 2.

Endpoint `page_envelope.v1`: a route is the loader's page (rows), its KV tile (rows), the split
cap of the runtime split rule and the padding columns the runtime writes per table row.
`checkRoute_sound` + `route_safe`: for an accepted route every launch, any request length and any
split count up to the cap reads only inside its padded row. The loader's read set is the model
above, not derived from machine code; the word before row 0 is the runtime's to write.
-/
import Lean.Data.Json

namespace Plow.PageEnvelope

/-- Columns of the loader's read set for one (request, tile, split). -/
def per (nmax S : Nat) : Nat := if nmax = 0 then 0 else (nmax + S - 1) / S
def lo (nmax S s : Nat) : Nat := s * per nmax S
def hi (nmax S s : Nat) : Nat := min (lo nmax S s + per nmax S) nmax

def Reads (nmax S s : Nat) (c : Int) : Prop :=
  c = (hi nmax S s : Int) - 1 ∨ c = (lo nmax S s : Int) ∨
    ((lo nmax S s : Int) ≤ c ∧ c < (hi nmax S s : Int))

theorem lo_bound (nmax S s : Nat) (hS : 0 < S) (hs : s < S) (h0 : nmax ≠ 0) :
    lo nmax S s + 1 < nmax + S := by
  obtain ⟨k, rfl⟩ : ∃ k, S = k + 1 := ⟨S - 1, by omega⟩
  unfold lo per
  simp only [h0, if_false]
  generalize hq : (nmax + (k + 1) - 1) / (k + 1) = q
  have hdiv : q * (k + 1) ≤ nmax + (k + 1) - 1 := hq ▸ Nat.div_mul_le_self _ _
  have h1 : 1 ≤ q := by
    rw [← hq]; apply (Nat.le_div_iff_mul_le hS).mpr; omega
  have : s * q ≤ k * q := Nat.mul_le_mul_right _ (by omega)
  have hm : q * (k + 1) = k * q + q := by rw [Nat.mul_succ, Nat.mul_comm]
  omega

theorem hi_pos (nmax S s : Nat) (hS : 0 < S) (h : nmax ≠ 0) : 1 ≤ hi nmax S s := by
  unfold hi lo per
  simp only [h, if_false]
  have hq : 1 ≤ (nmax + S - 1) / S := by
    apply (Nat.le_div_iff_mul_le hS).mpr
    omega
  omega

/-- Every column the loader reads lies in the padded envelope; `−1` only for an empty tile. -/
theorem read_in_envelope {nmax N width pad S s : Nat} {c : Int}
    (hN0 : 0 < N) (hN : nmax ≤ N) (hw : N ≤ width) (hS : 0 < S) (hSp : S ≤ pad + 1) (hs : s < S)
    (hr : Reads nmax S s c) :
    (-1 ≤ c ∧ c < (width : Int) + pad) ∧ (c = -1 → nmax = 0) := by
  by_cases h0 : nmax = 0
  · subst h0
    have e1 : hi 0 S s = 0 := by unfold hi; simp
    have e2 : lo 0 S s = 0 := by unfold lo per; simp
    unfold Reads at hr
    rw [e1, e2] at hr
    refine ⟨?_, fun _ => rfl⟩
    rcases hr with e | e | ⟨a, b⟩ <;> omega
  · have hlo := lo_bound nmax S s hS hs h0
    have hhi : hi nmax S s ≤ nmax := Nat.min_le_right _ _
    have hp := hi_pos nmax S s hS h0
    unfold Reads at hr
    refine ⟨?_, ?_⟩
    · rcases hr with e | e | ⟨a, b⟩ <;> omega
    · intro e
      rcases hr with e' | e' | ⟨a, b⟩ <;> omega

/-- A row written as `own pages [0, N) | pad copies of page N − 1`: the entry a non-negative read
    finds is one of the request's own pages. -/
def paddedEntry (own : Nat → Nat) (N : Nat) (c : Nat) : Nat :=
  if c < N then own c else own (N - 1)

theorem padded_entry_own {own : Nat → Nat} {nmax N width pad S s : Nat} {c : Int}
    (hN0 : 0 < N) (hN : nmax ≤ N) (hw : N ≤ width) (hS : 0 < S) (hSp : S ≤ pad + 1) (hs : s < S)
    (hr : Reads nmax S s c) (hc : 0 ≤ c) :
    c.toNat < width + pad ∧ ∃ j, j < N ∧ paddedEntry own N c.toNat = own j := by
  obtain ⟨⟨_, hlt⟩, _⟩ := read_in_envelope hN0 hN hw hS hSp hs hr
  refine ⟨by omega, ?_⟩
  unfold paddedEntry
  by_cases h : c.toNat < N
  · exact ⟨c.toNat, h, by simp [h]⟩
  · exact ⟨N - 1, by omega, by simp [h]⟩

/-- Without padding the loader reads past its row: one page, two splits, split 1 reads column 1
    of a width-1 table (the entry after the request's row, not written this launch). -/
theorem unpadded_overreads : Reads 1 2 1 1 ∧ ¬ ((1 : Int) < (1 : Nat) + (0 : Nat)) := by
  refine ⟨?_, by decide⟩
  unfold Reads hi lo per
  decide

/-- The envelope is tight: with `S = pad + 2` some read leaves it (so `pad ≥ max splits − 1`). -/
theorem envelope_tight (pad : Nat) :
    Reads 1 (pad + 2) (pad + 1) ((pad + 1 : Nat) : Int) ∧
      ¬ (((pad + 1 : Nat) : Int) < (1 : Nat) + pad) := by
  refine ⟨?_, by omega⟩
  unfold Reads hi lo per
  right; left
  have : (1 + (pad + 2) - 1) / (pad + 2) = 1 := by
    rw [show 1 + (pad + 2) - 1 = pad + 2 by omega]; exact Nat.div_self (by omega)
  simp [this]

/-! ## Endpoint `page_envelope.v1` -/

structure Route where
  page : Nat
  kvTile : Nat
  maxSplits : Nat
  pad : Nat
  deriving Repr

/-- One KV block is one page (the TMA paged path), and the pad covers the split cap. -/
def Route.Contract (r : Route) : Prop :=
  0 < r.page ∧ r.kvTile = r.page ∧ 0 < r.maxSplits ∧ r.maxSplits ≤ r.pad + 1

instance (r : Route) : Decidable r.Contract := by unfold Route.Contract; exact inferInstance

def checkRoute (r : Route) : Bool := decide r.Contract

theorem checkRoute_sound (r : Route) (h : checkRoute r = true) : r.Contract :=
  of_decide_eq_true h

/-- A request of `kvLen > 0` rows owns `ceil(kvLen / page)` columns of a row `width` wide; a tile
    reads at most `ceil(kvLen / kvTile)` KV blocks; the rule picks any `S ≤ maxSplits`. -/
theorem route_safe (r : Route) (h : r.Contract) {kvLen nmax width S s : Nat} {c : Int}
    (hk : 0 < kvLen) (hn : nmax ≤ (kvLen + r.kvTile - 1) / r.kvTile)
    (hw : (kvLen + r.page - 1) / r.page ≤ width) (hS : 0 < S) (hSm : S ≤ r.maxSplits)
    (hs : s < S) (hr : Reads nmax S s c) :
    (-1 ≤ c ∧ c < (width : Int) + r.pad) ∧ (c = -1 → nmax = 0) := by
  obtain ⟨hp, ht, _, hpad⟩ := h
  rw [ht] at hn
  have hN0 : 0 < (kvLen + r.page - 1) / r.page :=
    (Nat.le_div_iff_mul_le hp).mpr (by omega)
  exact read_in_envelope hN0 hn hw hS (by omega) hs hr

def exactKeys (j : Lean.Json) (keys : List String) : Except String Unit := do
  let obj ← j.getObj?
  let present := obj.toArray.map (·.1) |>.toList
  for k in present do
    if !keys.contains k then throw s!"unknown field {k}"
  for k in keys do
    if !present.contains k then throw s!"missing field {k}"

def nat (j : Lean.Json) (k : String) : Except String Nat := j.getObjValAs? Nat k

def parseRoute (j : Lean.Json) : Except String (String × Route) := do
  exactKeys j ["abi", "page", "kv_tile", "max_splits", "pad"]
  return (← j.getObjValAs? String "abi", {
    page := (← nat j "page"), kvTile := (← nat j "kv_tile"),
    maxSplits := (← nat j "max_splits"), pad := (← nat j "pad") })

def run (j : Lean.Json) : Except String String := do
  exactKeys j ["schema", "routes"]
  if (← nat j "schema") != 1 then throw "unsupported page_envelope schema"
  let raw ← j.getObjValAs? (List Lean.Json) "routes"
  if raw.isEmpty then throw "no routes"
  let routes ← raw.mapM parseRoute
  for (abi, r) in routes do
    if r.kvTile != r.page then
      throw s!"{abi}: KV tile {r.kvTile} is not the page {r.page}; the paged read model does not apply"
    if !checkRoute r then
      throw s!"{abi}: pad {r.pad} does not cover {r.maxSplits} splits (needs pad ≥ splits − 1, envelope_tight)"
  return s!"checkRoute_sound + route_safe: {routes.length} paged routes read only their own padded page-table row; loader read set modelled, not derived from machine code"

end Plow.PageEnvelope
