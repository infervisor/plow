/-
# Plow.Knobs.Resolve — cli > env > production_default > static, knob by knob in registry order.

A production default's `when` may read knobs declared EARLIER in the registry (`ordered`), so
resolution is a single left-to-right pass and the theorems can speak about the final config:
`resolve_production` says each defaulted value is the first case that holds IN THE RESOLVED
CONFIG, not in some intermediate one.
-/
import Plow.Knobs.Formula

namespace Plow.Knobs

def srcGet : Sources → String → Source
  | [], _ => {}
  | (k, s) :: rest, x => if x = k then s else srcGet rest x

def pickDefault (c : Config) (t : Target) : List DefaultCase → Val → Val
  | [], o => o
  | d :: ds, o => if eval c t d.when then d.value else pickDefault c t ds o

def resolveOne (c : Config) (t : Target) (s : Source) (k : KnobSpec) : Val :=
  match s.cli, s.env, k.dflt with
  | some v, _, _ => v
  | none, some v, _ => v
  | none, none, .static v => v
  | none, none, .production cs o => pickDefault c t cs o

def resolveAux (src : Sources) (t : Target) : Config → List KnobSpec → Config
  | acc, [] => acc
  | acc, k :: ks => resolveAux src t ((k.id, resolveOne acc t (srcGet src k.id) k) :: acc) ks

def resolve (specs : List KnobSpec) (src : Sources) (t : Target) : Config :=
  resolveAux src t [] specs

def casesVars : List DefaultCase → List String
  | [] => []
  | d :: ds => vars d.when ++ casesVars ds

def dfltVars : Dflt → List String
  | .static _ => []
  | .production cs _ => casesVars cs

def dfltValues : Dflt → List Val
  | .static v => [v]
  | .production cs o => o :: cs.map (·.value)

/-- Ids are unique, and every default reads only knobs declared before it. -/
def ordered : List String → List KnobSpec → Bool
  | _, [] => true
  | seen, k :: ks =>
    !memB k.id seen && (dfltVars k.dflt).all (fun x => memB x seen) && ordered (k.id :: seen) ks

def optAdmits (d : Domain) : Option Val → Bool
  | none => true
  | some v => admits d v

def defaultsInDomain (specs : List KnobSpec) : Bool :=
  specs.all fun k => (dfltValues k.dflt).all (admits k.domain)

def sourcesInDomain (specs : List KnobSpec) (src : Sources) : Bool :=
  specs.all fun k => optAdmits k.domain (srcGet src k.id).cli && optAdmits k.domain (srcGet src k.id).env

/-! ## Lemmas -/

theorem mem_split {α : Type} {a : α} : ∀ {l : List α}, a ∈ l → ∃ s t, l = s ++ a :: t
  | b :: l, h => by
    rcases List.mem_cons.mp h with rfl | hm
    · exact ⟨[], l, rfl⟩
    · obtain ⟨s, t, rfl⟩ := mem_split hm
      exact ⟨b :: s, t, rfl⟩

theorem resolveAux_append (src : Sources) (t : Target) :
    ∀ (acc : Config) (l₁ l₂ : List KnobSpec),
      resolveAux src t acc (l₁ ++ l₂) = resolveAux src t (resolveAux src t acc l₁) l₂
  | _, [], _ => rfl
  | acc, k :: l₁, l₂ => by
    simp only [List.cons_append, resolveAux]
    exact resolveAux_append src t _ l₁ l₂

theorem find_resolveAux_fresh (src : Sources) (t : Target) (x : String) :
    ∀ (acc : Config) (ks : List KnobSpec),
      (∀ k ∈ ks, k.id ≠ x) → find (resolveAux src t acc ks) x = find acc x
  | _, [], _ => rfl
  | acc, k :: ks, h => by
    simp only [resolveAux]
    rw [find_resolveAux_fresh src t x _ ks (fun k' hk' => h k' (List.mem_cons_of_mem _ hk'))]
    have hne : x ≠ k.id := fun e => h k (List.mem_cons_self _ _) e.symm
    simp [find, hne]

theorem ordered_cons (seen : List String) (k : KnobSpec) (ks : List KnobSpec) :
    ordered seen (k :: ks) = true ↔
      k.id ∉ seen ∧ (∀ x ∈ dfltVars k.dflt, x ∈ seen) ∧ ordered (k.id :: seen) ks = true := by
  simp [ordered, memB_iff, memB_eq_false_iff, List.all_eq_true, and_assoc]

theorem ordered_fresh : ∀ (seen : List String) (l : List KnobSpec),
    ordered seen l = true → ∀ k ∈ l, k.id ∉ seen
  | _, [], _, _, hk => absurd hk (List.not_mem_nil _)
  | seen, k :: ks, h, k', hk' => by
    obtain ⟨h1, _, h3⟩ := (ordered_cons seen k ks).mp h
    rcases List.mem_cons.mp hk' with rfl | hm
    · exact h1
    · exact fun hs => ordered_fresh _ ks h3 k' hm (List.mem_cons_of_mem _ hs)

theorem ordered_split : ∀ (seen : List String) (pre : List KnobSpec) (k : KnobSpec)
    (post : List KnobSpec), ordered seen (pre ++ k :: post) = true →
      (∀ k' ∈ post, k'.id ≠ k.id) ∧
      (∀ x ∈ dfltVars k.dflt, x ∈ seen ∨ ∃ q ∈ pre, q.id = x) ∧
      (∀ q ∈ pre, ∀ k' ∈ k :: post, k'.id ≠ q.id)
  | seen, [], k, post, h => by
    obtain ⟨_, h2, h3⟩ := (ordered_cons seen k post).mp h
    refine ⟨fun k' hk' e => ordered_fresh _ post h3 k' hk' (e ▸ List.mem_cons_self _ _),
      fun x hx => Or.inl (h2 x hx), fun q hq => absurd hq (List.not_mem_nil _)⟩
  | seen, p :: pre, k, post, h => by
    obtain ⟨_, _, h3⟩ := (ordered_cons seen p (pre ++ k :: post)).mp h
    obtain ⟨ih1, ih2, ih3⟩ := ordered_split (p.id :: seen) pre k post h3
    refine ⟨ih1, fun x hx => ?_, fun q hq k' hk' => ?_⟩
    · rcases ih2 x hx with hs | ⟨q, hq, hqx⟩
      · rcases List.mem_cons.mp hs with rfl | hs
        · exact Or.inr ⟨p, List.mem_cons_self _ _, rfl⟩
        · exact Or.inl hs
      · exact Or.inr ⟨q, List.mem_cons_of_mem _ hq, hqx⟩
    · rcases List.mem_cons.mp hq with rfl | hq
      · have hin : k' ∈ pre ++ k :: post := List.mem_append_right _ hk'
        exact fun e => ordered_fresh _ _ h3 k' hin (e ▸ List.mem_cons_self _ _)
      · exact ih3 q hq k' hk'

theorem casesVars_mem : ∀ (cs : List DefaultCase) (d : DefaultCase) (x : String),
    d ∈ cs → x ∈ vars d.when → x ∈ casesVars cs
  | [], _, _, hd, _ => absurd hd (List.not_mem_nil _)
  | c :: cs, d, x, hd, hx => by
    rcases List.mem_cons.mp hd with rfl | hd
    · simp [casesVars, hx]
    · simp [casesVars, casesVars_mem cs d x hd hx]

theorem pickDefault_congr (t : Target) (c₁ c₂ : Config) (o : Val) : ∀ cs : List DefaultCase,
    (∀ d ∈ cs, eval c₁ t d.when = eval c₂ t d.when) → pickDefault c₁ t cs o = pickDefault c₂ t cs o
  | [], _ => rfl
  | d :: ds, h => by
    simp only [pickDefault]
    rw [h d (List.mem_cons_self _ _),
      pickDefault_congr t c₁ c₂ o ds (fun d' hd' => h d' (List.mem_cons_of_mem _ hd'))]

theorem pickDefault_spec (c : Config) (t : Target) (o : Val) : ∀ cs : List DefaultCase,
    pickDefault c t cs o = o ∨ ∃ d ∈ cs, eval c t d.when = true ∧ pickDefault c t cs o = d.value
  | [] => Or.inl rfl
  | d :: ds => by
    by_cases hw : eval c t d.when = true
    · exact Or.inr ⟨d, List.mem_cons_self _ _, hw, by simp [pickDefault, hw]⟩
    · have hr : pickDefault c t (d :: ds) o = pickDefault c t ds o := by simp [pickDefault, hw]
      rcases pickDefault_spec c t o ds with h | ⟨d', hd', hw', hv⟩
      · exact Or.inl (hr.trans h)
      · exact Or.inr ⟨d', List.mem_cons_of_mem _ hd', hw', hr.trans hv⟩

/-- Where a registered knob's value comes from: `resolveOne` over the config of the knobs
    before it, and that prefix config agrees with the final one on every earlier knob. -/
theorem resolve_find (specs : List KnobSpec) (src : Sources) (t : Target) (k : KnobSpec)
    (hwf : ordered [] specs = true) (hk : k ∈ specs) :
    ∃ pre post, specs = pre ++ k :: post ∧
      find (resolve specs src t) k.id =
        some (resolveOne (resolveAux src t [] pre) t (srcGet src k.id) k) ∧
      (∀ x ∈ dfltVars k.dflt, get (resolve specs src t) x = get (resolveAux src t [] pre) x) := by
  obtain ⟨pre, post, rfl⟩ := mem_split hk
  obtain ⟨hpost, hvars, hlater⟩ := ordered_split [] pre k post hwf
  refine ⟨pre, post, rfl, ?_, ?_⟩
  · simp only [resolve, resolveAux_append, resolveAux]
    rw [find_resolveAux_fresh _ _ _ _ post hpost]
    simp [find]
  · intro x hx
    rcases hvars x hx with hs | ⟨q, hq, hqx⟩
    · exact absurd hs (List.not_mem_nil _)
    · simp only [get, resolve, resolveAux_append]
      rw [find_resolveAux_fresh _ _ _ _ (k :: post)
        (fun k' hk' => by rw [← hqx]; exact hlater q hq k' hk')]

/-! ## Universal theorems -/

/-- Every registered knob gets a value, and that value is in its domain. -/
theorem resolve_total (specs : List KnobSpec) (src : Sources) (t : Target) (k : KnobSpec)
    (hwf : ordered [] specs = true) (hdef : defaultsInDomain specs = true)
    (hsrc : sourcesInDomain specs src = true) (hk : k ∈ specs) :
    ∃ v, find (resolve specs src t) k.id = some v ∧ admits k.domain v = true := by
  obtain ⟨pre, _, _, hfind, _⟩ := resolve_find specs src t k hwf hk
  refine ⟨_, hfind, ?_⟩
  have hd := (List.all_eq_true.mp hdef) k hk
  have hs := (List.all_eq_true.mp hsrc) k hk
  simp only [Bool.and_eq_true] at hs
  obtain ⟨hcli, henv⟩ := hs
  cases hc : (srcGet src k.id).cli with
  | some v => simp only [resolveOne, hc]; rw [hc] at hcli; simpa [optAdmits] using hcli
  | none =>
    cases he : (srcGet src k.id).env with
    | some v => simp only [resolveOne, hc, he]; rw [he] at henv; simpa [optAdmits] using henv
    | none =>
      cases hdf : k.dflt with
      | static v => simp only [resolveOne, hc, he, hdf]; rw [hdf] at hd; simpa [dfltValues] using hd
      | production cs o =>
        simp only [resolveOne, hc, he, hdf]
        rw [hdf] at hd
        have hall := List.all_eq_true.mp hd
        rcases pickDefault_spec (resolveAux src t [] pre) t o cs with h | ⟨d, hdm, _, hv⟩
        · rw [h]; exact hall o (by simp [dfltValues])
        · rw [hv]; exact hall d.value (by simp [dfltValues]; exact Or.inr ⟨d, hdm, rfl⟩)

theorem resolve_cli (specs : List KnobSpec) (src : Sources) (t : Target) (k : KnobSpec) (v : Val)
    (hwf : ordered [] specs = true) (hk : k ∈ specs) (hc : (srcGet src k.id).cli = some v) :
    get (resolve specs src t) k.id = v := by
  obtain ⟨_, _, _, hfind, _⟩ := resolve_find specs src t k hwf hk
  simp [get, hfind, resolveOne, hc]

theorem resolve_env (specs : List KnobSpec) (src : Sources) (t : Target) (k : KnobSpec) (v : Val)
    (hwf : ordered [] specs = true) (hk : k ∈ specs)
    (hc : (srcGet src k.id).cli = none) (he : (srcGet src k.id).env = some v) :
    get (resolve specs src t) k.id = v := by
  obtain ⟨_, _, _, hfind, _⟩ := resolve_find specs src t k hwf hk
  simp [get, hfind, resolveOne, hc, he]

theorem resolve_static (specs : List KnobSpec) (src : Sources) (t : Target) (k : KnobSpec) (v : Val)
    (hwf : ordered [] specs = true) (hk : k ∈ specs)
    (hc : (srcGet src k.id).cli = none) (he : (srcGet src k.id).env = none)
    (hd : k.dflt = .static v) :
    get (resolve specs src t) k.id = v := by
  obtain ⟨_, _, _, hfind, _⟩ := resolve_find specs src t k hwf hk
  simp [get, hfind, resolveOne, hc, he, hd]

/-- With no cli and no env value, a production-defaulted knob takes the first case whose `when`
    holds in the RESOLVED config, else `otherwise`. -/
theorem resolve_production (specs : List KnobSpec) (src : Sources) (t : Target) (k : KnobSpec)
    (cs : List DefaultCase) (o : Val)
    (hwf : ordered [] specs = true) (hk : k ∈ specs)
    (hc : (srcGet src k.id).cli = none) (he : (srcGet src k.id).env = none)
    (hd : k.dflt = .production cs o) :
    get (resolve specs src t) k.id = pickDefault (resolve specs src t) t cs o := by
  obtain ⟨pre, _, _, hfind, hagree⟩ := resolve_find specs src t k hwf hk
  have hval : get (resolve specs src t) k.id = pickDefault (resolveAux src t [] pre) t cs o := by
    simp [get, hfind, resolveOne, hc, he, hd]
  rw [hval]
  apply pickDefault_congr
  intro d hdm
  apply eval_congr
  intro x hx
  have hx' : x ∈ dfltVars k.dflt := by rw [hd]; exact casesVars_mem cs d x hdm hx
  exact (hagree x hx').symm

/-- cli beats env beats production_default beats static. -/
theorem resolve_precedence (specs : List KnobSpec) (src : Sources) (t : Target) (k : KnobSpec)
    (hwf : ordered [] specs = true) (hk : k ∈ specs) :
    get (resolve specs src t) k.id =
      match (srcGet src k.id).cli, (srcGet src k.id).env, k.dflt with
      | some v, _, _ => v
      | none, some v, _ => v
      | none, none, .static v => v
      | none, none, .production cs o => pickDefault (resolve specs src t) t cs o := by
  cases hc : (srcGet src k.id).cli with
  | some v => exact resolve_cli specs src t k v hwf hk hc
  | none =>
    cases he : (srcGet src k.id).env with
    | some v => exact resolve_env specs src t k v hwf hk hc he
    | none =>
      cases hd : k.dflt with
      | static v => exact resolve_static specs src t k v hwf hk hc he hd
      | production cs o => exact resolve_production specs src t k cs o hwf hk hc he hd

/-- A production default applies only when one of its cases holds in the resolved config. -/
theorem production_default_scoped (specs : List KnobSpec) (src : Sources) (t : Target)
    (k : KnobSpec) (cs : List DefaultCase) (o : Val)
    (hwf : ordered [] specs = true) (hk : k ∈ specs)
    (hc : (srcGet src k.id).cli = none) (he : (srcGet src k.id).env = none)
    (hd : k.dflt = .production cs o) (hne : get (resolve specs src t) k.id ≠ o) :
    ∃ d ∈ cs, eval (resolve specs src t) t d.when = true ∧
      get (resolve specs src t) k.id = d.value := by
  have hp := resolve_production specs src t k cs o hwf hk hc he hd
  rcases pickDefault_spec (resolve specs src t) t o cs with h | ⟨d, hdm, hw, hv⟩
  · exact absurd (hp.trans h) hne
  · exact ⟨d, hdm, hw, hp.trans hv⟩

theorem resolveAux_congr (src₁ src₂ : Sources) (t : Target)
    (h : ∀ x, srcGet src₁ x = srcGet src₂ x) :
    ∀ (acc : Config) (ks : List KnobSpec), resolveAux src₁ t acc ks = resolveAux src₂ t acc ks
  | _, [] => rfl
  | acc, k :: ks => by
    simp only [resolveAux, h k.id]
    exact resolveAux_congr src₁ src₂ t h _ ks

/-- Resolution depends on the sources only through their per-knob lookup: entry order and
    unregistered entries cannot change the config. -/
theorem resolve_deterministic (specs : List KnobSpec) (src₁ src₂ : Sources) (t : Target)
    (h : ∀ x, srcGet src₁ x = srcGet src₂ x) : resolve specs src₁ t = resolve specs src₂ t :=
  resolveAux_congr src₁ src₂ t h [] specs

end Plow.Knobs
