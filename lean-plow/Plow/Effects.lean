import Plow.Verify

namespace Plow.Effects

open Plow Plow.Protocol Plow.Verify

structure Lease (tg : TaskGraph) where
  pool : Nat
  allocation : Nat
  generation : Nat
  owner : Nat
  offset : Nat
  size : Nat
  acquire : Fin tg.n
  retire : Fin tg.n
  cancel : Option (Fin tg.n)

structure Access (tg : TaskGraph) where
  task : Fin tg.n
  pool : Nat
  allocation : Nat
  generation : Nat
  owner : Nat
  offset : Nat
  size : Nat
  write : Bool

def containsAccess {tg : TaskGraph} (lease : Lease tg) (access : Access tg) : Bool :=
  decide (lease.pool = access.pool) && decide (lease.allocation = access.allocation) &&
  decide (lease.generation = access.generation) && decide (lease.owner = access.owner) &&
  decide (lease.offset ≤ access.offset) && decide (access.offset + access.size ≤ lease.offset + lease.size)

def activeAccess {tg : TaskGraph} (p : CounterProtocol tg) (paths : List (PathWitness tg))
    (lease : Lease tg) (access : Access tg) : Bool :=
  containsAccess lease access &&
  (decide (lease.acquire = access.task) || witnessedBefore p paths lease.acquire access.task) &&
  witnessedBefore p paths access.task lease.retire

def conflict {tg : TaskGraph} (a b : Access tg) : Bool :=
  decide (a.pool = b.pool) && bytesOverlapB a.offset a.size b.offset b.size && (a.write || b.write)

def checkConflicts {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (accesses : List (Access tg)) : Bool :=
  accesses.all fun a => accesses.all fun b =>
    !(decide (a.task.val < b.task.val) && conflict a b) || witnessedBefore p paths a.task b.task

def ConflictsOrdered {tg : TaskGraph} (p : CounterProtocol tg) (accesses : List (Access tg)) : Prop :=
  ∀ a ∈ accesses, ∀ b ∈ accesses, a.task.val < b.task.val → conflict a b = true →
    happensBefore p a.task b.task

theorem checkConflicts_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (accesses : List (Access tg))
    (h : checkConflicts p paths accesses = true) : ConflictsOrdered p accesses := by
  intro a ha b hb order overlaps
  have pair := (List.all_eq_true.mp ((List.all_eq_true.mp h) a ha)) b hb
  simp only [order, decide_true, overlaps, Bool.true_and, Bool.not_true, Bool.false_or] at pair
  exact witnessedBefore_sound p paths a.task b.task pair

def checkAccesses {tg : TaskGraph} (p : CounterProtocol tg) (paths : List (PathWitness tg))
    (leases : List (Lease tg)) (accesses : List (Access tg)) : Bool :=
  accesses.all fun access => decide (0 < access.size) &&
    leases.any fun lease => activeAccess p paths lease access

def AccessesLive {tg : TaskGraph} (p : CounterProtocol tg)
    (leases : List (Lease tg)) (accesses : List (Access tg)) : Prop :=
  ∀ access ∈ accesses, ∃ lease ∈ leases, containsAccess lease access = true ∧
    (lease.acquire = access.task ∨ happensBefore p lease.acquire access.task) ∧
    happensBefore p access.task lease.retire

theorem checkAccesses_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (leases : List (Lease tg)) (accesses : List (Access tg))
    (h : checkAccesses p paths leases accesses = true) : AccessesLive p leases accesses := by
  intro access ha
  have checked := List.all_eq_true.mp h access ha
  simp only [Bool.and_eq_true, List.any_eq_true, activeAccess, Bool.or_eq_true,
    decide_eq_true_eq] at checked
  obtain ⟨_, lease, hl, ⟨hc, hbegin⟩, hend⟩ := checked
  refine ⟨lease, hl, hc, ?_, witnessedBefore_sound p paths access.task lease.retire hend⟩
  rcases hbegin with same | ordered
  · exact Or.inl same
  · exact Or.inr (witnessedBefore_sound p paths lease.acquire access.task ordered)

def sameGeneration {tg : TaskGraph} (a b : Lease tg) : Bool :=
  decide (a.pool = b.pool) && decide (a.allocation = b.allocation) && decide (a.generation = b.generation)

def leasesOverlap {tg : TaskGraph} (a b : Lease tg) : Bool :=
  decide (a.pool = b.pool) && bytesOverlapB a.offset a.size b.offset b.size

def checkReuse {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (leases : List (Lease tg)) : Bool :=
  leases.all fun a => leases.all fun b =>
    sameGeneration a b || !leasesOverlap a b ||
    witnessedBefore p paths a.retire b.acquire || witnessedBefore p paths b.retire a.acquire

def ReuseRetired {tg : TaskGraph} (p : CounterProtocol tg) (leases : List (Lease tg)) : Prop :=
  ∀ a ∈ leases, ∀ b ∈ leases, sameGeneration a b = false → leasesOverlap a b = true →
    happensBefore p a.retire b.acquire ∨ happensBefore p b.retire a.acquire

theorem checkReuse_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (leases : List (Lease tg))
    (h : checkReuse p paths leases = true) : ReuseRetired p leases := by
  intro a ha b hb different overlap
  have pair := (List.all_eq_true.mp ((List.all_eq_true.mp h) a ha)) b hb
  simp only [different, overlap, Bool.not_true, Bool.false_or, Bool.or_eq_true] at pair
  rcases pair with forward | backward
  · exact Or.inl (witnessedBefore_sound p paths a.retire b.acquire forward)
  · exact Or.inr (witnessedBefore_sound p paths b.retire a.acquire backward)

def checkRetirements {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (leases : List (Lease tg)) : Bool :=
  leases.all fun lease => decide (0 < lease.size) && witnessedBefore p paths lease.acquire lease.retire &&
    match lease.cancel with
    | none => true
    | some cancel => witnessedBefore p paths cancel lease.retire

theorem cancellation_retired {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (leases : List (Lease tg))
    (h : checkRetirements p paths leases = true) (lease : Lease tg) (hl : lease ∈ leases)
    (cancel : Fin tg.n) (hc : lease.cancel = some cancel) : happensBefore p cancel lease.retire := by
  have checked := List.all_eq_true.mp h lease hl
  simp only [Bool.and_eq_true, hc] at checked
  exact witnessedBefore_sound p paths cancel lease.retire checked.2

end Plow.Effects
