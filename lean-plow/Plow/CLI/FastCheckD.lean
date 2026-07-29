/-
# Plow.CLI.FastCheckD — scalable executable twin of the checkpoint-D verifier.

`Plow.Verify.verifyAddressMap` / `readersWritersDisjointB` are the
proof-bearing reference definitions: fuel-bounded, list-based, quadratic in
entries and linear-recursive in `tg.n`. On a real schedule (~590k tasks,
~900k counter edges, ~330k byte-overlapping entry pairs) `reachableUpTo`
overflows the stack (recursion depth = `tg.n`) and the pairwise
happens-before probing is asymptotically infeasible.

This module computes the SAME booleans with scalable structures:

* The happens-before edge relation `directEdgeB` is factored through an
  expanded DAG instead of materializing dense pair edges:
  - counter edges: task `a` → counter-node `c` (c ∈ succs a) and
    counter-node `c` → task `b` (c ∈ waits b) — exactly the
    `∃ c, c ∈ succs a ∧ c ∈ waits b` clause;
  - resource stream order: tasks on one resource, grouped by equal
    `streamIdx`, chain through virtual group-nodes
    (`member of group g → G(g+1)`, `G(g) → members of g`, `G(g) → G(g+1)`)
    — the transitive closure over task nodes is exactly
    `resource a = resource b ∧ streamIdx a < streamIdx b`, with NO edge
    between equal-index tasks.
  Task-to-task reachability in this expanded DAG (plus reflexivity, which
  `reachableUpTo` includes via its fuel-0 base case) equals
  `happensBeforeB`.
* Reachability restricted to the target set (every writer of every entry)
  is computed by a blocked reverse-topological bitset DP — 1024 targets per
  pass, `O(V + E)` word-ops per pass — instead of one fuel-`n` list
  saturation per (reader, writer) pair. Each pass prunes to the ancestors
  of its targets first (`liveSet`).
* `readersWritersDisjointB` quantifies over CO-LOCATED pairs only (distinct
  names, bytes overlapping) — the same pairs `AddressMapSound` speaks about,
  and the same set this module's address-map pass already walks. It used to
  quantify over all pairs including `a = b`, which collapsed it to "the union
  of all reader sets and the union of all writer sets are disjoint"; that is
  strictly stronger than the theorem needs and rejected legitimate packets
  (see `disjointFast`).

The theorems in `Plow.Verify` continue to speak about the reference
definitions; `checkD`/`checkF` call this module for execution. A formal
equivalence proof between the two is future work — until then this module
is part of the same trusted computing base as the JSON bridge itself.

Cycles: a cyclic counter/resource graph (a deadlocking schedule) fails the
topological sort here and is REJECTED with an explicit reason. The
fuel-bounded reference would instead compute a partial closure; rejecting a
schedule whose ordering graph is cyclic is strictly conservative.

Phase timings are printed to stderr (`[fastD] …`) — they cost nothing
measurable and make the next performance conversation start from data.
-/
import Std.Data.HashMap
import Plow.Basic
import Plow.Protocol
import Plow.Memory
import Plow.CLI.Payload

namespace Plow.CLI.FastCheckD
open Plow

/-- Untyped view of one address-map entry (indices already validated by
    `Payload.parse` as `Fin tg.n`, carried here as raw `Nat`s). -/
structure EntryView where
  name    : String
  offset  : Nat
  size    : Nat
  readers : Array Nat
  writers : Array Nat
  deriving Inhabited

/-- The expanded happens-before DAG in CSR form plus its topological order. -/
structure Prep where
  nNodes : Nat
  adjOff : Array Nat
  adjDst : Array Nat
  /-- Reverse CSR (predecessors) — drives per-chunk ancestor pruning. -/
  revOff : Array Nat
  revDst : Array Nat
  /-- Forward topological order; `size < nNodes` iff the graph has a cycle. -/
  topo   : Array Nat

private def wordsPerRow : Nat := 16
/-- 63 usable bits per word: Lean boxes `UInt64` values ≥ 2^63 (tagged
    pointers carry 63 bits), and a boxed word turns every OR/AND in the hot
    loops into a heap allocation — measured 430ns/op vs ~5ns tagged. Keeping
    the top bit permanently clear costs 1/64 of the bit budget and keeps the
    whole DP allocation-free. -/
private def bitsPerWord : Nat := 63
private def allOnes : UInt64 := 0x7FFFFFFFFFFFFFFF

/-- Build the expanded DAG. `succsOf`/`waitsOf` give each task's counter
    lists; `resOf`/`idxOf` its resource and stream index. -/
def buildGraph (n : Nat) (succsOf waitsOf : Nat → List Nat)
    (resOf idxOf : Nat → Nat) : Prep := Id.run do
  -- Dense ids for the counters that appear at all.
  let mut cmap : Std.HashMap Nat Nat := {}
  for t in [0:n] do
    for c in succsOf t do
      if !cmap.contains c then cmap := cmap.insert c cmap.size
    for c in waitsOf t do
      if !cmap.contains c then cmap := cmap.insert c cmap.size
  let nC := cmap.size
  -- Stream groups: sort tasks by (resource, streamIdx); a group is a maximal
  -- run with equal resource AND equal streamIdx; one virtual node per group.
  let order := (Array.range n).qsort fun a b =>
    resOf a < resOf b || (resOf a == resOf b && idxOf a < idxOf b)
  let mut groupOf : Array Nat := Array.mkArray n 0 -- task → group id
  let mut groupResEnd : Array Bool := #[]          -- group ends its resource run
  let mut nG := 0
  for i in [0:n] do
    let t := order[i]!
    if i == 0 then
      nG := 1
    else
      let pt := order[i-1]!
      if resOf pt != resOf t then
        groupResEnd := groupResEnd.push true
        nG := nG + 1
      else if idxOf pt != idxOf t then
        groupResEnd := groupResEnd.push false
        nG := nG + 1
    groupOf := groupOf.set! t (nG - 1)
  if n > 0 then groupResEnd := groupResEnd.push true
  -- Node layout: [0,n) tasks, [n,n+nC) counters, [n+nC, n+nC+nG) groups.
  let nNodes := n + nC + nG
  let gnode := fun (g : Nat) => n + nC + g
  -- Edge emission (twice: degree count, then fill).
  let mut deg : Array Nat := Array.mkArray nNodes 0
  let bump := fun (deg : Array Nat) (v : Nat) => deg.set! v (deg[v]! + 1)
  for t in [0:n] do
    for _ in succsOf t do deg := bump deg t                 -- t → counter
    for c in waitsOf t do deg := bump deg (n + cmap[c]!)    -- counter → t
    deg := bump deg (gnode (groupOf[t]!))                   -- G(g) → t
    if !groupResEnd[groupOf[t]!]! then deg := bump deg t    -- t → G(g+1)
  for g in [0:nG] do
    if !groupResEnd[g]! then deg := bump deg (gnode g)      -- G(g) → G(g+1)
  let mut adjOff : Array Nat := Array.mkArray (nNodes + 1) 0
  for v in [0:nNodes] do
    adjOff := adjOff.set! (v+1) (adjOff[v]! + deg[v]!)
  let total := adjOff[nNodes]!
  let mut cur : Array Nat := adjOff -- running cursor per node (copy)
  let mut adjDst : Array Nat := Array.mkArray total 0
  let emit := fun (st : Array Nat × Array Nat) (v u : Nat) =>
    let (cur, adjDst) := st
    (cur.set! v (cur[v]! + 1), adjDst.set! (cur[v]!) u)
  for t in [0:n] do
    for c in succsOf t do
      let p := emit (cur, adjDst) t (n + cmap[c]!)
      cur := p.1; adjDst := p.2
    for c in waitsOf t do
      let p := emit (cur, adjDst) (n + cmap[c]!) t
      cur := p.1; adjDst := p.2
    let p := emit (cur, adjDst) (gnode (groupOf[t]!)) t
    cur := p.1; adjDst := p.2
    if !groupResEnd[groupOf[t]!]! then
      let p := emit (cur, adjDst) t (gnode (groupOf[t]! + 1))
      cur := p.1; adjDst := p.2
  for g in [0:nG] do
    if !groupResEnd[g]! then
      let p := emit (cur, adjDst) (gnode g) (gnode (g + 1))
      cur := p.1; adjDst := p.2
  -- Reverse CSR from the finished forward CSR.
  let mut rdeg : Array Nat := Array.mkArray nNodes 0
  for e in [0:total] do
    rdeg := rdeg.set! (adjDst[e]!) (rdeg[adjDst[e]!]! + 1)
  let mut revOff : Array Nat := Array.mkArray (nNodes + 1) 0
  for v in [0:nNodes] do
    revOff := revOff.set! (v+1) (revOff[v]! + rdeg[v]!)
  let mut rcur : Array Nat := revOff
  let mut revDst : Array Nat := Array.mkArray total 0
  for v in [0:nNodes] do
    let elo := adjOff[v]!
    let ehi := adjOff[v+1]!
    for e in [elo:ehi] do
      let u := adjDst[e]!
      revDst := revDst.set! (rcur[u]!) v
      rcur := rcur.set! u (rcur[u]! + 1)
  -- Kahn topological sort.
  let mut indeg : Array Nat := Array.mkArray nNodes 0
  for e in [0:total] do
    indeg := indeg.set! (adjDst[e]!) (indeg[adjDst[e]!]! + 1)
  let mut queue : Array Nat := #[]
  for v in [0:nNodes] do
    if indeg[v]! == 0 then queue := queue.push v
  let mut head := 0
  while head < queue.size do
    let v := queue[head]!
    head := head + 1
    let elo := adjOff[v]!
    let ehi := adjOff[v+1]!
    for e in [elo:ehi] do
      let u := adjDst[e]!
      indeg := indeg.set! u (indeg[u]! - 1)
      if indeg[u]! == 0 then queue := queue.push u
  return { nNodes := nNodes, adjOff := adjOff, adjDst := adjDst,
           revOff := revOff, revDst := revDst, topo := queue }

/-- Nodes that can reach at least one seed (reverse BFS). A pruned (dead)
    node's reachability row is all-zero by construction, so the DP skips it. -/
def liveSet (revOff revDst : Array Nat) (nNodes : Nat)
    (seeds : Array Nat) : Array Bool := Id.run do
  let mut live : Array Bool := Array.mkArray nNodes false
  let mut stack : Array Nat := #[]
  for s in seeds do
    if !live[s]! then
      live := live.set! s true
      stack := stack.push s
  let mut head := 0
  while head < stack.size do
    let v := stack[head]!
    head := head + 1
    let elo := revOff[v]!
    let ehi := revOff[v+1]!
    for e in [elo:ehi] do
      let p := revDst[e]!
      if !live[p]! then
        live := live.set! p true
        stack := stack.push p
  return live

/-- OR 16 consecutive words at `s` into 16 consecutive words at `d`
    (unchecked; callers guarantee `d + 16 ≤ size ∧ s + 16 ≤ size`). -/
unsafe def orRow16 (rows : Array UInt64) (d s : USize) : Array UInt64 :=
  let f := fun (rows : Array UInt64) (k : USize) =>
    rows.uset (d + k) ((rows.uget (d + k) lcProof) ||| (rows.uget (s + k) lcProof)) lcProof
  let rows := f rows 0;  let rows := f rows 1;  let rows := f rows 2
  let rows := f rows 3;  let rows := f rows 4;  let rows := f rows 5
  let rows := f rows 6;  let rows := f rows 7;  let rows := f rows 8
  let rows := f rows 9;  let rows := f rows 10; let rows := f rows 11
  let rows := f rows 12; let rows := f rows 13; let rows := f rows 14
  f rows 15

/-- Unchecked-index twin of `dpPass` — the DP is the verifier's hot loop
    (`(V+E) × words` array ops per chunk) and the checked accessors pay a
    bounds test per op. Indices are in range by construction: node ids come
    from `topo`/`adjDst` (< nNodes), edge ids from the CSR offsets, and rows
    has `nNodes * 16` slots. -/
unsafe def dpPassImpl (topo adjOff adjDst : Array Nat) (live : Array Bool)
    (_w : Nat) (rows0 : Array UInt64) : Array UInt64 := Id.run do
  let mut rows := rows0
  let mut ii := topo.size
  while ii > 0 do
    ii := ii - 1
    let v := topo.uget (USize.ofNat ii) lcProof
    if live.uget (USize.ofNat v) lcProof then
      let mut e := adjOff.uget (USize.ofNat v) lcProof
      let ehi := adjOff.uget (USize.ofNat (v + 1)) lcProof
      let vB := USize.ofNat (v * 16)
      while e < ehi do
        let u := adjDst.uget (USize.ofNat e) lcProof
        rows := orRow16 rows vB (USize.ofNat (u * 16))
        e := e + 1
  return rows

/-- Reverse-topological bitset propagation: after this pass, task row `t`
    has target-bit `k` set iff `t` reaches (or is) the chunk's `k`-th target.
    Reference implementation; execution uses `dpPassImpl`. -/
@[implemented_by dpPassImpl]
def dpPass (topo adjOff adjDst : Array Nat) (live : Array Bool)
    (w : Nat) (rows0 : Array UInt64) : Array UInt64 := Id.run do
  let mut rows := rows0
  let m := topo.size
  for ii in [0:m] do
    let v := topo[m - 1 - ii]!
    if live[v]! then
      let elo := adjOff[v]!
      let ehi := adjOff[v+1]!
      for e in [elo:ehi] do
        let u := adjDst[e]!
        for wi in [0:w] do
          let dst := v * w + wi
          rows := rows.set! dst (rows[dst]! ||| rows[u * w + wi]!)
  return rows

/-- AND 16 consecutive `rows` words at `s` into 16 `meets` words at `d`
    (unchecked; callers guarantee both spans are in range). -/
unsafe def andRow16 (meets : Array UInt64) (d : USize)
    (rows : Array UInt64) (s : USize) : Array UInt64 :=
  let f := fun (meets : Array UInt64) (k : USize) =>
    meets.uset (d + k) ((meets.uget (d + k) lcProof) &&& (rows.uget (s + k) lcProof)) lcProof
  let meets := f meets 0;  let meets := f meets 1;  let meets := f meets 2
  let meets := f meets 3;  let meets := f meets 4;  let meets := f meets 5
  let meets := f meets 6;  let meets := f meets 7;  let meets := f meets 8
  let meets := f meets 9;  let meets := f meets 10; let meets := f meets 11
  let meets := f meets 12; let meets := f meets 13; let meets := f meets 14
  f meets 15

/-- Unchecked twin of `meetKernel` (same justification as `dpPassImpl`). -/
unsafe def meetKernelImpl (needMeet : Array Bool) (entries : Array EntryView)
    (rows : Array UInt64) (meets0 : Array UInt64) : Array UInt64 := Id.run do
  let mut meets := meets0
  let ne := entries.size
  let mut iE := 0
  while iE < ne do
    if needMeet.uget (USize.ofNat iE) lcProof then
      let rd := (entries.uget (USize.ofNat iE) lcProof).readers
      let mBase := USize.ofNat (iE * 16)
      let nr := rd.size
      let mut ri := 0
      while ri < nr do
        let r := rd.uget (USize.ofNat ri) lcProof
        meets := andRow16 meets mBase rows (USize.ofNat (r * 16))
        ri := ri + 1
    iE := iE + 1
  return meets

/-- Intersect each needed entry's readers' reachability rows into its meet
    mask. Reference implementation; execution uses `meetKernelImpl`. -/
@[implemented_by meetKernelImpl]
def meetKernel (needMeet : Array Bool) (entries : Array EntryView)
    (rows : Array UInt64) (meets0 : Array UInt64) : Array UInt64 := Id.run do
  let mut meets := meets0
  let w := wordsPerRow
  for iE in [0:entries.size] do
    if needMeet[iE]! then
      for r in entries[iE]!.readers do
        for wi in [0:w] do
          let idx := iE * w + wi
          meets := meets.set! idx (meets[idx]! &&& rows[r * w + wi]!)
  return meets

/-- `readersWritersDisjointB`: for every CO-LOCATED pair (distinct names, bytes
    overlapping), the readers of one and the writers of the other are disjoint.
    Vacuously true when no two entries overlap.

    # This was an O(n) global union check, and it produced false rejections
    It used to read "no task appears in any entry's reader set AND any entry's
    writer set" — the faithful twin of the *unguarded* `readersWritersDisjointB`
    that Plow.Verify carried at the time. Both were far stronger than
    `AddressMapSound` needs, and the gap is not academic: on Gemma-4-12B
    `decode_b1_s1` the address map is `t7` at `[0, 6144)` and `kv_cache_L0` at
    `[6144, 3151872)` — ADJACENT, never overlapping — yet tasks 33-44 read and
    write the KV cache, so the global intersection was non-empty and checkpoint
    D rejected a packet with nothing wrong with it.

    The pair set here is exactly the one `verifyAddressMapFast` already walks
    (distinct name, bytes overlapping), so this adds no asymptotic term the
    checkpoint did not already pay — and on a real schedule that set is large
    (~330k pairs, see the header). Membership goes through a task-indexed
    scratch array cleared after each pair, so the cost is
    `O(pairs × (|readers| + |writers|))` rather than the quadratic
    `Array.contains` scan. -/
def disjointFast (n : Nat) (entries : Array EntryView) : Bool := Id.run do
  let mut mark : Array Bool := Array.mkArray n false
  for i in [0 : entries.size] do
    let a := entries[i]!
    for j in [i : entries.size] do
      let b := entries[j]!
      -- Same name ⇒ aliases of one buffer, which may share bytes by construction.
      if a.name == b.name then continue
      if !(a.offset < b.offset + b.size && b.offset < a.offset + a.size) then
        continue
      -- readers(a) ∩ writers(b)
      for w in b.writers do mark := mark.set! w true
      let mut bad := false
      for r in a.readers do
        if mark[r]! then bad := true
      for w in b.writers do mark := mark.set! w false
      if bad then return false
      -- readers(b) ∩ writers(a)
      for w in a.writers do mark := mark.set! w true
      let mut bad2 := false
      for r in b.readers do
        if mark[r]! then bad2 := true
      for w in a.writers do mark := mark.set! w false
      if bad2 then return false
  return true

/-- Allocate and fill the per-entry reader meets outside the caller's
    mutable-state scope (see the call-site comment). -/
def computeMeets (needMeet : Array Bool) (entries : Array EntryView)
    (rows : Array UInt64) (size : Nat) (ones : UInt64) : Array UInt64 :=
  meetKernel needMeet entries rows (Array.mkArray size ones)

/-- The address-map check: for every distinct-name byte-overlapping entry
    pair, all readers of one happen-before all writers of the other (in at
    least one direction). Errors on a cyclic ordering graph. Prints phase
    timings to stderr. -/
def verifyAddressMapFast (n : Nat) (prep : Prep)
    (entries : Array EntryView) : IO (Except String Bool) := do
  if prep.topo.size != prep.nNodes then
    return .error s!"ordering graph has a cycle ({prep.nNodes - prep.topo.size} nodes unsorted) — schedule would deadlock"
  -- Target set: every writer of every entry.
  let mut isTarget : Array Bool := Array.mkArray n false
  for e in entries do
    for w in e.writers do isTarget := isTarget.set! w true
  let mut targets : Array Nat := #[]
  for t in [0:n] do
    if isTarget[t]! then targets := targets.push t
  -- Distinct-name byte-overlapping pairs (unordered: the pair condition is
  -- symmetric, and a = b passes trivially via name equality).
  let mut pa : Array Nat := #[]
  let mut pb : Array Nat := #[]
  for i in [0:entries.size] do
    for j in [i+1:entries.size] do
      let a := entries[i]!
      let b := entries[j]!
      if a.name != b.name
          && a.offset < b.offset + b.size && b.offset < a.offset + a.size then
        pa := pa.push i
        pb := pb.push j
  let np := pa.size
  if np == 0 || targets.size == 0 then return .ok true
  -- okA: readers(pa) → writers(pb) holds so far; okB: the mirror direction.
  let mut okA : Array Bool := Array.mkArray np true
  let mut okB : Array Bool := Array.mkArray np true
  let w := wordsPerRow
  let chunkBits := w * bitsPerWord
  let nChunks := (targets.size + chunkBits - 1) / chunkBits
  let ones : UInt64 := allOnes
  let mut msLive := 0
  let mut msDp := 0
  let mut msMeet := 0
  let mut msPair := 0
  let mut nNeed := 0
  let mut nAndRows := 0
  for ch in [0:nChunks] do
    let base := ch * chunkBits
    let cnt := Nat.min chunkBits (targets.size - base)
    -- Which chunk bit (if any) each task owns.
    let mut taskBit : Array Nat := Array.mkArray n chunkBits
    for k in [0:cnt] do
      taskBit := taskBit.set! (targets[base + k]!) k
    -- Per-entry writer masks for this chunk + "has any bits" flags.
    let ne := entries.size
    let mut wmask : Array UInt64 := Array.mkArray (ne * w) 0
    let mut whas : Array Bool := Array.mkArray ne false
    for iE in [0:ne] do
      for wt in entries[iE]!.writers do
        let k := taskBit[wt]!
        if k < chunkBits then
          let idx := iE * w + k / bitsPerWord
          wmask := wmask.set! idx (wmask[idx]! ||| ((1 : UInt64) <<< UInt64.ofNat (k % bitsPerWord)))
          whas := whas.set! iE true
    -- Entries whose reader-meet we need this chunk.
    let mut needMeet : Array Bool := Array.mkArray ne false
    let mut anyNeed := false
    for p in [0:np] do
      if okA[p]! && whas[pb[p]!]! then
        needMeet := needMeet.set! (pa[p]!) true
        anyNeed := true
      if okB[p]! && whas[pa[p]!]! then
        needMeet := needMeet.set! (pb[p]!) true
        anyNeed := true
    if anyNeed then
      -- Ancestor pruning: only nodes that can reach a chunk target matter.
      let t0 ← IO.monoMsNow
      let mut seeds : Array Nat := #[]
      for k in [0:cnt] do
        seeds := seeds.push (targets[base + k]!)
      let live := liveSet prep.revOff prep.revDst prep.nNodes seeds
      let t1 ← IO.monoMsNow
      msLive := msLive + (t1 - t0)
      -- Blocked reverse-topological reachability DP.
      let mut rows : Array UInt64 := Array.mkArray (prep.nNodes * w) 0
      for k in [0:cnt] do
        let t := targets[base + k]!
        let idx := t * w + k / bitsPerWord
        rows := rows.set! idx (rows[idx]! ||| ((1 : UInt64) <<< UInt64.ofNat (k % bitsPerWord)))
      rows := dpPass prep.topo prep.adjOff prep.adjDst live w rows
      let t2 ← IO.monoMsNow
      msDp := msDp + (t2 - t1)
      -- Reader meets (all-ones when the reader set is empty — vacuous truth).
      for iE in [0:ne] do
        if needMeet[iE]! then
          nNeed := nNeed + 1
          nAndRows := nAndRows + entries[iE]!.readers.size
      -- `computeMeets` (not the mut var) owns the array while the kernel
      -- writes it: a `mut` bound during the call would keep a second
      -- reference in the loop's state tuple and turn every 16-word AND into
      -- a copy-on-write of the whole meets array (measured 10.6us vs 355ns
      -- per andRow16).
      let meets := computeMeets needMeet entries rows (ne * w) ones
      let t3 ← IO.monoMsNow
      msMeet := msMeet + (t3 - t2)
      -- Fold this chunk into the per-pair verdicts: direction holds iff the
      -- writer mask is a subset of the reader meet.
      let subset := fun (meets wmask : Array UInt64) (iMeet iMask : Nat) => Id.run do
        for wi in [0:w] do
          if wmask[iMask * w + wi]! &&& (allOnes ^^^ meets[iMeet * w + wi]!) != 0 then
            return false
        return true
      for p in [0:np] do
        if okA[p]! && whas[pb[p]!]! then
          if !subset meets wmask (pa[p]!) (pb[p]!) then okA := okA.set! p false
        if okB[p]! && whas[pa[p]!]! then
          if !subset meets wmask (pb[p]!) (pa[p]!) then okB := okB.set! p false
      let t4 ← IO.monoMsNow
      msPair := msPair + (t4 - t3)
  IO.eprintln s!"[fastD] chunks={nChunks} pairs={np} targets={targets.size} needSum={nNeed} andRows={nAndRows} live={msLive}ms dp={msDp}ms meet={msMeet}ms pair={msPair}ms"
  for p in [0:np] do
    if !(okA[p]! || okB[p]!) then return .ok false
  return .ok true

/-- Run both fast checks against a parsed checkpoint-D payload. Returns
    `(addressMapOk, disjointOk)`. -/
def run (d : Payload.Deserialized) : IO (Except String (Bool × Bool)) := do
  let n := d.taskGraph.n
  let liftF := fun (f : Fin d.taskGraph.n → List Nat) (i : Nat) =>
    if h : i < d.taskGraph.n then f ⟨i, h⟩ else []
  let liftN := fun (f : Fin d.taskGraph.n → Nat) (i : Nat) =>
    if h : i < d.taskGraph.n then f ⟨i, h⟩ else 0
  let entries : Array EntryView := d.entries.toArray.map fun e =>
    { name := e.name, offset := e.offset, size := e.size,
      readers := e.readers.toArray.map (·.val),
      writers := e.writers.toArray.map (·.val) }
  let t0 ← IO.monoMsNow
  let prep := buildGraph n (liftF d.protocol.succs) (liftF d.protocol.waits)
    (liftN d.protocol.resource) (liftN d.protocol.streamIdx)
  let t1 ← IO.monoMsNow
  IO.eprintln s!"[fastD] graph n={n} nodes={prep.nNodes} edges={prep.adjDst.size} build={t1 - t0}ms"
  match ← verifyAddressMapFast n prep entries with
  | .error e => return .error e
  | .ok amOk => return .ok (amOk, disjointFast n entries)

end Plow.CLI.FastCheckD
