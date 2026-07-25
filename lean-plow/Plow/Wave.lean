/-
# Plow.Wave — wave-quantized tile-cost theorems (§A1 A4 A5 E4).

Models the "wave" execution pattern on GPUs: `sm_count` SMs each process one
tile per wave; the number of waves is `⌈tiles / sm_count⌉`. Proves the lower
bound the extractor uses to prune tile candidates and the makespan identity
the scheduler relies on for cost-based ranking.
-/
import Plow.Basic
import Plow.TilePartition

namespace Plow.Wave

open Plow.TilePartition

/-- The machine's parallelism: how many tiles can execute per wave. -/
structure Machine where
  smCount   : Nat
  smPos     : 0 < smCount
  /-- Bytes per SM's tensor-memory register file (Blackwell `tmem`). -/
  tmemBytes : Nat

/-! ## A1 — Wave-count lower bound

    Every wave processes at most `smCount` tiles. Since covering the full
    `m × n × k` output requires at least `tileCount g t` tile-steps (proven
    by `tile_partition_covers`), the wave count satisfies

        `smCount · numWaves · bm · bn · bk ≥ m · n · k`

    (a monotone lower bound the extractor uses to prune tile candidates). -/

/-- Number of waves to cover the tile grid. -/
def numWaves (g : Gemm) (t : Tile) (m : Machine) : Nat :=
  ceilDiv (tileCount g t) m.smCount

/-- Ceiling division lower bound: `b · ⌈a/b⌉ ≥ a`. Mirrors
    `TilePartition.le_ceilDiv_mul` on the `Wave.Machine` side. -/
theorem smCount_mul_numWaves_ge (g : Gemm) (t : Tile) (m : Machine) :
    m.smCount * numWaves g t m ≥ tileCount g t := by
  unfold numWaves ceilDiv
  rw [if_neg (Nat.pos_iff_ne_zero.mp m.smPos)]
  have hdiv :
      m.smCount * ((tileCount g t + m.smCount - 1) / m.smCount)
      + (tileCount g t + m.smCount - 1) % m.smCount
      = tileCount g t + m.smCount - 1 :=
    Nat.div_add_mod _ m.smCount
  have hmod :
      (tileCount g t + m.smCount - 1) % m.smCount < m.smCount :=
    Nat.mod_lt _ m.smPos
  omega

/-- **A1**: covering the full GEMM output requires at least the wave count
    times per-wave tile-work. -/
theorem wave_count_covers_work (g : Gemm) (t : Tile) (m : Machine)
    (v : ValidPartition g t) :
    g.m * g.n * g.k ≤
      m.smCount * numWaves g t m * t.bm * t.bn * t.bk := by
  -- Chain: m·n·k ≤ tileCount·bm·bn·bk ≤ smCount·numWaves·bm·bn·bk.
  have h1 : g.m * g.n * g.k ≤ tileCount g t * t.bm * t.bn * t.bk :=
    tile_partition_covers g t v
  have h2 : tileCount g t ≤ m.smCount * numWaves g t m :=
    smCount_mul_numWaves_ge g t m
  have h3 : tileCount g t * t.bm * t.bn * t.bk ≤
            m.smCount * numWaves g t m * t.bm * t.bn * t.bk := by
    exact Nat.mul_le_mul_right t.bk
      (Nat.mul_le_mul_right t.bn (Nat.mul_le_mul_right t.bm h2))
  exact Nat.le_trans h1 h3

/-! ## A4 — TMEM budget compliance. -/

/-- A tile is compatible with the machine's TMEM budget iff the accumulator
    footprint fits. For a `BM × BN` output accumulator at `4` bytes per
    fp32 element, the budget check is `bm · bn · 4 ≤ tmemBytes`. -/
def tmemFits (t : Tile) (m : Machine) : Prop :=
  t.bm * t.bn * 4 ≤ m.tmemBytes

/-- **A4**: TMEM fit is monotone in tile area. Shrinking a tile that fits
    yields a tile that still fits. -/
theorem tmemFits_of_le (t t' : Tile) (m : Machine)
    (h_bm : t'.bm ≤ t.bm) (h_bn : t'.bn ≤ t.bn)
    (h : tmemFits t m) : tmemFits t' m := by
  unfold tmemFits at *
  have h1 : t'.bm * t'.bn ≤ t.bm * t.bn := Nat.mul_le_mul h_bm h_bn
  have h2 : t'.bm * t'.bn * 4 ≤ t.bm * t.bn * 4 := Nat.mul_le_mul_right 4 h1
  exact Nat.le_trans h2 h

/-! ## A5 — Tail-wave padding cost. -/

/-- Padding waste = tiles emitted minus useful tiles (`m · n / (bm · bn)`).
    Bounded by `smCount - 1` — the last wave has at most that many idle
    tile slots. -/
theorem tail_padding_bound (g : Gemm) (t : Tile) (m : Machine) :
    m.smCount * numWaves g t m ≤ tileCount g t + (m.smCount - 1) := by
  unfold numWaves ceilDiv
  rw [if_neg (Nat.pos_iff_ne_zero.mp m.smPos)]
  have hdiv :
      m.smCount * ((tileCount g t + m.smCount - 1) / m.smCount)
      + (tileCount g t + m.smCount - 1) % m.smCount
      = tileCount g t + m.smCount - 1 :=
    Nat.div_add_mod _ m.smCount
  have hmod :
      (tileCount g t + m.smCount - 1) % m.smCount < m.smCount :=
    Nat.mod_lt _ m.smPos
  omega

/-! ## E4 — Wave-count → makespan relationship. -/

/-- Per-tile compute time (cycles). The scheduler treats every wave as
    consuming exactly `waveTime`, so `makespan = numWaves · waveTime`. -/
def waveMakespan (g : Gemm) (t : Tile) (m : Machine) (waveTime : Nat) : Nat :=
  numWaves g t m * waveTime

/-- **E4**: makespan is monotone in wave count (given fixed per-wave time).
    Halving the wave count via a bigger tile at least halves the makespan
    (modulo per-tile compute time changes, absorbed into `waveTime`). -/
theorem waveMakespan_mono (g : Gemm) (t t' : Tile) (m : Machine)
    (waveTime : Nat)
    (h : numWaves g t' m ≤ numWaves g t m) :
    waveMakespan g t' m waveTime ≤ waveMakespan g t m waveTime := by
  unfold waveMakespan
  exact Nat.mul_le_mul_right waveTime h

end Plow.Wave
