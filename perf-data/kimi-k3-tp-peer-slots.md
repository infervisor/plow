# Kimi-K3 TP: the third peer slot, and the ordering dependency it would buy back

**Status: the third slot LANDED, but for a different reason than this note proposed, and the
ordering dependency below is still paid.** `PARTIAL_SLOTS` is now 3 and the host binds
`act.ug_tp` at `scratch_base + 2*slot_b` — not to give the shared expert's `down_proj` a private
slot, but to give the newly COLUMN-PARALLEL `routed_expert_up_proj` a peer-visible home for its
partial, which `d_xreduce` then ALL-GATHERS inside the shared expert's existing all-reduce
(`gcols`/`row_w`, `emit_xreduce_gather`). That shard is worth **-4.14 GB/rank/token** of decode
weight traffic, measured off the blob: the bf16 `Gemv` family goes 18.157 -> 14.021 GB.

So the layout change this note priced as a risk has now been taken anyway, and the peer region
is one slot bigger for every TP model. **The cheap A/B below is still the right next step** and
is still unrun: nothing about the gather removes the `c_cmb` edge, because the shared expert's
`down_proj` still writes slot 0. What a FOURTH slot would buy is exactly what a third was
proposed to buy here, and the arithmetic is unchanged.

Original note follows.

## What is there now

(As written, when `PARTIAL_SLOTS` was 2. Slot 2 now exists and is the up-projection gather slot;
the reuse of slot 0 by two collectives, and the dependency it costs, are unchanged.)

Two partial slots per rank, `PARTIAL_SLOTS = 2` (`crates/plowrt/src/exec/tp.rs:142`), each
`max_tokens * hidden * 2` bytes, with the cross-GPU counter region after them. The host binds
them by literal tensor name (`crates/plowrt/src/exec/amd.rs:2875-2876`):

    act.og_tp -> scratch_base            slot 0
    act.dg_tp -> scratch_base + slot_b   slot 1

`d_xreduce` (`runtime/amd/op_collective.h:132`) sums `peer_scratch[r] + slot_bytes` over every
rank and writes `out` **without reading it**, so a rank's own partial reaches the sum only by
being in its own peer region at that offset. That is the contract `21f1420` restored.

A K3 MoE layer now has **three** row-parallel producers and therefore three collectives:

| # | producer | writes | slot | reduced into |
|---|---|---|---|---|
| 1 | `o_proj` (attention) | `act.og_tp` | 0 | `act.l{n}.attn_full` |
| 2 | `MoeCombine` (routed experts, latent width) | `act.dg_tp` | 1 | `act.l{n}.moe.ylat` |
| 3 | `shared_experts.down_proj` | `act.og_tp` | 0 | `act.l{n}.moe.sh_down` |

Three collectives, two slots. Slot 0 is reused inside one layer.

## Why that reuse costs an ordering dependency

The one-shot all-reduce's gate says every peer **arrived**. It does not say every peer finished
**reading**. A rank may leave collective 1 as soon as it has read all eight slot-0 partials — a
peer may still be reading *its* slot 0 at that moment. So overwriting slot 0 is safe only after
something guarantees every peer has left collective 1, and the only thing that does is an
intervening collective: this rank cannot pass collective 2 until every peer has signalled it, and
a peer signals collective 2 only after leaving collective 1.

Hence `21f1420` makes the shared expert's down GEMV depend on `c_cmb` (collective 2) as well as
on its own GLU. Before that dependency the shared-expert chain hung off the block's `deps` alone
and could reach its slot-0 write while a peer was still reading slot 0 for the attention reduce.

**The cost is the lost overlap.** The shared expert's `gate|up|situ` GEMV still runs concurrently
with the routed chain; only the down GEMV is pushed behind it. Per MoE layer that adds the down
GEMV's own latency to the critical path.

## What a third slot would change

* `PARTIAL_SLOTS: 2 -> 3` (`exec/tp.rs:142`). Peer region grows by one `max_tokens * hidden * 2`
  slot per rank — 58.7 MB at `max_tokens = 4096, hidden = 7168`, against ~190 GiB of weights.
  Nothing.
* A third literal bind, `act.sg_tp -> scratch_base + 2 * slot_b` (`exec/amd.rs:2875`), and the
  matching `TpBind` field.
* `K3Tp` gains an `sg` handle; `emit_k3_latent_moe` writes the shared down GEMV to it, reduces
  from slot 2, and **drops the `c_cmb` dependency** — the overlap comes back because slot 2 is
  written once per layer and the next write is a whole layer away, behind two intervening
  collectives.
* `PeerLayout::new`'s `xctr_off` moves, which is why this is a layout change and not a constant
  bump: every rank's counter region shifts and the `partial_off` bound check changes with it.
  `plowrt/tests/multi_gpu.rs` pins the layout and will need its expectations updated.

Sanity: nothing else in the tree emits more than two collectives between reuses, so raising
`PARTIAL_SLOTS` is inert for GLM/Gemma/Qwen — but it is inert *at a different peer-region size*,
so every TP model re-binds. That is the risk, and it is why this waits for the re-baseline.

## The measurement I would expect, and the cheaper one to take first

**Take the cheap one first.** Price the dependency before paying for the layout change: add an
emit knob that drops the `c_cmb` edge (numerically unsafe — it is the race the dependency exists
to prevent — so it is an *instrument*, the same standing as `PLOW_K3_ABLATE` and
`PLOW_CHAIN_BYPASS`, and it must never be a serving mode). A/B the full 93-layer TP8 asset with
and without the edge. The delta IS the recovery a third slot would buy, exactly, with no host
change and no layout risk.

Expected magnitude, derived rather than measured:

* the shared down GEMV streams `hidden * (shared_inter / tp) * 2 B` = `7168 * 768 * 2` = **11.0 MB
  per layer per rank**; at MI355X HBM rates that is ~2 µs of body;
* one added serial packet on a decode chain costs ~5.3 µs of protocol by this tree's own
  measurement (`d_norm_residual_norm`'s note in `runtime/amd/op_norm.h`, +1.28 ms over 120 sites);
* so ~2–7 µs × 92 MoE layers = **0.2–0.65 ms/token**.

Against a decode token in the ~36–40 ms range that is **0.5–1.6%**. My recommendation: if the A/B
comes in under ~0.3 ms/token, do not do the host change at all — write the number down here and
keep the dependency, which is the version that is obviously correct.

> Caveat on the denominator: every K3 decode figure this campaign produced was measured on a model
> that was computing the wrong thing (unbound experts, or the shared expert overwritten by the
> attention output on 92 layers). Treat 36–40 ms/token as an order of magnitude, not a baseline,
> until the re-baseline lands.

## The regression gate this belongs behind

`scripts/k3_tp_equivalence.sh` — tp=1 vs tp=8 on the same asset, cosine floor on the full logit
vector. Any change to the peer-slot layout must keep it green. It is the only control in the tree
that can see a peer-slot contract violation: measured discrimination at `K3_NLAYERS=2` is
cos 0.9466 broken against 0.999986 fixed.
