# 21. Decode context parallelism (DCP)

DCP shards the MLA latent KV cache across ranks instead of replicating it, so long-context
concurrency stops being bounded by how much KV one rank can hold. It is a **capacity** lever, not
a latency one.

The full design — merge algebra, per-file work plan, staged rollout — is `/workspace/plow-glm53/archive/dcp-design/DCP.md`.
This chapter is the architectural summary: what DCP is, why the packet already contained the
primitive, and what it changes across the compiler, the runtime and admission.

## Why KV is replicated today

Tensor parallelism cuts every weight on a head or hidden axis. The MLA latent has no head axis to
cut: the `[512]` compressed KV and its `[64]` rope half are **shared by all heads**, so
`asset/shard.rs` leaves them replicated and every rank writes every row
(`crates/plowrt/src/asset/shard.rs:66-74`, `crates/devgen/src/mla.rs:4116-4118`).

Measured on GLM-5.3 TP8 / MI300X, that costs **55,608 B per token per rank** — the ~576 B latent
times ~92 layers, eight times over. Two consequences, both observed:

- KV admission seats **12** sequences at 70k context (3.87 GiB/seq against a 50.03 GiB budget).
- The KV pools reserve `max_ctx x ladder_top x 55,296 B` of VA per rank. At `max_ctx` 81920 and a
  decode ladder topping at 32 that is **135 GiB**, which on top of 93.41 GiB of weights does not
  fit a 192 GiB device — it fails at load with
  `hsa_amd_vmem_handle_create: HSA_STATUS_ERROR_OUT_OF_RESOURCES`.

The second is why the ladder top and `max_ctx` trade against each other, and why the 70k serving
configuration runs a narrower ladder than the hardware would otherwise allow.

## The idea

The only axis left is the KV **row** axis. Shard rows across the `d` ranks of a DCP group; each
rank attends over its own shard **for every head of the group**; one collective folds the per-rank
softmax partials back into the head-sharded layout everything downstream already expects.

## The packet already computes the right thing

Attention is not emitted as a single fused output. It is emitted as **partials plus softmax
statistics**, then merged:

```
FlashMlaPrefillFp8  Opart<-act.opart mlpart<-act.mlpart Qabs<-act.qa ...
MlaMergeFold        O<-act.oat Opart<-act.opart mlpart<-act.mlpart Wuv<-...v_absorb.weight
                    | n_head=8 V=256 nsplit=4
```

| tensor | layout | dtype |
|---|---|---|
| `act.opart` | `[b][t][head][nsplit][DK=512]` | f32 |
| `act.mlpart` | `[b][t][head][nsplit][2]` = (m, l) | f32 |
| `act.oat` | `[t][nh_l][V=256]` | bf16 |

`nsplit` is already a split **of the KV axis**, chosen by `glm_nsplit(ctx, nh_l)`. DCP is the same
split with the pieces on different devices, so the cross-rank fold is *the existing merge with a
permuted split axis* — not a second attention path. That is the central design claim, and it is
what keeps `o_proj`, `XReduceScatter` and the sequence-parallel seams byte-for-byte unchanged.

Three supporting pieces also already existed:

- **`DevOp::XFlashMerge = 27`** — a reserved stub whose own doc comment reads *"Context-parallel
  cross-GPU flash LSE-merge… folds N peers' (O_partial, m, l) over their KV-position shards"*.
  Dispatch present, body deferred.
- **`DevOp::XAllToAllHeads = 160`** — the cross-GPU head transpose, both directions.
- **`emit_glm_rowsplit_attn`** — already brackets one attention call with two head all-to-alls.
  It splits *query* rows so it needs no LSE merge, but its peer slots and scratch are the template.

## Lowering

Per attention site, within a DCP group of `d` ranks:

1. `XAllToAllHeads(Q, dir=0)` — gather the group's heads so this rank holds **all** `nh` heads.
2. Flash attention over **this rank's KV shard only**, at `ns_local` splits, producing
   `(O_partial, m, l)`.
3. `XAllToAllHeads(partials, dir=1)` — return to the head-sharded layout, now carrying `d` times
   as many splits.
4. `MlaMergeFold` with `nsplit = d * ns_local`.

Step 3 restores exactly the layout the non-DCP path presented at the `MlaMergeFold` input, which
is why nothing downstream changes. The merge happens in the 512-wide latent **before** the `W_uv`
absorb: absorbing first would halve the wire but requires a `wuv_full` replica (~1.5 GiB) — rejected.

`ns_local = glm_nsplit(ctx/d, nh_l*d)`: the DCP degree consumes the split budget, which also bounds
the wire cost.

## Sharding policy

Block-cyclic on the **absolute** row index at a page `P` (default 64):

```
shard(row)     = (row / P) % d
local_row(row) = (row / (P*d)) * P + row % P
```

Decisive argument: decode grows the sequence every step, so any policy needing migration when a
sequence crosses a shard boundary is disqualified. Block-cyclic growth is free. Imbalance is at
most `P` rows between any two shards at every length, and keying ownership on the *absolute* index
is what keeps the prefix-cache key rank-independent.

A contiguous range is rejected: the last shard holds every newly written row during decode, so one
rank does all the work.

## Merge math

Base-2 throughout, matching `fa_merge_ml` (`runtime/amd/op_attention_common.h:1252-1294`):
`M = max_j m_j`, `w_j = (m_j == FA_NEG_INF) ? 0 : exp2(m_j - M)`, accumulated in ascending
`j = shard*ns_local + split`.

Reproducible run to run for a fixed configuration. **Not** bit-identical to the non-DCP packet —
the split count and partition differ, so the summation order differs. That is stated rather than
claimed away. Empty partials stop being a corner case and become routine under DCP, since a short
sequence need not reach every shard.

## What it changes elsewhere

- **Admission.** `bytes_per_token` becomes per-shard. Seating must use the **worst** shard
  (`max_local_rows`), not the mean: `len/d` under-counts by up to a page and would OOM one rank
  while the budget still reported headroom.
- **KV pools.** Sized from `local_capacity(ctx)` rather than `ctx`. Identity at `d = 1`.
- **Prefix cache.** The sharp edge. `attach_ranks` requires all ranks to agree, and
  `snapshot_bytes` copies a contiguous `[0, rows)` scale range — a strided shard cannot express
  that as one copy, and a mistake there is silent corruption of a cache *hit*.
- **`kidx`.** The DSA indexer cache stays replicated: combining it across ranks is a distributed
  top-k, not an LSE merge. That leaves a 13.1 GiB/rank floor.

Pool attribution at the measured geometry: `ckv` 97.5 GiB, `krot` 24.375 GiB, `kidx` 13.125 GiB.
Sharding the first two covers 121.9 of the 135 GiB.

## Cost and benefit

Roughly neutral on latency — about 7 ms of latent bandwidth saved against 3-4 ms of added
collective. The prize is capacity: **135 GiB/rank of pools becomes ~17 GiB, and seats at 70k go
from 12 to ~96.**

Treat it as a throughput and context-length lever. It will not improve TTFT.

## Behaviour by configuration

| configuration | behaviour |
|---|---|
| single GPU | `d = 1`, every map the identity, packet byte-identical |
| TP, DCP off | today's replicated layout, unchanged |
| TP + DCP | `d` divides `tp`; groups of `d` consecutive ranks |
| NVIDIA | lowering is backend-neutral; device bodies are stage 4 |

MLA is what makes DCP attractive. For GQA/MHA the KV already has a head axis that TP cuts, so DCP
is only worth it once `tp` exceeds the KV head count.

## Knobs

- `PLOW_DCP` (`--dcp`) — DCP degree. Unset or 1 is the replicated layout.
- `PLOW_DCP_PAGE` (`--dcp-page`) — rows per page. Power of two, at least the flash KV tile (32),
  must divide the pool's block rows. Default 64.

Both registered in `crates/devgen/src/knob_spec.rs`, with `dcp_page_needs_dcp` tying the second to
the first.

## Status

Stage 1 — the algebra, the layout, the knob surface, KV sizing and the seating predicate — is
implemented: `crates/packet/src/dcp.rs`, `crates/plowrt/src/sched/kv_budget.rs`, and the emitter's
`local_capacity` sizing. `dcp_degree_one_emits_a_byte_identical_packet` pins the `d = 1` identity
over two full GLM emits by sha256.

`dcp_layout` **panics for `d > 1`**: the declaration exists but its attention lowering does not, and
a short extent must not be shippable. Stages 2-4 — the emitter lowering, the runtime,
`XFlashMerge`'s body, prefill, `1 < d < tp`, NVIDIA bodies and `kidx` — are tracked in the design
document.

A known shared prerequisite: the peer-slot table (`PeerLayout::with_slots`) accepts only 3 or 6
slots today, which also blocks the already-landed row-split arm.
