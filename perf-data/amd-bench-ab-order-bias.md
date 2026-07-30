# `amd-bench --checkpoint` A/Bs are ORDER-BIASED by 5–10 ms. Read this before trusting one.

**The trap:** run two arms sequentially in one script, each with `--checkpoint`, and the SECOND
of every pair is penalised by 5–10 ms/token. On Kimi-K3 at TP8 each bound run re-reads **195 GiB
per rank**, and the pair does it twice back to back. The penalty lands on whichever arm you put
second, so it looks exactly like a regression in that arm.

This is the shape almost every A/B in this campaign has used.

## The control that proves it, and the numbers

Same two blobs, same binary, same objects, 200 steps, `--checkpoint`, differing only in whether
K3's `routed_expert_up_proj` is column-parallel — a change independently measured at **-0.44
ms/token**, i.e. an order of magnitude smaller than the artefact.

| order within the pair | ctx | first arm | second arm |
|---|---|--:|--:|
| base, then shard | 8 000 | base 37.861 / 38.104 | shard 43.822 / **47.544** |
| shard, then base | 8 000 | shard 37.555 / 38.443 | base 37.636 / **44.086** |
| shard, then base | 32 000 | shard 40.898 / 41.122 | base 38.672 / **44.853** |

Base-first says the shard is 6–9 ms SLOWER. Shard-first says the opposite. **The sign follows the
ORDER, not the blob.** First-in-pair runs agree at 37.6–37.9 ms for both arms; the 44–47 ms
figures are all second-in-pair.

## What to do instead

**Prefer UNBOUND weights for an A/B of anything structural.** Omitting `--checkpoint` keeps the
schedule, the packet count, the workgroup counts and every buffer SIZE real — which is the whole
of what a sharding, fusion or geometry change touches — and only the VALUES are meaningless. It
also removes the ~100 s load per run, which is what makes 6 reps affordable. The same comparison
that was unresolvable bound came out clean unbound:

| ctx | replicated (min of 6) | column-parallel (min of 6) | delta | within-arm spread |
|---|--:|--:|--:|--:|
| 8 000 | 33.846 | 33.405 | **-0.441 ms** | 0.02–0.08 |
| 16 000 | 34.668 | 34.238 | **-0.430 ms** | 0.76–0.90 |
| 32 000 | 36.052 | 35.736 | **-0.316 ms** | 0.15–0.16 |

Sharded arm ahead in all 18 pairs.

Unbound is NOT valid for everything: routing is data-dependent, so with zeroed weights the router
picks a degenerate expert set. That changes which experts stream, identically in both arms — fine
for an A/B, wrong for an absolute number and wrong for anything whose cost depends on WHICH
experts are chosen.

**If you must run bound:**

1. **Balance the order** — run `A,B` then `B,A` and compare like positions, or discard
   second-in-pair entirely. A one-line reversal is what turned an apparent 6 ms regression into a
   measurement bug here.
2. **One load per process is the confound.** The penalty is per-load, not per-step, so more
   `--steps` in one process is nearly free while another rep is not.
3. **Report minima and spread, never a single rep.** Bound within-arm spread on a contended box
   reached 37.8 → 47.5 ms for the same blob.

## Why it is not just cache warmth

The GPU-side dispatch is identical between reps; what differs is the host: 195 GiB/rank of
checkpoint mmap re-faulted through the page cache and staged over the upload ring, twice per pair,
against a box other agents are also loading on. `PLOW_SHARE_CKPT` and the prefetch pool reduce it
and do not remove it. Treat any bound delta under ~10 ms as unmeasured until the order is
balanced.
