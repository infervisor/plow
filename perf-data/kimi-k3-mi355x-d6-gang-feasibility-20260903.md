# Decode fused-XReduce gang feasibility (MI355X, 2026-09-03)

Status: CPU/design gate only. No runtime integration, default change, or GPU result.

The compile-only carrier prototype is isolated in
`/tmp/plow-d6a.0EmPZl/src/runtime/bench/amd/d6_xreduce_attnres_resource.hip`; it is not a tracked
runtime arm. It calls the production one-shot XReduce, performs the per-XCD election and all four
AttnRes phases, and reserves the production interpreter's LDS envelope. Its typed arguments and
marker globals make the intended 14-workgroup, 512-thread, eight-XCD contract explicit.

## Eligible full-graph seams

The current T=1 BF16 TP8 graph has 278 one-shot XReduce packets. Exactly 186 use 14 workgroups
and have one adjacent coarse AttnRes consumer; the other 92 use seven workgroups and feed two
routed-expert GEMVs. All 186 eligible pairs have `T=1`, `H=7168`, TP8, a one-workgroup AttnRes,
and a fused post-RMSNorm. Their exact prefix contracts are:

| prefix contract | sites |
|---|---:|
| XReduce output is AttnRes `t1` directly | 8 |
| XReduce output is materialized-residual `t6` | 92 |
| XReduce output is materialized-residual `t7` | 86 |

All 186 are adjacent in topological instruction order. Eight carry the optional ring push. Live
row counts are one site at `nb=0`, 24 each at `nb=1..7`, and 18 at `nb=8`. Eligibility is the
graph/tensor/shape/dependency contract above, never a model or layer name. The expected structural
win is 186 removed packet boundaries and 186 removed one-workgroup AttnRes entries per token. The
initial AttnRes with no XReduce producer remains unchanged.

## D6-A carrier

Keep each XReduce packet's 14 entries and execute AttnRes before those entries return. L2 placement
puts two entries on domains 0..5 and one on domains 6..7. After each entry materializes its exact
BF16 XReduce range, a same-L2 arrival elects the last entry as that domain's leader. The eight
leaders execute AttnRes; the other six entries wait on an XCD-local completion flag. No follower
returns early, so the ordinary outgoing counter still reaches 14 only after the fused consumer is
complete. Downstream thresholds and counter meaning do not change.

Leader `d` emulates original wave `d`: active lanes use
`old_thread = 64*d + lane` and the original stride of 512. This preserves, without reassociation:

- residual materialization's nested BF16 boundaries;
- optional ring-push ownership;
- each wave's statistics accumulation and `wave_sum` tree;
- domain-0's wave fold in order 0 through 7;
- softmax's existing half-wave reductions;
- each output element's row accumulation order and BF16 materialization;
- RMSNorm accumulation over the re-read BF16 output, then the old wave fold order.

The fused-RMS route has these cross-XCD phases:

1. XReduce-domain ready: all 14 XReduce ranges are published before any leader reads a stripe.
2. Statistics: eight old-wave partials publish; domain 0 folds and publishes probabilities.
3. Mix/norm: leaders write disjoint BF16 output stripes and publish old-wave norm partials; domain
   0 folds and publishes the inverse RMS.
4. Scale: leaders overwrite their disjoint stripes, then release local followers.

## Admission and deadlock proof

This route is valid only for the gfx950 placed global queue with eight physical XCD windows and
`PLOW_GQ_BATCH=1`.

Each domain queue is stable op-major. Every one of the 14 carrier entries takes the ordinary full
predecessor gate before entering the fused body. A cursor cannot pass an entry without assigning
it to a resident workgroup. Therefore, if a later entry is claimed, every earlier carrier entry in
that domain is already owned. Once any carrier passes its predecessor gate, every producer slice
is complete, so no gang participant can be holding work needed by another carrier. At most two
workgroups per domain are held inside D6-A, well below the measured 32 workgroups/XCD. Downstream
workgroups may claim and wait, but cannot strand an unclaimed carrier: passing it proves ownership.

Static dispatch, an unplaced/global single queue, a non-eight-domain topology, fine wait/successor
lists, `PLOW_GQ_BATCH != 1`, or an object without the D6 marker must fail closed. HSA's
`launch_cooperative` wrapper does not request cooperative admission from ROCr. Safety therefore
also retains the existing exact-grid contract (`grid == device CUs == 256`), the measured 32/XCD
mapping, and the build/runtime resource checks. This limitation must be explicit; a function name
is not an admission guarantee.

## Scratch and counter ABI

No kernarg growth is required. Put a byte offset to per-op scratch in a currently unused fused
instruction integer slot; resolve it relative to the selected local-counter bank, so double-bank
re-arm and lifetime already match one decode dispatch.

One conservative, independently cache-lined site needs:

| region | generic maximum (`nb<=16`) | current graph (`nb<=8`) |
|---|---:|---:|
| statistics, `8 * 2 * (nb+1)` f32 | 1,088 B | 576 B |
| probabilities, rounded to 128 B | 128 B | 128 B |
| RMS partials, rounded to 128 B | 128 B | 128 B |
| five global phase lines | 640 B | 640 B |
| eight local-arrival + eight local-release lines | 2,048 B | 2,048 B |
| total, with statistics rounded to 128 B | 4,096 B | 3,584 B |

The simple per-site allocation is 651 KiB for the 186 current sites, or 1.27 MiB with two
banks. This is deliberately conservative. The existing HIER observe-election and observe-release
lines can potentially carry the local body arrival/release using advanced expected values after
all entries pass the input gate, reducing the increment to 279 KiB per bank, but that alias
must first be proved against the post-body publish path. Do not reuse HIER's publish-arrival line;
the ordinary successor signal still needs it starting at zero.

## Isolated gate required before integration

Build a typed-argument gfx950 object containing production one-shot TP8 XReduce followed by the
four D6-A phases. Compare it with the current two-packet producer+consumer path, not with either
body alone. Sweep every live `nb=0..8`, direct and both materialized-residual operand orders, ring
push on/off, and gamma on/off. The hard oracle is bit equality for the BF16 prefix, raw mix, normed
output, reduction scratch, and successor counter values on every rank. Also require deterministic
rank-order reduction and repeated-run identity.

The object must be wave64 with zero VGPR spills, zero SGPR spills, and zero private segment. Record
VGPR/SGPR/LDS and occupancy against control. Then compile the same arm into a clean decode
interpreter and reject it on any spill/private regression. Only a positive total-boundary timing
and unchanged interpreter resource envelope justify graph/runtime integration or a GPU full-model
gate.

The compile-only prototype passes its preliminary resource gate on ROCm 7.14/gfx950:

| wave | max WG | intended grid | XCDs | occupancy | VGPR | SGPR | VGPR spill | SGPR spill | private | LDS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 512 | 14 | 8 | 2 waves/SIMD | 48 | 58 | 0 | 0 | 0 B | 147,464 B |

Eight waves per 512-thread workgroup at two waves/SIMD, together with 147,464 B LDS, admits one
workgroup per CU: the same 256-workgroup placed-interpreter envelope in which the 14 carrier
entries run. The code-object globals contain wave `64`, grid `14`, and domains `8`; a future loader
must verify all three rather than trust a build flag. Compile logs and the unbundled object are
`/tmp/plow-d6a.0EmPZl/genco.log`, `/tmp/plow-d6a.0EmPZl/resources.log`, and
`/tmp/plow-d6a.0EmPZl/d6.elf`.

This is only a resource/admission result. Exactness and total producer+consumer timing remain
unmeasured; the prototype must not be integrated or enabled from this evidence alone.
