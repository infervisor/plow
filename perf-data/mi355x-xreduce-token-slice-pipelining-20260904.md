# Token-slice pipelining of a producer GEMM and its two-shot XReduce — design and prototype

Date 2026-09-04. Kimi-K3, 8x MI355X (gfx950, 256 CU, 8 XCDs), TP8, T8192 prefill. Plan item
v3 §5 P-C / P4. Base `f8aa24d` (branch `codex/d1-moe-decode-rule`).

## 0. Verdict up front

* The emit-side slicing is implemented and exact: `PLOW_XR_SLICES=2` splits every K3
  attention seam (`o_proj` → `XReduceTwoShot`) into two row bands with independent counter
  chains, the two-shot needs **no** new kernel operand (its row window `e0` already exists),
  and the unflagged packet is byte-identical to the published baseline
  (`sha256 f1bf783d…` on both). Tests pin the shape and the dependency graph.
* **In the current gfx950 execution model the bands cannot overlap.** The prefill grid is one
  workgroup per CU (147,504 B of the 160 KB LDS; 256 VGPR × 8 waves fills the 512-register
  file, so no second kernel can co-reside), the two-shot needs all 256 workgroups' loads in
  flight to reach its 0.73 ms fabric floor (`mi355x-xreduce-nwg-sweep-20260904.md`; the
  in-network `PLOW_XR_CUS=128` cell is +95 ms, bit-exact), and the global queue hands entries
  out on one monotonic cursor per (segment, XCD) — a workgroup inside a collective claims
  nothing else. So a 256-workgroup collective consumes the whole grid for its whole envelope,
  whatever the packet order. Slicing at packet granularity therefore buys only the tail
  overlap between band 1's GEMM stragglers and band 0's collective start (tens of µs), and
  pays one extra rendezvous pair per seam. Expected A/B: **+2 to +8 ms TTFT**, not a saving.
  That number is the slicing tax any real overlap mechanism (§6) must first recover.
* The "collective fully hidden" bound of ~200 ms is unreachable even with free concurrency:
  overlap can hide at most `min(producer, collective)` per seam, and every K3 producer is
  shorter than its collective (§4) — the honest ceiling is ~118 ms with infinite slices,
  ~61 ms at K=2, ~91 ms at K=4, all of them requiring a mechanism this tree does not have.

## 1. Which producer → collective pairs are sliceable

Per T8192 prefill (278 collectives, all `XReduceTwoShot` in the production packet — the FFN
seam's folded gather rides the two-shot since `PLOW_XR2_GATHER`):

| seam | producer (row-offset operand) | collective (band operand) | calls | sliceable |
|---|---|---|---:|---|
| attention | `o_proj` `GemmC5`/`GemmWide` bf16, `i4=a_row0 i5=c_row0` | `XReduceTwoShot` `[T,7168]` slot 0, `i5=e0` | 93 | **yes — prototyped** |
| dense layer-0 FFN | down `GemmC5` bf16 (`a_row0/c_row0`) | `XReduceTwoShot` `[T,7168]` slot 1 | 1 | yes, same mechanics, not worth a packet |
| MoE latent | `MoeCombinePf`, `i3=t_row0` | `XReduceTwoShot` `[T,3584]` slot 1 | 92 | yes in the packet (K2's `PLOW_GLM_XR_BAND` proves the emit), but the combine runs as the lean standalone segment (`lean MoE combine`, 92 segs) — a different kernel launch from its collective, so no in-grid overlap is possible without also un-isolating it |
| MoE FFN (folded gather) | shared-expert down `GemmC5` + up-shard `GemmC5` (both bf16, `a_row0/c_row0`) | `XReduceTwoShot` + gather, `i6=gslot i7=gcols`, `i5=e0` | 92 | yes with one emitter-side change: the gather arm indexes rows band-locally (`m = e / row_w`, `op_collective.h`), so band i needs `i6 = gslot + e0*2/tp`. No kernel change; the loader's slot-size inference reads `max(i2)` only, `i6` is free to carry the band offset |
| decode one-shots | — | `XReduce` | — | no (T=1) |

Constraints for any band: `T % (8K) == 0` so `T_slice % 8 == 0` and the per-rank reduce-scatter
ranges stay whole rows; `T_slice >= 512` so a band collective still saturates
(`elems/512 >= 256` workgroups) and a band is a whole number of GEMM tiles. Buckets that fail
either rule (512-row rung, decode) emit the unsliced seam, byte-identical.

## 2. How a slice is expressed in the packet

* GEMM band i: same opcode as the unsliced shape (`gfx950_prefill_tile(T, N, K)` is picked
  for the FULL `T`, so the K-loop and tile are identical), `i0 = T/K`, `i4 = a_row0 = i·T/K`,
  `i5 = c_row0 = i·T/K`, identical tensor handles (A, W, the `act.og_tp` partial).
* Collective band i: `i0 = n = (T/K)·H`, `i5 = e0 = i·(T/K)·H`, `i2` = region base
  (unchanged — the loader infers `slot_bytes` as `max(i2)`, so the band offset must ride in
  `e0`), fresh gate pair `(i3, i4)`. The dispatch already adds `e0` to `out`/`resid`/`out2`
  and `e0·2` to the window byte offset (`interp.hip` `PLOW_DOP_XREDUCE2`), so **the two-shot
  kernel needs no row-window argument** — `emit_xreduce_twoshot_band` (lib.rs) is the
  existing emit and the prototype reuses it. Every instrument (`PLOW_XR_RS_U`,
  `PLOW_XR_TRACE_PHASES`, NOWAIT/NOSIG, AGG) is untouched.
* Bit-exactness: each element is still reduced `r = 0..7` in f32 in strict rank order and
  rounded once; only WHICH rank owns the element's reduce-scatter changes (band-local
  `[e0 + n_b·r/8, e0 + n_b·(r+1)/8)` instead of `[N·r/8, …)`), and ownership never enters the
  arithmetic. The all-gather copies bf16. This is the same argument K2's banding shipped on
  (logits byte-identical), and the workgroup→element map inside a band is unchanged from the
  unbanded kernel's.

Emitted seam at T8192 (disasm of the sliced packet, program 8192, layer 0):

```
15 GemmC5          256 WG  M=4096 N=7168 K=1536  a_row0=c_row0=0
16 GemmC5          256 WG  M=4096 N=7168 K=1536  a_row0=c_row0=4096
17 XReduceTwoShot  256 WG  n=29360128 e0=0         gates (0,1)   waits: 15
18 XReduceTwoShot  256 WG  n=29360128 e0=29360128  gates (2,3)   waits: 16
19 AttnRes         256 WG                                        waits: 17, 18
```

## 3. Dependency graph, segments and the queue

Op order is **G0, G1, X0, X1** (both producers, then both collectives), not K2's
G0, X0, G1, X1. Counter edges: `G0, G1 ← output gate` (they share the producer), `X0 ← G0`,
`X1 ← G1`, `consumer ← X0, X1` (the prefix `Residual`, or `AttnRes` on a snapshot layer). The
test `sliced_seam_bands_are_independent_counter_chains` checks, from the packet's wait
table, that G1 is ready with only G0 retired, that X1 is ready with G1 retired while X0 is
still pending, and that the consumer needs both.

Same segment, not overlapping segments: all four ops are class-8 interpreter packets, so the
segmentation puts them in one ordered segment — the sliced 8192 program still has exactly 693
ordered segments, like the control. This is the only arrangement in which overlap is even
expressible: segments chain on the AQL barrier bit and segment-major dispatch only reorders
the enqueue across ranks, so two segments never run concurrently. Under
`PLOW_PHASE_OBJECTS=1`/`PLOW_XR_WAVE_RS=1` isolation each band becomes its own segment
(+1 boundary ≈ 80 µs per seam) and overlap is impossible by construction — do not combine.

What the global queue then does with this graph: the op-major stream is claimed on one
cursor per (segment, XCD). All 256 workgroups pass through G0 and G1 (256 entries each,
tiles round-robin), then claim X0's 256 entries — the first finishers of G1 enter the
collective while the last G1 tiles are still running, and that is the entire overlap
available: the two-shot holds every one of its workgroups until `gate_ag` sees
`nranks·nblk` arrivals, and a blocked workgroup claims nothing. K2's order G0, X0, G1, X1
only pipelines when the band collective runs on a CU-subset prefix so the remaining
workgroups walk past it to G1 (`glm52-band-pipeline-cusubset.md`), which on this part costs
the collective its fabric floor (+95 ms at 128 WG). With the collective at full width the two
orders are equivalent; producers-first keeps the two GEMMs back to back (no global-completion
wait between them) and gives X1's `gate_rs` a rendezvous that the ranks reach in lockstep.

Why nothing in the grid can run beside the collective: the prefill objects are
`256 VGPR / 108 SGPR, occupancy 2, 147,504 B LDS, 8 waves` — one workgroup per CU by LDS,
and 2 waves × 256 VGPR per SIMD fills the 512-register file, so a co-resident lean XR object
(`interp_xreduce_gq.elf`, 82 VGPR / 8,200 B LDS) on a second HSA queue cannot be placed
either. Occupancy 2 for the interpreter would need ≤ 80 KB LDS; co-residency of the lean
object would need the interpreter at ≤ 215 VGPR/wave.

## 4. Expected saving

Per-seam producer vs collective at T8192 (`kimi-k3-mi355x-current-attribution-20260904.md`,
`kimi-k3-mi355x-xreduce-phase-trace-20260904.md`):

| seam | producer | collective envelope | hideable = min(P, X) |
|---|---:|---:|---:|
| attention (93) | `GemmC5` ≈ 0.45 ms | full class ≈ 0.75 ms (RS 370 / gate2 80 / AG 300 µs) | 0.45 → 42 ms |
| MoE latent (92) | lean combine 0.42 ms | half class ≈ 0.42 ms | 0.37 → 34 ms |
| MoE FFN (92) | up-shard `GemmC5` ≈ 0.45 ms | gather class ≈ 1.0 ms | 0.45 → 41 ms |

* Fully hidden collective (~200 ms) — impossible: `P < X` on every seam, so even infinite
  slices and a free mechanism leave `X − P` exposed. Ceiling ≈ 118 ms.
* K slices with a real mechanism hide `P·(K−1)/K` per seam: K=2 ≈ 61 ms, K=4 ≈ 91 ms, before
  the added rendezvous (`13.6 + 0.0227·nblk` µs per pair isolated, ≈ 20 µs at 256; the
  in-network gate2 p50 is 80 µs but co-varies with data motion).
* This prototype (packet-level, current objects): overlap ≈ tail imbalance only; cost
  = +1 rendezvous pair + 2 packets per attention seam ≈ +20..+90 µs × 93 = **+2..+8 ms**.
  The A/B is still worth one seg-timing cell: it prices the slicing tax and confirms the band
  collectives run at the same per-byte rate as the whole one (each band moves half the
  bytes; the interpreter family should show +93 packets and ~unchanged XReduce2 time).

## 5. Prototype

* Flag: `PLOW_XR_SLICES=K` (emit_config `xr_slices`, threaded as `K3ModelCfg::xr_slices` →
  `K3Tp::oproj_slices(t)`; tests set the field, no env reads).
* `crates/devgen/src/k3.rs`: `emit_k3_oproj` (row-band `o_proj`, unsliced tile), both mixers
  return the band counters, `emit_k3_layer` emits one `emit_xreduce_twoshot_band` per band on
  the full `xr_cus` and hands every band's collective to the consumer.
  `crates/devgen/src/kda.rs`: the KDA mixer's two `o_proj` sites go through the same helper.
* Tests (`cargo test -p devgen --lib k3::tests::xr_slices k3::tests::sliced_seam`):
  `xr_slices_split_the_attention_seam_into_row_bands` (T=1024 → 93 seams, +186 packets, bands
  are T/2 rows with `T/2 % 8 == 0`, same tile as the control, `e0 ∈ {0, T/2·H}`, unique gate
  pairs, 512-row and decode buckets byte-identical) and
  `sliced_seam_bands_are_independent_counter_chains` (§3).
* Packet proof (`plowc --hf-dir /home/shaswot/models/Kimi-K3 --emit devblob --arch gfx950
  --gpu mi350 --num-gpus 8 --parallel tp --max-ctx 16384 --n-cu 256`, `K3_FULL=1`):
  unflagged `f1bf783dac96791b7116ffb549862c8206ba33351310c7c113504916611e8921` = control;
  `PLOW_XR_SLICES=2` → `440b45dbea711dd1b3459ce27e4b7261bec3375afdec416aa23247530bb445aa`,
  371 = 278 + 93 `XReduceTwoShot` in program 8192, 693 ordered segments (unchanged),
  `plow_config.h` identical — the control's packet-paired object set serves both.

## 6. What could actually overlap (not built)

1. **Fused band kernel**: a GEMM packet for band i+1 whose workgroups also stream band i's
   reduce-scatter (owned slice, 7/8 remote) — remote loads issued from inside the GEMM main
   loop, landing via direct-to-LDS loads or spare VGPRs, strict rank order kept per element.
   The only in-grid path given one workgroup per CU; it costs a new tile body and a register
   budget the 256-VGPR interpreter does not have, so it belongs in a lean standalone object
   (GEMM+RS), with the AG phase left as today's packet.
2. **Second HSA queue + co-resident lean XR object**: needs the interpreter at ≤ 215 VGPR/wave
   (or 4-wave XR at ≤ 128) and LDS ≤ 152 KB — dead until the fat prefill object shrinks.
3. **Fewer round trips** rather than overlap: fold the attention seam's prefix `Residual`
   into the two-shot's all-gather (`resid/out2` operands exist in the kernel; K3 does not
   wire them) — ~117 MB×3 of HBM traffic per layer, ≈ 0.1 ms × 93.

## 7. A/B command (parent session, 8-GPU lease)

Candidate assets: emit with `PLOW_XR_SLICES=2` into `<assets-xr2>` (copy the control's
`hsaco` pairing — `plow_config.h` is identical), then one exact 8192→1 cell per arm:

```sh
perf-data/tools/gpulease -n 8 xr-slices-seg env PLOW_PREFILL_SEG_TIMING=1 \
  plowrt bench --assets <assets> --rt-hsaco /tmp/k3-xr-phase-gate/hsaco-control \
    --random-input-len 8192 --output-len 1 --requests 1 --warmup-requests 1 \
    --concurrency 1 --parity-report --engine-diagnostics
```

Compare the interpreter family (expect +93 packets, `XReduce2` time ≈ unchanged) and the
endpoint TTFT (expect +2..+8 ms); the one-token checksum must match the control's
(`fnv1a64:35bef598a5574853` in the one-request timing mode).
