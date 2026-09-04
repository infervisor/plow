# Decode levers L7 (fused KDA decode arm) and L8 (claim-ahead GEMV weight prefetch) — status

Branch `codex/decode-l7-l8` (from `codex/amd-agent-harness` @ 98ddb25).
Plan rows: `decode-gap-plan-20260904.md` §4 L7 and L8. Flags: `PLOW_KDA_DECODE_FUSED_ARM`
and `PLOW_GEMV_PREFETCH` (emit env / `plowc` arg, and the decode object's `-D`). Default OFF
everywhere; the default packet is byte-identical (tested); flag-off objects carry no new code.

## L7 — the KDA decode chain as ONE dataflow-gated packet

Served per KDA layer (L3 folded or not): `KdaConv3` (256 WGs) -> `KdaStateStepG` (192 WGs) ->
`KdaGatedNorm` (2 WGs): three packets, three gates, `41 us/layer` on the critical path (plan §1),
vs vLLM's `fused_kda_decode` 8.4 us. The arm keeps op 112's slice map (one workgroup per
(head, value tile), H*D/BV = 192 at TP8) and folds the other two ops around it:

* **tile 0 of each head** runs the q/k conv for the head's 2·D channels (one channel per thread,
  `kda_fu_conv_chan` = `kda_conv_range`'s T=1 statements) and, under the L3 bit, the head's D f_b
  columns (`gemv_cols_wave`, L3's prologue verbatim, computed once instead of 16 times) and
  publishes each value as a tagged 8-byte word (32-bit round tag | bf16) — the L5 AttnRes
  rendezvous scheme (`op_k3.h`): relaxed agent-scope stores, no fences between the tiles;
* **every tile** convs its own BV v channels (each channel's window is rolled exactly once by
  exactly one workgroup), polls the head's q/k(/g) words (one word per thread), stages them
  into the SAME LDS layout op 112 uses and runs op 112's recurrence statement for statement
  (`block_sum` L2 norms, `wave_sum` over k, updated-state read-back, `f2bf`), then publishes
  its BV `o` words;
* the tile whose completion bump closes the head's `2·ntile` bumps for this token polls the
  head's D `o` words and runs op 103's row body (`d_kda_gated_norm`'s lane map, `wave_sum`,
  `rsqrtf(ss/D + eps)`, `norm_w`, `sigmoid(g_raw)`) into `y`.

Exactness: no arithmetic is re-derived — every conv, gate, step and norm expression is the served
op's expression on the same operands in the same order, and the exchange carries bf16 bits.
`o`, `state` and the conv windows are therefore bit-identical to the chain; the bench checks
that word for word (below). Deadlock: a tile waits only on a LOWER slice (its head's tile 0)
and never after the last-arriver election, so it is safe under `PLOW_GQ_BATCH=1` (the served
GQ object; `#error` otherwise) and under a claim-ahead queue; slices are `0..192` on distinct
CUs (24 per XCD). Round tag = `fetch_add(head counter) / (2·ntile) + 1`; scratch is
`act.<layer>kda.fused.scratch` (H·(1+4D)·8 = 49 KB/layer, loader-zeroed); a poll that never
completes (2^20) returns qNaN rather than hanging.

Packet (op 112, flags bit 3 `PLOW_KDA_F_FUSED_ARM`): `t0 = y`, `t1..t3` = raw q/k/v, `t4` =
f_raw (or f_a under bit 2), `t5` = beta_raw, `t6` = state, `t7` = u32 descriptor
`[wq, wk, wv, csq, csk, csv, A_log, dt_bias, norm_w, g_raw, scratch, eps_bits, W]`, `j1` = W_fb
under bit 2; deps = QKVG, f_b (or f_a), b_proj. `mix`/`o` are no longer written.
−2 packets/layer (−138/token; with L3 −207): flag-off emit from this tree 2073 decode
instructions -> 1866 flag-on (L7 + L3 + L8), Lean verified, 233 segments.

## L8 — claim-ahead weight prefetch for the dense GEMV

`decode-gap-plan` §3: a `Gemv` b=256 workgroup spends ~4.5 of its 9.1 us body on fixed
latency (descriptor, first HBM round trip, XCD ramp). With `PLOW_GEMV_PREFETCH=1` the
interpreter, right after claiming a `Gemv` entry and BEFORE polling its gate, issues that
slice's weight rows (`GV_BLOCKED` ownership `[slice·per, …)`, the map every bf16 GEMV body
uses) as `global_load_lds` into a dead 1 KB slot per wave (`gemv_prefetch_slice`, op_gemm.h:
no VGPRs, vmcnt-tracked), so the rows are L2-resident when the body streams them and the
fixed latency is paid under the gate wait. `cp_async_wait()` after the gate, before the arena
is reused. Loads only — the body, its bytes and its reduction order are untouched, so the
result is exact by construction (the bench still compares on == off word for word). The poll
is a vector atomic load, so its `vmcnt(0)` also drains the prefetch: a gate that is already
open pays the prefetch up front and the body then reads L2 — the `period = 0` row below is
that cost side. This is deliberately NOT the DMA-engine/`GV_DMA` idea op_gemm.h autopsies
(same bytes, same time, different issue pattern); it moves the first round trip under a wait
that the trace shows exists (8.77 ms of path gate per token).

Packet-inert: `features.gemv_prefetch` is recorded in the manifest from the emit setting so
the paired `plow_config.h` defaults `PLOW_GEMV_PREFETCH` for the decode object; no `requires`
(an unarmed object is still exact).

## Implemented

| piece | where |
|---|---|
| device: `d_kda_decode_fused_arm[_t<2>]`, `kda_fu_conv_chan`, `kda_fu_put/get`, descriptor enum, `plow_kda_decode_fused_arm_1` marker, `#error` on `PLOW_GQ_BATCH > 1` (all under `#if PLOW_KDA_DECODE_FUSED_ARM && PLOW_BUCKET_DECODE`) | `runtime/amd/op_kda.h` |
| device: `gemv_prefetch_slice`, `PLOW_GEMV_PF_MAX_BYTES` (128 KB) slice cap, `plow_gemv_prefetch_1` marker (under `#if PLOW_GEMV_PREFETCH && PLOW_BUCKET_DECODE`) | `runtime/amd/op_gemm.h` |
| interpreter: op 112 arm takes the fused body under flags bit 3; claim loop prefetches a `Gemv` slice above the HIER leader guard and waits `vmcnt(0)` after the gate | `runtime/amd/interp.hip` |
| CMake options `PLOW_KDA_DECODE_FUSED_ARM`, `PLOW_GEMV_PREFETCH` (OFF) -> `-D…=1` on decode objects | `runtime/CMakeLists.txt` |
| emit flags `kda_decode_fused_arm`, `gemv_prefetch` (default false) | `crates/devgen/src/emit_config.rs` |
| K3 mixer: decode + plain Conv3->StepG chain only (composes with L3) emits the single packet; `KdaCfg::supports_decode_fused_arm` (D=128, BV=8, W<=8, H·D/BV <= n_cu), `fused_arm_scratch_bytes` | `crates/devgen/src/kda.rs` |
| manifest: `features.kda_decode_fused_arm` (bit 3) -> `requires PLOW_KDA_DECODE_FUSED_ARM=1`, `plow_config.h` `PLOW_PACKET_REQUIRES_KDA_DECODE_FUSED_ARM` + `#ifndef` default; `features.gemv_prefetch` -> `#ifndef PLOW_GEMV_PREFETCH` default | `crates/devgen/src/manifest.rs` |
| loader: `DECODE_ARM_MARKERS` entry (refuses a fused packet on an object without the arm) | `crates/plowrt/src/exec/amd.rs` |
| coverage: a bit-3 step counts as the gated norm | `crates/plowc/src/fusion_coverage.rs` |
| op doc / flags reference | `crates/packet/src/dev.rs`, `docs/flags-reference.md` |
| tests: `kda::tests::fused_arm_replaces_the_chain_and_leaves_the_default_packet_byte_identical` (unset == "0" byte-for-byte, −2 packets, descriptor contents, +L3 composition, prefill untouched), `k3::tests::kda_decode_fused_arm_removes_two_packets_per_kda_layer` (69 layers, 192 slices each), `manifest::tests::kda_decode_fused_arm_is_a_decode_object_requirement` | devgen |
| 1-GPU microbenches | `runtime/bench/amd/kda_decode_fused_arm_bench.hip`, `runtime/bench/amd/gemv_prefetch_bench.hip` |

## Evidence (2026-09-04, MI355X, one GPU, `gpulease -n 1`)

### L7 — `/tmp/l78/kda_decode_fused_arm_bench 12 128 8 4 64 200`

H=12, D=128, BV=8, W=4 (conv3 at 256 WGs, step/fused at 192, norm at 2); 64 draws of fresh
random raw q/k/v, windows, taps, f_raw / f_a+W_fb (alternating), beta, A_log, dt_bias, o_norm,
output-gate logits and state; 200-rep timing:

```
conv3 alone (256 WGs)                                    3.06 us
step alone (192 WGs)                                     4.50 us
gated norm alone (2 WGs)                                 3.97 us
control: conv3 + step + norm (3 launches)                9.90 us
fused arm (1 launch)                                     8.33 us
control + f_b fold: conv3 + step(fold) + norm           14.05 us
fused arm + f_b fold (1 launch)                          9.37 us
exact: 13860864 words checked over 64 draws (fold alternating); fused != control:
       y 0, state 0, conv_state 0; non-finite y 0; status 0
```

* Exactness: every `y` word, every post-step `state` f32 and every post-conv window f32 of
  the fused arm equals the three-op chain's, with and without the L3 fold (32 draws each).
* Body: one launch at 8.3 us vs three at 9.9 (launches ≈ gates here). In the network the
  chain is conv3 body 6.8 + step 8.3 + norm 4.7 + three gates ≈ 24 us/layer on the path
  (plan §4 L7); the arm replaces it with one ≈ 8.3 us body + one gate ≈ −14 us/layer ×
  69 ≈ **−1.0 ms/token** (plan −0.9..−1.1). With L3 the fold costs the arm +1.0 us (9.4 vs
  8.3; each tile computes its 8 columns once, one load round, overlapped with its conv)
  where it costs the standalone step +3.8 (L3 status) — the arm keeps L3's packet saving
  and returns most of its body growth.
* gpulease flagged a foreign context on the leased card on all three runs (advisory lease;
  `foreign-before` 32 KB / 360 MB, `foreign-during` 1.9 GB); the three runs agree to
  0.05 us on every row (fused 8.36 / 8.33 / 8.34, fused+fold 9.37 / 9.35), so the numbers
  stand, with that caveat. Plan's vLLM reference: `fused_kda_decode` 8.4 us.

### L8 — `/tmp/l78/gemv_prefetch_bench 24 5`

256 persistent workgroups replay 24 `Gemv` packets per shape, each with its own cold weight
matrix (a 300 MB sweep between runs) and a simulated gate opening every `period` us; body =
`t_end − t_ready` per workgroup (mean / p50 over 23 packets × 256 WGs × 5 reps), span = the
packet's `max t_end − min t_ready`, makespan/pk = the chain's span per packet. `pf on` =
`gemv_prefetch_slice` before the poll (uncapped run; the byte cap only removes the 196 KB row):

| shape (per token) | KB/WG | period | body p50 off → on | pkt span off → on | makespan/pk off → on |
|---|---:|---:|---|---|---|
| N=7168 K=1536 o_proj (93) | 84 | 0 | 9.72 → 4.80 | 19.6 → 17.4 | 10.52 → 8.02 |
| | | 12 | 9.84 → 4.92 | 11.32 → 6.36 | 11.98 → 11.75 |
| N=7168 K=768 (92) | 42 | 0 | 7.72 → 4.56 | 15.4 → 11.9 | 8.42 → 6.70 |
| | | 12 | 7.80 → 4.64 | 9.02 → 5.37 | 11.87 → 11.71 |
| N=3584 K=7168 latent up (92) | 196 | 0 | 7.00 → 4.76 | 19.2 → 21.5 | 7.86 → 12.51 |
| | | 12 | 8.28 → 4.76 | 9.24 → 20.5 | 11.87 → 12.38 |
| N=1536 K=7168 (48) | 84 | 0 | 3.32 → 2.16 | 8.61 → 10.7 | 3.98 → 5.70 |
| | | 12 | 4.28 → 2.20 | 4.90 → 2.61 | 11.68 → 11.59 |
| N=1536 K=128 f_b (69, L3 off) | 1.5 | 0 | 2.44 → 2.08 | 5.02 → 4.40 | 2.89 → 3.01 |
| | | 12 | 2.40 → 2.12 | 3.00 → 2.63 | 11.61 → 11.59 |

* Exactness: prefetch-on == prefetch-off, every output word, every shape (loads only).
* Slices ≤ 84 KB/WG win: the o_proj body halves (9.7 → 4.8 us, ≈ the plan's "4.5 us fixed")
  and the packet span drops 5 us when the gate has slack; back to back (period 0) o_proj
  and K=768 still gain 2.5 / 1.7 us per packet, while the 6-row N=1536 K=7168 slice loses
  1.7 us back to back (its body is already 3.3 us) and wins 2.3 us with slack.
* The 196 KB latent-up slice LOSES at every period (32 WGs × 196 KB = 6.3 MB against a
  4 MB XCD L2; capping the issue at 96 KB per workgroup did not recover it: span 9.1 →
  16.3 at period 12). Hence `PLOW_GEMV_PF_MAX_BYTES` = 128 KB: that slice runs as before,
  everything in the census below it is prefetched. Re-run with the cap (`l8_bench3.log`):
  latent up on == off at every period (body 6.87 vs 6.89 us, span 17.3 vs 16.0 / 9.06 vs
  9.10), the ≤ 84 KB rows unchanged (o_proj p50 9.72 → 4.84); that run's means carry two
  contention outliers from a foreign process gpulease reported mid-run (advisory lease).
* In-network projection over the path (`decode-gap-plan` §1: 8.77 ms of gate/token, so most
  b=256 GEMVs claim into a closed gate): o_proj 93 × −5.0 + K=768 92 × −3.6 + N=1536 48 ×
  −2.3 ≈ −0.9 ms/token with slack, ≈ −0.4 ms if every gate were open at claim
  (period-0 makespan deltas). Plan L8: −1.0..−1.5.

Projected TPOT (served 25.25 ms, host 0.44 kept): L7 −1.0 and L8 −0.4..−0.9 →
**≈ 23.4–23.9 ms/token** from these two levers alone, additive with L1–L5 (separate
branches); the TP8 gate decides.

## Objects (served K3 decode GQ recipe from `/tmp/k3-stack3` build.make, this tree's source, served `plow_config.h`)

| object | bytes | VGPR | occ | vgpr spill | sgpr spill | private | LDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| flag off | 156488 | 248 | 2 | 0 | 110 | 216 B | 147504 |
| `-DPLOW_KDA_DECODE_FUSED_ARM=1` | 165440 | 248 | 2 | 0 | 110 | 216 B | 147504 |
| `-DPLOW_GEMV_PREFETCH=1` | 157560 | 248 | 2 | 0 | 112 | 216 B | 147504 |
| both + `-DPLOW_KDA_FB_FOLD=1` | 185504 | 248 | 2 | 0 | 112 | 284 B | 147504 |

All accepted by `hipcc_hsaco.sh`'s 256-register / occupancy-2 contract. VGPR, occupancy and
private segment unchanged by L7 and L8 (the 284 B row is L3's fold scratch, as in its status);
L7 is +8.9 KB of code, L8 +1.1 KB (+2 SGPR spills). The paired build against the flag-on emit's
own `plow_config.h` (no `-D`, the flags default from the packet) builds and is accepted too.

## Verified

`cargo build --release -p plowc`, `cargo build --release -p plowrt --features hsa`,
`cargo test -p devgen -p packet -p plowrt --features hsa` (devgen 305 passed incl. the three
new tests; packet, plowrt all green), `cargo fmt --all --check` clean. Flag-on emit
(`K3_FULL=1 PLOW_KDA_DECODE_FUSED_ARM=1 PLOW_GEMV_PREFETCH=1 PLOW_KDA_FB_FOLD=1 plowc …`):
Lean verified, `requires` carries `PLOW_KDA_DECODE_FUSED_ARM=1` and `PLOW_KDA_FB_FOLD=1`,
`plow_config.h` defaults all three flags.

## Pending

1. TP8 gate (main session owns the GPUs). Emit with the flags, build the paired objects, serve,
   3 alternating 8192->256 folds against the served bundle; checksum must be
   `fnv1a64:71a28c1449921c95`:

   ```
   docs/k3-mi355x-20260904/scripts/showdown_bundle.sh /tmp/k3-l7 PLOW_KDA_DECODE_FUSED_ARM=1
   docs/k3-mi355x-20260904/scripts/showdown_bundle.sh /tmp/k3-l8 PLOW_GEMV_PREFETCH=1
   docs/k3-mi355x-20260904/scripts/showdown_bundle.sh /tmp/k3-l78 PLOW_KDA_DECODE_FUSED_ARM=1 PLOW_GEMV_PREFETCH=1 PLOW_KDA_FB_FOLD=1
   ```
   (`showdown_bundle.sh` hard-codes `wt=` to the served worktree; point it at a checkout of this
   branch. No extra cmake `-D`: `plow_config.h` defaults the flags from the packet, decode
   objects only. `plowrt` refuses a fused packet on an object without
   `plow_kda_decode_fused_arm_1`.) Expect in the critpath: KDA family one packet per layer
   (KDA_STATE_STEP_G only), and under L8 the `Gemv` b=256 body/pk falling toward the L2-hit
   figure below on packets whose gate was closed at claim.
2. If a gate passes: default the flag on (`emit_config.rs`, CMake option ON), campaign row.
3. L7 headroom: the head's tile 0 carries the q/k conv AND (under L3) the 128-column fold on
   the critical path of its 15 siblings; distributing the fold across tiles is exact but needs a
   symmetric exchange, which is deadlock-free only without claim-ahead (the L8 variant here
   claims nothing ahead, so it would be safe today).
