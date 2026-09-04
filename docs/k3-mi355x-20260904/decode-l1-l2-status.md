# codex/decode-l1-l2 status (2026-09-04, deadline cut)

Branch `codex/decode-l1-l2` from `origin/codex/amd-agent-harness` (0b04dd2). Pushed.

## L1 — DONE (commit 8e18b50)

`runtime/amd/op_moe.h` `d_moe_router_topk`, `PLOW_MOE_ROUTER_SELECT=1` (the K3 default):
per-wave rank of the wave's own keys (uniform-address LDS broadcast reads) -> per-wave
top-k candidates -> wave 0 ranks the 8*16 candidates -> `wl[rank]`. 3 barriers (was 48).
Same packed key, same descending order, lowest-id tie-break, gate tail untouched.
Degenerate case (fewer than k nonzero keys, only reachable via group-mask of expert
n_exp-1) reproduced by pre-filling `wl` with n_exp-1. Old rounds arm kept as SELECT=2.

Evidence (gfx950, one GPU, `runtime/bench/amd/k3_router_topk_sweep`, 64 rotations of
smooth / uniform-random / 8-level-tied / spiky logits + bias, 21 samples):

    arm      median_us  p10     p90
    rounds   15.847     15.837  15.869   (control = SELECT=2, the served arm)
    merge    12.068     12.051  12.088   (SELECT=1, new)
    empty    2.584      (launch floor)   -> body 13.3 -> 9.5 us standalone
    byte_diff=0 over all 64 routing tables (ids, order, gates), oracle_id_diff=0

In-network the traced body is 24 us (interp object, occupancy-2 code), so the network
saving is expected to be larger than the standalone 3.8 us; the trace shows the GLU gate
would then move to the `xe` down-latent GEMV end (16 us/layer cap -> ~1.5 ms/token).

Resources (`/tmp/l1/obj/hsaco/*.resources.json` vs `/tmp/k3-stack3/hsaco`, same
plow_config.h): all objects identical vgpr/agpr/sgpr/occupancy/spills/private/LDS;
object_bytes +256 B on decode objects, +768..+1024 B on prefill MoE objects.

Rebuild the bench objects: `/tmp/l1/build_bench.sh` (needs `-DPLOW_LEAN_OBJECT=1` for the
hsaco ABI gate; the CMake recipe in `runtime/bench/CMakeLists.txt` was updated the same way).

## L2 — NOT WIRED (analysis only; default packet byte-identical)

Trace (`/tmp/decode-gap/trace-stack3.clean.raw`, `/tmp/decode-gap/layer_dump.py`,
`deps_dump.py`) contradicts part of the plan's L2 model:

1. At 1 WG/CU nearly every "small" projection is emitted at b=256 (f_b 256, q_rope 256,
   kv_a 256, g_proj 256; f_a 128, b_proj 12, k_rope 64). CU-time is conserved, so
   hoisting f_a/b_proj ahead of GemvQkvg saves only ~3 us/KDA layer (f_b still needs a
   full round) and q_absorb/q_rope vs kv_a/k_rope ordering is ~neutral.
2. `sh_down` (#37) is NOT queued behind up_proj by accident: it has a counter edge from
   the expert-combine XReduce (#34) — the slot-A WAR guard documented in
   `crates/devgen/src/k3.rs` ("ORDERED AFTER THE EXPERT-COMBINE COLLECTIVE"). Removing
   it needs a third `act.*_tp` peer slot (host change). Not an order-only lever.
3. THE REAL ORDER LEVER is the post-DOWN segment: `gq_asap_ranks` (devbuild.rs:630) ranks
   over the whole program, so `GemvGlu` (rank 3) sorts ahead of `MoeCombine` (rank 9)
   even though both are ready at the segment start (DOWN is a raw launch boundary).
   GemvGlu's 256 WGs delay Combine by ~11 us, then up_proj+sh_down (480 WGs) take two
   rounds: layer tail 369.4 -> 428.9 in the trace. With SEGMENT-RELATIVE ranks
   (a producer in an earlier segment contributes start 0) the window becomes
   [Combine, GemvGlu, XReduce, up_proj, sh_down, XReduce]: modelled tail ~412
   -> ~ -15 us/MoE layer x 92 = ~ -1.4 ms/token, order-only, exact by construction.
4. g_proj (#72) sits after the MLA specialist boundary (7.7 us) + 8.6 us body before
   MlaOutGate; emitting it before FlashMlaDecode nets ~ -5..-9 us/MLA layer x 24.
   Requires emit-order change in k3.rs AND ASAP not hoisting it to the window head
   (it is rank 0 there) — e.g. window sinks (no in-window consumer) ranked last.

Remaining steps for L2 (est. 0.5 day + emit/Lean + TP8 gate):
- devbuild.rs: compute `seg_of` before the ASAP ranks (it is built in `finish` at
  ~line 2163) and pass it to `gq_asap_ranks`; treat cross-segment producers as start 0.
  Add a unit test next to `gq_asap_order_hoists_ready_packets_ahead_of_gated_ones`.
- k3.rs: emit `f_a`/`b_proj` before `GemvQkvg` (same rank, stable sort keeps emit
  order); emit `g_proj` before the MLA specialist ops.
- Verify: `cargo test -p devgen -p packet`, then emit with PLOW_VERIFY_BIN
  (`showdown_bundle.sh`) and check `.lean.verified`; re-trace and re-run
  `/tmp/decode-gap/critpath_layers.py` for the post-DOWN window.
