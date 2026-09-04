# codex/prefill-kda-chain status (2026-09-04)

Branch `codex/prefill-kda-chain` from `origin/codex/amd-agent-harness` (986200d). Lever: the KDA
prefill Wu -> carry chain (v3 plan §7.0 carry follow-up "feed the key-factor hi/lo pair", TTFT lever
research "KDA key-factor/Wu chain"). Both steps bit-exact, default off, one flag each.

## What shipped

| step | mechanism | flag (emit) | objects |
| --- | --- | --- | --- |
| 2. lean Wu | `runtime/amd/op_kda_wu_lean.h`: one four-wave WG per (chunk, head); bf16(Ainv) and the transposed `beta k exp2(g)` / `beta v` operands staged once in LDS from coalesced loads; each wave forms its 16 output tiles with the shipping products in swapped A/B roles (a lane holds four consecutive channels of one row, 8-byte stores); q pre-scale fused on the same g loads | `PLOW_KDA_WU_LEAN=1` | `kda_chunk_wu_lean_gfx950.elf` |
| 1. key feed | the lean Wu also emits the carry's scaled-key hi/lo pair `k exp2(g_last - g)` (the `kda_chunk_key_factor` formula) into one reusable runtime scratch pair; `d_kda_chunk_carry_bt64_regstate<..., KEYFEED>` loads the pair rows one chunk ahead instead of rebuilding them from k and g (32 `exp2` + 64 RNE splits per lane per chunk, duplicated across the 8 V-tile WGs of a head) | `PLOW_KDA_CARRY_KEYFEED=1` (implies the lean Wu; needs `PLOW_KDA_CARRY_REGSTATE`, the default) | `kda_chunk_wu_lean_keys_gfx950.elf` + `kda_chunk_carry_regstate_keyfeed_gfx950.elf` |

The standalone key-factor Wu/carry pair (`PLOW_HSACO_KDA_KEY_FACTOR`) is not routed; the bench shows
why that screen lost: the interpreter Wu body at four waves is 1.114 vs 0.553 ms/layer at eight
(+0.56 ms x 69 = +39 ms of the observed +41 ms), independent of the carry.

Packet: `SE_KDA_WU_LEAN` (the shared opcode-disambiguated bit) on the pure Wu segment of an exact
qpre BT64/D128 pair; keyfeed sets the Wu's `i[5] = 1` (the interpreter ignores it). Manifest
`objects.lean.{kda_wu_lean,kda_carry_keyfeed}.required` -> `PLOW_PACKET_REQUIRES_KDA_WU_LEAN /
_KDA_CARRY_KEYFEED` -> CMake builds the packet-paired objects. Runtime (`plowrt` `exec/amd.rs`):
segment class 25, `promote_kda_wu_lean_routes` (after the regstate promotion) routes the Wu and,
for a key-emitting Wu, converts the following regstate carry route into `KdaChunkCarryKeyfeed`;
objects are gated on markers, pairing stamp, kernarg size, LDS and zero private, like regstate.
Ragged prefill chunks: the Wu object handles any T; the key-fed carry keeps regstate's
`t >= 512` rule (shorter tail chunk -> interpreter carry, exact).

## Per-layer numbers (one MI355X, `T8192 H12 D128 V128 BT64`, 21 order-rotated samples)

| kernel | before | after | speedup | bench |
| --- | ---: | ---: | ---: | --- |
| Wu (interpreter body WG512 grid 256 -> lean grid 768) | 0.553 ms | 0.056 ms | 9.86x | `runtime/bench/amd/kda_wu_lean` |
| Wu with key pair (lean keys, grid 1536) | 0.553 ms | 0.078 ms | 7.08x | same |
| carry (regstate -> regstate keyfeed) | 0.726 ms | 0.507 ms | 1.43x (3.81x vs the pre-regstate 1.931) | `runtime/bench/amd/kda_carry_regstate` |
| chain Wu + carry | 1.279 ms | 0.585 ms | 2.19x | |

Step 1 alone (carry consuming the pair, producer cost included): 0.726 -> 0.507 + 0.022 = -0.197
ms/layer. Step 2 alone: -0.497 ms/layer. Together -0.694 ms/layer -> about -48 ms TTFT over 69 KDA
layers if the network shape matches the bench (per-rank H=12), before launch/overlap effects; the
served Wu+carry attribution was 42.65 + 151.9 ms pre-regstate. Carry phase timers (wave-0 cycles
per chunk): `keys+loads` 7,025 -> 1,803, total 12,948 -> 7,677.

## Exactness evidence

- Wu: W, U, pre-scaled q bit-equal to the interpreter body on all 12,582,912 elements each, and the
  key pair bit-equal to the reference precompute, for structured, LCG-uniform and adversarial
  inputs (NaN/Inf/denormal/RNE-tie sprinkles, exp2 overflow/underflow gates, zero rows) at
  T=8192, T=8191 (63-row tail chunk) and T=100/H=3 (36-row tail); every grid arm.
- Carry keyfeed: out 0/12,582,912 and final f32 state 0/196,608 mismatches vs the shipping
  V16/WG512 carry, Aqk unchanged, same three input modes at T=8192 and T=8191.
- Flag-off objects: all 62 objects built from this tree against the stack-3 `plow_config.h` have
  `.text/.rodata/.data` identical to the base commit's build (`/tmp/plow-kda-chain/cmp_objs.sh`);
  the only whole-file difference is `__hip_cuid_*`, the source-path hash. The regstate carry
  object's non-KEYFEED instantiation stayed byte-identical only after keeping its prologue text
  verbatim in the `else` branch (a hoisted `c1` changed the schedule).
- Flag-on emit (`PLOW_KDA_CARRY_KEYFEED=1`, K3 TP8, 16384 ctx): Lean verified + oracle, tiles
  7650/7650 measured, `PLOW_PACKET_REQUIRES_KDA_CARRY_KEYFEED 1`; both objects build, stamped,
  markers present, resource gates pass.

## Object resources (gfx950)

| object | VGPR | SGPR | occupancy | LDS | private / spills |
| --- | ---: | ---: | ---: | ---: | --- |
| `kda_chunk_wu_lean_gfx950` | 168 | 44 | 3 | 46,080 B | 0 / 0 |
| `kda_chunk_wu_lean_keys_gfx950` | 230 | 48 | 2 | 46,080 B | 0 / 0 |
| `kda_chunk_carry_regstate_keyfeed_gfx950` | 242 | 48 | 2 | 43,520 B | 0 / 0 |
| `kda_chunk_carry_regstate_gfx950` (unchanged) | 229 | 52 | 2 | 43,520 B | 0 / 0 |

## Verification

`cargo build --release -p plowc`, `cargo build --release -p plowrt --features hsa`, `cargo fmt`,
`cargo test -p packet -p plowrt --features hsa` (all pass; new tests: packet
`kda_wu_lean_tests`, plowrt `kda_wu_lean_routes_the_marked_pair_and_feeds_the_regstate_carry`),
`cargo test -p devgen` (unit tests pass; the two `tuned_tile_selection` gfx942 integration tests
fail on TuneDB staleness of the gfx942 store, unrelated to this branch).

## TP8 gate (not run here; single-GPU rule)

Checkout of this branch in the bundle worktree (`showdown_bundle.sh` hardcodes `wt`), then:

```sh
docs/k3-mi355x-20260904/scripts/showdown_bundle.sh /tmp/k3-kda-chain PLOW_KDA_CARRY_KEYFEED=1
# expect: lean verified+oracle, tiles 7650/7650, objects 125 (keyfeed pair + regstate present),
#         emit.log: PLOW_PACKET_REQUIRES_KDA_CARRY_KEYFEED 1
cd /home/lava/plow && for tag in a b c; do for B in /tmp/k3-kda-chain /tmp/k3-stack3; do
  GPU_LEASE_TIMEOUT=14400 perf-data/tools/gpulease -n 8 "kda-chain-$tag" nix develop -c env RUST_LOG=info \
    /tmp/k3-kda-chain/bin/plowrt --rt-checkpoint /tmp/k3-farm.dvzmZN --rt-hsaco "$B/hsaco" --amd-tp-no-audit \
    bench --assets "$B/assets" --random-input-len 8192 --seed 20260904 \
    --concurrency 1 --requests 3 --warmup-requests 1 --output-len 256 > "/tmp/k3-xr-phase-gate/bench-kda-chain-$(basename $B)-$tag.log" 2>&1
  python3 docs/k3-mi355x-20260904/scripts/bench_fields.py "/tmp/k3-xr-phase-gate/bench-kda-chain-$(basename $B)-$tag.log"
done; done
```

Pass: checksum `fnv1a64:71a28c1449921c95` on every fold, TPOT neutral, TTFT below the served
bundle by roughly 40-50 ms; the runtime log must show `KDA Wu lean keys object accepted` and
`KDA carry keyfeed object accepted`. Roll back with `PLOW_KDA_CARRY_KEYFEED=0` (the default).
`PLOW_KDA_WU_LEAN=1` alone gates step 2 by itself (object `kda_chunk_wu_lean_gfx950`).

## Not done

- prepare/conv/gnorm on the interpreter (`KdaChunkPrepare`, `KdaConv3`, `KdaGatedNorm`) are
  untouched; the Wu prologue now reads q/k/g once per (chunk, head), so fusing prepare's L2
  normalization into it would need the gate prefix (a per-chunk scan over g_raw) first.
- The lean Wu's grid is fixed at min(items, 768); 512..1536 are within 5% on one GPU.
