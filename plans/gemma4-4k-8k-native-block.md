# Gemma-4-12B native 4K/8K block campaign

## Goal

Cut cold single-request prefill TTFT by 50% at the fixed M4096 and M8192 rungs on H100 using only Plow-native execution. Optimize isolated transformer blocks first; run the full model only after the block gate passes.

## 2026-09-12 checkpoint

The isolated-block timing gate passes with native segmented packets and native
lean objects. These numbers use one H100 lease, candidate first and control
second, with two warmups and ten measured prefill passes.

| Rung | Block | Control | Candidate | Reduction | Output check |
|---|---|---:|---:|---:|---|
| 4096 | sliding L0 | 13.174 ms | 3.802 ms | 71.1% | bit-identical |
| 4096 | global L5 | 15.991 ms | 4.564 ms | 71.5% | rel-L2 0.003002 |
| 8192 | sliding L0 | 26.511 ms | 7.487 ms | 71.8% | bit-identical |
| 8192 | global L5 | 44.541 ms | 10.923 ms | 75.5% | rel-L2 0.002877 |

Weighted by Gemma's 40 sliding and 8 global layers, the 4K score falls from
654.888 to 188.592 ms (71.2%) and the 8K score falls from 1416.768 to
386.864 ms (72.7%). No cuBLASLt role is loaded. The candidate keeps one whole
transformer block in the harness while using native segmented GEMM and
attention objects with a ten-node CUDA graph.

A production-path rerun with `PLOW_PF_SEG_DIR`, `PLOW_PF_SEG_PURE`,
`PLOW_PF_SEG_FA512`, and `PLOW_PF_SEG_GRAPH` all unset found and fixed two
selection defects. Partial attention roles had suppressed the bundled GEMM
pair, and the HD256-capable FA object was not self-selected. After the runtime
fix, the no-override medians are 3.792/7.474 ms for sliding and 4.558/10.974 ms
for global at 4K/8K. CUDA graph submission is therefore not required for the
fixed-rung win.

The isolated control was the universal interpreter, while the production
packet already carried segmented native objects. A matched full-model C1 run
therefore gives the production baseline: 202.34 ms at 4K and 425.05 ms at 8K.
Replacing only the HD512 role with the score-partitioned candidate gives
199.30 ms and 411.48 ms, a 1.5% / 3.2% reduction. Its greedy output checksum
changes at both rungs, so that role is rejected and must not be promoted. The
50% production target remains <=101.17 ms at 4K and <=212.52 ms at 8K.

The next experiment keeps the known-correct HD512 role and compiles the GEMM,
HD256 attention, and light segment objects against the exact packet-generated
`plow_config.h`. This removes unused opcode arms, stamps packet pairing, and
tests whether the broad light object's 2,360-byte stack frame is responsible
for part of the remaining packet overhead. Promotion requires zero stack/local
memory and identical full-model greedy output.

The packet-specific build exposed the light-object issue directly. Packet
inventory pruning alone left tiled GEMM arms in `PLOW_NV_FATLITE`, so ptxas
still allocated their WGMMA state: 128 registers, 2,360-byte stack. Compiling
those unreachable GEMM/GLU/MoE projection arms out reduces the BF16 packed
light object to 128 registers and a 24-byte stack; the analogous FP8 object is
16 bytes. In a same-lease full-model C1 A/B with the known-correct HD512 role,
the BF16 medians move 202.37 -> 199.41 ms at 4K (-1.46%) and 426.29 -> 422.42
ms at 8K (-0.91%). Prompt and output checksums are identical. This is promoted
as a safe native rung improvement, while the remaining small spill and the FA
object's 272-byte spill stay open.

The next attention split is also qualified. The packed FA object was executing
only HD256 segments because every HD512 segment already had the authenticated
standalone role, but it still compiled the unused HD512 body. An HD256-only
build drops the object from 255 registers / 272-byte stack to 240 registers /
zero stack. Against the promoted light object, full-model medians move 198.97
-> 196.81 ms at 4K (-1.08%) and 420.66 -> 416.22 ms at 8K (-1.05%), with
identical output checksums. The loader now accepts this object only when every
HD512 FlashPrefill segment names the packet's HD512 role. The cumulative
production reduction from the original segmented baseline is about 2.7% at
4K and 2.4% at 8K; the 50% target remains open.

The coalesced shared-memory GEMM epilogue is not a production win. In the
standalone exact-shape harness it raises most M4096/M8192 Gemma projection
cells to 0.91--1.05x cuBLASLt, with 160 registers and no spills. In the
packet-paired full model it changes 4K from 196.62 to 197.43 ms (+0.41%) and
8K from 416.26 to 415.79 ms (-0.11%); output checksums are identical. Do not
promote it. The remaining production headroom is attention and packet-level
activation/transition work, not the isolated GEMM epilogue.

The in-body HD512 score-producer/PV-consumer split is also rejected. A
one-producer/four-consumer, 640-thread object compiles at 220 registers with
no spills but cannot launch on H100 (`too many resources requested`). Reducing
it to one producer/two consumers at 384 threads compiles at 255 registers; an
entry cap of 168 makes it launchable only by introducing a 472-byte stack,
2,212-byte spill stores, and 3,512-byte spill loads per thread. The compiler
also serializes WGMMA around the divergent producer/consumer path. The next
HD512 design must separate the score and PV register regions structurally
(separate lean phases or a non-divergent pipeline) while keeping BKV32 tile
order for greedy agreement. Do not add `setmaxnreg` to the current body.

The exact HD256 BKV64 role is also rejected for production. Its 21-cell H100
SXM5 packet harness campaign passed single, packed-homogeneous, and
packed-ragged correctness and measured about 1.8x over the BKV32 control, but
the full Gemma-4 gate did not transfer: 4K was 195.87 vs 195.75 ms and 8K was
413.88 vs 415.42 ms. Both output checksums changed. No qualified TuneDB record
is published. The next HD256 candidate must preserve the BKV32 accumulation
order and be measured through the production 435-segment route before any
full-model promotion.

The exact HD256 BKV32 lean role passes that gate. Its 21-cell H100 SXM5 packet
campaign covers M4096/M8192, single, packed homogeneous, packed ragged, and
live-KV buckets 1K/4K/8K/16K. All cells pass with a median 1.252x standalone
speedup, 1.239x minimum, 1.258x maximum, and worst rel-L2 0.002730. The object
uses 157 registers, 103,424 bytes of shared memory, and zero stack, spills, or
local memory. A matched production packet changes only the 80 sliding HD256
roles: the first paired A/B improves 208.60 -> 204.36 ms at 4K and 439.55 ->
430.57 ms at 8K, both 2.04%. Across five prompt seeds, mean p50 reduction is
3.00% at 4K (2.27--4.05%) and 2.32% at 8K (1.95--2.57%). Every corresponding
prompt and output checksum is identical. Promote this role; the next structural
target remains phase fusion and HD512 because the packet still makes 435
segment launches and the BKV32 role alone cannot approach the 50% TTFT target.

The exact BF16 gate/up+GeGLU role now composes with that HD256 role at M4096
and M8192. It uses one TMA producer warpgroup and two WGMMA consumer
warpgroups, BM128/BN128/BK64, four stages, 160 registers, 197,696 bytes of
dynamic shared memory, and zero stack or spills. The compiler emits 48 role-12
segments at each target rung while retaining all 80 qualified role-11 segments
and all 112 existing role-6 segments. The fused opcode changes the authenticated
program digest, so the 21 HD256 cells were rerun under one H100 lease and
republished for the new exact packet; every cell passed. A matched three-rep
full-model A/B is checksum-identical and moves median TTFT 203.43 -> 195.56 ms
at 4K (-3.87%) and 427.02 -> 416.19 ms at 8K (-2.54%). The five-seed gate then
passes with every corresponding prompt/output checksum identical: mean p50
reduction is 3.31% at 4K (2.81--3.68%) and 3.27% at 8K (2.62--3.85%). The
TuneDB compile must use a CUDA toolchain
identity: the development shell currently exports a ROCm label even for SM90,
which correctly prevents those records from selecting unless overridden.

The existing `PLOW_PF_GFUSE` residual-plus-next-norm opcode is not the required
phase fusion on H100. It replaces 96 instructions but leaves the same 435
segment launches. A matched reverse-order production A/B regresses 4K from
208.27 to 220.20 ms (+5.73%) and 8K from 439.68 to 455.02 ms (+3.49%), and both
greedy output checksums change. Keep it off. The useful next candidates are a
lean norm object preserving the original WPR reduction order, or an output/down
projection epilogue that absorbs the first norm/residual work without adding a
second full-row pass inside one broad interpreter body.

The AMD host-path audit found no unmerged scheduler or mux work to cherry-pick:
`origin/worktree-tp-merge` at d6762aec is already an ancestor. CUDA already
uses the backend-neutral step planner, packed cross-request prefill, the shared
token-batch request/delivery contract, one timestamp per planning pass, and
demand-backed slab VMM. AMD's next-chunk mapping, deferred prefix publication,
deferred recycle, NUMA kernarg pools, and BAR staging are tied to ROCr queue and
mapping behavior. Keep them backend-specific until a CUDA driver A/B shows an
engine-thread stall they can remove.

Trace attribution on the control shows why both families are required. GEMM /
attention / remaining work is 53.7% / 36.9% / 9.4% for sliding 4K, 52.7% /
40.6% / 6.7% for sliding 8K, 44.9% / 46.9% / 8.2% for global 4K, and 33.2% /
63.0% / 3.8% for global 8K.

Next gates: packet-specialized cubin resource audit, full-model greedy
agreement, five-seed stability, sampled operator checks, then cold C1
full-model TTFT at exact 4K and 8K.

## Fixed comparison

- Branch/commit: `tp-bringup-mi300x` / `9fc89e68`.
- Model/dtype/device: Gemma-4-12B BF16, H100 SM90a, TP1.
- Rungs: exact M4096 and M8192. No chunk substitution or padding to another rung.
- Blocks: layer 0 sliding attention (HD256, GQA2, window1024) and layer 5 global attention (HD512, GQA16).
- Score: `40 * sliding_block_ms + 8 * global_block_ms` for each rung.
- Control: the commit's unsegmented universal native prefill interpreter. No cuBLASLt roles.
- Candidate: Plow-native segmented cubins and packet roles only. The cuBLASLt measurements remain a ceiling, not an execution dependency.
- Timing: one loaded engine, two warmups, at least 10 event-equivalent synchronized prefill passes, median and p95, exclusive `gpulease`.

## Acceptance gates

1. Weighted block median <= 50% of control at both M4096 and M8192.
2. Neither block kind may regress by more than 2% at either rung.
3. Candidate output passes full block reference checks at both rungs: finite, deterministic, sampled FP64 operator checks, block-output rel-L2 <= 1e-2, and identical greedy decision after reinsertion into the model.
4. Cubins have zero stack, spills and local memory. Resource metadata, object SHA, program digest and exact `(arch,dtype,M,N,K,attention geometry,KV bucket)` enter TuneDB.
5. Full-model cold C1 TTFT at 4K and 8K improves by >=50% with identical prompt/output counts. This final gate is run only after gates 1-4 pass.

## Baseline and attribution

1. Compile two block assets with `plowc --block 0` and `--block 5`, BF16, max context/chunk8192, native projection roles, and the 4096/8192 rungs.
2. Run `block_run bench --batch 1 --ctx 4096,8192` on the production native cubins.
3. Build trace-only twins with `PLOW_NV_TRACE=1`. Attribute synchronized block time to:
   - input norm + QKV projection + head norm/RoPE;
   - sliding/global attention;
   - output projection + residual;
   - FFN norm + gate/up GLU projection;
   - down projection + residual.
4. Record launch count, device time, tensor bytes, achieved TFLOP/s or bandwidth, registers, shared memory, occupancy, TMA transactions and barrier stalls. The untraced cubin supplies the promotion timing.

## Native optimization loop

### Projection segments

- Compile separate exact objects for M4096 and M8192 rather than sharing the general interpreter resource budget.
- Tune each of the eight Gemma projection shapes independently.
- Start from the proven TMA producer + two WGMMA consumer design: 128-byte swizzle, register donation, persistent tile cursor, prefetched tensor maps and 4/5/6-stage rings.
- Sweep BM/BN in `{64,128} x {128,256}`, consumer groups in `{2,3}`, cluster shape/multicast where both operands reuse, and Stream-K only when output-tile count underfills 132 SMs.
- Preserve fused QKV, fused gate/up GLU and eligible residual/norm epilogues when they remove a global-memory round trip. Do not split a fused segment unless the complete replacement is faster.
- Promote only a native cell that beats both the current native cell and the cuBLASLt ceiling by at least 5% across five seeds.

### Attention segments

- Sliding HD256: preserve the BKV32 accumulation order, specialize for query M4096/M8192 and effective KV1024, and avoid work for rows outside each query's causal window. The BKV64 role is a rejected throughput reference only.
- Global HD512: use the N64 score-partitioned two-consumer body with cross-warpgroup online-softmax merge. Tune BKV64, query tiles64/128, and sequence splits separately at 4K and 8K.
- Keep attention in its own lean object. Its register/shared-memory budget is incompatible with the projection object and combining them would lower occupancy.
- Gate packed and single-request metadata separately even though this campaign promotes only the single-request profiles.

### Block scheduling

- Keep one packet for the block, with direct native role transitions at projection and attention boundaries.
- Coalesce adjacent operations only when they use the same cubin and resource profile.
- Remove redundant interpreter queue scans, counter resets and global-memory round trips between fused producer/consumer pairs.
- Compare three packet layouts: current segmented block, three lean phases (`qkv+attention`, `o+residual`, `ffn`), and exact-role-per-hot-op. Select by complete block time.

## Execution order

1. Establish current native block baselines for all four `(layer kind, rung)` cells.
2. Capture trace attribution and compute the maximum possible gain from each operator family.
3. Optimize the top projection shapes until the weighted projection subtotal is >=2x faster.
4. Optimize HD256 and HD512 attention until the weighted attention subtotal is >=2x faster.
5. Reduce light-op and packet transition time enough for the whole weighted block to pass 2x.
6. Run block numerics, five-seed timing, and TuneDB authentication.
7. Compile a full-model native packet selecting only qualified cells; run cold 4K/8K C1 TTFT and correctness.

If the measured non-GEMM/non-attention floor exceeds 50% of current block time, the 50% TTFT target is impossible through kernel replacement alone. In that case the next required work is eliminating block-level activation round trips and packet transitions; the target is not weakened.

The prefill `PLOW_PF_GFUSE` follow-up with `PLOW_NV_NRN_WPR=1` is also rejected.
Its light object improves stack from 24 to 16 bytes but still uses 128 registers
and spills 12-byte stores/16-byte loads. A reverse-order production A/B remains
4.80% slower at 4K and 2.68% slower at 8K; greedy checksums still differ. The
candidate also lacks the compact terminal needed by CUDA unified token batching.
Do not promote either fused norm body. Preserve the split BF16 boundary until a
lean phase passes full-logit equality and runs inside the token-batch contract.

Reusing the qualified M512 BN128/six-stage producer-consumer body at the exact
M4096/M8192 projection cells is rejected as a general wide-rung policy. All 32
standalone body/segment/queue checks pass for each of the two configurations.
At M4096 it wins only N512/K3840 (1.431x), loses the other seven cells, and has
a 0.898 geomean. At M8192 it loses all eight cells with a 0.850 geomean. Keep
the current BN256 wide object. The isolated M4096/N512 result is only a search
lead until it beats cuBLASLt, passes five seeds, and has a packet-authenticated
exact role. Raw screen SHA256:
`7462361af7be7e0c6c600abe641fc63492a030e6716cf35bb6b3c50cb8214284`.

The M4096/N512/K3840 BN128/six-stage/band-8 cell beats tuned cuBLASLt in all
five standalone seeds: 1.058x minimum, 1.086x median, 1.218x maximum, with
zero output difference. It does not pass the production packet gate. The
existing global Q/KV segment mixes N8192 Q with the exact N512 projection, so
an ABI-bound object correctly selects zero work until plowc isolates the exact
projection. An opt-in packet split then selects exactly eight segments, one per
global layer, and leaves the 8K rung unselected. This increases the 4K launch
count from 435 to 443. A same-lease control/candidate/candidate/control A/B is
checksum-identical but changes mean TTFT from 211.13 to 211.66 ms (+0.25%);
p50 averages 211.28 vs 211.44 ms (+0.07%). Do not promote the object or packet
split. The next viable design is one lean 384-thread object that carries both
the existing BN256 Q body and the BN128 exact KV body within the original
segment launch; separate cubins cannot recover the extra launch cost at this
cell. Candidate packet SHA256
`181303948569dfcc25d60e1e1013d34a5b9560310c979efa7a4807569291e720`;
exact object SHA256
`83ceb8604c546e0540d8ea379029444728fe63f72515d932ba63ad9cbd8f98a4`.

The single-object hybrid BN256/S3 + exact BN128/S6 projection experiment also
fails the promotion gate. It preserves the original 435 launches and produces
identical 4K/8K checksums, but combining both bodies makes ptxas spill 44-byte
stores / 52-byte loads per thread. A same-lease control/Band8/Band16/control
run gives Band16 only about 0.6% lower 4K mean TTFT and 1.1% lower 8K mean TTFT;
Band8 regresses. The 8K movement, where the exact cell never selects, also
shows run-order noise dominates the claimed benefit. Do not promote the hybrid
object. HD512 attention remains the next structural target.

- HD512/BKV32 score-partition decision (2026-09-12): reject both two-warpgroup N-partition variants. The unordered variant compiled at 197 harness / 226 production registers with zero spills and passed sampled 4K/8K plus packed homogeneous/ragged correctness, but only improved isolated attention 1.7%/1.6%. The exact-order variant preserved the control BKV32 pair-sum order using 204,288 B dynamic arena, compiled at 180 harness registers with zero spills, and passed five-seed sampled correctness; same-lease p50 improved only 0.65% at both 4K and 8K. This is below the production-role threshold and cannot materially move the 50% TTFT goal. No source promotion. Raw evidence remains under /tmp/plow-hd512-bkv32-{nsplit,exact} and /tmp/hd512-bkv32-review.*.

- Paired-GQA2 exact-rung decision (2026-09-12): reject the existing shared-K/V
  GQA2 object as a replacement for role 11. A packet-bound object with the
  production packed/masked ABI and the fused role-12 GLU loaded successfully;
  unified token batching fired and every prompt/output checksum matched. Under
  one H100 lease, two rotated three-rep A/B seeds regressed TTFT by 6.15--6.39%
  at M4096 and 9.35--9.58% at M8192 versus the exact HD256/BKV32 role. Keep role
  11. Evidence is `/tmp/gemma4-gqa2-{control,candidate}-seed{17,19}.{json,log}`.
  The next attention experiment is the 384-thread HD512/BKV32 design with one
  TMA/score producer warpgroup and two PV consumer warpgroups; preserve the
  current BKV32 softmax order and screen it in the exact harness before adding
  a packet role.

- W8A8 direct-rung baseline (2026-09-12): the old 20K/C16 packet cannot load an
  exact M8192 activation arena on 80 GiB (planner requires 95.7 GiB). Re-emitting
  the same native W8A8 packet for max-context 16384 and decode rung C1 reduces
  declared KV from 25.0 to 5.25 GiB and loads with the existing general SM90a
  objects. Direct no-chunk three-rep p50 TTFT is 224.55 ms at M4096 and 546.32 ms
  at M8192. This is 14.5% and 31.3% slower than the promoted BF16 packet's
  five-seed means (~196.0/~415.8 ms), despite 12.0 vs 22.2 GiB of weights.
  W8A8 therefore needs structural quant/light and attention fusion before it is
  competitive; weight-only bandwidth savings do not offset its 192 quantization
  instructions and current attention path. Evidence:
  `/tmp/gemma4-w8a8-exact-c1-4k8k-v2{.log,.json,-serve.log}`.
  Direct segment timing attributes M4096 as 92.9 ms GEMM / 39.8 ms light /
  91.9 ms attention, and M8192 as 183.5 / 71.3 / 289.8 ms, over 193 / 242 /
  48 launches. Attention is already 41% of 4K and 53% of 8K, so FP8
  quantization fusion must compose with the BF16 attention work; it cannot make
  the 8K target by itself. Detailed site timings are in
  `/tmp/gemma4-w8a8-exact-profile.log`.

- W8A8 HD512 role screen (2026-09-12): pre-seeding the existing role-6 object
  into the exact W8A8 compile emits eight native HD512 segments per rung. It
  cuts p50 TTFT from 224.55 to 176.82 ms at M4096 (-21.3%) and 546.32 to
  371.64 ms at M8192 (-32.0%), enough to beat the current BF16 packet on this
  one seed. Do not promote it: output checksums differ at both rungs. The role
  changes the old general W8A8 attention path's accumulation geometry. This is
  strong evidence that FP8's first material kernel should be an exact lean
  HD512 object that preserves the accepted BKV16 reduction order, followed by
  full-logit and real-prompt greedy qualification. Evidence is
  `/tmp/gemma4-w8a8-exact-c1-hd512-v3{.json,-serve.log,-audit.json}`.
### 2026-09-13: packed exact-order HD512/BKV16 role

- Added the packed request ABI to the existing exact-order `BQ32/BKV16` HD512 role and routed packed requests through the production request mux.
- Exact packed W8A8 A/B on one leased H100, seed 29, 1 warmup + 3 reps:
  - 4K: 223.658 -> 219.098 ms mean TTFT (`-2.04%`).
  - 8K: 546.589 -> 530.501 ms mean TTFT (`-2.94%`).
  - Prompt and output checksums matched at both rungs.
- Evidence: `/tmp/w8a8-bkv16-packed-{control,candidate}-seed29.{json,log}`.
- Current packed mux compiles at 252 registers/thread. Keep the correctness win, then replace the broad mux with a narrow serial exact-shape wrapper to recover the unpacked object's 114-register budget.

### 2026-09-13: isolated W8A8 GLU-to-quant fold

- Split the existing T11 GLU-to-`QuantFp8` fold from `PLOW_QNORM_FUSE` with
  the default-off `PLOW_GLU_QUANT_FUSE` compiler knob. The old flag and AMD
  defaults retain their existing behavior; the new knob leaves both hidden-width
  RMSNorm quantization sites separate.
- Packet audit at M4096/M8192: 958 -> 910 instructions, exactly 48 `Glu`
  instructions removed, 48 `QuantFp8` instructions carry gate/up, and zero
  `RmsNorm` instructions carry quant outputs. The manifest requires
  `PLOW_T11_GLUQUANT=1`.
- Two same-lease H100 screens, three measured reps per rung, rotated order:
  - seed 31: 224.620 -> 221.441 ms at 4K (-1.42%); 547.451 -> 542.151 ms at
    8K (-0.97%).
  - seed 37: 224.909 -> 221.870 ms at 4K (-1.35%); 546.757 -> 543.305 ms at
    8K (-0.63%).
- Prompt and output checksums match at both rungs and both seeds. Promote as a
  safe opt-in, not as a default: the gain is consistent but small and cannot
  address the attention-dominated 8K gap. Evidence is
  `/tmp/w8a8-gluquant-{control,candidate}-seed{31,37}.{json,log}`.

### 2026-09-13: BF16 sliding QKV + HeadNorm/RoPE role review

The `f19d383a` harness is a valid kernel proof for Gemma-4 sliding layers. It
keeps the production WS384 `M128/N256/K64` WGMMA accumulation, explicitly
materializes each accumulator through BF16 in shared memory, then runs the
production HD256 pack-of-four norm/RoPE math. Its KV address is exactly the
packed formula `((pfslot[row] * nhead + head) * stride +
(pos[row] & mask)) * 256`; Q remains row-major. A clean Nix-shell build uses
the repository's required `env -i` nvcc isolation. On one leased H100, seed 23
is bit-exact at both rungs and repeats the original result:

- M4096: 0.551 -> 0.454 ms for the three sliding Q/K/V pairs, 1.213x.
- M8192: 1.072 -> 0.870 ms, 1.232x.
- 160 registers, zero stack/spills, 214,080 B dynamic shared memory, one CTA/SM.

Do not add a packet role yet. The existing packet has three `Gemm` instructions
in one segment followed by three `HeadNormRope` instructions in the next. A
role that only routes both segments still launches the split math and gets no
fusion benefit. A fused role must remove the HeadNorm/RoPE queue entries and
transfer their FlashPrefill dependency/successor obligations to the projection
tiles. The projection and HeadNorm/RoPE slice maps are different, so copying
successor lists by slice can under-synchronize attention. Overloading ordinary
`Gemm` fields would also make the packet's ISA meaning depend on the presence
of the role object. Neither change is safe as a post-build metadata rewrite.

The minimal safe implementation is an explicit paired operation at Builder
time, before `Builder::finish` constructs counters:

1. Add one SM90-only opt-in for BF16 Gemma-4 TP1 + TMA. Match only sliding
   HD256 layers, rows 4096/8192, K3840, Q N4096 and K/V N2048, no projection
   bias, and unique raw-projection consumers.
2. Emit one fused instruction for each Q/K/V projection with final output,
   input, weight, gamma/cos/sin/position, TMA maps, skip-norm, KV stride/mask,
   and packed slot-map operand. FlashPrefill depends directly on these three
   instructions. Omit the raw Q/K/V activation tensors from this opt-in packet.
   Use a dedicated opcode/slot schema so packet audit and CPU golden reject it
   explicitly instead of treating a private field overload as ordinary Gemm.
3. Add role 13 with a hash-bound `gemm_hnr_sm90_gemma4_slide_4k8k_v1` ABI,
   block 384, arena 214080, packed request ABI2 and masked-padding ABI1. The
   native object must trap on every shape/field mismatch.
4. Teach packed binding to patch the fused instruction's slot operand. Validate
   that every KV-writing fused site starts unbound, while Q has no slot/stride;
   keep the current per-row negative-slot padding guard.
5. Gate in this order: emitted segment/role counts (120 fused sites per rung),
   packed homogeneous and ragged slot correctness, full logits, five real
   prompt seeds, then same-lease full-model A/B. Promote only if checksums stay
   identical and the full packet wins beyond run-order noise.

The realistic full-model ceiling from the current trace is about 4.70 ms at 4K
and 9.70 ms at 8K, roughly 2.4%/2.3%. This is useful only after the role is
implemented without an extra segment launch; it cannot materially close the
50% TTFT target by itself.

### 2026-09-13: narrow packed HD512/BKV16 wrapper

- Replaced the broad packed attention mux in the exact BQ32/BKV16 HD512 role
  with a serial adapter that calls the same `d_flash_prefill_px4<512,32,16>`
  body selected by the unpacked role. A noinline device-call boundary keeps
  packed request metadata out of the px4 body's live range.
- ptxas resource use falls from 252 to 164 registers/thread with zero stack and
  zero spills. All fourteen prefill rungs retain eight role-6 segments.
- Two same-lease control A/B seeds remain prompt/output checksum-identical:
  M4096 improves 2.58% and 1.87%; M8192 improves 3.21% and 3.02%. A direct
  broad-vs-narrow run improves another 0.20%/0.17%; the large resource recovery
  only moves time slightly because the 99,376-byte arena already limits
  residency.
- Final evidence: `/tmp/gemma4-w8a8-hd512-bkv16-packed-narrow-v1/build-final.log`,
  `/tmp/gemma4-w8a8-exact-c1-hd512-bkv16-packed-narrow-v2-audit.json`,
  `/tmp/w8a8-bkv16-packed-narrow-v2-{control,candidate}-seed37.json`, and
  `/tmp/w8a8-bkv16-packed-{broad,narrow}-seed41.json`.

### 2026-09-13: HD512 global-scratch two-phase rejection

- An experiment-only split put QK plus online softmax in a 128-thread kernel
  and PV in a separate 256-thread kernel. The second kernel replayed KV tiles
  in order and applied each recorded online-softmax correction before the
  corresponding PV WGMMA. It compiled with 40/48 registers for score at
  BKV16/BKV32 and 154 registers for PV, with zero local memory or spills.
- Against QK-unroll-4 BQ64 WGMMA on one leased H100, it loses at every exact
  rung: BKV16 is 2.580 vs 2.170 ms at M4096 and 9.477 vs 7.870 ms at M8192;
  BKV32 is 1.902 vs 1.398 ms and 6.667 vs 4.998 ms. The split is bit-identical
  to its corresponding BQ64 WGMMA control, but not to the production
  BQ32/BKV16 px4 body (197,099 / 604,926 differing BF16 elements at 4K/8K).
- Compact triangular scratch still writes and rereads 613 MB / 2.435 GB at
  BKV16 and 579 MB / 2.300 GB at BKV32 for 4K/8K. Even the ideal 3.35 TB/s HBM
  floor is about 0.18/0.73 ms and 0.17/0.69 ms before the second CTA wave and
  launch. The measured penalty is larger. Reject the two-launch split; keep
  the non-divergent fused BQ64/BKV16 unroll search and qualify its changed
  px4 numerics at full-logit/greedy level. Raw results:
  `/tmp/hd512-twophase/{smoke,u4}.jsonl`.

### 2026-09-13: HD512 producer/consumer barrier diagnosis

- The opt-in 384-thread BQ64/BKV32 prototype deadlocked because `kv_full`,
  `p_full`, `empty`, and `inv_full` were function-local `__shared__` objects in
  `d_flash_prefill_sm90_pc512<PROD>`. NVCC instantiated disjoint storage for the
  producer and consumer specializations: ptxas reported 96 static bytes, equal
  to the producer's 56-byte set plus the consumer's live 40-byte subset. The
  consumers therefore waited on `p_full` barriers the producer never reached.
- A bounded experiment moved all barriers to the common kernel wrapper and
  passed them into both specializations. It completed and passed the sampled
  FP64 oracle at exact M4096/KV4096 and M8192/KV8192; candidate and control had
  identical worst-rel-L2 and max-absolute error for seed 29.
- Reject the design and keep production off. Same-lease p50 was 7505.38 vs
  1923.01 us at 4K (3.90x slower) and 29189.73 vs 6893.09 us at 8K (4.23x
  slower). CUDA 13 ptxas emitted C7512 WGMMA serialization, 64 bytes local
  state, 128-byte spill stores, and 64-byte spill loads. The score-producing
  warpgroup and PV-consuming warpgroups both issue WGMMA on divergent paths, so
  this arrangement does not provide the intended cross-tile overlap. The
  rejected source delta was reverted; evidence remains in
  `/tmp/hd512-pc-fixed-03ce25f9`.

### 2026-09-13: W8A8 HD512 BQ64/BKV16 WGMMA screen

- Exposed the latent Hopper `d_flash_prefill_sm90<512,64,16>` body as an
  experimental role geometry. The object is opt-in, uses the existing BKV16 KV
  tile order, `nsplit=1`, and QK unroll 4. Its packed wrapper compiles at 218
  registers with zero stack/spills and claims 134,144 bytes of shared memory.
  BKV16 stays on cp.async because the packet's KV tensor-map box is 32 rows.
- Direct control comparison against the accepted `BQ32/BKV16` px4 body shows
  roughly 1.50--1.63x for the original U1 build and 2.25--2.32x for U4. It is
  not operator-bit-exact: five U4 seeds have rel-L2 at most 8.18e-5 and max
  absolute BF16 difference 0.00390625. `nsplit=2/4/8` all lose decisively to
  `nsplit=1` once merge time is included.
- Packed full-model U4 single-chain seed 43 improves M4096 TTFT
  219.313 -> 183.025 ms (-16.55%) and M8192 528.978 -> 392.201 ms
  (-25.86%). The 8K greedy/output checksum matches; the 4K checksum differs.
- Splitting WGMMA QK into the same two HD256 accumulator chains as px4 reduces
  direct mismatches to 12,952/31,779 at 4K/8K and rel-L2 to
  4.72e-5/5.13e-5, but still is not exact. It costs about 3--4% in isolated
  attention and compiles at 226 packed-role registers with zero spills.
- The two-chain packed gate is also negative. Across seeds 17, 19, and 43 it
  wins 16.91% mean at 4K and 25.91% at 8K, but greedy/output agreement is 0/3
  at 4K and 2/3 at 8K. Do not promote either BQ64/BKV16 body until a full-logit
  and real-prompt numerical policy is established; exact px4 stays selected.
  Evidence is `/tmp/gemma4_hd512_wgmma_bkv16.qkunroll-five.jsonl`,
  `/tmp/w8a8-hd512-wg16-u4-{candidate,control}-seed43.{json,log}`, and
  `/tmp/w8a8-hd512-wg16-u4-halves-{candidate,control}-seed{17,19,43}.{json,log}`.

### 2026-09-13: exact HD512 px4 cp.async promotion

- The accepted BQ32/BKV16 `d_flash_prefill_px4<512,32,16>` role was still
  staging each 16-row KV tile through the TMA row-bulk path. Switching only
  that staging path to cp.async preserves the exact traversal, accumulator
  chain, online-softmax order, and BF16 stores.
- Five direct seeds are full-output bit-identical at both exact rungs. The
  production-like role improves 4661.1 -> 3833.3 us at M4096 and
  17182.7 -> 14102.4 us at M8192 (1.216x/1.218x). The native 70,672-byte arena
  gives the same result. The packed object uses 168 registers, 1024 bytes
  static shared memory, and zero stack/spills/local memory; the TMA control
  uses 164 registers.
- The production packet A/B explicitly replays `PLOW_UNISEG=0`,
  `PLOW_SEG_CLASS_SLICE=1`, `PLOW_SEG_FA512=all`, `PLOW_SEG_PURE_GEMM=1`, and
  `PLOW_SEG_SLICE_ALL=1`. Both variants retain 483 launches and 197,337
  workgroup packets at M4096/M8192. Across seeds 17/23/29, prompt and output
  checksums match exactly. Mean TTFT improves 219.132 -> 200.569 ms at M4096
  (-8.47%, 1.093x) and 529.396 -> 458.292 ms at M8192 (-13.43%, 1.155x).
- Promote cp.async for the exact px4 object and bind its true 70,672-byte arena.
  Preserve the larger arenas for the separate BQ64 WGMMA geometries.
  Evidence: `/tmp/px4-exact-sweep/multiseed.jsonl`,
  `/tmp/pfattn-px4-role-ab/{control,candidate}` and
  `/tmp/px4-fast-{control,candidate}-seed{17,23,29}.{json,log}`.

### 2026-09-13: W8A8 activation round-trip audit

The exact W8A8 packet still has three material activation boundaries per
layer. The concrete first-layer sequence is:

1. `RmsNorm(act.x -> act.hn)` and `QuantFp8(act.hn -> act.xqh, act.ash)` run in
   the fat/light segment; Q/K/V `GemmFp8` then run in the class-8 lean segment.
2. After attention, `QuantFp8(act.at -> act.xqo, act.aso)` is a standalone
   fat/light segment; O `GemmFp8` is the next class-8 lean segment.
3. The pre-MLP `NormResidual + RmsNorm + QuantFp8` fat/light segment produces
   `act.xqh/act.ash`; gate and up are two `GemmFp8` instructions in one class-8
   lean segment and materialize `act.gt/act.ut`; the next fat/light segment is
   `Glu(act.gt, act.ut -> act.fu) + QuantFp8(act.fu -> act.xqi, act.asi)`; the
   following class-8 lean segment is down `GemmFp8(act.xqi, act.asi -> act.dg)`.

This is a packet construction issue, not a missing runtime fence. The exact
program has 958 coarse counters and no fine counters. `Builder::finish` derives
the queue successors and segment waits from those instruction dependencies, so
fusion must be emitted before `finish` (or explicitly rebuild every dependency);
a role-metadata rewrite cannot safely delete a producer after the fact.

Generic on-chip handoff between the current tasks is not viable. `tilegraph_stat`
reports 480 KiB for `act.xqh` into Q/K/V or gate/up, 512/1024 KiB for
`act.xqo` into local/global O projection, and 1920 KiB for `act.xqi` into down.
These exceed the H100 packet arena limit (about 99 KiB), and the current consumer
K contraction has at most 32 busy CUs. Merely putting adjacent instructions in
one host segment therefore keeps the HBM stores/loads; forcing their existing
tile maps into one CTA would sacrifice machine occupancy. A useful fusion needs
a compound kernel with its own internal tile pipeline.

The smallest safe first change is an opt-in exact Gemma-4 SM90 W8A8 gate/up+GLU
role for M4096/M8192, N15360, K3840, GeGLU and TP1. Emit the already-defined
packet opcode `GemmGluFp8` at Builder time even though the pure-GEMM packet uses
`PLOW_NO_GLU_FUSE=1`, then route only that instruction to a separate hash-bound
lean cubin. Allocate the next free role/ABI at implementation time. Do not add
the fused body back to the generic class-8 object: its register footprint was
the reason pure-GEMM packets disabled GLU fusion. The fused instruction consumes
the same packed row-dense `act.xqh/act.ash` and mapped gate/up weights and writes
the same BF16-rounded `act.fu`; the existing `QuantFp8` and down projection stay
unchanged. `QuantFp8` depends directly on the new instruction, so coarse counters,
packed-request row ordering, and the interpreter/role dispatch contract remain
unchanged. This is the W8A8 analogue of the isolated BF16 Gemma-4 `GemmGlu` role,
but it must match `GemmGluFp8` and FP8 TMA descriptors rather than reuse the BF16
role's eligibility contract.

The direct gate/up+GLU role removes `act.gt/act.ut` stores and reloads: exactly
`8*M*15360*48` bytes, or 22.5 GiB at 4K and 45.0 GiB at 8K. At the H100's
3.35 TB/s peak this traffic alone floors at 7.21/14.42 ms, 3.21%/2.64% of the
224.55/546.32 ms TTFT baselines. This is an upper opportunity estimate because
L2 reuse can reduce physical HBM traffic. The hard trace ceiling is the entire
48-site `Glu+Quant` subtotal, 16.253/31.282 ms = 7.24%/5.74% of packet time;
the first role cannot realize that ceiling because it retains row quantization.
For comparison, the existing T11 GLU-to-quant fold removes only one BF16
`act.fu` reread (5.625/11.25 GiB) and measured about 3.0 ms at 4K and
3.5--5.3 ms at 8K, consistent with the traffic accounting.

Do not fold output quantization into this first role. Dynamic per-row scaling
over 15360 columns needs a cross-CTA maximum before E4M3 conversion. Avoiding
`act.fu` global storage would require a new deterministic reduction protocol or
retaining an entire row, while the current T11 path already preserves the BF16
rounding boundary. Treat a compound `GemmGluFp8 + QuantFp8` role as a second
experiment only after the producer-epilogue role passes exact tensor checks,
full logits, real-prompt greedy agreement, and same-lease packet A/B.

The other boundaries have lower or structurally harder ceilings:

- Fusing RMSNorm and Quant saves only one BF16 `act.hn` reread: 2.8125/5.625
  GiB over the 96 norm sites, a 0.90/1.80 ms peak-bandwidth floor. It still must
  materialize BF16-rounded `act.hn` while computing the row maximum, and prior
  serving evidence was mixed. Keep it behind its existing opt-in.
- Post-attention quant totals only 2.880/4.756 ms in the trace, so eliminating
  that whole segment is capped at 1.28%/0.87% TTFT. Its 512/1024 KiB handoff is
  also too large for the interpreter arena. It is lower priority than attention.
- Sharing `act.xqh` across Q/K/V or gate/up is best left as a global/L2 handoff
  for now. Its logical write plus consumer reads are 2.8125/5.625 GiB for QKV
  and 2.109/4.219 GiB for gate/up, only 0.90/1.80 and 0.68/1.35 ms at peak HBM.
  A real removal requires an exact compound projection role that quantizes and
  contracts internal BK tiles; scheduler co-location alone cannot do it.

Evidence: `/tmp/gemma4-w8a8-exact-profile.log`,
`/tmp/w8a8-tilegraph-{4096,8192}.txt`, and
`/tmp/w8a8-inst-4096.txt` from packet
`/tmp/gemma4-w8a8-exact-c1-4k8k-v2`.

### 2026-09-13: exact W8A8 gate/up+GeGLU role promotion

- Added a default-off, hash-bound SM90a role for exact Gemma-4 W8A8
  `GemmGluFp8` at M4096/M8192, N15360, K3840. Plowc fuses gate/up before
  packet finalization, keeps the instruction on one 132-CTA grid, and routes
  each of the 48 layer sites to a separate 384-thread lean object.
- The role uses one 128-thread TMA producer and two 128-thread WGMMA consumers,
  BM128/BN128/BK128 and four stages. Its epilogue preserves the interpreter's
  separate BF16 rounding of scaled gate and up before GeGLU. The cubin uses
  160 registers, 1040 bytes static shared memory, 197,696 bytes dynamic arena,
  and zero stack/spills/local memory.
- The direct full-tensor harness is bit-identical and improves 1.248 -> 0.951 ms
  at M4096 (1.312x) and 2.419 -> 1.857 ms at M8192 (1.303x).
- Production packets replay the same five fast segmentation settings and hold
  the exact cp.async attention object constant. The candidate replaces the 96
  separate gate/up sites at the two rungs with 96 role segments while runtime
  still executes 483 packet launches. Across seeds 17/23/29, prompt and output
  checksums are identical. Mean TTFT improves 200.758 -> 186.666 ms at M4096
  (-7.02%, 1.076x) and 458.348 -> 429.166 ms at M8192 (-6.37%, 1.068x).
- The packed/live-KV contract now authenticates all three E4M3 tensor maps
  `(activation, gate, up)` and their source extents. This is required for CUDA
  prefix cache plus packed prefill; partial, aliased, duplicate, or stale maps
  fail before engine load.
- Promote the exact fused role. Combined with exact HD512 cp.async, the old
  narrow packet's 219.132/529.396 ms becomes 186.666/429.166 ms, a cumulative
  14.81%/18.93% reduction. The 50% target remains open.
  Evidence: `/tmp/gemma4_w8a8_glu_exact`,
  `/tmp/gemma4-w8a8-glu-fast-v2-{control,candidate}` and
  `/tmp/w8glu-fast-v2-{control,candidate}-seed{17,23,29}.{json,log}`.

### 2026-09-13: exact 4K/8K W8A8 GEMM tile and stage sweep

An actual lean-object sweep on the H100 compared the shipped WS384 FP8 body
(BM128/BN256/BK128, four TMA stages, one producer plus two consumer
warpgroups) with the existing occ-2 BM128/BN128 role, BM64/BN256, shallower
WS384 rings, and WS384 BM128/BN128. Every compiled object used native Plow TMA
and WGMMA. No library fallback was involved.

The resource gate rejects two candidates before timing. The 256-thread
BM64/BN256 body cannot compile at the 128-register entry needed for two
CTAs/SM (`ptxas C7602`, minimum 154). Raising its entry to 160 makes it one
CTA/SM and only 426 TF/s on M4096 q_proj. The valid 256-thread BM128/BN128
object is 128 registers, 99,376 B dynamic shared memory and zero spills, but
only 665.9 TF/s versus 879.3 TF/s for WS384 on the same q_proj. Duplicating the
producer per math warpgroup does not compensate for the weaker CTA tile.

The useful WS384 configurations all compile at 160 registers with zero stack,
local memory, or spills. BN256/NS4 claims 197,696 B dynamic shared memory;
BN256/NS3 claims 148,528 B; BN128/NS6 claims 197,728 B. Their exact-shape
segment medians were:

| shape | M | control us / TF/s | best screened candidate us / TF/s | result |
|---|---:|---:|---:|---:|
| q, N4096 K3840 | 4096 | 146.880 / 877.2 | NS3 151.456 / 850.7 | keep NS4 |
| q, N4096 K3840 | 8192 | 275.616 / 935.0 | NS3 279.168 / 923.1 | keep NS4 |
| global k/v, N512 K3840 | 4096 | 43.904 / 366.8 | BN128/NS6 27.360 / 588.7 | **1.605x** |
| global k/v, N512 K3840 | 8192 | 48.672 / 661.8 | BN128/NS6 46.848 / 687.6 | **1.039x** |
| sliding k/v, N2048 K3840 | 4096 | 79.200 / 813.4 | BN128/NS6 82.656 / 779.4 | keep BN256 |
| sliding k/v, N2048 K3840 | 8192 | 139.680 / 922.5 | BN128/NS6 164.544 / 783.1 | keep BN256 |
| gate/up, N15360 K3840 | 4096 | 554.496 / 871.4 | NS3 566.560 / 852.8 | keep NS4 |
| gate/up, N15360 K3840 | 8192 | 1054.752 / 916.2 | NS3 1065.696 / 906.8 | keep NS4 |
| down, N3840 K15360 | 4096 | 422.912 / 1142.5 | NS3 410.752 / 1176.3 | **1.030x** |
| down, N3840 K15360 | 8192 | 883.264 / 1094.1 | NS3 853.984 / 1131.6 | **1.034x** |
| o, N3840 K4096 | 4096 | 167.936 / 767.3 | NS3 164.384 / 783.8 | 1.022x |
| o, N3840 K4096 | 8192 | 327.072 / 787.9 | BN128/NS6 318.912 / 808.1 | 1.026x |

The BN128 ring sweep at global-KV M4096 was NS2/3/4/5/6 =
37.600/29.984/27.872/27.776/27.488 us; at M8192 it was
78.912/56.064/49.984/47.200/46.880 us. Six stages is the stable selection.
Full-output FNV hashes match the production control for BN128/NS6 global-KV
and BN256/NS3 down at both M values. All screened objects also pass 257
deterministic f64-reference samples with rel-L2 0.00178--0.00326. These bodies
preserve the same four `m64n*k32` issues per BK128 stage and therefore the same
FP32 accumulation and BF16 rounding order.

The packet contains only eight N512 projections, 48 down projections, and 40 O
projections. Applying every isolated win therefore saves only about 0.86 ms at
4K and 1.75 ms at 8K: 0.9% of GEMM time and 0.32--0.38% of total TTFT. Tile and
ring selection alone cannot deliver the requested 50% rung improvement.

Production recommendation: record BN128/NS6 for `(M,N,K)=(4096,512,3840)` in
TuneDB, but select it only when it can reuse an existing launch/role; a new
segment launch can erase its 0.13 ms packet-wide gain. A BN256/NS3 long-K down
role is the stronger integration candidate because down is already its own
contiguous segment and saves 0.58/1.41 ms at 4K/8K. Keep the shipped BN256/NS4
role for q, sliding k/v, and gate/up. Reuse the existing GemmFp8 opcode and
packetized segment protocol; select a lean object/role from the tuned shape
rather than adding a new math opcode. Reaching a material rung win still needs
the compound gate/up+GLU role described above or a new WS384 mainloop that
closes the remaining 1.3--1.6x gap, rather than more BM/BN/stage enumeration.

Temporary evidence: `/tmp/gemma4_w8a8_gemm_tiles.cu`,
`/tmp/gemma4_segment_gemm_ws{128,384}.cu`, `/tmp/pfgemm_ws*.cubin`, and the
`fa512-w8gemm-*` gpulease runs in this agent transcript. No tracked source was
changed.

### 2026-09-13: rung-agent harness contract

`scripts/gemma4_h100_kernel_tuner.py` now treats every packet-derived rung
profile as a reproducible search cell. A candidate must declare one hypothesis,
one lever, and expected counter movement. Both arms bind packet, binary, cubin,
symbol, launch, tile, pipeline, TMA/swizzle, register, shared-memory, stack, and
spill identity. Five-seed correctness finishes before four balanced-order
10-warmup/50-sample trials under one `gpulease`. Cache state and SM/memory
clocks must match; the default gate rejects stack/spill traffic. The external
summary retains all verification hashes and timing samples, ranks
occurrence-weighted savings in `rung_rollup`, and labels an isolated winner
`kernel-qualified-candidate` pending the packet-role/block gate. This is the
required loop for the remaining 4K/8K work and for extending the same method to
the other packet rungs.

The first light-op lever run through this contract is rejected. An opt-in
packet handoff made each `NormResidual` accumulate the BF16 residual's row-square
sum into the existing activation-scale buffer, and the following `RmsNorm`
skipped its first hidden-row read and reduction. Default-off emission reproduced
the control packet byte for byte. The candidate kept all 483 segment launches
and every paired prompt/output checksum matched. Across seeds 53/59/61, mean
p50 TTFT changed 158.839 -> 159.331 ms at 4K (+0.31%) and 374.209 -> 373.573 ms
at 8K (-0.17%). It also raised the light object's spill stores/loads from
12/16 to 16/20 bytes. Remove the lever: its extra residual-side square and
reduction cost cancels the saved RMSNorm pass. Evidence:
`/tmp/normstat-{control,candidate}-seed{53,59,61}.{json,log}`.

### 2026-09-13: exact paired-GQA2 HD256/BKV32 role

- Added a packet-authenticated role for the 40 sliding HD256 attention sites at
  exact M4096/M8192. It pairs the two GQA query heads sharing each KV head while
  retaining BQ64/BKV32 accumulation order, packed metadata, and the production
  KV slot contract.
- The dedicated object uses 256 threads, 141,312 B dynamic shared memory, 244
  registers, 16 barriers, and zero stack, spills, or local memory. Replacing two
  local shared-state arrays with bitfields removed the original 16-byte stack
  without changing the algorithm.
- Same-lease alternating control/candidate runs across seeds 17/23/29, one
  warmup and three repetitions, preserve every prompt and output checksum.
  Median TTFT moves 186.759 -> 170.180 ms at M4096 (-8.88%, 1.097x) and
  428.072 -> 395.360 ms at M8192 (-7.64%, 1.083x).
- `--replay-knobs` does not reproduce unrecorded environment choices. Exact
  reproduction of this packet also requires `PLOW_UNISEG=0`,
  `PLOW_SEG_CLASS_SLICE=1`, `PLOW_SEG_FA512=all`, `PLOW_SEG_PURE_GEMM=1`, and
  `PLOW_SEG_SLICE_ALL=1`. With those settings the control packet hash is stable
  and both packets carry 483 segments.
- Evidence: `/tmp/gemma4-w8a8-gqa2-role14-{control-v4,v5}`,
  `/tmp/gqa2-role14-{control-v4,candidate-v5}-audit.json`, and
  `/tmp/gqa2-role14-v5-*.log`.

### Next: non-GEMM/non-attention packet work

The W8A8 4K/8K packet still executes 192 row quantizations, 144
HeadNorm/RoPE operations, 96 residual operations, 97 RMSNorm operations, and
48 GLU sites per prefill program. Prior measurements reject the broad
residual-plus-next-norm fusion because it changes reduction order and regresses
TTFT. Optimize compound boundaries that remove an activation materialization
while preserving the existing arithmetic order. Rank candidates by complete
packet time, require zero local memory/spills, and qualify exact output hashes
before promotion.

### 2026-09-13: light role rejection and packed FP8 conversion

An exact SM90 role for the 95 residual/RMSNorm segment pairs is rejected. The
256-thread object used 56 registers, 61,440 bytes of shared memory, and no
stack or spills. The packet retained 483 segments and routed 95 of them to the
role. At two CTAs/SM it regressed seed-17 TTFT from 168.535 to 181.692 ms at
4K and from 391.157 to 401.125 ms at 8K. Removing its shared staging changed
less than 0.2 ms. The loss is the extra role/module transition at 95 sites,
not the arithmetic. Keep light operations in the resident packed light object.

FP8 encode now uses four native `float2 -> e4m3x2` conversions per 8-element
vector instead of eight scalar conversions. SASS confirms four packed
`F2FP.SATFINITE.E4M3` instructions instead of eight. Standalone exact-shape
tests are byte-identical and improve by 0.6--3.0% for vLLM-compatible division
quantization and 0.1--4.4% for multiply-by-reciprocal quantization across
M4096/M8192, K3840/4096/15360, and 132/264 blocks. The vector path remains
spill-free at 28--32 registers.

The packet-specialized light build initially faulted on segment 4. With launch
blocking, the failing segment is the first `QuantFp8`: the segment builder had
compiled an exact W8A8 packet without `PLOW_NV_W8A8=1`, so the required arm was
absent and the interpreter trapped. The build now derives W8A8 from either
`PLOW_BUILD_W8A8=1` or packet metadata requiring `QuantFp8`. The corrected
object is 128 registers with a 16-byte stack and launches at 264 CTAs.

A three-seed matched A/B between scalar and packed conversion in otherwise
identical packet-specialized light objects preserves every prompt/output hash.
Mean p50 TTFT is 163.265 -> 162.432 ms at 4K (-0.51%) and 382.931 -> 382.283
ms at 8K (-0.17%). Evidence: `/tmp/plow_quant_fp8x2_ab.cu`,
`/tmp/quant-exact-{scalar,packed}-seed{17,23,29}.{json,log}` and
`/tmp/quant-fp8x2-{control-objects-v1,objects-v2}`.

### 2026-09-13: exact-width cooperative FP8 quantization

The resident packed light object now selects cooperative row groups for the two
wide Gemma-4 activation shapes: four warps per row at K4096 and eight at
K15360. K3840 retains one warp per row. At the production 264-block grid, the
standalone K15360 kernel improves 156.221 -> 114.448 us at M4096 and 306.368 ->
219.443 us at M8192; K4096 improves 32.726 -> 31.280 us and 62.083 -> 57.914
us. All tested row scales and FP8 bytes are identical. The helper is noinline,
so the exact packet object retains the control resource profile: 128 registers,
16-byte stack, 12-byte spill stores, and 16-byte spill loads.

A matched three-seed packet A/B preserves every corresponding prompt and output
checksum. Mean p50 TTFT improves 160.805 -> 159.068 ms at 4K (-1.08%) and
378.790 -> 374.629 ms at 8K (-1.10%). The normal segment builder produces the
same packed-light SHA256 as the manually isolated object:
`f048914fca621436de35e5b3299747e9b0b91c47b5efc607819700de15b227b3`.
Evidence: `/tmp/plow_quant_wpr_ab.cu`, `/tmp/quant-wpr-{control,candidate}-seed*.{json,log}`,
and `/tmp/quant-wpr-builder-final`.

### 2026-09-13: rung harness control anchors and resource gate

The exact-profile H100 tuner now treats its output as T2 evidence only. Every
candidate supplies predicted isolated savings and a directional counter
hypothesis. Four trials run control-before/candidate/control-after; qualification
requires a gain above the anchor-derived absolute noise floor and the declared
counter movement in at least three trials. The external rollup keeps predicted,
noise-floor, and realized occurrence-weighted savings so the next lever is
chosen from the remaining rung gap rather than raw speedup.

Tile changes now carry the hardware facts that control whether the result can
run: SM count, launch blocks, block size/warps, driver-reported blocks per SM,
cluster shape, registers and dynamic shared memory. The tuner rejects impossible
H100 register/shared-memory occupancy and grid/cluster combinations. Devgen's
resource audit confirms that AMD tile selection is bounded by
`hwspec::ArchGeometry` and the probed opcode inventory. NVIDIA generic prefill
emits one dtype opcode whose tile remains cubin-wide; exact tile variants must
therefore use a separate role object exporting block/arena ABI. Plowrt reads
those globals and recomputes occupancy from the loaded cubin. Do not publish a
NVIDIA exact-shape TuneDB winner until the T3 packet role uses that same entry,
block, arena, and object hash.

Devgen now checks every selected SM90 attention role against the named GPU's
`hwspec`: ISA, `warps * warp_lanes == block_threads`, maximum resident threads,
and configurable shared-memory capacity. This makes a BQ64/512-thread tile carry
16 warps and its own arena contract instead of inheriting the BQ32/256-thread
resources. Register-limited occupancy remains a runtime driver check; the cubin
reader does not expose per-entry register allocation.

### 2026-09-13: HD512 px4 BQ64 packet-role integration

Role 15 carries the standalone BQ64 candidate into the exact M4096/M8192
packet rungs. Devgen requires SM90a, packed metadata, 512 threads/16 warps,
BQ64/BKV16, and a 108,048-byte arena; it hashes the exact cubin and does not
select the role for other rungs. Plowrt independently reads the cubin globals,
sets the declared dynamic shared memory, and uses the CUDA occupancy API to
require exactly one 512-thread CTA per each of H100's 132 SMs. The production
object compiles at 128 registers with a 32-byte entry stack and 12-byte entry
spill traffic; its noinline attention helper has 24-byte spill stores and
32-byte spill loads.

The driver smoke and three-seed alternating packet A/B passed under `gpulease`
and preserved every paired prompt/output checksum. The candidate is rejected
for promotion: median p50 TTFT is 170.373 -> 170.898 ms at 4K (+0.31%) and
395.391 -> 398.050 ms at 8K (+0.67%). Per-segment event attribution at 8K shows
the eight HD512 sites moving 136.692 -> 140.068 ms, while other class totals
are stable. The standalone direct-entry gain therefore does not survive the
packet wrapper: the BQ64 direct kernel was 1.439x faster and spill-free, but the
packet helper is slightly slower than BQ32. Keep role 15 default-off as the
integration harness. Next work is to eliminate the helper's spill/liveness cost
without merging the attention body into the counter wrapper; a naive inline
variant was rejected at compile time because it increased the entry to a
384-byte stack with 384/576-byte spill traffic.

Evidence: `/tmp/gemma4_hd512_px4_bq64-five.jsonl`,
`/tmp/px4-role15-v1-{control,candidate}-seed{17,23,29}.{json,log}`,
`/tmp/px4-role15-segtime-{control,candidate}.log`, and candidate packet
`/tmp/gemma4-w8a8-px4-bq64-role15-v1`.

### 2026-09-13: HD512 packed exact-state recovery

The role-15 regression was caused by compiling a generic nullable packed helper,
not by BQ64 geometry. The object contained two complete attention bodies and
kept heads/KV-heads/window/split dynamic across a noinline boundary: 127 KiB,
128 reciprocal slow-path call sites, a 32-byte entry stack with 12/12-byte
spills, and 24/32-byte helper spills. Removing the unreachable null-request
body reduced it to 69 KiB. Binding the packet-authenticated Gemma-4 invariants
(16 heads, one KV head, global attention, one split) then produces a 127-register
object with zero stack or spills.

The exact role ABI is now v2 and exports those fixed invariants plus packed-only
as cubin globals. Devgen selects it only for matching M4096/M8192 instructions;
plowrt checks the instruction again and validates every global before load.
A three-seed alternating packet A/B against the prior BQ32 role preserves every
prompt/output checksum. Median p50 TTFT improves 170.375 -> 166.271 ms at 4K
(-2.41%) and 394.935 -> 378.686 ms at 8K (-4.11%). At 8K, the eight HD512
attention segments improve 136.612 -> 121.015 ms (-11.42%); unrelated segment
class totals stay stable. Evidence: `/tmp/px4-role15-v3-*.{json,log}` and
`/tmp/px4-role15-v3-segtime-*.{json,log}`.

### 2026-09-13: HD512 direct segment entry

The remaining BQ64 loss was the packet device-call boundary. Force-inlining the
attention helper removed its stack and spills but raised the entry to 128
registers and regressed each 8K global site from about 15.05 to 16.17 ms. Keep
that variant rejected.

Role 15 ABI v3 adds an exact host-marshaled segment entry. Plowrt authenticates
the object globals and packet hash, checks one ordered 132-entry FlashPrefill
instruction with no cross-grid counters, and passes only the packed request
table, tensor pointers, segment entries, successors, and counters. The kernel
still consumes packed requests and publishes the packet's successor counters;
all other segments stay on their existing packet interpreters. It compiles at
115 registers with zero stack or spills, versus 127 registers for the packet
device-call entry.

One-lease segment timing moves each 8K HD512 site from 15.20 to 12.82 ms
(-15.7%). Alternating graph-mode runs across seeds 17/23/29 preserve every
paired prompt and output checksum. Mean p50 TTFT improves 166.307 -> 160.703 ms
at 4K (-3.37%) and 379.791 -> 361.405 ms at 8K (-4.84%). A separate B2 packet
formed an R=2, 4096-row packed batch, ran the direct graph without a device
fault, completed all requests, and matched the fallback aggregate output hash.
Evidence: `/tmp/hd512-direct-v3-*.{json,log}` and
`/tmp/hd512-direct-v3-packed-b2-*.{json,log}`.

### 2026-09-13: W8A8 fused-GLU WGMMA queue-depth screen

The exact Gemma-4 fused W8A8 GLU role already uses the better one-group
consumer wait. Allowing two committed WGMMA groups in flight preserved the
8K prompt and output checksums, stayed at 160 registers with zero stack or
spills, but regressed the 48 fused-GLU sites from 88.752 to 90.163 ms (+1.59%)
and TTFT from 370.166 to 371.522 ms (+0.37%). Reject the two-group variant;
do not add it to tunedb. Evidence: `/tmp/gemmglu-wait2-seg-{control,candidate}.{json,log}`.

### 2026-09-13: HD256 paired-GQA2 direct segment entry

The paired sliding-attention role ABI v2 fixes the packet-authenticated Gemma-4
geometry (HD256, 16 query heads, 8 KV heads, window 1024, BQ64/BKV32, nsplit 1)
and adds a packed host-marshaled entry. Plowrt launches that entry only when the
packed request table is present; otherwise it retains the packet entry. Both
entries consume packet segment successors, and the direct object remains hash
bound. The direct entry compiles at 228 registers with zero stack or spills,
versus 244 registers for the packet entry.

One-lease segment timing preserves the prompt/output checksum and reduces the
40 HD256 sites from 29.396 to 28.308 ms (-3.70%). Balanced graph-mode runs over
seeds 17/23/29 preserve every paired checksum and improve mean p50 TTFT from
161.004 to 160.248 ms at 4K (-0.47%) and from 361.801 to 360.656 ms at 8K
(-0.32%). An R=2 packed run formed a 4096-row batch from two 2048-row requests,
completed four requests without a device fault, and retained aggregate output
hash `fnv1a64:30b1695c24b2aa4e`. Evidence:
`/tmp/hd256-gqa2-direct-*.{json,log}` and candidate assets
`/tmp/gemma4-w8a8-hd256-gqa2-direct-v2{a,-b2}`.

### 2026-09-13: W8A8 fused-GLU tile-raster screen

The exact fused-GLU standalone harness compared L2 raster bands 8, 16, 32,
and 64 at M4096/M8192 over seeds 17/23/29. Band 8 was the only possible
candidate, measuring 0.53% faster than the current band 16 at 4K and 0.11%
faster at 8K with bit-exact output. The packet screen reversed that result:
the 48 fused-GLU sites regressed 45.219 -> 46.142 ms at 4K (+2.04%) and
88.414 -> 89.764 ms at 8K (+1.53%); TTFT moved 170.337 -> 172.122 ms and
366.125 -> 366.541 ms. Keep band 16 and do not add band 8 to TuneDB.
Evidence: `/tmp/gemma4-w8a8-glu-band-screen.txt` and
`/tmp/gemma4-glu-band8-segtime-{control,candidate}.{json,log}`.

### 2026-09-13: W8A8 fused-GLU direct segment entry

Role 13 ABI v2 binds the exact BM128/BN128/BK128, four-stage, raster-band-16
object and adds a host-marshaled entry for its isolated M4096/M8192 segments.
The direct entry receives the seven output/TMA-map/scale addresses plus the
row count, runs the same producer/consumer WGMMA body at the packet's 132-CTA
grid, and signals the original packet successor counters. Plowrt requires an
ordered one-instruction grid and the object hash before selecting it. Both
packet and direct entries compile at 160 registers with zero stack or spills.

One-lease segment timing preserved both prompt/output hashes and reduced the
48 fused-GLU sites from 45.292 to 44.773 ms at 4K (-1.15%) and 88.509 to
86.456 ms at 8K (-2.32%). Alternating warmed graph-mode runs over seeds
17/23/29 preserved every paired hash and improved mean p50 TTFT from 160.551
to 159.293 ms at 4K (-0.78%) and from 360.655 to 358.060 ms at 8K (-0.72%).
An R=2 packed run formed a 4096-row rung from two 2048-row prompts, completed
4/4 requests, and matched aggregate output hash
`fnv1a64:30b1695c24b2aa4e`. Evidence:
`/tmp/gemma4-glu-direct-*.{json,log}` and candidate assets
`/tmp/gemma4-w8a8-glu-direct-v2{-b2,}`.

### 2026-09-13: W8A8 down-projection NS3 packet screen

The exact `(N,K)=(3840,15360)` W8A8 down projection was screened at M4096 and
M8192 as a dedicated WS384/TMA/WGMMA packet interpreter object. The object
compiled at 160 registers with zero stack or spills, selected exactly 48
segments at each rung, and preserved both prompt and output hashes. It did not
retain the earlier standalone gain inside the packet ladder: the 48-site
subtotal regressed 18.611 -> 19.409 ms at 4K (+4.29%) and 37.817 -> 38.172 ms
at 8K (+0.94%); cold TTFT moved 170.113 -> 172.783 ms and 363.560 -> 366.154
ms. Reject ABI 5/NS3 and retain the current four-stage packet object. Do not add
this result to TuneDB. Evidence:
`/tmp/gemma4-w8a8-down-ns3-{control,candidate}.log`.


### 2026-09-13: HD512 WGMMA numerical rejection

The BQ64 WGMMA candidates retain their large packet-time gain but do not retain
the accepted greedy sequence. On three natural-text chat prompts at each exact
rung, with 32 generated tokens, the single-chain candidate agrees with the
control on 83.33% of 4K tokens and 61.46% of 8K tokens. Splitting the head into
two reduction chains improves 4K agreement to 95.83% but leaves 8K at 61.46%.
The combined agreement is 72.40% and 78.65%, respectively. Reject both despite
the speedup; preserving the existing score/PV reduction order remains the
correctness constraint. Evidence: `/tmp/hd512-wgmma-quality-*.jsonl`,
`/tmp/hd512-wgmma-quality-compare.json`, and
`/tmp/hd512-wgmma-quality-halves-compare.json`.

### 2026-09-13: HD512 warp-cooperative TMA row issue

The exact BQ64/BKV16 PX4 object now lets warp zero issue the 16 K or V row-bulk
TMA transfers in parallel instead of making lane zero issue all of them. The
score, softmax, and PV instruction order and shared-memory layout are unchanged.
The direct entry compiles at 117 registers with zero stack or spills.

The one-lease packet screen preserved prompt/output hashes and reduced the eight
HD512 sites from 28.013 to 26.298 ms at 4K (-6.12%) and from 103.199 to 96.163
ms at 8K (-6.82%). Across balanced warmed graph runs at seeds 17/23/29, every
paired checksum matched and mean p50 TTFT improved 159.590 -> 157.621 ms at 4K
(-1.23%) and 358.602 -> 350.987 ms at 8K (-2.12%). A separate candidate run
formed an R=2, 4096-row packet from two 2048-row requests, completed 4/4, and
matched the control output hash `fnv1a64:30b1695c24b2aa4e`. Evidence:
`/tmp/hd512-rowwarp-{control,candidate}.log`,
`/tmp/hd512-rowwarp-graph-*.log`, and `/tmp/hd512-rowwarp-b2-*`.


### 2026-09-13: devgen hardware-resource audit

Devgen's analytical AMD GEMM picker is hardware-aware for ISA/dtype capability,
SM/CU count, matrix throughput, measured HBM bandwidth, per-CU LDS capacity, and
the target's single/double stage-buffer rule. It models wave fill through output
tile rounds and rejects tiles whose staged operands exceed LDS. Attention role
binding separately rejects mismatched architecture, warp/thread geometry, and
arena sizes above the target shared-memory ceiling.

Compiled resource use is not yet an input to generic tile ranking. The probed
inventory leaves every `KernelSpec.resource` unset; `ProbedObject` contains the
preprocessed arms and tile macros but no ptxas/ROCm resource envelope. Therefore
registers, spills, resident blocks/waves, max-warps/max-blocks, and compiler-
specific occupancy cannot change a tile decision. The Python H100 campaign and
TuneDB object records retain these facts, but devgen does not consume those
records for the generic picker. On NVIDIA, Gemm/Med/Small are correctly collapsed
as one object-wide body, so there is no honest per-opcode tile decision until
lean objects with distinct hashes and measured profiles are supplied.

Do not add an analytical register estimate: the exact objects have already shown
compiler-dependent register cliffs and spills. The correct follow-up is to make
the object builder publish an authenticated compiler resource envelope beside
the object SHA, attach it to the inventory/role candidate, apply the architecture
resource gate before ranking, and use measured packet performance for candidates
that remain legal. Until then, the current harness resource gate stays mandatory
for every 4K/8K promotion.

### Rejected: HD512 row-TMA issue restricted to warp 0 (2026-09-13)

Wrapping the row-TMA issue path in `tid < 32` raised the direct entry from 117 to 121 registers and regressed the exact HD512 sites despite exact output hashes:

- 4K: 26.308 ms -> 27.204 ms (+3.41%)
- 8K: 95.572 ms -> 99.476 ms (+4.09%)
- full rung: 168.418 ms -> 169.082 ms (4K), 356.737 ms -> 360.269 ms (8K)

Rejected and restored. Evidence: `/tmp/hd512-rowwarp-control.log`, `/tmp/hd512-rowwarp-candidate.log`; candidate asset `/tmp/gemma4-w8a8-hd512-rowwarp-w0-v1`.

### 2026-09-13: live-KV attention dispatch decision

The exact prefill programs and role metadata specialize query rungs M4096/M8192,
but the promoted attention objects do not switch on live KV history at runtime.
Packed request metadata correctly supplies each request's qlen/slot/kvlen. The
HD512 direct role nevertheless remains one BQ64/BKV16, 512-thread, 132-CTA,
nsplit=1 cubin for both rungs and all histories. The HD256 sliding role remains
BQ64/BKV32, 256-thread, nsplit=1; after its 1024-token window saturates this is
largely independent of total context.

Add an authenticated attention variant registry keyed by `(arch, dtype, kv_dtype,
query_rung, live_kv_bucket, head_dim, gqa, window, packed_topology)`. Each variant
carries object SHA, BQ/BKV, block/warps, nsplit, arena/register/spill envelope and
timing certificate. At launch, select by the packed request table's live history;
for mixed histories choose a qualified ragged variant or partition requests only
when the extra launch and merge pass win. Packet dependency/counter semantics do
not change. Current fixed roles remain the fallback until every required cell is
qualified.

Next HD512 candidates, in order: measure the current direct object; cluster adjacent
head CTAs and multicast their shared K/V TMA tiles for GQA16 without changing the
BKV16 arithmetic order; then test a non-divergent producer/consumer schedule. Tune
nsplit only at live-KV buckets where query-tile parallelism underfills or the shorter
KV slice repays FlashMerge. For exact 4K/8K full-query traffic, nsplit=1 is expected
to remain competitive because query/head tiles already saturate 132 SMs.

Next exact GEMM candidates: ping-pong consumers to overlap one tile's epilogue with
the next tile's WGMMA; stmatrix-to-smem plus TMA output store; cluster multicast on
the reused operand; and fused Q+KV/norm+quant roles when they eliminate a packet
activation pass. Stream-K/split-K is reserved for small-M/underfilled shapes; the
4K/8K wide projection grids already have ample output tiles and should not pay a
reduction pass without measured evidence.

### 2026-09-13: HD512 exact-object Nsight attribution

The accepted row-cooperative TMA cubin was profiled through a system-glibc CUDA
Driver API harness because root Nsight Compute aborts the Nix-glibc `plowrt`
binary. The harness launches the production direct symbol with grid 132, block
512, 108,048 bytes of dynamic shared memory, and M=KV=8192. Its unprofiled
median is 12.011 ms, matching the approximately 12 ms production segment site.

The profiled kernel takes 11.91 ms at 117 registers/thread, zero local/shared
spills, and one register-limited block per SM (25% occupancy). It reaches 27.35%
compute throughput, 62.47% memory throughput, 66.86% L1/TEX, but only 0.71%
DRAM with a 98.99% L2 hit rate. Tensor-pipe utilization is 14.12% and TMA-pipe
utilization is 1.18%. Schedulers have an eligible warp in only 29.26% of cycles
(0.54 eligible warps/scheduler). Per-issued-instruction stalls are led by short
scoreboard 4.71, barrier 2.85, wait 1.98, and MIO throttle 0.91.

Source counters report 436,862,976 excessive shared-memory wavefronts. The
dominant entries are the scalar score-tile `LDS` sequence (three-way conflicts,
approximately 25.1 million excessive wavefronts per instruction). The largest
barrier sample is the Q/K `LDSM.16.M88.2`; V's transposed `LDSM` also contributes.
This makes shared layout/access and phase dependencies the next one-variable
screen. HBM bandwidth and GQA multicast are not the first constraint for the
full-query 8K cell. Preserve BKV16 score/PV arithmetic order and reject any
candidate that raises the 117-register envelope. Evidence:
`/tmp/hd512-rowwarp-ncu{6,7,8}.ncu-rep`,
`/tmp/hd512-rowwarp-ncu8-source.csv`, and
`/tmp/hd512_direct_profile`.
