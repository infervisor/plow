# Sequence-parallel TP seams for K3 TP8 prefill — CPU feasibility gate (2026-09-04)

Branch `codex/seq-parallel-seams` off `codex/amd-agent-harness` @ 73c71c0. No GPU work. Sources:
the T=8192 candidate program (`/tmp/ttft-research/cand8192.json`, 2971 insts), its all-rank trace
(`/tmp/plow-tp-seg-major-trace/candidate.trace`, `candidate-report.txt`), `pf_idle.py` /
`gemm_shapes.py` re-run on them, `crates/devgen/src/k3.rs`, `crates/devgen/src/lib.rs`
(`emit_xreduce*`), `runtime/amd/op_collective.h` (`d_xreduce_twoshot_mega`), `runtime/amd/op_k3.h`
(`d_attn_res`), `runtime/amd/op_moe.h` (`d_moe_router_topk_pf`, `d_moe_align_pf`),
`crates/plowrt/src/exec/{amd.rs,amd_tp.rs,tp.rs}` (peer slots, gate accounting).

**Verdict: go-with-conditions.** Projected served TTFT 1113 → ~995 ms (−90..−150 ms), bit-exact
by construction with two named exceptions (GEMM family pinning; the FFN-seam gather fold must
keep its bf16 double rounding), no weight replication needed (the "+4.7 GB/rank up-proj
replication" in the brief is NOT required — see §1.4), +336 MiB/rank peer-mapped scratch.
The emitter half is implemented behind `PLOW_SEQ_PAR_SEAMS=1` (default off); the runtime half
(two interpreter arms + host band binding + gate bookkeeping) is ~250 lines and is specified in §4.

---

## 1. Data-flow analysis (emitted K3 TP8 prefill graph, T=8192)

Terms: `T`=8192 tokens, `H`=7168, `L`=3584 (latent), `E`=896 experts, `k`=16, `tp`=8, band
`B = T/8 = 1024` rows. Slot A = `act.og_tp` (peer slot 0), slot B = `act.dg_tp`
(slot 1, byte offset 117,440,512), slot C = `act.ug_tp` (slot 2). The two-shot
(`d_xreduce_twoshot_mega`) partitions the FLAT `[n]` message into 8 contiguous slices; with
`n = T*H` and `T % 8 == 0` slice `r` is exactly rows `[r*B, (r+1)*B)` — rank `r` OWNS row band `r`
after phase 1 (reduced values written IN PLACE into its own slot A/B, peer-visible).

### 1.1 One KDA MoE layer (layer 1: insts 23..55 of cand8192) — ops from o_proj to MoE stage-1

| # | op | row domain today | operands (in → out) | bytes moved / rank | band-legal? |
|---|---|---|---|---|---|
| 37 | `GemmC5` o_proj M=T N=H K=1536 | all T (K-sharded partial) | `act.pf.y` → `act.og_tp` (slot A) | R 25 MiB W 112 MiB | stays all-T (partial over local heads) |
| 38a | `XReduceTwoShot` RS phase (gate_rs) | band `r` | 8 peers' slot A band → own slot A band | 7×14.7 MiB remote R, 14.7 MiB W | **already band** |
| 38b | `XReduceTwoShot` AG phase (gate_ag) | all T | 8 slot-A bands → `act.pf.attn_full` | 7×14.7 MiB remote R, 112 MiB W | replaced by AG of results |
| (fused) | `Residual` → materialized into AttnRes `res_a/res_b` | all T | `act.xnext + attn_full` → `act.pf.prefix` | 3×112 MiB | **yes** (per row) |
| 39 | `AttnRes` (post-attn mix + fused post_ln), nb=1..8 | all T | `prefix`, `kv.blkres[T][nbcap][H]`, `mlp_res_score` → `act.pf.h2` | (nb+2)×112 MiB R, 112 MiB W | **yes** (one WG per token; §3) |
| 40 | `GemmWide` router M=T N=E K=H | all T, weight REPLICATED | `h2`, `gate.weight` → `act.pf.moe.logit` | 112 MiB R, 14.7 MiB W; 105 GFLOP | **yes** |
| 41 | `GemmC5` latent xe M=T N=L K=H | all T, weight REPLICATED | `h2`, `routed_expert_down_proj` → `act.pf.moe.xe` | 112 MiB R, 56 MiB W; 421 GFLOP | **yes** |
| 42 | `MoeRouterTopkPf` (k=16, flags=3, no atom/det) | all T | `logit`, `e_score_correction_bias` → `act.pf.moe.route_tab` (f32 `[T][k][2]`, 1 MiB) | 14.7 MiB R, 1 MiB W | **yes** (per token; §3) |
| 43 | `MoeAlignPf` b=1 | all T (global sort) | `route_tab` → `moe_meta`, `rowtok`, `rowpart`, `rowgate` | 1 MiB R, ~4.6 MiB W | **no** — needs the FULL route table |
| 44 | `MoeGroupGluPf` (lean stage-1, A4 reuse) | all T×k sorted rows | `xe` gathered by `rowtok`, local I/8 expert shards | 56 MiB+ R | **no** — TP-sharded experts: every rank runs every token |

After stage-1: 45 `MoeGroupDownPf`, 46 `MoeCombinePf` → slot B partial, 47 `XReduceTwoShot`
(latent, n=T*L, 56 MiB), 48 `RmsNorm` latent (all T, replicated), 49 `GemmWide` up-proj
M=T N=L→H/8 K=L (column-sharded, all T, `act.ug_tp` slot C), 50-53 shared experts
(column/K-sharded, all T, `→ act.og_tp` slot A), 54 `XReduceTwoShot` with folded gather
(`gslot`=slot C, `gcols`=896) → `act.pf.moe.sh_down`, 55 next layer's pre-attn `AttnRes`
(materialized `prefix = sh_down + prefix`, push at snapshot layers, fused `input_layernorm`) →
`act.pf.h_a`, then the next layer's four QKVG GEMMs (column-sharded, all T).

### 1.2 One MLA MoE layer (layer 3: insts 87..119)

Identical seam structure. Differences are all INSIDE the mixer (all-T, unchanged): `q_a_proj`,
`kv_a_proj_with_mqa`, `k_rope_down` read `h_a` (full T), the `kv.{l}.ckv/krot` cache rows are
written for all T, `FlashMlaPrefill` (raw V2/TR16), `MlaMergeFold`, `g_proj`, `MlaOutGate`,
then o_proj → slot A → the same 38..55 chain. Nothing in the MLA mixer reads the residual stream
or the ring, so the seam analysis is identical for KDA and MLA layers.

### 1.3 Which rows every consumer needs → what must be all-gathered

| consumer | reads | needs rows | ⇒ after the seam we must gather |
|---|---|---|---|
| shared-expert gate/up GEMMs (col-sharded), dense gate/up (layer 0) | `h2` | all T | **h2** (`T×H` bf16 = 112 MiB) — same bytes as today's `attn_full` AG |
| `MoeAlignPf` | `route_tab` | all T | **route_tab** (1 MiB) |
| `MoeGroupGluPf` | `xe` via `rowtok` | all T (TP experts) | **xe** (`T×L` bf16 = 56 MiB) |
| next layer QKVG / q_a / kv_a GEMMs, KDA conv, MLA cache | `h_a` | all T | **h_a** (112 MiB) — same bytes as today's `sh_down` AG |
| both `AttnRes`, final `AttnRes` | `prefix`, `x`, `xnext`, `kv.blkres` | **own band only** | nothing — the residual stream and ring are only ever read by AttnRes at the token that owns them |
| lm_head `Gemv` (a_row0 = T−1) | `act.xn` row T−1 | row T−1 (band 7) | **xn** after the final AttnRes (112 MiB, once) so every rank samples the same id |
| `MoeCombinePf`, latent `RmsNorm`, up-proj | `ylat`/`yn` | all T | unchanged latent two-shot |

Extra all-gather bytes per MoE layer vs today: `xe` 56 MiB + `route_tab` 1 MiB = **57 MiB**
(each rank receives 7/8 → 50 MiB over xGMI); 92 layers → 5.1 GiB per prefill. `h2`/`h_a`
gathers replace `attn_full`/`sh_down` gathers byte-for-byte. The FFN seam's column gather of
`ug_tp` SHRINKS 8×: today each rank pulls all T rows of 7 peers' 896-column slices
(7×14.7 MiB = 103 MB/layer); band mode pulls only its band (12.8 MB/layer) → −90 MB/layer.

### 1.4 Weight replication is NOT needed

The brief assumed the up-proj (`routed_expert_up_proj`, 3584→7168, column-sharded, 51.4 MB/layer)
must be replicated so the band can be up-projected locally. It need not: keep the up-proj
column-sharded on all T (it is TP-sharded work, not replicated work — FLOPs are identical
either way) and fold its gather into the FFN-seam reduce-scatter at BAND rows (§4.2). The only
replicated work left at the latent seam is the latent `RmsNorm` (~7 ms) — not worth 4.7 GB/rank.
Footprint impact of the recommended design: +3 peer slots × 112 MiB = **+336 MiB/rank** of
peer-mapped scratch (`PARTIAL_SLOTS` 3→6); 224.52 GiB → 224.85 GiB against 268.2 GiB (288 GB)
physical. Fits. (Even the rejected +4.4 GiB variant would fit; the EP dual-layout +224.5 GiB does
not — `gfx950-moe-ep-prefill-design-20260904.md:111-115`.)

### 1.5 Existing windows vs new plumbing

Already there:
- Row-band ownership of the flat message in phase 1 (`my_lo/my_hi = n*rank/nranks`), in-place
  reduced band in the rank's own slot: the band result is at `act.og_tp[r*B..]` with no copy.
- `e0` (`i5`) + `in->i[2] + e0*2` window offset in the interpreter (`interp.hip:4502`): a two-shot
  over a row window of the slot (token-slice pipelining design, `emit_xreduce_twoshot_band`).
- `resid`/`out2` (`t1`/`t2`): the AG writes `bf16(resid + reduced)`; `PLOW_XR_ATTNRES` arm has the
  `rows_per_rank / owner` row-band arithmetic; `a_row0`/`c_row0`/`t_row0` on GEMM/Combine for
  row-band packets; lm_head `a_row0` is host-patched per chunk.
- The folded gather (`gslot`/`gcols`) with its bf16-round-then-add arithmetic.
- The `Residual → AttnRes` materialized-input fusion in `Builder::finish` matches on handles, so
  band views fuse exactly as full tensors do.

New plumbing (none of it is a kernel body change; §4):
1. **Rank-relative band views of activations.** The blob is one packet for eight ranks; "rank r
   reads rows [rB,(r+1)B) of X" is not expressible today (`k3.rs:1395-1400` says exactly this).
   Emitter side (done): a tensor named `<base>@band<t>` with `bytes = (t/tp)·row_bytes`. Host side:
   bind it as a view at `base + rank·bytes` (same literal-name contract as `act.og_tp`).
2. **RS-only and AG-only collective arms**: `DevOp::XReduceScatter` (25) and `DevOp::XAllGather`
   (26) — reserved ABI numbers, unimplemented until now; the two-shot body is split at its
   existing phase boundary.
3. **Three more peer slots** `act.h2_tp`, `act.xe_tp`, `act.rt_tp` (slots 3/4/5) so band results
   are peer-visible for the AG; `PARTIAL_SLOTS` 3→6 in `tp.rs`.
4. Gate bookkeeping for the two new ops (`count_xgates`, `gate_expectations`) — gate COUNT per
   layer is unchanged (RS+AG = 2, as the two-shot), so `PeerLayout::counters_for` holds.

---

## 2. Cost model

Per-family costs from the T=8192 all-rank trace (`candidate-report.txt`, per packet wall) and
`gemm_shapes.py`; AttnRes uses the promoted f32-mix phase object (0.260 ms/site,
`campaign-summary:123`), everything else the served interpreter/lean route.

| moved op (per MoE layer) | today, all T | on band B=1024 (est.) | saved/layer | ×92 (×186 AttnRes) |
|---|---:|---:|---:|---:|
| AttnRes f32-mix (2 sites/layer) | 0.260 ms | ~0.045 ms (1/8 bytes + ramp) | 0.215 | **40 ms** |
| router GemmWide 8192×896×7168 | 0.178 ms | ~0.08 ms (32 tiles of 128×256 on 256 CUs: tile-bound) | 0.10 | **9 ms** |
| latent xe GemmC5 8192×3584×7168 | 0.503 ms | ~0.13 ms (84 tiles of 192×256, tile-bound) | 0.37 | **34 ms** |
| MoeRouterTopkPf | 0.564 ms | ~0.09 ms (4 tokens/WG instead of 32) | 0.47 | **43 ms** |
| FFN-seam ug column gather | ~103 MB remote R | 12.8 MB | ~0.15 ms | **10..20 ms** |
| **compute + gather saved** | | | | **−136..−146 ms** |

Costs added:

| item | per layer | ×92/93 |
|---|---:|---:|
| AG of `xe` + `route_tab` (57 MiB; 50 MiB received/rank) at the in-network AG rate (0.575 ms per 112 MiB) | +0.29 ms | **+27 ms** (pessimistic; +12 if xGMI-bound at ~450 GB/s) |
| +2 packets/layer (RS and AG split, 3-way AG is ONE packet) | ~5 µs each | +1 ms |
| rendezvous count | unchanged (2/seam) | 0 |
| band GEMM under-fill at 32/84 tiles vs 256 CUs | included above | — |

Net: **−110 ms (range −90..−150)** → served TTFT 1113 → **~1000 ms (960..1025)**. Error bars come
from (a) the AG rate in-network (12..27 ms), (b) band GEMM tile-bound estimates (±15 ms), (c) the
ug gather saving (10..20). Not counted: the sh_down/attn_full AG bytes are unchanged; the critical
path SHRINKS (band compute is 1/8 of the all-T compute that today sits between AG and stage-1).
Footprint: +336 MiB/rank peer scratch, no weights. Gate count unchanged (556 per prefill).

---

## 3. Exactness argument

Bit-exactness target: the served packet checksum `fnv1a64:71a28c1449921c95` (TP8 8192→256).
A row `t` is computed by exactly one rank (its owner) with the same kernel body and the same
operand order as the replicated computation; every other rank receives its bytes. Op by op:

| op | per-row arithmetic identical on the band? | why |
|---|---|---|
| RS phase | yes | unchanged code (`PLOW_XR_RS_U` strict rank order 0..7 f32 acc → bf16); only the AG is split off |
| `Residual` (materialized into AttnRes) | yes | `d_materialize_residual`: `f2bf(bf2f(a)+bf2f(b))` per element, token loop `t = slice..T step nblk` |
| `AttnRes` (interp) / `plow_attn_res_f32mix` | yes | one WG per token; per-row variance/score reductions are intra-WG; ring per token `[t][NBCAP][H]`; the softmax couples ring rows of ONE token, never rows of different tokens; the push writes row `nb` of the same token. `T` and the grid only change which WG owns which token |
| router `GemmWide`, xe `GemmC5` | yes, **if the family is pinned** | `d_gemm_t` accumulates each output element over `kt = 0..K/BK` in one accumulator chain; tile position only selects rows/cols. The emitter must NOT re-pick the tile at M=B (`pick_gemm_emit_plan(B,…)` could choose a different family / c8) — the prototype picks at the FULL M and emits with `i0 = B` (`emit_k3_linear_rows`). **Exception #1** |
| `MoeRouterTopkPf` | yes, **if `pf_fuse == None`** | per-token top-k under a token loop; the ATOMIC/DET arms zero the `[T,L]` accumulator with `T` — on a band they would zero 1/8 of it. The emitter asserts `MoePfFuse::None` (the served default). **Exception #2** |
| `MoeAlignPf`, stage-1/2, combine | unchanged | run on the FULL table/xe after the AG; within-expert row order is still absorbed by `row_partidx` |
| FFN-seam gather fold | yes, **if the RS arm rounds twice** | today phase 2 stores `f2bf(bf2f(reduced_bf16) + bf2f(g))`; the RS-with-fold arm must store `f2bf(bf2f(f2bf(acc)) + bf2f(g))` — round the reduction to bf16 FIRST (k3.rs:1571-1576 explains the 1-ULP/layer token flip otherwise). **Exception #3** (kernel-side contract) |
| final AttnRes + lm_head | yes | band mix, then AG `xn`; every rank reads the same row T−1 |
| AG ordering | value-independent | the AG copies bytes; peer visit order (`(slice+rank+i)%nranks`) never touches values. Determinism of the packet stream is unchanged: same DAG, counter-gated, one more Coarse edge per seam |

NOT band-legal (kept all-T, by construction): `MoeAlignPf` (global sort), stage-1/2 (TP-sharded
experts), latent norm + up-proj (feed the column-sharded up-proj), shared-expert GEMMs, all
mixer ops. Anything that reduces ACROSS rows: none of the moved ops do. Rank-local state: the
ring and residual stream become band-valid only — they have no reader outside AttnRes.
Unsupported → refused/fallback: `T % tp != 0`, `RowKind::Sequences`, decode (`t == 1`).

WAR safety of the new slots (protocol, not timing): a rank rewrites `act.h2_tp` at the next seam's
band AttnRes, which runs after that seam's RS rendezvous, which every peer signals only after
finishing the previous AG that read the slot. `act.xe_tp`/`act.rt_tp` are rewritten at the next
layer's seam 1, fenced by two intervening rendezvous. Same argument as slot A reuse
(`k3.rs:1490-1508`).

---

## 4. Implementation

### 4.1 Emitter (DONE, `PLOW_SEQ_PAR_SEAMS`, default off)
- `crates/devgen/src/emit_config.rs`: `seq_par_seams`.
- `crates/devgen/src/lib.rs`: `emit_xreduce_scatter` (op 25: `t0=slot tensor i0=n i1=n_gpu i2=slot
  i3=gate_rs i6=gslot? i7=gcols?`), `emit_xall_gather` (op 26: `t0..t2=dst i0..i2=n i3=gate i4=n_gpu
  i5..i7=src slot byte offsets`, one rendezvous for up to three arrays).
- `crates/packet/src/{dev.rs,slots.rs}`: operand docs; 25/26 leave `RESERVED`.
- `crates/packet/src/devbuild.rs`: `Builder::tensor_name`.
- `crates/devgen/src/k3.rs`: `K3Tp { seq_par, h2s, xes, rts }` (slots 3/4/5, `act.h2_tp`,
  `act.xe_tp`, `act.rt_tp`); `band()` views `<base>@band<t>`; seam 1 = RS → band Residual →
  band AttnRes → band router/xe/top-k → one 3-way AG (MoE) or 1-way AG (dense layer 0);
  seam 3 = RS with band gather fold → band Residual → (next layer) band AttnRes → AG `h_a`;
  final AttnRes on band → AG `xn`; latent seam unchanged. `emit_k3_linear_rows` pins the GEMM
  family at the full M.
- `crates/devgen/src/manifest.rs`: requirement `PLOW_SEQ_PAR_SEAMS=1` when the packet carries
  `XAllGather`, so an object without the arms refuses the blob instead of silently no-op'ing.
- `crates/plowrt/src/exec/amd_tp.rs`: `count_xgates` / `gate_expectations` know ops 25/26.

### 4.2 Runtime / object (NOT done; ~250 lines, no new kernel bodies)
1. `runtime/amd/op_collective.h`: `d_xreduce_scatter_mega` = rendezvous 1 + phase 1 of
   `d_xreduce_twoshot_mega` (+ optional band gather fold with the double rounding of §3), return.
   `d_xall_gather_mega` = one-WG-speaks signal on `gate` (the producers are earlier packets — the
   same argument as gate_rs) + wait `nranks` + the phase-2 copy loop for each `(dst, n, slot)`.
2. `runtime/amd/interp.hip`: `case PLOW_DOP_XREDUCESCATTER` / `PLOW_DOP_XALLGATHER` next to
   `PLOW_DOP_XREDUCE2` (both the `PLOW_BUCKET_XREDUCE` arm and the plain prefill arm);
   `dev_isa.h` already defines the opcodes. Add a `plow_seq_par_seams_arm` marker symbol and the
   `PLOW_SEQ_PAR_SEAMS=1` requirement string in the object inventory.
3. `crates/plowrt/src/exec/tp.rs`: `PARTIAL_SLOTS = 6`; `amd.rs:9336/9423`: bind `act.h2_tp`,
   `act.xe_tp`, `act.rt_tp` at slots 3/4/5; bind any `<base>@band<t>` tensor as
   `DeviceMem::view(base_addr + rank·bytes, bytes)` (resolve the base by name; the lean AttnRes
   f32-mix route resolves through the same table, so it needs nothing).
4. `crates/plowrt/src/asset/devblob.rs`: slot-bytes inference (`max(i[2])` over ops 24/29) is
   unaffected (op 25 carries `i2 ∈ {0, slot_b}`; op 26 carries slot offsets in `i5..i7`).
5. `scripts/build_gfx950.sh`: rebuild the prefill interpreter objects (interp_prefill*,
   `PLOW_BUCKET_XREDUCE`); the lean objects (MoE stage-1/2, KDA, MLA, AttnRes f32-mix) are untouched.
6. `scripts/k3_trace_report.py` / `xreduce_phase_report.py`: recognise ops 25/26 (cosmetic).

### 4.3 Gate plan
1. CPU: `cargo test -p devgen` (done) + emit the full TP8 packet with the flag and diff the
   op census against this doc (done: see §5).
2. Lean verify: `PLOW_TP_AUDIT` / `audit_xctr` with the new gate expectations on a 1-layer
   `K3_FULL=1 PLOW_K3_LAYERS=2` TP8 emit (`scripts/k3_tp_equivalence.sh` pattern), tp1 vs tp8
   token agreement.
3. TP8 exact gate: `plowrt bench` 8192→256 on the served packet vs the flag packet, checksum
   must equal `fnv1a64:71a28c1449921c95`; 3 order-alternated folds; TPOT must be neutral (decode
   programs are byte-identical — the flag is prefill-only).
4. Promote only if TTFT ≤ 1020 ms and exact; then the served showdown.

---

## 5. Prototype emit census (TP8, T=8192, `PLOW_SEQ_PAR_SEAMS=1`)

Full K3 packet, `plowc --hf-dir /home/shaswot/models/Kimi-K3 --emit devblob --arch gfx950 --gpu
mi350 --num-gpus 8 --parallel tp --batch 1 --seq 8192 --max-ctx 16384 --n-cu 256` (K3_FULL
default on), control vs candidate at `/tmp/seqpar-emit/{control,candidate}`; instruction-level
counts from `plowrt disasm --program 8192 --format json` (static, no GPU):

| | control | candidate |
|---|---:|---:|
| insts (T=8192 program) | 2971 | 3158 (+187: one extra packet per seam) |
| segments (T=8192) | 1067 | 1067 (unchanged → no new drains) |
| `XReduceTwoShot` / `XReduceScatter` / `XAllGather` | 278 / 0 / 0 | 92 / 186 / 187 |
| RS with band gather fold (`gslot`=slot C, `gcols`=896) | — | 92 |
| RS with local band copy (snapshot layers) | — | 8 |
| 3-way `XAllGather` (h2 112 MiB + xe 56 MiB + route 1 MiB, slots 3/4/5) | — | 92 |
| `AttnRes` rows | 187 × 8192 | 187 × 1024 |
| router `GemmWide` / xe `GemmC5` M | 8192 (full) | 1024 (band of `act.h2_tp`), same families |
| `MoeRouterTopkPf` T | 8192 | 1024 |
| latent `RmsNorm` / up-proj `GemmWide` M | 8192 | 8192 (unchanged, column-sharded) |
| xctr gates | 556 | 557 (all distinct) |
| all-gather bytes written locally per prefill | 25,984 MiB | 31,340 MiB (+5,356 = 92×57 + 112) |
| tensor table | +0 | +3 peer slots, +15 band views per bucket |
| manifest `requires` | … | + `PLOW_SEQ_PAR_SEAMS=1` |
| model.pkt | 217.1 MB | 224.9 MB |

Emitted seam-1 sequence (layer 1): `GemmC5(o_proj) → XReduceScatter(slot A) → AttnRes(band) →
GemmWide(router, band) → GemmC5(xe, band) → MoeRouterTopkPf(band) → XAllGather(h2, xe, route)
→ MoeAlignPf → MoeGroupGluPf …`. Decode program: byte-identical (2165 insts, 233 segments).
Note: at the 512/1024 buckets the band is 64/128 rows and the f32-mix AttnRes phase-object
route is not taken (`norm_residual` segments 187 → 0 there); the 8192 bucket keeps all 187.
`cargo test -p devgen` passes (299 lib tests; the `tuned_tile_selection`
`gfx942_mi325x_mxfp4_measurements_reach_the_compiler` failure is a pre-existing gfx942 tunedb
record issue, untouched here); `cargo test -p packet` 102 passed.
