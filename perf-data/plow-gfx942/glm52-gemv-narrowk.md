# GLM-5.2 gfx942 — narrow-K lane groups for the bf16 decode GEMV (`PLOW_GEMV_LG`)

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **AMD-GENERAL** — lane-group maps for narrow K assume wave64, which is CDNA-wide. The -1.61% is gfx942.

Branch `gemv-narrowk` off `worktree-glm52-bringup` @ 7ccd601. Worktree
`/app/plow/.claude/worktrees/gemv-narrowk`. Box: 8x MI300X, gfx942, ROCm 7.2.4.

The bf16 twin of `PLOW_MOE_DEC_LG`, one kernel over. `glm52-packet-protocol-xcd.md` §6 named
this as the next lever after LG: with LG in, the largest `busy` row in the GLM decode program is
`Gemv` at ~11.3k CU-us/layer, and the shared-expert down projection runs it at N=6144, K=256 —
the same narrow-K lane defect LG had just fixed in the fp8 MoE DOWN body.

---

## 0. WHICH OBJECT — the mandatory first step

`Variant::detect` (`crates/plowrt/src/exec/amd.rs:276`) scans for `DevOp::GemvFp8` and the three
fp8-KV flash ops. GLM-5.2's fp8 is BLOCK-scaled (`GemvFp8Blk`) and its MLA is bf16, so zero
opcodes match, the blob detects as `Variant::Bf16`, and decode runs on **`interp_decode_gq.elf`**,
not `interp_decode_fp8_gq.elf`.

Verified EMPIRICALLY, not by reading the code — `plowrt` rebuilt from this worktree so the
`code object` INFO line (added by the protocol-xcd landing) is present, then run against the real
GLM blob:

```
$ PLOW_MLA_PF_V2=1 PLOW_L2_PLACE_DISPATCH=1 plowrt amd-bench \
    --blob /workspace/assets/gfx942/glm52-tp8-final2/model.pkt \
    --hsaco /root/.claude/jobs/b09a4bcc/tmp/hsaco_glm18 --tp 8 --ctx 1024 --steps 8
INFO plowrt::exec::amd: code object object=interp_decode_gq.elf phase=Decode variant=Bf16 \
                        prefill_arm=MlaMoe sched=GlobalQueue
INFO plowrt::exec::amd: code object object=interp_prefill_mla_moe_gq.elf phase=Prefill ...
INFO plowrt::exec::amd: code object object=interp_flash_gq.elf phase=Flash ...
```

All 8 ranks log `interp_decode_gq.elf` for `phase=Decode`. Every ISA number below is taken from
that object, disassembled with
`/opt/rocm-7.2.4/llvm/bin/llvm-objdump --disassemble --mcpu=gfx942`.

`gemv_rows` is NOT a `.symtab` symbol — unlike `d_gemv_glu` / `d_gemv_qkvg` / `d_gemv_fp8_blk`,
which survive inlining as LOCAL FUNCs, `d_gemv_t<1>` is fully inlined into `_ZL9plow_exec…`. The
bodies below were located inside `plow_exec` by their signature (a 14-load `buffer_load_dwordx4`
clause with two descriptors for the control; the unique 8-`global_load_dwordx4` +
40-`ds_bpermute_b32` window for the arm — that window exists in the arm object and in NO window of
the control object).

Objects built OUTSIDE nix with the canonical recipe:

```
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1 JOBS=16 \
    PLOW_ROWS_ONLY=interp_decode bash scripts/build_gfx942.sh <outdir>          # control
env ... PLOW_GEMV_LG=1 ... bash scripts/build_gfx942.sh <outdir>                # arm
```

`PLOW_ROWS_ONLY=interp_decode` — NOT `interp_decode_fp8`. That is the row that produces
`interp_decode_gq.elf`.

**Default byte-identity.** Both a build of the un-patched tree and a build of the patched tree with
`PLOW_GEMV_LG` unset were made from the SAME directory (hipcc embeds the source path, so a
different directory manufactures a spurious mismatch). All six `interp_decode*.elf` compare
byte-identical (`cmp`). Against the canonical `hsaco_glm18/interp_decode_gq.elf` the ELF differs in
97 bytes — all inside the note/metadata section (embedded absolute source path) — and the
**disassembly is byte-identical**, so the control arm is the canonical decode object at ISA level.

---

## 1. WHICH CALL SITES ARE NARROW

`plowrt disasm <blob> --program 1` over the shipped asset (`glm52-tp8-final2`), grouped by shape.
There are exactly four `Gemv` shapes in the 2523-instruction decode program:

| site | shape | b | packets/token | nchunk = ceil(K/512) | lanes live |
|---|---|---|---|---|---|
| `self_attn.o_proj` | M=1 N=6144 K=2048 | 304 | 78 | 4 | 64/64 |
| `mlp.gate` (router) | M=1 N=256 K=6144 | 304 | 75 | 12 | 64/64 |
| **`mlp.shared_experts.down_proj`** | **M=1 N=6144 K=256** | **48** | **75** | **1** | **32/64** |
| `lm_head` | M=1 N=19360 K=6144 | 304 | 1 | 12 | 64/64 |

Traced with `PLOW_TRACE_RAW` (ctx 1024, one steady-state decode step, records grouped by `inst`,
never by `pc`), per packet:

| site | busy CU-us | span us | bytes | rate |
|---|---|---|---|---|
| router gate N=256 K=6144 | 5057.6 | 21.99 | 3.1 MB | — |
| o_proj N=6144 K=2048 | 4907.4 | 20.26 | 25.2 MB | 1.24 TB/s |
| **shared down N=6144 K=256** | **1506.3** | **35.01** | **3.1 MB** | **86 GB/s** |
| lm_head (1/token) | 22420.9 | 79.11 | — | — |

Sum of the three per-layer rows = 11471 CU-us, which is the `Gemv` 11285 CU-us/layer row of
`glm52-packet-protocol-xcd.md` §4. So `Gemv` DOES aggregate several shapes, and the shared down is
only 13% of the family's busy — but it has the **LONGEST span of any Gemv packet in a GLM MoE
layer** (35.0 us, more than o_proj's 20.3 while moving an EIGHTH of the bytes), and span is what
the serial packet chain pays.

**Shapes NOT fixed, and why.** o_proj (K=2048) and the router gate (K=6144) do NOT have this
defect: at K >= 512 the lane map `k = 8*lane` covers a full chunk and every lane is live. o_proj
already runs at 1.24 TB/s, i.e. near what 304 CUs get out of HBM; its remaining inefficiency is
`nchunk = 4 < UN = 7`, a shortened-K-loop problem the R-split already addresses and which no lane
remap can touch. The router's defect is narrow **N** (gv_per = ceil(256/304) = 1 row per
workgroup, so 7 of 8 waves idle) — a different lever, out of scope here. `lm_head` runs once a
token. The arm is guarded to `K <= (64/RG)*8 = 256`, so all three fall through to the shipped body
by construction.

---

## 2. THE BEFORE-STATE, from `interp_decode_gq.elf`

At K=256 the dispatch in `d_gemv_t` takes `gemv_rows_rs<MM=1, XLDS=true, UN=GV_RS_UN=7,
R=GV_RS_R=2>` (`nchunk = 1 < GV_RS_MAXNCH = 8`). The inlined body is at `0x767bc..0x77074`
(k-loop), `0x7671c..0x767b8` (row-loop latch) and `0x7707c..0x771dc` (two `wave_sum`s + stores).
`nchunk = 1`, so the k-loop runs EXACTLY ONCE per two output rows.

```
k-loop trip (= 2 output rows)  425 insts  342 VALU  14 vload  7 ds_read  3 full vmcnt(0)
  shape: W0(0) Lx13 r L W0(14) W6(0)x2 r W5(0)x2 r W4(0)x2 r W3(0)x2 r W2(0)x2 r W1(0)x2 r W0(0)
row-loop latch                  33 insts   18 VALU  1 vstore
epilogue (2 x 6-step wave_sum)  67 insts   25 VALU  12 ds_bpermute  1 vstore
```

Per OUTPUT ROW, at K=256:

* **263 instructions**, **193 VALU**
* **7.00 `buffer_load_dwordx4`** — of the 14 a trip issues, TWELVE are entirely past
  `num_records = 2K bytes` and fetch nothing; they occupy an issue slot and return zeros that are
  then widened and multiplied.
* **lanes used: 32 of 64.** `k = 8*lane`, so `lane >= 32` gives `k >= 256 = K` and the descriptor
  returns zero. Those 32 lanes still run all 16 VALU of widen+FMA per fragment.
* **bytes covered per load vs load width: 512 of 1024 B** (32 live lanes x 16 B) on the 2 loads
  that fetch at all; across the whole 14-load clause, 1024 useful bytes out of 14 x 1024 = **7.1%**.
* **max outstanding VMEM: 14 issued, 2 real.**
* **1.50 full `vmcnt(0)` drains per row** (3 per 2-row trip). The `W0(14)` in the shape string is a
  drain on all 14 loads before ANY is consumed — the software pipeline the unroll exists to build
  cannot form when `nchunk = 1`.
* 6 `ds_bpermute_b32` per row (the 64-lane `wave_sum`), of which the leading `off = 32` step adds
  an exact +0.0.

The defect is real, and it is worse than the fp8 one LG fixed: there the wave wasted 48 of 64
lanes but still issued one useful load; here half the lanes are dead AND 12 of every 14 loads are
no-ops.

---

## 3. THE FIX — `PLOW_GEMV_LG`, `op_gemm.h` [BF16-GEMV-NARROWK-LG]

Split the 64-lane wave into `RG` row-groups of `LPG = 64/RG` lanes. Group `g = lane/LPG` owns its
own output row; lane `sub = lane%LPG` owns `k = 8*sub`. At **RG = 2** that is LPG = 32 lanes x 8
halves = 256 = exactly GLM's contraction, so every lane is live and one `global_load_dwordx4`
retires **two whole consecutive rows** as a fully-coalesced 1024-byte fetch. `UNR = 8` row-batches
are issued before any is consumed, so `RG*UNR = 16` rows are in flight.

`UNR = 8` is not arbitrary at this shape: `gv_per = 6144/48 = 128` rows a workgroup over
`PLOW_WAVES = 8` waves is exactly 16 rows a wave, so every wave runs ONE balanced pass and there is
no ragged tail. It is also register-CHEAPER than what it replaces (8 x `bf16v8` = 32 VGPRs of
weight against the R-split's `R*UN = 14` vectors = 56).

Weight rows are read as plain 16-byte global vectors (`GV_LDW` = `ld_glob8_nt`), not through a
`buf_rsrc`: the row index is now LANE-VARYING and a descriptor from a divergent base compiles to
the readfirstlane waterfall this file documents at length. The `live` guard replaces the
descriptor's bounds check.

Guarded to `M == 1`, `K % 8 == 0` and `K <= LPG*8 = 256`. Tried BEFORE the R-split, because the
R-split cannot help at `nchunk = 1`.

### Bit-identity

* Same `(lane -> k)` fragment set per row: lane `sub` holds exactly what lane `sub` held.
* Same `dot8` seeded at `+0.0f`, same accumulation order. (The control writes
  `acc += live ? dot8(...) : 0.0f` from `acc = 0.0f`; `0.0f + x` is bitwise `x` because a `dot8`
  chain seeded at `+0.0f` can never return `-0.0` — every addend is a product added to a running
  sum that starts at `+0.0`, and IEEE round-to-nearest gives `a + (-a) = +0.0`.)
* Same xor-butterfly. The 64-lane `wave_sum` runs `off = 32,16,8,4,2,1`; the arm runs
  `off = 16,8,4,2,1`. At K=256 the dropped `off = 32` step adds lanes 32..63, whose weight
  fragments the descriptor returned as ZERO, so their `dot8` is an exact `+0.0` and `v + 0.0` is
  bitwise `v`. The remaining five steps are the identical tree over lanes 0..31 (and its mirror
  over 32..63).
* `C[n]` is written by exactly one lane in both bodies (`lane == 0` vs `sub == 0`); which wave owns
  which row moves, which is unobservable.
* Column OWNERSHIP `[gv_n0, gv_n1)` is unchanged, so `PLOW_FINE`'s gemv->headnorm dependency map is
  unaffected. Row coverage inside the block is complete and disjoint: 8 waves x 16 rows = 128 =
  `gv_per`.

### `ds_bpermute` EXEC discipline — checked in the ISA, not assumed

`__shfl_xor` lowers to `ds_bpermute_b32`, which honours EXEC on the READ side, so a lane sourcing a
masked-off lane gets nothing. Two places could have got this wrong, and the ISA shows both are
right:

1. The per-pass loop guard `n0 < gv_n1` is **wave-uniform** — in the ISA the predicate is built
   from `v55 = n0` where the wave term is `((threadIdx.x >> 6) * 16)`, so EXEC is either full or
   empty for the whole wave across every butterfly. There is no partial-lane EXEC around any
   `ds_bpermute`.
2. LLVM did turn the `live ? dot8(...) : 0.0f` select into an `s_and_saveexec_b64` region around
   each `u`'s dot — but it restores EXEC (`s_or_b64 exec, exec, s[8:9]` at `0x78c44`,
   `0x78cf8`, ...) **BEFORE** the following `ds_bpermute_b32`, and the accumulator is initialised
   (`v_mov_b32_e32 v33, 0`) outside the region. Every one of the 40 `ds_bpermute_b32` in the body
   is issued at full wave EXEC. At GLM's K=256 the predicate is uniformly true anyway (all 64 lanes
   have `k < K`), so the region is never divergent in the shipping shape; the ISA check matters for
   the guard's correctness at a hypothetical K < 256.

---

## 4. THE AFTER-STATE — same object, same tooling

The arm's body is the unique window in `plow_exec` with 8 `global_load_dwordx4` and 40
`ds_bpermute_b32` (`0x78a3c..0x79534`, latch branched out to `0xb8a3c`). No such window exists
anywhere in the control object.

```
one pass (= 16 output rows)  511 insts  297 VALU  8 vload  8 vstore  40 ds_bpermute
                             3 full vmcnt(0)  12 partial  max outstanding 9
  shape: S Lx8 W0(9) px5 W6(0)x2 px5 W5(0)x2 px5 W4(0)x2 px5 W3(0)x2 px5 W2(0)x2 px5
         W1(0)x2 px5 W0(0)x2 px5 Sx7
```

Per OUTPUT ROW at K=256, control vs arm:

| per output row | shipped `gemv_rows_rs` | `PLOW_GEMV_LG=1` | ratio |
|---|---|---|---|
| instructions | 263 | 32.0 | **8.2x** |
| VALU | 193 | 18.6 | **10.4x** |
| VMEM loads | 7.00 | 0.50 | **14x** |
| full `vmcnt(0)` drains | 1.50 | 0.19 | **8x** |
| `ds_bpermute` | 6 | 2.5 | 2.4x |
| lanes used | 32 / 64 | **64 / 64** | 2x |
| bytes covered / load width | 512 of 1024 B (7.1% of the clause) | 1024 of 1024 B (100%) | 14x |
| max outstanding VMEM | 14 issued / 2 real | 9 issued / 8 real | 4x real |

### Registers and occupancy

`scripts/build_gfx942.sh`'s own kernel-resource table, all six decode rows, both arms:

```
object                               vgpr   agpr       lds   spill
interp_decode                         108      0     30768       0
interp_decode_gq                      108      0     30776       0
interp_decode_fp8                     108      0     30768       0
interp_decode_fp8_gq                  108      0     30776       0
interp_decode_fp8kv                   108      0     30768       0
interp_decode_fp8kv_gq                108      0     30776       0
```

Identical in both arms. VGPR flat at 108 of 256, 0 AGPR, 0 spill, LDS unchanged — occupancy
unchanged (the `PLOW_OCC4` profile). The arm is register-NEGATIVE against the body it replaces
(32 vs 56 VGPRs of live weight vectors), which is why nothing moved.

---

## 5. TRACED PACKET DELTA — the kernel-level result, independent of the wall

`PLOW_TRACE_RAW`, `amd-bench --tp 8 --ctx 1024 --steps 24`, both arms on the same blob and the same
prefill/flash objects; records grouped by `inst` (NOT `pc`), median over the 75 shared-down packets
of one steady-state decode step.

| Gemv site | busy CU-us ctl -> lg | span us ctl -> lg | gate-wait CU-us |
|---|---|---|---|
| **shared down N=6144 K=256 (b=48)** | **1506.3 -> 385.8  (-74.4%)** | **35.01 -> 9.67  (-72.4%)** | 824.1 -> 815.4 |
| o_proj N=6144 K=2048 (b=304) | 4907.4 -> 4945.2 (+0.8%) | 20.26 -> 20.40 | flat |
| router gate N=256 K=6144 (b=304) | 5057.6 -> 5187.8 (+2.6%) | 21.99 -> 22.49 | flat |
| lm_head N=19360 K=6144 (1/token) | 22420.9 -> 22117.5 | 79.11 -> 76.79 | flat |
| GemvQkv (b=146 / b=149) | 2564.7 -> 2540.9 / 2209.0 -> 2209.8 | flat | flat |
| FlashMlaDecode (b=128) | 4196.7 -> 4213.9 | 35.70 -> 35.83 | flat |
| MoeExpertGluFp8Blk / DownFp8Blk | flat | flat | flat |

The targeted packet is the only row that moves outside run-to-run noise, in exactly the direction
and roughly the magnitude the ISA predicts. Gemv family busy per MoE layer:
11471 -> 10519 CU-us (-8.3%). The packet's rate goes 86 GB/s -> 313 GB/s over its 48-CU slice.

**Reconciliation with the wall.** 25.3 us of span removed x 75 layers = 1.90 ms if the packet were
fully serial, i.e. -7.1% of a 26.8 ms token. The measured wall delta is ~-1.5%. The difference is
overlap and it is expected: the shared-expert down is emitted on a 48-CU slice
(`GLM_SHARED_CUS=48`) that runs CONCURRENTLY with the 256-CU routed-expert slices, so most of its
span was already hidden. Roughly a fifth of it was not, and that fifth is what the token shows.
This is the same ceiling `glm52-packet-protocol-xcd.md` §3 describes: shortening a co-resident
slice only pays where it was the last slice to finish.

---

## 6. NUMERICS — character-identity gate

Claim is bit-identity, so the gate is CHARACTER-identity, temp 0, `max_tokens=128`, both arms
served from the same blob/checkpoint with only `hsaco/` swapped (ports 8195 ctl / 8196 lg, one
server at a time).

| # | prompt | prompt_tokens | ctl == lg |
|---|---|---|---|
| P0 | "What is the capital of France? Answer in one short sentence." | 19 | **identical** |
| P1 | "Compute 17*23 and give only the number." | 17 | **identical** |
| P2 | "Name the chemical symbol for gold and one common use." | 17 | **identical** |
| P3 | "Write two sentences about why the sky appears blue." | 16 | **identical** |
| P4 | long-context technical log + question | **14143** | **identical** |

```
P0  The capital of France is Paris.
P1  391
P2  The chemical symbol for gold is **Au**.
    One common use for gold is in **jewelry**, due to its beautiful luster, malleability, and
    resistance to tarnishing.
P3  The sky appears blue due to a phenomenon called Rayleigh scattering, where gases and particles
    in Earth's atmosphere scatter sunlight in all directions. Blue light travels as shorter,
    smaller waves, ...
P4  Every record reported that the subsystem had nominal throughput and a stable queue depth.
```

GATE: **PASS**, 5/5 byte-for-byte on the full 128-token completions. `amd-bench` additionally
reports all 8 ranks token-identical on every step of 24 for both arms.

Off-device, `(slice, wave, u, g)` row-coverage was walked at eight geometries — including the GLM
one (N=6144, nblk=48), a ragged N (6143), a ragged nblk (49), N < nblk (256/304) and N=17/nblk=3 —
and the arm's written-row set is COMPLETE, DISJOINT and EQUAL to the shipped body's at every one.
The lane->k maps of both row-groups equal the shipped body's live-lane map, and the shipped body's
lanes 32..63 are confirmed universally out of range at K=256 (which is what makes the dropped
`off=32` butterfly step an exact `+0.0`).

---

## 7. PERF — interleaved A/B, 3 rounds, `scripts/bench_speed.sh`

`IN_LENS="1024 4096 8192"`, CONCS=1, NPROMPT=8, OUTLEN=128, ctl/lg alternating, one `plowrt serve`
at a time (asserted `pgrep -x plowrt` == 0 before every arm and after every arm). TPOT ms/token:

| ctx | ctl (3 rounds) | ctl median | **ctl spread** | lg (3 rounds) | lg median | delta | ranges |
|---|---|---|---|---|---|---|---|
| 1024 | 26.78 26.80 26.77 | 26.780 | **0.112%** | 26.36 26.38 26.35 | 26.360 | **-1.57%** | DISJOINT |
| 4096 | 29.02 29.08 29.07 | 29.070 | **0.206%** | 28.69 29.01 28.68 | 28.690 | **-1.31%** | DISJOINT |
| 8192 | 29.32 29.38 29.36 | 29.360 | **0.204%** | 28.96 28.97 28.95 | 28.960 | **-1.36%** | DISJOINT |

The win is 6-14x the control's own round-to-round spread and the per-context ranges do not overlap
(lg max < ctl min at all three contexts). The one soft cell is lg/4096/round-2 (29.01, against
28.68/28.69 in the other two rounds; its reported `itl_p99` of 28.80 is BELOW its own mean TPOT,
which is internally inconsistent and points at one request with a long first inter-token gap). On
means rather than medians that context reads -0.91% instead of -1.31%; every other cell is stable
to +/-0.02 ms.

TTFT is unmoved, which is the expected sign: `PLOW_GEMV_LG` lands on `AX_DECODE` only.
ctl 343.7/973.7/1688.9 vs lg 340.7/967.1/1682.6 ms at 1k/4k/8k — inside the control spread.

A NOTE ON PROCESS. An earlier battery of this A/B was VOIDED: two copies of the driver script were
launched 89 seconds apart and both had a server up on port 8196, so either arm could have been
answered by the other's server. Every number in this section comes from a single re-run after the
duplicate was reaped, with an mkdir-based single-instance guard on the driver and an explicit
`pgrep -x plowrt` assertion (not `pgrep -f "plowrt serve"`, which false-positives on any sibling
shell whose command line merely CONTAINS that string) before and after each arm. This is the third
recorded instance of the shared-port co-tenancy failure in this campaign.

---

## 8. EXACT RECIPE

```bash
export PATH=/nix/var/nix/profiles/default/bin:/root/.nix-profile/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib
export ROCM_PATH=/opt/rocm-7.2.4 HIP_PATH=/opt/rocm-7.2.4 ROCM_HOME=/opt/rocm-7.2.4

# objects, OUTSIDE nix
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1 PLOW_GEMV_LG=1 \
    JOBS=16 bash scripts/build_gfx942.sh <outdir>
#   (PLOW_ROWS_ONLY=interp_decode is enough for a decode-only arm — it is the row that
#    produces interp_decode_gq.elf, the object a GLM blob actually loads)

# serve / bench, INSIDE nix
export PLOW_MLA_PF_V2=1                      # the blob carries the causal KV-split
IN_LENS="1024 4096 8192" bash scripts/bench_speed.sh <asset-dir> 8195 auto

# trace (amd-bench also needs PLOW_L2_PLACE_DISPATCH=1 because the blob is L2-placed)
PLOW_MLA_PF_V2=1 PLOW_L2_PLACE_DISPATCH=1 PLOW_TRACE_RAW=tr.bin \
  plowrt amd-bench --blob .../model.pkt --hsaco <objdir> --checkpoint <ckpt> --tp 8 --ctx 1024 --steps 24
```

Knobs: `PLOW_GEMV_LG_RG` (default 2 — LPG = 64/RG lanes per row, and RG > 2 makes LPG*8 < 256 so
GLM's shape stops qualifying), `PLOW_GEMV_LG_UNR` (default 8 = 16 rows in flight = exactly
`gv_per/PLOW_WAVES` at the GLM geometry).

---

## 9. VERDICT AND WHAT IS LEFT

**ADOPT as an opt-in; a default-on flip is a coordinator call.** -1.3 to -1.6% TPOT at every
context against a 0.11-0.21% control spread, disjoint ranges, character-identical on 5/5 including
14.1k tokens of context, register- and occupancy-neutral, default object byte-identical.

It is smaller than `PLOW_MOE_DEC_LG`'s -7.5% on the same defect one kernel over, and the reason is
visible in the trace rather than hypothetical: the fp8 DOWN was 11095 CU-us/layer on a 32-CU slice
per expert with eight of them, whereas this packet is 1506 CU-us/layer on ONE 48-CU slice that
already runs concurrently with the 256-CU routed-expert slices. The KERNEL improved by 74%; only
the unhidden fifth of its span reaches the token. The right reading is that the narrow-K lane
defect is now fixed everywhere it exists in the GLM decode program, and the residue is a placement
question, not a kernel one.

NOT worth reopening from here:
* o_proj (K=2048) and the router gate (K=6144) do not have this defect — every lane is already
  live. o_proj runs at 1.24 TB/s. Do not remap their lanes.
* The router's real defect is `gv_per = 1` (7 of 8 waves idle at N=256 over 304 blocks). That is
  the narrow-N lever, and `PLOW_GEMV_WG` was already falsified on Gemma; it would need a
  GLM-specific re-test.

Possibly worth a look, cheap:
* The arm still takes 3 full `vmcnt(0)` drains per 16-row pass because LLVM sinks the `u=0` load
  into the `s_and_saveexec` region it built for the `live` select and then drains on all 8. Writing
  the guard as a float multiply rather than a ternary would likely remove the region. Worth ~0.19
  drains/row, i.e. small; not done here.
* `GLM_SHARED_CUS` is 48 today. With the down projection now 3.6x faster, the shared slice's
  CU budget is worth re-sweeping — that is an emit-side axis and a different agent's lane.
