# GLM-5.2 TP8 collectives on 8x MI300X: are they tuned for this silicon?

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **CDNA3 / MI300X-SPECIFIC** — XGMI topology, 7 peer links and the measured 240.5 GB/s all-gather rate are this box. The MECHANISM -- fabric limited by cross-thread request concurrency, not per-thread depth -- is AMD-general and has held on three architectures.

2026-08-08, branch `coll-tune` off `worktree-glm52-bringup` @ a5b0423. Box: 8x MI300X
(gfx942, 304 CU), ROCm 7.2.4. Model GLM-5.2-FP8, TP8, hidden 6144, 78 layers
(3 dense + 75 MoE), **2 all-reduces per layer = 156 collectives per launch**.

**Headline: yes, they are — and the one plausible-looking defect left in the reduce
body is not a defect.** Two independent hypotheses (push instead of pull; add memory
parallelism across peers) were built and measured, and both are nulls. The fabric on
this box is limited by REQUEST CONCURRENCY ACROSS THREADS, and the shipped kernel
already supplies all of it the fabric will take. The implemented arm ships default OFF
as the record of that falsification.

Probe: `perf-data/plow-gfx942/probes/xgmi_dir.hip` (new). Its pull-path numbers
reproduce `xrbw.hip` to 3%, and the whole probe reproduces itself across two separate
lock holds to ~1%.

## 1. The topology, measured

`rocm-smi --showtopo`: **all 64 ordered pairs are XGMI, 1 hop, weight 15** — a fully
connected mesh. No ring, no near/far tiers, no fabric-side NUMA structure (the Numa Node
column is host-socket affinity, not a GPU-fabric tier). So a RING all-reduce is strictly
wrong here, plow does not use one, and there is no peer ordering to prefer — which is
exactly why the shipped stagger `(wg + rank + i) % N`, whose only job is to keep all
seven links busy at once, is the right shape.

Ceilings, 12.6 MB messages (= one all-gather slice at the widest GLM prefill bucket),
304 WG x 256 thr, best of 3 timed reps:

| what | GB/s |
|---|--:|
| ONE pair active, 2 B pull (GPU0 reads GPU1) | 44.2 |
| ONE pair active, 2 B push (GPU0 writes GPU1) | 47.9 |
| ONE pair active, 16 B pull / push | 48.5 / 48.2 |
| 8 devices, all-gather shape, **2 B pull, staggered (SHIPS)** | **239.7 per rank** |
| 8 devices, all-gather shape, 2 B push, staggered | 212.6 per rank |
| 8 devices, all-gather shape, 16 B pull | 119.1 per rank |
| 8 devices, all-gather shape, 16 B push | 182.7 per rank |
| 8 devices, reduce-scatter shape, **2 B pull + f32 reduce (SHIPS)** | **289.4 per rank** |

One link tops out near 48 GB/s in either direction. With all seven live the device
sustains 239.7 GB/s on the all-gather, of which 7/8 = 209.7 GB/s is remote = **29.9 GB/s
per link, 62% of what a link does alone** — the all-links ceiling is the device's shared
fabric concurrency, not the algorithm. The reduce-scatter runs FASTER than the
all-gather (289.4) because it is a permutation, not a broadcast, and it writes only one
slice locally where the all-gather writes all eight.

### PUSH vs PULL: a null, and it retires the question

plow's two-shot is PULL-only in both phases; RCCL and vLLM's custom all-reduce are
PUSH-based, and a remote read is a request plus a response where a remote write is
posted. "The algorithm is wrong by direction" was a live hypothesis. **It is false on
this fabric.** At the link level push and pull are within 8%; at the all-to-all shape the
SHIPPED form is the fastest of the four combinations, with push 11% behind at 2 B and
still 24% behind at 16 B. The `xrbw` width result also reproduces and now has a
direction: narrow beats wide **on the pull path** (239.7 vs 119.1); the push path prefers
wide (212.6 vs 182.7) and never catches pull. **Do not convert the collective to push.**

### The fabric is request-concurrency limited, and that governs everything else

All-gather (2 B pull, shipped shape) against participating workgroups:

| WG | ms | GB/s per rank |
|--:|--:|--:|
| 304 | 0.419 | 240.5 |
| 152 | 0.710 | 141.8 |
| 76 | 1.387 | 72.6 |
| 38 | 2.754 | 36.5 |
| 19 | 5.473 | 18.4 |

**Dead linear from 19 to 304 workgroups.** The fabric rate is proportional to the number
of threads issuing remote loads; it never saturates on this machine. Three consequences,
and they are the spine of the rest of this document:

1. the collective genuinely needs the whole machine — running it on fewer CUs costs
   fabric time roughly 1:1, so `PLOW_XR_CUS` narrowing is **not free**;
2. per-thread request DEPTH is not the currency — thread COUNT is (section 3);
3. there is no bandwidth headroom to unlock by reshaping the transfer. The only way to
   spend less time in the collective is to move fewer bytes (section 4) or to overlap it
   (section 6).

## 2. What plow actually does, per call site

Opcodes (`runtime/common/dev_isa.h`): XREDUCE 24 (one-shot), XFLASHMERGE 27,
XARGMAX_FIN 28, XREDUCE2 29 (two-shot), XREDUCE_ADD_NORM 116 (fused AR+residual+norm).

| # | call site | phase | algorithm | op | msg n | remote B/rank | class |
|---|---|---|---|--:|--:|--:|---|
| 1 | attn seam (o_proj partial -> `og_tp`, slot 0) | prefill | two-shot RS+AG, staggered AG | 29 | T*6144 | 3.5n | bandwidth |
| 2 | MoE / dense-FFN seam (`MoeCombinePf`/down -> `dg_tp`, slot_b) | prefill | two-shot RS+AG, staggered AG | 29 | T*6144 | 3.5n | bandwidth |
| 3 | attn seam (`GLM_FUSE_XRN=1`) | decode | one-shot **fused with residual+RMSNorm**, 16 B peer loads | 116 | 6144 | 86 kB | latency |
| 3b | attn seam (fusion off) | decode | one-shot | 24 | 6144 | 86 kB | latency |
| 4 | MoE / dense-FFN seam | decode | one-shot | 24 | 6144 | 86 kB | latency |
| 5 | lm_head argmax fold | decode | publish 8 B + N-way max | 28 | 8 B/seq | 56 B | latency |
| 6 | XFLASHMERGE | — | **dead opcode** | 27 | — | — | — |

Per collective per rank (bf16, N=8; one-shot = (N-1)*2n, two-shot = 3.5n):

| T | n | partial | one-shot remote | two-shot remote |
|--:|--:|--:|--:|--:|
| 1 (decode) | 6,144 | 12.3 kB | 86.0 kB | 21.5 kB |
| 128 | 786,432 | 1.57 MB | 11.0 MB | 2.75 MB |
| 512 | 3,145,728 | 6.29 MB | 44.0 MB | 11.0 MB |
| 1024 | 6,291,456 | 12.6 MB | 88.1 MB | 22.0 MB |
| 2048 | 12,582,912 | 25.2 MB | 176.2 MB | 44.0 MB |
| 4096 | 25,165,824 | 50.3 MB | 352.3 MB | 88.1 MB |
| 8192 | 50,331,648 | 100.7 MB | 704.6 MB | 176.2 MB |

Two structural facts fall out.

**(a) There is only ONE prefill shape.** The attention seam and the MoE/dense-FFN seam
are both `[T, 6144]` bf16 into the same peer window — they cannot want different
algorithms because they are the same message. The only genuinely different shapes are
decode's `[1, 6144]` and XArgmaxFin's 8 bytes, and both are already on the latency-side
algorithm. The "different call sites want different algorithms" hypothesis has nowhere
to land beyond the prefill/decode split the emitter already makes.

**(b) XFLASHMERGE (27) is dead on every path.** `interp.hip` dispatches it to an empty
body marked STUB, and no emitter constructs `DevOp::XFlashMerge` — a whole-tree grep
finds only the enum, the name table and the slots-spec row. Reserved opcode for a CP
flash LSE merge that never landed. Worth knowing before someone measures it.

**(c) The two size regimes want OPPOSITE load widths, and plow already has both.** 2 B
scalar is right for the large prefill message (239.7 vs 119.1 GB/s); 16 B vector is right
for decode's 6144-element message, and op 116 already uses it for exactly that reason
("nranks * XRN_VEC independent issues per thread"). These are opposite answers to the
same question at different sizes, and both are in the tree.

### The one place the selection predicate is arguably wrong

`emit_xreduce` picks the algorithm from the `decode` FLAG, not from message size. The
two-shot saves `(N-1)*2n - 3.5n = 10.5n` bytes of remote traffic and pays ONE extra
rendezvous, so break-even is

    n* = R_extra * 209.7e9 / 10.5  =>  T* ~ 170 rows  (R_extra = 51.8 us, gate_ag as built)
                                   =>  T* ~  27 rows  (R_extra =  8.2 us, a 1-signaller gate)

Decode (T=1) is two orders below either and T>=512 is two orders above: **the split is
right for every bucket the benches touch.** The exception is the `T = 128` prefill
bucket, which lands inside the crossover band and, on the as-built gate, on the wrong
side of it (one-shot ~61 us vs two-shot ~74 us of modelled cost). T=128 only runs for
short prompts and ragged chunk tails, so the prize is small; recorded, not spent. Note
that fixing gate_ag (section 6) moves the crossover to T~27 and dissolves the question.

## 3. The implemented arm: `PLOW_XR_MLP` — NULL, and a useful one

### The finding (ISA, not inference)

`d_xreduce` (every decode MoE/dense-FFN seam) and `d_xreduce_twoshot_mega`'s PHASE 1
(the reduce-scatter, half of every prefill collective's remote bytes) both walk peers as

```c
for (uint32_t r = 0; r < nranks; r++) {
    const bf16* part = (const bf16*)((const char*)peer_scratch[r] + slot_bytes);
    acc += bf2f(as_glob(part)[e]);
}
```

`nranks` is a runtime value so LLVM cannot unroll, and `peer_scratch` is a pointer table
in memory. `llvm-objdump -d --mcpu=gfx942` of a probe TU gives, **per element per peer**:

```
global_load_dwordx2 v[4:5], v1, s[14:15]   ; re-read peer_scratch[r] from memory
s_waitcnt vmcnt(0)                          ; stall on the POINTER
v_lshl_add_u64 ...                          ; + slot_bytes, + e*2
global_load_ushort v4, v[4:5], off          ; the remote bf16
s_waitcnt vmcnt(0)                          ; stall on the DATA
v_add_f32_e32 v3, v3, v4
```

Two full memory stalls per peer, the pointer table re-read eight times per element, and
**exactly one outstanding remote read per thread** — the eight links walked one round
trip after another. It reads like the all-gather's lockstep-peer bug one level down, and
the stagger note's claim that "the RS phase reads all 8 peers per element, so its links
were already concurrent" was read off the source, not the ISA. It is also a shape the
tree had already fixed *locally* in `d_xreduce_add_norm_mega` ("the MLP the scalar form
lacked") without ever fixing the scalar form.

### The arm

`PLOW_XR_MLP=1` — build axis on the DECODE and PREFILL objects (`scripts/build_gfx942.sh`),
default OFF, unset = objects code-identical to a build from before the axis existed
(verified: disassembly of the default-off header equals HEAD~1's). On `nranks == 8` only
(anything else takes the untouched loop) it hoists the eight peer bases into registers
once per thread and issues all eight remote loads before consuming any.

**Numerics class: BIT-IDENTICAL by construction** — same `r = 0..N-1` f32 accumulate,
same element->thread map, same 2 B scalar load width.

ISA audit of the arm (`-DPLOW_XR_MLP=1`): eight back-to-back `global_load_ushort` with
staged `s_waitcnt vmcnt(7)`, `vmcnt(6)`, ... consumption; no scratch; the generic loop
preserved for `nranks != 8`; the SHUFFLE probe axis produces identical text with and
without the arm. Object cost, canonical recipe (`PLOW_OCC4=1 PLOW_L2HIER=1
PLOW_MLA_PF_SV=1`):

| object | vgpr | vgpr spill | scratch | .text |
|---|--:|--:|--:|--:|
| `interp_decode_fp8_gq` ctl / MLP | 108 / 108 | 0 / 0 | 988 / 988 | 773,024 / 774,176 |
| `interp_prefill_fp8_mla_moe_gq` ctl / MLP | 256 / 256 | 2 / 2 | 1292 / 1292 | 511,072 / 512,288 |

Occupancy unchanged on both, no new scratch, +1.2 KB `.text`. Flash objects byte-identical
(`op_collective.h` is excluded from `PLOW_BUCKET_FLASH`).

### Measured: NULL, trending slightly negative

Microbench first, and it is the explanation. `xgmi_dir` [C], reduce-scatter at the real
GLM shape, 8 devices concurrent:

| reduce-scatter variant | GB/s per rank |
|---|--:|
| 2 B pull, **serial peers (SHIPS)** | **289.4** |
| 2 B pull, batched peers (`PLOW_XR_MLP`) | 181.7 |

*(A first cut of this probe reported 175.6 vs 179.4 and was WRONG: its "serial" control
passed `NR` as a compile-time constant, so LLVM unrolled it and the probe compared two
batched forms. The control now takes a runtime `nr`, exactly as the kernel takes
`nranks`, and emits the shipped serialised loop. The corrected control is the reason this
arm was measured on the GPU rather than shipped on the strength of an ISA diff.)*

Served, `scripts/bench_speed.sh`, IN_LENS 1024/4096/8192, conc 1, 4 prompts, 64 out
tokens, PLOW_MLA_PF_V2=1, **2 interleaved rounds ctl/mlp/ctl/mlp in one lock hold**,
TTFT median ms (per-round medians; the two 5xxx *means* in the raw log are single
first-request cold-start outliers that the median rejects):

| in_tok | ctl r1 | ctl r2 | ctl spread | mlp r1 | mlp r2 | ctl med | mlp med | delta |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 348.3 | 352.7 | 1.3% | 354.4 | 347.4 | 350.5 | 350.9 | **+0.1%** |
| 4096 | 1014.0 | 1006.7 | 0.7% | 1029.4 | 1013.5 | 1010.4 | 1021.5 | **+1.1%** |
| 8192 | 1738.5 | 1719.9 | 1.1% | 1761.3 | 1753.0 | 1729.2 | 1757.2 | **+1.6%** |

TPOT 28.4–33.5 across all eight cells with no arm-correlated ordering — within noise.
Every arm passed its own Paris gate, and the extended gate (below) is character-identical.

**Verdict: NULL, and if anything a small regression at the large buckets.** Default
stays OFF. Do not re-derive: adding per-thread request depth on this fabric does not add
concurrency — the [E] scaling says concurrency comes from THREAD COUNT and the machine
already supplies all of it. This is the **third independent** confirmation of the same
rule, after `xrbw`'s "wider loads lose" and `xrbw`'s "2 B x4 unroll loses". The
serialised, scalar, depth-1 peer walk is not a defect; **it is the right shape for this
fabric**, and it now has an ISA-level explanation on record instead of a source-level
assumption.

### Numerics gate (both arms, one lock hold, character-diffed)

Served from `plowrt serve` on each arm and diffed byte-for-byte:

* "capital of France" -> `The capital of France is Paris.`
* "17 multiplied by 23" -> `391`
* "chemical symbol for gold + one use" -> `**Au**` ... `**jewelry**` ...
* a ~250-word real-text passage + "summarize in three sentences, then explain why a ring
  all-reduce would be wrong on this topology" -> a 300-token coherent answer that
  correctly identifies the mesh and the five idle links per device.

`diff ctl mlp` => **CHARACTER-IDENTICAL on all four**, which is what a bit-identical
change must produce and is the real gate — a Paris prompt alone would not have exercised
156 collectives x many chunks the way the 300-token generation does.

## 4. QuickReduce-style quantized all-reduce: NO, and the arithmetic says why

The recorded ">1.2x TTFT" claim is against a **one-shot / ring** baseline. plow already
banked the 4x fabric reduction the two-shot gives (14n -> 3.5n remote bytes), so the
headroom a quantizer can still address here is 4x smaller than the headroom that claim
was measured over. At T=8192:

* two-shot remote bytes/collective/rank = 176.2 MB; at the measured rates the two phases
  cost 0.55 + 0.42 = 0.97 ms of pure transfer against a ~1.2 ms measured in-kernel span
  — i.e. the shipped collective is already at ~81% of its own two-phase microbench floor;
* fp8 halves ONLY the transfer term: best case ~0.49 ms/collective, x156 = **-76 ms** on
  a ~1729 ms TTFT @8k = **-4.4%, an upper bound** that assumes quantize/dequantize are
  free (they are VALU, on a path this tree has repeatedly measured as issue-bound);
* decode gets **nothing** — 86 kB against an 8 us rendezvous is not a bandwidth problem.

Against that: activations are bf16 (the weights are fp8, the activations are not), and
the all-reduced tensor is a PRE-RESIDUAL activation summed 156 times per prefill and 156
times per decoded token. fp8 e4m3 carries 3 mantissa bits — relative precision 2^-4 =
6.25% against bf16's 2^-8 = 0.39%, a **16x coarser** all-reduce at every seam. The tree
already has the calibration point: `PLOW_MOE_PF_PART16` (f32 -> bf16 on the MoE part
scatter — a far gentler change on a less exposed tensor) **flipped top-1 and was declared
unshippable**. A quantized AR needs per-group scales plus error feedback to have a
chance: a multi-day build for an upper bound of -4.4% TTFT and 0% TPOT, gated on a full
quality suite rather than a Paris prompt.

**Verdict: do not build.** If the transfer term ever has to shrink, the cheaper lever is
overlap (section 6) — same bytes, hidden rather than halved.

## 5. aiter's fused `allreduce_rmsnorm_N8192`: installed, inspected, NOT adoptable — and its shape is already in plow

aiter IS on this box: `/usr/local/lib/python3.12/dist-packages/aiter` (0.1.0), with
`aiter_meta/hsa/{all_reduce, allreduce_rmsnorm_N8192, allreduce_rmsnorm_qnt_N8192,
allreduce_layernorm_N8192}.co`, all **gfx942** (ELF flags 0x54c, verified). Reading the
host wrapper (`jit/build/aiter_/build/srcs/asm_communication.hip`):

1. **`all_reduce.co` is the same algorithm plow ships.** `stride_GPU = input_size /
   world_size`, with the source comment "gpu0 focus on 0~15; gpu1 focus on 16~31" — a
   reduce-scatter over a per-rank contiguous slice then a gather, on a fixed gdx=64 grid.
   An asm implementation of plow's two-shot, not a better algorithm. (And gdx=64 against
   the [E] scaling curve is 64/304 of the machine's fabric concurrency.)
2. **`allreduce_rmsnorm_N8192` does not fit GLM.** Grid is `gdx = M / world_size`, so it
   needs M >= 8 rows and is **structurally unusable at decode (M=1)** — the one place the
   fusion would be worth most. And it is specialised to N = 8192; GLM-5.2's hidden is
   **6144**, and there is no 6144 variant in the shipped `.co` set.
3. **The fusion it offers is already in plow, twice.** `PLOW_DOP_XREDUCE_ADD_NORM` (op
   116) is AR + residual + RMSNorm as one single-workgroup decode packet, already using
   16 B vector peer loads for the same reason aiter's kernel does.
   `PLOW_GLM_XR_RES` is the prefill half: the two-shot's all-gather writes
   `out2 = resid + reduced` and the Residual packet is not emitted at all.
4. Integration is blocked for the reason `glm52-experiments.md (consolidated; aiter/Tensile ceilings also in gemv-mlp-and-tensile.md)` recorded and
   this audit re-confirms: host-orchestrated kernels with their own workspace ABI and
   vLLM `CustomAllreduce` buffer registration. Linking one into the persistent megakernel
   breaks the counter-dep model that is plow's edge.

**Verdict: nothing to import, nothing left to copy — the shape was already copied.**

## 6. Ranked remaining opportunity

1. **Overlap — stop the collective owning the machine.** This is the only lever left
   whose prize is tens of percent, and [E] now prices its cost precisely: the fabric rate
   is LINEAR in participating workgroups, so moving a collective onto k of 304 CUs
   multiplies its transfer time by 304/k. Overlap therefore only pays when the freed CUs
   do more work than the collective loses — which is exactly why banded seams
   (`PLOW_GLM_XR_BAND=K`) were net negative at +3..+8% with every one of the 304
   GlobalQueue claimants piling into each band's rendezvous, and why the composed
   experiment (`PLOW_GLM_XR_BAND=K` + a CU subset) has to be measured rather than
   assumed. A separate run (`band-pipeline`) is mid-flight on exactly that sweep.
   **Blocker it must clear first: `PLOW_XR_CUS` does not reach prefill (section 7).**
2. **gate_ag's 304 signallers.** Measured: a 1-signaller N-way gate round costs 8.2 us;
   the 304-signaller gate the two-shot's second rendezvous actually uses costs
   **51.8 us** — 6.3x, from 2432 remote system-scope RMWs per rank per collective.
   x156 = ~8 ms/launch (~2.3% of TTFT @1k, ~0.5% @8k). Reducible with NO emitter or blob
   change: `PLOW_CTR_STRIDE` is 32 words and only word 0 is used, so word 1 of the same
   counter line is a free device-local aggregation counter. Each workgroup keeps its own
   system-scope RELEASE fence (load-bearing — it is what pushes its PHASE 1 stores past
   the XCD L2), then does an AGENT-scope RMW on word 1; the workgroup that closes the
   count issues the 8 remote signals and the threshold drops from `nranks*nblk` to
   `nranks`. Bit-identical. Not built here: alone the prize sits inside this box's
   +/-20% DVFS noise, so it should ride with opportunity 1. It also moves the
   one-shot/two-shot crossover from T~170 to T~27 and dissolves opportunity 4.
3. **Push-based reduce-scatter — DEMOTED by measurement.** [C] gives 2 B push scatter at
   212.9 GB/s against the shipped pull+reduce at 289.4 GB/s, so the pull form the tree
   ships is 36% FASTER even before counting the staging window a push RS would need
   (+(N-1)/N of a partial in the peer window, ~+44% at two slots) and the extra local
   reduce pass. Do not build. (The all-gather must stay pull for the separate reason in
   section 1.)
4. **The T=128 bucket's algorithm choice** (section 2) — small, and subsumed by 2.
5. **Quantized AR** (section 4) — upper bound -4.4% TTFT, 0% TPOT, numerics-breaking.
   Ranked last on purpose.

## 7. Latent defect found on the way (not fixed here)

`PLOW_XR_CUS` — the knob opportunity 1 needs — **never reaches the prefill program**.
`crates/devgen/src/mla.rs` reads `emit_config::active().xr_cus` once, at line 5009,
inside the DECODE program builder; the prefill builder hardcodes

```rust
let pxr: Vec<u32> = pall.clone();     // mla.rs:4936
```

so every prefill two-shot claims all 304 CUs regardless of the environment. Setting
`PLOW_XR_CUS=k` today silently narrows decode only — a knob that looks set and is not.
Same omission class the file records twice already ("the emitter never got it, exactly as
it never got `emit_xreduce`'s sizing"). One line to fix; left alone here because the
`band-pipeline` agent is mid-flight on the experiment that consumes it and two agents
editing the same three lines is how a merge conflict eats an afternoon.

## 8. Do not re-derive (measured here)

* **push instead of pull, all-gather** — 212.6 vs 239.7 GB/s at 2 B, 182.7 vs 239.7 at 16 B.
* **push instead of pull, reduce-scatter** — 212.9 vs 289.4 GB/s.
* **wider loads on the pull path** — 119.1 GB/s at 16 B vs 239.7 at 2 B (reproduces `xrbw`).
* **per-thread peer batching in the reduce** (`PLOW_XR_MLP`) — 181.7 vs 289.4 GB/s in the
  microbench, +1.1%/+1.6% TTFT @4k/8k served. Concurrency is thread count, not depth.
* **a ring** — every pair is 1 hop.
* **aiter's collectives** — same algorithm, wrong N, wrong M, incompatible ABI.
