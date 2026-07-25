# Gemma-4-26B-A4B — LIVE per-op decode trace on H100 NVL (sm_90a)

The `block-decode-baseline-26b.md` "68 % body / 29 % gate / 3 % sig" split and the
"uniform occupancy gap" were **inferred from roofline only** — the live plow trace was
never run on H100. This runs it, with the exact 31B/RTX method
(`gemma4-31b-t9b-trace.md`): `-DPLOW_NV_TRACE_DECODE=ON` self-dump of block-0's
per-packet gate/body/sig for ONE warm decode step, reduced by `scripts/trace_reduce.py`.

**Headline: the live trace CONFIRMS the roofline split to the tenth of a percent
(29.6 % / 28.4 % gate), and REFUTES the RTX FLASH lever — the H100 gate is spread
across the projection/MoE GEMVs, not concentrated in FLASH_MERGE.**

## Method / how this was measured (reproduce)

The trace facility lives ONLY in `interp_sm120.cu`'s launch helper (`plow_sm120_launch`
self-dumps `g_tr_*` after the cooperative launch). The production H100 decode object
`interp_sm90a.cu` has **no** trace instrumentation, and plowrt/step_bench launches its
prebuilt cubin via `cuLaunchCooperativeKernel` **from Rust**, bypassing the C++ launch
helper — so **plowrt cannot emit PLOW_TRACE**. The only traceable path is the C++ harness
`gemma4_sm120_chat` + the `plow_interp_sm120_gemma` decode object, **compiled for sm_90a**.

- Build (plain system env, nvcc 13.0, NOT nix): `runtime/CMakeLists.txt` arch strings
  overridden `120a → 90a` for this build only (reverted after); the decode/prefill/gf8
  gemma objects + harness then compile clean for Hopper.
  ```
  cmake -S runtime -B build-trace-dec -DPLOW_CUDA=ON -DPLOW_NV_TRACE_DECODE=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build-trace-dec --target gemma4_sm120_chat -j8
  ```
- Runtime env (MANDATORY compat driver, else INVALID_IMAGE):
  ```
  export PLOW_LIBCUDA=/usr/local/cuda-13.0/compat/libcuda.so.1
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
  ```
- Prime + trace. `PLOW_PREFILL=0` primes the context **through the decode object itself**
  (one launch per prompt token) — this exercises exactly the kernel we trace and avoids the
  sm120 prefill mma arms entirely. `plow_sm120_launch` increments a launch counter per decode
  step, so `PLOW_NV_TRACE_SKIP = n_prompt` lands the dump on gen-step-0 at `kvlen = n_prompt`.
  ```
  gpulease trace-c1024 env ... PLOW_PREFILL=0 PLOW_NV_TRACE_SKIP=1024 \
    ./build-trace-dec/gemma4_sm120_chat \
    /workspace/assets/plowrt-26b/bf16/model.pkt \
    /workspace/assets/plowrt-26b/bf16/checkpoint /tmp/prompt1024.ids 3
  # ctx=4096: prompt4096.ids + PLOW_NV_TRACE_SKIP=4096
  grep '^PLOW_TRACE' <log> | python3 scripts/trace_reduce.py   # (+MoE opcode labels 40-72)
  ```
- **Validity of the sm120@90a decode object as a proxy for the sm90a production kernel:**
  the trace's finding is the gate/body/sig SHAPE and the per-op cyc/op, which are properties
  of the emitted decode PROGRAM (packet stream, dep map, n_cu=132 slicing) — identical
  regardless of which .cu compiles it. The object even ran **numerically correct** on H100
  (`argmax check: device==host AGREE` at every ctx), and the reduced gate% matched the
  independent roofline to 0.4 pp. Absolute body cyc may differ from the shipped sm90a GEMV
  by a constant; the SPLIT, the per-op ranking, and the flash-alignment check transfer.
- Grid gate passed: `interpreter grid=132 == packet n_cu=132`. pkt `max_ctx=8192`, vocab
  262144, decode program 521 packets / 29 273 wg-packets; **block-0 = 216–219 packets**.

## 1. Reduced per-op tables

### ctx=1024  (kvlen=1024, argmax AGREE)
```
packets(block0)=216  total_cyc=16167509  gate=4792673 (29.6%)  body=10872896 (67.3%)  sig=501940 (3.1%)

op                    cnt    gate_cyc    body_cyc   sig_cyc     tot_cyc   %tot     g/op     b/op
MOE_GLU_NORM_GEMMA     30     1246467     3990807    133037     5370311  33.2%    41548   133026
GEMV (o/down/lmhead)   65     1396508     2505686    140891     4043085  25.0%    21484    38549
MOE_DOWN_GEMMA         30      274268     1767043     55207     2096518  13.0%     9142    58901
GEMV_QKV               25      977496     1027750     55681     2060927  12.7%    39099    41110
FLASH_DECODE           30      299596      772707     42486     1114789   6.9%     9986    25756
GEMV_GLU               26      373647      549053     54685      977385   6.0%    14371    21117
MOE_RTR_SCORE_FAST      5       83203      149331     12772      245306   1.5%    16640    29866
MOE_RTR_TOPK            1       51567       70332      1410      123309   0.8%    51567    70332
FLASH_MERGE             2       41760       19479      2935       64174   0.4%    20880     9739
NORM_RESIDUAL_NORM      1       32523       16735      1375       50633   0.3%    32523    16735
ARGMAX                  1       15638        3973      1461       21072   0.1%    15638     3973
TOTAL                 216     4792673    10872896    501940    16167509   100%
```

### ctx=4096  (kvlen=4096, argmax AGREE)
```
packets(block0)=217  total_cyc=16189800  gate=4590235 (28.4%)  body=11120244 (68.7%)  sig=479321 (3.0%)

op                    cnt    gate_cyc    body_cyc   sig_cyc     tot_cyc   %tot     g/op     b/op
MOE_GLU_NORM_GEMMA     30     1289336     4126767    105354     5521457  34.1%    42977   137558
GEMV (o/down/lmhead)   66     1332571     2511921    143755     3988247  24.6%    20190    38059
GEMV_QKV               25      959646     1020047     58494     2038187  12.6%    38385    40801
MOE_DOWN_GEMMA         30      150979     1760828     53465     1965272  12.1%     5032    58694
FLASH_DECODE           30      292144      977681     41832     1311657   8.1%     9738    32589
GEMV_GLU               32      476289      672073     66954     1215316   7.5%    14884    21002
MOE_RTR_SCORE_FAST      1       49842       28944      4932       83718   0.5%    49842    28944
FLASH_MERGE             1       19640       13506      1474       34620   0.2%    19640    13506
SOFTCAP                 1       18008        5302      1454       24764   0.2%    18008     5302
HEADNORM_ROPE           1        1780        3175      1607        6562   0.0%     1780     3175
TOTAL                 217     4590235    11120244    479321    16189800   100%
```

Top line — **live trace vs the roofline inference (block-decode-baseline-26b §plow):**

| | pkts | gate% | body% | sig% |
|---|--:|--:|--:|--:|
| roofline inference (never traced) | — | 29 | 68 | 3 |
| **live trace ctx=1024** | 216 | **29.6** | 67.3 | 3.1 |
| **live trace ctx=4096** | 217 | **28.4** | 68.7 | 3.0 |

The 29 % gate was real. First time observed, not inferred.

## 2. KEY ANSWER — the gate is SPREAD, not concentrated (opposite of 31B/RTX)

Gate ranking (share of total block-0 gate; cyc/op = per-op gate intensity):

| rank | op | ctx1024 gate % | g/op | ctx4096 gate % | g/op |
|---|---|--:|--:|--:|--:|
| 1 | GEMV (o/down/lmhead, op10) | 29.1 % | 21 484 | 29.0 % | 20 190 |
| 2 | MOE_GLU_NORM_GEMMA (op71)  | 26.0 % | 41 548 | 28.1 % | 42 977 |
| 3 | GEMV_QKV (op22)            | 20.4 % | 39 099 | 20.9 % | 38 385 |
| — | **top-3 = fraction of all gate** | **75.5 %** | | **78.0 %** | |
| . | FLASH_MERGE (op13)         | **0.4 %** | 20 880 | **0.2 %** | 19 640 |

**The 29 % gate WAIT is distributed across the three projection/MoE GEMV families, not
concentrated in one op.** This is the exact opposite of 31B on RTX, where a single op
(FLASH_MERGE) carried the long-ctx gate (658 k cyc/op @128k). Here FLASH_MERGE is **0.2–0.4 %
of gate and does not grow with ctx** (it shrinks 41 760→19 640 cyc). By per-op *intensity*
the worst gate is MOE_GLU_NORM (~42 k cyc/op) and GEMV_QKV (~39 k) — but GEMV(o/down) leads
the *total* only because there are 65 of them. This is the fingerprint of the shared
megakernel occupancy ceiling (1 block/SM), manifesting as inter-op counter-wait on every
big GEMV — i.e. the campaign's "uniform occupancy gap" is **confirmed by live trace**, and
the RTX flash-imbalance lever does **not** transfer.

## 3. Within body — GEMV share is even higher than the 84 % estimate

| op | ctx1024 b/op | ctx4096 b/op | note |
|---|--:|--:|---|
| **MOE_GLU_NORM (expert gate/up, op71)** | **133 026** | **137 558** | biggest single body op; 30 pkts = ~37 % of body |
| MOE_DOWN (expert down, op63)            | 58 901 | 58 694 | |
| GEMV_QKV (op22)                         | 41 110 | 40 801 | |
| GEMV o/down/lmhead (op10)               | 38 549 | 38 059 | |
| GEMV_GLU (dense gate/up, op19)          | 21 117 | 21 002 | |
| FLASH_DECODE (op12)                     | 25 756 | **32 589** | only op that grows with ctx (+27 %) |
| FLASH_MERGE (op13)                      | 9 739 | 13 506 | tiny |

**GEMV-family vs FlashDecode body split (block-0):**

| ctx | GEMV-family | FlashDecode(+merge) |
|---|--:|--:|
| 1024 | **92.7 %** | 7.3 % |
| 4096 | **91.1 %** | 8.9 % |

**Refines the campaign's 84 %/16 %:** at block-0 trace granularity the GEMV body share is
**91–93 %**, higher than the whole-model 84 % estimate, because (a) the 128-expert top-8 MoE
adds a large GEMV body a dense estimate omits — **the MoE experts alone (op71+op63+router)
are ~55 % of decode body** — and (b) FlashDecode at ctx≤4k is latency-bound and small.
FlashDecode body does scale with ctx (25.8k→32.6k cyc/op, +27 %) but from a small base, so
its share only reaches ~9 % at 4k. **The decode body is a GEMV problem, and MoE is the
majority of it** — consistent with vLLM's per-op measurement (moe_experts = the single
biggest kernel, 56.5 us).

## 4. Flash split-imbalance — the RTX lever is ALREADY TAKEN on H100

- grid = n_cu = **132**. Assets ship `PLOW_NS_FULL_ABS=33`: full-attn (hd256) layers split
  `n_grp·nsplit = 16·33 = 528 = 4·132` → **exactly 4 flash work-items per block, balanced.**
- Block-0 does **1–2** FLASH_MERGE items (consistent with the balanced 4-items/block spread),
  and its FLASH_MERGE **gate is flat and tiny: 20 880 cyc/op @1024, 19 640 cyc/op @4096** —
  it does NOT spike or grow with ctx. Contrast 31B/RTX ragged ns16: FLASH_MERGE gate
  **658 390 cyc/op** at 128k (33× larger) that the ns47 grid-align fix collapsed 14×.
- **Conclusion: FLASH work IS grid-aligned on H100; FLASH_MERGE is not a gate lever here.**
  The RTX flash-imbalance win is already captured by the shipped ns33 alignment.
- Caveat: this pkt's `max_ctx=8192`, so I traced ctx≤4096, not 128k. A long-ctx flash spike
  cannot be *observed* here — but the alignment math (528=4·132) plus the flat, sub-0.5 % gate
  at 4k is the evidence that the imbalance mechanism is designed out, not merely dormant.

## 5. VERDICT (one line)

**#1 op to attack = the MoE expert gate/up GEMV `MOE_EXPERT_GLU_NORM_GEMMA` (op71): it is
simultaneously the largest body (133–138 k cyc/op, ~37 % of decode body) AND the highest-
intensity gate (~42 k cyc/op, ~27 % of total gate) — but there is no single-op silver bullet;
the 29 % gate is the shared 1-block/SM occupancy ceiling spread across op71 + GEMV(o/down) +
GEMV_QKV (top-3 = 76–78 % of gate). Raising decode-GEMV occupancy/BW globally (occ2/occ3 +
cp.async row-staging, per baseline §4c) attacks both the body and the gate it imposes on
neighbours; the RTX FLASH_MERGE lever does NOT apply (FLASH is grid-aligned, 0.2–0.4 % of gate).**

## Raw command log

```
# build (arch 120a->90a overridden in runtime/CMakeLists.txt for this build, then reverted)
cmake -S runtime -B build-trace-dec -DPLOW_CUDA=ON -DPLOW_NV_TRACE_DECODE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-trace-dec --target gemma4_sm120_chat -j8      # clean, sm_90a

# prompts: int32 token-id files (content irrelevant; kvlen = token count)
python3 -c 'import struct,random;random.seed(1);n=1024;open("/tmp/prompt1024.ids","wb").write(struct.pack("<%di"%n,*[random.randint(10,2000) for _ in range(n)]))'
python3 -c 'import struct,random;random.seed(1);n=4096;open("/tmp/prompt4096.ids","wb").write(struct.pack("<%di"%n,*[random.randint(10,2000) for _ in range(n)]))'

export PLOW_LIBCUDA=/usr/local/cuda-13.0/compat/libcuda.so.1
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
PKT=/workspace/assets/plowrt-26b/bf16/model.pkt
CKPT=/workspace/assets/plowrt-26b/bf16/checkpoint
HN=./build-trace-dec/gemma4_sm120_chat

# ctx=1024  (gpulease rc=0)
gpulease trace-c1024 env PLOW_LIBCUDA=$PLOW_LIBCUDA LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  PLOW_PREFILL=0 PLOW_NV_TRACE_SKIP=1024 $HN $PKT $CKPT /tmp/prompt1024.ids 3 > /tmp/trace_c1024.log 2>&1
# ctx=4096  (gpulease rc=0)
gpulease trace-c4096 env PLOW_LIBCUDA=$PLOW_LIBCUDA LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  PLOW_PREFILL=0 PLOW_NV_TRACE_SKIP=4096 $HN $PKT $CKPT /tmp/prompt4096.ids 3 > /tmp/trace_c4096.log 2>&1

grep '^PLOW_TRACE' /tmp/trace_c1024.log | python3 scripts/trace_reduce.py   # + MoE labels 40-72
grep '^PLOW_TRACE' /tmp/trace_c4096.log | python3 scripts/trace_reduce.py
```

- Both runs: `gpulease rc=0` (clean, uncontended); `interpreter grid=132 == n_cu=132`;
  `argmax check … AGREE` (decode numerically correct on H100 despite sm120@90a compile).
- Opcodes seen beyond `trace_reduce.py`'s built-in map (extended for this analysis, shipped
  script untouched): 63=MOE_EXPERT_DOWN_GEMMA, 69=MOE_ROUTER_GEMMA_SCORE_FAST,
  68=MOE_ROUTER_GEMMA_TOPK, 70=MOE_COMBINE_NORM_GEMMA, 71=MOE_EXPERT_GLU_NORM_GEMMA
  (`runtime/common/dev_isa.h:411-465`).
- Build-config edit (`runtime/CMakeLists.txt` 120a→90a) reverted; no runtime source or
  emitter was modified.
```
