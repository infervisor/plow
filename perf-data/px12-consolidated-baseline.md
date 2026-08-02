# PX-12 — the §2b cell re-run with the campaign's corrections applied

RTX 5090 (sm_120a, 170 SM, 32 GiB / 31.36 usable, driver 580.159.03) · 2026-07-26 ·
Gemma-4-12B-it, fp8 weights (W8A8). Every GPU run under `perf-data/harness/gpulease`.
Companion to `perf-data/gemma4-12b-longctx-5090.md` (§2b is the cell), `px10-batched-decode.md`,
`px11-flash-decode.md`. Those documents are **not edited**; corrections to them are stated here.

The cell: **8 concurrent requests, 126,976 input tokens each, 1024 output tokens each**,
`vllm bench serve --backend openai-chat --dataset-name random --random-range-ratio 0
--num-prompts 8 --max-concurrency 8 --ignore-eos --seed 0`, tokenizer `/root/gemma-4-12B-it`.

**TTFT and TPOT are NOT reported.** The chat backend stamps TTFT on the first SSE chunk that
carries a `choices` array without checking `delta.content`, and plowrt emits a role-only chunk
before generation — so plow's "TTFT" is time-to-headers and its whole prefill lands in the first
ITL sample. vLLM emits its role chunk after prefill, so only vLLM's TTFT is real. Aggregate output
tok/s, wall-clock duration and **median** ITL are the only client metrics used (median is robust to
the one poisoned sample; mean ITL and TPOT are not).

## 0. Two things had to be fixed before any of this was measurable

Neither is a tuning decision; both are prerequisites for the cell being the cell.

**0a. `/root/plow-out/lc-b8` is not the §2b asset.** Its packet is an **all-layer** fp8-KV blob,
not the mixed one §2b describes. Two independent proofs:

| check | lc-b8 | all-layer fp8 | mixed (bf16 sliding) | §2b claims |
|---|---|---|---|---|
| KV per sequence | **1.333 GiB** (`kv_gib` 10.66455 / 8) | 1.3203 GiB | 1.6328 GiB | 1.68 GiB |

and, decisively, installing a PIPE=1 fp8 prefill object on it fails at an **18-token** prompt with
`CUDA_ERROR_LAUNCH_FAILED` — that is `interp_sm120.cu:772`'s `if (i[6] == 512) ... else __trap()`,
which fires only when the packet emits an **hd256** fp8 prefill op. A mixed packet never does.

So the mixed packet was re-emitted (`PLOW_FP8_KV=1 PLOW_FP8_KV_FULL=1 PLOW_NS_FULL_ABS=32
PLOW_DECODE_BATCH=8`, default window-derived chunk): **KV 13.13 GiB = 1.641 GiB/seq**, which is
§2b's 1.68. Prefill buckets `[128, 512, 1024]`, so PX-10's `≥4096` fp8-KV crash is unreachable.

**0b. `scripts/build_sm120_cubin.sh` cannot build a usable long-context fp8-KV asset.** It
hardcodes `-DPLOW_NV_FA_PIPE=0` on the fp8-KV prefill object. On the mixed packet that costs
**5.4× on prefill** (§2), and the resulting build is *not* what §2b measured — §9b's statement that
`PLOW_FP8_KV_FASTPF` "was OFF for this campaign's runs" is contradicted by §2b's own wall clock
(see §2).

## 1. Arms

All cubins built from this worktree with `PLOW_ROOT=$PWD`. The **decode** object is the
`_fp8kv` build installed as `interp_sm120.cubin` — which is exactly what the shipped asset does:
`/root/plow-out/lc-b8/interp_sm120.cubin` is **md5-identical** (`482882b3…`) to a
`PLOW_BUILD_FP8KV=1 PLOW_EXTRA_DEFINES="-DGV_MM_MAX=16 -DPLOW_NV_W8A8=1"` build of this tree.

| arm | decode object | prefill object |
|---|---|---|
| **A — control** | deployed flags: `-DGV_MM_MAX=16`, `GF_FULL` = default **2** | FASTPF (PIPE=1) |
| **B — tuned** | `GV_MM_MAX` default 8 + `-DPLOW_NV_FA_GF_FULL=8 -DPLOW_FP8_LD16 -DPLOW_FP8_FAST` | FASTPF (PIPE=1) |
| **P0 — as the script builds it** | either | `-DPLOW_NV_FA_PIPE=0` (the script's hardcode) |

**Correction to PX-11: the deployed `GF_FULL` is 2, not 4.** `build_sm120_cubin.sh` passes
`-DPLOW_NV_FA_GF_FULL=4` only on the *plain* decode object; the `_fp8kv` object — the one that is
actually loaded — omits it, so it falls to `#define PLOW_NV_FA_GF_FULL PLOW_NV_FA_GF` = **2**
(`interp_sm120.cu:254-261`). Every PX-11 ratio quoted against "deployed GF=4" understates the win.

## 2. Prefill, concurrency 1 — the FASTPF result

One 126,976-token prompt, 8 output tokens, so wall ≈ prefill.

| build | wall (s) | vs FASTPF |
|---|---|---|
| mixed packet, **PIPE=0** prefill object (what the script emits) | **175.90** | 5.43× |
| lc-b8 all-layer packet, PIPE=0 prefill object | 177.50 | — |
| mixed packet, **FASTPF** prefill object | **32.39** | 1.00× |
| *`gemma4-12b-longctx-5090.md` §2, recorded at 126976* | *33.09* | — |

**FASTPF is worth 5.43× on a 127k prefill**, not the −21% PX-8 measured at 67k. PX-8's figure was
taken at a context where the px4 fast path barely matters; §3a of the campaign doc already says the
same thing from the other side ("at 64k it is 4×").

And the FASTPF number **reproduces §2's recorded 33.09 s to 2%**, while the PIPE=0 number is 5.3×
away from it. §2b's plow run therefore already had a FASTPF-equivalent prefill object.
`PLOW_FP8_KV_FASTPF` was not an unclaimed win sitting next to §2b — it was **already in §2b's
numbers**, and the arm the campaign shipped in `build_sm120_cubin.sh` is 5.4× slower than the arm
it measured.

## 3. The cell

All rows report `Total input tokens = 1,015,914` and `Total generated tokens = 8,192` — the same
counts §2b reports, so this is like-for-like with the recorded cell.

| engine / arm | aggregate out tok/s | wall (s) | median ITL (ms) |
|---|---|---|---|
| **vLLM 0.26.0** — same session | **42.49** | **192.8** | 18.87 |
| **plow A — control** (deployed flags) | 23.76 | 344.8 | 63.97 |
| **plow B — tuned** (`GV_MM_MAX` 8 + `GF_FULL=8` + `LD16` + `FAST`) | **25.96** | **315.6** | **43.16** |
| *plow, §2b as recorded* | *24.53* | *~334* | *53.22* |
| *vLLM, §2b as recorded* | *42.63* | *192* | *18.85* |

**Both columns reproduce their recorded values.** plow's control lands within 3% of §2b's 24.53
tok/s / ~334 s; vLLM lands within 0.3% of 42.63 tok/s / 192 s / 18.85 ms. Nothing about the machine
or the harness moved between sessions, so the tuned arm's delta is the flags.

vLLM's median TTFT here is 98.7 s — real (vLLM emits its role chunk after prefill) and reported
only to show that vLLM is *also* prefill-dominated on this cell; it queues 8 × 127k against ~4
resident slots. plow's TTFT column is omitted for the reason given at the top.

## 4. Per-arm attribution

**Tuned − control = +9.3% aggregate (23.76 → 25.96 tok/s), −8.5% wall, −32.5% median ITL.**

All of it is decode; none of it is prefill. The two arms share a byte-identical prefill object
(`a10c0d8a…`) — `GV_MM_MAX`, `GF_FULL`, `LD16` and `FAST` are **provably inert in the prefill
object**: rebuilding it with and without all four gives the same md5. So the entire delta lands in
the ~60 s of the 316 s wall that is not serial prefill, which is why a decode win measured at
1.48× on its own op shows up as 9.3% end-to-end.

The median ITL is the cleaner view of the same thing: **63.97 → 43.16 ms, 1.48×** — and that is
numerically the *same* 1.48× the isolated flash-decode bench gives for `GF=2 → GF=8` at the
packet's `ns=32` (§5). The GEMV (`GV_MM_MAX`) and flash (`GF_FULL`/`LD16`/`FAST`) contributions are
**not separated end-to-end** — one cell per arm was the GPU budget, and separating them needs two
more full cells. PX-10 and PX-11 own the per-op split; this note does not re-derive it.

**Prefill is untouched and still dominates.** At 32.4 s per 127k prompt (§2) and no
prefill/decode overlap, 8 × 32.4 = 259 s of the 316 s wall is serial prefill. That is 82% of the
cell. No decode flag can reach it, and it is why the answer to §2b is still "the deficit is
prefill" even after every decode correction in the campaign has landed.

## 5. Isolated-op cross-check (PX-11's bench, re-run on this GPU)

`runtime/bench/nvidia/px11_flash_decode_bench.cu`, FULL class, B=8, ctx 131072, fp8 KV — reproduces PX-11's
own cells to 2–4%, which is that document's stated noise band:

| GF | nsplit | ms | vs deployed | note |
|---|---|---|---|---|
| **2** | 32 | **3.4242** | 1.00× | the TRUE deployed config |
| 4 | 32 | 2.6806 | 1.28× | what PX-11 assumed was deployed |
| **8** | **32** | **2.3162** | **1.48×** | arm B — the packet's baked `NS_FULL_ABS` |
| 8 | 21 | 1.7969 | **1.91×** | needs an `NS_FULL_ABS=21` re-emit |

`maxdiff` vs the GF reference at fixed nsplit is `0.000e+00` in every cell, re-confirming PX-11's
bit-exactness claim on this GPU. The `ns=21` row is the free 1.29× that a re-emit would add on top
of arm B; it is **not** in any number in §3 and is **not** bit-exact (a different split boundary is
a different merge), so it needs its own greedy gate.

## 6. Gates

| gate | result |
|---|---|
| control reproduces §2b | **PASS** — 23.76 tok/s / 344.8 s vs the recorded 24.53 / ~334 (3%) |
| same offered load as §2b | **PASS** — `Total input tokens` 1,015,914 and `Total generated` 8,192 on every row, identical to §2b |
| deployed cubin reproduced bit-exactly | **PASS** — `PLOW_BUILD_FP8KV=1 PLOW_EXTRA_DEFINES="-DGV_MM_MAX=16 -DPLOW_NV_W8A8=1"` on this tree gives md5 `482882b3…` = `/root/plow-out/lc-b8/interp_sm120.cubin` |
| every `-D` provably reaches the decode object | **PASS** — one flag at a time off the control build, all four change the cubin: `482882b3` (base) / `c047fb4a` (+`GF_FULL=8`) / `cfa0c4e9` (+`LD16`+`FAST`) / `973d29e4` (`GV_MM_MAX` default) |
| flags provably inert in the PREFILL object | **PASS** — the fp8-KV pf object is md5-identical with and without all four (`a10c0d8a…`); only `PLOW_NV_FA_PIPE` changes it |
| **greedy-token parity, control vs tuned** | **PASS** — 124,956-token prompt, `temperature 0`: identical text and **identical token ids**, 34/34. No divergence index to report |
| long-context coherence at 125k | **PASS** — both arms answer the prompt correctly and fluently, not degenerate |
| FASTPF legality on the packet actually served | **PASS** — the mixed packet emits no hd256 fp8 prefill, so `interp_sm120.cu:772`'s `__trap()` is unreachable; verified by the arm running |
| FASTPF legality on `/root/plow-out/lc-b8` | **FAIL, and that is the finding** — `CUDA_ERROR_LAUNCH_FAILED` at an 18-token prompt. lc-b8 is an all-layer fp8-KV packet (§0a) |
| PX-11's isolated bench reproduces on this GPU | **PASS** — GF=4/ns=32 2.6806 vs PX-11's 2.6918; GF=8/ns=21 1.7969 vs 1.7656 (2–4%, inside PX-11's stated band) |
| GF bit-exactness at fixed nsplit | **PASS** — `maxdiff` 0.000e+00 in every cell of the re-run |
| GPU exclusive | **ENFORCED** — every run under `perf-data/harness/gpulease`, rc=0, no foreign-process warning |
| GPU health cross-check | **PASS** — the bandwidth ladder reproduces the in-tree ceiling, so the slow prefill in §2 is the build, not the card |
| TTFT / TPOT | **NOT REPORTED, deliberately** — invalid for plow (role-only SSE chunk poisons the first ITL sample) |
| **GEMV vs flash split of the tuned arm's +9.3%** | **NOT RUN** — needs two more full cells; one per arm was the budget |
| **`NS_FULL_ABS=21` re-emit at `GF_FULL=8`** | **NOT RUN** — worth a further **1.29×** on the flash-decode op in the isolated bench (§5). Not bit-exact, needs its own greedy gate |
| tuned decode × PIPE=0 prefill, full cell | **NOT RUN** — FASTPF was attributed at concurrency 1 instead (§2), where it is unambiguous and 25× cheaper in GPU time |
| numeric parity vs HF/vLLM | **NOT RUN** — no `transformers` reference in this env; pre-existing harness scope, same as PX-10 |
| PX-10's fp8-KV hd512 prefill crash at bucket ≥ 4096 | **NOT REACHED** — the emitted ladder is `[128, 512, 1024]`. Still open, still undiagnosed |

### Bug found and recorded

`scripts/build_sm120_cubin.sh` cannot produce a working long-context fp8-KV deployment. Its
hardcoded `-DPLOW_NV_FA_PIPE=0` costs **5.4× on a 127k prefill** (§2), and the asset the campaign
left on disk (`/root/plow-out/lc-b8`) is an all-layer fp8-KV packet on which the fast arm cannot
legally run at all. The script needs a `PLOW_FP8_KV_FASTPF=1` env hook mirroring
`runtime/CMakeLists.txt`'s option, and it should refuse to pair a FASTPF object with an all-layer
packet rather than `__trap()` at the first prefill.

## 7. Reproduce

    W=$PWD
    # decode + PIPE=0 prefill objects
    PLOW_ROOT=$W PLOW_BUILD_FP8KV=1 \
      PLOW_EXTRA_DEFINES="-DGV_MM_MAX=16 -DPLOW_NV_W8A8=1" \            # arm A
      scripts/build_sm120_cubin.sh <dir>/interp_sm120.cubin
    PLOW_ROOT=$W PLOW_BUILD_FP8KV=1 \
      PLOW_EXTRA_DEFINES="-DPLOW_NV_W8A8=1 -DPLOW_NV_FA_GF_FULL=8 -DPLOW_FP8_LD16 -DPLOW_FP8_FAST" \
      scripts/build_sm120_cubin.sh <dir>/interp_sm120.cubin             # arm B
    # FASTPF prefill object — NOT reachable through the script (it hardcodes PIPE=0)
    nvcc -arch=sm_120a -O3 -cubin -I runtime/common -I runtime/nvidia \
      -DPLOW_NV_PREFILL=1 -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 \
      -DPLOW_FP8_KV=1 -DPLOW_NV_W8A8=1 -o <dir>/interp_sm120_pf.cubin runtime/nvidia/interp_sm120.cu
    # MIXED fp8-KV packet (this is the §2b asset; lc-b8 is not)
    PLOW_UNISEG=1 PLOW_NS_FULL_ABS=32 PLOW_DECODE_BATCH=8 PLOW_FP8=1 PLOW_W8A8=1 \
    PLOW_FP8_KV=1 PLOW_FP8_KV_FULL=1 \
      plowc --hf-dir /root/gemma4-fp8-ckpt --gpu rtx5090 --emit devblob \
            --max-ctx 132096 --weight-dtype fp8 --out <dir>
    # serve + bench
    PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1 PLOW_UNISEG=1 PLOW_DEV_SAMPLE=1 \
      plowrt serve --assets <dir> --port 8200 --slo-ms 100000000
    vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8200 \
      --endpoint /v1/chat/completions --model gemma4-fp8-ckpt \
      --tokenizer /root/gemma-4-12B-it --dataset-name random \
      --random-input-len 126976 --random-output-len 1024 --random-range-ratio 0 \
      --num-prompts 8 --max-concurrency 8 --ignore-eos --seed 0

## 8. Verdict

**plow still LOSES this cell, by 1.64×** — 25.96 tok/s against vLLM's 42.49, both measured in this
session on the same offered load. The campaign's decode corrections are real and they are now
landed and gated, but they move the deficit from **1.74× to 1.64×**: 6% of a 74% gap.

| | §2b as recorded | PX-12, measured |
|---|---|---|
| plow | 24.53 | **25.96** |
| vLLM | 42.63 | **42.49** |
| **vLLM ahead by** | **1.74×** | **1.64×** |

Why the ceiling is so low: **82% of plow's 316 s wall is serial prefill** (8 × 32.4 s, no overlap
with decode). Every flag in this note is a decode flag. Even a decode step of zero would leave plow
at 8192/259 = 31.6 tok/s, still **1.34× behind vLLM** — so no combination of `GV_MM_MAX`,
`GF_FULL`, `LD16`, `FAST` or `NS_FULL_ABS` can win this cell. The bound is arithmetic, not a
measurement.

That is the honest reading of the campaign: PX-10 and PX-11 correctly attributed the batched-decode
gap and correctly sized its fix, and the fix works — median ITL fell 1.48×, exactly as the isolated
op predicted. It is simply not the gap that decides this cell. `gemma4-12b-longctx-5090.md` §6
already named the two things that do — the prefill GEMM at 38% of fp8 peak (occupancy 1, needs
segmented dispatch reachable from `serve`) and the absence of prefill/decode overlap — and both are
runtime work, not knobs. **Neither was touched here, and neither should be skipped in favour of
more kernel flags.**

Two things that are free and were measured but not spent:

* **`NS_FULL_ABS=21` at `GF_FULL=8`** — a further 1.29× on the flash-decode op (§5), one re-emit,
  needs a greedy gate because nsplit is not bit-exact.
* **The `-DPLOW_NV_FA_PIPE=0` hardcode in `scripts/build_sm120_cubin.sh`** — not a win to be
  claimed (§2b already had the fast arm) but a 5.4× landmine for anyone who builds a long-context
  fp8-KV asset from the script as written.
