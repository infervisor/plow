# DEFINITIVE decode-only sweep — plow vs vLLM, Gemma-4-31B bf16

**The one artifact for "why plow loses short-ctx decode but wins long-ctx."**
DECODE ONLY (TPOT ms/tok, lower better). Prefill/TTFT deliberately excluded.
9 contexts (1k–128k) × 3 TP degrees (1/4/8) × 2 engines = **54 decode cells, all filled.**

Measured 2026-07-17, 8× MI350X / gfx950 (CDNA4), bf16, batch 1, greedy, output_len 128.
plow branch `tp`, GLOBAL-QUEUE decode, packets compiled at max_ctx=131072, **device==host verified**
(TP8 ranks agree, rank0 device-argmax==host-argmax). vLLM `vllm/vllm-openai-rocm:latest` 0.25.1,
TRITON_ATTN, cudagraphs (not bit-exact). Model `gemma-4-31B-it-text`.

---

## THE HEADLINE

- **plow LOSES short-context decode, WINS long-context decode, at every TP degree.** The curves
  cross exactly once each.
- **The crossover moves RIGHT as TP grows: TP4 ≈ 42k, TP8 ≈ 81k, TP1 ≈ 119k.** (Not monotonic in the
  naive direction — see §2: vLLM's short-ctx decode gets *dramatically* faster with more GPUs, so plow
  has to climb further to overtake at high TP.)
- **The mechanism (§3): vLLM's decode scales BADLY with context, plow's scales WELL.** From 1k→128k
  vLLM adds **~12 ms/tok at every TP** (attention balloons); plow adds only **~6 ms** — plow's
  per-token attention is ~2× cheaper per unit context. vLLM's *multiplicative* growth is 1.9×/2.6×/2.9×
  (TP1/4/8); plow's is 1.40×/1.45×/1.52×.

---

## 1. THE FULL 54-CELL GRID — decode TPOT ms/tok (plow | vLLM | ratio plow/vLLM)

**ratio < 1.00 = plow wins (bold).** The single crossover per TP is where the column flips.

| ctx  | plow TP1 | vLLM TP1 | ratio | plow TP4 | vLLM TP4 | ratio | plow TP8 | vLLM TP8 | ratio |
|------|---------:|---------:|------:|---------:|---------:|------:|---------:|---------:|------:|
| 1k   | 18.294 | 13.690 | 1.336 | 12.561 | 7.530  | 1.668 | 11.755 | 6.380  | 1.843 |
| 4k   | 18.440 | 14.630 | 1.260 | 12.267 | 8.500  | 1.443 | 12.101 | 7.450  | 1.624 |
| 8k   | 18.821 | 15.940 | 1.181 | 12.585 | 9.780  | 1.287 | 12.374 | 8.830  | 1.401 |
| 16k  | 19.286 | 16.700 | 1.155 | 12.961 | 10.690 | 1.212 | 12.662 | 9.810  | 1.291 |
| 32k  | 20.166 | 19.500 | 1.034 | 13.705 | 13.560 | 1.011 | 13.422 | 12.670 | 1.059 |
| 48k  | 21.081 | 20.450 | 1.031 | 14.456 | 14.550 | **0.994** | 14.248 | 13.590 | 1.048 |
| 64k  | 22.056 | 21.560 | 1.023 | 15.167 | 15.610 | **0.972** | 15.018 | 14.790 | 1.015 |
| 96k  | 23.861 | 23.540 | 1.014 | 16.615 | 17.690 | **0.939** | 16.451 | 16.670 | **0.987** |
| 128k | 25.571 | 25.710 | **0.995** | 18.157 | 19.840 | **0.915** | 17.905 | 18.720 | **0.956** |

Data provenance: plow — all 27 cells from ONE fresh idle-node build/session (monotonic in ctx).
vLLM TP1 — fresh this session (single GPU, `HIP_VISIBLE_DEVICES=7`). vLLM TP4/TP8 1k–32k from
`gemma4-31b-vllm-tp.json`; 48k–128k from `gemma4-31b-longctx-sweep.json`.

> **Supersedes note:** the earlier `gemma4-31b-longctx-sweep.json` reported plow TP4/TP8 at 48k–128k
> ~0.5–1.0 ms/tok *faster* than here (e.g. TP8@128k 16.9 vs 17.9). Those did not lie about the
> *direction* but are not monotonic when joined to the mid-context points measured on this build, so
> this unified single-build sweep is the definitive plow curve. Using it moves the TP8 crossover from
> the old ~48k estimate out to ~81k.

---

## 2. THE CROSSOVER — where plow overtakes vLLM (per TP)

| TP | last vLLM win | first plow win | **crossover ctx** | plow lead at 128k |
|----|---------------|----------------|-------------------|-------------------|
| 1  | 96k (1.014)   | 128k (0.995)   | **≈ 119k**        | +0.5 %            |
| 4  | 32k (1.011)   | 48k (0.994)    | **≈ 42k**         | +8.5 %            |
| 8  | 64k (1.015)   | 96k (0.987)    | **≈ 81k**         | +4.3 %            |

**Read:** decode has a single, clean crossover at every TP. **TP4 crosses earliest (~42k), TP8 later
(~81k), TP1 latest (~119k).** The ordering is set by how fast vLLM's *short*-context decode is at each
TP (§2a), not by plow — plow's short-ctx cost barely moves with TP (18.3→12.6→11.8 at 1k) while vLLM's
plummets (13.7→7.5→6.4), so at TP8 plow starts from a 1.84× hole and needs more context to dig out.

### 2a. Why the short-ctx gap is WORST at high TP
Relative plow deficit at 1k: **TP1 +33.6 %, TP4 +66.8 %, TP8 +84.3 %.** But in *absolute* ms the gap
is nearly TP-invariant: **TP1 4.60 ms, TP4 5.03 ms, TP8 5.38 ms.** This is the audit's finding
(`vllm-decode-audit.md`): the short-ctx gap is a fixed ~5 ms of plow decode *structure* — the
persistent-megakernel per-op counter-gate tax (~1.5 ms, no vLLM cudagraph analog) plus a heavier
flash_decode→merge→o_proj attention tail — **not** TP/all-reduce overhead. vLLM's cudagraph packs
kernels with no cross-CU handshake and its TP splits the GEMV weight-stream, so vLLM's short-ctx TPOT
falls with TP while plow's fixed tax stays put → the *relative* hole deepens with TP.

---

## 3. THE SCALING — vLLM scales badly, plow scales well (the mechanism)

Growth from 1k → 128k (decode ms/tok):

| | plow mult | vLLM mult | plow Δms | vLLM Δms |
|---|----------:|----------:|---------:|---------:|
| TP1 | **1.40×** | 1.88× | +7.28 | +12.02 |
| TP4 | **1.45×** | 2.64× | +5.60 | +12.31 |
| TP8 | **1.52×** | 2.94× | +6.15 | +12.34 |

Two facts nail the story:

1. **vLLM adds ~12 ms/tok of decode latency going 1k→128k at EVERY TP degree** (12.02 / 12.31 / 12.34).
   plow adds only **~6 ms** (7.28 / 5.60 / 6.15). Per unit context that's **plow ≈ 0.047 ms/tok per +1k
   vs vLLM ≈ 0.095 ms/tok per +1k — plow's per-token attention scaling is ~2× cheaper.**
2. **vLLM's multiplicative growth WORSENS with TP (1.9×→2.6×→2.9×)** purely because its short-ctx base
   shrinks (6.4 ms at TP8) while the ~12 ms attention growth is TP-insensitive — the KV-read/token cost
   of Triton attention doesn't parallelize away. plow's head-major sharded KV keeps its growth flat
   (~1.4–1.5×) at all TP.

This is *why* plow wins long-ctx decode: at short context attention is trivial and plow's fixed ~5 ms
structural tax dominates → plow loses; as context grows, decode becomes attention/KV-read bound, plow's
~2×-cheaper-per-token attention overtakes vLLM's ballooning Triton attention → plow wins, and the lead
widens monotonically to 128k.

---

## 4. THE SINGLE-GPU (TP1) PICTURE — same shape, crossover pushed furthest out

Yes: the single-GPU curve has the **same shape** as TP (lose short, win long) but is the *mildest*
short-ctx gap (**+33.6 % @1k**, vs +66.8 %/+84.3 % at TP4/TP8) and the *latest* crossover (**~119k**).
Why: at TP1 there is no TP GEMV-parallelism helping vLLM and no plow collective overhead, so both
engines run structurally similar single-GPU forward steps — their curves stay close and near-parallel,
crossing only at the very top (plow 25.57 vs vLLM 25.71 at 128k, a 0.5 % plow edge). The single-GPU gap
is real and structural (5 ms of plow decode scheduling), it just never gets amplified by TP, and plow's
better attention scaling still flips it — barely — by 128k.

**Bottom line:** plow's decode disadvantage is a fixed ~5 ms structural tax that is worst *relative* at
high TP (where vLLM's decode is fastest) and best at TP1; plow's decode *scaling* is ~2× better than
vLLM's at every TP, so plow overtakes at ~42k (TP4), ~81k (TP8), ~119k (TP1) and leads by up to +8.5 %
at 128k. Decode-heavy long-context generation favours plow; short-context or prefill-heavy favours vLLM.
