# GLM-5.3 on MI300X: are plow's designed collectives actually paying?

Static packet analysis of the emitted device blobs — `plowrt disasm`, no GPU, no driver.
Blobs: `build-glm53/tp4m` and `build-glm53/tp8m` (GLM-5.3-FP8, gfx942, 304 CU, max-ctx 10240,
decode rungs 1/2/4, measured GEMM tiles). Dispatch width is the `b=` field, i.e. how many of
the 304 CUs that packet occupies.

## Answer in one line

**Prefill: yes.** **Decode: the collective is cheap and is not the bottleneck.**
**But none of the designed collective FOLDS were armed, and arming them removes 156 full-width
prefill packets (−7.8% of CU-weighted work) and 78 decode packet boundaries per token.**

## 1. What the emitted stream actually contains

| program | packets | mean dispatch | collective packets | collective share of count | of CU-weighted work |
|---|---:|---:|---:|---:|---:|
| TP4 decode T=1 | 2523 | 66 / 304 CU | 157 | 6.2% | 1.13% |
| TP8 decode T=1 | 2523 | 61 / 304 CU | 157 | 6.2% | 1.21% |
| TP4 prefill T=2048 | 2246 | 269 / 304 CU | 157 | 7.0% | 7.84% |
| TP8 prefill T=2048 | 2246 | 269 / 304 CU | 157 | 7.0% | 7.84% |

Two collectives per layer (attention seam + MoE seam), 78 layers, so 156 per program, plus one
`XArgmaxFin` for the vocab-parallel lm_head.

**Prefill uses `XReduceTwoShot` at full chip width.** `n=12582912` elements — 2048 tokens x 6144
hidden — dispatched `b=304`, `gate_ag=1`. That is plow's two-shot all-gather doing exactly what
it was designed for, and 7.8% of CU-weighted work for 156 seams of 25 MB each is proportionate.

**Decode uses the narrow `XReduce`.** `H=6144`, so the payload is one token's hidden state, ~24 KB,
dispatched on `b=12` — 3.9% of the chip. It is 6.2% of the packet count but 1.1% of CU-weighted
work. The decode collective is a LATENCY cost (156 serialization points per token), not a
bandwidth cost. Its width does not grow with rank count: `b=12` at both n_gpu=4 and n_gpu=8, and
its weighted share only moves 1.13% -> 1.21%.

## 2. The designed folds were NOT armed — this is the finding

`XReduceAddNorm` appeared **zero** times. In both programs the seam's residual/norm was a
SEPARATE packet immediately consuming the collective's output:

```
#10  XReduce         b=12   out<-act.attn      | H=6144 n_gpu=4
#11  AddNorm         b=1    b<-act.attn        | rows=1 feat=6144      <- decode, ONE workgroup
#14  XReduceTwoShot  b=304  out<-act.attn      | n=12582912 n_gpu=4
#15  Residual        b=304  b<-act.attn        | n=12582912            <- prefill, FULL width
```

In prefill that un-folded `Residual` costs 7.8% of CU-weighted work — the same as the entire
collective it follows. `PLOW_GLM_XR_RES` folds it into the two-shot; `GLM_FUSE_XRN` folds the
decode pair into `XReduceAddNorm`. Neither was set in the emit; both are in this tree.

Re-emitting with `PLOW_GLM_XR_RES=1 GLM_FUSE_XRN=1` (`build-glm53/tp4f`), measured by the same
static census:

| program | packets | change | CU-weighted work |
|---|---:|---:|---:|
| decode T=1 | 2523 -> 2445 | −78 (78 XReduce+AddNorm pairs -> 78 XReduceAddNorm) | −0.6% |
| prefill T=2048 | 2246 -> 2090 | −156 (every Residual folded away) | **−7.8%** |

The decode win is not the 0.6% of work; it is 78 fewer packet boundaries per token, on a chain
where the GLM-5.2 campaign priced the boundary at ~2.13 us each.

### And it measures — the folds are the best plow arm on this model

Served end to end at TP4 on the same four cards, all arms minutes apart, 8 prompts per cell,
128 output tokens. `tp4f` is the folded blob; `tp4m` is the same blob without the folds; `plow`
is the original analytical-tile baseline.

| cell | metric | plow | tp4m | **tp4f** | tp4f vs plow |
|---|---|---:|---:|---:|---:|
| 1024 / c1 | tok/s | 23.78 | 23.97 | **24.20** | +1.8% |
| 1024 / c1 | TPOT ms | 38.70 | 38.59 | **38.21** | −1.3% |
| 1024 / c4 | tok/s | 38.93 | 39.23 | **39.94** | +2.6% |
| 1024 / c4 | TPOT ms | 95.06 | 94.75 | **93.62** | −1.5% |
| 4096 / c1 | tok/s | 18.31 | 17.66 | **19.22** | **+5.0%** |
| 4096 / c1 | TPOT ms | 47.02 | 49.84 | **43.69** | **−7.1%** |
| 4096 / c4 | tok/s | 29.48 | 29.05 | **29.59** | +0.4% |
| 4096 / c4 | TPOT ms | 122.23 | 123.65 | 122.24 | 0.0% |

The folded arm is at least as good as the baseline in every cell and clearly better at 4096/c1.
The win is on the DECODE axis, which is what the packet census predicted: the fold removes
packet boundaries, and 78 of them per token is worth about 0.17 ms against a 38-47 ms TPOT
before any second-order effect.

**TTFT did not move** (383 -> 385, 1024 -> 1030, 1698 -> 1692). That is a real negative result
about the prefill half: folding `Residual` into `XReduceTwoShot` removes 7.8% of CU-weighted
prefill WORK and buys no TTFT, so prefill at these lengths is not CU-throughput bound. Do not
spend more effort on prefill packet-count reduction on this evidence.

**Not yet adopted.** `PLOW_GLM_XR_RES` is recorded byte-identical and `GLM_FUSE_XRN` requires
`fuse_b1` + tp>1 (both hold here), but that record is GLM-5.2's. On GLM-5.3 these need the
paired accuracy gate before the deltas above can be called wins.

## 3. Collectives are not where decode is losing

The same census says where decode actually spends the chip:

| | TP4 | TP8 |
|---|---:|---:|
| Gemv | 30.3% | 32.7% |
| GemvQkv | 27.7% | 29.3% |
| MoE expert walks (Glu+Down, block-fp8) | 23.0% | 24.8% |
| FlashMlaDecode | 7.5% | 4.0% |
| MlaMergeFold | 6.0% | 3.2% |
| **collectives** | **1.13%** | **1.21%** |

**75.3% of decode packets dispatch on 32 CUs or fewer**, mean width 66 of 304. 155 of them are
`AddNorm` on a SINGLE workgroup. Projections plus expert walks are 58% (TP4) / 62% (TP8) of
CU-weighted work, and the emit's own lean oracle puts the TP4 decode step at 16.0 GB touched
with a 3.0 ms HBM-bandwidth floor against 38.7 ms measured.

So decode is weight-streaming and packet-boundary bound, not collective bound. Going after the
collectives to fix decode TPOT would be attacking 1% of the chip. The levers that match the
evidence are the ones that cut the weight stream (`GLM_LINEAR_FP8`, already prepped here as
`GLM-5.3-plow-q`) or the packet count (the folds above, and the narrow-op problem).

## Reproducing

```bash
plowrt disasm build-glm53/tp4m --program 1      # decode
plowrt disasm build-glm53/tp4m --program 2048   # one prefill bucket
```
