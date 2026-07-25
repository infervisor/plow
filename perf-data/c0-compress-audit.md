# C-0 compressibility audit — gemma-4-12B (P9 v2 step 0)

Date 2026-07-21. CPU-only, real checkpoint `/workspace/models/gemma-4-12B-it`. Decode-path weight stream audited: 23.813 GB bf16 (328 projections + tied embed/lm_head).

## bf16 classes (per class x layer type; bytes-weighted within class)

| class | ltype | GB | H_hi | H_lo | H_exp | EXP_BASE16 | esc16 % | **sz12 ratio** | sz11(3b) ratio | esc8 % | zstd19 whole | zstd19 planes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| down | full | 0.944 | 2.7591 | 7.9711 | 2.6197 | 109..110 | 0.0150 | **1.3323x** | 1.3575x | 1.6346 | - | - |
| down | sliding | 4.719 | 2.9192 | 7.9712 | 2.8258 | 107..111 | 0.0159 | **1.3323x** | 1.3644x | 1.5100 | - | - |
| embed/lm_head | shared | 2.013 | 2.6484 | 7.9703 | 2.5663 | 108..108 | 0.0140 | **1.3317x** | 1.3771x | 1.2708 | - | - |
| gate/up | full | 1.887 | 2.7413 | 7.9711 | 2.6007 | 109..109 | 0.0117 | **1.3318x** | 1.3668x | 1.4534 | - | - |
| gate/up | sliding | 9.437 | 2.7961 | 7.9719 | 2.6903 | 108..109 | 0.0125 | **1.3317x** | 1.3662x | 1.4646 | - | - |
| o | full | 0.503 | 2.7899 | 7.9716 | 2.6774 | 109..109 | 0.0180 | **1.3319x** | 1.3478x | 1.8074 | - | - |
| o | sliding | 1.258 | 2.806 | 7.9719 | 2.7051 | 108..110 | 0.0178 | **1.3315x** | 1.3589x | 1.5962 | - | - |
| qkv | full | 0.535 | 2.8197 | 7.9715 | 2.7109 | 109..110 | 0.0184 | **1.3314x** | 1.3442x | 1.8636 | - | - |
| qkv | sliding | 2.517 | 2.8131 | 7.9719 | 2.7151 | 108..110 | 0.0179 | **1.3315x** | 1.3549x | 1.6677 | - | - |

## fp8 (e4m3) twins

| class | ltype | GB | H_byte | H_exp4 | EXP_BASE8 | esc8 % | **sz7 ratio** | zstd19 |
|---|---|---|---|---|---|---|---|---|
| down | full | 0.472 | 6.5928 | 2.6514 | 7..8 | 1.9395 | **1.0286x** | - |
| down | sliding | 2.359 | 6.5843 | 2.6434 | 7..8 | 1.8620 | **1.0327x** | - |
| gate/up | full | 0.944 | 6.5233 | 2.5874 | 8..8 | 1.4713 | **1.0531x** | - |
| gate/up | sliding | 4.719 | 6.555 | 2.6169 | 7..8 | 1.5844 | **1.0468x** | - |
| o | full | 0.252 | 6.6114 | 2.6676 | 7..8 | 1.9830 | **1.026x** | - |
| o | sliding | 0.629 | 6.5686 | 2.6288 | 7..8 | 1.6784 | **1.0418x** | - |
| qkv | full | 0.267 | 6.5461 | 2.6081 | 7..8 | 1.5603 | **1.0482x** | - |
| qkv | sliding | 1.258 | 6.5711 | 2.6312 | 7..8 | 1.6768 | **1.0418x** | - |

## Reference upper bounds (zstd-19, "are we near Shannon?")

Byte-plane-split zstd-19 (hi[] and lo[] compressed separately) is the near-Shannon
reference; whole-tensor zstd-19 is the naive baseline. The full per-class zstd sweep is
CPU-expensive (~15 min/class at -19 -T0) and was truncated after two representative
classes to free cores for the kernel build; both bf16 classes cluster tightly (H_exp
2.56–2.83 b across every class), so these two are representative:

| class | ltype | zstd19 whole | zstd19 planes (near-Shannon) | our sz12 |
|---|---|---|---|---|
| embed/lm_head | shared | 1.290x | **1.501x** | 1.332x |
| down | sliding | 1.286x | **1.483x** | 1.332x |

Reading: the plane-split ceiling is ~1.48–1.50x (matching the literature's 1.51x bf16
Shannon bound). Our fixed-length splitzip realizes **1.33x** — it captures ~88% of the
exponent entropy that a full variable-length entropy coder would, at ~4 fused SASS
ops/elem and zero smem tables (variable-length coders are killed by the <2 ops/elem
budget, plan Appendix §2/§5). Whole-tensor zstd (1.29x) is *below* our fused scheme —
the byte-plane split is what unlocks the exponent redundancy.

## Verdict (plan C-0 thresholds)

- splitzip-12b bytes-weighted mean ratio: **1.3318x** (threshold >= 1.25)
- worst class escape rate: **0.0184%** (threshold <= 0.5%)
- **C-1 GO**
- fp8 sz7 bytes-weighted ratio: **1.0421x** (threshold >= 1.12) => **C-3fp8 NO-GO**
- KV: deferred (needs GPU-side KV dump; weights alone gate C-1)
