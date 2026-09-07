# Gemma-4-12B: an MXFP4 tied lm_head (`PLOW_MX4_HEAD`)

Gemma-4 ties `lm_head` to `model.language_model.embed_tokens.weight`. On the 12B that tensor is
**2,013,265,920 bytes of bf16** (262144 vocab x 3840 hidden x 2), and it is in NEITHER quantized
twin — so every decode step streamed 2.01 GB for the output projection while every other weight in
the model was quantized. `cpu_profile --batch 1` put it at 17.06 ms busy with a 17.53 ms span: a
serial tail with essentially no overlap, 13% of the fp8 decode step and 21% of the MXFP4 one.

## The tie is not a blocker

The emitter already had the split, for fp8 (`PLOW_FP8_HEAD`, rtx-19): keep the bf16 table bound for
the `Embed` op — it reads ONE 7.7 KB row per token, so its bandwidth is irrelevant — and give the
final GEMV a separate quantized copy. Quantizing the tensor in place would have changed the
embedding lookup too; a second copy costs memory and changes nothing upstream. This adds the w4
version of that trade: `mxfp4/<embed>` (e2m1, `[vocab, hidden/2]`) + `_scale` (E8M0, one byte per
32-K block), the same twin layout the dense MXFP4 projections and the GPT-OSS head already use, so
`GEMV_MXFP4` (op 91) needed no kernel work on any target.

2.01 GB -> 0.53 GB, against 1.01 GB for the fp8 head. **MXFP4 wins on both twins**, including the
fp8 one — logit quantization is far less sensitive than weight quantization mid-network, and it is
the larger cut. `--mx4-head` is therefore ON by default under `--mxfp4` and available as
`--mx4-head 1` on any other body; `--mx4-head 0` restores the bf16 head.

## Decode, through `plowrt serve`

`bench.py --workload chat_short --concurrency 1 --requests 8 --max-tokens 64 --fresh-prompts`,
`--cpu-threads 16`, one server alone on the box, TPOT p50 ms. Three runs per configuration, median
in bold. llama.cpp figures are the c=1 cells from `h2h/SUMMARY.md`.

| configuration | run 1 | run 2 | run 3 | median | RSS | llama.cpp | margin |
|---|---|---|---|---|---|---|---|
| bf16 (control, unchanged) | 218 | 219 | 221 | **219** | 29.85 GB | 267 (bf16 GGUF) | 1.22x |
| fp8 | 132 | 133 | 132 | **132** | 19.27 GB | 133 (Q8_0) | 1.01x |
| fp8 + MXFP4 head | 121 | 121 | 121 | **121** | 19.79 GB | 133 (Q8_0) | **1.10x** |
| MXFP4 | 88 | 88 | 89 | **88** | 35.76 GB | 121 (Q4_K_M) | 1.38x |
| MXFP4 + MXFP4 head | 76 | 75 | 76 | **76** | 36.36 GB | 121 (Q4_K_M) | **1.59x** |

fp8 was our only data type that did not beat llama.cpp. It now does, by 10%, and MXFP4 goes from
1.38x to 1.59x. The 11 ms and 12 ms recovered are close to the 8-9 ms and 13 ms the byte counts
predict; the head's span was already nearly serial, so the saving lands almost whole.

Emitted blobs were checked byte-identical across the default change: `--mxfp4` now reproduces the
76 ms blob exactly, `--mxfp4 --mx4-head 0` the 88 ms one, `--w8a16` the 132 ms one, `--w8a16
--mx4-head 1` the 121 ms one, and the bf16 blob is unchanged (also pinned by
`gemma_dense_blob_is_stable`).

### Memory

+0.53 GB of resident set on the fp8 twin (19.27 -> 19.79 GB) and +0.60 GB on the MXFP4 one. The fp8
configuration stays well under llama.cpp Q8_0's 12.75 GB of weights plus its cache. The MXFP4
twin's 35.8 GB baseline is not from this change and is worth recording: its blob declares BOTH the
bf16 originals (which prefill's GEMM reads) and the mxfp4 decode twins, so it resides larger than
bf16 despite decoding faster.

## Quality

This cannot be bit-identical — quantizing the output projection changes logits, and argmax over
262144 vocab entries reroutes on near-ties. Five prompts, greedy, 64 output tokens, tokenized with
the Gemma-4 tokenizer:

| comparison | tokens differing | first divergence (per prompt) |
|---|---|---|
| fp8 -> fp8 + MXFP4 head | 41 / 257 (16.0%) | -, 50, -, 27, 46 |
| MXFP4 -> MXFP4 + MXFP4 head | 111 / 257 (43.2%) | -, 32, 52, 26, 24 |
| bf16 -> fp8 (existing, for scale) | 46 / 257 (17.9%) | -, 62, -, 49, 34 |
| bf16 -> MXFP4 (existing, for scale) | 185 / 257 (72.0%) | -, 30, 5, 25, 5 |

The head costs LESS divergence than the body quantization each twin already carries. The counts on
their own do not say whether the text got worse, so the text:

```
"Explain in three sentences why the sky is blue."
 fp8      ... Because blue light travels in shorter, smaller waves, it is scattered in all directions
 fp8+m4h  ... Because blue and violet light have shorter wavelengths, they are scattered in all directions,

"List three differences between TCP and UDP."
 fp8      ### 1. Reliability and Delivery Guarantee
          *  **TCP is connection-oriented:** It ensures that all data packets arrive at their
             destination in the correct order. ...
 fp8+m4h  ### 1. Reliability and Connection State
          *  **TCP is connection-oriented:** Before data can be sent, TCP establishes a connection
             between the sender and receiver using a "three-way handshake." ...

"Summarize the causes of the First World War in four sentences."
 mx4      ... exacerbated by a rapid buildup of military arms and the formation of rigid alliance
          systems that ensured a local conflict could escalate into a continental war.
 mx4+m4h  ... exacerbated by a rigid system of military alliances and an escalating arms race that
          prepared nations for conflict.
```

Every answer stays correct and fluent on both twins; "Paris" is still Paris on all five
configurations. The divergences are alternate phrasings of the same content picked up after a
near-tie, not degradation — in the first pair the quantized head's version is the more accurate
one. That is the bar this can meet, so the flag ships on rather than off.

## Reproducing

```
python perf-data/tools/quantize_mxfp4.py <hf-dir> <twin-dir>      # tied head included by default
plowc --hf-dir <hf-dir> --mxfp4 ...                               # head on
plowc --hf-dir <hf-dir> --w8a16 --mx4-head 1 ...                  # fp8 body, w4 head
```

To add the head to an already-quantized twin without rewriting it, the quantizer writes a
standalone file the loader picks up alongside the rest:

```
python perf-data/tools/quantize_mxfp4.py <hf-dir> <twin-dir> --no-layers \
  --out-name head_mx4.safetensors --extra model.language_model.embed_tokens.weight
```

A twin without those two keys fails at load with `MISSING WEIGHT:
mxfp4/model.language_model.embed_tokens.weight`, which names its own fix.
