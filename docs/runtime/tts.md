# TTS on NVIDIA (sm_90a)

Veena (`maya-research/Veena`), Chatterbox (`ResembleAI/chatterbox`, English) and Chatterbox
Multilingual V3 (23 languages) compile to plow packets and serve as OpenAI `POST /v1/audio/speech` from
`plowrt serve`. Design: [24 — TTS pipelines](../arch/24-tts-pipelines.md).
Every stage is a plowc packet; plowrt ships no model-specific code or native library.

## Recipes

The checked-in recipes are the source of truth for the knobs and rungs (the commands below
are the same steps by hand):

| model | recipe |
|---|---|
| Veena | `recipes/infervisor/veena/sm90a-h100-tp1.toml` |
| Chatterbox | `recipes/infervisor/chatterbox/sm90a-h100-tp1.toml` |
| Chatterbox Multilingual V3 | `recipes/infervisor/chatterbox-mtl/sm90a-h100-tp1.toml` |
| Qwen3-ASR | `recipes/infervisor/qwen3-asr/sm90a-h100-tp1.toml` |

```sh
# prep steps (exports) run first; build-record.json pins commit, prep and every hash
python3 scripts/campaign/campaign.py build recipes/infervisor/veena/sm90a-h100-tp1.toml --out $OUT
```

Hosts without nix: `PLOW_CAMPAIGN_NO_NIX=1`, `CARGO_TARGET_DIR` holding a release `plowc`,
`PYREF` (torch + snac) and, for Chatterbox, `CBX_PY` (chatterbox-tts 0.1.7; for the
multilingual recipe the upstream git package, whose `mtl_tts` loads `t3_mtl23ls_v3`).

Reproduced 2026-09-27 at 7c0a7f7e into fresh directories (H100): every packet
(`model.pkt`, `encoder.pkt`, `codec.pkt`, `s3gen.pkt`) byte-identical to the working assets,
cubins SASS-identical; Qwen3-ASR WER 3.913% (p50 60.1 ms), SNAC parity rel-L2 ≤ 5.3e-6,
Chatterbox CER 0.000, Veena CER 0.026 (the code-mixed prompt is translated by the Whisper judge).

## Veena

```sh
# SNAC export (torch), then LM packet + codec.pkt + paired sm_90a objects
python scripts/tts/snac_export.py $SNAC
PLOW_TTS_CODEC_DIR=$SNAC PLOW_TTS_PROFILE=veena PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 \
  PLOW_EMIT_PREFILL_CUBLASLT=1 PLOW_NO_GLU_FUSE=1 PLOW_FUSE_KV_HNR=1 \
  plowc --hf-dir $VEENA --gpu h100 --arch sm_90a --max-ctx 2048 \
        --emit devblob+cubin --served-name veena --out $ASSETS

plowrt serve --assets $ASSETS --port 8080
curl -s localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"<id from /v1/models>","input":"Hello there","voice":"kavya"}' > out.wav
```

`PLOW_EMIT_PREFILL_CUBLASLT` + `PLOW_NO_GLU_FUSE` route the prefill projections
(including unfused gate/up) to cuBLASLt: TTFT for a 60-token prompt 56.7 → 5.8 ms.
The runtime `dlopen`s `libcublasLt.so`, so the CUDA library directory must be
on `LD_LIBRARY_PATH`.

The recipe also sets `PLOW_EMIT_DECODE_CUBLASLT=1` with `PLOW_EMIT_DECODE_CUBLASLT_MIN_ROWS=48`.
With that, decode rungs of 48+ rows run their projections on cuBLASLt, and narrower rungs stay
in the interpreter.

| step_bench ctx 1024, ms | B=32 | B=48 | B=64 | B=96 | B=128 |
|---|---|---|---|---|---|
| interpreter only | 5.42 | 7.80 | 8.55 | 11.07 | 12.66 |
| cuBLASLt from 48 rows | 5.42 | 6.97 | 7.69 | 9.40 | 10.98 |
| cuBLASLt from 32 rows | 6.16 | | | | |

Served greedy output tok/s, ISL 128 / OSL 512, unique seed per cell:

| concurrency | 32 | 64 | 128 |
|---|---|---|---|
| interpreter only | 6550 | 7878 | 12055 |
| cuBLASLt from 48 rows | 6562 | 9756 | 14722 |
| vLLM | | 12863 | 19972 |

Between library calls a routed rung used to run each AddNorm, Glu and attention span as an
interpreter window, about 5-7 us of fixed cost each. `PLOW_DECODE_LIGHT` (default on) runs them as
ordinary launches of the same bodies: `plow_<arch>_light` for a lone AddNorm or Glu, and
`plow_<arch>_light_attn` for HeadNormRope + FlashDecode at the object's head dim. The rungs also
pair k|v and gate|up into one strided batch-2 cuBLASLt call (`PLOW_LT_PAIR`). The pair algorithm
is timed on cold weight copies and pinned from the widest rung. Tokens are unchanged: the
step_bench digest is equal at every rung.

| step_bench ms, ctx 384 / 1024 | B=64 | B=128 |
|---|---|---|
| cuBLASLt from 48 rows | 6.07 / 7.70 | 7.71 / 10.99 |
| + pairs, light kernels | 5.04 / 6.66 | 6.61 / 9.86 |

Served with the same client: c64 9755 -> 11544 and c128 14674 -> 16802 tok/s. Streaming c64 goes
from 66.1 to 70.8 aps. c1 TPOT is 3.35 -> 3.36 ms. CER median 0 (n=80).

The routed rungs' attention and head, since 0472d5c5 (fused q|k|v, lm_head on cuBLASLt, routes from
32 rows):

* `PLOW_DECODE_LIGHT_FLASH` (default on) runs a light attention segment's hd128 FlashDecode as
  `plow_<arch>_light_flash` (`d_flash_decode_stream`). One producer warp streams each block's K/V
  rows through a 4-stage smem ring (64 rows of K and V per stage) with `cp.async.bulk`, across item
  boundaries. The eight row-group warps run the row-group body's arithmetic unchanged. The
  register body keeps ~32 KiB in flight per SM and drains at every item, which leaves short
  contexts latency-bound.
* When that segment is the layer's q/k/v HeadNormRope (no norm, no gamma) plus the flash, the one
  launch also does the HeadNormRope. The producer ropes the item's q rows into the ring and writes
  the new k/v cache rows, then `fence.proxy.async` before it streams them.
* `PLOW_DECODE_HEAD_ARGMAX` (default on): the unaligned-lm_head kernel also computes the greedy
  argmax (the same packed keys) and writes `ids`. The packet's Argmax/ArgmaxFin window launches
  nothing. The kernel's copy is now 16-byte loads restaged through smem, with the tail dots
  batching their loads.

Outputs and new KV rows are bit-identical to the row-group body plus separate HeadNormRope
(`experiments/fa_stream_hnr_bench.cu`). The step_bench token digest is equal at every rung and
context below.

| hd128, 24/8 heads, stride 2048, us | B=64 ctx 384 | B=128 ctx 384 | B=128 ctx 1024 |
|---|---|---|---|
| HeadNormRope + row-group flash | 51.5 | 96.1 | 224.6 |
| streamed, HeadNormRope folded | 41.3 | 75.7 | 181.4 |
| KV floor (3.35 TB/s) | 30.1 | 60.1 | 160.3 |

In-model at B=128 ctx 384, per layer: HeadNormRope + flash 7.5 + 96-99 us -> 82.6 us. Per step,
lm_head tail + Argmax window: 54 + 52 us -> 46 us.

| step_bench ms, ctx 384 / 1024 | B=32 | B=64 | B=128 |
|---|---|---|---|
| 0472d5c5 | 4.28 / 5.08 | 4.89 / 6.52 | 6.30 / 9.57 |
| + streamed flash, HNR fold, head argmax | 4.01 / 4.83 | 4.51 / 6.03 | 5.65 / 8.62 |

Served output tok/s, same client and settings as above:

| | c1 TPOT ms | c64 | c128 | stream c64 aps |
|---|---|---|---|---|
| 0472d5c5 | 3.36 | 11928 | 17615 | 71.8 |
| this | 3.36 | 12871 | 19419 | 77.8 |
| this, `--decode-pipeline` | | 13307 | 20291 | 75.0 |

CER median 0 (n=80). E4B step_bench digests are unchanged at B=1/64/128.

The recipe's `[serve] extra_args` add `--decode-pipeline` and `--lt-rung-algos` for Veena only.
Also:

* Packed hd128 prefill runs one varlen `d_flash_prefill` pass over all requests instead of one
  pass per request.
* `PLOW_PREFILL_LIGHT` (default on) runs a prefill bucket's lone RmsNorm, Residual and SiLU Glu
  segments between cuBLASLt calls as `plow_<arch>_light_pf` launches from the decode object,
  instead of interpreter windows. They use the prefill object's bodies, with RmsNorm's
  warp-per-row path from 32 rows, so the step_bench digests do not change.

`--decode-pipeline` is off by default because it was added opt-in. It covers greedy rows only,
and an unforeseen stop costs one discarded step. Pipelined prefill stays off.

Pipeline A/B on the veena40_1 objects, two runs each:

| | c1 TPOT ms | c64 | c128 | stream c1 TTFA p50 ms | stream c64 aps | stream c64 TTFA p50/p90 ms |
|---|---|---|---|---|---|---|
| off | 3.42 / 3.38 | 12638 / 12780 | 18440 / 19334 | 66 / 60 | 74.6 / 76.8 | 177/226, 167/182 |
| `--decode-pipeline` | 3.36 / 3.35 | 13158 / 13266 | 20225 / 20267 | 58 / 57 | 73.7 / 74.2 | 149/176, 150/183 |

Served greedy text at c64/c128 already differs from run to run without the pipeline (about 167 of
192 outputs match), because the rung a request decodes on depends on arrival. c1 is identical.

`--lt-rung-algos` makes each routed rung time its own cuBLASLt algorithms instead of pinning the
widest rung's. It is off by default: Veena's digests are unchanged, but E4B's B=64 digest moves.
With it, step_bench B=32/48/64 goes from 4.01/4.29/4.50 to 3.84/4.07/4.36 ms.

Served, ctx 384, same client:

| | c1 TPOT ms | c64 | c128 | stream c64 aps | stream c64 TTFA p50/p90 ms |
|---|---|---|---|---|---|
| veena40_1, `--decode-pipeline` | 3.36 | 13307 | 20291 | 75.0 | |
| + `--lt-rung-algos` | | 13618-13693 | 20192-20308 | | |
| + varlen packed prefill | 3.36 | 13759 / 13726 | 20472 / 20434 | 76.9 / 71.5 | 132/174 |
| + prefill light | 3.35 | 13742 / 13818 | 20577 / 20601 | 76.2 / 76.3 | 139/171, 138/170 |

Tried and dropped:

* HeadNormRope in `light_pf`. It saved another ~0.1%, but its digests differ from the prefill
  object's HeadNormRope.
* Scheduler knobs:
  * `--pf-batch`, `--pf-interleave-adaptive` and `--pf-interleave 0/4096` gave no gain.
  * `--pf-defer-decode` gave c64 +2% but c128 -3%, and a TTFT hit.

With the recipe flags, the gate CER median is 0.005 (n=80). The step_bench digests are those of
veena40_1 at B=1/32/64/128, and E4B's are unchanged at B=1/64/128. In-model at B=128 ctx 384,
o_proj on cuBLASLt takes 14.5 us per layer, against 9.3-10 us in torch.

Voices are the checkpoint's speaker tags (`kavya`, `agastya`, `maitri`, `vinaya`).
Defaults: temperature 0.4, top_p 0.9, no repetition penalty (device sampling; a
penalty switches that request to host sampling).

## Chatterbox

```sh
python scripts/tts/chatterbox_prep.py $T3_HF          # Llama-shaped T3 checkpoint + voice rows
python scripts/tts/s3gen_export.py $S3GEN             # S3Gen weights + voices for the packet
PLOW_TTS_VOCODER_DIR=$S3GEN PLOW_EMIT_PREFILL_CUBLASLT=1 PLOW_NO_GLU_FUSE=1 \
  plowc --hf-dir $T3_HF --gpu h100 --arch sm_90a --max-ctx 2048 --emit devblob+cubin \
        --served-name chatterbox --out $CBX

plowrt serve --assets $CBX --port 8080
```

The T3 asset runs on the shared text mux as guided jobs (two slots per request for
guidance); `s3gen.pkt` renders audio on a render thread.

Each token is drawn from the pair's logits with the packet's `lm.*` chain (guidance weight,
repetition penalty, temperature, min_p, top_p). The host draws by default; `--cfg-device` /
`PLOW_CFG_DEVICE=1` draws on the device (`plow_sample_cfg` in `sample_sm120.cubin`: same
uniforms, per-slot penalty history, token written to both members; `--cfg-multistep` also runs
pairs in the multi-step quantum). It stays off because it lost served aps at c16..c64 (H100: the
host draw's gap paced T3 so utterances closed in batches; without it S3Gen renders batch smaller).
The repetition penalty applies once per distinct history token, as HF's
`RepetitionPenaltyLogitsProcessor` (gather / scatter); applying it per occurrence (before
2026-09-27) pushed repeated speech tokens away exponentially and ran 7/96 multilingual requests to
the 1000-token cap (reference: 0/96).

## Chatterbox Multilingual V3

```sh
python scripts/tts/chatterbox_mtl_prep.py $T3_HF      # T3 v3 checkpoint, voice rows, text frontend
python scripts/tts/s3gen_export.py $S3GEN --weights s3gen_v3.safetensors
PLOW_TTS_VOCODER_DIR=$S3GEN ... plowc --hf-dir $T3_HF ... --served-name chatterbox-mtl --out $MTL
curl -s localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"chatterbox-mtl","input":"你好，今天天气真不错。","voice":"default","language":"zh"}'
```

Same T3 shape and pipeline as English (the text table grows 704 -> 2454 rows); the multilingual
part is the text frontend, which is packet data. Requests pick the language with `language`
(ISO 639-1: ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh; default `en`,
unsupported -> 400). OpenAI's speech API has no language field (it infers it); `language` is the
common extension of OpenAI-compatible TTS servers.

Reference: upstream `resemble-ai/chatterbox` git (5de7a54; PyPI 0.1.7 only has v2),
`ChatterboxMultilingualTTS.from_local(..., t3_model="v3")`, with S3Gen weights from
`s3gen_v3.safetensors` (upstream `mtl_tts` still loads `s3gen.pt`; the two differ only in the HiFT
vocoder, 328 `mel2wav.*` tensors). V3 vs v2 (code): same architecture, tokenizer
(`grapheme_mtl_merged_expanded_v1.json`, 2454 ids; the repo's `mtl_tokenizer.json`, 2352 ids, is
unused) and conditioning; the alignment-stream analyzer (forced EOS on hallucination) is gone,
the repetition penalty default drops 2.0 -> 1.2, and the last speech token's audio is cut
(`lm.trim_tail_tokens` = 1). The V3 HF Space differs again (no capitalization, VAD-trimmed
reference audio, `text_preproc="NFKD,fullcase"`); plow follows the upstream package.

The frontend (`mtl_tts.punc_norm` + `MTLTokenizer.encode`) as rules (`text.rules`, one list per
language in `text.rules.lang.<code>`, `crates/plowrt/src/text/rules.rs`) over tables in the
packet's `text_tables.v1` metadata section: punc_norm (CJK sentence enders), lowercase, NFKD, the
language's rules, the `[lang]` prefix, spaces -> `[SPACE]`. zh: word segmentation by a CRF
(`segment_crf`, the pkuseg spacy_ontonotes model and merge dictionary, decoded exactly) then
Cangjie codes per glyph (`map_chars`); ja: pykakasi's kanji -> hiragana as a longest-match table
(`dict_longest`, values are the reference's `hiragana_normalize` output per segment); ko: strip
(NFKD already decomposes Hangul); he / ru: nothing (the reference skips diacritics / stress when
`dicta_onnx` / `russian_text_stresser` are absent, and the package does not depend on them).
Text ids equal the reference on 326/327 texts (prompts, 23-language demo sentences, edge cases,
~250 Wikipedia extracts incl. 30 zh + 30 ja); the one miss is pykakasi 2.3 re-emitting the
previous segment after an emoji / combining mark (a kakasi buffer bug plow does not copy).

H100, 2026-09-27 (branch d9ad0353 + this recipe, fresh clean-tree build):

| gate | plow | reference |
|---|---|---|
| T3 last-prefill logits, 47 prompts x 23 languages | rel-L2 cond median 0.016 / max 0.042, uncond max 0.023; top-1 46/47 (one 0.009 tie) | fp32; a bf16 run of the reference: 0.017 / 0.055 max |
| T3 greedy agreement, 60 steps | median 27 | bf16 reference vs fp32: median 23 |
| S3Gen v3 mel rel-L2 (11 cases) | 4.2e-6 .. 1.3e-5; batched == single | torch fp32 |
| Whisper CER, 24 prompts (en hi zh ja es fr ar de) | median 0.000 (stream + full) | median 0.005 |
| length, 96 requests (4 seeds x 24) | 358 s audio, none capped | 360 s |

Per-language median CER (plow / reference): ar 0.027 / 0.017, de 0 / 0, en 0 / 0, es 0 / 0,
fr 0.011 / 0.011, hi 0.114 / 0.108, ja 0 / 0.050, zh 0.028 / 0 (Whisper writes digits and
Latin loanwords for fr / hi / en; the same misses appear in both).

Served (`tts_bench.py --prompt-set chatterbox-mtl`, 24 prompts in 8 languages, ~3.7 s each):

| conc | full aps | full p50 latency | stream aps | stream TTFA p50 / p90 | failed (queue TTL 30 s) |
|---|---|---|---|---|---|
| 1 | 9.4 | 0.39 s | 3.1 | 260 / 261 ms | 0 |
| 8 | 29.8 | 0.96 s | 8.2 | 0.61 / 1.5 s | 0 |
| 16 | 33.4 | 1.8 s | 10.2 | 1.4 / 3.9 s | 0 |
| 32 | 30.7 | 3.1 s | 11.8 | 2.0 / 2.6 s | 0 |
| 64 | 33.7 | 5.8 s | 12.5 | 4.5 / 7.5 s | 0 |
| 128 | 32.1 | 13.9 s | 9.4 | 27 / 32 s | stream 59/256 |
| 200 | 32.6 | 21.9 s | 9.3 | 29 / 32 s | stream 200/400 |

The table above is before streaming windows and the codec readback fix (2026-09-27).

### Streaming windows

A stream renders only its new tokens: `s3gen.pkt` carries cached-prompt capacities
(`csynth.b{B}.t{T}`) whose CFM runs the item's own tokens plus the prompt's last 8 mel rows and
attends to the voice prompt's attention K/V (`AttentionF32` with a key prefix), which the
`prefill.v{V}` programs compute once at load from the prompt alone (the prompt rows do not attend
to the tokens, unlike the reference; the encoder still sees prompt + tokens). Each window is the
new tokens plus `stream.context_tokens` (8) of left context; its NSF source continues the stream's
harmonic phase at the seam (`phase` / `seam` / `next_seam` tensors), and the render crossfades
`stream.fade_samples` as before. Due windows share one launch; while every due window has
`render.slack_ms` (600) of audio buffered, or up to `render.window_hold_ms` (500) while most live
streams are not due, the launch waits for more (wider launches cost less per token).
Each launch fits its windows to one capacity length: of the cached capacities' frame counts it
takes the one delivering the most new tokens per unit of render cost (`render.launch_frames` 370 +
batch x (frames + `render.item_frames` 4), the H100 cost of a cached render in frames) and clips
every window to it; the rest of a clipped window's tokens go in a later launch. Before, a launch
padded every window to its longest one and rounded up to the next capacity (33 frames ran as 64):
at c200 only 34% of the rendered capacity frames became delivered audio (56% fitted).
With more live streams than one launch holds (the device is the bottleneck), a started stream
waits for a window that fills the widest capacity at the full batch (`b64.t64`: 53 new tokens
instead of `stream.chunk_tokens` 25), the lowest render cost per token: stream c128 51.5 -> 53.7,
c200 49.3 -> 52.7 aps (TTFA p50 9.9 -> 8.7 s), 200-request streamed CER median 0.000 either way.
`PLOW_TTS_STREAM_WINDOWS=0` restores prefix re-renders. Each CFM step is 4 programs (`CFM_PARTS`):
the LM's decode launches get in only between programs.

Gates (H100, 2bcf79b9 + this change): `scripts/tts/s3gen_cached_check.py` (packet vs a torch
implementation of the same computation) mel rel-L2 5e-6..1.2e-5. Whisper CER, 96 streamed
requests (4 x the 24-prompt set), windows vs prefix re-render: median 0.000 / 0.000; per
language ar 0.027 / 0.027, de 0 / 0, en 0 / 0, es 0 / 0, fr 0.011 / 0.011, hi 0.187 / 0.175,
ja 0.017 / 0.000, zh 0 / 0 (reference hi 0.108, ja 0.050); left context 16 / 32 tokens gave
hi 0.120 / 0.133, ja 0.017 / 0.000 at 27-29 vs 25 aps (c64) — within the run-to-run spread, so 8
stays. Seam spectral flux at window boundaries equals the mid-chunk control (p50 1.27 / 1.26).
English Chatterbox CER 0.000 (stream and full), Veena 0.008, Qwen3-ASR WER 3.913%.

Served, same client (MTL, 24 prompts in 8 languages, ~3.7 s each):

| conc | stream aps before | stream aps | TTFA p50 / p90 | failed | full aps |
|---|---|---|---|---|---|
| 1 | 3.1 | 7.2 | 165 / 168 ms | 0 | 11.5 |
| 16 | 10.2 | 31.4 | 0.86 / 1.0 s | 0 | |
| 32 | 11.8 | 32.0 | 1.6 / 1.7 s | 0 | |
| 64 | 12.5 | 31.6 | 3.1 / 4.5 s | 0 | 33.6 |
| 128 | 9.4 | 42.2 | 8.2 / 10.2 s | 0 (was 59/256) | |
| 200 | 9.3 | 37.3 | 15 / 16 s | 0 (was 200/400) | 35.1 |

English Chatterbox: stream c1 7.2 aps (TTFA 165 ms), c16 29.9, c64 33.1; full c64 34.3.
Closed-loop clients past ~64 streams queue at the 128-slot T3 rung (a CFG pair per request), so
TTFA there is mostly queueing.

Standalone GPU time (packet_bench): `synth.b32.t96` 1.49 s (0.49 ms per token), `csynth.b32.t64`
0.68 s (0.33 ms per token incl. 8 context rows); a CFM step 137 vs 59 ms. Served, the render and
the T3 decode launches contend for the device (T3 ~80 ms per token per stream at c64 vs a 6.5 ms
standalone step at 128 rows).

Stock reference (fp32, one request at a time): 1.31 aps, RTF 0.69 (T3 21 ms / token).

Fitted windows (5eea79c4 + this change, same client; 96-request streamed CER median 0.000 at c1
and at c16, per language ar 0.028, de 0, en 0, es 0, fr 0.011, hi 0.145, ja 0.017, zh 0):

| conc | stream aps before | stream aps | TTFA p50 / p90 | failed |
|---|---|---|---|---|
| 1 | 7.2 | 7.1 | 165 / 165 ms | 0 |
| 16 | 31.4 | 35.1 | 0.80 / 0.89 s | 0 |
| 32 | 32.0 | 35.8 | 1.4 / 1.8 s | 0 |
| 64 | 31.6 | 42.6 | 1.9 / 2.7 s | 0 |
| 128 | 42.2 | 49.9 | 4.7 / 6.0 s | 0 |
| 200 | 37.3 | 52.2 | 9.4 / 10.4 s | 0 |

English Chatterbox: stream c16 32.8, c64 43.4 aps (was 29.9 / 33.1), CER 0.000. TTS-only
`scripts/voice/call_sim.py` (3 turns per call, 3 s user turns, calls ramped over 10 s): 64 calls
TTFA p50 / p95 658 / 995 ms, no underrun; 128 calls 1.9 / 2.7 s, underrun p95 623 ms (138 of 384
turns over 100 ms); 200 calls 2.2 / 3.5 s, underrun p50 / p95 153 / 1009 ms (331 of 600). The codec's
per-launch trace (`RUST_LOG=plowrt::tts::codec_launch=debug`: jobs, frames, capacity, H2D / run /
D2H us) put the render at 89% of wall time at c200 with host copies under 0.3%.

### S3Gen per op

Per-op marginal GPU time (`packet_bench PACKET vocoder.synth ROLE N --sweep`: instruction i costs
us(cap i+1) - us(cap i)) against a roofline of max(FLOPs / 989 TFLOP/s bf16, bytes / 3.35 TB/s),
`csynth.b32.t64` (64 CFG items x 136 rows = 8704 rows), one CFM step (programs 5-8, 56 transformer
blocks + 14 resnets). Timings came from a shared GPU (other agents' unleased processes), so read
the ratios, not the last digit.

| op | shape (rows x K x N) | impl | n | GFLOP | roofline us | measured us | % roof | % step |
|---|---|---|---|---|---|---|---|---|
| Conv1dF32 qkv | 8704x256x1536 | split-bf16 x3 | 56 | 383 | 1069 | 8957 | 12 | 23.3 |
| Conv1dF32 ff1 +gelu | 8704x256x1024 | split-bf16 x3 | 56 | 256 | 762 | 8308 | 9 | 21.6 |
| Conv1dF32 ff2 +res | 8704x1024x256 | split-bf16 x3 | 56 | 256 | 911 | 6902 | 13 | 17.9 |
| Conv1dF32 out +res | 8704x512x256 | split-bf16 x3 | 56 | 128 | 605 | 4389 | 14 | 11.4 |
| LayerNormF32 | 8704x256 | f32 | 141 | 1.6 | 750 | 3057 | 25 | 7.9 |
| AttentionF32 (prefix 306) | b64 q136 kv136+306 h8x64 | f32 tc | 56 | 441 | 1234 | 1941 | 64 | 5.0 |
| Conv1dF32 resnet k3 | 8704x768x256 | split-bf16 x3 | 26 | 89 | 145 | 2345 | 6 | 6.1 |
| Conv1dF32 resnet res | 8704x256x256 | split-bf16 x3 | 12 | 14 | 97 | 683 | 14 | 1.8 |
| GatherRows (time emb add) | 8704x256 | f32 | 14 | 0 | 74 | 620 | 12 | 1.6 |
| Unary mish | 8704x256 | f32 | 29 | 0 | 154 | 469 | 33 | 1.2 |
| **step** | | | | 1593 | 5888 | 38498 | 15 | |

The encoder (program 0: conformer over prompt + tokens, 39 ms) and HiFT (program 41: 53 ms, its
k3..k11 convs at 8-15% of roofline, snake unaries at 27%) are 13% of the render; the 10 CFM steps
are 87%. Per delivered audio second a full `csynth.b32.t64` launch (53 new tokens per window) costs
10 ms of GPU, `b64.t32` (21 new tokens) 13 ms.

The GEMMs run at 9-14% of the bf16 roofline; standalone (`speech_f32_op_test --bench-cfm`) qkv
takes 185 us, of which 3 us is the MMAs (dropping them: 182 us), 15 us the global loads, 67 us the
epilogue, and 85 us the k-loop skeleton alone (split, shared stores, barriers, ldmatrix), with the
ldmatrix results consumed by the next instruction (8 warps per SM, one block). A single bf16 pass
(flag bit 16, tried) saved 7-22% per GEMM and 9-10% per render but moved the mel 3e-3 rel-L2 from
the torch reference (was 1e-5), so it was dropped.

The sweep above ran the CFM program on zeroed inputs, where the key lengths are 0 and attention
skips its keys. With the encoder run first (`PB_PRE=csynth.b32.t64.0 packet_bench ... --sweep`),
program 1 of `csynth.b32.t64` (16 transformer blocks) takes 16.8 ms, not 9.9 ms, and attention
is its largest op:

| op (program 1, real inputs) | n | mma.sync us | wgmma us |
|---|---|---|---|
| AttentionF32 (3xTF32, 128-query tiles) | 16 | 5899 | 6543 |
| Conv1dF32 qkv 256->1536 | 16 | 2594 | 1885 |
| Conv1dF32 ff1 256->1024 +gelu | 16 | 2377 | 1821 |
| Conv1dF32 ff2 1024->256 +res | 16 | 1939 | 1606 |
| Conv1dF32 out 512->256 +res | 16 | 1328 | 1082 |
| LayerNormF32 | 40 | 880 | 1350 |
| Conv1dF32 resnet k3 | 9 | 848 | 727 |
| **program** | | 16791 | 15938 |

Attention runs one block per (item, head, 128-query tile): 136 query rows take two tiles, the
second 8 rows full, so it does 1.9x the needed work.

Warpgroup MMA (flags bit 17, sm_90a): the split-bf16 3-pass product on `wgmma.m64nNk16` from
128-byte-swizzled bf16 operand stages (BK 64, double-buffered at N = 64), f32 loads of the next
k-tile in flight during the MMAs, and an epilogue staged in shared memory and written with
`cp.async.bulk`. It covers pointwise convs and tap-major k-tap convs with 64-channel groups
(implicit im2col). Tiles are 128x128, or 128x64 where a 128-wide grid would leave a short last
round. `wgmma` must be inlined into the interpreter kernel (in a called function ptxas serializes
it, C7510), so it is dispatched from `plow_exec` with 12 instantiations (ptxas ~2 h for the speech
cubin). Standalone (`speech_f32_op_test --bench-cfm`, 8704 rows):

| GEMM | mma.sync us | wgmma us |
|---|---|---|
| qkv 256->1536 | 165 | 112 |
| ff1 256->1024 +gelu | 141 | 103 |
| ff2 1024->256 +res | 124 | 90 |
| out 512->256 +res | 78 | 53 |
| resnet res 320->256 | 60 | 41 |
| resnet c1 k3 320->256 | 124 | 88 |

Numerics are unchanged (mel rel-L2 5e-6 to 1.2e-5 vs the torch reference; same pass order). The
render is 5.6% faster (`packet_bench --seq`: `csynth.b32.t64` 682 -> 644 ms, `b64.t32` 701 -> 663
ms): the GEMMs lose 25%, but in the same cubin LayerNorm and attention got slower (register
allocation of the shared interpreter kernel). Served, same client, same lease (MTL stream aps):

| conc | before | wgmma | TTFA p90 |
|---|---|---|---|
| 64 | 43.4 | 44.2 | 2.6 s |
| 128 | 50.5 | 52.5 | 5.6 s |
| 200 | 52.8 | 53.9 | 10.4 s |

c1 7.1, c16 35.8, c32 40.2 aps; full c64 37.9. CER median 0.000 (96 at c1, 200 at c200), per
language as before; English c16 / c64 35.4 / 44.2 aps, CER 0.000; Qwen3-ASR WER 3.913%; Veena CER
0.008.

`Codec::set_yield` splits each render into segments of `render.yield_programs` programs (4: one
CFM step) and calls the hook between them, so a co-scheduler can hand the device to T3 mid-render.
Without a hook a render stays one launch.

Attention with packed query tiles: a block's 8 warps take 8 consecutive 16-row query tiles of an
item's (head, tile) list, so 136 query rows cost 8.5 warp tiles instead of two 128-row tiles; the
at most 2 (q >= 128 rows, 32-key K/V tiles) or 5 (16-key tiles) heads a run touches are staged side
by side. Same 3xTF32 products and softmax order. The LayerNorm rows kernel is back to 2 rows per
warp (8 rows ran slower in the packet). Standalone (`--bench-cfm`, b64, 306-row prefix): q136 469
-> 286 us, q72 212 -> 176 us. `csynth.b32.t64` program 2 with real inputs: attention 4972 -> 3531
us, LayerNorm 1065 -> 820 us, program 11.85 -> 9.99 ms. Render (`--seq`): `b32.t64` 644 -> 556
ms, `b64.t32` 663 -> 603 ms, `b8.t32` 173 -> 157 ms. Served, same client, same lease (MTL stream
aps; the previous row is the wgmma build, rerun):

| conc | before | packed attention | TTFA p50 / p90 |
|---|---|---|---|
| 64 | 47.1 | 49.5 | 1.7 / 2.4 s |
| 128 | 55.7 | 56.5 | 3.9 / 5.3 s |
| 200 | 57.8 | 61.3 | 7.5 / 8.7 s |

c1 7.3, c16 37.7 aps. Mel rel-L2 5e-6 to 1e-5; CER median 0.000 (96 at c1, 200 at c200), per
language unchanged; English c64 49.9 aps, CER 0.000; Qwen3-ASR WER 3.913%; Veena CER 0.008, c64
57.7 aps. The same build served at 53.9 and 57.8 aps at c200 in two runs, so single A/B runs
carry about 7% noise.

Served throughput at c200 is GPU-bound (render ~80% of the device, T3 decode the rest), so it
follows the render cost: measured with `--n 800` (two runs each, `steady_aps` = audio per second
over the middle 60% of the run) the wgmma build served 55.4 aps and packed attention 63.3.
Decode projections on cuBLASLt (`PLOW_EMIT_DECODE_CUBLASLT`, rungs of 48+ rows) cut the T3 step
at 128 rows from 6.47 to 5.11 ms but served no faster (c200 61.4 aps), so the recipe keeps the
GEMV rungs.

Encoder relative-position bias read in place (AttentionF32 flags bit 2: the bias is a table
`[q_rows][heads][bias_head_stride]` and score `(h, r, j)` reads entry `kv_rows - 1 - r + j`; bits
8-15: key lengths per group of that many heads). The conformer's grouped 1x1 relpos convolution
now feeds the attention directly; the diagonal `CopyColsF32` into `[b*8][t][t]` and the key-mask
add (at b64.t32 9.5 and 7.4 ms of the 59 ms encoder) are gone. The softmax runs in log2 units on
`exp2f` (log2 e folded into the scale; standalone attention -4 to -11%, same error), and the wgmma
GEMM stages its bias in shared memory with the tile's first k-tile. Issuing the next tile's loads
before the epilogue (standalone qkv 112 -> 92 us) spilled registers in the interpreter kernel and
made every op slower (render +31%), so it is not in.

Render (`--seq`): `b64.t32` 602 -> 565 ms (encoder 59 -> 41 ms, CFM 496 -> 478), `b64.t64` 952 ->
894, `b32.t64` 557 -> 528. Served c200 (`--n 800`, two runs): 61.0 -> 66.7 steady aps (60.6 /
61.4 -> 64.5 / 65.8 aps), TTFA p50 7.9 -> 7.4 s; c128 61.6 aps. Mel rel-L2 5e-6 to 1.2e-5; CER
median 0.000 (96 at c1, 200 at c200); English c64 54.6 aps, CER 0.000; Qwen3-ASR WER 3.913%;
Veena CER 0.008, c64 57.9 aps.

Queries of 4 to 7 tiles (the 72-row window) run 24-key K/V tiles with 3 heads side by side
instead of 32-key tiles with 2. The encoder's grouped 1x1 relpos convolution (`[t][b*8][2t-1]`
per head, 64 input channels per group) runs on warpgroup MMA: the tile decode walks (row tile,
group), and the A operand steps by the full input row. The position table's per-head stride is
padded to a multiple of 4 (the wgmma path needs 4-aligned rows). The grouped case is its own
template instance so the ungrouped CFM GEMMs keep their registers (a shared runtime-grouped path
made them 3-20% slower). Standalone relpos `b8 512 -> 6048`: 3xTF32 2.25 ms -> 0.57 ms.

Render (`--seq`): `b64.t32` 565 -> 544 ms (encoder 41 -> 30 ms, CFM program 13.6 -> 13.3 ms),
`b64.t64` 894 -> 885, `b32.t64` 528 -> 523, `b8.t32` 157 -> 150. Served c200 (`--n 800`, two runs
each, same lease): the runtime-grouped variant served 65.8 -> 67.5 steady aps against the
previous build, and this build 66.4 -> 67.8 (67.0 / 67.6 aps) against that variant; TTFA p50 7.2
s. c128 67.9 aps, c64 55.5, c16 40.3, c1 7.6. Mel rel-L2 5e-6 to 1.1e-5; CER median 0.000 (96 at
c1, 200 at c200), per language unchanged; English c64 55.0 aps, CER 0.000; Qwen3-ASR
WER 3.913%; Veena CER 0.008, c64 57.9 aps.

Two extra cached capacities (`(64, 36)`, `(64, 68)`) cut render per new token by 13% and 5%, and
the fit picked `(64, 36)` for nearly every c200 launch, but served no faster (66.0 vs 66.3 steady
aps), so the recipe keeps the 32/64 ladder.

Pre-split weights (Conv1dF32 flags bit 18): devgen appends to an ungrouped split-bf16 conv's f32
weights the bf16 hi / lo halves of its B operand, per 64-column k-tile and in the 128-byte swizzle
wgmma reads (`pipeline::wgmma_weight_split`), and the wgmma GEMM copies a tile's halves with
`cp.async` into two B buffers, the next k-tile's under this one's MMAs, instead of loading and
splitting f32 rows per thread (same products, bit-identical). An ungrouped wgmma conv without the
image takes the other paths. The CFM resnet's LayerNorm, Mish and time-embedding add run as one
LayerNormF32 (flags bits 4-7: a parameterless activation after the affine; t4: a `[feat]` row
added after it), which drops the UnaryF32 and the accumulating GatherRowsF32. Standalone
(`--bench-cfm`): qkv 111 -> 98 us, ff1 99 -> 91, ff2 89 -> 81, out 52 -> 48.

The interpreter's stack grows from 4632 to 5272 bytes and its attention runs 6-8% slower in the
packet (the attention functions' code is unchanged; the interpreter body spills more), which eats
part of the GEMM gain. Render (`--seq`): `b64.t32` 544 -> 528 ms, `b32.t64` 524 -> 494, `b8.t32`
150 -> 143; HiFT 46 -> 42 ms. Served c200 (`--n 800`, six runs each, same lease): 67.6 -> 68.1
steady aps, TTFA p50 7.2 -> 7.1 s; c128 66.7 aps, c64 54.7, c16 41.8, c1 7.7. Mel rel-L2 5e-6 to
1.1e-5; CER median 0.000 (96 at c1, 200 at c200), per language unchanged; English c64 57.0 aps,
CER 0.000; Qwen3-ASR WER 3.913%; Veena CER 0.008, c64 58.0 aps.

The B copy of that path takes one per-thread source pointer per tile and 32-bit offsets (it was
recomputing 64-bit addresses per copy): the interpreter stack goes from 5272 to 4920 bytes, the
spill load it had put on AttentionF32 (6-8% in the packet) is gone, and the GEMM gain shows in the
render. Render (`--seq`): `b64.t32` 527 -> 500 ms, `b32.t64` 496 -> 479 (CFM program 9.65 -> 9.13
ms). Served c200 (`--n 800`, six runs each over two leases, alternating order): 68.7 -> 71.7 steady
aps (67.5 to 69.6 -> 70.3 to 73.0), TTFA p50 7.0 -> 6.8 s; c128 73.5 steady aps, c16
42.7, c1 8.0. Mel rel-L2 5e-6 to 1.1e-5; CER median 0.000 (96 at c1, 200 at c200), per language
unchanged; English c64 58.4 aps, CER 0.000; Qwen3-ASR WER 3.913%; Veena CER 0.008, c64 57.8 aps.

The speech object compiles only the FP32 ops its packet's sidecar programs use: devgen writes
their set into `plow_config.h` (`PLOW_SPEECH_OPS`, bit `op - 163`; absent = every arm) and the
other arms trap. ptxas time grows superlinearly with the arms called from the interpreter kernel
(whole-program calling conventions; an ABI boundary would serialize the inlined wgmma): with every
arm the speech cubin took 4-7 h of ptxas (the 3xFP16 attention alone added hours), with one
packet's set 8 min (Chatterbox), 7 min (Veena), 1.5 min (Qwen3-ASR).

## Concurrency

Speech requests take the LLM path on each model's mux: packed prefill (several requests' prompt
rows in one launch; the host overlay rows, CFG pairs' conditional and unconditional rows and
the position bases staged per launch row) and, in the same tick, the decode launch. All three
packets carry the packed-prefill contract (hd64/128 attention, `EmbedOverlayBf16` /
`EmbedPosBf16`). The unified token batch (decode rows inside the prefill launch) stays off for
them: it is qualified on hd256/512 only, and overlay rows would need its row order.

Packed prefill of 8 x 60-row prompts, one launch vs serial (H100): Veena 10.0 vs 50.0 ms,
Chatterbox T3 7.1 vs 29.4 ms, Qwen3-ASR 7.5 vs 38.9 ms. A CFG pair's two members are
separate packed requests; the request's first token is drawn once both rows are in.

Qwen3-ASR's audio encoder batches too: queued utterances share one `encoder.pkt` launch
(`packed.{chunks}.{stage}`, 4..192 100-frame chunks), back to back by chunk. `GroupedAttentionF32`
flag 8 keeps each utterance's attention inside its own windows (a group table), and
`DenseGemmF32` flag 128 sums each row's K as that utterance's single-utterance capacity splits
it (per-row reference rows), so every transcript is bit-identical to the request run alone; a
lone utterance runs its single-utterance capacity. The encoder's ops are fused 12 per program.
H100, 292 requests: RTFx 92 / 252 / 397 / 393 / 394 before, 98 / 505 / 670 / 702 / 776 after
at c1 / 16 / 32 / 64 / 128; p50 63 -> 59 ms at c1, 981 -> 532 ms at c64; WER 3.913% at every
level. c1 is decode-bound (~2.3 ms per token).

Admission: requests past the slots queue (4 engine batches of ingress; the ASR front bounds
its own requests in flight at 256 and waits for mux room instead of answering 429). A CFG
request takes two slots, so the 128-slot rung serves 64 Chatterbox requests; the rest queue
(`DECODE_RUNG_MAX` = 128 is the packet format's decode/prefill boundary).

## Long inputs and real-time admission (codec-LM: Veena, Orpheus)

`input` takes up to 4096 characters (`tts::MAX_INPUT_CHARS`, ~7 minutes of speech); longer is a
400. A request's token budget is `min(chars x tokens.per_char_frames x 7 + 21, tokens.max_new_cap)`;
an input past the length where the cap binds (`SpeechContract::segment_chars`: Veena 151, Orpheus
141 characters) is spoken as segments of at most that length: whole sentences (`. ! ? … । ॥`, line
breaks) packed greedily, a longer sentence split after clause marks, then between words, into
about equal parts (a lone tail word made Orpheus go mute whatever the seed). Segments
generate in order with the same voice (seed + k when a seed is given) and stream back to back; a
whole response is their concatenation. At a join, silence beyond a 0.5 s pause is dropped (Orpheus
segments often open with 1-2 s of silence), a segment that keeps silent for 3 s after speaking
ends, and one silent for 3 s from its start, or droning for 2 s (a hum or "rrrr" whose 53 ms
envelope varies under 15%; speech measures >= 36%), is generated once more with another seed,
then split in two (Orpheus goes mute on some long run-on segments, persistently across seeds, and
not on their halves), down to 40 characters. Genuine Orpheus segments open with up to 2.9 s of
silence, so mute detection stays at 3 s. A segment still without speech (or without frames) after
that fails the request (500, or a terminal stream error), never a 200 missing part of the text. A streamed request's LM drain
waits for the emitter's verdict on each segment attempt before it starts the next; cuts and
verdicts carry the (segment, attempt) they were raised for.
A request of one segment is unchanged. `max_tokens` applies per segment.

Profiles (`devgen::tts`): Veena's cap 700 -> 1400 tokens (700 cut 8.3 s, inputs from ~110
characters); Orpheus 1.3 -> 1.8 frames per character (its voices speak down to ~8 characters/s, so
1.3 clipped the slow tail) and cap 1200 -> 1800. Tokens before the old limit are unchanged.

Every codec-LM request shares the model's decode steps, so each one admitted slows the rest
(L40S, Llama-3.2-3B BF16: 9 ms per token at B=1, 13.8 at B=32 against 85 ms of audio per 7-token
frame). `tts::realtime` admits a request only while the step projected at one more generating
request keeps every playing stream ahead of playback: each stream's sent-but-unplayed audio covers
its remaining frames' deficit (`frames x (7 x step - 85 ms)`) with 0.25 s to spare (0.55 s with
a segment still to start, for its prefill and first window), and a new stream can bank its own
deficit by holding its first audio at most 1.5 s. A segment's frames are projected at the p95
frames per character of the last 256 finished segments, capped by its budget (at its budget until
16 have finished; a segment past its projection at its budget): Orpheus runs ~15% of segments to
the budget (mean 1.23, p90 1.78 of 1.8 frames per character), and mean-based projections let 5-24%
of its admitted streams underrun in soak. A multi-segment stream also banks its first join before
its first audio (TTFA ~580 vs ~330 ms alone). The step is fit online from the streams' own
frame times: a width with 8+ samples predicts its own measured step, an unmeasured one the line
in the width but never below a narrower measured step. A width past a decode-ladder rung runs on
the next rung (L40S: 13.8 ms at 32 rows, 18.5 ms at 33-64), which the line missed and which let
Orpheus streams underrun just past 32 generating requests. Admission grows at most 8 requests
beyond the widest measured width, so a cold burst admits 8 until the first samples arrive. The
per-frame bookkeeping is O(1) atomics on the request's own state (a step sample every 4 frames
takes the lock); with `PLOW_TTS_REALTIME=0` nothing is tracked. Requests wait in arrival order; past
`PLOW_TTS_ADMIT_WAIT_MS` (6000) a request gets 429 with `Retry-After` (the projected time to the
next stream finishing). `PLOW_TTS_REALTIME=0` admits everything (the previous behavior).

## Sessions and streaming

`X-Request-Id` / `X-Session-Id` (`docs/runtime/sessions.md`): a session's requests resume the KV
of its previous one (Chatterbox: the voice conditioning rows of both CFG members; Veena: the
prefix and speaker tokens; Qwen3-ASR: the prompt and completed audio windows). Qwen3-ASR streams
over HTTP (`stream=true` SSE, `append=true` / `final=true` uploads in a session) and the
WebSocket, both doing O(new audio) work per partial. Idle sessions go after
`PLOW_SESSION_TTL_MS` (60 s); a live request always evicts retained KV it needs.

## Model names

`plowc --served-name NAME` writes `served_name` into `weights.json`; `plowrt serve` registers
the bundle under it (default: the HF repo id for a hub-cache `--hf-dir`, else the network
slug). The recipes set `veena`, `chatterbox` and `qwen3-asr`.

## Co-serving

One `plowrt serve --assets $ASR --assets $VEENA --assets $CBX` serves all three from one GPU:
each model has its own mux and KV, and the device changes hands at launch boundaries
(`--co-sched free|rr`). `scripts/tts/plow_coserve_probe.sh` runs each model alone, the
switch latency (`switch_bench.py`) and all three under concurrent load in one lease.

H100, 2026-09-27, reproduced assets, one process:

| | solo (same process) | mixed, `--co-sched free` | mixed, `--co-sched rr` |
|---|---|---|---|
| Qwen3-ASR c4 (RTFx, p50) | 168.6, 134 ms | 21.6, 1323 ms | 25.3, 738 ms |
| Veena stream c8 (audio s/s) | 14.76 | 9.46 | 11.73 |
| Chatterbox full c4 (audio s/s) | 12.84 | 2.82 | 4.86 |

Gates hold co-served (WER 3.913%, CER 0.026 / 0.000); switching between models costs < 0.5 ms
(back-to-back requests alternating across the three vs one model). Under mixed load the GPU is
saturated: persistent cooperative kernels take the whole device, so models alternate, and the
solo-normalized shares sum to ~1.0 (`free`) and ~1.3 (`rr`). Use `rr` when co-serving speech.
Tried and reverted (no gain): single-step ticks while a co-tenant waits, and holding the turn for
a whole encoder/codec/vocoder sequence (worse: a 166 ms S3Gen render then blocks everyone).

### Voice agent co-serve

Qwen3-ASR + Gemma 4 E4B + Chatterbox-MTL in one `plowrt serve`: `--co-sched deadline`,
`PLOW_LIVE_CTX_MODELS=qwen3-asr=768,chatterbox-mtl=512`, `--session-ttl-ms 60000`, LLM assets
last. `scripts/voice/serve_voice_agent.sh` launches it under `gpulease` + `timeout`, runs
`scripts/voice/call_sim.py` per call count and prints the SLO table
(`scripts/voice/slo_table.py`):

```sh
# assets: campaign.py build recipes/infervisor/{qwen3-asr,gemma-4-e4b,chatterbox-mtl}/sm90a-h100-tp1.toml --out $VA_BUILD_ROOT/<model>
VA_BUILD_ROOT=<dir> MANIFEST=<asr clips.json> PY=<python with aiohttp numpy soundfile> \
  scripts/voice/serve_voice_agent.sh calls <resdir> 10 20 30   # or: serve <resdir>
```

Memory and scheduling rationale and results: [gemma4-e4b-h100.md](gemma4-e4b-h100.md) "Voice
co-serving".

## Validation tools (`scripts/tts/`)

| tool | use |
|---|---|
| `veena_ref.py`, `chatterbox_ref.py`, `t3_ref.py` | HF / vLLM / fp32 references |
| `asr_check.py` | Whisper round-trip CER gate |
| `tts_bench.py` | TTFA / RTF / audio s/s client for any `/v1/audio/speech` server |
| `plow_speech_probe.sh`, `vllm_speech_probe.sh` | plowrt vs vLLM+SNAC speech servers, same client, one lease each |
| `snac_export.py`, `s3gen_export.py` | exports the codec lowerings read (`PLOW_TTS_CODEC_DIR`, `PLOW_TTS_VOCODER_DIR`) |
| `s3gen_packet_check.py`, `crates/plowrt/examples/codec_check.rs` | codec packet numerics vs torch / the reference decoders |
| `s3gen_cached_check.py` | `s3gen.pkt` cached-prompt capacities (prefill + windows, NSF phase carry) vs torch |
| `crates/plowrt/examples/packet_bench.rs` | per-program GPU time; with `PLOW_DEBUG_MAX_INST` per-op costs |
| `snac_check.py`, `sample_kernel_bench.py` | reference SNAC library and sampler numerics/latency |
| `crates/plowrt/examples/t3_check.rs` | T3 logits gate vs `t3_ref.py` / `t3_mtl_ref.py` (per-prompt `language`) |
| `chatterbox_mtl_ref.py`, `t3_mtl_ref.py`, `mtl_prompts.py` | Multilingual V3 reference (timing, wavs, fp32 / bf16 T3), prompt sets |
| `mtl_text_ref.py`, `crates/plowrt/examples/t3_text_check.rs` | text frontend gate: packet rules + tokenizer vs the reference ids (CPU) |

Every GPU run goes through `perf-data/tools/gpulease`.
