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

200 calls at ~30% speaking need ~60 real-time streams: full synthesis tops out at ~33 aps (0.55x
the need), streaming at ~12.5 aps (0.2x). Streams re-render their whole prefix every chunk
(`stream.chunk_tokens`), so S3Gen spends ~2.7x the whole-utterance GPU per audio second, and past
64 streams renders queue behind decode and requests hit the 30 s queue TTL.

Stock reference (fp32, one request at a time): 1.31 aps, RTF 0.69 (T3 21 ms / token).

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

## Validation tools (`scripts/tts/`)

| tool | use |
|---|---|
| `veena_ref.py`, `chatterbox_ref.py`, `t3_ref.py` | HF / vLLM / fp32 references |
| `asr_check.py` | Whisper round-trip CER gate |
| `tts_bench.py` | TTFA / RTF / audio s/s client for any `/v1/audio/speech` server |
| `plow_speech_probe.sh`, `vllm_speech_probe.sh` | plowrt vs vLLM+SNAC speech servers, same client, one lease each |
| `snac_export.py`, `s3gen_export.py` | exports the codec lowerings read (`PLOW_TTS_CODEC_DIR`, `PLOW_TTS_VOCODER_DIR`) |
| `s3gen_packet_check.py`, `crates/plowrt/examples/codec_check.rs` | codec packet numerics vs torch / the reference decoders |
| `crates/plowrt/examples/packet_bench.rs` | per-program GPU time; with `PLOW_DEBUG_MAX_INST` per-op costs |
| `snac_check.py`, `sample_kernel_bench.py` | reference SNAC library and sampler numerics/latency |
| `crates/plowrt/examples/t3_check.rs` | T3 logits gate vs `t3_ref.py` / `t3_mtl_ref.py` (per-prompt `language`) |
| `chatterbox_mtl_ref.py`, `t3_mtl_ref.py`, `mtl_prompts.py` | Multilingual V3 reference (timing, wavs, fp32 / bf16 T3), prompt sets |
| `mtl_text_ref.py`, `crates/plowrt/examples/t3_text_check.rs` | text frontend gate: packet rules + tokenizer vs the reference ids (CPU) |

Every GPU run goes through `perf-data/tools/gpulease`.
