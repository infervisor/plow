# TTS on NVIDIA (sm_90a)

Veena (`maya-research/Veena`) and Chatterbox (`ResembleAI/chatterbox`, English)
compile to plow packets and serve as OpenAI `POST /v1/audio/speech` from
`plowrt serve`. Design: [24 — TTS pipelines](../arch/24-tts-pipelines.md).
Every stage is a plowc packet; plowrt ships no model-specific code or native library.

## Veena

```sh
# SNAC export (torch), then LM packet + codec.pkt + paired sm_90a objects
python scripts/tts/snac_export.py $SNAC
PLOW_TTS_CODEC_DIR=$SNAC PLOW_TTS_PROFILE=veena PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 \
  PLOW_EMIT_PREFILL_CUBLASLT=1 PLOW_NO_GLU_FUSE=1 PLOW_FUSE_KV_HNR=1 \
  plowc --hf-dir $VEENA --gpu h100 --arch sm_90a --max-ctx 2048 \
        --emit devblob+cubin --out $ASSETS

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
  plowc --hf-dir $T3_HF --gpu h100 --arch sm_90a --max-ctx 2048 --emit devblob+cubin --out $CBX

plowrt serve --assets $CBX --port 8080     # served under the asset directory name
```

The T3 asset is not a text model: `serve` starts a guided speech worker for it
(its own engine; two slots per request for guidance; `s3gen.pkt` renders audio) and routes
`/v1/audio/speech` requests whose `model` is the directory name to it.

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
| `crates/plowrt/examples/t3_check.rs` | T3 logits gate vs `t3_ref.py` |

Every GPU run goes through `perf-data/tools/gpulease`.
