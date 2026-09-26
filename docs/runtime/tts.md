# TTS on NVIDIA (sm_90a)

Veena (`maya-research/Veena`) and Chatterbox (`ResembleAI/chatterbox`, English)
compile to plow packets and serve as OpenAI `POST /v1/audio/speech` from
`plowrt serve`. Design: [24 — TTS pipelines](../arch/24-tts-pipelines.md).
Measured on H200 (the numbers do not transfer to other parts).

## Veena

```sh
# packet + paired sm_90a objects + the SNAC codec object
PLOW_TTS_PROFILE=veena PLOW_EMIT_PREFILL_CUBLASLT=1 PLOW_NO_GLU_FUSE=1 PLOW_FUSE_KV_HNR=1 \
  plowc --hf-dir $VEENA --gpu h200 --arch sm_90a --max-ctx 2048 \
        --emit devblob+cubin --out $ASSETS
# codec weights (torch; like quantize_fp8.py)
python scripts/tts/snac_prep.py $ASSETS/codec/snac24k.bin

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
plowc --hf-dir $T3_HF --gpu h200 --arch sm_90a --max-ctx 2048 --emit devblob+cubin --out $CBX
python scripts/tts/s3gen_prep.py $CBX/codec/s3gen.bin --voice-out $CBX/codec/voices/default.bin
# build runtime/nvidia/s3gen into $CBX/codec/libplow_s3gen.so (runtime/nvidia/s3gen/build.sh)

plowrt serve --assets $CBX --port 8080     # served under the asset directory name
```

The T3 asset is not a text model: `serve` starts a Chatterbox worker for it
(its own engine; two slots per request for guidance) and routes
`/v1/audio/speech` requests whose `model` is the directory name to it.

## Validation tools (`scripts/tts/`)

| tool | use |
|---|---|
| `veena_ref.py`, `chatterbox_ref.py`, `t3_ref.py` | HF / vLLM / fp32 references |
| `asr_check.py` | Whisper round-trip CER gate |
| `tts_bench.py` | TTFA / RTF / audio s/s client for any `/v1/audio/speech` server |
| `plow_speech_probe.sh`, `vllm_speech_probe.sh` | plowrt vs vLLM+SNAC speech servers, same client, one lease each |
| `snac_check.py`, `sample_kernel_bench.py` | codec and sampler numerics/latency |
| `crates/plowrt/examples/t3_check.rs` | T3 logits gate vs `t3_ref.py` |

Every GPU run goes through `perf-data/tools/gpulease`.
