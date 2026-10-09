# Known issues and limits

## Models and profiles

* **chatterbox-mtl request length in `voice-core`.** The profile serves Chatterbox Multilingual at
  a 1024-token context (to fit beside ASR and the LLM). Its text tokens are roughly one per
  character, and the speech it generates shares the same context. Measured with a repeated English
  paragraph: 195 characters gave 8.8 s of audio and 390 gave 17.4 s (complete). 780 characters gave
  only 19.6 s and 1170 gave 9.6 s: **the audio is cut short without an error**. At about 1500
  characters the request is refused with 400 `context_length_exceeded`. Keep each request to
  **at most ~400 characters** (2-3 sentences) and send longer text sentence by sentence, as
  `clients/voice_agent.py` does. The `single-chatterbox-mtl` profile serves the full 2048-token
  context (the model's own cap is 1000 speech tokens, ~40 s).
* **Veena** speaks long input as consecutive segments of whole sentences (no length issue). Its
  first audio arrives after ~81 ms at c1, by design: the first chunk is sized so the next one does
  not underrun.
* **Orpheus is not shipped** (removed in kit4).
* **Not every bundle fits at once.** All seven production bundles need ~107 GiB even with narrowed
  contexts. Profiles are fixed sets that fit, and `--pin-resident` refuses one that does not. Use
  one instance per GPU for more models (`CUDA_VISIBLE_DEVICES`, separate configs and ports).
* **`voice-veena` has the least headroom:** 76.5 GiB peak of 79.6 under the 64-call load, against
  74.4 GiB for `voice-core`. Its E4B KV budget is smaller, so long-prompt LLM bursts at high
  concurrency queue (TTFT p50 1.4 s at 64 concurrent 1000-token prompts).
* **Gemma-4 E4B throughput in the voice profiles** is 23% lower at 64 concurrent requests than
  alone (per-request sliding-window caches, DEPLOY.md section 4). Latency at low concurrency is
  unchanged (TPOT 5.7 ms, TTFT +3 ms).
* **qwen3-asr-0.6b** WER is 4.435% one request at a time and 4.261% batched (the single-row decode
  path rounds differently); both are within its 4.5% gate. qwen3-asr (1.7B) is 3.913% either way.

## API

* **Silero no-speech gate latency:** with the VAD loaded, every transcription upload is scanned by
  Silero first (about 2 ms of CPU per second of audio; +11 ms p50 at c1 on LibriSpeech clips), and
  uploads with under 250 ms of speech return an empty transcript without running the model. To
  trade the gate for latency, remove `silero-vad` from the profile's `MODELS` (this also turns
  `/v1/audio/vad` off and returns the streaming endpointers to the energy detector).
* **Realtime API scope:** transcription sessions only (`intent=transcription`). Not implemented:
  responses / conversation items, `semantic_vad`, noise reduction, `include` logprobs, usage.
  `prefix_padding_ms` is accepted and echoed but fixed at 200 ms of context.
* **Transcription length:** 30 s per utterance on `POST /v1/audio/transcriptions` and the native
  WebSocket utterance mode; longer audio needs the WebSocket continuous mode or the Realtime API.
* **LLM:** no tool calling (`tools` is refused with 400), no `response_format` / JSON mode, `n = 1`,
  no `/v1/embeddings`. Raw `/v1/completions` adds no BOS: start Gemma prompts with `<bos>`.
* **TTS:** output is 24 kHz mono 16-bit only (`wav` or `pcm`); `speed` must be 1.0; no mp3 / opus.
  TTS sampling is random per request unless `seed` is set.
* **Model cards:** voices are not listed in `/v1/models`; an unknown `voice` answers 400 with the
  valid list.

## Platform

* **GPU:** H100 80GB only (the packets are compiled for sm_90a with 132 SMs). Other GPUs need other
  builds.
* **Driver:** tested on 595.91.07. Drivers 525.60-574 should work under CUDA 12 minor-version
  compatibility but are untested; `hostcheck` warns.
* **OS:** glibc >= 2.34 (Ubuntu 22.04+, RHEL 9+). x86_64 only.
* **No TLS in the server:** terminate TLS in a reverse proxy (DEPLOY.md section 6).
* **Startup:** ~70 s for `voice-core` with the checkpoints in the page cache, longer from a cold
  disk. `/health` answers only once every model is loaded; requests before that are refused.
* **Panics end the process** (no partial recovery); systemd restarts it after 5 s. A fatal device
  fault turns `/health` to 503 until the restart.
