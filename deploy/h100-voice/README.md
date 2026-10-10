# plow H100 voice kit

Speech recognition, an LLM and speech synthesis served together on **one NVIDIA H100 80GB** by
`plowrt`, behind an OpenAI-compatible HTTP + WebSocket API: the building blocks of a voice agent
(audio in -> transcript -> reply -> audio out), all models resident on the GPU.

| | |
|---|---|
| ASR | Qwen3-ASR 1.7B (`qwen3-asr`), Qwen3-ASR 0.6B, Nemotron 3.5 streaming ASR; Silero VAD (CPU) |
| LLM | Gemma-4 E4B-it (`gemma-4-e4b`), 8K context |
| TTS | Chatterbox Multilingual V3 (`chatterbox-mtl`, 23 languages), Veena (Indic + English), Chatterbox (English) |
| LLM (own profile) | Gemma-4 26B-A4B-it FP8 (`gemma-4-26b-a4b-it-fp8`), 131072 context, tool calls: `PROFILE=gemma-26b-fp8` |
| Server | `plowrt/`: one binary plus the cuBLASLt library it loads; no Python, CUDA toolkit or container needed (NVIDIA driver, glibc >= 2.34) |

## Quick start

```bash
cd plow-h100-voice-kit-<sha>
sudo ./deploy/plow-voice.sh install          # host check + hash check, copy to /opt/plow-voice, start the systemd unit
curl -s localhost:8000/health                # "ok" once all models are loaded (~70 s)
curl -s localhost:8000/v1/audio/transcriptions -F model=qwen3-asr -F file=@clients/samples/sample_en.wav
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -d '{"model":"chatterbox-mtl","input":"Hello! How can I help you today?","voice":"default"}' -o hello.wav
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"gemma-4-e4b","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32}'
```

The default config listens on `127.0.0.1:8000` without API keys. Edit
`/etc/plow-voice/plow-voice.conf` (bind address, port, `API_KEYS`, profile) and
`sudo systemctl restart plow-voice`. No systemd: `./deploy/plow-voice.sh run <config>` runs the
server in the foreground.

## Contents

| path | |
|---|---|
| `docs/DEPLOY.md` | host requirements, install, configuration, profiles (what fits on the GPU), operations, TLS proxy |
| `docs/API.md` | every endpoint: transcription (HTTP, SSE, WebSocket, OpenAI Realtime), VAD, speech, chat/completions; errors and limits |
| `docs/EVAL.md` | quality evaluation: ASR WER, TTS round-trip CER, LLM checks; reference results |
| `docs/PERF.md` | performance: per-model and voice-agent load results on H100, and how to reproduce them |
| `docs/KNOWN_ISSUES.md` | limits and known issues |
| `deploy/` | `plow-voice.sh` (hostcheck, preflight, install, run), the systemd unit, config sample, profiles, nginx / Caddy samples |
| `clients/` | curl and Python examples for every endpoint, incl. WebSocket, Realtime, streaming TTS to file and a voice-agent round trip |
| `eval/`, `perf/` | the evaluation and benchmark scripts; `data/` holds 73 LibriSpeech clips for them |
| `models/` | the compiled model bundles (each with `SHIP.md` and `MANIFEST.json`: provenance, hashes, gate results) |
| `hf/` | the HF checkpoints the bundles serve against, one directory per distinct checkpoint (`deploy/checkpoints.map`) |
| `plowrt/` | the one runtime for every model: `plowrt`, `libcublasLt.so.13` (cuBLAS 13.4), `plow_verify` (load-time check of the packets' compiler receipts), `BUILD.json` |
| `KIT.json`, `PAIRING.txt`, `SHA256SUMS` | build provenance, the binary/packet pairing, hashes of every file |
| `BASELINE.md` | single-model H100 baseline of the bundles |

Client tooling: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
(`requirements-whisper.txt` adds the Whisper model for the TTS quality check), then e.g.
`.venv/bin/python clients/voice_agent.py clients/samples/sample_en.wav reply.wav`.
Set `PLOW_URL` (default `http://127.0.0.1:8000`) and `PLOW_API_KEY` for a remote or keyed server.

Verify the copy before installing: `./deploy/plow-voice.sh preflight --full` checks every file
against `SHA256SUMS`.
