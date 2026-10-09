# Deploying the plow H100 voice kit

The kit is self-contained and relocatable: the `plowrt` server binary, the cuBLASLt library it
loads, the model bundles, the deploy scripts, and the client/eval/perf tooling. Nothing is built
on the target machine, and no CUDA toolkit, Python, container runtime or network access is needed
by the server. Every path the scripts use is relative to the kit directory.

## 1. Host requirements

| | requirement | checked by `hostcheck` |
|---|---|---|
| GPU | NVIDIA H100 80GB (SXM5 or PCIe HBM3, sm_90a); the packets are compiled for 132 SMs | yes |
| Driver | NVIDIA driver >= 575 recommended; tested 595.91.07. 525.60-574 run under CUDA 12 minor-version compatibility (untested) | yes |
| OS | x86_64 Linux with glibc >= 2.34: Ubuntu 22.04 / 24.04 / 26.04, Debian 12+, RHEL / Rocky / Alma 9+ | yes |
| CPU RAM | >= 64 GB (checkpoints are memory-mapped and page-cached; the voice profiles touch ~22 GB) | yes |
| Disk | ~46 GB for the kit, plus the same again for the installed copy | yes |
| Tools | `bash`, `curl`, `sha256sum`, `systemd` (optional) | yes |
| Python | only for the clients, eval and perf scripts: Python >= 3.10 with `requirements.txt` | yes |

The GPU must be otherwise idle: a voice profile keeps ~65-75 GB of the 80 GB resident.
`libcuda.so.1` comes from the driver; set `LIBCUDA=/path/to/libcuda.so.1` in the config if it is not
on the default loader path. `plowrt/libcublasLt.so.12` (CUDA 12.9) ships beside the binary and is
loaded from there; keep the two files in the same directory.

## 2. Check, install, start

```bash
cd plow-h100-voice-kit-<sha>
./deploy/plow-voice.sh hostcheck            # GPU, driver, glibc, RAM, disk, port, tools
./deploy/plow-voice.sh preflight --full     # every file against SHA256SUMS + the plowrt/packet pairing
sudo ./deploy/plow-voice.sh install         # copy to /opt/plow-voice, write the config, start the unit
```

`install` (root) does, in order:

1. Runs the quick preflight (binary and packet hashes against `PAIRING.txt`) and refuses on a mismatch.
2. Creates the system user `plow-voice` (`--user` to use another).
3. Stops a running `plow-voice.service` (graceful drain), copies the kit to
   `/opt/plow-voice/<kit name>` (`--prefix DIR` for another filesystem), makes it read-only, and
   points `/opt/plow-voice/current` at it.
4. Writes `/etc/plow-voice/plow-voice.conf` from `deploy/plow-voice.conf.sample` if it does not
   exist (an existing config is kept), mode 0640 root:plow-voice.
5. Installs `/etc/systemd/system/plow-voice.service`, enables it and starts it. `systemctl start`
   returns once `/health` answers 200, i.e. every model of the profile is loaded (about 70 s warm,
   up to a few minutes on a cold page cache).

Options: `--prefix DIR`, `--config FILE`, `--user NAME`, `--no-systemd` (copy + config only),
`--no-start`.

Without systemd (a container, another supervisor), run the server in the foreground:

```bash
./deploy/plow-voice.sh run /path/to/plow-voice.conf      # execs plowrt; stop with SIGTERM
./deploy/plow-voice.sh wait-ready /path/to/plow-voice.conf 900
./deploy/plow-voice.sh print-cmd /path/to/plow-voice.conf # the exact plowrt command line
```

`run` starts `plowrt` with a clean environment (`env -i`), so stray `PLOW_*` variables in the
calling shell cannot change its behaviour; everything comes from the config and the profile.

## 3. Configuration

`/etc/plow-voice/plow-voice.conf`, `KEY=VALUE` lines (no shell evaluation). After editing:
`sudo systemctl restart plow-voice`.

| key | default | meaning |
|---|---|---|
| `PROFILE` | `voice-core` | model set, a file in `deploy/profiles/` (section 4) |
| `BIND` | `127.0.0.1` | listen address; `0.0.0.0` for every interface (then set `API_KEYS` and put TLS in front) |
| `PORT` | `8000` | HTTP + WebSocket port |
| `API_KEYS` | empty | comma-separated keys; when set every route except `/health` and `/healthz` needs `Authorization: Bearer <key>` or `x-api-key: <key>` |
| `LOG_LEVEL` | `info` | `error`, `warn`, `info`, `debug` (RUST_LOG syntax) |
| `LOG_FILE` | empty | empty = stdout (the journal); a path = append there |
| `DRAIN_TIMEOUT_MS` | `30000` | SIGTERM drain window for live requests |
| `ASR_REQUEST_TIMEOUT_MS` | `120000` | a transcription past this is cancelled and answered 504 |
| `SESSION_TTL_MS` | `60000` | retention of a finished `X-Session-Id` request's state for the session's next turn |
| `HTTP_MAX_CONNECTIONS` | `4096` | concurrent connections (HTTP + WebSocket) |
| `CUDA_VISIBLE_DEVICES` | empty | GPU index or UUID on a multi-GPU host |
| `LIBCUDA` | empty | path to `libcuda.so.1` when it is not on the loader path |
| `MODELS`, `LIVE_CTX` | empty | override the profile's model list / per-model context bounds |
| `EXTRA_ARGS` | empty | extra `plowrt serve` flags |

## 4. Profiles: what stays resident on the GPU

Every model of a profile is loaded at startup and stays resident (`--pin-resident`): requests never
wait for a model swap, and if the set does not fit the server refuses to start instead of
swapping. Measured on one H100 80GB (driver 595.91.07):

| profile | models | GPU memory after startup / under load | startup | notes |
|---|---|---|---|---|
| `voice-core` (default) | Silero VAD, qwen3-asr, chatterbox-mtl, gemma-4-e4b | 63.5 / 74.4 GiB | 70 s | the recommended voice agent: 23-language TTS; 32 concurrent calls within SLO (PERF.md) |
| `voice-veena` | Silero VAD, qwen3-asr, veena, gemma-4-e4b | 67.1 / 76.5 GiB | 34 s | Indic + English voices (Veena) instead of Chatterbox; ~16-30 calls |
| `asr` | Silero VAD, nemotron-3.5-asr, qwen3-asr, qwen3-asr-0.6b | see below | | transcription only, full context |
| `single-<model>` | one model at its full packet context (ASR ones with Silero VAD) | 37-43 GiB | 14-41 s | the BASELINE.md single-model configuration |
| `experimental-orpheus` | orpheus | ~37 GiB | | EXPERIMENTAL (KNOWN_ISSUES.md) |

How the voice profiles fit, and what each setting costs:

* Each bundle sizes its KV cache for 128 concurrent sequences at its compiled context. Alone,
  qwen3-asr takes 38 GiB, chatterbox-mtl 42 GiB and gemma-4-e4b 41 GiB, so the three only
  co-exist by narrowing that reservation per model.
* `LIVE_CTX` (`--live-ctx-models`) narrows a model's context: qwen3-asr to 768 tokens (30 s of
  audio plus ~330 transcript tokens: the transcription API's 30 s limit is unchanged) and
  chatterbox-mtl to 1024 tokens (text and speech share it: keep requests to ~400 characters, see
  KNOWN_ISSUES.md). veena keeps 1536 tokens, which holds its full 1400-token segment
  budget (Veena speaks long input as consecutive segments).
* gemma-4-e4b keeps its full 8K context; its sliding-window caches are allocated per live request
  instead of up front (`PLOW_VMM_LIVE_RINGS_MODELS=gemma-4-e4b`, 20 GiB less at load). Cost, E4B
  alone, ISL 1000 / OSL 128: 21% lower throughput at 64 concurrent requests (TPOT p50 14.7 -> 19.0
  ms) and +7 ms TTFT at 1 request. With up-front caches the set fits only with chatterbox-mtl at 512
  tokens and leaves ~3 GiB, and E4B ran out of KV memory on its first request; the default
  profile therefore takes the per-request caches.
* All bundles keep the VMM prefix cache on.
* Not every bundle fits at once: all seven production bundles need ~107 GiB even with narrowed
  contexts, and `--pin-resident` refuses that set at startup. Use two GPUs (one instance each,
  `CUDA_VISIBLE_DEVICES`) for more models.

A custom set: copy a profile file to `deploy/profiles/<name>.profile` in the installed kit (or set
`MODELS` / `LIVE_CTX` in the config); `plowrt` refuses to start if it does not fit.

## 5. Operations

**Health and readiness.** `GET /health` (and `/healthz`, never behind auth) answers 200 once every
model is loaded and 503 while shutting down or after an engine died (a fatal device fault). Probe
it from the load balancer; restart on repeated 503.

**Logs.** `journalctl -u plow-voice -f` (or `LOG_FILE`). Startup logs one line per model with its
measured memory (`planner: load measured`); each request logs at debug level. API keys and other
secrets (`*_KEY(S)`, `*_TOKEN(S)`, `*SECRET*`, `*PASSWORD*`) are logged as `<redacted>`. Kits before
kit3 printed `PLOW_API_KEYS` in the startup `serve replay` line: rotate those keys and clear old logs.

**Metrics.** `GET /metrics` (Prometheus text): per-model request counters and latency histograms,
queue depths, KV usage, plus vLLM-compatible `vllm:` series for the LLM. `GET /v1/models/status`
lists each model's residency and planned memory.

**Restart policy.** The unit restarts the server 5 s after any exit (`Restart=always`). The
release build aborts on a panic rather than continuing in an unknown state, and a fatal device
fault (a model's engine dead, `/health` 503) makes the server drain for at most 5 s and exit 1
(`PLOW_EXIT_ON_ENGINE_DEATH=1`, set by the launcher), so the supervisor is the recovery path. A
container or other supervisor must restart on a non-zero exit too.

**Graceful stop / drain.** `systemctl stop plow-voice` sends SIGTERM: `/health` turns 503, new
requests get 503, in-flight requests continue for up to `DRAIN_TIMEOUT_MS` (30 s), live WebSocket
sessions still receiving audio get a terminal error and close code 1001, then the process exits 0.
`TimeoutStopSec=60` covers the drain.

**Load shedding.** A full queue answers `429` with `Retry-After`; TTS admits a stream only while it
can keep every playing stream ahead of real time and answers 429 past a 6 s wait. Shutdown and a
dead engine answer `503`. API.md lists every status code and limit.

**Upgrade / rollback.** Install the new kit the same way: it lands in its own directory and
`current` is switched; the config is kept. To roll back, point `current` at the previous directory
and `systemctl restart plow-voice`.

**Uninstall.** `systemctl disable --now plow-voice; rm /etc/systemd/system/plow-voice.service;
rm -r /opt/plow-voice /etc/plow-voice; userdel plow-voice`.

## 6. TLS and reverse proxy

`plowrt` speaks plain HTTP. For anything beyond localhost, keep `BIND=127.0.0.1` and terminate TLS
in a reverse proxy on the same host:

* `deploy/nginx-plow-voice.conf.sample`: nginx with the WebSocket upgrade
  (`/v1/audio/transcriptions/stream`, `/v1/realtime`), buffering off for SSE and streamed audio,
  1 h read timeouts for long sessions, 64 MB bodies, `/metrics` kept internal.
* `deploy/Caddyfile.sample`: Caddy (automatic certificates; WebSockets and streaming work as is).

Browsers using the Realtime API can pass the key as the WebSocket subprotocol
`openai-insecure-api-key.<key>` beside `realtime`; the proxy must forward `Sec-WebSocket-Protocol`
(both samples do).
