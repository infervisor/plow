# Deploy runtime contract

One `plowrt` binary and one set of runtime libraries serve every bundle of a release. Switching
models changes the `--assets` arguments, never the binary or a per-model runtime directory. The
repo carries no per-model or per-customer launch profiles: a bundle's `serve.json` holds its serve
defaults, and the command line names the bundles.

## One runtime per release

A release ships exactly one runtime directory:

| file | what |
|---|---|
| `plowrt` | `cargo build --release -p plowrt --features cuda,gguf --bin plowrt` (no HSA); model-agnostic |
| `libcublasLt.so.13` | cuBLAS 13.4 (`cublasLtGetVersion` 130402). The only cuBLASLt of the release: every H100 bundle (dense, MoE grouped routes, speech) is validated on it |
| `plow_verify` | the Lean verifier the packets' compiler receipts name (`PLOW_VERIFY_BIN`) |
| `BUILD.json` | commit, branch, toolchain, `min_glibc`, `min_driver`, `cublaslt`, dynamic deps |
| `plowrt.sha256` | sha256 of every file above except `BUILD.json`; the runtime is exactly this set |

`plowrt` loads cuBLASLt from `<exe>/../lib`, then `<exe>`, then the system loader (`.so.13` before
`.so.12`), so the library beside the binary always wins. cuBLAS 13 needs a CUDA 13 driver (>= 580;
tested 595.91.07). No `LD_LIBRARY_PATH` is needed or wanted. At startup plowrt logs the library it
bound (`cuBLASLt path=... version=...`).

## What a bundle declares

A bundle is a directory of compiled assets: packets (`*.pkt`), cubins, `objects/`, `build.json`,
`weights.json`, `MANIFEST.json`. Its HF checkpoint is a separate input (next section).

| requirement | where | checked by |
|---|---|---|
| packet ↔ object pairing | `build.json` `pairing.hash`, stamped into each specialised cubin | plowrt refuses a module whose stamp disagrees |
| checkpoint identity | `serve.json` (a `model.pkt` section) `weights`: per shard, size + sha256 of the safetensors header (every tensor's name, dtype, shape, offset), recorded at emit | plowrt at load, before any weight is read |
| cuBLASLt minimum | `build.json` `runtime_requires.cublaslt` (`"13.4"`) when declared; otherwise derived from the emit knobs in `build.json` `knobs.values`: `emit.moe_pf_lt` or `emit.moe_dec_lt` (grouped cuBLASLt MoE routes) need 13.4. Bundles emitted before the field exist get the derived default | plowrt at startup, per bundle |
| minimum plowrt | `build.json` `runtime_requires.plowrt_contract` (integer); this plowrt implements contract 1. Absent = 1 | plowrt at startup, per bundle |
| speech contract (ASR, VAD) | packet pipeline parameter `contract` (+ the driver name), per pipeline: `vad.frame.v1` 1, `rnnt.greedy.v1` 1, audio-LM `causal.v1` 1 ([ASR/VAD packet contract](runtime/asr-packet-contract.md)) | plowrt when it loads the packet (`--assets`, `--asr-packet`, `--asr-vad-packet`): newer or older than implemented = refusal |
| serve settings | `serve.json` `serve_defaults` (registered `PLOW_*` knobs; the environment overrides) | plowrt at startup (unknown knob = refusal) |

## Assets + HF checkpoint

`plowrt serve --assets <dir>[,checkpoint=<hf dir>]` pairs each bundle with its checkpoint; repeat
`--assets` per model. `--asr-packet NAME=PATH.pkt,checkpoint=<hf dir>` does the same for a packet
audio-LM packet (`tokenizer=` is an alias); RNNT packets carry their vocabulary and need none.

Resolution, per bundle:

1. the explicit `checkpoint=` of that bundle (or the `checkpoint` of a control-plane load);
2. `<assets>/checkpoint`;
3. the legacy process-wide `PLOW_CHECKPOINT` / `--rt-checkpoint` (warns when it serves more than one
   bundle, and when a bundle's own `checkpoint/` shadows it).

Every consumer resolves through the same function (`asset::serve::checkpoint_dir`): LLM weights,
speech T3 weights, ASR decoder weights, `tokenizer.json` / Qwen2 vocab, `generation_config.json`,
chat template and processor configs. Startup logs `bundle checkpoint model=... checkpoint=...
source=...` for each bundle.

## What fails closed

| condition | result |
|---|---|
| checkpoint shard missing, different size, or a different tensor header than the packet pins | startup error naming the model, the checkpoint dir, where it came from and each mismatched file |
| packet records no pins (old emit) | warning; tensors are still checked by byte size at bind |
| bundle needs cuBLASLt >= X and the runtime loads an older one, or none | startup error naming the model, the requirement and why, and the library found |
| bundle needs a newer runtime contract | startup error |
| ASR/VAD packet with another speech contract (e.g. a contract-0 `vad.silero.v1` or GGUF-vocabulary RNNT packet), or a required speech op/parameter missing | startup error; re-emit, or `asr_packet_upgrade` for receipt-less packets (VAD, RNNT) |
| `--assets` value with an unknown key, or two different checkpoints for one bundle | startup error |
| packet/object pairing hash mismatch | module refused at load |

## Running

```bash
<runtime>/plowrt serve --bind 0.0.0.0 --port 8000 \
  --assets <bundle A>[,checkpoint=<hf dir>] --assets <bundle B> \
  [--asr-packet NAME=<packet>.pkt] [--asr-vad-packet <silero_vad.pkt>]
```

No `PLOW_*` environment and no `LD_LIBRARY_PATH` are needed: serve settings come from each
bundle's `serve.json`, cuBLASLt from beside the binary. `PLOW_API_KEYS=k1,k2` turns on bearer auth
(every route except `/health`, `/healthz`). `/health` answers `ok` once every bundle is loaded;
SIGTERM drains. Run it under any supervisor (`--exit-on-engine-death` for restart-on-fault); the
API is [serving-openai-compat.md](serving-openai-compat.md).

Check a runtime + bundle set before handing it over (GPU through the queue):

```bash
scripts/bench/gpuq.py submit smoke 1 scripts/serve_test/smoke.sh <runtime>/plowrt <out> \
    --assets <bundle A> --assets <bundle B> [--asr-vad-packet <vad.pkt>]
```

`smoke.sh` starts that command line in a clean environment (`env -i`), runs
`scripts/serve_test/smoke_client.py` against every endpoint `/v1/models` advertises
(`x_plow_endpoints`: chat/completions, audio/transcriptions incl. SSE, WebSocket and Realtime,
audio/speech, plus `/v1/audio/vad`), records the cuBLASLt it mapped, and stops it. `ASR_MANIFEST`
enables the ASR legs, `TTS_CHECK=1` Whisper-checks the speech output, `EVAL=all` adds
`scripts/serve_test/eval.py` (ASR WER, TTS round-trip CER, LLM checks). The voice-agent load
client is `scripts/voice/call_sim.py` (`scripts/voice/serve_voice_agent.sh calls`).
