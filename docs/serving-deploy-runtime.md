# Deploy runtime contract

One `plowrt` binary and one set of runtime libraries serve every bundle of a release. Switching
models changes the assets and the profile, never the binary or a per-model runtime directory.

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
| kit pairing | `KIT.json` / `PAIRING.txt`: the plowrt sha256 and the sha256 of every packet it was qualified with | `plow-voice.sh preflight` |

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
| a model listed by the profile is not in the kit, or its mapped checkpoint dir is missing | `plow-voice.sh` refuses to start |
| plowrt, a runtime library or a packet differs from `plowrt.sha256` / `PAIRING.txt`; an extra `*.so*` in `plowrt/` | `plow-voice.sh preflight` fails (`install` runs it) |

## Kit layout

```
<kit>/
  plowrt/                plowrt, libcublasLt.so.13, plow_verify, BUILD.json, plowrt.sha256
  models/<name>/         compiled assets only
  hf/<org>--<repo>@<rev>/  HF snapshots; hf/<name>@<content id>/ for converted checkpoints;
                         deduplicated by content (bundles with identical checkpoints share one)
  deploy/                plow-voice.sh, plow-voice.service, profiles/*.profile, checkpoints.map
  KIT.json PAIRING.txt SHA256SUMS
```

`deploy/checkpoints.map` (`<model> <dir>`) pairs every model with its `hf/` directory;
`CHECKPOINTS=model=dir,...` in a profile or the config overrides it. A profile names a model set
(`MODELS`), context bounds and serve flags; switching models is `PROFILE=<name>` in the config plus
a restart of the one unit, `plow-voice`.

`make_kit.sh <runtime dir> <audio dir> <out> <name>=<bundle>...` builds a kit. A kit update ships
only what changed: `plow-voice.sh adopt <old kit> <other bundle dir>...` hard-links every missing
file whose sha256 another directory's `SHA256SUMS` lists, then `preflight --full` re-hashes all of
it.
