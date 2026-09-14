# Gemma 4 31B IT: BF16 weights, FP8 KV checkpoint

The H100 checkpoint preserves the complete compiled packet: prefill buckets
128/512/1024 and decode rungs 1/2/4/8/16. Weights and activations are BF16;
all KV layers use E4M3 with FP32 row scales. The context limit is 32768 tokens,
including output. Sixteen physical slots are distinct from queued concurrency.

Compile on the H100 with the original Hugging Face snapshot:

```sh
nix develop -c cargo build -p plowc --bin plowc
nix develop -c env PLOW_UNISEG=1 PLOW_FP8_KV=1 \
  PLOW_DECODE_BATCH_LADDER=1,2,4,8,16 PLOW_MAX_CHUNK=1024 \
  target/debug/plowc --hf-dir "$GEMMA_SNAPSHOT" --gpu 'H100 SXM5' \
  --arch sm_90a --n-cu 132 --max-ctx 32768 --emit devblob+cubin \
  --out "$GEMMA_ASSETS"
```

The asset directory must contain `checkpoint/` pointing to the source snapshot.
Use a frozen release runtime built with `cuda,hsa,hub`. Package it with:

```sh
nix develop -c python3 scripts/package_gemma4_checkpoint.py \
  --assets "$GEMMA_ASSETS" --runtime "$PLOWRT_BINARY" --out "$CHECKPOINT"
```

`--evidence FILE` may be repeated to include qualification logs. Packaging copies
weights, tokenizer and chat template, all top-level emitted assets, runtime and
linked ELF libraries into regular files. It records SHA-256 hashes, source
revision and every rung in `manifest.json`. Build scratch directories are omitted.
The output directory must not exist.

```sh
cd "$CHECKPOINT"
python3 verify.py
./serve.sh
# Another terminal:
python3 workloads.py --concurrency 16
```

The launcher explicitly selects the FP8-KV decode and prefill cubins and enables
prefix reuse. FP8 KV requires single-segment ordinary prefill; the segmented
and packed paths are unavailable. The token-batch selector is on, but CUDA uses
the serving fallback executor. Multi-step decode is disabled for this checkpoint.

FP8-KV cold and warm logits can differ when the request uses different prefill
buckets. Cache correctness is checked against an independently computed resident
prefix with the same suffix bucket. Workloads report cold/warm text agreement
and require concurrent warm replay to agree. Document tasks cover support replies,
incident handoff and field extraction; they are smoke workloads, not a quality
benchmark or proof of sustained production readiness.

This is a local instance checkpoint, not an off-instance backup. Copy the entire
directory to preserve it. The target host needs Linux x86_64 and a compatible
NVIDIA driver; `PLOW_LIBCUDA` overrides the driver path. The bundle does not need
the source checkout, Hugging Face cache or original Nix store paths to launch.
The runtime binds TCP on `0.0.0.0`; use the deployment's controlled ingress.
`PORT`, `SLO_MS` and `MAX_QUEUED_REQUESTS` configure serving. The sample SLO is
600000 ms; choose the deployment's actual latency budget before exposing traffic.
