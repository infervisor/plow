# GLM batched sparse decode on MI300X

The unpooled BF16 indexer now supports a 1/2/4/8 decode ladder. Its projections,
norm, score buffers and selected indices carry a batch axis. Key-cache writes
use each slot's position, including rung 1 of a larger blob. Each row's top-k
selection runs in sequence over the existing shared histogram and control
storage; `IndexSelect.i[3]` identifies the row. Gathered attention already has
the matching batch layout.

Qualification exposed an existing MI300X defect: the decode score kernel used
78,464 bytes of LDS, exceeding both the 64 KiB hardware limit and the interpreter
arena. CDNA3 now uses a 128-position tile occupying 43,648 bytes. CDNA4 retains
its 256-position tile. The interpreter asserts that the scratch fits its arena.
The loader requires `plow_dsa_decode_batch_arm` for batched selection and for
CDNA3 decode scoring, including single-row packets. Old objects are rejected.

## Verification

- The GPU test checks exact top-k sets at live lengths
  0/1/129/2047/2048/8192/65537/70000, tied scores, poisoned tails, permuted logical
  slices and three reuses of the shared scratch.
- The score test checks every live position for eight distinct query/key/weight
  rows against an independent FP32 calculation, with absolute error <= 1e-7.
  Unwritten score tails retain their sentinel.
- Emitter tests cover scratch capacity, row strides, rung-1 cache writes in a
  B8 blob, and dependencies between successive selections. Loader and manifest
  tests cover the required object marker.
- TP8 concurrent retrieval passes **18/18** after the LDS fix, versus **2/18**
  with the overflowing tile. Actual prompt lengths are 5433–5438 and
  68797–68802, depths 0.1/0.5/0.9, three facts, concurrency 8, temperature 0,
  and 24 generated tokens. This is a limited retrieval screen.

The [result record](mi300x-results.json) includes per-case outputs and hashes of
the packet, tested runtime and decode objects. The final emitter reproduces
the measured B8 packet byte-for-byte.

## Serving screen

Same host, TP8 B8, BF16 KV, all-layer sparse prefill, no prefill interleaving,
TP audit retained. The client uses the reference's random 70k/700 lengths,
ratio 0.14, seed 0 and concurrency 20, with **20 requests**. Both runs produced
identical per-request input/output token counts: 1,414,538 input and 13,795 output.

| Metric | Previous: dense decode, interpreter prefill | Sparse decode + native prefill |
|---|---:|---:|
| Successful / failed | 20 / 0 | 20 / 0 |
| Duration, s | 583.69 | 489.35 |
| Output tok/s | 23.63 | **28.19** |
| Mean TTFT, s | 223.01 | 195.14 |
| Mean TPOT, ms | 264.88 | 218.68 |
| Median ITL, ms | 157.16 | 116.60 |

Recorded throughput increases **19.3%** and mean TPOT decreases **17.4%**.
This comparison combines native AITER prefill and sparse decode, including
disabling the incompatible q-RoPE fusion. It is one before/after comparison,
not a repeated A/B or an isolated decode-kernel gain. It does not establish
parity with the H200 100-request result.

## Reproduce

Inside `nix develop`, compile and run the GPU oracle under a lease:

```sh
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 \
  -Iruntime/amd -Iruntime/common \
  -c runtime/tests/dsa_batch_select_gfx942_test.hip -o /tmp/dsa-batch.o
c++ /tmp/dsa-batch.o -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" \
  -lamdhip64 -o /tmp/dsa-batch
perf-data/tools/gpulease -n 1 dsa-batch /tmp/dsa-batch
```

Build the main decode object and its smaller tiers beside the existing
qualified prefill/flash objects:

```sh
PLOW_DECODE_BATCH=16 PLOW_DECODE_TIERS=1,2,4,8 PLOW_ROWS_ONLY==interp_decode \
  scripts/build_gfx942.sh "$OBJECT_DIR"
```

Emit with the previous all-layer sparse TP8 B8 `build.json` as `REPLAY`:

```sh
PLOW_VERIFY_BIN=lean-plow/.lake/build/bin/plow_verify GLM_FULL=1 \
PLOW_MLA_PF_AITER=1 PLOW_MLA_PF_V2=1 PLOW_UNISEG=0 PLOW_GLM_DSA_PF_SPAN=3 \
  target/release/plowc --hf-dir /workspace/models/GLM-5.3-plow-lite \
    --emit devblob --gpu MI300X --arch gfx942 --max-ctx 81920 --num-gpus 8 \
    --replay-knobs "$REPLAY" --glm-dsa 1 --glm-fuse-rope=false --out "$ASSETS"
```

The existing q-RoPE fusion shares fields with gathered attention and must be
disabled. Preserve the native MLA objects described in the
[prefill adapter instructions](../mla_sparse_aiter/README.md#native-plow-runtime-route).
Start serving under an eight-GPU lease with:

```sh
PLOW_HSACO="$OBJECT_DIR" PLOW_MLA_PF_AITER=1 PLOW_MLA_PF_V2=1 \
PLOW_PF_CHUNK=8192 PLOW_PF_INTERLEAVE=0 \
  target/release/plowrt serve --assets "$ASSETS" --port 8080
```

Run the retrieval client against that server:

```sh
python runtime/bench/amd/mla_sparse_aiter/quality.py \
  http://127.0.0.1:8080 sparse quality.json --concurrency 8
```

With the same server, run the serving screen using the ID returned by
`/v1/models` (the emitted bundle here serves `glm-5.3-plow-lite`):

```sh
vllm bench serve --backend vllm --host 127.0.0.1 --port 8080 \
  --model glm-5.3-plow-lite --tokenizer zai-org/GLM-5.3 --trust-remote-code \
  --dataset-name random --num-prompts 20 --random-input-len 70000 \
  --random-output-len 700 --random-range-ratio 0.14 --max-concurrency 20 \
  --save-result --save-detailed --result-filename serve-20.json
```

TP audit remains enabled. NVIDIA batched DSA loading, pooled batched DSA,
MXFP4 indexers and batched FP8 KV remain unsupported. No speculative decoding
is added. These checks do not establish H200 serving parity.
