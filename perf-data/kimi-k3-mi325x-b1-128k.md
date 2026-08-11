# Kimi-K3 MI325X B1 128K-context serving

Date: 2026-08-11. Server: `plowrt serve` on 8 leased MI325X GPUs
(gfx942, 304 CUs/GPU). Client: flake-pinned vLLM 0.27.0 `bench serve`.
The client measured the OpenAI endpoint; vLLM was not the serving engine.

## Artifact and protocol

No existing K3 asset under `/home/lava/models` exceeded `max_ctx=32768`.
The campaign emitted `/home/lava/models/k3_mi325x_b1_ctx131072` from the real
`/home/lava/models/k3_farm` checkpoint with TP8, native MXFP4 weights, FP8 KV,
B1, `max_ctx=131072`, and `n_cu=304`. The packet has programs
`[128, 512, 1024, 2048, 4096, 8192, 1]`, 5,412 tensors, and one exact-width
decode rung. Lean ordering/LDS certificates passed for all seven programs.

Frozen campaign hashes:

- base checkout (working tree was dirty): `4d338bcfc847083521d269ddd4060bedd34a3295`
- `plowrt`: `85e473936dc8e5020fad739770811771c27c8b9b95673c7e14f1a9b3a75bc0b8`
- `model.pkt`: `ec1181202b9832a71c34cb8da3015215ebbb6333f658884e32a6bd65ad2eed28`
- static/GQ B1 decode objects: `a733705827bec917469a9c27ab1f49b7e2a2b9bd772ce443a0c1d58e0c810394` /
  `7604d152dcfcdf428ab222e10f8d12ec2d6ac655790a9e4cfd676a897c440f97`

The object inventory was built with:

```bash
nix develop -c env PLOW_DECODE_BATCH=1 PLOW_K3_DECODE_GROUPED=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 JOBS=8 \
  scripts/build_gfx942.sh /home/lava/plow/build-amd/k3-mi325x-b1-current
```

The asset was emitted with:

```bash
nix develop -c env K3_FULL=1 PLOW_FP8_KV=1 PLOW_MXFP4=1 \
  PLOW_L2_PLACE=1 PLOW_DECODE_BATCH=1 PLOW_DECODE_BATCH_LADDER=1 \
  PLOW_GEMV_MM=1 PLOW_GLM_GEMV_WG=128 \
  ./target/release/plowc --hf-dir /home/lava/models/k3_farm \
  --emit devblob --arch gfx942 --gpu MI325X --num-gpus 8 --parallel tp \
  --max-ctx 131072 --n-cu 304 \
  --out /home/lava/models/k3_mi325x_b1_ctx131072
```

The server used `PLOW_L2_PLACE_DISPATCH=1`, `PLOW_TP_AUDIT_COMPACT=1`,
`PLOW_CTR_DBUF=1`, and `PLOW_STATE_CLEAR_DEVICE=1`. Every cell used C1/N1,
128 forced output tokens, random-range ratio 0, seed 0, temperature 0,
ignore-EOS, and exactly one warmup. The client command shape was:

```bash
nix develop .#vllm -c env VLLM_TARGET_DEVICE=cpu vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8042 \
  --endpoint /v1/chat/completions --model k3_farm \
  --served-model-name k3_farm --tokenizer /home/lava/models/k3_tokz \
  --tokenizer-mode hf --dataset-name random --random-input-len INPUT \
  --random-output-len 128 --random-range-ratio 0 --request-rate inf \
  --max-concurrency 1 --num-prompts 1 --num-warmups 1 --ignore-eos \
  --temperature 0 --seed 0 --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 --save-result --save-detailed
```

## Results

| requested input | actual input | p50 TTFT (ms) | p50 TPOT (ms) | p50 ITL (ms) | p50 E2EL (ms) | output tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 8,213 | 4,867.96 | 56.331 | 56.325 | 12,021.99 | 10.6465 |
| 16,000 | 16,021 | 9,669.70 | 57.806 | 57.809 | 17,011.10 | 7.5243 |
| 32,000 | 32,021 | 22,164.99 | 61.341 | 61.353 | 29,955.32 | 4.2729 |
| 64,000 | 64,021 | 55,558.11 | 68.369 | 68.376 | 64,240.98 | 1.9925 |
| 128,000 | 128,021 | 156,630.08 | 82.575 | 82.596 | 167,117.11 | 0.7659 |

Output throughput includes TTFT and therefore is not steady-state decode
capacity. TPOT/ITL isolate decode behavior. All five measured requests completed
with exactly 128 output tokens and no failures. Detailed vLLM JSON represents a
successful empty per-request error as `[""]`; all entries were empty strings,
and generated text contained no `[error:` marker. Prompt plus output remained
below 131,072. Compact TP rank/counter audits were clean, the server log had no
rank disagreement, timeout, fatal, or device-fault signature, and the lease was
released with no foreign compute process.

Results and logs are in `/tmp/k3-b1-128k-campaign-20260811/`:
`ctx8192.json`, `ctx16000.json`, `ctx32000.json`, `ctx64000.json`,
`ctx128000.json`, `server.log`, `commands.log`, per-cell memory CSVs, and the
empty `audit-errors.txt`.

## Memory and chunk policy

Before GPU execution, packet tensor geometry was checked per rank:

- packet tensors: 32,395,621,236 B (30.17 GiB), of which 22.84 GiB are
  weights and 7.33 GiB runtime tensors
- FP8 KV/cache: 2.817 GiB total: 1.5 GiB MLA compressed KV, 384 MiB rotary
  keys, 12 MiB MLA scales, plus the remaining cache tensors
- KDA carried recurrent state: 59,351,040 B (56.60 MiB) across 276 tensors
- block-residual history: 896 MiB

Observed VRAM was 214.230 GB/GPU at server readiness and 215.037--215.038
GB/GPU after every measured cell. The large difference from packet tensor
geometry is checkpoint/devblob residency; load logs report 191.23 GiB uploaded
per rank. The full 128K cache was preallocated, so resident memory did not grow
with request length.

Prefill used a maximum 8,192-token packet and the smallest available bucket for
the ragged tail. Actual measured covers were:

- 8,213 = 8,192 + 21 in the T=128 bucket
- 16,021 = 8,192 + 7,829
- 32,021 = 3x8,192 + 7,445
- 64,021 = 7x8,192 + 6,677
- 128,021 = 15x8,192 + 5,141

At 128K, measured per-8K chunk drain rose from 4.56 s at offset 0 to 14.98 s
at offset 114,688; the final 5,141-row tail took 10.04 s. This is the expected
causal-attention growth, not packet launch overhead.

## Roofline and next experiment

The MI325X reference roofs are 4,164 GB/s HBM and 1,063 TF/s BF16 MFMA.
Current kernel references are 420.2 TF/s dense MXFP4 prefill, 219.1 TF/s
grouped GLU, 57.4 TF/s grouped DOWN, and about 570 GB/s B1 decode GEMV.
Using the decoded packet's 23.792 GB minimum bytes touched per token, served
TPOT corresponds to a lower-bound effective bandwidth of 422 GB/s at 8K,
412 GB/s at 16K, 388 GB/s at 32K, 348 GB/s at 64K, and 288 GB/s at 128K.
The percentage of the 4,164 GB/s roof falls from 10.1% to 6.9%; context KV
traffic is additional to this fixed minimum, so this is deliberately a lower
bound rather than a complete traffic model.

One bounded next experiment is a K3-capable FP8 MLA flash GQ object for the
unchanged packet and serving flags. The available specialized
`interp_flash_fp8kv_gq.elf` lacked `plow_k3_arms_1`, so the safety check rejected
it and all flash segments used the generic 8-wave K3 interpreter. Build only
that object with K3 op 104/110 capability, then A/B the same 64K and 128K cells.
Accept it only with byte-identical output, clean exact TP audits, and lower TTFT;
the increasing chunk drain makes this the most directly bounded long-context
kernel target. The asset emitter also reported a stale tunedb digest and used
analytical tile fallback, which is held fixed for this proposed A/B.
