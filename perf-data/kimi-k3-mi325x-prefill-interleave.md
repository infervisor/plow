# Kimi K3 MI325X cold-prefill interleave sweep

Date: 2026-08-11

## Decision

Keep `PLOW_PF_NO_INTERLEAVE=1` as an explicit cold-throughput mode, not the
serving default. On a closed C8 burst of 8K prompts it consistently reduces
TTFT and burst makespan, but it stalls already-live decode streams whenever a
new prefill arrives.

Eight MI325X GPUs, TP8, native MXFP4, FP8 KV, the adopted FP8 MLA V2 packet,
compact TP counter audit, counter double buffering, vLLM 0.27.0 client, and one
warmup per cell were used. Both arms used host recurrent-state clear because
the measured primary decode object does not export `plow_state_clear`.

## Results

Workload: random input 8192, output 128, C8/N8, seeds 0/1/2.

| arm | seed | duration (s) | output tok/s | median TTFT (ms) | median TPOT (ms) |
|---|---:|---:|---:|---:|---:|
| interleave | 0 | 55.245 | 18.535 | 21690.17 | 257.14 |
| interleave | 1 | 55.407 | 18.482 | 21782.74 | 257.63 |
| interleave | 2 | 55.202 | 18.550 | 21679.93 | 256.98 |
| no-interleave | 0 | 53.808 | 19.031 | 20971.75 | 258.53 |
| no-interleave | 1 | 53.777 | 19.042 | 20991.78 | 258.12 |
| no-interleave | 2 | 53.813 | 19.029 | 21002.66 | 258.33 |

Median-of-three delta for no-interleave:

- burst duration: 55.245 -> 53.808 s (`-2.60%`)
- output throughput: 18.535 -> 19.031 tok/s (`+2.67%`)
- median TTFT: 21690.17 -> 20991.78 ms (`-3.22%`)
- median TPOT: 257.14 -> 258.33 ms (`+0.46%`)
- median E2EL: 54346.89 -> 53805.55 ms (`-1.00%`)

All six cells completed 8/8 requests, generated exactly 1024 tokens, reported
empty errors, and contained no in-band error marker. Generated text differs
between arms because gradual interleave sends early requests through B1/B2/B4
before B8, while no-interleave begins decode at B8; grouped MoE accumulation
order is width-dependent. This experiment does not change model weights or
packet instructions.

## Commands

The server control omitted `PLOW_PF_NO_INTERLEAVE`; the candidate added only
`PLOW_PF_NO_INTERLEAVE=1`:

```bash
nix develop --command env -u PLOW_STATE_CLEAR_DEVICE \
  PLOW_MLA_PF_V2=1 \
  PLOW_HSACO=/home/lava/plow/build-amd/k3-mi325x-v2fp8-seg-hsaco \
  PLOW_L2_PLACE_DISPATCH=1 PLOW_TP_AUDIT_COMPACT=1 PLOW_CTR_DBUF=1 \
  PLOW_HSACO_LOWRUNG=/home/lava/plow/build-amd/k3-b1-ladder-grouped:1,/home/lava/plow/build-amd/k3-b2-ladder-grouped:2,/home/lava/plow/build-amd/k3-b4-ladder-grouped:4,/home/lava/plow/build-amd/k3-b8-ladder-grouped:8 \
  perf-data/harness/gpulease -n 8 k3-pf-interleave \
  target/release/plowrt serve \
  --assets /home/lava/models/k3_mi325x_ladder_router_v2fp8_seg2 --port 8054
```

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8054 \
  --endpoint /v1/chat/completions \
  --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random --random-input-len 8192 --random-output-len 128 \
  --random-range-ratio 0 --request-rate inf --max-concurrency 8 \
  --num-prompts 8 --num-warmups 1 --ignore-eos --temperature 0 --seed SEED \
  --save-result --save-detailed --result-dir RESULT_DIR \
  --result-filename seedSEED.json
```

Raw JSON: `/tmp/k3-pf-serial-sweep/{control,no-interleave}/seed{0,1,2}.json`.
