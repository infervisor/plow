# Kimi-K3 MI325X B16 MM8+WALK serving control

Date: 2026-08-11. Source: `bf9a98b17a2b` plus the then-current uncommitted
build-script validation. Hardware: 8 leased MI325X GPUs, gfx942, 304 CUs each.
Toolchain: flake-pinned ROCm 7.14.0. Client: flake-pinned vLLM 0.27.0
`bench serve`; vLLM is the HTTP load generator, not the serving engine.

## Matched arms

Both arms use the existing `/home/lava/models/k3_mi325x_b16/model.pkt`, real
Kimi-K3 checkpoint, native MXFP4 weights, FP8 KV, global-queue decode, and
`PLOW_K3_DECODE_MXFP4_PROJ=0`. Only the decode GEMV row strategy changes:

| arm | build flags | GQ object | bytes | private bytes | markers |
|---|---|---|---:|---:|---|
| control | `MM=16`, `WALK=0` | `build-amd/k3-b16-mm16-ctl/interp_decode_fp8kv_k3_gq.elf` | 1,791,288 | 5,712 | `plow_gemv_mm_cap_16` |
| candidate | `MM=8`, `WALK=1` | `build-amd/k3-b16-mm8-walk/interp_decode_fp8kv_k3_gq.elf` | 1,288,800 | 3,140 | `plow_gemv_mm_cap_8`, `plow_gemv_walk_1` |

Both objects report 256 VGPR and 64,568 B LDS. GQ SHA-256 is
`dd0ee625deabce3e0fea8c24fb6e5c1cca3166011301e69ea660470702ad4b6b`
for the control and
`76a10f0710611f545fc7f8c03a4e1780a2a2c94dceb7cf39fe4b3c21b2d07dea`
for the candidate. The static and GQ decode objects were both replaced in an
otherwise copied B16 `hsaco/`; packet, weights, checkpoint, tokenizer, prefill,
and flash objects were unchanged.

Build commands:

```bash
nix develop --command env PLOW_DECODE_BATCH=16 PLOW_GEMV_MM=16 \
  PLOW_GEMV_WALK=0 PLOW_K3_DECODE_MXFP4_PROJ=0 \
  scripts/build_gfx942.sh build-amd/k3-b16-mm16-ctl

nix develop --command env PLOW_DECODE_BATCH=16 PLOW_GEMV_MM=8 \
  PLOW_GEMV_WALK=1 PLOW_K3_DECODE_MXFP4_PROJ=0 \
  scripts/build_gfx942.sh build-amd/k3-b16-mm8-walk
```

The measured control asset is `/tmp/k3-b16-mm16ctl.yL0gEC`. It symlinks the
four non-HSACO asset entries to `/home/lava/models/k3_mi325x_b16`, copies that
asset's `hsaco/`, and replaces these two files:

```bash
cp build-amd/k3-b16-mm16-ctl/interp_decode_fp8kv_k3.elf \
  /tmp/k3-b16-mm16ctl.yL0gEC/hsaco/
cp build-amd/k3-b16-mm16-ctl/interp_decode_fp8kv_k3_gq.elf \
  /tmp/k3-b16-mm16ctl.yL0gEC/hsaco/
```

The MM8 arm was assembled analogously from `build-amd/k3-b16-mm8-walk`.

## Matched serving result

One seed-0 run per arm: random input length 32 before chat templating, 128
forced output tokens, 32 requests, concurrency 16, one warmup, greedy, infinite
request rate. Both detailed results contain the same 1,707 actual input tokens.

| arm | duration (s) | output tok/s | median TTFT (ms) | median TPOT (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|
| MM16 control | 58.309 | **70.246** | 872.64 | **212.79** | **524.95** |
| MM8+WALK | 59.697 | 68.614 | 877.86 | 217.53 | 529.80 |

MM16 vs MM8+WALK is +2.38% output throughput, -2.32% duration, and
-2.18% median TPOT. Input lengths, output lengths, and every generated text
are exactly equal across arms. Each arm completed 32/32 requests, generated
4,096/4,096 requested tokens, and has 32 empty error strings.

The client command was identical apart from the served asset:

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8026 \
  --endpoint /v1/chat/completions \
  --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random --random-input-len 32 --random-output-len 128 \
  --random-range-ratio 0 --request-rate inf \
  --max-concurrency 16 --num-prompts 32 --num-warmups 1 \
  --ignore-eos --temperature 0 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 --save-result --save-detailed \
  --result-dir /tmp/k3-b16-mm16ctl-result --result-filename seed0.json
```

The server ran inside one eight-GPU `gpulease`:

```bash
nix develop --command cp target/release/plowrt /tmp/k3-b16-mm16ctl-plowrt
nix develop --command perf-data/harness/gpulease -n 8 k3-b16-mm16ctl \
  env PLOW_TP_AUDIT_COMPACT=1 PLOW_DSTEP_EVERY=1 PLOW_DSTEP_LOG=1 \
  /tmp/k3-b16-mm16ctl-plowrt serve \
  --assets /tmp/k3-b16-mm16ctl.yL0gEC --port 8026
```

Detailed JSON gates required `completed == 32`, `failed == 0`,
`total_output_tokens == 4096`, 32 output lengths all equal to 128, 32 empty
errors, no generated text containing `[error:`, and the input-length sum equal
to `total_input_tokens`. The cross-arm gate additionally required equal
`input_lens`, `output_lens`, and `generated_texts` arrays.

The executable JSON gate was:

```bash
nix develop --command python3 - "$RESULT" "$REQUESTS" "$OUTLEN" "$TOTAL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
n, outlen, total = map(int, sys.argv[2:])
assert d["completed"] == n and d["failed"] == 0
assert d["total_output_tokens"] == total
assert len(d["output_lens"]) == n and all(x == outlen for x in d["output_lens"])
assert len(d["errors"]) == n and all(not x for x in d["errors"])
assert len(d["generated_texts"]) == n
assert all("[error:" not in x.lower() for x in d["generated_texts"])
assert len(d["input_lens"]) == n
assert sum(d["input_lens"]) == d["total_input_tokens"]
PY
```

For the short cell, `REQUESTS=32 OUTLEN=128 TOTAL=4096`; for the long cell,
`REQUESTS=16 OUTLEN=512 TOTAL=8192`.

Artifacts:

- MM8 detailed JSON: `/tmp/k3-b16-walk8/seed0.json` (its server log was not retained).
- MM16 detailed JSON: `/tmp/k3-b16-mm16ctl-result/seed0.json`.
- MM16 server/client logs: `/tmp/k3-b16-mm16ctl-server.log` and
  `/tmp/k3-b16-mm16ctl-client.log`.

## Long-output decode ceiling

The MM16 control then ran seed 0 at concurrency 16, 16 requests, 512 forced
output tokens, and one warmup. It completed 16/16 requests with 8,192/8,192
tokens and empty errors. Its prompts equal the first 16 short-cell prompts, and
each 512-token text starts with the corresponding 128-token control text.

| duration | output tok/s | median TTFT | median TPOT | P99 ITL |
|---:|---:|---:|---:|---:|
| 99.030 s | **82.723** | 4,184.58 ms | 182.98 ms | 513.46 ms |

The last 512 logged decode steps are exactly the main cell after vLLM's initial
probe and warmup. Their mean is 176.861 ms/step: 173.163 ms GPU drain, 3.502 ms
host work, including 1.750 ms compact TP audit. At 16 generated tokens per
step, that is a measured decode ceiling of 90.47 tok/s; the GPU-drain-only
ceiling is 92.40 tok/s.

Long client command differences:

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8026 \
  --endpoint /v1/chat/completions \
  --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random --random-input-len 32 --random-output-len 512 \
  --random-range-ratio 0 --request-rate inf \
  --max-concurrency 16 --num-prompts 16 --num-warmups 1 \
  --ignore-eos --temperature 0 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 --save-result --save-detailed \
  --result-dir /tmp/k3-b16-mm16ctl-long-result --result-filename seed0.json
```

The long hard gate changes to `completed == 16`, 16 lengths all equal to 512,
and `total_output_tokens == 8192`; the empty-error, generated-text, and input
accounting gates are unchanged. Artifacts are
`/tmp/k3-b16-mm16ctl-long-result/seed0.json`,
`/tmp/k3-b16-mm16ctl-long-server.log`, and
`/tmp/k3-b16-mm16ctl-long-client.log`.

## Conclusion

Reject this MM8+WALK object for B16 serving: despite a smaller object and less
private storage, it loses to matched MM16 on this one-seed serving control.
This does not reject other row caps or packet shapes.

Long output amortizes prefill and queue turnover enough to raise observed
throughput from 70.25 to 82.72 tok/s, but does not expose a 100 tok/s decode
path. A B16 step must be at most 160 ms. The measured step therefore needs a
16.86 ms (9.53%) total reduction; even removing all measured host work still
requires at least 13.16 ms from GPU drain. The next experiment should target
the decode body rather than repeat this MM8+WALK arm.
