# Kimi-K3 MI325X FP8 MLA prefill V2

Date: 2026-08-11. Hardware: 8 leased MI325X GPUs, gfx942, 304 CUs/GPU.
Server: `plowrt serve`, TP8, native MXFP4 weights, FP8 KV. Performance client:
flake-pinned vLLM 0.27.0 `bench serve`. Accuracy harness:
`scripts/bench_gsm8k.sh`.

## Decision

Accept the numerics-changing V2 MLA-prefill axis. It reduces p50 TTFT by
5.01%, 9.19%, 15.38%, 23.35%, and 33.14% at matched 8K, 16K, 32K, 64K,
and 128K contexts. Decode TPOT is effectively unchanged (+0.05% to +0.26%).
The production accuracy gate scores 197/200 GSM8K, exactly matching the
established B1 score and clearing the minimum 196/200 non-inferiority bar.

The ladder candidate's primary decode object lacks the optional device
recurrent-state-clear symbol, so the 8K--32K ladder A/B used matched host clear.
The exact B1 64K/128K control and candidate decode objects are byte-identical,
carry `plow_state_clear`, and used matched device clear.

## Frozen artifacts and configuration

- control packet: `/home/lava/models/k3_mi325x_ladder_router/model.pkt`,
  SHA256 `f1f260d69105dffab3a7bd7f256d5fcbc215609f44c033c2cbb025949d14c709`
- candidate packet: `/home/lava/models/k3_mi325x_ladder_router_v2fp8_seg2/model.pkt`,
  SHA256 `96f3ea10bd1d04e889ef4cb32836d4190c832366567a8caf2f04c9a49750ab6a`
- candidate objects: `/home/lava/plow/build-amd/k3-mi325x-v2fp8-seg-hsaco`
- V2 FP8 flash static/GQ SHA256:
  `ce505118cd2e3039124c19fdcc4d4aaa4d5a9f944a956af3bd83b5930b067d22` /
  `1e2a3848542c9520108772b9f74678b34bf1ab64931e5646a7de16fcfb820fa7`
- frozen runtime SHA256:
  `ec7f5cce917931bac8713e799c257301ca09c402754009b65863cf6cb7b2f1b4`
- 128K control packet:
  `/home/lava/models/k3_mi325x_b1_ctx131072/model.pkt`, SHA256
  `ec1181202b9832a71c34cb8da3015215ebbb6333f658884e32a6bd65ad2eed28`
- 128K candidate packet:
  `/home/lava/models/k3_mi325x_b1_ctx131072_v2fp8/model.pkt`, SHA256
  `87c781bf21476425d663249e9cb5bd7e904353d047b96b3ce123dcba2bdca9f7`
- byte-identical 128K control/candidate B1 decode GQ object SHA256:
  `7604d152dcfcdf428ab222e10f8d12ec2d6ac655790a9e4cfd676a897c440f97`

The final flash GQ object exports `plow_k3_arms_1`, `plow_fp8_kv_1`,
`plow_mla_pf_v2_arm_1`, and `plow_l2_place_dispatch_1`. Its audited resources
are 512 VGPR, 256 AGPR, 58,376 B LDS, and zero spill. Static is 58,368 B LDS;
the two-object ISA audit passed 2/2.

Both ladder arms used:

- `PLOW_L2_PLACE_DISPATCH=1`
- `PLOW_TP_AUDIT_COMPACT=1`
- `PLOW_CTR_DBUF=1`
- host recurrent-state clear (`PLOW_STATE_CLEAR_DEVICE` unset)
- the identical canonical B1/B2/B4/B8 `PLOW_HSACO_LOWRUNG` mapping
- C1/N1, output 128, random range ratio 0, seed 0, temperature 0,
  ignore-EOS, and exactly one warmup

The candidate additionally used `PLOW_MLA_PF_V2=1` and the object directory
above. The corrected packet keeps T128/512/1024 L2-domain placed and
single-launch. Only T2048/4096/8192 use 49 pure wave-class segments; decode
remains L2 placed.

## Served ladder A/B

| requested input | actual input | control TTFT (ms) | V2 TTFT (ms) | TTFT delta | control TPOT (ms) | V2 TPOT (ms) | TPOT delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 8,213 | 4,888.85 | 4,643.81 | **-5.01%** | 56.220 | 56.338 | +0.21% |
| 16,000 | 16,021 | 9,702.46 | 8,810.74 | **-9.19%** | 57.677 | 57.829 | +0.26% |
| 32,000 | 32,021 | 22,204.06 | 18,789.63 | **-15.38%** | 61.199 | 61.338 | +0.23% |

Candidate E2EL also improves 1.91%, 5.12%, and 11.33%, respectively. Every
cell completed one request with exactly 128 output tokens, zero failed
requests, vLLM's successful empty-error sentinel `[""]`, and no in-band
`[error:` marker. Candidate and control compact TP audit logs are empty, and
the server logs contain no rank disagreement, counter failure, timeout,
device fault, or fatal signature.

The composed V2 flash object loaded on all eight ranks with zero fallback.
Measured candidate prefill used 50, 98, and 196 segment launches at 8K, 16K,
and 32K including ragged tails; control used 2, 2, and 4 single-launch chunks.
T128 correctness remained single-launch after the packet fix.

Greedy generated text is not byte-identical across arms at any of the three
random contexts. The outputs are coherent and length-correct, but their
SHA256 values differ. This is an explicit numerics-changing adoption, not a
token-identity claim; the task-level accuracy gate below is the deciding
correctness evidence.

Detailed results and logs:

- `/tmp/k3-v2fp8-seg-ab-20260811/candidate-ctx{8192,16000,32000}.json`
- `/tmp/k3-v2fp8-seg-ab-20260811/control-ctx{8192,16000,32000}.json`
- `/tmp/k3-v2fp8-seg-ab-20260811/{candidate,control}-server.log`
- `/tmp/k3-v2fp8-seg-ab-20260811/commands.log`

## Exact B1 64K/128K A/B

The long-context run used the exact B1 assets above, their asset-local object
directories, and the same frozen runtime. Both arms enabled
`PLOW_L2_PLACE_DISPATCH=1`, `PLOW_TP_AUDIT_COMPACT=1`, `PLOW_CTR_DBUF=1`, and
`PLOW_STATE_CLEAR_DEVICE=1`; the candidate alone enabled
`PLOW_MLA_PF_V2=1`. The client geometry, seed, generation policy, and one
warmup matched the ladder A/B.

| requested input | actual input | control TTFT (ms) | V2 TTFT (ms) | TTFT delta | control TPOT (ms) | V2 TPOT (ms) | TPOT delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64,000 | 64,021 | 55,563.81 | 42,591.46 | **-23.35%** | 68.275 | 68.317 | +0.06% |
| 128,000 | 128,021 | 156,771.58 | 104,822.28 | **-33.14%** | 82.441 | 82.479 | +0.05% |

E2EL improves 20.19% and 31.06%. TTFT-inclusive output throughput improves
25.29% and 45.05%, respectively. All four measured cells completed one request
with exactly 128 output tokens, zero failures, successful empty-error sentinels,
and no in-band error. Both compact audit files are empty and the server logs
contain no rank, counter, timeout, fatal, or device-fault signature.

The candidate flash object loaded on all eight ranks with zero fallback.
Candidate prefill used 392 and 784 segment launches at 64K and 128K; control
used 8 and 16 single-launch chunks. Greedy generated text again differs across
arms, consistent with the accepted numerics-changing axis and the GSM8K gate.

Evidence:

- `/tmp/k3-v2fp8-b1-long-ab-20260811/{candidate,control}-ctx64000.json`
- `/tmp/k3-v2fp8-b1-long-ab-20260811/{candidate,control}-ctx128000.json`
- `/tmp/k3-v2fp8-b1-long-ab-20260811/{candidate,control}-server.log`
- `/tmp/k3-v2fp8-b1-long-ab-20260811/commands.log`

## Accuracy gate

Final candidate only: first 200 GSM8K test questions, 8-shot chain-of-thought,
greedy temperature 0, maximum 320 output tokens, concurrency 1. Exact match
uses the final parsed number.

| metric | result |
|---|---:|
| Paris coherence | PASS |
| exact match | **197/200 = 0.9850** |
| request errors | 0 |
| median latency/question | 6.00 s |
| mean latency/question | 6.53 s |
| measured wall time | 1,306 s |

The score equals the established single-rung B1 result (197/200) and exceeds
the predeclared 196/200 adoption bar. Compact rank/counter audit remained
clean, the V2 flash object loaded on all eight ranks without fallback, and the
post-run GPU audit found no foreign compute process.

Dataset SHA256: test
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`;
train `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`.
The exact accuracy invocation was:

```bash
nix develop -c env -u PLOW_STATE_CLEAR_DEVICE \
  PLOW_MLA_PF_V2=1 \
  PLOW_HSACO=/home/lava/plow/build-amd/k3-mi325x-v2fp8-seg-hsaco \
  PLOW_L2_PLACE_DISPATCH=1 PLOW_TP_AUDIT_COMPACT=1 PLOW_CTR_DBUF=1 \
  PLOW_HSACO_LOWRUNG=/home/lava/plow/build-amd/k3-b1-ladder-grouped:1,/home/lava/plow/build-amd/k3-b2-ladder-grouped:2,/home/lava/plow/build-amd/k3-b4-ladder-grouped:4,/home/lava/plow/build-amd/k3-b8-ladder-grouped:8 \
  N=200 SHOTS=8 MAXTOK=320 CONC=1 GSM8K_TEMPERATURE=0 \
  PLOWRT_BIN=/tmp/k3-v2fp8-seg-ab-plowrt \
  perf-data/harness/gpulease -n 8 k3-v2fp8-seg2-gsm8k \
  scripts/bench_gsm8k.sh \
  /home/lava/models/k3_mi325x_ladder_router_v2fp8_seg2 8053 auto 1800
```

Accuracy evidence:

- `/tmp/k3-v2fp8-seg-ab-20260811/gsm8k-client.log`, SHA256
  `9f2469aa872dce851e9db7ad4873a79e10258a308e50c0fd1661538fafe9eb0d`
- `/tmp/k3-v2fp8-seg-ab-20260811/gsm8k-server.log`, SHA256
  `abc545c7a2e510d4fe8c15aa100c258d7407bf2e8dc095390f03a19d14f3300a`
- `/tmp/k3-v2fp8-seg-ab-20260811/gsm8k-audit-errors.txt` (empty)

## Preflight refusals

No invalid run contributed a result:

1. Enabling device recurrent-state clear was refused because the candidate
   primary decode object lacks `plow_state_clear`. Both measured arms switched
   to matched host clear. Evidence:
   `/tmp/k3-v2fp8-seg-ab-20260811/candidate-server-state-clear-reject.log`.
2. The first corrected mixed-placement packet found that the V2 flash object
   lacked `plow_l2_place_dispatch_1`; the runtime rejected it and would have
   fallen back to the generic interpreter. The composed final objects add the
   marker. Evidence:
   `/tmp/k3-v2fp8-seg-ab-20260811/candidate-server-seg2-l2-reject.log`.

An earlier packet that unnecessarily segmented T128/512/1024 was also aborted
before HTTP readiness and produced no client result. The final seg2 packet is
the only candidate used in the tables and accuracy score.
