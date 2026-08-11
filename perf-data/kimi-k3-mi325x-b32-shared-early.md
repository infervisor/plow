# Kimi-K3 MI325X B32 early shared-expert experiment

Date: 2026-08-11. Hardware: 8 leased MI325X GPUs, gfx942, 304 CUs each.
Toolchain: flake-pinned ROCm 7.14.0. Server: `plowrt serve`. Client:
flake-pinned vLLM 0.27.0 `bench serve`, one warmup.

## Experiment

The candidate moved only the shared expert's split gate/up GEMVs ahead of
`MoeRouterTopkPf`. `SituGlu` and shared down stayed at their original position;
shared down still waited on the exact routed `XReduceTwoShot` after
`MoeCombinePf`, preserving TP slot-A ownership.

The control and candidate B32 packets both had 2,942 instructions, 73,550
counters, and 442,850 stream entries. Their sorted raw-instruction multisets
had the same SHA256:

```text
80183dc9d951bf06e15b03dd8705b2a904f7027dbb9d91dc6fa2ce77dabb5684
```

Lean ordering and staged-LDS checks accepted all seven candidate programs.
The candidate reused the control HSACO; only packet ordering changed.

## Served result

Both cells used random 32-token requests before chat framing, concurrency 32,
512 generated tokens/request, greedy generation, `--ignore-eos`, seed 0, and
one warmup.

| arm | completed | output tokens | duration | output tok/s | median TPOT |
|---|---:|---:|---:|---:|---:|
| control | 32/32 | 16,384 | 148.79 s | 110.11 | 265.17 ms |
| early shared gate/up | 32/32 | 16,384 | 142.15 s | **115.26** | 253.42 ms |

The candidate improved served throughput by 4.68%. Its last 512 logged decode
ticks averaged 242.347 ms total and 238.747 ms GPU drain; local counter rearm
was 1.494 ms and compact TP audit was 1.630 ms. The one-request warmup fell
from 123.45 s to 111.56 s.

## Invalid first run

This result is not a valid B32-only A/B. The first gate used `T>1`, so it also
reordered shared gate/up in every prefill bucket. All requests completed with
exact output lengths, empty errors, and a compact exact-counter audit every
step, but every response diverged from its prompt-matched control after an
initial 104--555 characters. The 115.26 tok/s result therefore combines changed
prompt construction with the decode reorder and cannot be promoted or used as
the decode-only speedup.

The corrected experiment gates the overlap on `RowKind::Sequences`. Disassembly
proves programs 128--8192 byte/order-identical to control; only program 32
changes order, while its sorted raw-instruction hash remains identical.

The corrected candidate also enables TP local-counter double buffering. Counter
bank selection stays before launch, while the inactive bank's clear overlaps the
resident megakernels. Peer-visible cross-rank counters remain single-buffered
and are reset before any rank launches.

| corrected arm | completed | output tokens | duration | output tok/s | median TPOT |
|---|---:|---:|---:|---:|---:|
| decode-only early shared + counter overlap | 32/32 | 16,384 | 140.44 s | **116.67** | 249.74 ms |

All 32 generated texts and input lengths exactly match the control. Output
lengths are all 512, errors are empty, and no response contains an in-band
`[error:` marker. The server executed 1,053 compact TP audits. Over the last
512 ticks, total step time averaged 238.235 ms, GPU drain 234.612 ms, inactive
counter clear 1.509 ms, and compact audit 1.634 ms. This result accepts the
decode-only ordering and counter overlap, but remains below the 130 tok/s goal.

## Rejected follow-up

A default-off grouped-MoE experiment skipped explicit padded-row metadata and
derived live rows from the existing expert counts. Its poisoned-padding oracle
matched control byte-for-byte over 1,126,400 output bytes, and production
resources were unchanged. The served cell remained exact but regressed to
115.09 tok/s (-1.35% vs the accepted 116.67). The extra live-row predicates
cost more than the removed ALIGN initialization, so the production flag was
removed.

Artifacts:

- control JSON: `/tmp/k3-b32-long-result/seed0.json`
- candidate JSON: `/tmp/k3-b32-shared-early-result/seed0.json`
- candidate server log: `/tmp/k3-b32-shared-early-server.log`
- candidate packet: `/home/lava/models/k3_mi325x_b32_shared_early/model.pkt`
- corrected JSON: `/tmp/k3-b32-shared-decode-dbuf-result/seed0.json`
- corrected server log: `/tmp/k3-b32-shared-decode-server.log`
- corrected packet: `/home/lava/models/k3_mi325x_b32_shared_decode/model.pkt`
- rejected implicit-pad JSON: `/tmp/k3-b32-ipad-result/seed0.json`

The measured client command was:

```bash
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8033 \
  --endpoint /v1/chat/completions \
  --model k3_farm --served-model-name k3_farm \
  --tokenizer /home/lava/models/k3_tokz --tokenizer-mode hf \
  --dataset-name random --random-input-len 32 --random-output-len 512 \
  --random-range-ratio 0 --request-rate inf --max-concurrency 32 \
  --num-prompts 32 --num-warmups 1 --ignore-eos --temperature 0 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed \
  --result-dir /tmp/k3-b32-shared-early-result --result-filename seed0.json
```
