# Kimi-K3 plowrt TP8 throughput gates (MI355X)

Current-source control artifact, 2026-09-03: gfx950, 8×MI355X, ROCm 7.14.0, BF16 KV,
MXFP4 experts, default global queue with L2-domain placement. Atomic/deterministic grouped-MoE
prefill fusion is off.

## Artifact and memory gate

- Asset: `/tmp/plow-k3-throughput-control-gq.37c4Il/assets`
- Shape: max context 16384; prefill rungs 128/512/1024/2048/4096/8192; decode rungs
  1/32/64/128; `PLOW_GEMV_WALK=1`.
- Offline runtime tensors: 3,067 tensors, 71,403,824,764 B = 66.500 GiB/rank.
- Largest extents: each of 24 MLA compressed-KV tensors is 2.0 GiB;
  `act.pf.moe.part` is 1.75 GiB.
- Projected device use: 66.50 GiB runtime + 191.23 GiB observed weights + about 0.34 GiB
  TP peer + 0.5 GiB KV pool = about 258.6 GiB, below the 288 GiB device budget.
- A guarded TP8 load/small-request smoke passed with batch capacity 128 and decode rungs
  `[1,32,64,128]`. Log: `/tmp/plow-k3-throughput-control-gq.37c4Il/load-smoke.log`.

Artifact SHA-256:

| Artifact | SHA-256 |
|:---|:---|
| `model.pkt` | `5e8193ffc2926384cf3669b9750b656d2396fb382f41559b546483404ed98c08` |
| `build.json` | `87d411559d427cb13b4056dae9f41244b68a7199ff813d5323d933765c96f4db` |
| K3 decode GQ object | `f854501e3742d38f18d9f7d8e52811977657097ba3709c6cc916b6ffddec172d` |
| K3 A4W4 prefill GQ object | `6c94a82591d82417d3a253716356f7449babb27b2388f420de9cb9464c64195d` |

## Decode admission sanity

Command cell: random input 128, concurrency/requests 128, warmup 1, output 32.

- Completed 128/128, failed 0; rejected 0; admission shedding 0.
- Actual/admission/occupied decode rung: 128/128/128; rung switches 2.
- Output throughput 39.6682 token/s; request throughput 1.23963 request/s.
- Mean TPOT 680.222 ms; mean TTFT 40,568.594 ms.
- Output checksum `fnv1a64:52c0e37cee9d8d95`.
- Log: `/tmp/plow-k3-throughput-control-gq.37c4Il/decode-throughput-sanity.log`.

## 8192→32 short-output gate

Command cell: random input 8192, concurrency/requests 128, warmup 1, output 32.

- Completed 128/128, failed 0; rejected 0; admission shedding 0.
- Actual/admission/occupied decode rung: 128/128/128; rung switches 1.
- 1,048,576 prompt and 4,096 output tokens in 807,226.184 ms.
- Total throughput 1,304.061 token/s; output throughput 5.07417 token/s;
  request throughput 0.158568 request/s.
- Mean TTFT 382,965.122 ms (p50 366,773.839; p90 713,460.322; p99 786,630.252).
- Mean TPOT 1,482.914 ms (p50 1,453.092; p90 1,629.156; p99 1,630.228).
- Output checksum `fnv1a64:1d577cc08174cba1`.
- Log: `/tmp/plow-k3-throughput-control-gq.37c4Il/throughput-8k-to-32.log`.

This is a **short-output admission/throughput gate**, not an apples-to-apples comparison with the
vLLM 8192→1024 baseline. Packed prefill dispatch remains disabled, so the 128 long prompts are
prefilled through the current serialized path; the TTFT distribution primarily exposes that gap.

